import argparse
from pathlib import Path
import time
import torch
from torch import nn
from torch.export import export, export_for_training, ExportedProgram, Dim
from executorch import exir
from executorch.exir import EdgeProgramManager, ExecutorchBackendConfig, to_edge, to_edge_transform_and_lower
from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner
from torch.ao.quantization.quantize_pt2e import convert_pt2e, prepare_pt2e
from torch.ao.quantization.quantizer.xnnpack_quantizer import (
    get_symmetric_quantization_config,
    XNNPACKQuantizer,
)
from safetensors.torch import load_model

from moshi.models.lm import LMModel

_lm_kwargs7b = {
    "dim": 4096,
    "text_card": 32000,
    "existing_text_padding_id": 3,
    "n_q": 16,
    "dep_q": 8,
    "card": 2048,
    "num_heads": 32,
    "num_layers": 32,
    "hidden_scale": 4.125,
    "causal": True,
    "layer_scale": None,
    "context": 3000,
    "max_period": 10000,
    "gating": "silu",
    "norm": "rms_norm_f32",
    "positional_embedding": "rope",
    "depformer_dim": 1024,
    "depformer_dim_feedforward": int(4.125 * 1024),
    "depformer_num_heads": 16,
    "depformer_num_layers": 6,
    "depformer_causal": True,
    "depformer_layer_scale": None,
    "depformer_multi_linear": True,
    "depformer_context": 8,
    "depformer_max_period": 10000,
    "depformer_gating": "silu",
    "depformer_pos_emb": "none",
    "depformer_weights_per_step": True,
    "delays": [0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1],
}

_lm_kwargs300m = {
    "dim": 1024,
    "text_card": 48000,
    "existing_text_padding_id": 3,
    "n_q": 16,
    "dep_q": 0,
    "card": 2048,
    "num_heads": 8,
    "num_layers": 16,
    "hidden_scale": 4.125,
    "causal": True,
    "layer_scale": None,
    "context": 750,
    "max_period": 100000,
    "gating": "silu",
    "norm": "rms_norm_f32",
    "positional_embedding": "rope",
    "depformer_dim": 1024,
    "depformer_dim_feedforward": int(4.125 * 1024),
    "depformer_num_heads": 16,
    "depformer_num_layers": 0,
    "depformer_causal": True,
    "depformer_layer_scale": None,
    "depformer_multi_linear": True,
    "depformer_context": 8,
    "depformer_max_period": 10000,
    "depformer_gating": "silu",
    "depformer_pos_emb": "none",
    "depformer_weights_per_step": True,
    "delays": [0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1],
}


_lm_kwargs = _lm_kwargs300m

def _is_safetensors(path: Path | str) -> bool:
    return Path(path).suffix in (".safetensors", ".sft", ".sfts")


def get_moshi_lm(filename, strict, device='cpu') -> LMModel:
    # dtype = torch.bfloat16
    dtype = torch.float
    model = LMModel(
        device=device,
        dtype=dtype,
        **_lm_kwargs,
    ).to(device=device, dtype=dtype)
    model.eval()
    model.streaming_forever(1);
    if filename is not None:
        if _is_safetensors(filename):
            load_model(model, filename, strict=strict)
        else:
            pkg = torch.load(
                filename,
                "cpu",
            )
            model.load_state_dict(pkg["fsdp_best_state"]["model"])
    return model

class LM(nn.Module):
    def __init__(self, lm_model):
        super().__init__()
        self._lm_model = lm_model

    def forward(self, x):
        _, logits = self._lm_model.forward_text(x)
        return logits

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--moshi-weights", type=str, help="Path to a local checkpoint file for Moshi.")
    parser.add_argument("--quantized", action="store_true")
    parser.add_argument("--relaxed", action="store_true")
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    model = get_moshi_lm(args.moshi_weights, not args.relaxed, args.device)
    model = LM(model)
    print("moshi model loaded")

    print("running the model")
    sample_codes = torch.zeros((1, 17, 1), dtype=torch.int64).to(args.device)
    for i in range(5):
        start_time = time.time()
        out_codes = model(sample_codes)
        dt = time.time() - start_time
        print(out_codes[0, 0, 0, :20])
        print(f"step {i} shape: {out_codes.shape} dt: {1000 * dt:.0f}ms")
    model._lm_model.reset_streaming()


    if args.quantized:
        print("exporting to aten and edge")
        aten_dialect: ExportedProgram = export_for_training(model, (sample_codes,)).module()
        print("quantizing")
        linear_config = get_symmetric_quantization_config(
            is_per_channel=False,
            is_dynamic=False,
        )
        # We only enable quantization for the linear layers. Otherwise the embedding layers
        # (ScaledEmbedding) seem to only return zeros.
        quantizer = XNNPACKQuantizer()
        quantizer.set_module_type(torch.nn.functional.linear, linear_config)
        prepared_graph = prepare_pt2e(aten_dialect, quantizer)
        converted_graph = convert_pt2e(prepared_graph)
        aten_dialect: ExportedProgram = export(converted_graph, (sample_codes,))

        edge_program: EdgeProgramManager = to_edge_transform_and_lower(
            aten_dialect,
            partitioner=[XnnpackPartitioner()],
            # When exporting the bfloat16 version, the ir validity check fails.
            # Maybe related: https://github.com/pytorch/executorch/issues/6685
            # compile_config=exir.EdgeCompileConfig(_check_ir_validity=False)
        )
    else:
        print("exporting to aten and edge")
        aten_dialect: ExportedProgram = export(model, (sample_codes,))
        edge_program: EdgeProgramManager = to_edge(aten_dialect)

    print("exporting to executorch")
    executorch_program: exir.ExecutorchProgramManager = edge_program.to_executorch(
        ExecutorchBackendConfig(
            passes=[],  # User-defined passes
        )
    )

    filename = "moshi-lm.pte"
    print(f"writing {filename}")
    with open(filename, "wb") as file:
        file.write(executorch_program.buffer)

with torch.no_grad():
    main()

