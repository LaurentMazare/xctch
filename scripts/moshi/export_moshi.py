import argparse
from pathlib import Path
import copy
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
from torch.nn.attention import SDPBackend
from executorch.extension.export_util.utils import export_to_edge
from executorch.devtools import generate_etrecord
from safetensors import safe_open
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
    "n_q": 32,
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
    "delays": [0] * 33,
}


_lm_kwargs = _lm_kwargs300m

def _is_safetensors(path: Path | str) -> bool:
    return Path(path).suffix in (".safetensors", ".sft", ".sfts")


def get_moshi_lm(filename, strict, device='cpu') -> LMModel:
    dtype = torch.bfloat16
    # dtype = torch.float
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
        return logits.to(dtype=torch.float)

def run_sample(model, sample_file, device):
    with safe_open(sample_file, framework="pt", device=device) as fobj:
        codes = fobj.get_tensor("codes").to(dtype=torch.int64)
        print(codes.shape, codes.dtype)

    _, codebooks, seqlen = codes.shape

    init_codes = [_lm_kwargs["text_card"]] + [_lm_kwargs["card"]] * _lm_kwargs["n_q"]
    init_codes = torch.tensor(init_codes, dtype=torch.int64, device=device)[None, :, None]
    print(init_codes.shape)
    _ = model(init_codes)

    last_token = -1
    for step in range(seqlen):
        text_token = -1
        if step >= 25:
            text_token = last_token
        text_codes = torch.tensor([[[text_token]]], dtype=torch.int64, device=device)
        step_codes = codes[:, :, step:step+1]
        pad_codes = -torch.ones((1, _lm_kwargs["n_q"] - codebooks, 1), dtype=torch.int64, device=device)
        in_codes = torch.cat([text_codes, step_codes, pad_codes], dim=1)
        logits = model(in_codes)
        last_token = torch.argmax(logits, -1).item()
        print(step, last_token)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--moshi-weights", type=str, help="Path to a local checkpoint file for Moshi.")
    parser.add_argument("--quantized", action="store_true")
    parser.add_argument("--relaxed", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--sample-file", type=str)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    torch.manual_seed(42)

    model = get_moshi_lm(args.moshi_weights, not args.relaxed, args.device)
    if args.verbose:
        print(model)
    model = LM(model)
    print("moshi model loaded")

    print("running the model")
    sample_codes = torch.zeros((1, 17, 1), dtype=torch.int64).to(args.device)
    if args.sample_file is not None:
        run_sample(model, args.sample_file, args.device)
    for i in range(5):
        start_time = time.time()
        out_codes = model(sample_codes)
        dt = time.time() - start_time
        # print(out_codes[0, 0, 0, :20])
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
        quantizer = XNNPACKQuantizer().set_global(linear_config)
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

        print("exporting to executorch")
        executorch_program: exir.ExecutorchProgramManager = edge_program.to_executorch(
            ExecutorchBackendConfig(
                passes=[],  # User-defined passes
            )
        )

    else:
        from torch._export import capture_pre_autograd_graph
        from executorch.backends.xnnpack._passes.convert_to_linear import ConvertToLinearPass
        from executorch.exir.passes.quant_fusion_pass import QuantFusionPass
        from executorch.exir.passes import MemoryPlanningPass
        from executorch.exir.passes.sym_shape_eval_pass import ConstraintBasedSymShapeEvalPass

        print("exporting to aten and edge")
        aten_dialect: ExportedProgram = capture_pre_autograd_graph(model, (sample_codes,))
        edge_config = exir.EdgeCompileConfig(
            _check_ir_validity=False,
            _skip_type_promotion=False,
            _skip_dim_order=True,
        )
        edge_manager = export_to_edge(
            aten_dialect,
            (sample_codes,),
            dynamic_shapes=None,
            edge_constant_methods={},
            edge_compile_config=edge_config,
            verbose=False,
        )
        edge_manager = edge_manager.to_backend(XnnpackPartitioner())
        if args.verbose:
            print(edge_manager.exported_program().graph)

        edge_manager_copy = copy.deepcopy(edge_manager)
        executorch_program = edge_manager.to_executorch(
            ExecutorchBackendConfig(
                extract_delegate_segments=True,
                passes=[
                    # If there are Linear operations left in the graph, let's execute
                    # them with the optimized op_linear rather than materializing a
                    # transpose followed by a regular op_mm.
                    ConvertToLinearPass(),
                    QuantFusionPass(),
                ],
                memory_planning_pass=MemoryPlanningPass(alloc_graph_input=False),
                sym_shape_eval_pass=ConstraintBasedSymShapeEvalPass(),
            )
        )
        etrecord_path = "etrecord.bin"
        generate_etrecord(etrecord_path, edge_manager_copy, executorch_program)


    filename = "moshi-lm.pte"
    print(f"writing {filename}")
    with open(filename, "wb") as file:
        file.write(executorch_program.buffer)

with torch.nn.attention.sdpa_kernel([SDPBackend.MATH]), torch.no_grad():
    main()

