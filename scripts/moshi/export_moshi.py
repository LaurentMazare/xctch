import argparse
from pathlib import Path
import torch
from torch import nn
from torch.export import export, ExportedProgram, Dim
from executorch.exir import EdgeProgramManager, ExecutorchBackendConfig, to_edge

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


_lm_kwargs = _lm_kwargs300m

def _is_safetensors(path: Path | str) -> bool:
    return Path(path).suffix in (".safetensors", ".sft", ".sfts")


def get_moshi_lm(filename, device='cpu') -> LMModel:
    # dtype = torch.bfloat16
    dtype = torch.float
    model = LMModel(
        device=device,
        dtype=dtype,
        **_lm_kwargs,
    ).to(device=device, dtype=dtype)
    model.eval()
    if filename is not None:
        if _is_safetensors(filename):
            load_model(model, filename)
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
    parser.add_argument("--moshi-weight", type=str, help="Path to a local checkpoint file for Moshi.")
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    model = get_moshi_lm(args.moshi_weight, args.device)
    model = LM(model)
    print("moshi model loaded")

    print("running the model")
    sample_codes = torch.zeros((1, 17, 1), dtype=torch.int64).to(args.device)
    out_codes = model(sample_codes)
    print(out_codes.shape)

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

