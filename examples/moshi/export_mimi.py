import argparse
import torch
from torch import nn
from torch.export import export, ExportedProgram
from executorch.exir import EdgeProgramManager, ExecutorchBackendConfig, to_edge

from huggingface_hub import hf_hub_download
from moshi.models import loaders


class MimiEncoder(nn.Module):
    def __init__(self, mimi):
        super().__init__()
        self.mimi = mimi

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mimi.encode(x)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mimi-weight", type=str, help="Path to a local checkpoint file for Mimi.")
    parser.add_argument("--hf-repo", type=str, default=loaders.DEFAULT_REPO,
                        help="HF repo to look into, defaults Moshiko. "
                             "Use this to select a different pre-trained model.")
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    if args.mimi_weight is None:
        args.mimi_weight = hf_hub_download(args.hf_repo, loaders.MIMI_NAME)
    mimi = loaders.get_mimi(args.mimi_weight, args.device)
    print("==== MIMI MODEL ====")
    print(mimi)

    mimi_encoder = MimiEncoder(mimi)

    sample_pcm = torch.zeros((1, 1, 1920), dtype=torch.float).to(args.device)

    print("==== RUN MODEL ====")
    _out = mimi_encoder(sample_pcm)
    print(_out.shape)

    print("==== ATEN DIALECT ====")
    aten_dialect: ExportedProgram = export(mimi_encoder, (sample_pcm,))
    print(aten_dialect)
    edge_program: EdgeProgramManager = to_edge(aten_dialect)

    print("==== EDGE PROGRAM ====")
    print(edge_program)

    executorch_program: exir.ExecutorchProgramManager = edge_program.to_executorch(
        ExecutorchBackendConfig(
            passes=[],  # User-defined passes
        )
    )

    with open("mimi.pte", "wb") as file:
        file.write(executorch_program.buffer)

with torch.no_grad():
    main()
