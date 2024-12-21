import argparse
import torch
from torch import nn
from torch.export import export, ExportedProgram, Dim
from executorch.exir import EdgeProgramManager, ExecutorchBackendConfig, to_edge

from huggingface_hub import hf_hub_download
from moshi.models import loaders


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--moshi-weight", type=str, help="Path to a local checkpoint file for Moshi.")
    parser.add_argument("--hf-repo", type=str, default=loaders.DEFAULT_REPO,
                        help="HF repo to look into, defaults Moshiko. "
                             "Use this to select a different pre-trained model.")
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    if args.moshi_weight is None:
        args.moshi_weight = hf_hub_download(args.hf_repo, loaders.MOSHI_NAME)
    moshi = loaders.get_moshi_lm(args.moshi_weight, args.device)
    print("moshi model loaded")

    print("running the model")
    sample_codes = torch.zeros((1, 8, 1), dtype=torch.int64).to(args.device)
    out_codes = moshi(sample_codes)
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

    filename = "moshi.pte"
    print(f"writing {filename}")
    with open(filename, "wb") as file:
        file.write(executorch_program.buffer)

with torch.no_grad():
    main()

