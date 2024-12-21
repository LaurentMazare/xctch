import argparse
import torch
from torch import nn
from torch.export import export, ExportedProgram, Dim
from executorch.exir import EdgeProgramManager, ExecutorchBackendConfig, to_edge

from huggingface_hub import hf_hub_download
from moshi.models import loaders


class MimiEncoder(nn.Module):
    def __init__(self, mimi):
        super().__init__()
        self.mimi = mimi

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mimi.encode(x)

class MimiDecoder(nn.Module):
    def __init__(self, mimi):
        super().__init__()
        self.mimi = mimi

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mimi.decode(x)


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
    print("mimi model loaded")

    seq_len = 24000
    mimi_encoder = MimiEncoder(mimi)
    mimi_decoder = MimiDecoder(mimi)
    sample_pcm = torch.zeros((1, 1, seq_len), dtype=torch.float).to(args.device)

    print("running the encoder")
    sample_codes = mimi_encoder(sample_pcm)
    print(sample_codes.shape)
    print("running the decoder")
    _pcm = mimi_decoder(sample_codes)
    print(_pcm.shape)

    for model, inp_, filename in [
            (mimi_encoder, sample_pcm, "mimi_encoder.pte"),
            (mimi_decoder, sample_codes, "mimi_decoder.pte")]:
        print("exporting to aten and edge")
        aten_dialect: ExportedProgram = export(
            model,
            (inp_,),
            # Dynamic shapes result in unresolvable constraints
            # dynamic_shapes={"x": {2: Dim("t", min=1, max=240000)}},
        )
        edge_program: EdgeProgramManager = to_edge(aten_dialect)

        print("exporting to executorch")
        executorch_program: exir.ExecutorchProgramManager = edge_program.to_executorch(
            ExecutorchBackendConfig(
                passes=[],  # User-defined passes
            )
        )

        print(f"writing {filename}")
        with open(filename, "wb") as file:
            file.write(executorch_program.buffer)

with torch.no_grad():
    main()
