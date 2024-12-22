import argparse
import torch
from torch import nn
from torch.export import export, ExportedProgram, Dim
from executorch.exir import EdgeProgramManager, ExecutorchBackendConfig, to_edge

from huggingface_hub import hf_hub_download
from moshi.modules import conv



def main():
    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    torch.manual_seed(42)

    model = conv.StreamingConv1d(1, 2, 3, causal=True)
    model.streaming_forever(1)
    seq_len = 20
    inp_ = torch.arange(0, seq_len, 1, dtype=torch.float).reshape((1, 1, seq_len))
    inp_ = inp_.cos()

    print("running the conv")
    for _ in range(3):
        sample_codes = model(inp_)
        print(sample_codes.shape)
        print(sample_codes[:, :, :10])
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

    with open("streaming_conv1d.pte", "wb") as file:
        file.write(executorch_program.buffer)

with torch.no_grad():
    main()

