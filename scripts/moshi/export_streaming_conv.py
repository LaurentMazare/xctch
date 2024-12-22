import math
import argparse
import torch
from torch import nn
from torch.export import export, ExportedProgram, Dim
from executorch.exir import EdgeProgramManager, ExecutorchBackendConfig, to_edge

class RawStreamingConv1d(nn.Conv1d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert self.padding[0] == 0, "Padding should be handled outside."
        assert (
            self.stride[0] <= self.kernel_size[0]
        ), "stride must be less than kernel_size."
        kernel = (self.kernel_size[0] - 1) * self.dilation[0]
        self.register_buffer("_input_buf", torch.zeros((1, self.in_channels, kernel), dtype=torch.float))

    # Assume that the seq-len used in the forward is always the same.
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        stride = self.stride[0]
        # Effective kernel size accounting for dilation.
        kernel = (self.kernel_size[0] - 1) * self.dilation[0] + 1
        input = torch.cat([self._input_buf, input], dim=-1)
        B, C, T = input.shape
        # We now compute the number of full convolution frames, i.e. the frames
        # that are ready to be computed.
        num_frames = max(0, (T - kernel) // stride + 1)
        offset = num_frames * stride
        # We will compute `num_frames` outputs, and we are advancing by `stride`
        # for each of the frame, so we know the data before `stride * num_frames`
        # will never be used again.
        self._input_buf[:] = input[..., offset:]
        input_length = (num_frames - 1) * stride + kernel
        out = super().forward(input[..., :input_length])
        return out


def main():
    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    torch.manual_seed(42)

    model = RawStreamingConv1d(1, 2, 3)
    seq_len = 5
    inp_ = torch.arange(0, seq_len, 1, dtype=torch.float).reshape((1, 1, seq_len))
    inp_ = inp_.cos()

    print("running the conv")
    for _ in range(3):
        sample_codes = model(inp_)
        print(sample_codes)
        print(sample_codes.shape)

    model._input_buf.fill_(0.0)
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

