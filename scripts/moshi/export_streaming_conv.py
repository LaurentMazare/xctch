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
        # Use some zeros for the initial values here as we don't
        # seem to have control on the actual value that ends up being used.
        # This is likely related to the following warning that executorch
        # produces:
        #     UserWarning: Mutation on a buffer in the model is detected.
        #     ExecuTorch assumes buffers that are mutated in the graph have
        #     a meaningless initial state, only the shape and dtype will be
        #     serialized.
        self.register_buffer("_input_buf", torch.zeros((1, self.in_channels, kernel), dtype=torch.float))
        self.register_buffer("_init", torch.zeros(1, dtype=torch.float))

    # Assume that the seq-len used in the forward is always the same.
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        input_buf = self._input_buf + (1.0 - self._init) * input[0, 0, 0]
        stride = self.stride[0]
        # Effective kernel size accounting for dilation.
        kernel = (self.kernel_size[0] - 1) * self.dilation[0] + 1
        input = torch.cat([input_buf, input], dim=-1)
        _, _, T = input.shape
        # We now compute the number of full convolution frames, i.e. the frames
        # that are ready to be computed.
        num_frames = max(0, (T - kernel) // stride + 1)
        offset = num_frames * stride
        # We will compute `num_frames` outputs, and we are advancing by `stride`
        # for each of the frame, so we know the data before `stride * num_frames`
        # will never be used again.
        self._input_buf[:] = input[..., offset:]
        self._init.fill_(1.0)
        input_length = (num_frames - 1) * stride + kernel
        out = super().forward(input[..., :input_length])
        return out

    def reset(self):
        self._input_buf.fill_(0.0)
        self._init.fill_(0.0)


class ModelReset(nn.Module):
    def __init__(self, model):
        super().__init__()
        self._inner = model

    def forward(self, x):
        self._inner.reset()
        return x

def main():
    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    torch.manual_seed(42)

    model = RawStreamingConv1d(1, 2, 3)
    m = ModelReset(model)
    seq_len = 5
    inp_ = torch.arange(0, seq_len, 1, dtype=torch.float).reshape((1, 1, seq_len))
    inp_ = inp_.cos()

    print("running the conv")
    for idx in range(6):
        print(">>>", model._init)
        sample_codes = model(inp_)
        print(sample_codes)
        print(sample_codes.shape)
        if idx == 3:
            m.forward(inp_)

    model.reset()
    aten_forward: ExportedProgram = export(model, (inp_,))
    aten_reset: ExportedProgram = export(m, args=(inp_,))
    print(aten_forward)
    print(aten_reset)
    edge_program: EdgeProgramManager = to_edge({
        "forward": aten_forward,
        "reset": aten_reset,
    })

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

