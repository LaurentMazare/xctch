# https://pytorch.org/executorch/stable/llm/getting-started.html
# export_nanogpt.py

# Load partitioner for Xnnpack backend
from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner

# Model to be delegated to specific backend should use specific edge compile config
from executorch.backends.xnnpack.utils.configs import get_xnnpack_edge_compile_config
from executorch.exir import EdgeCompileConfig, to_edge

import torch
from torch.export import export
from torch.nn.attention import sdpa_kernel, SDPBackend
from torch._export import capture_pre_autograd_graph

from model import GPT

# Load the nanoGPT model.
model = GPT.from_pretrained('gpt2')

# Create example inputs. This is used in the export process to provide
# hints on the expected shape of the model input.
example_inputs = (
        torch.randint(0, 100, (1, model.config.block_size - 1), dtype=torch.long),
    )

# Set up dynamic shape configuration. This allows the sizes of the input tensors
# to differ from the sizes of the tensors in `example_inputs` during runtime, as
# long as they adhere to the rules specified in the dynamic shape configuration.
# Here we set the range of 0th model input's 1st dimension as
# [0, model.config.block_size].
# See https://pytorch.org/executorch/main/concepts.html#dynamic-shapes
# for details about creating dynamic shapes.
dynamic_shape = (
    {1: torch.export.Dim("token_dim", max=model.config.block_size - 1)},
)

# Trace the model, converting it to a portable intermediate representation.
# The torch.no_grad() call tells PyTorch to exclude training-specific logic.
with torch.nn.attention.sdpa_kernel([SDPBackend.MATH]), torch.no_grad():
    m = capture_pre_autograd_graph(model, example_inputs, dynamic_shapes=dynamic_shape)
    traced_model = export(m, example_inputs, dynamic_shapes=dynamic_shape)

# Convert the model into a runnable ExecuTorch program.
# To be further lowered to Xnnpack backend, `traced_model` needs xnnpack-specific edge compile config
edge_config = get_xnnpack_edge_compile_config()
edge_manager = to_edge(traced_model) # , compile_config=edge_config)

# Delegate exported model to Xnnpack backend by invoking `to_backend` function with Xnnpack partitioner.
# edge_manager = edge_manager.to_backend(XnnpackPartitioner())
et_program = edge_manager.to_executorch()

# Save the Xnnpack-delegated ExecuTorch program to a file.
with open("scripts/nanogpt/nanogpt.pte", "wb") as file:
    file.write(et_program.buffer)
