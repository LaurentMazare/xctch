from torch.export import export
from executorch import exir
from executorch.exir.backend.test.demos.rpc.executor_backend_partitioner import (
    ExecutorBackendPartitioner,
)
import torch

def run():
    class SharedModule(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(2, 2)

        def forward(self, x):
            return self.linear(x)


    class Module1(torch.nn.Module):
        def __init__(self, shared_module):
            super().__init__()
            self.shared_module = shared_module

        def forward(self, x):
            return self.shared_module(x)


    class Module2(torch.nn.Module):
        def __init__(self, shared_module):
            super().__init__()
            self.shared_module = shared_module

        def forward(self, x):
            self.shared_module.linear.weight.fill_(0.0)
            return self.shared_module(x)

    shared_module = SharedModule()
    module_1 = Module1(shared_module)
    module_2 = Module2(shared_module)
    example_inputs = (torch.randn(2, 2),)
    module_1(*example_inputs)
    module_2(*example_inputs)

    ep1 = export(module_1, example_inputs)
    ep2 = export(module_2, example_inputs)

    edge_program_manager = exir.to_edge(
        {"forward1": ep1, "forward2": ep2},
        compile_config=exir.EdgeCompileConfig(
            _check_ir_validity=False, _use_edge_ops=True
        ),
    )

    edge_program_manager = edge_program_manager.to_backend(ExecutorBackendPartitioner()).to_executorch()

with torch.no_grad():
    run()
