import executorch
from executorch.exir.backend.test.demos.rpc.executor_backend_partitioner import (
    ExecutorBackendPartitioner,
)
import torch

class SharedModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self._v = torch.nn.Parameter(torch.ones(1, dtype=torch.float), requires_grad=False)


class Module1(torch.nn.Module):
    def __init__(self, shared_module):
        super().__init__()
        self.shared_module = shared_module

    def forward(self, x):
        self.shared_module._v[:] = self.shared_module._v + x
        return self.shared_module._v


class Module2(torch.nn.Module):
    def __init__(self, shared_module):
        super().__init__()
        self.shared_module = shared_module

    def forward(self, x):
        self.shared_module._v.fill_(0.0)
        return x

def export():
    shared_module = SharedModule()
    module_1 = Module1(shared_module)
    module_2 = Module2(shared_module)
    example_inputs = (torch.randn(1),)
    module_1(*example_inputs)
    module_2(*example_inputs)

    ep1 = torch.export.export_for_training(module_1, example_inputs)
    ep2 = torch.export.export_for_training(module_2, example_inputs)

    edge_program_manager = executorch.exir.to_edge(
        {
            "forward1": ep1,
            "forward2": ep2,
        },
        compile_config=executorch.exir.EdgeCompileConfig(
            _check_ir_validity=False, _use_edge_ops=True
        ),
    )
    edge_program_manager = edge_program_manager.to_backend(ExecutorBackendPartitioner()).to_executorch()

with torch.no_grad():
    export()
