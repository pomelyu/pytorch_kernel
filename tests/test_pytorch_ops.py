import torch
from torch import Tensor
from torch.testing._internal.common_utils import TestCase


def reference_multiadd(a: Tensor, b: Tensor, c: float):
    return a * b + c

class TestFusedMulAdd(TestCase):
    def sample_inputs(self, device, *, requires_grad=False):
        def make_tensor(*size):
            return torch.randn(size, device=device, requires_grad=requires_grad)
        
        def make_nodiff_tensor(*size):
            return torch.randn(size, device=device, requires_grad=False)

        return [
            [make_tensor(3), make_tensor(3), 1],
            [make_tensor(20), make_tensor(20), 3.14],
            [make_tensor(20), make_nodiff_tensor(20), -123],
            [make_nodiff_tensor(2, 3), make_tensor(2, 3), -0.3],
        ]

    def test_correctness_cpu(self):
        self._test_correctness("cpu")

    def test_correctness_cuda(self):
        self._test_correctness("cuda")

    def _test_correctness(self, device):
        import pytorch_ops
        samples = self.sample_inputs(device)
        for args in samples:
            result = pytorch_ops.fused_muladd(*args)
            expected = reference_multiadd(*args)
            torch.testing.assert_close(result, expected)
