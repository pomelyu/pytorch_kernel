import torch
from torch import Tensor

__all__ = ["fused_muladd"]

def fused_muladd(a: Tensor, b: Tensor, c: float) -> Tensor:
    return torch.ops.pytorch_ops_stable.fused_muladd.default(a, b, c)

@torch.library.register_fake("pytorch_ops_stable::fused_muladd")
def _(a, b, c):
    torch._check(a.shape == b.shape)
    torch._check(a.dtype == torch.float)
    torch._check(b.dtype == torch.float)
    torch._check(a.device == b.device)
    return torch.empty_like(a)
