import torch
from torch import Tensor

from . import _C

__all__ = ["fused_muladd"]


def fused_muladd(a: Tensor, b: Tensor, c: float) -> Tensor:
    """Fused multiply-add: result[i] = a[i] * b[i] + c

    Supports both CPU and CUDA tensors (float32 only).
    """
    return _C.fused_muladd(a, b, c)
