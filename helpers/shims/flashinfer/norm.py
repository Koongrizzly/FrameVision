"""
flashinfer.norm shim (pure PyTorch RMSNorm fallback).
"""
from __future__ import annotations
import torch

def rmsnorm(x: torch.Tensor, weight: torch.Tensor | None = None, eps: float = 1e-6, *args, **kwargs) -> torch.Tensor:
    orig_dtype = x.dtype
    y = x.float() if x.dtype in (torch.float16, torch.bfloat16) else x
    rms = torch.sqrt(torch.mean(y * y, dim=-1, keepdim=True) + eps)
    y = y / rms
    if weight is not None:
        y = y * weight
    return y.to(orig_dtype)
