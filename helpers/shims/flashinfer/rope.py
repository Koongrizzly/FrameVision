"""
flashinfer.rope shim (pure PyTorch fallback).

MonarchRT imports:
    from flashinfer.rope import apply_rope_with_cos_sin_cache_inplace

The real flashinfer provides fused kernels. This fallback is slower but functional.
"""
from __future__ import annotations
import torch

def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x_even = x[..., 0::2]
    x_odd  = x[..., 1::2]
    out = torch.empty_like(x)
    out[..., 0::2] = -x_odd
    out[..., 1::2] = x_even
    return out

def apply_rope_with_cos_sin_cache_inplace(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    *args, **kwargs
) -> torch.Tensor:
    if x is None:
        return x
    orig_dtype = x.dtype
    y = x.float() if x.dtype in (torch.float16, torch.bfloat16) else x

    cos_t = cos.to(device=y.device, dtype=y.dtype)
    sin_t = sin.to(device=y.device, dtype=y.dtype)

    d = y.shape[-1]
    if cos_t.shape[-1] * 2 == d:
        cos_full = torch.empty((*cos_t.shape[:-1], d), device=y.device, dtype=y.dtype)
        sin_full = torch.empty((*sin_t.shape[:-1], d), device=y.device, dtype=y.dtype)
        cos_full[..., 0::2] = cos_t
        cos_full[..., 1::2] = cos_t
        sin_full[..., 0::2] = sin_t
        sin_full[..., 1::2] = sin_t
        cos_t, sin_t = cos_full, sin_full

    while cos_t.ndim < y.ndim:
        cos_t = cos_t.unsqueeze(0)
        sin_t = sin_t.unsqueeze(0)

    y_rope = (y * cos_t) + (_rotate_half(y) * sin_t)
    if y_rope.dtype != orig_dtype:
        y_rope = y_rope.to(orig_dtype)
    x.copy_(y_rope)
    return x
