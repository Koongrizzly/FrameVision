
import torch
import warnings

# FlashAttention is deliberately disabled in this build.
FLASH_ATTN_3_AVAILABLE = False
FLASH_ATTN_2_AVAILABLE = False

__all__ = [
    "flash_attention",
    "attention",
]


def _sdpa_fallback(
    q,
    k,
    v,
    q_lens=None,
    k_lens=None,
    dropout_p: float = 0.0,
    softmax_scale=None,
    q_scale=None,
    causal: bool = False,
    window_size=(-1, -1),
    deterministic: bool = False,
    dtype=torch.bfloat16,
):
    """
    Fallback attention implementation using PyTorch SDPA.

    Expected shapes (same as original WAN code):
        q: [B, Lq, Nq, C]
        k: [B, Lk, Nk, C]
        v: [B, Lk, Nk, C]
    """

    # In the original FA path q_lens / k_lens are used for varlen sequences.
    # Here we ignore them and warn once, similar to WAN's existing fallback.
    if q_lens is not None or k_lens is not None:
        warnings.warn(
            "Padding mask is disabled when using scaled_dot_product_attention. "
            "It can have a significant impact on performance.",
            RuntimeWarning,
        )

    # Optional scaling of q
    if q_scale is not None:
        q = q * q_scale

    # Convert to [B, heads, L, C] and cast to the desired dtype
    q = q.transpose(1, 2).to(dtype)
    k = k.transpose(1, 2).to(dtype)
    v = v.transpose(1, 2).to(dtype)

    # softmax_scale is not exposed directly in SDPA; WAN usually leaves it None.
    out = torch.nn.functional.scaled_dot_product_attention(
        q,
        k,
        v,
        attn_mask=None,
        is_causal=causal,
        dropout_p=dropout_p,
    )

    # Back to [B, L, heads, C]
    out = out.transpose(1, 2).contiguous()
    return out


def flash_attention(
    q,
    k,
    v,
    q_lens=None,
    k_lens=None,
    dropout_p: float = 0.0,
    softmax_scale=None,
    q_scale=None,
    causal: bool = False,
    window_size=(-1, -1),
    deterministic: bool = False,
    dtype=torch.bfloat16,
    version=None,
):
    """
    Drop-in replacement for the original flash-attn based function.
    We ignore 'version' and always use the SDPA fallback.
    """

    return _sdpa_fallback(
        q=q,
        k=k,
        v=v,
        q_lens=q_lens,
        k_lens=k_lens,
        dropout_p=dropout_p,
        softmax_scale=softmax_scale,
        q_scale=q_scale,
        causal=causal,
        window_size=window_size,
        deterministic=deterministic,
        dtype=dtype,
    )


def attention(
    q,
    k,
    v,
    q_lens=None,
    k_lens=None,
    dropout_p: float = 0.0,
    softmax_scale=None,
    q_scale=None,
    causal: bool = False,
    window_size=(-1, -1),
    deterministic: bool = False,
    dtype=torch.bfloat16,
    fa_version=None,
):
    """
    High-level wrapper used throughout WAN.
    In the original code this dispatched to flash-attn if available;
    here we *always* use the SDPA fallback.
    """

    return flash_attention(
        q=q,
        k=k,
        v=v,
        q_lens=q_lens,
        k_lens=k_lens,
        dropout_p=dropout_p,
        softmax_scale=softmax_scale,
        q_scale=q_scale,
        causal=causal,
        window_size=window_size,
        deterministic=deterministic,
        dtype=dtype,
        version=fa_version,
    )
