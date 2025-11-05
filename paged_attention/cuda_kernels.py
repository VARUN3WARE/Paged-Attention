"""
CUDA kernels wrapper for fused block access.

This module provides a runtime wrapper that will call a compiled CUDA extension
if available, and otherwise falls back to a pure-PyTorch implementation that
works on CPU/GPU (but less optimized).

To implement and compile a real fused kernel, see the template files under
`paged_attention/cuda/` and the README notes.
"""
from typing import List
import torch

_has_extension = False
_ext = None
try:
    # Attempt to import compiled extension (optional)
    from . import _fused_kernels as _ext  # type: ignore
    _has_extension = True
except Exception:
    _has_extension = False


def is_available() -> bool:
    """Return True if native fused CUDA kernels are available."""
    return _has_extension


def fused_block_matmul(Q: torch.Tensor, K_blocks: List[torch.Tensor], scale: float = 1.0) -> torch.Tensor:
    """
    Compute concatenated blockwise matmul(Q, concat(K_blocks)) in a fused manner.

    Args:
        Q: [batch, heads, q_len, head_dim]
        K_blocks: list of tensors, each [block_len, head_dim] (projected K)
        scale: scaling factor to apply to the matmul result

    Returns:
        scores: [batch, heads, q_len, total_kv_len]

    Notes:
        - If a compiled CUDA extension is available it will be used.
        - Otherwise this Python fallback concatenates and performs a single
          batched matmul which is correct but not as optimized.
    """
    if _has_extension:
        # The extension is expected to accept Q and a list of K block tensors
        # and return the concatenated scores tensor.
        return _ext.fused_block_matmul(Q, K_blocks, float(scale))

    # Fallback: concatenate K blocks and do batched matmul.
    if not K_blocks:
        # Empty cache -> return empty scores
        batch, heads, q_len, _ = Q.shape
        return torch.empty((batch, heads, q_len, 0), device=Q.device, dtype=Q.dtype)

    # Concatenate along sequence dim
    K_cat = torch.cat([k for k in K_blocks], dim=0)  # [kv_len, head_dim]

    # Prepare K for batched matmul: [batch, heads, head_dim, kv_len]
    batch, heads, q_len, head_dim = Q.shape
    kv_len = K_cat.shape[0]
    K_t = K_cat.t().unsqueeze(0).unsqueeze(0).expand(batch, heads, head_dim, kv_len)

    # Perform batched matmul via einsum for clarity
    scores = torch.einsum('bhqd,bhdk->bhqk', Q, K_t) * scale
    return scores
