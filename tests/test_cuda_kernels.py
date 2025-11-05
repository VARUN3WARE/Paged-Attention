import sys
import os
import torch

# Ensure project root on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from paged_attention import cuda_kernels


def test_fused_block_matmul_shapes():
    # Small shapes to validate fallback logic works on CPU
    batch = 2
    heads = 2
    q_len = 3
    head_dim = 4

    Q = torch.randn(batch, heads, q_len, head_dim)

    # Create two K blocks
    K1 = torch.randn(5, head_dim)
    K2 = torch.randn(7, head_dim)
    scores = cuda_kernels.fused_block_matmul(Q, [K1, K2], scale=1.0)

    assert scores.shape == (batch, heads, q_len, 12)


def test_is_available_flag():
    # No assertion on value; just call to ensure function exists
    _ = cuda_kernels.is_available()
