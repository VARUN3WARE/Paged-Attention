"""
Correctness tests: Verify PagedAttention matches vanilla attention.
"""

import pytest
import torch
import torch.nn.functional as F

import sys
import os
# Ensure project root (one level up from tests/) is on sys.path so imports work
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from paged_attention import (
    PagedAttention, VanillaAttention,
    PagedKVCache, BlockAllocator
)


def test_paged_vs_vanilla_attention():
    """Test that paged attention produces same output as vanilla attention."""
    torch.manual_seed(42)
    
    # Config
    batch_size = 2
    seq_len = 32
    hidden_dim = 256
    num_heads = 8
    block_size = 16
    
    # Create models
    paged_attn = PagedAttention(hidden_dim, num_heads, block_size)
    vanilla_attn = VanillaAttention(hidden_dim, num_heads)
    
    # Share weights for fair comparison
    vanilla_attn.q_proj.weight.data = paged_attn.q_proj.weight.data.clone()
    vanilla_attn.k_proj.weight.data = paged_attn.k_proj.weight.data.clone()
    vanilla_attn.v_proj.weight.data = paged_attn.v_proj.weight.data.clone()
    vanilla_attn.out_proj.weight.data = paged_attn.out_proj.weight.data.clone()
    
    # Generate input
    x = torch.randn(batch_size, seq_len, hidden_dim)
    
    # Vanilla forward
    vanilla_output = vanilla_attn(x, x, x)
    
    # Paged forward: first populate cache
    allocator = BlockAllocator(total_blocks=32, block_size=block_size, 
                              hidden_dim=hidden_dim)
    kv_cache = PagedKVCache(block_size, hidden_dim, allocator)
    
    # Project K/V and populate cache
    with torch.no_grad():
        k = paged_attn.k_proj(x[0])  # [seq_len, hidden_dim]
        v = paged_attn.v_proj(x[0])
        
        for i in range(seq_len):
            kv_cache.append_token_kv(k[i], v[i])
    
    # Query on single token (for autoregressive)
    query = x[0:1, -1:, :]  # [1, 1, hidden_dim]
    paged_output = paged_attn.forward_paged(query, kv_cache)
    
    # Compare: get corresponding vanilla output
    vanilla_single = vanilla_attn(query, x[0:1], x[0:1])
    
    # Check close
    assert torch.allclose(paged_output, vanilla_single, atol=1e-4), \
        f"Outputs differ: max diff = {(paged_output - vanilla_single).abs().max()}"
    
    print("✓ Paged attention matches vanilla attention")


def test_blockwise_softmax():
    """Test that blockwise softmax computation is correct."""
    torch.manual_seed(42)
    
    # Generate logits
    seq_len = 48
    block_size = 16
    num_blocks = (seq_len + block_size - 1) // block_size
    
    logits = torch.randn(1, seq_len)
    
    # Vanilla softmax
    vanilla_softmax = F.softmax(logits, dim=-1)
    
    # Blockwise softmax (using running max/sum) - use tensors for numerics
    device = logits.device
    max_val = torch.tensor(float('-inf'), device=device)
    sum_exp = torch.tensor(0.0, device=device)
    block_results = []

    for i in range(num_blocks):
        start = i * block_size
        end = min(start + block_size, seq_len)
        block_logits = logits[:, start:end]

        block_max = block_logits.max()
        new_max = torch.maximum(max_val, block_max)

        # If the running max increases, scale previous stored block exps
        if not torch.isneginf(max_val).item():
            # scale previous block exps to new_max
            scale = torch.exp(max_val - new_max)
            if scale != 1.0:
                block_results = [br * scale for br in block_results]
            sum_exp = sum_exp * scale

        # Current block exp (with respect to new_max)
        block_exp = torch.exp(block_logits - new_max)
        sum_exp = sum_exp + block_exp.sum()

        block_results.append(block_exp)
        max_val = new_max

    # Normalize
    blockwise_softmax = torch.cat(block_results, dim=-1) / sum_exp
    
    assert torch.allclose(vanilla_softmax, blockwise_softmax, atol=1e-5), \
        f"Blockwise softmax incorrect: max diff = {(vanilla_softmax - blockwise_softmax).abs().max()}"
    
    print("✓ Blockwise softmax is correct")


def test_kv_cache_consistency():
    """Test that KV cache stores and retrieves values correctly."""
    torch.manual_seed(42)
    
    block_size = 8
    hidden_dim = 64
    num_tokens = 20
    
    allocator = BlockAllocator(total_blocks=10, block_size=block_size, 
                              hidden_dim=hidden_dim)
    cache = PagedKVCache(block_size, hidden_dim, allocator)
    
    # Store tokens
    keys = []
    values = []
    
    for i in range(num_tokens):
        k = torch.randn(hidden_dim)
        v = torch.randn(hidden_dim)
        keys.append(k)
        values.append(v)
        cache.append_token_kv(k, v)
    
    # Retrieve
    k_blocks, v_blocks = cache.read_blocks_for_attention()
    
    retrieved_keys = torch.cat(k_blocks, dim=0)
    retrieved_values = torch.cat(v_blocks, dim=0)
    
    expected_keys = torch.stack(keys)
    expected_values = torch.stack(values)
    
    assert torch.allclose(retrieved_keys, expected_keys, atol=1e-6), \
        "Retrieved keys don't match stored keys"
    assert torch.allclose(retrieved_values, expected_values, atol=1e-6), \
        "Retrieved values don't match stored values"
    
    print("✓ KV cache stores and retrieves correctly")


def test_multi_sequence_isolation():
    """Test that multiple sequences don't interfere with each other."""
    torch.manual_seed(42)
    
    block_size = 4
    hidden_dim = 32
    
    allocator = BlockAllocator(total_blocks=20, block_size=block_size, 
                              hidden_dim=hidden_dim)
    
    # Create two caches
    cache1 = PagedKVCache(block_size, hidden_dim, allocator, seq_id=1)
    cache2 = PagedKVCache(block_size, hidden_dim, allocator, seq_id=2)
    
    # Add different data to each
    for i in range(10):
        cache1.append_token_kv(torch.ones(hidden_dim) * 1.0, 
                              torch.ones(hidden_dim) * 1.0)
        cache2.append_token_kv(torch.ones(hidden_dim) * 2.0, 
                              torch.ones(hidden_dim) * 2.0)
    
    # Retrieve and check
    k1, v1 = cache1.get_all_keys_values()
    k2, v2 = cache2.get_all_keys_values()
    
    assert torch.allclose(k1, torch.ones_like(k1) * 1.0), "Cache 1 corrupted"
    assert torch.allclose(k2, torch.ones_like(k2) * 2.0), "Cache 2 corrupted"
    
    print("✓ Multiple sequences are isolated correctly")


if __name__ == '__main__':
    pytest.main([__file__, '-v'])