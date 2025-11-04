"""
Tests for copy-on-write functionality.
"""

import pytest
import torch

import sys
import os
# Ensure project root (one level up from tests/) is on sys.path so imports work
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from paged_attention import (
    PagedKVCache, BlockAllocator, 
    DecodingManager, ParallelSamplingManager
)


def test_cache_fork():
    """Test that forking creates shared blocks."""
    allocator = BlockAllocator(total_blocks=20, block_size=8, hidden_dim=64)
    cache = PagedKVCache(block_size=8, hidden_dim=64, allocator=allocator)
    
    # Add some tokens
    for i in range(10):
        cache.append_token_kv(torch.randn(64), torch.randn(64))
    
    # Fork
    forked_cache = cache.fork()
    
    # Check blocks are shared
    for i, entry in enumerate(cache.block_table):
        forked_entry = forked_cache.block_table[i]
        assert entry.phys_block_id == forked_entry.phys_block_id
        
        # Check refcount increased
        block = allocator.get_block(entry.phys_block_id)
        assert block.refcount >= 2
    
    print("✓ Cache forking creates shared blocks")


def test_cow_on_write():
    """Test that writing to shared block triggers copy."""
    allocator = BlockAllocator(total_blocks=20, block_size=8, hidden_dim=64)
    cache1 = PagedKVCache(block_size=8, hidden_dim=64, allocator=allocator)
    
    # Add tokens (not filling last block completely)
    for i in range(10):
        cache1.append_token_kv(torch.ones(64), torch.ones(64))
    
    # Fork
    cache2 = cache1.fork()
    
    # Get last block ID before write
    last_block_id_cache1 = cache1.block_table[-1].phys_block_id
    last_block_id_cache2 = cache2.block_table[-1].phys_block_id
    assert last_block_id_cache1 == last_block_id_cache2  # Shared
    
    # Write to cache2 (should trigger COW)
    cache2.cow_append(torch.ones(64) * 2, torch.ones(64) * 2)
    
    # Check blocks are now different
    new_last_block_cache2 = cache2.block_table[-1].phys_block_id
    assert new_last_block_cache2 != last_block_id_cache1
    
    # Check cache1 unchanged
    k1, v1 = cache1.get_all_keys_values()
    assert torch.allclose(k1, torch.ones_like(k1))
    
    print("✓ Copy-on-write triggers correctly")


def test_beam_search_sharing():
    """Test beam search memory sharing."""
    allocator = BlockAllocator(total_blocks=50, block_size=8, hidden_dim=64)
    decoding_mgr = DecodingManager(allocator, block_size=8, hidden_dim=64)
    
    # Initialize with prompt
    prompt_cache = PagedKVCache(block_size=8, hidden_dim=64, allocator=allocator)
    for i in range(20):
        prompt_cache.append_token_kv(torch.randn(64), torch.randn(64))
    
    # Initialize beam
    beam_id = decoding_mgr.initialize_beam(prompt_cache, initial_token=0)
    
    # Fork multiple beams
    beam_ids = [beam_id]
    for i in range(3):
        new_id = decoding_mgr.fork_beam(beam_id, token_id=i+1, score=float(i))
        beam_ids.append(new_id)
    
    # All beams should share prompt blocks
    beam0_blocks = set(e.phys_block_id for e in decoding_mgr.beams[beam_ids[0]].kv_cache.block_table)
    for bid in beam_ids[1:]:
        beam_blocks = set(e.phys_block_id for e in decoding_mgr.beams[bid].kv_cache.block_table)
        # Should have significant overlap
        overlap = beam0_blocks & beam_blocks
        assert len(overlap) > 0, "Beams should share blocks"
    
    print("✓ Beam search shares memory correctly")


def test_parallel_sampling():
    """Test parallel sampling memory sharing."""
    allocator = BlockAllocator(total_blocks=50, block_size=8, hidden_dim=64)
    sampling_mgr = ParallelSamplingManager(allocator, block_size=8, hidden_dim=64)
    
    # Create prompt cache
    prompt_cache = PagedKVCache(block_size=8, hidden_dim=64, allocator=allocator)
    for i in range(16):
        prompt_cache.append_token_kv(torch.randn(64), torch.randn(64))
    
    # Create multiple samples
    num_samples = 4
    sample_ids = sampling_mgr.create_samples(prompt_cache, num_samples)
    
    assert len(sample_ids) == num_samples
    
    # All samples should share prompt blocks
    prompt_block_ids = set(e.phys_block_id for e in prompt_cache.block_table)
    
    for sid in sample_ids:
        sample_cache = sampling_mgr.get_sample_cache(sid)
        sample_block_ids = set(e.phys_block_id for e in sample_cache.block_table[:len(prompt_cache.block_table)])
        assert prompt_block_ids == sample_block_ids, "Samples should share prompt blocks"
    
    print("✓ Parallel sampling shares memory correctly")


if __name__ == '__main__':
    pytest.main([__file__, '-v'])