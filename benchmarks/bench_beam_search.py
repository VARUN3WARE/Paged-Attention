"""
Benchmark beam search: COW memory savings.
"""

import torch
import numpy as np
from typing import List

import sys
import os
# Ensure project root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from paged_attention import (
    PagedKVCache, BlockAllocator,
    DecodingManager, ParallelSamplingManager,
    plot_beam_search_memory,
    print_stats_table
)


def benchmark_beam_search_memory(beam_widths: List[int] = [2, 4, 6, 8],
                                prompt_len: int = 50,
                                generation_len: int = 20,
                                block_size: int = 16,
                                hidden_dim: int = 512):
    """
    Benchmark memory usage for beam search with different widths.
    
    Args:
        beam_widths: List of beam widths to test
        prompt_len: Prompt length
        generation_len: Generation length
        block_size: Block size
        hidden_dim: Hidden dimension
    """
    print(f"\n{'='*60}")
    print(f"Beam Search Memory Benchmark")
    print(f"{'='*60}")
    print(f"Prompt Length: {prompt_len}")
    print(f"Generation Length: {generation_len}")
    print(f"Block Size: {block_size}, Hidden Dim: {hidden_dim}\n")
    
    naive_memory_mb = []
    paged_memory_mb = []
    
    for width in beam_widths:
        print(f"Testing beam width: {width}")
        
        # Naive approach: each beam gets full copy of cache
        bytes_per_token = hidden_dim * 2 * 4  # K + V, float32
        total_tokens = prompt_len + generation_len
        naive_memory = width * total_tokens * bytes_per_token
        naive_mb = naive_memory / (1024 * 1024)
        naive_memory_mb.append(naive_mb)
        
        # Paged approach with COW
        allocator = BlockAllocator(
            total_blocks=500,
            block_size=block_size,
            hidden_dim=hidden_dim
        )
        
        decoding_mgr = DecodingManager(allocator, block_size, hidden_dim)
        
        # Create prompt cache
        prompt_cache = PagedKVCache(block_size, hidden_dim, allocator)
        for i in range(prompt_len):
            k = torch.randn(hidden_dim)
            v = torch.randn(hidden_dim)
            prompt_cache.append_token_kv(k, v)
        
        # Initialize root beam
        root_beam = decoding_mgr.initialize_beam(prompt_cache, initial_token=0)
        
        # Fork beams
        beam_ids = [root_beam]
        for i in range(width - 1):
            new_id = decoding_mgr.fork_beam(root_beam, token_id=i+1, score=float(i))
            beam_ids.append(new_id)
        
        # Generate tokens (each beam diverges)
        for step in range(generation_len):
            for bid in beam_ids:
                k = torch.randn(hidden_dim)
                v = torch.randn(hidden_dim)
                decoding_mgr.append_token(bid, k, v)
        
        # Calculate actual memory used
        unique_blocks = set()
        for bid in beam_ids:
            beam = decoding_mgr.beams[bid]
            for entry in beam.kv_cache.block_table:
                unique_blocks.add(entry.phys_block_id)
        
        paged_memory = len(unique_blocks) * block_size * bytes_per_token
        paged_mb = paged_memory / (1024 * 1024)
        paged_memory_mb.append(paged_mb)
        
        # Get stats
        stats = decoding_mgr.get_stats()
        alloc_stats = allocator.get_stats()
        
        memory_saved = (1 - paged_memory / naive_memory) * 100
        
        print(f"  Naive Memory:  {naive_mb:.2f} MB")
        print(f"  Paged Memory:  {paged_mb:.2f} MB")
        print(f"  Memory Saved:  {memory_saved:.2f}%")
        print(f"  COW Copies:    {stats['num_cow_copies']}")
        print(f"  Total Forks:   {stats['num_forks']}\n")
        
        # Cleanup
        for bid in beam_ids:
            decoding_mgr.free_beam(bid)
    
    # Plot
    plot_beam_search_memory(
        beam_widths,
        naive_memory_mb,
        paged_memory_mb,
        title="Beam Search Memory Usage"
    )
    
    # Summary
    print(f"\n{'='*60}")
    print(f"Memory Savings Summary")
    print(f"{'='*60}")
    for i, width in enumerate(beam_widths):
        savings = (1 - paged_memory_mb[i] / naive_memory_mb[i]) * 100
        print(f"Width {width}: {savings:.2f}% saved ({naive_memory_mb[i]:.2f} MB → {paged_memory_mb[i]:.2f} MB)")
    print(f"{'='*60}\n")
    
    return {
        'beam_widths': beam_widths,
        'naive_memory_mb': naive_memory_mb,
        'paged_memory_mb': paged_memory_mb
    }


def benchmark_parallel_sampling(num_samples_list: List[int] = [2, 4, 6, 8],
                               prompt_len: int = 50,
                               generation_len: int = 20,
                               block_size: int = 16,
                               hidden_dim: int = 512):
    """
    Benchmark parallel sampling memory efficiency.
    
    Args:
        num_samples_list: List of sample counts to test
        prompt_len: Prompt length
        generation_len: Generation length per sample
        block_size: Block size
        hidden_dim: Hidden dimension
    """
    print(f"\n{'='*60}")
    print(f"Parallel Sampling Memory Benchmark")
    print(f"{'='*60}")
    print(f"Prompt Length: {prompt_len}")
    print(f"Generation Length: {generation_len}")
    print(f"Block Size: {block_size}, Hidden Dim: {hidden_dim}\n")
    
    for num_samples in num_samples_list:
        print(f"Testing {num_samples} parallel samples")
        
        bytes_per_token = hidden_dim * 2 * 4
        
        # Naive: each sample gets full copy
        naive_memory = num_samples * (prompt_len + generation_len) * bytes_per_token
        naive_mb = naive_memory / (1024 * 1024)
        
        # Paged: samples share prompt
        allocator = BlockAllocator(
            total_blocks=500,
            block_size=block_size,
            hidden_dim=hidden_dim
        )
        
        sampling_mgr = ParallelSamplingManager(allocator, block_size, hidden_dim)
        
        # Create prompt cache
        prompt_cache = PagedKVCache(block_size, hidden_dim, allocator)
        for i in range(prompt_len):
            k = torch.randn(hidden_dim)
            v = torch.randn(hidden_dim)
            prompt_cache.append_token_kv(k, v)
        
        # Create samples
        sample_ids = sampling_mgr.create_samples(prompt_cache, num_samples)
        
        # Generate tokens
        for step in range(generation_len):
            for sid in sample_ids:
                cache = sampling_mgr.get_sample_cache(sid)
                k = torch.randn(hidden_dim)
                v = torch.randn(hidden_dim)
                cache.cow_append(k, v)
        
        # Calculate memory
        unique_blocks = set()
        for sid in sample_ids:
            cache = sampling_mgr.get_sample_cache(sid)
            for entry in cache.block_table:
                unique_blocks.add(entry.phys_block_id)
        
        paged_memory = len(unique_blocks) * block_size * bytes_per_token
        paged_mb = paged_memory / (1024 * 1024)
        
        memory_saved = (1 - paged_memory / naive_memory) * 100
        
        print(f"  Naive Memory:  {naive_mb:.2f} MB")
        print(f"  Paged Memory:  {paged_mb:.2f} MB")
        print(f"  Memory Saved:  {memory_saved:.2f}%\n")
        
        # Cleanup
        sampling_mgr.free_all()
    
    print(f"{'='*60}\n")


def benchmark_cow_overhead(beam_width: int = 4,
                          prompt_len: int = 50,
                          generation_len: int = 30,
                          block_size: int = 16,
                          hidden_dim: int = 512):
    """
    Measure copy-on-write overhead.
    
    Args:
        beam_width: Beam width
        prompt_len: Prompt length
        generation_len: Generation length
        block_size: Block size
        hidden_dim: Hidden dimension
    """
    print(f"\n{'='*60}")
    print(f"Copy-on-Write Overhead Analysis")
    print(f"{'='*60}")
    print(f"Beam Width: {beam_width}, Prompt: {prompt_len}, Generation: {generation_len}\n")
    
    allocator = BlockAllocator(
        total_blocks=500,
        block_size=block_size,
        hidden_dim=hidden_dim
    )
    
    decoding_mgr = DecodingManager(allocator, block_size, hidden_dim)
    
    # Create prompt
    prompt_cache = PagedKVCache(block_size, hidden_dim, allocator)
    for i in range(prompt_len):
        k = torch.randn(hidden_dim)
        v = torch.randn(hidden_dim)
        prompt_cache.append_token_kv(k, v)
    
    # Initialize beams
    root_beam = decoding_mgr.initialize_beam(prompt_cache, initial_token=0)
    beam_ids = [root_beam]
    for i in range(beam_width - 1):
        new_id = decoding_mgr.fork_beam(root_beam, token_id=i+1, score=float(i))
        beam_ids.append(new_id)
    
    # Generate and track COW events
    cow_events = []
    initial_cow = allocator.num_cow_copies
    
    for step in range(generation_len):
        step_cow_start = allocator.num_cow_copies
        
        for bid in beam_ids:
            k = torch.randn(hidden_dim)
            v = torch.randn(hidden_dim)
            decoding_mgr.append_token(bid, k, v)
        
        step_cow = allocator.num_cow_copies - step_cow_start
        cow_events.append(step_cow)
    
    total_cow = allocator.num_cow_copies - initial_cow
    
    print(f"Total COW copies: {total_cow}")
    print(f"COW per step (avg): {np.mean(cow_events):.2f}")
    print(f"COW per step (max): {np.max(cow_events)}")
    print(f"COW per step (min): {np.min(cow_events)}")
    
    # Calculate overhead
    total_appends = beam_width * generation_len
    cow_rate = total_cow / total_appends
    print(f"\nCOW rate: {cow_rate:.2%} of appends triggered COW")
    
    print(f"{'='*60}\n")
    
    return {
        'total_cow_copies': total_cow,
        'cow_rate': cow_rate,
        'cow_per_step': cow_events
    }


if __name__ == '__main__':
    torch.manual_seed(42)
    np.random.seed(42)
    
    print("\n" + "="*60)
    print("PagedAttention Benchmarks - Beam Search & Sampling")
    print("="*60)
    
    # Run benchmarks
    print("\n[1/3] Beam Search Memory")
    beam_results = benchmark_beam_search_memory(
        beam_widths=[2, 4, 6, 8],
        prompt_len=50,
        generation_len=20,
        block_size=16,
        hidden_dim=512
    )
    
    print("\n[2/3] Parallel Sampling")
    benchmark_parallel_sampling(
        num_samples_list=[2, 4, 6, 8],
        prompt_len=50,
        generation_len=20,
        block_size=16,
        hidden_dim=512
    )
    
    print("\n[3/3] COW Overhead Analysis")
    cow_results = benchmark_cow_overhead(
        beam_width=4,
        prompt_len=50,
        generation_len=30,
        block_size=16,
        hidden_dim=512
    )
    
    print("\n" + "="*60)
    print("All benchmarks completed!")
    print("="*60 + "\n")