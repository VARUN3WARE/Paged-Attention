"""
Benchmark basic sampling: naive vs paged attention.
"""

import torch
import time
import numpy as np
from typing import List, Tuple

import sys
sys.path.insert(0, '../')

from paged_attention import (
    PagedAttention, VanillaAttention,
    PagedKVCache, BlockAllocator,
    generate_synthetic_workload,
    plot_memory_usage,
    compute_memory_metrics,
    print_stats_table
)


def benchmark_memory_usage(num_sequences: int = 10,
                          mean_seq_len: int = 100,
                          block_size: int = 16,
                          hidden_dim: int = 512):
    """
    Benchmark memory usage: naive vs paged.
    
    Args:
        num_sequences: Number of sequences to process
        mean_seq_len: Average sequence length
        block_size: Block size for paged attention
        hidden_dim: Model hidden dimension
    """
    print(f"\n{'='*60}")
    print(f"Memory Usage Benchmark")
    print(f"{'='*60}")
    print(f"Sequences: {num_sequences}, Avg Length: {mean_seq_len}")
    print(f"Block Size: {block_size}, Hidden Dim: {hidden_dim}\n")
    
    # Generate workload
    workload = generate_synthetic_workload(
        num_sequences, 
        mean_prompt_len=mean_seq_len,
        mean_output_len=0,  # Just prompts
        prompt_std=20
    )
    
    # Naive approach: contiguous allocation
    naive_memory = []
    naive_wasted = []
    timestamps = []
    
    total_allocated_naive = 0
    total_used_naive = 0
    
    for t, (seq_len, _) in enumerate(workload):
        # Allocate contiguous memory for full sequence
        allocated = seq_len * hidden_dim * 2 * 4  # K + V, float32
        used = seq_len * hidden_dim * 2 * 4
        
        total_allocated_naive += allocated
        total_used_naive += used
        
        naive_memory.append(total_allocated_naive)
        naive_wasted.append(total_allocated_naive - total_used_naive)
        timestamps.append(t)
    
    # Paged approach
    allocator = BlockAllocator(
        total_blocks=1000,
        block_size=block_size,
        hidden_dim=hidden_dim
    )
    
    caches = []
    paged_memory = []
    paged_wasted = []
    
    total_allocated_paged = 0
    total_used_paged = 0
    
    for t, (seq_len, _) in enumerate(workload):
        cache = PagedKVCache(block_size, hidden_dim, allocator)
        
        # Add tokens
        for i in range(seq_len):
            k = torch.randn(hidden_dim)
            v = torch.randn(hidden_dim)
            cache.append_token_kv(k, v)
        
        caches.append(cache)
        
        # Calculate memory
        used = seq_len * hidden_dim * 2 * 4
        allocated = cache.get_memory_usage()
        
        total_allocated_paged += allocated
        total_used_paged += used
        
        paged_memory.append(total_allocated_paged)
        paged_wasted.append(cache.get_wasted_memory())
        timestamps.append(t)
    
    # Compute metrics
    naive_metrics = compute_memory_metrics(total_allocated_naive, total_used_naive)
    paged_metrics = compute_memory_metrics(total_allocated_paged, total_used_paged)
    
    # Print results
    print("\n--- Naive (Contiguous) ---")
    print_stats_table(naive_metrics, "Naive Metrics")
    
    print("\n--- Paged (Blocks) ---")
    print_stats_table(paged_metrics, "Paged Metrics")
    
    # Calculate savings
    memory_saved = (1 - total_allocated_paged / total_allocated_naive) * 100
    print(f"\n{'='*60}")
    print(f"Memory Saved: {memory_saved:.2f}%")
    print(f"{'='*60}\n")
    
    # Plot
    plot_memory_usage(
        timestamps[:len(workload)],
        naive_memory[:len(workload)],
        paged_memory[:len(workload)],
        title=f"Memory Usage (Block Size={block_size})"
    )
    
    return {
        'naive_metrics': naive_metrics,
        'paged_metrics': paged_metrics,
        'memory_saved_pct': memory_saved
    }


