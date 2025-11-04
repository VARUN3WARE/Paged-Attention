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


def benchmark_throughput(batch_sizes: List[int] = [1, 2, 4, 8],
                        seq_len: int = 64,
                        hidden_dim: int = 512,
                        num_heads: int = 8,
                        block_size: int = 16,
                        num_iterations: int = 10):
    """
    Benchmark throughput: tokens/sec for different batch sizes.
    
    Args:
        batch_sizes: List of batch sizes to test
        seq_len: Sequence length
        hidden_dim: Model hidden dimension
        num_heads: Number of attention heads
        block_size: Block size for paged attention
        num_iterations: Number of iterations to average
    """
    print(f"\n{'='*60}")
    print(f"Throughput Benchmark")
    print(f"{'='*60}")
    print(f"Seq Length: {seq_len}, Hidden Dim: {hidden_dim}")
    print(f"Num Heads: {num_heads}, Block Size: {block_size}\n")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}\n")
    
    naive_throughputs = []
    paged_throughputs = []
    
    for batch_size in batch_sizes:
        print(f"Testing batch size: {batch_size}")
        
        # Vanilla attention
        vanilla_attn = VanillaAttention(hidden_dim, num_heads).to(device)
        
        # Paged attention
        paged_attn = PagedAttention(hidden_dim, num_heads, block_size).to(device)
        allocator = BlockAllocator(
            total_blocks=200,
            block_size=block_size,
            hidden_dim=hidden_dim,
            device=device
        )
        
        # Generate input
        x = torch.randn(batch_size, seq_len, hidden_dim, device=device)
        
        # Warmup
        for _ in range(3):
            _ = vanilla_attn(x, x, x)
        
        # Benchmark vanilla
        if device == 'cuda':
            torch.cuda.synchronize()
        
        start = time.time()
        for _ in range(num_iterations):
            _ = vanilla_attn(x, x, x)
            if device == 'cuda':
                torch.cuda.synchronize()
        
        vanilla_time = (time.time() - start) / num_iterations
        vanilla_throughput = (batch_size * seq_len) / vanilla_time
        naive_throughputs.append(vanilla_throughput)
        
        # Benchmark paged
        caches = []
        for b in range(batch_size):
            cache = PagedKVCache(block_size, hidden_dim, allocator)
            # Populate cache
            for i in range(seq_len):
                k = torch.randn(hidden_dim, device=device)
                v = torch.randn(hidden_dim, device=device)
                cache.append_token_kv(k, v)
            caches.append(cache)
        
        # Warmup
        for cache in caches:
            query = torch.randn(1, 1, hidden_dim, device=device)
            _ = paged_attn.forward_paged(query, cache)
        
        if device == 'cuda':
            torch.cuda.synchronize()
        
        start = time.time()
        for _ in range(num_iterations):
            for cache in caches:
                query = torch.randn(1, 1, hidden_dim, device=device)
                _ = paged_attn.forward_paged(query, cache)
            if device == 'cuda':
                torch.cuda.synchronize()
        
        paged_time = (time.time() - start) / num_iterations
        paged_throughput = (batch_size * seq_len) / paged_time
        paged_throughputs.append(paged_throughput)
        
        print(f"  Vanilla: {vanilla_throughput:.2f} tokens/sec")
        print(f"  Paged:   {paged_throughput:.2f} tokens/sec")
        print(f"  Speedup: {paged_throughput/vanilla_throughput:.2f}x\n")
        
        # Cleanup
        for cache in caches:
            cache.free_all()
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"Throughput Summary")
    print(f"{'='*60}")
    for i, bs in enumerate(batch_sizes):
        print(f"Batch {bs}: Naive={naive_throughputs[i]:.2f}, Paged={paged_throughputs[i]:.2f} tokens/sec")
    print(f"{'='*60}\n")
    
    return {
        'batch_sizes': batch_sizes,
        'naive_throughput': naive_throughputs,
        'paged_throughput': paged_throughputs
    }


def benchmark_fragmentation_vs_blocksize(block_sizes: List[int] = [8, 16, 32, 64],
                                        num_sequences: int = 50,
                                        mean_seq_len: int = 100,
                                        hidden_dim: int = 512):
    """
    Benchmark fragmentation for different block sizes.
    
    Args:
        block_sizes: List of block sizes to test
        num_sequences: Number of sequences
        mean_seq_len: Average sequence length
        hidden_dim: Hidden dimension
    """
    print(f"\n{'='*60}")
    print(f"Fragmentation vs Block Size")
    print(f"{'='*60}\n")
    
    workload = generate_synthetic_workload(
        num_sequences,
        mean_prompt_len=mean_seq_len,
        mean_output_len=0,
        prompt_std=20
    )
    
    results = []
    
    for block_size in block_sizes:
        print(f"Testing block size: {block_size}")
        
        allocator = BlockAllocator(
            total_blocks=1000,
            block_size=block_size,
            hidden_dim=hidden_dim
        )
        
        total_allocated = 0
        total_used = 0
        
        for seq_len, _ in workload:
            cache = PagedKVCache(block_size, hidden_dim, allocator)
            
            for i in range(seq_len):
                k = torch.randn(hidden_dim)
                v = torch.randn(hidden_dim)
                cache.append_token_kv(k, v)
            
            total_allocated += cache.get_memory_usage()
            total_used += seq_len * hidden_dim * 2 * 4
            
            cache.free_all()
        
        metrics = compute_memory_metrics(total_allocated, total_used)
        fragmentation = metrics['fragmentation']
        
        results.append(fragmentation)
        print(f"  Fragmentation: {fragmentation:.2f}%\n")
    
    print(f"\n{'='*60}")
    print(f"Fragmentation Summary")
    print(f"{'='*60}")
    for i, bs in enumerate(block_sizes):
        print(f"Block Size {bs}: {results[i]:.2f}% fragmentation")
    print(f"{'='*60}\n")
    
    return {
        'block_sizes': block_sizes,
        'fragmentation': results
    }


if __name__ == '__main__':
    torch.manual_seed(42)
    np.random.seed(42)
    
    print("\n" + "="*60)
    print("PagedAttention Benchmarks - Basic Sampling")
    print("="*60)
    
    # Run benchmarks
    print("\n[1/3] Memory Usage Benchmark")
    mem_results = benchmark_memory_usage(
        num_sequences=20,
        mean_seq_len=100,
        block_size=16,
        hidden_dim=512
    )
    
    print("\n[2/3] Throughput Benchmark")
    throughput_results = benchmark_throughput(
        batch_sizes=[1, 2, 4, 8],
        seq_len=64,
        hidden_dim=512,
        num_heads=8,
        block_size=16,
        num_iterations=10
    )
    
    print("\n[3/3] Fragmentation vs Block Size")
    frag_results = benchmark_fragmentation_vs_blocksize(
        block_sizes=[8, 16, 32, 64],
        num_sequences=50,
        mean_seq_len=100,
        hidden_dim=512
    )
    
    print("\n" + "="*60)
    print("All benchmarks completed!")
    print("="*60 + "\n")