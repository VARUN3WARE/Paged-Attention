"""
Run all benchmarks and generate comprehensive report.
"""

import sys
sys.path.insert(0, '../')

import torch
import numpy as np
from paged_attention import (
    plot_throughput,
    plot_fragmentation,
    plot_swap_vs_recompute
)

# Import benchmark functions
sys.path.insert(0, '../benchmarks')
from bench_basic_sampling import (
    benchmark_memory_usage,
    benchmark_throughput,
    benchmark_fragmentation_vs_blocksize
)
from bench_beam_search import (
    benchmark_beam_search_memory,
    benchmark_parallel_sampling,
    benchmark_cow_overhead
)


def main():
    """Run all benchmarks."""
    print("\n" + "="*70)
    print(" "*20 + "PagedAttention Benchmarks")
    print("="*70 + "\n")
    
    torch.manual_seed(42)
    np.random.seed(42)
    
    results = {}
    
    # ===== Part 1: Basic Benchmarks =====
    print("\n" + "="*70)
    print("PART 1: Basic Performance Benchmarks")
    print("="*70 + "\n")
    
    print("[1/6] Memory Usage Benchmark...")
    results['memory'] = benchmark_memory_usage(
        num_sequences=20,
        mean_seq_len=100,
        block_size=16,
        hidden_dim=512
    )
    
    print("\n[2/6] Throughput Benchmark...")
    results['throughput'] = benchmark_throughput(
        batch_sizes=[1, 2, 4, 8],
        seq_len=64,
        hidden_dim=512,
        num_heads=8,
        block_size=16,
        num_iterations=10
    )
    
    print("\n[3/6] Fragmentation vs Block Size...")
    results['fragmentation'] = benchmark_fragmentation_vs_blocksize(
        block_sizes=[8, 16, 32, 64],
        num_sequences=50,
        mean_seq_len=100,
        hidden_dim=512
    )
    
    # ===== Part 2: Advanced Benchmarks =====
    print("\n" + "="*70)
    print("PART 2: Beam Search & Sampling Benchmarks")
    print("="*70 + "\n")
    
    print("[4/6] Beam Search Memory...")
    results['beam_search'] = benchmark_beam_search_memory(
        beam_widths=[2, 4, 6, 8],
        prompt_len=50,
        generation_len=20,
        block_size=16,
        hidden_dim=512
    )
    
    print("\n[5/6] Parallel Sampling...")
    benchmark_parallel_sampling(
        num_samples_list=[2, 4, 6, 8],
        prompt_len=50,
        generation_len=20,
        block_size=16,
        hidden_dim=512
    )
    
    print("\n[6/6] COW Overhead Analysis...")
    results['cow'] = benchmark_cow_overhead(
        beam_width=4,
        prompt_len=50,
        generation_len=30,
        block_size=16,
        hidden_dim=512
    )
    
    # ===== Summary Report =====
    print("\n" + "="*70)
    print("BENCHMARK SUMMARY REPORT")
    print("="*70 + "\n")
    
    print("1. Memory Efficiency:")
    mem_saved = results['memory']['memory_saved_pct']
    print(f"   → PagedAttention saves {mem_saved:.1f}% memory vs naive allocation")
    print(f"   → Utilization: {results['memory']['paged_metrics']['utilization']:.1f}%")
    print(f"   → Fragmentation: {results['memory']['paged_metrics']['fragmentation']:.1f}%")
    
    print("\n2. Throughput:")
    throughput_data = results['throughput']
    avg_speedup = np.mean([p/n for p, n in zip(throughput_data['paged_throughput'], 
                                                 throughput_data['naive_throughput'])])
    print(f"   → Average speedup: {avg_speedup:.2f}x")
    print(f"   → Best speedup at batch size {throughput_data['batch_sizes'][-1]}: "
          f"{throughput_data['paged_throughput'][-1]/throughput_data['naive_throughput'][-1]:.2f}x")
    
    print("\n3. Fragmentation:")
    frag_data = results['fragmentation']
    best_block_size = frag_data['block_sizes'][np.argmin(frag_data['fragmentation'])]
    print(f"   → Optimal block size: {best_block_size}")
    print(f"   → Min fragmentation: {min(frag_data['fragmentation']):.2f}%")
    print(f"   → Max fragmentation: {max(frag_data['fragmentation']):.2f}%")
    
    print("\n4. Beam Search:")
    beam_data = results['beam_search']
    avg_beam_savings = np.mean([
        (1 - p/n) * 100 
        for p, n in zip(beam_data['paged_memory_mb'], beam_data['naive_memory_mb'])
    ])
    print(f"   → Average memory savings: {avg_beam_savings:.1f}%")
    print(f"   → Best savings at width {beam_data['beam_widths'][-1]}: "
          f"{(1 - beam_data['paged_memory_mb'][-1]/beam_data['naive_memory_mb'][-1])*100:.1f}%")
    
    print("\n5. Copy-on-Write:")
    cow_data = results['cow']
    print(f"   → COW rate: {cow_data['cow_rate']:.2%} of writes trigger copy")
    print(f"   → Total COW copies: {cow_data['total_cow_copies']}")
    
    print("\n" + "="*70)
    print("Key Findings:")
    print("="*70)
    print(f"✓ Memory savings: {mem_saved:.1f}% (up to {avg_beam_savings:.1f}% with beam search)")
    print(f"✓ Throughput improvement: {avg_speedup:.2f}x average")
    print(f"✓ Optimal block size: {best_block_size} tokens")
    print(f"✓ COW overhead: {cow_data['cow_rate']:.2%} (minimal)")
    print("="*70 + "\n")
    
    # Additional plots
    print("Generating additional plots...\n")
    
    # Plot throughput comparison
    plot_throughput(
        throughput_data['batch_sizes'],
        throughput_data['naive_throughput'],
        throughput_data['paged_throughput'],
        title="Throughput: Naive vs Paged"
    )
    
    # Plot fragmentation
    plot_fragmentation(
        frag_data['block_sizes'],
        frag_data['fragmentation'],
        title="Internal Fragmentation by Block Size"
    )
    
    print("\n" + "="*70)
    print("All benchmarks completed successfully! ✓")
    print("="*70 + "\n")
    
    return results


if __name__ == '__main__':
    results = main()