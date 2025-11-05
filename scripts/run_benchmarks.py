"""Run all benchmarks and generate comprehensive report.

This script accepts a few CLI flags to support faster/headless runs for CI:
  --outdir DIR     Save plots to DIR instead of showing them interactively
  --headless       Use a non-interactive Matplotlib backend
  --fast           Run a much smaller, faster workload (for smoke tests)
"""

import argparse
import sys
import os

# Parse args early so we can set a headless backend before matplotlib imports
parser = argparse.ArgumentParser()
parser.add_argument('--outdir', type=str, default=None,
                    help='Directory to save plots (if provided)')
parser.add_argument('--headless', action='store_true',
                    help='Run with non-interactive Matplotlib backend')
parser.add_argument('--fast', action='store_true',
                    help='Run a fast, reduced benchmark workload (smoke test)')
args = parser.parse_args()

if args.headless:
    # Set MPL backend environment variable before matplotlib is imported anywhere
    os.environ.setdefault('MPLBACKEND', 'Agg')

# Ensure project root and benchmarks dir are on sys.path (file-relative)
proj_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, proj_root)

import torch
import numpy as np
from paged_attention import (
    plot_throughput,
    plot_fragmentation,
    plot_swap_vs_recompute,
    plot_beam_search_memory
)

# Import benchmark functions (make benchmarks dir importable)
bench_dir = os.path.join(proj_root, 'benchmarks')
sys.path.insert(0, bench_dir)
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

    # Use a much smaller workload when --fast is passed
    if args.fast:
        mem_num_sequences = 4
        mem_mean_seq_len = 32
        throughput_iters = 2
        frag_num_sequences = 8
        fragment_block_sizes = [8, 16]
    else:
        mem_num_sequences = 20
        mem_mean_seq_len = 100
        throughput_iters = 10
        frag_num_sequences = 50
        fragment_block_sizes = [8, 16, 32, 64]

    print("[1/6] Memory Usage Benchmark...")
    results['memory'] = benchmark_memory_usage(
        num_sequences=mem_num_sequences,
        mean_seq_len=mem_mean_seq_len,
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
        num_iterations=throughput_iters
    )

    print("\n[3/6] Fragmentation vs Block Size...")
    results['fragmentation'] = benchmark_fragmentation_vs_blocksize(
        block_sizes=fragment_block_sizes,
        num_sequences=frag_num_sequences,
        mean_seq_len=mem_mean_seq_len,
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
    outdir = args.outdir
    save_prefix = None
    if outdir:
        os.makedirs(outdir, exist_ok=True)
        save_prefix = os.path.join(outdir, 'bench_')

    plot_throughput(
        throughput_data['batch_sizes'],
        throughput_data['naive_throughput'],
        throughput_data['paged_throughput'],
        title="Throughput: Naive vs Paged",
        save_path=(save_prefix + 'throughput.png') if save_prefix else None
    )
    
    # Plot fragmentation
    plot_fragmentation(
        frag_data['block_sizes'],
        frag_data['fragmentation'],
        title="Internal Fragmentation by Block Size",
        save_path=(save_prefix + 'fragmentation.png') if save_prefix else None
    )
    
    print("\n" + "="*70)
    print("All benchmarks completed successfully! ✓")
    print("="*70 + "\n")
    
    return results


if __name__ == '__main__':
    results = main()