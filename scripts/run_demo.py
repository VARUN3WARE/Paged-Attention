"""
Demo script: Shows basic PagedAttention functionality and correctness.
"""

import torch
import sys
import os
# Ensure project root is on sys.path (file-relative) so imports work when running the script
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from paged_attention import (
    PagedAttention, VanillaAttention,
    PagedKVCache, BlockAllocator,
    print_stats_table
)


def demo_correctness():
    """Demonstrate that paged attention matches vanilla attention."""
    print("\n" + "="*60)
    print("Demo 1: Correctness Test")
    print("="*60 + "\n")
    
    torch.manual_seed(42)
    
    # Config
    batch_size = 1
    seq_len = 32
    hidden_dim = 256
    num_heads = 8
    block_size = 16
    
    print(f"Configuration:")
    print(f"  Sequence Length: {seq_len}")
    print(f"  Hidden Dim: {hidden_dim}")
    print(f"  Num Heads: {num_heads}")
    print(f"  Block Size: {block_size}\n")
    
    # Create models
    paged_attn = PagedAttention(hidden_dim, num_heads, block_size)
    vanilla_attn = VanillaAttention(hidden_dim, num_heads)
    
    # Share weights
    vanilla_attn.q_proj.weight.data = paged_attn.q_proj.weight.data.clone()
    vanilla_attn.k_proj.weight.data = paged_attn.k_proj.weight.data.clone()
    vanilla_attn.v_proj.weight.data = paged_attn.v_proj.weight.data.clone()
    vanilla_attn.out_proj.weight.data = paged_attn.out_proj.weight.data.clone()
    
    # Generate input
    x = torch.randn(batch_size, seq_len, hidden_dim)
    
    # Vanilla forward
    print("Running vanilla attention...")
    vanilla_output = vanilla_attn(x, x, x)
    
    # Paged forward
    print("Running paged attention...")
    allocator = BlockAllocator(total_blocks=32, block_size=block_size, 
                              hidden_dim=hidden_dim)
    kv_cache = PagedKVCache(block_size, hidden_dim, allocator)
    
    # Populate cache
    with torch.no_grad():
        k = paged_attn.k_proj(x[0])
        v = paged_attn.v_proj(x[0])
        
        for i in range(seq_len):
            kv_cache.append_token_kv(k[i], v[i])
    
    # Query on last token
    query = x[0:1, -1:, :]
    paged_output = paged_attn.forward_paged(query, kv_cache)
    
    # Compare
    vanilla_single = vanilla_attn(query, x[0:1], x[0:1])
    
    max_diff = (paged_output - vanilla_single).abs().max().item()
    mean_diff = (paged_output - vanilla_single).abs().mean().item()
    
    print(f"\nResults:")
    print(f"  Max Difference: {max_diff:.2e}")
    print(f"  Mean Difference: {mean_diff:.2e}")
    
    if max_diff < 1e-4:
        print(f"  ✓ PASSED: Outputs match within tolerance!")
    else:
        print(f"  ✗ FAILED: Outputs differ significantly!")
    
    print("\n" + "="*60 + "\n")


def demo_memory_efficiency():
    """Demonstrate memory efficiency of paged approach."""
    print("\n" + "="*60)
    print("Demo 2: Memory Efficiency")
    print("="*60 + "\n")
    
    torch.manual_seed(42)
    
    # Simulate varying sequence lengths
    seq_lengths = [10, 25, 33, 47, 62, 78, 91, 105]
    block_size = 16
    hidden_dim = 512
    
    print(f"Block Size: {block_size}")
    print(f"Hidden Dim: {hidden_dim}")
    print(f"Testing {len(seq_lengths)} sequences with varying lengths\n")


    allocator = BlockAllocator(total_blocks=100, block_size=block_size, 
                              hidden_dim=hidden_dim)
    
    # Naive memory calculation
    naive_total = 0
    paged_total = 0
    
    caches = []
    
    print(f"{'Seq Length':<12} {'Naive (KB)':<15} {'Paged (KB)':<15} {'Waste (%)':<12}")
    print("-" * 60)
    
    for seq_len in seq_lengths:
        # Naive: contiguous allocation
        bytes_per_token = hidden_dim * 2 * 4  # K + V, float32
        naive_mem = seq_len * bytes_per_token
        naive_total += naive_mem
        
        # Paged: block-based allocation
        cache = PagedKVCache(block_size, hidden_dim, allocator)
        
        for i in range(seq_len):
            k = torch.randn(hidden_dim)
            v = torch.randn(hidden_dim)
            cache.append_token_kv(k, v)
        
        paged_mem = cache.get_memory_usage()
        wasted_mem = cache.get_wasted_memory()
        waste_pct = (wasted_mem / paged_mem * 100) if paged_mem > 0 else 0
        
        paged_total += paged_mem
        caches.append(cache)
        
        print(f"{seq_len:<12} {naive_mem/1024:<15.2f} {paged_mem/1024:<15.2f} {waste_pct:<12.2f}")
    
    print("-" * 60)
    print(f"{'TOTAL':<12} {naive_total/1024:<15.2f} {paged_total/1024:<15.2f}")
    
    memory_saved = (1 - paged_total / naive_total) * 100
    print(f"\nMemory saved with paging: {memory_saved:.2f}%")
    
    # Allocator stats
    stats = allocator.get_stats()
    print_stats_table(stats, "Allocator Statistics")
    
    # Cleanup
    for cache in caches:
        cache.free_all()
    
    print("="*60 + "\n")


def demo_copy_on_write():
    """Demonstrate copy-on-write for beam search."""
    print("\n" + "="*60)
    print("Demo 3: Copy-on-Write (Beam Search)")
    print("="*60 + "\n")
    
    torch.manual_seed(42)
    
    from paged_attention import DecodingManager
    
    block_size = 8
    hidden_dim = 256
    prompt_len = 20
    beam_width = 4
    
    print(f"Configuration:")
    print(f"  Prompt Length: {prompt_len}")
    print(f"  Beam Width: {beam_width}")
    print(f"  Block Size: {block_size}\n")
    
    allocator = BlockAllocator(total_blocks=100, block_size=block_size, 
                              hidden_dim=hidden_dim)
    decoding_mgr = DecodingManager(allocator, block_size, hidden_dim)
    
    # Create prompt cache
    print("Creating prompt cache...")
    prompt_cache = PagedKVCache(block_size, hidden_dim, allocator)
    for i in range(prompt_len):
        k = torch.randn(hidden_dim)
        v = torch.randn(hidden_dim)
        prompt_cache.append_token_kv(k, v)
    
    prompt_blocks = [e.phys_block_id for e in prompt_cache.block_table]
    print(f"Prompt uses {len(prompt_blocks)} blocks: {prompt_blocks}")
    
    # Initialize root beam
    root_beam = decoding_mgr.initialize_beam(prompt_cache, initial_token=0)
    print(f"\nRoot beam initialized (ID: {root_beam})")
    
    # Fork beams
    print(f"\nForking {beam_width - 1} beams...")
    beam_ids = [root_beam]
    for i in range(beam_width - 1):
        new_id = decoding_mgr.fork_beam(root_beam, token_id=i+1, score=float(i))
        beam_ids.append(new_id)
        print(f"  Forked beam {new_id}")
    
    # Check sharing
    print(f"\nChecking block sharing after fork...")
    for bid in beam_ids:
        beam = decoding_mgr.beams[bid]
        beam_blocks = [e.phys_block_id for e in beam.kv_cache.block_table]
        print(f"  Beam {bid}: blocks {beam_blocks}")
        
        # Check refcounts
        for phys_id in beam_blocks[:len(prompt_blocks)]:
            block = allocator.get_block(phys_id)
            print(f"    Block {phys_id}: refcount = {block.refcount}")
    
    # Generate tokens (triggers COW)
    print(f"\nGenerating tokens (this triggers COW)...")
    initial_cow = allocator.num_cow_copies
    
    for step in range(5):
        print(f"\nStep {step + 1}:")
        for bid in beam_ids:
            k = torch.randn(hidden_dim)
            v = torch.randn(hidden_dim)
            decoding_mgr.append_token(bid, k, v)
        
        cow_this_step = allocator.num_cow_copies - initial_cow
        print(f"  COW copies so far: {cow_this_step}")
    
    total_cow = allocator.num_cow_copies - initial_cow
    print(f"\nTotal COW copies: {total_cow}")
    print(f"COW rate: {total_cow / (beam_width * 5):.2%} of appends")
    
    # Final memory usage
    unique_blocks = set()
    for bid in beam_ids:
        beam = decoding_mgr.beams[bid]
        for entry in beam.kv_cache.block_table:
            unique_blocks.add(entry.phys_block_id)
    
    print(f"\nFinal statistics:")
    print(f"  Total unique blocks used: {len(unique_blocks)}")
    print(f"  Naive would use: {beam_width * ((prompt_len + 5 + block_size - 1) // block_size)} blocks")
    print(f"  Memory saved: {(1 - len(unique_blocks) / (beam_width * ((prompt_len + 5 + block_size - 1) // block_size))) * 100:.1f}%")
    
    # Cleanup
    for bid in beam_ids:
        decoding_mgr.free_beam(bid)
    
    print("\n" + "="*60 + "\n")


def demo_swap_recompute():
    """Demonstrate swap vs recompute tradeoff."""
    print("\n" + "="*60)
    print("Demo 4: Swap vs Recompute")
    print("="*60 + "\n")
    
    from paged_attention import SwapManager, RecomputeManager
    
    block_size = 16
    hidden_dim = 512
    num_blocks = 10
    
    allocator = BlockAllocator(total_blocks=50, block_size=block_size, 
                              hidden_dim=hidden_dim, device='cpu')
    
    # Create managers
    swap_mgr = SwapManager(allocator, gpu_to_cpu_bandwidth_gbps=25.0)
    recompute_mgr = RecomputeManager(compute_time_per_token_ms=0.1)
    
    print(f"Configuration:")
    print(f"  Block Size: {block_size}")
    print(f"  Hidden Dim: {hidden_dim}")
    print(f"  GPU-CPU Bandwidth: 25 GB/s")
    print(f"  Recompute Time: 0.1 ms/token\n")
    
    # Allocate some blocks
    block_ids = [allocator.allocate() for _ in range(num_blocks)]
    
    # Simulate swap
    print("Simulating swap operations...")
    total_swap_time = 0
    for bid in block_ids[:5]:
        swap_time = swap_mgr.swap_out_block(bid)
        total_swap_time += swap_time
        print(f"  Swapped out block {bid}: {swap_time*1000:.4f} ms")
    
    print(f"Total swap out time: {total_swap_time*1000:.4f} ms\n")
    
    # Simulate recompute
    print("Simulating recompute operations...")
    total_recompute_time = 0
    for bid in block_ids[5:]:
        k, v, recomp_time = recompute_mgr.recompute_block(bid, block_size, hidden_dim)
        total_recompute_time += recomp_time
        print(f"  Recomputed block {bid}: {recomp_time*1000:.4f} ms")
    
    print(f"Total recompute time: {total_recompute_time*1000:.4f} ms\n")
    
    # Compare
    print("Comparison:")
    print(f"  Swap: {total_swap_time*1000:.4f} ms for {len(block_ids[:5])} blocks")
    print(f"  Recompute: {total_recompute_time*1000:.4f} ms for {len(block_ids[5:])} blocks")
    print(f"  Swap avg: {total_swap_time*1000/5:.4f} ms/block")
    print(f"  Recompute avg: {total_recompute_time*1000/5:.4f} ms/block")
    
    if total_swap_time < total_recompute_time:
        print(f"\n  → Swap is faster by {(total_recompute_time/total_swap_time):.2f}x")
    else:
        print(f"\n  → Recompute is faster by {(total_swap_time/total_recompute_time):.2f}x")
    
    # Stats
    print("\nSwap Manager Stats:")
    print_stats_table(swap_mgr.get_stats(), "Swap Statistics")
    
    print("\nRecompute Manager Stats:")
    print_stats_table(recompute_mgr.get_stats(), "Recompute Statistics")
    
    print("="*60 + "\n")


def main():
    """Run all demos."""
    print("\n" + "="*60)
    print("PagedAttention Demo")
    print("="*60)
    
    try:
        demo_correctness()
        demo_memory_efficiency()
        demo_copy_on_write()
        demo_swap_recompute()
        
        print("\n" + "="*60)
        print("All demos completed successfully! ✓")
        print("="*60 + "\n")
        
    except Exception as e:
        print(f"\n✗ Demo failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()