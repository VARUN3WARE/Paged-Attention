"""
PagedAttention: Efficient Memory Management for LLM Inference

This package implements block-based KV cache management with:
- Non-contiguous memory allocation
- Copy-on-write semantics
- Swap and recompute strategies
- Reduced memory fragmentation
"""

__version__ = '0.1.0'

from .paged_attention import PagedAttention, VanillaAttention
from .kv_cache import PagedKVCache, BlockTableEntry
from .allocator import BlockAllocator, PhysicalBlock
from .scheduler import SimpleScheduler
from .decoding import DecodingManager, ParallelSamplingManager
from .swap_recompute import SwapManager, RecomputeManager
from .utils import (
    generate_synthetic_workload,
    plot_memory_usage,
    plot_throughput,
    plot_fragmentation,
    plot_swap_vs_recompute,
    plot_beam_search_memory,
    print_stats_table,
    create_causal_mask,
    compute_memory_metrics
)

__all__ = [
    'PagedAttention',
    'VanillaAttention',
    'PagedKVCache',
    'BlockTableEntry',
    'BlockAllocator',
    'PhysicalBlock',
    'SimpleScheduler',
    'DecodingManager',
    'ParallelSamplingManager',
    'SwapManager',
    'RecomputeManager',
    'print_stats_table',
    'create_causal_mask',
    'plot_fragmentation',
    'plot_swap_vs_recompute',
    'generate_synthetic_workload',
    'plot_memory_usage',
    'plot_throughput',
    'compute_memory_metrics'
]
