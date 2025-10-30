"""
PagedAttention: Efficient Memory Management for LLM Inference

This package implements block-based KV cache management with:
- Non-contiguous memory allocation
- Copy-on-write semantics
- Swap and recompute strategies
- Reduced memory fragmentation
"""

__version__ = '0.1.0'

from .paged_attention import PagedAttention
from .kv_cache import PagedKVCache, BlockTableEntry
from .allocator import BlockAllocator, PhysicalBlock
from .scheduler import SimpleScheduler
from .decoding import DecodingManager
from .swap_recompute import SwapManager, RecomputeManager
from .utils import (
    generate_synthetic_workload,
    plot_memory_usage,
    plot_throughput,
    compute_memory_metrics
)

__all__ = [
    'PagedAttention',
    'PagedKVCache',
    'BlockTableEntry',
    'BlockAllocator',
    'PhysicalBlock',
    'SimpleScheduler',
    'DecodingManager',
    'SwapManager',
    'RecomputeManager',
    'generate_synthetic_workload',
    'plot_memory_usage',
    'plot_throughput',
    'compute_memory_metrics'
]
