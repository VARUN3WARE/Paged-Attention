"""
Swap and recompute strategies for memory management.

Simulates swapping blocks to CPU memory and recomputation of KV values.
"""

import torch
from typing import Dict, List, Set, Optional
import time

from .allocator import BlockAllocator, PhysicalBlock
from .kv_cache import PagedKVCache


class SwapManager:
    """
    Manages swapping blocks between GPU and CPU memory.
    
    Simulates the cost of data transfer for performance analysis.
    """
    
    def __init__(self, allocator: BlockAllocator, gpu_to_cpu_bandwidth_gbps: float = 25.0):
        """
        Initialize swap manager.
        
        Args:
            allocator: Block allocator
            gpu_to_cpu_bandwidth_gbps: PCIe bandwidth in GB/s
        """
        self.allocator = allocator
        self.gpu_to_cpu_bandwidth_gbps = gpu_to_cpu_bandwidth_gbps
        
        # Track swapped blocks
        self.swapped_blocks: Dict[int, tuple] = {}  # block_id -> (cpu_key, cpu_val)
        
        # Statistics
        self.num_swaps_out = 0
        self.num_swaps_in = 0
        self.total_swap_time = 0.0
        self.total_bytes_swapped = 0
    
    def swap_out_block(self, block_id: int) -> float:
        """
        Swap block from GPU to CPU.
        
        Args:
            block_id: Block to swap out
            
        Returns:
            Simulated swap time in seconds
        """
        block = self.allocator.get_block(block_id)
        
        if block_id in self.swapped_blocks:
            return 0.0  # Already swapped
        
        # Copy to CPU
        cpu_key = block.key_data.cpu()
        cpu_val = block.value_data.cpu()
        
        self.swapped_blocks[block_id] = (cpu_key, cpu_val)
        
        # Simulate swap time
        bytes_transferred = block.key_data.numel() * 4 * 2  # K + V, float32
        swap_time = bytes_transferred / (self.gpu_to_cpu_bandwidth_gbps * 1e9)
        
        self.num_swaps_out += 1
        self.total_swap_time += swap_time
        self.total_bytes_swapped += bytes_transferred
        
        return swap_time
    
    def swap_in_block(self, block_id: int) -> float:
        """
        Swap block from CPU to GPU.
        
        Args:
            block_id: Block to swap in
            
        Returns:
            Simulated swap time in seconds
        """
        if block_id not in self.swapped_blocks:
            return 0.0  # Not swapped
        
        block = self.allocator.get_block(block_id)
        cpu_key, cpu_val = self.swapped_blocks[block_id]
        
        # Copy to GPU
        block.key_data.copy_(cpu_key)
        block.value_data.copy_(cpu_val)
        
        del self.swapped_blocks[block_id]
        
        # Simulate swap time
        bytes_transferred = block.key_data.numel() * 4 * 2
        swap_time = bytes_transferred / (self.gpu_to_cpu_bandwidth_gbps * 1e9)
        
        self.num_swaps_in += 1
        self.total_swap_time += swap_time
        self.total_bytes_swapped += bytes_transferred
        
        return swap_time
    
    def is_swapped(self, block_id: int) -> bool:
        """Check if block is currently swapped out."""
        return block_id in self.swapped_blocks
    
    def get_stats(self) -> Dict:
        """Return swap statistics."""
        return {
            'num_swaps_out': self.num_swaps_out,
            'num_swaps_in': self.num_swaps_in,
            'total_swap_time': self.total_swap_time,
            'total_bytes_swapped': self.total_bytes_swapped,
            'swapped_blocks': len(self.swapped_blocks)
        }


class RecomputeManager:
    """
    Manages recomputation of KV values instead of storing them.
    
    Trades computation for memory by recomputing KV on demand.
    """
    
    def __init__(self, compute_time_per_token_ms: float = 0.1):
        """
        Initialize recompute manager.
        
        Args:
            compute_time_per_token_ms: Time to recompute one token's KV (milliseconds)
        """
        self.compute_time_per_token_ms = compute_time_per_token_ms
        
        # Track which blocks are recomputed
        self.recomputed_blocks: Set[int] = set()
        
        # Statistics
        self.num_recomputes = 0
        self.total_recompute_time = 0.0
        self.tokens_recomputed = 0
    
    def recompute_block(self, block_id: int, num_tokens: int, 
                       hidden_dim: int) -> tuple:
        """
        Recompute KV for a block.
        
        Args:
            block_id: Block identifier
            num_tokens: Number of tokens in block
            hidden_dim: Hidden dimension
            
        Returns:
            (key_tensor, value_tensor, recompute_time)
        """
        # Simulate recomputation
        recompute_time = num_tokens * self.compute_time_per_token_ms / 1000.0
        
        # Generate dummy recomputed values
        key_tensor = torch.randn(num_tokens, hidden_dim)
        value_tensor = torch.randn(num_tokens, hidden_dim)
        
        self.recomputed_blocks.add(block_id)
        self.num_recomputes += 1
        self.total_recompute_time += recompute_time
        self.tokens_recomputed += num_tokens
        
        return key_tensor, value_tensor, recompute_time
    
    def mark_for_recompute(self, block_id: int):
        """Mark block for recomputation."""
        self.recomputed_blocks.add(block_id)
    
    def is_recomputed(self, block_id: int) -> bool:
        """Check if block is marked for recomputation."""
        return block_id in self.recomputed_blocks
    
    def get_stats(self) -> Dict:
        """Return recompute statistics."""
        return {
            'num_recomputes': self.num_recomputes,
            'total_recompute_time': self.total_recompute_time,
            'tokens_recomputed': self.tokens_recomputed,
            'avg_time_per_token': (self.total_recompute_time / self.tokens_recomputed 
                                  if self.tokens_recomputed > 0 else 0)
        }


class HybridMemoryManager:
    """
    Combines swapping and recomputation strategies.
    
    Decides whether to swap or recompute based on access patterns.
    """
    
    def __init__(self, swap_manager: SwapManager, recompute_manager: RecomputeManager):
        self.swap_manager = swap_manager
        self.recompute_manager = recompute_manager
        
        # Access frequency tracking
        self.access_counts: Dict[int, int] = {}
    
    def evict_block(self, block_id: int, num_tokens: int) -> str:
        """
        Decide whether to swap or mark for recompute.
        
        Args:
            block_id: Block to evict
            num_tokens: Number of tokens in block
            
        Returns:
            Strategy used ('swap' or 'recompute')
        """
        access_count = self.access_counts.get(block_id, 0)
        
        # Simple heuristic: frequently accessed blocks should be swapped
        # Rarely accessed blocks can be recomputed
        if access_count > 2:
            self.swap_manager.swap_out_block(block_id)
            return 'swap'
        else:
            self.recompute_manager.mark_for_recompute(block_id)
            return 'recompute'
    
    def access_block(self, block_id: int):
        """Record block access."""
        self.access_counts[block_id] = self.access_counts.get(block_id, 0) + 1
    
    def get_combined_stats(self) -> Dict:
        """Return combined statistics."""
        return {
            'swap_stats': self.swap_manager.get_stats(),
            'recompute_stats': self.recompute_manager.get_stats(),
            'total_evictions': (self.swap_manager.num_swaps_out + 
                               self.recompute_manager.num_recomputes)
        }