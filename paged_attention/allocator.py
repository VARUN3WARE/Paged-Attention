"""
Physical block allocator with reference counting.

Manages a pool of physical memory blocks and tracks their usage
through reference counting for efficient memory sharing.
"""

import torch
from collections import deque
from typing import Dict, Optional, List
from dataclasses import dataclass


@dataclass
class PhysicalBlock:
    """
    A fixed-size physical block of memory for storing KV pairs.
    
    Attributes:
        block_id: Unique identifier for this block
        block_size: Number of token slots in this block
        hidden_dim: Dimension of K and V vectors
        device: Device where block memory resides
        key_data: Tensor storing key vectors [block_size, hidden_dim]
        value_data: Tensor storing value vectors [block_size, hidden_dim]
        filled: Number of valid token slots currently used
        refcount: Reference count for copy-on-write
    """
    block_id: int
    block_size: int
    hidden_dim: int
    device: str
    key_data: torch.Tensor
    value_data: torch.Tensor
    filled: int = 0
    refcount: int = 0
    
    @staticmethod
    def create(block_id: int, block_size: int, hidden_dim: int, 
               device: str = 'cpu') -> 'PhysicalBlock':
        """Factory method to create a new physical block."""
        key_data = torch.zeros((block_size, hidden_dim), 
                               device=device, dtype=torch.float32)
        value_data = torch.zeros((block_size, hidden_dim), 
                                 device=device, dtype=torch.float32)
        return PhysicalBlock(
            block_id=block_id,
            block_size=block_size,
            hidden_dim=hidden_dim,
            device=device,
            key_data=key_data,
            value_data=value_data,
            filled=0,
            refcount=0
        )
    
    def is_full(self) -> bool:
        """Check if block is completely filled."""
        return self.filled >= self.block_size
    
    def is_empty(self) -> bool:
        """Check if block has no valid tokens."""
        return self.filled == 0
    
    def reset(self):
        """Reset block state for reuse."""
        self.filled = 0
        self.refcount = 0
        self.key_data.zero_()
        self.value_data.zero_()


class OutOfMemoryError(Exception):
    """Raised when no free blocks are available."""
    pass


class BlockAllocator:
    """
    Manages allocation and deallocation of physical memory blocks.
    
    Implements reference counting for copy-on-write semantics.
    Blocks can be shared across multiple sequences until modified.
    """
    
    def __init__(self, total_blocks: int, block_size: int, 
                 hidden_dim: int, device: str = 'cpu'):
        """
        Initialize block allocator.
        
        Args:
            total_blocks: Total number of physical blocks in pool
            block_size: Number of token slots per block
            hidden_dim: Dimension of K/V vectors
            device: Device for block memory ('cpu' or 'cuda')
        """
        self.total_blocks = total_blocks
        self.block_size = block_size
        self.hidden_dim = hidden_dim
        self.device = device
        
        # Create all physical blocks upfront
        self.blocks: Dict[int, PhysicalBlock] = {}
        for i in range(total_blocks):
            self.blocks[i] = PhysicalBlock.create(
                i, block_size, hidden_dim, device
            )
        
        # Free list of available block IDs
        self.free_list: deque = deque(range(total_blocks))
        
        # Statistics
        self.num_allocations = 0
        self.num_frees = 0
        self.num_cow_copies = 0
    
    def allocate(self) -> int:
        """
        Allocate a new physical block.
        
        Returns:
            Block ID of allocated block
            
        Raises:
            OutOfMemoryError: If no free blocks available
        """
        if not self.free_list:
            raise OutOfMemoryError(
                f"No free blocks available (total: {self.total_blocks})"
            )
        
        block_id = self.free_list.popleft()
        block = self.blocks[block_id]
        block.refcount = 1
        block.filled = 0
        
        self.num_allocations += 1
        return block_id
    
    def free(self, block_id: int):
        """
        Free a physical block (decrement refcount, return to pool if 0).
        
        Args:
            block_id: ID of block to free
        """
        if block_id not in self.blocks:
            raise ValueError(f"Invalid block_id: {block_id}")
        
        block = self.blocks[block_id]
        if block.refcount <= 0:
            raise ValueError(f"Block {block_id} already free (refcount={block.refcount})")
        
        block.refcount -= 1
        
        if block.refcount == 0:
            # Return to free pool
            block.reset()
            self.free_list.append(block_id)
            self.num_frees += 1
    
    def inc_ref(self, block_id: int):
        """
        Increment reference count (for sharing).
        
        Args:
            block_id: ID of block to increment
        """
        if block_id not in self.blocks:
            raise ValueError(f"Invalid block_id: {block_id}")
        
        self.blocks[block_id].refcount += 1
    
    def dec_ref(self, block_id: int):
        """
        Decrement reference count (alias for free).
        
        Args:
            block_id: ID of block to decrement
        """
        self.free(block_id)
    
    def get_block(self, block_id: int) -> PhysicalBlock:
        """
        Get physical block by ID.
        
        Args:
            block_id: Block identifier
            
        Returns:
            PhysicalBlock object
        """
        if block_id not in self.blocks:
            raise ValueError(f"Invalid block_id: {block_id}")
        return self.blocks[block_id]
    
    def copy_block(self, src_block_id: int) -> int:
        """
        Copy block for copy-on-write.
        
        Args:
            src_block_id: Source block to copy
            
        Returns:
            New block ID with copied data
        """
        src_block = self.get_block(src_block_id)
        new_block_id = self.allocate()
        new_block = self.get_block(new_block_id)
        
        # Copy data
        new_block.key_data[:src_block.filled] = src_block.key_data[:src_block.filled]
        new_block.value_data[:src_block.filled] = src_block.value_data[:src_block.filled]
        new_block.filled = src_block.filled
        
        self.num_cow_copies += 1
        return new_block_id
    
    def get_num_free_blocks(self) -> int:
        """Return number of free blocks."""
        return len(self.free_list)
    
    def get_stats(self) -> Dict:
        """Return allocator statistics."""
        return {
            'total_blocks': self.total_blocks,
            'free_blocks': len(self.free_list),
            'used_blocks': self.total_blocks - len(self.free_list),
            'num_allocations': self.num_allocations,
            'num_frees': self.num_frees,
            'num_cow_copies': self.num_cow_copies
        }
    
    def reset_stats(self):
        """Reset statistics counters."""
        self.num_allocations = 0
        self.num_frees = 0
        self.num_cow_copies = 0
