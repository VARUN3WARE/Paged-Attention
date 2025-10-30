"""
Paged KV cache implementation with logical-to-physical block mapping.

Each sequence maintains a block table mapping logical blocks to physical blocks.
Supports efficient append, read, and copy-on-write operations.
"""

import torch
from typing import List, Tuple, Optional
from dataclasses import dataclass

from .allocator import BlockAllocator, PhysicalBlock


@dataclass
class BlockTableEntry:
    """
    Entry in a sequence's block table.
    
    Attributes:
        logical_idx: Logical block index in sequence
        phys_block_id: Physical block ID in allocator
        filled: Number of valid tokens in this block
    """
    logical_idx: int
    phys_block_id: int
    filled: int


class PagedKVCache:
    """
    KV cache for a single sequence using paged memory.
    
    Manages a block table that maps logical blocks to physical blocks.
    Supports appending new tokens and reading blocks for attention.
    """
    
    def __init__(self, block_size: int, hidden_dim: int, 
                 allocator: BlockAllocator, seq_id: Optional[int] = None):
        """
        Initialize paged KV cache.
        
        Args:
            block_size: Number of token slots per block
            hidden_dim: Dimension of K/V vectors
            allocator: Block allocator to use
            seq_id: Optional sequence identifier
        """
        self.block_size = block_size
        self.hidden_dim = hidden_dim
        self.allocator = allocator
        self.seq_id = seq_id
        
        # Block table: list of logical block mappings
        self.block_table: List[BlockTableEntry] = []
        
        # Total tokens stored
        self.num_tokens = 0
    
    def append_token_kv(self, key_vec: torch.Tensor, val_vec: torch.Tensor):
        """
        Append K/V vectors for a new token.
        
        Args:
            key_vec: Key vector [hidden_dim]
            val_vec: Value vector [hidden_dim]
        """
        if key_vec.shape != (self.hidden_dim,):
            raise ValueError(f"Expected key shape ({self.hidden_dim},), got {key_vec.shape}")
        if val_vec.shape != (self.hidden_dim,):
            raise ValueError(f"Expected val shape ({self.hidden_dim},), got {val_vec.shape}")
        
        # Check if we need a new block
        if not self.block_table or self.block_table[-1].filled >= self.block_size:
            # Allocate new physical block
            phys_id = self.allocator.allocate()
            logical_idx = len(self.block_table)
            self.block_table.append(BlockTableEntry(
                logical_idx=logical_idx,
                phys_block_id=phys_id,
                filled=0
            ))
        
        # Get last block and append
        entry = self.block_table[-1]
        block = self.allocator.get_block(entry.phys_block_id)
        
        # Write K/V to block
        block.key_data[entry.filled] = key_vec
        block.value_data[entry.filled] = val_vec
        entry.filled += 1
        block.filled = entry.filled
        
        self.num_tokens += 1
    
    def append_token_kv_batch(self, keys: torch.Tensor, values: torch.Tensor):
        """
        Append multiple K/V pairs at once.
        
        Args:
            keys: Key vectors [num_tokens, hidden_dim]
            values: Value vectors [num_tokens, hidden_dim]
        """
        num_tokens = keys.shape[0]
        for i in range(num_tokens):
            self.append_token_kv(keys[i], values[i])
    
    def read_blocks_for_attention(self) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """
        Read all blocks for attention computation.
        
        Returns:
            (keys_list, values_list): Lists of tensors, one per block.
            Each tensor is [filled_slots, hidden_dim]
        """
        keys_list = []
        values_list = []
        
        for entry in self.block_table:
            block = self.allocator.get_block(entry.phys_block_id)
            # Only return filled portion
            keys_list.append(block.key_data[:entry.filled])
            values_list.append(block.value_data[:entry.filled])
        
        return keys_list, values_list
    
    def get_all_keys_values(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get all keys and values as contiguous tensors (for baseline comparison).
        
        Returns:
            (keys, values): Each [num_tokens, hidden_dim]
        """
        keys_list, values_list = self.read_blocks_for_attention()
        
        if not keys_list:
            device = self.allocator.device
            return (torch.zeros(0, self.hidden_dim, device=device),
                    torch.zeros(0, self.hidden_dim, device=device))
        
        keys = torch.cat(keys_list, dim=0)
        values = torch.cat(values_list, dim=0)
        return keys, values
    
    def fork(self) -> 'PagedKVCache':
        """
        Fork this cache for beam search or parallel sampling.
        Shares physical blocks via reference counting (copy-on-write).
        
        Returns:
            New PagedKVCache sharing blocks with this one
        """
        new_cache = PagedKVCache(
            self.block_size,
            self.hidden_dim,
            self.allocator,
            seq_id=None  # New sequence
        )
        
        # Share all blocks (increment refcounts)
        for entry in self.block_table:
            self.allocator.inc_ref(entry.phys_block_id)
            new_cache.block_table.append(BlockTableEntry(
                logical_idx=entry.logical_idx,
                phys_block_id=entry.phys_block_id,
                filled=entry.filled
            ))
        
        new_cache.num_tokens = self.num_tokens
        return new_cache
    
    def cow_append(self, key_vec: torch.Tensor, val_vec: torch.Tensor):
        """
        Append with copy-on-write: if last block is shared, copy it first.
        
        Args:
            key_vec: Key vector [hidden_dim]
            val_vec: Value vector [hidden_dim]
        """
        if self.block_table:
            last_entry = self.block_table[-1]
            last_block = self.allocator.get_block(last_entry.phys_block_id)
            
            # If shared and not full, copy before writing
            if last_block.refcount > 1 and not last_block.is_full():
                # Copy block
                new_block_id = self.allocator.copy_block(last_entry.phys_block_id)
                # Decrement old block refcount
                self.allocator.dec_ref(last_entry.phys_block_id)
                # Update table entry
                last_entry.phys_block_id = new_block_id
        
        # Now append normally
        self.append_token_kv(key_vec, val_vec)
    
    def free_all(self):
        """Free all blocks in this cache."""
        for entry in self.block_table:
            self.allocator.free(entry.phys_block_id)
        self.block_table.clear()
        self.num_tokens = 0
    
    def get_num_blocks(self) -> int:
        """Return number of logical blocks."""
        return len(self.block_table)
    
    def get_memory_usage(self) -> int:
        """Return memory usage in bytes (K + V data)."""
        num_blocks = len(self.block_table)
        bytes_per_element = 4  # float32
        memory = num_blocks * self.block_size * self.hidden_dim * 2 * bytes_per_element
        return memory
    
    def get_wasted_memory(self) -> int:
        """Return wasted memory due to internal fragmentation."""
        if not self.block_table:
            return 0
        
        total_allocated = len(self.block_table) * self.block_size
        total_used = self.num_tokens
        wasted_slots = total_allocated - total_used
        
        bytes_per_element = 4
        return wasted_slots * self.hidden_dim * 2 * bytes_per_element