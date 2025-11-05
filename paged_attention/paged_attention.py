"""
PagedAttention: Blockwise attention over non-contiguous KV cache.

Implements attention that operates on KV cache split into fixed-size blocks.
Uses numerically stable blockwise softmax computation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
import math

from .kv_cache import PagedKVCache


class PagedAttention(nn.Module):
    """
    Multi-head attention with paged KV cache support.
    
    Computes attention over KV blocks stored non-contiguously in memory.
    Uses blockwise softmax for numerical stability.
    """
    
    def __init__(self, hidden_dim: int, num_heads: int, block_size: int, 
                 dropout: float = 0.0):
        """
        Initialize PagedAttention.
        
        Args:
            hidden_dim: Model hidden dimension
            num_heads: Number of attention heads
            block_size: Size of KV cache blocks
            dropout: Attention dropout probability
        """
        super().__init__()
        
        if hidden_dim % num_heads != 0:
            raise ValueError(f"hidden_dim ({hidden_dim}) must be divisible by num_heads ({num_heads})")
        
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.block_size = block_size
        self.dropout = dropout
        self.scale = 1.0 / math.sqrt(self.head_dim)
        
        # Projection layers
        self.q_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        
        self.dropout_layer = nn.Dropout(dropout) if dropout > 0 else None
    
    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        """
        Split hidden dim into multiple heads.
        
        Args:
            x: [batch, seq_len, hidden_dim]
            
        Returns:
            [batch, num_heads, seq_len, head_dim]
        """
        batch_size, seq_len, _ = x.shape
        x = x.view(batch_size, seq_len, self.num_heads, self.head_dim)
        return x.permute(0, 2, 1, 3)
    
    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        """
        Merge multiple heads back to hidden dim.
        
        Args:
            x: [batch, num_heads, seq_len, head_dim]
            
        Returns:
            [batch, seq_len, hidden_dim]
        """
        batch_size, _, seq_len, _ = x.shape
        x = x.permute(0, 2, 1, 3).contiguous()
        return x.view(batch_size, seq_len, self.hidden_dim)
    
    def forward_vanilla(self, query: torch.Tensor, key: torch.Tensor, 
                       value: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Vanilla attention (for baseline comparison).
        
        Args:
            query: [batch, q_len, hidden_dim]
            key: [batch, kv_len, hidden_dim]
            value: [batch, kv_len, hidden_dim]
            mask: Optional attention mask
            
        Returns:
            [batch, q_len, hidden_dim]
        """
        batch_size = query.shape[0]
        
        # Project and split heads
        Q = self._split_heads(self.q_proj(query))  # [batch, heads, q_len, head_dim]
        K = self._split_heads(self.k_proj(key))    # [batch, heads, kv_len, head_dim]
        V = self._split_heads(self.v_proj(value))  # [batch, heads, kv_len, head_dim]
        
        # Attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale  # [batch, heads, q_len, kv_len]
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        attn_weights = F.softmax(scores, dim=-1)
        
        if self.dropout_layer is not None:
            attn_weights = self.dropout_layer(attn_weights)
        
        # Apply attention to values
        output = torch.matmul(attn_weights, V)  # [batch, heads, q_len, head_dim]
        output = self._merge_heads(output)      # [batch, q_len, hidden_dim]
        output = self.out_proj(output)
        
        return output
    
    def forward_paged(self, query: torch.Tensor, kv_cache: PagedKVCache, 
                     mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Paged attention over blockwise KV cache.
        
        Args:
            query: [batch, q_len, hidden_dim]
            kv_cache: PagedKVCache with blocks
            mask: Optional attention mask
            
        Returns:
            [batch, q_len, hidden_dim]
        """
        batch_size, q_len, _ = query.shape
        
        # Project query
        Q = self._split_heads(self.q_proj(query))  # [batch, heads, q_len, head_dim]
        
        # Read K/V blocks from cache
        K_blocks_list, V_blocks_list = kv_cache.read_blocks_for_attention()
        
        if not K_blocks_list:
            # Empty cache, return zeros
            return torch.zeros_like(query)
        
        # Compute attention blockwise (numerically stable)
        output = self._blockwise_attention(Q, K_blocks_list, V_blocks_list, mask)
        
        output = self._merge_heads(output)
        output = self.out_proj(output)
        
        return output
    
    def _blockwise_attention(self, Q: torch.Tensor, K_blocks: list, 
                            V_blocks: list, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute attention over KV blocks with numerically stable softmax.
        
        Uses the trick: softmax over concatenated blocks by tracking running max and sum.
        
        Args:
            Q: [batch, heads, q_len, head_dim]
            K_blocks: List of [block_size_i, hidden_dim] tensors
            V_blocks: List of [block_size_i, hidden_dim] tensors
            mask: Optional mask
            
        Returns:
            [batch, heads, q_len, head_dim]
        """
        batch_size, num_heads, q_len, head_dim = Q.shape
        device = Q.device
        
        # Initialize accumulators for numerically stable softmax
        max_score = torch.full((batch_size, num_heads, q_len, 1), 
                              float('-inf'), device=device)
        sum_exp = torch.zeros((batch_size, num_heads, q_len, 1), device=device)
        weighted_values = torch.zeros((batch_size, num_heads, q_len, head_dim), 
                                     device=device)
        
        current_pos = 0
        
        for block_idx, (K_block, V_block) in enumerate(zip(K_blocks, V_blocks)):
            # K_block: [block_len, hidden_dim], V_block: [block_len, hidden_dim]
            block_len = K_block.shape[0]
            
            # K_block and V_block are expected to be already-projected tensors
            # of shape [block_len, hidden_dim]. Split heads directly.
            K_proj = self._split_heads(K_block.unsqueeze(0))  # [1, heads, block_len, head_dim]
            V_proj = self._split_heads(V_block.unsqueeze(0))
            
            # Expand to batch size
            K_proj = K_proj.expand(batch_size, -1, -1, -1)
            V_proj = V_proj.expand(batch_size, -1, -1, -1)
            
            # Compute scores for this block
            block_scores = torch.matmul(Q, K_proj.transpose(-2, -1)) * self.scale
            # [batch, heads, q_len, block_len]
            
            if mask is not None:
                block_mask = mask[:, :, :, current_pos:current_pos + block_len]
                block_scores = block_scores.masked_fill(block_mask == 0, float('-inf'))
            
            # Update running max
            block_max = block_scores.max(dim=-1, keepdim=True)[0]  # [batch, heads, q_len, 1]
            new_max = torch.maximum(max_score, block_max)
            
            # Adjust previous sum_exp for new max
            sum_exp = sum_exp * torch.exp(max_score - new_max)
            
            # Compute exp for current block
            block_exp = torch.exp(block_scores - new_max)
            
            # Update sum
            sum_exp = sum_exp + block_exp.sum(dim=-1, keepdim=True)
            
            # Update weighted values
            weighted_values = weighted_values * torch.exp(max_score - new_max)
            weighted_values = weighted_values + torch.matmul(block_exp, V_proj)
            
            # Update max
            max_score = new_max
            current_pos += block_len
        
        # Final normalization
        output = weighted_values / sum_exp
        
        if self.dropout_layer is not None and self.training:
            # Note: This applies dropout to output, not attention weights
            # For true attention dropout, we'd need to store weights
            output = self.dropout_layer(output)
        
        return output
    
    def forward(self, query: torch.Tensor, 
               key: Optional[torch.Tensor] = None,
               value: Optional[torch.Tensor] = None,
               kv_cache: Optional[PagedKVCache] = None,
               mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass - automatically selects vanilla or paged attention.
        
        Args:
            query: [batch, q_len, hidden_dim]
            key: Optional [batch, kv_len, hidden_dim] for vanilla mode
            value: Optional [batch, kv_len, hidden_dim] for vanilla mode
            kv_cache: Optional PagedKVCache for paged mode
            mask: Optional attention mask
            
        Returns:
            [batch, q_len, hidden_dim]
        """
        if kv_cache is not None:
            return self.forward_paged(query, kv_cache, mask)
        elif key is not None and value is not None:
            return self.forward_vanilla(query, key, value, mask)
        else:
            raise ValueError("Must provide either kv_cache or (key, value)")


class VanillaAttention(nn.Module):
    """
    Standard multi-head attention for baseline comparison.
    Stores KV in contiguous memory.
    """
    
    def __init__(self, hidden_dim: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        
        if hidden_dim % num_heads != 0:
            raise ValueError(f"hidden_dim must be divisible by num_heads")
        
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)
        
        self.q_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        
        self.dropout_layer = nn.Dropout(dropout) if dropout > 0 else None
        
        # Contiguous KV cache
        self.k_cache = []
        self.v_cache = []
    
    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        x = x.view(batch_size, seq_len, self.num_heads, self.head_dim)
        return x.permute(0, 2, 1, 3)
    
    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, _, seq_len, _ = x.shape
        x = x.permute(0, 2, 1, 3).contiguous()
        return x.view(batch_size, seq_len, self.hidden_dim)
    
    def append_kv(self, key: torch.Tensor, value: torch.Tensor):
        """Append K/V to cache."""
        self.k_cache.append(key)
        self.v_cache.append(value)
    
    def forward(self, query: torch.Tensor, key: torch.Tensor, 
               value: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Standard attention forward."""
        Q = self._split_heads(self.q_proj(query))
        K = self._split_heads(self.k_proj(key))
        V = self._split_heads(self.v_proj(value))
        
        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        attn_weights = F.softmax(scores, dim=-1)
        
        if self.dropout_layer is not None:
            attn_weights = self.dropout_layer(attn_weights)
        
        output = torch.matmul(attn_weights, V)
        output = self._merge_heads(output)
        output = self.out_proj(output)
        
        return output
    
    def get_memory_usage(self) -> int:
        """Return memory usage in bytes."""
        if not self.k_cache:
            return 0
        total_tokens = sum(k.shape[1] for k in self.k_cache)
        bytes_per_element = 4  # float32
        return total_tokens * self.hidden_dim * 2 * bytes_per_element