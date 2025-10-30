"""
Simple batch scheduler for managing multiple sequences.

Emulates iteration-level scheduling for testing and benchmarking.
"""

from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import torch

from .kv_cache import PagedKVCache
from .allocator import BlockAllocator


@dataclass
class SequenceRequest:
    """
    Represents a single sequence generation request.
    
    Attributes:
        seq_id: Unique sequence identifier
        prompt_tokens: Initial prompt token IDs
        max_tokens: Maximum tokens to generate
        num_samples: Number of parallel samples (for parallel sampling)
        kv_cache: Associated KV cache
        generated_tokens: Tokens generated so far
        is_finished: Whether generation is complete
    """
    seq_id: int
    prompt_tokens: List[int]
    max_tokens: int
    num_samples: int = 1
    kv_cache: Optional[PagedKVCache] = None
    generated_tokens: List[int] = None
    is_finished: bool = False
    
    def __post_init__(self):
        if self.generated_tokens is None:
            self.generated_tokens = []


class SimpleScheduler:
    """
    FCFS (First-Come-First-Serve) scheduler for batch inference.
    
    Manages multiple sequence requests and schedules them for execution.
    Supports parallel sampling and beam search through request forking.
    """
    
    def __init__(self, allocator: BlockAllocator, max_batch_size: int = 8):
        """
        Initialize scheduler.
        
        Args:
            allocator: Block allocator for KV caches
            max_batch_size: Maximum number of sequences to batch
        """
        self.allocator = allocator
        self.max_batch_size = max_batch_size
        
        # Active requests
        self.requests: Dict[int, SequenceRequest] = {}
        self.next_seq_id = 0
        
        # Waiting queue
        self.waiting_queue: List[SequenceRequest] = []
        
        # Statistics
        self.total_requests = 0
        self.completed_requests = 0
    
    def add_request(self, prompt_tokens: List[int], max_tokens: int, 
                   num_samples: int = 1) -> int:
        """
        Add a new sequence request.
        
        Args:
            prompt_tokens: Prompt token IDs
            max_tokens: Maximum tokens to generate
            num_samples: Number of parallel samples
            
        Returns:
            Sequence ID
        """
        seq_id = self.next_seq_id
        self.next_seq_id += 1
        
        request = SequenceRequest(
            seq_id=seq_id,
            prompt_tokens=prompt_tokens,
            max_tokens=max_tokens,
            num_samples=num_samples
        )
        
        self.waiting_queue.append(request)
        self.total_requests += 1
        
        return seq_id
    
    def schedule_batch(self) -> List[SequenceRequest]:
        """
        Schedule next batch of sequences.
        
        Returns:
            List of requests to process in this iteration
        """
        # Move waiting requests to active if space available
        while (len(self.requests) < self.max_batch_size and 
               self.waiting_queue and 
               self.allocator.get_num_free_blocks() > 0):
            request = self.waiting_queue.pop(0)
            self.requests[request.seq_id] = request
        
        # Return all active requests
        return list(self.requests.values())
    
    def mark_finished(self, seq_id: int):
        """Mark a sequence as finished and free its resources."""
        if seq_id in self.requests:
            request = self.requests[seq_id]
            if request.kv_cache is not None:
                request.kv_cache.free_all()
            request.is_finished = True
            del self.requests[seq_id]
            self.completed_requests += 1
    
    def get_stats(self) -> Dict:
        """Return scheduler statistics."""
        return {
            'active_requests': len(self.requests),
            'waiting_requests': len(self.waiting_queue),
            'total_requests': self.total_requests,
            'completed_requests': self.completed_requests,
            'free_blocks': self.allocator.get_num_free_blocks()
        }
    
    def is_empty(self) -> bool:
        """Check if scheduler has no pending work."""
        return len(self.requests) == 0 and len(self.waiting_queue) == 0


class BatchProcessor:
    """
    Processes batches of sequences through attention layers.
    Handles prompt and generation phases.
    """
    
    def __init__(self, block_size: int, hidden_dim: int):
        self.block_size = block_size
        self.hidden_dim = hidden_dim
    
    def process_prompt_phase(self, request: SequenceRequest, 
                            allocator: BlockAllocator) -> PagedKVCache:
        """
        Process prompt tokens and initialize KV cache.
        
        Args:
            request: Sequence request
            allocator: Block allocator
            
        Returns:
            Initialized PagedKVCache
        """
        cache = PagedKVCache(
            block_size=self.block_size,
            hidden_dim=self.hidden_dim,
            allocator=allocator,
            seq_id=request.seq_id
        )
        
        # Simulate adding prompt tokens to cache
        # In real implementation, this would run through attention
        num_prompt_tokens = len(request.prompt_tokens)
        
        for _ in range(num_prompt_tokens):
            # Generate dummy K/V vectors
            k = torch.randn(self.hidden_dim)
            v = torch.randn(self.hidden_dim)
            cache.append_token_kv(k, v)
        
        request.kv_cache = cache
        return cache
    
    def process_generation_step(self, request: SequenceRequest, 
                               new_token_id: int):
        """
        Process one generation step (append new token to cache).
        
        Args:
            request: Sequence request
            new_token_id: Generated token ID
        """
        if request.kv_cache is None:
            raise ValueError("KV cache not initialized")
        
        # Simulate adding new token K/V
        k = torch.randn(self.hidden_dim)
        v = torch.randn(self.hidden_dim)
        request.kv_cache.append_token_kv(k, v)
        
        request.generated_tokens.append(new_token_id)