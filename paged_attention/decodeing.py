"""
Copy-on-Write (COW) decoding manager for beam search and parallel sampling.

Manages forking and sharing of KV caches across multiple decode paths.
"""

from typing import List, Dict, Set, Optional
from dataclasses import dataclass
import torch

from .kv_cache import PagedKVCache
from .allocator import BlockAllocator


@dataclass
class BeamHypothesis:
    """
    A single hypothesis in beam search.
    
    Attributes:
        beam_id: Unique beam identifier
        parent_id: Parent beam ID (for backtracking)
        token_id: Token generated at this step
        score: Cumulative log probability
        kv_cache: KV cache for this hypothesis
        is_finished: Whether beam is complete
    """
    beam_id: int
    parent_id: Optional[int]
    token_id: int
    score: float
    kv_cache: PagedKVCache
    is_finished: bool = False


class DecodingManager:
    """
    Manages copy-on-write semantics for beam search and parallel sampling.
    
    Handles forking caches, tracking shared blocks, and efficient memory usage.
    """
    
    def __init__(self, allocator: BlockAllocator, block_size: int, hidden_dim: int):
        """
        Initialize decoding manager.
        
        Args:
            allocator: Block allocator
            block_size: Size of KV blocks
            hidden_dim: Model hidden dimension
        """
        self.allocator = allocator
        self.block_size = block_size
        self.hidden_dim = hidden_dim
        
        # Track beams
        self.beams: Dict[int, BeamHypothesis] = {}
        self.next_beam_id = 0
        
        # Statistics
        self.num_forks = 0
        self.num_cow_copies = 0
    
    def initialize_beam(self, initial_cache: PagedKVCache, 
                       initial_token: int = 0) -> int:
        """
        Initialize root beam with existing cache.
        
        Args:
            initial_cache: Prompt KV cache
            initial_token: Initial token ID
            
        Returns:
            Beam ID
        """
        beam_id = self.next_beam_id
        self.next_beam_id += 1
        
        beam = BeamHypothesis(
            beam_id=beam_id,
            parent_id=None,
            token_id=initial_token,
            score=0.0,
            kv_cache=initial_cache
        )
        
        self.beams[beam_id] = beam
        return beam_id
    
    def fork_beam(self, parent_beam_id: int, token_id: int, 
                 score: float) -> int:
        """
        Fork a new beam from parent (copy-on-write).
        
        Args:
            parent_beam_id: Parent beam to fork from
            token_id: New token for this beam
            score: Beam score
            
        Returns:
            New beam ID
        """
        if parent_beam_id not in self.beams:
            raise ValueError(f"Parent beam {parent_beam_id} not found")
        
        parent_beam = self.beams[parent_beam_id]
        
        # Fork KV cache (shares blocks via refcounting)
        new_cache = parent_beam.kv_cache.fork()
        
        new_beam_id = self.next_beam_id
        self.next_beam_id += 1
        
        new_beam = BeamHypothesis(
            beam_id=new_beam_id,
            parent_id=parent_beam_id,
            token_id=token_id,
            score=score,
            kv_cache=new_cache
        )
        
        self.beams[new_beam_id] = new_beam
        self.num_forks += 1
        
        return new_beam_id
    
    def append_token(self, beam_id: int, key_vec: torch.Tensor, 
                    val_vec: torch.Tensor):
        """
        Append token to beam's cache (with COW if needed).
        
        Args:
            beam_id: Beam to append to
            key_vec: Key vector [hidden_dim]
            val_vec: Value vector [hidden_dim]
        """
        if beam_id not in self.beams:
            raise ValueError(f"Beam {beam_id} not found")
        
        beam = self.beams[beam_id]
        
        # Check if COW is needed
        old_cow_copies = self.allocator.num_cow_copies
        beam.kv_cache.cow_append(key_vec, val_vec)
        
        if self.allocator.num_cow_copies > old_cow_copies:
            self.num_cow_copies += 1
    
    def finish_beam(self, beam_id: int):
        """Mark beam as finished."""
        if beam_id in self.beams:
            self.beams[beam_id].is_finished = True
    
    def free_beam(self, beam_id: int):
        """Free a beam and its resources."""
        if beam_id in self.beams:
            beam = self.beams[beam_id]
            beam.kv_cache.free_all()
            del self.beams[beam_id]
    
    def get_active_beams(self) -> List[BeamHypothesis]:
        """Return all active (not finished) beams."""
        return [b for b in self.beams.values() if not b.is_finished]
    
    def get_best_beam(self) -> Optional[BeamHypothesis]:
        """Return beam with highest score."""
        active = self.get_active_beams()
        if not active:
            return None
        return max(active, key=lambda b: b.score)
    
    def get_stats(self) -> Dict:
        """Return decoding statistics."""
        return {
            'num_beams': len(self.beams),
            'active_beams': len(self.get_active_beams()),
            'num_forks': self.num_forks,
            'num_cow_copies': self.num_cow_copies
        }


class ParallelSamplingManager:
    """
    Manages parallel sampling (multiple independent samples from same prompt).
    
    Similar to beam search but samples are independent (no pruning).
    """
    
    def __init__(self, allocator: BlockAllocator, block_size: int, hidden_dim: int):
        self.allocator = allocator
        self.block_size = block_size
        self.hidden_dim = hidden_dim
        
        self.samples: Dict[int, PagedKVCache] = {}
        self.next_sample_id = 0
        
        self.num_forks = 0
    
    def create_samples(self, prompt_cache: PagedKVCache, 
                      num_samples: int) -> List[int]:
        """
        Create multiple samples sharing prompt cache.
        
        Args:
            prompt_cache: Shared prompt KV cache
            num_samples: Number of samples to create
            
        Returns:
            List of sample IDs
        """
        sample_ids = []
        
        for _ in range(num_samples):
            sample_id = self.next_sample_id
            self.next_sample_id += 1
            
            # Fork cache for this sample
            sample_cache = prompt_cache.fork()
            self.samples[sample_id] = sample_cache
            sample_ids.append(sample_id)
            self.num_forks += 1
        
        return sample_ids
    
    def get_sample_cache(self, sample_id: int) -> PagedKVCache:
        """Get cache for a sample."""
        if sample_id not in self.samples:
            raise ValueError(f"Sample {sample_id} not found")
        return self.samples[sample_id]
    
    def free_sample(self, sample_id: int):
        """Free a sample's resources."""
        if sample_id in self.samples:
            self.samples[sample_id].free_all()
            del self.samples[sample_id]
    
    def free_all(self):
        """Free all samples."""
        for sample_id in list(self.samples.keys()):
            self.free_sample(sample_id)
    
    def get_stats(self) -> Dict:
        """Return sampling statistics."""
        return {
            'num_samples': len(self.samples),
            'num_forks': self.num_forks
        }