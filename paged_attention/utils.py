"""
Utility functions for workload generation, metrics, and visualization.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple
import torch


def generate_synthetic_workload(num_requests: int, 
                                mean_prompt_len: int = 100,
                                mean_output_len: int = 50,
                                prompt_std: int = 30,
                                output_std: int = 20) -> List[Tuple[int, int]]:
    """
    Generate synthetic workload with varying sequence lengths.
    
    Args:
        num_requests: Number of requests to generate
        mean_prompt_len: Mean prompt length
        mean_output_len: Mean output length
        prompt_std: Standard deviation of prompt lengths
        output_std: Standard deviation of output lengths
        
    Returns:
        List of (prompt_len, output_len) tuples
    """
    workload = []
    
    for _ in range(num_requests):
        prompt_len = max(1, int(np.random.normal(mean_prompt_len, prompt_std)))
        output_len = max(1, int(np.random.normal(mean_output_len, output_std)))
        workload.append((prompt_len, output_len))
    
    return workload


def compute_memory_metrics(total_allocated: int, total_used: int) -> Dict:
    """
    Compute memory utilization metrics.

    Args:
        total_allocated: Total allocated memory in bytes
        total_used: Actually used memory in bytes
        
    Returns:
        Dictionary with memory metrics
    """
    if total_allocated == 0:
        return {
            'utilization': 0.0,
            'fragmentation': 0.0,
            'wasted_memory': 0,
            'wasted_percentage': 0.0
        }
    
    wasted = total_allocated - total_used
    utilization = total_used / total_allocated
    fragmentation = wasted / total_allocated
    
    return {
        'utilization': utilization * 100,  # Percentage
        'fragmentation': fragmentation * 100,
        'wasted_memory': wasted,
        'wasted_percentage': fragmentation * 100
    }


def plot_memory_usage(timestamps: List[float], 
                     naive_memory: List[int],
                     paged_memory: List[int],
                     title: str = "Memory Usage Over Time",
                     save_path: str = None):
    """
    Plot memory usage comparison between naive and paged approaches.
    
    Args:
        timestamps: Time points
        naive_memory: Memory usage for naive approach (bytes)
        paged_memory: Memory usage for paged approach (bytes)
        title: Plot title
        save_path: Optional path to save figure
    """
    plt.figure(figsize=(10, 6))
    
    # Convert bytes to MB
    naive_mb = [m / (1024 * 1024) for m in naive_memory]
    paged_mb = [m / (1024 * 1024) for m in paged_memory]
    
    plt.plot(timestamps, naive_mb, label='Naive (Contiguous)', 
             linewidth=2, marker='o', markersize=4)
    plt.plot(timestamps, paged_mb, label='Paged (Blocks)', 
             linewidth=2, marker='s', markersize=4)
    
    plt.xlabel('Time (steps)', fontsize=12)
    plt.ylabel('Memory Usage (MB)', fontsize=12)
    plt.title(title, fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_throughput(batch_sizes: List[int],
                   naive_throughput: List[float],
                   paged_throughput: List[float],
                   title: str = "Throughput Comparison",
                   save_path: str = None):
    """
    Plot throughput comparison.
    
    Args:
        batch_sizes: Batch sizes tested
        naive_throughput: Tokens/sec for naive approach
        paged_throughput: Tokens/sec for paged approach
        title: Plot title
        save_path: Optional path to save figure
    """
    plt.figure(figsize=(10, 6))
    
    plt.plot(batch_sizes, naive_throughput, label='Naive', 
             linewidth=2, marker='o', markersize=8)
    plt.plot(batch_sizes, paged_throughput, label='Paged', 
             linewidth=2, marker='s', markersize=8)
    
    plt.xlabel('Batch Size', fontsize=12)
    plt.ylabel('Throughput (tokens/sec)', fontsize=12)
    plt.title(title, fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_fragmentation(block_sizes: List[int],
                      fragmentation_percentages: List[float],
                      title: str = "Memory Fragmentation vs Block Size",
                      save_path: str = None):
    """
    Plot fragmentation vs block size.
    
    Args:
        block_sizes: Block sizes tested
        fragmentation_percentages: Fragmentation % for each size
        title: Plot title
        save_path: Optional path to save figure
    """
    plt.figure(figsize=(10, 6))
    
    plt.bar(range(len(block_sizes)), fragmentation_percentages, 
            color='steelblue', alpha=0.7)
    plt.xticks(range(len(block_sizes)), block_sizes)
    
    plt.xlabel('Block Size', fontsize=12)
    plt.ylabel('Internal Fragmentation (%)', fontsize=12)
    plt.title(title, fontsize=14)
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_swap_vs_recompute(block_sizes: List[int],
                          swap_times: List[float],
                          recompute_times: List[float],
                          title: str = "Swap vs Recompute Overhead",
                          save_path: str = None):
    """
    Plot swap vs recompute time comparison.
    
    Args:
        block_sizes: Block sizes tested
        swap_times: Time to swap for each size (ms)
        recompute_times: Time to recompute for each size (ms)
        title: Plot title
        save_path: Optional path to save figure
    """
    plt.figure(figsize=(10, 6))
    
    x = np.arange(len(block_sizes))
    width = 0.35
    
    plt.bar(x - width/2, swap_times, width, label='Swap', alpha=0.8)
    plt.bar(x + width/2, recompute_times, width, label='Recompute', alpha=0.8)
    
    plt.xlabel('Block Size', fontsize=12)
    plt.ylabel('Time (ms)', fontsize=12)
    plt.title(title, fontsize=14)
    plt.xticks(x, block_sizes)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_beam_search_memory(beam_widths: List[int],
                            naive_memory: List[float],
                            paged_memory: List[float],
                            title: str = "Beam Search Memory Usage",
                            save_path: str = None):
    """
    Plot memory usage for beam search with different widths.
    
    Args:
        beam_widths: Beam widths tested
        naive_memory: Memory for naive approach (MB)
        paged_memory: Memory for paged approach with COW (MB)
        title: Plot title
        save_path: Optional path to save figure
    """
    plt.figure(figsize=(10, 6))
    
    plt.plot(beam_widths, naive_memory, label='Naive (Full Copy)', 
             linewidth=2, marker='o', markersize=8)
    plt.plot(beam_widths, paged_memory, label='Paged (COW)', 
             linewidth=2, marker='s', markersize=8)
    
    plt.xlabel('Beam Width', fontsize=12)
    plt.ylabel('Memory Usage (MB)', fontsize=12)
    plt.title(title, fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def create_causal_mask(seq_len: int, device: str = 'cpu') -> torch.Tensor:
    """
    Create causal attention mask.
    
    Args:
        seq_len: Sequence length
        device: Device for tensor
        
    Returns:
        Mask tensor [seq_len, seq_len]
    """
    mask = torch.tril(torch.ones(seq_len, seq_len, device=device))
    return mask


def generate_random_embeddings(batch_size: int, seq_len: int, 
                              hidden_dim: int, device: str = 'cpu') -> torch.Tensor:
    """
    Generate random embeddings for testing.
    
    Args:
        batch_size: Batch size
        seq_len: Sequence length
        hidden_dim: Hidden dimension
        device: Device for tensor
        
    Returns:
        Random tensor [batch_size, seq_len, hidden_dim]
    """
    return torch.randn(batch_size, seq_len, hidden_dim, device=device)


def calculate_attention_flops(batch_size: int, seq_len: int, 
                             hidden_dim: int, num_heads: int) -> int:
    """
    Calculate FLOPs for attention computation.
    
    Args:
        batch_size: Batch size
        seq_len: Sequence length
        hidden_dim: Hidden dimension
        num_heads: Number of attention heads
        
    Returns:
        Estimated FLOPs
    """
    head_dim = hidden_dim // num_heads
    
    # Q @ K^T: batch * num_heads * seq_len * seq_len * head_dim
    qk_flops = batch_size * num_heads * seq_len * seq_len * head_dim
    
    # Softmax: negligible compared to matmuls
    
    # Attn @ V: batch * num_heads * seq_len * seq_len * head_dim
    av_flops = batch_size * num_heads * seq_len * seq_len * head_dim
    
    # Projections: batch * seq_len * hidden_dim * hidden_dim (for Q, K, V, Out)
    proj_flops = 4 * batch_size * seq_len * hidden_dim * hidden_dim
    
    return qk_flops + av_flops + proj_flops


class MemoryTracker:
    """
    Tracks memory usage over time for visualization.
    """
    
    def __init__(self):
        self.timestamps = []
        self.memory_usage = []
        self.num_tokens = []
        self.num_blocks = []
    
    def record(self, timestamp: float, memory_bytes: int, 
              num_tokens: int = 0, num_blocks: int = 0):
        """Record a memory measurement."""
        self.timestamps.append(timestamp)
        self.memory_usage.append(memory_bytes)
        self.num_tokens.append(num_tokens)
        self.num_blocks.append(num_blocks)
    
    def get_data(self) -> Dict:
        """Get tracked data."""
        return {
            'timestamps': self.timestamps,
            'memory_usage': self.memory_usage,
            'num_tokens': self.num_tokens,
            'num_blocks': self.num_blocks
        }
    
    def plot(self, title: str = "Memory Usage", save_path: str = None):
        """Plot tracked memory usage."""
        plt.figure(figsize=(10, 6))
        
        memory_mb = [m / (1024 * 1024) for m in self.memory_usage]
        
        plt.plot(self.timestamps, memory_mb, linewidth=2, marker='o', markersize=4)
        plt.xlabel('Time (steps)', fontsize=12)
        plt.ylabel('Memory Usage (MB)', fontsize=12)
        plt.title(title, fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()


def format_bytes(bytes_val: int) -> str:
    """
    Format bytes to human-readable string.
    
    Args:
        bytes_val: Number of bytes
        
    Returns:
        Formatted string (e.g., "1.5 MB")
    """
    for unit in ['B', 'KB', 'MB', 'GB']:
        if bytes_val < 1024.0:
            return f"{bytes_val:.2f} {unit}"
        bytes_val /= 1024.0
    return f"{bytes_val:.2f} TB"


def print_stats_table(stats: Dict, title: str = "Statistics"):
    """
    Pretty print statistics table.
    
    Args:
        stats: Dictionary of statistics
        title: Table title
    """
    print(f"\n{'='*60}")
    print(f"{title:^60}")
    print(f"{'='*60}")
    
    for key, value in stats.items():
        key_formatted = key.replace('_', ' ').title()
        
        if isinstance(value, float):
            if value < 1:
                print(f"{key_formatted:<40} {value:.6f}")
            else:
                print(f"{key_formatted:<40} {value:.2f}")
        elif isinstance(value, int) and key.endswith('bytes'):
            print(f"{key_formatted:<40} {format_bytes(value)}")
        else:
            print(f"{key_formatted:<40} {value}")
    
    print(f"{'='*60}\n")