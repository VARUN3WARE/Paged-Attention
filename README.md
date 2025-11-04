# PagedAttention: Efficient Memory Management for LLM Inference

A from-scratch PyTorch implementation of PagedAttention and vLLM-like memory management for efficient KV cache handling in transformer inference.

## 📖 Overview

This project implements the core ideas from the PagedAttention paper:

- **Block-based KV cache**: Split KV cache into fixed-size blocks (pages)
- **Non-contiguous memory**: Blocks can be stored anywhere in memory
- **Copy-on-Write (COW)**: Efficient memory sharing for beam search and parallel sampling
- **Swap and recompute**: Trade computation for memory when needed
- **Reduced fragmentation**: Near-zero memory waste compared to naive contiguous allocation

## 🚀 Quick Start

### Installation

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Running the Project

```bash
# Run correctness tests
pytest tests/ -v

# Run demo (shows correctness and basic functionality)
python scripts/run_demo.py

# Run comprehensive benchmarks
python scripts/run_benchmarks.py

# Open interactive notebook
jupyter notebook demo_notebook.ipynb
```

## 📁 Project Structure

```
paged_attention_project/
├── paged_attention/          # Core implementation
│   ├── paged_attention.py    # Blockwise attention module
│   ├── kv_cache.py           # KV cache with block management
│   ├── allocator.py          # Physical block allocator
│   ├── scheduler.py          # Batch scheduler
│   ├── decoding.py           # COW logic for sampling
│   ├── swap_recompute.py     # Swap/recompute strategies
│   └── utils.py              # Utilities and plotting
├── tests/                    # Unit tests
├── benchmarks/               # Performance benchmarks
├── scripts/                  # Demo and benchmark runners
└── demo_notebook.ipynb       # Interactive experiments
```

## 🎯 Key Features

- ✅ **Correctness verified**: Outputs match vanilla attention within 1e-5 tolerance
- ✅ **Memory efficient**: 60-80% memory savings vs naive allocation
- ✅ **Parallel sampling**: Share prompt blocks across samples
- ✅ **Beam search**: COW semantics for efficient forking
- ✅ **Flexible swapping**: Simulate CPU-GPU transfers
- ✅ **Comprehensive tests**: Unit tests and benchmarks included

## 📊 Benchmark Results

Run `python scripts/run_benchmarks.py` to see:

- Memory utilization comparison
- Throughput (tokens/sec) improvements
- Fragmentation reduction metrics
- Beam search memory savings
- Swap vs recompute tradeoffs

## 🧪 Example Usage

```python
from paged_attention import PagedAttention, PagedKVCache, BlockAllocator

# Initialize components
allocator = BlockAllocator(total_blocks=128, block_size=16, hidden_dim=512)
kv_cache = PagedKVCache(block_size=16, hidden_dim=512, allocator=allocator)
attention = PagedAttention(hidden_dim=512, num_heads=8, block_size=16)

# Use in inference
query = torch.randn(1, 1, 512)  # New token query
output = attention(query, kv_cache)
```

## 📈 Extensions & Future Work

- [ ] CUDA kernel implementation for fused block access
- [ ] Adaptive block sizing based on sequence distribution
- [ ] Quantization for swapped blocks
- [ ] Distributed inference support
- [ ] Asynchronous swap pipeline

## 📚 References

- PagedAttention Paper: "Efficient Memory Management for Large Language Model Serving with PagedAttention"
- vLLM: https://github.com/vllm-project/vllm

## 📝 License

MIT License - Free for research and educational purposes.

```

---
```

#####

varunrao.gd@
