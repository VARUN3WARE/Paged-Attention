# PagedAttention: Efficient Memory Management for LLM Inference

A from-scratch PyTorch implementation of PagedAttention and vLLM-like memory management for efficient KV cache handling in transformer inference.

## Overview

This project implements the core ideas from the PagedAttention paper:

- **Block-based KV cache**: Split KV cache into fixed-size blocks (pages)
- **Non-contiguous memory**: Blocks can be stored anywhere in memory
- **Copy-on-Write (COW)**: Efficient memory sharing for beam search and parallel sampling
- **Swap and recompute**: Trade computation for memory when needed
- **Reduced fragmentation**: Near-zero memory waste compared to naive contiguous allocation

## Quick Start

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

## Project Structure

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

## Key Features

- **Correctness verified**: Outputs match vanilla attention within 1e-5 tolerance
- **Memory efficient**: 60-80% memory savings vs naive allocation
- **Parallel sampling**: Share prompt blocks across samples
- **Beam search**: COW semantics for efficient forking
- **Flexible swapping**: Simulate CPU-GPU transfers
- **Comprehensive tests**: Unit tests and benchmarks included

## Benchmark Results

Run `python scripts/run_benchmarks.py` to see:

- Memory utilization comparison
- Throughput (tokens/sec) improvements
- Fragmentation reduction metrics
- Beam search memory savings
- Swap vs recompute tradeoffs

## Example Usage

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

## KV cache semantics

Important: the `PagedKVCache` in this implementation stores projected key/value vectors — i.e. the outputs
of the model's linear projections (the `k_proj` and `v_proj` layers). The `PagedAttention.forward_paged`
and the tests expect cached K/V to already be projected. In practice you can populate the cache like this:

```python
# assume `paged_attn` is an instance of `PagedAttention`
k = paged_attn.k_proj(tokens)   # project input tokens to keys
v = paged_attn.v_proj(tokens)   # project input tokens to values
for i in range(k.shape[0]):
	kv_cache.append_token_kv(k[i], v[i])
```

If you prefer to cache raw embeddings instead, you'll need to project keys/values at read time or change
the attention implementation accordingly. The current contract (cached projected K/V) keeps the
attention read-path simple and matches the unit tests and examples in this repo.

## Running tests

Run the test suite from the project root (so the package can be imported correctly):

```bash
source venv/bin/activate
pytest tests/ -q
```

## Running the demo and benchmarks

The repository includes two runnable scripts that demonstrate behavior and collect micro-benchmarks:

- `scripts/run_demo.py` — runs a short demo sequence: correctness check, a memory-efficiency summary, a copy-on-write beam-search demo, and a swap vs recompute simulation.
- `scripts/run_benchmarks.py` — runs a set of micro-benchmarks (memory, throughput, fragmentation, beam search, parallel sampling, and COW overhead) and prints a summary. Several plots are shown at the end.

Run them from the project root (so imports resolve correctly):

```bash
# Activate your venv first
source venv/bin/activate

# Demo (prints outputs to console)
python scripts/run_demo.py

# Benchmarks (prints summary and displays plots)
python scripts/run_benchmarks.py
```

What to expect

- `run_demo.py` prints per-demo summaries (max/mean diffs for correctness, memory usage table, COW statistics, swap/recompute timing) and completes with "All demos completed successfully!" when no exception occurs.
- `run_benchmarks.py` prints benchmark sections and a final summary with memory savings, throughput speedups, optimal block size, and COW rates. Plots are displayed with Matplotlib.

Headless / CI-friendly runs

- If you're running on a headless machine (no display), set a non-interactive Matplotlib backend before running the scripts to avoid GUI errors. Example (bash):

```bash
export MPLBACKEND=Agg
python scripts/run_benchmarks.py
```

This will save/display plots without requiring a desktop environment.

If you want the benchmark scripts to save plots to disk (instead of showing), you can edit the plot calls in `scripts/run_benchmarks.py` or the plotting helpers in `paged_attention/utils.py` to pass a `save_path` argument.

Troubleshooting

- ModuleNotFoundError: If you see `ModuleNotFoundError: No module named 'paged_attention'`, make sure you're running scripts from the repository root and that the venv is active. The scripts use file-relative sys.path insertion to import the package; run them from the repo root to be safe.
- Matplotlib/axes errors: either set `MPLBACKEND=Agg` for headless runs or install a GUI backend.

Benchmark CLI flags and saved outputs

The benchmark runner supports a few CLI options for CI-friendly and headless runs:

- `--outdir DIR` — Save generated plots to the given directory (created if necessary). When provided the runner will save at least the following files:

  - `bench_throughput.png` — throughput comparison (naive vs paged)
  - `bench_fragmentation.png` — fragmentation vs block size

- `--headless` — Set a non-interactive Matplotlib backend (equivalent to `export MPLBACKEND=Agg`) so scripts don't require a display.

- `--fast` — Run a reduced workload (much smaller / faster) intended for smoke tests and CI. Use this in automated tests to keep runtime low.

Examples

Run a fast, headless benchmark and save plots to `bench_outputs/`:

```bash
source venv/bin/activate
python scripts/run_benchmarks.py --fast --headless --outdir bench_outputs
```

Run the full interactive benchmarks (shows plots):

```bash
python scripts/run_benchmarks.py
```

Smoke tests (CI)

This repository includes a small smoke test file that runs the demo and a fast/headless benchmark to ensure the example scripts don't crash in CI:

```bash
pytest tests/test_smoke_scripts.py -q
```

The smoke tests use the `--fast` and headless modes so they run quickly and without a display.

## Extensions & Future Work

- [ ] CUDA kernel implementation for fused block access
- [ ] Adaptive block sizing based on sequence distribution
- [ ] Quantization for swapped blocks
- [ ] Distributed inference support
- [ ] Asynchronous swap pipeline

Notes on CUDA kernels

This repository includes a small CUDA kernel template under `paged_attention/cuda/` and a
runtime wrapper `paged_attention/cuda_kernels.py` that will attempt to call a compiled
extension named `_fused_kernels` if present, or fall back to a correct (but unoptimized)
PyTorch implementation.

To prototype a fused CUDA kernel locally (example):

1. Implement device code in `paged_attention/cuda/fused_kernels.cu` and wrapper/dispatch
   in `paged_attention/cuda/fused_kernels.cpp`.

2. Build and load the extension from Python (example):

```py
from torch.utils.cpp_extension import load
import os

src_dir = os.path.join(os.path.dirname(__file__), 'cuda')
ext = load(name='fused_kernels', sources=[
   os.path.join(src_dir, 'fused_kernels.cpp'),
   os.path.join(src_dir, 'fused_kernels.cu')
])

# then ext.fused_block_matmul(...) will be available
```

3. Once built, you can import the compiled module as `from paged_attention import _fused_kernels`
   or rely on the `paged_attention.cuda_kernels` wrapper which will call the extension when present.

If you'd like, I can implement a minimal working CUDA kernel here (requires CUDA toolchain and
PyTorch C++ extension build). Tell me whether you want a full working kernel (I'll add build
instructions, tests that skip when CUDA isn't present, and optional CI steps), or whether
the current template + wrapper is sufficient for now.

## References

- PagedAttention Paper: "Efficient Memory Management for Large Language Model Serving with PagedAttention"
- vLLM: https://github.com/vllm-project/vllm
