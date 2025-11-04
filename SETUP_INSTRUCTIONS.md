# PagedAttention Project - Setup & Run Instructions

## Prerequisites

- Python 3.10 or higher
- pip package manager
- (Optional) CUDA-compatible GPU for GPU tests

## Installation Steps

### 1. Clone or Create Project Directory

```bash
mkdir paged_attention_project
cd paged_attention_project
```

### 2. Create Virtual Environment

```bash
python -m venv venv

# On Linux/Mac:
source venv/bin/activate

# On Windows:
venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Verify Installation

```bash
python -c "import torch; print(f'PyTorch {torch.__version__} installed')"
python -c "import paged_attention; print('✓ Package imports successfully')"
```

## Running the Project

### Run All Tests

```bash
# Run all unit tests
pytest tests/ -v

# Run specific test file
pytest tests/test_correctness.py -v
```

### Run Demo

```bash
# Run complete demo (shows all features)
python scripts/run_demo.py
```

Expected output:

- ✓ Correctness test passed
- Memory efficiency comparison
- COW demonstration
- Swap vs recompute analysis

### Run Benchmarks

```bash
# Run comprehensive benchmarks
python scripts/run_benchmarks.py
```

This will:

- Test memory usage
- Measure throughput
- Analyze fragmentation
- Benchmark beam search
- Measure COW overhead

Runtime: ~5-10 minutes

### Run Jupyter Notebook

```bash
# Start Jupyter
jupyter notebook

# Open demo_notebook.ipynb in browser
# Run all cells to see interactive experiments
```

## Project Structure

```
paged_attention_project/
├── paged_attention/       # Core implementation
│   ├── paged_attention.py # Attention module
│   ├── kv_cache.py        # KV cache management
│   ├── allocator.py       # Block allocator
│   ├── scheduler.py       # Batch scheduler
│   ├── decoding.py        # COW logic
│   ├── swap_recompute.py  # Memory management
│   └── utils.py           # Utilities
├── tests/                 # Unit tests
├── benchmarks/            # Performance benchmarks
├── scripts/               # Demo and benchmark runners
└── demo_notebook.ipynb    # Interactive notebook
```

## Expected Results

### Tests

- All tests should pass
- Correctness: max difference < 1e-5
- Allocator: no memory leaks
- COW: proper sharing and isolation

### Benchmarks

- Memory savings: 60-80%
- Beam search savings: 40-70%
- Throughput: competitive or better
- Optimal block size: 16 tokens

## Troubleshooting

### Import Errors

If you see `ModuleNotFoundError: No module named 'paged_attention'`:

```bash
# Add project to Python path
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Or run from project root with -m flag
python -m pytest tests/
```

### CUDA Out of Memory

If GPU tests fail with OOM:

- Reduce batch sizes in benchmarks
- Use smaller hidden_dim (256 instead of 512)
- Run on CPU (slower but works)

### Slow Performance

- First run is slower (JIT compilation)
- Use GPU for best performance
- Reduce num_iterations in benchmarks

## Next Steps

1. **Experiment**: Modify hyperparameters in demo_notebook.ipynb
2. **Extend**: Add custom features (see README for ideas)
3. **Optimize**: Profile and optimize hot paths
4. **Deploy**: Integrate into your inference pipeline

## Support

For issues or questions:

1. Check test outputs for specific errors
2. Review code comments and docstrings
3. Consult PagedAttention paper for algorithm details

## Citation

If you use this implementation in your research:

```bibtex
@inproceedings{pagedattention2023,
  title={Efficient Memory Management for Large Language Model Serving with PagedAttention},
  author={Kwon et al.},
  year={2023}
}
```
