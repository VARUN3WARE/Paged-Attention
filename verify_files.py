import os
from pathlib import Path

required_files = {
    'paged_attention/__init__.py',
    'paged_attention/paged_attention.py',
    'paged_attention/kv_cache.py',
    'paged_attention/allocator.py',
    'paged_attention/scheduler.py',
    'paged_attention/decoding.py',
    'paged_attention/swap_recompute.py',
    'paged_attention/utils.py',
}

missing_files = []
for file in required_files:
    if not Path(file).exists():
        missing_files.append(file)
        print(f"❌ Missing: {file}")
    else:
        size = Path(file).stat().st_size
        print(f"✓ Found: {file} ({size} bytes)")

if missing_files:
    print(f"\n⚠️  Missing {len(missing_files)} files!")
    print("Missing files:", missing_files)
else:
    print("\n✓ All required files present!")
