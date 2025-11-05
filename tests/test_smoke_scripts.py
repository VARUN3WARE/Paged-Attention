"""Smoke tests: run demo and benchmarks in fast/headless mode to ensure scripts don't crash.

These tests are intentionally fast and headless so they can run in CI.
"""

import subprocess
import sys
import os


def run_cmd(cmd, env=None):
    print('Running:', ' '.join(cmd))
    res = subprocess.run(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    print(res.stdout)
    if res.returncode != 0:
        raise RuntimeError(f"Command failed: {cmd}\nOutput:\n{res.stdout}")


def test_run_demo_headless(tmp_path):
    # Run demo with non-interactive backend
    env = os.environ.copy()
    env['MPLBACKEND'] = 'Agg'
    cmd = [sys.executable, 'scripts/run_demo.py']
    run_cmd(cmd, env=env)


def test_run_benchmarks_fast_headless(tmp_path):
    env = os.environ.copy()
    env['MPLBACKEND'] = 'Agg'
    outdir = str(tmp_path / 'bench_out')
    cmd = [sys.executable, 'scripts/run_benchmarks.py', '--fast', '--headless', '--outdir', outdir]
    run_cmd(cmd, env=env)
