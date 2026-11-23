#!/usr/bin/env bash
set -euo pipefail

# Save this record with each run. No image data or patient identifiers.
echo '=== Git ==='
git rev-parse HEAD 2>/dev/null || true
echo '=== Rust ==='
rustc --version 2>/dev/null || true
cargo --version 2>/dev/null || true
echo '=== NVIDIA GPU and driver ==='
nvidia-smi --query-gpu=name,driver_version,compute_cap,memory.total --format=csv 2>/dev/null || true
echo '=== CUDA toolkit ==='
nvcc --version 2>/dev/null || true
echo '=== CPU ==='
lscpu 2>/dev/null | grep -E 'Architecture:|Model name:|CPU\(s\):' || true
