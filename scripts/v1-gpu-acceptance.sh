#!/usr/bin/env bash
# Phase 1 acceptance on owner's NVIDIA GPU; NO CPU fallback.
set -euo pipefail
cd "$(dirname "$0")/.."
out="${1:-results/v1-gpu}"
mkdir -p "$out"
command -v nvidia-smi >/dev/null || { echo "FAIL: no NVIDIA driver" >&2; exit 2; }
command -v nvcc >/dev/null || { echo "FAIL: CUDA toolkit not installed" >&2; exit 2; }
command -v cargo >/dev/null || { echo "FAIL: Rust/Cargo not installed" >&2; exit 2; }
{
    echo "Run date: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "Git commit: $(git rev-parse HEAD)"
    bash scripts/collect_env.sh
} > "$out/environment.txt"
echo "Running native CUDA Rust GPU probes (residual, softmax, matmul)."
echo "A GPU or JIT failure is a FAIL; output is not proof of full model inference."
cargo run --release --manifest-path cuda-kernels/Cargo.toml -- all \
    2>&1 | tee "$out/kernels.log"
echo "Kernel probes passed. Full DINOv2 CUDA inference is still a separate gate."
