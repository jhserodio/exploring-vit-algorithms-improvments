# Phase 0 research environment for NVIDIA Blackwell / GeForce RTX 5070 Ti.
# The host NVIDIA driver must support this CUDA runtime.
FROM nvidia/cuda:12.8.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    CUDA_COMPUTE_CAP=120 \
    CUDA_HOME=/usr/local/cuda \
    RUSTUP_HOME=/usr/local/rustup \
    CARGO_HOME=/usr/local/cargo
ENV PATH="${CUDA_HOME}/bin:${CARGO_HOME}/bin:${PATH}"
ENV LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}"

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates curl git build-essential cmake pkg-config libssl-dev \
    clang llvm-dev libopenblas-dev liblapack-dev \
 && rm -rf /var/lib/apt/lists/*

RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y \
 && rustup component add rustfmt clippy

WORKDIR /workspace/motor_inferencia
CMD ["cargo", "test", "--no-default-features"]
