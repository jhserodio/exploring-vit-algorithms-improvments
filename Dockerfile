# Native NVIDIA CUDA Rust research environment on Blackwell RTX 5070 Ti.
# CUDA Tile Rust requires CUDA 13.3+; host driver must support that runtime.
FROM nvidia/cuda:13.3.0-devel-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive \
    CUDA_COMPUTE_CAP=120 \
    CUDA_HOME=/usr/local/cuda \
    CUDA_TOOLKIT_PATH=/usr/local/cuda \
    RUSTUP_HOME=/usr/local/rustup \
    CARGO_HOME=/usr/local/cargo
ENV PATH="${CUDA_HOME}/bin:${CARGO_HOME}/bin:${PATH}"
ENV LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}"

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates curl git build-essential cmake pkg-config libssl-dev \
    clang llvm-dev libopenblas-dev liblapack-dev \
 && rm -rf /var/lib/apt/lists/*

RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain 1.100.0 \
 && rustup component add --toolchain 1.100.0 rustfmt clippy

WORKDIR /workspace/motor_inferencia
# CPU reference tests are the default; GPU prototype runs in vit-gpu service.
CMD ["cargo", "test"]
