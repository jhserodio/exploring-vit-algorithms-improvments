# Phase 1 — Native CUDA Rust and complete DINOv2 forward

**Status:** IN PROGRESS. V0 was closed with original checkpoint numerical
parity, but the complete DINOv2 Base CUDA Rust forward has **not yet passed**
GPU execution or per-layer numerical equivalence. See the
[technical plan](docs/phase1-design.md) and [canonical roadmap](roadmap.md).

## CPU operation contract and safety tests

```bash
cd motor_inferencia
cargo test
cargo fmt --all -- --check
cargo clippy --all-targets -- -D warnings
```

`operation_contract.rs` checks shapes, dtypes, layouts and explicit
CPU/GPU residency for GEMM, residual, softmax, LayerNorm and GELU. This
is *not* proof the GPU model is implemented.

## Native CUDA 13.3 source compilation (without GPU execution)

```bash
cargo check --manifest-path cuda-kernels/Cargo.toml --all-targets
```

The separate CUDA source CI is informative. Compilation alone cannot
validate the JIT kernel, driver runtime or RTX 5070 Ti performance.

## Run isolated kernels on the local RTX 5070 Ti

```bash
bash scripts/v1-gpu-acceptance.sh results/v1-gpu
# or
docker compose --profile gpu run --build --rm vit-gpu
```

The operation probe includes residual addition, last-axis softmax and
a tiled GEMM. These kernels are initial fixtures: they do not yet cover
the 1,370-token masked attention or whole-model execution.

The acceptance harness fails if NVIDIA/driver/CUDA prerequisites are
missing; logs and environment summaries remain local under `results/`.

## Blocking gates

- [ ] CUDA source build and typecheck succeeds.
- [ ] All initial probes execute on the actual RTX 5070 Ti.
- [ ] Complete LayerNorm/GELU, projection and attention kernels are validated.
- [ ] All 12 DINOv2 blocks execute with original weights on the GPU.
- [ ] CPU versus GPU full hidden-state and per-layer numerical parity passes.
- [ ] GPU memory, cold/warm timing and transfer accounting are captured.

No task above is considered complete based solely on CPU CI or code review.
