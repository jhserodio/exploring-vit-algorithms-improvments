# Fase 1 — CUDA Rust: architecture, tests and acceptance gates

**Scope:** The official DINOv2 Base FP32 518px CPU reference passed full
Hugging Face parity in [V0](v0-validation.md). V1 implements reusable CUDA Rust
kernels and the complete GPU model, retaining `[1,1370,768]` hidden features
and `[1,768,37,37]` spatial map.

## Acceptance stages

| Gate | Deliverable | Required evidence |
| --- | --- | --- |
| V1.A | Runtime-independent op contracts (shapes, dtype, layout, device) | CPU CI |
| V1.B | Native CUDA Rust source builds with CUDA 13.3 | [Passed source CI](https://github.com/jhserodio/exploring-vit-algorithms-improvments/actions/runs/45250132345); NO GPU claim |
| V1.C | Residual, stable softmax and GEMM probes on RTX 5070 Ti | Recorded GPU output, driver, toolkit and Git SHA |
| V1.D | Remaining FP32 LayerNorm, exact GELU, patch projection, attention | CPU/GPU operator fixtures and errors |
| V1.E | Complete 12-block DINOv2 Base forward on device | Original checkpoint and normalized input, no CPU fallback |
| V1.F | CLS, all spatial patches, each block and full model parity | max_abs/relative_L2, tolerance, VRAM and latency |

**V1 is NOT complete until V1.F passes.** Compiling code or running isolated
kernels does not prove that a full model executes on GPU.

## Operation contracts

The same `motor_inferencia/src/operation_contract.rs` module is imported
by CPU tests and the GPU probe; its supported operations are GEMM,
residual add, last-axis softmax, LayerNorm and GELU. The contract rejects
mismatched shape, dtype and device. GPU calls must never fall back to CPU.

The first GPU source probes (residual/softmax/GEMM) use
`cutile = "=0.5.0"`, NVIDIA CUDA Rust Tile. They are NOT yet connected
to the 12-block forward, and they do NOT yet implement production-length
(1,370-token) masked attention softmax. Padding and a full-row normalization
strategy are mandatory before substituting attention.

## GPU model execution

- Model: `facebook/dinov2-base`, pinned weights from the V0 report
- Image: one RGB input preprocessed into `[1,3,518,518]` FP32
- Patch projection: 1,369 token vectors, dimension 768
- CLS + positional embeddings: 1,370 tokens
- 12 encoder blocks: LN → 12-head Q/K/V attention (64/head) → projection →
  residual LayerScale → LN → GELU MLP (768→3072→768) → residual LayerScale
- Final layernorm and CLS/CHW spatial export
- GPU memory: keep all internal activations on GPU; transfer only input and
  requested outputs, not intermediate layer states in normal inference

Use the CPU for WSI/tile reading, RGB decode, normalization, geometry and
bounded batch prefetch. Use GPU for weights, forward operations and output
feature extraction. The RTX 5070 Ti has a 16 GB VRAM budget, but real memory
usage must be measured.

## Repeat the operation-only checks locally

```bash
bash scripts/v1-gpu-acceptance.sh results/v1-gpu
docker compose --profile gpu run --build --rm vit-gpu
```

The first script records Git SHA, CPU, GPU, driver and CUDA toolkit to local
files and fails if runtime prerequisites are missing. **These scripts only
run isolated kernels**; a further GPU full-checkpoint command and reference
parity report are blocking for closing the phase.

## Dependencies / research ordering

The OSCC data label and magnification audit (Phase 2) can proceed in
parallel with CUDA optimization. Comparing more than one backbone (Phase 3)
should happen before declaring a model-agnostic algorithm. Independent
operator parity and precision studies (Phase 4) expand the V1 blocking checks.
Subsequent patient-safe task training and metrics are separate from runtime
correctness. An unlabeled feature map is not a tumor mask or a clinical
diagnostic result.
