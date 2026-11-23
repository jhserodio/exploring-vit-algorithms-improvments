# Execution checklist — CUDA Rust migration and spatial features

**Source of truth:** [roadmap.md](roadmap.md). Updated 2026-11-23,
branch `feat/testing-background`.

## Implemented in source (not all validated)

- [x] Remove legacy neural-network runtime imports and dependencies from the new engine.
- [x] Add `ModelAdapter` interface and model-independent `DenseFeatures` output.
- [x] Preserve CLS, 1,369 ordered patches and [768,37,37] spatial map.
- [x] Add patch indexing, shape and CHW transpose tests.
- [x] Retain CPU resize/crop geometry and coordinate round-trip tests.
- [x] Implement FP32 original DINOv2 Base loading/CPU forward as reference.
- [x] Add native CUDA Rust Tile residual-add operation probe.
- [x] Extend Hugging Face parity script to check CLS and patch tokens.
- [x] Separate CPU CI and local GPU kernel probe.

## V0 completed; open GPU, data and model-generalization gates

- [x] CI green for the standalone CPU implementation: [14-test Rust CPU CI on 2026-11-23](https://github.com/jhserodio/exploring-vit-algorithms-improvments/actions/runs/45242097849).
- [x] Official DINOv2 Base weights load and the full FP32 CPU forward emits finite features — [pinned checkpoint CI](https://github.com/jhserodio/exploring-vit-algorithms-improvments/actions/runs/45245472357).
- [x] CLS, 1,369 patches, complete hidden state and spatial-map parity passed with exact CHW ordering — [validation record](docs/v0-validation.md).
- [ ] CUDA Rust Tile operation probe passes on RTX 5070 Ti with CUDA 13.3+.
- [ ] Full GPU DINOv2 forward implemented using native CUDA Rust.
- [ ] Per-operation and full-model CPU/CUDA numerical comparisons pass.
- [ ] WSI/tile geometry and patient-safe dataset manifests validated.
- [ ] Histopathology backbone (e.g. Hibou-B) loads and matches its own reference.
- [ ] One identical optimized kernel runs in at least two compatible backbones.
- [ ] OSCC task head and region metrics validated against eligible annotations.

`[x]` in the first section means source implementation exists; numerical
and CUDA success require a separate documented execution. Do not report
patient-level quality, kernel speedups or localization based on fixtures.
