# Roadmap reassessment — OSCC ViT algorithm laboratory

**Date:** 2026-11-23. **Branch:** `feat/testing-background`.
**Goal:** reusable computational improvements for compatible visual models,
with spatially traceable OSCC research under valid patient-level annotations.

## Findings and corrections

| Finding | Risk | Change |
| --- | --- | --- |
| Phase 1 combined CUDA setup, kernels, full 12-block inference and parity | A partial kernel could be misrepresented as a working model | Divide V1 into acceptance gates V1.A–V1.F |
| CPU CI validated model features, not GPU kernels | Invalid GPU performance/correctness claims | Separate CUDA 13.3 source build CI from real RTX 5070 Ti execution |
| Operation implementations tightly coupled to original architecture | Hard to experiment with another model | Add `KernelSpec` shape/dtype/layout/device validation |
| 1,370-token attention needs masked padding in Tile kernels | Incorrect normalization or lesion feature drift | Require full-row softmax, masking and per-layer parity before integration |
| OSCC dataset audit came after full GPU forward | Delay in verifying meaningful clinical task and slide geometry | Run dataset/annotation audit in parallel with GPU bring-up |
| Adapter portability was implied, not demonstrated | False claim of model-agnostic benefit | Require identical kernel execution with a second compatible backbone |
| Math tests mixed with scientific OSCC evaluation | Confuses numeric equivalence with lesion recognition | Distinguish operator parity, benchmark and patient-level OSCC quality |
| Performance, logging and environment capture arrived late | Irreproducible speedups | Record hashes/versions from first GPU kernel, and isolate H2D/D2H timing |
| Initial CPU container required a CUDA development image | Unnecessary resources in CPU CI and workstation | Separate `Dockerfile.cpu` and GPU `Dockerfile` |

## Critical path and parallelism

```text
V0 verified CPU oracle
 ├── V1.A contracts → V1.B CUDA build → V1.C GPU operation parity
 │                              → V1.D full operation coverage
 │                              → V1.E full-model GPU forward
 │                              → V1.F per-layer + final parity
 └── Phase 2 OSCC dataset audit and spatial registration (parallel)

V1.F + Phase 3 second backbone + Phase 4 ablation rigor
    → controlled transferable algorithm experiments
Phase 2 + Phase 5 supervised task head
    → valid OSCC classification/location metrics
Phase 6 performance, Phase 7 domain robustness, Phase 8 reproducibility
    → scientifically defensible comparisons
Phase 9 agent orchestration remains strictly optional.
```

## Scientific non-negotiables

- A feature map is not a probability map, mask or OSCC diagnosis.
- Keep the original checkpoint fixed while comparing kernels; an alternative
  pathology backbone is a separate experiment with its own native preprocess.
- For identical-model kernel ablations, compare CLS **and every patch token**;
  preserve the spatial/WSI coordinate mapping.
- No automatic CPU execution when GPU cannot initialize.
- Allow CPU RGB decode and tile prefetch; keep all intermediate transformer
  activations GPU-resident for a declared GPU model run.
- Do not rely on tile-level independent confidence intervals when samples come
  from the same patient or slide.
- Never close GPU gates with a CPU-only runner, a source-build result or a
  GPU kernel that does not perform a full DINOv2 inference.

## Acceptance policies

A milestone is **implemented** when code/config/documents exist. It is
**verified** only when a run ID, exact Git commit, workload, hardware, numerical
error and test result exist. It is **closed** only when every blocking acceptance
gate has passed. Failures and unsupported targets remain recorded rather than
removed from the denominator.

See [full roadmap](../roadmap.md),
[V1 operation architecture](phase1-design.md), and
[passed V0 checkpoint parity](v0-validation.md).
