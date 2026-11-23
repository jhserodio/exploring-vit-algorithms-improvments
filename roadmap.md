# ViT Research Engine — CUDA Rust / Spatial OSCC Research Roadmap

**Branch:** `feat/testing-background`  
**Research:** mathematically controlled improvements to reusable Vision Transformer
operations, and their effect on histopathological patterns of oral squamous cell
carcinoma (OSCC).  
**Hardware target:** RTX 5070 Ti (`sm_120`), Ryzen 7 7700X3D-class CPU
(actual host details to confirm).  
**Implementation decision (2026-11-23):** use a standalone Rust/ndarray CPU
reference and NVIDIA CUDA Rust Tile kernels (`cutile-rs`) on CUDA 13.3+.
The previous runtime implementation remains accessible in Git history. **Do not claim that the full model runs on CUDA yet.**

The full model/operation source is intentionally under active refactor.
A completed source-code checkbox does not imply model-to-reference parity,
actual CUDA execution, clinical validity or a verified performance improvement.

**Roadmap audit (2026-11-23):** Phase 1 has been split into six
acceptance gates instead of combining kernel creation and full GPU inference
in one unchecked item. Dataset auditing is now explicitly parallel to GPU
bring-up. No GPU validation is inferred from CPU CI.

## 1. Scientific questions

1. **Kernel equivalence:** can native CUDA Rust improve execution of mathematically
   equivalent operations (matmul, residual add, LayerNorm, attention) while
   preserving numeric results?
2. **Backbone portability:** can the *same* kernel implement a compatible operator
   on multiple ViTs without rewriting the kernel? Capabilities and differences
   (QKV layout, register tokens, positional encoding, dtype, shape) are explicit.
3. **OSCC task quality:** do any proposed changes preserve region and lesion
   coverage under the actual available labels, patient-safe dataset split and
   imaging scale? Feature maps are not tumor masks.
4. **CPU/GPU orchestration:** can a bounded Ryzen tile/decode/normalization pipeline
   feed the RTX 5070 Ti without transferring model intermediates to host?

We must keep separate (a) same model+weights, alternate kernel, (b) different
backbones, and (c) different end-to-end CPU/GPU pipelines.

## 2. Runtime architecture

```text
ExperimentConfig / manifests / hashes / grouped patient splits
                  |
CPU: WSI/tile decode, provenance, geometry, prefetch
                  |
            ModelAdapter registry
            /               \
    DINOv2 Base         Pathology ViT adapter (planned)
            \               /
       Model-specific preprocessing and typed DenseFeatures
       CLS / patch tokens / CHW spatial map / geometry
                  |
       OperationBackend capability dispatch
         /                       \
Rust CPU reference           NVIDIA CUDA Rust (Tile)
         \                       /
   Per-operation parity, GPU timing, peak VRAM
                  |
  TaskHead (only if trained with eligible OSCC labels)
                  |
      Region/patient metrics and experiment reports
```

**API expectations:** ModelAdapter owns config/checkpoint/preprocessing/register
handling; OperationBackend owns mathematical semantics, dtype, layouts, device
and stream. The same operation must reject incompatible shapes or capabilities
rather than silently falling back. Implement these boundaries incrementally,
not as a speculative full framework.

## 3. Implementation phases

### V0 — DINOv2 Base FP32 reference and spatial feature contract [P0]

- [x] Remove legacy neural-network runtime dependencies and source imports.
- [x] Implement `DenseFeatures` for CLS, row-major patch tokens and CHW grid.
- [x] Implement 518px CPU image preprocessing and crop/source geometry.
- [x] Implement standalone FP32 CPU DINOv2 Base forward with original
      `safetensors` loader (source-level implementation only).
- [x] Add deterministic synthetic tests for CLS/patch/map ordering and
      geometric mappings (test code implemented).
- [x] Extend the independent HF parity harness to all 1,370 normalized tokens.
- [x] New CPU CI (10 unit + 4 CLI tests, fmt, Clippy, Python help) passed on [14-test Rust CPU CI on 2026-11-23](https://github.com/jhserodio/exploring-vit-algorithms-improvments/actions/runs/45242097849).
- [x] Full original `facebook/dinov2-base` checkpoint loaded and completed finite CPU inference: [pinned official checkpoint CI](https://github.com/jhserodio/exploring-vit-algorithms-improvments/actions/runs/45245472357).
- [x] Validated CLS, all 1,369 patch tokens and the CHW spatial map against identical-input Hugging Face reference; max absolute error 3.071e-4 (limit 1e-3): [numerical evidence](https://github.com/jhserodio/exploring-vit-algorithms-improvments/actions/runs/45245472357).

**V0 exit gate: PASSED (2026-11-23).** A real, revision-pinned official checkpoint was loaded and its full 1,370-token output was verified numerically against the Hugging Face implementation. Only the **test image** is synthetic; no clinical validation is claimed. See [V0 validation record](docs/v0-validation.md).

### Phase 1 — Native CUDA Rust: operation kernels, full DINOv2 and GPU parity [P0]

**Status:** IN PROGRESS. See [Phase 1 architecture and acceptance matrix](docs/phase1-design.md).
This phase cannot be closed on CPU-only CI. It has six independently evidenced
gates, and **the entire original model must run on the RTX 5070 Ti**.

**V1.A — Runtime-independent operation contracts**
- [x] Implement model-independent FP32 operation shape, dtype, layout and
      device validation for GEMM, residual, full-row softmax, LayerNorm and GELU.
- [x] Add synthetic tests for valid and invalid ViT tensor dimensions and
      explicit rejection of CPU/GPU mismatch (source/tests; CPU CI gate).
- [ ] Wire the operation contracts to a complete on-device execution interface
      with device buffers, streams, capability errors and no implicit fallback.

**V1.B — CUDA Rust source compilation**
- [x] Pin `cutile = "=0.5.0"` and add CUDA 13.3 development container.
- [x] Add native CUDA Rust Tile source for elementwise residual, stable
      last-axis softmax and tiled GEMM (unverified until built/run).
- [x] Add separate GPU-source-only CI in CUDA 13.3 container.
- [ ] GPU kernel crate passes source typechecking and source formatting
      in CI. **This is not equivalent to GPU execution.**

**V1.C — Real-device kernel correctness**
- [x] Add strict GPU-only local conformance runner with environment capture.
- [ ] Run residual, row softmax, and GEMM on the RTX 5070 Ti and attach
      synchronized GPU result logs, commit SHA and CUDA/driver/toolkit versions.
- [ ] Compare nontrivial numerical fixtures (not only all-ones), including
      large logits, different GEMM shapes and invalid inputs.

**V1.D — Coverage of all DINOv2 mathematical operations**
- [ ] Implement correct FP32 LayerNorm (eps=1e-6), exact GELU, patch
      projection, positional/CLS embeddings, Q/K/V, attention and LayerScale.
- [ ] Handle 1,370-token attention rows with correct padding/masking
      and no accidental softmax contribution from padding.
- [ ] Add independent operator fixtures, boundary cases, precision
      tolerance thresholds and actual GPU execution evidence.

**V1.E — Whole model GPU inference**
- [ ] Execute all 12 original DINOv2 Base blocks with original pinned
      FP32 weights, using GPU-resident intermediates and only explicit H2D/D2H.
- [ ] Return CLS, all 1,369 ordered spatial patches and CHW feature map.
- [ ] Expose GPU capability explicitly; unsupported execution must error.

**V1.F — Final numerical and resource acceptance**
- [ ] Compare each GPU layer, CLS, all patches and spatial CHW layout to
      pinned V0 reference; publish max_abs and relative_L2 tolerances.
- [ ] Verify no hidden CPU fallback; capture warm/cold GPU timing and
      peak VRAM without conflating it with host decode/preprocessing.

**Exit gate:** V1.A–V1.F proven by executable results. The CUDA source-build
CI and isolated probes are necessary but **not** evidence of a working GPU
DINOv2. The dataset audit in Phase 2 may run concurrently.

### Phase 2 — OSCC dataset integrity and spatial registration [P0, parallel to V1.B–V1.F]

- [ ] Inventory real OSCC images, labels, licensing, scanners, magnifications,
      patient/slide IDs, tile coordinates and segmentation annotations.
- [ ] Define immutable dataset manifests: `sample_id`, `patient_id`,
      `slide_id`, `source`, `x`, `y`, pyramid level, micrometers/pixel
      (when known), geometry transform, labels, checksum and annotation source.
- [ ] Add patient-safe splits; detect derivative tiles, duplicate images,
      contradictory labels, slide overlap and accidental leakage.
- [ ] Read WSIs by tiles rather than loading entire slides; benchmark CPU
      worker counts 1/2/4/6/8 with bounded prefetch and memory.
- [ ] Verify original slide→tile→processed patch→mask alignment and round trips.
- [ ] Do not prune tissue or resize/crop away small lesions without measuring
      label-preserving coverage.

**Gate:** held-out patient-level data splits and auditable spatial provenance.

### Phase 3 — Model-independent adapters and 2-backbone evidence [P0]

- [ ] Extract typed `ModelAdapter` and `ModelRegistry` with metadata and
      model-specific normalization/positional embeddings.
- [ ] Verify DINOv2 reference and CUDA implementations behind the adapter.
- [ ] Add a second architecture-family-compatible backbone such as Hibou-B
      (access/licence and register tokens must be checked first).
- [ ] Evaluate Phikon-v2 as an additional larger pathology baseline,
      conditional on access, resource budget and its checkpoint semantics.
- [ ] Conformance suite checks CLS + patches, register handling, grid order,
      finiteness, repeated results, checkpoint IDs and task compatibility.
- [ ] Execute the **same** operation under at least two compatible adapters
      without duplicating its mathematical implementation.

**Gate:** prove reuse by real test execution, not just matching trait names.

### Phase 4 — Cross-backbone numerical correctness and controlled ablations [P0]

- [x] Add simple FP32 CPU softmax and residual tests in the reference module
      (CPU fixtures passed in CI; independent GPU parity remains open).
- [ ] Add independently reviewed fixtures for GEMM, attention, softmax,
      normalization, GELU, patch embedding and precision conversions.
- [ ] Compare CPU reference versus GPU on edge cases, realistic sizes, layouts,
      NaN/inf inputs and BF16/FP16 where supported.
- [ ] Record per-layer and per-token errors, not only CLS.
- [ ] Verify selected optimized code path **actually executes**, without
      implicit CPU fallback or different preprocessing.
- [ ] Distinguish exact/reordered kernels from approximations or token pruning.

**Gate:** numerical parity is blocking for equivalent kernel proposals.

### Phase 5 — Supervised OSCC evaluation [P1]

- [ ] Choose tasks only after label audit: tile binary classification,
      slide-level MIL, region detection or mask segmentation.
- [ ] Train task heads/probes using training patients only; select thresholds
      on validation data and freeze before final test evaluation.
- [ ] Report sensitivity, specificity, AUROC/AUPRC and uncertainty grouped by
      patient/slide; Dice/IoU only for annotated masks.
- [ ] Measure lesion recall and false negatives for any spatial-token pruning.
- [ ] Do not present unlabeled spatial features or attention maps as tumor masks
      or a diagnostic system.

### Phase 6 — CPU/GPU performance and resource envelope [P1]

- [ ] Separate decode, crop, host normalization, batch assembly, H2D,
      GPU forward, D2H and whole-pipeline latency.
- [ ] Use synchronized GPU events, warmup, interleaved repetitions, p50/p95,
      occupancy/clock/power context and VRAM peak with reproducible reports.
- [ ] Benchmark 1/2/4/6/8 CPU workers, batch sizes 1→supported maximum
      without OOM, and multiple token sequence lengths.
- [ ] Compare against a strong library baseline, not only a naive kernel.
- [ ] Measure the extra VRAM/materialization cost of dense outputs.

### Phase 7 — Histology-aware robustness and algorithmic improvements [P1]

- [ ] Evaluate patch-border shifts, magnification and µm/px, stain changes,
      blur, scanner/site differences and label-preserving perturbations.
- [ ] Start with exact memory-efficient attention; then explore adaptive
      token selection/multiscale routing with strict lesion-recall gates.
- [ ] Hold model weights, patients, scoring protocol and preprocessing fixed
      in operator A/B experiments.
- [ ] Preserve negative findings and unexpected lesion misses.

### Phase 8 — Reproducibility and reports [P2]

- [ ] Version experiment configs, operation/build IDs, model revision/checkpoint,
      dataset hashes, Rust toolchain, CUDA/driver, GPU and CPU details.
- [ ] Persist per-sample output, per-region geometry, numeric errors, timing
      repetitions, errors/OOM and summary with cluster-aware intervals.
- [ ] Add regression gates for numerical parity and application quality;
      don't use single noisy timings as a hard CI failure.
- [ ] Separate CPU GitHub Actions from optional local GPU test reports.

### Phase 9 — Optional agent/orchestration [P2]

- [ ] Only after stable experiment executor: restricted CLI/config-driven
      experiment planner (Ollama/llama.cpp optional).
- [ ] Do not allow agents to alter ground truth, patient splits, scoring or
      hardware timing isolation.
- [ ] Keep single-workstation Docker/Podman deployment until scaling is justified.

## 4. Test matrix (aligned with both technical reports)

| Test | Gate | Current evidence |
| --- | --- | --- |
| SP-01 CLS versus independent Hugging Face reference | Pinned full checkpoint | Verified — [V0 record](docs/v0-validation.md) |
| SP-02 Spatial shape, patch order, CLS exclusion | CPU fixtures | Implemented |
| SP-03 All-token Hugging Face + spatial CHW parity | Full 1370x768 output | Verified — [CI](https://github.com/jhserodio/exploring-vit-algorithms-improvments/actions/runs/45245472357) |
| SP-04 Geometry tile / crop / source mapping | CPU coordinate round-trip | Implemented |
| SP-05 Mask-to-patch registration | Labeled masks | Planned |
| SP-06 Patient/slide leakage | Dataset manifests | Planned |
| SP-07 Model adapter conformance | Two actual models | Planned |
| SP-08 Register token exclusion | Hibou-B/compatible models | Planned |
| SP-09 Matmul/attention/norm and CUDA parity | CPU/GPU | CUDA residual + GEMM + softmax in source; no GPU validation yet |
| SP-10 CPU/GPU staged pipeline | Measured worker sweeps | Planned |
| SP-11 OSCC localization/segmentation | Appropriate annotations | Planned |
| SP-12 Histology and spatial robustness | Label-preserving transformations | Planned |
| SP-13 Same kernel in two ViTs | No silent fallback | Planned |
| SP-14 Dense-output cost | Latency and memory | Planned |
| SP-15 Pruning preserves lesions | Lesion recall | Planned |
| SP-16 Provenance/reproducibility | Model/data/toolchain hashes | Partial scripts |

## 5. First implementation sprint (refactor)

1. [x] R01 — introduce model-independent dense token output and geometry.
2. [x] R02 — replace the previous runtime with an independent Rust CPU implementation.
3. [x] R03 — add native CUDA Rust Tile residual operation probe.
4. [x] R04 — prepare full-token PyTorch parity script.
5. [x] R05 — CPU CI and original-checkpoint parity verified; [14-test CPU CI](https://github.com/jhserodio/exploring-vit-algorithms-improvments/actions/runs/45245472506) and [full spatial parity](https://github.com/jhserodio/exploring-vit-algorithms-improvments/actions/runs/45245472357).
6. [ ] R06 — compile CUDA Rust Tile source and execute GPU kernel probes on RTX 5070 Ti.
7. [ ] R07 — implement full GPU attention and DINOv2 forward.
8. [ ] R08 — audit OSCC patient/slide labels and spatial metadata.
9. [ ] R09 — add Hibou-B adapter and independent conformance suite.
10. [ ] R10 — run controlled ablation, OSCC quality and performance study.

**Historical baseline:** Previous inference implementation in
[commit 7b16388](https://github.com/jhserodio/exploring-vit-algorithms-improvments/commit/7b16388f3675cc6af104fb4b2176eeb18bbff254);
prior CPU CI reported eight tests passing. Those results **do not** establish
the current CPU or GPU correctness after this migration.

## 6. Cross-phase dependencies and execution priority

1. **Parallel tracks:** CUDA Rust V1.B–V1.E and Phase 2 dataset/annotation audit
   should advance independently. Neither should fabricate test evidence.
2. **Before algorithm claims:** V1.F (correct GPU model) + Phase 3 (two
   compatible backbones) + Phase 4 (operator equivalence).
3. **Before OSCC claims:** Phase 2 (patient/slide split and legitimate labels)
   + Phase 5 (task-specific training and held-out metrics).
4. **Before performance conclusions:** Phase 6 isolates host decode, H2D,
   kernel, D2H, memory use, warmup, clocks and latency distributions.
5. **Before algorithmic pathology optimization:** Phase 7 must preserve lesion
   recall with registered region labels, rather than treating feature maps as
   tumor masks.
6. **Reporting is cross-cutting:** start versioning experiment metadata in
   Phase 1 rather than waiting for the eventual Phase 8 report.

[Full dependency review](docs/roadmap-reassessment.md) ·
[Phase 1 acceptance plan](docs/phase1-design.md).

**Truthfulness rule:** A checkbox marked for an implementation means source
code exists. Phases close only after their acceptance tests run successfully
and reports are attached. CUDA execution and OSCC sensitivity are not inferred.
