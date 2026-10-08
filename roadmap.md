# ViT Research Engine — Model-Agnostic Testing Roadmap

**Repository:** [`jhserodio/exploring-vit-algorithms-improvments`](https://github.com/jhserodio/exploring-vit-algorithms-improvments)  
**Research domain:** Vision Transformers (ViT), histopathology, Oral Squamous Cell Carcinoma (OSCC)  
**Primary implementation:** Rust + Hugging Face Candle + CUDA  
**Initial reference backbone:** `facebook/dinov2-base` (replaceable)  
**Reference hardware:** NVIDIA RTX 5070 Ti (Blackwell, compute capability 12.0); AMD Ryzen 7 (exact CPU SKU to be confirmed)  
**Status:** Implementation plan — not a statement that the described modules or tests already exist.

## 1. Research purpose

Build a reproducible, **model-agnostic experimental framework** to answer three related questions:

1. **Recognition:** How reliably do different vision backbones identify patterns relevant to OSCC in labeled histopathology images?
2. **Robustness and generalization:** How do models respond to controlled changes in image appearance and to new patients, slides, scanners, or institutions?
3. **Algorithmic optimization:** Can proposed changes to linear algebra, attention, memory layout, or execution strategy reduce runtime and resource use **without unacceptable numerical or predictive degradation**?

The test framework must outlive any particular model or optimization. DINOv2 is the first baseline, **not** a hard-coded assumption.

### Non-negotiable design principles

- **Swap models without rewriting tests.** Each model plugs into a common interface; the benchmark runner depends only on capabilities and task contracts.
- **Swap optimized operations independently of models.** Compare an original kernel/implementation with an experimental alternative under an otherwise identical model, checkpoint, and workload.
- **Separate three comparisons:** (a) backbone versus backbone, (b) implementation versus implementation for the *same* backbone, and (c) end-to-end application pipeline versus pipeline.
- **Respect model-specific preprocessing.** Same patient/image sample does not imply forcibly identical normalization, tokenization, or input size. Record the preprocessing profile and provide an explicitly named standardized-input experiment only where technically valid.
- **Compare only compatible outputs.** Embeddings, classification scores, detection boxes, and segmentation masks have different valid metrics. Do not call a feature extractor a tumor classifier.
- **Keep benchmarks deterministic and auditable.** A language model may *request* a predefined experiment, but must not modify ground truth, scoring definitions, splits, or timing protocol.
- **Preserve medical-data boundaries.** De-identify identifiers in exported records; never commit patient imagery or sensitive metadata.
- **Never report scientific conclusions from synthetic labels.** Synthetic images are excellent for smoke tests and numerical checks, not a substitute for validated OSCC annotations.

## 2. Scientific experiments and their boundaries

| Experiment class | What changes? | What remains fixed? | Primary outcome |
|---|---|---|---|
| **Backbone comparison** | Model architecture/checkpoint | Patient-level dataset split, task definition, evaluation protocol, hardware | Recognition and robustness relative to cost |
| **Frozen-feature / linear-probe comparison** | Frozen backbone and its native preprocessing | Label definitions, training/evaluation partitions, specified probe-training protocol | Quality of transferable representations |
| **Operation ablation** | Matmul, softmax, attention, patch embedding, normalization, memory layout, etc. | Model/checkpoint, preprocessing, samples, dtype where applicable | Numerical equivalence, latency, VRAM |
| **Precision study** | FP32/FP16/BF16 or supported precision modes | Model, samples, metric code and declared conversion procedure | Accuracy-versus-speed/memory trade-off |
| **Pipeline study** | Decode workers, batch size, queuing, transfers | Backbone, dataset, task and reported environment | End-to-end latency/throughput |
| **External-domain study** | Evaluation cohort/source | Frozen trained model and scoring procedure | Generalization |

**Do not mix these into a single leaderboard without labeling the study type.** A smaller architecture being faster does not prove that an attention kernel is more efficient.

## 3. Intended architecture

```text
experiment.toml / CLI / optional Ollama-assisted planner
                      |
              Experiment Validator
                      |
                  Test Runner
      +---------------+----------------+
      |               |                |
 DatasetProvider   ModelRegistry    OperationRegistry
      |               |                |
 Patient/slide    ModelAdapter       OperationBackend
 metadata +       (DINOv2, other     (reference, optimized)
 annotations       ViT models)
      |               |                |
      +----> Task-specific evaluator <-+
                       |
               Metrics + Timings
                       |
          JSONL predictions + report +
          hashes/versions/experiment manifest
```

### Separation of contracts

**`ModelAdapter` (one per supported architecture/checkpoint family)**

- Describe model ID, revision/checkpoint digest, architecture, supported tasks, input requirements, tensor precision and output semantics.
- Own or provide model-specific image preprocessing; expose a consistent call to load and execute inference.
- Report **typed** outputs, such as `Embedding`, `ClassScores`, `Detection`, or `Segmentation`.
- Expose optional supported operation overrides; reject unsupported variants explicitly.
- Do not implement evaluation metrics or dataset splitting.

**`OperationBackend` (one per tested implementation)**

- Name and version each original/optimized operation.
- Specify supported devices, dimensions, precision, memory layouts and numerical tolerances.
- Provide a reference path against which results can be tested.
- Ensure that selecting an optimized variant actually changes the executed code path; record this in the experiment report.

**`DatasetProvider` / `EvaluationTask`**

- Read immutable manifests with anonymized patient ID, slide ID, image/tile ID, source and annotation references.
- Produce stratified/grouped train/validation/test partitions without patient/slide leakage.
- Define task-specific targets and metrics independent of model internals.
- Support detection or segmentation only when corresponding regional annotations exist.

**`BenchmarkRunner`**

- Run the same named evaluation protocol for every eligible adapter, validate capabilities before execution, and persist per-sample predictions.
- Perform warm-up and synchronized GPU measurement outside of initialization/loading time, unless evaluating cold start explicitly.
- Separate decode, CPU preprocessing, host-to-device transfer, GPU forward pass and end-to-end timing.
- Never silently substitute a model, precision, operation or preprocessing profile.

### Proposed code organization (not currently present)

```text
motor_inferencia/
├── src/
│   ├── models/{mod.rs,adapter.rs,registry.rs,dinov2.rs,...}
│   ├── operations/{mod.rs,backend.rs,reference.rs,optimized.rs}
│   ├── data/{manifest.rs,split.rs,loader.rs}
│   ├── preprocessing/
│   ├── inference/
│   ├── tasks/{features.rs,classification.rs,segmentation.rs}
│   ├── metrics/{classification.rs,segmentation.rs,numerical.rs}
│   ├── benchmark/{runner.rs,timing.rs,resource.rs,report.rs}
│   └── bin/{infer.rs,benchmark.rs}
├── tests/{fixtures,unit,integration}/
├── benches/{cpu_ops.rs,gpu_ops.rs,model_inference.rs}
├── experiments/{baseline.toml,robustness.toml,ablation.toml}
└── Cargo.toml
```

The organization above is a **target layout**, not a requirement to create every module before producing the first meaningful result.

## 4. Implementation phases

### Phase 0 — Restore a trustworthy runnable baseline **[P0]**

**Current evidence:** `motor_inferencia/src/main.rs` is an incomplete image-to-tensor prototype; `Dockerfile` refers to CUDA 12.4 / capability `89`; the current RTX 5070 Ti requires Blackwell-aware tooling (`sm_120`).

- [ ] Fix Rust compile errors and return the preprocessed tensor with correct `Result<Tensor>` handling.
- [ ] Replace dummy image generation with deterministic in-repository fixtures; assert shape, channel order, dtype, range and known pixel mappings.
- [ ] Decouple CPU-only unit tests from mandatory CUDA device creation.
- [ ] Verify NVIDIA driver, compatible CUDA toolkit (12.8+ for native Blackwell support), Candle submodule and `cudarc` build compatibility. Pin tested versions.
- [ ] Remove unnecessary privileged-container defaults for ordinary runs; keep profiling privileges separate if required.
- [ ] Add `cargo fmt`, `cargo clippy`, build/smoke tests and a CPU-only CI path. Run actual CUDA tests on the local GPU.
- [ ] Implement a real DINOv2 forward pass (feature extraction) and verify shape and numeric finiteness.

**Exit gate:** Clean build; input-to-embedding inference works on the GPU; deterministic fixtures pass on CPU; environment versions are captured. **No OSCC accuracy claims yet.**

### Phase 1 — Testing infrastructure and immutable data manifests **[P0]**

- [ ] Introduce versioned experiment configuration (`.toml` or `.yaml`) and reject unknown/invalid fields.
- [ ] Build a dataset manifest format: `sample_id`, `patient_id`, `slide_id`, `source_id`, `image_path`, `annotation_path`, `split`, `label_scheme`, `checksum`.
- [ ] Add dataset-integrity checks: file/checksum verification, label vocabulary, missing values, duplicate hashes, ID consistency.
- [ ] Detect train/validation/test leakage at **patient** level (and slide level); check derivatives and augmented tiles inherit their original grouping.
- [ ] Store seeds, hashes, Git commit, model revision, exact adapter/preprocessor versions, hardware/driver/CUDA/Candle versions.
- [ ] Define result schema with per-sample predictions, metric summary, timing samples and failure records.
- [ ] Create tiny synthetic fixtures for unit tests; reserve labeled OSCC images for genuine experiments.

**Exit gate:** Repeated runs with the same manifest/config are traceable to the same inputs and settings; test suite rejects intentional patient leakage.

### Phase 2 — Model adapters and proof of model independence **[P0]**

- [ ] Design the `ModelAdapter` and `ModelRegistry` contracts for supported input/output types and feature extraction.
- [ ] Implement DINOv2 as the first adapter, including native preprocessing and checkpoint loading.
- [ ] Implement a **second supported vision backbone adapter** to prove portability (choose based on actual Candle/checkpoint compatibility, not merely theoretical availability).
- [ ] Run identical adapter conformance tests on both models: load, preprocess, tensor metadata, repeated prediction, finite output, serialization.
- [ ] Define a fixed downstream classifier/linear-probe protocol before comparing OSCC recognition; maintain identical patient partitions and training/evaluation rules.
- [ ] Fail visibly on incompatible tasks (e.g., comparing embeddings directly to segmentation masks) rather than coercing outputs.
- [ ] Add a conformance test that runs the same task test suite against all registered eligible adapters **without modifying test implementations**.

**Exit gate:** A new model requires only its adapter, metadata and experiment config; existing eligible tests run unchanged. Both implemented adapters pass the same conformance suite.

### Phase 3 — Mathematical and numerical correctness **[P0]**

- [ ] Implement trusted reference fixtures for matmul, scaled dot-product attention, softmax, layer norm and patch embedding.
- [ ] Test tensor shapes, layouts, degenerate sizes, invalid inputs, non-finite values and relevant boundary cases.
- [ ] Compare reference CPU, reference CUDA and candidate optimized operations when supported.
- [ ] Record maximum absolute error, norm-based relative error and failure count, with **dtype- and operation-specific** thresholds fixed before analysis.
- [ ] Add deterministic inference checks for repeated execution in supported deterministic modes; document any nondeterministic kernels.
- [ ] Compare DINOv2 embeddings with a trustworthy checkpoint-equivalent reference implementation or validated fixtures to catch model-loading/preprocessing errors.
- [ ] Verify unsupported operation injection is rejected; assert registered optimized code path actually executes.

**Exit gate:** No optimization can be benchmarked as a scientific improvement before it passes its corresponding correctness tests.

### Phase 4 — OSCC recognition and clinical-data evaluation **[P1]**

- [ ] Inventory available labels **before** choosing tasks: image classification, slide classification, detection and/or segmentation.
- [ ] Establish a patient-grouped training/validation/test split; keep a separate external cohort if possible.
- [ ] Start with frozen-feature extraction + a specified probe/classifier (same training budget/protocol for each eligible backbone).
- [ ] Measure sensitivity, specificity, precision, F1, balanced accuracy, AUROC and AUPRC for *binary classification*, where applicable.
- [ ] Add Dice/IoU only for pixel-level annotated segmentation; add mAP only for supported detection annotations.
- [ ] Choose classification thresholds on validation data only; freeze them before final test evaluation.
- [ ] Report confusion matrices and results by patient/slide, not only tile-level means.
- [ ] Compute patient-/slide-cluster-aware confidence intervals; use paired comparisons when models predict the same held-out units.
- [ ] Record class prevalence, label provenance and annotation limitations; avoid presenting research predictions as clinical diagnoses.

**Exit gate:** Both models are evaluated on the **same held-out patients** under a documented downstream protocol, with uncertainty estimates and no leakage.

### Phase 5 — CPU/GPU performance benchmarks **[P1]**

- [ ] Add CPU microbenchmarks for decoding, normalization, tiling and serial/parallel dataset loading.
- [ ] Add CUDA microbenchmarks for matrix multiplication, attention, softmax and candidate replacement kernels.
- [ ] Add model inference benchmarks by batch size, supported image resolution and precision.
- [ ] Separate cold start, model load, warm inference, transfer time and end-to-end timings.
- [ ] Use warm-up plus synchronized GPU events or correctly synchronized host timing; report medians, p95 and run-to-run dispersion.
- [ ] Capture peak allocated GPU memory if measurable, plus external VRAM utilization telemetry when available; distinguish these metrics.
- [ ] Sweep bounded CPU worker counts (e.g., 1/2/4/6), queues and batch sizes; stop gracefully at unsupported/OOM configurations.
- [ ] Isolate the GPU during **scientific** performance measurements: Ollama/llama.cpp and unrelated GPU applications must not compete for resources.
- [ ] Save device, driver, power/clock/temperature context where possible and rerun comparisons in interleaved or randomized order to reduce drift bias.

**Exit gate:** Re-running a benchmark with the same model and workload produces a comparable latency distribution and complete environment report, not a single uncontextualized stopwatch number.

### Phase 6 — Robustness and cross-domain generalization **[P1]**

- [ ] Create deterministic, seeded transforms: translation (including ViT patch-boundary shifts), scale, rotation, blur, noise, compression and partial occlusion.
- [ ] Add pathology-oriented stain/color perturbations, focus changes and scanner/image-source differences.
- [ ] Define defensible, label-preserving severity ranges; exclude transformations that alter histological meaning from the main robustness score.
- [ ] Re-evaluate the same held-out units under every severity level; record performance degradation from the clean baseline.
- [ ] Compare performance by source/site/scanner and on external labeled data where available.
- [ ] Inspect and archive disagreements and high-confidence errors, preserving privacy and annotation licenses.
- [ ] Analyze subgroup and image-source performance without implying general clinical validity from a small collection.

**Exit gate:** All eligible models complete an identical *transformation protocol* on the same held-out samples, with per-condition results and confidence intervals.

### Phase 7 — Algorithmic optimization and ablation experiments **[P1]**

- [ ] Introduce explicitly named `OperationBackend` selections (e.g., `reference_attention`, `optimized_attention_v1`) where Candle integration allows them.
- [ ] Start with one well-bounded operation; benchmark its isolated kernel before changing the full model.
- [ ] For each new variant, run numerical parity, stability, peak-memory, latency and end-to-end inference tests.
- [ ] Hold architecture/checkpoint/preprocessing/samples fixed for every kernel-versus-kernel experiment.
- [ ] Report speedup `baseline_time / candidate_time`, memory ratio, numerical error and downstream metric change.
- [ ] Test relevant tensor dimensions, sequence lengths, batch sizes, supported dtypes and realistic patch counts.
- [ ] Compare against a suitable strong baseline (including vendor/library optimizations when available), not only an intentionally naive kernel.
- [ ] Keep original and optimized variants available behind versioned configuration to support rollback and repeated experiments.

**Exit gate:** Every claimed speedup has a reproducible matched baseline, parity evidence, workload definition, and performance distribution.

### Phase 8 — Scientific reporting and automated regression gates **[P2]**

- [ ] Produce structured per-run artifacts plus human-readable Markdown/CSV reports and figures.
- [ ] Include model and variant IDs, exact dataset/split digests, preprocessing, environment, failures and confidence intervals.
- [ ] Define a statistical analysis plan **before** interpreting final results (primary outcome, comparisons, multiplicity approach and practical equivalence margins).
- [ ] Support paired comparisons at patient/slide level for prediction quality and repeated-session comparisons for runtime.
- [ ] Implement regression gates based on predeclared tolerances: correctness **must** pass; severe inference/quality regressions block promotion; noisy GPU benchmarks provide evidence rather than brittle single-run CI failures.
- [ ] Add a reproducibility guide for recreating tables and figures from saved manifests and predictions.
- [ ] Keep GPU benchmark jobs separate from lightweight pull-request CI.

**Exit gate:** A researcher can identify exactly which model, operation, dataset, configuration and hardware produced a reported finding and independently rerun the procedure.

### Phase 9 — Optional automation/control plane **[P2, after core stability]**

- [ ] Expose a restricted experiment-runner CLI/API accepting validated, versioned configs.
- [ ] Permit Ollama or llama.cpp to suggest experiments and summarize completed results, **not** redefine scoring, alter manifests or run unapproved code.
- [ ] Keep the LLM process off the GPU during isolated performance benchmarks.
- [ ] Retain Docker Compose for local workflow initially; consider native Rust + systemd or Podman if Docker removal becomes a requirement. Distributed orchestration (Nomad, etc.) is unnecessary for one research workstation unless requirements change.

**Exit gate:** Optional automation cannot silently change experimental inputs, methodology or measurement isolation.

## 5. Test suites and execution tiers

| Suite | Examples | Device | Frequency / gate |
|---|---|---|---|
| **Unit correctness** | preprocessing, shapes, manifest parsing, metrics, CPU reference ops | CPU | Every change / blocking |
| **Adapter conformance** | load metadata, correct preprocess, typed output, invalid-capability rejection | CPU/GPU as supported | Every model change / blocking |
| **Numerical parity** | CPU↔GPU and reference↔optimized kernel | CPU/GPU | Every optimization / blocking |
| **Smoke inference** | fixed image → finite embedding/output | GPU | Local GPU check / blocking before experiments |
| **Dataset integrity** | leakage, duplicates, corrupt labels, group splits | CPU | Every dataset update / blocking |
| **Prediction-quality benchmark** | held-out OSCC classification/region task | GPU + CPU statistics | Per candidate release / scientific report |
| **Robustness** | seeded spatial/stain/noise severity sweeps | GPU | Per candidate release / scientific report |
| **Performance** | warm inference, latency distribution, throughput, peak memory | RTX 5070 Ti | Controlled workstation sessions |
| **Ablation** | same checkpoint, different operation backend | RTX 5070 Ti | Every numerical optimization proposal |

The existing project has not yet implemented these suites. Commands such as `cargo test` and a future `benchmark` binary should only be documented as runnable after their targets are created.

## 6. Minimum model-independent experiment specification

*Illustrative target configuration; fields and executable support are to be implemented.*

```toml
schema_version = 1
experiment_id = "oscc-frozen-probe-baseline"
seed = 20261007

[dataset]
manifest = "datasets/oscc/manifest-v1.csv"
split = "test"
split_digest = "REPLACE_WITH_ACTUAL_DIGEST"
task = "binary_classification" # requires appropriate labels

[protocol]
feature_mode = "frozen_backbone"
probe_id = "pretrained_or_fit_on_training_only"
threshold_source = "validation"
metrics = ["sensitivity", "specificity", "f1", "auroc", "auprc"]

[[models]]
id = "dinov2-base"
checkpoint = "facebook/dinov2-base"
revision = "PIN_EXACT_REVISION"
preprocessor = "model_native"

[[models]]
id = "second-backbone"
checkpoint = "PIN_SUPPORTED_MODEL"
revision = "PIN_EXACT_REVISION"
preprocessor = "model_native"

[runtime]
device = "cuda:0"
dtype = "f32"
batch_size = 8
operation_backend = "reference"
benchmark_mode = "isolated"

[results]
root = "results/oscc-frozen-probe-baseline"
save_per_sample = true
```

For **kernel ablation**, use the same model and checkpoint in separate runs and change only `operation_backend` (and a documented experiment variable, if required). Do not use the sample above as an assertion that a second backbone or classification head currently exists.

## 7. Minimum output artifacts per run

```text
results/<experiment-id>/<run-id>/
├── manifest.json           # frozen config, code/model/data/hardware digests
├── predictions.jsonl       # per-sample output and pseudonymous IDs
├── metrics.json            # values, denominators, intervals, failures
├── latency.jsonl           # synchronized, repeated timings and setup
├── numerical_parity.json   # relevant reference/optimized comparisons
├── environment.json        # CPU, GPU, CUDA, driver, Candle, git commit
└── report.md               # human-readable summary and caveats
```

Use split and dataset references rather than copying protected source images into artifacts. Failures (OOM, load error, unsupported adapter, NaN, etc.) should be recorded, not dropped from the denominator without disclosure.

## 8. First implementation sprint — concrete order

Work in this order; avoid creating the entire framework in advance.

1. [ ] **S01 — Fix baseline build:** repair `main.rs`, validate the input tensor; add CPU fixture tests.
2. [ ] **S02 — Fix Blackwell environment:** test CUDA/Candle compatibility on the RTX 5070 Ti; capture versions.
3. [ ] **S03 — Introduce minimal `ModelAdapter`:** DINOv2 loads and extracts embeddings from a fixed test image.
4. [ ] **S04 — Add manifest and leakage tests:** group patients/slides and validate labels/checksums.
5. [ ] **S05 — Add numerical reference tests:** matmul, softmax and attention on CPU, then CUDA parity.
6. [ ] **S06 — Implement a second actual adapter:** run the exact same conformance suite unchanged.
7. [ ] **S07 — Add controlled CUDA timing:** synchronize, warm up, log inference latency and memory.
8. [ ] **S08 — Add appropriate labeled OSCC task:** frozen features, train-only probe, fixed validation/test evaluation.
9. [ ] **S09 — Add seeded robustness transforms:** start with translation/blur and histology-aware color variation.
10. [ ] **S10 — Introduce first optimized operation:** compare against its reference in isolation and in the same model.

### Definition of done for the first scientifically useful milestone

- At least **two working, compatible model adapters** evaluated under one unchanged test protocol.
- Verified patient/slide-safe held-out dataset split and legitimate task annotations.
- Reproducible per-sample predictions and confidence intervals for the chosen primary metric.
- Controlled inference latency and memory evidence from the same GPU.
- At least one numerical reference-versus-candidate operation test, even if no speedup has been obtained yet.
- Recorded hashes, environment, limitations, and any failed runs.

## 9. Scientific interpretation rules

- **Performance is not accuracy:** faster kernels require a correctness check, not only a throughput gain.
- **Feature extraction is not tumor localization:** evaluate the precise task supported by labels and outputs.
- **More image tiles are not more independent patients:** avoid patch-level leakage and naïve tile-independent confidence intervals.
- **Native preprocessing is part of the model definition:** log it; do not quietly apply DINOv2 normalization to every architecture.
- **A failure to improve is a valid finding:** preserve negative experiments and settings.
- **Hardware context matters:** publish which GPU, CUDA, precision, batch, input shape and runtime were used.
- **Precision and numerical error must be considered together:** lower-precision acceleration can change logits or task scores.
- **LLM assistance is not a source of ground truth:** experiment methodology stays deterministic and reviewable.

## 10. Project progress tracker

| Phase | Priority | Status |
|---|---|---|
| 0. Runnable baseline and Blackwell verification | P0 | Not completed / to validate |
| 1. Manifests and reproducibility | P0 | Planned |
| 2. Model-independent adapters | P0 | Planned |
| 3. Numerical correctness | P0 | Planned |
| 4. Labeled OSCC evaluation | P1 | Planned |
| 5. CPU/GPU benchmarking | P1 | Planned |
| 6. Robustness and generalization | P1 | Planned |
| 7. Algorithmic ablation | P1 | Planned |
| 8. Scientific reporting | P2 | Planned |
| 9. Optional Ollama/control plane | P2 | Planned |

**Update this tracker only after running and reviewing the corresponding acceptance tests.**
