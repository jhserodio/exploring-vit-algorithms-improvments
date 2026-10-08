# ViT Research Engine — OSCC Pattern Recognition and Computational Optimization

**Rust + Hugging Face Candle + NVIDIA CUDA** research prototype for controlled comparisons of Vision Transformers and experimental numerical optimizations.

## Research goals

1. Evaluate feature representations and, with appropriate downstream training and labeled data, pattern recognition relevant to Oral Squamous Cell Carcinoma (OSCC).
2. Compare **different model architectures/checkpoints** under a defensible common evaluation protocol.
3. Compare **alternative implementations of mathematical operations** within the **same** model: numerical parity, latency, throughput, memory, accuracy and robustness.

The test framework is designed to be **model-agnostic**. DINOv2 Base is only the **first checkpoint**. The benchmarking system and its future task-specific metrics must not depend on it.

> **Research status:** Phase 0 is a proposed, unverified inference baseline. No medical classification, localization, validation, multi-model benchmark, or Ollama/llama.cpp agent is implemented yet. This is not a medical diagnostic application.

## Hardware and execution

- **GPU:** NVIDIA RTX 5070 Ti (Blackwell, SM 120). Native Blackwell compilation is available in CUDA Toolkit 12.8+; hardware/driver/Candle compatibility still needs workstation validation.
- **CPU:** AMD Ryzen 7 (exact SKU to confirm). CPU workers handle image decoding, metadata, preprocessing, experiment control and statistical analysis.
- **Engine:** Rust 2024; Hugging Face Candle at the repository's pinned Git submodule revision.
- **Development:** Docker Compose with separate CPU and GPU execution services. No privileged container is required for normal inference. Native execution remains supported.

Ollama or llama.cpp may become an **optional** experiment-control assistant; they are not replacements for Candle or the scientific benchmark runner. LLM workloads should not compete for GPU resources during isolated timing experiments.

## Phase 0 — Get a trustworthy starting point

Read **[PHASE0.md](PHASE0.md)** for the full instructions, exact preprocessing profile, known limitations and validation checklist.

CPU-only correctness tests:

```bash
git submodule update --init --recursive
cd motor_inferencia
cargo test --no-default-features
```

CUDA feature extraction of a local input with the original `facebook/dinov2-base` weights (after validating toolchain compatibility):

```bash
cd motor_inferencia
cargo run --release --features cuda -- \
  --cuda --weights ../models/model.safetensors --image ../data/example.png
```

This baseline intentionally uses **518 × 518** images to retain the model's original positional grid; it does not reproduce the default Hugging Face 224-pixel preprocessing. Output is a **CLS embedding `[1,768]`**, not a classification score. Do not treat a successful run as numerical equivalence until the independent reference-parity test passes.

## Planned research pipeline

```text
Patient/slide-grouped dataset + labels
            |
     Experiment validator
            |
 CPU decode and model-owned preprocessing
            |
        ModelAdapter  <-- DINOv2, future backbones
            |
        OperationBackend  <-- reference and optimized math
            |
     Task-specific evaluator
            |
 Accuracy/robustness + GPU timing/VRAM + reproducibility report
```

For scientific comparisons, preserve identical patients and scoring protocols, model-specific native preprocessing, explicit task/output compatibility, and immutable experiment configurations. Compare kernel optimizations against a fixed backbone/checkpoint, *separately* from backbone-vs-backbone comparisons.

See **[roadmap.md](roadmap.md)** for the phased implementation plan, scientific questions and acceptance gates.

## Data and checkpoints

Store image data and model weights outside Git (`data/` and `models/` are ignored). Keep patient data de-identified and use patient-/slide-disjoint splits. No result derived solely from synthetic test fixtures is evidence of OSCC recognition.

## Licensing and attribution

DINOv2: Meta AI / Hugging Face (`facebook/dinov2-base`). Hugging Face Candle is a separate upstream project included as a Git submodule. Follow dataset, code and model-specific licenses independently.
