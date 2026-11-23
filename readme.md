# ViT Research Engine — OSCC Spatial Features and CUDA Rust Algorithms

Research sandbox in **Rust**, focused on mathematically controlled, model-agnostic
algorithm experiments for oral squamous cell carcinoma (OSCC) histopathology.

**Current migration (2026-11-23):** [14-test Rust CPU CI on 2026-11-23](https://github.com/jhserodio/exploring-vit-algorithms-improvments/actions/runs/45242097849) passed (10 unit + 4 CLI tests, Rustfmt, Clippy, Python help). The project now has a standalone Rust CPU reference and an isolated native CUDA Rust package. There are now two explicitly separate paths:

1. **`motor_inferencia/`** — standalone Rust/ndarray DINOv2 Base **CPU numerical
   reference**, with original FP32 safetensors loading, full 12-block forward, and
   both global and spatial features. This path must still pass real-checkpoint
   reference parity; synthetic CI tests alone do not establish model equivalence.
2. **`cuda-kernels/`** — NVIDIA CUDA Rust Tile (`cutile-rs`) operation experiment,
   starting with a model-independent FP32 residual add over a realistic ViT
   hidden-state size. Requires CUDA 13.3+, compatible NVIDIA driver and GPU.
   This is **not** a full DINOv2 CUDA inference engine yet.

The previous inference baseline remains accessible in Git history at
[commit 7b16388](https://github.com/jhserodio/exploring-vit-algorithms-improvments/commit/7b16388f3675cc6af104fb4b2176eeb18bbff254),
but is not a current runtime dependency.

## Why dense spatial features?

For `facebook/dinov2-base` with 518x518 input and 14x14 patches:

- `cls` / global descriptor: `[1, 768]`
- `patch_tokens` in row-major token order: `[1, 1369, 768]`
- `spatial_map` in channel-first order: `[1, 768, 37, 37]`

The features are **not tumor probabilities or segmentation masks**. Any
clinical-data task requires suitable labels, an independently trained/validated
task head, patient-level splits and uncertainty estimates.

## Run CPU correctness tests

```bash
cd motor_inferencia
cargo fmt --all -- --check
cargo test
cargo clippy --all-targets -- -D warnings
```

## Run original DINOv2 checkpoint through the Rust CPU reference

Prepare local `models/model.safetensors` and `data/example.png` (see
[PHASE0.md](PHASE0.md)). A full 518px/12-layer CPU forward may be slow.

```bash
cd motor_inferencia
cargo run --release -- --cpu --weights ../models/model.safetensors --image ../data/example.png \
  --export-prefix ../results/reference/dinov2
```

This exports input, CLS, complete normalized hidden state and the channel-first
spatial map as local FP32 little-endian arrays. Files under `results/` and
`*.f32le` are ignored by Git; exported patient-derived tensors remain sensitive.

To independently compare all 1,370 output tokens against Hugging Face:

```bash
python3 scripts/compare_dinov2_reference.py \
  --image data/example.png --model-dir models --device cpu
```

This command requires optional Python dependencies `numpy`, `torch` and
`transformers`. The comparison has not been demonstrated with real weights here.

## Run the first native CUDA Rust kernel

CUDA Rust Tile needs **CUDA 13.3+** and NVIDIA GPU `sm_120` support. It is
kept out of CPU CI; it must run on the RTX 5070 Ti before being marked validated.

```bash
cargo run --release --manifest-path cuda-kernels/Cargo.toml
# Or, on Fedora with NVIDIA Container Toolkit:
docker compose --profile gpu run --build --rm vit-gpu
```

This validates a residual-add operation over `1370 * 768` FP32 values.
It does **not** execute a whole neural network. `motor_inferencia --cuda` fails
explicitly until the full model's GPU operation backend is implemented.

## Architecture and research gates

- `ModelAdapter`: model-owned weights, preprocessing and output semantics.
- `DenseFeatures`: framework-independent CLS, patch tokens and spatial map.
- `OperationBackend` (planned): reference or CUDA Rust operator with an
  explicit capability contract; never silent fallback to CPU.
- `DatasetProvider / Geometry`: slide coordinates, scale, annotated regions,
  patient-safe split and provenance.
- `TaskHead`: trained/evaluated separately from the backbone, depending on
  what labels are actually available.

For the scientific phases and SP-01 to SP-16 validation matrix, see
[roadmap.md](roadmap.md). For current acceptance gates see
[test-checklist-roadmap.md](test-checklist-roadmap.md).

## Hardware and reproducibility

Target: NVIDIA RTX 5070 Ti, Ryzen 7 CPU, host Fedora. CPU work handles image
decode, tiling, geometry and input batches; GPU work should eventually keep
model layers on the device without intermediate CPU round trips. Record
`scripts/collect_env.sh`, model hash, image hash, model-specific preprocessing,
dtypes and tested CUDA/Rust versions per result. A green CPU CI is **not**
proof of CUDA compatibility or OSCC quality.

Data and weights must stay outside Git; never commit unredacted patient images
or derived outputs. Check licenses and access for alternative pathology
backbones (e.g. Hibou-B or Phikon-v2).
