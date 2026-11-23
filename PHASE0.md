# Phase 0 — reproducible baseline and GPU acceptance gates

**Status (2026-11-23):** The CPU build, four unit tests, formatting, Clippy and CLI-help smoke passed in [GitHub Actions on 2026-10-08](https://github.com/jhserodio/exploring-vit-algorithms-improvments/actions/runs/37717374470). CUDA runtime, original checkpoint loading and numerical parity have **not** been verified on the RTX 5070 Ti. This project is research, not an OSCC diagnostic application.

## Implemented

- Rust/Candle DINOv2 Base with 12 transformer layers and CLS output `[1, 768]` for a single image; code exists but requires a real checkpoint run.
- The custom `dinov2-base-native518-v1` preprocessing profile: RGB, resize shorter side to 592, Catmull-Rom, center crop to 518×518, ImageNet channel normalization, FP32 CHW. This is **not** the default Hugging Face 224px preprocessor.
- CPU correctness tests, feature-gated CUDA, and independent CPU/GPU Compose services.

## CPU checks

```bash
git submodule update --init --recursive
cd motor_inferencia
cargo fmt --all -- --check
cargo test --no-default-features
cargo clippy --all-targets --no-default-features -- -D warnings
cargo run --no-default-features -- --help
```

Alternatively, from the repository root:

```bash
docker compose run --build --rm vit-cpu
```

These checks do **not** validate inference with the actual DINOv2 checkpoint.

## Prepare model and image inputs

Use dedicated directories outside sensitive host paths. Data and weights must not enter Git.

```bash
mkdir -p models data results
# After installing huggingface_hub CLI on the host:
hf download facebook/dinov2-base model.safetensors config.json --local-dir ./models
# Provide a permitted, non-sensitive local PNG at ./data/example.png
```

Pin the model revision and hash `models/model.safetensors` for reproducible scientific runs.

## Run actual inference

From the repository root:

```bash
# Full model CPU path (may be slow)
cd motor_inferencia
cargo run --release --no-default-features -- --cpu --weights ../models/model.safetensors --image ../data/example.png

# Full model CUDA path (requires a compatible host driver / toolkit)
cargo run --release --features cuda -- --cuda --weights ../models/model.safetensors --image ../data/example.png
```

Or use the dedicated GPU Compose command from the repository root:

```bash
docker compose --profile gpu run --build --rm vit-gpu
```

Unlike the CPU service, `vit-gpu` **always runs the CUDA feature and an actual inference command**. It fails if model weights, the image, GPU support or CUDA initialization are missing; never interpret Compose configuration as proof of successful GPU inference.

## Fedora SELinux mounts

The Docker Compose bind mounts use shared `:z` for the repository and `:ro,z` for input directories so both services can access the same content under enforcing SELinux. The `z` option **relabels host files**, including read-only mounts. Stage copies of pathology data into dedicated directories; do not use this pattern against arbitrary system paths or clinical-data storage whose labels must not change.

Configure `DATA_DIR` and `MODEL_DIR` in an ignored `.env` based on `.env.example` if required.

## Independent reference parity

Use the independent verification runner (`scripts/compare_dinov2_reference.py`) with the same **normalized tensor** supplied to Candle and Hugging Face PyTorch. This requires optional Python research dependencies (`torch`, `transformers`, `numpy`), the original checkpoint, and an explicit run; it is not part of the lightweight CPU CI.

The parity runner invokes the Rust CLI with `--export-prefix` to write a float32 little-endian input tensor and output embedding under `results/`, then feeds that same input to the local Hugging Face model. Those tensor dumps may encode patient information; treat them as sensitive and do not commit or share them. Example after setting up the optional Python environment:

```bash
python3 scripts/compare_dinov2_reference.py --image data/example.png --model-dir models --device cpu
python3 scripts/compare_dinov2_reference.py --image data/example.png --model-dir models --device cuda
```

Record maximum absolute error, relative L2 error, model/checkpoint hashes and environment. Do not assert equivalence until a matching checkpoint passes declared tolerances.

## Exit checklist

- [x] Four CPU unit tests plus fmt / Clippy / CLI-help passed on 2026-10-08.
- [x] CUDA 12.8.1 / `sm_120` is configured; no privileged container for ordinary execution.
- [x] DINOv2 Base forward and checkpoint loader code is present.
- [x] A specific GPU Compose run command and SELinux mount strategy are documented.
- [ ] New commit passes CI.
- [ ] A real CPU checkpoint run produces finite `[1,768]` embeddings.
- [ ] The RTX 5070 Ti CUDA run succeeds and environment is captured.
- [ ] Independent, identical-input reference parity passes.

See [roadmap.md](roadmap.md) for the canonical 10-phase plan and [test-checklist-roadmap.md](test-checklist-roadmap.md) for the compact execution checklist.
