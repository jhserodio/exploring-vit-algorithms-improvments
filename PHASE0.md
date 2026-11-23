# V0 — DINOv2 Base FP32 reference and spatial feature validation

**Current state:** architecture migration underway. Native CUDA model forward
is deliberately **not yet implemented**. Do not conflate the CUDA kernel probe
with real DINOv2 GPU inference.

## 1. CPU reference

```bash
cd motor_inferencia
cargo fmt --all -- --check
cargo test
cargo clippy --all-targets -- -D warnings
cargo run -- --help
```

The Rust reference uses `ndarray` for matrix operations, `safetensors` to
load original DINOv2 Base FP32 weights, and `image` to decode/normalize RGB.
The engine has no dependency on a neural-network runtime or implicit fallback.

Output for the single-image 518x518 profile:

| Name | Shape | Interpretation |
| --- | --- | --- |
| `cls` | `[1,768]` | Global descriptor |
| `patch_tokens` | `[1,1369,768]` | Row-major 37x37 spatial tokens |
| `spatial_map` | `[1,768,37,37]` | Channel-first view of patch features |

No output is a cancer classifier, pixel mask or clinically validated result.

## 2. Real checkpoint validation (open)

```bash
mkdir -p models data results
# After installing the Hugging Face CLI on the host:
hf download facebook/dinov2-base model.safetensors config.json --local-dir ./models
# Stage a non-sensitive local image at data/example.png

cargo run --release --manifest-path motor_inferencia/Cargo.toml -- \
  --cpu --weights models/model.safetensors --image data/example.png \
  --export-prefix results/reference/dinov2

python3 scripts/compare_dinov2_reference.py \
  --image data/example.png --model-dir models --device cpu
```

The Python reference comparison requires optional `numpy`, `torch` and
`transformers`. It compares the **entire hidden state** (CLS + all patch
tokens), not merely CLS. Tolerances `--atol` and `--rtol` are predeclared
before observing results. The checkpoint's keys, positional embeddings and
activation semantics must be validated, not assumed correct.

## 3. Native GPU kernel probe (not DINOv2 inference)

```bash
cargo run --release --manifest-path cuda-kernels/Cargo.toml
# or:
docker compose --profile gpu run --build --rm vit-gpu
```

This package uses NVIDIA `cutile-rs` and a Rust Tile kernel to add two
tensors of 1,052,160 elements (1,370 × 768). Output must equal 2.0 at
every element. Requires CUDA 13.3+, a compatible host NVIDIA driver and
an RTX 5070 Ti or other supported GPU. It is isolated from CPU CI.

The GPU operation is a genuine CUDA Rust program; it is **not currently
wired into the 12-block DINOv2 forward**. No full-GPU speedup or VRAM claim
can be made at this milestone.

## 4. Geometric fidelity for pathology

The 518px center crop is a model-specific baseline for a single image tile.
`CropGeometry` retains resize/crop dimensions, and supports processed-pixel
to source-image coordinate conversion and spatial-token centers. For WSI,
slide origin, pyramid level, micrometers per pixel, and mask registration
must be added to the dataset manifest. Never interpret a feature location
as a lesion without regional labels and a trained head.

CPU processing should prepare the next tile batch while the GPU works.
Avoid copying internal network activations to the host during inference.

## 5. Fedora SELinux

Compose volumes use shared `:z`, and input directories use `:ro,z`.
The `z` option relabels host files even when mounted read-only. Stage
copies in dedicated research folders; do not mount protected patient
storage or arbitrary system paths to be relabeled.

## 6. V0 acceptance gates

- [x] Standalone Rust CPU reference with no legacy tensor-framework dependency.
- [x] CLS, spatial tokens and CHW feature map contract + synthetic tests.
- [x] Crop/source geometry and round-trip tests.
- [x] CPU unit/integration CI passes Rustfmt and Clippy.
- [x] Automated original-checkpoint download, verification and full-token parity workflow added.
- [ ] Real checkpoint loads and completes finite FP32 inference on CPU.
- [ ] Identical-input reference parity passes for CLS, all 1,369 patches,
      the full hidden state and the CHW spatial map; report tolerances and hashes.

**V0 scope:** The CUDA Rust operation probe, full CUDA inference and
reference-versus-GPU parity are **Phase 1** gates. Patient-level dataset manifests
and OSCC labels are **Phase 2+** gates. They do not hold V0 open.

Read [roadmap.md](roadmap.md) for the full staged refactoring plan.
