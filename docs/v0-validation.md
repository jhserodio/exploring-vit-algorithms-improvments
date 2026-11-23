# V0 validation record — DINOv2 Base FP32 dense features

**Status:** PASSED — 2026-11-23.  
**Target code:** `feat/testing-background`, verified parity implementation at commit `c7cf334023cadc4d38b31f41f277ae03f732be86`.  
**Full parity CI:** [run 45245472357](https://github.com/jhserodio/exploring-vit-algorithms-improvments/actions/runs/45245472357) (completed successfully).  
**CPU-format-test-lint CI:** [run 45245472506](https://github.com/jhserodio/exploring-vit-algorithms-improvments/actions/runs/45245472506) (completed successfully).  

## Checkpoint provenance

- Model: `facebook/dinov2-base`, original FP32 `model.safetensors`.
- Fixed Hugging Face revision: `f9e44c814b77203eaa57a6bdbbd535f21ede1415`.
- Model weights SHA-256: `d73036b56966966d07975d696bde331762f37297e2f095de8cea0040c3aa0841`.
- Configuration SHA-256: `f7ff4cfa73d2f70647dbf6950541ad25d73082d54c2e7e9bded160c7656b2a70`.
- Test input SHA-256: `b76486bb9bd71eecd3be6c730620e751abadf95dbe3cef27482439c76040b35f`.
- Input: `motor_inferencia/tests/fixtures/pixel_fixture.ppm` (synthetic 2×2 RGB pixels expanded by the existing 518px preprocessing profile). **This is not an OSCC dataset.**
- Rust runtime: standalone `ndarray` FP32 CPU reference.
- Independent reference: PyTorch `2.9.1+cpu`, Transformers `4.57.1`.
- Shape: CLS `[1,768]`; patches `[1,1369,768]`; spatial map `[1,768,37,37]`.
- Both frameworks consumed the **identical normalized FP32 input tensor** exported by the Rust implementation.

## Numerical comparisons (same original checkpoint)

| Compared feature | Maximum absolute error | Relative L2 error |
| --- | ---: | ---: |
| CLS | `0.0000173226` | `0.00000256844` |
| All 1,369 patch tokens | `0.0003070831` | `0.00000688757` |
| Full 1,370-token hidden state | `0.0003070831` | `0.00000688988` |
| Channel-first spatial map (768×37×37) | `0.0003070831` | `0.00000688988` |

Pass thresholds were declared before running: `max_abs <= 0.001` AND
`relative_l2 <= 0.001` for all four outputs. Both tests passed.

The Rust CLS export exactly matched token 0 from its own full hidden-state
export, and the spatial map exactly matched its row-major tokens transformed
to CHW (`cls_export_exact: true`, `spatial_map_order_exact: true`).

These numerical errors may vary slightly with CPU BLAS/parallel execution,
but the declared tolerances apply independently to each rerun.

## Repeat locally

```bash
python3 -m pip install huggingface_hub numpy transformers
# Install PyTorch from the official CPU wheel index for reference tests.
python3 scripts/fetch_dinov2_checkpoint.py --output models \
  --revision f9e44c814b77203eaa57a6bdbbd535f21ede1415
python3 scripts/compare_dinov2_reference.py \
  --image motor_inferencia/tests/fixtures/pixel_fixture.ppm \
  --model-dir models --device cpu --output-prefix results/reference/v0-dinov2 \
  --atol 0.001 --rtol 0.001
```

The [V0 GitHub Actions workflow](../.github/workflows/v0-dinov2-reference.yml)
pins the Python dependencies and publishes only JSON metrics/provenance. It does
not upload private images or exported feature tensors.

## Boundaries

**V0 is complete as a numerical baseline only.** CUDA Rust Tile smoke probes
still require local execution on the RTX 5070 Ti; sandbox Phase 1 prepares
their environment and tooling. Full GPU DINOv2 inference, OSCC tumor
classification, segmentation and patient-level evaluation belong to future
experimental branches, not this sandbox roadmap. A synthetic
input and feature-vector parity are not evidence of cancer-detection quality.
