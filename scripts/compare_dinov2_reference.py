#!/usr/bin/env python3
"""Validate the complete DINOv2 CLS + patch tokens against Hugging Face.

Uses the identical model checkpoint and Rust-normalized FP32 input tensor.
Requires optional numpy, torch and transformers; not part of CPU unit CI.
"""
import argparse
import hashlib
import json
import subprocess
from pathlib import Path


def sha256(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True, type=Path)
    parser.add_argument("--model-dir", required=True, type=Path)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu",
                        help="Rust runtime; CUDA full forward is not implemented yet")
    parser.add_argument("--atol", type=float, default=1e-3)
    parser.add_argument("--rtol", type=float, default=1e-3)
    parser.add_argument("--output-prefix", type=Path,
                        default=Path("results/reference/dinov2-native518"))
    args = parser.parse_args()
    if args.atol <= 0 or args.rtol <= 0:
        parser.error("atol and rtol must be strictly positive")

    import numpy as np
    import torch
    import transformers
    from transformers import Dinov2Model

    root = Path(__file__).resolve().parent.parent
    image = args.image.expanduser().resolve()
    model_dir = args.model_dir.expanduser().resolve()
    weights = model_dir / "model.safetensors"
    config = model_dir / "config.json"
    prefix = (root / args.output_prefix).resolve()
    for path in (image, weights, config):
        if not path.is_file():
            parser.error(f"file missing: {path}")
    if args.device == "cuda":
        parser.error("native CUDA Rust full-model inference is not ready; use --device cpu")

    rust_cmd = [
        "cargo", "run", "--release", "--manifest-path",
        str(root / "motor_inferencia" / "Cargo.toml"), "--",
        "--cpu", "--image", str(image), "--weights", str(weights),
        "--export-prefix", str(prefix),
    ]
    print("Running independent Rust/ndarray reference...", flush=True)
    subprocess.run(rust_cmd, cwd=root, check=True)

    raw = np.fromfile(str(prefix) + ".input.f32le", dtype="<f4")
    if raw.size != 3 * 518 * 518:
        raise ValueError(f"invalid Rust input shape: {raw.size}")
    pixels = torch.from_numpy(raw.reshape(1, 3, 518, 518).copy())
    actual = np.fromfile(str(prefix) + ".hidden.f32le", dtype="<f4")
    if actual.size != 1370 * 768:
        raise ValueError(f"invalid Rust hidden state: {actual.size}")
    actual = actual.reshape(1370, 768).astype(np.float64)

    torch.set_num_threads(min(4, torch.get_num_threads()))
    model = Dinov2Model.from_pretrained(str(model_dir), local_files_only=True).float().eval()
    with torch.inference_mode():
        output = model(pixel_values=pixels).last_hidden_state
    expected = output.squeeze(0).detach().cpu().numpy().astype(np.float64)
    if expected.shape != actual.shape:
        raise ValueError(f"shape mismatch: torch {expected.shape}, rust {actual.shape}")
    if not np.isfinite(expected).all() or not np.isfinite(actual).all():
        raise ValueError("non-finite output")

    # Verify that the exported CLS and the spatial CHW map preserve the exact
    # bit patterns and patch ordering of the complete Rust hidden state.
    cls_export = np.fromfile(str(prefix) + ".embedding.f32le", dtype="<f4")
    if cls_export.size != 768 or not np.array_equal(cls_export, actual[0]):
        raise ValueError("Rust CLS export does not match hidden-state token 0")

    spatial = np.fromfile(str(prefix) + ".spatial.f32le", dtype="<f4")
    if spatial.size != 768 * 37 * 37:
        raise ValueError(f"invalid spatial map size: {spatial.size}")
    spatial = spatial.reshape(768, 37, 37)
    rust_chw = actual[1:].reshape(37, 37, 768).transpose(2, 0, 1)
    if not np.array_equal(spatial, rust_chw):
        raise ValueError("Rust spatial-map token order / CHW conversion is inconsistent")
    expected_chw = expected[1:].reshape(37, 37, 768).transpose(2, 0, 1)

    def measure(a, b):
        delta = a - b
        return {
            "max_abs": float(np.max(np.abs(delta))),
            "relative_l2": float(np.linalg.norm(delta) / max(np.linalg.norm(b), 1e-12)),
        }

    cls = measure(actual[:1], expected[:1])
    patches = measure(actual[1:], expected[1:])
    full = measure(actual, expected)
    spatial_parity = measure(spatial, expected_chw)
    passed = all(m["max_abs"] <= args.atol and m["relative_l2"] <= args.rtol
                 for m in (cls, patches, full, spatial_parity))
    report = {
        "experiment": "dinov2-native518-identical-tensor-parity",
        "passed": passed,
        "tensor_shape": [1, 1370, 768],
        "spatial_grid": [37, 37],
        "cls": cls,
        "patch_tokens": patches,
        "full_hidden": full,
        "spatial_map": spatial_parity,
        "cls_export_exact": True,
        "spatial_map_order_exact": True,
        "tolerances": {"max_abs": args.atol, "relative_l2": args.rtol},
        "checkpoint_sha256": sha256(weights),
        "config_sha256": sha256(config),
        "image_sha256": sha256(image),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "numpy": np.__version__,
        "rust_backend": "ndarray-cpu",
    }
    provenance = model_dir / "checkpoint-source.json"
    if provenance.exists():
        metadata = json.loads(provenance.read_text(encoding="utf-8"))
        if metadata.get("checkpoint_sha256") != report["checkpoint_sha256"]:
            raise ValueError("download provenance does not match checkpoint bytes")
        if metadata.get("config_sha256") != report["config_sha256"]:
            raise ValueError("download provenance does not match model config")
        report["model_id"] = metadata.get("model_id")
        report["checkpoint_revision"] = metadata.get("revision")
    report_path = Path(str(prefix) + ".parity.json")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
