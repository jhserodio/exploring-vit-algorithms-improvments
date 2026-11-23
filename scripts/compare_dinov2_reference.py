#!/usr/bin/env python3
"""Independent DINOv2 Base parity check: identical Rust-normalized FP32 tensor.

Optional research dependencies: pip install numpy torch transformers
Data exports can encode patient information. Keep results local and private.
"""
import argparse
import hashlib
import json
import subprocess
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True, help="Local HF config.json + model.safetensors")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu", help="Candle execution; PyTorch reference runs on CPU")
    parser.add_argument("--atol", type=float, default=1e-3, help="Predeclared max absolute error (FP32)")
    parser.add_argument("--rtol", type=float, default=1e-3, help="Predeclared relative L2 error (FP32)")
    parser.add_argument("--output-prefix", type=Path, default=Path("results/reference/dinov2-base"))
    args = parser.parse_args()

    import numpy as np
    import torch
    import transformers
    from transformers import Dinov2Model

    root = Path(__file__).resolve().parent.parent
    image = args.image.expanduser().resolve()
    model_dir = args.model_dir.expanduser().resolve()
    prefix = (root / args.output_prefix).resolve() if not args.output_prefix.is_absolute() else args.output_prefix
    weights = model_dir / "model.safetensors"
    config = model_dir / "config.json"
    for path in (image, weights, config):
        if not path.is_file():
            parser.error(f"required local file missing: {path}")
    if args.atol <= 0 or args.rtol <= 0:
        parser.error("atol and rtol must be strictly positive")

    cmd = [
        "cargo", "run", "--release", "--manifest-path", str(root / "motor_inferencia/Cargo.toml"),
    ]
    cmd += ["--features", "cuda"] if args.device == "cuda" else ["--no-default-features"]
    cmd += ["--", f"--{args.device}", "--weights", str(weights), "--image", str(image), "--export-prefix", str(prefix)]
    print("Running Rust/Candle inference with explicit tensor export...", flush=True)
    subprocess.run(cmd, cwd=root, check=True)

    raw_input = np.fromfile(str(prefix) + ".input.f32le", dtype="<f4")
    if raw_input.size != 3 * 518 * 518:
        raise ValueError(f"wrong input tensor size: {raw_input.size}")
    pixels = torch.from_numpy(raw_input.reshape(1, 3, 518, 518).copy())
    rust_embedding = np.fromfile(str(prefix) + ".embedding.f32le", dtype="<f4")
    if rust_embedding.size != 768:
        raise ValueError(f"wrong Rust embedding size: {rust_embedding.size}")

    print("Loading the same local checkpoint into Hugging Face PyTorch...", flush=True)
    torch.set_num_threads(min(4, torch.get_num_threads()))
    reference = Dinov2Model.from_pretrained(str(model_dir), local_files_only=True).float().eval()
    with torch.inference_mode():
        reference_cls = reference(pixel_values=pixels).last_hidden_state[:, 0, :]
    expected = reference_cls.detach().cpu().numpy().reshape(-1).astype(np.float64)
    actual = rust_embedding.astype(np.float64)
    if not np.isfinite(actual).all() or not np.isfinite(expected).all():
        raise ValueError("non-finite embedding values")
    diff = actual - expected
    max_abs = float(np.max(np.abs(diff)))
    relative_l2 = float(np.linalg.norm(diff) / max(np.linalg.norm(expected), 1e-12))
    passed = bool(max_abs <= args.atol and relative_l2 <= args.rtol)
    report = {
        "check": "dinov2-base-identical-input-cls-parity",
        "passed": passed,
        "rust_device": args.device,
        "pytorch_device": "cpu",
        "tensor_shape": [1, 3, 518, 518],
        "embedding_shape": [1, 768],
        "max_abs_error": max_abs,
        "relative_l2_error": relative_l2,
        "max_abs_tolerance": args.atol,
        "relative_l2_tolerance": args.rtol,
        "checkpoint_sha256": sha256(weights),
        "config_sha256": sha256(config),
        "image_sha256": sha256(image),
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "numpy_version": np.__version__,
    }
    report_path = Path(str(prefix) + ".parity.json")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    print(f"Saved locally: {report_path}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
