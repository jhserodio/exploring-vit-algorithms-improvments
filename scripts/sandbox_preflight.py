#!/usr/bin/env python3
"""Offline source/configuration preflight for the ViT research sandbox.

No GPU, model download, patient data or network access is needed.
CUDA execution is validated separately on the workstation, never simulated.
"""
import argparse
import json
from pathlib import Path
import sys
import tomllib

ROOT = Path(__file__).resolve().parent.parent

REQUIRED = (
    "motor_inferencia/Cargo.toml",
    "motor_inferencia/Cargo.lock",
    "motor_inferencia/src/operation_contract.rs",
    "motor_inferencia/src/spatial.rs",
    "cuda-kernels/Cargo.toml",
    "cuda-kernels/Cargo.lock",
    "cuda-kernels/src/main.rs",
    "Dockerfile",
    "Dockerfile.cpu",
    "docker-compose.yml",
    ".github/workflows/rust-cpu.yml",
    ".github/workflows/cuda-rust-source.yml",
    ".github/workflows/v0-dinov2-reference.yml",
    "scripts/v1-gpu-acceptance.sh",
    "scripts/collect_env.sh",
    "scripts/compare_dinov2_reference.py",
    "docs/v0-validation.md",
    "PHASE1.md",
    "roadmap.md",
)

def validate(root: Path) -> list[str]:
    errors: list[str] = []
    for name in REQUIRED:
        if not (root / name).is_file():
            errors.append(f"missing sandbox artifact: {name}")

    def read(name: str) -> str:
        file = root / name
        try:
            return file.read_text(encoding="utf-8")
        except OSError:
            return ""

    try:
        cpu = tomllib.loads(read("motor_inferencia/Cargo.toml"))
        gpu = tomllib.loads(read("cuda-kernels/Cargo.toml"))
    except (ValueError, tomllib.TOMLDecodeError) as exc:
        errors.append(f"invalid Cargo manifest: {exc}")
        return errors

    if cpu.get("package", {}).get("edition") != "2024":
        errors.append("CPU crate must use Rust edition 2024")
    legacy_name = bytes((99, 97, 110, 100, 108, 101)).decode("ascii")
    if legacy_name in str(cpu.get("dependencies", {})).lower():
        errors.append("obsolete inference framework dependency in CPU crate")
    if gpu.get("dependencies", {}).get("cutile") != "=0.5.0":
        errors.append("CUDA Rust Tile version must be pinned at cutile =0.5.0")
    if "13.3.0-devel" not in read("Dockerfile"):
        errors.append("CUDA Dockerfile must pin the CUDA 13.3 development image")
    if "FROM rust:" not in read("Dockerfile.cpu"):
        errors.append("CPU image must use independent Rust image")
    compose = read("docker-compose.yml")
    for item in ("vit-cpu:", "vit-gpu:", "Dockerfile.cpu", "dockerfile: Dockerfile", 'profiles: ["gpu"]', "capabilities: [gpu]"):
        if item not in compose:
            errors.append(f"Compose missing required profile/service: {item}")
    if "cutile::prelude" not in read("cuda-kernels/src/main.rs"):
        errors.append("CUDA Rust source does not expose the Tile toolchain probe")
    if "pub mod operation_contract;" not in read("motor_inferencia/src/lib.rs"):
        errors.append("CPU crate does not expose shared operation contract")
    ignored = read(".gitignore")
    for pattern in ("/models/", "/data/", "/results/", "*.safetensors", "*.f32le"):
        if pattern not in ignored:
            errors.append(f"Sensitive data/output ignore rule absent: {pattern}")
    return errors

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT, help="sandbox checkout")
    parser.add_argument("--json", action="store_true", help="machine-readable result")
    args = parser.parse_args()
    errors = validate(args.root.resolve())
    outcome = {
        "sandbox_source_configured": not errors,
        "cuda_runtime_executed": False,
        "gpu_full_model_implemented": False,
        "errors": errors,
    }
    if args.json:
        print(json.dumps(outcome, indent=2))
    else:
        print("Sandbox configuration: " + ("PASS" if not errors else "FAIL"))
        for error in errors:
            print(" - " + error)
        print("GPU runtime and whole-model inference are NOT tested by this preflight.")
    return 1 if errors else 0

if __name__ == "__main__":
    sys.exit(main())
