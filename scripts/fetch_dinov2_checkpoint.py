#!/usr/bin/env python3
"""Download a revision-pinned official DINOv2 Base FP32 checkpoint for Phase 0.

Uses only public Meta / Hugging Face model files. Records the exact repository
revision and SHA-256 hashes to make future parity runs reproducible.
"""
import argparse
import hashlib
import json
from pathlib import Path


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("models"))
    parser.add_argument("--model-id", default="facebook/dinov2-base")
    parser.add_argument("--revision", help="Optional fixed 40-character Hub commit; defaults to resolving main")
    args = parser.parse_args()

    from huggingface_hub import HfApi, snapshot_download

    info = HfApi().model_info(args.model_id, revision=args.revision or "main")
    revision = info.sha
    if not revision or len(revision) != 40:
        raise RuntimeError(f"unable to resolve a full model revision: {revision!r}")
    target = args.output.resolve()
    target.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=args.model_id,
        revision=revision,
        local_dir=target,
        allow_patterns=["config.json", "model.safetensors"],
    )
    weights = target / "model.safetensors"
    config_path = target / "config.json"
    for file in (weights, config_path):
        if not file.is_file():
            raise RuntimeError(f"checkpoint revision {revision} is missing {file.name}")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    required = {
        "model_type": "dinov2",
        "hidden_size": 768,
        "image_size": 518,
        "num_attention_heads": 12,
        "num_hidden_layers": 12,
        "patch_size": 14,
        "num_channels": 3,
        "hidden_act": "gelu",
        "layer_norm_eps": 1e-6,
        "layerscale_value": 1.0,
        "use_swiglu_ffn": False,
    }
    for key, expected in required.items():
        if config.get(key) != expected:
            raise RuntimeError(
                f"unsupported config {key}: expected {expected!r}, got {config.get(key)!r}"
            )
    metadata = {
        "model_id": args.model_id,
        "revision": revision,
        "checkpoint_sha256": digest(weights),
        "config_sha256": digest(config_path),
        "architecture": "dinov2-base-518-fp32",
    }
    (target / "checkpoint-source.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(metadata, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
