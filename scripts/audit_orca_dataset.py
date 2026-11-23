#!/usr/bin/env python3
"""Read-only audit of the ORCA 512x512 files already in this repository.

Checks actual PNG dimensions, pixels in grayscale annotation masks, filename
pairing, case-level split leakage, repeated bytes, and file validity.
Writes ONLY aggregate metadata; does not generate or upload images/masks.
No third-party Python dependencies are needed.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from hashlib import sha256
import json
from pathlib import Path
import re
import struct
import sys
import zlib

PNG_MAGIC = b"\x89PNG\r\n\x1a\n"
CASE_PATTERN = re.compile(r"TCGA-[A-Za-z0-9]{2}-[A-Za-z0-9]{4}")


def decode_png(path: Path, pixel_counts: bool) -> dict:
    raw = path.read_bytes()
    if not raw.startswith(PNG_MAGIC):
        raise ValueError(f"invalid PNG signature: {path}")
    offset = len(PNG_MAGIC)
    header = None
    idat = []
    ended = False
    while offset + 12 <= len(raw):
        n = struct.unpack_from(">I", raw, offset)[0]
        kind = raw[offset + 4:offset + 8]
        end = offset + 12 + n
        if end > len(raw):
            raise ValueError(f"truncated PNG chunk: {path}")
        chunk = raw[offset + 8:offset + 8 + n]
        crc = struct.unpack_from(">I", raw, offset + 8 + n)[0]
        if zlib.crc32(kind + chunk) != crc:
            raise ValueError(f"PNG CRC mismatch: {path}")
        if kind == b"IHDR":
            if header is not None or len(chunk) != 13:
                raise ValueError(f"invalid PNG IHDR: {path}")
            header = struct.unpack(">IIBBBBB", chunk)
        elif kind == b"IDAT":
            idat.append(chunk)
        elif kind == b"IEND":
            ended = True
            if end != len(raw):
                raise ValueError(f"trailing PNG data: {path}")
            break
        offset = end
    if not ended or header is None or not idat:
        raise ValueError(f"incomplete PNG: {path}")
    width, height, depth, color, compression, method, interlace = header
    if depth != 8 or color not in (0, 2) or compression or method or interlace:
        raise ValueError(f"unsupported PNG layout {header}: {path}")
    channels = 1 if color == 0 else 3
    stride = width * channels
    if not width or not height or stride > 100_000_000:
        raise ValueError(f"invalid PNG geometry: {path}")
    raw_pixels = zlib.decompress(b"".join(idat))
    if len(raw_pixels) != height * (stride + 1):
        raise ValueError(f"unexpected decompressed scanline length: {path}")
    hist = Counter()
    previous = bytearray(stride)
    for row in range(height):
        start = row * (stride + 1)
        method = raw_pixels[start]
        scanline = bytearray(raw_pixels[start + 1:start + 1 + stride])
        if method not in range(5):
            raise ValueError(f"bad PNG row filter {method}: {path}")
        if pixel_counts:
            for x in range(stride):
                left = scanline[x - channels] if x >= channels else 0
                up = previous[x]
                upper_left = previous[x - channels] if x >= channels else 0
                if method == 1:
                    scanline[x] = (scanline[x] + left) & 255
                elif method == 2:
                    scanline[x] = (scanline[x] + up) & 255
                elif method == 3:
                    scanline[x] = (scanline[x] + (left + up) // 2) & 255
                elif method == 4:
                    pred = left + up - upper_left
                    da, db, dc = abs(pred - left), abs(pred - up), abs(pred - upper_left)
                    predictor = left if da <= db and da <= dc else up if db <= dc else upper_left
                    scanline[x] = (scanline[x] + predictor) & 255
            hist.update(scanline)
        elif method in (2, 3, 4):
            # Image decoding is not required here: only PNG integrity and shape.
            pass
        previous = scanline
    return {
        "width": width, "height": height, "depth": depth,
        "color_type": color, "sha256": sha256(raw).hexdigest(),
        "pixel_counts": hist
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("ORCA_512x512"))
    parser.add_argument("--output", type=Path, default=Path("results/orca-audit.json"))
    parser.add_argument("--expected-pairs", type=int, default=200)
    args = parser.parse_args()

    split_summaries = {}
    global_colors = Counter()
    present_in_masks = Counter()
    all_case_sets = {}
    all_image_hashes = set()
    all_mask_hashes = set()
    errors = []
    pairs_total = 0
    for split in ("training", "testing"):
        images_dir = args.dataset / split / "tumor" / "tma"
        masks_dir = args.dataset / split / "lesion_annotations"
        images = {p.stem: p for p in images_dir.glob("*.png")}
        masks = {p.stem.removesuffix("_mask"): p for p in masks_dir.glob("*_mask.png")}
        missing = sorted(set(images) - set(masks))
        orphan = sorted(set(masks) - set(images))
        if missing or orphan:
            errors.append(f"{split}: {len(missing)} unmatched images, {len(orphan)} orphan masks")
        cases = defaultdict(int)
        slides = set()
        per_split_colors = Counter()
        for name in sorted(set(images) & set(masks)):
            try:
                image_meta = decode_png(images[name], pixel_counts=False)
                mask_meta = decode_png(masks[name], pixel_counts=True)
            except (OSError, ValueError, zlib.error) as exc:
                errors.append(f"{split}: invalid PNG pair: {exc}")
                continue
            if (image_meta["width"], image_meta["height"]) != (512, 512):
                errors.append(f"{split}: unexpected image dimension for {name}")
            if (mask_meta["width"], mask_meta["height"]) != (512, 512):
                errors.append(f"{split}: unexpected mask dimension for {name}")
            if image_meta["color_type"] != 2 or mask_meta["color_type"] != 0:
                errors.append(f"{split}: wrong image/mask color format for {name}")
            if image_meta["sha256"] in all_image_hashes:
                errors.append(f"{split}: duplicate image bytes for {name}")
            if mask_meta["sha256"] in all_mask_hashes:
                errors.append(f"{split}: duplicate mask bytes for {name}")
            all_image_hashes.add(image_meta["sha256"])
            all_mask_hashes.add(mask_meta["sha256"])
            case_match = CASE_PATTERN.search(name)
            if not case_match:
                errors.append(f"{split}: missing TCGA case token for {name}")
                continue
            cases[case_match.group()] += 1
            slides.add(name.rsplit("_", 1)[0])
            per_split_colors.update(mask_meta["pixel_counts"])
            for color in mask_meta["pixel_counts"]:
                present_in_masks[str(color)] += 1
            pairs_total += 1

        all_case_sets[split] = set(cases)
        global_colors.update(per_split_colors)
        split_summaries[split] = {
            "pairs": len(set(images) & set(masks)),
            "case_identifiers": len(cases),
            "slide_identifiers": len(slides),
            "cores_per_case_distribution": dict(sorted(Counter(cases.values()).items())),
            "mask_pixel_values": {str(c): n for c, n in sorted(per_split_colors.items())},
        }
    overlap = all_case_sets["training"] & all_case_sets["testing"]
    if overlap:
        errors.append(f"{len(overlap)} TCGA identifiers overlap between training/testing")
    if pairs_total != args.expected_pairs:
        errors.append(f"expected {args.expected_pairs} successfully decoded pairs, got {pairs_total}")
    report = {
        "schema_version": 1,
        "source": str(args.dataset),
        "source_types": {"images": "PNG RGB8", "masks": "PNG grayscale8"},
        "expected_size": [512, 512],
        "pairs_verified": pairs_total,
        "images_verified": pairs_total,
        "masks_verified": pairs_total,
        "unique_case_identifiers_from_filenames": len(all_case_sets["training"] | all_case_sets["testing"]),
        "cross_split_case_overlap_count": len(overlap),
        "mask_pixel_values": {str(c): n for c, n in sorted(global_colors.items())},
        "masks_containing_each_value": dict(sorted(present_in_masks.items(), key=lambda item: int(item[0]))),
        "splits": split_summaries,
        "errors": errors,
        "passed": not errors,
        "notes": [
            "Case IDs are inferred from filenames; patient/slide provenance is not independently verified.",
            "Mask class semantics and provenance must be checked against original ORCA documentation.",
            "No patient images or individual patient IDs are included in this aggregate output.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if not errors else 1


if __name__ == "__main__":
    sys.exit(main())
