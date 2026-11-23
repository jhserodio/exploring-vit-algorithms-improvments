#!/usr/bin/env python3
"""Fail CI on references to the retired tensor runtime in tracked source/docs.

The name is intentionally built from ASCII codes so this check does not
match its own source. Only tracked UTF-8 files are inspected.
"""
from pathlib import Path
import subprocess
import sys

root = Path(__file__).resolve().parent.parent
retired_name = bytes([99, 97, 110, 100, 108, 101]).decode("ascii")
tracked = subprocess.check_output(["git", "ls-files", "-z"], cwd=root).split(b"\0")
found = []
for raw_name in tracked:
    if not raw_name:
        continue
    name = raw_name.decode("utf-8")
    path = root / name
    if not path.is_file():
        continue
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (UnicodeError, OSError):
        continue
    found.extend(
        f"{name}:{number}: {line.strip()[:120]}"
        for number, line in enumerate(lines, 1)
        if retired_name in line.casefold()
    )

if found:
    print("Stale runtime-framework references found:")
    print("\n".join(found))
    sys.exit(1)

print("Tracked source and documentation contain no stale runtime references.")
