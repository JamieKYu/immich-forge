#!/usr/bin/env python
"""Fetch model weights into the weights/ directory, in the foreground.

The server downloads missing weights by itself in the background (with
progress on its setup page), so this is only needed to pre-fill a weights
directory — e.g. for an offline host run with FORGE_SKIP_WEIGHT_DOWNLOAD=1.

    python scripts/download_weights.py
"""
from __future__ import annotations

import sys
from pathlib import Path

# Allow running from the server/ dir without installing the package.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.models_download import WEIGHTS, download_file  # noqa: E402


def main() -> int:
    out = Path(__file__).resolve().parent.parent / "weights"
    out.mkdir(exist_ok=True)
    failed = 0
    for w in WEIGHTS:
        dest = out / w.name
        if dest.exists():
            print(f"✓ {w.name} already present")
            continue
        print(f"↓ {w.name} ({w.label}) ...", flush=True)
        try:
            download_file(w.url, dest)
        except Exception as exc:  # noqa: BLE001 - report and carry on with the rest
            print(f"✕ {w.name}: {exc}", file=sys.stderr)
            failed += 1
            continue
        print(f"✓ {w.name} -> {dest}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
