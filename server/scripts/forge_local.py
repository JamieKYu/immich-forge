#!/usr/bin/env python
"""Run the Forge pipeline on a local image file — no Immich required.

The fastest way to confirm the GPU / model wiring works before testing the full
extension → server → Immich loop.

    python scripts/forge_local.py input.jpg output.jpg --denoise --face --upscale 4

Requires the ML deps + weights for the stages you enable; otherwise those stages
fall back (denoise -> NLM, upscale -> Lanczos, face -> no-op) and it still runs.
"""
from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

# Allow running from the server/ dir without installing the package.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import get_settings  # noqa: E402
from app.pipeline import Pipeline  # noqa: E402
from app.schemas import ForgeOperations  # noqa: E402


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("input", type=Path)
    ap.add_argument("output", type=Path)
    ap.add_argument("--denoise", action="store_true")
    ap.add_argument("--denoise-strength", type=float, default=1.0, help="0..1 blend")
    ap.add_argument("--low-light", action="store_true", help="CLAHE + gamma brighten")
    ap.add_argument("--colorize", action="store_true")
    ap.add_argument("--upscale", type=int, default=0, help="factor 2 or 4 (0 = off)")
    ap.add_argument("--face", action="store_true")
    ap.add_argument("--fidelity", type=float, default=0.5)
    args = ap.parse_args()

    ops = ForgeOperations(
        denoise=args.denoise,
        denoise_strength=args.denoise_strength,
        low_light=args.low_light,
        colorize=args.colorize,
        upscale=args.upscale > 0,
        upscale_factor=args.upscale if args.upscale in (2, 4) else 4,
        face_restore=args.face,
        face_fidelity=args.fidelity,
    )

    pipeline = Pipeline(get_settings())
    data = args.input.read_bytes()

    def progress(p: float, stage: str) -> None:
        print(f"  [{int(p * 100):3d}%] {stage}")

    print(f"forging {args.input} -> {args.output}  ({ops})")
    out, notes = await pipeline.run(data, ops, progress)
    for note in notes:
        print(f"  note: {note}")
    args.output.write_bytes(out)
    print(f"done: {len(out) / 1024:.0f} KB written to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
