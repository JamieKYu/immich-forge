#!/usr/bin/env python
"""Print Auto-mode suggestions and the raw measurements for local photos — no
Immich required. Use it to sanity-check the suggestions on your own library and
to tune the FORGE_AUTO_* cutoffs (it reads them from the environment / .env,
like the server).

    python scripts/suggest_report.py ~/Pictures/old-scans photo.jpg
    python scripts/suggest_report.py --metrics photos/        # also show raw numbers

Face detection runs if the ML deps are installed and
detection_Resnet50_Final.pth is in FORGE_WEIGHTS_DIR (download_weights.py);
otherwise faces are reported as unavailable.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow running from the server/ dir without installing the package.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import get_settings  # noqa: E402
from app.pipeline import Pipeline  # noqa: E402
from app.pipeline.suggest import FaceDetector, decide, face_plan, measure  # noqa: E402

_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".tif", ".tiff", ".bmp"}
_OPS = ("dejpeg", "denoise", "low_light", "upscale", "face_restore", "colorize")
_LABELS = {
    "dejpeg": "JPEG cleanup",
    "denoise": "Reduce noise",
    "low_light": "Brighten",
    "upscale": "Enlarge",
    "face_restore": "Restore faces",
    "colorize": "Colorize",
}


def _files(paths: list[Path]) -> list[Path]:
    out = []
    for p in paths:
        if p.is_dir():
            out += sorted(f for f in p.rglob("*") if f.suffix.lower() in _EXTS)
        else:
            out.append(p)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("paths", nargs="+", type=Path, help="image files or folders")
    ap.add_argument("--metrics", action="store_true", help="also print raw measurements")
    args = ap.parse_args()

    settings = get_settings()
    detector = FaceDetector(settings.weights_dir, settings.auto_face_detection)
    files = _files(args.paths)
    if not files:
        print("no images found", file=sys.stderr)
        return 1

    print(f"weights dir: {settings.weights_dir.resolve()}")
    face_status_shown = False
    for path in files:
        data = path.read_bytes()
        try:
            img = Pipeline.decode(data)
        except ValueError:
            print(f"{path}: can't decode, skipped\n")
            continue
        signals = measure(img, data, detector)
        if not face_status_shown:
            # Known after the first detection attempt.
            print(f"face detection: {detector.unavailable_reason or 'ready (cpu)'}\n")
            face_status_shown = True
        suggestion = decide(signals, settings)
        ops = suggestion.operations
        print(path)
        for op in _OPS:
            on = getattr(ops, op)
            label = _LABELS[op] + (f" ×{ops.upscale_factor}" if op == "upscale" and on else "")
            print(f"  {'✓' if on else '·'} {label:<16} {suggestion.reasons[op]}")
        if args.metrics:
            metrics = ", ".join(f"{k}={v:.2f}" for k, v in signals.metrics().items())
            print(f"    {metrics}")
            faces = signals.faces or []
            _, _, labels = face_plan(faces, settings)
            for i, (face, label) in enumerate(zip(faces, labels), 1):
                print(
                    f"    face {i}: {face.width:.0f}px wide, sharpness {face.sharpness:.0f}"
                    f"  -> {label}"
                )
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
