#!/usr/bin/env python
"""Generate the toolbar icons from the design sheet `forge-icons.png`.

The sheet is a flattened preview (the "F" logo on a transparency checkerboard)
showing an active (multi-color gradient) and inactive (grayscale) variant. This
script extracts each "F", keys out the checkerboard to restore transparency, and
writes square 16/32/48/128 PNGs into ../public/icons/ (copied verbatim into
dist/icons/ by Vite).

  - active   : keyed by colorfulness — the gradient F is saturated, the checker
               (white ~244 / grey ~195) is not.
  - inactive : the F is grey too, so it's isolated as the largest dark connected
               component and hole-filled into a clean silhouette.

Both keep only the largest component, so stray title/label text is dropped.

Run with a Python that has Pillow + numpy + OpenCV (the server venv works):
    server/.venv/bin/python extension/icons/generate_icons.py
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
SRC = HERE / "forge-icons.png"
OUT = HERE.parent / "public" / "icons"
SIZES = (16, 32, 48, 128)

# Crop boxes (x0, y0, x1, y1) around the 128px renderings in forge-icons.png,
# tuned to the sheet's layout (excludes the neighbouring smaller renderings).
ACTIVE_BOX = (14, 50, 350, 335)
INACTIVE_BOX = (14, 430, 350, 720)


def keep_largest(alpha: np.ndarray) -> np.ndarray:
    mask = (alpha > 0.4).astype(np.uint8)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    if n <= 1:
        return alpha
    big = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return alpha * (lab == big)


def alpha_active(rgb: np.ndarray) -> np.ndarray:
    mx = rgb.max(2)
    mn = rgb.min(2)
    sat = (mx - mn) / np.maximum(mx, 1.0)          # colorfulness
    dark = np.clip((110.0 - mx) / 40.0, 0, 1)       # only truly dark pixels
    return np.clip(np.maximum(sat * 2.5, dark), 0, 1)


def alpha_inactive(rgb: np.ndarray) -> np.ndarray:
    mx = rgb.max(2).astype(np.uint8)
    core = (mx < 170).astype(np.uint8)              # solid F core (below grey ~195)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(core, 8)
    if n > 1:
        big = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        core = (lab == big).astype(np.uint8)
    core = cv2.morphologyEx(core, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    fill = core.copy()
    m = np.zeros((core.shape[0] + 2, core.shape[1] + 2), np.uint8)
    cv2.floodFill(fill, m, (0, 0), 1)               # flood background from corner
    sil = ((core == 1) | (fill == 0)).astype(np.float32)
    return cv2.GaussianBlur(sil, (0, 0), 1.2)        # feather edges


def extract(im: np.ndarray, box, alpha_fn) -> Image.Image:
    x0, y0, x1, y1 = box
    rgb = im[y0:y1, x0:x1].astype(np.float32)
    a = keep_largest(alpha_fn(rgb))
    rgba = np.dstack([rgb, np.clip(a * 255, 0, 255)]).astype(np.uint8)
    ys, xs = np.where(rgba[:, :, 3] > 10)
    rgba = rgba[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    # Center on a square, transparent canvas with a little padding.
    h, w = rgba.shape[:2]
    side = int(max(h, w) * 1.12)
    canvas = np.zeros((side, side, 4), np.uint8)
    oy, ox = (side - h) // 2, (side - w) // 2
    canvas[oy:oy + h, ox:ox + w] = rgba
    return Image.fromarray(canvas, "RGBA")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    im = np.array(Image.open(SRC).convert("RGB"))
    for state, fn in (("active", alpha_active), ("inactive", alpha_inactive)):
        box = ACTIVE_BOX if state == "active" else INACTIVE_BOX
        icon = extract(im, box, fn)
        for size in SIZES:
            path = OUT / f"forge-{state}-{size}.png"
            icon.resize((size, size), Image.LANCZOS).save(path)
            print(f"wrote {path}")


if __name__ == "__main__":
    main()
