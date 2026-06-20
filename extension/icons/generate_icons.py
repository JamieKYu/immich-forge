#!/usr/bin/env python
"""Generate the toolbar icons for the extension.

Draws a "spark/sparkle" mark (enhance) in two states — active (colored) and
inactive (grey) — at 16/32/48/128px, supersampled for crisp edges, and writes
them to ../public/icons/ so Vite copies them verbatim into dist/icons/.

Run with any Python that has Pillow, e.g. the server venv:
    server/.venv/bin/python extension/icons/generate_icons.py
"""
from __future__ import annotations

import math
from pathlib import Path

from PIL import Image, ImageDraw

OUT = Path(__file__).resolve().parent.parent / "public" / "icons"
SIZES = (16, 32, 48, 128)
SS = 8  # supersample factor

ACTIVE = (79, 124, 255, 255)   # accent blue
INACTIVE = (140, 146, 158, 255)  # muted grey


def star_points(cx: float, cy: float, outer: float, inner: float, n: int = 4):
    """Vertices of an n-point star centered at (cx, cy), first tip pointing up."""
    pts = []
    for i in range(n * 2):
        r = outer if i % 2 == 0 else inner
        ang = -math.pi / 2 + i * math.pi / n
        pts.append((cx + r * math.cos(ang), cy + r * math.sin(ang)))
    return pts


def draw(size: int, color: tuple[int, int, int, int]) -> Image.Image:
    s = size * SS
    img = Image.new("RGBA", (s, s), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    # Main 4-point sparkle, slightly above center.
    d.polygon(star_points(s * 0.46, s * 0.52, s * 0.44, s * 0.13), fill=color)
    # Small secondary sparkle, upper-right.
    d.polygon(star_points(s * 0.80, s * 0.22, s * 0.16, s * 0.05), fill=color)
    return img.resize((size, size), Image.LANCZOS)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for state, color in (("active", ACTIVE), ("inactive", INACTIVE)):
        for size in SIZES:
            path = OUT / f"forge-{state}-{size}.png"
            draw(size, color).save(path)
            print(f"wrote {path}")


if __name__ == "__main__":
    main()
