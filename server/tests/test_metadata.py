"""Tests for best-effort EXIF transplant onto forged JPEGs."""
from __future__ import annotations

import io

import piexif
import pytest
from PIL import Image

from app.metadata import _safe_dump, transplant_exif

from .conftest import make_jpeg, make_jpeg_with_exif


def _load_exif(jpeg: bytes) -> dict:
    return piexif.load(jpeg)


def test_transplants_original_exif_onto_forged():
    original = make_jpeg_with_exif(make="Canon")
    forged = make_jpeg(128, 96)  # upscaled, no EXIF of its own

    out = transplant_exif(original, forged)
    exif = _load_exif(out)

    assert exif["0th"][piexif.ImageIFD.Make] == b"Canon"


def test_orientation_is_reset_to_upright():
    # Original says 6 (rotate 90°); cv2 already baked rotation into the pixels,
    # so the forged output must declare Orientation=1 to avoid a double-rotate.
    original = make_jpeg_with_exif(orientation=6)
    forged = make_jpeg(128, 96)

    exif = _load_exif(transplant_exif(original, forged))

    assert exif["0th"][piexif.ImageIFD.Orientation] == 1


def test_dimensions_rewritten_to_forged_size():
    original = make_jpeg_with_exif(width=64, height=48)
    forged = make_jpeg(256, 192)  # 4x

    exif = _load_exif(transplant_exif(original, forged))

    assert exif["0th"][piexif.ImageIFD.ImageWidth] == 256
    assert exif["0th"][piexif.ImageIFD.ImageLength] == 192
    assert exif["Exif"][piexif.ExifIFD.PixelXDimension] == 256
    assert exif["Exif"][piexif.ExifIFD.PixelYDimension] == 192


def test_stale_thumbnail_is_dropped():
    original = make_jpeg_with_exif(with_thumbnail=True)
    forged = make_jpeg(128, 96)

    exif = _load_exif(transplant_exif(original, forged))

    assert exif["thumbnail"] is None
    assert exif["1st"] == {}


def test_forged_pixels_are_preserved():
    # Transplant must only touch metadata, never the image pixels.
    original = make_jpeg_with_exif()
    forged = make_jpeg(128, 96, color=(5, 200, 5))

    out = transplant_exif(original, forged)

    with Image.open(io.BytesIO(out)) as im:
        assert im.size == (128, 96)
        # Center pixel keeps the forged color (JPEG is lossy; allow slack).
        r, g, b = im.getpixel((64, 48))
        assert g > 150 and r < 60 and b < 60


def test_non_jpeg_original_returns_forged_unchanged():
    # A PNG original has no piexif-loadable EXIF -> forged bytes pass through.
    png = io.BytesIO()
    Image.new("RGB", (32, 32), (1, 2, 3)).save(png, format="PNG")
    forged = make_jpeg()

    assert transplant_exif(png.getvalue(), forged) == forged


def test_malformed_original_returns_forged_unchanged():
    forged = make_jpeg()
    assert transplant_exif(b"not an image at all", forged) == forged


def test_original_with_empty_exif_still_yields_valid_jpeg():
    # A plain PIL JPEG has a loadable-but-empty EXIF, so transplant proceeds
    # (rather than bailing): the result must still decode and be marked upright.
    original = make_jpeg()
    forged = make_jpeg(128, 96)

    out = transplant_exif(original, forged)
    exif = _load_exif(out)

    assert exif["0th"][piexif.ImageIFD.Orientation] == 1
    with Image.open(io.BytesIO(out)) as im:
        assert im.size == (128, 96)


def test_safe_dump_retries_after_stripping_undumpable_tags():
    # SceneType stored as an int is loadable but makes piexif.dump raise; _safe_dump
    # should strip the offenders and succeed on the retry.
    exif = {
        "0th": {piexif.ImageIFD.Make: b"X"},
        "Exif": {piexif.ExifIFD.SceneType: 1},  # not bytes -> dump raises
        "GPS": {},
        "1st": {},
        "thumbnail": None,
    }
    with pytest.raises(Exception):
        piexif.dump(exif)  # sanity: the bad tag really does break a plain dump

    out = _safe_dump(exif, piexif)
    assert isinstance(out, bytes) and len(out) > 0
