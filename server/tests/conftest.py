"""Shared fixtures and helpers for the Forge server test suite.

Everything here runs on the lightweight core deps (no torch / no GPU). Images are
synthesised in-memory with Pillow + piexif so tests need no fixture files.
"""
from __future__ import annotations

import io

import piexif
import pytest
from PIL import Image

from app.config import Settings


def make_jpeg(width: int = 64, height: int = 48, *, color=(120, 30, 200)) -> bytes:
    """A bare JPEG with no EXIF."""
    buf = io.BytesIO()
    Image.new("RGB", (width, height), color).save(buf, format="JPEG")
    return buf.getvalue()


def make_jpeg_with_exif(
    width: int = 64,
    height: int = 48,
    *,
    orientation: int = 6,
    make: str = "TestCam",
    with_thumbnail: bool = True,
) -> bytes:
    """A JPEG carrying EXIF: an Orientation tag (default 6 = rotate 90°), a Make
    string, dimensions, and (optionally) an embedded thumbnail — i.e. the shape
    `transplant_exif` has to fix up."""
    exif: dict = {
        "0th": {
            piexif.ImageIFD.Make: make.encode(),
            piexif.ImageIFD.Orientation: orientation,
            piexif.ImageIFD.ImageWidth: width,
            piexif.ImageIFD.ImageLength: height,
        },
        "Exif": {
            piexif.ExifIFD.PixelXDimension: width,
            piexif.ExifIFD.PixelYDimension: height,
        },
        "GPS": {},
        "1st": {},
        "thumbnail": None,
    }
    if with_thumbnail:
        thumb = io.BytesIO()
        Image.new("RGB", (8, 6), (0, 0, 0)).save(thumb, format="JPEG")
        exif["thumbnail"] = thumb.getvalue()
        exif["1st"] = {piexif.ImageIFD.Orientation: orientation}

    img = io.BytesIO()
    Image.new("RGB", (width, height), (10, 20, 30)).save(
        img, format="JPEG", exif=piexif.dump(exif)
    )
    return img.getvalue()


@pytest.fixture
def jpeg_no_exif() -> bytes:
    return make_jpeg()


@pytest.fixture
def jpeg_with_exif() -> bytes:
    return make_jpeg_with_exif()


@pytest.fixture
def settings() -> Settings:
    """A Settings instance with deterministic test values (ignores any ambient
    .env — we pass explicit values)."""
    return Settings(
        IMMICH_BASE_URL="http://immich-test:2283",
        IMMICH_API_KEY="test-key",
        FORGE_API_TOKEN="secret-token",
        FORGE_DEVICE="cpu",
    )
