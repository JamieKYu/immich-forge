"""Direction tests for the source-quality precheck.

The absolute thresholds are heuristic and tuned in config, so these assert the
*ordering* the metrics must preserve — a crisp, detailed image must score
higher (and read as not-low-quality) than a blurred or flat version of it —
rather than pinning exact scores.
"""
from __future__ import annotations

import cv2
import numpy as np

from app.config import Settings
from app.pipeline.quality import assess_quality

_S = Settings(FORGE_API_TOKEN="t", IMMICH_API_KEY="k", FORGE_DEVICE="cpu")


def _detailed() -> np.ndarray:
    """A high-frequency, sharp image (deterministic pseudo-random texture)."""
    rng = np.random.default_rng(0)
    return rng.integers(0, 256, size=(512, 512, 3), dtype=np.uint8)


def test_sharp_detailed_image_is_not_low_quality():
    report = assess_quality(_detailed(), _S)
    assert report.low_quality is False
    assert report.width == 512 and report.height == 512


def test_blurring_lowers_sharpness_and_flags_low_quality():
    sharp = _detailed()
    blurred = cv2.GaussianBlur(sharp, (0, 0), sigmaX=8)
    sharp_r = assess_quality(sharp, _S)
    blurred_r = assess_quality(blurred, _S)

    assert blurred_r.sharpness < sharp_r.sharpness
    assert blurred_r.hf_ratio < sharp_r.hf_ratio
    assert blurred_r.low_quality is True


def test_flat_image_is_low_quality():
    flat = np.full((256, 256, 3), 128, np.uint8)
    report = assess_quality(flat, _S)
    assert report.sharpness == 0.0
    assert report.low_quality is True


def test_tiny_image_does_not_crash():
    # Below the blockiness crop guard — must return a verdict, not raise.
    report = assess_quality(np.full((8, 8, 3), 200, np.uint8), _S)
    assert report.blockiness == 0.0
    assert isinstance(report.low_quality, bool)
