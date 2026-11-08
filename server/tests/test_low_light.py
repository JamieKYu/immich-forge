"""Tests for the low-light stage's classical paths and the full-res transfer —
pure NumPy/OpenCV, no torch. (The HVI-CIDNet model path is covered by
test_low_light_hvicidnet.py, which needs the ML stack.)
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from app.pipeline.low_light import (
    LowLightEnhancer,
    classical_low_light,
    transfer_to_full_res,
)


def _img(h: int = 64, w: int = 96, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, size=(h, w, 3), dtype=np.uint8)


def test_classical_backend_is_classical():
    img = _img()
    out = LowLightEnhancer("classical", Path("/nonexistent-weights"), "cpu")(img)
    assert np.array_equal(out, classical_low_light(img))


def test_missing_weight_falls_back_to_classical_before_importing_torch():
    # The weight-existence check short-circuits the load, so this stays GPU/torch-free.
    enh = LowLightEnhancer("hvicidnet", Path("/nonexistent-weights"), "cpu")
    img = _img()
    assert np.array_equal(enh(img), classical_low_light(img))
    assert enh._model is None and not enh._disabled  # transient: retried later


def test_transfer_identity_when_model_changes_nothing():
    src = _img(120, 160)
    lo = cv2.resize(src, (40, 30), interpolation=cv2.INTER_AREA)
    # out_lo == lo: no brightening, gain 1, so the source comes back exactly.
    assert np.array_equal(transfer_to_full_res(src, lo, lo), src)


def test_transfer_scales_detail_by_local_gain():
    rng = np.random.default_rng(0)
    base = np.full((120, 160, 3), 40, np.uint8)
    src = np.clip(base + rng.normal(0, 4, base.shape), 0, 255).astype(np.uint8)
    lo = cv2.resize(src, (40, 30), interpolation=cv2.INTER_AREA)
    out_lo = cv2.multiply(lo, 2.0)  # the "model" doubles brightness
    out = transfer_to_full_res(src, lo, out_lo)
    assert out.shape == src.shape
    assert abs(out.mean() - 2 * src.mean()) < 2  # brightness follows the model
    # Detail is carried over, amplified roughly by the ~2x gain.
    ratio = out.astype(float).std() / src.astype(float).std()
    assert 1.5 < ratio < 2.5


def test_transfer_caps_detail_gain_on_near_black():
    src = np.zeros((80, 80, 3), np.uint8)
    src[::2, ::2] = 2  # faint noise on black
    lo = cv2.resize(src, (20, 20), interpolation=cv2.INTER_AREA)
    out_lo = np.full_like(lo, 200)  # an extreme brightening
    out = transfer_to_full_res(src, lo, out_lo)
    # Without the cap, gain ~ (200+4)/(0.5+4) ≈ 45 would blow the noise up to
    # ~±45; the 4x cap keeps it within a few levels of the flat 200.
    assert np.abs(out.astype(int) - 200).max() <= 8
