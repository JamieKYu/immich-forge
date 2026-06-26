"""Tests for the Denoiser stage — strength blend, low-light, and classical
fallback selection. All paths here stay on the classical (NLM/CLAHE) code so
the tests never import torch or SCUNet:

  * backend="nlm" forces the classical denoiser;
  * backend="none" disables denoising;
  * backend="scunet" with a missing weight falls back to NLM *before* it would
    import torch (the weight-existence check short-circuits the load).
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from app.pipeline.denoise import Denoiser


def _denoiser(backend: str, weights_dir: Path | None = None) -> Denoiser:
    # tile=0 is irrelevant on the classical path; weights_dir defaults to a
    # path with no .pth so SCUNet never loads.
    return Denoiser(backend, weights_dir or Path("/nonexistent-weights"), "cpu", tile=0)


def _noisy(seed: int = 0, h: int = 48, w: int = 48) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, size=(h, w, 3), dtype=np.uint8)


def _flat(value: int, h: int = 48, w: int = 48) -> np.ndarray:
    return np.full((h, w, 3), value, dtype=np.uint8)


def test_backend_none_is_identity():
    img = _noisy()
    out = _denoiser("none")(img, strength=1.0, low_light=False)
    assert np.array_equal(out, img)


def test_backend_none_still_applies_low_light():
    img = _flat(100)
    out = _denoiser("none")(img, strength=1.0, low_light=True)
    # gamma 0.7 brightens midtones; a flat mid-gray must get brighter.
    assert out.mean() > img.mean()


def test_strength_zero_returns_original_exactly():
    img = _noisy()
    out = _denoiser("nlm")(img, strength=0.0, low_light=False)
    assert np.array_equal(out, img)


def test_strength_one_is_full_nlm():
    img = _noisy()
    out = _denoiser("nlm")(img, strength=1.0, low_light=False)
    expected = cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21)
    assert np.array_equal(out, expected)


def test_strength_half_is_blend_of_original_and_denoised():
    img = _noisy()
    denoised = cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21)
    expected = cv2.addWeighted(denoised, 0.5, img, 0.5, 0.0)
    out = _denoiser("nlm")(img, strength=0.5, low_light=False)
    assert np.array_equal(out, expected)


def test_nlm_reduces_noise_on_structured_image():
    # NLM removes noise by averaging similar patches, so it works on structure,
    # not pure random. A flat base + additive noise is the canonical case: the
    # denoised result should sit closer to the (constant) base.
    rng = np.random.default_rng(0)
    base = _flat(128)
    noise = rng.normal(0, 25, base.shape)
    noisy = np.clip(base + noise, 0, 255).astype(np.uint8)
    out = _denoiser("nlm")(noisy, strength=1.0, low_light=False)
    assert np.abs(out.astype(int) - 128).mean() < np.abs(noisy.astype(int) - 128).mean()


def test_scunet_missing_weight_falls_back_to_nlm():
    # No scunet_color_real_psnr.pth on this path -> classical NLM, no torch import.
    img = _noisy()
    out = _denoiser("scunet")(img, strength=1.0, low_light=False)
    expected = cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21)
    assert np.array_equal(out, expected)


def test_low_light_brightens_dark_image():
    dark = _flat(40)
    out = Denoiser._enhance_low_light(dark)
    assert out.mean() > dark.mean()
