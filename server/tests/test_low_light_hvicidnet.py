"""Tests for the HVI-CIDNet low-light path: load, whole-image vs downscaled
runs, the OOM ladder, and fallbacks.

Needs torch + spandrel (requirements-ml.txt), so it's skipped in the GPU-free
CI. A randomly-initialised HVI-CIDNet stands in for the released weights —
spandrel detects the architecture from the state_dict either way — and
everything runs on CPU.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("spandrel")

from safetensors.torch import save_file  # noqa: E402
from spandrel.architectures.HVICIDNet import HVICIDNet  # noqa: E402

from app.pipeline.low_light import WEIGHTS, LowLightEnhancer, classical_low_light  # noqa: E402


@pytest.fixture
def weights_dir(tmp_path: Path) -> Path:
    torch.manual_seed(0)
    save_file(HVICIDNet().state_dict(), tmp_path / WEIGHTS)
    return tmp_path


def _enhancer(weights_dir: Path, max_pixels: int = 0) -> LowLightEnhancer:
    enh = LowLightEnhancer("hvicidnet", weights_dir, "cpu", max_pixels)
    enh._model = enh._load()
    assert enh._model is not None
    return enh


def _img(h: int = 60, w: int = 84, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, 40, size=(h, w, 3), dtype=np.uint8)


def _record_infer(enh: LowLightEnhancer, fail_above: int | None = None) -> list:
    """Wrap _infer to record input sizes and optionally OOM above a size."""
    real, seen = enh._infer, []

    def infer(img):
        seen.append(img.shape[0] * img.shape[1])
        if fail_above is not None and seen[-1] > fail_above:
            raise torch.OutOfMemoryError("CUDA out of memory")
        return real(img)

    enh._infer = infer
    return seen


def test_loads_and_keeps_shape(weights_dir):
    enh = _enhancer(weights_dir)
    assert enh._model.architecture.name == "HVI-CIDNet"
    img = _img()  # 60x84: not a multiple of 8, so spandrel pads and crops
    out = enh(img)
    assert out.shape == img.shape and out.dtype == np.uint8


def test_whole_image_under_max_pixels(weights_dir):
    enh = _enhancer(weights_dir, max_pixels=10_000)
    seen = _record_infer(enh)
    enh(_img())  # 5040 px
    assert seen == [60 * 84]


def test_downscaled_over_max_pixels(weights_dir):
    enh = _enhancer(weights_dir, max_pixels=1_000)
    seen = _record_infer(enh)
    out = enh(_img())
    assert out.shape == (60, 84, 3)
    assert len(seen) == 1 and seen[0] <= 1_100


def test_oom_halves_working_size_and_recovers(weights_dir, monkeypatch):
    monkeypatch.setattr("app.pipeline.low_light._MIN_PIXELS", 500)
    enh = _enhancer(weights_dir, max_pixels=4_000)
    seen = _record_infer(enh, fail_above=1_500)
    out = enh(_img())
    assert out.shape == (60, 84, 3)
    assert seen[0] > 1_500 and seen[-1] <= 1_500  # tried big, finished small


def test_oom_at_min_size_falls_back_to_classical(weights_dir, monkeypatch):
    monkeypatch.setattr("app.pipeline.low_light._MIN_PIXELS", 500)
    enh = _enhancer(weights_dir, max_pixels=4_000)
    _record_infer(enh, fail_above=0)
    img = _img()
    assert np.array_equal(enh(img), classical_low_light(img))


def test_unrecognised_checkpoint_falls_back_to_classical(tmp_path):
    save_file({"not": torch.zeros(1)}, tmp_path / WEIGHTS)
    enh = LowLightEnhancer("hvicidnet", tmp_path, "cpu")
    img = _img()
    assert np.array_equal(enh(img), classical_low_light(img))
    assert enh._disabled
