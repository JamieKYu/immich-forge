"""Tests for the spandrel-backed upscale path: model load, tiling, OOM ladder.

Needs torch + spandrel (requirements-ml.txt), so it's skipped in the GPU-free
CI. A tiny randomly-initialised ESRGAN stands in for RealESRGAN_x4plus.pth —
spandrel detects the architecture from the state_dict either way — and
everything runs on CPU.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("spandrel")

from spandrel.architectures.ESRGAN import ESRGAN  # noqa: E402

from app.pipeline import tiling  # noqa: E402
from app.pipeline.tiling import TILE_PAD  # noqa: E402
from app.pipeline.upscale import Upscaler  # noqa: E402


@pytest.fixture
def weights_dir(tmp_path: Path) -> Path:
    torch.manual_seed(0)
    net = ESRGAN(num_filters=8, num_blocks=1, scale=4)
    torch.save(net.state_dict(), tmp_path / "RealESRGAN_x4plus.pth")
    return tmp_path


def _upscaler(weights_dir: Path, tile: int = 0) -> Upscaler:
    return Upscaler("realesrgan", weights_dir, "cpu", tile)


def _img(h: int = 40, w: int = 56, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, size=(h, w, 3), dtype=np.uint8)


def test_loads_checkpoint_and_upscales_x4(weights_dir):
    up = _upscaler(weights_dir)
    out = up(_img(), 4)
    assert up._model is not None and up._model.scale == 4
    assert out.shape == (160, 224, 3) and out.dtype == np.uint8


def test_factor_below_model_scale_is_resampled(weights_dir):
    out = _upscaler(weights_dir)(_img(), 2)
    assert out.shape == (80, 112, 3)


def test_tiled_matches_whole_image_away_from_seams(weights_dir):
    img = _img(300, 260)
    whole = _upscaler(weights_dir, tile=0)(img, 4)
    tiled = _upscaler(weights_dir, tile=128)(img, 4)
    assert tiled.shape == whole.shape
    # Each tile sees TILE_PAD px of context, less than the net's receptive
    # field, so seams may differ slightly — but only slightly.
    diff = np.abs(tiled.astype(int) - whole.astype(int))
    assert diff.mean() < 1.0
    # A tile interior far from any seam is identical.
    assert np.array_equal(tiled[100:400, 100:400], whole[100:400, 100:400])


def test_oom_shrinks_tile_and_recovers(weights_dir, monkeypatch):
    up = _upscaler(weights_dir, tile=256)
    up._model = up._load_realesrgan()
    real_infer = tiling.infer
    seen = []

    def infer(model, patch):
        seen.append(patch.shape[-1])
        if patch.shape[-1] > 128 + 2 * TILE_PAD:
            raise torch.OutOfMemoryError("CUDA out of memory")
        return real_infer(model, patch)

    monkeypatch.setattr(tiling, "infer", infer)
    out = up(_img(300, 300), 4)
    assert out.shape == (1200, 1200, 3)
    fits = 128 + 2 * TILE_PAD
    assert max(seen) > fits and seen[-1] <= fits  # tried 256, finished at 128


def test_oom_at_every_tile_falls_back_to_lanczos(weights_dir, monkeypatch):
    up = _upscaler(weights_dir, tile=256)
    up._model = up._load_realesrgan()

    def infer(model, patch):
        raise torch.OutOfMemoryError("CUDA out of memory")

    monkeypatch.setattr(tiling, "infer", infer)
    out = up(_img(), 4)
    assert out.shape == (160, 224, 3)  # Lanczos still delivers the factor


def test_non_oom_error_propagates(weights_dir, monkeypatch):
    up = _upscaler(weights_dir, tile=256)
    up._model = up._load_realesrgan()

    def infer(model, patch):
        raise ValueError("bad shape")

    monkeypatch.setattr(tiling, "infer", infer)
    with pytest.raises(ValueError):
        up(_img(), 4)


def test_unrecognised_checkpoint_disables_to_lanczos(tmp_path):
    torch.save({"not": torch.zeros(1)}, tmp_path / "RealESRGAN_x4plus.pth")
    up = _upscaler(tmp_path)
    out = up(_img(), 4)
    assert up._model is None and up._disabled
    assert out.shape == (160, 224, 3)
