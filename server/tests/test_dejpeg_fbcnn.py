"""Tests for the FBCNN de-JPEG path: load, tiling on the JPEG grid, the OOM
ladder, and fallbacks.

Needs torch + spandrel (requirements-ml.txt), so it's skipped in the GPU-free
CI. A small randomly-initialised FBCNN stands in for fbcnn_color.pth —
spandrel detects the architecture from the state_dict either way — and
everything runs on CPU.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("spandrel")

from spandrel.architectures.FBCNN import FBCNN  # noqa: E402

from app.pipeline import tiling  # noqa: E402
from app.pipeline.dejpeg import WEIGHTS, JpegCleaner  # noqa: E402
from app.pipeline.tiling import TILE_PAD  # noqa: E402


@pytest.fixture(scope="module")
def weights_dir(tmp_path_factory) -> Path:
    # spandrel only detects FBCNN with the standard nb=4, and its QF head is
    # hard-wired to 512 channels at the deepest level, so this is as small as
    # it gets (~230MB) — saved once per module, not per test.
    d = tmp_path_factory.mktemp("fbcnn")
    torch.manual_seed(0)
    torch.save(FBCNN(nc=[8, 16, 32, 512], nb=4).state_dict(), d / WEIGHTS)
    return d


def _cleaner(weights_dir: Path, tile: int = 0) -> JpegCleaner:
    return JpegCleaner("fbcnn", weights_dir, "cpu", tile)


def _img(h: int = 300, w: int = 260, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, size=(h, w, 3), dtype=np.uint8)


def test_loads_and_keeps_shape(weights_dir):
    c = _cleaner(weights_dir)
    img = _img(61, 83)  # not a multiple of 8: FBCNN pads internally
    out = c(img)
    assert c._model is not None and c._model.architecture.name == "FBCNN"
    assert out.shape == img.shape and out.dtype == np.uint8


def test_tiles_stay_on_jpeg_block_grid(weights_dir, monkeypatch):
    c = _cleaner(weights_dir, tile=500)  # deliberately off-grid
    run_tiled, tiles = tiling.run_tiled, []

    def spy_run_tiled(model, x, tile):
        tiles.append(tile)
        return run_tiled(model, x, tile)

    monkeypatch.setattr(tiling, "run_tiled", spy_run_tiled)
    c(_img(600, 600))
    assert tiles == [496]  # rounded down onto the 8px grid


def test_tiled_close_to_whole_image(weights_dir):
    img = _img()
    whole = _cleaner(weights_dir, tile=0)(img)
    tiled = _cleaner(weights_dir, tile=128)(img)
    assert tiled.shape == whole.shape
    # Not bit-exact even in tile interiors: FBCNN predicts the JPEG quality
    # from each tile it's given. With the real weights every tile predicts the
    # same QF as the whole image and tiled/whole PSNR is identical (max diff
    # 7/255 on 0.02% of pixels at q=10); random weights just drift a little.
    assert np.abs(tiled.astype(int) - whole.astype(int)).mean() < 1.0


def test_oom_shrinks_tile_and_recovers(weights_dir, monkeypatch):
    c = _cleaner(weights_dir, tile=256)
    real, seen = tiling.infer, []

    def infer(model, patch):
        seen.append(patch.shape[-1])
        if patch.shape[-1] > 128 + 2 * TILE_PAD:
            raise torch.OutOfMemoryError("CUDA out of memory")
        return real(model, patch)

    monkeypatch.setattr(tiling, "infer", infer)
    img = _img()
    out = c(img)
    assert out.shape == img.shape
    assert seen[-1] <= 128 + 2 * TILE_PAD


def test_oom_at_every_tile_returns_input(weights_dir, monkeypatch):
    c = _cleaner(weights_dir, tile=256)

    def infer(model, patch):
        raise torch.OutOfMemoryError("CUDA out of memory")

    monkeypatch.setattr(tiling, "infer", infer)
    img = _img()
    assert np.array_equal(c(img), img)


def test_unrecognised_checkpoint_is_noop(tmp_path):
    torch.save({"not": torch.zeros(1)}, tmp_path / WEIGHTS)
    c = JpegCleaner("fbcnn", tmp_path, "cpu")
    img = _img(40, 40)
    assert np.array_equal(c(img), img)
    assert c._disabled
