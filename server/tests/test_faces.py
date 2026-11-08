"""Tests for faces.py's crop <-> tensor conversion — the [-1, 1] range GFPGAN
and CodeFormer were trained on. Needs torch + facexlib (requirements-ml.txt),
so it's skipped in the GPU-free CI.
"""
from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("facexlib")

from app.pipeline.faces import crop_to_tensor, tensor_to_crop  # noqa: E402


def _crop() -> np.ndarray:
    rng = np.random.default_rng(0)
    return rng.integers(0, 256, size=(64, 48, 3), dtype=np.uint8)


def test_crop_to_tensor_is_rgb_in_minus_one_to_one():
    crop = _crop()
    t = crop_to_tensor(crop, torch.device("cpu"))
    assert t.shape == (1, 3, 64, 48) and t.dtype == torch.float32
    assert t.min() >= -1.0 and t.max() <= 1.0
    # Channel 0 is red: BGR -> RGB.
    assert torch.allclose(t[0, 0], torch.from_numpy(crop[..., 2]).float() / 127.5 - 1)


def test_round_trip_is_exact():
    crop = _crop()
    assert np.array_equal(tensor_to_crop(crop_to_tensor(crop, torch.device("cpu"))), crop)


def test_tensor_to_crop_clamps_out_of_range():
    t = torch.full((1, 3, 4, 4), 1.7)
    assert (tensor_to_crop(t) == 255).all()
    assert (tensor_to_crop(-t) == 0).all()


def test_works_on_inference_tensors():
    # Model outputs come from inference_mode; conversion must not mutate them.
    with torch.inference_mode():
        t = crop_to_tensor(_crop(), torch.device("cpu")) * 1.0
    assert tensor_to_crop(t).shape == (64, 48, 3)
