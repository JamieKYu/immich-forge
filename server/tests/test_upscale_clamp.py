"""Tests for the upscale clamp math and OOM detection — pure logic, no GPU.

The Upscaler is constructed with a non-realesrgan backend so it stays in its
permanently-disabled (classical) state and never touches torch.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from app.pipeline.upscale import Upscaler, _looks_like_oom


def _upscaler(max_output_pixels: int) -> Upscaler:
    return Upscaler(
        backend="lanczos",
        weights_dir=Path("weights"),
        device="cpu",
        tile=0,
        max_output_pixels=max_output_pixels,
    )


def _img(h: int, w: int) -> np.ndarray:
    return np.zeros((h, w, 3), dtype=np.uint8)


def test_no_cap_returns_requested_factor():
    up = _upscaler(max_output_pixels=0)
    assert up.effective_factor(_img(1000, 1000), 4) == 4


def test_factor_unclamped_when_output_fits():
    up = _upscaler(max_output_pixels=100_000_000)
    # 1000x1000 * 4x = 16MP, well under 100MP.
    assert up.effective_factor(_img(1000, 1000), 4) == 4


def test_factor_clamped_when_output_would_exceed_cap():
    up = _upscaler(max_output_pixels=10_000_000)  # 10MP cap
    # 2000x2000 source = 4MP; x4 -> 64MP (too big). sqrt(10MP/4MP) = 1.58 -> 1.
    assert up.effective_factor(_img(2000, 2000), 4) == 1


def test_factor_clamped_to_intermediate_value():
    up = _upscaler(max_output_pixels=16_000_000)  # 16MP cap
    # 1000x1000 = 1MP. allowed = sqrt(16MP/1MP) = 4. Request 4 -> stays 4.
    assert up.effective_factor(_img(1000, 1000), 4) == 4
    # 2000x2000 = 4MP. allowed = sqrt(16MP/4MP) = 2. Request 4 -> clamp to 2.
    assert up.effective_factor(_img(2000, 2000), 4) == 2


def test_factor_never_below_one():
    up = _upscaler(max_output_pixels=1)  # absurdly small
    assert up.effective_factor(_img(4000, 4000), 4) == 1


def test_requested_factor_two_not_raised():
    up = _upscaler(max_output_pixels=100_000_000)
    # Clamp only lowers; a request of 2 with lots of headroom stays 2.
    assert up.effective_factor(_img(500, 500), 2) == 2


def test_looks_like_oom_detects_direct_and_disguised():
    assert _looks_like_oom(RuntimeError("CUDA out of memory. Tried to allocate"))
    # Real-ESRGAN's tile loop masks the OOM as an UnboundLocalError on output_tile.
    assert _looks_like_oom(
        UnboundLocalError("local variable 'output_tile' referenced before assignment")
    )

    class OutOfMemoryError(Exception):
        pass

    assert _looks_like_oom(OutOfMemoryError("boom"))


def test_looks_like_oom_ignores_unrelated_errors():
    assert not _looks_like_oom(ValueError("bad shape"))
    assert not _looks_like_oom(RuntimeError("some other failure"))
