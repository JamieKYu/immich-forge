"""Tests for the shared tiling helpers' pure logic — no torch, no GPU.
(Tiled inference itself is covered by the spandrel stage tests.)
"""
from __future__ import annotations

from app.pipeline.tiling import MIN_TILE, TILE_PAD, looks_like_oom, tile_ladder


def test_ladder_halves_down_to_min_tile():
    assert tile_ladder(512) == [512, 256, 128]


def test_ladder_tiling_off_is_single_whole_image_attempt():
    assert tile_ladder(0) == [0]
    assert tile_ladder(-1) == [0]


def test_ladder_below_min_tile_is_empty():
    # Configured smaller than MIN_TILE: nothing to try (caller falls back).
    assert tile_ladder(MIN_TILE - 1) == []


def test_ladder_rounds_to_multiple_for_jpeg_grid():
    # 500 isn't on the 8x8 block grid; every rung must be.
    ladder = tile_ladder(500, multiple=8)
    assert ladder == [496, 248]  # 500 -> 250 -> 125 < MIN_TILE stops
    assert all(t % 8 == 0 for t in ladder)


def test_tile_pad_keeps_jpeg_grid_alignment():
    # Tile origins are y0 - TILE_PAD with y0 a multiple of the tile size.
    assert TILE_PAD % 8 == 0


def test_looks_like_oom_detects_by_message_and_type():
    assert looks_like_oom(RuntimeError("CUDA out of memory. Tried to allocate"))

    class OutOfMemoryError(Exception):
        pass

    assert looks_like_oom(OutOfMemoryError("boom"))


def test_looks_like_oom_ignores_unrelated_errors():
    assert not looks_like_oom(ValueError("bad shape"))
    assert not looks_like_oom(RuntimeError("some other failure"))
