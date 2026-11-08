"""Tests for the de-JPEG stage's no-op paths and its place in the pipeline —
no torch. (The FBCNN model path is covered by test_dejpeg_fbcnn.py, which
needs the ML stack.)
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from app.pipeline.dejpeg import JpegCleaner
from app.pipeline.orchestrator import Pipeline
from app.schemas import ForgeOperations
from tests.conftest import make_jpeg


def _img() -> np.ndarray:
    rng = np.random.default_rng(0)
    return rng.integers(0, 256, size=(48, 64, 3), dtype=np.uint8)


def test_backend_none_is_identity():
    img = _img()
    assert np.array_equal(JpegCleaner("none", Path("/nonexistent-weights"), "cpu")(img), img)


def test_missing_weight_is_noop_before_importing_torch():
    cleaner = JpegCleaner("fbcnn", Path("/nonexistent-weights"), "cpu")
    img = _img()
    assert np.array_equal(cleaner(img), img)
    assert cleaner._model is None and not cleaner._disabled  # transient: retried later


def test_dejpeg_runs_first(settings):
    pipe = Pipeline(settings)
    calls = []

    def stage(name):
        def run(img, *args):
            calls.append(name)
            return img
        return run

    pipe.dejpeg = stage("dejpeg")
    pipe.denoiser = stage("denoise")
    pipe.colorizer = stage("colorize")
    pipe.upscaler = stage("upscale")
    pipe.upscaler.effective_factor = lambda img, f: f
    ops = ForgeOperations(dejpeg=True, denoise=True, colorize=True, upscale=True)
    pipe._run_sync(make_jpeg(), ops, lambda p, s: None)
    assert calls == ["dejpeg", "denoise", "colorize", "upscale"]


def test_dejpeg_off_by_default():
    assert ForgeOperations().dejpeg is False
