"""Tests for the FaceRestorer stage's wiring and no-op behavior — no torch,
facexlib or model weights:

  * backend="none" is permanently disabled (returns the input untouched);
  * a missing weight no-ops (the weight-existence check short-circuits before
    importing the ML stack), while still honouring an owed upscale;
  * with the models and the face pipeline stubbed out: which models run per
    backend, in what order, with what fidelity, and what reaches paste-back.

Real restoration needs the weights and the ML stack (which CI omits); the
crop <-> tensor conversion is covered by test_faces.py.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from app.pipeline.face_restore import FaceRestorer


def _restorer(backend: str) -> FaceRestorer:
    # A weights_dir with no .pth so no backend ever loads.
    return FaceRestorer(backend, Path("/nonexistent-weights"), "cpu")


def _img(seed: int = 0, h: int = 48, w: int = 48) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, size=(h, w, 3), dtype=np.uint8)


def test_backend_none_is_identity():
    img = _img()
    out = _restorer("none")(img, fidelity=0.5)
    assert np.array_equal(out, img)


def test_codeformer_missing_weight_is_noop():
    img = _img()
    r = _restorer("codeformer")
    # No codeformer.pth present: returns the input unchanged for any fidelity.
    assert np.array_equal(r(img, fidelity=0.0), img)
    assert np.array_equal(r(img, fidelity=1.0), img)


def test_gfpgan_missing_weight_is_noop():
    img = _img()
    out = _restorer("gfpgan")(img, fidelity=0.5)
    assert np.array_equal(out, img)


def test_combo_missing_weights_is_noop():
    img = _img()
    # gfpgan+codeformer with neither weight present: both passes skip, input
    # is returned unchanged (degrade-to-available also covers the one-present case).
    out = _restorer("gfpgan+codeformer")(img, fidelity=0.4)
    assert np.array_equal(out, img)


def test_missing_backend_with_upscale_falls_back_to_upscaler():
    # Combined face+upscale path when the face model can't load: the stage must
    # still honour the requested upscale via the supplied upscaler rather than
    # silently returning the source resolution. No ML deps are touched (the
    # missing-weight check short-circuits before any import).
    img = _img(h=10, w=10)
    calls: list[int] = []

    def fake_upscaler(im: np.ndarray, factor: int) -> np.ndarray:
        calls.append(factor)
        return np.repeat(np.repeat(im, factor, axis=0), factor, axis=1)

    for backend in ("codeformer", "gfpgan", "gfpgan+codeformer"):
        calls.clear()
        out = _restorer(backend)(img, fidelity=0.5, factor=2, upscaler=fake_upscaler)
        assert calls == [2], backend
        assert out.shape == (20, 20, 3), backend


def test_factor_without_upscaler_is_plain_restore():
    # factor > 1 but no upscaler => no upscale owed, no upscaler calls, in-place.
    img = _img()
    out = _restorer("codeformer")(img, fidelity=0.5, factor=4)
    assert np.array_equal(out, img)


class _Crop:
    """Stands in for an aligned-crop tensor; records which models touched it."""

    def __init__(self, trail=()):
        self.trail = list(trail)

    def clamp(self, lo, hi):
        return self


class _FakeFaces:
    """Stands in for FacePipeline: runs restore_crop once, records the call."""

    def __init__(self):
        self.calls = []
        self.trail = None

    def restore(self, img, restore_crop, factor, bg):
        self.calls.append((factor, None if bg is None else bg.shape))
        self.trail = restore_crop(_Crop()).trail
        return bg if bg is not None else img


def _stubbed(backend: str, gfpgan: bool = True, codeformer: bool = True):
    r = _restorer(backend)
    faces, seen = _FakeFaces(), {}

    def fake_gfpgan(t, return_rgb):
        assert return_rgb is False
        return (_Crop(t.trail + ["gfpgan"]),)

    def fake_codeformer(t, w, adain):
        seen["w"] = w
        return (_Crop(t.trail + ["codeformer"]),)

    r._get_faces = lambda: faces
    r._get_gfpgan = lambda: fake_gfpgan if gfpgan else None
    r._get_codeformer = lambda: fake_codeformer if codeformer else None
    return r, faces, seen


def test_each_backend_runs_only_its_models():
    for backend, trail in (
        ("gfpgan", ["gfpgan"]),
        ("codeformer", ["codeformer"]),
        ("gfpgan+codeformer", ["gfpgan", "codeformer"]),
    ):
        r, faces, _ = _stubbed(backend)
        r(_img(), fidelity=0.5)
        assert faces.trail == trail, backend
        assert len(faces.calls) == 1, backend  # one detect/paste pass, even for the combo


def test_fidelity_is_codeformers_w():
    r, _, seen = _stubbed("gfpgan+codeformer")
    r(_img(), fidelity=0.3)
    assert seen["w"] == 0.3


def test_combo_degrades_to_whichever_model_loaded():
    r, faces, _ = _stubbed("gfpgan+codeformer", gfpgan=False)
    r(_img(), fidelity=0.5)
    assert faces.trail == ["codeformer"]
    r, faces, _ = _stubbed("gfpgan+codeformer", codeformer=False)
    r(_img(), fidelity=0.5)
    assert faces.trail == ["gfpgan"]


def test_combined_upscale_pastes_onto_upscaled_background():
    img = _img(h=10, w=10)
    calls = []

    def fake_upscaler(im, factor):
        calls.append(factor)
        return np.repeat(np.repeat(im, factor, axis=0), factor, axis=1)

    r, faces, _ = _stubbed("codeformer")
    out = r(img, fidelity=0.5, factor=2, upscaler=fake_upscaler)
    assert calls == [2]  # the background is upscaled exactly once
    assert faces.calls == [(2, (20, 20, 3))]  # faces pasted at x2 onto it
    assert out.shape == (20, 20, 3)
