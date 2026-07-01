"""Tests for the FaceRestorer stage's no-op behavior. These stay off the deep
models — they never import torch/gfpgan/basicsr — by exercising only the paths
that short-circuit before a load:

  * backend="none" is permanently disabled (returns the input untouched);
  * backend="codeformer"/"gfpgan" with a missing weight no-op (the
    weight-existence check short-circuits before importing the ML stack).

The actual restoration + fidelity behavior needs the model weights and the ML
deps, which CI intentionally omits (GPU-free); that path is covered by loading
the real codeformer.pth into the vendored arch during development.
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
