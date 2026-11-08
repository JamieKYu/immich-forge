"""JPEG artifact removal (de-JPEG). Runs **first** in the pipeline, so the
blocking and ringing a heavy JPEG re-save left behind aren't brightened,
"denoised" into smears, or upscaled into visible tiling by later stages.

Wired backend: **FBCNN** (ICCV 2021, Apache-2.0; `fbcnn_color.pth`), loaded
via `spandrel`. It's *blind*: it predicts the JPEG quality factor from the
image itself and conditions its restoration on that, so it needs no setting
and adapts per photo. It does lightly smooth an already-clean image, which is
why this stage is opt-in (`dejpeg`) rather than always on. `FORGE_DEJPEG_BACKEND`
is `fbcnn` or `none`.

Tiled via `tiling.py` (FBCNN is a plain CNN, so tiles agree), with tile sizes
kept on the 8x8 JPEG block grid so every tile sees blocks where they really
are. When the model/weights are unavailable this stage is a logged no-op —
classical deblocking filters aren't worth substituting.

All I/O is BGR uint8 (the pipeline convention).
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from .tiling import enhance_tiled

log = logging.getLogger("forge.dejpeg")

WEIGHTS = "fbcnn_color.pth"
# JPEG's DCT block size; tiles must stay aligned to it.
_JPEG_BLOCK = 8


class JpegCleaner:
    """Lazy-loaded de-JPEG stage. `backend` selects the implementation."""

    def __init__(self, backend: str, weights_dir: Path, device: str, tile: int = 512):
        self.backend = backend
        self.weights_dir = weights_dir
        self.device = device
        self.tile = tile
        self._model = None                     # spandrel descriptor, cached on load
        self._disabled = backend != "fbcnn"   # permanent: backend off / libs absent
        self._warned_missing = False          # warn once, not every call

    def _load(self):
        """Load FBCNN. Returns None if the lib/weights aren't (yet) available."""
        weights = self.weights_dir / WEIGHTS
        if not weights.exists():
            # Transient — retry if the weight is added later (no restart needed).
            if not self._warned_missing:
                log.warning("missing %s; de-JPEG no-ops until present", weights)
                self._warned_missing = True
            return None

        try:
            import torch
            from spandrel import ImageModelDescriptor, ModelLoader
        except ImportError as exc:
            log.warning("spandrel unavailable (%r); de-JPEG disabled", exc)
            self._disabled = True  # permanent for this process
            return None

        try:
            model = ModelLoader().load_from_file(weights)
            if not isinstance(model, ImageModelDescriptor) or model.scale != 1:
                raise TypeError(f"{model.architecture.name} is not a 1x restoration model")
        except Exception as exc:  # noqa: BLE001 - bad checkpoint / unsupported arch
            log.warning("de-JPEG model load failed (%r); de-JPEG disabled", exc)
            self._disabled = True  # permanent for this process
            return None

        use_cuda = self.device == "cuda" and torch.cuda.is_available()
        # fp32: spandrel's FBCNN half-precision support is unverified ("TODO").
        model.to(torch.device("cuda" if use_cuda else "cpu")).eval()
        log.info("de-JPEG model %s loaded on %s", model.architecture.name, model.device)
        return model

    def __call__(self, img: np.ndarray) -> np.ndarray:
        if self._model is None and not self._disabled:
            self._model = self._load()
        if self._model is None:
            return img  # logged no-op
        out = enhance_tiled(
            self._model, img, self.tile, stage="de-JPEG", multiple=_JPEG_BLOCK
        )
        if out is None:
            log.warning("de-JPEG skipped: out of GPU memory at every tile size")
            return img
        return out
