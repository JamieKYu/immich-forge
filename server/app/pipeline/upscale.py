"""Upscaling stage. Prefers Real-ESRGAN on GPU; falls back to Lanczos.

The network is loaded with `spandrel` (the chaiNNer/ComfyUI model loader),
which detects the architecture from the checkpoint itself — no `realesrgan` /
`basicsr` import, and no torchvision shim. Tiling (with OOM tile-halving) is in
`tiling.py`, shared with the de-JPEG stage.
"""
from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np

from .tiling import enhance_tiled

log = logging.getLogger("forge.upscale")


class Upscaler:
    """Lazy-loaded upscaler. `backend` selects the implementation."""

    def __init__(
        self, backend: str, weights_dir: Path, device: str, tile: int,
        max_output_pixels: int = 0,
    ):
        self.backend = backend
        self.weights_dir = weights_dir
        self.device = device
        self.tile = tile
        self.max_output_pixels = max_output_pixels
        self._model = None                       # spandrel descriptor, cached on load
        self._disabled = backend != "realesrgan"  # permanent: backend off / libs absent
        self._warned_missing = False             # warn once, not every call

    def _load_realesrgan(self):
        """Load Real-ESRGAN. Returns None if the lib/weights aren't (yet) available."""
        weights = self.weights_dir / "RealESRGAN_x4plus.pth"
        if not weights.exists():
            # Transient — retry if the weight is added later (no restart needed).
            if not self._warned_missing:
                log.warning("missing %s; upscaling with lanczos until present", weights)
                self._warned_missing = True
            return None

        try:
            import torch
            from spandrel import ImageModelDescriptor, ModelLoader
        except ImportError as exc:
            log.warning("spandrel unavailable, using lanczos: %r", exc)
            self._disabled = True  # permanent for this process
            return None

        try:
            model = ModelLoader().load_from_file(weights)
            if not isinstance(model, ImageModelDescriptor):
                raise TypeError(f"{model.architecture.name} is not an image-to-image model")
        except Exception as exc:  # noqa: BLE001 - bad checkpoint / unsupported arch
            log.warning("upscale model load failed (%r); using lanczos", exc)
            self._disabled = True  # permanent for this process
            return None

        use_cuda = self.device == "cuda" and torch.cuda.is_available()
        model.to(torch.device("cuda" if use_cuda else "cpu")).eval()
        if use_cuda and model.supports_half:
            model.half()
        log.info(
            "upscale model %s x%d loaded on %s",
            model.architecture.name, model.scale, model.device,
        )
        return model

    def effective_factor(self, img: np.ndarray, factor: int) -> int:
        """The upscale factor clamped so the output stays under max_output_pixels.

        Pure (no logging) so callers can use it both to decide and to report.
        """
        if self.max_output_pixels <= 0:
            return factor
        h, w = img.shape[:2]
        allowed = int((self.max_output_pixels / (h * w)) ** 0.5)
        return max(1, min(factor, allowed))

    def __call__(self, img: np.ndarray, factor: int) -> np.ndarray:
        requested = factor
        factor = self.effective_factor(img, factor)
        if factor < requested:
            h, w = img.shape[:2]
            log.warning(
                "clamping upscale x%d -> x%d: source %dx%d would exceed %d output px",
                requested, factor, w, h, self.max_output_pixels,
            )
        if factor <= 1:
            return img  # source already at/over the output cap; leave as-is
        if self._model is None and not self._disabled:
            self._model = self._load_realesrgan()
        if self._model is not None:
            out = self._enhance(img, factor)
            if out is not None:
                return out
            # GPU couldn't fit it at any tile size — degrade to Lanczos below.
        # Fallback: classical Lanczos resize.
        h, w = img.shape[:2]
        return cv2.resize(
            img, (w * factor, h * factor), interpolation=cv2.INTER_LANCZOS4
        )

    def _enhance(self, img: np.ndarray, factor: int) -> np.ndarray | None:
        """Run the model (tiled, halving the tile on OOM). Returns None to tell
        the caller to fall back to Lanczos (GPU too contended to fit a tile).
        """
        out = enhance_tiled(self._model, img, self.tile, stage="upscale")
        if out is None:
            return None
        if factor != self._model.scale:
            # The network has a fixed scale (x4); resample to the requested
            # factor, as RealESRGANer's `outscale` did.
            h, w = img.shape[:2]
            out = cv2.resize(
                out, (w * factor, h * factor), interpolation=cv2.INTER_LANCZOS4
            )
        return out
