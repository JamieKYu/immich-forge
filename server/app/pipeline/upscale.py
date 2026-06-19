"""Upscaling stage. Prefers Real-ESRGAN on GPU; falls back to Lanczos."""
from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np

log = logging.getLogger("forge.upscale")


class Upscaler:
    """Lazy-loaded upscaler. `backend` selects the implementation."""

    def __init__(self, backend: str, weights_dir: Path, device: str, tile: int):
        self.backend = backend
        self.weights_dir = weights_dir
        self.device = device
        self.tile = tile
        self._model = None  # RealESRGANer, loaded on first use

    def _load_realesrgan(self):
        """Load Real-ESRGAN. Returns None if the lib/weights are unavailable."""
        try:
            import torch
            from basicsr.archs.rrdbnet_arch import RRDBNet
            from realesrgan import RealESRGANer
        except ImportError:
            log.warning("realesrgan/basicsr not installed; falling back to lanczos")
            return None

        weights = self.weights_dir / "RealESRGAN_x4plus.pth"
        if not weights.exists():
            log.warning("missing %s; falling back to lanczos", weights)
            return None

        model = RRDBNet(
            num_in_ch=3, num_out_ch=3, num_feat=64,
            num_block=23, num_grow_ch=32, scale=4,
        )
        half = self.device == "cuda" and torch.cuda.is_available()
        return RealESRGANer(
            scale=4,
            model_path=str(weights),
            model=model,
            tile=self.tile,
            tile_pad=10,
            half=half,
            device=self.device if half else "cpu",
        )

    def __call__(self, img: np.ndarray, factor: int) -> np.ndarray:
        if self.backend == "realesrgan":
            if self._model is None:
                self._model = self._load_realesrgan() or "lanczos"
            if self._model != "lanczos":
                out, _ = self._model.enhance(img, outscale=factor)
                return out
        # Fallback: classical Lanczos resize.
        h, w = img.shape[:2]
        return cv2.resize(
            img, (w * factor, h * factor), interpolation=cv2.INTER_LANCZOS4
        )
