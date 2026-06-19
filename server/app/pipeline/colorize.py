"""Colorization stage. Prefers DeOldify; falls back to passthrough.

DeOldify pins older fastai/torch and is best isolated. The scaffold leaves it as
an explicit integration point and no-ops when unavailable.
"""
from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np

log = logging.getLogger("forge.colorize")


class Colorizer:
    def __init__(self, backend: str, weights_dir: Path, device: str):
        self.backend = backend
        self.weights_dir = weights_dir
        self.device = device
        self._model = None

    def _load(self):
        if self.backend == "none":
            return None
        try:
            # from deoldify.visualize import get_image_colorizer
            # return get_image_colorizer(artistic=True)
            raise ImportError("deoldify integration not wired in scaffold")
        except ImportError:
            log.warning("colorize backend %r unavailable; skipping colorize", self.backend)
            return None

    def __call__(self, img: np.ndarray) -> np.ndarray:
        if self._model is None:
            self._model = self._load() or "skip"
        if self._model == "skip":
            # Only meaningful on a grayscale source; ensure 3-channel output.
            if img.ndim == 2:
                return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            return img
        # DeOldify works on PIL/RGB; orchestrator hands us BGR ndarray.
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        out_rgb = self._model.get_transformed_image_from_ndarray(rgb)
        return cv2.cvtColor(np.asarray(out_rgb), cv2.COLOR_RGB2BGR)
