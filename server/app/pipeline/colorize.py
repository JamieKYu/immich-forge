"""Colorization stage.

Wired backend: **DDColor** (ICCV 2023), vendored under `ddcolor/` — a
self-contained torch implementation (no basicsr/timm dependency, so it doesn't
collide with the basicsr used by Real-ESRGAN). When the model/weights are
unavailable this stage is a logged no-op (it only ensures a 3-channel output).
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
        self._pipe = None  # ColorizationPipeline | "skip"

    def _load(self):
        """Build the DDColor pipeline, or None if unavailable."""
        if self.backend == "none":
            return None

        weights = self.weights_dir / "ddcolor_modelscope.pth"
        if not weights.exists():
            log.warning("missing %s; skipping colorize", weights)
            return None

        try:
            import torch

            from .ddcolor import ColorizationPipeline, DDColor, build_ddcolor_model
        except ImportError as exc:
            log.warning("ddcolor unavailable (%r); skipping colorize", exc)
            return None

        use_cuda = self.device == "cuda" and torch.cuda.is_available()
        device = torch.device("cuda" if use_cuda else "cpu")
        # The modelscope checkpoint is the "large" (convnext-l) model at 512px,
        # matching build_ddcolor_model's defaults (MultiScaleColorDecoder).
        model = build_ddcolor_model(
            DDColor, model_path=str(weights), model_size="large",
            input_size=512, device=device,
        )
        log.info("DDColor loaded on %s", device)
        return ColorizationPipeline(model, input_size=512, device=device)

    def __call__(self, img: np.ndarray) -> np.ndarray:
        if self._pipe is None:
            self._pipe = self._load() or "skip"
        if self._pipe == "skip":
            # Only meaningful on a grayscale source; ensure 3-channel output.
            if img.ndim == 2:
                return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            return img
        # DDColor's pipeline takes and returns BGR uint8 — our convention.
        return self._pipe.process(img)
