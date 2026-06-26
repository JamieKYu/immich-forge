"""Denoise / low-light stage. Runs **first** in the pipeline so later stages —
the upscaler especially — don't amplify sensor noise.

Wired backend: **SCUNet** (Swin-Conv-UNet, blind real-image denoising), vendored
under `scunet/`. Unlike colorize/face-restore, the classical fallback here is a
real denoiser — OpenCV Non-Local Means — so a missing weight or unavailable lib
degrades to `nlm` rather than a no-op. `backend="nlm"` forces that path; `"none"`
disables denoising entirely (low-light can still run).

`strength` blends the denoised result back toward the original (1 = fully
denoised, 0 = original) to dial down over-smoothing. `low_light` applies a
classical CLAHE + gamma brightening pass *after* denoising — the deep denoisers
clean noise but don't brighten, so low-light is handled with classical CV.

All I/O is BGR uint8 (the pipeline convention).
"""
from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np

log = logging.getLogger("forge.denoise")


class Denoiser:
    def __init__(self, backend: str, weights_dir: Path, device: str, tile: int = 512):
        self.backend = backend
        self.weights_dir = weights_dir
        self.device = device
        self.tile = tile
        self._pipe = None                        # cached SCUNet pipeline on load
        self._disabled = backend == "none"       # permanent: denoising off
        self._use_nlm = backend == "nlm"         # classical path (forced or fallen-back)
        self._scunet_failed = False              # permanent SCUNet load/run failure
        self._warned_missing = False             # warn once, not every call

    def _load_scunet(self):
        """Build the SCUNet pipeline. Returns None if it can't (yet)."""
        weights = self.weights_dir / "scunet_color_real_psnr.pth"
        if not weights.exists():
            # Transient — don't disable; meanwhile we denoise with NLM. Retry if
            # the weight is added later (no restart needed). Warn only once.
            if not self._warned_missing:
                log.warning("missing %s; denoising with NLM until present", weights)
                self._warned_missing = True
            return None

        try:
            import torch

            from .scunet import ScunetPipeline, build_scunet_model
        except ImportError as exc:
            log.warning("SCUNet unavailable (%r); denoising with NLM", exc)
            self._scunet_failed = True  # permanent for this process
            return None

        try:
            use_cuda = self.device == "cuda" and torch.cuda.is_available()
            device = torch.device("cuda" if use_cuda else "cpu")
            model = build_scunet_model(str(weights), device)
            log.info("SCUNet loaded on %s", device)
            return ScunetPipeline(model, device=device, tile=self.tile)
        except Exception as exc:  # noqa: BLE001 - bad checkpoint / arch mismatch
            log.warning("SCUNet load failed (%r); denoising with NLM", exc)
            self._scunet_failed = True  # permanent for this process
            return None

    def _denoise(self, img: np.ndarray) -> np.ndarray:
        """Return the denoised image (deep if available, else NLM)."""
        if self._disabled:
            return img
        if not self._use_nlm and not self._scunet_failed:
            if self._pipe is None:
                self._pipe = self._load_scunet()
            if self._pipe is not None:
                try:
                    return self._pipe.process(img)
                except Exception as exc:  # noqa: BLE001 - OOM at min tile, etc.
                    log.warning("SCUNet denoise failed (%r); falling back to NLM", exc)
        # Classical Non-Local Means (genuine denoiser, not a no-op).
        return cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21)

    def __call__(
        self, img: np.ndarray, strength: float = 1.0, low_light: bool = False
    ) -> np.ndarray:
        out = img
        if not self._disabled:
            denoised = self._denoise(img)
            # Blend back toward the original to soften the effect. strength>=1
            # keeps the full denoise; <=0 keeps the original.
            if strength >= 1.0:
                out = denoised
            elif strength <= 0.0:
                out = img
            else:
                out = cv2.addWeighted(denoised, strength, img, 1.0 - strength, 0.0)
        if low_light:
            out = self._enhance_low_light(out)
        return out

    @staticmethod
    def _enhance_low_light(img: np.ndarray) -> np.ndarray:
        """Classical low-light lift: CLAHE on the LAB L-channel for local
        contrast, then a brightening gamma. Cheap, deterministic, no model."""
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        out = cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)

        gamma = 0.7  # <1 brightens midtones/shadows
        lut = np.array(
            [((i / 255.0) ** gamma) * 255 for i in range(256)], dtype=np.uint8
        )
        return cv2.LUT(out, lut)
