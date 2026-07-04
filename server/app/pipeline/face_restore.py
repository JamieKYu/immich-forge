"""Face-restoration stage.

`FORGE_FACE_BACKEND` selects the backend:

  * **codeformer** (`codeformer.pth`, vendored under `codeformer/`) — the
    default. Honours `face_fidelity` as CodeFormer's `w` (0 = max quality/most
    restored, 1 = closest to the original face).
  * **gfpgan** (`GFPGANv1.4.pth`, via `gfpgan.GFPGANer`) — fast and robust, but
    has **no fidelity knob**: its architecture ignores the `weight` argument, so
    `face_fidelity` has no effect on GFPGAN output.
  * **gfpgan+codeformer** — a two-pass combo (community technique): GFPGAN first
    to fix eye positioning / facial symmetry, then CodeFormer over that result to
    add back realistic skin texture on top of GFPGAN's smoother "plastic" finish.
    Here `face_fidelity` drives the *second* (CodeFormer) pass — set it low
    (~0.3-0.5) to let CodeFormer's prior add texture. If either weight is missing
    the stage degrades to whichever backend loaded.

When no model/weights are available this stage is a logged no-op rather than a
degraded approximation — classical CV cannot meaningfully restore a face.
All I/O is BGR uint8 (the pipeline convention).
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from .compat import patch_basicsr_torchvision

log = logging.getLogger("forge.face")

# Backends that run each sub-model. A combo runs both, in this order.
_USES_GFPGAN = {"gfpgan", "gfpgan+codeformer"}
_USES_CODEFORMER = {"codeformer", "gfpgan+codeformer"}


class FaceRestorer:
    def __init__(self, backend: str, weights_dir: Path, device: str):
        self.backend = backend
        self.weights_dir = weights_dir
        self.device = device
        self._disabled = backend == "none"       # permanent: backend off
        # Sub-models are loaded lazily and cached; each has an "off" flag set on
        # a permanent failure (missing lib / bad checkpoint) so we don't retry
        # every call. A missing *weight* stays transient (retried, warned once).
        self._gfpgan = None
        self._gfpgan_off = False
        self._codeformer = None
        self._codeformer_off = False
        self._warned: set[str] = set()           # warn-once keys

    def _warn_once(self, key: str, msg: str, *args) -> None:
        if key not in self._warned:
            log.warning(msg, *args)
            self._warned.add(key)

    def _resolve_device(self):
        import torch

        use_cuda = self.device == "cuda" and torch.cuda.is_available()
        return torch.device("cuda" if use_cuda else "cpu")

    def _get_gfpgan(self):
        """Lazily build the GFPGANer. Returns it, or None if it can't (yet)."""
        if self._gfpgan is not None or self._gfpgan_off:
            return self._gfpgan
        weights = self.weights_dir / "GFPGANv1.4.pth"
        if not weights.exists():
            # Transient — retry if the weight is added later (no restart needed).
            self._warn_once(
                "gfpgan-missing", "missing %s; GFPGAN no-ops until present", weights
            )
            return None

        try:
            patch_basicsr_torchvision()  # must precede the gfpgan import
            from gfpgan import GFPGANer
        except ImportError as exc:
            log.warning("gfpgan unavailable (%s); GFPGAN disabled", exc)
            self._gfpgan_off = True  # permanent for this process
            return None

        # arch/channel_multiplier match the v1.4 "clean" model. upscale=1 because
        # the pipeline's upscale stage handles resolution; here we only restore
        # faces in-place. bg_upsampler=None leaves the background untouched.
        #
        # On first run, facexlib downloads its face detection + parsing weights
        # (detection_Resnet50_Final.pth, parsing_parsenet.pth).
        try:
            self._gfpgan = GFPGANer(
                model_path=str(weights),
                upscale=1,
                arch="clean",
                channel_multiplier=2,
                bg_upsampler=None,
                device=self._resolve_device(),
            )
            return self._gfpgan
        except Exception as exc:  # noqa: BLE001 - bad checkpoint / arch mismatch
            log.warning("GFPGAN load failed (%r); GFPGAN disabled", exc)
            self._gfpgan_off = True  # permanent for this process
            return None

    def _get_codeformer(self):
        """Lazily build the CodeFormerPipeline. Returns it, or None if it can't (yet)."""
        if self._codeformer is not None or self._codeformer_off:
            return self._codeformer
        weights = self.weights_dir / "codeformer.pth"
        if not weights.exists():
            # Transient — retry if the weight is added later (no restart needed).
            self._warn_once(
                "codeformer-missing", "missing %s; CodeFormer no-ops until present", weights
            )
            return None

        try:
            patch_basicsr_torchvision()  # must precede the basicsr/facexlib imports
            from .codeformer import CodeFormerPipeline
        except ImportError as exc:
            log.warning("codeformer deps unavailable (%s); CodeFormer disabled", exc)
            self._codeformer_off = True  # permanent for this process
            return None

        try:
            self._codeformer = CodeFormerPipeline(
                str(weights), self._resolve_device(), model_rootpath=str(self.weights_dir)
            )
            return self._codeformer
        except Exception as exc:  # noqa: BLE001 - bad checkpoint / arch mismatch
            log.warning("CodeFormer load failed (%r); CodeFormer disabled", exc)
            self._codeformer_off = True  # permanent for this process
            return None

    @staticmethod
    def _run_gfpgan(model, img: np.ndarray) -> np.ndarray:
        # GFPGANer works on BGR uint8 and returns
        # (cropped_faces, restored_faces, restored_full_bgr). GFPGAN's arch has no
        # fidelity knob, so `face_fidelity` is intentionally not passed here.
        _, _, restored = model.enhance(
            img,
            has_aligned=False,
            only_center_face=False,
            paste_back=True,
        )
        # GFPGAN returns None for restored_img only if no face is found; in that
        # case it still returns the original-shaped image, but guard anyway.
        return restored if restored is not None else img

    def __call__(self, img: np.ndarray, fidelity: float) -> np.ndarray:
        if self._disabled:
            return img

        out = img
        # GFPGAN first (structure/symmetry). It ignores fidelity.
        if self.backend in _USES_GFPGAN:
            model = self._get_gfpgan()
            if model is not None:
                out = self._run_gfpgan(model, out)
        # CodeFormer second. In combo mode this is the texture pass over GFPGAN's
        # output; `fidelity` is its w knob.
        if self.backend in _USES_CODEFORMER:
            model = self._get_codeformer()
            if model is not None:
                out = model.restore(out, fidelity)
        return out
