"""Face-restoration stage.

Two backends share this stage, selected by `FORGE_FACE_BACKEND`:

  * **gfpgan** (`GFPGANv1.4.pth`, via `gfpgan.GFPGANer`) — the default. Fast and
    robust, but has **no fidelity knob**: its architecture ignores the `weight`
    argument, so `face_fidelity` has no effect on GFPGAN output.
  * **codeformer** (`codeformer.pth`, vendored under `codeformer/`) — honours
    `face_fidelity` as CodeFormer's `w` (0 = max quality/most restored,
    1 = closest to the original face).

When the model/weights are unavailable this stage is a logged no-op rather than
a degraded approximation — classical CV cannot meaningfully restore a face.
All I/O is BGR uint8 (the pipeline convention).
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from .compat import patch_basicsr_torchvision

log = logging.getLogger("forge.face")


class FaceRestorer:
    def __init__(self, backend: str, weights_dir: Path, device: str):
        self.backend = backend
        self.weights_dir = weights_dir
        self.device = device
        self._model = None                       # cached on successful load
        self._disabled = backend == "none"       # permanent: backend off / libs absent
        self._warned_missing = False             # warn once, not every call

    def _resolve_device(self):
        import torch

        use_cuda = self.device == "cuda" and torch.cuda.is_available()
        return torch.device("cuda" if use_cuda else "cpu")

    def _load_gfpgan(self):
        """Load GFPGAN. Returns the GFPGANer, or None if it can't (yet)."""
        weights = self.weights_dir / "GFPGANv1.4.pth"
        if not weights.exists():
            # Transient — retry if the weight is added later (no restart needed).
            if not self._warned_missing:
                log.warning("missing %s; face restore no-ops until present", weights)
                self._warned_missing = True
            return None

        try:
            patch_basicsr_torchvision()  # must precede the gfpgan import
            from gfpgan import GFPGANer
        except ImportError as exc:
            log.warning("gfpgan unavailable (%s); face restore disabled", exc)
            self._disabled = True  # permanent for this process
            return None

        # arch/channel_multiplier match the v1.4 "clean" model. upscale=1 because
        # the pipeline's upscale stage handles resolution; here we only restore
        # faces in-place. bg_upsampler=None leaves the background untouched.
        #
        # On first run, facexlib downloads its face detection + parsing weights
        # (detection_Resnet50_Final.pth, parsing_parsenet.pth) into gfpgan/weights/.
        try:
            return GFPGANer(
                model_path=str(weights),
                upscale=1,
                arch="clean",
                channel_multiplier=2,
                bg_upsampler=None,
                device=self._resolve_device(),
            )
        except Exception as exc:  # noqa: BLE001 - bad checkpoint / arch mismatch
            log.warning("GFPGAN load failed (%r); face restore disabled", exc)
            self._disabled = True  # permanent for this process
            return None

    def _load_codeformer(self):
        """Load CodeFormer. Returns a CodeFormerPipeline, or None if it can't (yet)."""
        weights = self.weights_dir / "codeformer.pth"
        if not weights.exists():
            # Transient — retry if the weight is added later (no restart needed).
            if not self._warned_missing:
                log.warning("missing %s; face restore no-ops until present", weights)
                self._warned_missing = True
            return None

        try:
            patch_basicsr_torchvision()  # must precede the basicsr/facexlib imports
            from .codeformer import CodeFormerPipeline
        except ImportError as exc:
            log.warning("codeformer deps unavailable (%s); face restore disabled", exc)
            self._disabled = True  # permanent for this process
            return None

        try:
            return CodeFormerPipeline(
                str(weights), self._resolve_device(), model_rootpath=str(self.weights_dir)
            )
        except Exception as exc:  # noqa: BLE001 - bad checkpoint / arch mismatch
            log.warning("CodeFormer load failed (%r); face restore disabled", exc)
            self._disabled = True  # permanent for this process
            return None

    def _load(self):
        if self.backend == "codeformer":
            return self._load_codeformer()
        return self._load_gfpgan()

    def __call__(self, img: np.ndarray, fidelity: float) -> np.ndarray:
        if self._model is None and not self._disabled:
            self._model = self._load()
        if self._model is None:
            return img

        if self.backend == "codeformer":
            # CodeFormer consumes `fidelity` as its w knob.
            return self._model.restore(img, fidelity)

        # GFPGANer works on BGR uint8 and returns
        # (cropped_faces, restored_faces, restored_full_bgr). GFPGAN's arch has no
        # fidelity knob, so `fidelity` is intentionally not passed here.
        _, _, restored = self._model.enhance(
            img,
            has_aligned=False,
            only_center_face=False,
            paste_back=True,
        )
        # GFPGAN returns None for restored_img only if no face is found; in that
        # case it still returns the original-shaped image, but guard anyway.
        return restored if restored is not None else img
