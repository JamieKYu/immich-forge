"""Face-restoration stage.

Wired backend: **GFPGAN** (`GFPGANv1.4.pth`). CodeFormer shares the same
`GFPGANer` runtime and is left as a config option. When the model/weights are
unavailable this stage is a logged no-op rather than a degraded approximation —
classical CV cannot meaningfully restore a face.
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

    def _load(self):
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
            import torch
            from gfpgan import GFPGANer
        except ImportError as exc:
            log.warning("gfpgan unavailable (%s); face restore disabled", exc)
            self._disabled = True  # permanent for this process
            return None

        use_cuda = self.device == "cuda" and torch.cuda.is_available()
        device = torch.device("cuda" if use_cuda else "cpu")
        # arch/channel_multiplier match the v1.4 "clean" model. upscale=1 because
        # the pipeline's upscale stage handles resolution; here we only restore
        # faces in-place. bg_upsampler=None leaves the background untouched.
        #
        # On first run, facexlib downloads its face detection + parsing weights
        # (detection_Resnet50_Final.pth, parsing_parsenet.pth) into gfpgan/weights/.
        return GFPGANer(
            model_path=str(weights),
            upscale=1,
            arch="clean",
            channel_multiplier=2,
            bg_upsampler=None,
            device=device,
        )

    def __call__(self, img: np.ndarray, fidelity: float) -> np.ndarray:
        if self._model is None and not self._disabled:
            self._model = self._load()
        if self._model is None:
            return img

        # GFPGANer works on BGR uint8 (our pipeline convention) and returns
        # (cropped_faces, restored_faces, restored_full_bgr).
        # `weight` is the CodeFormer fidelity knob; ignored by the GFPGAN arch.
        _, _, restored = self._model.enhance(
            img,
            has_aligned=False,
            only_center_face=False,
            paste_back=True,
            weight=fidelity,
        )
        # GFPGAN returns None for restored_img only if no face is found; in that
        # case it still returns the original-shaped image, but guard anyway.
        return restored if restored is not None else img
