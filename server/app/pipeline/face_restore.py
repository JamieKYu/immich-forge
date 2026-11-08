"""Face-restoration stage.

`FORGE_FACE_BACKEND` selects the backend:

  * **codeformer** (`codeformer.pth`, vendored under `codeformer/`) — the
    default. Honours `face_fidelity` as CodeFormer's `w` (0 = max quality/most
    restored, 1 = closest to the original face).
  * **gfpgan** (`GFPGANv1.4.pth`, loaded via `spandrel`) — fast and robust, but
    has **no fidelity knob**, so `face_fidelity` has no effect on GFPGAN output.
  * **gfpgan+codeformer** — a combo (community technique): GFPGAN fixes eye
    positioning / facial symmetry, then CodeFormer adds back realistic skin
    texture on top of GFPGAN's smoother "plastic" finish. Both run on the same
    aligned crop, one after the other, before a single paste-back.
    `face_fidelity` drives the CodeFormer pass — set it low (~0.3-0.5) to let
    its prior add texture. If either weight is missing the stage degrades to
    whichever model loaded.

Detection, alignment and paste-back are shared (`faces.py`, facexlib); the
models only see aligned 512px crops. When no model/weights are available this
stage is a logged no-op rather than a degraded approximation — classical CV
cannot meaningfully restore a face. All I/O is BGR uint8 (the pipeline
convention).

**Combined face+upscale.** When the caller passes a `factor > 1` and an
`upscaler`, this stage restores *and* upscales in one pass: the background is
upscaled by Real-ESRGAN and the restored 512px face crops are pasted onto that
upscaled canvas last. The general upscaler therefore never runs over face
pixels (which turns them waxy/"un-human"). With `factor == 1` (or no upscaler)
it's a plain in-place restore at the source resolution.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

log = logging.getLogger("forge.face")

GFPGAN_WEIGHTS = "GFPGANv1.4.pth"
CODEFORMER_WEIGHTS = "codeformer.pth"

# Backends that run each sub-model. A combo runs both, in this order.
_USES_GFPGAN = {"gfpgan", "gfpgan+codeformer"}
_USES_CODEFORMER = {"codeformer", "gfpgan+codeformer"}


class FaceRestorer:
    def __init__(self, backend: str, weights_dir: Path, device: str):
        self.backend = backend
        self.weights_dir = weights_dir
        self.device = device
        self._disabled = backend == "none"       # permanent: backend off
        # Models (and the shared face pipeline) are loaded lazily and cached;
        # each has an "off" flag set on a permanent failure (missing lib / bad
        # checkpoint) so we don't retry every call. A missing *weight* stays
        # transient (retried, warned once).
        self._faces = None
        self._faces_off = False
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

    def _weight(self, name: str, key: str):
        """Path to a weight, or None (warned once) if it isn't there yet."""
        path = self.weights_dir / name
        if path.exists():
            return path
        # Transient — retry if the weight is added later (no restart needed).
        self._warn_once(f"{key}-missing", "missing %s; %s no-ops until present", path, key)
        return None

    def _get_gfpgan(self):
        """Lazily load GFPGAN via spandrel. Returns the raw net, or None."""
        if self._gfpgan is not None or self._gfpgan_off:
            return self._gfpgan
        weights = self._weight(GFPGAN_WEIGHTS, "GFPGAN")
        if weights is None:
            return None
        try:
            from spandrel import ModelLoader
        except ImportError as exc:
            log.warning("spandrel unavailable (%r); GFPGAN disabled", exc)
            self._gfpgan_off = True  # permanent for this process
            return None
        try:
            model = ModelLoader().load_from_file(weights)
            if model.architecture.id != "GFPGAN":
                raise TypeError(f"{model.architecture.name} is not GFPGAN")
            # Use the raw net, not the descriptor's __call__: GFPGAN works in
            # [-1, 1] and spandrel's wrapper passes its [0, 1] input straight
            # through. faces.py hands crops over already in [-1, 1].
            net = model.model.to(self._resolve_device()).eval()
        except Exception as exc:  # noqa: BLE001 - bad checkpoint / arch mismatch
            log.warning("GFPGAN load failed (%r); GFPGAN disabled", exc)
            self._gfpgan_off = True  # permanent for this process
            return None
        log.info("GFPGAN loaded on %s", self._resolve_device())
        self._gfpgan = net
        return net

    def _get_codeformer(self):
        """Lazily build the vendored CodeFormer net. Returns it, or None."""
        if self._codeformer is not None or self._codeformer_off:
            return self._codeformer
        weights = self._weight(CODEFORMER_WEIGHTS, "CodeFormer")
        if weights is None:
            return None
        try:
            from .codeformer import build_codeformer_model
        except ImportError as exc:
            log.warning("codeformer deps unavailable (%r); CodeFormer disabled", exc)
            self._codeformer_off = True  # permanent for this process
            return None
        try:
            self._codeformer = build_codeformer_model(str(weights), self._resolve_device())
        except Exception as exc:  # noqa: BLE001 - bad checkpoint / arch mismatch
            log.warning("CodeFormer load failed (%r); CodeFormer disabled", exc)
            self._codeformer_off = True  # permanent for this process
            return None
        log.info("CodeFormer loaded on %s", self._resolve_device())
        return self._codeformer

    def _get_faces(self):
        """Lazily build the shared detection/alignment/paste pipeline."""
        if self._faces is not None or self._faces_off:
            return self._faces
        try:
            from .faces import FacePipeline

            # facexlib fetches its detection + parsing weights into the
            # weights volume on first use (download_weights.py pre-fetches).
            self._faces = FacePipeline(
                self._resolve_device(), model_rootpath=str(self.weights_dir)
            )
        except Exception as exc:  # noqa: BLE001 - facexlib missing / weight fetch failed
            log.warning("face detection unavailable (%r); face restore disabled", exc)
            self._faces_off = True  # permanent for this process
        return self._faces

    def __call__(
        self,
        img: np.ndarray,
        fidelity: float,
        factor: int = 1,
        upscaler=None,
    ) -> np.ndarray:
        """Restore faces, optionally upscaling in the same pass.

        `factor > 1` together with `upscaler` runs the combined face+upscale
        path: the background is upscaled and the restored faces are pasted onto
        it (see the module docstring). Otherwise it's an in-place restore at the
        source resolution."""
        if self._disabled:
            return img

        # An effective upscale is owed only when both a factor and an upscaler
        # are supplied (the orchestrator's combined stage). Otherwise upscale=1.
        upscale = factor if factor > 1 and upscaler is not None else 1

        gfpgan = self._get_gfpgan() if self.backend in _USES_GFPGAN else None
        codeformer = self._get_codeformer() if self.backend in _USES_CODEFORMER else None
        faces = self._get_faces() if gfpgan is not None or codeformer is not None else None
        if faces is None:
            # No face model: still honour an owed upscale.
            return upscaler(img, upscale) if upscale > 1 else img

        def restore_crop(t):
            # GFPGAN first (structure/symmetry; ignores fidelity), then
            # CodeFormer (texture; `fidelity` is its w knob).
            if gfpgan is not None:
                t = gfpgan(t, return_rgb=False)[0].clamp(-1.0, 1.0)
            if codeformer is not None:
                t = codeformer(t, w=fidelity, adain=True)[0]
            return t

        bg = upscaler(img, upscale) if upscale > 1 else None
        return faces.restore(img, restore_crop, upscale, bg)
