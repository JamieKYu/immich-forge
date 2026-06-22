"""Upscaling stage. Prefers Real-ESRGAN on GPU; falls back to Lanczos."""
from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np

from .compat import patch_basicsr_torchvision

log = logging.getLogger("forge.upscale")

# Smallest tile we'll retry with before giving up on the GPU entirely.
_MIN_TILE = 128


def _looks_like_oom(exc: BaseException) -> bool:
    """True if `exc` is (or masks) a CUDA out-of-memory.

    Real-ESRGAN's tile loop catches the OOM RuntimeError, prints it, then crashes
    with `UnboundLocalError: ... 'output_tile' ...` because the tile output never
    got assigned. So an OOM reaches us either directly or wearing that disguise.
    """
    name = type(exc).__name__
    msg = str(exc).lower()
    return (
        name == "OutOfMemoryError"
        or "out of memory" in msg
        or "output_tile" in msg
    )


def _free_cuda() -> None:
    """Release cached CUDA memory so the next (smaller) attempt has room."""
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


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
        self._model = None                       # cached on successful load
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
            patch_basicsr_torchvision()  # must precede the basicsr/realesrgan import
            import torch
            from basicsr.archs.rrdbnet_arch import RRDBNet
            from realesrgan import RealESRGANer
        except ImportError as exc:
            # Log the real exception — "not installed" vs the torchvision
            # functional_tensor removal vs a numpy-2 incompatibility look very
            # different and need different fixes.
            log.warning("Real-ESRGAN unavailable, using lanczos: %r", exc)
            self._disabled = True  # permanent for this process
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
        """Run Real-ESRGAN, halving the tile size on each OOM. Returns None to
        tell the caller to fall back to Lanczos (GPU too contended to fit a tile).
        """
        # Tile ladder: configured size down to _MIN_TILE. tile<=0 means tiling is
        # off, so there's a single whole-image attempt with nothing to shrink.
        tiles = [0]
        if self.tile > 0:
            tiles = []
            t = self.tile
            while t >= _MIN_TILE:
                tiles.append(t)
                t //= 2
        for tile in tiles:
            try:
                self._model.tile_size = tile
                out, _ = self._model.enhance(img, outscale=factor)
                if tile != self.tile:
                    log.warning("upscale recovered at tile=%d (configured %d)", tile, self.tile)
                return out
            except Exception as exc:  # noqa: BLE001 - re-raised below unless it's an OOM
                if not _looks_like_oom(exc):
                    raise
                _free_cuda()
                log.warning("upscale OOM at tile=%d (%s); retrying smaller", tile, type(exc).__name__)
        log.warning("upscale out of GPU memory at every tile size; falling back to lanczos")
        return None
