"""Upscaling stage. Prefers Real-ESRGAN on GPU; falls back to Lanczos.

The network is loaded with `spandrel` (the chaiNNer/ComfyUI model loader),
which detects the architecture from the checkpoint itself — no `realesrgan` /
`basicsr` import, and no torchvision shim. Tiling is done here: each tile is
inferred with `_TILE_PAD` px of context on every side and only its centre is
kept, so tiles butt together without seams (the same scheme as `RealESRGANer`).
"""
from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np

log = logging.getLogger("forge.upscale")

# Smallest tile we'll retry with before giving up on the GPU entirely.
_MIN_TILE = 128
# Context (input px) inferred around each tile and then cropped away. RRDBNet's
# receptive field is wider than this, so tiles can't match a whole-image pass
# exactly; 32 keeps the drift invisible (RealESRGANer used 10, which shows).
_TILE_PAD = 32


def _looks_like_oom(exc: BaseException) -> bool:
    """True if `exc` is a CUDA out-of-memory."""
    name = type(exc).__name__
    msg = str(exc).lower()
    return name == "OutOfMemoryError" or "out of memory" in msg


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
        """Run the model, halving the tile size on each OOM. Returns None to
        tell the caller to fall back to Lanczos (GPU too contended to fit a tile).
        """
        import torch

        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        x = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0)  # uint8, CPU
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
                with torch.inference_mode():
                    out = self._tiled(x, tile)
                if tile != self.tile:
                    log.warning("upscale recovered at tile=%d (configured %d)", tile, self.tile)
                break
            except Exception as exc:  # noqa: BLE001 - re-raised below unless it's an OOM
                if not _looks_like_oom(exc):
                    raise
                _free_cuda()
                log.warning("upscale OOM at tile=%d (%s); retrying smaller", tile, type(exc).__name__)
        else:
            log.warning("upscale out of GPU memory at every tile size; falling back to lanczos")
            return None

        out = cv2.cvtColor(out, cv2.COLOR_RGB2BGR)
        if factor != self._model.scale:
            # The network has a fixed scale (x4); resample to the requested
            # factor, as RealESRGANer's `outscale` did.
            h, w = img.shape[:2]
            out = cv2.resize(
                out, (w * factor, h * factor), interpolation=cv2.INTER_LANCZOS4
            )
        return out

    def _tiled(self, x, tile: int) -> np.ndarray:
        """Upscale uint8 RGB `x` (1,3,h,w) tile by tile into an HxWx3 uint8 array.

        The output is written straight into a uint8 buffer, so host memory is
        just the final image — no float accumulator at x4 resolution.
        """
        s = self._model.scale
        _, _, h, w = x.shape
        if tile <= 0:
            tile = max(h, w)  # tiling off: one whole-image tile
        out = np.empty((h * s, w * s, 3), dtype=np.uint8)
        for y0 in range(0, h, tile):
            for x0 in range(0, w, tile):
                y1, x1 = min(y0 + tile, h), min(x0 + tile, w)
                # Infer on the tile plus surrounding context, keep the centre.
                py0, px0 = max(0, y0 - _TILE_PAD), max(0, x0 - _TILE_PAD)
                py1, px1 = min(h, y1 + _TILE_PAD), min(w, x1 + _TILE_PAD)
                pred = self._infer(x[:, :, py0:py1, px0:px1])
                oy, ox = (y0 - py0) * s, (x0 - px0) * s
                out[y0 * s:y1 * s, x0 * s:x1 * s] = pred[
                    oy:oy + (y1 - y0) * s, ox:ox + (x1 - x0) * s
                ]
        return out

    def _infer(self, patch) -> np.ndarray:
        """uint8 RGB (1,3,h,w) CPU tensor -> uint8 RGB HxWx3 array at model scale."""
        m = self._model
        inp = patch.to(m.device, dtype=m.dtype).div_(255.0)
        pred = m(inp)  # spandrel guarantees [0, 1] and strips any size padding
        return (
            pred.mul(255.0).round_().byte().squeeze(0).permute(1, 2, 0).cpu().numpy()
        )
