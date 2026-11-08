"""Tiled inference for spandrel image models, shared by the upscale and de-JPEG
stages.

Each tile is inferred with `TILE_PAD` px of context on every side and only its
centre is kept, so tiles butt together without seams (the `RealESRGANer`
scheme). Output is written straight into a uint8 buffer, so host memory is
just the final image — no float accumulator at x4 resolution. A CUDA OOM
halves the tile size and retries, down to `MIN_TILE`.

All I/O is BGR uint8 (the pipeline convention). torch is imported lazily so
this module stays importable on the GPU-free CI.
"""
from __future__ import annotations

import logging

import cv2
import numpy as np

log = logging.getLogger("forge.tiling")

# Smallest tile we'll retry with before giving up on the GPU entirely.
MIN_TILE = 128
# Context (input px) inferred around each tile and then cropped away. Wide
# receptive fields (RRDBNet's especially) mean tiles can't match a whole-image
# pass exactly; 32 keeps the drift invisible (RealESRGANer used 10, which
# shows). A multiple of 8, so tile origins stay on the JPEG 8x8 block grid.
TILE_PAD = 32


def looks_like_oom(exc: BaseException) -> bool:
    """True if `exc` is a CUDA out-of-memory."""
    name = type(exc).__name__
    msg = str(exc).lower()
    return name == "OutOfMemoryError" or "out of memory" in msg


def free_cuda() -> None:
    """Release cached CUDA memory so the next (smaller) attempt has room."""
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


def tile_ladder(tile: int, multiple: int = 1) -> list[int]:
    """Tile sizes to try: `tile`, halving down to MIN_TILE, each rounded down
    to `multiple`. tile<=0 means tiling is off: one whole-image attempt."""
    if tile <= 0:
        return [0]
    sizes = []
    t = tile
    while t >= MIN_TILE:
        sizes.append(max(multiple, t // multiple * multiple))
        t //= 2
    return sizes


def infer(model, patch) -> np.ndarray:
    """uint8 RGB (1,3,h,w) CPU tensor -> uint8 RGB HxWx3 array at model scale."""
    inp = patch.to(model.device, dtype=model.dtype).div_(255.0)
    pred = model(inp)  # spandrel guarantees [0, 1] and strips any size padding
    return pred.mul(255.0).round_().byte().squeeze(0).permute(1, 2, 0).cpu().numpy()


def run_tiled(model, x, tile: int) -> np.ndarray:
    """Run `model` over uint8 RGB `x` (1,3,h,w) tile by tile into an
    (h*scale)x(w*scale)x3 uint8 RGB array."""
    s = model.scale
    _, _, h, w = x.shape
    if tile <= 0:
        tile = max(h, w)  # tiling off: one whole-image tile
    out = np.empty((h * s, w * s, 3), dtype=np.uint8)
    for y0 in range(0, h, tile):
        for x0 in range(0, w, tile):
            y1, x1 = min(y0 + tile, h), min(x0 + tile, w)
            # Infer on the tile plus surrounding context, keep the centre.
            py0, px0 = max(0, y0 - TILE_PAD), max(0, x0 - TILE_PAD)
            py1, px1 = min(h, y1 + TILE_PAD), min(w, x1 + TILE_PAD)
            pred = infer(model, x[:, :, py0:py1, px0:px1])
            oy, ox = (y0 - py0) * s, (x0 - px0) * s
            out[y0 * s:y1 * s, x0 * s:x1 * s] = pred[
                oy:oy + (y1 - y0) * s, ox:ox + (x1 - x0) * s
            ]
    return out


def enhance_tiled(
    model, img: np.ndarray, tile: int, *, stage: str, multiple: int = 1
) -> np.ndarray | None:
    """Run `model` over BGR uint8 `img`, halving the tile size on each OOM.

    Returns the BGR uint8 result at model scale, or None if the GPU couldn't
    fit even MIN_TILE (the caller then falls back). Non-OOM errors propagate.
    `multiple` keeps tile sizes on a grid (8 for JPEG blocks).
    """
    import torch

    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    x = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0)  # uint8, CPU
    ladder = tile_ladder(tile, multiple)
    for t in ladder:
        try:
            with torch.inference_mode():
                out = run_tiled(model, x, t)
            if t != ladder[0]:
                log.warning("%s recovered at tile=%d (configured %d)", stage, t, tile)
            return cv2.cvtColor(out, cv2.COLOR_RGB2BGR)
        except Exception as exc:  # noqa: BLE001 - re-raised below unless it's an OOM
            if not looks_like_oom(exc):
                raise
            free_cuda()
            log.warning("%s OOM at tile=%d (%s); retrying smaller", stage, t, type(exc).__name__)
    log.warning("%s out of GPU memory at every tile size", stage)
    return None
