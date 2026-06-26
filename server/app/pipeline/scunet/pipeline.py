"""Build + run SCUNet for denoising.

SCUNet works in RGB float [0,1] (the KAIR convention); this pipeline takes and
returns BGR uint8 to match the rest of the Forge pipeline. Large images are
processed in overlapping tiles with a feathered blend so VRAM stays bounded and
no seams appear; an out-of-memory error halves the tile size and retries, and
the whole run raises if it can't fit even the smallest tile (the Denoiser stage
then falls back to classical Non-Local Means).
"""
from __future__ import annotations

import logging

import cv2
import numpy as np
import torch

from .network_scunet import SCUNet

log = logging.getLogger("forge.denoise")

# Smallest tile we'll retry with before giving up on the GPU.
_MIN_TILE = 128


def _looks_like_oom(exc: BaseException) -> bool:
    name = type(exc).__name__
    msg = str(exc).lower()
    return name == "OutOfMemoryError" or "out of memory" in msg


def build_scunet_model(model_path: str, device, *, in_nc: int = 3):
    """Build SCUNet with the released colour-model config and load weights.

    The `scunet_color_real_psnr`/`_gan` checkpoints are raw state_dicts for the
    config=[4,4,4,4,4,4,4], dim=64 model. Loaded strictly so a mismatched
    checkpoint raises here (the caller degrades to classical denoising) rather
    than silently producing garbage.
    """
    model = SCUNet(in_nc=in_nc, config=[4, 4, 4, 4, 4, 4, 4], dim=64)
    state_dict = torch.load(model_path, map_location="cpu")
    if isinstance(state_dict, dict) and "params" in state_dict:
        state_dict = state_dict["params"]
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model.to(device)


class ScunetPipeline:
    def __init__(self, model, *, device, tile: int = 512, tile_overlap: int = 32):
        self.model = model
        self.device = device
        self.tile = tile
        self.tile_overlap = tile_overlap

    def process(self, img_bgr: np.ndarray) -> np.ndarray:
        rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        x = (
            torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).float().div_(255.0)
        )

        # Tile ladder: configured size down to _MIN_TILE. tile<=0 (or an image
        # smaller than the tile) means a single whole-image pass.
        h, w = x.shape[-2:]
        tile = self.tile if self.tile > 0 else 0
        while True:
            try:
                with torch.inference_mode():
                    if tile <= 0 or (h <= tile and w <= tile):
                        out = self._infer(x.to(self.device))
                    else:
                        out = self._tiled(x, tile)
                break
            except Exception as exc:  # noqa: BLE001 - re-raised below unless OOM
                if not _looks_like_oom(exc):
                    raise
                self._free_cuda()
                if tile <= _MIN_TILE:
                    raise
                tile = max(_MIN_TILE, (tile or max(h, w)) // 2)
                log.warning("denoise OOM; retrying at tile=%d", tile)

        out = out.clamp_(0.0, 1.0).mul_(255.0).round_().squeeze(0)
        out_rgb = out.permute(1, 2, 0).byte().cpu().numpy()
        return cv2.cvtColor(out_rgb, cv2.COLOR_RGB2BGR)

    def _infer(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x).float().cpu()

    def _tiled(self, x: torch.Tensor, tile: int) -> torch.Tensor:
        """Overlapping-tile inference with a linear feather blend (CPU accum)."""
        _, c, h, w = x.shape
        overlap = min(self.tile_overlap, tile // 2)
        stride = max(1, tile - overlap)
        out = torch.zeros((1, c, h, w), dtype=torch.float32)
        weight = torch.zeros((1, 1, h, w), dtype=torch.float32)

        ys = list(range(0, max(1, h - overlap), stride))
        xs = list(range(0, max(1, w - overlap), stride))
        for y in ys:
            for xx in xs:
                y0, x0 = y, xx
                y1, x1 = min(y0 + tile, h), min(x0 + tile, w)
                # Pull the tile back to a full `tile` window where possible so
                # edge tiles keep the model's receptive field.
                y0, x0 = max(0, y1 - tile), max(0, x1 - tile)
                patch = x[:, :, y0:y1, x0:x1].to(self.device)
                pred = self._infer(patch)
                mask = self._feather(y1 - y0, x1 - x0, overlap)
                out[:, :, y0:y1, x0:x1] += pred * mask
                weight[:, :, y0:y1, x0:x1] += mask
        return out / weight.clamp_(min=1e-8)

    @staticmethod
    def _feather(h: int, w: int, overlap: int) -> torch.Tensor:
        """A 1-in-the-middle, ramps-to-~0-at-the-edges weight, so overlapping
        tiles cross-fade instead of leaving seams."""
        if overlap <= 0:
            return torch.ones((1, 1, h, w), dtype=torch.float32)

        def ramp(n: int) -> torch.Tensor:
            r = torch.ones(n, dtype=torch.float32)
            k = min(overlap, n // 2)
            if k > 0:
                edge = torch.linspace(1.0 / (k + 1), 1.0, k)
                r[:k] = edge
                r[-k:] = edge.flip(0)
            return r

        wy = ramp(h).view(1, 1, h, 1)
        wx = ramp(w).view(1, 1, 1, w)
        return wy * wx

    @staticmethod
    def _free_cuda() -> None:
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:  # noqa: BLE001
            pass
