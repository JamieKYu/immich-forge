"""Low-light enhancement. Runs inside the denoise stage, *before* denoising.

`FORGE_LOW_LIGHT_BACKEND` selects the backend:

  * **hvicidnet** (`hvi_cidnet_generalization.safetensors`, loaded via
    `spandrel`) — the default. HVI-CIDNet (CVPR 2025) brightens in its own
    learned colour space, so it lifts shadows without washing out colour, and
    it was trained on noisy low-light pairs. The "Generalization" checkpoint is
    the one its authors recommend for real-world photos.
  * **classical** — CLAHE on the LAB L-channel + a brightening gamma. Also the
    fallback whenever the model/weights/libs are unavailable.

Why before denoising: on a dark photo the signal sits barely above the noise,
so a denoiser smears the faint detail into flat, plastic patches before the
brightener ever sees it. HVI-CIDNet handles noisy dark input itself; the
denoiser then cleans up whatever noise the brightening exposed.

**Large images.** HVI-CIDNet is not tileable (its attention is global, so
independent tiles would each pick a different exposure) and needs ~1.35GB of
VRAM per megapixel in fp32 (linear; measured on an RTX 4090). Above `max_pixels` it runs on a downscaled copy, and the
result is carried back to full resolution: the upsampled model output supplies
brightness/colour, and the source's own high-frequency detail is added back,
scaled by the local brightening gain. An OOM halves the working size and
retries.

All I/O is BGR uint8 (the pipeline convention).
"""
from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np

from .tiling import free_cuda, looks_like_oom

log = logging.getLogger("forge.low_light")

WEIGHTS = "hvi_cidnet_generalization.safetensors"

# Smallest working size (pixels) we'll retry with before giving up on the GPU.
_MIN_PIXELS = 250_000
# Cap on how much the detail layer is amplified in the downscaled path; past
# this, detail transfer is mostly amplified noise.
_MAX_DETAIL_GAIN = 4.0
# BT.601 luma weights, in BGR order.
_LUMA_BGR = np.array([0.114, 0.587, 0.299], dtype=np.float32)


def classical_low_light(img: np.ndarray) -> np.ndarray:
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


def _resize(img: np.ndarray, w: int, h: int, interpolation: int) -> np.ndarray:
    return cv2.resize(img, (w, h), interpolation=interpolation)


def transfer_to_full_res(
    src: np.ndarray, lo: np.ndarray, out_lo: np.ndarray
) -> np.ndarray:
    """Carry an enhancement computed on `lo` (a downscale of `src`) back to
    `src`'s resolution.

    up(out_lo) supplies the low-frequency brightness/colour; `src - up(lo)` is
    the detail the downscale threw away, re-added scaled by the local luma gain
    (out/in) so it matches the brightened exposure. Pure NumPy/OpenCV.
    """
    h, w = src.shape[:2]
    up_in = _resize(lo, w, h, cv2.INTER_LINEAR).astype(np.float32)
    up_out = _resize(out_lo, w, h, cv2.INTER_LINEAR).astype(np.float32)
    # einsum, not `@`: numpy's matmul on macOS/Accelerate spuriously warns.
    luma_in = np.einsum("hwc,c->hw", up_in, _LUMA_BGR)
    luma_out = np.einsum("hwc,c->hw", up_out, _LUMA_BGR)
    # +4 keeps near-black pixels from producing huge, noise-driven gains.
    gain = np.clip((luma_out + 4.0) / (luma_in + 4.0), 0.0, _MAX_DETAIL_GAIN)
    out = up_out + gain[..., None] * (src.astype(np.float32) - up_in)
    return np.clip(out, 0, 255).round().astype(np.uint8)


class LowLightEnhancer:
    """Lazy-loaded low-light enhancer. `backend` selects the implementation."""

    def __init__(
        self, backend: str, weights_dir: Path, device: str, max_pixels: int = 4_000_000
    ):
        self.backend = backend
        self.weights_dir = weights_dir
        self.device = device
        self.max_pixels = max_pixels
        self._model = None                         # spandrel descriptor, cached on load
        self._disabled = backend != "hvicidnet"   # permanent: classical / libs absent
        self._warned_missing = False              # warn once, not every call

    def _load(self):
        """Load HVI-CIDNet. Returns None if the lib/weights aren't (yet) available."""
        weights = self.weights_dir / WEIGHTS
        if not weights.exists():
            # Transient — retry if the weight is added later (no restart needed).
            if not self._warned_missing:
                log.warning("missing %s; low-light uses classical CLAHE until present", weights)
                self._warned_missing = True
            return None

        try:
            import torch
            from spandrel import ImageModelDescriptor, ModelLoader
        except ImportError as exc:
            log.warning("spandrel unavailable, low-light uses classical CLAHE: %r", exc)
            self._disabled = True  # permanent for this process
            return None

        try:
            model = ModelLoader().load_from_file(weights)
            if not isinstance(model, ImageModelDescriptor) or model.scale != 1:
                raise TypeError(f"{model.architecture.name} is not a 1x restoration model")
        except Exception as exc:  # noqa: BLE001 - bad checkpoint / unsupported arch
            log.warning("low-light model load failed (%r); using classical CLAHE", exc)
            self._disabled = True  # permanent for this process
            return None

        use_cuda = self.device == "cuda" and torch.cuda.is_available()
        # fp32: spandrel doesn't mark HVI-CIDNet half-safe (its HVI colour
        # transform uses pow/atan2, which fp16 handles poorly).
        model.to(torch.device("cuda" if use_cuda else "cpu")).eval()
        log.info("low-light model %s loaded on %s", model.architecture.name, model.device)
        return model

    def __call__(self, img: np.ndarray) -> np.ndarray:
        if self._model is None and not self._disabled:
            self._model = self._load()
        if self._model is not None:
            try:
                return self._enhance(img)
            except Exception as exc:  # noqa: BLE001 - OOM at min size, etc.
                log.warning("HVI-CIDNet failed (%r); using classical CLAHE", exc)
        return classical_low_light(img)

    def _enhance(self, img: np.ndarray) -> np.ndarray:
        """Run HVI-CIDNet on the whole image if it fits under max_pixels, else on
        a downscale carried back to full res. Halves the working size on OOM."""
        h, w = img.shape[:2]
        budget = self.max_pixels if self.max_pixels > 0 else h * w
        while True:
            scale = min(1.0, (budget / (h * w)) ** 0.5)
            lo = img
            if scale < 1.0:
                lo = _resize(
                    img, max(8, round(w * scale)), max(8, round(h * scale)), cv2.INTER_AREA
                )
            try:
                out_lo = self._infer(lo)
                break
            except Exception as exc:  # noqa: BLE001 - re-raised below unless OOM
                if not looks_like_oom(exc) or budget <= _MIN_PIXELS:
                    raise
                free_cuda()
                budget = max(_MIN_PIXELS, budget // 2)
                log.warning("low-light OOM; retrying at %d px", budget)
        if lo is img:
            return out_lo
        log.info(
            "low-light ran at %dx%d for a %dx%d source; transferring to full res",
            lo.shape[1], lo.shape[0], w, h,
        )
        return transfer_to_full_res(img, lo, out_lo)

    def _infer(self, img: np.ndarray) -> np.ndarray:
        """BGR uint8 -> BGR uint8, same size. spandrel pads to its size multiple."""
        import torch

        m = self._model
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        x = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0)
        with torch.inference_mode():
            pred = m(x.to(m.device, dtype=m.dtype).div_(255.0))
            out = pred.mul(255.0).round_().byte().squeeze(0).permute(1, 2, 0).cpu().numpy()
        return cv2.cvtColor(out, cv2.COLOR_RGB2BGR)
