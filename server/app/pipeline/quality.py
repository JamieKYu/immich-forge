"""No-reference source-quality precheck.

The forge pipeline can only ever redistribute the detail that's already in the
source — it can't invent a face that was never captured, un-blur an out-of-focus
shot, or recover detail a heavy JPEG re-save threw away. When the source is
already degraded the "after" barely beats the "before", so we warn the user
up front rather than let the result speak for a promise the input can't keep.

This is a *no-reference* judgement (there is no ground-truth "good" version to
compare against), so it leans on three cheap, classical signals:

  * **sharpness** — variance of the Laplacian. Blur (focus/motion) and upstream
    over-smoothing both crush high-frequency energy, so a low value flags images
    the upscaler/face-restorer have nothing to sharpen.
  * **hf_ratio** — the fraction of FFT magnitude living at high spatial
    frequencies. This is what catches the "big but soft" case: an image already
    upscaled once carries large *dimensions* but little real detail, so pixel
    count alone (which we deliberately don't gate on — small originals are the
    whole use case for upscaling) would be misleading.
  * **blockiness** — 8x8 JPEG block-edge strength, measured on native pixels
    (a resize destroys the grid). Heavy compression that the restorer would
    otherwise amplify into visible tiling.

Everything is CPU-only OpenCV/NumPy on a downscaled working copy, so the
precheck stays cheap enough to run synchronously before a job is submitted and
never touches the GPU.

The verdict is deliberately **conservative**: `low_quality` is true unless the
image clears *all* the bars. Real-world libraries skew low-quality far more
often than high, and a spurious "results may be limited" note costs the user
nothing (they're never blocked from trying), whereas silence on a genuinely
poor source over-promises. Thresholds are heuristic defaults, tunable via the
`FORGE_QUALITY_*` settings.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

# Longest side (px) of the working copy for the resolution-invariant metrics.
# Metrics like Laplacian variance scale with image size, so we normalise every
# image to the same working resolution before measuring; only downscaled, never
# upscaled (a small original is measured as-is).
_WORK_LONG_SIDE = 1024
# Native-pixel window (px) for blockiness. A resize would smear the 8x8 JPEG
# grid, so blockiness is measured on an unresampled centre crop of this size.
_BLOCK_CROP = 1024
# FFT radius (as a fraction of the max radius) above which energy counts as
# "high frequency" for hf_ratio.
_HF_CUTOFF = 0.25


@dataclass(frozen=True)
class QualityReport:
    """Per-image metrics plus the conservative verdict. `low_quality` is the
    only field the UI acts on; the raw metrics are returned for tuning/telemetry."""

    low_quality: bool
    sharpness: float
    hf_ratio: float
    blockiness: float
    width: int
    height: int


def _to_gray(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        return img
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def _work_copy(gray: np.ndarray) -> np.ndarray:
    """Downscale to a canonical longest-side so size-sensitive metrics compare
    across images. INTER_AREA is the right filter for shrinking."""
    h, w = gray.shape[:2]
    long_side = max(h, w)
    if long_side <= _WORK_LONG_SIDE:
        return gray
    scale = _WORK_LONG_SIDE / long_side
    return cv2.resize(gray, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA)


def _sharpness(gray_work: np.ndarray) -> float:
    """Variance of the Laplacian — the classic focus measure. Higher = crisper."""
    return float(cv2.Laplacian(gray_work, cv2.CV_64F).var())


def _hf_ratio(gray_work: np.ndarray) -> float:
    """Fraction of FFT magnitude beyond `_HF_CUTOFF` of the spectrum radius.

    Natural images concentrate energy at low frequencies, so a soft/upsampled
    image leaves very little here even when its pixel dimensions are large."""
    f = np.fft.fftshift(np.fft.fft2(gray_work.astype(np.float32)))
    mag = np.abs(f)
    total = float(mag.sum())
    if total <= 0.0:
        return 0.0
    h, w = mag.shape
    cy, cx = h / 2.0, w / 2.0
    yy, xx = np.ogrid[:h, :w]
    radius = np.sqrt(((yy - cy) / cy) ** 2 + ((xx - cx) / cx) ** 2)
    return float(mag[radius > _HF_CUTOFF].sum() / total)


def _blockiness(gray_full: np.ndarray) -> float:
    """8x8 JPEG block-edge strength on native (unresampled) pixels.

    Compares the mean adjacent-pixel difference *on* the 8-pixel grid lines to
    the mean *off* them; a clean image has no preferred grid so the ratio sits
    near 0, while heavy JPEG compression makes the block edges stand out."""
    h, w = gray_full.shape[:2]
    # Centre crop to bound cost without disturbing the 8x8 alignment relative to
    # itself (absolute phase doesn't matter — we compare grid vs non-grid).
    if h > _BLOCK_CROP:
        top = (h - _BLOCK_CROP) // 2
        gray_full = gray_full[top:top + _BLOCK_CROP, :]
        h = _BLOCK_CROP
    if w > _BLOCK_CROP:
        left = (w - _BLOCK_CROP) // 2
        gray_full = gray_full[:, left:left + _BLOCK_CROP]
        w = _BLOCK_CROP
    if h < 16 or w < 16:
        return 0.0
    g = gray_full.astype(np.float32)

    def axis_blockiness(diff: np.ndarray, axis_len: int, start: int) -> float:
        # diff[k] is |pixel[k+1] - pixel[k]| averaged over the other axis.
        idx = np.arange(diff.shape[0])
        on_grid = (idx + 1 + start) % 8 == 0  # boundaries fall between blocks
        on = diff[on_grid].mean() if on_grid.any() else 0.0
        off = diff[~on_grid].mean() if (~on_grid).any() else 0.0
        return max(0.0, float((on - off) / (off + 1e-6)))

    dh = np.abs(np.diff(g, axis=1)).mean(axis=0)  # vertical block edges
    dv = np.abs(np.diff(g, axis=0)).mean(axis=1)  # horizontal block edges
    return (axis_blockiness(dh, w, 0) + axis_blockiness(dv, h, 0)) / 2.0


def assess_quality(img: np.ndarray, settings) -> QualityReport:
    """Score a decoded BGR image and return the conservative verdict.

    `settings` supplies the thresholds (`quality_sharpness_min`,
    `quality_hf_ratio_min`, `quality_blockiness_max`). The image is flagged
    `low_quality` if it fails *any* bar — the image must look good on every axis
    to earn a clean bill of health."""
    h, w = img.shape[:2]
    gray = _to_gray(img)
    work = _work_copy(gray)

    sharpness = _sharpness(work)
    hf_ratio = _hf_ratio(work)
    blockiness = _blockiness(gray)

    low_quality = (
        sharpness < settings.quality_sharpness_min
        or hf_ratio < settings.quality_hf_ratio_min
        or blockiness > settings.quality_blockiness_max
    )
    return QualityReport(
        low_quality=low_quality,
        sharpness=sharpness,
        hf_ratio=hf_ratio,
        blockiness=blockiness,
        width=w,
        height=h,
    )
