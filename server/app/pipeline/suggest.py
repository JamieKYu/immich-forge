"""Auto-mode suggestions: which enhancements a photo probably needs, and why.

`/analyze` runs this next to the source-quality precheck. It's split in two:

  * `measure()` takes cheap, no-reference measurements of the original —
    resolution, JPEG quality (from the file's own quantization tables) and
    visible blocking, noise, exposure, colorfulness, camera ISO and faces;
  * `decide()` turns those into a full `ForgeOperations` plus a short,
    user-facing reason for every option, on *or* off ("640×480 source",
    "already in color"). It's pure, so the cutoffs are unit-testable.

Everything is CPU-only (face detection included — see `FaceDetector`), so like
the precheck it never waits on the GPU while a forge job runs.

The cutoffs are heuristics, set in config (`FORGE_AUTO_*`). They lean
conservative for the two options whose false positives hurt: JPEG cleanup
softens photos that are already clean, and face restoration can change how a
person looks. A suggestion is only a starting point — the user can flip any of
them.
"""
from __future__ import annotations

import io
import logging
import math
import threading
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from ..schemas import ForgeOperations

log = logging.getLogger("forge.suggest")

# Working size for the size-independent colour/exposure statistics.
_WORK_LONG_SIDE = 1024
# Native-pixel window for the noise and blocking estimates: downscaling
# averages noise away and smears the 8x8 JPEG grid, so both are measured on an
# unresampled centre crop.
_NATIVE_CROP = 1024
# Faces are detected on a copy with this long side (as facexlib's helper does).
_FACE_DETECT_SIDE = 640
# Ignore detections narrower than this (source px): too small to be worth
# restoring, and the likeliest false positives.
_FACE_MIN_PX = 20

# The IJG standard luminance quantization table at quality 50, in natural
# (row-major) order — the order Pillow returns a file's tables in. Encoders
# scale it by quality, so comparing a file's table against it estimates the
# quality setting.
_STD_LUMA = (
    16, 11, 10, 16, 24, 40, 51, 61,
    12, 12, 14, 19, 26, 58, 60, 55,
    14, 13, 16, 24, 40, 57, 69, 56,
    14, 17, 22, 29, 51, 87, 80, 62,
    18, 22, 37, 56, 68, 109, 103, 77,
    24, 35, 55, 64, 81, 104, 113, 92,
    49, 64, 78, 87, 103, 121, 120, 101,
    72, 92, 95, 98, 112, 100, 103, 99,
)


@dataclass(frozen=True)
class Face:
    width: float      # bounding-box width in source pixels
    sharpness: float  # Laplacian variance of the face resized to 128px


@dataclass(frozen=True)
class Signals:
    width: int
    height: int
    jpeg_quality: float | None  # None: not a JPEG / no tables
    jpeg_blocking: float        # extra step across 8x8 block edges, grey levels
    noise_sigma: float
    iso: int | None
    luma_median: float
    luma_p95: float
    chroma_p95: float
    hue_concentration: float
    faces: list[Face] | None    # None: face detection unavailable

    def metrics(self) -> dict[str, float]:
        """The raw measurements, for /analyze's `metrics` (tuning/debugging)."""
        out = {
            "jpeg_blocking": self.jpeg_blocking,
            "noise_sigma": self.noise_sigma,
            "luma_median": self.luma_median,
            "luma_p95": self.luma_p95,
            "chroma_p95": self.chroma_p95,
            "hue_concentration": self.hue_concentration,
        }
        if self.jpeg_quality is not None:
            out["jpeg_quality"] = self.jpeg_quality
        if self.iso is not None:
            out["iso"] = float(self.iso)
        if self.faces is not None:
            out["faces"] = float(len(self.faces))
        return out


@dataclass
class Suggestion:
    operations: ForgeOperations
    # Keyed by operation field: dejpeg, denoise, low_light, upscale,
    # face_restore, colorize.
    reasons: dict[str, str] = field(default_factory=dict)


# --- measurements ---------------------------------------------------------


def estimate_jpeg_quality(data: bytes) -> float | None:
    """Estimated JPEG quality (1-100) from the luminance quantization table, or
    None if `data` isn't a JPEG. Exact for IJG/libjpeg-style encoders; a close
    approximation for cameras and phones with their own tables."""
    try:
        from PIL import Image

        with Image.open(io.BytesIO(data)) as im:
            if im.format != "JPEG":
                return None
            tables = getattr(im, "quantization", None) or {}
    except Exception:  # noqa: BLE001 - not an image Pillow can read
        return None
    luma = tables.get(0)
    if not luma or len(luma) != 64:
        return None
    # libjpeg: table = std * scale / 100 (clamped to 1..255), where
    # scale = 5000/Q (Q < 50) or 200 - 2Q (Q >= 50). Invert it from the average
    # scaling of the entries that weren't clamped (at very low quality many hit
    # 255, which would read as a higher quality than the file really has).
    pairs = [(q, std) for q, std in zip(luma, _STD_LUMA) if 1 < q < 255] or list(zip(luma, _STD_LUMA))
    scale = 100.0 * sum(q for q, _ in pairs) / sum(std for _, std in pairs)
    quality = (200.0 - scale) / 2.0 if scale <= 100.0 else 5000.0 / scale
    return float(min(100.0, max(1.0, quality)))


def read_iso(data: bytes) -> int | None:
    """Camera ISO from EXIF, or None (no EXIF, HEIC/RAW, malformed)."""
    try:
        import piexif

        value = piexif.load(data)["Exif"].get(piexif.ExifIFD.ISOSpeedRatings)
    except Exception:  # noqa: BLE001
        return None
    if isinstance(value, (tuple, list)):
        value = value[0] if value else None
    return int(value) if isinstance(value, int) and value > 0 else None


def _work_copy(img: np.ndarray) -> np.ndarray:
    h, w = img.shape[:2]
    long_side = max(h, w)
    if long_side <= _WORK_LONG_SIDE:
        return img
    s = _WORK_LONG_SIDE / long_side
    return cv2.resize(img, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA)


def _centre_crop(img: np.ndarray, size: int) -> np.ndarray:
    h, w = img.shape[:2]
    top, left = max(0, (h - size) // 2), max(0, (w - size) // 2)
    return img[top:top + size, left:left + size]


_NOISE_KERNEL = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], np.float32)


def estimate_noise_sigma(img: np.ndarray) -> float:
    """Noise standard deviation (0-255 scale, averaged over colour channels),
    by Immerkær's fast method on the flatter half of a native-resolution crop.

    The 3x3 kernel is a difference of two Laplacians, so it cancels smooth
    gradients and keeps mostly noise; restricting it to low-gradient pixels
    keeps edges and texture from reading as noise. Channels are measured
    separately because converting to grey averages independent colour noise
    down by about a third, under-reporting what you see."""
    crop = _centre_crop(img, _NATIVE_CROP).astype(np.float32)
    if crop.shape[0] < 8 or crop.shape[1] < 8:
        return 0.0
    gray = crop if crop.ndim == 2 else cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0)[1:-1, 1:-1]
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1)[1:-1, 1:-1]
    grad = np.hypot(gx, gy)
    flat = grad <= np.percentile(grad, 50)
    if not flat.any():
        return 0.0
    channels = [crop] if crop.ndim == 2 else [crop[..., c] for c in range(crop.shape[2])]
    sigmas = [
        math.sqrt(math.pi / 2.0)
        * np.abs(cv2.filter2D(ch, -1, _NOISE_KERNEL))[1:-1, 1:-1][flat].mean()
        / 6.0
        for ch in channels
    ]
    return float(np.mean(sigmas))


def jpeg_blocking(img: np.ndarray) -> float:
    """How much stronger the brightness step is across 8x8 block edges than
    elsewhere, in grey levels, on a native-resolution crop.

    Absolute rather than a ratio (cf. quality.py's blockiness): a ratio blows up
    on smooth areas like sky, where the off-grid step is near zero, and would
    flag clean photos. On textured images this reads about 2 or more at JPEG
    quality 60 and below, and stays well under 1 on smooth ones at any quality."""
    gray = img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    g = _centre_crop(gray, _NATIVE_CROP).astype(np.float32)
    if g.shape[0] < 16 or g.shape[1] < 16:
        return 0.0

    def axis_step(diff: np.ndarray) -> float:
        # diff[k] is |pixel[k+1] - pixel[k]|; block edges fall after every 8th.
        on = (np.arange(diff.shape[0]) + 1) % 8 == 0
        return float(diff[on].mean() - diff[~on].mean())

    across_cols = np.abs(np.diff(g, axis=1)).mean(axis=0)
    across_rows = np.abs(np.diff(g, axis=0)).mean(axis=1)
    return max(0.0, (axis_step(across_cols) + axis_step(across_rows)) / 2.0)


def exposure(img: np.ndarray) -> tuple[float, float]:
    """(median, 95th percentile) brightness, 0-255."""
    gray = cv2.cvtColor(_work_copy(img), cv2.COLOR_BGR2GRAY)
    return float(np.median(gray)), float(np.percentile(gray, 95))


def color_stats(img: np.ndarray) -> tuple[float, float]:
    """(95th-percentile chroma, hue concentration).

    Chroma is the distance from neutral grey in Lab's a/b plane (0 for a true
    black-and-white image). Hue concentration is the chroma-weighted mean
    resultant length of the hue angles: near 1 when all the colour is one hue
    (sepia, a cyanotype), lower for a normal colour photo.

    Chroma shrinks with lightness, so a dark colour photo would read as
    black-and-white. The copy is first scaled up so its highlights sit at a
    normal brightness (a plain RGB gain keeps hue and relative saturation, and a
    real black-and-white photo stays at zero chroma), then near-black and
    blown-out pixels are left out."""
    work = _work_copy(img).astype(np.float32)
    bright = float(np.percentile(work.max(axis=-1), 99))
    if 0 < bright < 200:
        work = np.clip(work * (200.0 / bright), 0, 255)
    lab = cv2.cvtColor(work.astype(np.uint8), cv2.COLOR_BGR2LAB).astype(np.float32)
    lightness = lab[..., 0]
    keep = (lightness >= 40) & (lightness <= 235)  # OpenCV 8-bit L is 0-255
    if keep.mean() < 0.01:
        keep = np.ones_like(keep)  # almost all near-black/white: use everything
    a, b = lab[..., 1][keep] - 128.0, lab[..., 2][keep] - 128.0
    chroma = np.hypot(a, b)
    total = float(chroma.sum())
    if total < 1e-6:
        return 0.0, 1.0
    # sum(chroma * unit hue vector) == sum of the (a, b) vectors themselves.
    concentration = math.hypot(float(a.sum()), float(b.sum())) / total
    return float(np.percentile(chroma, 95)), float(concentration)


def measure(
    img: np.ndarray,
    data: bytes,
    face_detector: "FaceDetector | None" = None,
) -> Signals:
    h, w = img.shape[:2]
    luma_median, luma_p95 = exposure(img)
    chroma_p95, hue = color_stats(img)
    return Signals(
        width=w,
        height=h,
        jpeg_quality=estimate_jpeg_quality(data),
        jpeg_blocking=jpeg_blocking(img),
        noise_sigma=estimate_noise_sigma(img),
        iso=read_iso(data),
        luma_median=luma_median,
        luma_p95=luma_p95,
        chroma_p95=chroma_p95,
        hue_concentration=hue,
        faces=face_detector.detect(img) if face_detector else None,
    )


# --- decisions -------------------------------------------------------------


def decide(s: Signals, settings) -> Suggestion:
    """Suggested operations, with a reason for every option."""
    ops = ForgeOperations(upscale=False)  # every option starts off
    reasons: dict[str, str] = {}

    def off_on_server(backend: str) -> bool:
        return backend == "none"

    # Colorize: only black-and-white or single-tint photos. Colorizing a colour
    # photo would replace its real colours.
    if s.chroma_p95 <= settings.auto_gray_chroma_max:
        ops.colorize, reasons["colorize"] = True, "black-and-white photo"
    elif (
        s.chroma_p95 <= settings.auto_tint_chroma_max
        and s.hue_concentration >= settings.auto_tint_hue_min
    ):
        ops.colorize, reasons["colorize"] = True, "single-tint photo (e.g. sepia)"
    else:
        reasons["colorize"] = "already in color"
    if ops.colorize and off_on_server(settings.colorize_backend):
        ops.colorize, reasons["colorize"] = False, "turned off on this server"

    # Enlarge: the largest factor the resolution calls for that also fits the
    # output-size cap (the pipeline would clamp it anyway).
    size = f"{s.width}×{s.height} source"
    long_side = max(s.width, s.height)
    factor = 4 if long_side <= settings.auto_x4_max_side else (
        2 if long_side <= settings.auto_x2_max_side else 0
    )
    while factor > 1 and s.width * s.height * factor * factor > settings.max_output_pixels:
        factor //= 2
    if factor >= 2:
        ops.upscale, ops.upscale_factor, reasons["upscale"] = True, factor, size
    else:
        reasons["upscale"] = f"already {s.width}×{s.height}"

    # JPEG cleanup: low estimated quality, or visible blocking (a compressed
    # photo later re-saved at high quality keeps its blocks).
    if s.jpeg_quality is not None and s.jpeg_quality < settings.auto_jpeg_quality_max:
        ops.dejpeg, reasons["dejpeg"] = True, f"heavily compressed (JPEG quality ≈{s.jpeg_quality:.0f})"
    elif s.jpeg_blocking >= settings.auto_jpeg_blocking_min:
        ops.dejpeg, reasons["dejpeg"] = True, "visible JPEG blocking"
    elif s.jpeg_quality is not None:
        reasons["dejpeg"] = f"lightly compressed (JPEG quality ≈{s.jpeg_quality:.0f})"
    else:
        reasons["dejpeg"] = "no visible JPEG blocking"
    if ops.dejpeg and off_on_server(settings.dejpeg_backend):
        ops.dejpeg, reasons["dejpeg"] = False, "turned off on this server"

    # Brighten: dark overall, without bright highlights (a silhouette against a
    # bright sky isn't underexposed).
    if s.luma_median <= settings.auto_dark_median_max and s.luma_p95 <= settings.auto_dark_p95_max:
        ops.low_light = True
        reasons["low_light"] = f"underexposed (median brightness {s.luma_median / 2.55:.0f}%)"
    else:
        reasons["low_light"] = "well exposed"

    # Reduce noise: measured noise, a high camera ISO, or noise that
    # brightening will make visible.
    noise_min = settings.auto_noise_sigma_min
    if s.noise_sigma >= noise_min:
        ops.denoise, reasons["denoise"] = True, "visible grain or noise"
    elif s.iso is not None and s.iso >= settings.auto_iso_min:
        ops.denoise, reasons["denoise"] = True, f"shot at ISO {s.iso}"
    elif ops.low_light and s.noise_sigma >= 0.6 * noise_min:
        ops.denoise, reasons["denoise"] = True, "brightening will reveal grain"
    else:
        reasons["denoise"] = "little noise"
    if ops.denoise and off_on_server(settings.denoise_backend):
        ops.denoise, reasons["denoise"] = False, "turned off on this server"

    # Restore faces: see face_plan().
    if off_on_server(settings.face_backend):
        reasons["face_restore"] = "turned off on this server"
    elif s.faces is None:
        reasons["face_restore"] = "face detection unavailable"
    else:
        ops.face_restore, reasons["face_restore"], _ = face_plan(s.faces, settings)

    return Suggestion(ops, reasons)


def face_plan(faces: list[Face], settings) -> tuple[bool, str, list[str]]:
    """Whether to restore faces in this photo, the reason, and a label per face.

    Judged per photo, not per face, because restoration runs on every face:
      * faces narrower than `auto_face_min_px` are ignored — too little to work
        from, so the model invents features;
      * if the largest face is big and sharp, the photo's subject is already
        fine and any softer faces are likely deliberate background blur, so
        nothing is restored;
      * otherwise restore if any face is small (under `auto_face_small_px`) or
        soft. Restoration can change how a person looks, so large sharp faces
        alone never trigger it."""
    min_px = settings.auto_face_min_px
    small_px = settings.auto_face_small_px
    soft = settings.auto_face_sharpness_min
    if not faces:
        return False, "no faces found", []

    usable = [f for f in faces if f.width >= min_px]
    labels = []
    main = max(usable, key=lambda f: f.width) if usable else None
    main_is_fine = main is not None and main.width >= small_px and main.sharpness >= soft
    for f in faces:
        if f.width < min_px:
            labels.append(f"too small (< {min_px}px)")
        elif main_is_fine:
            labels.append("leave (main face is sharp)" if f is not main else "leave (sharp main face)")
        elif f.width < small_px:
            labels.append(f"restore (< {small_px}px)")
        elif f.sharpness < soft:
            labels.append(f"restore (soft < {soft:g})")
        else:
            labels.append("leave (large and sharp)")

    n = len(faces)
    if not usable:
        return False, "face too small to restore well" if n == 1 else "faces too small to restore well", labels
    if main_is_fine:
        return False, "face already sharp" if n == 1 else "main face already sharp", labels
    needy = sum(1 for label in labels if label.startswith("restore"))
    if not needy:
        return False, "face already sharp" if n == 1 else f"{n} faces, all sharp", labels
    if n == 1:
        return True, "small or soft face", labels
    return True, f"{needy} of {n} faces small or soft", labels


# --- face detection ----------------------------------------------------------


class FaceDetector:
    """CPU RetinaFace (facexlib) for the Restore faces suggestion.

    Separate from the face-restore stage's detector, which lives on the GPU
    behind the pipeline semaphore: this one must answer while a forge job
    runs. It only loads when the ML deps and `detection_Resnet50_Final.pth` are
    already present — it never triggers a download from a request. Unavailable
    means `detect()` returns None and the suggestion says so."""

    WEIGHTS = "detection_Resnet50_Final.pth"

    def __init__(self, weights_dir: Path, enabled: bool = True):
        self.weights_dir = weights_dir
        self._model = None
        self._off = not enabled
        self._lock = threading.Lock()  # /analyze calls run in worker threads
        # Why detect() returns None, for logs and scripts/suggest_report.py.
        self.unavailable_reason: str | None = (
            None if enabled else "turned off (FORGE_AUTO_FACE_DETECTION=false)"
        )
        self._warned_missing = False

    def _load(self):
        weights = self.weights_dir / self.WEIGHTS
        if not weights.exists():
            # Transient: retried once the weight is downloaded. Warn once.
            self.unavailable_reason = f"missing {weights.resolve()}"
            if not self._warned_missing:
                log.warning("auto-mode face detection unavailable: %s", self.unavailable_reason)
                self._warned_missing = True
            return None
        try:
            import warnings

            import torch  # noqa: F401
            from facexlib.detection import init_detection_model

            # facexlib trips torchvision's 'pretrained' deprecation and
            # torch.load's weights_only notice; both are noise here (the
            # checkpoint is our own pinned download).
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                warnings.simplefilter("ignore", FutureWarning)
                model = init_detection_model(
                    "retinaface_resnet50", device="cpu", model_rootpath=str(self.weights_dir)
                )
            log.info("auto-mode face detector loaded (cpu)")
            self.unavailable_reason = None
            return model
        except Exception as exc:  # noqa: BLE001 - ML deps missing / bad checkpoint
            self.unavailable_reason = f"failed to load ({exc!r})"
            log.warning("auto-mode face detection unavailable: %s", self.unavailable_reason)
            self._off = True  # permanent for this process
            return None

    def detect(self, img: np.ndarray) -> list[Face] | None:
        if self._off:
            return None
        with self._lock:
            if self._model is None:
                self._model = self._load()
            if self._model is None:
                return None
            h, w = img.shape[:2]
            scale = min(1.0, _FACE_DETECT_SIDE / max(h, w))
            small = img if scale == 1.0 else cv2.resize(
                img, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA
            )
            try:
                import torch

                with torch.inference_mode():
                    boxes = self._model.detect_faces(small, 0.97)
            except Exception as exc:  # noqa: BLE001 - never fail /analyze over faces
                log.warning("auto-mode face detection failed (%r)", exc)
                return None
        return faces_from_boxes(img, boxes, scale)


def faces_from_boxes(img: np.ndarray, boxes, scale: float) -> list[Face]:
    """Turn detector boxes (x1, y1, x2, y2, score, ... on the `scale`d copy)
    into Faces measured on the source image."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    faces = []
    for box in boxes if boxes is not None else []:
        x1, y1, x2, y2 = (float(v) / scale for v in box[:4])
        width = x2 - x1
        if width < _FACE_MIN_PX:
            continue
        crop = gray[max(0, int(y1)):min(h, int(y2)), max(0, int(x1)):min(w, int(x2))]
        if crop.size == 0:
            continue
        shrink = crop.shape[1] > 128
        crop = cv2.resize(
            crop, (128, 128), interpolation=cv2.INTER_AREA if shrink else cv2.INTER_LINEAR
        )
        faces.append(Face(width=width, sharpness=float(cv2.Laplacian(crop, cv2.CV_64F).var())))
    return faces
