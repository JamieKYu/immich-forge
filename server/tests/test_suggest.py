"""Tests for Auto-mode suggestions (pipeline/suggest.py).

Measurements are checked on synthetic images with known properties (a JPEG
saved at a known quality, Gaussian noise of a known sigma, a grey or sepia
image). Decisions are checked by varying one signal at a time from a "clean,
large colour photo" baseline. All of it runs on the core deps — the real
RetinaFace detector needs torch + facexlib + weights and is exercised on the GPU
host via scripts/suggest_report.py.
"""
from __future__ import annotations

import io
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np
import piexif
import pytest
from PIL import Image

from app.config import Settings
from app.pipeline.suggest import (
    Face,
    FaceDetector,
    Signals,
    color_stats,
    decide,
    estimate_jpeg_quality,
    estimate_noise_sigma,
    exposure,
    faces_from_boxes,
    jpeg_blocking,
    measure,
    read_iso,
)

_S = Settings(FORGE_API_TOKEN="t", IMMICH_API_KEY="k", FORGE_DEVICE="cpu")


def _gradient(w: int = 512, h: int = 384) -> np.ndarray:
    """A smooth colour gradient (BGR uint8): no noise, no edges."""
    x = np.linspace(0, 1, w, dtype=np.float32)[None, :]
    y = np.linspace(0, 1, h, dtype=np.float32)[:, None]
    b = 60 + 120 * x + 0 * y
    g = 80 + 100 * y + 0 * x
    r = 200 - 100 * x * y
    return np.clip(np.stack([b, g, r], -1), 0, 255).astype(np.uint8)


def _jpeg(img_bgr: np.ndarray, quality: int, exif: bytes | None = None) -> bytes:
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    buf = io.BytesIO()
    kwargs = {"exif": exif} if exif else {}
    Image.fromarray(rgb).save(buf, format="JPEG", quality=quality, **kwargs)
    return buf.getvalue()


# --- measurements ----------------------------------------------------------


@pytest.mark.parametrize("quality", [10, 30, 50, 75, 90])
def test_jpeg_quality_estimate_matches_the_saved_quality(quality):
    est = estimate_jpeg_quality(_jpeg(_gradient(), quality))
    assert est == pytest.approx(quality, abs=2)


def test_jpeg_quality_is_none_for_non_jpeg():
    buf = io.BytesIO()
    Image.new("RGB", (32, 32)).save(buf, format="PNG")
    assert estimate_jpeg_quality(buf.getvalue()) is None
    assert estimate_jpeg_quality(b"not an image") is None


def test_read_iso_from_exif():
    exif = piexif.dump({"Exif": {piexif.ExifIFD.ISOSpeedRatings: 6400}})
    assert read_iso(_jpeg(_gradient(), 90, exif)) == 6400
    assert read_iso(_jpeg(_gradient(), 90)) is None


def test_noise_estimate_recovers_known_sigma():
    rng = np.random.default_rng(0)
    clean = _gradient().astype(np.float32)
    noisy = np.clip(clean + rng.normal(0, 10, clean.shape), 0, 255).astype(np.uint8)
    assert estimate_noise_sigma(noisy) == pytest.approx(10, abs=2)


def test_noise_estimate_is_low_for_clean_images_with_edges():
    assert estimate_noise_sigma(_gradient()) < 1.0
    # Hard edges (a checkerboard) must not read as noise.
    board = (np.indices((384, 512)).sum(0) // 32 % 2 * 200 + 20).astype(np.uint8)
    assert estimate_noise_sigma(cv2.cvtColor(board, cv2.COLOR_GRAY2BGR)) < 2.0


def _texture(w: int = 1024, h: int = 768) -> np.ndarray:
    """Photo-like multi-scale texture (BGR uint8)."""
    rng = np.random.default_rng(0)
    acc = np.zeros((h, w), np.float32)
    for step, amp in ((2, 6), (8, 18), (32, 30), (128, 50)):
        n = rng.normal(0, 1, (h // step + 2, w // step + 2)).astype(np.float32)
        acc += amp * cv2.resize(n, (w, h), interpolation=cv2.INTER_CUBIC)
    g = np.clip(128 + acc, 0, 255)
    return np.stack([g * 0.8 + 20, g, g * 0.9 + 10], -1).clip(0, 255).astype(np.uint8)


def _decoded(img: np.ndarray, quality: int) -> np.ndarray:
    return cv2.imdecode(np.frombuffer(_jpeg(img, quality), np.uint8), cv2.IMREAD_COLOR)


def test_jpeg_blocking_tracks_compression_on_texture():
    heavy, light = jpeg_blocking(_decoded(_texture(), 25)), jpeg_blocking(_decoded(_texture(), 95))
    assert heavy >= 2.0 > 0.5 > light


def test_jpeg_blocking_stays_low_on_smooth_images():
    # A ratio-based measure explodes here (sky-like areas); this one mustn't.
    for quality in (10, 40, 75):
        assert jpeg_blocking(_decoded(_gradient(), quality)) < 1.0  # cutoff is 2.0


def test_jpeg_blocking_survives_a_high_quality_resave():
    resaved = _decoded(_decoded(_texture(), 30), 95)
    assert jpeg_blocking(resaved) >= 2.0


def test_exposure_reports_median_and_p95():
    dark = np.full((100, 100, 3), 30, np.uint8)
    median, p95, black, lit = exposure(dark)
    assert median == pytest.approx(30, abs=1) and p95 == pytest.approx(30, abs=1)
    assert black == 0.0 and lit == pytest.approx(30, abs=1)


def test_exposure_separates_black_background_from_lit_subject():
    img = np.zeros((100, 100, 3), np.uint8)
    img[30:70, 40:60] = 120  # a lit subject on black
    _, _, black, lit = exposure(img)
    assert black == pytest.approx(0.92, abs=0.01) and lit == pytest.approx(120, abs=1)


def test_color_stats_grey_sepia_and_colour():
    grey = cv2.cvtColor(cv2.cvtColor(_gradient(), cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
    chroma, _ = color_stats(grey)
    assert chroma < 2

    # Sepia: grey luminance with one warm tint.
    lum = cv2.cvtColor(_gradient(), cv2.COLOR_BGR2GRAY).astype(np.float32)
    sepia = np.clip(np.stack([lum * 0.75, lum * 0.9, lum * 1.05], -1), 0, 255).astype(np.uint8)
    chroma, concentration = color_stats(sepia)
    assert 6 < chroma <= 20 and concentration > 0.95

    rng = np.random.default_rng(0)
    colourful = rng.integers(0, 256, size=(128, 128, 3), dtype=np.uint8)
    chroma, concentration = color_stats(colourful)
    assert chroma > 20 and concentration < 0.5


def test_dark_colour_photo_does_not_read_as_black_and_white():
    # An ordinary colour scene at ~10% exposure: unadjusted, its chroma falls
    # under the black-and-white cutoff.
    dark = (_gradient().astype(np.float32) * 0.1).astype(np.uint8)
    chroma, _ = color_stats(dark)
    assert chroma > 3 * _S.auto_gray_chroma_max
    # ...while a dark black-and-white photo still reads as one.
    grey = cv2.cvtColor(cv2.cvtColor(dark, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
    chroma, _ = color_stats(grey)
    assert chroma <= _S.auto_gray_chroma_max


def test_measure_combines_signals():
    img = _gradient()
    s = measure(img, _jpeg(img, 40), face_detector=None)
    assert (s.width, s.height) == (512, 384)
    assert s.jpeg_quality == pytest.approx(40, abs=2)
    assert s.faces is None
    assert {"noise_sigma", "jpeg_quality", "jpeg_blocking", "chroma_p95"} <= set(s.metrics())
    assert "faces" not in s.metrics()


# --- decisions ---------------------------------------------------------------

# A clean, large colour photo with no faces: nothing should be suggested.
_CLEAN = Signals(
    width=4000, height=3000, jpeg_quality=92.0, jpeg_blocking=0.2, noise_sigma=1.5,
    iso=100, luma_median=120.0, luma_p95=230.0, black_fraction=0.02, lit_median=125.0,
    chroma_p95=45.0,
    hue_concentration=0.3, faces=[],
)
_OPS = ("dejpeg", "denoise", "low_light", "upscale", "face_restore", "colorize")


def _on(sig: Signals, settings: Settings = _S) -> set[str]:
    ops = decide(sig, settings).operations
    return {op for op in _OPS if getattr(ops, op)}


def test_clean_photo_gets_no_suggestions_but_a_reason_for_every_option():
    sug = decide(_CLEAN, _S)
    assert _on(_CLEAN) == set()
    assert set(sug.reasons) == set(_OPS)
    assert sug.reasons["colorize"] == "already in color"
    assert sug.reasons["upscale"] == "already 4000×3000"


@pytest.mark.parametrize(
    ("w", "h", "factor"),
    [(640, 480, 4), (1200, 900, 4), (2000, 1500, 2), (2500, 1600, 2), (3000, 2000, None)],
)
def test_enlarge_factor_follows_resolution(w, h, factor):
    sug = decide(replace(_CLEAN, width=w, height=h), _S)
    assert sug.operations.upscale is (factor is not None)
    if factor:
        assert sug.operations.upscale_factor == factor
        assert sug.reasons["upscale"] == f"{w}×{h} source"


def test_enlarge_respects_output_cap():
    capped = Settings(FORGE_API_TOKEN="t", FORGE_MAX_OUTPUT_PIXELS=5_000_000)
    sug = decide(replace(_CLEAN, width=1000, height=800), capped)  # x4 = 12.8MP, x2 = 3.2MP
    assert sug.operations.upscale and sug.operations.upscale_factor == 2


def test_jpeg_cleanup_on_low_quality_or_visible_blocking():
    sug = decide(replace(_CLEAN, jpeg_quality=35.0), _S)
    assert sug.operations.dejpeg and "≈35" in sug.reasons["dejpeg"]
    sug = decide(replace(_CLEAN, jpeg_quality=95.0, jpeg_blocking=3.0), _S)
    assert sug.operations.dejpeg and sug.reasons["dejpeg"] == "visible JPEG blocking"
    sug = decide(replace(_CLEAN, jpeg_quality=None), _S)
    assert not sug.operations.dejpeg and sug.reasons["dejpeg"] == "no visible JPEG blocking"


def test_brighten_needs_dark_midtones_and_no_bright_highlights():
    assert "low_light" in _on(replace(_CLEAN, luma_median=30.0, luma_p95=120.0))
    # Silhouette against a bright sky: dark median, bright highlights.
    assert "low_light" not in _on(replace(_CLEAN, luma_median=30.0, luma_p95=250.0))


def test_lit_subject_on_black_background_is_not_brightened():
    sug = decide(
        replace(_CLEAN, luma_median=5.0, luma_p95=150.0, black_fraction=0.7, lit_median=80.0), _S
    )
    assert not sug.operations.low_light
    assert sug.reasons["low_light"] == "dark background, subject well lit"


def test_reduce_noise_on_noise_iso_or_brightening():
    sug = decide(replace(_CLEAN, noise_sigma=10.0), _S)
    assert sug.operations.denoise and sug.reasons["denoise"] == "visible grain or noise"
    sug = decide(replace(_CLEAN, iso=6400), _S)
    assert sug.operations.denoise and sug.reasons["denoise"] == "shot at ISO 6400"
    dark_mild_noise = replace(_CLEAN, luma_median=30.0, luma_p95=120.0, noise_sigma=5.5)
    sug = decide(dark_mild_noise, _S)
    assert sug.operations.denoise and sug.reasons["denoise"] == "brightening will reveal grain"
    assert "denoise" not in _on(replace(_CLEAN, noise_sigma=5.5))
    # Fine film grain on an old scan (sigma ~6.4) isn't worth removing.
    assert "denoise" not in _on(replace(_CLEAN, noise_sigma=6.4))


def test_colorize_only_for_black_and_white_or_single_tint():
    sug = decide(replace(_CLEAN, chroma_p95=2.0), _S)
    assert sug.operations.colorize and sug.reasons["colorize"] == "black-and-white photo"
    assert "colorize" in _on(replace(_CLEAN, chroma_p95=12.0, hue_concentration=0.98))
    # Muted colour photo: low chroma but mixed hues.
    assert "colorize" not in _on(replace(_CLEAN, chroma_p95=12.0, hue_concentration=0.6))


# Readings from real sample photos, with the owner's verdict.
@pytest.mark.parametrize(
    ("name", "signals", "want"),
    [
        ("sepia print", dict(chroma_p95=11.70, hue_concentration=0.99), {"colorize"}),
        ("black-and-white scan", dict(chroma_p95=0.0, hue_concentration=1.0), {"colorize"}),
        # Pale goat against blue sky: low chroma, fairly one-hued — still colour.
        ("pale colour photo", dict(chroma_p95=19.21, hue_concentration=0.92), set()),
        ("dark-mode screenshot", dict(chroma_p95=70.94, hue_concentration=0.96, luma_median=21.0, luma_p95=255.0), set()),
        # Fine film grain the owner didn't consider worth removing.
        ("old scan with grain", dict(noise_sigma=6.38), set()),
        ("small web image", dict(noise_sigma=4.13), set()),
        # Flash/torch-lit people against a black cave or night: leave dark.
        ("lit family in a cave", dict(luma_median=2.0, luma_p95=157.0, black_fraction=0.737, lit_median=82.0), set()),
        ("flash group at night", dict(luma_median=12.0, luma_p95=138.0, black_fraction=0.581, lit_median=60.0), set()),
        # Genuinely dim scenes the owner wanted brightened.
        ("dim room", dict(luma_median=22.0, luma_p95=169.0, black_fraction=0.473, lit_median=140.0), {"low_light"}),
        ("dark concert", dict(luma_median=4.0, luma_p95=24.0, black_fraction=0.929, lit_median=27.0), {"low_light"}),
        ("lamp-lit living room", dict(luma_median=53.0, luma_p95=168.0, black_fraction=0.28, lit_median=79.0), {"low_light"}),
        # Two photos saved at quality ~72: one with visible damage from an
        # earlier save, one clean. Indistinguishable by every measure, so both
        # are left alone rather than soften the clean one.
        ("quality-72 photo", dict(jpeg_quality=71.96, jpeg_blocking=1.11), set()),
        ("lightly compressed scan", dict(jpeg_quality=88.96, jpeg_blocking=0.0), set()),
    ],
)
def test_matches_calibration_readings(name, signals, want):
    on = _on(replace(_CLEAN, **signals)) - {"upscale"}  # _CLEAN is large
    assert on == want, name


def _faces(*specs: tuple[float, float]) -> list[Face]:
    return [Face(width=w, sharpness=sh) for w, sh in specs]


# Per-face (width px, sharpness) measured on real sample photos, with whether
# their owner wanted faces restored. The rule must reproduce every verdict.
_CALIBRATION = {
    "group photo, tiny faces": (_faces((27, 32), (23, 24), (21, 23), (22, 21)), False),
    "two tiny faces": (_faces((29, 9), (28, 10)), False),
    "one small distant face": (_faces((67, 10)), False),
    "sharp subject + soft background face": (_faces((232, 294), (104, 25)), False),
    "old group photo, small sharp faces": (
        _faces((120, 770), (126, 679), (114, 593), (124, 918)), True,
    ),
    "screenshot of a group, small faces": (
        _faces((140, 26), (155, 31), (116, 185), (126, 32), (124, 24), (107, 180), (96, 126), (94, 124)),
        True,
    ),
}


@pytest.mark.parametrize("name", list(_CALIBRATION))
def test_face_restore_matches_calibration_photos(name):
    faces, want = _CALIBRATION[name]
    assert decide(replace(_CLEAN, faces=faces), _S).operations.face_restore is want


def test_face_restore_reasons():
    reason = lambda faces: decide(replace(_CLEAN, faces=faces), _S).reasons["face_restore"]  # noqa: E731
    assert reason(_faces((120, 300))) == "small or soft face"
    assert reason(_faces((120, 300), (130, 300), (90, 300))) == "3 of 3 faces small or soft"
    assert reason(_faces((40, 20))) == "face too small to restore well"
    assert reason(_faces((40, 20), (30, 20))) == "faces too small to restore well"
    assert reason(_faces((600, 300))) == "face already sharp"
    assert reason(_faces((600, 300), (110, 20))) == "main face already sharp"
    assert reason([]) == "no faces found"
    assert decide(replace(_CLEAN, faces=None), _S).reasons["face_restore"] == "face detection unavailable"


def test_large_soft_face_is_restored():
    # An out-of-focus portrait: big face, but soft.
    assert decide(replace(_CLEAN, faces=_faces((600, 10))), _S).operations.face_restore is True


def test_options_turned_off_on_the_server_are_never_suggested():
    off = Settings(
        FORGE_API_TOKEN="t", FORGE_DEJPEG_BACKEND="none", FORGE_COLORIZE_BACKEND="none",
        FORGE_DENOISE_BACKEND="none", FORGE_FACE_BACKEND="none",
    )
    needy = replace(
        _CLEAN, jpeg_quality=20.0, chroma_p95=1.0, noise_sigma=12.0,
        faces=[Face(width=80, sharpness=5)],
    )
    sug = decide(needy, off)
    for op in ("dejpeg", "colorize", "denoise", "face_restore"):
        assert getattr(sug.operations, op) is False
        assert sug.reasons[op] == "turned off on this server"


def test_suggestion_keeps_default_strengths():
    ops = decide(replace(_CLEAN, noise_sigma=9.0, faces=[Face(120, 5)]), _S).operations
    assert ops.denoise_strength == 1.0 and ops.face_fidelity == 0.95


# --- face detection plumbing -------------------------------------------------


def test_faces_from_boxes_scales_back_to_source_pixels():
    img = _gradient(1280, 960)
    # Boxes on a half-size copy: a 100px face there is 200px in the source.
    boxes = np.array([[100, 100, 200, 200, 0.99], [10, 10, 15, 15, 0.99]])
    faces = faces_from_boxes(img, boxes, 0.5)
    assert len(faces) == 1  # the 10px detection is dropped as too small
    assert faces[0].width == pytest.approx(200)


def test_face_detector_unavailable_without_weights(tmp_path: Path):
    det = FaceDetector(tmp_path)
    assert det.detect(_gradient()) is None
    assert det.unavailable_reason.startswith("missing ")
    assert det.unavailable_reason.endswith("detection_Resnet50_Final.pth")
    off = FaceDetector(tmp_path, enabled=False)
    assert off.detect(_gradient()) is None
    assert "FORGE_AUTO_FACE_DETECTION" in off.unavailable_reason
