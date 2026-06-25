"""Best-effort EXIF preservation for forged images.

The pipeline decodes to a raw pixel array (OpenCV) and re-encodes a bare JPEG,
which drops the original's EXIF entirely. Immich then sees a forged asset with
no GPS / camera / lens / capture-time metadata — and, worse, no Orientation tag,
so an EXIF-rotated original comes back *sideways* on the forged asset (which
becomes the stack's primary).

`transplant_exif` copies the original's EXIF onto the forged JPEG, fixing up only
what the pipeline invalidated:

  - orientation — RESET to 1 (upright). `cv2.imdecode` auto-applies the EXIF
    Orientation tag, so the pipeline's pixels are already rotated to their
    display orientation. Carrying the original's tag (e.g. 6 = rotate 90°) onto
    those already-upright pixels makes Immich rotate them a *second* time, so the
    forged asset shows up sideways. Forcing Orientation=1 matches the baked-in
    pixels;
  - pixel dimensions — upscale changes them (and orientation may swap W/H), so
    ImageWidth/Length and PixelX/YDimension are rewritten to the forged size;
  - the embedded thumbnail — baked for the original resolution, so it's dropped.

Strictly best-effort: any failure (a non-JPEG original such as HEIC/PNG/RAW, or
malformed EXIF) returns the forged bytes unchanged. Metadata loss must never
fail a forge.
"""
from __future__ import annotations

import io
import logging

log = logging.getLogger("forge.metadata")


def transplant_exif(original: bytes, forged_jpeg: bytes) -> bytes:
    """Return `forged_jpeg` with the original's EXIF grafted on (dimensions and
    thumbnail corrected). Returns `forged_jpeg` unchanged on any failure."""
    try:
        import piexif
    except ImportError:
        return forged_jpeg

    try:
        exif = piexif.load(original)
    except Exception as exc:  # noqa: BLE001 - non-JPEG (HEIC/RAW/PNG) or malformed
        log.info(
            "no transplantable EXIF on original (%s); forged asset will have none",
            type(exc).__name__,
        )
        return forged_jpeg

    # Forged pixel dimensions — header-only read, doesn't decode the image.
    width = height = None
    try:
        from PIL import Image

        with Image.open(io.BytesIO(forged_jpeg)) as im:
            width, height = im.size
    except Exception:  # noqa: BLE001 - dimension fix-up is non-essential
        pass

    if width and height:
        exif["0th"][piexif.ImageIFD.ImageWidth] = width
        exif["0th"][piexif.ImageIFD.ImageLength] = height
        exif["Exif"][piexif.ExifIFD.PixelXDimension] = width
        exif["Exif"][piexif.ExifIFD.PixelYDimension] = height

    # cv2.imdecode already rotated the pixels to their display orientation, so the
    # forged pixels are upright — the output must say Orientation=1, not echo the
    # original's tag (which would make Immich rotate a second time).
    exif["0th"][piexif.ImageIFD.Orientation] = 1
    # Drop the stale thumbnail (sized/oriented for the original).
    exif["1st"] = {}
    exif["thumbnail"] = None

    try:
        exif_bytes = _safe_dump(exif, piexif)
        out = io.BytesIO()
        piexif.insert(exif_bytes, forged_jpeg, out)
        return out.getvalue()
    except Exception as exc:  # noqa: BLE001 - keep the forge; just lose metadata
        log.warning("EXIF transplant failed (%s); forged asset keeps no metadata", exc)
        return forged_jpeg


def _safe_dump(exif: dict, piexif) -> bytes:
    """`piexif.dump`, retried once with a few notoriously un-dumpable tags
    stripped (some cameras store SceneType/FileSource/MakerNote in a form piexif
    can round-trip on load but not on dump)."""
    try:
        return piexif.dump(exif)
    except Exception:  # noqa: BLE001 - drop the offenders and try once more
        for tag in (
            piexif.ExifIFD.SceneType,
            piexif.ExifIFD.FileSource,
            piexif.ExifIFD.MakerNote,
        ):
            exif.get("Exif", {}).pop(tag, None)
        return piexif.dump(exif)
