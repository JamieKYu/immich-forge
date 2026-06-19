"""Runs the selected enhancement stages in a sensible fixed order.

Order: colorize -> upscale -> face_restore. Colorizing first gives the upscaler
real color detail to sharpen; restoring faces last operates on the final-res
image so detail isn't lost to a later resize.

All stages exchange BGR uint8 ndarrays (OpenCV convention). A single GPU
semaphore is held for the whole pipeline so concurrent jobs don't OOM the card.
"""
from __future__ import annotations

import asyncio
import logging
from typing import Callable

import cv2
import numpy as np

from ..config import Settings
from ..schemas import ForgeOperations
from .colorize import Colorizer
from .face_restore import FaceRestorer
from .upscale import Upscaler

log = logging.getLogger("forge.pipeline")

ProgressCb = Callable[[float, str], None]


class Pipeline:
    def __init__(self, settings: Settings):
        self.settings = settings
        tile = settings.tile_size if settings.tile_size > 0 else 0
        self.upscaler = Upscaler(
            settings.upscale_backend, settings.weights_dir, settings.device, tile
        )
        self.face = FaceRestorer(
            settings.face_backend, settings.weights_dir, settings.device
        )
        self.colorizer = Colorizer(
            settings.colorize_backend, settings.weights_dir, settings.device
        )
        # Serialize GPU access across jobs.
        self._gpu = asyncio.Semaphore(settings.max_concurrent_gpu_jobs)

    @staticmethod
    def decode(data: bytes) -> np.ndarray:
        arr = np.frombuffer(data, np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("could not decode image bytes")
        return img

    @staticmethod
    def encode(img: np.ndarray, quality: int = 95) -> bytes:
        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
        if not ok:
            raise ValueError("could not encode image")
        return buf.tobytes()

    async def run(
        self, data: bytes, ops: ForgeOperations, progress: ProgressCb
    ) -> bytes:
        async with self._gpu:
            # Heavy CPU/GPU work off the event loop.
            return await asyncio.to_thread(self._run_sync, data, ops, progress)

    def _run_sync(
        self, data: bytes, ops: ForgeOperations, progress: ProgressCb
    ) -> bytes:
        img = self.decode(data)
        h, w = img.shape[:2]
        if h * w > self.settings.max_image_pixels:
            raise ValueError(
                f"image too large ({w}x{h} > {self.settings.max_image_pixels}px)"
            )

        stages = [s for s, on in (
            ("colorize", ops.colorize),
            ("upscale", ops.upscale),
            ("face_restore", ops.face_restore),
        ) if on]
        if not stages:
            raise ValueError("no operations selected")

        progress(0.0, "starting")
        for i, stage in enumerate(stages):
            progress(i / len(stages), stage)
            if stage == "colorize":
                img = self.colorizer(img)
            elif stage == "upscale":
                img = self.upscaler(img, ops.upscale_factor)
            elif stage == "face_restore":
                img = self.face(img, ops.face_fidelity)
            log.info("stage %s done -> %s", stage, img.shape)

        progress(1.0, "encoding")
        return self.encode(img)
