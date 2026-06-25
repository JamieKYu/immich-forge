"""In-process async job queue + result store.

Sufficient for single-host single-GPU use. Swap the store for Redis and the
worker loop for RQ/Celery if you need persistence or multi-GPU scale-out.
"""
from __future__ import annotations

import asyncio
import logging
import time
import uuid
from dataclasses import dataclass, field

from .config import Settings
from .immich import ImmichClient
from .metadata import transplant_exif
from .pipeline import Pipeline
from .schemas import ForgeOperations, JobInfo, JobStatus

log = logging.getLogger("forge.jobs")


@dataclass
class Job:
    job_id: str
    asset_id: str
    operations: ForgeOperations
    status: JobStatus = JobStatus.queued
    progress: float = 0.0
    stage: str | None = None
    error: str | None = None
    result_bytes: bytes | None = None
    notes: list[str] = field(default_factory=list)
    original_filename: str = "asset.jpg"
    file_created_at: str | None = None
    new_asset_id: str | None = None
    stack_id: str | None = None
    created_at: float = field(default_factory=time.monotonic)

    def info(self) -> JobInfo:
        return JobInfo(
            job_id=self.job_id,
            asset_id=self.asset_id,
            status=self.status,
            progress=self.progress,
            stage=self.stage,
            error=self.error,
            notes=self.notes,
            new_asset_id=self.new_asset_id,
            stack_id=self.stack_id,
        )


class JobManager:
    def __init__(self, settings: Settings, pipeline: Pipeline, immich: ImmichClient):
        self.settings = settings
        self.pipeline = pipeline
        self.immich = immich
        self._jobs: dict[str, Job] = {}
        self._queue: asyncio.Queue[str] = asyncio.Queue()
        self._worker: asyncio.Task | None = None

    def start(self) -> None:
        self._worker = asyncio.create_task(self._run_worker())

    async def stop(self) -> None:
        if self._worker:
            self._worker.cancel()

    def get(self, job_id: str) -> Job | None:
        return self._jobs.get(job_id)

    async def submit(self, asset_id: str, operations: ForgeOperations) -> Job:
        # Fetch metadata up front so we can fail fast on a bad asset id and
        # carry the original capture date onto the forged upload.
        meta = await self.immich.get_asset(asset_id)
        job = Job(
            job_id=uuid.uuid4().hex,
            asset_id=asset_id,
            operations=operations,
            original_filename=meta.get("originalFileName", f"{asset_id}.jpg"),
            file_created_at=meta.get("fileCreatedAt") or meta.get("localDateTime"),
        )
        self._jobs[job.job_id] = job
        await self._queue.put(job.job_id)
        return job

    async def _run_worker(self) -> None:
        while True:
            job_id = await self._queue.get()
            job = self._jobs.get(job_id)
            if job is None:
                continue
            try:
                await self._process(job)
            except Exception as exc:  # noqa: BLE001 - surface any failure to the client
                log.exception("job %s failed", job_id)
                job.status = JobStatus.error
                job.error = str(exc)
            finally:
                self._evict_expired()

    async def _process(self, job: Job) -> None:
        job.status = JobStatus.running

        def on_progress(p: float, stage: str) -> None:
            job.progress = p
            job.stage = stage

        original = await self.immich.download_original(job.asset_id)
        forged, job.notes = await self.pipeline.run(
            original, job.operations, on_progress
        )
        # OpenCV re-encodes a bare JPEG, so carry the original's EXIF (GPS,
        # camera, capture time, orientation) onto the forged result. Best-effort:
        # returns `forged` unchanged if the original has no transplantable EXIF.
        job.result_bytes = transplant_exif(original, forged)
        job.status = JobStatus.done
        job.progress = 1.0
        job.stage = "done"

    async def accept(self, job: Job) -> Job:
        """Upload the forged result and stack it as primary over the original."""
        if job.status != JobStatus.done or job.result_bytes is None:
            raise ValueError("job is not in a completed state")

        name = f"forged-{job.asset_id}-{job.original_filename}"
        if not name.lower().endswith((".jpg", ".jpeg", ".png")):
            name += ".jpg"

        new_id, is_duplicate = await self.immich.upload_asset(
            job.result_bytes, name, file_created_at=job.file_created_at
        )
        # If Immich saw the forged bytes as a checksum duplicate it returns an
        # EXISTING asset id (often the original), not a new one. Stacking that
        # puts the same image on both sides of the stack — which only happens
        # when the pipeline made no real change. Surface it instead of creating
        # a bogus stack.
        if is_duplicate or new_id == job.asset_id:
            raise ValueError(
                "forged image is identical to an existing asset — the enhancement "
                "was a no-op (a model likely fell back). Check the server logs for "
                "'falling back' / 'skipping'. Nothing was stacked."
            )
        # Forged asset becomes primary; original stays as a child of the stack.
        stack_id = await self.immich.create_stack(new_id, [new_id, job.asset_id])
        job.new_asset_id = new_id
        job.stack_id = stack_id
        return job

    def _evict_expired(self) -> None:
        ttl = self.settings.job_ttl_seconds
        now = time.monotonic()
        for jid in [j for j, job in self._jobs.items() if now - job.created_at > ttl]:
            self._jobs.pop(jid, None)
