"""Tests for JobManager — the in-process job lifecycle and the no-op/duplicate
guard. Pipeline and Immich are faked; no GPU, no network.
"""
from __future__ import annotations

import time

import pytest

from app.jobs import Job, JobManager
from app.schemas import ForgeOperations, JobStatus

from .conftest import make_jpeg, make_jpeg_with_exif


class FakeImmich:
    def __init__(self, *, asset_meta=None, upload=("new-id", False)):
        self.asset_meta = asset_meta or {
            "originalFileName": "photo.jpg",
            "fileCreatedAt": "2020-01-01T00:00:00Z",
        }
        self._upload = upload
        self.uploaded = None
        self.stacked = None

    async def get_asset(self, asset_id):
        return self.asset_meta

    async def download_original(self, asset_id):
        return self._original

    async def upload_asset(self, data, filename, *, file_created_at=None, is_favorite=False):
        self.uploaded = {"data": data, "filename": filename, "created": file_created_at}
        return self._upload

    async def create_stack(self, primary_asset_id, asset_ids):
        self.stacked = {"primary": primary_asset_id, "ids": asset_ids}
        return "stack-1"


class FakePipeline:
    def __init__(self, forged=b"forged", notes=None):
        self.forged = forged
        self.notes = notes or []
        self.ran_with = None

    async def run(self, original, operations, on_progress):
        self.ran_with = original
        on_progress(0.5, "upscaling")
        return self.forged, self.notes


def _manager(settings, *, immich=None, pipeline=None) -> JobManager:
    return JobManager(settings, pipeline or FakePipeline(), immich or FakeImmich())


def _job(**kw) -> Job:
    base = dict(job_id="j1", asset_id="a1", operations=ForgeOperations())
    base.update(kw)
    return Job(**base)


# --- submit -----------------------------------------------------------------

async def test_submit_pulls_metadata_and_queues(settings):
    immich = FakeImmich(
        asset_meta={"originalFileName": "vacation.jpg", "fileCreatedAt": "2019-05-05T10:00:00Z"}
    )
    mgr = _manager(settings, immich=immich)

    job = await mgr.submit("a1", ForgeOperations())

    assert job.asset_id == "a1"
    assert job.original_filename == "vacation.jpg"
    assert job.file_created_at == "2019-05-05T10:00:00Z"
    assert job.status == JobStatus.queued
    assert mgr.get(job.job_id) is job


async def test_submit_falls_back_to_local_datetime(settings):
    immich = FakeImmich(asset_meta={"localDateTime": "2018-01-01T00:00:00Z"})
    mgr = _manager(settings, immich=immich)
    job = await mgr.submit("a1", ForgeOperations())
    assert job.file_created_at == "2018-01-01T00:00:00Z"
    assert job.original_filename == "a1.jpg"  # default when absent


# --- _process ---------------------------------------------------------------

async def test_process_runs_pipeline_and_transplants_exif(settings):
    original = make_jpeg_with_exif(make="Nikon")
    forged = make_jpeg(128, 96)
    immich = FakeImmich()
    immich._original = original
    pipeline = FakePipeline(forged=forged, notes=["upscale clamped x4 -> x2"])
    mgr = _manager(settings, immich=immich, pipeline=pipeline)
    job = _job()

    await mgr._process(job)

    assert job.status == JobStatus.done
    assert job.progress == 1.0
    assert job.notes == ["upscale clamped x4 -> x2"]
    # transplant_exif ran: result carries the original's EXIF, so it differs from
    # the bare forged bytes.
    assert job.result_bytes != forged
    import piexif

    assert piexif.load(job.result_bytes)["0th"][piexif.ImageIFD.Make] == b"Nikon"


# --- accept -----------------------------------------------------------------

async def test_accept_uploads_and_stacks_forged_as_primary(settings):
    immich = FakeImmich(upload=("forged-id", False))
    mgr = _manager(settings, immich=immich)
    job = _job(status=JobStatus.done, result_bytes=b"bytes", original_filename="p.jpg")

    out = await mgr.accept(job)

    assert out.new_asset_id == "forged-id"
    assert out.stack_id == "stack-1"
    # Forged asset is the stack primary; original is a child.
    assert immich.stacked == {"primary": "forged-id", "ids": ["forged-id", "a1"]}


async def test_accept_rejects_when_immich_reports_duplicate(settings):
    immich = FakeImmich(upload=("whatever", True))
    mgr = _manager(settings, immich=immich)
    job = _job(status=JobStatus.done, result_bytes=b"bytes")

    with pytest.raises(ValueError, match="no-op"):
        await mgr.accept(job)
    assert immich.stacked is None  # nothing stacked


async def test_accept_rejects_when_upload_returns_original_id(settings):
    # Immich dedup can return the EXISTING (original) id without a duplicate flag.
    immich = FakeImmich(upload=("a1", False))
    mgr = _manager(settings, immich=immich)
    job = _job(asset_id="a1", status=JobStatus.done, result_bytes=b"bytes")

    with pytest.raises(ValueError, match="no-op"):
        await mgr.accept(job)
    assert immich.stacked is None


async def test_accept_requires_completed_job(settings):
    mgr = _manager(settings)
    job = _job(status=JobStatus.running, result_bytes=None)
    with pytest.raises(ValueError, match="completed state"):
        await mgr.accept(job)


async def test_accept_appends_jpg_extension_when_missing(settings):
    immich = FakeImmich(upload=("forged-id", False))
    mgr = _manager(settings, immich=immich)
    job = _job(status=JobStatus.done, result_bytes=b"x", original_filename="raw.dng")

    await mgr.accept(job)
    assert immich.uploaded["filename"].endswith(".jpg")


# --- eviction ----------------------------------------------------------------

def test_evict_expired_drops_only_old_jobs(settings):
    mgr = _manager(settings)
    settings_ttl = settings.job_ttl_seconds
    fresh = _job(job_id="fresh")
    stale = _job(job_id="stale", created_at=time.monotonic() - settings_ttl - 10)
    mgr._jobs = {"fresh": fresh, "stale": stale}

    mgr._evict_expired()

    assert "fresh" in mgr._jobs
    assert "stale" not in mgr._jobs
