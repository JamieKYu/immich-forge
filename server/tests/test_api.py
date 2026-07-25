"""FastAPI endpoint tests via TestClient — focused on the fail-closed auth and
the job-state error codes. app.state is populated directly so the real lifespan
(which would build an ImmichClient/Pipeline/GPU) never runs.

TestClient is used WITHOUT a `with` block on purpose: that skips the lifespan
context, leaving the app.state we set by hand in place.
"""
from __future__ import annotations

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.main import app
from app.schemas import JobStatus

AUTH = {"Authorization": "Bearer secret-token"}


def _jpeg(img: np.ndarray) -> bytes:
    ok, buf = cv2.imencode(".jpg", img)
    assert ok
    return buf.tobytes()


class FakeImmich:
    def __init__(self, original: bytes | None = None):
        self._original = original

    async def ping(self):
        return True

    async def download_original(self, asset_id: str) -> bytes:
        assert self._original is not None, "test did not seed an original"
        return self._original


class FakeJob:
    def __init__(self, *, status=JobStatus.done, result_bytes=b"img"):
        self.status = status
        self.result_bytes = result_bytes

    def info(self):
        from app.schemas import JobInfo

        return JobInfo(job_id="j1", asset_id="a1", status=self.status)


class FakeJobs:
    def __init__(self, jobs=None):
        self._jobs = jobs or {}

    def get(self, job_id):
        return self._jobs.get(job_id)


def _settings(token="secret-token") -> Settings:
    return Settings(FORGE_API_TOKEN=token, IMMICH_API_KEY="k", FORGE_DEVICE="cpu")


@pytest.fixture
def client():
    app.state.settings = _settings()
    app.state.immich = FakeImmich()
    app.state.jobs = FakeJobs()
    return TestClient(app)


# --- auth (require_token) ----------------------------------------------------

def test_models_requires_token(client):
    assert client.get("/models").status_code == 401


def test_models_rejects_wrong_token(client):
    r = client.get("/models", headers={"Authorization": "Bearer nope"})
    assert r.status_code == 401


def test_models_accepts_correct_token(client):
    r = client.get("/models", headers=AUTH)
    assert r.status_code == 200
    assert set(r.json()) == {"denoise", "upscale", "face", "colorize"}


def test_api_disabled_when_token_unset(client):
    # Fail closed: an unconfigured server must not serve the proxy at all.
    app.state.settings = _settings(token="")
    r = client.get("/models", headers=AUTH)
    assert r.status_code == 503


def test_health_needs_no_token(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is True and body["immich"] is True


# --- job-state error codes ---------------------------------------------------

def test_job_status_404_when_unknown(client):
    assert client.get("/forge/missing", headers=AUTH).status_code == 404


def test_job_result_409_when_not_ready(client):
    app.state.jobs = FakeJobs({"j1": FakeJob(status=JobStatus.running, result_bytes=None)})
    r = client.get("/forge/j1/result", headers=AUTH)
    assert r.status_code == 409


def test_job_result_returns_jpeg_when_ready(client):
    app.state.jobs = FakeJobs({"j1": FakeJob(status=JobStatus.done, result_bytes=b"\xff\xd8x")})
    r = client.get("/forge/j1/result", headers=AUTH)
    assert r.status_code == 200
    assert r.headers["content-type"] == "image/jpeg"
    assert r.content == b"\xff\xd8x"


# --- /analyze precheck -------------------------------------------------------

def test_analyze_requires_token(client):
    assert client.post("/analyze", json={"asset_id": "a1"}).status_code == 401


def test_analyze_flags_flat_image_low_quality(client):
    flat = np.full((128, 128, 3), 128, np.uint8)
    app.state.immich = FakeImmich(original=_jpeg(flat))
    r = client.post("/analyze", headers=AUTH, json={"asset_id": "a1"})
    assert r.status_code == 200
    body = r.json()
    assert body["low_quality"] is True
    assert set(body["metrics"]) >= {"sharpness", "hf_ratio", "blockiness"}


def test_analyze_passes_detailed_image(client):
    rng = np.random.default_rng(1)
    detailed = rng.integers(0, 256, size=(256, 256, 3), dtype=np.uint8)
    app.state.immich = FakeImmich(original=_jpeg(detailed))
    r = client.post("/analyze", headers=AUTH, json={"asset_id": "a1"})
    assert r.status_code == 200
    assert r.json()["low_quality"] is False


def test_analyze_400_on_undecodable_bytes(client):
    app.state.immich = FakeImmich(original=b"not an image")
    r = client.post("/analyze", headers=AUTH, json={"asset_id": "a1"})
    assert r.status_code == 400
