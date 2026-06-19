"""Forge API — FastAPI app.

Endpoints:
  GET  /health                  liveness + Immich connectivity + GPU status
  GET  /models                  which backends/weights are active
  POST /forge                   submit a forge job  -> {job_id}
  GET  /forge/{job_id}          job status / progress
  GET  /forge/{job_id}/result   the forged image bytes (when done)
  POST /forge/{job_id}/accept   upload to Immich + stack as primary

  Immich-proxy reads for the extension's browser UI (broker keeps the key):
  GET  /immich/search           proxied asset search
  GET  /immich/thumbnail/{id}   proxied thumbnail bytes
"""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, HTTPException, Query, Request, Response
from fastapi.middleware.cors import CORSMiddleware

import httpx

from .config import get_settings
from .immich import ImmichClient, ImmichError
from .jobs import JobManager
from .pipeline import Pipeline
from .schemas import AcceptResponse, ForgeRequest, JobInfo

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("forge")


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    immich = ImmichClient(settings)
    pipeline = Pipeline(settings)
    jobs = JobManager(settings, pipeline, immich)
    jobs.start()
    app.state.settings = settings
    app.state.immich = immich
    app.state.pipeline = pipeline
    app.state.jobs = jobs
    log.info("forge ready (device=%s)", settings.device)
    try:
        yield
    finally:
        await jobs.stop()
        await immich.aclose()


app = FastAPI(title="Immich Forge", version="0.1.0", lifespan=lifespan)

# The extension talks only to Forge; allow the extension origin(s).
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten to chrome-extension://<id> in production
    allow_methods=["*"],
    allow_headers=["*"],
)


def require_token(request: Request) -> None:
    settings = request.app.state.settings
    if not settings.forge_api_token:
        return  # auth disabled (dev)
    sent = request.headers.get("authorization", "").removeprefix("Bearer ").strip()
    if sent != settings.forge_api_token:
        raise HTTPException(status_code=401, detail="invalid forge token")


@app.get("/health")
async def health(request: Request):
    immich: ImmichClient = request.app.state.immich
    settings = request.app.state.settings
    try:
        immich_ok = await immich.ping()
    except Exception:  # noqa: BLE001
        immich_ok = False
    gpu = _gpu_status(settings.device)
    return {"ok": True, "immich": immich_ok, "gpu": gpu}


@app.get("/models")
async def models(request: Request, _: None = Depends(require_token)):
    s = request.app.state.settings
    return {
        "upscale": s.upscale_backend,
        "face": s.face_backend,
        "colorize": s.colorize_backend,
    }


@app.post("/forge", response_model=JobInfo)
async def forge(req: ForgeRequest, request: Request, _: None = Depends(require_token)):
    jobs: JobManager = request.app.state.jobs
    try:
        job = await jobs.submit(req.asset_id, req.operations)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=str(exc))
    return job.info()


@app.get("/forge/{job_id}", response_model=JobInfo)
async def job_status(job_id: str, request: Request, _: None = Depends(require_token)):
    job = request.app.state.jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="job not found")
    return job.info()


@app.get("/forge/{job_id}/result")
async def job_result(job_id: str, request: Request, _: None = Depends(require_token)):
    job = request.app.state.jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="job not found")
    if job.result_bytes is None:
        raise HTTPException(status_code=409, detail=f"job not ready ({job.status})")
    return Response(content=job.result_bytes, media_type="image/jpeg")


@app.post("/forge/{job_id}/accept", response_model=AcceptResponse)
async def job_accept(job_id: str, request: Request, _: None = Depends(require_token)):
    jobs: JobManager = request.app.state.jobs
    job = jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="job not found")
    try:
        job = await jobs.accept(job)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=str(exc))
    return AcceptResponse(new_asset_id=job.new_asset_id, stack_id=job.stack_id)


# --- Immich read proxies so the browser UI never holds the API key ---


@app.get("/immich/search")
async def immich_search(
    request: Request,
    page: int = Query(1, ge=1),
    size: int = Query(50, ge=1, le=200),
    _: None = Depends(require_token),
):
    immich: ImmichClient = request.app.state.immich
    r = await immich._client.post(
        "/api/search/metadata",
        headers={**immich._headers, "Content-Type": "application/json"},
        json={"page": page, "size": size, "type": "IMAGE"},
    )
    if r.status_code != 200:
        raise HTTPException(status_code=502, detail=f"immich search: {r.text}")
    return r.json()


@app.get("/immich/asset/{asset_id}")
async def immich_asset(asset_id: str, request: Request, _: None = Depends(require_token)):
    """Asset metadata — lets the extension show the filename and confirm the id
    the browser is viewing actually resolves on the server's Immich connection."""
    immich: ImmichClient = request.app.state.immich
    try:
        return await immich.get_asset(asset_id)
    except ImmichError as exc:
        # Immich responded, but not 200: wrong API key, asset not found, or the
        # asset belongs to a different user than the key.
        raise HTTPException(status_code=404, detail=str(exc))
    except httpx.HTTPError as exc:
        # Couldn't reach Immich at all: wrong IMMICH_BASE_URL or network/DNS.
        raise HTTPException(
            status_code=502,
            detail=f"cannot reach Immich at {immich._base}: {exc!r}",
        )


@app.get("/immich/thumbnail/{asset_id}")
async def immich_thumbnail(asset_id: str, request: Request, _: None = Depends(require_token)):
    immich: ImmichClient = request.app.state.immich
    r = await immich._client.get(
        f"/api/assets/{asset_id}/thumbnail", headers=immich._headers
    )
    if r.status_code != 200:
        raise HTTPException(status_code=502, detail="thumbnail fetch failed")
    return Response(content=r.content, media_type=r.headers.get("content-type", "image/jpeg"))


def _gpu_status(device: str) -> dict:
    try:
        import torch

        avail = torch.cuda.is_available()
        return {
            "requested": device,
            "cuda_available": avail,
            "name": torch.cuda.get_device_name(0) if avail else None,
        }
    except ImportError:
        return {"requested": device, "cuda_available": False, "name": None}
