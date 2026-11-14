"""Forge API — FastAPI app.

Endpoints:
  GET  /                        setup & management page (HTML)
  GET  /health                  liveness + Immich connectivity + GPU + setup/model status
  GET  /models                  which backends/weights are active
  POST /analyze                 quality precheck + Auto-mode suggestions (CPU-only)
  POST /forge                   submit a forge job  -> {job_id}
  GET  /forge/{job_id}          job status / progress
  GET  /forge/{job_id}/result   the forged image bytes (when done)
  POST /forge/{job_id}/accept   upload to Immich + stack as primary

  Setup page API (see setup_config.py; first visitor until a token exists):
  GET  /setup/status            setup state + model download progress (public)
  GET  /setup/config            saved connection details (key masked)
  POST /setup/test              test an Immich address + key
  POST /setup/immich            save them (creates the Forge token on first run)
  POST /setup/token             set your own Forge token, or generate a new one
  POST /setup/models/retry      retry failed model downloads

  Immich-proxy reads for the extension's browser UI (broker keeps the key):
  GET  /immich/search           proxied asset search
  GET  /immich/thumbnail/{id}   proxied thumbnail bytes
  GET  /immich/original/{id}    proxied full-resolution original bytes
"""
from __future__ import annotations

import asyncio
import logging
import secrets
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException, Query, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

import httpx

from .config import get_settings
from .immich import ImmichClient, ImmichError, check_connection
from .jobs import JobManager
from .models_download import ModelDownloader
from .setup_config import SetupConfig
from .pipeline import Pipeline
from .pipeline.quality import assess_quality
from .pipeline.suggest import FaceDetector, decide, measure
from .schemas import AcceptResponse, ForgeAnalysis, ForgeRequest, JobInfo

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("forge")


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    setup = SetupConfig(settings)
    # Missing model weights download in the background, so the server (and the
    # setup page, which shows progress) is up straight away. Forging waits.
    downloader = ModelDownloader(settings.weights_dir, skip=settings.skip_weight_download)
    downloader.start()
    immich = ImmichClient(setup.connection)
    pipeline = Pipeline(settings)
    jobs = JobManager(settings, pipeline, immich)
    jobs.start()
    app.state.settings = settings
    app.state.setup = setup
    app.state.downloader = downloader
    app.state.immich = immich
    app.state.pipeline = pipeline
    app.state.jobs = jobs
    # CPU face detector for Auto-mode suggestions (loads lazily on first use).
    app.state.face_detector = FaceDetector(settings.weights_dir, settings.auto_face_detection)
    log.info("forge ready (device=%s)", settings.device)
    if setup.setup_required:
        log.info("setup required: open this server's address (port 8000) in a browser")
    try:
        yield
    finally:
        await jobs.stop()
        await app.state.immich.aclose()


app = FastAPI(title="Forge for Immich", version="0.1.0", lifespan=lifespan)

# CORS is OFF by default. The extension reaches Forge through its granted host
# permission (not subject to CORS), so we don't need to open the API to web
# origins — doing so would let any website read the Immich proxy responses.
# Set FORGE_CORS_ORIGINS (comma-separated) only if a non-extension web client
# needs access.
_cors = get_settings().cors_origin_list
if _cors:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=_cors,
        allow_methods=["*"],
        allow_headers=["*"],
    )


SETUP_REQUIRED = (
    "setup required: open this server's address in a browser to connect Immich "
    "and create a Forge token"
)


def _bearer(request: Request) -> str:
    return request.headers.get("authorization", "").removeprefix("Bearer ").strip()


def require_token(request: Request) -> None:
    conn = request.app.state.setup.connection
    # Fail closed: until setup is complete (a token, and Immich connected) the
    # API serves nothing — no unauthenticated window onto the Immich library.
    if not conn.forge_api_token or not conn.immich_configured:
        raise HTTPException(status_code=503, detail=SETUP_REQUIRED)
    if not secrets.compare_digest(_bearer(request), conn.forge_api_token):
        raise HTTPException(status_code=401, detail="invalid forge token")


def require_setup_access(request: Request) -> None:
    """Setup page access: open to the first visitor until a Forge token
    exists, then the token is required."""
    token = request.app.state.setup.connection.forge_api_token
    if token and not secrets.compare_digest(_bearer(request), token):
        raise HTTPException(status_code=401, detail="invalid forge token")


def _swap_immich_client(app_: FastAPI) -> None:
    """Use freshly saved Immich details right away, without a restart."""
    old = app_.state.immich
    new = ImmichClient(app_.state.setup.connection)
    app_.state.immich = new
    app_.state.jobs.immich = new

    async def close_later():
        await asyncio.sleep(60)  # let in-flight requests on the old client finish
        await old.aclose()

    asyncio.get_running_loop().create_task(close_later())


# --- setup & management page ---

_SETUP_PAGE = Path(__file__).parent / "static" / "setup.html"


class TokenChoice(BaseModel):
    token: str | None = None  # omitted: generate one


class ImmichDetails(BaseModel):
    immich_base_url: str
    immich_api_key: str | None = None  # empty keeps the saved key
    skip_test: bool = False


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def setup_page():
    return HTMLResponse(_SETUP_PAGE.read_text(encoding="utf-8"))


@app.get("/setup/status")
async def setup_status(request: Request):
    setup: SetupConfig = request.app.state.setup
    conn = setup.connection
    return {
        "setup_required": setup.setup_required,
        "token_set": bool(conn.forge_api_token),
        "immich_configured": conn.immich_configured,
        "models": request.app.state.downloader.status(),
    }


@app.get("/setup/config", dependencies=[Depends(require_setup_access)])
async def setup_get_config(request: Request):
    setup: SetupConfig = request.app.state.setup
    conn = setup.connection
    key = conn.immich_api_key
    return {
        "immich_base_url": conn.immich_base_url,
        "immich_api_key_set": bool(key),
        # Only the last few characters, so you can tell keys apart.
        "immich_api_key_hint": key[-4:] if len(key) >= 8 else "",
        "token_set": bool(conn.forge_api_token),
        "from_env": {
            name: setup.from_env(name)
            for name in ("immich_base_url", "immich_api_key", "forge_api_token")
        },
    }


@app.post("/setup/test", dependencies=[Depends(require_setup_access)])
async def setup_test(details: ImmichDetails, request: Request):
    key = details.immich_api_key or request.app.state.setup.connection.immich_api_key
    return await check_connection(details.immich_base_url, key)


@app.post("/setup/immich", dependencies=[Depends(require_setup_access)])
async def setup_save_immich(details: ImmichDetails, request: Request):
    setup: SetupConfig = request.app.state.setup
    key = details.immich_api_key or setup.connection.immich_api_key
    if not details.skip_test:
        result = await check_connection(details.immich_base_url, key)
        if not result["ok"]:
            raise HTTPException(status_code=400, detail=result["message"])
    try:
        setup.save_immich(details.immich_base_url, details.immich_api_key)
        new_token = setup.ensure_token()  # first run: the token is created here
    except PermissionError as exc:
        raise HTTPException(status_code=409, detail=str(exc))
    _swap_immich_client(request.app)
    # A newly created token is returned once, for the page to show.
    return {"ok": True, "token": new_token}


@app.post("/setup/token", dependencies=[Depends(require_setup_access)])
async def setup_set_token(request: Request, choice: TokenChoice | None = None):
    """Set the Forge token: the user's own (pasted) or a newly generated one.
    Either way the old token stops working at once."""
    setup: SetupConfig = request.app.state.setup
    try:
        if choice and choice.token:
            return {"token": setup.set_token(choice.token)}
        return {"token": setup.regenerate_token()}
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except PermissionError as exc:
        raise HTTPException(status_code=409, detail=str(exc))


@app.post("/setup/models/retry", dependencies=[Depends(require_setup_access)])
async def setup_retry_models(request: Request):
    request.app.state.downloader.retry()
    return request.app.state.downloader.status()


@app.get("/health")
async def health(request: Request):
    immich: ImmichClient = request.app.state.immich
    settings = request.app.state.settings
    setup: SetupConfig = request.app.state.setup
    immich_ok = False
    if setup.connection.immich_configured:
        try:
            immich_ok = await immich.ping()
        except Exception:  # noqa: BLE001
            immich_ok = False
    models = request.app.state.downloader.status()
    return {
        "ok": True,
        "immich": immich_ok,
        "gpu": _gpu_status(settings.device),
        "setup_required": setup.setup_required,
        "models": {k: models[k] for k in ("state", "done_bytes", "total_bytes")},
    }


@app.get("/models")
async def models(request: Request, _: None = Depends(require_token)):
    s = request.app.state.settings
    return {
        "dejpeg": s.dejpeg_backend,
        "denoise": s.denoise_backend,
        "upscale": s.upscale_backend,
        "face": s.face_backend,
        "colorize": s.colorize_backend,
        "low_light": s.low_light_backend,
    }


@app.post("/analyze", response_model=ForgeAnalysis)
async def analyze(req: ForgeRequest, request: Request, _: None = Depends(require_token)):
    """Pre-submit check the extension runs on the photo screen: a source-quality
    verdict (warn when a low-quality original means results will be limited)
    plus Auto-mode suggestions with reasons. CPU only (face detection included)
    — it never touches the pipeline semaphore, so it stays responsive even while
    a forge job is running."""
    immich: ImmichClient = request.app.state.immich
    settings = request.app.state.settings
    try:
        data = await immich.download_original(req.asset_id)
    except ImmichError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except httpx.HTTPError as exc:
        raise HTTPException(status_code=502, detail=f"cannot reach Immich: {exc!r}")

    face_detector = getattr(request.app.state, "face_detector", None)

    def _assess() -> ForgeAnalysis:
        img = Pipeline.decode(data)  # raises ValueError on undecodable bytes
        report = assess_quality(img, settings)
        signals = measure(img, data, face_detector)
        suggestion = decide(signals, settings)
        return ForgeAnalysis(
            low_quality=report.low_quality,
            metrics={
                "sharpness": report.sharpness,
                "hf_ratio": report.hf_ratio,
                "blockiness": report.blockiness,
                "width": float(report.width),
                "height": float(report.height),
                **signals.metrics(),
            },
            suggested=suggestion.operations,
            reasons=suggestion.reasons,
        )

    try:
        return await asyncio.to_thread(_assess)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.post("/forge", response_model=JobInfo)
async def forge(req: ForgeRequest, request: Request, _: None = Depends(require_token)):
    downloader: ModelDownloader = request.app.state.downloader
    if not downloader.ready_to_forge:
        models = downloader.status()
        pct = round(100 * models["done_bytes"] / max(1, models["total_bytes"]))
        detail = (
            f"models still downloading ({pct}%) — try again in a few minutes"
            if models["state"] == "downloading"
            else "some models failed to download — retry on the server's setup page"
        )
        raise HTTPException(status_code=503, detail=detail)
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


@app.get("/immich/original/{asset_id}")
async def immich_original(asset_id: str, request: Request, _: None = Depends(require_token)):
    """Full-resolution original — used for a 1:1-quality preview in the panel.
    Note: may be a non-web format (HEIC/RAW); the client falls back to the
    thumbnail if the browser can't render it."""
    immich: ImmichClient = request.app.state.immich
    r = await immich._client.get(
        f"/api/assets/{asset_id}/original", headers=immich._headers
    )
    if r.status_code != 200:
        raise HTTPException(status_code=502, detail="original fetch failed")
    return Response(
        content=r.content,
        media_type=r.headers.get("content-type", "application/octet-stream"),
    )


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
