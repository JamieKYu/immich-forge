# Forge Server

FastAPI service that pulls an original from Immich, runs a GPU enhancement
pipeline (upscale / face-restore / colorize), and — on accept — uploads the
result and stacks it as the new primary asset.

## Run with Docker (GPU host)

Requires the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).

```bash
cp .env.example .env        # set IMMICH_BASE_URL, IMMICH_API_KEY, FORGE_API_TOKEN
docker compose up --build
```

On first boot the container downloads the model weights into the mounted
`weights/` volume (idempotent — subsequent starts are instant). Set
`FORGE_SKIP_WEIGHT_DOWNLOAD=1` to bypass this and pre-populate `weights/`
yourself (e.g. air-gapped hosts). The `basicsr` `functional_tensor` import is
patched at build time so Real-ESRGAN / GFPGAN import cleanly.

## Run locally without a GPU (fallback mode)

The classical-CV fallbacks let you exercise the full job loop and Immich
round-trip without CUDA or model weights:

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
FORGE_DEVICE=cpu FORGE_UPSCALE_BACKEND=lanczos \
  FORGE_FACE_BACKEND=none FORGE_COLORIZE_BACKEND=none \
  uvicorn app.main:app --reload
```

## API

| Method | Path | Purpose |
|--------|------|---------|
| GET  | `/health` | liveness + Immich connectivity + GPU status |
| GET  | `/models` | active backends |
| POST | `/forge` | submit a job `{asset_id, operations}` → `{job_id}` |
| GET  | `/forge/{id}` | status / progress |
| GET  | `/forge/{id}/result` | forged image bytes |
| POST | `/forge/{id}/accept` | upload + stack as primary |
| GET  | `/immich/search` | proxied asset search (browser UI) |
| GET  | `/immich/thumbnail/{id}` | proxied thumbnail |

All non-health endpoints require `Authorization: Bearer $FORGE_API_TOKEN` when a
token is configured.

### Example

```bash
TOKEN=...; API=http://localhost:8000
# submit
JOB=$(curl -s -X POST $API/forge -H "Authorization: Bearer $TOKEN" \
  -H 'Content-Type: application/json' \
  -d '{"asset_id":"<immich-asset-id>","operations":{"upscale":true,"upscale_factor":4,"face_restore":true}}' \
  | jq -r .job_id)
# poll
curl -s $API/forge/$JOB -H "Authorization: Bearer $TOKEN" | jq
# preview
curl -s $API/forge/$JOB/result -H "Authorization: Bearer $TOKEN" -o forged.jpg
# accept -> becomes the primary asset in a new stack
curl -s -X POST $API/forge/$JOB/accept -H "Authorization: Bearer $TOKEN" | jq
```

## Pipeline order

`colorize → upscale → face_restore`, all exchanging BGR uint8 ndarrays. A single
GPU semaphore serializes jobs so concurrent requests don't OOM the card. Large
images are tiled (`FORGE_TILE_SIZE`).

## Models

| Stage | Backend | Source |
|-------|---------|--------|
| upscale | Real-ESRGAN (`realesrgan` \| `lanczos`) | pip `realesrgan` + `RealESRGAN_x4plus.pth` |
| face restore | GFPGAN (`gfpgan` \| `codeformer` \| `none`) | pip `gfpgan` + `GFPGANv1.4.pth` |
| colorize | DDColor (`ddcolor` \| `none`) | **vendored** under `app/pipeline/ddcolor/` + `ddcolor_modelscope.pth` |

DDColor (ICCV 2023) is vendored as a self-contained torch implementation — no
`basicsr`/`timm` dependency, so it doesn't collide with the `basicsr` Real-ESRGAN
uses. DeOldify was the original plan but needs fastai 1.x / torch 1.x, which is
incompatible with the torch 2.x this image runs on.

Each backend falls back to a classical/no-op impl when its weights are missing.
To enable the deep models: `python scripts/download_weights.py` (or the
container does it on first boot), then set the `FORGE_*_BACKEND` env vars.
