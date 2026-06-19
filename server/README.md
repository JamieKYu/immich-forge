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

## Swapping in real models

1. `python scripts/download_weights.py` (or drop `.pth` files into `weights/`).
2. Set `FORGE_*_BACKEND` env vars.
3. For face restore / colorize, finish the integration points marked in
   `app/pipeline/face_restore.py` and `app/pipeline/colorize.py` (kept optional
   because their deps are heavy and version-sensitive — DeOldify especially is
   best isolated in its own container).
