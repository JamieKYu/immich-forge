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
FORGE_API_TOKEN=dev-token FORGE_DEVICE=cpu FORGE_UPSCALE_BACKEND=lanczos \
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

All non-health endpoints require `Authorization: Bearer $FORGE_API_TOKEN`.
`FORGE_API_TOKEN` is mandatory: if it's unset the API **fails closed** (HTTP 503)
rather than serving the Immich proxy unauthenticated. CORS is closed by default;
set `FORGE_CORS_ORIGINS` only if a non-extension web client needs access.

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

`denoise → colorize → face_restore → upscale`, all exchanging BGR uint8 ndarrays.
Denoise runs *first* so sensor noise is removed before later stages — the
upscaler especially — amplify it; it's tiled (`FORGE_TILE_SIZE`) so VRAM stays
bounded on large sources. Face restore runs *before* upscale so face detection
works on the original-resolution image — detecting faces on a 4×-upscaled
(~300MP) image OOMs the GPU. Upscale is last and tiled.

A single GPU semaphore serializes jobs, the CUDA cache is freed between stages,
and the upscale factor is clamped so output stays under `FORGE_MAX_OUTPUT_PIXELS`
(default 100MP) — e.g. a 19MP source requested at ×4 is clamped to ×2.

## Models

| Stage | Backend | Source |
|-------|---------|--------|
| denoise | SCUNet (`scunet` \| `nlm` \| `none`) | **vendored** under `app/pipeline/scunet/` + `scunet_color_real_psnr.pth` |
| upscale | Real-ESRGAN (`realesrgan` \| `lanczos`) | pip `realesrgan` + `RealESRGAN_x4plus.pth` |
| face restore | CodeFormer (`codeformer` \| `gfpgan` \| `none`) | **vendored** under `app/pipeline/codeformer/` + `codeformer.pth` (default); or pip `gfpgan` + `GFPGANv1.4.pth` |
| colorize | DDColor (`ddcolor` \| `none`) | **vendored** under `app/pipeline/ddcolor/` + `ddcolor_modelscope.pth` |

DDColor (ICCV 2023) is vendored as a self-contained torch implementation — no
`basicsr`/`timm` dependency, so it doesn't collide with the `basicsr` Real-ESRGAN
uses. DeOldify was the original plan but needs fastai 1.x / torch 1.x, which is
incompatible with the torch 2.x this image runs on.

SCUNet (Swin-Conv-UNet) is vendored the same way — torch + `einops` only, with
`timm`'s `trunc_normal_`/`DropPath` inlined. It's a *blind* real-image denoiser,
so it needs no noise-level input. The denoise stage's classical fallback is a
real denoiser (OpenCV Non-Local Means), so a missing weight degrades to `nlm`
rather than a no-op. SCUNet/NAFNet-style models denoise but don't brighten, so
**low-light enhancement is a classical CLAHE + gamma pass** (`low_light`) applied
after denoising. `denoise_strength` (0–1) blends the denoised result back toward
the original to soften over-smoothing.

CodeFormer (default face backend) is vendored the same way — the arch is copied
from sczhou/CodeFormer with the `basicsr` registry decorators stripped, and it's
driven via `facexlib` (detection/align/paste-back) + `basicsr` img helpers. It's
the default because it's the only face backend that honors `face_fidelity` (its
`w` knob: 0 = max quality/most restored, 1 = closest to the original face).
GFPGAN has no such control and ignores `face_fidelity` entirely.

**Model licenses** differ — see the table in the [root README](../README.md#third-party-components).
Note the **default** `codeformer` face backend uses code + weights under the
**non-commercial** S-Lab License; switch to `gfpgan` (Apache-2.0) for commercial
use — the Fidelity control then has no effect.

Each backend falls back to a classical/no-op impl when its weights are missing.
To enable the deep models: `python scripts/download_weights.py` (or the
container does it on first boot), then set the `FORGE_*_BACKEND` env vars.
