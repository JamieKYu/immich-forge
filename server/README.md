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
yourself (e.g. air-gapped hosts).

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
| POST | `/analyze` | quality precheck + Auto-mode suggestions `{asset_id}` (CPU only) |
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

`dejpeg → denoise → colorize → (face_restore + upscale)`, all exchanging BGR
uint8 ndarrays. De-JPEG (opt-in) runs *first*: JPEG blocking and ringing are
part of the source as stored, so they're removed before anything brightens,
smooths or enlarges them. Denoise runs next so sensor noise is removed before
later stages — the upscaler especially — amplify it. Both are tiled
(`FORGE_TILE_SIZE`) so VRAM stays bounded on large sources.

When both **face restore** and **upscale** are enabled they run as a single
stage: faces are detected/aligned on the original-resolution image (cheap, and
it avoids detecting on a 4×-upscaled ~300MP image, which OOMs the GPU), the
**background** is upscaled by Real-ESRGAN, and the restored 512px face crops are
pasted onto that upscaled canvas *last*. This keeps the *general* upscaler off
the face pixels — running it over faces turns them waxy/"un-human". If only one
of the two is enabled it runs on its own (an in-place face restore, or a plain
upscale). Both are tiled.

A single GPU semaphore serializes jobs, the CUDA cache is freed between stages,
and the upscale factor is clamped so output stays under `FORGE_MAX_OUTPUT_PIXELS`
(default 100MP) — e.g. a 19MP source requested at ×4 is clamped to ×2.

## Auto-mode suggestions

`/analyze` suggests which enhancements a photo probably needs, with a short
reason for every option, on or off. The extension's Auto mode shows these.

```json
{
  "low_quality": true,
  "suggested": { "dejpeg": true, "upscale": true, "upscale_factor": 4, "colorize": true, "...": "..." },
  "reasons": {
    "dejpeg": "heavily compressed (JPEG quality ≈20)",
    "upscale": "800×600 source",
    "colorize": "black-and-white photo",
    "face_restore": "no faces found",
    "...": "..."
  },
  "metrics": { "jpeg_quality": 20.0, "noise_sigma": 0.4, "chroma_p95": 0.0, "...": 0 }
}
```

| Option | Suggested when (defaults) |
|--------|---------------------------|
| JPEG cleanup | estimated JPEG quality < 60 (from the file's quantization tables), or a visible 8×8 block-edge step ≥ 2 grey levels |
| Reduce noise | estimated noise σ ≥ 8 (0–255), camera ISO ≥ 3200, or σ ≥ 4.8 when Brighten is also suggested |
| Brighten | median brightness ≤ 60 and 95th percentile ≤ 190 (a silhouette against a bright sky isn't "dark"), except a lit subject on a black background: ≥ 52% near-black with the rest at median ≥ 45 (flash or torch-lit photos at night) |
| Enlarge | ×4 up to 1200 px on the long side, ×2 up to 2500 px, within `FORGE_MAX_OUTPUT_PIXELS` |
| Restore faces | a face 80–200 px wide, or soft — unless the largest face is itself over 200 px and sharp (softer faces are then likely background blur). Faces under 80 px are ignored as too small to restore well |
| Colorize | black-and-white (95th-percentile chroma ≤ 6), or sepia-toned (chroma ≤ 15, hue concentration ≥ 0.97, a warm hue between 20° and 110°) — and never if ≥ 0.1% of pixels are strongly coloured (real colour somewhere, e.g. a coloured house or icon) |

It's all CPU work (about 0.2 s for a 24 MP photo, plus face detection), so it
never waits on the GPU while a job runs. Face detection uses facexlib's
RetinaFace on the CPU and loads only if `detection_Resnet50_Final.pth` is
already in the weights folder (`download_weights.py` fetches it); otherwise the
reason reads "face detection unavailable". Options whose backend is `none` are
never suggested.

The cutoffs lean conservative for JPEG cleanup and face restoration, whose
false positives cost the most (softening a clean photo, changing how someone
looks). They're tunable with the `FORGE_AUTO_*` settings in `app/config.py`. To
see what Auto would pick for your own photos, with the raw numbers:

```bash
python scripts/suggest_report.py --metrics ~/Pictures/scans
```

## Models

| Stage | Backend | Source |
|-------|---------|--------|
| de-JPEG | FBCNN (`fbcnn` \| `none`) | pip `spandrel` + `fbcnn_color.pth` |
| denoise | SCUNet (`scunet` \| `nlm` \| `none`) | **vendored** under `app/pipeline/scunet/` + `scunet_color_real_psnr.pth` |
| upscale | Real-ESRGAN (`realesrgan` \| `lanczos`) | pip `spandrel` + `RealESRGAN_x4plus.pth` |
| face restore | CodeFormer (`codeformer` \| `gfpgan` \| `gfpgan+codeformer` \| `none`) | **vendored** under `app/pipeline/codeformer/` + `codeformer.pth` (default); or `spandrel` + `GFPGANv1.4.pth`; pip `facexlib` for detection/paste-back |
| colorize | DDColor (`ddcolor` \| `none`) | **vendored** under `app/pipeline/ddcolor/` + `ddcolor_modelscope.pth` |
| low-light | HVI-CIDNet (`hvicidnet` \| `classical`) | pip `spandrel` + `hvi_cidnet_generalization.safetensors` |

DDColor (ICCV 2023) is vendored as a self-contained torch implementation, with
no `basicsr`/`timm` dependency. DeOldify was the original plan but needs fastai 1.x / torch 1.x, which is
incompatible with the torch 2.x this image runs on.

SCUNet (Swin-Conv-UNet) is vendored the same way — torch + `einops` only, with
`timm`'s `trunc_normal_`/`DropPath` inlined. It's a *blind* real-image denoiser,
so it needs no noise-level input. The denoise stage's classical fallback is a
real denoiser (OpenCV Non-Local Means), so a missing weight degrades to `nlm`
rather than a no-op. `denoise_strength` (0–1) blends the denoised result back
toward the original to soften over-smoothing.

**Low-light** (`low_light`) is **HVI-CIDNet** (CVPR 2025, MIT; the authors'
"Generalization" weights, loaded via `spandrel`), with classical CLAHE + gamma
as the fallback (`FORGE_LOW_LIGHT_BACKEND=hvicidnet|classical`). It runs inside
the denoise stage *before* denoising: on a dark photo the denoiser would smear
the faint detail into flat patches, whereas HVI-CIDNet handles noisy dark input
itself and the denoiser then cleans up the noise brightening exposes. The model
can't be tiled (its attention is global), so above `FORGE_LOW_LIGHT_MAX_PIXELS`
(default 4MP, ~1.35GB VRAM per MP) it runs on a downscale whose brightness/colour
is carried back to full resolution with the source's own detail re-added.

**De-JPEG** (`dejpeg`, opt-in) is **FBCNN** (ICCV 2021, Apache-2.0;
`fbcnn_color.pth`, loaded via `spandrel`). It's *blind* — it predicts the JPEG
quality factor from the image and conditions on it — so it needs no setting.
It removes blocking and colour blotches from heavily compressed images (q=10:
29.8 → 32.1 dB), but lightly smooths an already-clean one, hence opt-in. Tiles
stay on the 8×8 JPEG block grid. Without the weights it's a logged no-op
(`FORGE_DEJPEG_BACKEND=fbcnn|none`).

CodeFormer (default face backend) is vendored the same way — the arch is copied
from sczhou/CodeFormer with the `basicsr` registry decorators stripped. Every
face backend shares one `facexlib` flow (`app/pipeline/faces.py`): detect,
align each face to a 512px crop, restore the crop, paste it back through a
face-parsing mask. GFPGAN's network is loaded via `spandrel`; neither `gfpgan`
nor `basicsr` is installed. CodeFormer is the default because it's the only face backend that honors `face_fidelity` (its
`w` knob: 0 = max quality/most restored, 1 = closest to the original face).
GFPGAN has no such control and ignores `face_fidelity` entirely.

`gfpgan+codeformer` runs both on each aligned face crop before a single
paste-back — GFPGAN first (it's good at fixing eye positioning / facial symmetry
but leaves a smooth "plastic" skin), then CodeFormer over that result to add
realistic skin texture back. `face_fidelity`
drives the CodeFormer pass here, so set it **low** (~0.3-0.5) to let the prior
add texture. If either weight is absent the stage degrades to whichever loaded.

**Model licenses** differ — see the table in the [root README](../README.md#third-party-components).
Note the **default** `codeformer` face backend uses code + weights under the
**non-commercial** S-Lab License; switch to `gfpgan` (Apache-2.0) for commercial
use — the Fidelity control then has no effect.

Each backend falls back to a classical/no-op impl when its weights are missing.
To enable the deep models: `python scripts/download_weights.py` (or the
container does it on first boot), then set the `FORGE_*_BACKEND` env vars.
