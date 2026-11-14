# Forge for Immich — Server

**Repair and enhance old, blurry, noisy or low-resolution photos in your self-hosted [Immich](https://immich.app) library**

Open a photo in Immich, click the Forge icon, and Forge can:

- **Enlarge** small photos (×2 / ×4)
- **Restore faces** that are small or soft
- **Colorize** black-and-white and sepia photos
- **Clean up JPEG blocks**, **reduce noise**, and **brighten** dark shots

<br>

You will also need the **[Forge for Immich](https://chromewebstore.google.com/detail/forge-for-immich/ceaoooljelkcaagcljambkdldadpblgc)** Chrome extension.

---

## Getting started

### Step 1 — Gather three things

1. **Immich IP address.** The address you open Immich at, e.g. `http://192.168.1.50:2283`.
   <br>⚠️ Use the computer's **network IP address**, not `localhost`
<br>
2. **Immich API key.** In Immich: **your avatar (top-right) → Account Settings → API Keys → New API Key**. Copy it.
<br>
3. **Forge token** — a password you make up, so only you can use your Forge server. Generate one and **save it**; you'll paste it into the extension too:
   - **Mac / Linux:** `openssl rand -hex 32`
   - **Windows (PowerShell):** `[guid]::NewGuid().ToString("N") + [guid]::NewGuid().ToString("N")`

### Step 2 — Start the server

Pick the command for your computer and replace the three `paste-` with values from step 1.

#### 🐧 Linux with an NVIDIA GPU (recommended)

Install the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) first, then:

```bash
docker run -d --name immich-forge --gpus all -p 8000:8000 \
  -e IMMICH_BASE_URL=http://paste-your-immich-ip:2283 \
  -e IMMICH_API_KEY=paste-your-immich-api-key \
  -e FORGE_API_TOKEN=paste-your-forge-token \
  -v immich-forge-weights:/app/weights \
  --restart unless-stopped \
  jamiekyu/immich-forge:latest
```

#### 🪟 Windows with an NVIDIA GPU

Use [Docker Desktop](https://www.docker.com/products/docker-desktop/) (WSL 2 backend, the default) with a current NVIDIA driver; GPU support works out of the box
<br>
In PowerShell:
```powershell
docker run -d --name immich-forge --gpus all -p 8000:8000 `
  -e IMMICH_BASE_URL=http://paste-your-immich-ip:2283 `
  -e IMMICH_API_KEY=paste-your-immich-api-key `
  -e FORGE_API_TOKEN=paste-your-forge-token `
  -v immich-forge-weights:/app/weights `
  --restart unless-stopped `
  jamiekyu/immich-forge:latest
```

#### 🍎 Mac with Apple Silicon

The image is built for Intel/AMD (`linux/amd64`) because it uses NVIDIA's CUDA. On an Apple Silicon Mac, Docker runs it through emulation, on the CPU:

1. In **Docker Desktop → Settings → General**, turn on **"Use Rosetta for x86_64/amd64 emulation on Apple Silicon"** (much faster than the default emulation).
2. Run it with `--platform linux/amd64` and `FORGE_DEVICE=cpu`:

```bash
docker run -d --name immich-forge --platform linux/amd64 -p 8000:8000 \
  -e FORGE_DEVICE=cpu \
  -e IMMICH_BASE_URL=http://paste-your-immich-ip:2283 \
  -e IMMICH_API_KEY=paste-your-immich-api-key \
  -e FORGE_API_TOKEN=paste-your-forge-token \
  -v immich-forge-weights:/app/weights \
  --restart unless-stopped \
  jamiekyu/immich-forge:latest
```

Without `--platform linux/amd64`, `docker pull` fails with *"no matching manifest for linux/arm64/v8"*. Expect enhancements to be **slow** (minutes per photo).

#### 💻 Intel Mac, or any computer without an NVIDIA GPU

Drop `--gpus all` and add `FORGE_DEVICE=cpu`. Runs on the CPU, so may be slow:

```bash
docker run -d --name immich-forge -p 8000:8000 \
  -e FORGE_DEVICE=cpu \
  -e IMMICH_BASE_URL=http://paste-your-immich-ip:2283 \
  -e IMMICH_API_KEY=paste-your-immich-api-key \
  -e FORGE_API_TOKEN=paste-your-forge-token \
  -v immich-forge-weights:/app/weights \
  --restart unless-stopped \
  jamiekyu/immich-forge:latest
```

### Step 3 — Check it's running

On first start Forge downloads its AI models (**about 2.3 GB**, once). Watch progress:

```bash
docker logs -f immich-forge
```

It's ready when you see `forge ready` and `Uvicorn running on http://0.0.0.0:8000` (press Ctrl+C to stop watching; the server keeps running).

### Step 4 — Connect the Chrome extension

1. Install **[Forge for Immich](https://chromewebstore.google.com/detail/forge-for-immich/ceaoooljelkcaagcljambkdldadpblgc)** from the Chrome Web Store.
2. Click the Forge toolbar icon → **Set up Forge server**.
3. Enter the **server address** — `http://YOUR_FORGE_IP:8000` (or `http://localhost:8000` if Forge runs on the same computer as your browser) — and your **Forge token** from step 1, then click **Save & test**. Allow access when Chrome asks.
4. Open any photo in Immich and click the Forge icon. 🎉

---

## Using Forge

- **Auto** (the default) suggests enhancements for each photo, with the reason under every option. **Custom** starts from your own defaults, set on the extension's options page. Change anything per photo before you forge.
- Click **Forge**, then compare **before / after** by dragging across the image.
- **Save to Immich** adds the result as the primary photo of a stack, with your original kept inside it. **Discard** throws the result away.

---

## docker compose

Save as `docker-compose.yml`, fill in your values, and run `docker compose up -d`:

```yaml
services:
  forge:
    image: jamiekyu/immich-forge:latest
    # platform: linux/amd64         # uncomment on Apple Silicon Macs
    ports:
      - "8000:8000"
    environment:
      IMMICH_BASE_URL: http://paste-your-immich-ip:2283
      IMMICH_API_KEY: paste-your-immich-api-key
      FORGE_API_TOKEN: paste-your-forge-token
      # FORGE_DEVICE: cpu            # uncomment if you have no NVIDIA GPU
    volumes:
      - immich-forge-weights:/app/weights
    # Delete this "deploy" block if you have no NVIDIA GPU:
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]
    restart: unless-stopped

volumes:
  immich-forge-weights:
```

---

## Troubleshooting

| Problem | Fix |
| --- | --- |
| `no matching manifest for linux/arm64/v8` | You're on an Apple Silicon Mac. Add `--platform linux/amd64` (see step 2). |
| *"The requested image's platform (linux/amd64) does not match…"* warning | Harmless on Apple Silicon — it's running through emulation. Add `--platform linux/amd64` to silence it. |
| `/health` says `"immich": false` | Forge can't reach Immich. Check `IMMICH_BASE_URL` uses your Immich computer's **IP address** (not `localhost`) and port `2283`, then recreate the container. |
| Extension says **token rejected** | The token in the extension must exactly match `FORGE_API_TOKEN`. |
| Extension says the server has **no token set** (HTTP 503) | `FORGE_API_TOKEN` is empty. Forge refuses all requests until it's set — this keeps your library private. |
| Extension **can't reach** the server | Check the container is running (`docker ps`), the address uses port `8000`, and no firewall blocks it. From another computer, use the Forge machine's IP, not `localhost`. |
| `"cuda_available": false` on an NVIDIA machine | Add `--gpus all` (or the compose `deploy` block), and install the NVIDIA Container Toolkit on Linux. |
| Out of GPU memory | Lower `FORGE_TILE_SIZE` to `256`, and `FORGE_LOW_LIGHT_MAX_PIXELS` to `2000000`. |
| Auto says **"face detection unavailable"** | The face model is missing from the weights volume. Restart the container so it downloads, or check the logs for a failed download. |
| Changed a setting but nothing happened | Settings are read at start: recreate the container (`docker rm -f` + run again, or `docker compose up -d`). A plain `docker restart` keeps the old settings. |

---

## Settings

Set these with `-e NAME=value` (or under `environment:` in compose).

**Essentials**

| Variable | Default | What it does |
| --- | --- | --- |
| `IMMICH_BASE_URL` | `http://immich-server:2283` | Your Immich address. **Set this.** |
| `IMMICH_API_KEY` | _(empty)_ | Your Immich API key. **Set this.** |
| `FORGE_API_TOKEN` | _(empty)_ | The token the extension must send. **Required** — when empty, Forge refuses all requests (HTTP 503). |
| `FORGE_DEVICE` | `cuda` | `cuda` (NVIDIA GPU) or `cpu`. |

**Performance and limits**

| Variable | Default | What it does |
| --- | --- | --- |
| `FORGE_TILE_SIZE` | `512` | Images are processed in tiles this big. Lower (e.g. `256`) if the GPU runs out of memory; `0` turns tiling off. |
| `FORGE_LOW_LIGHT_MAX_PIXELS` | `4000000` | Largest photo brightened in one piece (~1.35 GB VRAM per megapixel). Bigger photos use a downscale carried back to full size. Raise on a big GPU (e.g. `12000000` on 24 GB). |
| `FORGE_MAX_IMAGE_PIXELS` | `40000000` | Largest photo Forge accepts (40 MP). |
| `FORGE_MAX_OUTPUT_PIXELS` | `100000000` | Largest result (100 MP). Enlarge ×4 drops to ×2 (or off) to fit. |
| `FORGE_MAX_CONCURRENT_GPU_JOBS` | `1` | Jobs processed at once. |
| `FORGE_JOB_TTL_SECONDS` | `3600` | How long a forge result is kept (counted from when the job started) before it's discarded if not saved. |

**AI models** — each can be swapped, and most turned off (`none` greys that option out in the extension)

| Variable | Default | Options |
| --- | --- | --- |
| `FORGE_DEJPEG_BACKEND` | `fbcnn` | `fbcnn` (FBCNN, Apache-2.0) · `none` |
| `FORGE_DENOISE_BACKEND` | `scunet` | `scunet` (SCUNet) · `nlm` (classic non-AI denoiser) · `none` |
| `FORGE_LOW_LIGHT_BACKEND` | `hvicidnet` | `hvicidnet` (HVI-CIDNet, MIT) · `classical` (CLAHE + gamma) |
| `FORGE_UPSCALE_BACKEND` | `realesrgan` | `realesrgan` (Real-ESRGAN) · `lanczos` (classic resize) |
| `FORGE_FACE_BACKEND` | `codeformer` | `codeformer` (best results; **non-commercial** S-Lab license) · `gfpgan` (Apache-2.0) · `gfpgan+codeformer` · `none` |
| `FORGE_COLORIZE_BACKEND` | `ddcolor` | `ddcolor` (DDColor) · `none` |

**Auto mode**

| Variable | Default | What it does |
| --- | --- | --- |
| `FORGE_AUTO_FACE_DETECTION` | `true` | Look for faces (on the CPU) to decide whether to suggest **Restore faces**. `false` skips it. |
| `FORGE_AUTO_*` | — | Fine-tune when Auto suggests each enhancement (e.g. `FORGE_AUTO_NOISE_SIGMA_MIN`). See the [full list](https://github.com/JamieKYu/immich-forge/blob/main/server/app/config.py) and [how the rules work](https://github.com/JamieKYu/immich-forge/blob/main/server/README.md#auto-mode-suggestions). |

**Advanced**

| Variable | Default | What it does |
| --- | --- | --- |
| `FORGE_SKIP_WEIGHT_DOWNLOAD` | `0` | `1` skips the model download at start (for offline hosts with a pre-filled weights volume). |
| `FORGE_WEIGHTS_DIR` | `weights` | Where the AI models are stored inside the container (`/app/weights`). Keep the volume mounted there. |
| `FORGE_CORS_ORIGINS` | _(empty)_ | Leave empty. The extension doesn't need it; set only for a custom web client. |
