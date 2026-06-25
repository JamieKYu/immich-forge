# Forge for Immich — Server

GPU image that **enhances your Immich photos**: AI upscaling, face restoration, and black‑and‑white colorization. It runs as a small broker next to your self‑hosted Immich — it holds your Immich API key so the browser never has to — and pairs with the **Forge for Immich** Chrome extension.

`upscale (Real‑ESRGAN) · face restore (GFPGAN) · colorize (DDColor)`

---

## Requirements

- A self‑hosted **Immich** server (you only need its URL and an API key).
- An **NVIDIA GPU** with the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) installed on the Docker host.
  - No GPU? You can still run it in CPU mode for testing — see [CPU‑only](#cpu-only-no-gpu) below.

---

## 1. Get your Immich API key

In the Immich web app: click your **avatar (top‑right) → Account Settings → API Keys → New API Key**. Give it a name and copy the key — you'll paste it below.

Also note your Immich URL. For a typical self‑hosted setup it's `http://YOUR_SERVER_IP:2283`.

## 2. Run the server

```bash
docker run -d \
  --name immich-forge \
  --gpus all \
  -p 8000:8000 \
  -e IMMICH_BASE_URL=http://YOUR_SERVER_IP:2283 \
  -e IMMICH_API_KEY=paste-your-immich-api-key \
  -e FORGE_API_TOKEN=$(openssl rand -hex 32) \
  -v immich-forge-weights:/app/weights \
  --restart unless-stopped \
  jamiekyu/immich-forge:latest
```

- `IMMICH_BASE_URL` / `IMMICH_API_KEY` — point Forge at your Immich and let it act on your behalf.
- `FORGE_API_TOKEN` — **required**; a secret that protects this server (anyone who can reach it could otherwise use your GPU and read your Immich library). The command above generates a random one; **set your own and save it** — you'll enter it in the extension. If it's unset the server fails closed (HTTP 503).
- `-v immich-forge-weights:/app/weights` — model weights (~2–3 GB) download here on first start. Keep this volume so they aren't re‑downloaded.

Check it's healthy:

```bash
curl http://YOUR_SERVER_IP:8000/health
# {"ok": true, "immich": true, ...}
```

> First boot downloads the model weights, so give it a minute before the GPU stages work. Until they're present, Forge falls back to basic (non‑AI) processing.

## 3. Install the Chrome extension

Install **“Forge for Immich”** from the **Chrome Web Store**:
https://chromewebstore.google.com/detail/forge-for-immich/ceaoooljelkcaagcljambkdldadpblgc

Then open a single photo in your Immich web app, click the **Forge** toolbar icon, open **settings**, and enter:

- **Forge server URL** — `http://YOUR_SERVER_IP:8000`
- **Forge API token** — the `FORGE_API_TOKEN` value from step 2

---

## Configuration

| Variable | Default | Description |
| --- | --- | --- |
| `IMMICH_BASE_URL` | `http://immich-server:2283` | Your Immich server URL. **Set this.** |
| `IMMICH_API_KEY` | _(empty)_ | Immich API key. **Set this.** |
| `FORGE_API_TOKEN` | _(empty)_ | Bearer token clients must send. **Required** — unset = API fails closed (503). |
| `FORGE_DEVICE` | `cuda` | `cuda` or `cpu`. |
| `FORGE_WEIGHTS_DIR` | `weights` | Where model weights live (mount a volume here). |
| `FORGE_TILE_SIZE` | `512` | Upscale tile size; lower it (e.g. `256`) if you hit GPU out‑of‑memory. |
| `FORGE_MAX_OUTPUT_PIXELS` | `100000000` | Caps output size; the upscale factor is clamped to fit. |
| `FORGE_UPSCALE_BACKEND` | `realesrgan` | `realesrgan` or `lanczos`. |
| `FORGE_FACE_BACKEND` | `gfpgan` | `gfpgan`, `codeformer`, or `none`. |
| `FORGE_COLORIZE_BACKEND` | `ddcolor` | `ddcolor` or `none`. |
| `FORGE_MAX_CONCURRENT_GPU_JOBS` | `1` | Simultaneous GPU jobs. |
| `FORGE_JOB_TTL_SECONDS` | `3600` | How long finished results are kept. |

The server listens on **port 8000**. `/health` is public; all other endpoints require the `FORGE_API_TOKEN` bearer token when one is set.

## docker compose

```yaml
services:
  forge:
    image: jamiekyu/immich-forge:latest
    ports:
      - "8000:8000"
    environment:
      IMMICH_BASE_URL: http://YOUR_SERVER_IP:2283
      IMMICH_API_KEY: paste-your-immich-api-key
      FORGE_API_TOKEN: a-long-random-token
    volumes:
      - immich-forge-weights:/app/weights
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

## CPU‑only (no GPU)

Drop `--gpus all` and set `FORGE_DEVICE=cpu`. AI stages are very slow on CPU and some fall back to classical methods — fine for trying it out, not for everyday use.

```bash
docker run -d --name immich-forge -p 8000:8000 \
  -e FORGE_DEVICE=cpu \
  -e IMMICH_BASE_URL=http://YOUR_SERVER_IP:2283 \
  -e IMMICH_API_KEY=paste-your-immich-api-key \
  -e FORGE_API_TOKEN=$(openssl rand -hex 32) \
  -v immich-forge-weights:/app/weights \
  jamiekyu/immich-forge:latest
```
