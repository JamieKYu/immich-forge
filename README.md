# Immich Forge

A companion to [Immich](https://immich.app) that "forges" low-quality photos —
**upscale**, **face-restore**, and **colorize** old or blurry images on your own
GPU, then stacks the enhanced result as the new primary asset in your library.

## Architecture

```
┌──────────────────┐    ┌──────────────────────┐    ┌─────────────────┐
│ Chrome Extension │───▶│  Forge Server        │───▶│  Immich Server  │
│ (browse, pick,   │    │  (FastAPI + GPU       │    │  (REST API)     │
│  review, accept) │◀───│   PyTorch pipeline)   │◀───│                 │
└──────────────────┘    └──────────────────────┘    └─────────────────┘
        talks only to Forge          holds the Immich API key,
                                      pulls originals / uploads / stacks
```

Two design choices (see the build plan):

- **Server-as-broker** — the Forge server holds the Immich API key. The browser
  extension never sees it; it only talks to Forge.
- **Async job queue** — `POST /forge` returns a `jobId`; the client polls for
  progress. GPU work can take seconds to minutes without blocking.

## Repo layout

```
server/      FastAPI service + GPU pipeline + Immich client  (Docker, NVIDIA)
extension/   Manifest V3 Chrome extension (Vite + React + TS)
```

See [`server/README.md`](server/README.md) and
[`extension/README.md`](extension/README.md) to run each component.

## Quick start

```bash
# 1. Forge server (GPU host with NVIDIA Container Toolkit)
cd server
cp .env.example .env          # set IMMICH_BASE_URL, IMMICH_API_KEY, FORGE_API_TOKEN
docker compose up --build

# 2. Extension
cd extension
npm install
npm run build                 # load extension/dist as an unpacked extension in Chrome
```

## Status

Milestone 1–2 scaffold: end-to-end upscale + Immich round-trip. The GPU pipeline
ships with **classical-CV fallbacks** (Lanczos upscale, etc.) so the whole loop
runs and is testable *before* you download the deep-learning model weights. Swap
in Real-ESRGAN / GFPGAN-CodeFormer / DeOldify by dropping weights into
`server/weights/` and setting the backend env vars.
