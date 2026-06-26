# Forge for Immich

A companion to [Immich](https://immich.app) that "forges" low-quality photos —
**denoise**, **upscale**, **face-restore**, and **colorize** old, noisy, or
blurry images on your own GPU, then stacks the enhanced result as the new primary
asset in your library.

> **Unofficial / independent project.** "Forge for Immich" is a community-built
> companion. It is **not affiliated with, endorsed by, or sponsored by** Immich
> or FUTO. "Immich" is a trademark of its respective owner; it is used here only
> to describe interoperability.

**Your originals are never modified or deleted.** Forge uploads the enhanced copy
as a *new* asset and stacks it as the primary; the original stays in the stack.
Unstack in Immich at any time to revert.

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

`FORGE_API_TOKEN` is **required** — the server fails closed (HTTP 503) without
it, so the Immich proxy is never exposed unauthenticated. Generate one with
`openssl rand -hex 32`.

The GPU pipeline ships with **classical-CV fallbacks** (Lanczos upscale, etc.) so
the whole loop runs and is testable *before* you download the deep-learning model
weights. Enable the deep models by dropping weights into `server/weights/` and
setting the backend env vars.

## Git hooks

Run once after cloning to enable the pre-commit hook, which runs the test suite
for whichever subproject (`server/` and/or `extension/`) has staged changes:

```bash
./scripts/install-git-hooks.sh
```

The server suite expects `server/.venv` with dev deps installed; the extension
suite expects `extension/node_modules`. Bypass for a single commit with
`git commit --no-verify`.

## Releases

Publishing is **tag-gated** and the two components release **independently**,
each from its own prefixed version tag. Building locally or merging to `main`
never ships anything — a merge just runs the tests (and uploads the extension
`.zip` as a build artifact).

| Push this tag | Triggers | Result |
| --- | --- | --- |
| `api-vX.Y.Z` | server tests → Docker build | `jamiekyu/immich-forge:latest` **and** `:vX.Y.Z` pushed to Docker Hub |
| `ext-vX.Y.Z` | extension tests → build | extension `.zip` attached to a **draft** GitHub Release (the Chrome Web Store upload stays a manual step) |

The two families never cross-trigger: an `api-v` tag can't ship the extension,
and an `ext-v` tag can't ship the image. Cut a release with the helper scripts
(run them on a clean, up-to-date `main`):

```bash
scripts/release-server.sh            # patch bump -> pushes api-vX.Y.(Z+1)
scripts/release-server.sh minor      # -> api-vX.(Y+1).0
scripts/release-extension.sh 1.0.0   # writes 1.0.0 into the manifest, commits, tags ext-v1.0.0
```

Both default to a **patch** bump; pass `major`, `minor`, or an explicit `X.Y.Z`.
The extension script additionally writes the new version into
`extension/manifest.config.ts` and `extension/package.json` and commits it before
tagging, so the built `manifest.json` carries the released version. Pull
`:latest` for the newest server release, or pin `:vX.Y.Z`.

## License

This project is licensed under the [MIT License](LICENSE).

### Third-party components

Forge talks to Immich **only over its public REST API** — no Immich source code
is included — so this project's license is independent of Immich's.

Model code and weights it can use carry their own licenses; review them before
redistribution or commercial use:

| Component | Used as | License |
| --- | --- | --- |
| SCUNet (denoise) | vendored under `server/app/pipeline/scunet/` | Apache-2.0 ([LICENSE](server/app/pipeline/scunet/LICENSE)) |
| DDColor (colorize) | vendored under `server/app/pipeline/ddcolor/` | Apache-2.0 ([LICENSE](server/app/pipeline/ddcolor/LICENSE)) |
| Real-ESRGAN (upscale) | pip dependency + downloaded weights | BSD-3-Clause |
| GFPGAN (face restore) | pip dependency + downloaded weights | Apache-2.0 |
| CodeFormer (optional face backend) | downloaded weights | **S-Lab License 1.0 — non-commercial** |

> ⚠️ The optional CodeFormer backend's weights are licensed for **non-commercial
> use only**. The default face backend is GFPGAN.
