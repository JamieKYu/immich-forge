#!/bin/sh
# Ensure model weights are present before the server starts, so the pipeline
# never silently falls back to classical CV because a weight was missing.
#
# download_weights.py is idempotent (skips files already in /app/weights, which
# is the mounted ./weights volume), so this only downloads on the first boot.
# Set FORGE_SKIP_WEIGHT_DOWNLOAD=1 to bypass (e.g. air-gapped hosts that
# pre-populate the volume).
set -e

if [ "${FORGE_SKIP_WEIGHT_DOWNLOAD:-0}" != "1" ]; then
  python scripts/download_weights.py \
    || echo "WARN: weight download failed; missing stages will use classical fallbacks"
fi

exec uvicorn app.main:app --host 0.0.0.0 --port 8000
