#!/bin/sh
# Start the server straight away. Missing model weights download in the
# background (progress on the setup page at /), and forging waits until they're
# all present — see app/models_download.py. Set FORGE_SKIP_WEIGHT_DOWNLOAD=1 to
# skip downloads (e.g. offline hosts with a pre-filled weights volume).
set -e

exec uvicorn app.main:app --host 0.0.0.0 --port 8000
