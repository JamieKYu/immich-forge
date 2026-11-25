# CUDA runtime base. Ubuntu 24.04 ships Python 3.12 (numpy 2.5+ needs >= 3.12).
# The torch wheels bring their own CUDA libraries, so the cu124 torch below runs
# fine on this 12.6 base. The base image sets the minimum host driver (560+).
FROM nvidia/cuda:12.6.3-cudnn-runtime-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.12 python3.12-venv \
    libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# 24.04 doesn't allow system-wide pip installs (PEP 668), so install into a venv.
# It's first on PATH, so `python`, `pip` and `uvicorn` all resolve to it.
RUN python3.12 -m venv /opt/venv
ENV PATH=/opt/venv/bin:$PATH
RUN python -m pip install --upgrade pip

WORKDIR /app

# Core deps first (good layer caching).
COPY requirements.txt .
RUN pip install -r requirements.txt

# Torch built for the CUDA in the base image, then the model libs.
# Pin the index URL to the matching CUDA version (cu124 here). 2.6.0 is the
# last torch release with cu124 wheels; newer torch needs a newer CUDA base.
RUN pip install torch==2.6.0 torchvision==0.21.0 \
    --index-url https://download.pytorch.org/whl/cu124
COPY requirements-ml.txt .
# A failed install fails the build: an image without its models would quietly
# serve only the classical fallbacks.
RUN pip install -r requirements-ml.txt

COPY app ./app
COPY scripts ./scripts
RUN chmod +x scripts/entrypoint.sh

EXPOSE 8000
# Ensures model weights are present, then launches uvicorn.
CMD ["/app/scripts/entrypoint.sh"]
