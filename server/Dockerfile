# CUDA runtime base — provides the NVIDIA libraries Torch links against.
FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 python3-pip python3.11-dev \
    libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/bin/python3.11 /usr/bin/python && python -m pip install --upgrade pip

WORKDIR /app

# Core deps first (good layer caching).
COPY requirements.txt .
RUN pip install -r requirements.txt

# Torch built for the CUDA in the base image, then the model libs.
# Pin the index URL to the matching CUDA version (cu124 here).
RUN pip install torch==2.5.1 torchvision==0.20.1 \
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
