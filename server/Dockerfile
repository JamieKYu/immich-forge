# Plain Ubuntu 26.04 (Python 3.14), no NVIDIA CUDA base image: the torch wheels
# bring their own CUDA runtime and cuDNN, so the host only needs the NVIDIA
# driver (525+ for these CUDA 12 wheels) and the NVIDIA Container Toolkit.
FROM ubuntu:26.04

# NVIDIA_* tell the NVIDIA Container Toolkit to mount the driver's compute
# libraries into the container (the nvidia/cuda images set these for you).
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility

# Only Python: OpenCV is the headless build, so no libGL / X11 / Qt libraries.
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.14 python3.14-venv \
    && rm -rf /var/lib/apt/lists/*

# Ubuntu doesn't allow system-wide pip installs (PEP 668), so install into a venv.
# It's first on PATH, so `python`, `pip` and `uvicorn` all resolve to it.
RUN python3.14 -m venv /opt/venv
ENV PATH=/opt/venv/bin:$PATH
RUN python -m pip install --upgrade pip

WORKDIR /app

# Core deps first (good layer caching).
COPY requirements.txt .
RUN pip install -r requirements.txt

# Torch from the CUDA 12.6 wheel index (the newest torch still built for CUDA
# 12). torchvision must come from the same index: PyPI's build targets a
# different CUDA.
RUN pip install torch==2.14.1 torchvision==0.29.1 \
    --index-url https://download.pytorch.org/whl/cu126
COPY requirements-ml.txt requirements-ml-nodeps.txt ./
# A failed install fails the build: an image without its models would quietly
# serve only the classical fallbacks.
RUN pip install -r requirements-ml.txt \
    && pip install --no-deps -r requirements-ml-nodeps.txt \
    && ! pip show -q opencv-python
# (The last check fails the build if anything pulled in the full opencv-python.)

COPY app ./app
COPY scripts ./scripts
RUN chmod +x scripts/entrypoint.sh

EXPOSE 8000
# Ensures model weights are present, then launches uvicorn.
CMD ["/app/scripts/entrypoint.sh"]
