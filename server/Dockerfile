# CUDA runtime base — provides the NVIDIA libraries Torch links against.
FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

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
RUN pip install --upgrade setuptools wheel
COPY requirements-ml.txt .
# basicsr's setup.py imports torch at build time and breaks under pip's build
# isolation (the isolated env has no torch). torch is already installed above,
# so disable isolation. NOTE: this stack requires Python 3.11 (the base image) —
# basicsr 1.4.2's setup.py fails on Python 3.13+ due to a PEP 667 locals() change.
RUN pip install --no-build-isolation -r requirements-ml.txt || \
    echo "WARN: ML deps failed to install; server will use classical fallbacks"

COPY app ./app
COPY scripts ./scripts

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
