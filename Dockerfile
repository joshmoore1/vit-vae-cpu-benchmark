FROM python:3.11-slim

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH="/opt/ComfyUI"

# 1. System utilities (including zstd for fast cache extraction)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    wget \
    git \
    ffmpeg \
    rclone \
    jq \
    ca-certificates \
    zstd \
 && rm -rf /var/lib/apt/lists/*

# 2. PyTorch CPU
RUN pip install --no-cache-dir \
    torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cpu

# 3. Clone ComfyUI core
RUN git clone --depth 1 https://github.com/comfyanonymous/ComfyUI.git /opt/ComfyUI

# 4. Install ComfyUI requirements + your benchmark utilities
RUN pip install --no-cache-dir -r /opt/ComfyUI/requirements.txt && \
    pip install --no-cache-dir safetensors psutil scipy numpy

WORKDIR /__w
