FROM python:3.11-slim

# Prevent interactive prompts during apt install
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH="/opt/ComfyUI"

# 1. Install system utilities and tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    git \
    ffmpeg \
    rclone \
    jq \
    ca-certificates \
 && rm -rf /var/lib/apt/lists/*

# 2. Install PyTorch CPU from PyTorch's wheel index
RUN pip install --no-cache-dir \
    torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cpu

# 3. Install other dependencies from standard PyPI
RUN pip install --no-cache-dir \
    safetensors \
    einops \
    psutil \
    scipy \
    numpy

# 4. Bake ComfyUI core directly into /opt/ComfyUI
RUN git clone --depth 1 https://github.com/comfyanonymous/ComfyUI.git /opt/ComfyUI

WORKDIR /__w
