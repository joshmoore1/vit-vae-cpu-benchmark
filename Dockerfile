FROM python:3.11-slim

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH="/opt/ComfyUI"

# 1. System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    wget \
    git \
    ffmpeg \
    rclone \
    jq \
    ca-certificates \
 && rm -rf /var/lib/apt/lists/*

# 2. PyTorch CPU
RUN pip install --no-cache-dir \
    torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cpu

# 3. Python libraries
RUN pip install --no-cache-dir \
    safetensors \
    einops \
    psutil \
    scipy \
    numpy

# 4. ComfyUI core
RUN git clone --depth 1 https://github.com/comfyanonymous/ComfyUI.git /opt/ComfyUI

# 5. Bake the 5.21 GB VAE model directly into the image
RUN curl -L -C - "https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/main/vae/minimax_h3_video_vae_fp16.safetensors" \
    -o /opt/minimax_h3_video_vae_fp16.safetensors

WORKDIR /__w
