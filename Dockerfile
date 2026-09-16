FROM python:3.11-slim

# Prevent interactive prompts during apt install
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH="/opt/ComfyUI:$PYTHONPATH"

# 1. Install system utilities and tools (including rclone and ffmpeg)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    git \
    ffmpeg \
    rclone \
    jq \
    ca-certificates \
 && rm -rf /var/lib/apt/lists/*

# 2. Pre-install PyTorch CPU & required runtime packages
RUN pip install --no-cache-dir \
    torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu \
    safetensors \
    einops \
    psutil \
    scipy \
    numpy

# 3. Bake ComfyUI core directly into /opt/ComfyUI
RUN git clone --depth 1 https://github.com/comfyanonymous/ComfyUI.git /opt/ComfyUI

# (Optional: If public repo and you want the 5.21GB model baked in, uncomment below:)
# RUN curl -L -C - "https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/main/vae/minimax_h3_video_vae_fp16.safetensors" \
#     -o /opt/minimax_h3_video_vae_fp16.safetensors

WORKDIR /__w
