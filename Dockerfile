FROM python:3.11-slim

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH="/opt/ComfyUI"

# 1. System utilities and tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    wget \
    git \
    ffmpeg \
    rclone \
    jq \
    ca-certificates \
 && rm -rf /var/lib/apt/lists/*

# 2. Pre-install PyTorch CPU (must come before requirements.txt so pip doesn't pull CUDA torch)
RUN pip install --no-cache-dir \
    torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cpu

# 3. Clone ComfyUI core
RUN git clone --depth 1 https://github.com/comfyanonymous/ComfyUI.git /opt/ComfyUI

# 4. Install ComfyUI requirements (includes comfy-aimdo) + your benchmark utilities
RUN pip install --no-cache-dir -r /opt/ComfyUI/requirements.txt && \
    pip install --no-cache-dir safetensors psutil scipy numpy

# 5. Bake the 5.21 GB VAE model directly into the image
RUN curl -L -C - "https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/main/vae/minimax_h3_video_vae_fp16.safetensors" \
    -o /opt/minimax_h3_video_vae_fp16.safetensors

WORKDIR /__w
