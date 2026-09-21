# vit-vae-cpu-benchmark

A benchmarking suite for evaluating CPU inference latency, memory scaling (RSS), and SIMD vectorization (AVX2 / AVX-512) for large-scale 3D Vision Transformer Autoencoders (Currently testing MiniMax-H3 VAE).

## Overview

This benchmark profiles the CPU execution characteristics of multi-layer 3D Vision Transformer decoders:
- **Layer-wise forward latency ($s/\text{block}$)**: Granular forward-hook tracking across all decoder transformer blocks.
- **Spatiotemporal tiling & self-attention**: Profiles causal 3D attention and patch-wise spatial and temporal tiling strategies.
- **Memory scaling**: Real-time background heartbeat tracking Resident Set Size (RSS), virtual memory percentage, and swap dynamics.
- **SIMD capability detection**: Identifies AVX2 (256-bit), AVX-512 (512-bit), and FMA instruction set availability across cloud runner allocations.

## Baseline Telemetry

Empirical measurements gathered on commodity cloud CI runners (4 vCPU / 16 GB RAM):

| Platform / CPU | SIMD Engine | Precision | Block Latency | Peak RSS |
| :--- | :--- | :--- | :--- | :--- |
| **AMD EPYC 7763** (4 cores) | AVX2 / FMA | Float32 | ~5.45 s / block | ~12.5 GB |
| **AMD EPYC 9V74** (4 cores) | AVX-512 / FMA | Float32 | ~4.80 s / block | ~12.6 GB |
| **Intel Xeon Platinum 8272CL** | AVX-512 / FMA | Float32 | ~5.80 s / block | ~12.5 GB |

## Setup & Prerequisites

### 1. Dependencies & Execution Backend

Install Python dependencies:
```bash
pip install -r requirements.txt
```

For native execution outside the container, clone the ComfyUI execution backend into `./ComfyUI`:
```bash
git clone --depth 1 https://github.com/comfyanonymous/ComfyUI.git ./ComfyUI
pip install -r ./ComfyUI/requirements.txt
```

### 2. Model Weights

If using the official MiniMax-H3 AutoencoderKL FP16 weights:
```bash
wget -c "https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/main/vae/minimax_h3_video_vae_fp16.safetensors"
```

## Usage

```bash
python run_benchmark.py input_tensor.safetensors \
  --dtype float32 \
  --output eval_artifact.safetensors \
  --metrics-out benchmark_metrics.json
```

### Arguments

* `latent_path`: Path to input latent tensor file (`.safetensors`).
* `--vae_path`: Path to VAE model weights (defaults to `/opt/weights.safetensors` or `./minimax_h3_video_vae_fp16.safetensors`).
* `--output`: Output artifact path (defaults to `eval_artifact.safetensors`).
* `--dtype`: Numerical precision (`float32` or `bfloat16`, default: `float32`).
* `--metrics-out`: Path to export granular JSON telemetry and layer profiles (default: `benchmark_metrics.json`).

## Evaluation Artifacts & Telemetry

### Safetensors Output Schema
When writing to `.safetensors`, the output file contains:
* `decoded`: Reconstructed tensor of shape `[T, H, W, 3]` with `uint8` values $[0, 255]$.
* Header Metadata: Tensor geometry (`shape_format`, `width`, `height`, `temporal_slices`) and execution precision.

### Telemetry JSON Export (`benchmark_metrics.json`)
Exports structured execution telemetry including:
* **Hardware topology**: CPU model, core counts, SIMD flags (AVX2, AVX-512, FMA).
* **Execution throughput**: Total decode latency, layer evaluation rate ($s/\text{block}$), and pass counts.
* **Memory dynamics**: Background heartbeat series tracking process RSS, RAM %, and swap utilization.
* **Layer profiles**: Individual timestamps and rates across all evaluated transformer blocks.

## License

MIT
