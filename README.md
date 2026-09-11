# vit-vae-cpu-benchmarks

A benchmarking suite for evaluating CPU inference latency, memory scaling (RSS), and SIMD vectorization (AVX2 / AVX-512) for 3D Vision Transformer Autoencoders (Video VAEs).

## Overview

This benchmark profiles the CPU execution characteristics of multi-layer 3D Vision Transformer decoders on standard x86_64 cloud hypervisors:
- Layer-wise forward evaluation rate ($s/\text{block}$) across 36 ViT transformer blocks.
- Spatio-temporal self-attention scaling under full float32 precision.
- Resident set size (RSS) memory footprint and multi-threaded scaling.

## Installation

```bash
pip install -r requirements.txt
```

## Usage

```bash
python run_benchmark.py input_tensor.safetensors --dtype float32 --output result.mp4
```

## License

MIT
