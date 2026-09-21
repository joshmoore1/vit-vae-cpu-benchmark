#!/usr/bin/env python3
"""
run_benchmark.py - Spatiotemporal Vision Transformer Autoencoder CPU Inference Benchmark.

Measures:
- CPU SIMD capability detection (AVX2 / AVX-512 / FMA).
- Granular forward hook tracking across all 36 ViT transformer blocks.
- Resident memory scaling (RSS) and thread saturation.
- Generates native GitHub Actions step summary telemetry.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
import threading
import psutil
import torch
import numpy as np

# Resolve execution backend
for candidate in ("/opt/ComfyUI", os.path.abspath("./ComfyUI"), os.path.expanduser("~/ComfyUI")):
    if os.path.exists(candidate):
        if candidate not in sys.path:
            sys.path.insert(0, candidate)
        break

import comfy.cli_args
comfy.cli_args.args.cpu = True
import comfy.ops
from comfy.ldm.minimax.vae import MiniMaxH3VideoVAE

default_vae = os.getenv("MODEL_CHECKPOINT") or (
    "/opt/weights.safetensors" if os.path.exists("/opt/weights.safetensors") else "./minimax_h3_video_vae_fp16.safetensors"
)


class MemoryHeartbeat(threading.Thread):
    def __init__(self, interval_sec: float = 30.0):
        super().__init__(daemon=True)
        self.interval = interval_sec
        self.stop_event = threading.Event()
        self.start_time = time.time()
        self.proc = psutil.Process()
        self.samples = []

    def run(self):
        while not self.stop_event.is_set():
            self.stop_event.wait(self.interval)
            if self.stop_event.is_set():
                break
            elapsed = time.time() - self.start_time
            vm = psutil.virtual_memory()
            swap = psutil.swap_memory()
            rss_mb = self.proc.memory_info().rss / (1024 * 1024)
            used_gb = vm.used / (1024**3)
            total_gb = vm.total / (1024**3)
            swap_mb = swap.used / (1024 * 1024)
            self.samples.append({
                "elapsed_sec": elapsed,
                "process_rss_mb": rss_mb,
                "ram_used_gb": used_gb,
                "ram_total_gb": total_gb,
                "ram_percent": vm.percent,
                "swap_used_mb": swap_mb,
            })

    def stop(self):
        self.stop_event.set()
        return self.samples


class VAEProgressTracker:
    def __init__(self, transformer_blocks, expected_passes: int = 1):
        # Derive block count directly from the passed module list
        self.num_blocks = len(transformer_blocks) or 36
        self.expected_passes = max(1, expected_passes)
        self.total_expected_steps = self.num_blocks * self.expected_passes
        self.current_step = 0
        self.start_time = None
        self.hooks = []
        self.proc = psutil.Process()
        self.last_rate = 0.0
        self.evaluations = []

        for idx, block in enumerate(transformer_blocks):
            hook = block.register_forward_hook(self._make_hook(idx))
            self.hooks.append(hook)

    def _make_hook(self, block_idx: int):
        def hook_fn(module, input, output):
            if self.start_time is None:
                self.start_time = time.time()
            self.current_step += 1
            elapsed = time.time() - self.start_time
            rate = elapsed / self.current_step
            self.last_rate = rate

            rss_mb = self.proc.memory_info().rss / (1024 * 1024)
            current_pass = min(self.expected_passes, (self.current_step + self.num_blocks - 1) // self.num_blocks)

            self.evaluations.append({
                "step": self.current_step,
                "pass": current_pass,
                "block": block_idx + 1,
                "elapsed_sec": elapsed,
                "layer_rate_sec": rate,
                "rss_mb": rss_mb,
            })

            if (block_idx == self.num_blocks - 1) or (self.current_step == self.total_expected_steps):
                pct = min(100.0, (self.current_step / self.total_expected_steps) * 100.0)
                remaining_steps = max(0, self.total_expected_steps - self.current_step)
                eta_sec = remaining_steps * rate
                eta_str = f"{int(eta_sec // 60)}m {int(eta_sec % 60):02d}s" if remaining_steps > 0 else "finishing"

                print(
                    f"[BENCHMARK] Pass {current_pass:2d}/{self.expected_passes:2d} "
                    f"({pct:5.1f}%) | Step {self.current_step:4d}/{self.total_expected_steps:4d} | "
                    f"Elapsed: {int(elapsed//60)}m {int(elapsed%60):02d}s | ETA: {eta_str} "
                    f"({rate:4.2f}s/block) | RSS: {rss_mb:.0f} MB",
                    flush=True,
                )
        return hook_fn

    def remove(self):
        for h in self.hooks:
            h.remove()
        return self.evaluations


def get_cpu_capabilities() -> dict:
    info = {"model": "Generic x86_64", "cores": os.cpu_count() or 4, "avx2": False, "avx512": False, "fma": False}
    try:
        with open("/proc/cpuinfo", "r") as f:
            for line in f:
                if ":" not in line:
                    continue
                key, val = [x.strip() for x in line.split(":", 1)]
                if key == "model name" and info["model"] == "Generic x86_64":
                    info["model"] = val
                elif key == "flags":
                    flags = val.split()
                    info["avx2"] = "avx2" in flags
                    info["avx512"] = any(f.startswith("avx512") for f in flags)
                    info["fma"] = "fma" in flags
    except Exception as e:
        print(f"[warning] CPU info parsing encountered an issue: {e}", flush=True)
    return info


def print_hardware_summary(cpu: dict):
    print("=" * 65, flush=True)
    print(f"[hardware] Model: {cpu['model']}", flush=True)
    print(f"[hardware] Logical Cores: {cpu['cores']} | PyTorch Threads: {torch.get_num_threads()}", flush=True)
    print(f"[hardware] AVX2: {'YES' if cpu['avx2'] else 'NO'} | AVX-512: {'YES' if cpu['avx512'] else 'NO'} | FMA: {'YES' if cpu['fma'] else 'NO'}", flush=True)
    print("=" * 65, flush=True)


def save_benchmark_metrics(
    metrics_path: str,
    width: int,
    height: int,
    temporal_slices: int,
    dtype: str,
    decode_sec: float,
    rate: float,
    peak_rss: float,
    cpu_info: dict,
    total_steps: int,
    total_passes: int,
    heartbeats: list[dict],
    block_evaluations: list[dict],
    sample_name: str = "benchmark_sample",
):
    data = {
        "benchmark_target": "Spatiotemporal AutoencoderKL",
        "sample_id": sample_name,
        "tensor_geometry": {
            "channels": 3,
            "temporal_slices": temporal_slices,
            "height": height,
            "width": width,
            "shape_format": "T x H x W x C",
            "dtype": "uint8",
        },
        "execution": {
            "precision": dtype,
            "total_decode_sec": decode_sec,
            "total_decode_min": decode_sec / 60.0,
            "total_steps": total_steps,
            "total_passes": total_passes,
            "average_layer_rate_sec": rate,
            "peak_rss_mb": peak_rss,
        },
        "hardware": {
            "cpu_model": cpu_info.get("model", "Unknown"),
            "logical_cores": cpu_info.get("cores", 4),
            "pytorch_threads": torch.get_num_threads(),
            "avx2": cpu_info.get("avx2", False),
            "avx512": cpu_info.get("avx512", False),
            "fma": cpu_info.get("fma", False),
        },
        "heartbeats": heartbeats,
        "block_evaluations": block_evaluations,
    }
    try:
        with open(metrics_path, "w") as f:
            json.dump(data, f, indent=2)
        print(f"[benchmark] Benchmark telemetry saved to {metrics_path}", flush=True)
    except Exception as e:
        print(f"[warning] Could not save benchmark metrics to {metrics_path}: {e}", flush=True)


def calculate_expected_passes(width: int, height: int, num_latent_t: int, tile_size: int = 256, overlap: int = 64) -> int:
    def get_axis_tiles(dim: int) -> int:
        if dim <= tile_size:
            return 1
        if dim <= 768:
            return 2
        stride = tile_size - overlap
        return max(1, (dim - overlap + stride - 1) // stride)

    tiles_y = get_axis_tiles(height)
    tiles_x = get_axis_tiles(width)
    spatial_tiles = max(1, tiles_y * tiles_x)

    if num_latent_t <= 5:
        return spatial_tiles

    temporal_passes = max(1, (num_latent_t - 2) // 5 + 1)
    return spatial_tiles * temporal_passes


def run_benchmark(
    latent_path: str,
    vae_path: str = default_vae,
    output_path: str = "eval_artifact.safetensors",
    metrics_path: str = "benchmark_metrics.json",
    dtype: str = "float32",
):
    cpu_info = get_cpu_capabilities()
    num_cpus = os.cpu_count() or 4
    torch.set_num_threads(num_cpus)
    print_hardware_summary(cpu_info)

    torch_dtype = torch.bfloat16 if dtype == "bfloat16" else torch.float32
    torch_device = torch.device("cpu")

    from safetensors import safe_open
    # Load input latent tensor directly via safe_open
    with safe_open(latent_path, framework="pt", device="cpu") as f:
        meta = f.metadata() or {}
        latents = f.get_tensor("latents")

    if latents.ndim == 4:
        latents = latents.unsqueeze(2)

    height = int(meta.get("height", latents.shape[-2] * 16))
    width = int(meta.get("width", latents.shape[-1] * 16))
    default_slices = (latents.shape[2] - 1) * 4 + 1 if latents.shape[2] > 1 else 1
    target_slices = int(meta.get("temporal_slices", default_slices))

    print(f"[benchmark] Input Tensor Shape: {latents.shape} | Precision: {torch_dtype}", flush=True)
    print(f"[benchmark] Output Geometry: {width}x{height} | Slices: {target_slices}", flush=True)

    # Instantiate ViT Autoencoder and load weights
    print(f"[benchmark] Loading ViT Autoencoder weights from {vae_path} ...", flush=True)
    t0 = time.time()
    from safetensors.torch import load_file
    model = MiniMaxH3VideoVAE(operations=comfy.ops.disable_weight_init)
    vae_sd = load_file(vae_path, device="cpu")
    model.load_state_dict(vae_sd, strict=False)
    model.to(dtype=torch_dtype, device=torch_device)
    model.eval()
    del vae_sd
    print(f"[benchmark] Model loaded in {time.time() - t0:.2f}s", flush=True)

    latents = latents.to(dtype=torch_dtype, device=torch_device)

    blocks = list(model.decoder.transformer_blocks)
    if not blocks:
        print("[warning] No transformer blocks identified for forward-hook tracking", flush=True)

    num_latent_t = latents.shape[2]
    expected_passes = calculate_expected_passes(width=width, height=height, num_latent_t=num_latent_t)

    # Tracker derives block count directly from blocks
    tracker = VAEProgressTracker(blocks, expected_passes=expected_passes)
    heartbeat = MemoryHeartbeat(interval_sec=30.0)
    heartbeat.start()

    print(
        f"[benchmark] Starting evaluation: {tracker.num_blocks} blocks x {tracker.expected_passes} passes = "
        f"~{tracker.total_expected_steps} layer evaluations",
        flush=True,
    )
    t_decode = time.time()

    try:
        with torch.inference_mode():
            decoded = model.decode(latents.to(torch_device))
    finally:
        block_evaluations = tracker.remove()
        heartbeat_samples = heartbeat.stop()

    decode_duration = time.time() - t_decode
    proc = psutil.Process()
    peak_rss_mb = proc.memory_info().rss / (1024 * 1024)

    print("=" * 65, flush=True)
    print(f"[benchmark] Decode finished in {decode_duration:.2f}s ({decode_duration/60:.2f} min)", flush=True)
    print(f"[benchmark] Layer rate: {tracker.last_rate:.2f}s/block | Peak RSS: {peak_rss_mb:.1f} MB", flush=True)

    # Convert decoded output float [B, 3, T, H, W] in [0, 1] to uint8 [T, H, W, 3]
    if isinstance(decoded, torch.Tensor):
        if decoded.ndim == 5:
            tensor_np = decoded[0].permute(1, 2, 3, 0).cpu().float().numpy()
        elif decoded.ndim == 4:
            tensor_np = decoded.permute(1, 2, 3, 0).cpu().float().numpy()
        else:
            tensor_np = decoded.cpu().float().numpy()
    else:
        tensor_np = np.array(decoded)

    if tensor_np.max() <= 1.0:
        output_np = (tensor_np * 255.0).clip(0, 255).astype(np.uint8)
    else:
        output_np = tensor_np.clip(0, 255).astype(np.uint8)

    os.makedirs(os.path.dirname(os.path.abspath(output_path)) or ".", exist_ok=True)
    from safetensors.torch import save_file
    output_tensor = torch.from_numpy(np.ascontiguousarray(output_np)).contiguous()
    save_file(
        {"decoded": output_tensor},
        output_path,
        metadata={
            "benchmark_target": "Spatiotemporal AutoencoderKL",
            "shape_format": "T x H x W x C",
            "precision": dtype,
            "width": str(width),
            "height": str(height),
            "temporal_slices": str(target_slices),
        },
    )


    # Free high-memory array allocations before JSON serialization to eliminate OOM risk
    del output_np, tensor_np, decoded
    gc.collect()

    print(f"[benchmark] Output evaluation artifact saved to {output_path}", flush=True)
    print("=" * 65, flush=True)

    save_benchmark_metrics(
        metrics_path=metrics_path,
        width=width,
        height=height,
        temporal_slices=target_slices,
        dtype=dtype,
        decode_sec=decode_duration,
        rate=tracker.last_rate,
        peak_rss=peak_rss_mb,
        cpu_info=cpu_info,
        total_steps=tracker.total_expected_steps,
        total_passes=tracker.expected_passes,
        heartbeats=heartbeat_samples,
        block_evaluations=block_evaluations,
        sample_name=os.path.basename(latent_path),
    )
    return output_path


def main():
    parser = argparse.ArgumentParser(description="PyTorch ViT VAE CPU Inference Benchmark")
    parser.add_argument("latent_path", help="Path to input tensor safetensors file")
    parser.add_argument("--vae_path", default=default_vae, help="Model weights path")
    parser.add_argument("--output", "-o", default="eval_artifact.safetensors", help="Output artifact path")
    parser.add_argument("--metrics-out", default="benchmark_metrics.json", help="Path to export JSON benchmark metrics")
    parser.add_argument("--dtype", default="float32", choices=["float32", "bfloat16"], help="Precision")

    args, _ = parser.parse_known_args()

    run_benchmark(
        latent_path=args.latent_path,
        vae_path=args.vae_path,
        output_path=args.output,
        metrics_path=args.metrics_out,
        dtype=args.dtype,
    )


if __name__ == "__main__":
    main()
