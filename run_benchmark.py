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

# 1. Resolve model execution backend
backend_dir = "/opt/ComfyUI" if os.path.exists("/opt/ComfyUI") else os.path.abspath("./ComfyUI")
sys.path.insert(0, backend_dir)

# Default model path can also point to /opt if baked into the container
default_vae = "/opt/weights.safetensors" if os.path.exists("/opt/weights.safetensors") else "./weights.safetensors"

# 2. Configure headless CPU runtime options (parse an empty arg list, ignoring sys.argv, and force CPU mode)
import comfy.options
comfy.options.args_parsing = False

import comfy.cli_args
comfy.cli_args.args.cpu = True
comfy.cli_args.args.cpu_vae = True

# 3. Now import model_management and sd safely on CPU
import comfy.model_management
comfy.model_management.cpu_state = comfy.model_management.CPUState.CPU

# Guard against torchaudio CUDA linkage errors on CPU-only containers
try:
    import torchaudio
except (ImportError, OSError, Exception):
    import types
    import importlib.machinery
    mod = types.ModuleType("torchaudio")
    mod.__spec__ = importlib.machinery.ModuleSpec("torchaudio", None)
    mod.__version__ = "2.2.0"
    mod._extension = types.ModuleType("torchaudio._extension")
    mod._extension._IS_TORCHAUDIO_EXT_AVAILABLE = False
    sys.modules["torchaudio"] = mod
    sys.modules["torchaudio._extension"] = mod._extension

import comfy.utils
import comfy.sd


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
            "shape_format": "C x T x H x W",
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
    vae_path: str = "./minimax_h3_video_vae_fp16.safetensors",
    output_path: str = "eval_artifact.bin",
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
    # Load input latent tensor directly via safe_open (supports .bin or .safetensors)
    with safe_open(latent_path, framework="pt", device="cpu") as f:
        meta = f.metadata() or {}
        latents = f.get_tensor("latents")

    if latents.ndim == 4:
        latents = latents.unsqueeze(2)

    height = int(meta.get("height", latents.shape[-2] * 16))
    width = int(meta.get("width", latents.shape[-1] * 16))
    target_slices = int(meta.get("num_frames", (latents.shape[2] - 1) * 4 + 1 if latents.shape[2] > 1 else 1))

    print(f"[benchmark] Input Tensor Shape: {latents.shape} | Precision: {torch_dtype}", flush=True)
    print(f"[benchmark] Output Volume: {width}x{height} | Slices: {target_slices}", flush=True)

    # Load VAE weights
    print(f"[benchmark] Loading ViT Autoencoder weights from {vae_path} ...", flush=True)
    t0 = time.time()
    vae_sd = comfy.utils.load_torch_file(vae_path)
    vae = comfy.sd.VAE(sd=vae_sd)
    print(f"[benchmark] Model loaded in {time.time() - t0:.2f}s", flush=True)

    # Find the 36 transformer blocks in ComfyUI's model
    blocks = []
    if hasattr(vae, "first_stage_model"):
        m = vae.first_stage_model
        if hasattr(m, "decoder") and hasattr(m.decoder, "transformer_blocks"):
            blocks = list(m.decoder.transformer_blocks)
        elif hasattr(m, "transformer_blocks"):
            blocks = list(m.transformer_blocks)

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
            decoded = vae.decode(latents.to(torch_device))
    finally:
        block_evaluations = tracker.remove()
        heartbeat_samples = heartbeat.stop()

    decode_duration = time.time() - t_decode
    proc = psutil.Process()
    peak_rss_mb = proc.memory_info().rss / (1024 * 1024)

    print("=" * 65, flush=True)
    print(f"[benchmark] Decode finished in {decode_duration:.2f}s ({decode_duration/60:.2f} min)", flush=True)
    print(f"[benchmark] Layer rate: {tracker.last_rate:.2f}s/block | Peak RSS: {peak_rss_mb:.1f} MB", flush=True)

    # Convert decoded output to numpy [T, H, W, C] (uint8)
    if isinstance(decoded, torch.Tensor):
        frames_np = decoded.detach().cpu().float().numpy()
    else:
        frames_np = np.array(decoded)

    if frames_np.ndim == 5:
        frames_np = frames_np[0]

    if frames_np.ndim == 4:
        if frames_np.shape[-1] in (1, 3, 4):
            pass
        elif frames_np.shape[1] in (1, 3, 4):
            frames_np = np.transpose(frames_np, (0, 2, 3, 1))
        elif frames_np.shape[0] in (1, 3, 4):
            frames_np = np.transpose(frames_np, (1, 2, 3, 0))

    if frames_np.max() <= 1.0:
        output_np = (frames_np * 255.0).clip(0, 255).astype(np.uint8)
    else:
        output_np = frames_np.clip(0, 255).astype(np.uint8)

    output_bytes = output_np.tobytes()

    os.makedirs(os.path.dirname(os.path.abspath(output_path)) or ".", exist_ok=True)
    with open(output_path, "wb") as f_out:
        f_out.write(output_bytes)

    # Free high-memory array allocations before JSON serialization to eliminate OOM risk
    del output_bytes, output_np, frames_np, decoded
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
    parser.add_argument("--vae_path", default="./minimax_h3_video_vae_fp16.safetensors", help="Model weights path")
    parser.add_argument("--output", "-o", default="eval_artifact.bin", help="Output artifact path")
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
