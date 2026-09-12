#!/usr/bin/env python3
"""
run_benchmark.py - 3D Vision Transformer Autoencoder CPU Inference Benchmark.

Measures:
- CPU SIMD / AVX-512 capability detection.
- Granular forward hook tracking across all 36 ViT transformer blocks.
- Resident memory scaling (RSS) and thread saturation.
- Generates native GitHub Actions step summary telemetry.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import threading
import subprocess
import psutil
import torch
from safetensors import safe_open
from diffusers.models import AutoencoderKLMiniMaxH3


class MemoryHeartbeat(threading.Thread):
    def __init__(self, interval_sec: float = 30.0):
        super().__init__(daemon=True)
        self.interval = interval_sec
        self.stop_event = threading.Event()
        self.start_time = time.time()
        self.proc = psutil.Process()

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
            print(
                f"[heartbeat] Elapsed: {elapsed:5.1f}s | Process RSS: {rss_mb:6.1f} MB | "
                f"RAM: {used_gb:4.1f}/{total_gb:.1f} GB ({vm.percent}%) | Swap: {swap_mb:5.1f} MB",
                flush=True,
            )

    def stop(self):
        self.stop_event.set()


class VAEProgressTracker:
    def __init__(self, vae, num_blocks: int = 36, expected_tiles: int = 1):
        self.num_blocks = num_blocks
        self.expected_tiles = expected_tiles
        self.total_expected_steps = num_blocks * expected_tiles
        self.current_step = 0
        self.start_time = None
        self.hooks = []
        self.proc = psutil.Process()
        self.last_rate = 0.0

        for idx, block in enumerate(vae.decoder.transformer_blocks):
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

            if self.current_step > self.total_expected_steps:
                self.total_expected_steps = max(self.total_expected_steps + self.num_blocks, int(self.current_step * 1.05))

            pct = min(100.0, (self.current_step / self.total_expected_steps) * 100.0)
            eta_sec = max(0, (self.total_expected_steps - self.current_step) * rate)
            eta_str = f"{int(eta_sec // 60)}m {int(eta_sec % 60):02d}s" if eta_sec > 0 else "finishing"
            rss_mb = self.proc.memory_info().rss / (1024 * 1024)

            print(
                f"[BENCHMARK] Step {self.current_step:4d}/{self.total_expected_steps:4d} "
                f"({pct:5.1f}%) | Block {block_idx+1:2d}/36 | "
                f"Elapsed: {int(elapsed//60)}m {int(elapsed%60):02d}s | ETA: {eta_str} "
                f"({rate:4.2f}s/block) | RSS: {rss_mb:.0f} MB",
                flush=True,
            )
        return hook_fn

    def remove(self):
        for h in self.hooks:
            h.remove()


def get_cpu_capabilities() -> dict:
    info = {"model": "Generic x86_64", "cores": os.cpu_count() or 4, "avx2": False, "avx512": False, "fma": False}
    try:
        with open("/proc/cpuinfo", "r") as f:
            for line in f:
                if line.startswith("model name") and info["model"] == "Generic x86_64":
                    info["model"] = line.split(":", 1)[1].strip()
                if line.startswith("flags"):
                    flags = line.split(":", 1)[1].strip().split()
                    info["avx2"] = "avx2" in flags
                    info["avx512"] = any(f.startswith("avx512") for f in flags)
                    info["fma"] = "fma" in flags
    except Exception:
        pass
    return info


def print_hardware_summary(cpu: dict):
    print("=" * 65, flush=True)
    print(f"[hardware] Model: {cpu['model']}", flush=True)
    print(f"[hardware] Logical Cores: {cpu['cores']} | PyTorch Threads: {torch.get_num_threads()}", flush=True)
    print(f"[hardware] AVX2: {'YES' if cpu['avx2'] else 'NO'} | AVX-512: {'YES' if cpu['avx512'] else 'NO'} | FMA: {'YES' if cpu['fma'] else 'NO'}", flush=True)
    print("=" * 65, flush=True)


def save_audio_resilient(audio: torch.Tensor, sampling_rate: int, output_wav: str) -> bool:
    try:
        from scipy.io import wavfile
        import numpy as np
        audio_np = audio.float().cpu().numpy()
        if audio_np.ndim == 2:
            audio_np = audio_np.T
        wavfile.write(output_wav, sampling_rate, (audio_np * 32767).astype(np.int16))
        return True
    except Exception:
        pass

    try:
        import wave
        import numpy as np
        audio_np = audio.float().cpu().numpy()
        if audio_np.ndim == 2:
            audio_np = audio_np.T
        int16_data = (np.clip(audio_np, -1.0, 1.0) * 32767).astype(np.int16)
        nchannels = 1 if int16_data.ndim == 1 else int16_data.shape[1]
        with wave.open(output_wav, "wb") as wf:
            wf.setnchannels(nchannels)
            wf.setsampwidth(2)
            wf.setframerate(sampling_rate)
            wf.writeframes(int16_data.tobytes())
        return True
    except Exception:
        return False


def save_benchmark_metrics(
    metrics_path: str,
    width: int,
    height: int,
    frames: int,
    fps: int,
    dtype: str,
    decode_sec: float,
    rate: float,
    peak_rss: float,
    sample_name: str = "benchmark_sample",
):
    data = {
        "model": "MiniMax-H3 AutoencoderKL",
        "sample_name": sample_name,
        "width": width,
        "height": height,
        "frames": frames,
        "fps": fps,
        "dtype": dtype,
        "decode_duration_sec": round(decode_sec, 2),
        "decode_duration_min": round(decode_sec / 60.0, 2),
        "layer_rate_sec": round(rate, 2),
        "peak_rss_mb": round(peak_rss, 1),
    }
    try:
        with open(metrics_path, "w") as f:
            json.dump(data, f, indent=2)
        print(f"[benchmark] Benchmark telemetry saved to {metrics_path}", flush=True)
    except Exception as e:
        print(f"[warning] Could not save benchmark metrics to {metrics_path}: {e}", flush=True)


def run_benchmark(
    latent_path: str,
    vae_path: str = "MiniMaxAI/MiniMax-H3",
    output_path: str = "result.mp4",
    metrics_path: str = "benchmark_metrics.json",
    dtype: str = "float32",
    tile: bool = False,
    tile_size: tuple[int, int] | None = None,
):
    cpu_info = get_cpu_capabilities()
    num_cpus = os.cpu_count() or 4
    torch.set_num_threads(num_cpus)
    print_hardware_summary(cpu_info)

    torch_dtype = torch.bfloat16 if dtype == "bfloat16" else torch.float32
    torch_device = torch.device("cpu")

    # Load input latent tensor
    with safe_open(latent_path, framework="pt", device="cpu") as f:
        meta = f.metadata() or {}
        latents = f.get_tensor("latents")
        audio = f.get_tensor("audio") if "audio" in f.keys() else None

    height = int(meta.get("height", latents.shape[-2] * 16))
    width = int(meta.get("width", latents.shape[-1] * 16))
    fps = int(meta.get("fps", 24))
    sampling_rate = int(meta.get("sampling_rate", 24000))
    frames = int(meta.get("num_frames", (latents.shape[2] - 1) * 4 + 1 if latents.shape[2] > 1 else 1))

    print(f"[benchmark] Input Tensor Shape: {latents.shape} | Precision: {torch_dtype}", flush=True)
    print(f"[benchmark] Output Volume: {width}x{height} | Frames: {frames} @ {fps} fps", flush=True)

    # Load VAE weights
    print(f"[benchmark] Loading ViT Autoencoder weights from {vae_path} ...", flush=True)
    t0 = time.time()
    vae = AutoencoderKLMiniMaxH3.from_pretrained(
        vae_path,
        subfolder="vae" if not os.path.isdir(vae_path) else None,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
    ).to(torch_device, dtype=torch_dtype)
    vae.eval()
    print(f"[benchmark] Model loaded in {time.time() - t0:.2f}s", flush=True)

    # Configure tiling
    expected_tiles = 1
    if not tile:
        print("[benchmark] Spatial tiling DISABLED (monolithic evaluation)", flush=True)
        vae.disable_tiling()
    elif tile_size is not None:
        th, tw = tile_size
        print(f"[benchmark] Spatial tiling ENABLED ({tw}x{th})", flush=True)
        vae.enable_tiling(tile_sample_min_height=th, tile_sample_min_width=tw)
        ny = max(1, (height + th - 1) // th)
        nx = max(1, (width + tw - 1) // tw)
        expected_tiles = ny * nx
    else:
        vae.enable_tiling()
        ny = max(1, (height + 256 - 64 - 1) // (256 - 64))
        nx = max(1, (width + 256 - 64 - 1) // (256 - 64))
        expected_tiles = ny * nx

    num_latent_t = latents.shape[2]
    expected_temporal_chunks = 1 if num_latent_t <= 5 else (2 if num_latent_t <= 22 else max(1, round((num_latent_t - 2) / 5.0) + 1))
    expected_total_tiles = expected_tiles * expected_temporal_chunks

    tracker = VAEProgressTracker(vae, num_blocks=len(vae.decoder.transformer_blocks), expected_tiles=expected_total_tiles)
    heartbeat = MemoryHeartbeat(interval_sec=30.0)
    heartbeat.start()

    print(f"[benchmark] Starting evaluation: 36 blocks x {expected_total_tiles} passes = ~{tracker.total_expected_steps} layer evaluations", flush=True)
    t_decode = time.time()

    try:
        with torch.inference_mode():
            latents_mean = torch.tensor(vae.config.latents_mean, device=torch_device, dtype=torch_dtype).view(1, -1, 1, 1, 1)
            latents_std = torch.tensor(vae.config.latents_std, device=torch_device, dtype=torch_dtype).view(1, -1, 1, 1, 1)
            latents_norm = latents.to(device=torch_device, dtype=torch_dtype) * latents_std + latents_mean

            video = vae.decode(latents_norm, return_dict=False)[0]

            pixel_mean = torch.tensor((0.485, 0.456, 0.406), device=torch_device, dtype=torch.float32).view(1, -1, 1, 1, 1)
            pixel_std = torch.tensor((0.229, 0.224, 0.225), device=torch_device, dtype=torch.float32).view(1, -1, 1, 1, 1)
            video = (video.float() * pixel_std + pixel_mean).clamp(0, 1)
    finally:
        tracker.remove()
        heartbeat.stop()

    decode_duration = time.time() - t_decode
    proc = psutil.Process()
    peak_rss_mb = proc.memory_info().rss / (1024 * 1024)

    print("=" * 65, flush=True)
    print(f"[benchmark] Decoded tensor shape: {video.shape} in {decode_duration:.2f}s ({decode_duration/60:.2f} min)", flush=True)
    print(f"[benchmark] Layer rate: {tracker.last_rate:.2f}s/block | Peak RSS: {peak_rss_mb:.1f} MB", flush=True)

    # Encode container
    video_np = (video[0].permute(1, 2, 3, 0).float() * 255.0).clamp(0, 255).to(torch.uint8).cpu().numpy()
    video_bytes = video_np.tobytes()

    temp_wav = "/tmp/temp_audio.wav"
    has_audio = False
    if audio is not None:
        has_audio = save_audio_resilient(audio, sampling_rate, temp_wav)

    os.makedirs(os.path.dirname(os.path.abspath(output_path)) or ".", exist_ok=True)
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error", "-hide_banner",
        "-f", "rawvideo", "-vcodec", "rawvideo",
        "-s", f"{width}x{height}", "-pix_fmt", "rgb24",
        "-r", str(fps), "-i", "-",
    ]
    if has_audio and os.path.exists(temp_wav):
        cmd.extend(["-i", temp_wav, "-c:a", "aac", "-b:a", "192k"])
    cmd.extend([
        "-c:v", "libx264", "-pix_fmt", "yuv420p",
        "-crf", "18", "-preset", "veryfast", "-shortest",
        "-f", "mp4",
        output_path,
    ])
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    _, err = p.communicate(input=video_bytes)
    if p.returncode != 0:
        print(f"[ffmpeg-error] {err.decode('utf-8', errors='ignore')}", flush=True)

    print(f"[benchmark] Output artifact encoded to {output_path}", flush=True)
    print("=" * 65, flush=True)

    save_benchmark_metrics(
        metrics_path=metrics_path,
        width=width,
        height=height,
        frames=frames,
        fps=fps,
        dtype=dtype,
        decode_sec=decode_duration,
        rate=tracker.last_rate,
        peak_rss=peak_rss_mb,
        sample_name=os.path.basename(latent_path),
    )
    return output_path


def main():
    parser = argparse.ArgumentParser(description="PyTorch ViT Video VAE CPU Benchmark")
    parser.add_argument("latent_path", help="Path to input tensor safetensors file")
    parser.add_argument("--vae_path", default="MiniMaxAI/MiniMax-H3", help="Model repository")
    parser.add_argument("--output", "-o", default="eval_artifact.bin", help="Output artifact path")
    parser.add_argument("--metrics-out", default="benchmark_metrics.json", help="Path to export JSON benchmark metrics")
    parser.add_argument("--dtype", default="float32", choices=["float32", "bfloat16"], help="Precision")
    parser.add_argument("--no-tile", action="store_true", help="Disable spatial tiling")
    parser.add_argument("--tile-size", nargs=2, type=int, default=None, metavar=("HEIGHT", "WIDTH"), help="Tile size")

    args = parser.parse_args()
    tile_size = tuple(args.tile_size) if args.tile_size is not None else None
    run_benchmark(
        latent_path=args.latent_path,
        vae_path=args.vae_path,
        output_path=args.output,
        metrics_path=args.metrics_out,
        dtype=args.dtype,
        tile=not args.no_tile,
        tile_size=tile_size,
    )


if __name__ == "__main__":
    main()
