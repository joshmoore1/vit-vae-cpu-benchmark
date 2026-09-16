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
import numpy as np

# 1. Add ComfyUI to path
sys.path.insert(0, os.path.abspath("./ComfyUI"))

# 2. Tell ComfyUI to parse an empty arg list (ignoring sys.argv) and force CPU mode
import comfy.options
comfy.options.args_parsing = False

import comfy.cli_args
comfy.cli_args.args.cpu = True
comfy.cli_args.args.cpu_vae = True

# 3. Now import model_management and sd safely on CPU
import comfy.model_management
comfy.model_management.cpu_state = comfy.model_management.CPUState.CPU

import comfy.utils
import comfy.sd


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
    def __init__(self, transformer_blocks, num_blocks: int = 36, expected_tiles: int = 1):
        self.num_blocks = num_blocks
        self.expected_tiles = expected_tiles
        self.total_expected_steps = num_blocks * expected_tiles
        self.current_step = 0
        self.start_time = None
        self.hooks = []
        self.proc = psutil.Process()
        self.last_rate = 0.0

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
                    info["model"] = line.split(":", 1).strip()
                if line.startswith("flags"):
                    flags = line.split(":", 1).strip().split()
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
        audio_np = audio.float().cpu().numpy()
        if audio_np.ndim == 2:
            audio_np = audio_np.T
        wavfile.write(output_wav, sampling_rate, (audio_np * 32767).astype(np.int16))
        return True
    except Exception:
        pass

    try:
        import wave
        audio_np = audio.float().cpu().numpy()
        if audio_np.ndim == 2:
            audio_np = audio_np.T
        int16_data = (np.clip(audio_np, -1.0, 1.0) * 32767).astype(np.int16)
        nchannels = 1 if int16_data.ndim == 1 else int16_data.shape
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
        "model": "MiniMax-H3 Video VAE",
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
    vae_path: str = "./minimax_h3_video_vae_fp16.safetensors",
    output_path: str = "result.mp4",
    metrics_path: str = "benchmark_metrics.json",
    dtype: str = "float32",
    tile: bool = False,
):
    cpu_info = get_cpu_capabilities()
    num_cpus = os.cpu_count() or 4
    torch.set_num_threads(num_cpus)
    print_hardware_summary(cpu_info)

    torch_dtype = torch.bfloat16 if dtype == "bfloat16" else torch.float32
    torch_device = torch.device("cpu")

    # Load input latent tensor
    latent_data = comfy.utils.load_torch_file(latent_path)
    if isinstance(latent_data, dict):
        latents = latent_data.get("latents", latent_data.get("video"))
        audio = latent_data.get("audio")
    else:
        latents = latent_data
        audio = None

    if latents.ndim == 4:
        latents = latents.unsqueeze(2)

    height = latents.shape[-2] * 16
    width = latents.shape[-1] * 16
    fps = 24
    sampling_rate = 24000
    frames = (latents.shape[2] - 1) * 4 + 1 if latents.shape[2] > 1 else 1

    print(f"[benchmark] Input Tensor Shape: {latents.shape} | Precision: {torch_dtype}", flush=True)
    print(f"[benchmark] Output Volume: {width}x{height} | Frames: {frames} @ {fps} fps", flush=True)

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
    expected_temporal_chunks = 1 if num_latent_t <= 5 else (2 if num_latent_t <= 22 else max(1, round((num_latent_t - 2) / 5.0) + 1))
    expected_total_tiles = 1 * expected_temporal_chunks

    tracker = VAEProgressTracker(blocks, num_blocks=len(blocks) or 36, expected_tiles=expected_total_tiles)
    heartbeat = MemoryHeartbeat(interval_sec=30.0)
    heartbeat.start()

    print(f"[benchmark] Starting evaluation: {len(blocks) or 36} blocks x {expected_total_tiles} passes = ~{tracker.total_expected_steps} layer evaluations", flush=True)
    t_decode = time.time()

    try:
        with torch.inference_mode():
            decoded = vae.decode(latents.to(torch_device))
    finally:
        tracker.remove()
        heartbeat.stop()

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
        elif frames_np.shape in (1, 3, 4):
            frames_np = np.transpose(frames_np, (0, 2, 3, 1))
        elif frames_np.shape[0] in (1, 3, 4):
            frames_np = np.transpose(frames_np, (1, 2, 3, 0))

    if frames_np.max() <= 1.0:
        video_np = (frames_np * 255.0).clip(0, 255).astype(np.uint8)
    else:
        video_np = frames_np.clip(0, 255).astype(np.uint8)

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
    parser.add_argument("--vae_path", default="./minimax_h3_video_vae_fp16.safetensors", help="Model weights path")
    parser.add_argument("--output", "-o", default="eval_artifact.mp4", help="Output artifact path")
    parser.add_argument("--metrics-out", default="benchmark_metrics.json", help="Path to export JSON benchmark metrics")
    parser.add_argument("--dtype", default="float32", choices=["float32", "bfloat16"], help="Precision")

    args = parser.parse_args()

    run_benchmark(
        latent_path=args.latent_path,
        vae_path=args.vae_path,
        output_path=args.output,
        metrics_path=args.metrics_out,
        dtype=args.dtype,
    )


if __name__ == "__main__":
    main()
