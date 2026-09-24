#!/usr/bin/env python3
"""Decode MiniMax H3 packed latents with the ComfyUI H3 VAEs.

The released audio checkpoint stores weight-normalized convolutions as
``weight_g``/``weight_v``.  ComfyUI's H3 audio implementation intentionally
uses plain convolutions, so this command folds the parametrization in memory
and refuses to continue when any real checkpoint key is missing.

The command emits JSONL progress records on stdout.  Diagnostics from
PyTorch/ComfyUI are left on stderr so an API worker can consume progress
without parsing human logs.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import subprocess
import sys
import wave
from pathlib import Path
from typing import Any, Iterable


def event(stage: str, current: int, total: int, message: str, **extra: Any) -> None:
    payload = {"event": "progress", "stage": stage, "current": current, "total": total, "message": message}
    payload.update(extra)
    print(json.dumps(payload, ensure_ascii=False), flush=True)


def result(**values: Any) -> None:
    payload = {"event": "result"}
    payload.update(values)
    print(json.dumps(payload, ensure_ascii=False), flush=True)


def fold_weight_norm(state: dict[str, Any], *, torch: Any) -> dict[str, Any]:
    """Fold all ``weight_g``/``weight_v`` pairs (PyTorch weight_norm dim=0)."""
    out: dict[str, Any] = {}
    handled: set[str] = set()
    for key, value in state.items():
        if key in handled or not key.endswith(".weight_g"):
            continue
        base = key[: -len("_g")]
        vkey = base + "_v"
        if vkey not in state:
            raise RuntimeError(f"audio checkpoint has {key} but no matching {vkey}")
        v = state[vkey]
        # weight_norm(..., dim=0): g has [out,1,...], norm covers all other
        # axes.  This works for Conv1d and ConvTranspose1d alike.
        dims = tuple(range(1, v.ndim))
        norm = torch.linalg.vector_norm(v, dim=dims, keepdim=True)
        out[base] = value * v / norm.clamp_min(torch.finfo(v.dtype).eps)
        handled.update({key, vkey})
    for key, value in state.items():
        if key not in handled:
            if key.endswith(".weight_v") or key.endswith(".weight_g"):
                raise RuntimeError(f"unpaired weight-normalization tensor: {key}")
            out[key] = value
    return out


def strict_keys(module: Any, state: dict[str, Any], *, allowed_missing: Iterable[str] = ()) -> None:
    expected = set(module.state_dict())
    actual = set(state)
    missing = expected - actual
    unexpected = actual - expected
    allowed = set(allowed_missing)
    real_missing = missing - allowed
    if real_missing or unexpected:
        raise RuntimeError(
            "VAE checkpoint key mismatch: "
            f"missing={sorted(real_missing)[:12]} unexpected={sorted(unexpected)[:12]}"
        )


def load_safetensors(path: Path, *, torch: Any) -> dict[str, Any]:
    from safetensors.torch import load_file

    if not path.is_file():
        raise FileNotFoundError(path)
    return load_file(str(path), device="cpu")


def add_latent_stats(state: dict[str, Any], checkpoint: Path, *, torch: Any) -> None:
    """Load the published per-channel normalization constants from config.json."""
    config_path = checkpoint.parent / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8")) if config_path.is_file() else {}
    # The official video weights live in ``video_vae/source`` while the
    # wrapper-level latent statistics live one directory above it.
    if not isinstance(config.get("latents_mean"), list) and (checkpoint.parent.parent / "config.json").is_file():
        config_path = checkpoint.parent.parent / "config.json"
        config = json.loads(config_path.read_text(encoding="utf-8"))
    if not config_path.is_file():
        raise FileNotFoundError(f"latent statistics config is required: {config_path}")
    for name in ("latents_mean", "latents_std"):
        values = config.get(name)
        if not isinstance(values, list) or not values:
            raise RuntimeError(f"{config_path} has no valid {name}")
        state[name] = torch.tensor(values, dtype=torch.float32)


def write_wav(path: Path, audio: Any, *, torch: Any) -> None:
    import numpy as np

    # Comfy returns [B, stereo, samples]. Keep the first batch and clip only
    # at the PCM boundary; the floating-point tensor remains available above.
    samples = audio[0].detach().float().cpu().clamp(-1, 1).numpy()
    pcm = (np.clip(samples.T, -1.0, 1.0) * 32767.0).round().astype(np.int16)
    with wave.open(str(path), "wb") as f:
        f.setnchannels(2)
        f.setsampwidth(2)
        f.setframerate(32000)
        f.writeframes(pcm.tobytes())


def ffmpeg_path() -> str:
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return "ffmpeg"


def encode_video(
    frames: Any,
    audio_wav: Path,
    output: Path,
    *,
    torch: Any,
    lossless: bool,
    fps: int = 24,
    crf: int = 17,
    rgb_npy: Path | None = None,
) -> None:
    """Pipe [1,T,H,W,3] float frames to ffmpeg without a giant conversion copy."""
    import numpy as np

    if frames.ndim != 5 or frames.shape[0] != 1 or frames.shape[-1] != 3:
        raise ValueError(f"expected video [1,T,H,W,3], got {tuple(frames.shape)}")
    _, count, height, width, _ = frames.shape
    output.parent.mkdir(parents=True, exist_ok=True)
    ffmpeg = ffmpeg_path()
    vcodec = "libx264rgb" if lossless else "libx264"
    pix_out = "rgb24" if lossless else "yuv444p"
    quality = ["-crf", "0"] if lossless else ["-crf", str(crf)]
    cmd = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s:v",
        f"{width}x{height}",
        "-r",
        str(fps),
        "-i",
        "-",
        "-i",
        str(audio_wav),
        "-map",
        "0:v:0",
        "-map",
        "1:a:0",
        "-c:v",
        vcodec,
        *quality,
        "-preset",
        "medium" if not lossless else "slow",
        "-pix_fmt",
        pix_out,
        "-c:a",
        "aac",
        "-b:a",
        "320k",
        "-shortest",
        str(output),
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    assert proc.stdin is not None
    try:
        if rgb_npy is not None:
            rgb_npy.parent.mkdir(parents=True, exist_ok=True)
            # NPY is a portable exact archive of the 8-bit RGB stream.
            np.save(str(rgb_npy), np.empty((0,), dtype=np.uint8))
            rgb_npy.unlink()
            archive = np.lib.format.open_memmap(str(rgb_npy), mode="w+", dtype=np.uint8, shape=(count, height, width, 3))
        else:
            archive = None
        for index in range(count):
            frame = frames[0, index].detach().float().cpu().clamp(0, 1).mul(255).round().to(torch.uint8).numpy()
            if archive is not None:
                archive[index] = frame
            proc.stdin.write(np.ascontiguousarray(frame).tobytes())
            if index == 0 or index + 1 == count or index % 2 == 0:
                event("encode", index + 1, count, f"编码视频帧 {index + 1}/{count}")
        if archive is not None:
            archive.flush()
        proc.stdin.close()
        code = proc.wait()
    except Exception:
        proc.kill()
        proc.wait()
        raise
    if code != 0:
        raise RuntimeError(f"ffmpeg exited with code {code}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--latents", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--comfy-root", type=Path, required=True)
    p.add_argument("--lightx-root", type=Path, required=True)
    p.add_argument("--video-vae", type=Path, required=True)
    p.add_argument("--audio-vae", type=Path, required=True)
    p.add_argument("--frames", type=int, required=True)
    p.add_argument("--width", type=int, required=True)
    p.add_argument("--height", type=int, required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--lossless", action="store_true", help="also write libx264rgb CRF0 and an exact uint8 RGB NPY archive")
    p.add_argument("--crf", type=int, default=17, help="preview H.264 CRF (default 17)")
    p.add_argument("--keep-latent", action="store_true", help="copy a latent reference into output-dir")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    # Must be set before ComfyUI imports torch extensions.
    os.environ.setdefault("PYTHONPATH", str(args.comfy_root))
    os.environ.setdefault("LIGHTX2V_MINIMAL_IMPORT", "1")
    sys.path.insert(0, str(args.comfy_root))
    sys.path.insert(0, str(args.lightx_root))
    import torch
    from safetensors import safe_open

    from lightx2v.models.networks.minimax_h3.packing import (
        audio_latent_num_frames,
        unpack_audio_tokens,
        unpatchify_video_tokens,
        video_latent_num_frames,
    )
    import comfy.sd
    import comfy.utils
    from comfy.ldm.minimax.audio_vae import MiniMaxH3AudioVAE

    args.output_dir.mkdir(parents=True, exist_ok=True)
    event("decode", 0, 4, "读取并展开 H3 latent")
    with safe_open(str(args.latents), framework="pt", device="cpu") as f:
        video_rows = f.get_tensor("video_rows")
        audio_rows = f.get_tensor("audio_rows")
    latent_frames = video_latent_num_frames(args.frames)
    video = unpatchify_video_tokens(video_rows, latent_frames, args.height // 16, args.width // 16, channels=24)
    audio = unpack_audio_tokens(audio_rows, audio_latent_num_frames(args.frames)).permute(1, 0, 2).unsqueeze(0).contiguous()
    event("decode", 1, 4, f"latent 已展开 video={tuple(video.shape)} audio={tuple(audio.shape)}")

    device = torch.device(args.device)
    video_sd = load_safetensors(args.video_vae, torch=torch)
    add_latent_stats(video_sd, args.video_vae, torch=torch)
    from comfy.ldm.minimax.vae import MiniMaxH3VideoVAE

    video_module = MiniMaxH3VideoVAE()
    strict_keys(video_module, video_sd)
    del video_module
    event("decode", 2, 4, "视频 VAE 键校验通过，开始解码")
    video_vae = comfy.sd.VAE(sd=video_sd, device=device, dtype=torch.float16)
    del video_sd
    with torch.inference_mode():
        decoded = video_vae.decode(video)
    if decoded.shape[-1] != 3 or not torch.isfinite(decoded).all():
        raise RuntimeError(f"video decode invalid: shape={tuple(decoded.shape)}")
    event("decode", 3, 4, f"视频解码完成 shape={tuple(decoded.shape)} range={decoded.min().item():.4f}..{decoded.max().item():.4f}")
    del video_vae, video
    gc.collect()
    torch.cuda.empty_cache()

    raw_audio = load_safetensors(args.audio_vae, torch=torch)
    audio_sd = fold_weight_norm(raw_audio, torch=torch)
    add_latent_stats(audio_sd, args.audio_vae, torch=torch)
    del raw_audio
    audio_module = MiniMaxH3AudioVAE()
    strict_keys(audio_module, audio_sd)
    missing, unexpected = audio_module.load_state_dict(audio_sd, strict=True)
    if missing or unexpected:
        raise RuntimeError(f"audio strict load reported missing={missing} unexpected={unexpected}")
    audio_module.eval().to(device=device, dtype=torch.float32)
    event("decode", 4, 4, "音频 VAE 键校验通过，开始解码")
    with torch.inference_mode():
        decoded_audio = audio_module.decode(audio.to(device=device, dtype=torch.float32))
    if not torch.isfinite(decoded_audio).all() or float(decoded_audio.abs().max()) < 1e-5:
        raise RuntimeError("audio decode is non-finite or near-zero")

    wav = args.output_dir / "audio.wav"
    write_wav(wav, decoded_audio, torch=torch)
    preview = args.output_dir / "h3-preview.mp4"
    rgb_archive = args.output_dir / "h3-rgb8.npy" if args.lossless else None
    encode_video(decoded, wav, preview, torch=torch, lossless=False, crf=args.crf, rgb_npy=rgb_archive)
    lossless_path = None
    if args.lossless:
        lossless_path = args.output_dir / "h3-lossless.mp4"
        encode_video(decoded, wav, lossless_path, torch=torch, lossless=True, crf=args.crf)
    metadata = {
        "frames": args.frames,
        "fps": 24,
        "width": args.width,
        "height": args.height,
        "audio_sample_rate": 32000,
        "video_shape": list(decoded.shape),
        "audio_shape": list(decoded_audio.shape),
        "preview": str(preview),
        "lossless": str(lossless_path) if lossless_path else None,
        "rgb8": str(rgb_archive) if rgb_archive else None,
    }
    (args.output_dir / "media.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    result(**metadata)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
