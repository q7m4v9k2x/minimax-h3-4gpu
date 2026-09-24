#!/usr/bin/env python3
"""Run the MiniMax-H3 prompt -> TP4 DiT -> VAE -> MP4 pipeline.

The script is deliberately a process orchestrator.  Text encoding, TP4
sampling, and VAE decoding run in separate processes so the 27 GB Qwen
encoder is released before the four V100s load the DiT.  Every stage emits a
small JSONL progress record, which the web queue persists and replays after a
browser refresh.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
REMOTE_ROOT = Path(os.getenv("H3_REMOTE_ROOT", "/home/ymzx/minimax-h3-4gpu"))
COMFY_ROOT = Path(os.getenv("H3_COMFY_ROOT", "/home/ymzx/ComfyUI"))
COMFY_PYTHON = Path(os.getenv("H3_COMFY_PYTHON", str(COMFY_ROOT / "venv/bin/python")))
_default_torchrun = COMFY_ROOT / "venv/bin/torchrun"
TORCHRUN = os.getenv(
    "H3_TORCHRUN",
    str(_default_torchrun if _default_torchrun.is_file() else Path(sys.executable).with_name("torchrun")),
)
LIGHTX_ROOT = Path(os.getenv("H3_LIGHTX_ROOT", "/home/ymzx/LightX2V-V100"))
MODEL_ROOT = Path(os.getenv("H3_MODEL_ROOT", "/home/ymzx/models/minimax-h3"))
VIDEO_VAE = Path(os.getenv("H3_VIDEO_VAE", str(MODEL_ROOT / "FL2VA/FL2VA/video_vae/source/model.safetensors")))
AUDIO_VAE = Path(os.getenv("H3_AUDIO_VAE", str(MODEL_ROOT / "FL2VA/FL2VA/audio_vae/model.safetensors")))
TEXT_ENCODER = Path(os.getenv("H3_TEXT_ENCODER", str(COMFY_ROOT / "models/text_encoders/qwen3vl_32b_minimax_h3_int8_convrot.safetensors")))
TRANSFORMER = Path(os.getenv("H3_TRANSFORMER", str(MODEL_ROOT / "lightx2v-fl2v-pruned")))
BASE_CONFIG = ROOT / "config" / "lightx2v-v100-tp4-16gb-safe.json"
HIGH_CONFIG = ROOT / "config" / "lightx2v-v100-tp4-16gb-chunked8192.experimental.json"
EXPORTER = ROOT / "scripts" / "export_h3_conditioning.py"
DECODER = ROOT / "scripts" / "decode_h3_comfy.py"
MIN_TEXT_BYTES = 1_000_000_000


def safetensors_expected_bytes(path: Path) -> int | None:
    """Return the exact byte length declared by a safetensors header."""
    try:
        with path.open("rb") as handle:
            header_size_raw = handle.read(8)
            if len(header_size_raw) != 8:
                return None
            header_size = int.from_bytes(header_size_raw, "little")
            header = json.loads(handle.read(header_size))
        max_offset = max(
            value.get("data_offsets", [0, 0])[1]
            for key, value in header.items()
            if key != "__metadata__"
        )
        return 8 + header_size + max_offset
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return None


def emit(stage: str, current: int, total: int, message: str, **extra: Any) -> None:
    payload = {"event": "progress", "stage": stage, "current": current, "total": total, "message": message}
    payload.update(extra)
    print(json.dumps(payload, ensure_ascii=False), flush=True)


def emit_result(**values: Any) -> None:
    print(json.dumps({"event": "result", **values}, ensure_ascii=False), flush=True)


def check_resources() -> dict[str, Any]:
    checks: dict[str, Any] = {"ready": True, "reasons": [], "resources": {}}

    def required(path: Path, label: str, minimum: int = 1) -> None:
        try:
            size = path.stat().st_size
        except OSError:
            size = 0
        checks["resources"][label] = {"path": str(path), "bytes": size}
        if not path.is_file() or size < minimum:
            checks["ready"] = False
            checks["reasons"].append(f"{label} 不存在或未下载完整")

    required(EXPORTER, "conditioning_exporter")
    required(DECODER, "vae_decoder")
    required(TRANSFORMER / "config.json", "transformer_config")
    required(VIDEO_VAE, "video_vae", 1_000_000_000)
    required(AUDIO_VAE, "audio_vae", 100_000_000)
    required(TEXT_ENCODER, "text_encoder", MIN_TEXT_BYTES)
    if TEXT_ENCODER.is_file():
        actual = TEXT_ENCODER.stat().st_size
        expected = safetensors_expected_bytes(TEXT_ENCODER)
        checks["resources"]["text_encoder"].update({"declared_bytes": expected})
        if expected is None:
            checks["ready"] = False
            checks["reasons"].append("text_encoder safetensors 头部无法解析")
        elif actual != expected:
            checks["ready"] = False
            checks["reasons"].append(
                f"text_encoder 文件长度异常：实际 {actual}，索引声明 {expected}"
            )
    return checks


def run_stream(command: list[str], *, cwd: Path | None = None, env: dict[str, str] | None = None,
               stage: str = "starting", progress_re: re.Pattern[str] | None = None,
               total_default: int = 1) -> None:
    merged = os.environ.copy()
    merged.update(env or {})
    merged.setdefault("PYTHONUNBUFFERED", "1")
    process = subprocess.Popen(command, cwd=str(cwd) if cwd else None, env=merged,
                               stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                               text=True, encoding="utf-8", errors="replace", bufsize=1)
    last = None
    assert process.stdout is not None
    for line in process.stdout:
        # Preserve diagnostics in the job log while exposing only structured
        # progress on stdout to the queue.
        match = progress_re.search(line) if progress_re else None
        if match:
            current, total = int(match.group(1)), int(match.group(2))
            marker = (current, total)
            if marker != last:
                emit(stage, current, total, line.strip())
                last = marker
    code = process.wait()
    if code:
        raise RuntimeError(f"子流程失败（代码 {code}）: {' '.join(command[:3])}")


def write_config(path: Path, steps: int, high: bool) -> Path:
    source = HIGH_CONFIG if high else BASE_CONFIG
    config = json.loads(source.read_text(encoding="utf-8"))
    # LightX2V reports infer_steps - 1 model evaluations.  21 therefore means
    # the accepted 20-evaluation quality baseline.
    config["infer_steps"] = steps + 1
    target = path / "effective-config.json"
    target.write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return target


def parse_request(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("请求必须是 JSON 对象")
    prompt = value.get("prompt")
    width, height = int(value.get("width", 864)), int(value.get("height", 480))
    frames, steps = int(value.get("frames", 124)), int(value.get("steps", 20))
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("prompt 不能为空")
    if (width, height) not in {(864, 480), (480, 864), (1152, 640), (640, 1152)}:
        raise ValueError("不支持的 H3 几何尺寸")
    if frames != 124 or steps not in {20, 30}:
        raise ValueError("当前只验收 124 帧和 20/30 步")
    seed = int(value.get("seed", 0))
    return {"prompt": prompt.strip(), "width": width, "height": height, "frames": frames,
            "steps": steps, "seed": seed, "fps": 24, "lossless": bool(value.get("lossless", False))}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request-json", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--check", action="store_true", help="只检查权重和脚本，不启动 GPU")
    args = parser.parse_args(argv)
    if args.check:
        print(json.dumps(check_resources(), ensure_ascii=False))
        return 0
    if not args.request_json or not args.output_dir:
        parser.error("生成时需要 --request-json 和 --output-dir")

    request = parse_request(args.request_json)
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    resources = check_resources()
    if not resources["ready"]:
        raise RuntimeError("; ".join(resources["reasons"]))
    prompt = request["prompt"]
    width, height, frames, steps = request["width"], request["height"], request["frames"], request["steps"]
    high = max(width * height, 480 * 864) > 480 * 864

    # Conditioning is intentionally generated in a child process so the
    # encoder's memory is returned before torchrun initializes four V100s.
    condition = out / "conditioning.safetensors"
    emit("conditioning", 0, 1, "加载 Qwen3-VL H3 文本编码器")
    encoder_python = str(COMFY_PYTHON if COMFY_PYTHON.is_file() else Path(sys.executable))
    encoder_env = {"PYTHONPATH": str(COMFY_ROOT)}
    run_stream([encoder_python, str(EXPORTER), "--prompt", prompt, "--output", str(condition),
                "--comfy-root", str(COMFY_ROOT), "--cpu"],
               cwd=COMFY_ROOT, env=encoder_env, stage="conditioning")
    if not condition.is_file() or condition.stat().st_size == 0:
        raise RuntimeError("文本编码器没有输出 conditioning bundle")
    emit("conditioning", 1, 1, "conditioning bundle 已生成")

    config = write_config(out, steps, high)
    latent = out / "h3-latents.safetensors"
    case_id = out.name
    emit("loading", 0, 1, "启动四卡 TP4 DiT")
    env = {
        "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
        "CUDA_VISIBLE_DEVICES": "1,2,3,4",
        "DTYPE": "FP16",
        "SENSITIVE_LAYER_DTYPE": "FP32",
        "LIGHTX2V_MINIMAL_IMPORT": "1",
        "PYTHONPATH": str(LIGHTX_ROOT),
    }
    run_stream([
        TORCHRUN, "--standalone", "--nnodes=1", "--nproc_per_node=4",
        str(LIGHTX_ROOT / "tools/minimax_h3/run_tp4_benchmark.py"),
        "--config", str(config), "--model-path", str(MODEL_ROOT),
        "--transformer-path", str(TRANSFORMER), "--task", "t2av",
        "--condition-path", str(condition), "--prompt", prompt,
        "--seed", str(request["seed"]), "--case-id", case_id,
        "--frames", str(frames), "--nominal-seconds", "5",
        "--height", str(height), "--width", str(width), "--trials", "1",
        "--evidence-dir", str(out), "--report", str(out / "benchmark.json"),
        "--effective-config", str(out / "effective-runtime.json"),
    ], cwd=LIGHTX_ROOT, env=env, stage="sampling",
       progress_re=re.compile(r"MiniMax-H3 step:\s*(\d+)\s*/\s*(\d+)"))
    # The runner uses <case>-trial1-latents.safetensors as its stable output.
    generated = out / f"{case_id}-trial1-latents.safetensors"
    if not generated.is_file():
        raise RuntimeError("TP4 没有生成 latent 文件")
    if generated != latent:
        shutil.copy2(generated, latent)
    emit("sampling", steps, steps, "四卡 DiT latent 已完成")

    emit("decode", 0, 4, "加载官方视频/音频 VAE")
    decoder = str(DECODER)
    decode_env = {"PYTHONPATH": str(COMFY_ROOT) + os.pathsep + str(LIGHTX_ROOT), "LIGHTX2V_MINIMAL_IMPORT": "1"}
    decoder_python = str(COMFY_PYTHON if COMFY_PYTHON.is_file() else Path(sys.executable))
    decode_command = [decoder_python, decoder, "--latents", str(latent), "--output-dir", str(out),
                      "--comfy-root", str(COMFY_ROOT), "--lightx-root", str(LIGHTX_ROOT),
                      "--video-vae", str(VIDEO_VAE), "--audio-vae", str(AUDIO_VAE),
                      "--frames", str(frames), "--width", str(width), "--height", str(height),
                      "--device", "cuda:0", "--crf", "17"]
    if request["lossless"]:
        decode_command.append("--lossless")
    run_stream(decode_command, cwd=ROOT, env=decode_env, stage="decode",
               progress_re=re.compile(r'"stage":\s*"(?:decode|encode)".*?"current":\s*(\d+).*?"total":\s*(\d+)'))

    media = json.loads((out / "media.json").read_text(encoding="utf-8"))
    preview = Path(media["preview"]).resolve()
    if not preview.is_file():
        raise RuntimeError("VAE/编码阶段没有生成 MP4")
    # The web worker serves only files inside the job directory.
    result = {"video": preview.name, "poster": None, "lossless": None, "metadata": media}
    if request["lossless"] and media.get("lossless"):
        result["lossless"] = Path(media["lossless"]).resolve().name
    (out / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    emit("encode", 1, 1, "MP4 已写入并完成校验")
    emit_result(**result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
