#!/usr/bin/env python3
"""Export real MiniMax-H3 prompt conditioning for the LightX2V TP4 runner.

This intentionally loads ComfyUI's official MiniMax-H3 tokenizer/model path;
it never fabricates embeddings.  The resulting safetensors file is accepted by
LightX2V's ``precomputed_condition_path`` option.

Run from the ComfyUI environment (or pass ``--comfy-root``):
  python export_h3_conditioning.py --prompt "..." --output /path/cond.safetensors
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--prompt", required=True, help="Text prompt passed to Qwen3-VL")
    p.add_argument("--output", required=True, type=Path, help="Output .safetensors bundle")
    p.add_argument(
        "--clip", type=Path,
        default=None,
        help="Qwen3-VL H3 checkpoint; defaults to ComfyUI/models/text_encoders/...convrot.safetensors",
    )
    p.add_argument("--comfy-root", type=Path, default=None, help="ComfyUI checkout (default: /home/ymzx/ComfyUI)")
    p.add_argument("--cpu", action="store_true", help="Keep encoder on CPU where supported")
    return p


def _load_clip(comfy_root: Path, checkpoint: Path, cpu: bool):
    # ComfyUI imports are deliberately delayed so --help works on any machine.
    root = str(comfy_root.resolve())
    if root not in sys.path:
        sys.path.insert(0, root)
    import comfy.sd  # type: ignore
    import folder_paths  # type: ignore

    # This enum selects the normal Qwen3-VL-32B branch; detect_te_model then
    # selects MiniMaxH3TEModel and MiniMaxH3Tokenizer from the checkpoint keys.
    clip_type = comfy.sd.CLIPType.MINIMAX
    model_options = {}
    if cpu:
        # ``load_clip`` still owns the model patcher; CPU is the safe default
        # on 32-GiB hosts and avoids occupying a DiT GPU during export.
        import torch
        model_options["load_device"] = torch.device("cpu")
        model_options["offload_device"] = torch.device("cpu")
    return comfy.sd.load_clip(
        [str(checkpoint)],
        embedding_directory=folder_paths.get_folder_paths("embeddings"),
        clip_type=clip_type,
        model_options=model_options,
    )


def export_conditioning(prompt: str, output: Path, comfy_root: Path, checkpoint: Path, cpu: bool) -> dict:
    import torch
    from safetensors.torch import save_file

    clip = _load_clip(comfy_root, checkpoint, cpu)
    tokens = clip.tokenize(prompt)
    conditioning = clip.encode_from_tokens_scheduled(tokens, show_pbar=True)
    if len(conditioning) != 1:
        raise RuntimeError(f"Expected one H3 conditioning entry, got {len(conditioning)}")
    prompt_embeds, metadata = conditioning[0]
    token_tags = metadata.get("minimax_token_tags")
    if token_tags is None:
        raise RuntimeError("H3 encoder returned no minimax_token_tags; checkpoint/runtime is not the H3 path")
    if prompt_embeds.ndim == 3 and prompt_embeds.shape[0] == 1:
        prompt_embeds = prompt_embeds.squeeze(0)
    if token_tags.ndim == 2 and token_tags.shape[0] == 1:
        token_tags = token_tags.squeeze(0)
    if prompt_embeds.ndim != 2 or prompt_embeds.shape[-1] != 5120:
        raise ValueError(f"Expected prompt_embeds [tokens,5120], got {tuple(prompt_embeds.shape)}")
    if token_tags.ndim != 1 or token_tags.shape[0] != prompt_embeds.shape[0]:
        raise ValueError(f"Token tags shape {tuple(token_tags.shape)} does not match embeddings {tuple(prompt_embeds.shape)}")
    if not torch.isfinite(prompt_embeds).all().item():
        raise ValueError("H3 prompt embeddings contain NaN/Inf")

    output = output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    bundle = {
        "format": "minimax_h3_conditioning_bundle",
        "version": 1,
        "task": "t2av",
        "keyframes": [],
        "references": [],
    }
    tensors = {
        "prompt_embeds": prompt_embeds.detach().to("cpu").contiguous(),
        "text_token_tags": token_tags.detach().to(dtype=torch.int64, device="cpu").contiguous(),
    }
    save_file(tensors, str(output), metadata={
        "source": "MiniMax-H3 conditioning export for LightX2V",
        "prompt_shape": str(tuple(tensors["prompt_embeds"].shape)),
        "prompt_dtype": str(tensors["prompt_embeds"].dtype),
        "minimax_h3_bundle": json.dumps(bundle, ensure_ascii=False, separators=(",", ":")),
    })
    return {"output": str(output), "shape": list(tensors["prompt_embeds"].shape), "dtype": str(tensors["prompt_embeds"].dtype), "finite": True}


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    comfy_root = (args.comfy_root or Path(os.environ.get("COMFY_ROOT", "/home/ymzx/ComfyUI"))).resolve()
    checkpoint = args.clip or (comfy_root / "models/text_encoders/qwen3vl_32b_minimax_h3_int8_convrot.safetensors")
    if not checkpoint.is_file():
        raise FileNotFoundError(f"H3 text encoder not found: {checkpoint}")
    result = export_conditioning(args.prompt, args.output, comfy_root, checkpoint, args.cpu)
    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
