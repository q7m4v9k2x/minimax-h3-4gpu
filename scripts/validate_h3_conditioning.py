#!/usr/bin/env python3
"""Validate a MiniMax-H3 LightX2V conditioning bundle without loading models."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def validate(path: Path) -> dict:
    import torch
    from safetensors import safe_open

    with safe_open(str(path), framework="pt", device="cpu") as f:
        keys = set(f.keys())
        missing = {"prompt_embeds", "text_token_tags"} - keys
        if missing:
            raise ValueError(f"missing tensors: {sorted(missing)}")
        embeds = f.get_tensor("prompt_embeds")
        tags = f.get_tensor("text_token_tags")
        metadata = dict(f.metadata() or {})
    if embeds.ndim != 2 or embeds.shape[-1] != 5120:
        raise ValueError(f"prompt_embeds must be [tokens,5120], got {tuple(embeds.shape)}")
    if tags.ndim != 1 or tags.shape[0] != embeds.shape[0]:
        raise ValueError(f"text_token_tags must match rows: embeds={tuple(embeds.shape)}, tags={tuple(tags.shape)}")
    if tags.dtype not in (torch.int64, torch.int32, torch.int16, torch.int8):
        raise ValueError(f"text_token_tags must be integer, got {tags.dtype}")
    if not torch.isfinite(embeds).all().item():
        raise ValueError("prompt_embeds contains NaN/Inf")
    raw = metadata.get("minimax_h3_bundle")
    bundle = json.loads(raw) if raw else {}
    if bundle.get("format") != "minimax_h3_conditioning_bundle":
        raise ValueError("missing/invalid minimax_h3_bundle metadata")
    return {"path": str(path.resolve()), "shape": list(embeds.shape), "dtype": str(embeds.dtype), "tags_dtype": str(tags.dtype), "keys": sorted(keys), "bundle": bundle}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("bundle", type=Path)
    args = p.parse_args()
    print(json.dumps(validate(args.bundle), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
