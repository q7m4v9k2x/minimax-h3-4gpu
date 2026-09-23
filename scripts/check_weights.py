#!/usr/bin/env python3
"""Inspect a local H3 directory without downloading or loading weights."""
from __future__ import annotations

import argparse
from pathlib import Path


def size_gib(p: Path) -> float:
    return sum(x.stat().st_size for x in p.rglob("*") if x.is_file()) / 2**30


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("model_dir", type=Path)
    args = ap.parse_args()
    root = args.model_dir
    if not root.exists():
        print(f"missing: {root}")
        return 2
    print(f"root: {root}\nsize: {size_gib(root):.2f} GiB")
    for name in ("FL2VA", "Ref2VA", "transformer", "transformer_ref", "text_encoder", "vae", "video_vae", "audio_vae"):
        p = root / name
        if p.exists():
            print(f"  {name:16} {size_gib(p):8.2f} GiB")
    files = list(root.rglob("*.safetensors"))
    print(f"safetensors: {len(files)}")
    if not files:
        print("warning: no safetensors found")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

