#!/usr/bin/env python3
"""Conservative H3 weight/RAM estimate; it does not claim runtime fit."""
from __future__ import annotations

import argparse
from dataclasses import dataclass

GIB = 2**30


@dataclass(frozen=True)
class Weight:
    name: str
    gib: float
    note: str


WEIGHTS = {
    "dit_pruned_fp16": Weight("AdaLN-pruned FP16 DiT", 37.46, "community file statistic; TP4 theory only"),
    "dit_pruned_int8": Weight("pruned INT8 ConvRot DiT", 20.97e9 / GIB, "community measurement; V100-compatible kernel required"),
    "dit_bf16": Weight("full BF16 DiT", 66.28e9 / GIB, "official-style storage; not a V100 Tensor Core format"),
    "te_int8": Weight("INT8 ConvRot text encoder", 27.14e9 / GIB, "community measurement"),
    "te_bf16": Weight("BF16 text encoder", 51.51e9 / GIB, "Qwen3-VL based; V100 needs FP16/CPU handling"),
    "te_precomputed": Weight("precomputed text conditioning", 0.0, "encoder is outside the DiT process"),
    "vae": Weight("official video + audio VAE", (10.42e9 + 0.61e9) / GIB, "official HF metadata; community quantized variants may be smaller"),
}


def fmt(v: float) -> str:
    return f"{v:6.1f} GiB"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", default="v100-16gb", choices=["v100-16gb", "32gb-host"])
    ap.add_argument("--ram-gib", type=float, default=32)
    args = ap.parse_args()
    print("MiniMax H3 memory estimate (upper bound, not a benchmark)\n")
    for dit_key, te_key in (("dit_pruned_fp16", "te_precomputed"), ("dit_pruned_int8", "te_int8"), ("dit_bf16", "te_bf16")):
        dit, te, vae = WEIGHTS[dit_key], WEIGHTS[te_key], WEIGHTS["vae"]
        all_held = dit.gib + te.gib + vae.gib
        sequential = max(dit.gib, te.gib) + vae.gib
        print(f"{dit.name} + {te.name}")
        print(f"  all held in host RAM: {fmt(all_held)}")
        print(f"  largest stage + VAE:   {fmt(sequential)}")
        print(f"  TP4 weight-only/card:   {fmt((dit.gib + te.gib) / 4)}")
        print(f"  host RAM {args.ram_gib:.0f} GiB minus 4 GiB OS: {'fits lower bound' if sequential <= args.ram_gib - 4 else 'DOES NOT FIT lower bound'}")
        print()
    print("Context token estimate (hidden=5376, bf16 conditioning):")
    for w, h in ((640, 480), (960, 544), (1344, 768), (2048, 2048)):
        tokens = (w // 32) * (h // 32)
        mib = tokens * 5376 * 2 / 2**20
        print(f"  {w}x{h}: {tokens:,} image tokens, ~{mib:.1f} MiB per reference image")
    print("\nWarnings:")
    print("  * Total VRAM is not a shared pool; true TP/sharding is required.")
    print("  * INT8/FP8 storage does not guarantee a Volta-compatible compute kernel.")
    print("  * Add activation, allocator, VAE and communication headroom before declaring a fit.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
