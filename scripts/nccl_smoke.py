#!/usr/bin/env python3
"""Small single-node NCCL all-reduce gate for the selected four GPUs."""
from __future__ import annotations

import argparse
import os
import time

import torch
import torch.distributed as dist


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--megabytes", type=int, default=32)
    ap.add_argument("--iters", type=int, default=10)
    args = ap.parse_args()
    rank = int(os.environ["RANK"])
    world = int(os.environ["WORLD_SIZE"])
    local = int(os.environ.get("LOCAL_RANK", rank))
    torch.cuda.set_device(local)
    dist.init_process_group("nccl")
    n = args.megabytes * 1024 * 1024 // 2
    x = torch.ones(n, dtype=torch.float16, device="cuda") * (rank + 1)
    for _ in range(3):
        dist.all_reduce(x)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(args.iters):
        dist.all_reduce(x)
    torch.cuda.synchronize()
    elapsed = (time.perf_counter() - t0) / args.iters
    expected = world * (world + 1) // 2
    err = float((x[0] - expected).abs().item())
    if rank == 0:
        aggregate_gib_s = (args.megabytes / 1024) * 2 * (world - 1) / world / elapsed
        print({"world": world, "bytes_per_tensor": args.megabytes * 1024 * 1024, "avg_seconds": elapsed, "aggregate_gib_s": aggregate_gib_s, "max_error": err})
    dist.destroy_process_group()
    return 0 if err == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
