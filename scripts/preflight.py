#!/usr/bin/env python3
"""Read-only V100/P2P/PyTorch preflight for MiniMax H3 TP4."""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


def run(cmd: list[str], timeout: int = 30) -> tuple[int, str, str]:
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
        return 127, "", str(exc)
    return p.returncode, p.stdout.strip(), p.stderr.strip()


def nvidia_rows() -> list[dict]:
    fields = ["index", "name", "memory.total", "compute_cap"]
    code, out, err = run(["nvidia-smi", f"--query-gpu={','.join(fields)}", "--format=csv,noheader,nounits"])
    if code != 0:
        raise RuntimeError(f"nvidia-smi failed: {err or out}")
    rows = []
    for line in out.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) != len(fields):
            continue
        rows.append(dict(zip(fields, parts)))
    return rows


def p2p_matrix() -> str:
    code, out, err = run(["nvidia-smi", "topo", "-p2p", "r"], timeout=20)
    return out if code == 0 else f"unavailable: {err or out}"


def topo() -> str:
    code, out, err = run(["nvidia-smi", "topo", "-m"], timeout=20)
    return out if code == 0 else f"unavailable: {err or out}"


def torch_probe() -> dict:
    try:
        import torch  # type: ignore
    except Exception as exc:
        return {"available": False, "error": str(exc)}
    result = {
        "available": True,
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "device_count": torch.cuda.device_count(),
        "devices": [],
        "p2p": [],
    }
    for i in range(torch.cuda.device_count()):
        prop = torch.cuda.get_device_properties(i)
        result["devices"].append({
            "index": i,
            "name": prop.name,
            "cc": f"{prop.major}.{prop.minor}",
            "total_gib": round(prop.total_memory / 2**30, 2),
        })
    for i in range(torch.cuda.device_count()):
        for j in range(torch.cuda.device_count()):
            if i != j:
                try:
                    result["p2p"].append({"src": i, "dst": j, "can_access": bool(torch.cuda.can_device_access_peer(i, j))})
                except Exception as exc:
                    result["p2p"].append({"src": i, "dst": j, "error": str(exc)})
    return result


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", type=Path, help="write a machine-readable report")
    args = ap.parse_args()
    report: dict = {"timestamp_utc": datetime.now(timezone.utc).isoformat(), "env": {k: os.environ.get(k) for k in ("CUDA_VISIBLE_DEVICES", "CUDA_DEVICE_ORDER")}}
    try:
        report["nvidia_smi"] = nvidia_rows()
        report["p2p_matrix"] = p2p_matrix()
        report["topology"] = topo()
    except Exception as exc:
        report["nvidia_error"] = str(exc)
    report["torch"] = torch_probe()

    rows = report.get("nvidia_smi", [])
    v100 = [r for r in rows if "V100" in r.get("name", "") and float(r.get("memory.total", "0")) >= 14000]
    warnings = []
    notes = []
    if len(v100) != 4:
        warnings.append(f"expected 4 V100 >=14,000 MiB, found {len(v100)}")
    if any(r.get("compute_cap") not in ("7.0", "7.0 ") for r in v100):
        warnings.append("one or more selected GPUs are not SM70")
    if "K620" in " ".join(r.get("name", "") for r in rows):
        notes.append("Quadro K620 detected: keep it out of CUDA_VISIBLE_DEVICES")
    report["selected_v100"] = v100
    report["warnings"] = warnings
    report["notes"] = notes
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return 0 if not warnings else 2


if __name__ == "__main__":
    raise SystemExit(main())
