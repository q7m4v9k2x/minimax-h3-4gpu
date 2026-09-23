"""Summarize MiniMax-H3 stage timings and identify the measured bottleneck.

The tool consumes the JSON emitted by run_tp4_benchmark.py. It deliberately
does not invent missing stages: a DiT-only report is marked as incomplete when
VAE/encoding/output timings are absent.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


def _elapsed(mapping: dict[str, Any] | None) -> float:
    if not mapping:
        return 0.0
    return float(mapping.get("elapsed_seconds", 0.0) or 0.0)


def _gib(value: int | float) -> float:
    return float(value or 0.0) / (1024**3)


def _gb(value: int | float) -> float:
    return float(value or 0.0) / 1_000_000_000


def _max_stage(ranks: list[dict[str, Any]], stage: str) -> float:
    values = []
    for rank in ranks:
        for trial in rank.get("trials", []):
            values.append(_elapsed(trial.get(stage)))
    return max(values, default=0.0)


def _max_peak_memory(ranks: list[dict[str, Any]], key: str) -> int:
    values = []
    for rank in ranks:
        for trial in rank.get("trials", []):
            values.append(int(trial.get("cuda_memory", {}).get(key, 0) or 0))
    return max(values, default=0)


def summarize_report(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        report = json.load(handle)

    ranks = report.get("ranks", [])
    if not ranks:
        raise ValueError(f"{path}: report has no ranks")
    trial_count = max((len(rank.get("trials", [])) for rank in ranks), default=0)
    stage_names = ("condition_wait", "conditioning", "prepare_dit", "dit_segment", "pipeline")
    stages = {name: _max_stage(ranks, name) for name in stage_names}
    residual = max(
        stages["pipeline"] - stages["conditioning"] - stages["prepare_dit"] - stages["dit_segment"],
        0.0,
    )
    stages["pipeline_overhead"] = residual

    step_times: list[float] = []
    for rank in ranks:
        for trial in rank.get("trials", []):
            step_times.extend(
                float(item.get("elapsed_seconds", 0.0) or 0.0)
                for item in trial.get("model_evaluations", [])
            )
    step_times.sort()
    step_count = len(step_times)
    step_p50 = step_times[(step_count - 1) // 2] if step_times else 0.0
    step_p95 = step_times[min(step_count - 1, math.ceil(step_count * 0.95) - 1)] if step_times else 0.0

    measured = {
        name: value
        for name, value in stages.items()
        if name not in {"condition_wait", "pipeline", "pipeline_overhead"} and value > 0
    }
    bottleneck = max(measured, key=measured.get) if measured else None
    peak_allocated = _max_peak_memory(ranks, "peak_allocated_bytes")
    peak_reserved = _max_peak_memory(ranks, "peak_reserved_bytes")
    first_trial = ranks[0].get("trials", [{}])[0] if ranks[0].get("trials") else {}
    metadata = first_trial.get("latent_metadata", {})
    return {
        "source": str(path),
        "status": report.get("status", "unknown"),
        "case_id": report.get("case_id"),
        "geometry": report.get("geometry", {}),
        "benchmark": report.get("benchmark", {}),
        "runtime": report.get("runtime", {}),
        "trial_count": trial_count,
        "stages_seconds": stages,
        "bottleneck": bottleneck,
        "bottleneck_seconds": measured.get(bottleneck, 0.0) if bottleneck else 0.0,
        "step_count": step_count,
        "step_p50_seconds": step_p50,
        "step_p95_seconds": step_p95,
        "peak_allocated": {"bytes": peak_allocated, "gib": _gib(peak_allocated), "gb": _gb(peak_allocated)},
        "peak_reserved": {"bytes": peak_reserved, "gib": _gib(peak_reserved), "gb": _gb(peak_reserved)},
        "latent_finite": metadata.get("finite"),
        "has_vae_stage": False,
        "has_output_stage": False,
        "warnings": [
            "report is DiT-only; VAE, media encoding, atomic persistence and API response are not measured",
        ],
    }


def to_markdown(summary: dict[str, Any]) -> str:
    geometry = summary["geometry"]
    stages = summary["stages_seconds"]
    lines = [
        f"## {summary.get('case_id') or 'H3 report'}",
        "",
        f"- source: {summary['source']}",
        f"- status: {summary['status']}",
        f"- geometry: {geometry.get('height')}×{geometry.get('width')}×{geometry.get('frames')} frames",
        f"- bottleneck: {summary['bottleneck']} ({summary['bottleneck_seconds']:.3f} s)",
        f"- peak allocated: {summary['peak_allocated']['gib']:.2f} GiB / {summary['peak_allocated']['gb']:.2f} GB",
        f"- peak reserved: {summary['peak_reserved']['gib']:.2f} GiB / {summary['peak_reserved']['gb']:.2f} GB",
        f"- step P50/P95: {summary['step_p50_seconds']:.3f} / {summary['step_p95_seconds']:.3f} s",
        "",
        "| Stage | Seconds | Share of pipeline |",
        "|---|---:|---:|",
    ]
    pipeline = stages.get("pipeline", 0.0)
    for name in ("condition_wait", "conditioning", "prepare_dit", "dit_segment", "pipeline_overhead", "pipeline"):
        value = stages.get(name, 0.0)
        share = value / pipeline * 100 if pipeline else 0.0
        lines.append(f"| {name} | {value:.3f} | {share:.1f}% |")
    lines.extend(["", "- VAE/output stage: not measured (DiT-only report)", ""])
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reports", nargs="+", type=Path, help="benchmark report JSON files")
    parser.add_argument("--json-out", type=Path, help="write combined summary JSON")
    parser.add_argument("--markdown-out", type=Path, help="write a Markdown summary")
    args = parser.parse_args()

    summaries = [summarize_report(path) for path in args.reports]
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(summaries, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    markdown = "\n".join(to_markdown(item) for item in summaries)
    if args.markdown_out:
        args.markdown_out.parent.mkdir(parents=True, exist_ok=True)
        args.markdown_out.write_text(markdown + "\n", encoding="utf-8")
    print(markdown)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
