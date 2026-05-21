from __future__ import annotations

import argparse
import json
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
RUNTIME = ROOT / "runtimes" / "circleworld_proto"
for path in (ROOT, RUNTIME):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from benchmark_circleworld_real_anchor import run_benchmark
from evaluate_circleworld import evaluate_config
from export_circleworld_audio import _load_circle_cfg
from train_circleworld_real_anchor import _evaluate_transfer_probe_repeated, _selection_gate_status


def _parse_float_list(value: str) -> list[float]:
    parts = [chunk.strip() for chunk in value.split(",") if chunk.strip()]
    return [float(part) for part in parts]


def _load_payload(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _set_mix(payload: dict[str, Any], parent_mix: float, parent_mix_early: float) -> dict[str, Any]:
    out = deepcopy(payload)
    cfg = out["config"] if "config" in out else out
    cfg["child_parent_mix"] = float(parent_mix)
    cfg["child_parent_mix_early"] = float(parent_mix_early)
    return out


def _variant_name(parent_mix: float, parent_mix_early: float) -> str:
    return f"parentmix_{int(round(parent_mix * 1000)):03d}_{int(round(parent_mix_early * 1000)):03d}"


def _write_markdown(path: Path, rows: list[dict[str, Any]], base_config_path: Path) -> None:
    lines = [
        "# Circleworld Parent-Mix Replay Sweep",
        "",
        f"Base config: `{base_config_path}`",
        "",
        "| variant | parent_mix | early_mix | gate | margin | naked_parent_div_min | naked_writeback_min | naked_branch_min | eval_naked_parent_div | eval_branch | bench_corr | bench_mae |",
        "| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            "| {variant} | {parent_mix:.3f} | {parent_mix_early:.3f} | {gate} | {margin:.6f} | "
            "{naked_parent_div_min:.6f} | {naked_writeback_min:.6f} | {naked_branch_min:.6f} | "
            "{eval_naked_parent_div:.6f} | {eval_branch:.6f} | {bench_corr:.6f} | {bench_mae:.6f} |".format(
                variant=row["variant"],
                parent_mix=row["child_parent_mix"],
                parent_mix_early=row["child_parent_mix_early"],
                gate="pass" if row["gate_passed"] else "fail",
                margin=row["gate_margin"],
                naked_parent_div_min=row["probe"]["naked_parent_div_min"],
                naked_writeback_min=row["probe"]["naked_writeback_min"],
                naked_branch_min=row["probe"]["naked_branch_min"],
                eval_naked_parent_div=row["heldout"]["by_source"]["naked_rafa"]["mean_child_parent_divergence"],
                eval_branch=row["heldout"]["mean_real_branch_fraction"],
                bench_corr=row["benchmark"]["mean_corr"],
                bench_mae=row["benchmark"]["mean_mae"],
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_sweep(
    base_config_path: Path,
    out_dir: Path,
    device_name: str,
    time_steps: int,
    clip_seconds: int,
    phase_blend: float,
    parent_mix_values: list[float],
    early_mix_values: list[float],
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    configs_dir = out_dir / "configs"
    configs_dir.mkdir(parents=True, exist_ok=True)

    payload = _load_payload(base_config_path)
    score_cfg = dict(payload.get("score_cfg", {}))
    probe_cfg = dict(payload.get("heldout_probe_cfg", {}))
    rows: list[dict[str, Any]] = []

    for parent_mix in parent_mix_values:
        for early_mix in early_mix_values:
            if early_mix >= parent_mix:
                continue
            variant = _variant_name(parent_mix, early_mix)
            variant_payload = _set_mix(payload, parent_mix=parent_mix, parent_mix_early=early_mix)
            config_path = configs_dir / f"{variant}.json"
            config_path.write_text(json.dumps(variant_payload, indent=2), encoding="utf-8")

            heldout_dir = out_dir / variant / "heldout_eval"
            benchmark_dir = out_dir / variant / "benchmark"
            heldout = evaluate_config(config_path=config_path, out_dir=heldout_dir, time_steps=time_steps, device_name=device_name)
            circle_cfg = _load_circle_cfg(config_path)
            probe = _evaluate_transfer_probe_repeated(
                circle_cfg,
                device_name=device_name,
                score_cfg=score_cfg,
                probe_cfg=probe_cfg,
            )
            gate = _selection_gate_status(probe, probe_cfg=probe_cfg, score_cfg=score_cfg)
            benchmark = run_benchmark(
                config_path=config_path,
                out_dir=benchmark_dir,
                device_name=device_name,
                clip_seconds=clip_seconds,
                phase_blend=phase_blend,
            )
            rows.append(
                {
                    "variant": variant,
                    "config_path": str(config_path),
                    "child_parent_mix": float(parent_mix),
                    "child_parent_mix_early": float(early_mix),
                    "heldout": heldout,
                    "probe": probe,
                    "gate_passed": bool(gate["passed"]),
                    "gate_margin": float(gate["margin"]),
                    "gate_failed_checks": list(gate["failed_checks"]),
                    "benchmark": benchmark,
                }
            )

    rows.sort(
        key=lambda row: (
            1 if row["gate_passed"] else 0,
            row["gate_margin"],
            row["probe"].get("naked_parent_div_min", 0.0),
            row["benchmark"].get("mean_corr", 0.0),
        ),
        reverse=True,
    )

    summary = {
        "base_config_path": str(base_config_path),
        "device": device_name,
        "time_steps": int(time_steps),
        "clip_seconds": int(clip_seconds),
        "phase_blend": float(phase_blend),
        "parent_mix_values": parent_mix_values,
        "early_mix_values": early_mix_values,
        "rows": rows,
    }
    (out_dir / "parentmix_sweep_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    _write_markdown(out_dir / "parentmix_sweep_summary.md", rows=rows, base_config_path=base_config_path)
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description="Focused Circleworld replay-safe parent-mix sweep.")
    ap.add_argument("--base-config", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--time-steps", type=int, default=128)
    ap.add_argument("--clip-seconds", type=int, default=10)
    ap.add_argument("--phase-blend", type=float, default=1.0)
    ap.add_argument("--parent-mix-values", default="0.18,0.22,0.26")
    ap.add_argument("--early-mix-values", default="0.10,0.14,0.18")
    args = ap.parse_args()

    summary = run_sweep(
        base_config_path=Path(args.base_config),
        out_dir=Path(args.out_dir),
        device_name=args.device,
        time_steps=int(args.time_steps),
        clip_seconds=int(args.clip_seconds),
        phase_blend=float(args.phase_blend),
        parent_mix_values=_parse_float_list(args.parent_mix_values),
        early_mix_values=_parse_float_list(args.early_mix_values),
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
