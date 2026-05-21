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


def _parse_int_list(value: str) -> list[int]:
    parts = [chunk.strip() for chunk in value.split(",") if chunk.strip()]
    return [int(part) for part in parts]


def _load_payload(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _set_params(
    payload: dict[str, Any],
    *,
    support_window: int,
    support_decay: float,
    writeback_gain: float,
) -> dict[str, Any]:
    out = deepcopy(payload)
    cfg = out["config"] if "config" in out else out
    cfg["child_support_window"] = int(support_window)
    cfg["child_support_decay"] = float(support_decay)
    cfg["child_writeback_gain"] = float(writeback_gain)
    return out


def _variant_name(support_window: int, support_decay: float, writeback_gain: float) -> str:
    decay_tag = int(round(support_decay * 1000))
    gain_tag = int(round(writeback_gain * 1000))
    return f"survival_w{support_window:02d}_d{decay_tag:03d}_g{gain_tag:03d}"


def _write_markdown(path: Path, rows: list[dict[str, Any]], base_config_path: Path) -> None:
    lines = [
        "# Circleworld Child-Survival Replay Sweep",
        "",
        f"Base config: `{base_config_path}`",
        "",
        "| variant | support_window | support_decay | writeback_gain | gate | margin | naked_parent_div_min | naked_meso_min | naked_writeback_min | bench_corr | bench_mae |",
        "| --- | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        bench = row.get("benchmark") or {}
        lines.append(
            "| {variant} | {support_window:d} | {support_decay:.4f} | {writeback_gain:.4f} | {gate} | {margin:.6f} | "
            "{naked_parent_div_min:.6f} | {naked_meso_min:.6f} | {naked_writeback_min:.6f} | {bench_corr} | {bench_mae} |".format(
                variant=row["variant"],
                support_window=row["child_support_window"],
                support_decay=row["child_support_decay"],
                writeback_gain=row["child_writeback_gain"],
                gate="pass" if row["gate_passed"] else "fail",
                margin=row["gate_margin"],
                naked_parent_div_min=row["probe"].get("naked_parent_div_min", 0.0),
                naked_meso_min=row["probe"].get("naked_meso_min", 0.0),
                naked_writeback_min=row["probe"].get("naked_writeback_min", 0.0),
                bench_corr=f"{bench.get('mean_corr', 0.0):.6f}" if bench else "-",
                bench_mae=f"{bench.get('mean_mae', 0.0):.6f}" if bench else "-",
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_sweep(
    *,
    base_config_path: Path,
    out_dir: Path,
    device_name: str,
    time_steps: int,
    clip_seconds: int,
    phase_blend: float,
    support_window_values: list[int],
    support_decay_values: list[float],
    writeback_gain_values: list[float],
    benchmark_top_k: int,
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    configs_dir = out_dir / "configs"
    configs_dir.mkdir(parents=True, exist_ok=True)

    payload = _load_payload(base_config_path)
    score_cfg = dict(payload.get("score_cfg", {}))
    probe_cfg = dict(payload.get("heldout_probe_cfg", {}))
    rows: list[dict[str, Any]] = []

    for support_window in support_window_values:
        for support_decay in support_decay_values:
            for writeback_gain in writeback_gain_values:
                variant = _variant_name(support_window, support_decay, writeback_gain)
                variant_payload = _set_params(
                    payload,
                    support_window=int(support_window),
                    support_decay=float(support_decay),
                    writeback_gain=float(writeback_gain),
                )
                config_path = configs_dir / f"{variant}.json"
                config_path.write_text(json.dumps(variant_payload, indent=2), encoding="utf-8")

                heldout_dir = out_dir / variant / "heldout_eval"
                heldout = evaluate_config(
                    config_path=config_path,
                    out_dir=heldout_dir,
                    time_steps=time_steps,
                    device_name=device_name,
                )
                circle_cfg = _load_circle_cfg(config_path)
                probe = _evaluate_transfer_probe_repeated(
                    circle_cfg,
                    device_name=device_name,
                    score_cfg=score_cfg,
                    probe_cfg=probe_cfg,
                )
                gate = _selection_gate_status(probe, probe_cfg=probe_cfg, score_cfg=score_cfg)
                rows.append(
                    {
                        "variant": variant,
                        "config_path": str(config_path),
                        "child_support_window": int(support_window),
                        "child_support_decay": float(support_decay),
                        "child_writeback_gain": float(writeback_gain),
                        "heldout": heldout,
                        "probe": probe,
                        "gate_passed": bool(gate["passed"]),
                        "gate_margin": float(gate["margin"]),
                        "gate_failed_checks": list(gate["failed_checks"]),
                    }
                )

    rows.sort(
        key=lambda row: (
            1 if row["gate_passed"] else 0,
            row["gate_margin"],
            row["probe"].get("naked_parent_div_min", 0.0),
            row["probe"].get("naked_meso_min", 0.0),
            row["probe"].get("naked_writeback_min", 0.0),
        ),
        reverse=True,
    )

    for row in rows[: max(0, int(benchmark_top_k))]:
        benchmark_dir = out_dir / row["variant"] / "benchmark"
        row["benchmark"] = run_benchmark(
            config_path=Path(row["config_path"]),
            out_dir=benchmark_dir,
            device_name=device_name,
            clip_seconds=clip_seconds,
            phase_blend=phase_blend,
        )

    summary = {
        "base_config_path": str(base_config_path),
        "device": device_name,
        "time_steps": int(time_steps),
        "clip_seconds": int(clip_seconds),
        "phase_blend": float(phase_blend),
        "support_window_values": support_window_values,
        "support_decay_values": support_decay_values,
        "writeback_gain_values": writeback_gain_values,
        "benchmark_top_k": int(benchmark_top_k),
        "rows": rows,
    }
    (out_dir / "childsurvival_sweep_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    _write_markdown(out_dir / "childsurvival_sweep_summary.md", rows=rows, base_config_path=base_config_path)
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description="Focused Circleworld replay-safe child-survival sweep.")
    ap.add_argument("--base-config", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--time-steps", type=int, default=128)
    ap.add_argument("--clip-seconds", type=int, default=10)
    ap.add_argument("--phase-blend", type=float, default=1.0)
    ap.add_argument("--support-window-values", default="24,26,28")
    ap.add_argument("--support-decay-values", default="0.010,0.014,0.018")
    ap.add_argument("--writeback-gain-values", default="0.56,0.64")
    ap.add_argument("--benchmark-top-k", type=int, default=6)
    args = ap.parse_args()

    summary = run_sweep(
        base_config_path=Path(args.base_config),
        out_dir=Path(args.out_dir),
        device_name=args.device,
        time_steps=int(args.time_steps),
        clip_seconds=int(args.clip_seconds),
        phase_blend=float(args.phase_blend),
        support_window_values=_parse_int_list(args.support_window_values),
        support_decay_values=_parse_float_list(args.support_decay_values),
        writeback_gain_values=_parse_float_list(args.writeback_gain_values),
        benchmark_top_k=int(args.benchmark_top_k),
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
