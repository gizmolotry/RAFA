from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
CORE = ROOT / "core"
LINEAGE = ROOT / "lineages" / "04_positive_replacement"
for path in (ROOT, CORE, LINEAGE):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))



def _exit_if_help_requested_without_runtime() -> None:
    if any(arg in ("-h", "--help") for arg in sys.argv[1:]):
        print(f"usage: {Path(__file__).name} [runtime-coupled contract options]")
        print()
        print("This contract is coupled to the Circleworld runtime/signature lane.")
        print("Run it from a Circleworld integration worktree for full argument parsing and execution.")
        raise SystemExit(0)


_exit_if_help_requested_without_runtime()

from ablate_formalization import _generate_seed_phase, make_seed_rafa
from export_circleworld_audio import _load_circle_cfg
from test_nested_commitment import _child_signal_metrics, _run_depth_trace


def _safe_device(requested: str) -> torch.device:
    if requested == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _parse_config_spec(spec: str) -> tuple[str, Path]:
    if "=" in spec:
        label, path = spec.split("=", 1)
        return label.strip(), Path(path.strip())
    path = Path(spec)
    label = path.parent.name if path.name.endswith(".json") else path.stem
    return label, path


def _float(v: Any, default: float = 0.0) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return float(default)


def _audit_one_seed(
    *,
    config_path: Path,
    label: str,
    seed_source: str,
    seed: int,
    time_steps: int,
    depth: int,
    mode: str,
    live_child_threshold: float,
    device: torch.device,
) -> dict[str, Any]:
    cfg = _load_circle_cfg(config_path)
    run_mode = cfg.branching_mode if cfg.branching_mode in {"native_multimode", "native_multimode_childworld"} else mode
    rafa_core = make_seed_rafa(dev=device.type) if seed_source == "naked_rafa" else None
    z0 = _generate_seed_phase(
        rafa_core=rafa_core,
        batch_size=1,
        time_steps=int(time_steps),
        device=device,
        seed_source=seed_source,
        seed=int(seed),
    ).detach()
    trace = _run_depth_trace(z0, cfg=cfg, depth=max(1, int(depth)), mode=run_mode)
    rows: list[dict[str, Any]] = []
    best: dict[str, Any] | None = None
    for idx, block in enumerate(trace["history"], start=1):
        signal = _child_signal_metrics(block, live_child_threshold)
        row = {
            "depth": int(idx),
            "live_child_fraction": _float(signal.get("live_child_fraction")),
            "child_world_count": _float(signal.get("child_world_count")),
            "child_score": _float(signal.get("child_score")),
            "live_child_gate_passed": bool(_float(signal.get("live_child_gate_passed")) > 0.0),
        }
        rows.append(row)
        if best is None or row["child_score"] > _float(best.get("child_score")):
            best = row
    best = dict(best or {})
    return {
        "config_label": label,
        "config_path": str(config_path),
        "branching_mode": str(getattr(cfg, "branching_mode", "")),
        "child_local_ifs_enabled": bool(getattr(cfg, "child_local_ifs_enabled", False)),
        "child_local_steps": int(getattr(cfg, "child_local_steps", 0)),
        "child_local_coherence_retention_enabled": bool(getattr(cfg, "child_local_coherence_retention_enabled", False)),
        "child_local_coherence_retention_mix": _float(getattr(cfg, "child_local_coherence_retention_mix", 0.0)),
        "child_local_coherence_floor": _float(getattr(cfg, "child_local_coherence_floor", 0.0)),
        "child_local_coherence_floor_support": _float(getattr(cfg, "child_local_coherence_floor_support", 0.18)),
        "child_local_coherence_causal_gate_enabled": bool(
            getattr(cfg, "child_local_coherence_causal_gate_enabled", False)
        ),
        "child_local_coherence_causal_min_delta": _float(
            getattr(cfg, "child_local_coherence_causal_min_delta", 0.02)
        ),
        "child_local_coherence_causal_full_delta": _float(
            getattr(cfg, "child_local_coherence_causal_full_delta", 0.18)
        ),
        "child_local_coherence_causal_gate_floor": _float(
            getattr(cfg, "child_local_coherence_causal_gate_floor", 0.0)
        ),
        "run_mode": str(run_mode),
        "seed_source": seed_source,
        "seed": int(seed),
        "time_steps": int(time_steps),
        "depth": int(depth),
        "live_child_threshold": float(live_child_threshold),
        "per_depth": rows,
        "best_depth": int(best.get("depth", 0) or 0),
        "best_child_score": _float(best.get("child_score")),
        "best_live_child_fraction": _float(best.get("live_child_fraction")),
        "best_child_world_count": _float(best.get("child_world_count")),
        "best_live_child_gate_passed": bool(best.get("live_child_gate_passed", False)),
        "any_live_child_gate_passed": any(bool(row["live_child_gate_passed"]) for row in rows),
        "any_child_signal": any(_float(row["child_score"]) > 0.0 for row in rows),
    }


def _aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {
            "seed_count": 0,
            "live_seed_fraction": 0.0,
            "signal_seed_fraction": 0.0,
            "mean_best_child_score": 0.0,
            "mean_best_live_child_fraction": 0.0,
            "mean_best_child_world_count": 0.0,
            "mean_live_depth_fraction": 0.0,
        }
    live_depth_flags = [
        1.0 if bool(depth_row.get("live_child_gate_passed", False)) else 0.0
        for row in rows
        for depth_row in row.get("per_depth", [])
    ]
    return {
        "seed_count": int(len(rows)),
        "live_seed_fraction": float(np.mean([1.0 if row.get("any_live_child_gate_passed") else 0.0 for row in rows])),
        "signal_seed_fraction": float(np.mean([1.0 if row.get("any_child_signal") else 0.0 for row in rows])),
        "mean_best_child_score": float(np.mean([_float(row.get("best_child_score")) for row in rows])),
        "mean_best_live_child_fraction": float(np.mean([_float(row.get("best_live_child_fraction")) for row in rows])),
        "mean_best_child_world_count": float(np.mean([_float(row.get("best_child_world_count")) for row in rows])),
        "mean_live_depth_fraction": float(np.mean(live_depth_flags)) if live_depth_flags else 0.0,
    }


def _write_markdown(payload: dict[str, Any], path: Path) -> None:
    lines = [
        "# Seeded Child-Substrate Audit",
        "",
        "This audit replays the fixed seeded substrate and records whether live child worlds exist at each recurrence depth.",
        "",
        "## Aggregate",
        "",
        "| checkpoint | live seed fraction | signal seed fraction | mean best score | mean best live child | mean child worlds | live depth fraction |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for label, agg in payload.get("aggregate_by_config", {}).items():
        lines.append(
            f"| {label} | {agg.get('live_seed_fraction', 0.0):.4f} | "
            f"{agg.get('signal_seed_fraction', 0.0):.4f} | "
            f"{agg.get('mean_best_child_score', 0.0):.6f} | "
            f"{agg.get('mean_best_live_child_fraction', 0.0):.4f} | "
            f"{agg.get('mean_best_child_world_count', 0.0):.4f} | "
            f"{agg.get('mean_live_depth_fraction', 0.0):.4f} |"
        )
    lines.extend(["", "## Per Seed", ""])
    for row in payload.get("rows", []):
        lines.append(
            f"- `{row['config_label']}` seed `{row['seed']}`: best_depth={row['best_depth']}, "
            f"best_score={row['best_child_score']:.6f}, live={row['any_live_child_gate_passed']}, "
            f"signal={row['any_child_signal']}"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description="Audit fixed seeded Circleworld child-world substrate stability.")
    ap.add_argument("--config", action="append", required=True, help="Config path or label=path. Repeatable.")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed-source", default="naked_rafa")
    ap.add_argument("--seeds", default="9100,9101,9102")
    ap.add_argument("--time-steps", type=int, default=128)
    ap.add_argument("--depth", type=int, default=5)
    ap.add_argument("--mode", default="native_multimode_childworld", choices=["active_packets", "native_multimode", "native_multimode_childworld"])
    ap.add_argument("--live-child-threshold", type=float, default=0.03)
    args = ap.parse_args()

    device = _safe_device(args.device)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    seeds = [int(part.strip()) for part in str(args.seeds).split(",") if part.strip()]
    specs = [_parse_config_spec(spec) for spec in args.config]

    rows: list[dict[str, Any]] = []
    for label, config_path in specs:
        for seed in seeds:
            rows.append(
                _audit_one_seed(
                    config_path=config_path,
                    label=label,
                    seed_source=str(args.seed_source),
                    seed=int(seed),
                    time_steps=int(args.time_steps),
                    depth=int(args.depth),
                    mode=str(args.mode),
                    live_child_threshold=float(args.live_child_threshold),
                    device=device,
                )
            )

    aggregate_by_config: dict[str, Any] = {}
    for label, _ in specs:
        aggregate_by_config[label] = _aggregate([row for row in rows if row.get("config_label") == label])
    payload = {
        "schema": "circleworld_seeded_child_substrate_audit_v1",
        "device": str(device),
        "seed_source": str(args.seed_source),
        "seeds": seeds,
        "time_steps": int(args.time_steps),
        "depth": int(args.depth),
        "mode": str(args.mode),
        "live_child_threshold": float(args.live_child_threshold),
        "aggregate_by_config": aggregate_by_config,
        "rows": rows,
    }
    (out_dir / "seeded_child_substrate_audit.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    _write_markdown(payload, out_dir / "SEEDED_CHILD_SUBSTRATE_AUDIT.md")
    print(json.dumps({"out_dir": str(out_dir), "aggregate_by_config": aggregate_by_config}, indent=2))


if __name__ == "__main__":
    main()
