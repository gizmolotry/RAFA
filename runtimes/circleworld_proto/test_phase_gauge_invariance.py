from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Callable

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
from circleworld import circleworld_loss, recurse_circleworld, summarize_circleworld_run
from evaluate_circleworld import _load_circle_cfg, _safe_device


Q_METRICS = [
    "dominant_q_share",
    "q_entropy",
    "loss",
    "l_q_dom",
    "l_q_entropy",
    "num_law_families",
    "law_family_entropy",
    "dominant_law_q_share",
    "law_top_q_entropy",
    "num_law_top_q_unique",
]

ARC_METRICS = [
    "initial_major_mass",
    "final_major_mass",
    "initial_minor_residue",
    "final_minor_residue",
    "initial_promotability",
    "final_promotability",
    "major_gain",
    "residue_drop",
    "promotability_gain",
    "major_saturation",
    "final_harmonic_ratio",
    "final_concentration",
    "final_sharpness",
    "l_major_sat",
]

BRANCH_METRICS = [
    "mean_real_branch_fraction",
    "parent_real_branch_fraction",
    "child_real_branch_fraction",
    "real_branch_fraction",
    "meso_branch_effect",
    "mean_branch_positive_mask",
    "mean_branch_negative_mask",
    "mean_branch_defect",
    "mean_branch_world_grad",
    "mean_branch_q_disagreement",
    "mean_branch_phase_wall",
    "mean_branch_seed_energy",
    "mean_phase_only_branch_surface",
    "mean_phase_only_branch_surface_peak",
    "mean_phase_only_branch_phase_wall",
    "mean_phase_only_branch_support_balance",
    "mean_phase_only_branch_distinctness",
    "mean_phase_only_branch_distinctness_balanced",
    "phase_only_real_branch_fraction",
    "phase_only_excess_branch_fraction",
    "decorative_slot2_low_phase_fraction",
    "final_phase_only_branch_surface",
    "final_phase_only_branch_surface_peak",
    "final_phase_only_branch_phase_wall",
    "final_phase_only_branch_live_fraction",
    "final_phase_only_branch_support_balance",
    "final_phase_only_branch_distinctness",
    "final_phase_only_branch_distinctness_balanced",
    "mean_child_world_count",
    "mean_live_child_fraction",
    "mean_child_age",
    "mean_child_writeback_mass",
    "mean_child_support_writeback_mass",
    "mean_child_parent_divergence",
    "mean_child_sibling_divergence",
    "child_spawn_count",
    "child_kill_count",
    "child_writeback_count",
    "child_max_age",
]

METRIC_GROUPS = {
    "q": Q_METRICS,
    "arc": ARC_METRICS,
    "branch": BRANCH_METRICS,
}

ALL_SCALAR_METRICS = list(dict.fromkeys(Q_METRICS + ARC_METRICS + BRANCH_METRICS))


def _phasor_normalize(z: torch.Tensor) -> torch.Tensor:
    return z / torch.linalg.vector_norm(z, dim=-1, keepdim=True).clamp_min(1e-8)


def _rotate_phasor(z: torch.Tensor, angle: float) -> torch.Tensor:
    cos_a = math.cos(float(angle))
    sin_a = math.sin(float(angle))
    x = z[..., 0]
    y = z[..., 1]
    return torch.stack((x * cos_a - y * sin_a, x * sin_a + y * cos_a), dim=-1)


def _phase_preserving_rescale(z: torch.Tensor, scale: float) -> torch.Tensor:
    return _phasor_normalize(z * float(scale))


def _phase_preserving_taper(z: torch.Tensor) -> torch.Tensor:
    f_bins = z.size(1)
    t_len = z.size(2)
    freq = torch.linspace(0.75, 1.25, steps=f_bins, device=z.device, dtype=z.dtype).view(1, f_bins, 1, 1)
    time = torch.linspace(1.20, 0.80, steps=t_len, device=z.device, dtype=z.dtype).view(1, 1, t_len, 1)
    return _phasor_normalize(z * freq * time)


def _tensor_mean(value: torch.Tensor) -> float:
    return float(value.detach().mean().item())


def _q_mass(final_arc: dict[str, torch.Tensor]) -> list[float]:
    per_q_abs = final_arc["per_q"].detach().abs()
    q_mass = per_q_abs.mean(dim=(0, 2))
    q_mass = q_mass / q_mass.sum().clamp_min(1e-8)
    return [float(v) for v in q_mass.detach().cpu().tolist()]


def _evaluate_phase_state(
    phase_state: torch.Tensor,
    cfg: Any,
) -> dict[str, Any]:
    run_mode = cfg.branching_mode if cfg.branching_mode in {"native_multimode", "native_multimode_childworld"} else "active_packets"
    run = recurse_circleworld(phase_state, cfg=cfg, depth=cfg.recursion_depth, mode=run_mode)
    summary = summarize_circleworld_run(run)
    losses = circleworld_loss(run)
    final_arc = run["final_arc"]

    scalars: dict[str, float] = {}
    scalars.update({key: float(value) for key, value in summary.items() if isinstance(value, (int, float))})
    scalars.update(
        {
            "loss": float(losses["loss"].detach().item()),
            "l_q_dom": float(losses["l_q_dom"].detach().item()),
            "l_q_entropy": float(losses["l_q_entropy"].detach().item()),
            "l_major_sat": float(losses["l_major_sat"].detach().item()),
            "final_harmonic_ratio": _tensor_mean(final_arc["harmonic_ratio"]),
            "final_concentration": _tensor_mean(final_arc["concentration"]),
            "final_sharpness": _tensor_mean(final_arc["sharpness"]),
        }
    )

    return {
        "mode": run_mode,
        "scalar_metrics": {key: scalars[key] for key in ALL_SCALAR_METRICS if key in scalars},
        "q_mass": _q_mass(final_arc),
        "packet_counts": [len(level) for level in run["packets"]],
    }


def _scalar_diff(baseline: dict[str, float], variant: dict[str, float]) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for key in ALL_SCALAR_METRICS:
        if key not in baseline or key not in variant:
            continue
        signed = float(variant[key] - baseline[key])
        out[key] = {"signed": signed, "abs": abs(signed)}
    return out


def _group_max_abs(diff: dict[str, dict[str, float]]) -> dict[str, float]:
    grouped: dict[str, float] = {}
    for group, keys in METRIC_GROUPS.items():
        grouped[group] = max((diff[key]["abs"] for key in keys if key in diff), default=0.0)
    return grouped


def _vector_diff(baseline: list[float], variant: list[float]) -> dict[str, float]:
    pairs = list(zip(baseline, variant))
    abs_diffs = [abs(float(right) - float(left)) for left, right in pairs]
    return {
        "q_mass_l1": float(sum(abs_diffs)),
        "q_mass_linf": float(max(abs_diffs) if abs_diffs else 0.0),
    }


def _jsonable(value: Any) -> Any:
    if torch.is_tensor(value):
        if value.numel() == 1:
            return float(value.detach().cpu().item())
        return value.detach().cpu().tolist()
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_jsonable(v) for v in value]
    if isinstance(value, tuple):
        return [_jsonable(v) for v in value]
    return value


def _parse_seeds(raw: str, source: str) -> list[int]:
    raw = str(raw).strip()
    if not raw:
        raise ValueError("--seeds must be a positive count or a comma-separated seed list")
    if "," in raw:
        return [int(part.strip()) for part in raw.split(",") if part.strip()]
    count = int(raw)
    if count <= 0:
        raise ValueError("--seeds count must be positive")
    base = 6100 if source == "naked_rafa" else 5100
    return [base + idx for idx in range(count)]


def _seed_plan(seed_source: str, seeds: str) -> list[tuple[str, int]]:
    sources = ["synthetic", "naked_rafa"] if seed_source == "both" else [seed_source]
    plan: list[tuple[str, int]] = []
    for source in sources:
        plan.extend((source, seed) for seed in _parse_seeds(seeds, source))
    return plan


def _transform_specs() -> list[tuple[str, dict[str, Any], Callable[[torch.Tensor], torch.Tensor]]]:
    return [
        (
            "global_rotation_pi_over_7",
            {"kind": "global_phasor_rotation", "angle_radians": math.pi / 7.0},
            lambda z: _rotate_phasor(z, math.pi / 7.0),
        ),
        (
            "global_rotation_pi_over_2",
            {"kind": "global_phasor_rotation", "angle_radians": math.pi / 2.0},
            lambda z: _rotate_phasor(z, math.pi / 2.0),
        ),
        (
            "global_rotation_minus_2pi_over_3",
            {"kind": "global_phasor_rotation", "angle_radians": -2.0 * math.pi / 3.0},
            lambda z: _rotate_phasor(z, -2.0 * math.pi / 3.0),
        ),
        (
            "phase_preserving_rescale_half",
            {"kind": "phase_preserving_uniform_rescale", "scale": 0.5},
            lambda z: _phase_preserving_rescale(z, 0.5),
        ),
        (
            "phase_preserving_rescale_double",
            {"kind": "phase_preserving_uniform_rescale", "scale": 2.0},
            lambda z: _phase_preserving_rescale(z, 2.0),
        ),
        (
            "phase_preserving_freq_time_taper",
            {"kind": "phase_preserving_nonuniform_rescale", "freq_range": [0.75, 1.25], "time_range": [1.20, 0.80]},
            _phase_preserving_taper,
        ),
    ]


def _aggregate(cases: list[dict[str, Any]]) -> dict[str, Any]:
    by_variant: dict[str, list[dict[str, Any]]] = {}
    for case in cases:
        for variant in case["variants"]:
            by_variant.setdefault(variant["name"], []).append(variant)

    aggregate: dict[str, Any] = {}
    for name, rows in by_variant.items():
        metric_names = sorted({key for row in rows for key in row["diff"]["scalar_metrics"]})
        metric_summary: dict[str, dict[str, float]] = {}
        for metric in metric_names:
            signed = [row["diff"]["scalar_metrics"][metric]["signed"] for row in rows if metric in row["diff"]["scalar_metrics"]]
            abs_vals = [abs(v) for v in signed]
            metric_summary[metric] = {
                "mean_signed": float(sum(signed) / max(1, len(signed))),
                "mean_abs": float(sum(abs_vals) / max(1, len(abs_vals))),
                "max_abs": float(max(abs_vals) if abs_vals else 0.0),
            }

        aggregate[name] = {
            "num_cases": int(len(rows)),
            "metrics": metric_summary,
            "groups": {
                group: {
                    "mean_max_abs": float(sum(row["diff"]["group_max_abs"].get(group, 0.0) for row in rows) / max(1, len(rows))),
                    "max_abs": float(max((row["diff"]["group_max_abs"].get(group, 0.0) for row in rows), default=0.0)),
                }
                for group in METRIC_GROUPS
            },
            "q_mass": {
                "mean_l1": float(sum(row["diff"]["q_mass"]["q_mass_l1"] for row in rows) / max(1, len(rows))),
                "max_l1": float(max((row["diff"]["q_mass"]["q_mass_l1"] for row in rows), default=0.0)),
                "mean_linf": float(sum(row["diff"]["q_mass"]["q_mass_linf"] for row in rows) / max(1, len(rows))),
                "max_linf": float(max((row["diff"]["q_mass"]["q_mass_linf"] for row in rows), default=0.0)),
            },
        }
    return aggregate


def _brief_aggregate(aggregate: dict[str, Any]) -> dict[str, Any]:
    brief: dict[str, Any] = {}
    for name, row in aggregate.items():
        groups = row.get("groups", {})
        q_mass = row.get("q_mass", {})
        brief[name] = {
            "num_cases": int(row.get("num_cases", 0)),
            "group_max_abs": {group: float(values.get("max_abs", 0.0)) for group, values in groups.items()},
            "q_mass_max_l1": float(q_mass.get("max_l1", 0.0)),
            "q_mass_max_linf": float(q_mass.get("max_linf", 0.0)),
        }
    return brief


def run_phase_gauge_invariance(
    *,
    config_path: Path,
    out_dir: Path,
    device_name: str,
    time_steps: int,
    seed_source: str,
    seeds: str,
) -> dict[str, Any]:
    cfg = _load_circle_cfg(config_path)
    device = _safe_device(device_name)
    plan = _seed_plan(seed_source, seeds)
    needs_rafa = any(source == "naked_rafa" for source, _ in plan)
    rafa_core = make_seed_rafa(dev=device.type) if needs_rafa else None
    transforms = _transform_specs()

    cases: list[dict[str, Any]] = []
    with torch.no_grad():
        for source, seed in plan:
            phase_state = _generate_seed_phase(
                rafa_core=rafa_core,
                batch_size=1,
                time_steps=time_steps,
                device=device,
                seed_source=source,
                seed=seed,
            ).detach()
            baseline = _evaluate_phase_state(phase_state, cfg)
            variants = []
            for name, transform, fn in transforms:
                metrics = _evaluate_phase_state(fn(phase_state).detach(), cfg)
                scalar_diff = _scalar_diff(baseline["scalar_metrics"], metrics["scalar_metrics"])
                variants.append(
                    {
                        "name": name,
                        "transform": transform,
                        "metrics": metrics,
                        "diff": {
                            "scalar_metrics": scalar_diff,
                            "group_max_abs": _group_max_abs(scalar_diff),
                            "q_mass": _vector_diff(baseline["q_mass"], metrics["q_mass"]),
                            "packet_counts_changed": metrics["packet_counts"] != baseline["packet_counts"],
                        },
                    }
                )
            cases.append(
                {
                    "source": source,
                    "seed": int(seed),
                    "baseline": baseline,
                    "variants": variants,
                }
            )

    report = {
        "runtime": "circleworld_proto",
        "test": "phase_gauge_invariance_v1",
        "config_path": str(config_path),
        "device": str(device),
        "requested_device": str(device_name),
        "time_steps": int(time_steps),
        "seed_source": seed_source,
        "seed_plan": [{"source": source, "seed": int(seed)} for source, seed in plan],
        "metric_groups": METRIC_GROUPS,
        "notes": [
            "Global phasor rotations should preserve Circleworld metrics that depend only on relative phase.",
            "Phase-preserving rescale controls are normalized before evaluation; drift there indicates normalization/control sensitivity.",
        ],
        "cases": cases,
        "aggregate_differences": _aggregate(cases),
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = out_dir / "phase_gauge_invariance_report.json"
    report["report_path"] = str(report_path)
    report_path.write_text(json.dumps(_jsonable(report), indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Circleworld phase-only/gauge-invariance checks.")
    parser.add_argument("--config", required=True, help="Circleworld config JSON.")
    parser.add_argument("--out-dir", required=True, help="Directory for phase gauge report JSON.")
    parser.add_argument("--device", default="cuda", help="Requested torch device; falls back to CPU if CUDA is unavailable.")
    parser.add_argument("--time-steps", type=int, default=128)
    parser.add_argument("--seed-source", choices=["synthetic", "naked_rafa", "both"], default="synthetic")
    parser.add_argument("--seeds", default="3", help="Positive count per source, or a comma-separated explicit seed list.")
    args = parser.parse_args()

    report = run_phase_gauge_invariance(
        config_path=Path(args.config),
        out_dir=Path(args.out_dir),
        device_name=str(args.device),
        time_steps=int(args.time_steps),
        seed_source=str(args.seed_source),
        seeds=str(args.seeds),
    )
    print(json.dumps(_jsonable({
        "report_path": report["report_path"],
        "device": report["device"],
        "time_steps": report["time_steps"],
        "seed_plan": report["seed_plan"],
        "aggregate_brief": _brief_aggregate(report["aggregate_differences"]),
    }), indent=2))


if __name__ == "__main__":
    main()
