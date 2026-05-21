from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any, Sequence

ROOT = Path(__file__).resolve().parents[2]
CORE = ROOT / "core"
LINEAGE = ROOT / "lineages" / "04_positive_replacement"
RUNTIME = ROOT / "runtimes" / "circleworld_proto"
for path in (ROOT, CORE, LINEAGE, RUNTIME):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from common_io import as_float as _as_float  # noqa: E402
from common_io import mean as _mean  # noqa: E402
from common_io import parse_gain_csv as _parse_csv_floats  # noqa: E402
from benchmark_audio_continuation import _jsonify, _load_circle_cfg  # noqa: E402
from run_audio_circle_delta_probe import CORE_MASK_MODES, MASK_MODES, run_probe  # noqa: E402


OUTPUT_JSON = "audio_delta_objective_scout.json"
OUTPUT_MD = "AUDIO_DELTA_OBJECTIVE_SCOUT.md"


FIXED_CANDIDATES_V1: dict[str, dict[str, Any]] = {
    "copyphase_delta_base": {},
    "phase_writeback_off_support_only": {
        "child_writeback_phase_delta_gain": 0.0,
        "child_operator_seed_gain": 0.0,
        "child_operator_promotability_gain": 0.0,
        "child_writeback_operator_mix": 0.0,
        "child_writeback_phase_floor_gain": 0.0,
    },
    "raw_child_phase_only": {
        "child_support_writeback_gain": 0.0,
        "child_operator_seed_gain": 0.0,
        "child_operator_promotability_gain": 0.0,
        "child_writeback_operator_mix": 0.0,
        "child_writeback_phase_floor_gain": 0.0,
    },
    "operator_floor_only": {
        "child_support_writeback_gain": 0.0,
        "child_writeback_phase_delta_gain": 0.0,
    },
    "support_operator_no_floor": {
        "child_writeback_phase_floor_gain": 0.0,
    },
    "support_operator_floor_high": {
        "child_writeback_phase_floor_gain": 3.25,
    },
    "branch_conservative_delta_low": {
        "split_pressure": 0.08,
        "split_seed_scale": 0.25,
        "defect_phase_gain": 0.25,
        "instability_seed_scale": 0.06,
        "merge_pressure": 0.20,
    },
    "branch_aggressive_delta_high": {
        "split_pressure": 0.26,
        "split_seed_scale": 0.90,
        "defect_phase_gain": 0.60,
        "instability_seed_scale": 0.20,
        "merge_pressure": 0.10,
    },
    "relation_qtrace_only_control": {
        "branch_kernel_version": "qtrace_only",
    },
    "relation_phase_only_control": {
        "branch_kernel_version": "phase_only",
    },
    "compact_fourierish_q_basis": {
        "branch_kernel_version": "ramanujan",
        "qset": [2, 4, 8, 12],
        "q_weights": [
            1.9012945150466065,
            1.9012945150466065,
            1.9012945150466065,
            1.9012945150466065,
        ],
        "q_trace_rank": 4,
    },
}


def _parse_csv(raw: str, allowed: Sequence[str], label: str) -> list[str]:
    values = [item.strip() for item in raw.split(",") if item.strip()]
    if not values:
        raise ValueError(f"At least one {label} is required")
    unknown = sorted(set(values) - set(allowed))
    if unknown:
        raise ValueError(f"Unknown {label}: {unknown}; expected one of {tuple(allowed)}")
    return values
def _load_candidate_config(base_config_path: Path, overrides: dict[str, Any]) -> dict[str, Any]:
    cfg = _load_circle_cfg(base_config_path)
    payload = asdict(cfg)
    payload.update(overrides)
    return {
        "runtime": "circleworld_proto",
        "schema": "circleworld_v0",
        "schema_note": "audio_delta_objective_scout_candidate_v0",
        "base_config_path": str(base_config_path),
        "candidate_overrides": overrides,
        "config": payload,
    }


def _write_candidate_config(base_config_path: Path, candidate_dir: Path, overrides: dict[str, Any]) -> Path:
    candidate_dir.mkdir(parents=True, exist_ok=True)
    config_path = candidate_dir / "circleworld_config.json"
    payload = _jsonify(_load_candidate_config(base_config_path, overrides))
    config_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return config_path


def _row_key(row: dict[str, Any]) -> tuple[str, str, float]:
    return (
        str(row.get("magnitude_mode")),
        str(row.get("mask_mode")),
        round(_as_float(row.get("gain")), 8),
    )


def _find_row(
    aggregate: Sequence[dict[str, Any]],
    *,
    magnitude_mode: str,
    mask_mode: str,
    gain: float,
) -> dict[str, Any]:
    target = (magnitude_mode, mask_mode, round(float(gain), 8))
    for row in aggregate:
        if _row_key(row) == target:
            return row
    return {}


def _case_objective_deltas(
    probe: dict[str, Any],
    *,
    magnitude_mode: str,
    mask_mode: str,
    gain: float,
) -> list[dict[str, float | str]]:
    rows: list[dict[str, float | str]] = []
    for case in probe.get("cases", []) or []:
        if not isinstance(case, dict):
            continue
        case_rows = [row for row in case.get("rows", []) or [] if isinstance(row, dict)]
        selected = _find_row(case_rows, magnitude_mode=magnitude_mode, mask_mode=mask_mode, gain=gain)
        baseline = _find_row(case_rows, magnitude_mode=magnitude_mode, mask_mode=mask_mode, gain=0.0)
        if not selected or not baseline:
            continue
        corr_delta = _as_float(selected.get("target_corr")) - _as_float(baseline.get("target_corr"))
        mse_delta = _as_float(selected.get("target_mse")) - _as_float(baseline.get("target_mse"))
        rows.append(
            {
                "name": str(case.get("name", selected.get("name", ""))),
                "target_corr_delta_vs_gain0": corr_delta,
                "target_mse_delta_vs_gain0": mse_delta,
            }
        )
    return rows


def _median(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(float(value) for value in values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[mid]
    return 0.5 * (ordered[mid - 1] + ordered[mid])


def _case_robustness(case_deltas: Sequence[dict[str, float | str]]) -> dict[str, Any]:
    corr = [_as_float(row.get("target_corr_delta_vs_gain0")) for row in case_deltas]
    mse = [_as_float(row.get("target_mse_delta_vs_gain0")) for row in case_deltas]
    total_gain = sum(max(0.0, value) for value in corr)
    max_share = max((max(0.0, value) / total_gain for value in corr), default=0.0) if total_gain > 1.0e-12 else 0.0
    leave_one_out = []
    if len(corr) > 1:
        for idx in range(len(corr)):
            subset = [value for j, value in enumerate(corr) if j != idx]
            leave_one_out.append(_mean(subset))
    return {
        "case_count": len(case_deltas),
        "mean_corr_delta": _mean(corr),
        "median_corr_delta": _median(corr),
        "min_corr_delta": min(corr, default=0.0),
        "corr_positive_fraction": _mean([1.0 if value > 0.0 else 0.0 for value in corr]),
        "max_positive_corr_gain_share": max_share,
        "min_leave_one_out_corr_delta": min(leave_one_out, default=_mean(corr)),
        "mean_mse_delta": _mean(mse),
        "median_mse_delta": _median(mse),
        "mse_improve_fraction": _mean([1.0 if value < 0.0 else 0.0 for value in mse]),
        "rows": list(case_deltas),
    }


def _leakage_detected(payload: dict[str, Any]) -> bool:
    return bool(payload.get("future_target_magnitude_reused")) or bool(
        payload.get("target_future_stft_magnitude_accessed")
    ) or bool(payload.get("future_target_phase_reused")) or bool(payload.get("target_future_stft_phase_accessed"))


def _score_probe(
    probe: dict[str, Any],
    *,
    objective_mode: str,
    objective_mask: str,
    objective_gain: float,
    min_corr_delta: float,
    min_corr_win_fraction: float,
    max_loop_delta: float,
    max_reentry_delta: float,
) -> dict[str, Any]:
    aggregate = [row for row in probe.get("aggregate", []) or [] if isinstance(row, dict)]
    primary = _find_row(aggregate, magnitude_mode=objective_mode, mask_mode=objective_mask, gain=objective_gain)
    primary_base = _find_row(aggregate, magnitude_mode=objective_mode, mask_mode=objective_mask, gain=0.0)
    nonzero = [row for row in aggregate if abs(_as_float(row.get("gain"))) > 1.0e-12]
    case_deltas = _case_objective_deltas(
        probe,
        magnitude_mode=objective_mode,
        mask_mode=objective_mask,
        gain=objective_gain,
    )
    robustness = _case_robustness(case_deltas)

    leakage = _leakage_detected(probe)
    corr_delta = _as_float(primary.get("mean_target_corr_delta_vs_gain0"))
    mse_delta = _as_float(primary.get("mean_target_mse_delta_vs_gain0"))
    corr_wins = _as_float(primary.get("target_corr_win_fraction_vs_gain0"))
    mse_wins = _as_float(primary.get("target_mse_win_fraction_vs_gain0"))
    carrier_corr = _as_float(primary.get("mean_vs_gain0_corr"))
    loop_delta = _as_float(primary.get("mean_loop_autocorr_peak")) - _as_float(primary_base.get("mean_loop_autocorr_peak"))
    reentry_delta = _as_float(primary.get("mean_first_chunk_reentry")) - _as_float(
        primary_base.get("mean_first_chunk_reentry")
    )

    mean_corr_delta = _mean([_as_float(row.get("mean_target_corr_delta_vs_gain0")) for row in nonzero])
    mean_mse_delta = _mean([_as_float(row.get("mean_target_mse_delta_vs_gain0")) for row in nonzero])
    corr_positive_fraction = _mean(
        [1.0 if _as_float(row.get("mean_target_corr_delta_vs_gain0")) > 0.0 else 0.0 for row in nonzero]
    )
    mse_positive_fraction = _mean(
        [1.0 if _as_float(row.get("mean_target_mse_delta_vs_gain0")) < 0.0 else 0.0 for row in nonzero]
    )

    # Transparent ranking score, not a success criterion. Success uses thresholds below.
    score = 0.0
    score += 1000.0 * corr_delta
    score += 10000.0 * max(0.0, -mse_delta)
    score -= 10000.0 * max(0.0, mse_delta)
    score += 3.0 * (corr_wins - 0.5)
    score += 1.5 * (mse_wins - 0.5)
    score += 300.0 * mean_corr_delta
    score += 1.0 * (corr_positive_fraction - 0.5)
    score -= 5.0 * max(0.0, loop_delta - max_loop_delta)
    score -= 2.0 * max(0.0, reentry_delta - max_reentry_delta)
    if carrier_corr >= 0.95 and abs(corr_delta) <= 0.005:
        score -= 1.0
    if leakage:
        score -= 1.0e6

    passes_primary = (
        not leakage
        and corr_delta >= min_corr_delta
        and corr_wins >= min_corr_win_fraction
        and _as_float(robustness.get("median_corr_delta")) > 0.0
        and _as_float(robustness.get("min_leave_one_out_corr_delta")) > 0.0
        and _as_float(robustness.get("max_positive_corr_gain_share")) <= 0.50
        and mse_delta <= 0.0
        and loop_delta <= max_loop_delta
        and reentry_delta <= max_reentry_delta
    )
    if passes_primary:
        status = "audio_delta_candidate"
    elif leakage:
        status = "invalid_future_leakage"
    elif mse_delta < 0.0 and corr_delta > 0.0:
        status = "weak_delta_tradeoff_signal"
    elif mse_delta < 0.0:
        status = "mse_only_or_corr_tradeoff"
    else:
        status = "not_improved"

    best_corr_row = max(nonzero, key=lambda row: _as_float(row.get("mean_target_corr_delta_vs_gain0")), default={})
    best_mse_row = min(nonzero, key=lambda row: _as_float(row.get("mean_target_mse_delta_vs_gain0")), default={})
    return {
        "status": status,
        "score": score,
        "probe_status": probe.get("status"),
        "objective_mode": objective_mode,
        "objective_mask": objective_mask,
        "objective_gain": float(objective_gain),
        "primary_row": primary,
        "primary_gain0_row": primary_base,
        "primary_corr_delta": corr_delta,
        "primary_mse_delta": mse_delta,
        "primary_corr_win_fraction": corr_wins,
        "primary_mse_win_fraction": mse_wins,
        "primary_vs_gain0_corr": carrier_corr,
        "primary_loop_delta": loop_delta,
        "primary_reentry_delta": reentry_delta,
        "case_robustness": robustness,
        "mean_corr_delta_all_nonzero": mean_corr_delta,
        "mean_mse_delta_all_nonzero": mean_mse_delta,
        "corr_positive_row_fraction": corr_positive_fraction,
        "mse_positive_row_fraction": mse_positive_fraction,
        "best_corr_delta": best_corr_row.get("mean_target_corr_delta_vs_gain0"),
        "best_corr_delta_row": best_corr_row,
        "best_mse_delta": best_mse_row.get("mean_target_mse_delta_vs_gain0"),
        "best_mse_delta_row": best_mse_row,
        "future_target_magnitude_reused": bool(probe.get("future_target_magnitude_reused")),
        "target_future_stft_magnitude_accessed": bool(probe.get("target_future_stft_magnitude_accessed")),
        "future_target_phase_reused": bool(probe.get("future_target_phase_reused")),
        "target_future_stft_phase_accessed": bool(probe.get("target_future_stft_phase_accessed")),
    }


def _candidate_items(profile: str) -> list[tuple[str, dict[str, Any]]]:
    if profile != "fixed_v1":
        raise ValueError("Only fixed_v1 candidate profile is currently implemented")
    return list(FIXED_CANDIDATES_V1.items())


def _overall_status(rows: Sequence[dict[str, Any]]) -> str:
    if any(row.get("objective", {}).get("status") == "audio_delta_candidate" for row in rows):
        return "candidate_found"
    if any(row.get("objective", {}).get("status") == "weak_delta_tradeoff_signal" for row in rows):
        return "tradeoff_signal_only"
    if any(row.get("objective", {}).get("status") == "invalid_future_leakage" for row in rows):
        return "invalid_future_leakage"
    return "no_candidate_found"


def _write_markdown(summary: dict[str, Any], path: Path) -> None:
    def fmt(value: Any) -> str:
        if value is None:
            return "NA"
        if isinstance(value, (int, float)):
            return f"{float(value):.6g}"
        return str(value)

    lines = [
        "# Audio Delta Objective Scout",
        "",
        "This scout ranks Circleworld config candidates by whether they improve over the legal gain-0 copyphase carrier.",
        "The objective is declared before ranking; all gain/mask rows remain available inside each candidate probe artifact.",
        "",
        f"- status: `{summary.get('status')}`",
        f"- candidate profile: `{summary.get('candidate_profile')}`",
        f"- cases: `{summary.get('case_count')}`",
        f"- objective row: `{summary.get('objective_mode')}` / `{summary.get('objective_mask')}` / gain `{summary.get('objective_gain')}`",
        f"- minimum corr delta: `{summary.get('min_corr_delta')}`",
        f"- minimum corr win fraction: `{summary.get('min_corr_win_fraction')}`",
        "",
        "## Ranked Candidates",
        "",
        "| rank | candidate | status | score | corr delta | MSE delta | corr wins | MSE wins | vs gain0 corr | loop d | reentry d | best corr d | best MSE d | probe | overrides |",
        "|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|",
    ]
    for index, row in enumerate(summary.get("ranked_candidates", []) or [], start=1):
        obj = row.get("objective", {}) if isinstance(row.get("objective"), dict) else {}
        lines.append(
            "| {rank} | `{name}` | `{status}` | {score} | {corr} | {mse} | {cwins} | {mwins} | {carrier} | {loop} | {reentry} | {bestcorr} | {bestmse} | `{probe_status}` | `{overrides}` |".format(
                rank=index,
                name=row.get("candidate"),
                status=obj.get("status"),
                score=fmt(obj.get("score")),
                corr=fmt(obj.get("primary_corr_delta")),
                mse=fmt(obj.get("primary_mse_delta")),
                cwins=fmt(obj.get("primary_corr_win_fraction")),
                mwins=fmt(obj.get("primary_mse_win_fraction")),
                carrier=fmt(obj.get("primary_vs_gain0_corr")),
                loop=fmt(obj.get("primary_loop_delta")),
                reentry=fmt(obj.get("primary_reentry_delta")),
                bestcorr=fmt(obj.get("best_corr_delta")),
                bestmse=fmt(obj.get("best_mse_delta")),
                probe_status=obj.get("probe_status"),
                overrides=json.dumps(row.get("overrides", {}), sort_keys=True),
            )
        )
    lines.extend(
        [
            "",
            "## Interpretation Guard",
            "",
            "- `audio_delta_candidate` requires the declared objective row to beat gain 0 on correlation, win fraction, MSE, and recurrence guards.",
            "- Candidate status also requires positive median case delta, positive leave-one-out mean, and no single case contributing more than half of positive corr gain.",
            "- MSE-only or correlation/MSE tradeoffs are reported as weak signals, not promotion evidence.",
            "- The scout does not tune gains or masks per case; candidate ranking uses the fixed objective row named above.",
            "- Future target magnitude/phase access invalidates a candidate regardless of score.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_scout(
    *,
    config_path: Path,
    out_dir: Path,
    device_name: str,
    num_cases: int,
    prefix_seconds: float,
    future_seconds: float,
    magnitude_modes: Sequence[str],
    mask_modes: Sequence[str],
    gains: Sequence[float],
    candidate_profile: str,
    candidate_limit: int,
    objective_mode: str,
    objective_mask: str,
    objective_gain: float,
    min_corr_delta: float,
    min_corr_win_fraction: float,
    max_loop_delta: float,
    max_reentry_delta: float,
    cases_json: Path | None = None,
    write_only: bool = False,
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    candidates = _candidate_items(candidate_profile)
    if candidate_limit > 0:
        candidates = candidates[:candidate_limit]

    rows: list[dict[str, Any]] = []
    for name, overrides in candidates:
        candidate_dir = out_dir / "candidates" / name
        candidate_config = _write_candidate_config(config_path, candidate_dir, overrides)
        row: dict[str, Any] = {
            "candidate": name,
            "config_path": str(candidate_config),
            "overrides": overrides,
        }
        if not write_only:
            probe_dir = candidate_dir / "probe"
            probe = run_probe(
                config_path=candidate_config,
                out_dir=probe_dir,
                device_name=device_name,
                num_cases=num_cases,
                prefix_seconds=prefix_seconds,
                future_seconds=future_seconds,
                magnitude_modes=magnitude_modes,
                mask_modes=mask_modes,
                gains=gains,
                cases_json=cases_json,
            )
            row["probe_json"] = str(probe_dir / "audio_circle_delta_probe.json")
            row["probe_markdown"] = str(probe_dir / "AUDIO_CIRCLE_DELTA_PROBE.md")
            row["objective"] = _score_probe(
                probe,
                objective_mode=objective_mode,
                objective_mask=objective_mask,
                objective_gain=objective_gain,
                min_corr_delta=min_corr_delta,
                min_corr_win_fraction=min_corr_win_fraction,
                max_loop_delta=max_loop_delta,
                max_reentry_delta=max_reentry_delta,
            )
        rows.append(row)

    ranked = sorted(rows, key=lambda row: _as_float(row.get("objective", {}).get("score"), -1.0e9), reverse=True)
    summary = _jsonify(
        {
            "runtime": "circleworld_proto",
            "schema": "circleworld_audio_delta_objective_scout_v0",
            "status": "write_only" if write_only else _overall_status(ranked),
            "base_config_path": str(config_path),
            "out_dir": str(out_dir),
            "device": device_name,
            "requested_device": device_name,
            "case_count": int(num_cases),
            "prefix_seconds": float(prefix_seconds),
            "future_seconds": float(future_seconds),
            "magnitude_modes": list(magnitude_modes),
            "mask_modes": list(mask_modes),
            "gains": [float(gain) for gain in gains],
            "candidate_profile": candidate_profile,
            "candidate_count": len(ranked),
            "objective_mode": objective_mode,
            "objective_mask": objective_mask,
            "objective_gain": float(objective_gain),
            "min_corr_delta": float(min_corr_delta),
            "min_corr_win_fraction": float(min_corr_win_fraction),
            "max_loop_delta": float(max_loop_delta),
            "max_reentry_delta": float(max_reentry_delta),
            "future_target_audio_used_for_metrics_only": True,
            "ranked_candidates": ranked,
        }
    )
    (out_dir / OUTPUT_JSON).write_text(json.dumps(summary, indent=2), encoding="utf-8")
    _write_markdown(summary, out_dir / OUTPUT_MD)
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description="Scout Circleworld configs against the audio delta-over-copyphase objective.")
    ap.add_argument("--config", required=True, help="Base Circleworld config JSON.")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--num-cases", type=int, default=3)
    ap.add_argument("--prefix-seconds", type=float, default=1.0)
    ap.add_argument("--future-seconds", type=float, default=1.0)
    ap.add_argument("--magnitude-modes", default="prefix_hold,flat")
    ap.add_argument("--masks", default=",".join(CORE_MASK_MODES))
    ap.add_argument("--gains", default="-1,0,0.25,0.5,1,2")
    ap.add_argument("--candidate-profile", default="fixed_v1")
    ap.add_argument("--candidate-limit", type=int, default=0)
    ap.add_argument("--objective-mode", default="prefix_hold")
    ap.add_argument("--objective-mask", default="all_bins")
    ap.add_argument("--objective-gain", type=float, default=2.0)
    ap.add_argument("--min-corr-delta", type=float, default=0.01)
    ap.add_argument("--min-corr-win-fraction", type=float, default=0.60)
    ap.add_argument("--max-loop-delta", type=float, default=0.02)
    ap.add_argument("--max-reentry-delta", type=float, default=0.02)
    ap.add_argument("--write-only", action="store_true")
    ap.add_argument(
        "--cases-json",
        default=None,
        help="Optional JSON mapping case names to real anchor WAV paths.",
    )
    args = ap.parse_args()

    summary = run_scout(
        config_path=Path(args.config),
        out_dir=Path(args.out_dir),
        device_name=str(args.device),
        num_cases=int(args.num_cases),
        prefix_seconds=float(args.prefix_seconds),
        future_seconds=float(args.future_seconds),
        magnitude_modes=_parse_csv(str(args.magnitude_modes), ("prefix_hold", "flat"), "magnitude mode"),
        mask_modes=_parse_csv(str(args.masks), MASK_MODES, "mask mode"),
        gains=_parse_csv_floats(str(args.gains)),
        candidate_profile=str(args.candidate_profile),
        candidate_limit=int(args.candidate_limit),
        objective_mode=str(args.objective_mode),
        objective_mask=str(args.objective_mask),
        objective_gain=float(args.objective_gain),
        min_corr_delta=float(args.min_corr_delta),
        min_corr_win_fraction=float(args.min_corr_win_fraction),
        max_loop_delta=float(args.max_loop_delta),
        max_reentry_delta=float(args.max_reentry_delta),
        cases_json=Path(args.cases_json) if args.cases_json else None,
        write_only=bool(args.write_only),
    )
    print(
        json.dumps(
            {
                "json": str(Path(args.out_dir) / OUTPUT_JSON),
                "markdown": str(Path(args.out_dir) / OUTPUT_MD),
                "status": summary["status"],
                "candidate_count": summary["candidate_count"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
