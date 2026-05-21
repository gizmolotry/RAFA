from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable

from common_io import as_float as _as_float
from common_io import fmt as _fmt
from common_io import load_json as _load
from common_io import mean as _mean

OUTPUT_JSON = "internal_phase_law_objective_score.json"
OUTPUT_MD = "INTERNAL_PHASE_LAW_OBJECTIVE_SCORE.md"
TARGET_ROW = ("flat", "low_energy_bins", "raw", 2.0)


def _candidate_key(row: dict[str, Any]) -> tuple[str, str, str, float]:
    return (
        str(row.get("magnitude_mode", "")),
        str(row.get("mask_mode", "")),
        str(row.get("mechanism", "")),
        float(_as_float(row.get("gain"))),
    )


def _target_row_dict(target_row: tuple[str, str, str, float]) -> dict[str, Any]:
    return {
        "magnitude_mode": target_row[0],
        "mask_mode": target_row[1],
        "mechanism": target_row[2],
        "gain": target_row[3],
    }


def _is_bad_dominated(row: dict[str, Any]) -> bool:
    status = str(row.get("status") or "")
    return bool(row.get("bad_baseline_rescue_dominated")) or "bad_baseline_rescue" in status


def _bin_rows(row: dict[str, Any]) -> list[dict[str, Any]]:
    return [item for item in row.get("baseline_correlation_bins", []) or [] if isinstance(item, dict)]


def _bin_by_name(row: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {str(item.get("bin")): item for item in _bin_rows(row)}


def _weighted_bin_mean(bins: Iterable[dict[str, Any]], key: str) -> float:
    total = 0.0
    weight = 0.0
    for item in bins:
        count = max(0.0, _as_float(item.get("case_count")))
        total += count * _as_float(item.get(key))
        weight += count
    return float(total / weight) if weight > 0.0 else 0.0


def _nonbad_bins(row: dict[str, Any]) -> list[dict[str, Any]]:
    return [item for item in _bin_rows(row) if str(item.get("bin")) != "bad_baseline"]


def _best_abs_row(rows: list[dict[str, Any]], run: str, *, allow_bad: bool) -> dict[str, Any]:
    local = [row for row in rows if str(row.get("run_label")) == run]
    if not allow_bad:
        local = [row for row in local if not _is_bad_dominated(row)]
    if not local:
        return {}
    return max(
        local,
        key=lambda row: (
            _as_float(row.get("mean_corr_delta")),
            _as_float(row.get("median_corr_delta")),
            _as_float(row.get("corr_win_fraction")),
            -_as_float(row.get("mean_mse_delta")),
        ),
    )


def _matching_abs_row(rows: list[dict[str, Any]], direct_row: dict[str, Any]) -> dict[str, Any]:
    run = str(direct_row.get("run_label"))
    key = _candidate_key(direct_row)
    for row in rows:
        if str(row.get("run_label")) == run and _candidate_key(row) == key:
            return row
    return {}


def _target_direct_row(rows: list[dict[str, Any]], target_row: tuple[str, str, str, float]) -> dict[str, Any]:
    for row in rows:
        if _candidate_key(row) == target_row:
            return row
    return {}


def _target_decision(row: dict[str, Any]) -> str:
    if not row.get("target_low_energy_present"):
        return "target_row_missing"
    if _as_float(row.get("target_low_energy_direct_mean_corr_delta")) <= 0.0:
        return "hold_target_direct_nonpositive"
    if _as_float(row.get("target_low_energy_direct_median_corr_delta")) <= 0.0:
        return "hold_target_direct_median"
    if _as_float(row.get("target_low_energy_direct_corr_win_fraction")) < 0.55:
        return "hold_target_direct_win_fraction"
    if _as_float(row.get("target_low_energy_absolute_mean_corr_delta")) <= 0.0:
        return "hold_target_absolute_nonpositive"
    if _as_float(row.get("target_low_energy_nonbad_bin_mean_corr_delta")) <= 0.0:
        return "hold_target_nonbad_bin_regression"
    if _as_float(row.get("target_low_energy_good_bin_mean_corr_delta")) < -0.01:
        return "hold_target_good_bin_regression"
    if _as_float(row.get("target_low_energy_absolute_corr_win_fraction")) < 0.55:
        return "hold_target_absolute_win_fraction"
    if bool(row.get("target_low_energy_bad_dominated")):
        return "hold_target_bad_baseline_dominated"
    return "target_row_candidate"


def _decision(row: dict[str, Any]) -> str:
    if _as_float(row.get("absolute_best_nonbad_mean_corr_delta")) <= 0.0:
        return "hold_absolute_negative"
    if _as_float(row.get("absolute_best_nonbad_median_corr_delta")) <= 0.0:
        return "hold_absolute_median"
    if _as_float(row.get("absolute_best_nonbad_corr_win_fraction")) < 0.55:
        return "hold_absolute_win_fraction"
    if _as_float(row.get("direct_overall_mean_corr_delta")) <= 0.0:
        return "hold_direct_nonpositive"
    if _as_float(row.get("direct_best_mean_corr_delta")) < 0.001:
        return "hold_direct_too_small"
    if _as_float(row.get("matched_absolute_mean_corr_delta")) <= 0.0:
        return "hold_matched_absolute_negative"
    if bool(row.get("matched_absolute_bad_dominated")):
        return "hold_matched_bad_baseline_dominated"
    if _as_float(row.get("matched_absolute_nonbad_bin_mean_corr_delta")) <= 0.0:
        return "hold_matched_nonbad_bin_regression"
    if _as_float(row.get("matched_absolute_good_bin_mean_corr_delta")) < -0.01:
        return "hold_matched_good_bin_regression"
    return "candidate_for_locked_followup"


def score_objective(
    *,
    absolute_compare: Path,
    variant_compare: Path,
    out_dir: Path,
    target_row: tuple[str, str, str, float] = TARGET_ROW,
) -> dict[str, Any]:
    abs_payload = _load(absolute_compare)
    variant_payload = _load(variant_compare)
    abs_candidates = [row for row in abs_payload.get("candidate_summaries", []) or [] if isinstance(row, dict)]
    direct_overall = [
        row
        for row in variant_payload.get("overall_summaries", []) or []
        if isinstance(row, dict)
    ]
    direct_candidates = [
        row
        for row in variant_payload.get("candidate_summaries", []) or []
        if isinstance(row, dict)
    ]
    runs = sorted({str(row.get("run_label")) for row in direct_overall if row.get("run_label")})
    rows: list[dict[str, Any]] = []
    for run in runs:
        direct = next((row for row in direct_overall if str(row.get("run_label")) == run), {})
        direct_local = [row for row in direct_candidates if str(row.get("run_label")) == run]
        direct_best = max(
            direct_local,
            key=lambda row: _as_float(row.get("mean_corr_delta_vs_baseline_config")),
            default={},
        )
        abs_best = _best_abs_row(abs_candidates, run, allow_bad=True)
        abs_best_nonbad = _best_abs_row(abs_candidates, run, allow_bad=False)
        matched_abs = _matching_abs_row(abs_candidates, direct_best)
        target_direct = _target_direct_row(direct_local, target_row)
        target_abs = _matching_abs_row(abs_candidates, target_direct) if target_direct else {}
        absolute_nonbad_mean = _as_float(abs_best_nonbad.get("mean_corr_delta"))
        direct_mean = _as_float(direct.get("mean_corr_delta_vs_baseline_config"))
        matched_abs_mean = _as_float(matched_abs.get("mean_corr_delta"))
        matched_bins = _bin_by_name(matched_abs)
        matched_nonbad_bins = _nonbad_bins(matched_abs)
        matched_nonbad_bin_mean = _weighted_bin_mean(matched_nonbad_bins, "mean_corr_delta")
        target_bins = _bin_by_name(target_abs)
        target_nonbad_bins = _nonbad_bins(target_abs)
        target_abs_mean = _as_float(target_abs.get("mean_corr_delta"))
        target_direct_mean = _as_float(target_direct.get("mean_corr_delta_vs_baseline_config"))
        target_nonbad_mean = _weighted_bin_mean(target_nonbad_bins, "mean_corr_delta")
        target_good_mean = _as_float(target_bins.get("good_baseline", {}).get("mean_corr_delta"))
        target_score = (
            3.0 * target_direct_mean
            + 3.0 * target_abs_mean
            + 4.0 * min(0.0, target_nonbad_mean)
            + 2.0 * min(0.0, target_good_mean)
            + 0.003 * (_as_float(target_direct.get("corr_win_fraction_vs_baseline_config")) - 0.5)
            + 0.004 * (_as_float(target_abs.get("corr_win_fraction")) - 0.5)
            - 0.002 * (1.0 if _is_bad_dominated(target_abs) else 0.0)
        ) if target_direct else -1.0
        score = (
            2.0 * direct_mean
            + _as_float(direct_best.get("mean_corr_delta_vs_baseline_config"))
            + 4.0 * min(0.0, absolute_nonbad_mean)
            + 2.0 * min(0.0, matched_abs_mean)
            + 4.0 * min(0.0, matched_nonbad_bin_mean)
            + 2.0 * min(0.0, _as_float(matched_bins.get("good_baseline", {}).get("mean_corr_delta")))
            - 0.001 * (1.0 if _is_bad_dominated(abs_best) else 0.0)
        )
        row = {
            "run_label": run,
            "score": float(score),
            "direct_overall_mean_corr_delta": direct_mean,
            "direct_overall_median_corr_delta": direct.get("median_corr_delta_vs_baseline_config"),
            "direct_overall_corr_win_fraction": direct.get("corr_win_fraction_vs_baseline_config"),
            "direct_best_mode": direct_best.get("magnitude_mode"),
            "direct_best_mask": direct_best.get("mask_mode"),
            "direct_best_mechanism": direct_best.get("mechanism"),
            "direct_best_gain": direct_best.get("gain"),
            "direct_best_mean_corr_delta": direct_best.get("mean_corr_delta_vs_baseline_config"),
            "direct_best_median_corr_delta": direct_best.get("median_corr_delta_vs_baseline_config"),
            "direct_best_corr_win_fraction": direct_best.get("corr_win_fraction_vs_baseline_config"),
            "absolute_best_mean_corr_delta": abs_best.get("mean_corr_delta"),
            "absolute_best_status": abs_best.get("status"),
            "absolute_best_bad_dominated": _is_bad_dominated(abs_best),
            "absolute_best_nonbad_mean_corr_delta": abs_best_nonbad.get("mean_corr_delta"),
            "absolute_best_nonbad_median_corr_delta": abs_best_nonbad.get("median_corr_delta"),
            "absolute_best_nonbad_corr_win_fraction": abs_best_nonbad.get("corr_win_fraction"),
            "absolute_best_nonbad_mean_mse_delta": abs_best_nonbad.get("mean_mse_delta"),
            "matched_absolute_mean_corr_delta": matched_abs.get("mean_corr_delta"),
            "matched_absolute_median_corr_delta": matched_abs.get("median_corr_delta"),
            "matched_absolute_corr_win_fraction": matched_abs.get("corr_win_fraction"),
            "matched_absolute_mean_mse_delta": matched_abs.get("mean_mse_delta"),
            "matched_absolute_status": matched_abs.get("status"),
            "matched_absolute_bad_dominated": _is_bad_dominated(matched_abs),
            "matched_absolute_bad_bin_mean_corr_delta": matched_bins.get("bad_baseline", {}).get("mean_corr_delta"),
            "matched_absolute_nonbad_bin_mean_corr_delta": matched_nonbad_bin_mean,
            "matched_absolute_nonbad_bin_corr_win_fraction": _weighted_bin_mean(matched_nonbad_bins, "corr_win_fraction"),
            "matched_absolute_weak_bin_mean_corr_delta": matched_bins.get("weak_baseline", {}).get("mean_corr_delta"),
            "matched_absolute_moderate_bin_mean_corr_delta": matched_bins.get("moderate_baseline", {}).get("mean_corr_delta"),
            "matched_absolute_good_bin_mean_corr_delta": matched_bins.get("good_baseline", {}).get("mean_corr_delta"),
            "target_low_energy_present": bool(target_direct),
            "target_low_energy_score": float(target_score),
            "target_low_energy_mode": target_row[0],
            "target_low_energy_mask": target_row[1],
            "target_low_energy_mechanism": target_row[2],
            "target_low_energy_gain": target_row[3],
            "target_low_energy_direct_mean_corr_delta": target_direct.get("mean_corr_delta_vs_baseline_config"),
            "target_low_energy_direct_median_corr_delta": target_direct.get("median_corr_delta_vs_baseline_config"),
            "target_low_energy_direct_corr_win_fraction": target_direct.get("corr_win_fraction_vs_baseline_config"),
            "target_low_energy_absolute_mean_corr_delta": target_abs.get("mean_corr_delta"),
            "target_low_energy_absolute_median_corr_delta": target_abs.get("median_corr_delta"),
            "target_low_energy_absolute_corr_win_fraction": target_abs.get("corr_win_fraction"),
            "target_low_energy_absolute_mean_mse_delta": target_abs.get("mean_mse_delta"),
            "target_low_energy_status": target_abs.get("status"),
            "target_low_energy_bad_dominated": _is_bad_dominated(target_abs),
            "target_low_energy_bad_bin_mean_corr_delta": target_bins.get("bad_baseline", {}).get("mean_corr_delta"),
            "target_low_energy_nonbad_bin_mean_corr_delta": target_nonbad_mean,
            "target_low_energy_nonbad_bin_corr_win_fraction": _weighted_bin_mean(
                target_nonbad_bins, "corr_win_fraction"
            ),
            "target_low_energy_weak_bin_mean_corr_delta": target_bins.get("weak_baseline", {}).get("mean_corr_delta"),
            "target_low_energy_moderate_bin_mean_corr_delta": target_bins.get("moderate_baseline", {}).get(
                "mean_corr_delta"
            ),
            "target_low_energy_good_bin_mean_corr_delta": target_bins.get("good_baseline", {}).get("mean_corr_delta"),
        }
        row["decision"] = _decision(row)
        row["target_low_energy_decision"] = _target_decision(row)
        row.update(
            {
                "target_row_present": row["target_low_energy_present"],
                "target_row_score": row["target_low_energy_score"],
                "target_row_mode": row["target_low_energy_mode"],
                "target_row_mask": row["target_low_energy_mask"],
                "target_row_mechanism": row["target_low_energy_mechanism"],
                "target_row_gain": row["target_low_energy_gain"],
                "target_row_direct_mean_corr_delta": row["target_low_energy_direct_mean_corr_delta"],
                "target_row_direct_median_corr_delta": row["target_low_energy_direct_median_corr_delta"],
                "target_row_direct_corr_win_fraction": row["target_low_energy_direct_corr_win_fraction"],
                "target_row_absolute_mean_corr_delta": row["target_low_energy_absolute_mean_corr_delta"],
                "target_row_absolute_median_corr_delta": row["target_low_energy_absolute_median_corr_delta"],
                "target_row_absolute_corr_win_fraction": row["target_low_energy_absolute_corr_win_fraction"],
                "target_row_absolute_mean_mse_delta": row["target_low_energy_absolute_mean_mse_delta"],
                "target_row_status": row["target_low_energy_status"],
                "target_row_bad_dominated": row["target_low_energy_bad_dominated"],
                "target_row_bad_bin_mean_corr_delta": row["target_low_energy_bad_bin_mean_corr_delta"],
                "target_row_nonbad_bin_mean_corr_delta": row["target_low_energy_nonbad_bin_mean_corr_delta"],
                "target_row_nonbad_bin_corr_win_fraction": row[
                    "target_low_energy_nonbad_bin_corr_win_fraction"
                ],
                "target_row_weak_bin_mean_corr_delta": row["target_low_energy_weak_bin_mean_corr_delta"],
                "target_row_moderate_bin_mean_corr_delta": row[
                    "target_low_energy_moderate_bin_mean_corr_delta"
                ],
                "target_row_good_bin_mean_corr_delta": row["target_low_energy_good_bin_mean_corr_delta"],
                "target_row_decision": row["target_low_energy_decision"],
            }
        )
        rows.append(row)

    rows.sort(key=lambda row: _as_float(row.get("score")), reverse=True)
    target_rows = [row for row in rows if row.get("target_low_energy_present")]
    target_rows.sort(key=lambda row: _as_float(row.get("target_low_energy_score")), reverse=True)
    target_candidates = [row for row in target_rows if row.get("target_low_energy_decision") == "target_row_candidate"]
    target_best = target_rows[0] if target_rows else {}
    summary = {
        "schema": "circleworld_internal_phase_law_objective_score_v0",
        "status": "candidate_found" if any(row.get("decision") == "candidate_for_locked_followup" for row in rows) else "no_locked_candidate",
        "absolute_compare": str(absolute_compare),
        "variant_compare": str(variant_compare),
        "run_count": len(rows),
        "best_run": rows[0].get("run_label") if rows else None,
        "best_decision": rows[0].get("decision") if rows else None,
        "best_score": rows[0].get("score") if rows else None,
        "mean_run_score": _mean(_as_float(row.get("score")) for row in rows),
        "mean_matched_absolute_nonbad_bin_corr_delta": _mean(
            _as_float(row.get("matched_absolute_nonbad_bin_mean_corr_delta")) for row in rows
        ),
        "target_row": _target_row_dict(target_row),
        "target_low_energy_row": _target_row_dict(target_row),
        "target_row_count": len(target_rows),
        "target_row_candidate_count": len(target_candidates),
        "target_row_best_run": target_best.get("run_label"),
        "target_row_best_decision": target_best.get("target_row_decision"),
        "target_row_best_score": target_best.get("target_row_score"),
        "target_row_best_direct_mean_corr_delta": target_best.get("target_row_direct_mean_corr_delta"),
        "target_row_best_direct_median_corr_delta": target_best.get("target_row_direct_median_corr_delta"),
        "target_row_best_direct_corr_win_fraction": target_best.get("target_row_direct_corr_win_fraction"),
        "target_row_best_absolute_mean_corr_delta": target_best.get("target_row_absolute_mean_corr_delta"),
        "target_row_best_absolute_median_corr_delta": target_best.get("target_row_absolute_median_corr_delta"),
        "target_row_best_absolute_corr_win_fraction": target_best.get("target_row_absolute_corr_win_fraction"),
        "target_row_best_nonbad_bin_mean_corr_delta": target_best.get("target_row_nonbad_bin_mean_corr_delta"),
        "target_row_best_nonbad_bin_corr_win_fraction": target_best.get(
            "target_row_nonbad_bin_corr_win_fraction"
        ),
        "target_row_best_good_bin_mean_corr_delta": target_best.get("target_row_good_bin_mean_corr_delta"),
        "target_low_energy_row_count": len(target_rows),
        "target_low_energy_candidate_count": len(target_candidates),
        "target_low_energy_best_run": target_best.get("run_label"),
        "target_low_energy_best_decision": target_best.get("target_low_energy_decision"),
        "target_low_energy_best_score": target_best.get("target_low_energy_score"),
        "target_low_energy_best_direct_mean_corr_delta": target_best.get(
            "target_low_energy_direct_mean_corr_delta"
        ),
        "target_low_energy_best_direct_median_corr_delta": target_best.get(
            "target_low_energy_direct_median_corr_delta"
        ),
        "target_low_energy_best_direct_corr_win_fraction": target_best.get(
            "target_low_energy_direct_corr_win_fraction"
        ),
        "target_low_energy_best_absolute_mean_corr_delta": target_best.get(
            "target_low_energy_absolute_mean_corr_delta"
        ),
        "target_low_energy_best_absolute_median_corr_delta": target_best.get(
            "target_low_energy_absolute_median_corr_delta"
        ),
        "target_low_energy_best_absolute_corr_win_fraction": target_best.get(
            "target_low_energy_absolute_corr_win_fraction"
        ),
        "target_low_energy_best_nonbad_bin_mean_corr_delta": target_best.get(
            "target_low_energy_nonbad_bin_mean_corr_delta"
        ),
        "target_low_energy_best_nonbad_bin_corr_win_fraction": target_best.get(
            "target_low_energy_nonbad_bin_corr_win_fraction"
        ),
        "target_low_energy_best_good_bin_mean_corr_delta": target_best.get(
            "target_low_energy_good_bin_mean_corr_delta"
        ),
        "rows": rows,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / OUTPUT_JSON).write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (out_dir / OUTPUT_MD).write_text(_markdown(summary), encoding="utf-8")
    return summary


def _markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# Internal Phase-Law Objective Score",
        "",
        "This locks the next internal-law decision to two conditions: direct recurrence improvement and absolute audio improvement.",
        "",
        f"- status: `{summary['status']}`",
        f"- best run: `{summary.get('best_run')}`",
        f"- best decision: `{summary.get('best_decision')}`",
        f"- best score: `{_fmt(summary.get('best_score'))}`",
        f"- target row: `{summary['target_row']['magnitude_mode']}` / `{summary['target_row']['mask_mode']}` / `{summary['target_row']['mechanism']}` / gain `{_fmt(summary['target_row']['gain'])}`",
        f"- target candidates: `{summary.get('target_row_candidate_count')}` / `{summary.get('target_row_count')}`",
        f"- target best run / decision / score: `{summary.get('target_row_best_run')}` / `{summary.get('target_row_best_decision')}` / `{_fmt(summary.get('target_row_best_score'))}`",
        f"- target best direct mean / median / wins: `{_fmt(summary.get('target_row_best_direct_mean_corr_delta'))}` / `{_fmt(summary.get('target_row_best_direct_median_corr_delta'))}` / `{_fmt(summary.get('target_row_best_direct_corr_win_fraction'))}`",
        f"- target best absolute mean / median / wins: `{_fmt(summary.get('target_row_best_absolute_mean_corr_delta'))}` / `{_fmt(summary.get('target_row_best_absolute_median_corr_delta'))}` / `{_fmt(summary.get('target_row_best_absolute_corr_win_fraction'))}`",
        f"- target best nonbad / nonbad wins / good corr d: `{_fmt(summary.get('target_row_best_nonbad_bin_mean_corr_delta'))}` / `{_fmt(summary.get('target_row_best_nonbad_bin_corr_win_fraction'))}` / `{_fmt(summary.get('target_row_best_good_bin_mean_corr_delta'))}`",
        "",
        "| run | decision | score | direct mean d | direct best d | abs nonbad mean d | matched abs d | matched nonbad bin d | matched good d | target decision | target direct d | target nonbad d | matched status |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---|",
    ]
    for row in summary.get("rows", []):
        lines.append(
            "| {run} | {decision} | {score} | {direct} | {best} | {absmean} | {matched} | {nonbad} | {good} | {target_decision} | {target_direct} | {target_nonbad} | {matched_status} |".format(
                run=row.get("run_label"),
                decision=row.get("decision"),
                score=_fmt(row.get("score")),
                direct=_fmt(row.get("direct_overall_mean_corr_delta")),
                best=_fmt(row.get("direct_best_mean_corr_delta")),
                absmean=_fmt(row.get("absolute_best_nonbad_mean_corr_delta")),
                matched=_fmt(row.get("matched_absolute_mean_corr_delta")),
                nonbad=_fmt(row.get("matched_absolute_nonbad_bin_mean_corr_delta")),
                good=_fmt(row.get("matched_absolute_good_bin_mean_corr_delta")),
                target_decision=row.get("target_row_decision"),
                target_direct=_fmt(row.get("target_row_direct_mean_corr_delta")),
                target_nonbad=_fmt(row.get("target_row_nonbad_bin_mean_corr_delta")),
                matched_status=row.get("matched_absolute_status"),
            )
        )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description="Score internal phase-law scouts against a locked direct+absolute objective.")
    ap.add_argument("--absolute-compare", required=True)
    ap.add_argument("--variant-compare", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--target-mode", default=TARGET_ROW[0])
    ap.add_argument("--target-mask", default=TARGET_ROW[1])
    ap.add_argument("--target-mechanism", default=TARGET_ROW[2])
    ap.add_argument("--target-gain", type=float, default=TARGET_ROW[3])
    args = ap.parse_args()
    summary = score_objective(
        absolute_compare=Path(args.absolute_compare),
        variant_compare=Path(args.variant_compare),
        out_dir=Path(args.out_dir),
        target_row=(str(args.target_mode), str(args.target_mask), str(args.target_mechanism), float(args.target_gain)),
    )
    print(
        json.dumps(
            {
                "json": str(Path(args.out_dir) / OUTPUT_JSON),
                "markdown": str(Path(args.out_dir) / OUTPUT_MD),
                "status": summary["status"],
                "best_run": summary.get("best_run"),
                "best_decision": summary.get("best_decision"),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
