from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable


OUTPUT_JSON = "internal_phase_law_joint_row_diagnostics.json"
OUTPUT_MD = "INTERNAL_PHASE_LAW_JOINT_ROW_DIAGNOSTICS.md"
TARGET_ROW = ("flat", "low_energy_bins", "raw", 2.0)


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _fmt(value: Any) -> str:
    if value is None:
        return "NA"
    if isinstance(value, (int, float)):
        return f"{float(value):.6g}"
    return str(value)


def _load(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"Expected JSON object: {path}")
    return payload


def _row_key(row: dict[str, Any]) -> tuple[str, str, str, str, float]:
    return (
        str(row.get("run_label", "")),
        str(row.get("magnitude_mode", "")),
        str(row.get("mask_mode", "")),
        str(row.get("mechanism", "")),
        float(_as_float(row.get("gain"))),
    )


def _target_row_key(row: dict[str, Any]) -> tuple[str, str, str, float]:
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


def _decision(row: dict[str, Any]) -> str:
    if _as_float(row.get("direct_mean_corr_delta")) <= 0.0:
        return "hold_direct_nonpositive"
    if _as_float(row.get("direct_median_corr_delta")) <= 0.0:
        return "hold_direct_median"
    if _as_float(row.get("direct_corr_win_fraction")) < 0.55:
        return "hold_direct_win_fraction"
    if _as_float(row.get("absolute_mean_corr_delta")) <= 0.0:
        return "hold_absolute_nonpositive"
    if _as_float(row.get("absolute_nonbad_bin_mean_corr_delta")) <= 0.0:
        return "hold_absolute_nonbad_bin_regression"
    if _as_float(row.get("absolute_moderate_bin_mean_corr_delta")) < -0.01:
        return "hold_moderate_baseline_regression"
    if _as_float(row.get("absolute_good_bin_mean_corr_delta")) < -0.01:
        return "hold_good_baseline_regression"
    if _as_float(row.get("absolute_median_corr_delta")) <= 0.0:
        return "hold_absolute_median"
    if _as_float(row.get("absolute_corr_win_fraction")) < 0.55:
        return "hold_absolute_win_fraction"
    if _is_bad_dominated(row):
        return "hold_bad_baseline_dominated"
    if _as_float(row.get("absolute_mean_mse_delta")) > 0.0:
        return "hold_absolute_mse_regression"
    return "candidate_joint_row"


def _mean(values: Iterable[float]) -> float:
    vals = [float(value) for value in values]
    return float(sum(vals) / len(vals)) if vals else 0.0


def diagnose_joint_rows(
    *,
    absolute_compare: Path,
    variant_compare: Path,
    out_dir: Path,
    target_row: tuple[str, str, str, float] = TARGET_ROW,
) -> dict[str, Any]:
    absolute = _load(absolute_compare)
    variant = _load(variant_compare)
    abs_rows = {
        _row_key(row): row
        for row in absolute.get("candidate_summaries", []) or []
        if isinstance(row, dict)
    }

    rows: list[dict[str, Any]] = []
    missing_absolute = 0
    for direct in variant.get("candidate_summaries", []) or []:
        if not isinstance(direct, dict):
            continue
        key = _row_key(direct)
        absolute_row = abs_rows.get(key)
        if absolute_row is None:
            missing_absolute += 1
            continue
        bins = _bin_by_name(absolute_row)
        nonbad_bins = _nonbad_bins(absolute_row)
        row = {
            "run_label": key[0],
            "magnitude_mode": key[1],
            "mask_mode": key[2],
            "mechanism": key[3],
            "gain": key[4],
            "direct_status": direct.get("status"),
            "direct_mean_corr_delta": direct.get("mean_corr_delta_vs_baseline_config"),
            "direct_median_corr_delta": direct.get("median_corr_delta_vs_baseline_config"),
            "direct_corr_win_fraction": direct.get("corr_win_fraction_vs_baseline_config"),
            "direct_mean_mse_delta": direct.get("mean_mse_delta_vs_baseline_config"),
            "absolute_status": absolute_row.get("status"),
            "absolute_mean_corr_delta": absolute_row.get("mean_corr_delta"),
            "absolute_median_corr_delta": absolute_row.get("median_corr_delta"),
            "absolute_corr_win_fraction": absolute_row.get("corr_win_fraction"),
            "absolute_mean_mse_delta": absolute_row.get("mean_mse_delta"),
            "absolute_bad_baseline_dominated": _is_bad_dominated(absolute_row),
            "absolute_bad_bin_mean_corr_delta": bins.get("bad_baseline", {}).get("mean_corr_delta"),
            "absolute_nonbad_bin_mean_corr_delta": _weighted_bin_mean(nonbad_bins, "mean_corr_delta"),
            "absolute_nonbad_bin_corr_win_fraction": _weighted_bin_mean(nonbad_bins, "corr_win_fraction"),
            "absolute_weak_bin_mean_corr_delta": bins.get("weak_baseline", {}).get("mean_corr_delta"),
            "absolute_moderate_bin_mean_corr_delta": bins.get("moderate_baseline", {}).get("mean_corr_delta"),
            "absolute_good_bin_mean_corr_delta": bins.get("good_baseline", {}).get("mean_corr_delta"),
        }
        row["decision"] = _decision(row)
        row["score"] = float(
            2.0 * _as_float(row.get("direct_mean_corr_delta"))
            + _as_float(row.get("direct_median_corr_delta"))
            + 3.0 * _as_float(row.get("absolute_mean_corr_delta"))
            + _as_float(row.get("absolute_median_corr_delta"))
            + 4.0 * min(0.0, _as_float(row.get("absolute_nonbad_bin_mean_corr_delta")))
            + 2.0 * min(0.0, _as_float(row.get("absolute_good_bin_mean_corr_delta")))
            + 0.002 * (_as_float(row.get("direct_corr_win_fraction")) - 0.5)
            + 0.004 * (_as_float(row.get("absolute_corr_win_fraction")) - 0.5)
            - 0.002 * (1.0 if row.get("absolute_bad_baseline_dominated") else 0.0)
            - max(0.0, _as_float(row.get("absolute_mean_mse_delta")))
        )
        rows.append(row)

    rows.sort(
        key=lambda row: (
            row.get("decision") == "candidate_joint_row",
            _as_float(row.get("score")),
            _as_float(row.get("direct_mean_corr_delta")),
            _as_float(row.get("absolute_mean_corr_delta")),
        ),
        reverse=True,
    )
    decisions: dict[str, int] = {}
    for row in rows:
        decision = str(row.get("decision"))
        decisions[decision] = decisions.get(decision, 0) + 1
    candidates = [row for row in rows if row.get("decision") == "candidate_joint_row"]
    target_rows = [row for row in rows if _target_row_key(row) == target_row]
    target_rows.sort(
        key=lambda row: (
            row.get("decision") == "candidate_joint_row",
            _as_float(row.get("score")),
            _as_float(row.get("direct_mean_corr_delta")),
            _as_float(row.get("absolute_nonbad_bin_mean_corr_delta")),
        ),
        reverse=True,
    )
    target_candidates = [row for row in target_rows if row.get("decision") == "candidate_joint_row"]
    target_best = target_rows[0] if target_rows else {}
    summary = {
        "schema": "circleworld_internal_phase_law_joint_row_diagnostics_v0",
        "status": "candidate_joint_row_found" if candidates else "no_joint_row_candidate",
        "absolute_compare": str(absolute_compare),
        "variant_compare": str(variant_compare),
        "row_count": len(rows),
        "missing_absolute_count": missing_absolute,
        "candidate_count": len(candidates),
        "decision_counts": decisions,
        "best_row": rows[0] if rows else None,
        "mean_direct_corr_delta": _mean(_as_float(row.get("direct_mean_corr_delta")) for row in rows),
        "mean_absolute_corr_delta": _mean(_as_float(row.get("absolute_mean_corr_delta")) for row in rows),
        "mean_absolute_nonbad_bin_corr_delta": _mean(_as_float(row.get("absolute_nonbad_bin_mean_corr_delta")) for row in rows),
        "target_row": _target_row_dict(target_row),
        "target_low_energy_row": _target_row_dict(target_row),
        "target_row_count": len(target_rows),
        "target_row_candidate_count": len(target_candidates),
        "target_row_best_row": target_best,
        "target_row_best_run": target_best.get("run_label"),
        "target_row_best_decision": target_best.get("decision"),
        "target_row_best_score": target_best.get("score"),
        "target_row_best_direct_mean_corr_delta": target_best.get("direct_mean_corr_delta"),
        "target_row_best_direct_median_corr_delta": target_best.get("direct_median_corr_delta"),
        "target_row_best_direct_corr_win_fraction": target_best.get("direct_corr_win_fraction"),
        "target_row_best_absolute_mean_corr_delta": target_best.get("absolute_mean_corr_delta"),
        "target_row_best_absolute_median_corr_delta": target_best.get("absolute_median_corr_delta"),
        "target_row_best_absolute_corr_win_fraction": target_best.get("absolute_corr_win_fraction"),
        "target_row_best_nonbad_bin_mean_corr_delta": target_best.get("absolute_nonbad_bin_mean_corr_delta"),
        "target_row_best_nonbad_bin_corr_win_fraction": target_best.get("absolute_nonbad_bin_corr_win_fraction"),
        "target_row_best_good_bin_mean_corr_delta": target_best.get("absolute_good_bin_mean_corr_delta"),
        "target_low_energy_row_count": len(target_rows),
        "target_low_energy_candidate_count": len(target_candidates),
        "target_low_energy_best_row": target_best,
        "target_low_energy_best_run": target_best.get("run_label"),
        "target_low_energy_best_decision": target_best.get("decision"),
        "target_low_energy_best_score": target_best.get("score"),
        "target_low_energy_best_direct_mean_corr_delta": target_best.get("direct_mean_corr_delta"),
        "target_low_energy_best_direct_median_corr_delta": target_best.get("direct_median_corr_delta"),
        "target_low_energy_best_direct_corr_win_fraction": target_best.get("direct_corr_win_fraction"),
        "target_low_energy_best_absolute_mean_corr_delta": target_best.get("absolute_mean_corr_delta"),
        "target_low_energy_best_absolute_median_corr_delta": target_best.get("absolute_median_corr_delta"),
        "target_low_energy_best_absolute_corr_win_fraction": target_best.get("absolute_corr_win_fraction"),
        "target_low_energy_best_nonbad_bin_mean_corr_delta": target_best.get(
            "absolute_nonbad_bin_mean_corr_delta"
        ),
        "target_low_energy_best_nonbad_bin_corr_win_fraction": target_best.get(
            "absolute_nonbad_bin_corr_win_fraction"
        ),
        "target_low_energy_best_good_bin_mean_corr_delta": target_best.get("absolute_good_bin_mean_corr_delta"),
        "rows": rows,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / OUTPUT_JSON).write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (out_dir / OUTPUT_MD).write_text(_markdown(summary), encoding="utf-8")
    return summary


def _markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# Internal Phase-Law Joint Row Diagnostics",
        "",
        "This joins direct variant-vs-noop rows to absolute carrier rows by run/mode/mask/mechanism/gain.",
        "",
        f"- status: `{summary['status']}`",
        f"- rows: `{summary['row_count']}`",
        f"- candidates: `{summary['candidate_count']}`",
        f"- missing absolute rows: `{summary['missing_absolute_count']}`",
        f"- target row: `{summary['target_row']['magnitude_mode']}` / `{summary['target_row']['mask_mode']}` / `{summary['target_row']['mechanism']}` / gain `{_fmt(summary['target_row']['gain'])}`",
        f"- target candidates: `{summary.get('target_row_candidate_count')}` / `{summary.get('target_row_count')}`",
        f"- target best run / decision / score: `{summary.get('target_row_best_run')}` / `{summary.get('target_row_best_decision')}` / `{_fmt(summary.get('target_row_best_score'))}`",
        f"- target best direct mean / median / wins: `{_fmt(summary.get('target_row_best_direct_mean_corr_delta'))}` / `{_fmt(summary.get('target_row_best_direct_median_corr_delta'))}` / `{_fmt(summary.get('target_row_best_direct_corr_win_fraction'))}`",
        f"- target best absolute mean / median / wins: `{_fmt(summary.get('target_row_best_absolute_mean_corr_delta'))}` / `{_fmt(summary.get('target_row_best_absolute_median_corr_delta'))}` / `{_fmt(summary.get('target_row_best_absolute_corr_win_fraction'))}`",
        f"- target best nonbad / nonbad wins / good corr d: `{_fmt(summary.get('target_row_best_nonbad_bin_mean_corr_delta'))}` / `{_fmt(summary.get('target_row_best_nonbad_bin_corr_win_fraction'))}` / `{_fmt(summary.get('target_row_best_good_bin_mean_corr_delta'))}`",
        "",
        "## Decision Counts",
        "",
    ]
    for key, value in sorted((summary.get("decision_counts") or {}).items()):
        lines.append(f"- `{key}`: `{value}`")
    lines.extend(
        [
            "",
            "## Top Rows",
            "",
            "| run | mode | mask | gain | decision | score | direct mean d | direct wins | abs mean d | nonbad bin d | weak d | moderate d | good d | abs wins | abs status |",
            "|---|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
        ]
    )
    for row in (summary.get("rows") or [])[:24]:
        lines.append(
            "| {run} | {mode} | {mask} | {gain} | {decision} | {score} | {direct} | {dwins} | {absmean} | {nonbad} | {weak} | {moderate} | {good} | {awins} | {status} |".format(
                run=row.get("run_label"),
                mode=row.get("magnitude_mode"),
                mask=row.get("mask_mode"),
                gain=_fmt(row.get("gain")),
                decision=row.get("decision"),
                score=_fmt(row.get("score")),
                direct=_fmt(row.get("direct_mean_corr_delta")),
                dwins=_fmt(row.get("direct_corr_win_fraction")),
                absmean=_fmt(row.get("absolute_mean_corr_delta")),
                nonbad=_fmt(row.get("absolute_nonbad_bin_mean_corr_delta")),
                weak=_fmt(row.get("absolute_weak_bin_mean_corr_delta")),
                moderate=_fmt(row.get("absolute_moderate_bin_mean_corr_delta")),
                good=_fmt(row.get("absolute_good_bin_mean_corr_delta")),
                awins=_fmt(row.get("absolute_corr_win_fraction")),
                status=row.get("absolute_status"),
            )
        )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose joint direct+absolute internal phase-law row candidates.")
    parser.add_argument("--absolute-compare", required=True)
    parser.add_argument("--variant-compare", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--target-mode", default=TARGET_ROW[0])
    parser.add_argument("--target-mask", default=TARGET_ROW[1])
    parser.add_argument("--target-mechanism", default=TARGET_ROW[2])
    parser.add_argument("--target-gain", type=float, default=TARGET_ROW[3])
    args = parser.parse_args()
    summary = diagnose_joint_rows(
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
                "candidate_count": summary["candidate_count"],
                "row_count": summary["row_count"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
