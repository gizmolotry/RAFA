from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence

from common_io import as_float as _as_float
from common_io import fmt as _fmt
from common_io import load_json as _load
from common_io import mean as _mean
from common_io import median as _median


OUTPUT_JSON = "internal_phase_law_failure_atlas.json"
OUTPUT_MD = "INTERNAL_PHASE_LAW_FAILURE_ATLAS.md"


def _summarize_values(rows: Sequence[dict[str, Any]], corr_key: str, mse_key: str | None = None) -> dict[str, Any]:
    corr = [_as_float(row.get(corr_key)) for row in rows]
    mse = [_as_float(row.get(mse_key)) for row in rows] if mse_key else []
    return {
        "row_count": len(rows),
        "mean_corr_delta": _mean(corr),
        "median_corr_delta": _median(corr),
        "corr_win_fraction": _mean(1.0 if value > 0.0 else 0.0 for value in corr),
        "mean_mse_delta": _mean(mse) if mse else None,
        "mse_win_fraction": _mean(1.0 if value < 0.0 else 0.0 for value in mse) if mse else None,
    }


def _group_by(rows: Iterable[dict[str, Any]], keys: Sequence[str]) -> dict[tuple[Any, ...], list[dict[str, Any]]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[tuple(row.get(key) for key in keys)].append(row)
    return grouped


def _best_row(rows: Sequence[dict[str, Any]], key: str = "mean_corr_delta") -> dict[str, Any]:
    if not rows:
        return {}
    return max(
        rows,
        key=lambda row: (
            _as_float(row.get(key)),
            _as_float(row.get("median_corr_delta")),
            _as_float(row.get("corr_win_fraction")),
            -_as_float(row.get("mean_mse_delta")),
        ),
    )


def _worst_row(rows: Sequence[dict[str, Any]], key: str = "mean_corr_delta") -> dict[str, Any]:
    if not rows:
        return {}
    return min(
        rows,
        key=lambda row: (
            _as_float(row.get(key)),
            _as_float(row.get("median_corr_delta")),
            _as_float(row.get("corr_win_fraction")),
        ),
    )


def _candidate_key(row: dict[str, Any]) -> tuple[str, str, str, float]:
    return (
        str(row.get("magnitude_mode", "")),
        str(row.get("mask_mode", "")),
        str(row.get("mechanism", "")),
        float(_as_float(row.get("gain"))),
    )


def _run_sort_key(row: dict[str, Any]) -> tuple[float, float, float]:
    return (
        _as_float(row.get("joint_best_score")),
        _as_float(row.get("direct_mean_corr_delta")),
        _as_float(row.get("absolute_best_mean_corr_delta")),
    )


def _family_sort_key(row: dict[str, Any]) -> tuple[float, float, float]:
    return (
        _as_float(row.get("direct_mean_corr_delta")),
        _as_float(row.get("absolute_best_mean_corr_delta")),
        _as_float(row.get("absolute_best_corr_win_fraction")),
    )


def build_atlas(
    *,
    absolute_compare: Path,
    variant_compare: Path,
    joint_rows: Path,
    out_dir: Path,
    top_k: int = 16,
) -> dict[str, Any]:
    absolute = _load(absolute_compare)
    variant = _load(variant_compare)
    joint = _load(joint_rows)

    direct_overall = {
        str(row.get("run_label")): row
        for row in variant.get("overall_summaries", []) or []
        if isinstance(row, dict) and row.get("run_label")
    }
    abs_candidates_by_run = _group_by(
        [row for row in absolute.get("candidate_summaries", []) or [] if isinstance(row, dict)],
        ("run_label",),
    )
    joint_by_run = _group_by([row for row in joint.get("rows", []) or [] if isinstance(row, dict)], ("run_label",))
    direct_cases_by_run_group = _group_by(
        [row for row in variant.get("case_deltas", []) or [] if isinstance(row, dict)],
        ("run_label", "group"),
    )
    abs_groups_by_run_group = _group_by(
        [row for row in absolute.get("groups", []) or [] if isinstance(row, dict) and row.get("group")],
        ("run_label", "group"),
    )

    run_rows: list[dict[str, Any]] = []
    for run in sorted(direct_overall):
        direct = direct_overall.get(run, {})
        abs_rows = abs_candidates_by_run.get((run,), [])
        best_abs = _best_row(abs_rows)
        non_bad_abs = [row for row in abs_rows if not bool(row.get("bad_baseline_rescue_dominated"))]
        best_non_bad = _best_row(non_bad_abs)
        joint_local = joint_by_run.get((run,), [])
        best_joint = max(joint_local, key=lambda row: _as_float(row.get("score")), default={})
        run_rows.append(
            {
                "run_label": run,
                "direct_mean_corr_delta": direct.get("mean_corr_delta_vs_baseline_config"),
                "direct_median_corr_delta": direct.get("median_corr_delta_vs_baseline_config"),
                "direct_corr_win_fraction": direct.get("corr_win_fraction_vs_baseline_config"),
                "direct_mean_mse_delta": direct.get("mean_mse_delta_vs_baseline_config"),
                "absolute_best_mean_corr_delta": best_abs.get("mean_corr_delta"),
                "absolute_best_median_corr_delta": best_abs.get("median_corr_delta"),
                "absolute_best_corr_win_fraction": best_abs.get("corr_win_fraction"),
                "absolute_best_status": best_abs.get("status"),
                "absolute_best_key": list(_candidate_key(best_abs)) if best_abs else None,
                "absolute_nonbad_best_mean_corr_delta": best_non_bad.get("mean_corr_delta"),
                "absolute_nonbad_best_corr_win_fraction": best_non_bad.get("corr_win_fraction"),
                "joint_best_decision": best_joint.get("decision"),
                "joint_best_score": best_joint.get("score"),
                "joint_best_absolute_median_corr_delta": best_joint.get("absolute_median_corr_delta"),
                "joint_best_absolute_corr_win_fraction": best_joint.get("absolute_corr_win_fraction"),
                "joint_best_absolute_status": best_joint.get("absolute_status"),
            }
        )
    run_rows.sort(key=_run_sort_key, reverse=True)

    family_rows: list[dict[str, Any]] = []
    for (run, group), direct_rows in direct_cases_by_run_group.items():
        if not run or not group:
            continue
        direct_summary = _summarize_values(direct_rows, "corr_delta_vs_baseline_config", "mse_delta_vs_baseline_config")
        abs_group_rows = abs_groups_by_run_group.get((run, group), [])
        best_abs_group = _best_row(abs_group_rows)
        worst_abs_group = _worst_row(abs_group_rows)
        family_rows.append(
            {
                "run_label": run,
                "group": group,
                "direct_case_count": len({row.get("case_key") for row in direct_rows}),
                "direct_mean_corr_delta": direct_summary["mean_corr_delta"],
                "direct_median_corr_delta": direct_summary["median_corr_delta"],
                "direct_corr_win_fraction": direct_summary["corr_win_fraction"],
                "absolute_best_mean_corr_delta": best_abs_group.get("mean_corr_delta"),
                "absolute_best_median_corr_delta": best_abs_group.get("median_corr_delta"),
                "absolute_best_corr_win_fraction": best_abs_group.get("corr_win_fraction"),
                "absolute_best_status": best_abs_group.get("status"),
                "absolute_best_key": list(_candidate_key(best_abs_group)) if best_abs_group else None,
                "absolute_worst_mean_corr_delta": worst_abs_group.get("mean_corr_delta"),
                "absolute_worst_median_corr_delta": worst_abs_group.get("median_corr_delta"),
                "absolute_worst_corr_win_fraction": worst_abs_group.get("corr_win_fraction"),
                "absolute_worst_status": worst_abs_group.get("status"),
            }
        )
    family_rows.sort(key=_family_sort_key, reverse=True)

    bin_rows: list[dict[str, Any]] = []
    for row in absolute.get("baseline_correlation_bins", []) or []:
        if isinstance(row, dict):
            bin_rows.append(
                {
                    "bin": row.get("bin"),
                    "case_count": row.get("case_count"),
                    "mean_baseline_target_corr": row.get("mean_baseline_target_corr"),
                    "mean_corr_delta": row.get("mean_corr_delta"),
                    "median_corr_delta": row.get("median_corr_delta"),
                    "corr_win_fraction": row.get("corr_win_fraction"),
                    "mean_mse_delta": row.get("mean_mse_delta"),
                    "positive_outlier_share": row.get("positive_outlier_share"),
                }
            )

    top_joint = [row for row in joint.get("rows", []) or [] if isinstance(row, dict)][:top_k]
    summary = {
        "schema": "circleworld_internal_phase_law_failure_atlas_v0",
        "status": "atlas_built",
        "absolute_compare": str(absolute_compare),
        "variant_compare": str(variant_compare),
        "joint_rows": str(joint_rows),
        "absolute_status": absolute.get("status"),
        "variant_status": variant.get("status"),
        "joint_status": joint.get("status"),
        "run_count": len(run_rows),
        "family_row_count": len(family_rows),
        "joint_candidate_count": joint.get("candidate_count"),
        "joint_decision_counts": joint.get("decision_counts"),
        "baseline_bins": bin_rows,
        "top_runs": run_rows[:top_k],
        "top_family_positive": family_rows[:top_k],
        "top_family_negative": sorted(
            family_rows,
            key=lambda row: _as_float(row.get("absolute_worst_mean_corr_delta")),
        )[:top_k],
        "top_joint_rows": top_joint,
        "all_runs": run_rows,
        "family_rows": family_rows,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / OUTPUT_JSON).write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (out_dir / OUTPUT_MD).write_text(_markdown(summary), encoding="utf-8")
    return summary


def _markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# Internal Phase-Law Failure Atlas",
        "",
        "This atlas separates direct recurrence improvements from absolute carrier robustness.",
        "",
        f"- status: `{summary['status']}`",
        f"- absolute status: `{summary.get('absolute_status')}`",
        f"- variant status: `{summary.get('variant_status')}`",
        f"- joint status: `{summary.get('joint_status')}`",
        f"- joint candidates: `{_fmt(summary.get('joint_candidate_count'))}`",
        f"- run count: `{_fmt(summary.get('run_count'))}`",
        f"- family rows: `{_fmt(summary.get('family_row_count'))}`",
        "",
        "## Baseline Bins",
        "",
        "| bin | cases | mean base corr | mean corr d | median corr d | corr wins | mean MSE d | outlier share |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary.get("baseline_bins", []):
        lines.append(
            "| {bin} | {cases} | {base} | {mean} | {median} | {wins} | {mse} | {outlier} |".format(
                bin=row.get("bin"),
                cases=_fmt(row.get("case_count")),
                base=_fmt(row.get("mean_baseline_target_corr")),
                mean=_fmt(row.get("mean_corr_delta")),
                median=_fmt(row.get("median_corr_delta")),
                wins=_fmt(row.get("corr_win_fraction")),
                mse=_fmt(row.get("mean_mse_delta")),
                outlier=_fmt(row.get("positive_outlier_share")),
            )
        )

    lines.extend(
        [
            "",
            "## Top Runs",
            "",
            "| run | direct mean d | direct wins | abs best mean d | abs best median d | abs best wins | abs status | joint decision | joint abs median | joint abs wins |",
            "|---|---:|---:|---:|---:|---:|---|---|---:|---:|",
        ]
    )
    for row in summary.get("top_runs", []):
        lines.append(
            "| {run} | {direct} | {dwins} | {absmean} | {absmed} | {awins} | {astatus} | {jdecision} | {jmed} | {jwins} |".format(
                run=row.get("run_label"),
                direct=_fmt(row.get("direct_mean_corr_delta")),
                dwins=_fmt(row.get("direct_corr_win_fraction")),
                absmean=_fmt(row.get("absolute_best_mean_corr_delta")),
                absmed=_fmt(row.get("absolute_best_median_corr_delta")),
                awins=_fmt(row.get("absolute_best_corr_win_fraction")),
                astatus=row.get("absolute_best_status"),
                jdecision=row.get("joint_best_decision"),
                jmed=_fmt(row.get("joint_best_absolute_median_corr_delta")),
                jwins=_fmt(row.get("joint_best_absolute_corr_win_fraction")),
            )
        )

    lines.extend(
        [
            "",
            "## Strongest Family Positives",
            "",
            "| run | group | direct mean d | direct wins | abs best mean d | abs best median d | abs best wins | abs status | key |",
            "|---|---|---:|---:|---:|---:|---:|---|---|",
        ]
    )
    for row in summary.get("top_family_positive", []):
        lines.append(
            "| {run} | {group} | {direct} | {dwins} | {absmean} | {absmed} | {awins} | {status} | `{key}` |".format(
                run=row.get("run_label"),
                group=row.get("group"),
                direct=_fmt(row.get("direct_mean_corr_delta")),
                dwins=_fmt(row.get("direct_corr_win_fraction")),
                absmean=_fmt(row.get("absolute_best_mean_corr_delta")),
                absmed=_fmt(row.get("absolute_best_median_corr_delta")),
                awins=_fmt(row.get("absolute_best_corr_win_fraction")),
                status=row.get("absolute_best_status"),
                key=row.get("absolute_best_key"),
            )
        )

    lines.extend(
        [
            "",
            "## Strongest Family Negatives",
            "",
            "| run | group | direct mean d | direct wins | abs worst mean d | abs worst median d | abs worst wins | abs status |",
            "|---|---|---:|---:|---:|---:|---:|---|",
        ]
    )
    for row in summary.get("top_family_negative", []):
        lines.append(
            "| {run} | {group} | {direct} | {dwins} | {absmean} | {absmed} | {awins} | {status} |".format(
                run=row.get("run_label"),
                group=row.get("group"),
                direct=_fmt(row.get("direct_mean_corr_delta")),
                dwins=_fmt(row.get("direct_corr_win_fraction")),
                absmean=_fmt(row.get("absolute_worst_mean_corr_delta")),
                absmed=_fmt(row.get("absolute_worst_median_corr_delta")),
                awins=_fmt(row.get("absolute_worst_corr_win_fraction")),
                status=row.get("absolute_worst_status"),
            )
        )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build an atlas of internal phase-law failure modes.")
    parser.add_argument("--absolute-compare", required=True)
    parser.add_argument("--variant-compare", required=True)
    parser.add_argument("--joint-rows", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--top-k", type=int, default=16)
    args = parser.parse_args()
    summary = build_atlas(
        absolute_compare=Path(args.absolute_compare),
        variant_compare=Path(args.variant_compare),
        joint_rows=Path(args.joint_rows),
        out_dir=Path(args.out_dir),
        top_k=int(args.top_k),
    )
    print(
        json.dumps(
            {
                "json": str(Path(args.out_dir) / OUTPUT_JSON),
                "markdown": str(Path(args.out_dir) / OUTPUT_MD),
                "status": summary["status"],
                "run_count": summary["run_count"],
                "family_row_count": summary["family_row_count"],
                "joint_status": summary["joint_status"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
