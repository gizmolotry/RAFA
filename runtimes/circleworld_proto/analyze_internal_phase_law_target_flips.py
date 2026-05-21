from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

from common_io import as_float as _as_float
from common_io import fmt_float_only as _fmt
from common_io import load_json as _load
from common_io import mean as _mean
from common_io import median as _median


OUTPUT_JSON = "internal_phase_law_target_flip_diagnostics.json"
OUTPUT_MD = "INTERNAL_PHASE_LAW_TARGET_FLIP_DIAGNOSTICS.md"
TARGET_MODE = "flat"
TARGET_MASK = "low_energy_bins"
TARGET_MECHANISM = "raw"
TARGET_GAIN = 2.0


def _win_fraction(values: Iterable[float]) -> float:
    vals = list(values)
    return float(sum(1 for value in vals if value > 0.0) / len(vals)) if vals else 0.0


def _target_candidate_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in payload.get("candidate_summaries", []) or []:
        if not isinstance(row, dict):
            continue
        if row.get("magnitude_mode") != TARGET_MODE:
            continue
        if row.get("mask_mode") != TARGET_MASK:
            continue
        if row.get("mechanism") != TARGET_MECHANISM:
            continue
        rows.append(row)
    return rows


def _target_case_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in payload.get("case_deltas", []) or []:
        if not isinstance(row, dict):
            continue
        if row.get("magnitude_mode") != TARGET_MODE:
            continue
        if row.get("mask_mode") != TARGET_MASK:
            continue
        if row.get("mechanism") != TARGET_MECHANISM:
            continue
        rows.append(row)
    return rows


def _select_run(candidate_rows: list[dict[str, Any]], requested: str | None) -> str | None:
    if requested:
        return requested
    gain2 = [row for row in candidate_rows if abs(_as_float(row.get("gain")) - TARGET_GAIN) < 1e-9]
    if not gain2:
        return None
    gain2.sort(
        key=lambda row: (
            _as_float(row.get("corr_win_fraction_vs_baseline_config")),
            _as_float(row.get("median_corr_delta_vs_baseline_config")),
            _as_float(row.get("mean_corr_delta_vs_baseline_config")),
        ),
        reverse=True,
    )
    return str(gain2[0].get("run_label"))


def _summarize_values(values: list[float]) -> dict[str, Any]:
    return {
        "case_count": len(values),
        "mean_corr_delta": _mean(values),
        "median_corr_delta": _median(values),
        "corr_win_fraction": _win_fraction(values),
        "positive_count": sum(1 for value in values if value > 0.0),
        "negative_count": sum(1 for value in values if value < 0.0),
        "zero_count": sum(1 for value in values if value == 0.0),
    }


def _case_key(row: dict[str, Any]) -> str:
    return str(row.get("case_key") or row.get("case_name") or "")


def analyze(variant_compare: Path, out_dir: Path, target_run: str | None = None) -> dict[str, Any]:
    payload = _load(variant_compare)
    candidate_rows = _target_candidate_rows(payload)
    case_rows = _target_case_rows(payload)
    selected_run = _select_run(candidate_rows, target_run)
    selected_rows = [row for row in case_rows if row.get("run_label") == selected_run]

    gain_summaries: list[dict[str, Any]] = []
    for row in sorted(candidate_rows, key=lambda item: (str(item.get("run_label")), _as_float(item.get("gain")))):
        gain_summaries.append(
            {
                "run_label": row.get("run_label"),
                "gain": row.get("gain"),
                "case_count": row.get("case_count"),
                "mean_corr_delta": row.get("mean_corr_delta_vs_baseline_config"),
                "median_corr_delta": row.get("median_corr_delta_vs_baseline_config"),
                "corr_win_fraction": row.get("corr_win_fraction_vs_baseline_config"),
                "mean_mse_delta": row.get("mean_mse_delta_vs_baseline_config"),
                "status": row.get("status"),
            }
        )

    best_by_gain: list[dict[str, Any]] = []
    gains = sorted({_as_float(row.get("gain")) for row in candidate_rows})
    for gain in gains:
        rows = [row for row in candidate_rows if abs(_as_float(row.get("gain")) - gain) < 1e-9]
        rows.sort(
            key=lambda row: (
                _as_float(row.get("corr_win_fraction_vs_baseline_config")),
                _as_float(row.get("median_corr_delta_vs_baseline_config")),
                _as_float(row.get("mean_corr_delta_vs_baseline_config")),
            ),
            reverse=True,
        )
        if rows:
            row = rows[0]
            best_by_gain.append(
                {
                    "run_label": row.get("run_label"),
                    "gain": row.get("gain"),
                    "case_count": row.get("case_count"),
                    "mean_corr_delta": row.get("mean_corr_delta_vs_baseline_config"),
                    "median_corr_delta": row.get("median_corr_delta_vs_baseline_config"),
                    "corr_win_fraction": row.get("corr_win_fraction_vs_baseline_config"),
                    "mean_mse_delta": row.get("mean_mse_delta_vs_baseline_config"),
                    "status": row.get("status"),
                }
            )

    by_case: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in selected_rows:
        by_case[_case_key(row)].append(row)
    case_flip_rows: list[dict[str, Any]] = []
    gain1 = min(gains) if gains else 0.0
    gain2 = TARGET_GAIN
    for case_key, rows in by_case.items():
        rows.sort(key=lambda row: _as_float(row.get("gain")))
        values = {_as_float(row.get("gain")): _as_float(row.get("corr_delta_vs_baseline_config")) for row in rows}
        first = values.get(gain1)
        last = values.get(gain2)
        signs = ["positive" if value > 0 else "negative" if value < 0 else "zero" for value in values.values()]
        if first is None or last is None:
            category = "missing_endpoint"
        elif first <= 0.0 < last:
            category = "rescued_by_gain"
        elif first > 0.0 >= last:
            category = "lost_by_gain"
        elif all(sign == "positive" for sign in signs):
            category = "positive_all_gains"
        elif all(sign in ("negative", "zero") for sign in signs):
            category = "nonpositive_all_gains"
        else:
            category = "mixed_nonendpoint"
        first_row = rows[0] if rows else {}
        case_flip_rows.append(
            {
                "case_key": case_key,
                "case_name": first_row.get("case_name"),
                "group": first_row.get("group"),
                "category": category,
                "gain_min_corr_delta": first,
                "gain_target_corr_delta": last,
                "max_corr_delta": max(values.values()) if values else None,
                "min_corr_delta": min(values.values()) if values else None,
                "gain_values": values,
            }
        )

    category_counts: dict[str, int] = defaultdict(int)
    for row in case_flip_rows:
        category_counts[str(row["category"])] += 1

    target_gain_rows = [
        row
        for row in selected_rows
        if abs(_as_float(row.get("gain")) - TARGET_GAIN) < 1e-9
    ]
    target_values = [_as_float(row.get("corr_delta_vs_baseline_config")) for row in target_gain_rows]
    by_group: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in target_gain_rows:
        by_group[str(row.get("group") or "unknown")].append(row)
    group_rows: list[dict[str, Any]] = []
    leave_one_out_rows: list[dict[str, Any]] = []
    for group, rows in sorted(by_group.items()):
        values = [_as_float(row.get("corr_delta_vs_baseline_config")) for row in rows]
        group_summary = _summarize_values(values)
        group_summary["group"] = group
        group_rows.append(group_summary)
        remaining = [
            _as_float(row.get("corr_delta_vs_baseline_config"))
            for row in target_gain_rows
            if str(row.get("group") or "unknown") != group
        ]
        loo = _summarize_values(remaining)
        loo["excluded_group"] = group
        leave_one_out_rows.append(loo)

    top_negative = sorted(
        target_gain_rows,
        key=lambda row: _as_float(row.get("corr_delta_vs_baseline_config")),
    )[:12]
    top_positive = sorted(
        target_gain_rows,
        key=lambda row: _as_float(row.get("corr_delta_vs_baseline_config")),
        reverse=True,
    )[:12]

    summary = {
        "schema": "circleworld_internal_phase_law_target_flip_diagnostics_v0",
        "status": "completed",
        "variant_compare": str(variant_compare),
        "target_row": {
            "magnitude_mode": TARGET_MODE,
            "mask_mode": TARGET_MASK,
            "mechanism": TARGET_MECHANISM,
            "target_gain": TARGET_GAIN,
        },
        "selected_run": selected_run,
        "gain_count": len(gains),
        "candidate_row_count": len(candidate_rows),
        "case_delta_row_count": len(case_rows),
        "selected_target_gain_summary": _summarize_values(target_values),
        "category_counts": dict(sorted(category_counts.items())),
        "best_by_gain": best_by_gain,
        "gain_summaries": gain_summaries,
        "group_rows": group_rows,
        "leave_one_out_rows": leave_one_out_rows,
        "top_negative_cases": [
            {
                "case_key": row.get("case_key"),
                "case_name": row.get("case_name"),
                "group": row.get("group"),
                "corr_delta": row.get("corr_delta_vs_baseline_config"),
                "mse_delta": row.get("mse_delta_vs_baseline_config"),
            }
            for row in top_negative
        ],
        "top_positive_cases": [
            {
                "case_key": row.get("case_key"),
                "case_name": row.get("case_name"),
                "group": row.get("group"),
                "corr_delta": row.get("corr_delta_vs_baseline_config"),
                "mse_delta": row.get("mse_delta_vs_baseline_config"),
            }
            for row in top_positive
        ],
        "case_flip_rows": case_flip_rows,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / OUTPUT_JSON).write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (out_dir / OUTPUT_MD).write_text(_markdown(summary), encoding="utf-8")
    return summary


def _markdown(summary: dict[str, Any]) -> str:
    target = summary["target_row"]
    selected = summary.get("selected_target_gain_summary", {})
    lines = [
        "# Internal Phase-Law Target Flip Diagnostics",
        "",
        f"- status: `{summary.get('status')}`",
        f"- selected run: `{summary.get('selected_run')}`",
        f"- target row: `{target['magnitude_mode']}` / `{target['mask_mode']}` / `{target['mechanism']}` / gain `{_fmt(target['target_gain'])}`",
        f"- candidate rows: `{summary.get('candidate_row_count')}`",
        f"- case delta rows: `{summary.get('case_delta_row_count')}`",
        f"- target gain mean / median / wins: `{_fmt(selected.get('mean_corr_delta'))}` / `{_fmt(selected.get('median_corr_delta'))}` / `{_fmt(selected.get('corr_win_fraction'))}`",
        f"- target gain positive / negative / zero cases: `{selected.get('positive_count')}` / `{selected.get('negative_count')}` / `{selected.get('zero_count')}`",
        "",
        "## Gain Winners",
        "",
        "| gain | run | mean corr d | median corr d | wins | mean MSE d |",
        "|---:|---|---:|---:|---:|---:|",
    ]
    for row in summary.get("best_by_gain", []) or []:
        lines.append(
            "| {gain} | {run} | {mean} | {median} | {wins} | {mse} |".format(
                gain=_fmt(row.get("gain")),
                run=row.get("run_label"),
                mean=_fmt(row.get("mean_corr_delta")),
                median=_fmt(row.get("median_corr_delta")),
                wins=_fmt(row.get("corr_win_fraction")),
                mse=_fmt(row.get("mean_mse_delta")),
            )
        )

    lines.extend(["", "## Sign Categories", "", "| category | cases |", "|---|---:|"])
    for category, count in (summary.get("category_counts") or {}).items():
        lines.append(f"| {category} | {count} |")

    lines.extend(
        [
            "",
            "## Family Rows At Target Gain",
            "",
            "| group | cases | mean corr d | median corr d | wins | positives | negatives | zeros |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in summary.get("group_rows", []) or []:
        lines.append(
            "| {group} | {cases} | {mean} | {median} | {wins} | {pos} | {neg} | {zero} |".format(
                group=row.get("group"),
                cases=row.get("case_count"),
                mean=_fmt(row.get("mean_corr_delta")),
                median=_fmt(row.get("median_corr_delta")),
                wins=_fmt(row.get("corr_win_fraction")),
                pos=row.get("positive_count"),
                neg=row.get("negative_count"),
                zero=row.get("zero_count"),
            )
        )

    lines.extend(
        [
            "",
            "## Leave One Out At Target Gain",
            "",
            "| excluded group | cases | mean corr d | median corr d | wins |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for row in summary.get("leave_one_out_rows", []) or []:
        lines.append(
            "| {group} | {cases} | {mean} | {median} | {wins} |".format(
                group=row.get("excluded_group"),
                cases=row.get("case_count"),
                mean=_fmt(row.get("mean_corr_delta")),
                median=_fmt(row.get("median_corr_delta")),
                wins=_fmt(row.get("corr_win_fraction")),
            )
        )

    for title, key in [
        ("Top Negative Cases", "top_negative_cases"),
        ("Top Positive Cases", "top_positive_cases"),
    ]:
        lines.extend(["", f"## {title}", "", "| group | case | corr d | MSE d |", "|---|---|---:|---:|"])
        for row in summary.get(key, []) or []:
            lines.append(
                "| {group} | {case} | {corr} | {mse} |".format(
                    group=row.get("group"),
                    case=row.get("case_name"),
                    corr=_fmt(row.get("corr_delta")),
                    mse=_fmt(row.get("mse_delta")),
                )
            )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze target-row gain sign flips for internal phase-law scouts.")
    parser.add_argument("--variant-compare", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--target-run", default="")
    args = parser.parse_args()
    summary = analyze(
        variant_compare=Path(args.variant_compare),
        out_dir=Path(args.out_dir),
        target_run=args.target_run or None,
    )
    print(
        json.dumps(
            {
                "json": str(Path(args.out_dir) / OUTPUT_JSON),
                "markdown": str(Path(args.out_dir) / OUTPUT_MD),
                "status": summary["status"],
                "selected_run": summary.get("selected_run"),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
