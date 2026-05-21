from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

from common_io import as_float as _as_float
from common_io import fmt_float_only as _fmt
from common_io import load_json as _load
from common_io import mean as _mean
from common_io import median as _median
from common_io import write_json as _write_json


OUTPUT_JSON = "internal_phase_law_family_mask_diagnostics.json"
OUTPUT_MD = "INTERNAL_PHASE_LAW_FAMILY_MASK_DIAGNOSTICS.md"

TARGET_MODE = "flat"
TARGET_MECHANISM = "raw"
TARGET_GAIN = 2.0
TARGET_MASK = "low_energy_bins"
MASKS = ("all_bins", "high_energy_bins", "low_energy_bins")


def _win_fraction(values: Iterable[float]) -> float:
    vals = list(values)
    return float(sum(1 for value in vals if value > 0.0) / len(vals)) if vals else 0.0


def _case_key(row: dict[str, Any]) -> str:
    return str(row.get("case_key") or row.get("case_name") or "")


def _delta(row: dict[str, Any]) -> float:
    return _as_float(row.get("corr_delta_vs_baseline_config"))


def _mse_delta(row: dict[str, Any]) -> float:
    return _as_float(row.get("mse_delta_vs_baseline_config"))


def _summarize_values(values: list[float], mse_values: list[float] | None = None) -> dict[str, Any]:
    mse_values = mse_values or []
    return {
        "case_count": len(values),
        "mean_corr_delta": _mean(values),
        "median_corr_delta": _median(values),
        "corr_win_fraction": _win_fraction(values),
        "positive_count": sum(1 for value in values if value > 0.0),
        "negative_count": sum(1 for value in values if value < 0.0),
        "zero_count": sum(1 for value in values if value == 0.0),
        "mean_mse_delta": _mean(mse_values),
        "mse_win_fraction": _win_fraction([-value for value in mse_values]),
    }


def _row_sort_key(row: dict[str, Any]) -> tuple[float, float, float]:
    return (
        _as_float(row.get("corr_win_fraction_vs_baseline_config")),
        _as_float(row.get("median_corr_delta_vs_baseline_config")),
        _as_float(row.get("mean_corr_delta_vs_baseline_config")),
    )


def _candidate_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in payload.get("candidate_summaries", []) or []:
        if not isinstance(row, dict):
            continue
        if row.get("magnitude_mode") != TARGET_MODE:
            continue
        if row.get("mechanism") != TARGET_MECHANISM:
            continue
        if row.get("mask_mode") not in MASKS:
            continue
        rows.append(row)
    return rows


def _case_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in payload.get("case_deltas", []) or []:
        if not isinstance(row, dict):
            continue
        if row.get("magnitude_mode") != TARGET_MODE:
            continue
        if row.get("mechanism") != TARGET_MECHANISM:
            continue
        if row.get("mask_mode") not in MASKS:
            continue
        rows.append(row)
    return rows


def _select_target_run(
    candidate_rows: list[dict[str, Any]],
    requested: str | None,
    objective_score: dict[str, Any] | None = None,
) -> tuple[str | None, str]:
    if requested:
        return requested, "explicit --selected-run"
    objective_run = (objective_score or {}).get("target_low_energy_best_run")
    if objective_run:
        return str(objective_run), "objective-score target_low_energy_best_run"
    target_rows = [
        row
        for row in candidate_rows
        if row.get("mask_mode") == TARGET_MASK and abs(_as_float(row.get("gain")) - TARGET_GAIN) < 1e-9
    ]
    target_rows.sort(key=_row_sort_key, reverse=True)
    return (
        str(target_rows[0].get("run_label")) if target_rows else None,
        "best low_energy_bins target-gain row by win, median, mean",
    )


def _best_candidate_row(candidate_rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    rows = list(candidate_rows)
    rows.sort(key=_row_sort_key, reverse=True)
    return rows[0] if rows else None


def _summarize_grouped(
    rows: list[dict[str, Any]],
    keys: tuple[str, ...],
    extra: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    buckets: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        buckets[tuple(row.get(key) for key in keys)].append(row)

    summaries: list[dict[str, Any]] = []
    for key_values, group_rows in sorted(buckets.items(), key=lambda item: tuple(str(value) for value in item[0])):
        values = [_delta(row) for row in group_rows]
        mse_values = [_mse_delta(row) for row in group_rows]
        summary = _summarize_values(values, mse_values)
        for key, value in zip(keys, key_values):
            summary[key] = value
        if extra:
            summary.update(extra)
        summaries.append(summary)
    return summaries


def _target_gain_rows(rows: list[dict[str, Any]], selected_run: str | None) -> list[dict[str, Any]]:
    return [
        row
        for row in rows
        if row.get("run_label") == selected_run and abs(_as_float(row.get("gain")) - TARGET_GAIN) < 1e-9
    ]


def _pair_target_masks(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_case: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        by_case[_case_key(row)][str(row.get("mask_mode"))] = row

    paired: list[dict[str, Any]] = []
    for case_key, masks in sorted(by_case.items()):
        first = next(iter(masks.values()), {})
        values = {mask: _delta(masks[mask]) for mask in MASKS if mask in masks}
        if not values:
            continue
        best_mask, best_delta = max(values.items(), key=lambda item: (item[1], item[0]))
        low = values.get("low_energy_bins")
        high = values.get("high_energy_bins")
        all_bins = values.get("all_bins")
        row = {
            "case_key": case_key,
            "case_name": first.get("case_name"),
            "group": first.get("group"),
            "role": first.get("role"),
            "best_mask": best_mask,
            "best_corr_delta": best_delta,
            "mask_values": values,
            "low_minus_all": None if low is None or all_bins is None else low - all_bins,
            "low_minus_high": None if low is None or high is None else low - high,
            "high_minus_all": None if high is None or all_bins is None else high - all_bins,
            "low_positive_all_nonpositive": bool(low is not None and all_bins is not None and low > 0.0 >= all_bins),
            "all_positive_low_nonpositive": bool(low is not None and all_bins is not None and all_bins > 0.0 >= low),
            "low_positive_high_nonpositive": bool(low is not None and high is not None and low > 0.0 >= high),
            "high_positive_low_nonpositive": bool(low is not None and high is not None and high > 0.0 >= low),
        }
        paired.append(row)
    return paired


def _best_mask_counts(paired_rows: list[dict[str, Any]]) -> dict[str, Any]:
    total = Counter(str(row.get("best_mask")) for row in paired_rows)
    by_family: dict[str, Counter[str]] = defaultdict(Counter)
    for row in paired_rows:
        by_family[str(row.get("group") or "unknown")][str(row.get("best_mask"))] += 1
    return {
        "overall": dict(sorted(total.items())),
        "by_family": {family: dict(sorted(counts.items())) for family, counts in sorted(by_family.items())},
    }


def _best_mask_by_family(family_mask_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_family: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in family_mask_rows:
        by_family[str(row.get("group") or "unknown")].append(row)

    best_rows: list[dict[str, Any]] = []
    for family, rows in sorted(by_family.items()):
        ranked = sorted(
            rows,
            key=lambda row: (
                _as_float(row.get("corr_win_fraction")),
                _as_float(row.get("median_corr_delta")),
                _as_float(row.get("mean_corr_delta")),
            ),
            reverse=True,
        )
        best = dict(ranked[0]) if ranked else {"group": family}
        best["ranked_masks"] = ranked
        best_rows.append(best)
    return best_rows


def _leave_one_family_out_by_mask(target_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    families = sorted({str(row.get("group") or "unknown") for row in target_rows})
    output: list[dict[str, Any]] = []
    for mask in MASKS:
        mask_rows = [row for row in target_rows if row.get("mask_mode") == mask]
        for family in families:
            remaining = [row for row in mask_rows if str(row.get("group") or "unknown") != family]
            summary = _summarize_values([_delta(row) for row in remaining], [_mse_delta(row) for row in remaining])
            summary["mask_mode"] = mask
            summary["excluded_group"] = family
            output.append(summary)
    return output


def _top_cases(paired_rows: list[dict[str, Any]], key: str, reverse: bool, limit: int = 12) -> list[dict[str, Any]]:
    rows = [row for row in paired_rows if row.get(key) is not None]
    rows.sort(key=lambda row: _as_float(row.get(key)), reverse=reverse)
    return rows[:limit]


def analyze(
    variant_compare: Path,
    out_dir: Path,
    selected_run: str | None = None,
    target_gain: float = TARGET_GAIN,
    objective_score_path: Path | None = None,
) -> dict[str, Any]:
    global TARGET_GAIN
    TARGET_GAIN = target_gain

    payload = _load(variant_compare)
    objective_score = _load(objective_score_path) if objective_score_path else None
    candidate_rows = _candidate_rows(payload)
    case_rows = _case_rows(payload)
    selected, selected_basis = _select_target_run(candidate_rows, selected_run, objective_score)
    global_best = _best_candidate_row(candidate_rows)

    selected_rows = [row for row in case_rows if row.get("run_label") == selected]
    selected_target_rows = _target_gain_rows(case_rows, selected)
    selected_global_rows = [
        row for row in selected_rows if abs(_as_float(row.get("gain")) - TARGET_GAIN) < 1e-9
    ]
    paired_target_rows = _pair_target_masks(selected_target_rows)
    family_mask_target_rows = _summarize_grouped(selected_target_rows, ("group", "mask_mode"))
    mask_target_rows = _summarize_grouped(selected_target_rows, ("mask_mode",))

    global_best_run = str(global_best.get("run_label")) if global_best else None
    global_best_mask = str(global_best.get("mask_mode")) if global_best else None
    global_best_gain = _as_float(global_best.get("gain")) if global_best else 0.0
    global_best_rows = [
        row
        for row in case_rows
        if row.get("run_label") == global_best_run
        and row.get("mask_mode") == global_best_mask
        and abs(_as_float(row.get("gain")) - global_best_gain) < 1e-9
    ]

    sign_flip_counts = {
        "low_positive_all_nonpositive": sum(1 for row in paired_target_rows if row["low_positive_all_nonpositive"]),
        "all_positive_low_nonpositive": sum(1 for row in paired_target_rows if row["all_positive_low_nonpositive"]),
        "low_positive_high_nonpositive": sum(1 for row in paired_target_rows if row["low_positive_high_nonpositive"]),
        "high_positive_low_nonpositive": sum(1 for row in paired_target_rows if row["high_positive_low_nonpositive"]),
    }

    summary = {
        "schema": "circleworld_internal_phase_law_family_mask_diagnostics_v0",
        "status": "completed",
        "variant_compare": str(variant_compare),
        "target_row": {
            "magnitude_mode": TARGET_MODE,
            "mechanism": TARGET_MECHANISM,
            "target_gain": TARGET_GAIN,
            "target_mask": TARGET_MASK,
            "masks": list(MASKS),
        },
        "selected_run": selected,
        "selected_run_basis": selected_basis,
        "objective_score": str(objective_score_path) if objective_score_path else None,
        "global_best_direct_row": global_best,
        "candidate_row_count": len(candidate_rows),
        "case_delta_row_count": len(case_rows),
        "selected_target_mask_rows": mask_target_rows,
        "selected_family_mask_target_rows": family_mask_target_rows,
        "selected_best_mask_by_family": _best_mask_by_family(family_mask_target_rows),
        "selected_best_mask_counts": _best_mask_counts(paired_target_rows),
        "selected_paired_case_mask_rows": paired_target_rows,
        "selected_sign_flip_counts": sign_flip_counts,
        "selected_leave_one_family_out_by_mask": _leave_one_family_out_by_mask(selected_target_rows),
        "global_best_direct_summary": _summarize_values(
            [_delta(row) for row in global_best_rows],
            [_mse_delta(row) for row in global_best_rows],
        ),
        "candidate_mask_gain_rows": [
            {
                "run_label": row.get("run_label"),
                "mask_mode": row.get("mask_mode"),
                "gain": row.get("gain"),
                "case_count": row.get("case_count"),
                "mean_corr_delta": row.get("mean_corr_delta_vs_baseline_config"),
                "median_corr_delta": row.get("median_corr_delta_vs_baseline_config"),
                "corr_win_fraction": row.get("corr_win_fraction_vs_baseline_config"),
                "mean_mse_delta": row.get("mean_mse_delta_vs_baseline_config"),
                "mse_win_fraction": row.get("mse_win_fraction_vs_baseline_config"),
                "status": row.get("status"),
            }
            for row in sorted(
                candidate_rows,
                key=lambda row: (str(row.get("run_label")), str(row.get("mask_mode")), _as_float(row.get("gain"))),
            )
        ],
        "top_low_vs_all_cases": _top_cases(paired_target_rows, "low_minus_all", reverse=True),
        "top_all_vs_low_cases": _top_cases(paired_target_rows, "low_minus_all", reverse=False),
        "top_high_vs_all_cases": _top_cases(paired_target_rows, "high_minus_all", reverse=True),
        "top_all_vs_high_cases": _top_cases(paired_target_rows, "high_minus_all", reverse=False),
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    _write_json(out_dir / OUTPUT_JSON, summary)
    (out_dir / OUTPUT_MD).write_text(render_markdown(summary), encoding="utf-8")
    return summary


def _table(headers: list[str], rows: list[list[Any]]) -> list[str]:
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(_fmt(value) for value in row) + " |")
    return lines


def render_markdown(summary: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append("# Internal Phase-Law Family/Support-Mask Diagnostics")
    lines.append("")
    lines.append(f"- Status: `{summary.get('status')}`")
    lines.append(f"- Variant compare: `{summary.get('variant_compare')}`")
    lines.append(f"- Selected run: `{summary.get('selected_run')}`")
    row = summary.get("target_row", {})
    lines.append(
        "- Target slice: "
        f"`{row.get('magnitude_mode')}/{row.get('mechanism')}/gain={row.get('target_gain')}`"
    )
    global_best = summary.get("global_best_direct_row") or {}
    if global_best:
        lines.append(
            "- Global best direct row: "
            f"`{global_best.get('run_label')}` / `{global_best.get('mask_mode')}` / "
            f"gain `{global_best.get('gain')}` / mean `{_fmt(global_best.get('mean_corr_delta_vs_baseline_config'))}` / "
            f"median `{_fmt(global_best.get('median_corr_delta_vs_baseline_config'))}` / "
            f"wins `{_fmt(global_best.get('corr_win_fraction_vs_baseline_config'))}`"
        )
    lines.append("")

    lines.append("## Selected Run Mask Summary")
    lines.extend(
        _table(
            ["mask", "cases", "mean", "median", "wins", "pos", "neg", "mse_mean"],
            [
                [
                    row.get("mask_mode"),
                    row.get("case_count"),
                    row.get("mean_corr_delta"),
                    row.get("median_corr_delta"),
                    row.get("corr_win_fraction"),
                    row.get("positive_count"),
                    row.get("negative_count"),
                    row.get("mean_mse_delta"),
                ]
                for row in summary.get("selected_target_mask_rows", [])
            ],
        )
    )
    lines.append("")

    lines.append("## Best Mask By Family")
    lines.extend(
        _table(
            ["family", "best_mask", "cases", "mean", "median", "wins", "pos", "neg"],
            [
                [
                    row.get("group"),
                    row.get("mask_mode"),
                    row.get("case_count"),
                    row.get("mean_corr_delta"),
                    row.get("median_corr_delta"),
                    row.get("corr_win_fraction"),
                    row.get("positive_count"),
                    row.get("negative_count"),
                ]
                for row in summary.get("selected_best_mask_by_family", [])
            ],
        )
    )
    lines.append("")

    lines.append("## Best Mask Counts")
    counts = summary.get("selected_best_mask_counts", {})
    lines.append(f"- Overall: `{json.dumps(counts.get('overall', {}), sort_keys=True)}`")
    for family, family_counts in (counts.get("by_family") or {}).items():
        lines.append(f"- `{family}`: `{json.dumps(family_counts, sort_keys=True)}`")
    lines.append("")

    lines.append("## Sign-Flip Counts")
    for key, value in sorted((summary.get("selected_sign_flip_counts") or {}).items()):
        lines.append(f"- `{key}`: `{value}`")
    lines.append("")

    lines.append("## Leave-One-Family-Out By Mask")
    loo_rows = summary.get("selected_leave_one_family_out_by_mask", [])
    lines.extend(
        _table(
            ["mask", "excluded", "cases", "mean", "median", "wins"],
            [
                [
                    row.get("mask_mode"),
                    row.get("excluded_group"),
                    row.get("case_count"),
                    row.get("mean_corr_delta"),
                    row.get("median_corr_delta"),
                    row.get("corr_win_fraction"),
                ]
                for row in loo_rows
            ],
        )
    )
    lines.append("")

    lines.append("## Top Low-Energy Mask Wins Over All-Bins")
    lines.extend(
        _table(
            ["case", "family", "low_minus_all", "best_mask"],
            [
                [row.get("case_name"), row.get("group"), row.get("low_minus_all"), row.get("best_mask")]
                for row in summary.get("top_low_vs_all_cases", [])
            ],
        )
    )
    lines.append("")

    lines.append("## Top All-Bins Wins Over Low-Energy Mask")
    lines.extend(
        _table(
            ["case", "family", "low_minus_all", "best_mask"],
            [
                [row.get("case_name"), row.get("group"), row.get("low_minus_all"), row.get("best_mask")]
                for row in summary.get("top_all_vs_low_cases", [])
            ],
        )
    )
    lines.append("")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant-compare", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--selected-run", default=None)
    parser.add_argument("--target-gain", type=float, default=TARGET_GAIN)
    parser.add_argument("--objective-score", type=Path, default=None)
    args = parser.parse_args()
    summary = analyze(args.variant_compare, args.out_dir, args.selected_run, args.target_gain, args.objective_score)
    print(
        json.dumps(
            {
                "status": summary.get("status"),
                "selected_run": summary.get("selected_run"),
                "global_best_direct_row": summary.get("global_best_direct_row"),
                "output_json": str(args.out_dir / OUTPUT_JSON),
                "output_md": str(args.out_dir / OUTPUT_MD),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
