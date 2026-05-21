from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from common_io import as_float as _as_float
from common_io import fmt_float_only as _fmt
from common_io import load_json as _load
from common_io import mean as _mean
from common_io import median as _median
from common_io import write_json as _write_json


OUTPUT_JSON = "internal_phase_law_support_router_oracle.json"
OUTPUT_MD = "INTERNAL_PHASE_LAW_SUPPORT_ROUTER_ORACLE.md"


def _win_fraction(values: Iterable[float]) -> float:
    vals = list(values)
    return float(sum(1 for value in vals if value > 0.0) / len(vals)) if vals else 0.0


def _summarize(values: list[float]) -> dict[str, Any]:
    return {
        "case_count": len(values),
        "mean_corr_delta": _mean(values),
        "median_corr_delta": _median(values),
        "corr_win_fraction": _win_fraction(values),
        "positive_count": sum(1 for value in values if value > 0.0),
        "negative_count": sum(1 for value in values if value < 0.0),
        "zero_count": sum(1 for value in values if value == 0.0),
    }


def _rank_summary(row: dict[str, Any]) -> tuple[float, float, float]:
    return (
        _as_float(row.get("corr_win_fraction")),
        _as_float(row.get("median_corr_delta")),
        _as_float(row.get("mean_corr_delta")),
    )


def _global_best_fixed_mask(mask_rows: list[dict[str, Any]]) -> str | None:
    rows = list(mask_rows)
    rows.sort(key=_rank_summary, reverse=True)
    return str(rows[0].get("mask_mode")) if rows else None


def _family_policy(family_rows: list[dict[str, Any]], fallback_mask: str | None) -> dict[str, str]:
    policy: dict[str, str] = {}
    for row in family_rows:
        family = str(row.get("group") or "unknown")
        mask = row.get("mask_mode") or fallback_mask
        if mask:
            policy[family] = str(mask)
    return policy


def _mask_values(row: dict[str, Any]) -> dict[str, float]:
    raw = row.get("mask_values") if isinstance(row.get("mask_values"), dict) else {}
    return {str(mask): _as_float(value) for mask, value in raw.items()}


def _apply_fixed(rows: list[dict[str, Any]], mask: str | None) -> tuple[list[float], list[dict[str, Any]]]:
    values: list[float] = []
    case_rows: list[dict[str, Any]] = []
    if not mask:
        return values, case_rows
    for row in rows:
        masks = _mask_values(row)
        if mask not in masks:
            continue
        value = masks[mask]
        values.append(value)
        case_rows.append(
            {
                "case_name": row.get("case_name"),
                "group": row.get("group"),
                "chosen_mask": mask,
                "corr_delta": value,
            }
        )
    return values, case_rows


def _apply_family_policy(
    rows: list[dict[str, Any]],
    policy: dict[str, str],
    fallback_mask: str | None,
) -> tuple[list[float], list[dict[str, Any]]]:
    values: list[float] = []
    case_rows: list[dict[str, Any]] = []
    for row in rows:
        family = str(row.get("group") or "unknown")
        chosen = policy.get(family, fallback_mask)
        masks = _mask_values(row)
        if chosen not in masks:
            continue
        value = masks[str(chosen)]
        values.append(value)
        case_rows.append(
            {
                "case_name": row.get("case_name"),
                "group": family,
                "chosen_mask": chosen,
                "corr_delta": value,
            }
        )
    return values, case_rows


def _apply_case_oracle(rows: list[dict[str, Any]], reverse: bool) -> tuple[list[float], list[dict[str, Any]]]:
    values: list[float] = []
    case_rows: list[dict[str, Any]] = []
    for row in rows:
        masks = _mask_values(row)
        if not masks:
            continue
        chosen, value = sorted(masks.items(), key=lambda item: (item[1], item[0]), reverse=reverse)[0]
        values.append(value)
        case_rows.append(
            {
                "case_name": row.get("case_name"),
                "group": row.get("group"),
                "chosen_mask": chosen,
                "corr_delta": value,
                "mask_values": masks,
            }
        )
    return values, case_rows


def _delta_rows(
    proposed_rows: list[dict[str, Any]],
    baseline_rows: list[dict[str, Any]],
    limit: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    baseline_by_case = {str(row.get("case_name")): row for row in baseline_rows}
    rows: list[dict[str, Any]] = []
    for row in proposed_rows:
        base = baseline_by_case.get(str(row.get("case_name")))
        if not base:
            continue
        delta = _as_float(row.get("corr_delta")) - _as_float(base.get("corr_delta"))
        rows.append(
            {
                "case_name": row.get("case_name"),
                "group": row.get("group"),
                "chosen_mask": row.get("chosen_mask"),
                "baseline_mask": base.get("chosen_mask"),
                "corr_delta": row.get("corr_delta"),
                "baseline_corr_delta": base.get("corr_delta"),
                "delta_vs_baseline_mask": delta,
            }
        )
    rows.sort(key=lambda item: _as_float(item.get("delta_vs_baseline_mask")), reverse=True)
    return rows[:limit], list(reversed(rows[-limit:]))


def _policy_counts(rows: list[dict[str, Any]]) -> dict[str, Any]:
    overall = Counter(str(row.get("chosen_mask")) for row in rows)
    by_family: dict[str, Counter[str]] = {}
    for row in rows:
        family = str(row.get("group") or "unknown")
        by_family.setdefault(family, Counter())[str(row.get("chosen_mask"))] += 1
    return {
        "overall": dict(sorted(overall.items())),
        "by_family": {family: dict(sorted(counts.items())) for family, counts in sorted(by_family.items())},
    }


def analyze(family_mask_diagnostics: Path, out_dir: Path, limit: int = 15) -> dict[str, Any]:
    payload = _load(family_mask_diagnostics)
    paired_rows = [
        row for row in payload.get("selected_paired_case_mask_rows", []) or [] if isinstance(row, dict)
    ]
    mask_rows = [row for row in payload.get("selected_target_mask_rows", []) or [] if isinstance(row, dict)]
    family_rows = [
        row for row in payload.get("selected_best_mask_by_family", []) or [] if isinstance(row, dict)
    ]

    global_mask = _global_best_fixed_mask(mask_rows)
    low_values, low_case_rows = _apply_fixed(paired_rows, "low_energy_bins")
    global_values, global_case_rows = _apply_fixed(paired_rows, global_mask)
    policy = _family_policy(family_rows, global_mask)
    family_values, family_case_rows = _apply_family_policy(paired_rows, policy, global_mask)
    oracle_values, oracle_case_rows = _apply_case_oracle(paired_rows, reverse=True)
    anti_values, anti_case_rows = _apply_case_oracle(paired_rows, reverse=False)

    top_family_vs_global, bottom_family_vs_global = _delta_rows(family_case_rows, global_case_rows, limit)
    top_oracle_vs_global, bottom_oracle_vs_global = _delta_rows(oracle_case_rows, global_case_rows, limit)

    global_summary = _summarize(global_values)
    family_summary = _summarize(family_values)
    oracle_summary = _summarize(oracle_values)
    low_summary = _summarize(low_values)
    anti_summary = _summarize(anti_values)
    family_win_gain = _as_float(family_summary.get("corr_win_fraction")) - _as_float(
        global_summary.get("corr_win_fraction")
    )
    oracle_win_gain = _as_float(oracle_summary.get("corr_win_fraction")) - _as_float(
        global_summary.get("corr_win_fraction")
    )
    family_mean_gain = _as_float(family_summary.get("mean_corr_delta")) - _as_float(
        global_summary.get("mean_corr_delta")
    )
    if family_win_gain >= 0.03 and family_mean_gain >= 0.0:
        status = "family_router_promising"
    elif oracle_win_gain >= 0.10:
        status = "case_router_promising_family_router_insufficient"
    else:
        status = "routing_ceiling_small"

    summary = {
        "schema": "circleworld_internal_phase_law_support_router_oracle_v0",
        "status": status,
        "family_mask_diagnostics": str(family_mask_diagnostics),
        "selected_run": payload.get("selected_run"),
        "target_row": payload.get("target_row"),
        "global_fixed_mask": global_mask,
        "family_policy": policy,
        "summaries": {
            "low_energy_fixed": low_summary,
            "global_fixed_best": global_summary,
            "family_oracle_router": family_summary,
            "case_oracle_router": oracle_summary,
            "case_anti_oracle": anti_summary,
        },
        "deltas_vs_global_fixed": {
            "family_mean_corr_delta_gain": family_mean_gain,
            "family_corr_win_fraction_gain": family_win_gain,
            "case_oracle_mean_corr_delta_gain": _as_float(oracle_summary.get("mean_corr_delta"))
            - _as_float(global_summary.get("mean_corr_delta")),
            "case_oracle_corr_win_fraction_gain": oracle_win_gain,
            "low_energy_mean_corr_delta_gap": _as_float(low_summary.get("mean_corr_delta"))
            - _as_float(global_summary.get("mean_corr_delta")),
            "low_energy_corr_win_fraction_gap": _as_float(low_summary.get("corr_win_fraction"))
            - _as_float(global_summary.get("corr_win_fraction")),
        },
        "policy_counts": {
            "family_oracle_router": _policy_counts(family_case_rows),
            "case_oracle_router": _policy_counts(oracle_case_rows),
            "case_anti_oracle": _policy_counts(anti_case_rows),
        },
        "top_family_router_wins_vs_global": top_family_vs_global,
        "top_family_router_losses_vs_global": bottom_family_vs_global,
        "top_case_oracle_wins_vs_global": top_oracle_vs_global,
        "top_case_oracle_losses_vs_global": bottom_oracle_vs_global,
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
    lines = [
        "# Internal Phase-Law Support Router Oracle",
        "",
        f"- Status: `{summary.get('status')}`",
        f"- Selected run: `{summary.get('selected_run')}`",
        f"- Global fixed mask: `{summary.get('global_fixed_mask')}`",
        f"- Source: `{summary.get('family_mask_diagnostics')}`",
        "",
        "## Router Summaries",
    ]
    summaries = summary.get("summaries", {})
    lines.extend(
        _table(
            ["router", "cases", "mean", "median", "wins", "pos", "neg"],
            [
                [
                    name,
                    row.get("case_count"),
                    row.get("mean_corr_delta"),
                    row.get("median_corr_delta"),
                    row.get("corr_win_fraction"),
                    row.get("positive_count"),
                    row.get("negative_count"),
                ]
                for name, row in summaries.items()
                if isinstance(row, dict)
            ],
        )
    )
    lines.extend(["", "## Deltas Vs Global Fixed"])
    for key, value in (summary.get("deltas_vs_global_fixed") or {}).items():
        lines.append(f"- `{key}`: `{_fmt(value)}`")
    lines.extend(["", "## Family Policy"])
    for family, mask in sorted((summary.get("family_policy") or {}).items()):
        lines.append(f"- `{family}` -> `{mask}`")
    lines.extend(["", "## Top Family-Router Wins Vs Global"])
    lines.extend(
        _table(
            ["case", "family", "chosen", "global", "delta"],
            [
                [
                    row.get("case_name"),
                    row.get("group"),
                    row.get("chosen_mask"),
                    row.get("baseline_mask"),
                    row.get("delta_vs_baseline_mask"),
                ]
                for row in summary.get("top_family_router_wins_vs_global", [])
            ],
        )
    )
    lines.extend(["", "## Top Family-Router Losses Vs Global"])
    lines.extend(
        _table(
            ["case", "family", "chosen", "global", "delta"],
            [
                [
                    row.get("case_name"),
                    row.get("group"),
                    row.get("chosen_mask"),
                    row.get("baseline_mask"),
                    row.get("delta_vs_baseline_mask"),
                ]
                for row in summary.get("top_family_router_losses_vs_global", [])
            ],
        )
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--family-mask-diagnostics", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=15)
    args = parser.parse_args()
    summary = analyze(args.family_mask_diagnostics, args.out_dir, args.limit)
    print(
        json.dumps(
            {
                "status": summary.get("status"),
                "global_fixed_mask": summary.get("global_fixed_mask"),
                "summaries": summary.get("summaries"),
                "output_json": str(args.out_dir / OUTPUT_JSON),
                "output_md": str(args.out_dir / OUTPUT_MD),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
