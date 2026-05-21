from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

from score_phase_native_audio_reentry_guard import (  # noqa: E402
    _as_float,
    _collect_delta_rows,
    _copy_last_baseline,
    _future_access_clean,
    _key,
    _load_required_artifacts,
    _mean,
    _median,
    _win_fraction,
)


OUTPUT_JSON = "phase_native_audio_reentry_oracle_score.json"
OUTPUT_MD = "PHASE_NATIVE_AUDIO_REENTRY_ORACLE_SCORE.md"


def _json_load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _case_group(case_name: str) -> str:
    return str(case_name).split("__", 1)[0]


def _row_objective(row: dict[str, Any]) -> float:
    corr = _as_float(row.get("corr_delta_vs_copy_last"))
    mse = _as_float(row.get("mse_delta_vs_copy_last"))
    loop = _as_float(row.get("loop_delta_vs_copy_last"))
    reentry = _as_float(row.get("reentry_delta_vs_copy_last"))
    corr_gain0 = _as_float(row.get("corr_delta_vs_gain0"))
    return (
        corr
        + 0.25 * min(-mse, 0.05)
        + 0.10 * max(corr_gain0, 0.0)
        - 1.25 * max(reentry, 0.0)
        - 0.35 * max(loop, 0.0)
        - 0.35 * max(mse, 0.0)
    )


def _aggregate_selected(rows: Sequence[dict[str, Any]], label: str) -> dict[str, Any]:
    corr = [_as_float(row.get("corr_delta_vs_copy_last")) for row in rows]
    mse = [_as_float(row.get("mse_delta_vs_copy_last")) for row in rows]
    loop = [_as_float(row.get("loop_delta_vs_copy_last")) for row in rows]
    reentry = [_as_float(row.get("reentry_delta_vs_copy_last")) for row in rows]
    corr_gain0 = [_as_float(row.get("corr_delta_vs_gain0")) for row in rows]
    source_counts: dict[str, int] = defaultdict(int)
    key_counts: dict[str, int] = defaultdict(int)
    group_counts: dict[str, int] = defaultdict(int)
    for row in rows:
        source_counts[str(row.get("delta_source", ""))] += 1
        key_counts[" / ".join(str(part) for part in _key(row))] += 1
        group_counts[str(row.get("group", _case_group(str(row.get("case_name", "")))))] += 1
    strict = bool(
        rows
        and _mean(corr) > 0.0
        and _mean(mse) <= 0.0
        and _mean(loop) <= 0.0
        and _mean(reentry) <= 0.0
        and _mean(corr_gain0) >= 0.0
    )
    return {
        "label": label,
        "case_count": len(rows),
        "mean_corr_delta_vs_copy_last": _mean(corr),
        "median_corr_delta_vs_copy_last": _median(corr),
        "corr_win_fraction_vs_copy_last": _win_fraction(corr, positive=True),
        "mean_mse_delta_vs_copy_last": _mean(mse),
        "mse_win_fraction_vs_copy_last": _win_fraction(mse, positive=False),
        "mean_loop_delta_vs_copy_last": _mean(loop),
        "loop_win_fraction_vs_copy_last": _win_fraction(loop, positive=False),
        "mean_reentry_delta_vs_copy_last": _mean(reentry),
        "reentry_win_fraction_vs_copy_last": _win_fraction(reentry, positive=False),
        "mean_corr_delta_vs_gain0": _mean(corr_gain0),
        "strict_guard_pass": strict,
        "mean_row_objective": _mean([_row_objective(row) for row in rows]),
        "source_counts": dict(sorted(source_counts.items())),
        "key_counts_top": dict(sorted(key_counts.items(), key=lambda item: item[1], reverse=True)[:12]),
        "group_counts": dict(sorted(group_counts.items())),
    }


def _best_global(rows: Sequence[dict[str, Any]]) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
    grouped: dict[tuple[str, str, str, float], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if abs(_as_float(row.get("gain"))) <= 1.0e-12:
            continue
        grouped[_key(row)].append(row)
    candidates = [_aggregate_selected(group, "global") | {"key": key} for key, group in grouped.items()]
    if not candidates:
        return None, []
    best = max(candidates, key=lambda item: _as_float(item.get("mean_row_objective")))
    best_rows = grouped[best["key"]]
    best["key"] = {
        "magnitude_mode": best["key"][0],
        "mask_mode": best["key"][1],
        "mechanism": best["key"][2],
        "gain": best["key"][3],
    }
    return best, best_rows


def _best_per_group(rows: Sequence[dict[str, Any]]) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    by_group_key: dict[str, dict[tuple[str, str, str, float], list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        if abs(_as_float(row.get("gain"))) <= 1.0e-12:
            continue
        by_group_key[str(row["group"])][_key(row)].append(row)
    selected: list[dict[str, Any]] = []
    choices: dict[str, Any] = {}
    for group, key_rows in sorted(by_group_key.items()):
        aggregates = [(_aggregate_selected(group_rows, group), key, group_rows) for key, group_rows in key_rows.items()]
        best_agg, best_key, best_rows = max(aggregates, key=lambda item: _as_float(item[0].get("mean_row_objective")))
        selected.extend(best_rows)
        choices[group] = {
            "key": {
                "magnitude_mode": best_key[0],
                "mask_mode": best_key[1],
                "mechanism": best_key[2],
                "gain": best_key[3],
            },
            "summary": best_agg,
        }
    return _aggregate_selected(selected, "family_oracle"), choices, selected


def _best_per_case(rows: Sequence[dict[str, Any]]) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    by_case: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if abs(_as_float(row.get("gain"))) <= 1.0e-12:
            continue
        by_case[str(row["case_name"])].append(row)
    selected: list[dict[str, Any]] = []
    choices_by_group: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for case_name, case_rows in sorted(by_case.items()):
        best = max(case_rows, key=_row_objective)
        selected.append(best)
        choices_by_group[str(best["group"])][" / ".join(str(part) for part in _key(best))] += 1
    return _aggregate_selected(selected, "case_oracle"), selected, {
        group: dict(sorted(counts.items(), key=lambda item: item[1], reverse=True)[:8])
        for group, counts in sorted(choices_by_group.items())
    }


def _group_breakdown(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["group"])].append(row)
    return {group: _aggregate_selected(group_rows, group) for group, group_rows in sorted(grouped.items())}


def _collect_all_delta_rows(
    suite_json: Path,
    delta_jsons: Sequence[Path],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    prefix, flat, _ = _load_required_artifacts(suite_json)
    copy_last = _copy_last_baseline(prefix)
    payloads = [prefix, flat]
    rows: list[dict[str, Any]] = []
    for delta_json in delta_jsons:
        payload = _json_load(delta_json)
        payloads.append(payload)
        source_rows = _collect_delta_rows(payload, copy_last)
        for row in source_rows:
            row["delta_source"] = str(delta_json)
        rows.extend(source_rows)
    return {"future_access_clean": _future_access_clean(payloads), "copy_last_case_count": len(copy_last)}, rows


def _markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# Phase-Native Audio Reentry Oracle Score",
        "",
        "This is an oracle/ceiling analysis over already executed phase-delta rows.",
        "It asks whether the failure is global selection, family routing, or missing mechanism capacity.",
        "",
        "## Summary",
        "",
        f"- Status: `{summary['status']}`",
        f"- Future access clean: `{summary['future_access_clean']}`",
        f"- Delta rows: `{summary['delta_row_count']}`",
        f"- Global strict pass: `{summary['global']['strict_guard_pass'] if summary.get('global') else False}`",
        f"- Family-oracle strict pass: `{summary['family_oracle']['strict_guard_pass']}`",
        f"- Case-oracle strict pass: `{summary['case_oracle']['strict_guard_pass']}`",
        "",
        "## Main Results",
        "",
        "| selector | corr-copy | mse-copy | loop-copy | reentry-copy | corr-gain0 | reentry wins | strict |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for label in ("global", "family_oracle", "case_oracle"):
        row = summary.get(label)
        if not row:
            continue
        lines.append(
            "| `{label}` | {corr:.9f} | {mse:.9f} | {loop:.9f} | {reentry:.9f} | {cgain:.9f} | {rwin:.6f} | `{strict}` |".format(
                label=label,
                corr=_as_float(row.get("mean_corr_delta_vs_copy_last")),
                mse=_as_float(row.get("mean_mse_delta_vs_copy_last")),
                loop=_as_float(row.get("mean_loop_delta_vs_copy_last")),
                reentry=_as_float(row.get("mean_reentry_delta_vs_copy_last")),
                cgain=_as_float(row.get("mean_corr_delta_vs_gain0")),
                rwin=_as_float(row.get("reentry_win_fraction_vs_copy_last")),
                strict=bool(row.get("strict_guard_pass")),
            )
        )
    lines.extend(["", "## Global Key", "", "```json", json.dumps(summary.get("global", {}).get("key", {}), indent=2), "```"])
    lines.extend(["", "## Group Breakdown For Case Oracle", ""])
    lines.extend(
        [
            "| group | corr-copy | mse-copy | loop-copy | reentry-copy | strict |",
            "| --- | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    for group, row in summary["case_oracle_group_breakdown"].items():
        lines.append(
            "| `{group}` | {corr:.9f} | {mse:.9f} | {loop:.9f} | {reentry:.9f} | `{strict}` |".format(
                group=group,
                corr=_as_float(row.get("mean_corr_delta_vs_copy_last")),
                mse=_as_float(row.get("mean_mse_delta_vs_copy_last")),
                loop=_as_float(row.get("mean_loop_delta_vs_copy_last")),
                reentry=_as_float(row.get("mean_reentry_delta_vs_copy_last")),
                strict=bool(row.get("strict_guard_pass")),
            )
        )
    lines.extend(["", "## Interpretation", "", summary["interpretation"]])
    return "\n".join(lines) + "\n"


def score_oracle(suite_json: Path, delta_jsons: Sequence[Path], out_dir: Path) -> dict[str, Any]:
    meta, rows = _collect_all_delta_rows(suite_json, delta_jsons)
    global_summary, global_rows = _best_global(rows)
    family_summary, family_choices, family_rows = _best_per_group(rows)
    case_summary, case_rows, case_choices = _best_per_case(rows)
    if bool(case_summary.get("strict_guard_pass")):
        status = "case_oracle_reentry_guard_possible"
        interpretation = (
            "The mechanism pool contains enough per-case choices to pass the reentry guard. "
            "The next step should train a selector/router, not invent a new operator."
        )
    elif bool(family_summary.get("strict_guard_pass")):
        status = "family_oracle_reentry_guard_possible"
        interpretation = (
            "Family routing can pass the guard. The next step should learn or predeclare family-conditioned routing."
        )
    elif global_summary and bool(global_summary.get("strict_guard_pass")):
        status = "global_reentry_guard_possible"
        interpretation = "A single global row can pass the guard; rerun it as a narrow predeclared profile."
    else:
        status = "oracle_reentry_guard_not_possible"
        interpretation = (
            "Even case-oracle routing over the current mechanism pool does not pass the strict guard. "
            "This points to missing operator capacity or an overly strict first-chunk copy-last floor, not just bad selection."
        )
    summary = {
        "schema": "phase_native_audio_reentry_oracle_score_v1",
        "status": status,
        "suite_json": str(suite_json),
        "delta_jsons": [str(path) for path in delta_jsons],
        "future_access_clean": bool(meta["future_access_clean"]),
        "copy_last_case_count": int(meta["copy_last_case_count"]),
        "delta_row_count": len(rows),
        "global": global_summary,
        "family_oracle": family_summary,
        "case_oracle": case_summary,
        "family_choices": family_choices,
        "case_oracle_choice_counts_by_group": case_choices,
        "global_group_breakdown": _group_breakdown(global_rows) if global_rows else {},
        "family_oracle_group_breakdown": _group_breakdown(family_rows),
        "case_oracle_group_breakdown": _group_breakdown(case_rows),
        "interpretation": interpretation,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / OUTPUT_JSON).write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (out_dir / OUTPUT_MD).write_text(_markdown(summary), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Oracle ceiling analysis for phase-native audio reentry.")
    parser.add_argument("--suite-json", required=True, type=Path)
    parser.add_argument("--delta-json", required=True, action="append", type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    args = parser.parse_args()
    summary = score_oracle(args.suite_json, args.delta_json, args.out_dir)
    print(
        json.dumps(
            {
                "status": summary["status"],
                "json": str(Path(args.out_dir) / OUTPUT_JSON),
                "markdown": str(Path(args.out_dir) / OUTPUT_MD),
                "global_strict": bool(summary.get("global", {}).get("strict_guard_pass")),
                "family_strict": bool(summary["family_oracle"].get("strict_guard_pass")),
                "case_strict": bool(summary["case_oracle"].get("strict_guard_pass")),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
