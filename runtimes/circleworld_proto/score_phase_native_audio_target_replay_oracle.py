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
from score_phase_native_audio_reentry_metric_audit import _wave_metrics  # noqa: E402


OUTPUT_JSON = "phase_native_audio_target_replay_oracle_score.json"
OUTPUT_MD = "PHASE_NATIVE_AUDIO_TARGET_REPLAY_ORACLE_SCORE.md"
UNSPECIFIED_SOURCE = "<unspecified>"


def _json_load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _case_group(case_name: str) -> str:
    return str(case_name).split("__", 1)[0]


def _target_reentry_by_case(continuation_summary: dict[str, Any]) -> dict[str, float]:
    out: dict[str, float] = {}
    for row in continuation_summary.get("rows", []):
        target = Path(str(row.get("target_future_wav", "")))
        if target.exists():
            out[str(row.get("name"))] = _as_float(_wave_metrics(target).get("first_chunk_reentry"))
    return out


def _copy_last_harm_by_case(
    continuation_summary: dict[str, Any],
    target_reentry: dict[str, float],
    *,
    margin: float,
) -> dict[str, float]:
    out: dict[str, float] = {}
    for row in continuation_summary.get("rows", []):
        case_name = str(row.get("name"))
        target = target_reentry.get(case_name, 0.0)
        loop = row.get("methods", {}).get("baseline_copy_last", {}).get("loop_reentry", {})
        pred = _as_float(loop.get("first_chunk_reentry"))
        out[case_name] = max(0.0, pred - target - float(margin))
    return out


def _augment_rows(
    rows: Sequence[dict[str, Any]],
    target_reentry: dict[str, float],
    copy_last_harm: dict[str, float],
    *,
    margin: float,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in rows:
        case_name = str(row.get("case_name"))
        target = target_reentry.get(case_name, 0.0)
        pred = _as_float(row.get("first_chunk_reentry"))
        harm = max(0.0, pred - target - float(margin))
        copy_harm = copy_last_harm.get(case_name, 0.0)
        item = dict(row)
        item.update(
            {
                "target_first_chunk_reentry": target,
                "target_norm_reentry_delta": pred - target,
                "harmful_replay_excess": harm,
                "copy_last_harmful_replay_excess": copy_harm,
                "harmful_replay_excess_delta_vs_copy_last": harm - copy_harm,
            }
        )
        out.append(item)
    return out


def _row_objective(row: dict[str, Any]) -> float:
    corr = _as_float(row.get("corr_delta_vs_copy_last"))
    mse = _as_float(row.get("mse_delta_vs_copy_last"))
    loop = _as_float(row.get("loop_delta_vs_copy_last"))
    harm_delta = _as_float(row.get("harmful_replay_excess_delta_vs_copy_last"))
    corr_gain0 = _as_float(row.get("corr_delta_vs_gain0"))
    return (
        corr
        + 0.20 * min(-mse, 0.05)
        + 0.10 * max(corr_gain0, 0.0)
        - 1.50 * max(harm_delta, 0.0)
        - 0.25 * max(loop, 0.0)
        - 0.25 * max(mse, 0.0)
    )


def _dedupe_best_by_case(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    best_by_case: dict[str, dict[str, Any]] = {}
    for row in rows:
        case_name = str(row.get("case_name", ""))
        if not case_name:
            continue
        if case_name not in best_by_case or _row_objective(row) > _row_objective(best_by_case[case_name]):
            best_by_case[case_name] = row
    return [best_by_case[name] for name in sorted(best_by_case)]


def _source_count_key(row: dict[str, Any]) -> str:
    source = str(row.get("delta_source", "")).strip()
    return source if source else UNSPECIFIED_SOURCE


def _aggregate_selected(rows: Sequence[dict[str, Any]], label: str) -> dict[str, Any]:
    raw_row_count = len(rows)
    rows = _dedupe_best_by_case(rows)
    corr = [_as_float(row.get("corr_delta_vs_copy_last")) for row in rows]
    mse = [_as_float(row.get("mse_delta_vs_copy_last")) for row in rows]
    loop = [_as_float(row.get("loop_delta_vs_copy_last")) for row in rows]
    reentry = [_as_float(row.get("reentry_delta_vs_copy_last")) for row in rows]
    harm = [_as_float(row.get("harmful_replay_excess")) for row in rows]
    copy_harm = [_as_float(row.get("copy_last_harmful_replay_excess")) for row in rows]
    harm_delta = [_as_float(row.get("harmful_replay_excess_delta_vs_copy_last")) for row in rows]
    corr_gain0 = [_as_float(row.get("corr_delta_vs_gain0")) for row in rows]
    key_counts: dict[str, int] = defaultdict(int)
    group_counts: dict[str, int] = defaultdict(int)
    source_counts: dict[str, int] = defaultdict(int)
    for row in rows:
        key_counts[" / ".join(str(part) for part in _key(row))] += 1
        group_counts[str(row.get("group", _case_group(str(row.get("case_name", "")))))] += 1
        source_counts[_source_count_key(row)] += 1
    strict = bool(
        rows
        and _mean(corr) > 0.0
        and _mean(mse) <= 0.0
        and _mean(loop) <= 0.0
        and _mean(harm_delta) <= 0.0
        and _mean(corr_gain0) >= 0.0
    )
    diagnostic = bool(rows and _mean(corr) > 0.0 and _mean(mse) <= 0.0 and _mean(harm_delta) <= 0.0)
    return {
        "label": label,
        "case_count": len(rows),
        "raw_selected_row_count": raw_row_count,
        "mean_corr_delta_vs_copy_last": _mean(corr),
        "median_corr_delta_vs_copy_last": _median(corr),
        "corr_win_fraction_vs_copy_last": _win_fraction(corr, positive=True),
        "mean_mse_delta_vs_copy_last": _mean(mse),
        "mse_win_fraction_vs_copy_last": _win_fraction(mse, positive=False),
        "mean_loop_delta_vs_copy_last": _mean(loop),
        "loop_win_fraction_vs_copy_last": _win_fraction(loop, positive=False),
        "mean_reentry_delta_vs_copy_last": _mean(reentry),
        "reentry_win_fraction_vs_copy_last": _win_fraction(reentry, positive=False),
        "mean_harmful_replay_excess": _mean(harm),
        "mean_copy_last_harmful_replay_excess": _mean(copy_harm),
        "mean_harmful_replay_excess_delta_vs_copy_last": _mean(harm_delta),
        "harmful_replay_win_fraction_vs_copy_last": _win_fraction(harm_delta, positive=False),
        "mean_corr_delta_vs_gain0": _mean(corr_gain0),
        "strict_target_replay_pass": strict,
        "diagnostic_target_replay_pass": diagnostic,
        "mean_row_objective": _mean([_row_objective(row) for row in rows]),
        "key_counts_top": dict(sorted(key_counts.items(), key=lambda item: item[1], reverse=True)[:12]),
        "group_counts": dict(sorted(group_counts.items())),
        "source_counts": dict(sorted(source_counts.items())),
    }


def _best_global(rows: Sequence[dict[str, Any]]) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
    grouped: dict[tuple[str, str, str, float], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if abs(_as_float(row.get("gain"))) <= 1.0e-12:
            continue
        grouped[_key(row)].append(row)
    if not grouped:
        return None, []
    candidates = [_aggregate_selected(group, "global") | {"key_tuple": key} for key, group in grouped.items()]
    best = max(candidates, key=lambda item: _as_float(item.get("mean_row_objective")))
    key = best.pop("key_tuple")
    best["key"] = {
        "magnitude_mode": key[0],
        "mask_mode": key[1],
        "mechanism": key[2],
        "gain": key[3],
    }
    return best, grouped[key]


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


def _collect_rows(
    suite_json: Path,
    delta_jsons: Sequence[Path],
    margin: float,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    prefix, flat, _ = _load_required_artifacts(suite_json)
    target_reentry = _target_reentry_by_case(prefix)
    copy_last_harm = _copy_last_harm_by_case(prefix, target_reentry, margin=margin)
    copy_last = _copy_last_baseline(prefix)
    payloads = [prefix, flat]
    rows: list[dict[str, Any]] = []
    for delta_json in delta_jsons:
        payload = _json_load(delta_json)
        payloads.append(payload)
        source_rows = _collect_delta_rows(payload, copy_last)
        for row in source_rows:
            row["delta_source"] = str(delta_json)
        rows.extend(_augment_rows(source_rows, target_reentry, copy_last_harm, margin=margin))
    return {
        "future_access_clean": _future_access_clean(payloads),
        "copy_last_case_count": len(copy_last),
        "target_reentry_case_count": len(target_reentry),
    }, rows


def _markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# Phase-Native Audio Target Replay Oracle Score",
        "",
        "This oracle optimizes target-normalized harmful replay excess rather than the",
        "old self-only first-chunk reentry scalar.",
        "",
        "## Summary",
        "",
        f"- Status: `{summary['status']}`",
        f"- Future access clean: `{summary['future_access_clean']}`",
        f"- Target margin: `{summary['target_margin']}`",
        f"- Delta rows: `{summary['delta_row_count']}`",
        "",
        "## Main Results",
        "",
        "| selector | corr-copy | MSE-copy | loop-copy | reentry-copy | harm-delta | harm wins | corr-gain0 | strict |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for label in ("global", "family_oracle", "case_oracle"):
        row = summary.get(label)
        if not row:
            continue
        lines.append(
            "| `{label}` | {corr:.9f} | {mse:.9f} | {loop:.9f} | {reentry:.9f} | {harm:.9f} | {hwin:.6f} | {cgain:.9f} | `{strict}` |".format(
                label=label,
                corr=_as_float(row.get("mean_corr_delta_vs_copy_last")),
                mse=_as_float(row.get("mean_mse_delta_vs_copy_last")),
                loop=_as_float(row.get("mean_loop_delta_vs_copy_last")),
                reentry=_as_float(row.get("mean_reentry_delta_vs_copy_last")),
                harm=_as_float(row.get("mean_harmful_replay_excess_delta_vs_copy_last")),
                hwin=_as_float(row.get("harmful_replay_win_fraction_vs_copy_last")),
                cgain=_as_float(row.get("mean_corr_delta_vs_gain0")),
                strict=bool(row.get("strict_target_replay_pass")),
            )
        )
    lines.extend(["", "## Global Key", "", "```json", json.dumps(summary.get("global", {}).get("key", {}), indent=2), "```"])
    lines.extend(["", "## Case Oracle Group Breakdown", ""])
    lines.extend(
        [
            "| group | corr-copy | MSE-copy | loop-copy | harm-delta | strict |",
            "| --- | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    for group, row in summary["case_oracle_group_breakdown"].items():
        lines.append(
            "| `{group}` | {corr:.9f} | {mse:.9f} | {loop:.9f} | {harm:.9f} | `{strict}` |".format(
                group=group,
                corr=_as_float(row.get("mean_corr_delta_vs_copy_last")),
                mse=_as_float(row.get("mean_mse_delta_vs_copy_last")),
                loop=_as_float(row.get("mean_loop_delta_vs_copy_last")),
                harm=_as_float(row.get("mean_harmful_replay_excess_delta_vs_copy_last")),
                strict=bool(row.get("strict_target_replay_pass")),
            )
        )
    lines.extend(["", "## Interpretation", "", summary["interpretation"]])
    return "\n".join(lines) + "\n"


def score_target_replay_oracle(
    suite_json: Path,
    delta_jsons: Sequence[Path],
    out_dir: Path,
    *,
    margin: float,
) -> dict[str, Any]:
    meta, rows = _collect_rows(suite_json, delta_jsons, margin)
    global_summary, global_rows = _best_global(rows)
    family_summary, family_choices, family_rows = _best_per_group(rows)
    case_summary, case_rows, case_choices = _best_per_case(rows)
    if global_summary and bool(global_summary.get("strict_target_replay_pass")):
        status = "global_target_replay_pass"
        interpretation = "A single global row passes the target-normalized harmful replay guard; rerun it narrowly."
    elif bool(family_summary.get("strict_target_replay_pass")):
        status = "family_oracle_target_replay_pass"
        interpretation = (
            "Family-level routing can pass the target-normalized harmful replay guard. "
            "The next step should learn or predeclare family-conditioned routing."
        )
    elif bool(case_summary.get("strict_target_replay_pass")):
        status = "case_oracle_target_replay_pass"
        interpretation = (
            "The current mechanism pool can pass the target-normalized harmful replay guard under per-case routing. "
            "The next step should train a router/selector on prefix-only features."
        )
    else:
        status = "target_replay_oracle_not_possible"
        interpretation = (
            "Even target-normalized harmful replay does not pass under the current oracle pool. "
            "The next step needs a new operator or a different audio objective."
        )
    summary = {
        "schema": "phase_native_audio_target_replay_oracle_score_v1",
        "status": status,
        "suite_json": str(suite_json),
        "delta_jsons": [str(path) for path in delta_jsons],
        "future_access_clean": bool(meta["future_access_clean"]),
        "target_margin": float(margin),
        "copy_last_case_count": int(meta["copy_last_case_count"]),
        "target_reentry_case_count": int(meta["target_reentry_case_count"]),
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
    parser = argparse.ArgumentParser(description="Oracle ceiling for target-normalized harmful replay.")
    parser.add_argument("--suite-json", required=True, type=Path)
    parser.add_argument("--delta-json", required=True, action="append", type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--target-margin", type=float, default=0.01)
    args = parser.parse_args()
    summary = score_target_replay_oracle(args.suite_json, args.delta_json, args.out_dir, margin=args.target_margin)
    print(
        json.dumps(
            {
                "status": summary["status"],
                "json": str(Path(args.out_dir) / OUTPUT_JSON),
                "markdown": str(Path(args.out_dir) / OUTPUT_MD),
                "global_strict": bool(summary.get("global", {}).get("strict_target_replay_pass")),
                "family_strict": bool(summary["family_oracle"].get("strict_target_replay_pass")),
                "case_strict": bool(summary["case_oracle"].get("strict_target_replay_pass")),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
