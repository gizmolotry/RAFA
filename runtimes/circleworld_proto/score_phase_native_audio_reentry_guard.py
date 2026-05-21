from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence


ROOT = Path(__file__).resolve().parents[2]
OUTPUT_JSON = "phase_native_audio_reentry_guard_score.json"
OUTPUT_MD = "PHASE_NATIVE_AUDIO_REENTRY_GUARD_SCORE.md"

DEFAULT_VIEWS = {
    "all": set(),
    "no_razor": {"razor_single_source_probe"},
    "no_razor_no_buzzy": {"razor_single_source_probe", "buzzy_synth_control"},
}


def _json_load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(value):
        return default
    return value


def _mean(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    return float(sum(float(value) for value in values) / len(values))


def _median(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(float(value) for value in values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[mid]
    return 0.5 * (ordered[mid - 1] + ordered[mid])


def _win_fraction(values: Sequence[float], *, positive: bool = True) -> float:
    if not values:
        return 0.0
    if positive:
        return float(sum(1 for value in values if float(value) > 0.0) / len(values))
    return float(sum(1 for value in values if float(value) <= 0.0) / len(values))


def _case_group(case_name: str) -> str:
    return str(case_name).split("__", 1)[0]


def _task_artifacts(suite: dict[str, Any]) -> dict[str, Path]:
    artifacts: dict[str, Path] = {}
    for row in suite.get("scorecard", {}).get("task_scores", []):
        name = str(row.get("name", ""))
        artifact = row.get("artifact")
        if name and artifact:
            artifacts[name] = Path(str(artifact))
    for row in suite.get("commands", []):
        name = str(row.get("name", ""))
        artifact = row.get("expected_json")
        artifacts.setdefault(name, Path(str(artifact)) if artifact else Path())
    return artifacts


def _load_required_artifacts(
    suite_json: Path,
    delta_json_override: Path | None = None,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    suite = _json_load(suite_json)
    artifacts = _task_artifacts(suite)
    try:
        prefix = _json_load(artifacts["continuation_prefix_hold_copyphase"])
        flat = _json_load(artifacts["continuation_flat_copyphase"])
        delta = _json_load(delta_json_override) if delta_json_override is not None else _json_load(artifacts["delta_mechanism_probe_core"])
    except KeyError as exc:
        raise RuntimeError(f"Suite JSON is missing expected task artifact: {exc}") from exc
    return prefix, flat, delta


def _copy_last_baseline(continuation_summary: dict[str, Any]) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for row in continuation_summary.get("rows", []):
        item = row.get("methods", {}).get("baseline_copy_last", {})
        metrics = item.get("metrics", {})
        loop = item.get("loop_reentry", {})
        out[str(row.get("name"))] = {
            "corr": _as_float(metrics.get("corr")),
            "mse": _as_float(metrics.get("mse")),
            "loop_autocorr_peak": _as_float(loop.get("loop_autocorr_peak")),
            "first_chunk_reentry": _as_float(loop.get("first_chunk_reentry")),
        }
    return out


def _future_access_clean(payloads: Iterable[dict[str, Any]]) -> bool:
    keys = (
        "future_target_magnitude_reused",
        "target_future_stft_magnitude_accessed",
        "future_target_phase_reused",
        "target_future_stft_phase_accessed",
    )
    return not any(any(bool(payload.get(key)) for key in keys) for payload in payloads)


def _gain0_index(case_rows: Sequence[dict[str, Any]]) -> dict[tuple[str, str, str], dict[str, Any]]:
    index: dict[tuple[str, str, str], dict[str, Any]] = {}
    for row in case_rows:
        if abs(_as_float(row.get("gain"))) <= 1.0e-12:
            index[(str(row.get("magnitude_mode")), str(row.get("mask_mode")), str(row.get("mechanism")))] = row
    return index


def _collect_delta_rows(
    delta_summary: dict[str, Any],
    copy_last: dict[str, dict[str, float]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for case in delta_summary.get("cases", []):
        case_name = str(case.get("name"))
        baseline = copy_last.get(case_name)
        if baseline is None:
            continue
        gain0 = _gain0_index(case.get("rows", []))
        for row in case.get("rows", []):
            mode = str(row.get("magnitude_mode"))
            mask = str(row.get("mask_mode"))
            mechanism = str(row.get("mechanism"))
            base = gain0.get((mode, mask, mechanism))
            if base is None:
                continue
            rows.append(
                {
                    "case_name": case_name,
                    "group": _case_group(case_name),
                    "magnitude_mode": mode,
                    "mask_mode": mask,
                    "mechanism": mechanism,
                    "gain": _as_float(row.get("gain")),
                    "target_corr": _as_float(row.get("target_corr")),
                    "target_mse": _as_float(row.get("target_mse")),
                    "loop_autocorr_peak": _as_float(row.get("loop_autocorr_peak")),
                    "first_chunk_reentry": _as_float(row.get("first_chunk_reentry")),
                    "corr_delta_vs_copy_last": _as_float(row.get("target_corr")) - baseline["corr"],
                    "mse_delta_vs_copy_last": _as_float(row.get("target_mse")) - baseline["mse"],
                    "loop_delta_vs_copy_last": _as_float(row.get("loop_autocorr_peak")) - baseline["loop_autocorr_peak"],
                    "reentry_delta_vs_copy_last": _as_float(row.get("first_chunk_reentry"))
                    - baseline["first_chunk_reentry"],
                    "corr_delta_vs_gain0": _as_float(row.get("target_corr")) - _as_float(base.get("target_corr")),
                    "mse_delta_vs_gain0": _as_float(row.get("target_mse")) - _as_float(base.get("target_mse")),
                    "loop_delta_vs_gain0": _as_float(row.get("loop_autocorr_peak"))
                    - _as_float(base.get("loop_autocorr_peak")),
                    "reentry_delta_vs_gain0": _as_float(row.get("first_chunk_reentry"))
                    - _as_float(base.get("first_chunk_reentry")),
                    "future_target_magnitude_reused": bool(row.get("future_target_magnitude_reused")),
                    "target_future_stft_magnitude_accessed": bool(row.get("target_future_stft_magnitude_accessed")),
                    "future_target_phase_reused": bool(row.get("future_target_phase_reused")),
                    "target_future_stft_phase_accessed": bool(row.get("target_future_stft_phase_accessed")),
                }
            )
    return rows


def _aggregate(rows: Sequence[dict[str, Any]], view: str, excluded_groups: set[str]) -> list[dict[str, Any]]:
    by_key: dict[tuple[str, str, str, float], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row["group"] in excluded_groups:
            continue
        if abs(_as_float(row["gain"])) <= 1.0e-12:
            continue
        key = (row["magnitude_mode"], row["mask_mode"], row["mechanism"], _as_float(row["gain"]))
        by_key[key].append(row)

    out: list[dict[str, Any]] = []
    for (mode, mask, mechanism, gain), group in by_key.items():
        corr_copy = [_as_float(row["corr_delta_vs_copy_last"]) for row in group]
        mse_copy = [_as_float(row["mse_delta_vs_copy_last"]) for row in group]
        loop_copy = [_as_float(row["loop_delta_vs_copy_last"]) for row in group]
        reentry_copy = [_as_float(row["reentry_delta_vs_copy_last"]) for row in group]
        corr_gain0 = [_as_float(row["corr_delta_vs_gain0"]) for row in group]
        mse_gain0 = [_as_float(row["mse_delta_vs_gain0"]) for row in group]
        reentry_gain0 = [_as_float(row["reentry_delta_vs_gain0"]) for row in group]

        mean_corr = _mean(corr_copy)
        mean_mse = _mean(mse_copy)
        mean_loop = _mean(loop_copy)
        mean_reentry = _mean(reentry_copy)
        mean_corr_gain0 = _mean(corr_gain0)
        strict = bool(
            mean_corr > 0.0
            and mean_mse <= 0.0
            and mean_loop <= 0.0
            and mean_reentry <= 0.0
            and mean_corr_gain0 >= 0.0
            and _win_fraction(corr_copy, positive=True) >= 0.5
            and _win_fraction(reentry_copy, positive=False) >= 0.5
        )
        diagnostic = bool(mean_corr > 0.0 and mean_mse <= 0.0 and mean_reentry <= 0.0)
        guard_score = (
            mean_corr
            - max(mean_reentry, 0.0)
            - 0.25 * max(mean_loop, 0.0)
            - 0.25 * max(mean_mse, 0.0)
            + 0.25 * min(-mean_mse, 0.05)
        )
        out.append(
            {
                "view": view,
                "magnitude_mode": mode,
                "mask_mode": mask,
                "mechanism": mechanism,
                "gain": gain,
                "case_count": len(group),
                "mean_corr_delta_vs_copy_last": mean_corr,
                "median_corr_delta_vs_copy_last": _median(corr_copy),
                "corr_win_fraction_vs_copy_last": _win_fraction(corr_copy, positive=True),
                "mean_mse_delta_vs_copy_last": mean_mse,
                "mse_win_fraction_vs_copy_last": _win_fraction(mse_copy, positive=False),
                "mean_loop_delta_vs_copy_last": mean_loop,
                "loop_win_fraction_vs_copy_last": _win_fraction(loop_copy, positive=False),
                "mean_reentry_delta_vs_copy_last": mean_reentry,
                "reentry_win_fraction_vs_copy_last": _win_fraction(reentry_copy, positive=False),
                "mean_corr_delta_vs_gain0": mean_corr_gain0,
                "median_corr_delta_vs_gain0": _median(corr_gain0),
                "mean_mse_delta_vs_gain0": _mean(mse_gain0),
                "mean_reentry_delta_vs_gain0": _mean(reentry_gain0),
                "strict_reentry_guard_candidate": strict,
                "diagnostic_reentry_candidate": diagnostic,
                "guard_score": guard_score,
            }
        )
    return sorted(out, key=lambda row: _as_float(row["guard_score"]), reverse=True)


def _key(row: dict[str, Any]) -> tuple[str, str, str, float]:
    return (
        str(row["magnitude_mode"]),
        str(row["mask_mode"]),
        str(row["mechanism"]),
        _as_float(row["gain"]),
    )


def _robust_candidates(view_rows: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    strict_by_view = {
        view: {_key(row): row for row in rows if row.get("strict_reentry_guard_candidate")}
        for view, rows in view_rows.items()
    }
    if not strict_by_view:
        return []
    shared = set.intersection(*(set(items) for items in strict_by_view.values()))
    rows: list[dict[str, Any]] = []
    for key in shared:
        all_row = strict_by_view.get("all", {}).get(key) or next(iter(strict_by_view.values()))[key]
        rows.append(
            {
                "magnitude_mode": key[0],
                "mask_mode": key[1],
                "mechanism": key[2],
                "gain": key[3],
                "all_view_guard_score": all_row.get("guard_score", 0.0),
                "views": {view: strict_by_view[view][key] for view in strict_by_view if key in strict_by_view[view]},
            }
        )
    return sorted(rows, key=lambda row: _as_float(row["all_view_guard_score"]), reverse=True)


def _markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# Phase-Native Audio Reentry Guard Score",
        "",
        "This scorer joins the full phase-native audio reset suite back to copy-last",
        "baselines and asks whether any nonzero phase-delta row preserves correlation,",
        "MSE, loop, and first-chunk reentry simultaneously.",
        "",
        "## Summary",
        "",
        f"- Status: `{summary['status']}`",
        f"- Suite JSON: `{summary['suite_json']}`",
        f"- Future access clean: `{summary['future_access_clean']}`",
        f"- Raw delta rows scored: `{summary['raw_delta_rows']}`",
        f"- Robust strict candidates: `{len(summary['robust_strict_candidates'])}`",
        "",
        "## Candidate Counts",
        "",
        "| View | Rows | Strict | Diagnostic | Top guard score |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for view, item in summary["views"].items():
        top = item["top_rows"][0]["guard_score"] if item["top_rows"] else 0.0
        lines.append(
            f"| `{view}` | {item['row_count']} | {item['strict_candidate_count']} | {item['diagnostic_candidate_count']} | {top:.9f} |"
        )
    lines.extend(["", "## Top Rows By View", ""])
    for view, item in summary["views"].items():
        lines.extend(
            [
                f"### {view}",
                "",
                "| mode | mask | mechanism | gain | guard | corr-copy | mse-copy | loop-copy | reentry-copy | corr-gain0 | strict |",
                "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
            ]
        )
        for row in item["top_rows"][:12]:
            lines.append(
                "| `{mode}` | `{mask}` | `{mech}` | {gain:.3f} | {guard:.9f} | {corr:.9f} | {mse:.9f} | {loop:.9f} | {reentry:.9f} | {cgain:.9f} | `{strict}` |".format(
                    mode=row["magnitude_mode"],
                    mask=row["mask_mode"],
                    mech=row["mechanism"],
                    gain=_as_float(row["gain"]),
                    guard=_as_float(row["guard_score"]),
                    corr=_as_float(row["mean_corr_delta_vs_copy_last"]),
                    mse=_as_float(row["mean_mse_delta_vs_copy_last"]),
                    loop=_as_float(row["mean_loop_delta_vs_copy_last"]),
                    reentry=_as_float(row["mean_reentry_delta_vs_copy_last"]),
                    cgain=_as_float(row["mean_corr_delta_vs_gain0"]),
                    strict=bool(row["strict_reentry_guard_candidate"]),
                )
            )
        lines.append("")
    lines.extend(
        [
            "## Interpretation",
            "",
            summary["interpretation"],
        ]
    )
    return "\n".join(lines) + "\n"


def score_suite(suite_json: Path, out_dir: Path, delta_json: Path | None = None) -> dict[str, Any]:
    prefix, flat, delta = _load_required_artifacts(suite_json, delta_json_override=delta_json)
    copy_last = _copy_last_baseline(prefix)
    delta_rows = _collect_delta_rows(delta, copy_last)
    view_rows = {view: _aggregate(delta_rows, view, excluded) for view, excluded in DEFAULT_VIEWS.items()}
    robust = _robust_candidates(view_rows)
    future_clean = _future_access_clean([prefix, flat, delta])
    if robust and future_clean:
        status = "strict_reentry_guard_candidate_found"
        interpretation = (
            "At least one nonzero phase-delta row satisfies the strict reentry guard in all stratified views. "
            "This is a candidate for rerun with a narrower predeclared suite, not a checkpoint promotion by itself."
        )
    elif any(row.get("diagnostic_reentry_candidate") for rows in view_rows.values() for row in rows):
        status = "diagnostic_reentry_candidate_only"
        interpretation = (
            "Some rows improve correlation/MSE while not worsening reentry in at least one view, but no row is strict "
            "across all/no-razor/no-buzzy views. Treat this as search-direction evidence only."
        )
    else:
        status = "no_reentry_guard_candidate"
        interpretation = (
            "The current phase-delta grid does not contain a nonzero row that preserves correlation, MSE, loop, "
            "and first-chunk reentry against copy-last under the reentry guard."
        )
    summary = {
        "schema": "phase_native_audio_reentry_guard_score_v1",
        "status": status,
        "suite_json": str(suite_json),
        "delta_json_override": str(delta_json) if delta_json is not None else None,
        "future_access_clean": future_clean,
        "raw_delta_rows": len(delta_rows),
        "copy_last_case_count": len(copy_last),
        "robust_strict_candidates": robust[:20],
        "views": {
            view: {
                "excluded_groups": sorted(DEFAULT_VIEWS[view]),
                "row_count": len(rows),
                "strict_candidate_count": sum(1 for row in rows if row.get("strict_reentry_guard_candidate")),
                "diagnostic_candidate_count": sum(1 for row in rows if row.get("diagnostic_reentry_candidate")),
                "top_rows": rows[:25],
            }
            for view, rows in view_rows.items()
        },
        "interpretation": interpretation,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / OUTPUT_JSON).write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (out_dir / OUTPUT_MD).write_text(_markdown(summary), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Score phase-native audio rows against a copy-last reentry guard.")
    parser.add_argument("--suite-json", required=True, type=Path)
    parser.add_argument("--delta-json", default=None, type=Path, help="Optional alternate audio_delta_mechanism_probe.json.")
    parser.add_argument("--out-dir", required=True, type=Path)
    args = parser.parse_args()
    summary = score_suite(args.suite_json, args.out_dir, delta_json=args.delta_json)
    print(
        json.dumps(
            {
                "status": summary["status"],
                "json": str(Path(args.out_dir) / OUTPUT_JSON),
                "markdown": str(Path(args.out_dir) / OUTPUT_MD),
                "robust_strict_candidates": len(summary["robust_strict_candidates"]),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
