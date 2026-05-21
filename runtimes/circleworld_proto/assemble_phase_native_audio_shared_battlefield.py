from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


OUTPUT_JSON = "phase_native_audio_shared_battlefield.json"
OUTPUT_MD = "PHASE_NATIVE_AUDIO_SHARED_BATTLEFIELD.md"


def _json_load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _num(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _raw_row(path: Path) -> dict[str, Any]:
    payload = _json_load(path)
    score = payload.get("scorecard", {})
    task_counts = [
        row.get("case_count")
        for row in score.get("task_scores", [])
        if isinstance(row, dict) and row.get("case_count") is not None
    ]
    case_count = task_counts[0] if task_counts else payload.get("num_cases")
    return {
        "label": "raw_circleworld",
        "path": str(path),
        "schema": payload.get("schema"),
        "status": payload.get("status"),
        "case_count": case_count,
        "future_access_clean": bool(score.get("future_access_clean")),
        "promotion_candidate": bool(score.get("promotion_candidate")),
        "mean_corr_delta_vs_copy_last": score.get("mean_circleworld_corr_delta_vs_copy_last"),
        "mean_mse_delta_vs_copy_last": score.get("mean_circleworld_mse_delta_vs_copy_last"),
        "mean_loop_delta_vs_copy_last": score.get("mean_circleworld_loop_peak_delta_vs_copy_last"),
        "mean_reentry_delta_vs_copy_last": score.get("mean_circleworld_reentry_delta_vs_copy_last"),
        "mean_harmful_replay_excess_delta_vs_copy_last": None,
        "strict_target_replay_pass": False,
    }


def _selected_row(path: Path) -> dict[str, Any]:
    payload = _json_load(path)
    score = payload.get("score", {}).get("aggregate", {})
    return {
        "label": "routed_circleworld_objective_knn5",
        "path": str(path),
        "schema": payload.get("schema"),
        "status": payload.get("status"),
        "case_count": payload.get("case_count"),
        "future_access_clean": bool(payload.get("future_access_clean")),
        "promotion_candidate": bool(score.get("strict_target_replay_pass")),
        "mean_corr_delta_vs_copy_last": score.get("mean_corr_delta_vs_copy_last"),
        "mean_mse_delta_vs_copy_last": score.get("mean_mse_delta_vs_copy_last"),
        "mean_loop_delta_vs_copy_last": score.get("mean_loop_delta_vs_copy_last"),
        "mean_reentry_delta_vs_copy_last": score.get("mean_reentry_delta_vs_copy_last"),
        "mean_harmful_replay_excess_delta_vs_copy_last": score.get(
            "mean_harmful_replay_excess_delta_vs_copy_last"
        ),
        "mean_corr_delta_vs_gain0": score.get("mean_corr_delta_vs_gain0"),
        "strict_target_replay_pass": bool(score.get("strict_target_replay_pass")),
    }


def _graduation_row(path: Path) -> dict[str, Any]:
    payload = _json_load(path)
    comp = payload.get("comparisons", {}).get("graduation_vs_copy_last", {})
    return {
        "label": "graduation_compat14_bridge",
        "path": str(path),
        "schema": payload.get("schema"),
        "status": payload.get("status"),
        "case_count": payload.get("case_count"),
        "future_access_clean": bool(payload.get("future_access_clean")),
        "promotion_candidate": payload.get("status") == "graduation_bridge_smoke_pass",
        "mean_corr_delta_vs_copy_last": comp.get("mean_corr_delta"),
        "mean_mse_delta_vs_copy_last": comp.get("mean_mse_delta"),
        "mean_loop_delta_vs_copy_last": comp.get("mean_loop_delta"),
        "mean_reentry_delta_vs_copy_last": comp.get("mean_reentry_delta"),
        "mean_harmful_replay_excess_delta_vs_copy_last": None,
        "strict_target_replay_pass": False,
        "adapter_caveat": "compat14_restore_forced_into_prefix_only_continuation_canvas",
    }


def _beats(a: dict[str, Any], b: dict[str, Any]) -> dict[str, bool]:
    return {
        "corr": (_num(a.get("mean_corr_delta_vs_copy_last")) or 0.0)
        > (_num(b.get("mean_corr_delta_vs_copy_last")) or 0.0),
        "mse": (_num(a.get("mean_mse_delta_vs_copy_last")) or 0.0)
        < (_num(b.get("mean_mse_delta_vs_copy_last")) or 0.0),
        "loop": (_num(a.get("mean_loop_delta_vs_copy_last")) or 0.0)
        < (_num(b.get("mean_loop_delta_vs_copy_last")) or 0.0),
        "reentry": (_num(a.get("mean_reentry_delta_vs_copy_last")) or 0.0)
        < (_num(b.get("mean_reentry_delta_vs_copy_last")) or 0.0),
    }


def _status(rows: dict[str, dict[str, Any]]) -> str:
    routed = rows["routed_circleworld_objective_knn5"]
    graduation = rows["graduation_compat14_bridge"]
    raw = rows["raw_circleworld"]
    routed_vs_grad = _beats(routed, graduation)
    routed_vs_raw = _beats(routed, raw)
    if bool(routed.get("strict_target_replay_pass")) and all(routed_vs_grad.values()) and all(routed_vs_raw.values()):
        return "routed_circleworld_beats_raw_and_current_adapter_not_graduation_baseline"
    if bool(routed.get("strict_target_replay_pass")) and all(routed_vs_raw.values()):
        return "routed_circleworld_component_beats_raw_only"
    return "shared_battlefield_mixed"


def _fmt(value: Any) -> str:
    parsed = _num(value)
    return "NA" if parsed is None else f"{parsed:+.9f}"


def _markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# Phase-Native Audio Shared Battlefield",
        "",
        f"- Status: `{summary['status']}`",
        f"- Case count: `{summary['case_count']}`",
        "",
        "| method | status | corr-copy | MSE-copy | loop-copy | reentry-copy | harm-delta |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summary["rows"]:
        lines.append(
            "| `{label}` | `{status}` | {corr} | {mse} | {loop} | {reentry} | {harm} |".format(
                label=row["label"],
                status=row["status"],
                corr=_fmt(row.get("mean_corr_delta_vs_copy_last")),
                mse=_fmt(row.get("mean_mse_delta_vs_copy_last")),
                loop=_fmt(row.get("mean_loop_delta_vs_copy_last")),
                reentry=_fmt(row.get("mean_reentry_delta_vs_copy_last")),
                harm=_fmt(row.get("mean_harmful_replay_excess_delta_vs_copy_last")),
            )
        )
    lines.extend(
        [
            "",
            "## Caveat",
            "",
            summary["caveat"],
            "",
            "## Interpretation",
            "",
            summary["interpretation"],
        ]
    )
    return "\n".join(lines) + "\n"


def assemble(*, raw_suite: Path, selected_route: Path, graduation_bridge: Path, out_dir: Path) -> dict[str, Any]:
    rows_by_label = {
        "raw_circleworld": _raw_row(raw_suite),
        "routed_circleworld_objective_knn5": _selected_row(selected_route),
        "graduation_compat14_bridge": _graduation_row(graduation_bridge),
    }
    labels = ["raw_circleworld", "routed_circleworld_objective_knn5", "graduation_compat14_bridge"]
    case_counts = {row.get("case_count") for row in rows_by_label.values()}
    summary = {
        "schema": "phase_native_audio_shared_battlefield_v1",
        "status": _status(rows_by_label),
        "case_count": next(iter(case_counts)) if len(case_counts) == 1 else None,
        "case_count_set": sorted(str(item) for item in case_counts),
        "rows": [rows_by_label[label] for label in labels],
        "pairwise": {
            "routed_vs_raw": _beats(rows_by_label["routed_circleworld_objective_knn5"], rows_by_label["raw_circleworld"]),
            "routed_vs_graduation_adapter": _beats(
                rows_by_label["routed_circleworld_objective_knn5"], rows_by_label["graduation_compat14_bridge"]
            ),
        },
        "caveat": (
            "Graduation here is the compat14 restore runtime forced into a prefix-only continuation canvas. "
            "This is stronger than having no shared battlefield at all, but it is not yet a certified "
            "native Graduation continuation baseline and must not be read as a Graduation RAFA failure."
        ),
        "interpretation": (
            "On this shared lockbox battlefield, routed Circleworld objective_knn5 beats raw Circleworld and "
            "the current experimental compat14 adapter on corr/MSE/loop/reentry deltas versus copy-last. "
            "The valid claim is component-level: the Circleworld route selector is useful under this harness. "
            "This does not evaluate the protected Graduation branch as a sound engine; the adapter needs "
            "its own calibration before any Graduation baseline comparison can be made."
        ),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / OUTPUT_JSON).write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (out_dir / OUTPUT_MD).write_text(_markdown(summary), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Assemble Graduation/raw/routed phase-native audio comparison.")
    parser.add_argument("--raw-suite", required=True, type=Path)
    parser.add_argument("--selected-route", required=True, type=Path)
    parser.add_argument("--graduation-bridge", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    args = parser.parse_args()
    summary = assemble(
        raw_suite=args.raw_suite,
        selected_route=args.selected_route,
        graduation_bridge=args.graduation_bridge,
        out_dir=args.out_dir,
    )
    print(
        json.dumps(
            {
                "status": summary["status"],
                "json": str(Path(args.out_dir) / OUTPUT_JSON),
                "markdown": str(Path(args.out_dir) / OUTPUT_MD),
                "pairwise": summary["pairwise"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
