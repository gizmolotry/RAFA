from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


OUTPUT_JSON = "internal_phase_law_candidate_classes.json"
OUTPUT_MD = "INTERNAL_PHASE_LAW_CANDIDATE_CLASSES.md"


def _load(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"Expected JSON object: {path}")
    return payload


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


def _is_bad_dominated(row: dict[str, Any]) -> bool:
    return bool(row.get("absolute_bad_baseline_dominated"))


def _strict_audio_candidate(row: dict[str, Any]) -> bool:
    return (
        _as_float(row.get("direct_mean_corr_delta")) >= 0.001
        and _as_float(row.get("direct_median_corr_delta")) > 0.0
        and _as_float(row.get("direct_corr_win_fraction")) >= 0.55
        and _as_float(row.get("absolute_mean_corr_delta")) > 0.0
        and _as_float(row.get("absolute_median_corr_delta")) > 0.0
        and _as_float(row.get("absolute_corr_win_fraction")) >= 0.55
        and _as_float(row.get("absolute_nonbad_bin_mean_corr_delta")) > 0.0
        and _as_float(row.get("absolute_good_bin_mean_corr_delta")) >= -0.01
        and not _is_bad_dominated(row)
    )


def _phase_support_candidate(row: dict[str, Any]) -> bool:
    return (
        _as_float(row.get("direct_mean_corr_delta")) >= 0.00075
        and _as_float(row.get("direct_median_corr_delta")) > 0.0
        and _as_float(row.get("direct_corr_win_fraction")) >= 0.55
        and _as_float(row.get("absolute_mean_corr_delta")) >= 0.0025
        and _as_float(row.get("absolute_median_corr_delta")) > 0.0
        and _as_float(row.get("absolute_corr_win_fraction")) >= 0.55
        and _as_float(row.get("absolute_nonbad_bin_mean_corr_delta")) >= 0.00075
        and _as_float(row.get("absolute_good_bin_mean_corr_delta")) >= 0.0
        and not _is_bad_dominated(row)
    )


def _classify(row: dict[str, Any]) -> str:
    if _strict_audio_candidate(row):
        return "strict_audio_candidate"
    if _phase_support_candidate(row):
        return "phase_support_candidate_not_promotion"
    if _as_float(row.get("direct_mean_corr_delta")) < 0.00075:
        return "hold_direct_size"
    if _as_float(row.get("absolute_nonbad_bin_mean_corr_delta")) < 0.00075:
        return "hold_phase_support_nonbad"
    if _as_float(row.get("absolute_corr_win_fraction")) < 0.55:
        return "hold_absolute_win_fraction"
    if _as_float(row.get("direct_corr_win_fraction")) < 0.55:
        return "hold_direct_win_fraction"
    if _is_bad_dominated(row):
        return "hold_bad_baseline_dominated"
    return "hold_other"


def _score(row: dict[str, Any]) -> float:
    return float(
        2.0 * _as_float(row.get("direct_mean_corr_delta"))
        + _as_float(row.get("direct_median_corr_delta"))
        + 3.0 * _as_float(row.get("absolute_mean_corr_delta"))
        + _as_float(row.get("absolute_median_corr_delta"))
        + 4.0 * _as_float(row.get("absolute_nonbad_bin_mean_corr_delta"))
        + 0.004 * (_as_float(row.get("absolute_corr_win_fraction")) - 0.5)
        + 0.002 * (_as_float(row.get("direct_corr_win_fraction")) - 0.5)
    )


def classify(joint_rows: Path, out_dir: Path, limit: int = 24) -> dict[str, Any]:
    payload = _load(joint_rows)
    rows: list[dict[str, Any]] = []
    counts: dict[str, int] = {}
    for raw in payload.get("rows", []) or []:
        if not isinstance(raw, dict):
            continue
        row = dict(raw)
        row["candidate_class"] = _classify(row)
        row["class_score"] = _score(row)
        counts[row["candidate_class"]] = counts.get(row["candidate_class"], 0) + 1
        rows.append(row)
    rows.sort(
        key=lambda row: (
            row.get("candidate_class") == "strict_audio_candidate",
            row.get("candidate_class") == "phase_support_candidate_not_promotion",
            _as_float(row.get("class_score")),
        ),
        reverse=True,
    )
    strict = [row for row in rows if row.get("candidate_class") == "strict_audio_candidate"]
    phase_support = [
        row for row in rows if row.get("candidate_class") == "phase_support_candidate_not_promotion"
    ]
    status = (
        "strict_audio_candidate_found"
        if strict
        else "phase_support_candidate_found_not_promotion"
        if phase_support
        else "no_candidate_class_found"
    )
    summary = {
        "schema": "circleworld_internal_phase_law_candidate_classes_v0",
        "status": status,
        "joint_rows": str(joint_rows),
        "row_count": len(rows),
        "class_counts": counts,
        "strict_audio_candidate_count": len(strict),
        "phase_support_candidate_count": len(phase_support),
        "best_row": rows[0] if rows else None,
        "rows": rows[:limit],
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / OUTPUT_JSON).write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (out_dir / OUTPUT_MD).write_text(_markdown(summary), encoding="utf-8")
    return summary


def _markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# Internal Phase-Law Candidate Classes",
        "",
        "This is a diagnostic classifier only. It does not change checkpoint promotion rules.",
        "",
        f"- status: `{summary.get('status')}`",
        f"- rows: `{summary.get('row_count')}`",
        f"- strict audio candidates: `{summary.get('strict_audio_candidate_count')}`",
        f"- phase-support candidates: `{summary.get('phase_support_candidate_count')}`",
        "",
        "## Class Counts",
        "",
    ]
    for key, value in sorted((summary.get("class_counts") or {}).items()):
        lines.append(f"- `{key}`: `{value}`")
    lines.extend(
        [
            "",
            "## Top Rows",
            "",
            "| class | run | mask | gain | class score | direct mean | direct wins | abs mean | abs wins | nonbad | good |",
            "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in summary.get("rows", []):
        lines.append(
            "| {klass} | {run} | {mask} | {gain} | {score} | {direct} | {dwins} | {absmean} | {awins} | {nonbad} | {good} |".format(
                klass=row.get("candidate_class"),
                run=row.get("run_label"),
                mask=row.get("mask_mode"),
                gain=_fmt(row.get("gain")),
                score=_fmt(row.get("class_score")),
                direct=_fmt(row.get("direct_mean_corr_delta")),
                dwins=_fmt(row.get("direct_corr_win_fraction")),
                absmean=_fmt(row.get("absolute_mean_corr_delta")),
                awins=_fmt(row.get("absolute_corr_win_fraction")),
                nonbad=_fmt(row.get("absolute_nonbad_bin_mean_corr_delta")),
                good=_fmt(row.get("absolute_good_bin_mean_corr_delta")),
            )
        )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Classify internal phase-law joint rows into predeclared candidate classes.")
    parser.add_argument("--joint-rows", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--limit", type=int, default=24)
    args = parser.parse_args()
    summary = classify(joint_rows=Path(args.joint_rows), out_dir=Path(args.out_dir), limit=int(args.limit))
    print(
        json.dumps(
            {
                "json": str(Path(args.out_dir) / OUTPUT_JSON),
                "markdown": str(Path(args.out_dir) / OUTPUT_MD),
                "status": summary["status"],
                "strict_audio_candidate_count": summary["strict_audio_candidate_count"],
                "phase_support_candidate_count": summary["phase_support_candidate_count"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
