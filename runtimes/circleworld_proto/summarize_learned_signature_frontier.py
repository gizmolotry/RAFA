from __future__ import annotations

import argparse
import glob
import json
import math
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SCHEMA = "rafa_learned_signature_frontier_v0"
TOKENBURST_ROOT = Path(r"C:\Users\Andrew\AppData\Local\Temp\rafa_tokenburst")
DEFAULT_INPUT_GLOB = str(TOKENBURST_ROOT / "learned_signature_scout*" / "learned_signature_scout.json")
DEFAULT_OUT_DIR = TOKENBURST_ROOT / "learned_signature_frontier_2026_05_07"

STABILITY_EPSILON = 1.0e-4
SEPARATION_EPSILON = 1.0e-4
MAX_DOMINANT_CLUSTER_DELTA = 0.02
MAX_NEAR_DUPLICATE_DELTA = 0.02
MIN_EFFECTIVE_RANK_DELTA = -0.05
MIN_PAIRWISE_DISTANCE_DELTA = -0.02


def _utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return float(default)
        result = float(value)
    except (TypeError, ValueError):
        return float(default)
    if math.isnan(result) or math.isinf(result):
        return float(default)
    return result


def _as_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _recover_seed(profile: str, payload: dict[str, Any], train: dict[str, Any]) -> int | None:
    for value in (payload.get("seed"), train.get("seed")):
        seed = _as_int(value)
        if seed is not None:
            return seed
    match = re.search(r"(?:^|_)seed(\d+)(?:_|$)", profile)
    return int(match.group(1)) if match else None


def _recover_hidden_dim(profile: str, payload: dict[str, Any], train: dict[str, Any]) -> int | None:
    for value in (train.get("hidden_dim"), payload.get("hidden_dim")):
        hidden_dim = _as_int(value)
        if hidden_dim is not None:
            return hidden_dim
    match = re.search(r"(?:^|_)h(\d+)(?:_|$)", profile)
    return int(match.group(1)) if match else None


def _axis_margins(
    stability_delta: float,
    separation_delta: float,
    collapse_dominant_delta: float,
    collapse_near_duplicate_delta: float,
    collapse_effective_rank_delta: float,
    collapse_mean_pairwise_distance_delta: float,
) -> dict[str, float]:
    anti_margins = {
        "collapse_dominant_cluster_margin": MAX_DOMINANT_CLUSTER_DELTA - collapse_dominant_delta,
        "collapse_near_duplicate_margin": MAX_NEAR_DUPLICATE_DELTA - collapse_near_duplicate_delta,
        "collapse_effective_rank_margin": collapse_effective_rank_delta - MIN_EFFECTIVE_RANK_DELTA,
        "collapse_mean_pairwise_distance_margin": collapse_mean_pairwise_distance_delta - MIN_PAIRWISE_DISTANCE_DELTA,
    }
    return {
        "stability_margin": stability_delta - STABILITY_EPSILON,
        "separation_margin": separation_delta - SEPARATION_EPSILON,
        "anti_collapse_margin": min(anti_margins.values()),
        **anti_margins,
    }


def _criteria_from_metrics(margins: dict[str, float]) -> dict[str, bool]:
    stability = margins["stability_margin"] > 0.0
    separation = margins["separation_margin"] > 0.0
    anti_collapse = margins["anti_collapse_margin"] >= 0.0
    return {
        "learned_beats_metadata_stability": stability,
        "learned_beats_metadata_separation": separation,
        "learned_no_worse_collapse": anti_collapse,
        "all_axes": stability and separation and anti_collapse,
    }


def _bool_criterion(criteria: dict[str, Any], key: str, fallback: bool) -> bool:
    value = criteria.get(key)
    return bool(value) if isinstance(value, bool) else bool(fallback)


def _blocking_margin(margins: dict[str, float], all_axes: bool) -> float:
    axis_margins = [
        margins["stability_margin"],
        margins["separation_margin"],
        margins["anti_collapse_margin"],
    ]
    if all_axes:
        return min(axis_margins)
    failed = [margin for margin in axis_margins if margin < 0.0]
    return max(failed) if failed else min(axis_margins)


def _blocking_axis(margins: dict[str, float], all_axes: bool) -> str | None:
    axis_margins = {
        "stability": margins["stability_margin"],
        "separation": margins["separation_margin"],
        "anti_collapse": margins["anti_collapse_margin"],
    }
    if all_axes:
        return None
    failed = {key: value for key, value in axis_margins.items() if value < 0.0}
    if not failed:
        return None
    return max(failed.items(), key=lambda item: item[1])[0]


def _row(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    profile = path.parent.name
    train = _dict(payload.get("train_summary"))
    aggregate = _dict(payload.get("aggregate"))
    stability = _dict(aggregate.get("stability"))
    separation = _dict(aggregate.get("separation"))
    collapse = _dict(aggregate.get("collapse"))
    collapse_delta = _dict(collapse.get("delta_learned_minus_metadata"))
    verdict = _dict(payload.get("verdict"))
    verdict_criteria = _dict(verdict.get("criteria"))

    stability_delta = _as_float(stability.get("delta_learned_minus_metadata"))
    separation_delta = _as_float(separation.get("delta_learned_minus_metadata"))
    collapse_dominant_delta = _as_float(collapse_delta.get("dominant_cluster_share"))
    collapse_near_duplicate_delta = _as_float(collapse_delta.get("near_duplicate_pair_share"))
    collapse_effective_rank_delta = _as_float(collapse_delta.get("effective_rank"))
    collapse_mean_pairwise_distance_delta = _as_float(collapse_delta.get("mean_pairwise_distance"))

    margins = _axis_margins(
        stability_delta,
        separation_delta,
        collapse_dominant_delta,
        collapse_near_duplicate_delta,
        collapse_effective_rank_delta,
        collapse_mean_pairwise_distance_delta,
    )
    metric_criteria = _criteria_from_metrics(margins)
    stability_axis = _bool_criterion(
        verdict_criteria,
        "learned_beats_metadata_stability",
        metric_criteria["learned_beats_metadata_stability"],
    )
    separation_axis = _bool_criterion(
        verdict_criteria,
        "learned_beats_metadata_separation",
        metric_criteria["learned_beats_metadata_separation"],
    )
    anti_collapse_axis = _bool_criterion(
        verdict_criteria,
        "learned_no_worse_collapse",
        metric_criteria["learned_no_worse_collapse"],
    )
    all_axes = _bool_criterion(
        verdict_criteria,
        "all_axes",
        stability_axis and separation_axis and anti_collapse_axis,
    )
    split_mode = str(payload.get("split_mode") or verdict.get("split_mode") or "unknown")
    is_heldout = split_mode.startswith("heldout")
    axis_coverage_count = int(stability_axis) + int(separation_axis) + int(anti_collapse_axis)
    heldout_candidate = bool(is_heldout and all_axes)
    heldout_near_candidate = bool(is_heldout and not all_axes and axis_coverage_count >= 2)
    closest_blocking_margin = _blocking_margin(margins, all_axes)

    return {
        "profile": profile,
        "path": str(path),
        "split_mode": split_mode,
        "seed": _recover_seed(profile, payload, train),
        "case_offset": _as_int(payload.get("case_offset")),
        "case_shuffle_seed": _as_int(payload.get("case_shuffle_seed")),
        "hidden_dim": _recover_hidden_dim(profile, payload, train),
        "verdict_status": verdict.get("status", "unknown"),
        "verdict_reasons": [str(reason) for reason in _list(verdict.get("reasons"))],
        "all_axis_candidate": all_axes,
        "heldout_candidate": heldout_candidate,
        "heldout_near_candidate": heldout_near_candidate,
        "axis_coverage_count": axis_coverage_count,
        "axis_coverage": {
            "stability": stability_axis,
            "separation": separation_axis,
            "anti_collapse": anti_collapse_axis,
        },
        "closest_blocking_margin": closest_blocking_margin,
        "closest_blocking_axis": _blocking_axis(margins, all_axes),
        "stability_delta": stability_delta,
        "separation_delta": separation_delta,
        "collapse_deltas": {
            "dominant_cluster_share": collapse_dominant_delta,
            "near_duplicate_pair_share": collapse_near_duplicate_delta,
            "effective_rank": collapse_effective_rank_delta,
            "mean_pairwise_distance": collapse_mean_pairwise_distance_delta,
            "effective_cluster_count": _as_float(collapse_delta.get("effective_cluster_count")),
        },
        "margins": margins,
        "num_cases": payload.get("num_cases"),
        "train_case_count": payload.get("train_case_count"),
        "heldout_case_count": payload.get("heldout_case_count"),
        "train_packet_count": train.get("train_packet_count"),
        "created_utc": payload.get("created_utc"),
    }


def _rank_key(row: dict[str, Any]) -> tuple[Any, ...]:
    # Ordered to surface the frontier requested by the scout workflow: all-axis
    # passes first, then held-out evidence, then held-out near misses, then the
    # broadest axis coverage and the smallest remaining blocker.
    return (
        int(bool(row.get("all_axis_candidate"))),
        int(bool(row.get("heldout_candidate"))),
        int(bool(row.get("heldout_near_candidate"))),
        int(row.get("axis_coverage_count") or 0),
        _as_float(row.get("closest_blocking_margin")),
        _as_float(row.get("stability_delta")),
        _as_float(row.get("separation_delta")),
        str(row.get("profile") or ""),
    )


def _rank_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    ranked = sorted(rows, key=_rank_key, reverse=True)
    for index, row in enumerate(ranked, start=1):
        row["frontier_rank"] = index
    return ranked


def _compact_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "frontier_rank": row.get("frontier_rank"),
        "profile": row.get("profile"),
        "split_mode": row.get("split_mode"),
        "seed": row.get("seed"),
        "case_offset": row.get("case_offset"),
        "case_shuffle_seed": row.get("case_shuffle_seed"),
        "hidden_dim": row.get("hidden_dim"),
        "all_axis_candidate": row.get("all_axis_candidate"),
        "heldout_candidate": row.get("heldout_candidate"),
        "heldout_near_candidate": row.get("heldout_near_candidate"),
        "axis_coverage_count": row.get("axis_coverage_count"),
        "axis_coverage": row.get("axis_coverage"),
        "closest_blocking_margin": row.get("closest_blocking_margin"),
        "closest_blocking_axis": row.get("closest_blocking_axis"),
        "stability_delta": row.get("stability_delta"),
        "separation_delta": row.get("separation_delta"),
        "collapse_deltas": row.get("collapse_deltas"),
        "verdict_status": row.get("verdict_status"),
        "verdict_reasons": row.get("verdict_reasons"),
        "path": row.get("path"),
    }


def _frontier_status(rows: list[dict[str, Any]]) -> str:
    if any(row.get("heldout_candidate") for row in rows):
        return "heldout_candidate_found"
    if any(row.get("all_axis_candidate") for row in rows):
        return "in_sample_candidate_found_holdout_open"
    if any(row.get("heldout_near_candidate") for row in rows):
        return "heldout_near_candidate_frontier"
    if rows:
        return "no_all_axis_candidate"
    return "no_scout_outputs_found"


def summarize_frontier(input_glob: str, out_dir: Path) -> dict[str, Any]:
    paths = sorted(
        Path(path)
        for path in realsorted_glob(input_glob)
        if "smoke" not in Path(path).parent.name.lower()
    )
    rows: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    for path in paths:
        try:
            rows.append(_row(path))
        except (OSError, json.JSONDecodeError, TypeError, KeyError) as exc:
            errors.append({"path": str(path), "error": f"{type(exc).__name__}: {exc}"})

    ranked_rows = _rank_rows(rows)
    compact_rows = [_compact_row(row) for row in ranked_rows]
    heldout_rows = [row for row in ranked_rows if str(row.get("split_mode", "")).startswith("heldout")]
    summary = {
        "schema": SCHEMA,
        "created_utc": _utc_timestamp(),
        "input_glob": input_glob,
        "status": _frontier_status(ranked_rows),
        "scout_output_count": len(paths),
        "parsed_row_count": len(ranked_rows),
        "parse_error_count": len(errors),
        "all_axis_candidate_count": sum(1 for row in ranked_rows if row.get("all_axis_candidate")),
        "heldout_candidate_count": sum(1 for row in ranked_rows if row.get("heldout_candidate")),
        "heldout_near_candidate_count": sum(1 for row in ranked_rows if row.get("heldout_near_candidate")),
        "heldout_row_count": len(heldout_rows),
        "best_profile": ranked_rows[0].get("profile") if ranked_rows else None,
        "best_heldout_profile": heldout_rows[0].get("profile") if heldout_rows else None,
        "rank_definition": [
            "all_axis_candidate descending",
            "heldout_candidate descending",
            "heldout_near_candidate descending",
            "axis_coverage_count descending",
            "closest_blocking_margin descending",
            "stability_delta descending",
            "separation_delta descending",
            "profile descending for deterministic ties",
        ],
        "thresholds": {
            "stability_delta_gt": STABILITY_EPSILON,
            "separation_delta_gt": SEPARATION_EPSILON,
            "collapse_dominant_cluster_delta_lte": MAX_DOMINANT_CLUSTER_DELTA,
            "collapse_near_duplicate_delta_lte": MAX_NEAR_DUPLICATE_DELTA,
            "collapse_effective_rank_delta_gte": MIN_EFFECTIVE_RANK_DELTA,
            "collapse_mean_pairwise_distance_delta_gte": MIN_PAIRWISE_DISTANCE_DELTA,
        },
        "frontier_rows": compact_rows,
        "parse_errors": errors,
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "learned_signature_frontier.json"
    md_path = out_dir / "LEARNED_SIGNATURE_FRONTIER.md"
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    _write_markdown(md_path, summary)
    return summary


def realsorted_glob(pattern: str) -> list[str]:
    return sorted(glob.glob(pattern), key=lambda item: item.lower())


def _fmt(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        return f"{value:+.6f}"
    return str(value)


def _axis_flags(row: dict[str, Any]) -> str:
    axes = row.get("axis_coverage") if isinstance(row.get("axis_coverage"), dict) else {}
    labels = []
    if axes.get("stability"):
        labels.append("S")
    if axes.get("separation"):
        labels.append("P")
    if axes.get("anti_collapse"):
        labels.append("A")
    return "".join(labels) or "-"


def _reason_summary(row: dict[str, Any]) -> str:
    reasons = row.get("verdict_reasons") if isinstance(row.get("verdict_reasons"), list) else []
    if not reasons:
        return ""
    return "; ".join(str(reason) for reason in reasons)


def _write_markdown(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# Learned Signature Frontier",
        "",
        f"- schema: `{summary['schema']}`",
        f"- status: `{summary['status']}`",
        f"- input glob: `{summary['input_glob']}`",
        f"- scout outputs: `{summary['scout_output_count']}`",
        f"- parsed rows: `{summary['parsed_row_count']}`",
        f"- parse errors: `{summary['parse_error_count']}`",
        f"- all-axis candidates: `{summary['all_axis_candidate_count']}`",
        f"- heldout candidates: `{summary['heldout_candidate_count']}`",
        f"- heldout near candidates: `{summary['heldout_near_candidate_count']}`",
        f"- best profile: `{summary.get('best_profile')}`",
        f"- best heldout profile: `{summary.get('best_heldout_profile')}`",
        "",
        "Rank order: all-axis candidate, heldout candidate, heldout near candidate, axis coverage, closest blocking margin.",
        "",
        "| rank | profile | split | seed | offset | shuffle | h | all | heldout | near | axes | blocker | block margin | stability d | separation d | dom d | dup d | rank d | dist d | verdict |",
        "|---:|---|---|---:|---:|---:|---:|---|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in summary.get("frontier_rows", []):
        collapse = row.get("collapse_deltas") if isinstance(row.get("collapse_deltas"), dict) else {}
        lines.append(
            "| {rank} | {profile} | {split} | {seed} | {offset} | {shuffle} | {hidden} | {all_axes} | {heldout} | {near} | {axes} | {blocker} | {margin} | {stable} | {sep} | {dom} | {dup} | {rank_d} | {dist} | {verdict} |".format(
                rank=row.get("frontier_rank"),
                profile=row.get("profile"),
                split=row.get("split_mode"),
                seed=_fmt(row.get("seed")),
                offset=_fmt(row.get("case_offset")),
                shuffle=_fmt(row.get("case_shuffle_seed")),
                hidden=_fmt(row.get("hidden_dim")),
                all_axes=_fmt(row.get("all_axis_candidate")),
                heldout=_fmt(row.get("heldout_candidate")),
                near=_fmt(row.get("heldout_near_candidate")),
                axes=f"{row.get('axis_coverage_count')}:{_axis_flags(row)}",
                blocker=_fmt(row.get("closest_blocking_axis")),
                margin=_fmt(row.get("closest_blocking_margin")),
                stable=_fmt(row.get("stability_delta")),
                sep=_fmt(row.get("separation_delta")),
                dom=_fmt(collapse.get("dominant_cluster_share")),
                dup=_fmt(collapse.get("near_duplicate_pair_share")),
                rank_d=_fmt(collapse.get("effective_rank")),
                dist=_fmt(collapse.get("mean_pairwise_distance")),
                verdict=row.get("verdict_status"),
            )
        )
    if summary.get("parse_errors"):
        lines.extend(["", "## Parse Errors", ""])
        for error in summary["parse_errors"]:
            lines.append(f"- `{error['path']}`: {error['error']}")
    lines.extend(["", "## Verdict Reasons", ""])
    for row in summary.get("frontier_rows", []):
        reasons = _reason_summary(row)
        if reasons:
            lines.append(f"- `{row.get('frontier_rank')}` `{row.get('profile')}`: {reasons}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Summarize learned_signature_scout.json outputs into a compact frontier report."
    )
    parser.add_argument("--input-glob", default=DEFAULT_INPUT_GLOB, help="Glob for learned_signature_scout.json files.")
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR), help="Directory for JSON and Markdown reports.")
    args = parser.parse_args()

    summary = summarize_frontier(args.input_glob, Path(args.out_dir))
    print(
        json.dumps(
            {
                "schema": summary["schema"],
                "saved": str(Path(args.out_dir) / "learned_signature_frontier.json"),
                "report": str(Path(args.out_dir) / "LEARNED_SIGNATURE_FRONTIER.md"),
                "status": summary["status"],
                "scout_output_count": summary["scout_output_count"],
                "parsed_row_count": summary["parsed_row_count"],
                "all_axis_candidate_count": summary["all_axis_candidate_count"],
                "heldout_candidate_count": summary["heldout_candidate_count"],
                "heldout_near_candidate_count": summary["heldout_near_candidate_count"],
                "best_profile": summary["best_profile"],
                "best_heldout_profile": summary["best_heldout_profile"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
