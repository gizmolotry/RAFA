from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


TOKENBURST_ROOT = Path(r"C:\Users\Andrew\AppData\Local\Temp\rafa_tokenburst")
DEFAULT_OUT_DIR = TOKENBURST_ROOT / "internal_phase_law_predeclared_diagnostic_class_v10_v13_2026_05_07"


THRESHOLDS = {
    "direct_mean_corr_delta_min": 0.00075,
    "direct_median_corr_delta_min_exclusive": 0.0,
    "direct_corr_win_fraction_min": 0.55,
    "absolute_mean_corr_delta_min": 0.0025,
    "absolute_median_corr_delta_min_exclusive": 0.0,
    "absolute_corr_win_fraction_min": 0.55,
    "absolute_nonbad_bin_mean_corr_delta_min": 0.00075,
    "absolute_good_bin_mean_corr_delta_min": 0.0,
}

STRICT_THRESHOLDS = {
    **THRESHOLDS,
    "direct_mean_corr_delta_min": 0.001,
}


DEFAULT_JOINT_ROWS = {
    "v10_phase_router": TOKENBURST_ROOT
    / "internal_phase_law_objective_scout_v10_phase_router_masks_v1_2026_05_07"
    / "joint_diagnostics"
    / "internal_phase_law_joint_row_diagnostics.json",
    "v11_phase_router_direct": TOKENBURST_ROOT
    / "internal_phase_law_objective_scout_v11_phase_router_direct_v1_2026_05_07"
    / "joint_diagnostics"
    / "internal_phase_law_joint_row_diagnostics.json",
    "v12_phase_reentry_direct": TOKENBURST_ROOT
    / "internal_phase_law_objective_scout_v12_phase_reentry_direct_v1_2026_05_07"
    / "joint_diagnostics"
    / "internal_phase_law_joint_row_diagnostics.json",
    "v13_causal_reentry_direct": TOKENBURST_ROOT
    / "internal_phase_law_objective_scout_v13_causal_reentry_direct_v1_2026_05_07"
    / "joint_diagnostics"
    / "internal_phase_law_joint_row_diagnostics.json",
}


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
    if isinstance(value, (float, int)):
        return f"{float(value):.6g}"
    return str(value)


def _bad_dominated(row: dict[str, Any]) -> bool:
    return bool(row.get("absolute_bad_baseline_dominated"))


def _passes_thresholds(row: dict[str, Any], thresholds: dict[str, float]) -> bool:
    return (
        _as_float(row.get("direct_mean_corr_delta")) >= thresholds["direct_mean_corr_delta_min"]
        and _as_float(row.get("direct_median_corr_delta")) > thresholds["direct_median_corr_delta_min_exclusive"]
        and _as_float(row.get("direct_corr_win_fraction")) >= thresholds["direct_corr_win_fraction_min"]
        and _as_float(row.get("absolute_mean_corr_delta")) >= thresholds["absolute_mean_corr_delta_min"]
        and _as_float(row.get("absolute_median_corr_delta")) > thresholds["absolute_median_corr_delta_min_exclusive"]
        and _as_float(row.get("absolute_corr_win_fraction")) >= thresholds["absolute_corr_win_fraction_min"]
        and _as_float(row.get("absolute_nonbad_bin_mean_corr_delta"))
        >= thresholds["absolute_nonbad_bin_mean_corr_delta_min"]
        and _as_float(row.get("absolute_good_bin_mean_corr_delta"))
        >= thresholds["absolute_good_bin_mean_corr_delta_min"]
        and not _bad_dominated(row)
    )


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


def _summarize_track(label: str, path: Path) -> dict[str, Any]:
    payload = _load(path)
    rows = [row for row in payload.get("rows", []) or [] if isinstance(row, dict)]
    diagnostic_rows: list[dict[str, Any]] = []
    strict_rows: list[dict[str, Any]] = []
    for row in rows:
        enriched = dict(row)
        enriched["track"] = label
        enriched["diagnostic_class"] = "small_direct_strong_absolute_nonbad_not_promotion"
        enriched["diagnostic_score"] = _score(enriched)
        if _passes_thresholds(enriched, THRESHOLDS):
            diagnostic_rows.append(enriched)
        if _passes_thresholds(enriched, STRICT_THRESHOLDS):
            strict_rows.append(enriched)
    diagnostic_rows.sort(key=lambda row: _as_float(row.get("diagnostic_score")), reverse=True)
    strict_rows.sort(key=lambda row: _as_float(row.get("diagnostic_score")), reverse=True)
    best = diagnostic_rows[0] if diagnostic_rows else None
    return {
        "track": label,
        "joint_rows": str(path),
        "joint_status": payload.get("status"),
        "row_count": len(rows),
        "diagnostic_count": len(diagnostic_rows),
        "strict_audio_count": len(strict_rows),
        "best_diagnostic_row": best,
        "diagnostic_rows": diagnostic_rows,
        "strict_audio_rows": strict_rows,
    }


def build_report(joint_rows: dict[str, Path], out_dir: Path, limit: int) -> dict[str, Any]:
    tracks = [_summarize_track(label, path) for label, path in joint_rows.items()]
    all_diagnostic_rows = [
        row for track in tracks for row in track["diagnostic_rows"]
    ]
    all_strict_rows = [row for track in tracks for row in track["strict_audio_rows"]]
    all_diagnostic_rows.sort(key=lambda row: _as_float(row.get("diagnostic_score")), reverse=True)
    best_track = max(tracks, key=lambda track: (track["diagnostic_count"], track["strict_audio_count"]))
    payload = {
        "schema": "circleworld_internal_phase_law_predeclared_diagnostic_class_v0",
        "status": (
            "strict_audio_candidate_found"
            if all_strict_rows
            else "diagnostic_acceptance_found_not_promotion"
            if all_diagnostic_rows
            else "no_diagnostic_acceptance"
        ),
        "promotion_effect": "none",
        "class_name": "small_direct_strong_absolute_nonbad_not_promotion",
        "definition": {
            "intent": (
                "Predeclared diagnostic class for rows with small but nonzero direct recurrence, "
                "strong absolute carrier-beating support, positive nonbad bins, and no good-bin damage."
            ),
            "not_promotion": True,
            "thresholds": THRESHOLDS,
            "strict_audio_thresholds": STRICT_THRESHOLDS,
        },
        "track_count": len(tracks),
        "total_row_count": sum(track["row_count"] for track in tracks),
        "diagnostic_count": len(all_diagnostic_rows),
        "strict_audio_count": len(all_strict_rows),
        "best_track_by_count": best_track["track"],
        "tracks": [
            {
                key: value
                for key, value in track.items()
                if key not in {"diagnostic_rows", "strict_audio_rows"}
            }
            for track in tracks
        ],
        "top_rows": all_diagnostic_rows[:limit],
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "internal_phase_law_predeclared_diagnostic_class.json"
    md_path = out_dir / "INTERNAL_PHASE_LAW_PREDECLARED_DIAGNOSTIC_CLASS.md"
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    md_path.write_text(_markdown(payload), encoding="utf-8")
    return payload


def _markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Internal Phase-Law Predeclared Diagnostic Class",
        "",
        "This report defines a diagnostic class only. It does not change checkpoint promotion rules.",
        "",
        f"- status: `{payload.get('status')}`",
        f"- class: `{payload.get('class_name')}`",
        f"- promotion effect: `{payload.get('promotion_effect')}`",
        f"- tracks: `{payload.get('track_count')}`",
        f"- total rows: `{payload.get('total_row_count')}`",
        f"- diagnostic rows: `{payload.get('diagnostic_count')}`",
        f"- strict audio rows: `{payload.get('strict_audio_count')}`",
        f"- best track by count: `{payload.get('best_track_by_count')}`",
        "",
        "## Thresholds",
        "",
    ]
    for key, value in (payload.get("definition") or {}).get("thresholds", {}).items():
        lines.append(f"- `{key}`: `{value}`")
    lines.extend(
        [
            "",
            "## Track Summary",
            "",
            "| track | joint status | rows | diagnostic | strict | best run | best gain | best direct | best abs | best nonbad | best good |",
            "|---|---|---:|---:|---:|---|---:|---:|---:|---:|---:|",
        ]
    )
    for track in payload.get("tracks", []):
        best = track.get("best_diagnostic_row") if isinstance(track.get("best_diagnostic_row"), dict) else {}
        lines.append(
            "| {track} | {status} | {rows} | {diag} | {strict} | {run} | {gain} | {direct} | {absmean} | {nonbad} | {good} |".format(
                track=track.get("track"),
                status=track.get("joint_status"),
                rows=_fmt(track.get("row_count")),
                diag=_fmt(track.get("diagnostic_count")),
                strict=_fmt(track.get("strict_audio_count")),
                run=best.get("run_label"),
                gain=_fmt(best.get("gain")),
                direct=_fmt(best.get("direct_mean_corr_delta")),
                absmean=_fmt(best.get("absolute_mean_corr_delta")),
                nonbad=_fmt(best.get("absolute_nonbad_bin_mean_corr_delta")),
                good=_fmt(best.get("absolute_good_bin_mean_corr_delta")),
            )
        )
    lines.extend(
        [
            "",
            "## Top Diagnostic Rows",
            "",
            "| track | run | mask | gain | score | direct | direct wins | abs | abs wins | nonbad | good |",
            "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in payload.get("top_rows", []):
        lines.append(
            "| {track} | {run} | {mask} | {gain} | {score} | {direct} | {dwins} | {absmean} | {awins} | {nonbad} | {good} |".format(
                track=row.get("track"),
                run=row.get("run_label"),
                mask=row.get("mask_mode"),
                gain=_fmt(row.get("gain")),
                score=_fmt(row.get("diagnostic_score")),
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


def _parse_joint_row_args(values: list[str]) -> dict[str, Path]:
    if not values:
        return dict(DEFAULT_JOINT_ROWS)
    out: dict[str, Path] = {}
    for raw in values:
        if "=" not in raw:
            raise RuntimeError(f"Expected --joint-row label=path, got {raw!r}")
        label, path = raw.split("=", 1)
        out[label] = Path(path)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Predeclare and audit the small-direct/strong-absolute diagnostic class.")
    parser.add_argument("--joint-row", action="append", default=[], help="Optional label=path override. Repeatable.")
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--limit", type=int, default=32)
    args = parser.parse_args()
    payload = build_report(_parse_joint_row_args(args.joint_row), Path(args.out_dir), int(args.limit))
    print(
        json.dumps(
            {
                "json": str(Path(args.out_dir) / "internal_phase_law_predeclared_diagnostic_class.json"),
                "markdown": str(Path(args.out_dir) / "INTERNAL_PHASE_LAW_PREDECLARED_DIAGNOSTIC_CLASS.md"),
                "status": payload["status"],
                "diagnostic_count": payload["diagnostic_count"],
                "strict_audio_count": payload["strict_audio_count"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
