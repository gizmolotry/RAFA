from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SCHEMA = "rafa_learned_signature_split_suite_compare_v0"
TOKENBURST_ROOT = Path(r"C:\Users\Andrew\AppData\Local\Temp\rafa_tokenburst")
DEFAULT_OUT_DIR = TOKENBURST_ROOT / "learned_signature_split_suite_compare_2026_05_07"


def _utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return float(default)
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _mean(values: list[float]) -> float | None:
    finite = [float(value) for value in values if isinstance(value, (int, float))]
    return float(sum(finite) / len(finite)) if finite else None


def _child_train_metric_mean(payload: dict[str, Any], metric: str) -> float | None:
    values: list[float] = []
    for row in payload.get("rows", []) or []:
        if not isinstance(row, dict):
            continue
        direct = row.get(metric)
        if direct is not None:
            values.append(_as_float(direct))
            continue
        path = row.get("path")
        if not path:
            continue
        try:
            child = json.loads(Path(path).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        train_summary = child.get("train_summary") if isinstance(child.get("train_summary"), dict) else {}
        if metric in train_summary:
            values.append(_as_float(train_summary.get(metric)))
    return _mean(values)


def _default_suite_paths() -> list[Path]:
    paths = []
    for path in sorted(TOKENBURST_ROOT.glob("learned_signature_split_suite*/learned_signature_split_suite.json")):
        name = path.parent.name.lower()
        if "smoke" in name:
            continue
        paths.append(path)
    return paths


def _row(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rollup = payload.get("rollup") if isinstance(payload.get("rollup"), dict) else {}
    recipe = payload.get("recipe") if isinstance(payload.get("recipe"), dict) else {}
    hard_cases = [row for row in rollup.get("hard_stability_cases", []) or [] if isinstance(row, dict)]
    return {
        "path": str(path),
        "profile": payload.get("profile", path.parent.name),
        "directory": path.parent.name,
        "status": payload.get("status", rollup.get("status")),
        "promotion_effect": payload.get("promotion_effect"),
        "split_set": payload.get("split_set"),
        "split_count": rollup.get("split_count"),
        "completed_count": rollup.get("completed_count"),
        "failed_count": rollup.get("failed_count"),
        "candidate_count": rollup.get("candidate_count"),
        "candidate_fraction": rollup.get("candidate_fraction"),
        "heldout_row_count": rollup.get("heldout_row_count"),
        "heldout_candidate_count": rollup.get("heldout_candidate_count"),
        "heldout_two_axis_candidate_count": rollup.get("heldout_two_axis_candidate_count"),
        "heldout_stability_anticollapse_candidate_count": rollup.get(
            "heldout_stability_anticollapse_candidate_count"
        ),
        "seed_candidate_fraction": rollup.get("seed_candidate_fraction"),
        "partition_candidate_fraction": rollup.get("partition_candidate_fraction"),
        "mean_stability_delta_learned_minus_metadata": rollup.get(
            "mean_stability_delta_learned_minus_metadata"
        ),
        "mean_separation_delta_learned_minus_metadata": rollup.get(
            "mean_separation_delta_learned_minus_metadata"
        ),
        "mean_collapse_effective_rank_delta": rollup.get("mean_collapse_effective_rank_delta"),
        "mean_collapse_pairwise_distance_delta": rollup.get("mean_collapse_pairwise_distance_delta"),
        "mean_final_law_signature_loss": rollup.get("mean_final_law_signature_loss")
        if rollup.get("mean_final_law_signature_loss") is not None
        else _child_train_metric_mean(payload, "final_law_signature_loss"),
        "mean_final_operator_seed_loss": rollup.get("mean_final_operator_seed_loss")
        if rollup.get("mean_final_operator_seed_loss") is not None
        else _child_train_metric_mean(payload, "final_operator_seed_loss"),
        "mean_final_tail_law_signature_loss": rollup.get("mean_final_tail_law_signature_loss")
        if rollup.get("mean_final_tail_law_signature_loss") is not None
        else _child_train_metric_mean(payload, "final_tail_law_signature_loss"),
        "mean_final_tail_operator_seed_loss": rollup.get("mean_final_tail_operator_seed_loss")
        if rollup.get("mean_final_tail_operator_seed_loss") is not None
        else _child_train_metric_mean(payload, "final_tail_operator_seed_loss"),
        "mean_final_tail_operator_alignment_loss": rollup.get("mean_final_tail_operator_alignment_loss")
        if rollup.get("mean_final_tail_operator_alignment_loss") is not None
        else _child_train_metric_mean(payload, "final_tail_operator_alignment_loss"),
        "mean_final_matryoshka_loss": rollup.get("mean_final_matryoshka_loss")
        if rollup.get("mean_final_matryoshka_loss") is not None
        else _child_train_metric_mean(payload, "final_matryoshka_loss"),
        "mean_prefix640_stability_score": rollup.get("mean_prefix640_stability_score"),
        "mean_tail640_stability_score": rollup.get("mean_tail640_stability_score"),
        "mean_prefix640_separation_score": rollup.get("mean_prefix640_separation_score"),
        "mean_tail640_separation_score": rollup.get("mean_tail640_separation_score"),
        "mean_prefix640_effective_rank": rollup.get("mean_prefix640_effective_rank"),
        "mean_tail640_effective_rank": rollup.get("mean_tail640_effective_rank"),
        "metamer_consistency_weight": recipe.get("metamer_consistency_weight"),
        "run_contrastive_weight": recipe.get("run_contrastive_weight"),
        "base_svd_rank_weight": recipe.get("base_svd_rank_weight"),
        "factor_geometry_weight": recipe.get("factor_geometry_weight"),
        "redact_law_operator_inputs": bool(recipe.get("redact_law_operator_inputs", False)),
        "metamer_consistency_slice": recipe.get("metamer_consistency_slice", "full"),
        "case_separation_slice": recipe.get("case_separation_slice", "full"),
        "run_contrastive_slice": recipe.get("run_contrastive_slice", "full"),
        "selection_objective": recipe.get("selection_objective"),
        "hard_stability_cases": hard_cases[:5],
    }


def _rank_score(row: dict[str, Any]) -> tuple[float, float, float, float, float]:
    return (
        _as_float(row.get("candidate_fraction")),
        _as_float(row.get("seed_candidate_fraction")),
        _as_float(row.get("partition_candidate_fraction")),
        _as_float(row.get("mean_stability_delta_learned_minus_metadata")),
        _as_float(row.get("mean_separation_delta_learned_minus_metadata")),
    )


def compare_split_suites(paths: list[Path], out_dir: Path) -> dict[str, Any]:
    rows = [_row(path) for path in paths if path.exists()]
    robust_rows = [row for row in rows if row.get("status") in {"split_suite_all_passed", "split_suite_majority_passed"}]
    fragile_rows = [row for row in rows if row.get("status") == "split_suite_fragile_candidate"]
    partial_rows = [row for row in rows if row.get("status") == "split_suite_partial_run_failed"]
    redacted_rows = [row for row in rows if row.get("redact_law_operator_inputs")]
    best = max(rows, key=_rank_score, default={})
    best_stability = max(
        rows,
        key=lambda row: _as_float(row.get("mean_stability_delta_learned_minus_metadata")),
        default={},
    )
    best_partition = max(
        rows,
        key=lambda row: _as_float(row.get("partition_candidate_fraction")),
        default={},
    )
    if robust_rows:
        status = "robust_split_suite_candidate_found"
    elif fragile_rows:
        status = "fragile_split_suite_candidate_only"
    elif partial_rows:
        status = "split_suite_compare_partial_failures"
    elif rows:
        status = "no_split_suite_candidate"
    else:
        status = "missing"
    payload = {
        "schema": SCHEMA,
        "created_utc": _utc_timestamp(),
        "status": status,
        "promotion_effect": "none",
        "row_count": len(rows),
        "robust_candidate_count": len(robust_rows),
        "fragile_candidate_count": len(fragile_rows),
        "partial_failure_count": len(partial_rows),
        "redacted_suite_count": len(redacted_rows),
        "redacted_candidate_count": len([row for row in redacted_rows if _as_float(row.get("candidate_count")) > 0.0]),
        "best_profile": best.get("profile"),
        "best_status": best.get("status"),
        "best_candidate_fraction": best.get("candidate_fraction"),
        "best_seed_candidate_fraction": best.get("seed_candidate_fraction"),
        "best_partition_candidate_fraction": best.get("partition_candidate_fraction"),
        "best_mean_stability_delta": best.get("mean_stability_delta_learned_minus_metadata"),
        "best_stability_profile": best_stability.get("profile"),
        "best_stability_delta": best_stability.get("mean_stability_delta_learned_minus_metadata"),
        "best_partition_profile": best_partition.get("profile"),
        "best_partition_fraction": best_partition.get("partition_candidate_fraction"),
        "rows": sorted(rows, key=_rank_score, reverse=True),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    json_text = json.dumps(payload, indent=2)
    (out_dir / "learned_signature_split_suite_compare.json").write_text(json_text, encoding="utf-8")
    (out_dir / "track_compare.json").write_text(json_text, encoding="utf-8")
    (out_dir / "track_summary.json").write_text(json_text, encoding="utf-8")
    _write_markdown(out_dir / "LEARNED_SIGNATURE_SPLIT_SUITE_COMPARE.md", payload)
    (out_dir / "TRACK_REPORT.md").write_text(
        (out_dir / "LEARNED_SIGNATURE_SPLIT_SUITE_COMPARE.md").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    return payload


def _fmt(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def _write_markdown(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# Learned Signature Split Suite Compare",
        "",
        f"- schema: `{payload['schema']}`",
        f"- status: `{payload['status']}`",
        f"- rows: `{payload['row_count']}`",
        f"- robust candidates: `{payload['robust_candidate_count']}`",
        f"- fragile candidates: `{payload['fragile_candidate_count']}`",
        f"- redacted suites: `{payload.get('redacted_suite_count', 0)}`",
        f"- best profile: `{payload.get('best_profile')}`",
        f"- best candidate fraction: `{_fmt(payload.get('best_candidate_fraction'))}`",
        f"- best seed candidate fraction: `{_fmt(payload.get('best_seed_candidate_fraction'))}`",
        f"- best partition candidate fraction: `{_fmt(payload.get('best_partition_candidate_fraction'))}`",
        f"- best mean stability delta: `{_fmt(payload.get('best_mean_stability_delta'))}`",
        "",
        "| profile | status | redacted | slices | cand frac | seed frac | partition frac | mean stability d | mean separation d | tail align | hard cases |",
        "|---|---|---:|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in payload.get("rows", []) or []:
        hard = ", ".join(str(item.get("case")) for item in row.get("hard_stability_cases", [])[:3])
        lines.append(
            "| {profile} | {status} | {redacted} | {slices} | {cand} | {seed} | {part} | {stable} | {sep} | {align} | {hard} |".format(
                profile=row.get("profile"),
                status=row.get("status"),
                redacted="yes" if row.get("redact_law_operator_inputs") else "no",
                slices="{}/{}/{}".format(
                    row.get("metamer_consistency_slice", "full"),
                    row.get("case_separation_slice", "full"),
                    row.get("run_contrastive_slice", "full"),
                ),
                cand=_fmt(row.get("candidate_fraction")),
                seed=_fmt(row.get("seed_candidate_fraction")),
                part=_fmt(row.get("partition_candidate_fraction")),
                stable=_fmt(row.get("mean_stability_delta_learned_minus_metadata")),
                sep=_fmt(row.get("mean_separation_delta_learned_minus_metadata")),
                align=_fmt(row.get("mean_final_tail_operator_alignment_loss")),
                hard=hard,
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description="Compare learned RAFA signature split-suite artifacts.")
    ap.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    ap.add_argument("--suite-json", action="append", default=[])
    args = ap.parse_args()
    paths = [Path(item) for item in args.suite_json] if args.suite_json else _default_suite_paths()
    payload = compare_split_suites(paths, Path(args.out_dir))
    print(
        json.dumps(
            {
                "schema": payload["schema"],
                "saved": str(Path(args.out_dir) / "learned_signature_split_suite_compare.json"),
                "status": payload["status"],
                "row_count": payload["row_count"],
                "best_profile": payload.get("best_profile"),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
