from __future__ import annotations

import argparse
import json
import math
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
CORE = ROOT / "core"
LINEAGE = ROOT / "lineages" / "04_positive_replacement"
RUNTIME = ROOT / "runtimes" / "circleworld_proto"
for path in (ROOT, CORE, LINEAGE, RUNTIME):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from train_learned_signature_scout import (  # noqa: E402
    DEFAULT_CONFIG,
    TOKENBURST_ROOT,
    _default_cases_for_scout,
    train_learned_signature_scout,
)


SCHEMA = "rafa_learned_signature_split_suite_v0"
METRIC_EPSILON = 1.0e-4
DEFAULT_OUT_DIR = TOKENBURST_ROOT / "learned_signature_split_suite_v24_tf6_2026_05_07"


def _utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return float(default)
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _default_split_specs(split_set: str) -> list[dict[str, Any]]:
    smoke = [
        {
            "split_id": "smoke_seed1729_offset0",
            "kind": "smoke",
            "seed": 1729,
            "case_offset": 0,
            "case_shuffle_seed": None,
        }
    ]
    seeds = [
        {
            "split_id": "seed1729_offset0",
            "kind": "seed",
            "seed": 1729,
            "case_offset": 0,
            "case_shuffle_seed": None,
        },
        {
            "split_id": "seed2718_offset0",
            "kind": "seed",
            "seed": 2718,
            "case_offset": 0,
            "case_shuffle_seed": None,
        },
        {
            "split_id": "seed3141_offset0",
            "kind": "seed",
            "seed": 3141,
            "case_offset": 0,
            "case_shuffle_seed": None,
        },
    ]
    partitions = [
        {
            "split_id": "offset3_seed1729",
            "kind": "rotation",
            "seed": 1729,
            "case_offset": 3,
            "case_shuffle_seed": None,
        },
        {
            "split_id": "offset6_seed1729",
            "kind": "rotation",
            "seed": 1729,
            "case_offset": 6,
            "case_shuffle_seed": None,
        },
        {
            "split_id": "offset12_seed1729",
            "kind": "rotation",
            "seed": 1729,
            "case_offset": 12,
            "case_shuffle_seed": None,
        },
        {
            "split_id": "shuffle101_seed1729",
            "kind": "shuffle",
            "seed": 1729,
            "case_offset": 0,
            "case_shuffle_seed": 101,
        },
        {
            "split_id": "shuffle202_seed1729",
            "kind": "shuffle",
            "seed": 1729,
            "case_offset": 0,
            "case_shuffle_seed": 202,
        },
    ]
    split_set = str(split_set)
    if split_set == "smoke":
        return smoke
    if split_set == "seed":
        return seeds
    if split_set == "partition":
        return partitions
    if split_set == "core":
        return [*seeds, *partitions]
    raise ValueError(f"unsupported split set: {split_set!r}")


def _parse_extra_split(raw: str) -> dict[str, Any]:
    parts = [part.strip() for part in str(raw).split(":")]
    if len(parts) != 4:
        raise ValueError("extra split must have form split_id:seed:case_offset:case_shuffle_seed_or_none")
    split_id, seed, case_offset, shuffle = parts
    return {
        "split_id": split_id,
        "kind": "extra",
        "seed": int(seed),
        "case_offset": int(case_offset),
        "case_shuffle_seed": None if shuffle.lower() in {"", "none", "null", "-"} else int(shuffle),
    }


def _collapse_deltas(summary: dict[str, Any]) -> dict[str, float]:
    collapse = summary.get("aggregate", {}).get("collapse", {})
    delta = collapse.get("delta_learned_minus_metadata") if isinstance(collapse, dict) else {}
    if not isinstance(delta, dict):
        delta = {}
    return {
        "dominant_cluster_share": _as_float(delta.get("dominant_cluster_share")),
        "near_duplicate_pair_share": _as_float(delta.get("near_duplicate_pair_share")),
        "effective_rank": _as_float(delta.get("effective_rank")),
        "mean_pairwise_distance": _as_float(delta.get("mean_pairwise_distance")),
    }


def _metric_row(summary: dict[str, Any], *, split_id: str, kind: str, path: Path) -> dict[str, Any]:
    aggregate = summary.get("aggregate") if isinstance(summary.get("aggregate"), dict) else {}
    stability = aggregate.get("stability") if isinstance(aggregate.get("stability"), dict) else {}
    separation = aggregate.get("separation") if isinstance(aggregate.get("separation"), dict) else {}
    slice_views = aggregate.get("slice_views") if isinstance(aggregate.get("slice_views"), dict) else {}
    prefix640 = slice_views.get("prefix640") if isinstance(slice_views.get("prefix640"), dict) else {}
    tail640 = slice_views.get("tail640") if isinstance(slice_views.get("tail640"), dict) else {}
    prefix640_sep = prefix640.get("separation") if isinstance(prefix640.get("separation"), dict) else {}
    tail640_sep = tail640.get("separation") if isinstance(tail640.get("separation"), dict) else {}
    prefix640_collapse = prefix640.get("collapse") if isinstance(prefix640.get("collapse"), dict) else {}
    tail640_collapse = tail640.get("collapse") if isinstance(tail640.get("collapse"), dict) else {}
    collapse_delta = _collapse_deltas(summary)
    verdict = summary.get("verdict") if isinstance(summary.get("verdict"), dict) else {}
    train_summary = summary.get("train_summary") if isinstance(summary.get("train_summary"), dict) else {}
    contract = train_summary.get("contract") if isinstance(train_summary.get("contract"), dict) else {}
    stability_delta = _as_float(stability.get("delta_learned_minus_metadata"))
    separation_delta = _as_float(separation.get("delta_learned_minus_metadata"))
    anti_collapse_pass = (
        collapse_delta["dominant_cluster_share"] <= 0.02
        and collapse_delta["near_duplicate_pair_share"] <= 0.02
        and collapse_delta["effective_rank"] >= -0.05
        and collapse_delta["mean_pairwise_distance"] >= -0.02
    )
    stability_pass = stability_delta > METRIC_EPSILON
    separation_pass = separation_delta > METRIC_EPSILON
    candidate = stability_pass and separation_pass and anti_collapse_pass
    heldout_evidence = str(summary.get("split_mode", "")).startswith("heldout")
    worst_cases = []
    for case in summary.get("cases", []):
        case_stability = case.get("stability") if isinstance(case, dict) else {}
        if not isinstance(case_stability, dict):
            continue
        worst_cases.append(
            {
                "case": case.get("case"),
                "stability_delta_learned_minus_metadata": _as_float(
                    case_stability.get("delta_learned_minus_metadata")
                ),
            }
        )
    worst_cases = sorted(worst_cases, key=lambda row: row["stability_delta_learned_minus_metadata"])[:5]
    return {
        "split_id": split_id,
        "kind": kind,
        "path": str(path),
        "status": verdict.get("status"),
        "promotion_effect": verdict.get("promotion_effect"),
        "seed": summary.get("seed"),
        "case_offset": summary.get("case_offset"),
        "case_shuffle_seed": summary.get("case_shuffle_seed"),
        "split_mode": summary.get("split_mode"),
        "train_case_count": summary.get("train_case_count"),
        "heldout_case_count": summary.get("heldout_case_count"),
        "train_case_names": summary.get("train_case_names", []),
        "heldout_case_names": summary.get("heldout_case_names", []),
        "stability_delta_learned_minus_metadata": stability_delta,
        "separation_delta_learned_minus_metadata": separation_delta,
        "collapse_dominant_cluster_delta": collapse_delta["dominant_cluster_share"],
        "collapse_near_duplicate_delta": collapse_delta["near_duplicate_pair_share"],
        "collapse_effective_rank_delta": collapse_delta["effective_rank"],
        "collapse_mean_pairwise_distance_delta": collapse_delta["mean_pairwise_distance"],
        "final_law_signature_loss": _as_float(train_summary.get("final_law_signature_loss")),
        "final_operator_seed_loss": _as_float(train_summary.get("final_operator_seed_loss")),
        "final_tail_law_signature_loss": _as_float(train_summary.get("final_tail_law_signature_loss")),
        "final_tail_operator_seed_loss": _as_float(train_summary.get("final_tail_operator_seed_loss")),
        "final_tail_operator_alignment_loss": _as_float(train_summary.get("final_tail_operator_alignment_loss")),
        "final_matryoshka_loss": _as_float(train_summary.get("final_matryoshka_loss")),
        "prefix640_stability_score": _as_float(prefix640.get("stability_score")),
        "tail640_stability_score": _as_float(tail640.get("stability_score")),
        "prefix640_separation_score": _as_float(prefix640_sep.get("mean_pairwise_distance")),
        "tail640_separation_score": _as_float(tail640_sep.get("mean_pairwise_distance")),
        "prefix640_effective_rank": _as_float(prefix640_collapse.get("effective_rank")),
        "tail640_effective_rank": _as_float(tail640_collapse.get("effective_rank")),
        "redact_law_operator_inputs": bool(train_summary.get("redact_law_operator_inputs", False)),
        "tail_loss_semantics": train_summary.get("tail_loss_semantics"),
        "metamer_consistency_slice": train_summary.get("metamer_consistency_slice", train_summary.get("metamer_loss_view")),
        "case_separation_slice": train_summary.get("case_separation_slice", train_summary.get("case_separation_loss_view")),
        "run_contrastive_slice": train_summary.get("run_contrastive_slice", train_summary.get("run_contrastive_loss_view")),
        "law_signature_shape": contract.get("law_signature_shape"),
        "operator_seed_shape": contract.get("operator_seed_shape"),
        "stability_pass": stability_pass,
        "separation_pass": separation_pass,
        "anti_collapse_pass": anti_collapse_pass,
        "candidate": candidate,
        "two_axis_candidate": stability_pass and separation_pass,
        "stability_anticollapse_candidate": stability_pass and anti_collapse_pass,
        "heldout_evidence": heldout_evidence,
        "heldout_candidate": heldout_evidence and candidate,
        "heldout_two_axis_candidate": heldout_evidence and stability_pass and separation_pass,
        "heldout_stability_anticollapse_candidate": heldout_evidence and stability_pass and anti_collapse_pass,
        "worst_stability_cases": worst_cases,
        "verdict_reasons": verdict.get("reasons", []),
    }


def _mean(rows: list[dict[str, Any]], key: str) -> float:
    values = [_as_float(row.get(key)) for row in rows if row.get(key) is not None]
    return float(sum(values) / len(values)) if values else 0.0


def _summarize_rows(rows: list[dict[str, Any]], failed_rows: list[dict[str, Any]]) -> dict[str, Any]:
    split_count = len(rows) + len(failed_rows)
    candidate_rows = [row for row in rows if row.get("candidate")]
    heldout_rows = [row for row in rows if row.get("heldout_evidence")]
    heldout_candidate_rows = [row for row in rows if row.get("heldout_candidate")]
    heldout_two_axis_rows = [row for row in rows if row.get("heldout_two_axis_candidate")]
    heldout_stability_anticollapse_rows = [
        row for row in rows if row.get("heldout_stability_anticollapse_candidate")
    ]
    two_axis_rows = [row for row in rows if row.get("two_axis_candidate")]
    stability_anticollapse_rows = [row for row in rows if row.get("stability_anticollapse_candidate")]
    seed_rows = [row for row in rows if row.get("kind") == "seed"]
    partition_rows = [row for row in rows if row.get("kind") in {"rotation", "shuffle"}]
    hard_cases: dict[str, dict[str, Any]] = {}
    for row in rows:
        for case in row.get("worst_stability_cases", []):
            name = str(case.get("case"))
            if not name or name == "None":
                continue
            bucket = hard_cases.setdefault(
                name,
                {
                    "case": name,
                    "hit_count": 0,
                    "min_stability_delta_learned_minus_metadata": math.inf,
                    "split_ids": [],
                },
            )
            bucket["hit_count"] += 1
            bucket["min_stability_delta_learned_minus_metadata"] = min(
                _as_float(bucket["min_stability_delta_learned_minus_metadata"]),
                _as_float(case.get("stability_delta_learned_minus_metadata")),
            )
            bucket["split_ids"].append(row.get("split_id"))
    hard_case_rows = sorted(
        hard_cases.values(),
        key=lambda row: (row["min_stability_delta_learned_minus_metadata"], -row["hit_count"]),
    )[:10]
    if failed_rows:
        status = "split_suite_partial_run_failed" if rows else "split_suite_failed_to_run"
    elif rows and not heldout_rows:
        status = "split_suite_in_sample_only_needs_heldout"
    elif split_count and len(heldout_candidate_rows) == split_count:
        status = "split_suite_all_passed"
    elif split_count and len(heldout_candidate_rows) >= max(1, math.ceil(split_count / 2)):
        status = "split_suite_majority_passed"
    elif heldout_candidate_rows:
        status = "split_suite_fragile_candidate"
    elif heldout_two_axis_rows and heldout_stability_anticollapse_rows:
        status = "split_suite_complementary_near_candidates"
    elif heldout_two_axis_rows:
        status = "split_suite_two_axis_collapse_blocked"
    elif rows:
        status = "split_suite_no_candidate"
    else:
        status = "split_suite_failed_to_run"
    return {
        "status": status,
        "split_count": split_count,
        "completed_count": len(rows),
        "failed_count": len(failed_rows),
        "candidate_count": len(candidate_rows),
        "candidate_fraction": (len(candidate_rows) / split_count) if split_count else 0.0,
        "two_axis_candidate_count": len(two_axis_rows),
        "stability_anticollapse_candidate_count": len(stability_anticollapse_rows),
        "heldout_row_count": len(heldout_rows),
        "heldout_candidate_count": len(heldout_candidate_rows),
        "heldout_two_axis_candidate_count": len(heldout_two_axis_rows),
        "heldout_stability_anticollapse_candidate_count": len(heldout_stability_anticollapse_rows),
        "seed_split_count": len(seed_rows),
        "seed_candidate_count": len([row for row in seed_rows if row.get("heldout_candidate")]),
        "seed_candidate_fraction": (
            len([row for row in seed_rows if row.get("heldout_candidate")]) / len(seed_rows) if seed_rows else 0.0
        ),
        "partition_split_count": len(partition_rows),
        "partition_candidate_count": len([row for row in partition_rows if row.get("heldout_candidate")]),
        "partition_candidate_fraction": (
            len([row for row in partition_rows if row.get("heldout_candidate")]) / len(partition_rows)
            if partition_rows
            else 0.0
        ),
        "mean_stability_delta_learned_minus_metadata": _mean(rows, "stability_delta_learned_minus_metadata"),
        "mean_separation_delta_learned_minus_metadata": _mean(rows, "separation_delta_learned_minus_metadata"),
        "mean_collapse_dominant_cluster_delta": _mean(rows, "collapse_dominant_cluster_delta"),
        "mean_collapse_near_duplicate_delta": _mean(rows, "collapse_near_duplicate_delta"),
        "mean_collapse_effective_rank_delta": _mean(rows, "collapse_effective_rank_delta"),
        "mean_collapse_pairwise_distance_delta": _mean(rows, "collapse_mean_pairwise_distance_delta"),
        "mean_final_law_signature_loss": _mean(rows, "final_law_signature_loss"),
        "mean_final_operator_seed_loss": _mean(rows, "final_operator_seed_loss"),
        "mean_final_tail_law_signature_loss": _mean(rows, "final_tail_law_signature_loss"),
        "mean_final_tail_operator_seed_loss": _mean(rows, "final_tail_operator_seed_loss"),
        "mean_final_tail_operator_alignment_loss": _mean(rows, "final_tail_operator_alignment_loss"),
        "mean_final_matryoshka_loss": _mean(rows, "final_matryoshka_loss"),
        "mean_prefix640_stability_score": _mean(rows, "prefix640_stability_score"),
        "mean_tail640_stability_score": _mean(rows, "tail640_stability_score"),
        "mean_prefix640_separation_score": _mean(rows, "prefix640_separation_score"),
        "mean_tail640_separation_score": _mean(rows, "tail640_separation_score"),
        "mean_prefix640_effective_rank": _mean(rows, "prefix640_effective_rank"),
        "mean_tail640_effective_rank": _mean(rows, "tail640_effective_rank"),
        "hard_stability_cases": hard_case_rows,
    }


def _write_markdown(path: Path, suite: dict[str, Any]) -> None:
    rollup = suite["rollup"]
    lines = [
        "# Learned Signature Split Suite",
        "",
        f"- schema: `{suite['schema']}`",
        f"- profile: `{suite['profile']}`",
        f"- status: `{rollup['status']}`",
        f"- split set: `{suite['split_set']}`",
        f"- split count: `{rollup['split_count']}`",
        f"- completed: `{rollup['completed_count']}`",
        f"- failed: `{rollup['failed_count']}`",
        f"- candidates: `{rollup['candidate_count']}` / `{rollup['split_count']}`",
        f"- seed candidates: `{rollup['seed_candidate_count']}` / `{rollup['seed_split_count']}`",
        f"- partition candidates: `{rollup['partition_candidate_count']}` / `{rollup['partition_split_count']}`",
        f"- redact law/operator inputs: `{suite.get('recipe', {}).get('redact_law_operator_inputs', False)}`",
        f"- metamer consistency slice: `{suite.get('recipe', {}).get('metamer_consistency_slice', 'full')}`",
        f"- case separation slice: `{suite.get('recipe', {}).get('case_separation_slice', 'full')}`",
        f"- run contrastive slice: `{suite.get('recipe', {}).get('run_contrastive_slice', 'full')}`",
        f"- mean stability delta: `{rollup['mean_stability_delta_learned_minus_metadata']:+.6f}`",
        f"- mean separation delta: `{rollup['mean_separation_delta_learned_minus_metadata']:+.6f}`",
        f"- mean effective-rank delta: `{rollup['mean_collapse_effective_rank_delta']:+.6f}`",
        f"- mean final law signature loss: `{rollup.get('mean_final_law_signature_loss', 0.0):.6f}`",
        f"- mean final operator seed loss: `{rollup.get('mean_final_operator_seed_loss', 0.0):.6f}`",
        f"- mean final tail law signature loss: `{rollup.get('mean_final_tail_law_signature_loss', 0.0):.6f}`",
        f"- mean final tail operator seed loss: `{rollup.get('mean_final_tail_operator_seed_loss', 0.0):.6f}`",
        f"- mean final tail operator alignment loss: `{rollup.get('mean_final_tail_operator_alignment_loss', 0.0):.6f}`",
        f"- mean prefix640 stability score: `{rollup.get('mean_prefix640_stability_score', 0.0):.6f}`",
        f"- mean tail640 stability score: `{rollup.get('mean_tail640_stability_score', 0.0):.6f}`",
        f"- mean prefix640 separation score: `{rollup.get('mean_prefix640_separation_score', 0.0):.6f}`",
        f"- mean tail640 separation score: `{rollup.get('mean_tail640_separation_score', 0.0):.6f}`",
        "",
        "## Splits",
        "",
        "| split | kind | status | candidate | stability | separation | rank delta | worst cases |",
        "|---|---:|---|---:|---:|---:|---:|---|",
    ]
    for row in suite.get("rows", []):
        worst = ", ".join(str(case.get("case")) for case in row.get("worst_stability_cases", [])[:3])
        lines.append(
            "| {split_id} | {kind} | {status} | {candidate} | {stability:+.6f} | {separation:+.6f} | {rank:+.6f} | {worst} |".format(
                split_id=row.get("split_id"),
                kind=row.get("kind"),
                status=row.get("status"),
                candidate="yes" if row.get("candidate") else "no",
                stability=_as_float(row.get("stability_delta_learned_minus_metadata")),
                separation=_as_float(row.get("separation_delta_learned_minus_metadata")),
                rank=_as_float(row.get("collapse_effective_rank_delta")),
                worst=worst,
            )
        )
    if suite.get("failed_rows"):
        lines.extend(["", "## Failed Splits", ""])
        for row in suite["failed_rows"]:
            lines.append(f"- `{row.get('split_id')}`: {row.get('error_type')}: {row.get('error')}")
    lines.extend(["", "## Hard Stability Cases", ""])
    for row in rollup.get("hard_stability_cases", []):
        lines.append(
            f"- `{row['case']}`: min stability delta `{row['min_stability_delta_learned_minus_metadata']:+.6f}`, hits `{row['hit_count']}`"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_split_suite(
    *,
    config_path: Path,
    cases_path: Path,
    out_dir: Path,
    profile: str,
    split_set: str,
    extra_splits: list[str],
    reuse_existing: bool,
    stop_on_failure: bool,
    device_name: str,
    num_cases: int,
    heldout_cases: int,
    num_transforms: int,
    clip_seconds: int,
    epochs: int,
    hidden_dim: int,
    learning_rate: float,
    batch_size: int,
    metamer_consistency_weight: float,
    case_separation_weight: float,
    separation_margin: float,
    run_contrastive_weight: float,
    run_contrastive_temperature: float,
    rank_entropy_weight: float,
    rank_entropy_floor: float,
    run_rank_entropy_weight: float,
    run_rank_effective_dim_floor: float,
    base_rank_entropy_weight: float,
    base_rank_effective_dim_floor: float,
    base_svd_rank_weight: float,
    base_svd_rank_floor: float,
    factor_geometry_weight: float,
    metamer_consistency_slice: str,
    case_separation_slice: str,
    run_contrastive_slice: str,
    redact_law_operator_inputs: bool,
    selection_objective: str,
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    split_specs = [*_default_split_specs(split_set), *[_parse_extra_split(raw) for raw in extra_splits]]
    rows: list[dict[str, Any]] = []
    failed_rows: list[dict[str, Any]] = []
    for spec in split_specs:
        split_id = str(spec["split_id"])
        split_out_dir = out_dir / split_id
        split_json = split_out_dir / "learned_signature_scout.json"
        try:
            summary = None
            if reuse_existing and split_json.exists():
                cached_summary = json.loads(split_json.read_text(encoding="utf-8"))
                cached_train = (
                    cached_summary.get("train_summary")
                    if isinstance(cached_summary.get("train_summary"), dict)
                    else {}
                )
                cached_redacted = bool(cached_train.get("redact_law_operator_inputs", False))
                cached_slices_match = (
                    str(cached_train.get("metamer_consistency_slice", cached_train.get("metamer_loss_view", "full")))
                    == str(metamer_consistency_slice)
                    and str(cached_train.get("case_separation_slice", cached_train.get("case_separation_loss_view", "full")))
                    == str(case_separation_slice)
                    and str(cached_train.get("run_contrastive_slice", cached_train.get("run_contrastive_loss_view", "full")))
                    == str(run_contrastive_slice)
                )
                if cached_redacted == bool(redact_law_operator_inputs) and cached_slices_match:
                    summary = cached_summary
            if summary is None:
                summary = train_learned_signature_scout(
                    config_path=config_path,
                    cases_path=cases_path,
                    out_dir=split_out_dir,
                    device_name=device_name,
                    num_cases=int(num_cases),
                    heldout_cases=int(heldout_cases),
                    num_transforms=int(num_transforms),
                    clip_seconds=int(clip_seconds),
                    epochs=int(epochs),
                    hidden_dim=int(hidden_dim),
                    learning_rate=float(learning_rate),
                    batch_size=int(batch_size),
                    metamer_consistency_weight=float(metamer_consistency_weight),
                    case_separation_weight=float(case_separation_weight),
                    separation_margin=float(separation_margin),
                    run_contrastive_weight=float(run_contrastive_weight),
                    run_contrastive_temperature=float(run_contrastive_temperature),
                    rank_entropy_weight=float(rank_entropy_weight),
                    rank_entropy_floor=float(rank_entropy_floor),
                    run_rank_entropy_weight=float(run_rank_entropy_weight),
                    run_rank_effective_dim_floor=float(run_rank_effective_dim_floor),
                    base_rank_entropy_weight=float(base_rank_entropy_weight),
                    base_rank_effective_dim_floor=float(base_rank_effective_dim_floor),
                    base_svd_rank_weight=float(base_svd_rank_weight),
                    base_svd_rank_floor=float(base_svd_rank_floor),
                    factor_geometry_weight=float(factor_geometry_weight),
                    metamer_loss_view=str(metamer_consistency_slice),
                    case_separation_loss_view=str(case_separation_slice),
                    run_contrastive_loss_view=str(run_contrastive_slice),
                    redact_law_operator_inputs=bool(redact_law_operator_inputs),
                    seed=int(spec["seed"]),
                    case_offset=int(spec["case_offset"]),
                    case_shuffle_seed=spec.get("case_shuffle_seed"),
                    selection_objective=str(selection_objective),
                )
            rows.append(_metric_row(summary, split_id=split_id, kind=str(spec.get("kind")), path=split_json))
        except Exception as exc:  # pragma: no cover - diagnostics should survive partial suites.
            failed_rows.append(
                {
                    "split_id": split_id,
                    "kind": str(spec.get("kind")),
                    "seed": spec.get("seed"),
                    "case_offset": spec.get("case_offset"),
                    "case_shuffle_seed": spec.get("case_shuffle_seed"),
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                }
            )
            if stop_on_failure:
                raise
        finally:
            try:
                import torch

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass
    rollup = _summarize_rows(rows, failed_rows)
    suite = {
        "schema": SCHEMA,
        "status": rollup["status"],
        "promotion_effect": "none",
        "created_utc": _utc_timestamp(),
        "profile": str(profile),
        "split_set": str(split_set),
        "config_path": str(config_path),
        "cases_path": str(cases_path),
        "out_dir": str(out_dir),
        "metric_epsilon": METRIC_EPSILON,
        "recipe": {
            "device": str(device_name),
            "num_cases": int(num_cases),
            "heldout_cases": int(heldout_cases),
            "num_transforms": int(num_transforms),
            "clip_seconds": int(clip_seconds),
            "epochs": int(epochs),
            "hidden_dim": int(hidden_dim),
            "learning_rate": float(learning_rate),
            "batch_size": int(batch_size),
            "metamer_consistency_weight": float(metamer_consistency_weight),
            "case_separation_weight": float(case_separation_weight),
            "separation_margin": float(separation_margin),
            "run_contrastive_weight": float(run_contrastive_weight),
            "run_contrastive_temperature": float(run_contrastive_temperature),
            "rank_entropy_weight": float(rank_entropy_weight),
            "rank_entropy_floor": float(rank_entropy_floor),
            "run_rank_entropy_weight": float(run_rank_entropy_weight),
            "run_rank_effective_dim_floor": float(run_rank_effective_dim_floor),
            "base_rank_entropy_weight": float(base_rank_entropy_weight),
            "base_rank_effective_dim_floor": float(base_rank_effective_dim_floor),
            "base_svd_rank_weight": float(base_svd_rank_weight),
            "base_svd_rank_floor": float(base_svd_rank_floor),
            "factor_geometry_weight": float(factor_geometry_weight),
            "metamer_consistency_slice": str(metamer_consistency_slice),
            "case_separation_slice": str(case_separation_slice),
            "run_contrastive_slice": str(run_contrastive_slice),
            "redact_law_operator_inputs": bool(redact_law_operator_inputs),
            "selection_objective": str(selection_objective),
        },
        "split_specs": split_specs,
        "rollup": rollup,
        "rows": rows,
        "failed_rows": failed_rows,
    }
    json_path = out_dir / "learned_signature_split_suite.json"
    md_path = out_dir / "LEARNED_SIGNATURE_SPLIT_SUITE.md"
    json_text = json.dumps(suite, indent=2)
    json_path.write_text(json_text, encoding="utf-8")
    (out_dir / "track_summary.json").write_text(json_text, encoding="utf-8")
    _write_markdown(md_path, suite)
    (out_dir / "TRACK_REPORT.md").write_text(md_path.read_text(encoding="utf-8"), encoding="utf-8")
    return suite


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Run a learned RAFA signature scout recipe across seeds, rotations, and shuffled heldout splits."
    )
    ap.add_argument("--config", default=str(DEFAULT_CONFIG))
    ap.add_argument("--cases-json", default=str(_default_cases_for_scout()))
    ap.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    ap.add_argument("--profile", default="v24_tf6_baseline")
    ap.add_argument("--split-set", choices=("core", "seed", "partition", "smoke"), default="core")
    ap.add_argument("--extra-split", action="append", default=[])
    ap.add_argument("--reuse-existing", action="store_true")
    ap.add_argument("--stop-on-failure", action="store_true")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--num-cases", type=int, default=12)
    ap.add_argument("--heldout-cases", type=int, default=6)
    ap.add_argument("--num-transforms", type=int, default=6)
    ap.add_argument("--clip-seconds", type=int, default=1)
    ap.add_argument("--epochs", type=int, default=24)
    ap.add_argument("--hidden-dim", type=int, default=96)
    ap.add_argument("--learning-rate", type=float, default=3.0e-3)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--metamer-consistency-weight", type=float, default=0.0)
    ap.add_argument("--case-separation-weight", type=float, default=0.0)
    ap.add_argument("--separation-margin", type=float, default=0.65)
    ap.add_argument("--run-contrastive-weight", type=float, default=1.25)
    ap.add_argument("--run-contrastive-temperature", type=float, default=0.12)
    ap.add_argument("--rank-entropy-weight", type=float, default=0.0)
    ap.add_argument("--rank-entropy-floor", type=float, default=0.02)
    ap.add_argument("--run-rank-entropy-weight", type=float, default=0.0)
    ap.add_argument("--run-rank-effective-dim-floor", type=float, default=3.0)
    ap.add_argument("--base-rank-entropy-weight", type=float, default=0.0)
    ap.add_argument("--base-rank-effective-dim-floor", type=float, default=3.0)
    ap.add_argument("--base-svd-rank-weight", type=float, default=0.12)
    ap.add_argument("--base-svd-rank-floor", type=float, default=2.5)
    ap.add_argument("--factor-geometry-weight", type=float, default=0.0)
    ap.add_argument("--metamer-consistency-slice", default="full")
    ap.add_argument("--case-separation-slice", default="full")
    ap.add_argument("--run-contrastive-slice", default="full")
    ap.add_argument("--redact-law-operator-inputs", action="store_true")
    ap.add_argument("--selection-objective", choices=("final", "train_proxy"), default="final")
    args = ap.parse_args()
    suite = run_split_suite(
        config_path=Path(args.config),
        cases_path=Path(args.cases_json),
        out_dir=Path(args.out_dir),
        profile=str(args.profile),
        split_set=str(args.split_set),
        extra_splits=list(args.extra_split),
        reuse_existing=bool(args.reuse_existing),
        stop_on_failure=bool(args.stop_on_failure),
        device_name=str(args.device),
        num_cases=int(args.num_cases),
        heldout_cases=int(args.heldout_cases),
        num_transforms=int(args.num_transforms),
        clip_seconds=int(args.clip_seconds),
        epochs=int(args.epochs),
        hidden_dim=int(args.hidden_dim),
        learning_rate=float(args.learning_rate),
        batch_size=int(args.batch_size),
        metamer_consistency_weight=float(args.metamer_consistency_weight),
        case_separation_weight=float(args.case_separation_weight),
        separation_margin=float(args.separation_margin),
        run_contrastive_weight=float(args.run_contrastive_weight),
        run_contrastive_temperature=float(args.run_contrastive_temperature),
        rank_entropy_weight=float(args.rank_entropy_weight),
        rank_entropy_floor=float(args.rank_entropy_floor),
        run_rank_entropy_weight=float(args.run_rank_entropy_weight),
        run_rank_effective_dim_floor=float(args.run_rank_effective_dim_floor),
        base_rank_entropy_weight=float(args.base_rank_entropy_weight),
        base_rank_effective_dim_floor=float(args.base_rank_effective_dim_floor),
        base_svd_rank_weight=float(args.base_svd_rank_weight),
        base_svd_rank_floor=float(args.base_svd_rank_floor),
        factor_geometry_weight=float(args.factor_geometry_weight),
        metamer_consistency_slice=str(args.metamer_consistency_slice),
        case_separation_slice=str(args.case_separation_slice),
        run_contrastive_slice=str(args.run_contrastive_slice),
        redact_law_operator_inputs=bool(args.redact_law_operator_inputs),
        selection_objective=str(args.selection_objective),
    )
    print(
        json.dumps(
            {
                "schema": suite["schema"],
                "saved": str(Path(args.out_dir) / "learned_signature_split_suite.json"),
                "report": str(Path(args.out_dir) / "LEARNED_SIGNATURE_SPLIT_SUITE.md"),
                "status": suite["rollup"]["status"],
                "candidate_count": suite["rollup"]["candidate_count"],
                "split_count": suite["rollup"]["split_count"],
                "failed_count": suite["rollup"]["failed_count"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
