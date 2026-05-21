from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SCHEMA = "rafa_dense_signature_failure_modes_v0"
TOKENBURST_ROOT = Path(r"C:\Users\Andrew\AppData\Local\Temp\rafa_tokenburst")
DEFAULT_INPUT = (
    TOKENBURST_ROOT
    / "dense_signature_claim_broad18_cuda_2026_05_06"
    / "dense_signature_claim.json"
)
DEFAULT_OUT_DIR = TOKENBURST_ROOT / "dense_signature_failure_modes_2026_05_07"
EPS = 1.0e-12


def _utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return float(default)
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return float(sum(float(v) for v in values) / float(len(values)))


def _stddev(values: list[float]) -> float:
    if not values:
        return 0.0
    avg = _mean(values)
    return float((sum((float(v) - avg) ** 2 for v in values) / float(len(values))) ** 0.5)


def _fmt(value: Any, digits: int = 6) -> str:
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _safe_get(mapping: dict[str, Any], *keys: str, default: Any = None) -> Any:
    cur: Any = mapping
    for key in keys:
        if not isinstance(cur, dict):
            return default
        cur = cur.get(key)
    return default if cur is None else cur


def _dense_required_axes(payload: dict[str, Any]) -> dict[str, Any]:
    aggregate = payload.get("aggregate") if isinstance(payload.get("aggregate"), dict) else {}
    collapse_delta = _safe_get(aggregate, "collapse", "delta_dense_minus_metadata", default={})
    if not isinstance(collapse_delta, dict):
        collapse_delta = {}
    dense_collapse = _safe_get(aggregate, "collapse", "dense", default={})
    metadata_collapse = _safe_get(aggregate, "collapse", "metadata", default={})
    if not isinstance(dense_collapse, dict):
        dense_collapse = {}
    if not isinstance(metadata_collapse, dict):
        metadata_collapse = {}

    stability_delta = _as_float(_safe_get(aggregate, "stability", "delta_dense_minus_metadata"))
    separation_delta = _as_float(_safe_get(aggregate, "separation", "delta_dense_minus_metadata"))
    dominant_delta = _as_float(collapse_delta.get("dominant_cluster_share"))
    duplicate_delta = _as_float(collapse_delta.get("near_duplicate_pair_share"))
    rank_delta = _as_float(collapse_delta.get("effective_rank"))
    packet_distance_delta = _as_float(collapse_delta.get("mean_pairwise_distance"))
    effective_cluster_delta = _as_float(collapse_delta.get("effective_cluster_count"))

    thresholds = {
        "min_stability_delta": 1.0e-4,
        "min_separation_delta": 1.0e-4,
        "max_dominant_cluster_share_delta": 0.02,
        "max_near_duplicate_pair_share_delta": 0.02,
        "min_effective_rank_delta": -0.05,
        "min_packet_pairwise_distance_delta": -0.02,
    }
    axes = {
        "enough_cases": int(_as_float(payload.get("num_cases"))) >= 2,
        "stability_beats_metadata": stability_delta > thresholds["min_stability_delta"],
        "separation_beats_metadata": separation_delta > thresholds["min_separation_delta"],
        "dominant_cluster_not_worse": dominant_delta <= thresholds["max_dominant_cluster_share_delta"],
        "near_duplicate_not_worse": duplicate_delta <= thresholds["max_near_duplicate_pair_share_delta"],
        "effective_rank_not_worse": rank_delta >= thresholds["min_effective_rank_delta"],
        "packet_pairwise_distance_not_worse": packet_distance_delta >= thresholds["min_packet_pairwise_distance_delta"],
    }
    axes["anti_collapse_not_worse"] = all(
        bool(axes[name])
        for name in (
            "dominant_cluster_not_worse",
            "near_duplicate_not_worse",
            "effective_rank_not_worse",
            "packet_pairwise_distance_not_worse",
        )
    )
    axes["all_required_axes_pass"] = all(
        bool(axes[name])
        for name in (
            "enough_cases",
            "stability_beats_metadata",
            "separation_beats_metadata",
            "anti_collapse_not_worse",
        )
    )
    return {
        "thresholds": thresholds,
        "axes": axes,
        "deltas": {
            "stability_delta": stability_delta,
            "separation_delta": separation_delta,
            "dominant_cluster_share_delta": dominant_delta,
            "near_duplicate_pair_share_delta": duplicate_delta,
            "effective_rank_delta": rank_delta,
            "packet_pairwise_distance_delta": packet_distance_delta,
            "effective_cluster_count_delta": effective_cluster_delta,
        },
        "dense": {
            "stability_score": _as_float(_safe_get(aggregate, "stability", "dense_score")),
            "separation_score": _as_float(_safe_get(aggregate, "separation", "dense_score")),
            "dominant_cluster_share": _as_float(dense_collapse.get("dominant_cluster_share")),
            "near_duplicate_pair_share": _as_float(dense_collapse.get("near_duplicate_pair_share")),
            "effective_rank": _as_float(dense_collapse.get("effective_rank")),
            "mean_pairwise_distance": _as_float(dense_collapse.get("mean_pairwise_distance")),
            "effective_cluster_count": _as_float(dense_collapse.get("effective_cluster_count")),
        },
        "metadata": {
            "stability_score": _as_float(_safe_get(aggregate, "stability", "metadata_score")),
            "separation_score": _as_float(_safe_get(aggregate, "separation", "metadata_score")),
            "dominant_cluster_share": _as_float(metadata_collapse.get("dominant_cluster_share")),
            "near_duplicate_pair_share": _as_float(metadata_collapse.get("near_duplicate_pair_share")),
            "effective_rank": _as_float(metadata_collapse.get("effective_rank")),
            "mean_pairwise_distance": _as_float(metadata_collapse.get("mean_pairwise_distance")),
            "effective_cluster_count": _as_float(metadata_collapse.get("effective_cluster_count")),
        },
    }


def _prefix_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for case in payload.get("cases", []) or []:
        if not isinstance(case, dict):
            continue
        case_name = str(case.get("case", "unknown"))
        for transform in case.get("transforms", []) or []:
            if not isinstance(transform, dict):
                continue
            dense = transform.get("dense") if isinstance(transform.get("dense"), dict) else {}
            prefix_cos = dense.get("prefix_cosines") if isinstance(dense.get("prefix_cosines"), dict) else {}
            parsed: dict[int, float] = {}
            for raw_dim, raw_value in prefix_cos.items():
                try:
                    dim = int(raw_dim)
                except (TypeError, ValueError):
                    continue
                parsed[dim] = _as_float(raw_value)
            if not parsed:
                continue
            dims = sorted(parsed)
            rows.append(
                {
                    "case": case_name,
                    "transform": transform.get("transform"),
                    "transform_family": transform.get("transform_family"),
                    "prefix_cosines": {str(dim): parsed[dim] for dim in dims},
                    "full_cosine": parsed[dims[-1]],
                    "early_cosine": parsed[dims[0]],
                    "early_minus_full": float(parsed[dims[0]] - parsed[dims[-1]]),
                    "prefix_spread": float(max(parsed.values()) - min(parsed.values())),
                    "order_violations": sum(
                        1
                        for left, right in zip(dims[:-1], dims[1:])
                        if parsed[left] + 1.0e-5 < parsed[right]
                    ),
                    "num_adjacent_pairs": max(0, len(dims) - 1),
                }
            )
    return rows


def _prefix_summary(payload: dict[str, Any]) -> dict[str, Any]:
    rows = _prefix_rows(payload)
    if not rows:
        return {
            "status": "missing_prefix_metrics",
            "prefix_dims": [],
            "prefix_monotonicity_score": 0.0,
            "prefix_spread_mean": 0.0,
            "early_minus_full_mean": 0.0,
            "rows": [],
            "by_dim": {},
            "by_family": {},
        }

    dims = sorted({int(dim) for row in rows for dim in row["prefix_cosines"].keys()})
    by_dim: dict[str, dict[str, float]] = {}
    for dim in dims:
        values = [_as_float(row["prefix_cosines"].get(str(dim))) for row in rows if str(dim) in row["prefix_cosines"]]
        by_dim[str(dim)] = {
            "mean_cosine": _mean(values),
            "std_cosine": _stddev(values),
            "min_cosine": min(values) if values else 0.0,
            "max_cosine": max(values) if values else 0.0,
        }

    by_family: dict[str, dict[str, float]] = {}
    families = sorted({str(row.get("transform_family", "unknown")) for row in rows})
    for family in families:
        family_rows = [row for row in rows if str(row.get("transform_family", "unknown")) == family]
        by_family[family] = {
            "row_count": float(len(family_rows)),
            "early_minus_full_mean": _mean([_as_float(row["early_minus_full"]) for row in family_rows]),
            "prefix_spread_mean": _mean([_as_float(row["prefix_spread"]) for row in family_rows]),
            "full_cosine_mean": _mean([_as_float(row["full_cosine"]) for row in family_rows]),
            "early_cosine_mean": _mean([_as_float(row["early_cosine"]) for row in family_rows]),
        }

    total_pairs = sum(int(row["num_adjacent_pairs"]) for row in rows)
    total_violations = sum(int(row["order_violations"]) for row in rows)
    monotonicity = 1.0 - (float(total_violations) / float(max(1, total_pairs)))
    spread_mean = _mean([_as_float(row["prefix_spread"]) for row in rows])
    early_minus_full = _mean([_as_float(row["early_minus_full"]) for row in rows])
    if monotonicity >= 0.95 and rows:
        status = "prefix_order_contract_present"
    else:
        status = "prefix_order_contract_weak"
    if spread_mean <= 1.0e-4:
        status = "prefix_order_present_but_undifferentiated"
    return {
        "status": status,
        "prefix_dims": [int(dim) for dim in dims],
        "row_count": len(rows),
        "prefix_monotonicity_score": monotonicity,
        "prefix_order_violation_fraction": 1.0 - monotonicity,
        "prefix_spread_mean": spread_mean,
        "early_minus_full_mean": early_minus_full,
        "by_dim": by_dim,
        "by_family": by_family,
        "rows": rows[:50],
    }


def _transform_family_summary(payload: dict[str, Any]) -> dict[str, Any]:
    families: dict[str, list[dict[str, Any]]] = {}
    for case in payload.get("cases", []) or []:
        if not isinstance(case, dict):
            continue
        for transform in case.get("transforms", []) or []:
            if not isinstance(transform, dict):
                continue
            family = str(transform.get("transform_family", "unknown"))
            families.setdefault(family, []).append(transform)
    summary: dict[str, Any] = {}
    for family, rows in sorted(families.items()):
        deltas = [_as_float(row.get("delta_dense_minus_metadata_stability")) for row in rows]
        winners = [str(row.get("winner", "unknown")) for row in rows]
        summary[family] = {
            "row_count": len(rows),
            "mean_stability_delta_dense_minus_metadata": _mean(deltas),
            "std_stability_delta_dense_minus_metadata": _stddev(deltas),
            "dense_wins": sum(1 for winner in winners if winner == "dense"),
            "metadata_wins": sum(1 for winner in winners if winner == "metadata"),
            "ties": sum(1 for winner in winners if winner == "tie"),
            "dense_win_fraction": float(sum(1 for winner in winners if winner == "dense") / max(1, len(winners))),
        }
    return summary


def _blocking_failures(axis_block: dict[str, Any], prefix_block: dict[str, Any]) -> list[str]:
    axes = axis_block["axes"]
    deltas = axis_block["deltas"]
    failures: list[str] = []
    if not axes["enough_cases"]:
        failures.append("not_enough_cases_for_aggregate_separation")
    if not axes["stability_beats_metadata"]:
        failures.append("dense_does_not_beat_metadata_on_metamer_stability")
    if not axes["separation_beats_metadata"]:
        failures.append("dense_does_not_beat_metadata_on_case_separation")
    if not axes["dominant_cluster_not_worse"]:
        failures.append("dense_has_higher_dominant_cluster_share_than_metadata")
    if not axes["near_duplicate_not_worse"]:
        failures.append("dense_has_higher_near_duplicate_pair_share_than_metadata")
    if not axes["effective_rank_not_worse"]:
        failures.append("dense_effective_rank_regresses_vs_metadata")
    if not axes["packet_pairwise_distance_not_worse"]:
        failures.append("dense_packet_pairwise_distance_regresses_vs_metadata")
    if prefix_block.get("status") == "missing_prefix_metrics":
        failures.append("prefix_metrics_missing")
    elif _as_float(prefix_block.get("prefix_spread_mean")) <= 1.0e-4 and not axes["separation_beats_metadata"]:
        failures.append("prefixes_are_stable_but_undifferentiated_under_current_harness")
    if deltas["separation_delta"] < 0.0 and deltas["near_duplicate_pair_share_delta"] > 0.0:
        failures.append("dense_body_is_more_invariant_but_less_discriminative_than_explicit_metadata")
    return failures


def _recommended_next_losses(failures: list[str]) -> list[str]:
    losses = [
        "contrastive_anti_collapse_loss: separate different source families/cases in dense h while preserving metamer invariance",
        "matryoshka_prefix_distillation_loss: make h[:128]/h[:256]/... recover coarse, q, temporal, support, branch, and operator-head targets without hard semantic slots",
        "metamer_prefix_consistency_loss: strongly match early prefixes under gain/time/phase transforms while allowing calibrated tail drift",
        "operator_tail_predictive_loss: require h[640:768] or projected operator_seed to improve next-window continuation over explicit metadata baselines",
    ]
    if "dense_does_not_beat_metadata_on_case_separation" in failures:
        losses.append("case_separation_margin_loss: require pooled dense signatures from different wav families to exceed metadata pairwise distance")
    if "dense_has_higher_near_duplicate_pair_share_than_metadata" in failures:
        losses.append("near_duplicate_floor_loss: penalize dense packet pairs above the merge threshold unless they share source/support identity")
    if "prefixes_are_stable_but_undifferentiated_under_current_harness" in failures:
        losses.append("prefix_spread_calibration_loss: prevent all prefixes from becoming equally invariant by assigning prefix-specific prediction heads")
    return losses


def analyze_dense_signature_failure_modes(input_json: Path, out_dir: Path) -> dict[str, Any]:
    payload = json.loads(input_json.read_text(encoding="utf-8"))
    axis_block = _dense_required_axes(payload)
    prefix_block = _prefix_summary(payload)
    family_summary = _transform_family_summary(payload)
    failures = _blocking_failures(axis_block, prefix_block)
    dense_beats_all_axes = bool(axis_block["axes"]["all_required_axes_pass"])
    verdict = payload.get("verdict") if isinstance(payload.get("verdict"), dict) else {}
    source_dense_wins = bool(verdict.get("dense_wins"))
    claim_passes = dense_beats_all_axes and source_dense_wins

    if claim_passes:
        claim_status = "dense_body_claim_candidate"
    elif axis_block["axes"]["stability_beats_metadata"] and failures:
        claim_status = "dense_stability_only_not_token_claim"
    else:
        claim_status = "dense_body_claim_not_established"

    if prefix_block.get("status") == "missing_prefix_metrics":
        matryoshka_status = "not_measured"
    elif claim_passes and _as_float(prefix_block.get("prefix_monotonicity_score")) >= 0.95:
        matryoshka_status = "candidate_contract_present"
    elif _as_float(prefix_block.get("prefix_monotonicity_score")) >= 0.95:
        matryoshka_status = "prefix_trace_present_but_claim_axes_fail"
    else:
        matryoshka_status = "prefix_trace_needs_loss_pressure"

    summary = {
        "schema": SCHEMA,
        "created_utc": _utc_timestamp(),
        "source_path": str(input_json),
        "source_schema": payload.get("schema"),
        "source_verdict": verdict,
        "claim_status": claim_status,
        "promotion_effect": "none",
        "dense_beats_metadata_on_all_required_axes": dense_beats_all_axes,
        "blocking_failures": failures,
        "required_axes": axis_block,
        "prefix_diagnostics": prefix_block,
        "matryoshka_readiness_status": matryoshka_status,
        "operator_tail_distinctness": {
            "status": "not_measured_in_current_artifact",
            "reason": "dense_signature_claim.json contains prefix stability and aggregate collapse metrics but not raw h-tail/operator-seed separability or continuation-usefulness probes.",
            "required_next_probe": "compare h[640:768]/operator_seed against metadata on next-window continuation and low-rank law prediction.",
        },
        "transform_family_stability": family_summary,
        "recommended_next_losses": _recommended_next_losses(failures),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "dense_signature_failure_modes.json"
    md_path = out_dir / "DENSE_SIGNATURE_FAILURE_MODES.md"
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    _write_markdown(md_path, summary)
    return summary


def _write_markdown(path: Path, summary: dict[str, Any]) -> None:
    deltas = summary["required_axes"]["deltas"]
    axes = summary["required_axes"]["axes"]
    prefix = summary["prefix_diagnostics"]
    lines = [
        "# Dense Signature Failure Modes",
        "",
        f"- schema: `{summary['schema']}`",
        f"- source: `{summary['source_path']}`",
        f"- source verdict: `{summary['source_verdict'].get('status')}`",
        f"- claim status: `{summary['claim_status']}`",
        f"- promotion effect: `{summary['promotion_effect']}`",
        f"- matryoshka readiness: `{summary['matryoshka_readiness_status']}`",
        f"- operator tail distinctness: `{summary['operator_tail_distinctness']['status']}`",
        "",
        "## Required Axes",
        "",
        f"- stability delta dense-minus-metadata: `{_fmt(deltas['stability_delta'])}`",
        f"- separation delta dense-minus-metadata: `{_fmt(deltas['separation_delta'])}`",
        f"- dominant cluster share delta: `{_fmt(deltas['dominant_cluster_share_delta'])}`",
        f"- near-duplicate pair share delta: `{_fmt(deltas['near_duplicate_pair_share_delta'])}`",
        f"- effective rank delta: `{_fmt(deltas['effective_rank_delta'])}`",
        f"- packet pairwise distance delta: `{_fmt(deltas['packet_pairwise_distance_delta'])}`",
        f"- all required axes pass: `{axes['all_required_axes_pass']}`",
        "",
        "## Prefix Diagnostics",
        "",
        f"- prefix status: `{prefix.get('status')}`",
        f"- prefix monotonicity score: `{_fmt(prefix.get('prefix_monotonicity_score'))}`",
        f"- prefix spread mean: `{_fmt(prefix.get('prefix_spread_mean'))}`",
        f"- early-minus-full mean: `{_fmt(prefix.get('early_minus_full_mean'))}`",
        "",
        "## Blocking Failures",
        "",
    ]
    lines.extend([f"- `{item}`" for item in summary.get("blocking_failures", [])] or ["- none"])
    lines.extend(["", "## Transform Families", ""])
    for family, row in summary.get("transform_family_stability", {}).items():
        lines.append(
            "- `{family}`: mean stability delta `{delta}`, dense wins `{dense}` / metadata wins `{metadata}` / ties `{ties}`".format(
                family=family,
                delta=_fmt(row.get("mean_stability_delta_dense_minus_metadata")),
                dense=row.get("dense_wins"),
                metadata=row.get("metadata_wins"),
                ties=row.get("ties"),
            )
        )
    lines.extend(["", "## Recommended Next Losses", ""])
    lines.extend([f"- {item}" for item in summary.get("recommended_next_losses", [])])
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Analyze dense RAFA signature claim failures separately from explicit q/law metadata."
    )
    ap.add_argument("--input-json", default=str(DEFAULT_INPUT))
    ap.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    args = ap.parse_args()
    summary = analyze_dense_signature_failure_modes(
        input_json=Path(args.input_json),
        out_dir=Path(args.out_dir),
    )
    print(
        json.dumps(
            {
                "schema": summary["schema"],
                "saved": str(Path(args.out_dir) / "dense_signature_failure_modes.json"),
                "report": str(Path(args.out_dir) / "DENSE_SIGNATURE_FAILURE_MODES.md"),
                "claim_status": summary["claim_status"],
                "matryoshka_readiness_status": summary["matryoshka_readiness_status"],
                "blocking_failures": summary["blocking_failures"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
