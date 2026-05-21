from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SCHEMA = "rafa_learned_signature_scout_compare_v0"
TOKENBURST_ROOT = Path(r"C:\Users\Andrew\AppData\Local\Temp\rafa_tokenburst")
METRIC_EPSILON = 1.0e-4
DEFAULT_SCOUTS = [
    TOKENBURST_ROOT / "learned_signature_scout_cuda_2026_05_07" / "learned_signature_scout.json",
    TOKENBURST_ROOT / "learned_signature_scout_v2_metamer_calibrated_cuda_2026_05_07" / "learned_signature_scout.json",
    TOKENBURST_ROOT / "learned_signature_scout_v3_balanced_cuda_2026_05_07" / "learned_signature_scout.json",
    TOKENBURST_ROOT / "learned_signature_scout_v4_separation_heavy_cuda_2026_05_07" / "learned_signature_scout.json",
    TOKENBURST_ROOT / "learned_signature_scout_v5_runcontrast_cuda_2026_05_07" / "learned_signature_scout.json",
    TOKENBURST_ROOT / "learned_signature_scout_v6_runcontrast_separation_cuda_2026_05_07" / "learned_signature_scout.json",
    TOKENBURST_ROOT / "learned_signature_scout_v7_runcontrast_rank_cuda_2026_05_07" / "learned_signature_scout.json",
    TOKENBURST_ROOT / "learned_signature_scout_v8_runcontrast_runrank_cuda_2026_05_07" / "learned_signature_scout.json",
    TOKENBURST_ROOT / "learned_signature_scout_v9_runcontrast_baserank_cuda_2026_05_07" / "learned_signature_scout.json",
    TOKENBURST_ROOT / "learned_signature_scout_v10_runcontrast_basesvd_cuda_2026_05_07" / "learned_signature_scout.json",
    TOKENBURST_ROOT / "learned_signature_scout_v11_runcontrast_basesvd_soft_cuda_2026_05_07" / "learned_signature_scout.json",
    TOKENBURST_ROOT / "learned_signature_scout_v12_runcontrast_strong_basesvd_soft_cuda_2026_05_07" / "learned_signature_scout.json",
    TOKENBURST_ROOT / "learned_signature_scout_v13_runcontrast_mid_basesvd_soft_cuda_2026_05_07" / "learned_signature_scout.json",
    TOKENBURST_ROOT / "learned_signature_scout_v14_runcontrast_lowmid_basesvd_soft_cuda_2026_05_07" / "learned_signature_scout.json",
    TOKENBURST_ROOT / "learned_signature_scout_v14_seed1729_runcontrast_lowmid_basesvd_soft_cuda_2026_05_07" / "learned_signature_scout.json",
    TOKENBURST_ROOT / "learned_signature_scout_v14_seed2718_runcontrast_lowmid_basesvd_soft_cuda_2026_05_07" / "learned_signature_scout.json",
    TOKENBURST_ROOT / "learned_signature_scout_v14_seed3141_runcontrast_lowmid_basesvd_soft_cuda_2026_05_07" / "learned_signature_scout.json",
    TOKENBURST_ROOT / "learned_signature_scout_v14_seed1729_holdout6_runcontrast_lowmid_basesvd_soft_cuda_2026_05_07" / "learned_signature_scout.json",
    TOKENBURST_ROOT / "learned_signature_scout_v12_seed1729_holdout6_runcontrast_strong_basesvd_soft_cuda_2026_05_07" / "learned_signature_scout.json",
    TOKENBURST_ROOT / "learned_signature_scout_v15_seed1729_holdout6_geometry04_cuda_2026_05_07" / "learned_signature_scout.json",
    TOKENBURST_ROOT / "learned_signature_scout_v15_seed1729_holdout6_geometry20_cuda_2026_05_07" / "learned_signature_scout.json",
    TOKENBURST_ROOT / "learned_signature_scout_v15_seed1729_holdout6_strong_geometry04_cuda_2026_05_07" / "learned_signature_scout.json",
    TOKENBURST_ROOT / "learned_signature_scout_v16_seed1729_train12_holdout6_lowmid_cuda_2026_05_07" / "learned_signature_scout.json",
    TOKENBURST_ROOT / "learned_signature_scout_v16_seed1729_train12_holdout6_strong_cuda_2026_05_07" / "learned_signature_scout.json",
    TOKENBURST_ROOT / "learned_signature_scout_v17_seed1729_train12_holdout6_strong_basesvd45_cuda_2026_05_07" / "learned_signature_scout.json",
    TOKENBURST_ROOT / "learned_signature_scout_v17_seed1729_train12_holdout6_strong_basesvd45_floor35_cuda_2026_05_07" / "learned_signature_scout.json",
    TOKENBURST_ROOT / "learned_signature_scout_v18_seed1729_train12_holdout6_strong_h64_cuda_2026_05_07" / "learned_signature_scout.json",
    TOKENBURST_ROOT / "learned_signature_scout_v18_seed1729_train12_holdout6_strong_h96_cuda_2026_05_07" / "learned_signature_scout.json",
    TOKENBURST_ROOT / "learned_signature_scout_v19_seed1729_train12_holdout6_h96_run100_cuda_2026_05_07" / "learned_signature_scout.json",
    TOKENBURST_ROOT / "learned_signature_scout_v19_seed1729_train12_holdout6_h96_run085_cuda_2026_05_07" / "learned_signature_scout.json",
    TOKENBURST_ROOT / "learned_signature_scout_v20_seed1729_train12_holdout6_h128_run125_cuda_2026_05_07" / "learned_signature_scout.json",
    TOKENBURST_ROOT / "learned_signature_scout_v20_seed1729_train12_holdout6_h128_run110_cuda_2026_05_07" / "learned_signature_scout.json",
    TOKENBURST_ROOT / "learned_signature_scout_v21_seed1729_train12_holdout6_h96_run125_meta002_cuda_2026_05_07" / "learned_signature_scout.json",
    TOKENBURST_ROOT / "learned_signature_scout_v21_seed1729_train12_holdout6_h96_run125_meta005_cuda_2026_05_07" / "learned_signature_scout.json",
    TOKENBURST_ROOT / "learned_signature_scout_v22_seed1729_train12_holdout6_h96_run125_epochs48_cuda_2026_05_07" / "learned_signature_scout.json",
    TOKENBURST_ROOT / "learned_signature_scout_v23_seed1729_train12_holdout6_h96_run125_epochs12_cuda_2026_05_07" / "learned_signature_scout.json",
    TOKENBURST_ROOT / "learned_signature_scout_v23_seed1729_train12_holdout6_h96_run125_epochs18_cuda_2026_05_07" / "learned_signature_scout.json",
]
DEFAULT_OUT_DIR = TOKENBURST_ROOT / "learned_signature_scout_compare_2026_05_07"


def _default_scout_paths() -> list[Path]:
    discovered = sorted(TOKENBURST_ROOT.glob("learned_signature_scout*/learned_signature_scout.json"))
    paths: list[Path] = []
    seen: set[str] = set()
    for path in [*DEFAULT_SCOUTS, *discovered]:
        if "smoke" in path.parent.name.lower():
            continue
        key = str(path.resolve()) if path.exists() else str(path)
        if key in seen:
            continue
        seen.add(key)
        paths.append(path)
    return paths


def _utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return float(default)
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _row(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    aggregate = payload.get("aggregate") if isinstance(payload.get("aggregate"), dict) else {}
    stability = aggregate.get("stability") if isinstance(aggregate.get("stability"), dict) else {}
    separation = aggregate.get("separation") if isinstance(aggregate.get("separation"), dict) else {}
    collapse = aggregate.get("collapse") if isinstance(aggregate.get("collapse"), dict) else {}
    collapse_delta = (
        collapse.get("delta_learned_minus_metadata")
        if isinstance(collapse.get("delta_learned_minus_metadata"), dict)
        else {}
    )
    learned_collapse = collapse.get("learned") if isinstance(collapse.get("learned"), dict) else {}
    operator_tail = aggregate.get("operator_tail") if isinstance(aggregate.get("operator_tail"), dict) else {}
    train = payload.get("train_summary") if isinstance(payload.get("train_summary"), dict) else {}
    verdict = payload.get("verdict") if isinstance(payload.get("verdict"), dict) else {}
    stability_delta = _as_float(stability.get("delta_learned_minus_metadata"))
    separation_delta = _as_float(separation.get("delta_learned_minus_metadata"))
    dominant_delta = _as_float(collapse_delta.get("dominant_cluster_share"))
    duplicate_delta = _as_float(collapse_delta.get("near_duplicate_pair_share"))
    rank_delta = _as_float(collapse_delta.get("effective_rank"))
    distance_delta = _as_float(collapse_delta.get("mean_pairwise_distance"))
    if stability_delta > METRIC_EPSILON and separation_delta <= METRIC_EPSILON:
        profile_class = "metamer_stable_but_collapsed"
    elif stability_delta <= METRIC_EPSILON and separation_delta > METRIC_EPSILON:
        profile_class = "separating_but_unstable"
    elif stability_delta > METRIC_EPSILON and separation_delta > METRIC_EPSILON:
        profile_class = "candidate_tradeoff"
    else:
        profile_class = "dual_regression"
    if dominant_delta > 0.02 or duplicate_delta > 0.02 or rank_delta < -0.05 or distance_delta < -0.02:
        profile_class += "_anti_collapse_failed"
    elif stability_delta > METRIC_EPSILON and separation_delta <= METRIC_EPSILON:
        profile_class = "stable_rank_preserved_but_not_separating"
    return {
        "path": str(path),
        "profile": path.parent.name,
        "status": verdict.get("status"),
        "promotion_effect": verdict.get("promotion_effect"),
        "split_mode": payload.get("split_mode"),
        "num_cases": payload.get("num_cases"),
        "num_transforms": payload.get("num_transforms"),
        "train_packet_count": train.get("train_packet_count"),
        "metamer_consistency_weight": train.get("metamer_consistency_weight"),
        "case_separation_weight": train.get("case_separation_weight"),
        "run_contrastive_weight": train.get("run_contrastive_weight"),
        "run_contrastive_temperature": train.get("run_contrastive_temperature"),
        "separation_margin": train.get("separation_margin"),
        "factor_geometry_weight": train.get("factor_geometry_weight"),
        "final_factor_geometry_cosine": train.get("final_factor_geometry_cosine"),
        "seed": train.get("seed"),
        "case_offset": payload.get("case_offset"),
        "case_shuffle_seed": payload.get("case_shuffle_seed"),
        "train_loss_reduction": train.get("loss_reduction"),
        "final_run_positive_cosine": train.get("final_run_positive_cosine"),
        "final_run_negative_cosine": train.get("final_run_negative_cosine"),
        "stability_delta_learned_minus_metadata": stability_delta,
        "separation_delta_learned_minus_metadata": separation_delta,
        "collapse_dominant_cluster_delta": dominant_delta,
        "collapse_near_duplicate_delta": duplicate_delta,
        "collapse_effective_rank_delta": rank_delta,
        "collapse_mean_pairwise_distance_delta": distance_delta,
        "learned_dominant_cluster_share": learned_collapse.get("dominant_cluster_share"),
        "learned_near_duplicate_pair_share": learned_collapse.get("near_duplicate_pair_share"),
        "operator_tail_effective_rank": operator_tail.get("effective_rank"),
        "operator_tail_effective_cluster_count": operator_tail.get("effective_cluster_count"),
        "profile_class": profile_class,
        "reasons": verdict.get("reasons", []),
    }


def _dominates(left: dict[str, Any], right: dict[str, Any]) -> bool:
    left_pair = (
        _as_float(left.get("stability_delta_learned_minus_metadata")),
        _as_float(left.get("separation_delta_learned_minus_metadata")),
    )
    right_pair = (
        _as_float(right.get("stability_delta_learned_minus_metadata")),
        _as_float(right.get("separation_delta_learned_minus_metadata")),
    )
    return left_pair[0] >= right_pair[0] and left_pair[1] >= right_pair[1] and left_pair != right_pair


def _seed_family(profile: Any) -> str:
    return re.sub(r"_seed\d+", "_seed*", str(profile or ""))


def _seed_stability(rows: list[dict[str, Any]], candidate_rows: list[dict[str, Any]]) -> dict[str, Any]:
    candidate_ids = {row.get("path") for row in candidate_rows}
    by_family: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        if row.get("seed") is None:
            continue
        by_family.setdefault(_seed_family(row.get("profile")), []).append(row)
    families = []
    for family, family_rows in sorted(by_family.items()):
        candidates = [row for row in family_rows if row.get("path") in candidate_ids]
        failures = [row for row in family_rows if row.get("path") not in candidate_ids]
        families.append(
            {
                "family": family,
                "seed_count": len(family_rows),
                "candidate_seed_count": len(candidates),
                "candidate_seed_fraction": (len(candidates) / len(family_rows)) if family_rows else 0.0,
                "candidate_seeds": [row.get("seed") for row in candidates],
                "failed_seeds": [row.get("seed") for row in failures],
            }
        )
    return {
        "families": families,
        "seeded_family_count": len(families),
        "best_seeded_family": max(families, key=lambda row: row.get("candidate_seed_fraction", 0.0), default={}),
    }


def compare_learned_signature_scouts(paths: list[Path], out_dir: Path) -> dict[str, Any]:
    rows = [_row(path) for path in paths if path.exists()]
    pareto = [
        row
        for row in rows
        if not any(_dominates(other, row) for other in rows if other is not row)
    ]
    best_separation = max(rows, key=lambda row: _as_float(row.get("separation_delta_learned_minus_metadata")), default={})
    best_stability = max(rows, key=lambda row: _as_float(row.get("stability_delta_learned_minus_metadata")), default={})
    candidate_rows = [
        row
        for row in rows
        if _as_float(row.get("stability_delta_learned_minus_metadata")) > METRIC_EPSILON
        and _as_float(row.get("separation_delta_learned_minus_metadata")) > METRIC_EPSILON
        and _as_float(row.get("collapse_dominant_cluster_delta")) <= 0.02
        and _as_float(row.get("collapse_near_duplicate_delta")) <= 0.02
        and _as_float(row.get("collapse_effective_rank_delta")) >= -0.05
        and _as_float(row.get("collapse_mean_pairwise_distance_delta")) >= -0.02
    ]
    two_axis_rows = [
        row
        for row in rows
        if _as_float(row.get("stability_delta_learned_minus_metadata")) > METRIC_EPSILON
        and _as_float(row.get("separation_delta_learned_minus_metadata")) > METRIC_EPSILON
    ]
    stability_anticollapse_rows = [
        row
        for row in rows
        if _as_float(row.get("stability_delta_learned_minus_metadata")) > METRIC_EPSILON
        and _as_float(row.get("collapse_dominant_cluster_delta")) <= 0.02
        and _as_float(row.get("collapse_near_duplicate_delta")) <= 0.02
        and _as_float(row.get("collapse_effective_rank_delta")) >= -0.05
        and _as_float(row.get("collapse_mean_pairwise_distance_delta")) >= -0.02
    ]
    heldout_rows = [row for row in rows if str(row.get("split_mode", "")).startswith("heldout")]
    heldout_candidate_rows = [row for row in candidate_rows if str(row.get("split_mode", "")).startswith("heldout")]
    heldout_two_axis_rows = [row for row in two_axis_rows if str(row.get("split_mode", "")).startswith("heldout")]
    heldout_stability_anticollapse_rows = [
        row for row in stability_anticollapse_rows if str(row.get("split_mode", "")).startswith("heldout")
    ]
    status = "no_learned_signature_candidate"
    if heldout_candidate_rows:
        status = "heldout_candidate_found_needs_seed"
    elif heldout_rows and candidate_rows:
        status = "in_sample_candidate_holdout_blocked"
    elif candidate_rows:
        status = "candidate_tradeoff_found_needs_holdout"
    elif two_axis_rows and stability_anticollapse_rows:
        status = "complementary_near_candidates_mapped"
    elif two_axis_rows:
        status = "two_axis_near_candidate_rank_collapse_blocked"
    elif rows:
        status = "tradeoff_bracket_mapped"
    seed_stability = _seed_stability(rows, candidate_rows)
    heldout_seed_stability = _seed_stability(heldout_rows, heldout_candidate_rows)
    if heldout_candidate_rows:
        interpretation = (
            "A held-out learned signature candidate is now present. This does not prove the RAFA-token claim yet; "
            "it upgrades the lane to seed-replication and downstream predictive-usefulness checks."
        )
    elif candidate_rows:
        interpretation = (
            "Learned signatures have in-sample scout candidates when run-center contrastive pressure is balanced "
            "against base-SVD rank preservation. Held-out rows still block promotion."
        )
    else:
        interpretation = (
            "Learned signatures show a real separation/tail signal, but the current scout losses trade it against "
            "metamer stability. No row satisfies stability, separation, and anti-collapse together."
        )
    summary = {
        "schema": SCHEMA,
        "created_utc": _utc_timestamp(),
        "status": status,
        "promotion_effect": "none",
        "row_count": len(rows),
        "candidate_count": len(candidate_rows),
        "two_axis_candidate_count": len(two_axis_rows),
        "stability_anticollapse_candidate_count": len(stability_anticollapse_rows),
        "heldout_row_count": len(heldout_rows),
        "heldout_candidate_count": len(heldout_candidate_rows),
        "heldout_two_axis_candidate_count": len(heldout_two_axis_rows),
        "heldout_stability_anticollapse_candidate_count": len(heldout_stability_anticollapse_rows),
        "pareto_count": len(pareto),
        "best_separation_profile": best_separation.get("profile"),
        "best_stability_profile": best_stability.get("profile"),
        "best_separation_delta": best_separation.get("separation_delta_learned_minus_metadata"),
        "best_stability_delta": best_stability.get("stability_delta_learned_minus_metadata"),
        "seed_stability": seed_stability,
        "heldout_seed_stability": heldout_seed_stability,
        "interpretation": interpretation,
        "pareto_rows": pareto,
        "candidate_rows": candidate_rows,
        "heldout_candidate_rows": heldout_candidate_rows,
        "two_axis_rows": two_axis_rows,
        "stability_anticollapse_rows": stability_anticollapse_rows,
        "rows": rows,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "learned_signature_scout_compare.json"
    md_path = out_dir / "LEARNED_SIGNATURE_SCOUT_COMPARE.md"
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    _write_markdown(md_path, summary)
    return summary


def _fmt(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def _write_markdown(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# Learned Signature Scout Compare",
        "",
        f"- schema: `{summary['schema']}`",
        f"- status: `{summary['status']}`",
        f"- promotion effect: `{summary['promotion_effect']}`",
        f"- rows: `{summary['row_count']}`",
        f"- candidates: `{summary['candidate_count']}`",
        f"- two-axis candidates: `{summary['two_axis_candidate_count']}`",
        f"- stability+anti-collapse candidates: `{summary['stability_anticollapse_candidate_count']}`",
        f"- heldout rows: `{summary['heldout_row_count']}`",
        f"- heldout candidates: `{summary['heldout_candidate_count']}`",
        f"- heldout two-axis candidates: `{summary['heldout_two_axis_candidate_count']}`",
        f"- heldout stability+anti-collapse candidates: `{summary['heldout_stability_anticollapse_candidate_count']}`",
        f"- pareto rows: `{summary['pareto_count']}`",
        f"- best separation profile: `{summary['best_separation_profile']}`",
        f"- best stability profile: `{summary['best_stability_profile']}`",
        f"- best seeded family: `{summary.get('seed_stability', {}).get('best_seeded_family', {}).get('family')}`",
        f"- best seeded candidate fraction: `{_fmt(summary.get('seed_stability', {}).get('best_seeded_family', {}).get('candidate_seed_fraction'))}`",
        f"- best heldout seeded family: `{summary.get('heldout_seed_stability', {}).get('best_seeded_family', {}).get('family')}`",
        f"- best heldout seeded candidate fraction: `{_fmt(summary.get('heldout_seed_stability', {}).get('best_seeded_family', {}).get('candidate_seed_fraction'))}`",
        "",
        summary["interpretation"],
        "",
        "| profile | class | stability d | separation d | dom d | dup d | rank d | dist d | op clusters | run w |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary.get("rows", []):
        lines.append(
            "| {profile} | {klass} | {stable} | {sep} | {dom} | {dup} | {rank} | {dist} | {op} | {runw} |".format(
                profile=row.get("profile"),
                klass=row.get("profile_class"),
                stable=_fmt(row.get("stability_delta_learned_minus_metadata")),
                sep=_fmt(row.get("separation_delta_learned_minus_metadata")),
                dom=_fmt(row.get("collapse_dominant_cluster_delta")),
                dup=_fmt(row.get("collapse_near_duplicate_delta")),
                rank=_fmt(row.get("collapse_effective_rank_delta")),
                dist=_fmt(row.get("collapse_mean_pairwise_distance_delta")),
                op=_fmt(row.get("operator_tail_effective_cluster_count")),
                runw=_fmt(row.get("run_contrastive_weight")),
            )
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description="Compare learned RAFA signature scout profiles.")
    ap.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    ap.add_argument("--scout-json", action="append", default=[])
    args = ap.parse_args()
    paths = [Path(item) for item in args.scout_json] if args.scout_json else _default_scout_paths()
    summary = compare_learned_signature_scouts(paths, Path(args.out_dir))
    print(
        json.dumps(
            {
                "schema": summary["schema"],
                "saved": str(Path(args.out_dir) / "learned_signature_scout_compare.json"),
                "report": str(Path(args.out_dir) / "LEARNED_SIGNATURE_SCOUT_COMPARE.md"),
                "status": summary["status"],
                "candidate_count": summary["candidate_count"],
                "pareto_count": summary["pareto_count"],
                "best_separation_profile": summary["best_separation_profile"],
                "best_stability_profile": summary["best_stability_profile"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
