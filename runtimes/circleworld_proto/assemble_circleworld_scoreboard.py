import argparse
import json
from collections import OrderedDict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(r"D:\RAFA")
OUTPUTS = ROOT / "outputs" / "circleworld_proto"
DEFAULT_OUTDIR = OUTPUTS / "tokenburst_2026-05-04_w10_scoreboard"


def _load_json(path: Path | None) -> dict[str, Any] | None:
    if path is None or not path.exists():
        return None
    with path.open("r", encoding="utf-8-sig") as fh:
        return json.load(fh)


def _first_number(data: dict[str, Any] | None, *keys: str) -> float | None:
    if not data:
        return None
    for key in keys:
        value = data.get(key)
        if isinstance(value, (int, float)):
            return float(value)
    return None


def _round_or_none(value: float | None, digits: int = 6) -> float | None:
    if value is None:
        return None
    return round(value, digits)


def _nested_lookup(track_summary: dict[str, Any] | None, checkpoint_name: str) -> dict[str, Any] | None:
    if not track_summary:
        return None
    for row in track_summary.get("checkpoints", []):
        if row.get("name") == checkpoint_name:
            return row
    return None


def _metamer_lookup(track_summary: dict[str, Any] | None, run_name: str) -> dict[str, Any] | None:
    if not track_summary:
        return None
    runs = track_summary.get("runs", {})
    if isinstance(runs, dict):
        return runs.get(run_name)
    return None


def _library_stats(path: Path | None) -> dict[str, float | int | None]:
    payload = _load_json(path)
    if not payload:
        return {
            "aggregate_families": None,
            "mean_confidence": None,
            "mean_branch_mass": None,
        }
    aggregate_families = None
    mean_confidence = None
    mean_branch_mass = None
    for key in (
        "num_families",
        "aggregate_num_signature_families",
        "aggregate_num_relational_signature_families",
        "aggregate_num_families",
    ):
        value = payload.get(key)
        if isinstance(value, (int, float)):
            aggregate_families = int(value)
            break
    for key in (
        "mean_relational_signature_confidence",
        "aggregate_mean_relational_signature_confidence",
    ):
        value = payload.get(key)
        if isinstance(value, (int, float)):
            mean_confidence = float(value)
            break
    for key in (
        "mean_relational_branch_mass",
        "aggregate_mean_relational_branch_mass",
    ):
        value = payload.get(key)
        if isinstance(value, (int, float)):
            mean_branch_mass = float(value)
            break
    return {
        "aggregate_families": aggregate_families,
        "mean_confidence": mean_confidence,
        "mean_branch_mass": mean_branch_mass,
    }


def _collect_checkpoint_rows() -> list[dict[str, Any]]:
    w4_summary = _load_json(OUTPUTS / "tokenburst_2026-05-04_w4_signature_analysis" / "track_summary.json")
    w5_summary = _load_json(OUTPUTS / "tokenburst_2026-05-04_w5_nested_ontology" / "track_summary.json")
    active_library = _library_stats(
        OUTPUTS
        / "relational_signature_full_2026-05-04"
        / "relational_signature_library.json"
    )
    baseline_library = _library_stats(
        OUTPUTS
        / "relational_signature_full_parentmix_2026-05-04"
        / "relational_signature_library.json"
    )
    relsig_library = _library_stats(
        OUTPUTS
        / "relsig_scout_balance_2026-05-04"
        / "law_token_library"
        / "relational_signature_library.json"
    )

    specs = [
        {
            "name": "agreement_scout_v1",
            "heldout_path": OUTPUTS / "relational_eval_active_2026-05-04" / "heldout_summary.json",
            "benchmark_path": OUTPUTS / "training_run_2026-04-24_agreement_scout_v1" / "benchmark_expanded" / "benchmark_summary.json",
            "continuity_path": OUTPUTS / "training_run_2026-04-24_agreement_scout_v1" / "benchmark_expanded" / "continuity_circleworld.json",
            "nested_name": "agreement_scout_v1",
            "metamer_name": "agreement_scout_v1",
            "library_stats": active_library,
        },
        {
            "name": "parentmix_220_100",
            "heldout_path": OUTPUTS / "parentmix_sweep_2026-04-24_v1" / "parentmix_220_100" / "heldout_eval" / "heldout_summary.json",
            "benchmark_path": OUTPUTS / "parentmix_sweep_2026-04-24_v1" / "parentmix_220_100" / "benchmark" / "benchmark_summary.json",
            "continuity_path": OUTPUTS / "parentmix_sweep_2026-04-24_v1" / "parentmix_220_100" / "benchmark" / "continuity_circleworld.json",
            "nested_name": "parentmix_220_100",
            "metamer_name": "parentmix_220_100",
            "library_stats": baseline_library,
        },
        {
            "name": "relsig_balance_guard_v1",
            "heldout_path": OUTPUTS / "relsig_scout_balance_2026-05-04" / "heldout_eval" / "heldout_summary.json",
            "benchmark_path": OUTPUTS / "relsig_scout_balance_2026-05-04" / "benchmark_expanded" / "benchmark_summary.json",
            "continuity_path": OUTPUTS / "relsig_scout_balance_2026-05-04" / "benchmark_expanded" / "continuity_circleworld.json",
            "nested_name": "relsig_balance_guard_v1",
            "metamer_name": "relsig_balance",
            "library_stats": relsig_library,
        },
    ]

    rows: list[dict[str, Any]] = []
    for spec in specs:
        heldout = _load_json(spec["heldout_path"])
        benchmark = _load_json(spec["benchmark_path"])
        continuity = _load_json(spec["continuity_path"])
        nested = _nested_lookup(w5_summary, spec["nested_name"])
        metamer = _metamer_lookup(w4_summary, spec["metamer_name"])
        row = OrderedDict(
            name=spec["name"],
            row_type="checkpoint",
            benchmark_corr=_round_or_none(_first_number(benchmark, "mean_corr")),
            benchmark_mae=_round_or_none(_first_number(benchmark, "mean_mae")),
            heldout_branch_fraction=_round_or_none(_first_number(heldout, "mean_real_branch_fraction")),
            heldout_naked_branch_fraction=_round_or_none(
                _first_number((heldout or {}).get("by_source", {}).get("naked_rafa"), "mean_real_branch_fraction")
            ),
            heldout_child_writeback=_round_or_none(_first_number(heldout, "mean_child_writeback_mass")),
            heldout_parent_divergence=_round_or_none(_first_number(heldout, "mean_child_parent_divergence")),
            heldout_signature_families=_round_or_none(_first_number(heldout, "mean_num_relational_signature_families")),
            heldout_signature_confidence=_round_or_none(_first_number(heldout, "mean_relational_signature_confidence")),
            continuity_repeat=_round_or_none(_first_number(continuity, "mean_nonlocal_chunk_repeat")),
            continuity_reentry=_round_or_none(_first_number(continuity, "mean_first_chunk_reentry")),
            continuity_adjacent=_round_or_none(_first_number(continuity, "mean_adjacent_chunk_similarity")),
            nested_sibling_fraction=_round_or_none(_first_number(nested, "mean_nested_sibling_fraction")),
            nested_qualified_carry=_round_or_none(_first_number(nested, "mean_branch_identity_qualified_carry")),
            nested_evidence_status=(nested or {}).get("evidence_status"),
            metamer_full_cos=_round_or_none(_first_number(metamer, "mean_pooled_full_cos")),
            metamer_prefix_cos=_round_or_none(_first_number(metamer, "mean_prefix_cos")),
            metamer_confidence_drift=_round_or_none(_first_number(metamer, "mean_confidence_delta")),
            export_signature_families=spec["library_stats"]["aggregate_families"],
            export_signature_confidence=_round_or_none(spec["library_stats"]["mean_confidence"]),
            export_branch_mass=_round_or_none(spec["library_stats"]["mean_branch_mass"]),
        )
        rows.append(row)
    return rows


def _summarize_track_payload(summary: dict[str, Any]) -> dict[str, Any]:
    track = summary.get("track", "unknown")
    row = OrderedDict(name=track, row_type="track")
    row["status"] = summary.get("status", "available")
    row["top_level_keys"] = len(summary.keys())
    row["notes"] = []

    if track == "w1_branch_profiles":
        row["profiles_added"] = 3
        row["source_file"] = summary.get("source_file")
        row["notes"].append("smoke validated profile registration and config preparation")
    elif track == "w2_branch_trainer":
        row["source_file"] = summary.get("source_file")
        row["new_score_knobs"] = len(summary.get("new_score_cfg_knobs", []))
        row["new_probe_knobs"] = len(summary.get("new_heldout_probe_cfg_knobs", []))
        row["notes"].append("adds optional score/probe JSON overlays and branch-first floor logic")
    elif track == "w3_signature_profiles":
        row["profiles_added"] = len(summary.get("profiles", [])) if isinstance(summary.get("profiles"), list) else 3
        row["reference_profile"] = summary.get("reference_profile")
        row["notes"].append("adds export anti-collapse signature search profiles")
    elif track == "w4_signature_analysis":
        leaderboard = summary.get("leaderboard", {})
        row["metamer_leader"] = leaderboard.get("highest_mean_pooled_full_cos")
        row["low_drift_leader"] = leaderboard.get("lowest_abs_confidence_drift")
        row["transform_count"] = len(summary.get("transform_suite", []))
        row["notes"].append("refreshes 10-transform smoke metamer lane and pairwise verdicts")
    elif track == "w5_nested_ontology":
        checkpoints = summary.get("checkpoints", [])
        row["checkpoint_count"] = len(checkpoints)
        row["agreement_nested_fraction"] = _round_or_none(
            _first_number(_nested_lookup(summary, "agreement_scout_v1"), "mean_nested_sibling_fraction")
        )
        row["parentmix_nested_fraction"] = _round_or_none(
            _first_number(_nested_lookup(summary, "parentmix_220_100"), "mean_nested_sibling_fraction")
        )
        row["notes"].append("strict carry preserves agreement_scout_v1 and invalidates parentmix nested evidence")
    elif track == "w6_benchmark_continuity":
        active = ((summary.get("sources") or {}).get("active") or {})
        baseline = ((summary.get("sources") or {}).get("baseline") or {})
        active_headline = active.get("benchmark_headline", {})
        baseline_headline = baseline.get("benchmark_headline", {})
        row["active_benchmark_corr"] = _round_or_none(_first_number(active_headline, "mean_corr"))
        row["baseline_benchmark_corr"] = _round_or_none(_first_number(baseline_headline, "mean_corr"))
        row["active_recurrence_band"] = (((active.get("circleworld") or {}).get("recurrence_severity_band")))
        row["baseline_recurrence_band"] = (((baseline.get("circleworld") or {}).get("recurrence_severity_band")))
        row["notes"].append("adds multi-window continuity summaries and severity bands")
    elif track == "w7_seed_determinism":
        seed_paths = summary.get("seed_paths", {})
        branch_seed = seed_paths.get("branch_first_active", {})
        row["plan_sha256"] = branch_seed.get("plan_sha256")
        row["notes"].append("adds explicit seed metadata and a reproducibility audit")
    elif track == "w8_library_export":
        row["aggregate_relational_effective_family_count"] = summary.get("aggregate_relational_effective_family_count")
        row["aggregate_relational_dominant_family_pressure"] = summary.get("aggregate_relational_dominant_family_pressure")
        row["notes"].append("adds export-side family pressure, entropy, and divergence diagnostics")
    elif track == "w9_semantic_projector":
        contract = summary.get("contract", {})
        row["control_count"] = len(contract.get("control_names", []))
        row["prompt_family_count"] = len(contract.get("allowed_prompt_families", []))
        row["notes"].append("narrows the semantic steering contract to the 5-control surface")
    else:
        row["notes"].append("generic tokenburst track summary")
    return row


def _collect_track_rows() -> list[dict[str, Any]]:
    track_dirs = [
        OUTPUTS / "tokenburst_2026-05-04_w1_branch_profiles",
        OUTPUTS / "tokenburst_2026-05-04_w2_branch_trainer",
        OUTPUTS / "tokenburst_2026-05-04_w3_signature_profiles",
        OUTPUTS / "tokenburst_2026-05-04_w4_signature_analysis",
        OUTPUTS / "tokenburst_2026-05-04_w5_nested_ontology",
        OUTPUTS / "tokenburst_2026-05-04_w6_benchmark_continuity",
        OUTPUTS / "tokenburst_2026-05-04_w7_seed_determinism",
        OUTPUTS / "tokenburst_2026-05-04_w8_library_export",
        OUTPUTS / "tokenburst_2026-05-04_w9_semantic_projector",
    ]
    rows: list[dict[str, Any]] = []
    for track_dir in track_dirs:
        payload = _load_json(track_dir / "track_summary.json")
        if payload:
            payload.setdefault("track", track_dir.name.replace("tokenburst_2026-05-04_", "", 1))
            rows.append(_summarize_track_payload(payload))
    return rows


def _make_markdown(checkpoint_rows: list[dict[str, Any]], track_rows: list[dict[str, Any]]) -> str:
    lines = [
        "# Circleworld Tokenburst Scoreboard",
        "",
        f"- generated_utc: `{datetime.now(timezone.utc).isoformat()}`",
        "",
        "## Checkpoint Matrix",
        "",
        "| name | bench corr | bench mae | heldout branch | naked branch | nested frac | qual carry | sig fams | sig conf | metamer cos | export fams | nested status |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in checkpoint_rows:
        lines.append(
            "| {name} | {benchmark_corr} | {benchmark_mae} | {heldout_branch_fraction} | {heldout_naked_branch_fraction} | {nested_sibling_fraction} | {nested_qualified_carry} | {heldout_signature_families} | {heldout_signature_confidence} | {metamer_full_cos} | {export_signature_families} | {nested_evidence_status} |".format(
                **{k: ("" if v is None else v) for k, v in row.items()}
            )
        )
    lines.extend(
        [
            "",
            "## Track Matrix",
            "",
            "| track | status | notes |",
            "|---|---|---|",
        ]
    )
    for row in track_rows:
        notes = row.get("notes", [])
        if isinstance(notes, list):
            notes_text = "; ".join(str(x) for x in notes)
        else:
            notes_text = str(notes)
        lines.append(f"| {row.get('name','')} | {row.get('status','')} | {notes_text} |")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Assemble a Circleworld tokenburst scoreboard from existing artifacts.")
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTDIR),
        help="Directory where scoreboard artifacts will be written.",
    )
    args = parser.parse_args()

    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    checkpoint_rows = _collect_checkpoint_rows()
    track_rows = _collect_track_rows()

    summary = OrderedDict(
        track="w10_scoreboard",
        cycle="tokenburst_2026-05-04",
        generated_utc=datetime.now(timezone.utc).isoformat(),
        scope="circleworld_only",
        checkpoint_matrix=checkpoint_rows,
        track_matrix=track_rows,
        inputs=OrderedDict(
            w4_summary=str(OUTPUTS / "tokenburst_2026-05-04_w4_signature_analysis" / "track_summary.json"),
            w5_summary=str(OUTPUTS / "tokenburst_2026-05-04_w5_nested_ontology" / "track_summary.json"),
            w6_summary=str(OUTPUTS / "tokenburst_2026-05-04_w6_benchmark_continuity" / "track_summary.json"),
            active_heldout=str(OUTPUTS / "relational_eval_active_2026-05-04" / "heldout_summary.json"),
            parentmix_heldout=str(OUTPUTS / "parentmix_sweep_2026-04-24_v1" / "parentmix_220_100" / "heldout_eval" / "heldout_summary.json"),
            relsig_heldout=str(OUTPUTS / "relsig_scout_balance_2026-05-04" / "heldout_eval" / "heldout_summary.json"),
        ),
    )

    compare = OrderedDict(
        branch_reference="agreement_scout_v1",
        diversity_reference="parentmix_220_100",
        signature_reference="relsig_balance_guard_v1",
        notes=[
            "Checkpoint rows combine heldout, benchmark, continuity, nested, metamer, and export-library evidence when available.",
            "Track rows summarize tokenburst worker outputs rather than re-scoring them as checkpoints.",
        ],
    )

    (outdir / "track_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (outdir / "track_compare.json").write_text(json.dumps(compare, indent=2), encoding="utf-8")
    (outdir / "TRACK_REPORT.md").write_text(_make_markdown(checkpoint_rows, track_rows), encoding="utf-8")


if __name__ == "__main__":
    main()
