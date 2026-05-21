from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
RUNTIME = ROOT / "runtimes" / "circleworld_proto"
for path in (ROOT, RUNTIME):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from benchmark_audio_continuity import benchmark_folder
from benchmark_circleworld_real_anchor import _load_cases_json, run_benchmark
from build_law_token_library import build_library
from evaluate_circleworld import evaluate_config
from run_childworld_experiment import _run_nested_commitment


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _load_config_payload(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    return payload if "config" in payload else {"config": payload}


def _candidate_metrics(row: dict[str, Any]) -> dict[str, float]:
    transfer = row.get("transfer_probe", {})
    val = row.get("val", {})
    return {
        "naked_branch": float(transfer.get("naked_branch", 0.0) or 0.0),
        "naked_writeback": float(transfer.get("naked_writeback", 0.0) or 0.0),
        "naked_parent_div": float(transfer.get("naked_parent_div", 0.0) or 0.0),
        "naked_live_child": float(transfer.get("naked_live_child", 0.0) or 0.0),
        "naked_law_families": float(transfer.get("naked_law_families", 0.0) or 0.0),
        "naked_law_entropy": float(transfer.get("naked_law_entropy", 0.0) or 0.0),
        "synthetic_law_families": float(transfer.get("synth_law_families", 0.0) or 0.0),
        "synthetic_law_entropy": float(transfer.get("synth_law_entropy", 0.0) or 0.0),
        "mean_corr": float(val.get("mean_corr", 0.0) or 0.0),
        "mean_mae": float(val.get("mean_mae", 999.0) or 999.0),
        "mean_meso_branch_effect": float(val.get("mean_meso_branch_effect", 0.0) or 0.0),
        "mean_num_law_families": float(val.get("aggregate_num_law_families", 0.0) or 0.0),
        "mean_law_entropy": float(val.get("aggregate_law_family_entropy", 0.0) or 0.0),
    }


def _selection_profiles(args: argparse.Namespace) -> list[tuple[str, dict[str, float]]]:
    return [
        (
            "branchfloor_audio",
            {
                "min_naked_branch": float(args.min_naked_branch),
                "min_naked_writeback": float(args.min_naked_writeback),
                "min_naked_parent_div": 0.0,
                "min_naked_law_families": 0.0,
                "min_naked_law_entropy": 0.0,
            },
        ),
        (
            "branchfloor_law",
            {
                "min_naked_branch": float(args.min_naked_branch),
                "min_naked_writeback": 0.0,
                "min_naked_parent_div": 0.0,
                "min_naked_law_families": float(args.min_naked_law_families),
                "min_naked_law_entropy": 0.0,
            },
        ),
        (
            "lawfirst_relaxed",
            {
                "min_naked_branch": float(args.relaxed_naked_branch),
                "min_naked_writeback": float(args.relaxed_naked_writeback),
                "min_naked_parent_div": float(args.relaxed_naked_parent_div),
                "min_naked_law_families": float(args.min_naked_law_families),
                "min_naked_law_entropy": float(args.min_naked_law_entropy),
            },
        ),
    ]


def _meets(metrics: dict[str, float], constraints: dict[str, float]) -> bool:
    return (
        metrics["naked_branch"] >= constraints["min_naked_branch"]
        and metrics["naked_writeback"] >= constraints["min_naked_writeback"]
        and metrics["naked_parent_div"] >= constraints["min_naked_parent_div"]
        and metrics["naked_law_families"] >= constraints["min_naked_law_families"]
        and metrics["naked_law_entropy"] >= constraints["min_naked_law_entropy"]
    )


def _profile_sort_key(profile_name: str, metrics: dict[str, float]) -> tuple[float, ...]:
    if profile_name == "branchfloor_audio":
        return (
            metrics["mean_corr"],
            -metrics["mean_mae"],
            metrics["naked_branch"],
            metrics["naked_writeback"],
            metrics["naked_law_families"],
            metrics["naked_law_entropy"],
        )
    if profile_name == "branchfloor_law":
        return (
            metrics["naked_law_families"],
            metrics["naked_law_entropy"],
            metrics["naked_branch"],
            metrics["mean_corr"],
            -metrics["mean_mae"],
        )
    return (
        metrics["naked_law_families"],
        metrics["naked_law_entropy"],
        metrics["naked_parent_div"],
        metrics["naked_writeback"],
        metrics["mean_corr"],
        -metrics["mean_mae"],
    )


def _candidate_slug(profile_name: str, row_index: int, iteration: int | None) -> str:
    iter_label = "na" if iteration is None else str(iteration)
    return f"{profile_name}_idx{row_index:02d}_iter{iter_label}"


def _select_candidates(rows: list[dict[str, Any]], args: argparse.Namespace) -> list[dict[str, Any]]:
    picks: list[dict[str, Any]] = []
    seen_indices: set[int] = set()
    for profile_name, constraints in _selection_profiles(args):
        eligible: list[tuple[dict[str, Any], dict[str, float], int]] = []
        for idx, row in enumerate(rows):
            metrics = _candidate_metrics(row)
            if _meets(metrics, constraints):
                eligible.append((row, metrics, idx))
        if not eligible:
            continue
        eligible.sort(key=lambda item: _profile_sort_key(profile_name, item[1]), reverse=True)
        for row, metrics, idx in eligible:
            if idx in seen_indices:
                continue
            picks.append(
                {
                    "profile_name": profile_name,
                    "row_index": idx,
                    "iteration": row.get("iteration"),
                    "constraints": constraints,
                    "metrics": metrics,
                    "row": row,
                }
            )
            seen_indices.add(idx)
            break

    if len(picks) >= int(args.max_candidates):
        return picks[: int(args.max_candidates)]

    eligible_all: list[tuple[dict[str, Any], dict[str, float], int]] = []
    for idx, row in enumerate(rows):
        metrics = _candidate_metrics(row)
        if metrics["naked_branch"] >= float(args.min_naked_branch):
            eligible_all.append((row, metrics, idx))
    eligible_all.sort(
        key=lambda item: (
            item[1]["naked_law_families"],
            item[1]["naked_law_entropy"],
            item[1]["mean_corr"],
            -item[1]["mean_mae"],
        ),
        reverse=True,
    )
    for row, metrics, idx in eligible_all:
        if idx in seen_indices:
            continue
        picks.append(
            {
                "profile_name": "fill_branchfloor",
                "row_index": idx,
                "iteration": row.get("iteration"),
                "constraints": {"min_naked_branch": float(args.min_naked_branch)},
                "metrics": metrics,
                "row": row,
            }
        )
        seen_indices.add(idx)
        if len(picks) >= int(args.max_candidates):
            break
    return picks


def _materialize_candidate_config(base_payload: dict[str, Any], pick: dict[str, Any], out_path: Path) -> None:
    payload = json.loads(json.dumps(base_payload))
    payload["config"] = pick["row"]["materialized_cfg"]
    payload["retrospective_selection"] = {
        "profile_name": pick["profile_name"],
        "row_index": pick["row_index"],
        "iteration": pick["iteration"],
        "constraints": pick["constraints"],
        "metrics": pick["metrics"],
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _evaluate_candidate(
    config_path: Path,
    out_dir: Path,
    cases_path: Path,
    device_name: str,
    clip_seconds: int,
    phase_blend: float,
) -> dict[str, Any]:
    heldout = evaluate_config(
        config_path=config_path,
        out_dir=out_dir / "heldout_eval",
        time_steps=128,
        device_name=device_name,
    )
    benchmark = run_benchmark(
        config_path=config_path,
        out_dir=out_dir / "benchmark_expanded",
        device_name=device_name,
        clip_seconds=clip_seconds,
        phase_blend=phase_blend,
        rerender_check=True,
        cases=_load_cases_json(cases_path),
    )
    continuity_circleworld = benchmark_folder(
        folder=out_dir / "benchmark_expanded",
        pattern="*_circleworld.wav",
        out_path=out_dir / "benchmark_expanded" / "continuity_circleworld.json",
        min_loop_seconds=0.5,
        max_loop_seconds=4.0,
        chunk_seconds=2.0,
    )
    continuity_reference = benchmark_folder(
        folder=out_dir / "benchmark_expanded",
        pattern="*_reference.wav",
        out_path=out_dir / "benchmark_expanded" / "continuity_reference.json",
        min_loop_seconds=0.5,
        max_loop_seconds=4.0,
        chunk_seconds=2.0,
    )
    nested = _run_nested_commitment(
        config_path=config_path,
        out_dir=out_dir / "nested_commitment_livefork",
        device_name=device_name,
        case_json=cases_path,
        fork_selector="first_live_child",
        live_child_threshold=0.05,
    )
    law_tokens = build_library(
        config_path=config_path,
        out_dir=out_dir / "law_token_library",
        cases_path=cases_path,
        device_name=device_name,
        clip_seconds=clip_seconds,
    )
    summary = {
        "config_path": str(config_path),
        "heldout_summary": str(out_dir / "heldout_eval" / "heldout_summary.json"),
        "benchmark_summary": str(out_dir / "benchmark_expanded" / "benchmark_summary.json"),
        "continuity_circleworld": str(out_dir / "benchmark_expanded" / "continuity_circleworld.json"),
        "continuity_reference": str(out_dir / "benchmark_expanded" / "continuity_reference.json"),
        "nested_summary": str(out_dir / "nested_commitment_livefork" / "nested_commitment_report.json"),
        "law_token_library": str(out_dir / "law_token_library" / "law_token_library.json"),
        "heldout": {
            "mean_real_branch_fraction": heldout.get("mean_real_branch_fraction", 0.0),
            "mean_meso_branch_effect": heldout.get("mean_meso_branch_effect", 0.0),
            "mean_child_world_count": heldout.get("mean_child_world_count", 0.0),
            "mean_live_child_fraction": heldout.get("mean_live_child_fraction", 0.0),
            "mean_child_writeback_mass": heldout.get("mean_child_writeback_mass", 0.0),
            "mean_child_parent_divergence": heldout.get("mean_child_parent_divergence", 0.0),
            "mean_num_law_families": heldout.get("mean_num_law_families", 0.0),
            "mean_law_family_entropy": heldout.get("mean_law_family_entropy", 0.0),
            "by_source": heldout.get("by_source", {}),
        },
        "benchmark": {
            "mean_corr": benchmark["mean_corr"],
            "mean_mae": benchmark["mean_mae"],
            "mean_mse": benchmark["mean_mse"],
        },
        "continuity": {
            "mean_loop_autocorr_peak": continuity_circleworld["mean_loop_autocorr_peak"],
            "mean_nonlocal_chunk_repeat": continuity_circleworld["mean_nonlocal_chunk_repeat"],
            "mean_first_chunk_reentry": continuity_circleworld["mean_first_chunk_reentry"],
            "reference_mean_nonlocal_chunk_repeat": continuity_reference["mean_nonlocal_chunk_repeat"],
        },
        "nested": {
            "overall_read": nested.get("overall_read"),
            "verdict_counts": nested.get("verdict_counts", {}),
        },
        "law_tokens": {
            "mean_num_law_packets": law_tokens["mean_num_law_packets"],
            "mean_num_law_families": law_tokens["mean_num_law_families"],
            "aggregate_num_families": law_tokens["aggregate_library"]["num_families"],
        },
    }
    _write_json(out_dir / "candidate_summary.json", summary)
    return summary


def run_retrospective(
    search_history_path: Path,
    base_config_path: Path,
    out_dir: Path,
    cases_path: Path,
    device_name: str,
    clip_seconds: int,
    phase_blend: float,
    args: argparse.Namespace,
) -> dict[str, Any]:
    rows = json.loads(search_history_path.read_text(encoding="utf-8-sig"))
    base_payload = _load_config_payload(base_config_path)
    out_dir.mkdir(parents=True, exist_ok=True)

    picks = _select_candidates(rows, args)
    evaluated: list[dict[str, Any]] = []
    for pick in picks:
        slug = _candidate_slug(pick["profile_name"], int(pick["row_index"]), pick.get("iteration"))
        candidate_dir = out_dir / slug
        config_path = candidate_dir / "circleworld_real_anchor_config_cem_v1.json"
        _materialize_candidate_config(base_payload, pick, config_path)
        summary = _evaluate_candidate(
            config_path=config_path,
            out_dir=candidate_dir,
            cases_path=cases_path,
            device_name=device_name,
            clip_seconds=clip_seconds,
            phase_blend=phase_blend,
        )
        evaluated.append(
            {
                "slug": slug,
                "profile_name": pick["profile_name"],
                "row_index": pick["row_index"],
                "iteration": pick["iteration"],
                "selection_metrics": pick["metrics"],
                "evaluation": summary,
            }
        )

    report = {
        "mode": "retrospective_childworld_eval",
        "search_history_path": str(search_history_path),
        "base_config_path": str(base_config_path),
        "cases_path": str(cases_path),
        "device": device_name,
        "clip_seconds": clip_seconds,
        "phase_blend": phase_blend,
        "selection": {
            "min_naked_branch": float(args.min_naked_branch),
            "min_naked_writeback": float(args.min_naked_writeback),
            "min_naked_law_families": float(args.min_naked_law_families),
            "min_naked_law_entropy": float(args.min_naked_law_entropy),
            "relaxed_naked_branch": float(args.relaxed_naked_branch),
            "relaxed_naked_writeback": float(args.relaxed_naked_writeback),
            "relaxed_naked_parent_div": float(args.relaxed_naked_parent_div),
            "max_candidates": int(args.max_candidates),
        },
        "num_history_rows": len(rows),
        "selected_candidates": evaluated,
    }
    _write_json(out_dir / "retrospective_report.json", report)
    return report


def main() -> None:
    ap = argparse.ArgumentParser(description="Recover and evaluate alternate Circleworld childworld candidates from search history.")
    ap.add_argument("--search-history", required=True)
    ap.add_argument("--base-config", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--cases-json", default=str(ROOT / "outputs" / "circleworld_proto" / "expanded_anchor_cases_2026-04-09.json"))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--clip-seconds", type=int, default=10)
    ap.add_argument("--phase-blend", type=float, default=1.0)
    ap.add_argument("--max-candidates", type=int, default=3)
    ap.add_argument("--min-naked-branch", type=float, default=0.333)
    ap.add_argument("--min-naked-writeback", type=float, default=0.075)
    ap.add_argument("--min-naked-law-families", type=float, default=2.0)
    ap.add_argument("--min-naked-law-entropy", type=float, default=0.12)
    ap.add_argument("--relaxed-naked-branch", type=float, default=0.28)
    ap.add_argument("--relaxed-naked-writeback", type=float, default=0.07)
    ap.add_argument("--relaxed-naked-parent-div", type=float, default=0.04)
    args = ap.parse_args()

    report = run_retrospective(
        search_history_path=Path(args.search_history),
        base_config_path=Path(args.base_config),
        out_dir=Path(args.out_dir),
        cases_path=Path(args.cases_json),
        device_name=args.device,
        clip_seconds=int(args.clip_seconds),
        phase_blend=float(args.phase_blend),
        args=args,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
