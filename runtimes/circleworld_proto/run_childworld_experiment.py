from __future__ import annotations

import argparse
import json
import subprocess
import sys
import traceback
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
from train_circleworld_real_anchor import train_circleworld_real_anchor


DEFAULT_RAMANUJAN_INIT = (
    ROOT
    / "checkpoints_circleworld_proto"
    / "training_run_2026-04-21_token_diverse_ramanujan"
    / "circleworld_real_anchor_config_cem_v1.json"
)
DEFAULT_QTRACE_INIT = (
    ROOT
    / "outputs"
    / "circleworld_proto"
    / "branchlaw_ablation_series_2026-04-17"
    / "relational_qkv_v2_qtrace_only"
    / "checkpoint"
    / "circleworld_real_anchor_config_cem_v1.json"
)
DEFAULT_CASES = ROOT / "outputs" / "circleworld_proto" / "expanded_anchor_cases_2026-04-09.json"
BRANCH_PROFILE_NAMES = {
    "branch_identity_guard_v1",
    "branch_writeback_floor_v1",
    "branch_balance_guard_v1",
    "branch_recovery_soft_guard_v1",
    "naked_phase_nested_recovery_v1",
    "mode_replace_conversion_v1",
}


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _load_case_payload(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _seeded_nested_case_suite(suite_name: str = "default") -> list[dict[str, Any]]:
    normalized = str(suite_name or "default").strip().lower()
    if normalized in {"default", "seeded_branch_active"}:
        return [
            {
                "name": "naked_seed_9100",
                "seed_source": "naked_rafa",
                "seed": 9100,
                "time_steps": 128,
                "warmup_depth": 5,
                "continuation_depth": 3,
                "require_live_child_start": True,
            },
            {
                "name": "naked_seed_9101",
                "seed_source": "naked_rafa",
                "seed": 9101,
                "time_steps": 128,
                "warmup_depth": 5,
                "continuation_depth": 3,
                "require_live_child_start": True,
            },
            {
                "name": "naked_seed_9102",
                "seed_source": "naked_rafa",
                "seed": 9102,
                "time_steps": 128,
                "warmup_depth": 5,
                "continuation_depth": 3,
                "require_live_child_start": True,
            },
        ]
    if normalized in {"branch_recovery_soft_guard_v1", "branch_recovery_soft_guard", "naked_phase_nested_recovery_v1"}:
        return [
            {
                "name": "naked_seed_9200",
                "seed_source": "naked_rafa",
                "seed": 9200,
                "time_steps": 128,
                "warmup_depth": 5,
                "continuation_depth": 3,
                "require_live_child_start": True,
            },
            {
                "name": "naked_seed_9201",
                "seed_source": "naked_rafa",
                "seed": 9201,
                "time_steps": 128,
                "warmup_depth": 5,
                "continuation_depth": 3,
                "require_live_child_start": True,
            },
            {
                "name": "naked_seed_9202",
                "seed_source": "naked_rafa",
                "seed": 9202,
                "time_steps": 128,
                "warmup_depth": 5,
                "continuation_depth": 3,
                "require_live_child_start": True,
            },
        ]
    raise ValueError(f"Unknown selection_nested_case_suite={suite_name!r}")


def _default_seeded_nested_cases() -> list[dict[str, Any]]:
    return _seeded_nested_case_suite("default")


def _resolve_nested_case_json(
    *,
    cases_path: Path,
    out_dir: Path,
    profile: str,
    nested_case_json: Path | None,
    nested_case_plan: list[dict[str, Any]] | None = None,
) -> Path:
    if nested_case_json is not None:
        return nested_case_json
    if profile in BRANCH_PROFILE_NAMES:
        seeded_path = out_dir / "nested_seeded_branch_active_cases.json"
        if nested_case_plan is not None:
            seeded_cases = nested_case_plan
        else:
            suite_name = (
                "branch_recovery_soft_guard_v1"
                if profile in {"branch_recovery_soft_guard_v1", "naked_phase_nested_recovery_v1"}
                else "default"
            )
            seeded_cases = _seeded_nested_case_suite(suite_name)
        seeded_path.write_text(json.dumps(seeded_cases, indent=2), encoding="utf-8")
        return seeded_path
    return cases_path


def _materialize_nested_case_json(case_json: Path, out_dir: Path, limit: int | None, stem: str) -> Path:
    if limit is None or limit <= 0:
        return case_json
    raw_cases = _load_case_payload(case_json)
    if isinstance(raw_cases, list):
        trimmed = raw_cases[: int(limit)]
    elif isinstance(raw_cases, dict):
        trimmed = dict(list(raw_cases.items())[: int(limit)])
    else:
        trimmed = raw_cases
    trimmed_path = out_dir / f"{stem}.json"
    trimmed_path.write_text(json.dumps(trimmed, indent=2), encoding="utf-8")
    return trimmed_path


def _nested_report_stub(
    *,
    config_path: Path,
    out_dir: Path,
    status: str,
    overall_read: str,
    message: str,
    device_name: str,
    case_json: Path,
    fork_selector: str,
    live_child_threshold: float,
    attempts: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    report = {
        "config_path": str(config_path),
        "device": str(device_name),
        "source_case_json": str(case_json),
        "fork_selector": str(fork_selector),
        "live_child_threshold": float(live_child_threshold),
        "status": str(status),
        "overall_read": str(overall_read),
        "message": str(message),
        "verdict_counts": {},
        "cases": [],
        "num_live_child_start_cases": 0,
        "num_eligible_nested_cases": 0,
        "mean_child_response_score": 0.0,
        "max_child_response_score": 0.0,
        "mean_child_active_fraction": 0.0,
        "mean_child_meso_response": 0.0,
        "mean_coarse_preservation": 0.0,
        "mean_world_jump_penalty": 0.0,
        "mean_nested_sibling_fraction": 0.0,
        "mean_over_rigid_fraction": 0.0,
        "mean_child_survival_signal": 0.0,
        "mean_readout_sibling_response": 0.0,
        "mean_branch_identity_carry": 0.0,
        "mean_branch_identity_qualified_carry": 0.0,
        "mean_branch_identity_budget_retained": 0.0,
        "mean_branch_identity_parent_div": 0.0,
        "mean_branch_identity_sibling_div": 0.0,
        "mean_readout_branch_count": 0.0,
        "attempts": attempts or [],
    }
    _write_json(out_dir / "nested_commitment_report.json", report)
    return report


def _run_nested_attempt(cmd: list[str], out_dir: Path, label: str) -> subprocess.CompletedProcess[str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    stdout_path = out_dir / f"{label}_stdout.txt"
    stderr_path = out_dir / f"{label}_stderr.txt"
    (out_dir / f"{label}_command.txt").write_text(" ".join(cmd), encoding="utf-8")
    with stdout_path.open("w", encoding="utf-8") as stdout_handle, stderr_path.open("w", encoding="utf-8") as stderr_handle:
        return subprocess.run(
            cmd,
            check=True,
            cwd=str(ROOT),
            stdout=stdout_handle,
            stderr=stderr_handle,
            text=True,
        )


def _attempt_record(
    *,
    label: str,
    cmd: list[str],
    device_name: str,
    case_json: Path,
    status: str,
    exc: BaseException | None = None,
) -> dict[str, Any]:
    record: dict[str, Any] = {
        "label": str(label),
        "device": str(device_name),
        "case_json": str(case_json),
        "status": str(status),
        "command": list(cmd),
    }
    if exc is not None:
        record["error_type"] = type(exc).__name__
        record["error_message"] = str(exc)
        if isinstance(exc, subprocess.CalledProcessError):
            record["returncode"] = int(exc.returncode)
        record["traceback"] = traceback.format_exc()
    return record


def _childworld_score_cfg(profile: str = "childworld_v1") -> dict[str, float]:
    cfg = {
        "target_corr": 0.74,
        "target_mae": 0.094,
        "target_dom": 0.60,
        "target_entropy": 0.56,
        "min_promotions": 8.0,
        "w_corr": 0.45,
        "w_mae": 1.6,
        "w_dom": 0.25,
        "w_entropy": 0.25,
        "w_promote": 0.05,
        "w_residue": 0.14,
        "w_promotability": 0.10,
        "w_major": 0.10,
        "w_prefix_alignment": 0.20,
        "w_prefix_delta": 0.10,
        "corr_low": 0.45,
        "corr_high": 0.95,
        "mae_low": 0.040,
        "mae_high": 0.160,
        "w_corr_band": 0.8,
        "w_mae_band": 1.2,
        "w_baseline_corr": 1.2,
        "w_baseline_mae": 1.2,
        "baseline_corr_margin": 0.0,
        "baseline_mae_margin": 0.0,
        "w_real_branch": 5.0,
        "w_meso_branch": 90000.0,
        "w_slot2_live": 0.50,
        "w_silent_singlepath": 0.50,
        "w_branch_positive_mask": 2.5,
        "w_branch_negative_mask": 0.8,
        "w_decorative_slot2": 1.6,
        "w_relation_handoff": 5.0,
        "w_relation_attn_10": 0.75,
        "w_branch_defect": 0.8,
        "w_branch_world_grad": 0.4,
        "w_branch_q_disagreement": 0.8,
        "w_branch_phase_wall": 0.6,
        "w_branch_seed_energy": 0.3,
        "w_child_world_count": 1.8,
        "w_live_child": 2.6,
        "w_child_age": 0.9,
        "w_child_writeback": 1.8,
        "w_child_parent_div": 1.4,
        "w_child_sibling_div": 1.6,
        "w_defect_without_branch": 1.8,
        "w_branch_delay": 0.3,
        "target_case_law_families": 2.5,
        "target_case_law_entropy": 0.24,
        "target_case_law_top_q_entropy": 0.10,
        "target_case_law_dominant_family_share": 0.82,
        "target_case_law_dominant_q_share": 0.62,
        "target_aggregate_law_families": 7.0,
        "target_aggregate_law_entropy": 0.46,
        "target_aggregate_law_top_q_entropy": 0.16,
        "target_aggregate_law_top_q_unique": 2.0,
        "target_aggregate_law_dominant_family_share": 0.65,
        "target_aggregate_law_dominant_q_share": 0.60,
        "w_case_law_family_shortfall": 0.4,
        "w_case_law_entropy_shortfall": 0.5,
        "w_case_law_top_q_entropy_shortfall": 0.4,
        "w_case_law_dominant_family": 0.3,
        "w_case_law_dominant_q": 0.3,
        "w_aggregate_law_family_shortfall": 0.6,
        "w_aggregate_law_entropy_shortfall": 0.7,
        "w_aggregate_law_top_q_entropy_shortfall": 0.6,
        "w_aggregate_law_top_q_unique_shortfall": 0.5,
        "w_aggregate_law_dominant_family": 0.7,
        "w_aggregate_law_dominant_q": 0.6,
        "w_transfer_min_branch": 0.0,
        "w_transfer_min_meso": 0.0,
        "w_transfer_min_writeback": 0.0,
        "w_transfer_min_parent_div": 0.0,
        "w_transfer_min_live_child": 0.0,
        "w_transfer_branch_gap": 0.0,
        "w_transfer_meso_gap": 0.0,
        "w_transfer_writeback_gap": 0.0,
        "w_transfer_parent_div_gap": 0.0,
        "w_transfer_branch_shortfall": 0.0,
        "w_transfer_writeback_shortfall": 0.0,
        "w_transfer_parent_div_shortfall": 0.0,
        "w_transfer_meso_shortfall": 0.0,
        "w_transfer_live_child_shortfall": 0.0,
        "w_transfer_naked_zero_branch": 0.0,
        "w_transfer_naked_zero_writeback": 0.0,
    }
    if profile == "writeback_v2":
        cfg.update(
            {
                "w_corr": 0.35,
                "w_mae": 1.2,
                "w_baseline_corr": 1.0,
                "w_baseline_mae": 1.0,
                "w_real_branch": 6.0,
                "w_meso_branch": 110000.0,
                "w_slot2_live": 0.45,
                "w_silent_singlepath": 0.55,
                "w_branch_positive_mask": 2.8,
                "w_branch_negative_mask": 0.9,
                "w_decorative_slot2": 1.9,
                "w_relation_handoff": 5.5,
                "w_child_world_count": 1.4,
                "w_live_child": 2.1,
                "w_child_age": 0.2,
                "w_child_writeback": 10.0,
                "w_child_parent_div": 1.8,
                "w_child_sibling_div": 2.0,
                "w_defect_without_branch": 2.3,
                "w_branch_delay": -0.7,
                "target_case_law_families": 2.2,
                "target_case_law_entropy": 0.20,
                "target_aggregate_law_families": 6.0,
                "target_aggregate_law_entropy": 0.40,
            }
        )
    elif profile == "writeback_v3_identity":
        cfg.update(
            {
                "w_corr": 0.30,
                "w_mae": 1.0,
                "w_baseline_corr": 0.8,
                "w_baseline_mae": 0.8,
                "w_real_branch": 8.0,
                "w_meso_branch": 120000.0,
                "w_slot2_live": 0.55,
                "w_silent_singlepath": 0.70,
                "w_branch_positive_mask": 3.0,
                "w_branch_negative_mask": 1.0,
                "w_decorative_slot2": 2.1,
                "w_relation_handoff": 5.8,
                "w_child_world_count": 1.2,
                "w_live_child": 2.0,
                "w_child_age": 0.1,
                "w_child_writeback": 8.0,
                "w_child_parent_div": 4.0,
                "w_child_sibling_div": 3.5,
                "w_defect_without_branch": 2.5,
                "w_branch_delay": -0.9,
                "target_case_law_families": 2.0,
                "target_case_law_entropy": 0.18,
                "target_aggregate_law_families": 6.0,
                "target_aggregate_law_entropy": 0.36,
            }
        )
    elif profile == "writeback_v4_real_anchor_transfer":
        cfg.update(
            {
                "w_corr": 0.28,
                "w_mae": 0.95,
                "w_baseline_corr": 0.75,
                "w_baseline_mae": 0.75,
                "w_real_branch": 7.0,
                "w_meso_branch": 90000.0,
                "w_slot2_live": 0.45,
                "w_silent_singlepath": 0.75,
                "w_branch_positive_mask": 3.1,
                "w_branch_negative_mask": 1.1,
                "w_decorative_slot2": 2.0,
                "w_relation_handoff": 5.6,
                "w_child_world_count": 1.0,
                "w_live_child": 1.8,
                "w_child_age": 0.08,
                "w_child_writeback": 7.0,
                "w_child_parent_div": 4.5,
                "w_child_sibling_div": 3.8,
                "w_defect_without_branch": 2.7,
                "w_branch_delay": -0.8,
                "target_case_law_families": 2.0,
                "target_case_law_entropy": 0.18,
                "target_aggregate_law_families": 6.0,
                "target_aggregate_law_entropy": 0.36,
                "w_transfer_min_branch": 12.0,
                "w_transfer_min_meso": 160000.0,
                "w_transfer_min_writeback": 14.0,
                "w_transfer_min_parent_div": 10.0,
                "w_transfer_min_live_child": 6.0,
                "w_transfer_branch_gap": 3.5,
                "w_transfer_meso_gap": 90000.0,
                "w_transfer_writeback_gap": 3.0,
                "w_transfer_parent_div_gap": 2.5,
                "w_transfer_branch_shortfall": 7.0,
                "w_transfer_writeback_shortfall": 9.0,
                "w_transfer_parent_div_shortfall": 7.0,
                "w_transfer_meso_shortfall": 120000.0,
                "w_transfer_live_child_shortfall": 5.0,
                "w_transfer_naked_zero_branch": 5.0,
                "w_transfer_naked_zero_writeback": 5.0,
            }
        )
    elif profile == "writeback_v5_thresholded_real_branch":
        cfg.update(
            {
                "w_corr": 0.28,
                "w_mae": 0.95,
                "w_baseline_corr": 0.75,
                "w_baseline_mae": 0.75,
                "w_real_branch": 8.5,
                "w_meso_branch": 115000.0,
                "w_slot2_live": 0.45,
                "w_silent_singlepath": 0.75,
                "w_branch_positive_mask": 3.1,
                "w_branch_negative_mask": 1.1,
                "w_decorative_slot2": 2.0,
                "w_relation_handoff": 5.8,
                "w_child_world_count": 1.0,
                "w_live_child": 2.0,
                "w_child_age": 0.08,
                "w_child_writeback": 8.0,
                "w_child_parent_div": 4.8,
                "w_child_sibling_div": 4.0,
                "w_defect_without_branch": 2.7,
                "w_branch_delay": -0.8,
                "target_case_law_families": 2.0,
                "target_case_law_entropy": 0.18,
                "target_aggregate_law_families": 6.0,
                "target_aggregate_law_entropy": 0.36,
                "w_transfer_min_branch": 13.0,
                "w_transfer_min_meso": 190000.0,
                "w_transfer_min_writeback": 14.0,
                "w_transfer_min_parent_div": 10.0,
                "w_transfer_min_live_child": 6.0,
                "w_transfer_branch_gap": 3.0,
                "w_transfer_meso_gap": 80000.0,
                "w_transfer_writeback_gap": 2.5,
                "w_transfer_parent_div_gap": 2.0,
                "w_transfer_branch_shortfall": 7.0,
                "w_transfer_writeback_shortfall": 8.0,
                "w_transfer_parent_div_shortfall": 6.0,
                "w_transfer_meso_shortfall": 120000.0,
                "w_transfer_live_child_shortfall": 5.0,
                "w_transfer_naked_zero_branch": 5.0,
                "w_transfer_naked_zero_writeback": 5.0,
            }
        )
    elif profile == "writeback_v6_naked_survival_curriculum":
        cfg.update(
            {
                "w_corr": 0.24,
                "w_mae": 0.85,
                "w_baseline_corr": 0.65,
                "w_baseline_mae": 0.65,
                "w_real_branch": 7.5,
                "w_meso_branch": 105000.0,
                "w_slot2_live": 0.40,
                "w_silent_singlepath": 0.80,
                "w_branch_positive_mask": 3.2,
                "w_branch_negative_mask": 1.2,
                "w_decorative_slot2": 2.1,
                "w_relation_handoff": 5.8,
                "w_child_world_count": 1.0,
                "w_live_child": 2.1,
                "w_child_age": 0.12,
                "w_child_writeback": 8.0,
                "w_child_parent_div": 5.2,
                "w_child_sibling_div": 4.0,
                "w_defect_without_branch": 2.9,
                "w_branch_delay": -0.8,
                "target_case_law_families": 2.0,
                "target_case_law_entropy": 0.18,
                "target_aggregate_law_families": 6.0,
                "target_aggregate_law_entropy": 0.38,
                "w_transfer_min_branch": 10.0,
                "w_transfer_min_meso": 150000.0,
                "w_transfer_min_writeback": 12.0,
                "w_transfer_min_parent_div": 9.0,
                "w_transfer_min_live_child": 5.0,
                "w_transfer_branch_gap": 4.0,
                "w_transfer_meso_gap": 95000.0,
                "w_transfer_writeback_gap": 3.0,
                "w_transfer_parent_div_gap": 2.4,
                "w_transfer_branch_shortfall": 8.0,
                "w_transfer_writeback_shortfall": 8.0,
                "w_transfer_parent_div_shortfall": 6.0,
                "w_transfer_meso_shortfall": 130000.0,
                "w_transfer_live_child_shortfall": 5.0,
                "w_transfer_naked_zero_branch": 8.0,
                "w_transfer_naked_zero_writeback": 8.0,
                "w_naked_min_branch": 12.0,
                "w_naked_min_meso": 190000.0,
                "w_naked_min_writeback": 18.0,
                "w_naked_min_parent_div": 14.0,
                "w_naked_min_live_child": 9.0,
                "w_naked_defect_without_branch": 11.0,
                "w_naked_branch_delay": 2.0,
                "w_naked_branch_shortfall": 12.0,
                "w_naked_writeback_shortfall": 15.0,
                "w_naked_parent_div_shortfall": 12.0,
                "w_naked_meso_shortfall": 220000.0,
                "w_naked_live_child_shortfall": 10.0,
                "w_naked_zero_child_worlds": 9.0,
                "w_naked_high_defect_zero_branch": 28.0,
                "w_naked_high_defect_zero_writeback": 24.0,
            }
        )
    elif profile == "writeback_v7_audio_recovery_with_branch_floor":
        cfg.update(
            {
                "w_corr": 0.30,
                "w_mae": 1.05,
                "w_baseline_corr": 0.90,
                "w_baseline_mae": 0.90,
                "w_real_branch": 7.0,
                "w_meso_branch": 95000.0,
                "w_slot2_live": 0.42,
                "w_silent_singlepath": 0.72,
                "w_branch_positive_mask": 3.0,
                "w_branch_negative_mask": 1.1,
                "w_decorative_slot2": 1.9,
                "w_relation_handoff": 5.8,
                "w_child_world_count": 1.0,
                "w_live_child": 2.0,
                "w_child_age": 0.12,
                "w_child_writeback": 7.2,
                "w_child_parent_div": 4.8,
                "w_child_sibling_div": 3.8,
                "w_defect_without_branch": 2.7,
                "w_branch_delay": -0.7,
                "target_case_law_families": 2.0,
                "target_case_law_entropy": 0.18,
                "target_aggregate_law_families": 6.0,
                "target_aggregate_law_entropy": 0.38,
                "w_transfer_min_branch": 11.0,
                "w_transfer_min_meso": 145000.0,
                "w_transfer_min_writeback": 12.0,
                "w_transfer_min_parent_div": 9.0,
                "w_transfer_min_live_child": 5.0,
                "w_transfer_branch_gap": 3.2,
                "w_transfer_meso_gap": 88000.0,
                "w_transfer_writeback_gap": 2.8,
                "w_transfer_parent_div_gap": 2.2,
                "w_transfer_branch_shortfall": 9.0,
                "w_transfer_writeback_shortfall": 9.0,
                "w_transfer_parent_div_shortfall": 7.0,
                "w_transfer_meso_shortfall": 145000.0,
                "w_transfer_live_child_shortfall": 5.5,
                "w_transfer_naked_zero_branch": 10.0,
                "w_transfer_naked_zero_writeback": 10.0,
                "w_naked_min_branch": 14.0,
                "w_naked_min_meso": 210000.0,
                "w_naked_min_writeback": 18.0,
                "w_naked_min_parent_div": 16.0,
                "w_naked_min_live_child": 10.0,
                "w_naked_defect_without_branch": 12.0,
                "w_naked_branch_delay": 2.0,
                "w_naked_branch_shortfall": 14.0,
                "w_naked_writeback_shortfall": 16.0,
                "w_naked_parent_div_shortfall": 14.0,
                "w_naked_meso_shortfall": 240000.0,
                "w_naked_live_child_shortfall": 11.0,
                "w_naked_zero_child_worlds": 10.0,
                "w_naked_high_defect_zero_branch": 30.0,
                "w_naked_high_defect_zero_writeback": 26.0,
            }
        )
    elif profile == "writeback_v8_recovery_with_law_floor":
        cfg.update(
            {
                "w_corr": 0.31,
                "w_mae": 1.05,
                "w_baseline_corr": 0.92,
                "w_baseline_mae": 0.92,
                "w_real_branch": 7.0,
                "w_meso_branch": 93000.0,
                "w_slot2_live": 0.42,
                "w_silent_singlepath": 0.72,
                "w_branch_positive_mask": 3.0,
                "w_branch_negative_mask": 1.1,
                "w_decorative_slot2": 1.9,
                "w_relation_handoff": 5.8,
                "w_child_world_count": 1.0,
                "w_live_child": 2.0,
                "w_child_age": 0.12,
                "w_child_writeback": 7.2,
                "w_child_parent_div": 4.8,
                "w_child_sibling_div": 3.8,
                "w_defect_without_branch": 2.7,
                "w_branch_delay": -0.7,
                "target_case_law_families": 2.0,
                "target_case_law_entropy": 0.18,
                "target_case_law_top_q_entropy": 0.08,
                "target_aggregate_law_families": 7.0,
                "target_aggregate_law_entropy": 0.42,
                "target_aggregate_law_top_q_entropy": 0.14,
                "target_aggregate_law_top_q_unique": 2.0,
                "w_case_law_family_shortfall": 0.7,
                "w_case_law_entropy_shortfall": 0.8,
                "w_case_law_top_q_entropy_shortfall": 0.6,
                "w_case_law_dominant_family": 0.45,
                "w_case_law_dominant_q": 0.4,
                "w_aggregate_law_family_shortfall": 1.0,
                "w_aggregate_law_entropy_shortfall": 1.1,
                "w_aggregate_law_top_q_entropy_shortfall": 0.9,
                "w_aggregate_law_top_q_unique_shortfall": 0.8,
                "w_aggregate_law_dominant_family": 0.9,
                "w_aggregate_law_dominant_q": 0.8,
                "w_transfer_min_branch": 11.0,
                "w_transfer_min_meso": 145000.0,
                "w_transfer_min_writeback": 12.0,
                "w_transfer_min_parent_div": 9.0,
                "w_transfer_min_live_child": 5.0,
                "w_transfer_min_law_families": 5.0,
                "w_transfer_min_law_entropy": 4.0,
                "w_transfer_branch_gap": 3.2,
                "w_transfer_meso_gap": 88000.0,
                "w_transfer_writeback_gap": 2.8,
                "w_transfer_parent_div_gap": 2.2,
                "w_transfer_law_family_gap": 2.5,
                "w_transfer_law_entropy_gap": 2.5,
                "w_transfer_branch_shortfall": 9.0,
                "w_transfer_writeback_shortfall": 9.0,
                "w_transfer_parent_div_shortfall": 7.0,
                "w_transfer_meso_shortfall": 145000.0,
                "w_transfer_live_child_shortfall": 5.5,
                "w_transfer_law_family_shortfall": 5.0,
                "w_transfer_law_entropy_shortfall": 5.0,
                "w_transfer_naked_zero_branch": 10.0,
                "w_transfer_naked_zero_writeback": 10.0,
                "w_naked_min_branch": 14.0,
                "w_naked_min_meso": 210000.0,
                "w_naked_min_writeback": 18.0,
                "w_naked_min_parent_div": 16.0,
                "w_naked_min_live_child": 10.0,
                "w_naked_min_law_families": 7.0,
                "w_naked_min_law_entropy": 6.0,
                "w_naked_defect_without_branch": 12.0,
                "w_naked_branch_delay": 2.0,
                "w_naked_branch_shortfall": 14.0,
                "w_naked_writeback_shortfall": 16.0,
                "w_naked_parent_div_shortfall": 14.0,
                "w_naked_meso_shortfall": 240000.0,
                "w_naked_live_child_shortfall": 11.0,
                "w_naked_law_family_shortfall": 10.0,
                "w_naked_law_entropy_shortfall": 9.0,
                "w_naked_zero_child_worlds": 10.0,
                "w_naked_high_defect_zero_branch": 30.0,
                "w_naked_high_defect_zero_writeback": 26.0,
            }
        )
    elif profile == "writeback_v9_aligned_probe_recovery":
        cfg.update(
            {
                "use_hard_selection_gate": 1.0,
                "w_corr": 0.28,
                "w_mae": 0.98,
                "w_baseline_corr": 0.82,
                "w_baseline_mae": 0.82,
                "w_real_branch": 7.6,
                "w_meso_branch": 98000.0,
                "w_slot2_live": 0.44,
                "w_silent_singlepath": 0.76,
                "w_branch_positive_mask": 3.1,
                "w_branch_negative_mask": 1.1,
                "w_decorative_slot2": 2.0,
                "w_relation_handoff": 5.9,
                "w_child_world_count": 1.1,
                "w_live_child": 2.2,
                "w_child_age": 0.14,
                "w_child_writeback": 7.6,
                "w_child_parent_div": 5.0,
                "w_child_sibling_div": 4.0,
                "w_defect_without_branch": 2.9,
                "w_branch_delay": -0.8,
                "target_case_law_families": 2.1,
                "target_case_law_entropy": 0.20,
                "target_case_law_top_q_entropy": 0.09,
                "target_aggregate_law_families": 7.0,
                "target_aggregate_law_entropy": 0.44,
                "target_aggregate_law_top_q_entropy": 0.15,
                "target_aggregate_law_top_q_unique": 2.0,
                "w_case_law_family_shortfall": 0.8,
                "w_case_law_entropy_shortfall": 0.9,
                "w_case_law_top_q_entropy_shortfall": 0.7,
                "w_case_law_dominant_family": 0.5,
                "w_case_law_dominant_q": 0.45,
                "w_aggregate_law_family_shortfall": 1.1,
                "w_aggregate_law_entropy_shortfall": 1.2,
                "w_aggregate_law_top_q_entropy_shortfall": 1.0,
                "w_aggregate_law_top_q_unique_shortfall": 0.9,
                "w_aggregate_law_dominant_family": 0.95,
                "w_aggregate_law_dominant_q": 0.85,
                "w_transfer_min_branch": 13.0,
                "w_transfer_min_meso": 160000.0,
                "w_transfer_min_writeback": 14.0,
                "w_transfer_min_parent_div": 12.0,
                "w_transfer_min_live_child": 6.0,
                "w_transfer_min_law_families": 7.0,
                "w_transfer_min_law_entropy": 6.0,
                "w_transfer_branch_gap": 4.5,
                "w_transfer_meso_gap": 105000.0,
                "w_transfer_writeback_gap": 3.2,
                "w_transfer_parent_div_gap": 2.8,
                "w_transfer_law_family_gap": 3.4,
                "w_transfer_law_entropy_gap": 3.2,
                "w_transfer_branch_shortfall": 13.0,
                "w_transfer_writeback_shortfall": 13.0,
                "w_transfer_parent_div_shortfall": 10.0,
                "w_transfer_meso_shortfall": 190000.0,
                "w_transfer_live_child_shortfall": 7.0,
                "w_transfer_law_family_shortfall": 8.0,
                "w_transfer_law_entropy_shortfall": 7.0,
                "w_transfer_naked_zero_branch": 15.0,
                "w_transfer_naked_zero_writeback": 12.0,
                "w_naked_min_branch": 20.0,
                "w_naked_min_meso": 260000.0,
                "w_naked_min_writeback": 22.0,
                "w_naked_min_parent_div": 20.0,
                "w_naked_min_live_child": 13.0,
                "w_naked_min_law_families": 10.0,
                "w_naked_min_law_entropy": 8.0,
                "w_naked_defect_without_branch": 16.0,
                "w_naked_branch_delay": 2.5,
                "w_naked_branch_shortfall": 20.0,
                "w_naked_writeback_shortfall": 20.0,
                "w_naked_parent_div_shortfall": 18.0,
                "w_naked_meso_shortfall": 300000.0,
                "w_naked_live_child_shortfall": 15.0,
                "w_naked_law_family_shortfall": 14.0,
                "w_naked_law_entropy_shortfall": 12.0,
                "w_naked_zero_child_worlds": 14.0,
                "w_naked_high_defect_zero_branch": 36.0,
                "w_naked_high_defect_zero_writeback": 32.0,
            }
        )
    elif profile == "writeback_v11_hard_gate_seeded":
        cfg.update(
            {
                "use_hard_selection_gate": 1.0,
                "w_corr": 0.27,
                "w_mae": 0.96,
                "w_baseline_corr": 0.80,
                "w_baseline_mae": 0.80,
                "w_real_branch": 7.8,
                "w_meso_branch": 105000.0,
                "w_slot2_live": 0.44,
                "w_silent_singlepath": 0.78,
                "w_branch_positive_mask": 3.1,
                "w_branch_negative_mask": 1.1,
                "w_decorative_slot2": 2.0,
                "w_relation_handoff": 5.9,
                "w_child_world_count": 1.1,
                "w_live_child": 2.3,
                "w_child_age": 0.14,
                "w_child_writeback": 7.8,
                "w_child_parent_div": 5.2,
                "w_child_sibling_div": 4.0,
                "w_defect_without_branch": 3.0,
                "w_branch_delay": -0.8,
                "target_case_law_families": 2.0,
                "target_case_law_entropy": 0.18,
                "target_case_law_top_q_entropy": 0.09,
                "target_aggregate_law_families": 6.5,
                "target_aggregate_law_entropy": 0.40,
                "target_aggregate_law_top_q_entropy": 0.14,
                "target_aggregate_law_top_q_unique": 2.0,
                "w_case_law_family_shortfall": 0.7,
                "w_case_law_entropy_shortfall": 0.8,
                "w_case_law_top_q_entropy_shortfall": 0.7,
                "w_case_law_dominant_family": 0.45,
                "w_case_law_dominant_q": 0.40,
                "w_aggregate_law_family_shortfall": 0.9,
                "w_aggregate_law_entropy_shortfall": 1.0,
                "w_aggregate_law_top_q_entropy_shortfall": 0.9,
                "w_aggregate_law_top_q_unique_shortfall": 0.8,
                "w_aggregate_law_dominant_family": 0.85,
                "w_aggregate_law_dominant_q": 0.80,
                "w_transfer_min_branch": 13.0,
                "w_transfer_min_meso": 170000.0,
                "w_transfer_min_writeback": 15.0,
                "w_transfer_min_parent_div": 13.0,
                "w_transfer_min_live_child": 7.0,
                "w_transfer_min_law_families": 7.0,
                "w_transfer_min_law_entropy": 6.0,
                "w_transfer_branch_gap": 4.5,
                "w_transfer_meso_gap": 105000.0,
                "w_transfer_writeback_gap": 3.5,
                "w_transfer_parent_div_gap": 3.0,
                "w_transfer_law_family_gap": 3.0,
                "w_transfer_law_entropy_gap": 3.0,
                "w_transfer_branch_shortfall": 14.0,
                "w_transfer_writeback_shortfall": 14.0,
                "w_transfer_parent_div_shortfall": 12.0,
                "w_transfer_meso_shortfall": 210000.0,
                "w_transfer_live_child_shortfall": 8.0,
                "w_transfer_law_family_shortfall": 8.0,
                "w_transfer_law_entropy_shortfall": 7.0,
                "w_transfer_naked_zero_branch": 18.0,
                "w_transfer_naked_zero_writeback": 14.0,
                "w_naked_min_branch": 22.0,
                "w_naked_min_meso": 300000.0,
                "w_naked_min_writeback": 24.0,
                "w_naked_min_parent_div": 22.0,
                "w_naked_min_live_child": 15.0,
                "w_naked_min_law_families": 10.0,
                "w_naked_min_law_entropy": 8.0,
                "w_naked_defect_without_branch": 18.0,
                "w_naked_branch_delay": 2.5,
                "w_naked_branch_shortfall": 22.0,
                "w_naked_writeback_shortfall": 22.0,
                "w_naked_parent_div_shortfall": 20.0,
                "w_naked_meso_shortfall": 340000.0,
                "w_naked_live_child_shortfall": 16.0,
                "w_naked_law_family_shortfall": 14.0,
                "w_naked_law_entropy_shortfall": 12.0,
                "w_naked_zero_child_worlds": 16.0,
                "w_naked_high_defect_zero_branch": 40.0,
                "w_naked_high_defect_zero_writeback": 36.0,
            }
        )
    elif profile == "writeback_v12_verified_gate_seeded":
        cfg.update(
            {
                "use_hard_selection_gate": 1.0,
                "w_corr": 0.26,
                "w_mae": 0.94,
                "w_baseline_corr": 0.78,
                "w_baseline_mae": 0.78,
                "w_real_branch": 8.0,
                "w_meso_branch": 108000.0,
                "w_slot2_live": 0.45,
                "w_silent_singlepath": 0.80,
                "w_branch_positive_mask": 3.1,
                "w_branch_negative_mask": 1.1,
                "w_decorative_slot2": 2.0,
                "w_relation_handoff": 5.9,
                "w_child_world_count": 1.1,
                "w_live_child": 2.3,
                "w_child_age": 0.14,
                "w_child_writeback": 7.9,
                "w_child_parent_div": 5.3,
                "w_child_sibling_div": 4.0,
                "w_defect_without_branch": 3.0,
                "w_branch_delay": -0.8,
                "target_case_law_families": 2.0,
                "target_case_law_entropy": 0.18,
                "target_case_law_top_q_entropy": 0.09,
                "target_aggregate_law_families": 6.5,
                "target_aggregate_law_entropy": 0.40,
                "target_aggregate_law_top_q_entropy": 0.14,
                "target_aggregate_law_top_q_unique": 2.0,
                "w_case_law_family_shortfall": 0.7,
                "w_case_law_entropy_shortfall": 0.8,
                "w_case_law_top_q_entropy_shortfall": 0.7,
                "w_case_law_dominant_family": 0.45,
                "w_case_law_dominant_q": 0.40,
                "w_aggregate_law_family_shortfall": 0.9,
                "w_aggregate_law_entropy_shortfall": 1.0,
                "w_aggregate_law_top_q_entropy_shortfall": 0.9,
                "w_aggregate_law_top_q_unique_shortfall": 0.8,
                "w_aggregate_law_dominant_family": 0.85,
                "w_aggregate_law_dominant_q": 0.80,
                "w_transfer_min_branch": 13.0,
                "w_transfer_min_meso": 170000.0,
                "w_transfer_min_writeback": 15.0,
                "w_transfer_min_parent_div": 13.0,
                "w_transfer_min_live_child": 7.0,
                "w_transfer_min_law_families": 7.0,
                "w_transfer_min_law_entropy": 6.0,
                "w_transfer_branch_gap": 4.5,
                "w_transfer_meso_gap": 105000.0,
                "w_transfer_writeback_gap": 3.5,
                "w_transfer_parent_div_gap": 3.0,
                "w_transfer_law_family_gap": 3.0,
                "w_transfer_law_entropy_gap": 3.0,
                "w_transfer_branch_shortfall": 14.0,
                "w_transfer_writeback_shortfall": 14.0,
                "w_transfer_parent_div_shortfall": 12.0,
                "w_transfer_meso_shortfall": 210000.0,
                "w_transfer_live_child_shortfall": 8.0,
                "w_transfer_law_family_shortfall": 8.0,
                "w_transfer_law_entropy_shortfall": 7.0,
                "w_transfer_naked_zero_branch": 18.0,
                "w_transfer_naked_zero_writeback": 14.0,
                "w_naked_min_branch": 22.0,
                "w_naked_min_meso": 300000.0,
                "w_naked_min_writeback": 24.0,
                "w_naked_min_parent_div": 22.0,
                "w_naked_min_live_child": 15.0,
                "w_naked_min_law_families": 10.0,
                "w_naked_min_law_entropy": 8.0,
                "w_naked_defect_without_branch": 18.0,
                "w_naked_branch_delay": 2.5,
                "w_naked_branch_shortfall": 22.0,
                "w_naked_writeback_shortfall": 22.0,
                "w_naked_parent_div_shortfall": 20.0,
                "w_naked_meso_shortfall": 340000.0,
                "w_naked_live_child_shortfall": 16.0,
                "w_naked_law_family_shortfall": 14.0,
                "w_naked_law_entropy_shortfall": 12.0,
                "w_naked_zero_child_worlds": 16.0,
                "w_naked_high_defect_zero_branch": 40.0,
                "w_naked_high_defect_zero_writeback": 36.0,
            }
        )
    elif profile == "writeback_v13_targeted_replay_recovery":
        cfg.update(
            {
                "use_hard_selection_gate": 1.0,
                "w_corr": 0.27,
                "w_mae": 0.95,
                "w_baseline_corr": 0.80,
                "w_baseline_mae": 0.80,
                "w_real_branch": 7.9,
                "w_meso_branch": 106000.0,
                "w_slot2_live": 0.45,
                "w_silent_singlepath": 0.79,
                "w_branch_positive_mask": 3.1,
                "w_branch_negative_mask": 1.1,
                "w_decorative_slot2": 2.0,
                "w_relation_handoff": 5.9,
                "w_child_world_count": 1.1,
                "w_live_child": 2.3,
                "w_child_age": 0.14,
                "w_child_writeback": 8.1,
                "w_child_parent_div": 5.6,
                "w_child_sibling_div": 4.0,
                "w_defect_without_branch": 3.0,
                "w_branch_delay": -0.8,
                "target_case_law_families": 2.0,
                "target_case_law_entropy": 0.18,
                "target_case_law_top_q_entropy": 0.09,
                "target_aggregate_law_families": 6.5,
                "target_aggregate_law_entropy": 0.40,
                "target_aggregate_law_top_q_entropy": 0.14,
                "target_aggregate_law_top_q_unique": 2.0,
                "w_case_law_family_shortfall": 0.6,
                "w_case_law_entropy_shortfall": 0.7,
                "w_case_law_top_q_entropy_shortfall": 0.6,
                "w_case_law_dominant_family": 0.4,
                "w_case_law_dominant_q": 0.38,
                "w_aggregate_law_family_shortfall": 0.8,
                "w_aggregate_law_entropy_shortfall": 0.9,
                "w_aggregate_law_top_q_entropy_shortfall": 0.8,
                "w_aggregate_law_top_q_unique_shortfall": 0.7,
                "w_aggregate_law_dominant_family": 0.8,
                "w_aggregate_law_dominant_q": 0.75,
                "w_transfer_min_branch": 13.0,
                "w_transfer_min_meso": 170000.0,
                "w_transfer_min_writeback": 16.0,
                "w_transfer_min_parent_div": 16.0,
                "w_transfer_min_live_child": 7.0,
                "w_transfer_min_law_families": 6.0,
                "w_transfer_min_law_entropy": 5.0,
                "w_transfer_branch_gap": 4.5,
                "w_transfer_meso_gap": 105000.0,
                "w_transfer_writeback_gap": 3.4,
                "w_transfer_parent_div_gap": 3.2,
                "w_transfer_law_family_gap": 2.6,
                "w_transfer_law_entropy_gap": 2.5,
                "w_transfer_branch_shortfall": 14.0,
                "w_transfer_writeback_shortfall": 18.0,
                "w_transfer_parent_div_shortfall": 18.0,
                "w_transfer_meso_shortfall": 210000.0,
                "w_transfer_live_child_shortfall": 8.0,
                "w_transfer_law_family_shortfall": 7.0,
                "w_transfer_law_entropy_shortfall": 6.0,
                "w_transfer_naked_zero_branch": 18.0,
                "w_transfer_naked_zero_writeback": 16.0,
                "w_naked_min_branch": 22.0,
                "w_naked_min_meso": 300000.0,
                "w_naked_min_writeback": 30.0,
                "w_naked_min_parent_div": 30.0,
                "w_naked_min_live_child": 15.0,
                "w_naked_min_law_families": 8.0,
                "w_naked_min_law_entropy": 6.0,
                "w_naked_defect_without_branch": 18.0,
                "w_naked_branch_delay": 2.5,
                "w_naked_branch_shortfall": 22.0,
                "w_naked_writeback_shortfall": 30.0,
                "w_naked_parent_div_shortfall": 30.0,
                "w_naked_meso_shortfall": 340000.0,
                "w_naked_live_child_shortfall": 16.0,
                "w_naked_law_family_shortfall": 10.0,
                "w_naked_law_entropy_shortfall": 9.0,
                "w_naked_zero_child_worlds": 16.0,
                "w_naked_high_defect_zero_branch": 40.0,
                "w_naked_high_defect_zero_writeback": 40.0,
            }
        )
    elif profile == "writeback_v15_nested_divergence_parentmix":
        cfg.update(
            {
                "use_hard_selection_gate": 1.0,
                "w_corr": 0.20,
                "w_mae": 0.82,
                "w_baseline_corr": 0.68,
                "w_baseline_mae": 0.68,
                "w_real_branch": 8.2,
                "w_meso_branch": 145000.0,
                "w_slot2_live": 0.50,
                "w_silent_singlepath": 0.90,
                "w_branch_positive_mask": 3.4,
                "w_branch_negative_mask": 1.0,
                "w_decorative_slot2": 2.2,
                "w_relation_handoff": 6.6,
                "w_child_world_count": 1.2,
                "w_live_child": 2.5,
                "w_child_age": 0.18,
                "w_child_writeback": 8.6,
                "w_child_parent_div": 7.2,
                "w_child_sibling_div": 8.8,
                "w_defect_without_branch": 3.4,
                "w_branch_delay": -1.2,
                "target_case_law_families": 2.2,
                "target_case_law_entropy": 0.22,
                "target_case_law_top_q_entropy": 0.10,
                "target_aggregate_law_families": 7.0,
                "target_aggregate_law_entropy": 0.45,
                "target_aggregate_law_top_q_entropy": 0.16,
                "target_aggregate_law_top_q_unique": 2.0,
                "w_case_law_family_shortfall": 0.8,
                "w_case_law_entropy_shortfall": 0.9,
                "w_case_law_top_q_entropy_shortfall": 0.7,
                "w_case_law_dominant_family": 0.35,
                "w_case_law_dominant_q": 0.32,
                "w_aggregate_law_family_shortfall": 1.0,
                "w_aggregate_law_entropy_shortfall": 1.1,
                "w_aggregate_law_top_q_entropy_shortfall": 0.9,
                "w_aggregate_law_top_q_unique_shortfall": 0.8,
                "w_aggregate_law_dominant_family": 0.7,
                "w_aggregate_law_dominant_q": 0.68,
                "w_transfer_min_branch": 13.0,
                "w_transfer_min_meso": 260000.0,
                "w_transfer_min_writeback": 16.0,
                "w_transfer_min_parent_div": 17.0,
                "w_transfer_min_live_child": 8.0,
                "w_transfer_min_law_families": 7.0,
                "w_transfer_min_law_entropy": 6.0,
                "w_transfer_branch_gap": 4.8,
                "w_transfer_meso_gap": 135000.0,
                "w_transfer_writeback_gap": 3.6,
                "w_transfer_parent_div_gap": 3.5,
                "w_transfer_law_family_gap": 2.8,
                "w_transfer_law_entropy_gap": 2.7,
                "w_transfer_branch_shortfall": 16.0,
                "w_transfer_writeback_shortfall": 18.0,
                "w_transfer_parent_div_shortfall": 20.0,
                "w_transfer_meso_shortfall": 300000.0,
                "w_transfer_live_child_shortfall": 10.0,
                "w_transfer_law_family_shortfall": 8.0,
                "w_transfer_law_entropy_shortfall": 7.0,
                "w_transfer_naked_zero_branch": 20.0,
                "w_transfer_naked_zero_writeback": 18.0,
                "w_naked_min_branch": 24.0,
                "w_naked_min_meso": 420000.0,
                "w_naked_min_writeback": 28.0,
                "w_naked_min_parent_div": 28.0,
                "w_naked_min_live_child": 17.0,
                "w_naked_min_law_families": 10.0,
                "w_naked_min_law_entropy": 8.0,
                "w_naked_defect_without_branch": 20.0,
                "w_naked_branch_delay": 3.0,
                "w_naked_branch_shortfall": 24.0,
                "w_naked_writeback_shortfall": 28.0,
                "w_naked_parent_div_shortfall": 28.0,
                "w_naked_meso_shortfall": 440000.0,
                "w_naked_live_child_shortfall": 18.0,
                "w_naked_law_family_shortfall": 12.0,
                "w_naked_law_entropy_shortfall": 10.0,
                "w_naked_zero_child_worlds": 18.0,
                "w_naked_high_defect_zero_branch": 42.0,
                "w_naked_high_defect_zero_writeback": 42.0,
            }
        )
    elif profile == "writeback_v16_benchmark_guard_nested":
        cfg.update(
            {
                "use_hard_selection_gate": 1.0,
                "target_corr": 0.90,
                "target_mae": 0.050,
                "corr_low": 0.82,
                "corr_high": 0.97,
                "mae_low": 0.030,
                "mae_high": 0.085,
                "w_corr": 0.55,
                "w_mae": 2.2,
                "w_baseline_corr": 2.8,
                "w_baseline_mae": 3.0,
                "baseline_corr_margin": 0.0,
                "baseline_mae_margin": 0.0,
                "w_dom": 0.20,
                "w_entropy": 0.20,
                "w_real_branch": 7.8,
                "w_meso_branch": 185000.0,
                "w_slot2_live": 0.48,
                "w_silent_singlepath": 0.92,
                "w_branch_positive_mask": 3.0,
                "w_branch_negative_mask": 0.9,
                "w_decorative_slot2": 2.0,
                "w_relation_handoff": 6.2,
                "w_child_world_count": 1.0,
                "w_live_child": 2.1,
                "w_child_age": 0.14,
                "w_child_writeback": 7.4,
                "w_child_parent_div": 6.0,
                "w_child_sibling_div": 9.2,
                "w_defect_without_branch": 3.0,
                "w_branch_delay": -1.0,
                "target_case_law_families": 2.1,
                "target_case_law_entropy": 0.20,
                "target_case_law_top_q_entropy": 0.10,
                "target_aggregate_law_families": 6.5,
                "target_aggregate_law_entropy": 0.42,
                "target_aggregate_law_top_q_entropy": 0.15,
                "target_aggregate_law_top_q_unique": 2.0,
                "w_case_law_family_shortfall": 0.7,
                "w_case_law_entropy_shortfall": 0.8,
                "w_case_law_top_q_entropy_shortfall": 0.6,
                "w_case_law_dominant_family": 0.32,
                "w_case_law_dominant_q": 0.30,
                "w_aggregate_law_family_shortfall": 0.9,
                "w_aggregate_law_entropy_shortfall": 1.0,
                "w_aggregate_law_top_q_entropy_shortfall": 0.8,
                "w_aggregate_law_top_q_unique_shortfall": 0.7,
                "w_aggregate_law_dominant_family": 0.65,
                "w_aggregate_law_dominant_q": 0.62,
                "w_transfer_min_branch": 13.0,
                "w_transfer_min_meso": 220000.0,
                "w_transfer_min_writeback": 16.0,
                "w_transfer_min_parent_div": 16.0,
                "w_transfer_min_live_child": 8.0,
                "w_transfer_min_law_families": 7.0,
                "w_transfer_min_law_entropy": 6.0,
                "w_transfer_branch_gap": 4.0,
                "w_transfer_meso_gap": 115000.0,
                "w_transfer_writeback_gap": 3.0,
                "w_transfer_parent_div_gap": 3.0,
                "w_transfer_law_family_gap": 2.6,
                "w_transfer_law_entropy_gap": 2.4,
                "w_transfer_branch_shortfall": 15.0,
                "w_transfer_writeback_shortfall": 17.0,
                "w_transfer_parent_div_shortfall": 18.0,
                "w_transfer_meso_shortfall": 260000.0,
                "w_transfer_live_child_shortfall": 9.0,
                "w_transfer_law_family_shortfall": 7.0,
                "w_transfer_law_entropy_shortfall": 6.0,
                "w_transfer_naked_zero_branch": 20.0,
                "w_transfer_naked_zero_writeback": 18.0,
                "w_naked_min_branch": 24.0,
                "w_naked_min_meso": 360000.0,
                "w_naked_min_writeback": 26.0,
                "w_naked_min_parent_div": 26.0,
                "w_naked_min_live_child": 16.0,
                "w_naked_min_law_families": 10.0,
                "w_naked_min_law_entropy": 8.0,
                "w_naked_defect_without_branch": 18.0,
                "w_naked_branch_delay": 2.8,
                "w_naked_branch_shortfall": 22.0,
                "w_naked_writeback_shortfall": 24.0,
                "w_naked_parent_div_shortfall": 24.0,
                "w_naked_meso_shortfall": 390000.0,
                "w_naked_live_child_shortfall": 17.0,
                "w_naked_law_family_shortfall": 12.0,
                "w_naked_law_entropy_shortfall": 10.0,
                "w_naked_zero_child_worlds": 18.0,
                "w_naked_high_defect_zero_branch": 40.0,
                "w_naked_high_defect_zero_writeback": 40.0,
            }
        )
    elif profile == "writeback_v17_nested_response_gate":
        cfg.update(
            {
                "use_hard_selection_gate": 1.0,
                "target_corr": 0.90,
                "target_mae": 0.050,
                "corr_low": 0.84,
                "corr_high": 0.97,
                "mae_low": 0.030,
                "mae_high": 0.085,
                "w_corr": 0.60,
                "w_mae": 2.3,
                "w_baseline_corr": 3.0,
                "w_baseline_mae": 3.2,
                "baseline_corr_margin": 0.0,
                "baseline_mae_margin": 0.0,
                "w_dom": 0.20,
                "w_entropy": 0.20,
                "w_real_branch": 7.8,
                "w_meso_branch": 185000.0,
                "w_slot2_live": 0.48,
                "w_silent_singlepath": 0.92,
                "w_branch_positive_mask": 3.0,
                "w_branch_negative_mask": 0.9,
                "w_decorative_slot2": 2.0,
                "w_relation_handoff": 6.2,
                "w_child_world_count": 1.0,
                "w_live_child": 2.1,
                "w_child_age": 0.14,
                "w_child_writeback": 7.4,
                "w_child_parent_div": 5.2,
                "w_child_sibling_div": 4.2,
                "w_defect_without_branch": 2.9,
                "w_branch_delay": -0.5,
                "target_case_law_families": 2.0,
                "target_case_law_entropy": 0.18,
                "target_case_law_top_q_entropy": 0.09,
                "target_aggregate_law_families": 6.0,
                "target_aggregate_law_entropy": 0.36,
                "target_aggregate_law_top_q_entropy": 0.12,
                "target_aggregate_law_top_q_unique": 2.0,
                "w_case_law_family_shortfall": 0.6,
                "w_case_law_entropy_shortfall": 0.7,
                "w_case_law_top_q_entropy_shortfall": 0.6,
                "w_case_law_dominant_family": 0.4,
                "w_case_law_dominant_q": 0.35,
                "w_aggregate_law_family_shortfall": 0.8,
                "w_aggregate_law_entropy_shortfall": 0.9,
                "w_aggregate_law_top_q_entropy_shortfall": 0.8,
                "w_aggregate_law_top_q_unique_shortfall": 0.7,
                "w_aggregate_law_dominant_family": 0.8,
                "w_aggregate_law_dominant_q": 0.75,
                "w_transfer_min_branch": 13.0,
                "w_transfer_min_meso": 180000.0,
                "w_transfer_min_writeback": 15.0,
                "w_transfer_min_parent_div": 13.0,
                "w_transfer_min_live_child": 6.0,
                "w_transfer_min_law_families": 7.0,
                "w_transfer_min_law_entropy": 6.0,
                "w_transfer_branch_gap": 4.0,
                "w_transfer_meso_gap": 100000.0,
                "w_transfer_writeback_gap": 3.0,
                "w_transfer_parent_div_gap": 2.8,
                "w_transfer_law_family_gap": 2.8,
                "w_transfer_law_entropy_gap": 2.6,
                "w_transfer_branch_shortfall": 14.0,
                "w_transfer_writeback_shortfall": 14.0,
                "w_transfer_parent_div_shortfall": 12.0,
                "w_transfer_meso_shortfall": 200000.0,
                "w_transfer_live_child_shortfall": 8.0,
                "w_transfer_law_family_shortfall": 8.0,
                "w_transfer_law_entropy_shortfall": 7.0,
                "w_transfer_naked_zero_branch": 18.0,
                "w_transfer_naked_zero_writeback": 14.0,
                "w_naked_min_branch": 22.0,
                "w_naked_min_meso": 300000.0,
                "w_naked_min_writeback": 24.0,
                "w_naked_min_parent_div": 22.0,
                "w_naked_min_live_child": 15.0,
                "w_naked_min_law_families": 10.0,
                "w_naked_min_law_entropy": 8.0,
                "w_naked_defect_without_branch": 18.0,
                "w_naked_branch_delay": 2.5,
                "w_naked_branch_shortfall": 22.0,
                "w_naked_writeback_shortfall": 22.0,
                "w_naked_parent_div_shortfall": 20.0,
                "w_naked_meso_shortfall": 340000.0,
                "w_naked_live_child_shortfall": 16.0,
                "w_naked_law_family_shortfall": 14.0,
                "w_naked_law_entropy_shortfall": 12.0,
                "w_naked_zero_child_worlds": 16.0,
                "w_naked_high_defect_zero_branch": 40.0,
                "w_naked_high_defect_zero_writeback": 36.0,
                "w_nested_probe_response": 2.5,
                "w_nested_probe_active_fraction": 24.0,
                "w_nested_probe_meso_response": 240000.0,
                "w_nested_probe_sibling_fraction": 10.0,
                "w_nested_probe_coarse_preservation": 1.5,
                "target_nested_probe_coarse_preservation": 0.90,
                "w_nested_probe_over_rigid": 2.5,
                "w_nested_probe_world_jump": 5.0,
                "w_nested_probe_response_shortfall": 4.0,
                "w_nested_probe_active_shortfall": 24.0,
                "w_nested_probe_meso_shortfall": 240000.0,
            }
        )
    elif profile in {"writeback_v18_nested_probe_objective", "writeback_v19_seeded_nested_substrate", "writeback_v20_child_record_perturb"}:
        cfg.update(
            {
                "use_hard_selection_gate": 1.0,
                "target_corr": 0.90,
                "target_mae": 0.050,
                "corr_low": 0.84,
                "corr_high": 0.97,
                "mae_low": 0.030,
                "mae_high": 0.085,
                "w_corr": 0.58,
                "w_mae": 2.25,
                "w_baseline_corr": 2.9,
                "w_baseline_mae": 3.1,
                "baseline_corr_margin": 0.0,
                "baseline_mae_margin": 0.0,
                "w_dom": 0.20,
                "w_entropy": 0.20,
                "w_real_branch": 7.8,
                "w_meso_branch": 185000.0,
                "w_slot2_live": 0.48,
                "w_silent_singlepath": 0.92,
                "w_branch_positive_mask": 3.0,
                "w_branch_negative_mask": 0.9,
                "w_decorative_slot2": 2.0,
                "w_relation_handoff": 6.2,
                "w_child_world_count": 1.0,
                "w_live_child": 2.1,
                "w_child_age": 0.14,
                "w_child_writeback": 7.4,
                "w_child_parent_div": 5.2,
                "w_child_sibling_div": 4.2,
                "w_defect_without_branch": 2.9,
                "w_branch_delay": -0.5,
                "target_case_law_families": 2.0,
                "target_case_law_entropy": 0.18,
                "target_case_law_top_q_entropy": 0.09,
                "target_aggregate_law_families": 6.0,
                "target_aggregate_law_entropy": 0.36,
                "target_aggregate_law_top_q_entropy": 0.12,
                "target_aggregate_law_top_q_unique": 2.0,
                "w_case_law_family_shortfall": 0.6,
                "w_case_law_entropy_shortfall": 0.7,
                "w_case_law_top_q_entropy_shortfall": 0.6,
                "w_case_law_dominant_family": 0.4,
                "w_case_law_dominant_q": 0.35,
                "w_aggregate_law_family_shortfall": 0.8,
                "w_aggregate_law_entropy_shortfall": 0.9,
                "w_aggregate_law_top_q_entropy_shortfall": 0.8,
                "w_aggregate_law_top_q_unique_shortfall": 0.7,
                "w_aggregate_law_dominant_family": 0.8,
                "w_aggregate_law_dominant_q": 0.75,
                "w_transfer_min_branch": 13.0,
                "w_transfer_min_meso": 180000.0,
                "w_transfer_min_writeback": 15.0,
                "w_transfer_min_parent_div": 13.0,
                "w_transfer_min_live_child": 6.0,
                "w_transfer_min_law_families": 7.0,
                "w_transfer_min_law_entropy": 6.0,
                "w_transfer_branch_gap": 4.0,
                "w_transfer_meso_gap": 100000.0,
                "w_transfer_writeback_gap": 3.0,
                "w_transfer_parent_div_gap": 2.8,
                "w_transfer_law_family_gap": 2.8,
                "w_transfer_law_entropy_gap": 2.6,
                "w_transfer_branch_shortfall": 14.0,
                "w_transfer_writeback_shortfall": 14.0,
                "w_transfer_parent_div_shortfall": 12.0,
                "w_transfer_meso_shortfall": 200000.0,
                "w_transfer_live_child_shortfall": 8.0,
                "w_transfer_law_family_shortfall": 8.0,
                "w_transfer_law_entropy_shortfall": 7.0,
                "w_transfer_naked_zero_branch": 18.0,
                "w_transfer_naked_zero_writeback": 14.0,
                "w_naked_min_branch": 22.0,
                "w_naked_min_meso": 300000.0,
                "w_naked_min_writeback": 24.0,
                "w_naked_min_parent_div": 22.0,
                "w_naked_min_live_child": 15.0,
                "w_naked_min_law_families": 10.0,
                "w_naked_min_law_entropy": 8.0,
                "w_naked_defect_without_branch": 18.0,
                "w_naked_branch_delay": 2.5,
                "w_naked_branch_shortfall": 22.0,
                "w_naked_writeback_shortfall": 22.0,
                "w_naked_parent_div_shortfall": 20.0,
                "w_naked_meso_shortfall": 340000.0,
                "w_naked_live_child_shortfall": 16.0,
                "w_naked_law_family_shortfall": 14.0,
                "w_naked_law_entropy_shortfall": 12.0,
                "w_naked_zero_child_worlds": 16.0,
                "w_naked_high_defect_zero_branch": 40.0,
                "w_naked_high_defect_zero_writeback": 36.0,
                "w_nested_probe_response": 3.0,
                "w_nested_probe_active_fraction": 32.0,
                "w_nested_probe_meso_response": 320000.0,
                "w_nested_probe_sibling_fraction": 12.0,
                "w_nested_probe_coarse_preservation": 1.5,
                "target_nested_probe_coarse_preservation": 0.90,
                "w_nested_probe_over_rigid": 3.0,
                "w_nested_probe_world_jump": 5.0,
                "w_nested_probe_response_shortfall": 5.0,
                "w_nested_probe_active_shortfall": 32.0,
                "w_nested_probe_meso_shortfall": 320000.0,
            }
        )
    elif profile == "branch_identity_guard_v1":
        cfg.update(
            {
                "use_hard_selection_gate": 1.0,
                "target_corr": 0.78,
                "target_mae": 0.088,
                "w_corr": 0.32,
                "w_mae": 1.12,
                "w_baseline_corr": 1.05,
                "w_baseline_mae": 0.98,
                "w_prefix_alignment": 0.28,
                "w_prefix_delta": 0.16,
                "w_real_branch": 7.8,
                "w_meso_branch": 135000.0,
                "w_slot2_live": 0.48,
                "w_silent_singlepath": 0.88,
                "w_branch_positive_mask": 3.2,
                "w_branch_negative_mask": 1.0,
                "w_decorative_slot2": 2.1,
                "w_relation_handoff": 6.4,
                "w_child_world_count": 1.1,
                "w_live_child": 2.4,
                "w_child_age": 0.18,
                "w_child_writeback": 8.2,
                "w_child_parent_div": 7.4,
                "w_child_sibling_div": 9.0,
                "w_defect_without_branch": 3.2,
                "w_branch_delay": -1.1,
                "target_case_law_families": 2.2,
                "target_case_law_entropy": 0.22,
                "target_case_law_top_q_entropy": 0.10,
                "target_aggregate_law_families": 7.0,
                "target_aggregate_law_entropy": 0.44,
                "target_aggregate_law_top_q_entropy": 0.16,
                "target_aggregate_law_top_q_unique": 2.0,
                "w_case_law_family_shortfall": 0.85,
                "w_case_law_entropy_shortfall": 0.95,
                "w_case_law_top_q_entropy_shortfall": 0.7,
                "w_case_law_dominant_family": 0.36,
                "w_case_law_dominant_q": 0.34,
                "w_aggregate_law_family_shortfall": 1.0,
                "w_aggregate_law_entropy_shortfall": 1.1,
                "w_aggregate_law_top_q_entropy_shortfall": 0.9,
                "w_aggregate_law_top_q_unique_shortfall": 0.8,
                "w_aggregate_law_dominant_family": 0.72,
                "w_aggregate_law_dominant_q": 0.70,
                "w_transfer_min_branch": 13.0,
                "w_transfer_min_meso": 230000.0,
                "w_transfer_min_writeback": 16.0,
                "w_transfer_min_parent_div": 18.0,
                "w_transfer_min_live_child": 8.0,
                "w_transfer_min_law_families": 7.0,
                "w_transfer_min_law_entropy": 6.0,
                "w_transfer_branch_gap": 4.6,
                "w_transfer_meso_gap": 120000.0,
                "w_transfer_writeback_gap": 3.5,
                "w_transfer_parent_div_gap": 3.4,
                "w_transfer_law_family_gap": 2.8,
                "w_transfer_law_entropy_gap": 2.7,
                "w_transfer_branch_shortfall": 16.0,
                "w_transfer_writeback_shortfall": 17.0,
                "w_transfer_parent_div_shortfall": 18.0,
                "w_transfer_meso_shortfall": 260000.0,
                "w_transfer_live_child_shortfall": 9.0,
                "w_transfer_law_family_shortfall": 8.0,
                "w_transfer_law_entropy_shortfall": 7.0,
                "w_transfer_naked_zero_branch": 20.0,
                "w_transfer_naked_zero_writeback": 18.0,
                "w_naked_min_branch": 24.0,
                "w_naked_min_meso": 380000.0,
                "w_naked_min_writeback": 28.0,
                "w_naked_min_parent_div": 28.0,
                "w_naked_min_live_child": 17.0,
                "w_naked_min_law_families": 10.0,
                "w_naked_min_law_entropy": 8.0,
                "w_naked_defect_without_branch": 20.0,
                "w_naked_branch_delay": 3.0,
                "w_naked_branch_shortfall": 24.0,
                "w_naked_writeback_shortfall": 28.0,
                "w_naked_parent_div_shortfall": 28.0,
                "w_naked_meso_shortfall": 420000.0,
                "w_naked_live_child_shortfall": 18.0,
                "w_naked_law_family_shortfall": 12.0,
                "w_naked_law_entropy_shortfall": 10.0,
                "w_naked_zero_child_worlds": 18.0,
                "w_naked_high_defect_zero_branch": 42.0,
                "w_naked_high_defect_zero_writeback": 40.0,
            }
        )
    elif profile == "branch_writeback_floor_v1":
        cfg.update(
            {
                "use_hard_selection_gate": 1.0,
                "w_corr": 0.20,
                "w_mae": 0.84,
                "w_baseline_corr": 0.70,
                "w_baseline_mae": 0.70,
                "w_real_branch": 8.1,
                "w_meso_branch": 150000.0,
                "w_slot2_live": 0.50,
                "w_silent_singlepath": 0.88,
                "w_branch_positive_mask": 3.3,
                "w_branch_negative_mask": 1.0,
                "w_decorative_slot2": 2.2,
                "w_relation_handoff": 6.4,
                "w_child_world_count": 1.3,
                "w_live_child": 2.7,
                "w_child_age": 0.16,
                "w_child_writeback": 10.6,
                "w_child_parent_div": 6.0,
                "w_child_sibling_div": 7.0,
                "w_defect_without_branch": 3.2,
                "w_branch_delay": -1.0,
                "target_case_law_families": 2.1,
                "target_case_law_entropy": 0.20,
                "target_case_law_top_q_entropy": 0.10,
                "target_aggregate_law_families": 6.8,
                "target_aggregate_law_entropy": 0.42,
                "target_aggregate_law_top_q_entropy": 0.15,
                "target_aggregate_law_top_q_unique": 2.0,
                "w_case_law_family_shortfall": 0.75,
                "w_case_law_entropy_shortfall": 0.85,
                "w_case_law_top_q_entropy_shortfall": 0.7,
                "w_case_law_dominant_family": 0.38,
                "w_case_law_dominant_q": 0.34,
                "w_aggregate_law_family_shortfall": 0.95,
                "w_aggregate_law_entropy_shortfall": 1.05,
                "w_aggregate_law_top_q_entropy_shortfall": 0.9,
                "w_aggregate_law_top_q_unique_shortfall": 0.8,
                "w_aggregate_law_dominant_family": 0.78,
                "w_aggregate_law_dominant_q": 0.74,
                "w_transfer_min_branch": 13.0,
                "w_transfer_min_meso": 220000.0,
                "w_transfer_min_writeback": 18.0,
                "w_transfer_min_parent_div": 14.0,
                "w_transfer_min_live_child": 8.0,
                "w_transfer_min_law_families": 7.0,
                "w_transfer_min_law_entropy": 6.0,
                "w_transfer_branch_gap": 4.4,
                "w_transfer_meso_gap": 120000.0,
                "w_transfer_writeback_gap": 4.2,
                "w_transfer_parent_div_gap": 3.0,
                "w_transfer_law_family_gap": 2.8,
                "w_transfer_law_entropy_gap": 2.6,
                "w_transfer_branch_shortfall": 15.0,
                "w_transfer_writeback_shortfall": 22.0,
                "w_transfer_parent_div_shortfall": 14.0,
                "w_transfer_meso_shortfall": 250000.0,
                "w_transfer_live_child_shortfall": 9.0,
                "w_transfer_law_family_shortfall": 8.0,
                "w_transfer_law_entropy_shortfall": 7.0,
                "w_transfer_naked_zero_branch": 18.0,
                "w_transfer_naked_zero_writeback": 24.0,
                "w_naked_min_branch": 23.0,
                "w_naked_min_meso": 360000.0,
                "w_naked_min_writeback": 32.0,
                "w_naked_min_parent_div": 24.0,
                "w_naked_min_live_child": 17.0,
                "w_naked_min_law_families": 10.0,
                "w_naked_min_law_entropy": 8.0,
                "w_naked_defect_without_branch": 19.0,
                "w_naked_branch_delay": 2.8,
                "w_naked_branch_shortfall": 23.0,
                "w_naked_writeback_shortfall": 32.0,
                "w_naked_parent_div_shortfall": 24.0,
                "w_naked_meso_shortfall": 400000.0,
                "w_naked_live_child_shortfall": 18.0,
                "w_naked_law_family_shortfall": 13.0,
                "w_naked_law_entropy_shortfall": 11.0,
                "w_naked_zero_child_worlds": 17.0,
                "w_naked_high_defect_zero_branch": 41.0,
                "w_naked_high_defect_zero_writeback": 44.0,
            }
        )
    elif profile == "branch_balance_guard_v1":
        cfg.update(
            {
                "use_hard_selection_gate": 1.0,
                "w_corr": 0.24,
                "w_mae": 0.90,
                "w_baseline_corr": 0.78,
                "w_baseline_mae": 0.78,
                "w_real_branch": 8.0,
                "w_meso_branch": 125000.0,
                "w_slot2_live": 0.48,
                "w_silent_singlepath": 0.86,
                "w_branch_positive_mask": 3.3,
                "w_branch_negative_mask": 1.0,
                "w_decorative_slot2": 2.1,
                "w_relation_handoff": 6.2,
                "w_child_world_count": 1.2,
                "w_live_child": 2.5,
                "w_child_age": 0.16,
                "w_child_writeback": 8.4,
                "w_child_parent_div": 6.4,
                "w_child_sibling_div": 6.8,
                "w_defect_without_branch": 3.1,
                "w_branch_delay": -1.0,
                "target_case_law_families": 2.1,
                "target_case_law_entropy": 0.20,
                "target_case_law_top_q_entropy": 0.10,
                "target_aggregate_law_families": 6.8,
                "target_aggregate_law_entropy": 0.42,
                "target_aggregate_law_top_q_entropy": 0.15,
                "target_aggregate_law_top_q_unique": 2.0,
                "w_case_law_family_shortfall": 0.8,
                "w_case_law_entropy_shortfall": 0.9,
                "w_case_law_top_q_entropy_shortfall": 0.7,
                "w_case_law_dominant_family": 0.40,
                "w_case_law_dominant_q": 0.36,
                "w_aggregate_law_family_shortfall": 0.95,
                "w_aggregate_law_entropy_shortfall": 1.05,
                "w_aggregate_law_top_q_entropy_shortfall": 0.9,
                "w_aggregate_law_top_q_unique_shortfall": 0.8,
                "w_aggregate_law_dominant_family": 0.78,
                "w_aggregate_law_dominant_q": 0.74,
                "w_transfer_min_branch": 13.0,
                "w_transfer_min_meso": 200000.0,
                "w_transfer_min_writeback": 16.0,
                "w_transfer_min_parent_div": 15.0,
                "w_transfer_min_live_child": 8.0,
                "w_transfer_min_law_families": 7.0,
                "w_transfer_min_law_entropy": 6.0,
                "w_transfer_branch_gap": 4.3,
                "w_transfer_meso_gap": 110000.0,
                "w_transfer_writeback_gap": 3.4,
                "w_transfer_parent_div_gap": 3.1,
                "w_transfer_law_family_gap": 2.9,
                "w_transfer_law_entropy_gap": 2.8,
                "w_transfer_branch_shortfall": 15.0,
                "w_transfer_writeback_shortfall": 16.0,
                "w_transfer_parent_div_shortfall": 15.0,
                "w_transfer_meso_shortfall": 240000.0,
                "w_transfer_live_child_shortfall": 9.0,
                "w_transfer_law_family_shortfall": 8.0,
                "w_transfer_law_entropy_shortfall": 7.0,
                "w_transfer_naked_zero_branch": 18.0,
                "w_transfer_naked_zero_writeback": 16.0,
                "w_naked_min_branch": 23.0,
                "w_naked_min_meso": 340000.0,
                "w_naked_min_writeback": 26.0,
                "w_naked_min_parent_div": 24.0,
                "w_naked_min_live_child": 16.0,
                "w_naked_min_law_families": 10.0,
                "w_naked_min_law_entropy": 8.0,
                "w_naked_defect_without_branch": 19.0,
                "w_naked_branch_delay": 2.7,
                "w_naked_branch_shortfall": 23.0,
                "w_naked_writeback_shortfall": 26.0,
                "w_naked_parent_div_shortfall": 24.0,
                "w_naked_meso_shortfall": 380000.0,
                "w_naked_live_child_shortfall": 17.0,
                "w_naked_law_family_shortfall": 13.0,
                "w_naked_law_entropy_shortfall": 11.0,
                "w_naked_zero_child_worlds": 17.0,
                "w_naked_high_defect_zero_branch": 41.0,
                "w_naked_high_defect_zero_writeback": 38.0,
            }
        )
    elif profile in {"branch_recovery_soft_guard_v1", "naked_phase_nested_recovery_v1"}:
        cfg.update(
            {
                "use_hard_selection_gate": 1.0,
                "target_corr": 0.82,
                "target_mae": 0.086,
                "w_corr": 0.30,
                "w_mae": 1.02,
                "w_baseline_corr": 0.92,
                "w_baseline_mae": 0.90,
                "w_prefix_alignment": 0.24,
                "w_prefix_delta": 0.14,
                "w_real_branch": 7.7,
                "w_phase_only_real_branch": 10.0,
                "w_phase_only_branch_distinctness": 16.0,
                "w_meso_branch": 118000.0,
                "w_slot2_live": 0.46,
                "w_silent_singlepath": 0.84,
                "w_branch_positive_mask": 3.2,
                "w_branch_negative_mask": 0.95,
                "w_decorative_slot2": 2.0,
                "w_relation_handoff": 6.0,
                "w_child_world_count": 1.15,
                "w_live_child": 2.4,
                "w_child_age": 0.15,
                "w_child_writeback": 8.0,
                "w_child_parent_div": 6.0,
                "w_child_sibling_div": 6.2,
                "w_defect_without_branch": 3.0,
                "w_branch_delay": -0.9,
                "target_case_law_families": 2.0,
                "target_case_law_entropy": 0.19,
                "target_case_law_top_q_entropy": 0.09,
                "target_aggregate_law_families": 6.6,
                "target_aggregate_law_entropy": 0.40,
                "target_aggregate_law_top_q_entropy": 0.14,
                "target_aggregate_law_top_q_unique": 2.0,
                "w_case_law_family_shortfall": 0.78,
                "w_case_law_entropy_shortfall": 0.88,
                "w_case_law_top_q_entropy_shortfall": 0.68,
                "w_case_law_dominant_family": 0.38,
                "w_case_law_dominant_q": 0.34,
                "w_aggregate_law_family_shortfall": 0.92,
                "w_aggregate_law_entropy_shortfall": 1.0,
                "w_aggregate_law_top_q_entropy_shortfall": 0.86,
                "w_aggregate_law_top_q_unique_shortfall": 0.78,
                "w_aggregate_law_dominant_family": 0.74,
                "w_aggregate_law_dominant_q": 0.70,
                "w_transfer_min_branch": 13.0,
                "w_transfer_min_phase_only_branch": 24.0,
                "w_transfer_min_phase_only_distinctness": 36.0,
                "w_transfer_min_meso": 200000.0,
                "w_transfer_min_writeback": 16.0,
                "w_transfer_min_parent_div": 15.0,
                "w_transfer_min_live_child": 8.0,
                "w_transfer_min_law_families": 7.0,
                "w_transfer_min_law_entropy": 6.0,
                "w_transfer_branch_gap": 4.2,
                "w_transfer_phase_only_branch_gap": 5.0,
                "w_transfer_phase_only_distinctness_gap": 7.5,
                "w_transfer_meso_gap": 105000.0,
                "w_transfer_writeback_gap": 3.3,
                "w_transfer_parent_div_gap": 3.0,
                "w_transfer_law_family_gap": 2.8,
                "w_transfer_law_entropy_gap": 2.7,
                "w_transfer_branch_shortfall": 15.0,
                "w_transfer_phase_only_branch_shortfall": 24.0,
                "w_transfer_phase_only_distinctness_shortfall": 36.0,
                "w_transfer_writeback_shortfall": 16.0,
                "w_transfer_parent_div_shortfall": 15.0,
                "w_transfer_meso_shortfall": 235000.0,
                "w_transfer_live_child_shortfall": 9.0,
                "w_transfer_law_family_shortfall": 8.0,
                "w_transfer_law_entropy_shortfall": 7.0,
                "w_transfer_naked_zero_branch": 18.0,
                "w_transfer_naked_zero_writeback": 16.0,
                "w_naked_min_branch": 23.0,
                "w_naked_min_phase_only_branch": 36.0,
                "w_naked_min_phase_only_distinctness": 48.0,
                "w_naked_min_meso": 340000.0,
                "w_naked_min_writeback": 26.0,
                "w_naked_min_parent_div": 24.0,
                "w_naked_min_live_child": 16.0,
                "w_naked_min_law_families": 10.0,
                "w_naked_min_law_entropy": 8.0,
                "w_naked_defect_without_branch": 18.0,
                "w_naked_branch_delay": 2.5,
                "w_naked_branch_shortfall": 22.0,
                "w_naked_phase_only_branch_shortfall": 36.0,
                "w_naked_phase_only_distinctness_shortfall": 48.0,
                "w_naked_writeback_shortfall": 25.0,
                "w_naked_parent_div_shortfall": 23.0,
                "w_naked_meso_shortfall": 360000.0,
                "w_naked_live_child_shortfall": 16.0,
                "w_naked_law_family_shortfall": 12.0,
                "w_naked_law_entropy_shortfall": 10.0,
                "w_naked_zero_child_worlds": 16.0,
                "w_naked_high_defect_zero_branch": 38.0,
                "w_naked_high_defect_zero_writeback": 35.0,
                "w_nested_probe_response": 3.0,
                "w_nested_probe_active_fraction": 32.0,
                "w_nested_probe_meso_response": 320000.0,
                "w_nested_probe_sibling_fraction": 12.0,
                "w_nested_probe_coarse_preservation": 1.5,
                "target_nested_probe_coarse_preservation": 0.90,
                "w_nested_probe_over_rigid": 3.0,
                "w_nested_probe_world_jump": 5.0,
                "w_nested_probe_response_shortfall": 5.0,
                "w_nested_probe_active_shortfall": 32.0,
                "w_nested_probe_meso_shortfall": 320000.0,
            }
        )
        if profile == "naked_phase_nested_recovery_v1":
            cfg.update(
                {
                    "w_phase_only_real_branch": 14.0,
                    "w_phase_only_branch_distinctness": 22.0,
                    "w_phase_only_branch_phase_wall": 3.0,
                    "w_transfer_min_phase_only_branch": 36.0,
                    "w_transfer_min_phase_only_distinctness": 52.0,
                    "w_transfer_phase_only_branch_gap": 6.0,
                    "w_transfer_phase_only_distinctness_gap": 9.0,
                    "w_transfer_phase_only_branch_shortfall": 38.0,
                    "w_transfer_phase_only_distinctness_shortfall": 54.0,
                    "w_naked_min_phase_only_branch": 56.0,
                    "w_naked_min_phase_only_distinctness": 72.0,
                    "w_naked_phase_only_branch_shortfall": 64.0,
                    "w_naked_phase_only_distinctness_shortfall": 84.0,
                    "w_nested_probe_response": 3.0,
                    "w_nested_probe_active_fraction": 40.0,
                    "w_nested_probe_meso_response": 420000.0,
                    "w_nested_probe_sibling_fraction": 10.0,
                    "w_nested_probe_response_shortfall": 5.0,
                    "w_nested_probe_active_shortfall": 44.0,
                    "w_nested_probe_meso_shortfall": 440000.0,
                }
            )
    elif profile == "mode_replace_conversion_v1":
        cfg.update(
            {
                "use_hard_selection_gate": 1.0,
                "target_corr": 0.80,
                "target_mae": 0.090,
                "w_corr": 0.24,
                "w_mae": 0.86,
                "w_baseline_corr": 0.74,
                "w_baseline_mae": 0.72,
                "w_real_branch": 7.6,
                "w_meso_branch": 115000.0,
                "w_slot2_live": 0.46,
                "w_silent_singlepath": 0.82,
                "w_branch_positive_mask": 3.0,
                "w_branch_negative_mask": 0.95,
                "w_decorative_slot2": 2.4,
                "w_relation_handoff": 6.0,
                "w_child_world_count": 1.1,
                "w_live_child": 2.3,
                "w_child_age": 0.14,
                "w_child_writeback": 7.0,
                "w_child_parent_div": 6.0,
                "w_child_sibling_div": 7.4,
                "w_defect_without_branch": 3.0,
                "w_branch_delay": -0.8,
                "target_case_law_families": 2.0,
                "target_case_law_entropy": 0.18,
                "target_aggregate_law_families": 6.4,
                "target_aggregate_law_entropy": 0.38,
                "w_case_law_family_shortfall": 0.72,
                "w_case_law_entropy_shortfall": 0.82,
                "w_aggregate_law_family_shortfall": 0.90,
                "w_aggregate_law_entropy_shortfall": 1.00,
                "w_nested_probe_response": 2.0,
                "w_nested_probe_active_fraction": 20.0,
                "w_nested_probe_meso_response": 180000.0,
                "w_nested_probe_mode_replace_sibling_fraction": 35.0,
                "w_nested_probe_parent_mode_conversion_sibling_fraction": 45.0,
                "w_nested_probe_mode_replace_max_readiness": 18.0,
                "w_nested_probe_mode_replace_readout_response": 7.0,
                "w_nested_probe_mode_replace_q_corr": 2.0,
                "w_nested_probe_child_record_survival_score": 16.0,
                "w_nested_probe_mode_replace_conversion_score": 60.0,
                "w_nested_probe_coarse_preservation": 2.0,
                "target_nested_probe_coarse_preservation": 0.90,
                "w_nested_probe_world_jump": 8.0,
                "w_nested_probe_over_rigid": 2.5,
                "w_nested_probe_readout_without_continuation_penalty": 8.0,
                "w_nested_probe_direct_readout_dependency_penalty": 12.0,
                "w_nested_probe_final_direct_mix_shortcut_penalty": 20.0,
                "w_nested_probe_mode_replace_readiness_shortfall": 16.0,
                "target_nested_probe_mode_replace_readiness": 0.58,
                "w_nested_probe_mode_replace_response_shortfall": 4.0,
                "target_nested_probe_mode_replace_readout_response": 0.16,
                "w_nested_probe_child_record_survival_shortfall": 10.0,
                "target_nested_probe_child_record_survival_score": 0.60,
                "w_nested_probe_mode_replace_conversion_shortfall": 28.0,
                "target_nested_probe_mode_replace_conversion_score": 0.08,
            }
        )
    return cfg


def _heldout_probe_cfg(profile: str = "childworld_v1") -> dict[str, Any]:
    cfg: dict[str, Any] = {
        "enabled": False,
        "time_steps": 128,
        # Mirror the current heldout evaluator seed families so search-time
        # probe decisions are directly comparable to the main retrospective lane.
        "plan": [
            ("synthetic", 5100),
            ("synthetic", 5101),
            ("synthetic", 5102),
            ("synthetic", 5103),
            ("synthetic", 5104),
            ("synthetic", 5105),
            ("naked_rafa", 6100),
            ("naked_rafa", 6101),
            ("naked_rafa", 6102),
        ],
        "min_branch_target": 0.15,
        "min_writeback_target": 0.02,
        "min_parent_div_target": 0.05,
        "min_meso_target": 0.004,
        "min_live_child_target": 0.10,
    }
    if profile in {"writeback_v4_real_anchor_transfer", "writeback_v5_thresholded_real_branch", "writeback_v6_naked_survival_curriculum", "writeback_v7_audio_recovery_with_branch_floor", "writeback_v8_recovery_with_law_floor", "writeback_v9_aligned_probe_recovery", "writeback_v11_hard_gate_seeded", "writeback_v12_verified_gate_seeded", "writeback_v13_targeted_replay_recovery", "writeback_v15_nested_divergence_parentmix", "writeback_v16_benchmark_guard_nested", "writeback_v17_nested_response_gate", "writeback_v18_nested_probe_objective", "writeback_v19_seeded_nested_substrate", "writeback_v20_child_record_perturb", "branch_identity_guard_v1", "branch_writeback_floor_v1", "branch_balance_guard_v1", "branch_recovery_soft_guard_v1", "naked_phase_nested_recovery_v1", "mode_replace_conversion_v1"}:
        cfg["enabled"] = True
        cfg["min_branch_target"] = 0.20
        cfg["min_writeback_target"] = 0.02
        cfg["min_parent_div_target"] = 0.06
        cfg["min_meso_target"] = 0.005
        cfg["min_live_child_target"] = 0.12
        cfg["selection_probe_repeats"] = 1
        cfg["selection_heldout_top_k"] = 4
        cfg["selection_require_heldout_agreement"] = True
        cfg["selection_nested_top_k"] = 0
        cfg["selection_require_nested_response"] = False
    if profile == "writeback_v5_thresholded_real_branch":
        cfg["min_branch_target"] = 0.25
        cfg["min_parent_div_target"] = 0.032
        cfg["min_meso_target"] = 0.002
    if profile == "writeback_v6_naked_survival_curriculum":
        cfg["min_branch_target"] = 0.25
        cfg["min_writeback_target"] = 0.03
        cfg["min_parent_div_target"] = 0.040
        cfg["min_meso_target"] = 0.003
        cfg["min_live_child_target"] = 0.16
        cfg["min_naked_branch_target"] = 0.20
        cfg["min_naked_writeback_target"] = 0.03
        cfg["min_naked_parent_div_target"] = 0.035
        cfg["min_naked_meso_target"] = 0.002
        cfg["min_naked_live_child_target"] = 0.18
        cfg["naked_defect_gate"] = 0.10
    if profile == "writeback_v7_audio_recovery_with_branch_floor":
        cfg["min_branch_target"] = 0.25
        cfg["min_writeback_target"] = 0.03
        cfg["min_parent_div_target"] = 0.040
        cfg["min_meso_target"] = 0.003
        cfg["min_live_child_target"] = 0.16
        cfg["min_naked_branch_target"] = 0.333
        cfg["min_naked_writeback_target"] = 0.075
        cfg["min_naked_parent_div_target"] = 0.040
        cfg["min_naked_meso_target"] = 0.0025
        cfg["min_naked_live_child_target"] = 0.30
        cfg["naked_defect_gate"] = 0.10
    if profile == "writeback_v8_recovery_with_law_floor":
        cfg["min_branch_target"] = 0.25
        cfg["min_writeback_target"] = 0.03
        cfg["min_parent_div_target"] = 0.040
        cfg["min_meso_target"] = 0.003
        cfg["min_live_child_target"] = 0.16
        cfg["min_law_family_target"] = 1.5
        cfg["min_law_entropy_target"] = 0.12
        cfg["min_naked_branch_target"] = 0.333
        cfg["min_naked_writeback_target"] = 0.075
        cfg["min_naked_parent_div_target"] = 0.040
        cfg["min_naked_meso_target"] = 0.0025
        cfg["min_naked_live_child_target"] = 0.30
        cfg["min_naked_law_family_target"] = 2.0
        cfg["min_naked_law_entropy_target"] = 0.12
        cfg["naked_defect_gate"] = 0.10
    if profile == "writeback_v9_aligned_probe_recovery":
        cfg["min_branch_target"] = 0.25
        cfg["min_writeback_target"] = 0.03
        cfg["min_parent_div_target"] = 0.040
        cfg["min_meso_target"] = 0.003
        cfg["min_live_child_target"] = 0.16
        cfg["min_law_family_target"] = 1.8
        cfg["min_law_entropy_target"] = 0.14
        cfg["min_naked_branch_target"] = 0.333
        cfg["min_naked_writeback_target"] = 0.075
        cfg["min_naked_parent_div_target"] = 0.040
        cfg["min_naked_meso_target"] = 0.003
        cfg["min_naked_live_child_target"] = 0.30
        cfg["min_naked_law_family_target"] = 2.0
        cfg["min_naked_law_entropy_target"] = 0.18
        cfg["naked_defect_gate"] = 0.10
    if profile == "writeback_v11_hard_gate_seeded":
        cfg["min_branch_target"] = 0.25
        cfg["min_writeback_target"] = 0.03
        cfg["min_parent_div_target"] = 0.040
        cfg["min_meso_target"] = 0.003
        cfg["min_live_child_target"] = 0.16
        cfg["min_law_family_target"] = 1.6
        cfg["min_law_entropy_target"] = 0.12
        cfg["min_naked_branch_target"] = 0.333
        cfg["min_naked_writeback_target"] = 0.078
        cfg["min_naked_parent_div_target"] = 0.040
        cfg["min_naked_meso_target"] = 0.003
        cfg["min_naked_live_child_target"] = 0.30
        cfg["min_naked_law_family_target"] = 2.0
        cfg["min_naked_law_entropy_target"] = 0.12
        cfg["naked_defect_gate"] = 0.10
    if profile == "writeback_v12_verified_gate_seeded":
        cfg["enabled"] = True
        cfg["selection_probe_repeats"] = 3
        cfg["min_branch_target"] = 0.25
        cfg["min_writeback_target"] = 0.03
        cfg["min_parent_div_target"] = 0.040
        cfg["min_meso_target"] = 0.003
        cfg["min_live_child_target"] = 0.16
        cfg["min_law_family_target"] = 1.6
        cfg["min_law_entropy_target"] = 0.12
        cfg["min_naked_branch_target"] = 0.333
        cfg["min_naked_writeback_target"] = 0.078
        cfg["min_naked_parent_div_target"] = 0.040
        cfg["min_naked_meso_target"] = 0.002
        cfg["min_naked_live_child_target"] = 0.30
        cfg["min_naked_law_family_target"] = 2.0
        cfg["min_naked_law_entropy_target"] = 0.12
        cfg["naked_defect_gate"] = 0.10
    if profile == "writeback_v13_targeted_replay_recovery":
        cfg["enabled"] = True
        cfg["selection_probe_repeats"] = 3
        cfg["min_branch_target"] = 0.25
        cfg["min_writeback_target"] = 0.03
        cfg["min_parent_div_target"] = 0.040
        cfg["min_meso_target"] = 0.003
        cfg["min_live_child_target"] = 0.16
        cfg["min_law_family_target"] = 1.5
        cfg["min_law_entropy_target"] = 0.10
        cfg["min_naked_branch_target"] = 0.333
        cfg["min_naked_writeback_target"] = 0.080
        cfg["min_naked_parent_div_target"] = 0.038
        cfg["min_naked_meso_target"] = 0.002
        cfg["min_naked_live_child_target"] = 0.30
        cfg["min_naked_law_family_target"] = 2.0
        cfg["min_naked_law_entropy_target"] = 0.10
        cfg["naked_defect_gate"] = 0.10
    if profile == "writeback_v15_nested_divergence_parentmix":
        cfg["enabled"] = True
        cfg["selection_probe_repeats"] = 3
        cfg["min_branch_target"] = 0.25
        cfg["min_writeback_target"] = 0.03
        cfg["min_parent_div_target"] = 0.040
        cfg["min_meso_target"] = 0.0032
        cfg["min_live_child_target"] = 0.18
        cfg["min_law_family_target"] = 2.0
        cfg["min_law_entropy_target"] = 0.16
        cfg["min_naked_branch_target"] = 0.333
        cfg["min_naked_writeback_target"] = 0.080
        cfg["min_naked_parent_div_target"] = 0.040
        cfg["min_naked_meso_target"] = 0.0030
        cfg["min_naked_live_child_target"] = 0.30
        cfg["min_naked_law_family_target"] = 2.0
        cfg["min_naked_law_entropy_target"] = 0.16
        cfg["naked_defect_gate"] = 0.10
    if profile == "branch_identity_guard_v1":
        cfg["enabled"] = True
        cfg["selection_probe_repeats"] = 3
        cfg["min_branch_target"] = 0.25
        cfg["min_writeback_target"] = 0.03
        cfg["min_parent_div_target"] = 0.045
        cfg["min_meso_target"] = 0.0032
        cfg["min_live_child_target"] = 0.18
        cfg["min_law_family_target"] = 2.0
        cfg["min_law_entropy_target"] = 0.18
        cfg["min_naked_branch_target"] = 0.333
        cfg["min_naked_writeback_target"] = 0.080
        cfg["min_naked_parent_div_target"] = 0.045
        cfg["min_naked_meso_target"] = 0.0030
        cfg["min_naked_live_child_target"] = 0.30
        cfg["min_naked_law_family_target"] = 2.0
        cfg["min_naked_law_entropy_target"] = 0.18
        cfg["naked_defect_gate"] = 0.10
    if profile == "branch_writeback_floor_v1":
        cfg["enabled"] = True
        cfg["selection_probe_repeats"] = 3
        cfg["min_branch_target"] = 0.25
        cfg["min_writeback_target"] = 0.035
        cfg["min_parent_div_target"] = 0.040
        cfg["min_meso_target"] = 0.0032
        cfg["min_live_child_target"] = 0.20
        cfg["min_law_family_target"] = 2.0
        cfg["min_law_entropy_target"] = 0.16
        cfg["min_naked_branch_target"] = 0.333
        cfg["min_naked_writeback_target"] = 0.090
        cfg["min_naked_parent_div_target"] = 0.040
        cfg["min_naked_meso_target"] = 0.0030
        cfg["min_naked_live_child_target"] = 0.30
        cfg["min_naked_law_family_target"] = 2.0
        cfg["min_naked_law_entropy_target"] = 0.16
        cfg["naked_defect_gate"] = 0.10
    if profile == "branch_balance_guard_v1":
        cfg["enabled"] = True
        cfg["selection_probe_repeats"] = 3
        cfg["min_branch_target"] = 0.25
        cfg["min_writeback_target"] = 0.03
        cfg["min_parent_div_target"] = 0.040
        cfg["min_meso_target"] = 0.0032
        cfg["min_live_child_target"] = 0.18
        cfg["min_law_family_target"] = 2.0
        cfg["min_law_entropy_target"] = 0.16
        cfg["min_naked_branch_target"] = 0.333
        cfg["min_naked_writeback_target"] = 0.080
        cfg["min_naked_parent_div_target"] = 0.040
        cfg["min_naked_meso_target"] = 0.0030
        cfg["min_naked_live_child_target"] = 0.30
        cfg["min_naked_law_family_target"] = 2.0
        cfg["min_naked_law_entropy_target"] = 0.16
        cfg["naked_defect_gate"] = 0.10
    if profile in {"branch_recovery_soft_guard_v1", "naked_phase_nested_recovery_v1"}:
        cfg["enabled"] = True
        cfg["selection_probe_repeats"] = 3
        cfg["min_branch_target"] = 0.24
        cfg["min_phase_only_branch_target"] = 0.010
        cfg["min_phase_only_distinctness_target"] = 0.030
        cfg["min_writeback_target"] = 0.028
        cfg["min_parent_div_target"] = 0.038
        cfg["min_meso_target"] = 0.0030
        cfg["min_live_child_target"] = 0.17
        cfg["min_law_family_target"] = 2.0
        cfg["min_law_entropy_target"] = 0.15
        cfg["min_naked_branch_target"] = 0.320
        cfg["min_naked_phase_only_branch_target"] = 0.002
        cfg["min_naked_phase_only_distinctness_target"] = 0.003
        cfg["min_naked_writeback_target"] = 0.078
        cfg["min_naked_parent_div_target"] = 0.038
        cfg["min_naked_meso_target"] = 0.0028
        cfg["min_naked_live_child_target"] = 0.28
        cfg["min_naked_law_family_target"] = 2.0
        cfg["min_naked_law_entropy_target"] = 0.15
        cfg["naked_defect_gate"] = 0.10
        cfg["nested_probe_top_k"] = 4
        cfg["selection_nested_top_k"] = 3
        cfg["selection_nested_eval_top_k"] = 4
        cfg["selection_require_nested_response"] = True
        cfg["selection_allow_missing_nested_fallback"] = True
        cfg["selection_nested_depth"] = 3
        cfg["selection_nested_fork_depth"] = 1
        cfg["selection_nested_fork_selector"] = "max_live_child"
        cfg["selection_nested_live_child_threshold"] = 0.03
        cfg["selection_nested_mode"] = "native_multimode_childworld"
        cfg["selection_nested_case_source"] = "seeded_branch_active"
        cfg["selection_nested_case_suite"] = "branch_recovery_soft_guard_v1"
        cfg["selection_nested_min_response"] = -1.87
        cfg["selection_nested_min_active_fraction"] = 0.02
        cfg["selection_nested_min_meso_response"] = 0.0001
        cfg["selection_nested_min_sibling_fraction"] = 0.0
        cfg["selection_nested_max_over_rigid"] = 0.95
        cfg["selection_nested_case_plan"] = _seeded_nested_case_suite("branch_recovery_soft_guard_v1")
        if profile == "naked_phase_nested_recovery_v1":
            cfg["min_phase_only_branch_target"] = 0.014
            cfg["min_phase_only_distinctness_target"] = 0.040
            cfg["min_naked_phase_only_branch_target"] = 0.006
            cfg["min_naked_phase_only_distinctness_target"] = 0.012
            cfg["nested_probe_top_k"] = 6
            cfg["selection_nested_min_active_fraction"] = 0.025
            cfg["selection_nested_min_meso_response"] = 0.00015
    if profile == "writeback_v16_benchmark_guard_nested":
        cfg["enabled"] = True
        cfg["selection_probe_repeats"] = 3
        cfg["min_branch_target"] = 0.25
        cfg["min_writeback_target"] = 0.03
        cfg["min_parent_div_target"] = 0.040
        cfg["min_meso_target"] = 0.0030
        cfg["min_live_child_target"] = 0.18
        cfg["min_law_family_target"] = 2.0
        cfg["min_law_entropy_target"] = 0.14
        cfg["min_naked_branch_target"] = 0.333
        cfg["min_naked_writeback_target"] = 0.080
        cfg["min_naked_parent_div_target"] = 0.040
        cfg["min_naked_meso_target"] = 0.0030
        cfg["min_naked_live_child_target"] = 0.30
        cfg["min_naked_law_family_target"] = 2.0
        cfg["min_naked_law_entropy_target"] = 0.14
        cfg["naked_defect_gate"] = 0.10
    if profile == "writeback_v17_nested_response_gate":
        cfg["enabled"] = True
        cfg["selection_probe_repeats"] = 3
        cfg["min_branch_target"] = 0.25
        cfg["min_writeback_target"] = 0.03
        cfg["min_parent_div_target"] = 0.040
        cfg["min_meso_target"] = 0.0030
        cfg["min_live_child_target"] = 0.18
        cfg["min_law_family_target"] = 2.0
        cfg["min_law_entropy_target"] = 0.14
        cfg["min_naked_branch_target"] = 0.333
        cfg["min_naked_writeback_target"] = 0.080
        cfg["min_naked_parent_div_target"] = 0.040
        cfg["min_naked_meso_target"] = 0.0030
        cfg["min_naked_live_child_target"] = 0.30
        cfg["min_naked_law_family_target"] = 2.0
        cfg["min_naked_law_entropy_target"] = 0.14
        cfg["naked_defect_gate"] = 0.10
        cfg["selection_nested_top_k"] = 2
        cfg["selection_require_nested_response"] = True
        cfg["selection_nested_depth"] = 3
        cfg["selection_nested_fork_depth"] = 1
        cfg["selection_nested_fork_selector"] = "first_live_child"
        cfg["selection_nested_live_child_threshold"] = 0.05
        cfg["selection_nested_mode"] = "active_packets"
        cfg["selection_nested_min_response"] = -1.87
        cfg["selection_nested_min_active_fraction"] = 0.05
        cfg["selection_nested_min_meso_response"] = 0.0005
        cfg["selection_nested_min_sibling_fraction"] = 0.0
        cfg["selection_nested_max_over_rigid"] = 0.95
    if profile in {"writeback_v18_nested_probe_objective", "writeback_v19_seeded_nested_substrate", "writeback_v20_child_record_perturb"}:
        cfg["enabled"] = True
        cfg["selection_probe_repeats"] = 3
        cfg["min_branch_target"] = 0.25
        cfg["min_writeback_target"] = 0.03
        cfg["min_parent_div_target"] = 0.040
        cfg["min_meso_target"] = 0.0030
        cfg["min_live_child_target"] = 0.18
        cfg["min_law_family_target"] = 2.0
        cfg["min_law_entropy_target"] = 0.14
        cfg["min_naked_branch_target"] = 0.333
        cfg["min_naked_writeback_target"] = 0.080
        cfg["min_naked_parent_div_target"] = 0.040
        cfg["min_naked_meso_target"] = 0.0030
        cfg["min_naked_live_child_target"] = 0.30
        cfg["min_naked_law_family_target"] = 2.0
        cfg["min_naked_law_entropy_target"] = 0.14
        cfg["naked_defect_gate"] = 0.10
        cfg["nested_probe_top_k"] = 4
        cfg["selection_nested_top_k"] = 2
        cfg["selection_require_nested_response"] = True
        cfg["selection_nested_depth"] = 3
        cfg["selection_nested_fork_depth"] = 1
        cfg["selection_nested_fork_selector"] = "first_live_child"
        cfg["selection_nested_live_child_threshold"] = 0.05
        cfg["selection_nested_mode"] = "active_packets"
        cfg["selection_nested_min_response"] = -1.87
        cfg["selection_nested_min_active_fraction"] = 0.05
        cfg["selection_nested_min_meso_response"] = 0.0005
        cfg["selection_nested_min_sibling_fraction"] = 0.0
        cfg["selection_nested_max_over_rigid"] = 0.95
    if profile in {"writeback_v19_seeded_nested_substrate", "writeback_v20_child_record_perturb"}:
        cfg["selection_nested_case_source"] = "seeded_branch_active"
        cfg["selection_nested_mode"] = "native_multimode_childworld"
        cfg["selection_nested_fork_selector"] = "max_live_child"
        cfg["selection_nested_live_child_threshold"] = 0.03
        cfg["selection_nested_case_plan"] = [
            {
                "name": "naked_seed_9200",
                "seed_source": "naked_rafa",
                "seed": 9200,
                "time_steps": 128,
                "warmup_depth": 5,
                "continuation_depth": 3,
                "require_live_child_start": True,
            },
            {
                "name": "naked_seed_9201",
                "seed_source": "naked_rafa",
                "seed": 9201,
                "time_steps": 128,
                "warmup_depth": 5,
                "continuation_depth": 3,
                "require_live_child_start": True,
            },
            {
                "name": "naked_seed_9202",
                "seed_source": "naked_rafa",
                "seed": 9202,
                "time_steps": 128,
                "warmup_depth": 5,
                "continuation_depth": 3,
                "require_live_child_start": True,
            },
        ]
    if profile == "writeback_v20_child_record_perturb":
        cfg["nested_probe_top_k"] = 6
        cfg["selection_nested_top_k"] = 3
        cfg["selection_nested_min_active_fraction"] = 0.02
        cfg["selection_nested_min_meso_response"] = 0.0001
    if profile == "mode_replace_conversion_v1":
        cfg["enabled"] = True
        cfg["selection_probe_repeats"] = 3
        cfg["min_branch_target"] = 0.24
        cfg["min_writeback_target"] = 0.02
        cfg["min_parent_div_target"] = 0.035
        cfg["min_meso_target"] = 0.0028
        cfg["min_live_child_target"] = 0.16
        cfg["nested_probe_top_k"] = 6
        cfg["selection_nested_top_k"] = 3
        cfg["selection_nested_eval_top_k"] = 4
        cfg["selection_require_nested_response"] = True
        cfg["selection_allow_missing_nested_fallback"] = True
        cfg["selection_nested_depth"] = 3
        cfg["selection_nested_fork_depth"] = 1
        cfg["selection_nested_fork_selector"] = "max_live_child"
        cfg["selection_nested_live_child_threshold"] = 0.03
        cfg["selection_nested_mode"] = "native_multimode_childworld"
        cfg["selection_nested_case_source"] = "seeded_branch_active"
        cfg["selection_nested_case_suite"] = "seeded_branch_active"
        cfg["selection_nested_case_plan"] = _seeded_nested_case_suite("seeded_branch_active")
        cfg["selection_nested_min_active_fraction"] = 0.02
        cfg["selection_nested_min_meso_response"] = 0.0001
        cfg["selection_nested_min_sibling_fraction"] = 0.0
        cfg["selection_nested_min_mode_replace_readiness"] = 0.40
        cfg["selection_nested_min_child_record_survival_score"] = 0.45
        cfg["selection_nested_min_mode_replace_conversion_score"] = 0.0
        cfg["selection_nested_max_direct_readout_dependency"] = 0.05
        cfg["selection_nested_max_final_direct_mix_shortcut"] = 0.20
        cfg["selection_nested_max_world_jump"] = 0.0
        cfg["selection_nested_max_over_rigid"] = 0.95
    return cfg


def _case_weights() -> dict[str, float]:
    return {
        "clarinet": 2.8,
        "metal-clang": 2.2,
        "mystic-chanting": 1.7,
        "anvil-impact": 1.5,
        "airplane": 1.3,
    }


def _extra_train_wavs() -> list[str]:
    return [
        str(ROOT / "wav_files" / "freewavesamples_ensoniq-zr-76-clarinet-c5_19f8dd5ddd.wav"),
    ]


def _extra_val_wavs() -> list[str]:
    return [
        str(ROOT / "wav_files" / "soundbible_mystic-chanting-4_7962fb220f.wav"),
        str(ROOT / "wav_files" / "soundbible_spooky-drone_7cc45e3229.wav"),
        str(ROOT / "wav_files" / "soundbible_muscle-car_2f21ca885f.wav"),
    ]


def _prepare_childworld_init_config(init_config_path: Path, out_dir: Path, profile: str = "childworld_v1") -> Path:
    payload = json.loads(init_config_path.read_text(encoding="utf-8-sig"))
    cfg = payload["config"] if "config" in payload else payload
    cfg["branching_mode"] = "native_multimode_childworld"
    cfg.setdefault("child_spawn_threshold", 0.34)
    cfg.setdefault("child_max_worlds", 4)
    cfg.setdefault("child_min_age_for_writeback", 2)
    cfg.setdefault("child_support_window", 12)
    cfg.setdefault("child_support_decay", 0.12)
    cfg.setdefault("child_survival_coherence_weight", 0.42)
    cfg.setdefault("child_survival_qtrace_weight", 0.24)
    cfg.setdefault("child_survival_residue_penalty", 0.22)
    cfg.setdefault("child_writeback_gain", 0.32)
    cfg.setdefault("child_writeback_rank", 12)
    cfg.setdefault("child_writeback_temperature", 0.85)
    cfg.setdefault("child_writeback_budget", 1.25)
    cfg.setdefault("child_kill_threshold", 0.08)
    if profile == "writeback_v2":
        cfg["child_spawn_threshold"] = min(float(cfg.get("child_spawn_threshold", 0.34)), 0.28)
        cfg["child_max_worlds"] = max(int(cfg.get("child_max_worlds", 4)), 6)
        cfg["child_min_age_for_writeback"] = 1
        cfg["child_support_window"] = max(int(cfg.get("child_support_window", 12)), 18)
        cfg["child_support_decay"] = min(float(cfg.get("child_support_decay", 0.12)), 0.08)
        cfg["child_survival_coherence_weight"] = max(float(cfg.get("child_survival_coherence_weight", 0.42)), 0.46)
        cfg["child_survival_qtrace_weight"] = max(float(cfg.get("child_survival_qtrace_weight", 0.24)), 0.28)
        cfg["child_survival_residue_penalty"] = min(float(cfg.get("child_survival_residue_penalty", 0.22)), 0.18)
        cfg["child_writeback_gain"] = max(float(cfg.get("child_writeback_gain", 0.32)), 0.55)
        cfg["child_writeback_rank"] = max(int(cfg.get("child_writeback_rank", 12)), 16)
        cfg["child_writeback_temperature"] = max(float(cfg.get("child_writeback_temperature", 0.85)), 1.10)
        cfg["child_writeback_budget"] = max(float(cfg.get("child_writeback_budget", 1.25)), 1.8)
        cfg["child_kill_threshold"] = min(float(cfg.get("child_kill_threshold", 0.08)), 0.03)
    elif profile == "writeback_v3_identity":
        cfg["child_spawn_threshold"] = min(float(cfg.get("child_spawn_threshold", 0.34)), 0.24)
        cfg["child_max_worlds"] = max(int(cfg.get("child_max_worlds", 4)), 6)
        cfg["child_min_age_for_writeback"] = 1
        cfg["child_support_window"] = max(int(cfg.get("child_support_window", 12)), 20)
        cfg["child_support_decay"] = min(float(cfg.get("child_support_decay", 0.12)), 0.06)
        cfg["child_survival_coherence_weight"] = max(float(cfg.get("child_survival_coherence_weight", 0.42)), 0.48)
        cfg["child_survival_qtrace_weight"] = max(float(cfg.get("child_survival_qtrace_weight", 0.24)), 0.30)
        cfg["child_survival_residue_penalty"] = min(float(cfg.get("child_survival_residue_penalty", 0.22)), 0.16)
        cfg["child_writeback_gain"] = max(float(cfg.get("child_writeback_gain", 0.32)), 0.48)
        cfg["child_writeback_rank"] = max(int(cfg.get("child_writeback_rank", 12)), 16)
        cfg["child_writeback_temperature"] = max(float(cfg.get("child_writeback_temperature", 0.85)), 1.05)
        cfg["child_writeback_budget"] = max(float(cfg.get("child_writeback_budget", 1.25)), 1.6)
        cfg["child_kill_threshold"] = min(float(cfg.get("child_kill_threshold", 0.08)), 0.02)
    elif profile == "writeback_v4_real_anchor_transfer":
        cfg["child_spawn_threshold"] = min(float(cfg.get("child_spawn_threshold", 0.34)), 0.22)
        cfg["child_max_worlds"] = max(int(cfg.get("child_max_worlds", 4)), 6)
        cfg["child_min_age_for_writeback"] = 1
        cfg["child_support_window"] = max(int(cfg.get("child_support_window", 12)), 20)
        cfg["child_support_decay"] = min(float(cfg.get("child_support_decay", 0.12)), 0.06)
        cfg["child_survival_coherence_weight"] = max(float(cfg.get("child_survival_coherence_weight", 0.42)), 0.50)
        cfg["child_survival_qtrace_weight"] = max(float(cfg.get("child_survival_qtrace_weight", 0.24)), 0.32)
        cfg["child_survival_residue_penalty"] = min(float(cfg.get("child_survival_residue_penalty", 0.22)), 0.15)
        cfg["child_writeback_gain"] = max(float(cfg.get("child_writeback_gain", 0.32)), 0.50)
        cfg["child_writeback_rank"] = max(int(cfg.get("child_writeback_rank", 12)), 18)
        cfg["child_writeback_temperature"] = max(float(cfg.get("child_writeback_temperature", 0.85)), 1.08)
        cfg["child_writeback_budget"] = max(float(cfg.get("child_writeback_budget", 1.25)), 1.8)
        cfg["child_kill_threshold"] = min(float(cfg.get("child_kill_threshold", 0.08)), 0.02)
    elif profile == "writeback_v5_thresholded_real_branch":
        cfg["child_spawn_threshold"] = min(float(cfg.get("child_spawn_threshold", 0.34)), 0.22)
        cfg["child_max_worlds"] = max(int(cfg.get("child_max_worlds", 4)), 6)
        cfg["child_min_age_for_writeback"] = 1
        cfg["child_support_window"] = max(int(cfg.get("child_support_window", 12)), 20)
        cfg["child_support_decay"] = min(float(cfg.get("child_support_decay", 0.12)), 0.06)
        cfg["child_survival_coherence_weight"] = max(float(cfg.get("child_survival_coherence_weight", 0.42)), 0.50)
        cfg["child_survival_qtrace_weight"] = max(float(cfg.get("child_survival_qtrace_weight", 0.24)), 0.32)
        cfg["child_survival_residue_penalty"] = min(float(cfg.get("child_survival_residue_penalty", 0.22)), 0.15)
        cfg["child_writeback_gain"] = max(float(cfg.get("child_writeback_gain", 0.32)), 0.50)
        cfg["child_writeback_rank"] = max(int(cfg.get("child_writeback_rank", 12)), 18)
        cfg["child_writeback_temperature"] = max(float(cfg.get("child_writeback_temperature", 0.85)), 1.08)
        cfg["child_writeback_budget"] = max(float(cfg.get("child_writeback_budget", 1.25)), 1.8)
        cfg["child_kill_threshold"] = min(float(cfg.get("child_kill_threshold", 0.08)), 0.02)
        cfg["child_branch_parent_threshold"] = min(float(cfg.get("child_branch_parent_threshold", cfg.get("real_branch_threshold", 0.12))), 0.032)
        cfg["child_branch_writeback_threshold"] = max(float(cfg.get("child_branch_writeback_threshold", 0.0)), 0.02)
        cfg["child_branch_meso_threshold"] = max(float(cfg.get("child_branch_meso_threshold", 0.0)), 0.0018)
        cfg["child_branch_live_threshold"] = max(float(cfg.get("child_branch_live_threshold", 0.0)), 0.18)
    elif profile == "writeback_v6_naked_survival_curriculum":
        cfg["child_spawn_threshold"] = min(float(cfg.get("child_spawn_threshold", 0.34)), 0.20)
        cfg["child_max_worlds"] = max(int(cfg.get("child_max_worlds", 4)), 6)
        cfg["child_min_age_for_writeback"] = 1
        cfg["child_support_window"] = max(int(cfg.get("child_support_window", 12)), 22)
        cfg["child_support_decay"] = min(float(cfg.get("child_support_decay", 0.12)), 0.05)
        cfg["child_survival_coherence_weight"] = max(float(cfg.get("child_survival_coherence_weight", 0.42)), 0.52)
        cfg["child_survival_qtrace_weight"] = max(float(cfg.get("child_survival_qtrace_weight", 0.24)), 0.34)
        cfg["child_survival_residue_penalty"] = min(float(cfg.get("child_survival_residue_penalty", 0.22)), 0.14)
        cfg["child_writeback_gain"] = max(float(cfg.get("child_writeback_gain", 0.32)), 0.56)
        cfg["child_writeback_rank"] = max(int(cfg.get("child_writeback_rank", 12)), 18)
        cfg["child_writeback_temperature"] = max(float(cfg.get("child_writeback_temperature", 0.85)), 1.10)
        cfg["child_writeback_budget"] = max(float(cfg.get("child_writeback_budget", 1.25)), 1.9)
        cfg["child_kill_threshold"] = min(float(cfg.get("child_kill_threshold", 0.08)), 0.02)
        cfg["child_branch_parent_threshold"] = min(float(cfg.get("child_branch_parent_threshold", cfg.get("real_branch_threshold", 0.12))), 0.030)
        cfg["child_branch_writeback_threshold"] = max(float(cfg.get("child_branch_writeback_threshold", 0.0)), 0.025)
        cfg["child_branch_meso_threshold"] = max(float(cfg.get("child_branch_meso_threshold", 0.0)), 0.0020)
        cfg["child_branch_live_threshold"] = max(float(cfg.get("child_branch_live_threshold", 0.0)), 0.20)
    elif profile == "writeback_v7_audio_recovery_with_branch_floor":
        cfg["child_spawn_threshold"] = min(float(cfg.get("child_spawn_threshold", 0.34)), 0.21)
        cfg["child_max_worlds"] = max(int(cfg.get("child_max_worlds", 4)), 6)
        cfg["child_min_age_for_writeback"] = 1
        cfg["child_support_window"] = max(int(cfg.get("child_support_window", 12)), 20)
        cfg["child_support_decay"] = min(float(cfg.get("child_support_decay", 0.12)), 0.05)
        cfg["child_survival_coherence_weight"] = max(float(cfg.get("child_survival_coherence_weight", 0.42)), 0.50)
        cfg["child_survival_qtrace_weight"] = max(float(cfg.get("child_survival_qtrace_weight", 0.24)), 0.34)
        cfg["child_survival_residue_penalty"] = min(float(cfg.get("child_survival_residue_penalty", 0.22)), 0.14)
        cfg["child_writeback_gain"] = max(float(cfg.get("child_writeback_gain", 0.32)), 0.54)
        cfg["child_writeback_rank"] = max(int(cfg.get("child_writeback_rank", 12)), 18)
        cfg["child_writeback_temperature"] = max(float(cfg.get("child_writeback_temperature", 0.85)), 1.08)
        cfg["child_writeback_budget"] = max(float(cfg.get("child_writeback_budget", 1.25)), 1.9)
        cfg["child_kill_threshold"] = min(float(cfg.get("child_kill_threshold", 0.08)), 0.02)
        cfg["child_branch_parent_threshold"] = min(float(cfg.get("child_branch_parent_threshold", cfg.get("real_branch_threshold", 0.12))), 0.030)
        cfg["child_branch_writeback_threshold"] = max(float(cfg.get("child_branch_writeback_threshold", 0.0)), 0.025)
        cfg["child_branch_meso_threshold"] = max(float(cfg.get("child_branch_meso_threshold", 0.0)), 0.0020)
        cfg["child_branch_live_threshold"] = max(float(cfg.get("child_branch_live_threshold", 0.0)), 0.20)
    elif profile == "writeback_v8_recovery_with_law_floor":
        cfg["child_spawn_threshold"] = min(float(cfg.get("child_spawn_threshold", 0.34)), 0.21)
        cfg["child_max_worlds"] = max(int(cfg.get("child_max_worlds", 4)), 6)
        cfg["child_min_age_for_writeback"] = 1
        cfg["child_support_window"] = max(int(cfg.get("child_support_window", 12)), 20)
        cfg["child_support_decay"] = min(float(cfg.get("child_support_decay", 0.12)), 0.05)
        cfg["child_survival_coherence_weight"] = max(float(cfg.get("child_survival_coherence_weight", 0.42)), 0.50)
        cfg["child_survival_qtrace_weight"] = max(float(cfg.get("child_survival_qtrace_weight", 0.24)), 0.34)
        cfg["child_survival_residue_penalty"] = min(float(cfg.get("child_survival_residue_penalty", 0.22)), 0.14)
        cfg["child_writeback_gain"] = max(float(cfg.get("child_writeback_gain", 0.32)), 0.54)
        cfg["child_writeback_rank"] = max(int(cfg.get("child_writeback_rank", 12)), 18)
        cfg["child_writeback_temperature"] = max(float(cfg.get("child_writeback_temperature", 0.85)), 1.08)
        cfg["child_writeback_budget"] = max(float(cfg.get("child_writeback_budget", 1.25)), 1.9)
        cfg["child_kill_threshold"] = min(float(cfg.get("child_kill_threshold", 0.08)), 0.02)
        cfg["child_branch_parent_threshold"] = min(float(cfg.get("child_branch_parent_threshold", cfg.get("real_branch_threshold", 0.12))), 0.030)
        cfg["child_branch_writeback_threshold"] = max(float(cfg.get("child_branch_writeback_threshold", 0.0)), 0.025)
        cfg["child_branch_meso_threshold"] = max(float(cfg.get("child_branch_meso_threshold", 0.0)), 0.0020)
        cfg["child_branch_live_threshold"] = max(float(cfg.get("child_branch_live_threshold", 0.0)), 0.20)
    elif profile == "writeback_v9_aligned_probe_recovery":
        cfg["child_spawn_threshold"] = min(float(cfg.get("child_spawn_threshold", 0.34)), 0.20)
        cfg["child_max_worlds"] = max(int(cfg.get("child_max_worlds", 4)), 6)
        cfg["child_min_age_for_writeback"] = 1
        cfg["child_support_window"] = max(int(cfg.get("child_support_window", 12)), 22)
        cfg["child_support_decay"] = min(float(cfg.get("child_support_decay", 0.12)), 0.05)
        cfg["child_survival_coherence_weight"] = max(float(cfg.get("child_survival_coherence_weight", 0.42)), 0.52)
        cfg["child_survival_qtrace_weight"] = max(float(cfg.get("child_survival_qtrace_weight", 0.24)), 0.35)
        cfg["child_survival_residue_penalty"] = min(float(cfg.get("child_survival_residue_penalty", 0.22)), 0.14)
        cfg["child_writeback_gain"] = max(float(cfg.get("child_writeback_gain", 0.32)), 0.58)
        cfg["child_writeback_rank"] = max(int(cfg.get("child_writeback_rank", 12)), 18)
        cfg["child_writeback_temperature"] = max(float(cfg.get("child_writeback_temperature", 0.85)), 1.10)
        cfg["child_writeback_budget"] = max(float(cfg.get("child_writeback_budget", 1.25)), 1.95)
        cfg["child_kill_threshold"] = min(float(cfg.get("child_kill_threshold", 0.08)), 0.02)
        cfg["child_branch_parent_threshold"] = min(float(cfg.get("child_branch_parent_threshold", cfg.get("real_branch_threshold", 0.12))), 0.030)
        cfg["child_branch_writeback_threshold"] = max(float(cfg.get("child_branch_writeback_threshold", 0.0)), 0.025)
        cfg["child_branch_meso_threshold"] = max(float(cfg.get("child_branch_meso_threshold", 0.0)), 0.0020)
        cfg["child_branch_live_threshold"] = max(float(cfg.get("child_branch_live_threshold", 0.0)), 0.20)
    elif profile == "writeback_v11_hard_gate_seeded":
        cfg["child_spawn_threshold"] = min(float(cfg.get("child_spawn_threshold", 0.34)), 0.20)
        cfg["child_max_worlds"] = max(int(cfg.get("child_max_worlds", 4)), 6)
        cfg["child_min_age_for_writeback"] = 1
        cfg["child_support_window"] = max(int(cfg.get("child_support_window", 12)), 22)
        cfg["child_support_decay"] = min(float(cfg.get("child_support_decay", 0.12)), 0.05)
        cfg["child_survival_coherence_weight"] = max(float(cfg.get("child_survival_coherence_weight", 0.42)), 0.52)
        cfg["child_survival_qtrace_weight"] = max(float(cfg.get("child_survival_qtrace_weight", 0.24)), 0.35)
        cfg["child_survival_residue_penalty"] = min(float(cfg.get("child_survival_residue_penalty", 0.22)), 0.14)
        cfg["child_writeback_gain"] = max(float(cfg.get("child_writeback_gain", 0.32)), 0.58)
        cfg["child_writeback_rank"] = max(int(cfg.get("child_writeback_rank", 12)), 18)
        cfg["child_writeback_temperature"] = max(float(cfg.get("child_writeback_temperature", 0.85)), 1.10)
        cfg["child_writeback_budget"] = max(float(cfg.get("child_writeback_budget", 1.25)), 1.95)
        cfg["child_kill_threshold"] = min(float(cfg.get("child_kill_threshold", 0.08)), 0.02)
        cfg["child_branch_parent_threshold"] = min(float(cfg.get("child_branch_parent_threshold", cfg.get("real_branch_threshold", 0.12))), 0.030)
        cfg["child_branch_writeback_threshold"] = max(float(cfg.get("child_branch_writeback_threshold", 0.0)), 0.025)
        cfg["child_branch_meso_threshold"] = max(float(cfg.get("child_branch_meso_threshold", 0.0)), 0.0020)
        cfg["child_branch_live_threshold"] = max(float(cfg.get("child_branch_live_threshold", 0.0)), 0.20)
    elif profile == "writeback_v12_verified_gate_seeded":
        cfg["child_spawn_threshold"] = min(float(cfg.get("child_spawn_threshold", 0.34)), 0.19)
        cfg["child_max_worlds"] = max(int(cfg.get("child_max_worlds", 4)), 6)
        cfg["child_min_age_for_writeback"] = 1
        cfg["child_support_window"] = max(int(cfg.get("child_support_window", 12)), 24)
        cfg["child_support_decay"] = min(float(cfg.get("child_support_decay", 0.12)), 0.05)
        cfg["child_survival_coherence_weight"] = max(float(cfg.get("child_survival_coherence_weight", 0.42)), 0.54)
        cfg["child_survival_qtrace_weight"] = max(float(cfg.get("child_survival_qtrace_weight", 0.24)), 0.36)
        cfg["child_survival_residue_penalty"] = min(float(cfg.get("child_survival_residue_penalty", 0.22)), 0.14)
        cfg["child_writeback_gain"] = max(float(cfg.get("child_writeback_gain", 0.32)), 0.60)
        cfg["child_writeback_rank"] = max(int(cfg.get("child_writeback_rank", 12)), 18)
        cfg["child_writeback_temperature"] = max(float(cfg.get("child_writeback_temperature", 0.85)), 1.10)
        cfg["child_writeback_budget"] = max(float(cfg.get("child_writeback_budget", 1.25)), 1.95)
        cfg["child_kill_threshold"] = min(float(cfg.get("child_kill_threshold", 0.08)), 0.02)
        cfg["child_branch_parent_threshold"] = min(float(cfg.get("child_branch_parent_threshold", cfg.get("real_branch_threshold", 0.12))), 0.030)
        cfg["child_branch_writeback_threshold"] = max(float(cfg.get("child_branch_writeback_threshold", 0.0)), 0.025)
        cfg["child_branch_meso_threshold"] = max(float(cfg.get("child_branch_meso_threshold", 0.0)), 0.0020)
        cfg["child_branch_live_threshold"] = max(float(cfg.get("child_branch_live_threshold", 0.0)), 0.20)
    elif profile == "writeback_v13_targeted_replay_recovery":
        cfg["child_spawn_threshold"] = min(float(cfg.get("child_spawn_threshold", 0.34)), 0.19)
        cfg["child_max_worlds"] = max(int(cfg.get("child_max_worlds", 4)), 6)
        cfg["child_min_age_for_writeback"] = 1
        cfg["child_support_window"] = max(int(cfg.get("child_support_window", 12)), 24)
        cfg["child_support_decay"] = min(float(cfg.get("child_support_decay", 0.12)), 0.045)
        cfg["child_survival_coherence_weight"] = max(float(cfg.get("child_survival_coherence_weight", 0.42)), 0.54)
        cfg["child_survival_qtrace_weight"] = max(float(cfg.get("child_survival_qtrace_weight", 0.24)), 0.38)
        cfg["child_survival_residue_penalty"] = min(float(cfg.get("child_survival_residue_penalty", 0.22)), 0.14)
        cfg["child_writeback_gain"] = max(float(cfg.get("child_writeback_gain", 0.32)), 0.64)
        cfg["child_writeback_rank"] = max(int(cfg.get("child_writeback_rank", 12)), 20)
        cfg["child_writeback_temperature"] = max(float(cfg.get("child_writeback_temperature", 0.85)), 1.12)
        cfg["child_writeback_budget"] = max(float(cfg.get("child_writeback_budget", 1.25)), 2.05)
        cfg["child_kill_threshold"] = min(float(cfg.get("child_kill_threshold", 0.08)), 0.02)
        cfg["child_branch_parent_threshold"] = min(float(cfg.get("child_branch_parent_threshold", cfg.get("real_branch_threshold", 0.12))), 0.030)
        cfg["child_branch_writeback_threshold"] = max(float(cfg.get("child_branch_writeback_threshold", 0.0)), 0.028)
        cfg["child_branch_meso_threshold"] = max(float(cfg.get("child_branch_meso_threshold", 0.0)), 0.0020)
        cfg["child_branch_live_threshold"] = max(float(cfg.get("child_branch_live_threshold", 0.0)), 0.20)
    elif profile == "writeback_v15_nested_divergence_parentmix":
        cfg["child_spawn_threshold"] = min(float(cfg.get("child_spawn_threshold", 0.34)), 0.18)
        cfg["child_max_worlds"] = max(int(cfg.get("child_max_worlds", 4)), 6)
        cfg["child_min_age_for_writeback"] = 1
        cfg["child_support_window"] = max(int(cfg.get("child_support_window", 12)), 28)
        cfg["child_support_decay"] = min(float(cfg.get("child_support_decay", 0.12)), 0.040)
        cfg["child_survival_coherence_weight"] = max(float(cfg.get("child_survival_coherence_weight", 0.42)), 0.56)
        cfg["child_survival_qtrace_weight"] = max(float(cfg.get("child_survival_qtrace_weight", 0.24)), 0.40)
        cfg["child_survival_residue_penalty"] = min(float(cfg.get("child_survival_residue_penalty", 0.22)), 0.12)
        cfg["child_writeback_gain"] = max(float(cfg.get("child_writeback_gain", 0.32)), 0.66)
        cfg["child_writeback_rank"] = max(int(cfg.get("child_writeback_rank", 12)), 20)
        cfg["child_writeback_temperature"] = max(float(cfg.get("child_writeback_temperature", 0.85)), 1.14)
        cfg["child_writeback_budget"] = max(float(cfg.get("child_writeback_budget", 1.25)), 2.10)
        cfg["child_parent_mix"] = max(float(cfg.get("child_parent_mix", 0.15)), 0.22)
        cfg["child_parent_mix_early"] = min(max(float(cfg.get("child_parent_mix_early", 0.08)), 0.10), float(cfg["child_parent_mix"]) - 0.02)
        cfg["child_kill_threshold"] = min(float(cfg.get("child_kill_threshold", 0.08)), 0.018)
        cfg["child_branch_parent_threshold"] = min(float(cfg.get("child_branch_parent_threshold", cfg.get("real_branch_threshold", 0.12))), 0.028)
        cfg["child_branch_writeback_threshold"] = max(float(cfg.get("child_branch_writeback_threshold", 0.0)), 0.030)
        cfg["child_branch_meso_threshold"] = max(float(cfg.get("child_branch_meso_threshold", 0.0)), 0.0025)
        cfg["child_branch_live_threshold"] = max(float(cfg.get("child_branch_live_threshold", 0.0)), 0.22)
    elif profile == "writeback_v16_benchmark_guard_nested":
        cfg["child_spawn_threshold"] = min(float(cfg.get("child_spawn_threshold", 0.34)), 0.20)
        cfg["child_max_worlds"] = max(int(cfg.get("child_max_worlds", 4)), 5)
        cfg["child_min_age_for_writeback"] = 1
        cfg["child_support_window"] = max(int(cfg.get("child_support_window", 12)), 24)
        cfg["child_support_decay"] = min(float(cfg.get("child_support_decay", 0.12)), 0.050)
        cfg["child_survival_coherence_weight"] = max(float(cfg.get("child_survival_coherence_weight", 0.42)), 0.54)
        cfg["child_survival_qtrace_weight"] = max(float(cfg.get("child_survival_qtrace_weight", 0.24)), 0.36)
        cfg["child_survival_residue_penalty"] = min(float(cfg.get("child_survival_residue_penalty", 0.22)), 0.14)
        cfg["child_writeback_gain"] = max(float(cfg.get("child_writeback_gain", 0.32)), 0.56)
        cfg["child_writeback_rank"] = max(int(cfg.get("child_writeback_rank", 12)), 18)
        cfg["child_writeback_temperature"] = max(float(cfg.get("child_writeback_temperature", 0.85)), 1.06)
        cfg["child_writeback_budget"] = max(float(cfg.get("child_writeback_budget", 1.25)), 1.85)
        cfg["child_parent_mix"] = max(float(cfg.get("child_parent_mix", 0.15)), 0.18)
        cfg["child_parent_mix_early"] = min(max(float(cfg.get("child_parent_mix_early", 0.08)), 0.09), float(cfg["child_parent_mix"]) - 0.02)
        cfg["child_kill_threshold"] = min(float(cfg.get("child_kill_threshold", 0.08)), 0.024)
        cfg["child_branch_parent_threshold"] = min(float(cfg.get("child_branch_parent_threshold", cfg.get("real_branch_threshold", 0.12))), 0.030)
        cfg["child_branch_writeback_threshold"] = max(float(cfg.get("child_branch_writeback_threshold", 0.0)), 0.028)
        cfg["child_branch_meso_threshold"] = max(float(cfg.get("child_branch_meso_threshold", 0.0)), 0.0025)
        cfg["child_branch_live_threshold"] = max(float(cfg.get("child_branch_live_threshold", 0.0)), 0.20)
    elif profile == "writeback_v17_nested_response_gate":
        cfg["child_spawn_threshold"] = min(float(cfg.get("child_spawn_threshold", 0.34)), 0.20)
        cfg["child_max_worlds"] = max(int(cfg.get("child_max_worlds", 4)), 5)
        cfg["child_min_age_for_writeback"] = 1
        cfg["child_support_window"] = max(int(cfg.get("child_support_window", 12)), 24)
        cfg["child_support_decay"] = min(float(cfg.get("child_support_decay", 0.12)), 0.050)
        cfg["child_survival_coherence_weight"] = max(float(cfg.get("child_survival_coherence_weight", 0.42)), 0.54)
        cfg["child_survival_qtrace_weight"] = max(float(cfg.get("child_survival_qtrace_weight", 0.24)), 0.36)
        cfg["child_survival_residue_penalty"] = min(float(cfg.get("child_survival_residue_penalty", 0.22)), 0.14)
        cfg["child_writeback_gain"] = max(float(cfg.get("child_writeback_gain", 0.32)), 0.56)
        cfg["child_writeback_rank"] = max(int(cfg.get("child_writeback_rank", 12)), 18)
        cfg["child_writeback_temperature"] = max(float(cfg.get("child_writeback_temperature", 0.85)), 1.06)
        cfg["child_writeback_budget"] = max(float(cfg.get("child_writeback_budget", 1.25)), 1.85)
        cfg["child_parent_mix"] = max(float(cfg.get("child_parent_mix", 0.15)), 0.18)
        cfg["child_parent_mix_early"] = min(max(float(cfg.get("child_parent_mix_early", 0.08)), 0.09), float(cfg["child_parent_mix"]) - 0.02)
        cfg["child_kill_threshold"] = min(float(cfg.get("child_kill_threshold", 0.08)), 0.024)
        cfg["child_branch_parent_threshold"] = min(float(cfg.get("child_branch_parent_threshold", cfg.get("real_branch_threshold", 0.12))), 0.030)
        cfg["child_branch_writeback_threshold"] = max(float(cfg.get("child_branch_writeback_threshold", 0.0)), 0.028)
        cfg["child_branch_meso_threshold"] = max(float(cfg.get("child_branch_meso_threshold", 0.0)), 0.0025)
        cfg["child_branch_live_threshold"] = max(float(cfg.get("child_branch_live_threshold", 0.0)), 0.20)
    elif profile in {"writeback_v18_nested_probe_objective", "writeback_v19_seeded_nested_substrate", "writeback_v20_child_record_perturb", "mode_replace_conversion_v1"}:
        cfg["child_spawn_threshold"] = min(float(cfg.get("child_spawn_threshold", 0.34)), 0.20)
        cfg["child_max_worlds"] = max(int(cfg.get("child_max_worlds", 4)), 5)
        cfg["child_min_age_for_writeback"] = 1
        cfg["child_support_window"] = max(int(cfg.get("child_support_window", 12)), 24)
        cfg["child_support_decay"] = min(float(cfg.get("child_support_decay", 0.12)), 0.050)
        cfg["child_survival_coherence_weight"] = max(float(cfg.get("child_survival_coherence_weight", 0.42)), 0.54)
        cfg["child_survival_qtrace_weight"] = max(float(cfg.get("child_survival_qtrace_weight", 0.24)), 0.36)
        cfg["child_survival_residue_penalty"] = min(float(cfg.get("child_survival_residue_penalty", 0.22)), 0.14)
        cfg["child_writeback_gain"] = max(float(cfg.get("child_writeback_gain", 0.32)), 0.56)
        cfg["child_writeback_rank"] = max(int(cfg.get("child_writeback_rank", 12)), 18)
        cfg["child_writeback_temperature"] = max(float(cfg.get("child_writeback_temperature", 0.85)), 1.06)
        cfg["child_writeback_budget"] = max(float(cfg.get("child_writeback_budget", 1.25)), 1.85)
        cfg["child_parent_mix"] = max(float(cfg.get("child_parent_mix", 0.15)), 0.18)
        cfg["child_parent_mix_early"] = min(max(float(cfg.get("child_parent_mix_early", 0.08)), 0.09), float(cfg["child_parent_mix"]) - 0.02)
        cfg["child_kill_threshold"] = min(float(cfg.get("child_kill_threshold", 0.08)), 0.024)
        cfg["child_branch_parent_threshold"] = min(float(cfg.get("child_branch_parent_threshold", cfg.get("real_branch_threshold", 0.12))), 0.030)
        cfg["child_branch_writeback_threshold"] = max(float(cfg.get("child_branch_writeback_threshold", 0.0)), 0.028)
        cfg["child_branch_meso_threshold"] = max(float(cfg.get("child_branch_meso_threshold", 0.0)), 0.0025)
        cfg["child_branch_live_threshold"] = max(float(cfg.get("child_branch_live_threshold", 0.0)), 0.20)
        if profile == "mode_replace_conversion_v1":
            cfg["child_support_window"] = max(int(cfg.get("child_support_window", 12)), 30)
            cfg["child_support_decay"] = min(float(cfg.get("child_support_decay", 0.12)), 0.026)
            cfg["child_writeback_gain"] = max(float(cfg.get("child_writeback_gain", 0.32)), 0.64)
            cfg["child_writeback_rank"] = max(int(cfg.get("child_writeback_rank", 12)), 22)
            cfg["child_writeback_budget"] = max(float(cfg.get("child_writeback_budget", 1.25)), 2.10)
            cfg["child_parent_mix"] = max(float(cfg.get("child_parent_mix", 0.15)), 0.28)
            cfg["child_parent_mix_early"] = min(max(float(cfg.get("child_parent_mix_early", 0.08)), 0.10), float(cfg["child_parent_mix"]) - 0.02)
            cfg["child_kill_threshold"] = min(float(cfg.get("child_kill_threshold", 0.08)), 0.018)
            cfg["child_branch_parent_threshold"] = min(float(cfg.get("child_branch_parent_threshold", cfg.get("real_branch_threshold", 0.12))), 0.026)
    elif profile == "branch_identity_guard_v1":
        cfg["child_spawn_threshold"] = min(float(cfg.get("child_spawn_threshold", 0.34)), 0.19)
        cfg["child_max_worlds"] = max(int(cfg.get("child_max_worlds", 4)), 6)
        cfg["child_min_age_for_writeback"] = 1
        cfg["child_support_window"] = max(int(cfg.get("child_support_window", 12)), 28)
        cfg["child_support_decay"] = min(float(cfg.get("child_support_decay", 0.12)), 0.030)
        cfg["child_survival_coherence_weight"] = max(float(cfg.get("child_survival_coherence_weight", 0.42)), 0.56)
        cfg["child_survival_qtrace_weight"] = max(float(cfg.get("child_survival_qtrace_weight", 0.24)), 0.34)
        cfg["child_survival_residue_penalty"] = min(float(cfg.get("child_survival_residue_penalty", 0.22)), 0.16)
        cfg["child_writeback_gain"] = max(float(cfg.get("child_writeback_gain", 0.32)), 0.60)
        cfg["child_writeback_rank"] = max(int(cfg.get("child_writeback_rank", 12)), 20)
        cfg["child_writeback_temperature"] = max(float(cfg.get("child_writeback_temperature", 0.85)), 1.14)
        cfg["child_writeback_budget"] = max(float(cfg.get("child_writeback_budget", 1.25)), 1.90)
        cfg["child_parent_mix"] = max(float(cfg.get("child_parent_mix", 0.15)), 0.24)
        cfg["child_parent_mix_early"] = min(max(float(cfg.get("child_parent_mix_early", 0.08)), 0.08), float(cfg["child_parent_mix"]) - 0.02)
        cfg["child_kill_threshold"] = min(float(cfg.get("child_kill_threshold", 0.08)), 0.020)
        cfg["child_branch_parent_threshold"] = min(float(cfg.get("child_branch_parent_threshold", cfg.get("real_branch_threshold", 0.12))), 0.028)
        cfg["child_branch_writeback_threshold"] = max(float(cfg.get("child_branch_writeback_threshold", 0.0)), 0.028)
        cfg["child_branch_meso_threshold"] = max(float(cfg.get("child_branch_meso_threshold", 0.0)), 0.0022)
        cfg["child_branch_live_threshold"] = max(float(cfg.get("child_branch_live_threshold", 0.0)), 0.20)
    elif profile == "branch_writeback_floor_v1":
        cfg["child_spawn_threshold"] = min(float(cfg.get("child_spawn_threshold", 0.34)), 0.19)
        cfg["child_max_worlds"] = max(int(cfg.get("child_max_worlds", 4)), 6)
        cfg["child_min_age_for_writeback"] = 1
        cfg["child_support_window"] = max(int(cfg.get("child_support_window", 12)), 28)
        cfg["child_support_decay"] = min(float(cfg.get("child_support_decay", 0.12)), 0.020)
        cfg["child_survival_coherence_weight"] = max(float(cfg.get("child_survival_coherence_weight", 0.42)), 0.55)
        cfg["child_survival_qtrace_weight"] = max(float(cfg.get("child_survival_qtrace_weight", 0.24)), 0.36)
        cfg["child_survival_residue_penalty"] = min(float(cfg.get("child_survival_residue_penalty", 0.22)), 0.15)
        cfg["child_writeback_gain"] = max(float(cfg.get("child_writeback_gain", 0.32)), 0.68)
        cfg["child_writeback_rank"] = max(int(cfg.get("child_writeback_rank", 12)), 22)
        cfg["child_writeback_temperature"] = max(float(cfg.get("child_writeback_temperature", 0.85)), 1.16)
        cfg["child_writeback_budget"] = max(float(cfg.get("child_writeback_budget", 1.25)), 2.20)
        cfg["child_parent_mix"] = max(float(cfg.get("child_parent_mix", 0.15)), 0.24)
        cfg["child_parent_mix_early"] = min(max(float(cfg.get("child_parent_mix_early", 0.08)), 0.08), float(cfg["child_parent_mix"]) - 0.02)
        cfg["child_kill_threshold"] = min(float(cfg.get("child_kill_threshold", 0.08)), 0.018)
        cfg["child_branch_parent_threshold"] = min(float(cfg.get("child_branch_parent_threshold", cfg.get("real_branch_threshold", 0.12))), 0.028)
        cfg["child_branch_writeback_threshold"] = max(float(cfg.get("child_branch_writeback_threshold", 0.0)), 0.030)
        cfg["child_branch_meso_threshold"] = max(float(cfg.get("child_branch_meso_threshold", 0.0)), 0.0025)
        cfg["child_branch_live_threshold"] = max(float(cfg.get("child_branch_live_threshold", 0.0)), 0.22)
    elif profile == "branch_balance_guard_v1":
        cfg["child_spawn_threshold"] = min(float(cfg.get("child_spawn_threshold", 0.34)), 0.20)
        cfg["child_max_worlds"] = max(int(cfg.get("child_max_worlds", 4)), 6)
        cfg["child_min_age_for_writeback"] = 1
        cfg["child_support_window"] = max(int(cfg.get("child_support_window", 12)), 26)
        cfg["child_support_decay"] = min(float(cfg.get("child_support_decay", 0.12)), 0.025)
        cfg["child_survival_coherence_weight"] = max(float(cfg.get("child_survival_coherence_weight", 0.42)), 0.54)
        cfg["child_survival_qtrace_weight"] = max(float(cfg.get("child_survival_qtrace_weight", 0.24)), 0.33)
        cfg["child_survival_residue_penalty"] = min(float(cfg.get("child_survival_residue_penalty", 0.22)), 0.16)
        cfg["child_writeback_gain"] = max(float(cfg.get("child_writeback_gain", 0.32)), 0.62)
        cfg["child_writeback_rank"] = max(int(cfg.get("child_writeback_rank", 12)), 20)
        cfg["child_writeback_temperature"] = max(float(cfg.get("child_writeback_temperature", 0.85)), 1.12)
        cfg["child_writeback_budget"] = max(float(cfg.get("child_writeback_budget", 1.25)), 1.95)
        cfg["child_parent_mix"] = max(float(cfg.get("child_parent_mix", 0.15)), 0.24)
        cfg["child_parent_mix_early"] = min(max(float(cfg.get("child_parent_mix_early", 0.08)), 0.09), float(cfg["child_parent_mix"]) - 0.02)
        cfg["child_kill_threshold"] = min(float(cfg.get("child_kill_threshold", 0.08)), 0.020)
        cfg["child_branch_parent_threshold"] = min(float(cfg.get("child_branch_parent_threshold", cfg.get("real_branch_threshold", 0.12))), 0.028)
        cfg["child_branch_writeback_threshold"] = max(float(cfg.get("child_branch_writeback_threshold", 0.0)), 0.028)
        cfg["child_branch_meso_threshold"] = max(float(cfg.get("child_branch_meso_threshold", 0.0)), 0.0025)
        cfg["child_branch_live_threshold"] = max(float(cfg.get("child_branch_live_threshold", 0.0)), 0.20)
    elif profile in {"branch_recovery_soft_guard_v1", "naked_phase_nested_recovery_v1"}:
        cfg["child_spawn_threshold"] = min(float(cfg.get("child_spawn_threshold", 0.34)), 0.215)
        cfg["child_max_worlds"] = max(int(cfg.get("child_max_worlds", 4)), 5)
        cfg["child_min_age_for_writeback"] = 1
        cfg["child_support_window"] = max(int(cfg.get("child_support_window", 12)), 24)
        cfg["child_support_decay"] = min(float(cfg.get("child_support_decay", 0.12)), 0.040)
        cfg["child_survival_coherence_weight"] = max(float(cfg.get("child_survival_coherence_weight", 0.42)), 0.52)
        cfg["child_survival_qtrace_weight"] = max(float(cfg.get("child_survival_qtrace_weight", 0.24)), 0.30)
        cfg["child_survival_residue_penalty"] = min(float(cfg.get("child_survival_residue_penalty", 0.22)), 0.18)
        cfg["child_writeback_gain"] = max(float(cfg.get("child_writeback_gain", 0.32)), 0.57)
        cfg["child_writeback_rank"] = max(int(cfg.get("child_writeback_rank", 12)), 18)
        cfg["child_writeback_temperature"] = max(float(cfg.get("child_writeback_temperature", 0.85)), 1.10)
        cfg["child_writeback_budget"] = max(float(cfg.get("child_writeback_budget", 1.25)), 1.75)
        cfg["child_parent_mix"] = max(float(cfg.get("child_parent_mix", 0.15)), 0.22)
        cfg["child_parent_mix_early"] = min(max(float(cfg.get("child_parent_mix_early", 0.08)), 0.08), float(cfg["child_parent_mix"]) - 0.02)
        cfg["child_kill_threshold"] = min(float(cfg.get("child_kill_threshold", 0.08)), 0.032)
        cfg["child_branch_parent_threshold"] = min(float(cfg.get("child_branch_parent_threshold", cfg.get("real_branch_threshold", 0.12))), 0.032)
        cfg["child_branch_writeback_threshold"] = max(float(cfg.get("child_branch_writeback_threshold", 0.0)), 0.025)
        cfg["child_branch_meso_threshold"] = max(float(cfg.get("child_branch_meso_threshold", 0.0)), 0.0020)
        cfg["child_branch_live_threshold"] = max(float(cfg.get("child_branch_live_threshold", 0.0)), 0.19)
        if profile == "naked_phase_nested_recovery_v1":
            cfg["child_spawn_threshold"] = min(float(cfg.get("child_spawn_threshold", 0.34)), 0.205)
            cfg["child_max_worlds"] = max(int(cfg.get("child_max_worlds", 4)), 6)
            cfg["child_support_window"] = max(int(cfg.get("child_support_window", 12)), 28)
            cfg["child_support_decay"] = min(float(cfg.get("child_support_decay", 0.12)), 0.035)
            cfg["child_writeback_gain"] = max(float(cfg.get("child_writeback_gain", 0.32)), 0.62)
            cfg["child_writeback_rank"] = max(int(cfg.get("child_writeback_rank", 12)), 20)
            cfg["child_writeback_budget"] = max(float(cfg.get("child_writeback_budget", 1.25)), 1.90)
            cfg["child_parent_mix"] = max(float(cfg.get("child_parent_mix", 0.15)), 0.24)
            cfg["child_kill_threshold"] = min(float(cfg.get("child_kill_threshold", 0.08)), 0.030)
            cfg["child_branch_parent_threshold"] = min(float(cfg.get("child_branch_parent_threshold", 0.12)), 0.030)
    out_payload = payload if "config" in payload else cfg
    if "config" in payload:
        out_payload["config"] = cfg
    out_path = out_dir / "childworld_init_config.json"
    out_path.write_text(json.dumps(out_payload, indent=2), encoding="utf-8")
    return out_path


def _run_nested_commitment(
    config_path: Path,
    out_dir: Path,
    device_name: str,
    case_json: Path,
    fork_selector: str = "fixed",
    live_child_threshold: float = 0.05,
    depth: int = 3,
    fork_depth: int = 1,
    skip: bool = False,
    fail_hard: bool = False,
    case_limit: int | None = None,
    fallback_case_limit: int | None = 4,
    fallback_device_name: str | None = None,
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    if skip:
        return _nested_report_stub(
            config_path=config_path,
            out_dir=out_dir,
            status="skipped",
            overall_read="nested_commitment_skipped",
            message="Nested commitment sidecar was skipped by runner configuration.",
            device_name=device_name,
            case_json=case_json,
            fork_selector=fork_selector,
            live_child_threshold=live_child_threshold,
        )

    attempts: list[dict[str, Any]] = []
    primary_case_json = _materialize_nested_case_json(case_json, out_dir, case_limit, "nested_primary_cases")
    primary_cmd = [
        sys.executable,
        str(RUNTIME / "test_nested_commitment.py"),
        "--config",
        str(config_path),
        "--out-dir",
        str(out_dir),
        "--device",
        device_name,
        "--depth",
        str(depth),
        "--fork-depth",
        str(fork_depth),
        "--fork-selector",
        str(fork_selector),
        "--live-child-threshold",
        str(live_child_threshold),
        "--mode",
        "native_multimode_childworld",
        "--case-json",
        str(primary_case_json),
    ]
    try:
        _run_nested_attempt(primary_cmd, out_dir, "nested_primary")
        report = json.loads((out_dir / "nested_commitment_report.json").read_text(encoding="utf-8"))
        report.setdefault("status", "ok")
        report.setdefault("attempts", [])
        report["attempts"].append(
            _attempt_record(
                label="primary",
                cmd=primary_cmd,
                device_name=device_name,
                case_json=primary_case_json,
                status="ok",
            )
        )
        _write_json(out_dir / "nested_commitment_report.json", report)
        return report
    except Exception as exc:
        attempts.append(
            _attempt_record(
                label="primary",
                cmd=primary_cmd,
                device_name=device_name,
                case_json=primary_case_json,
                status="failed",
                exc=exc,
            )
        )
        if fail_hard:
            raise

    retry_device = fallback_device_name or ("cpu" if str(device_name).lower() != "cpu" else str(device_name))
    retry_case_json = _materialize_nested_case_json(case_json, out_dir, fallback_case_limit, "nested_fallback_cases")
    retry_cmd = [
        sys.executable,
        str(RUNTIME / "test_nested_commitment.py"),
        "--config",
        str(config_path),
        "--out-dir",
        str(out_dir),
        "--device",
        str(retry_device),
        "--depth",
        str(depth),
        "--fork-depth",
        str(fork_depth),
        "--fork-selector",
        str(fork_selector),
        "--live-child-threshold",
        str(live_child_threshold),
        "--mode",
        "native_multimode_childworld",
        "--case-json",
        str(retry_case_json),
    ]
    try:
        _run_nested_attempt(retry_cmd, out_dir, "nested_fallback")
        report = json.loads((out_dir / "nested_commitment_report.json").read_text(encoding="utf-8"))
        report["status"] = "fallback_ok"
        report.setdefault("attempts", [])
        report["attempts"].extend(attempts)
        report["attempts"].append(
            _attempt_record(
                label="fallback",
                cmd=retry_cmd,
                device_name=str(retry_device),
                case_json=retry_case_json,
                status="ok",
            )
        )
        report["fallback_used"] = True
        _write_json(out_dir / "nested_commitment_report.json", report)
        return report
    except Exception as retry_exc:
        attempts.append(
            _attempt_record(
                label="fallback",
                cmd=retry_cmd,
                device_name=str(retry_device),
                case_json=retry_case_json,
                status="failed",
                exc=retry_exc,
            )
        )
        return _nested_report_stub(
            config_path=config_path,
            out_dir=out_dir,
            status="failed",
            overall_read="nested_commitment_failed",
            message="Nested commitment sidecar failed after primary and fallback attempts.",
            device_name=str(retry_device),
            case_json=retry_case_json,
            fork_selector=fork_selector,
            live_child_threshold=live_child_threshold,
            attempts=attempts,
        )


def run_experiment(
    out_dir: Path,
    checkpoint_dir: Path,
    init_config_path: Path,
    cases_path: Path,
    device_name: str,
    iterations: int,
    population: int,
    elite_count: int,
    seed: int,
    clip_seconds: int,
    phase_blend: float,
    profile: str = "childworld_v1",
    skip_nested_commitment: bool = False,
    nested_case_json: Path | None = None,
    nested_depth: int = 3,
    nested_fork_depth: int = 1,
    nested_fork_selector: str | None = None,
    nested_live_child_threshold: float = 0.05,
    nested_case_limit: int | None = None,
    nested_fallback_case_limit: int | None = 4,
    nested_device_name: str | None = None,
    nested_fail_hard: bool = False,
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    heldout_probe_cfg = _heldout_probe_cfg(profile=profile)
    prepared_init = _prepare_childworld_init_config(init_config_path, out_dir, profile=profile)
    train_summary = train_circleworld_real_anchor(
        out_dir=out_dir,
        checkpoint_dir=checkpoint_dir,
        iterations=iterations,
        population=population,
        elite_count=elite_count,
        seed=seed,
        device_name=device_name,
        clip_seconds=clip_seconds,
        phase_blend=phase_blend,
        init_config_path=prepared_init,
        score_cfg=_childworld_score_cfg(profile=profile),
        case_weights=_case_weights(),
        extra_train_wavs=_extra_train_wavs(),
        extra_val_wavs=_extra_val_wavs(),
        heldout_probe_cfg=heldout_probe_cfg,
    )

    config_path = Path(train_summary["checkpoint"])
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
    law_tokens = build_library(
        config_path=config_path,
        out_dir=out_dir / "law_token_library",
        cases_path=cases_path,
        device_name=device_name,
        clip_seconds=clip_seconds,
    )
    resolved_nested_case_json = _resolve_nested_case_json(
        cases_path=cases_path,
        out_dir=out_dir,
        profile=profile,
        nested_case_json=nested_case_json,
        nested_case_plan=heldout_probe_cfg.get("selection_nested_case_plan"),
    )
    resolved_fork_selector = nested_fork_selector or (
        str(heldout_probe_cfg.get("selection_nested_fork_selector"))
        if heldout_probe_cfg.get("selection_nested_fork_selector") is not None
        else (
            "first_live_child"
        if profile in {"writeback_v6_naked_survival_curriculum", "writeback_v7_audio_recovery_with_branch_floor", "writeback_v8_recovery_with_law_floor", "writeback_v9_aligned_probe_recovery", "writeback_v11_hard_gate_seeded", "writeback_v12_verified_gate_seeded", "writeback_v13_targeted_replay_recovery", "writeback_v15_nested_divergence_parentmix", "writeback_v16_benchmark_guard_nested", "writeback_v17_nested_response_gate", "writeback_v18_nested_probe_objective", "writeback_v19_seeded_nested_substrate", "writeback_v20_child_record_perturb", "branch_identity_guard_v1", "branch_writeback_floor_v1", "branch_balance_guard_v1", "branch_recovery_soft_guard_v1", "naked_phase_nested_recovery_v1", "mode_replace_conversion_v1"}
            else "fixed"
        )
    )
    resolved_live_child_threshold = float(nested_live_child_threshold)
    if heldout_probe_cfg.get("selection_nested_live_child_threshold") is not None and abs(float(nested_live_child_threshold) - 0.05) < 1e-9:
        resolved_live_child_threshold = float(heldout_probe_cfg["selection_nested_live_child_threshold"])
    nested = _run_nested_commitment(
        config_path=config_path,
        out_dir=out_dir / "nested_commitment",
        device_name=device_name,
        case_json=resolved_nested_case_json,
        fork_selector=resolved_fork_selector,
        live_child_threshold=resolved_live_child_threshold,
        depth=nested_depth,
        fork_depth=nested_fork_depth,
        skip=skip_nested_commitment,
        fail_hard=nested_fail_hard,
        case_limit=nested_case_limit,
        fallback_case_limit=nested_fallback_case_limit,
        fallback_device_name=nested_device_name,
    )

    summary = {
        "runtime": "circleworld_proto",
        "mode": "childworld_experiment",
        "profile": profile,
        "init_config_path": str(init_config_path),
        "prepared_init_config": str(prepared_init),
        "cases_path": str(cases_path),
        "device": device_name,
        "iterations": iterations,
        "population": population,
        "elite_count": elite_count,
        "seed": seed,
        "clip_seconds": clip_seconds,
        "phase_blend": phase_blend,
        "skip_nested_commitment": bool(skip_nested_commitment),
        "nested_case_json": str(resolved_nested_case_json),
        "nested_depth": int(nested_depth),
        "nested_fork_depth": int(nested_fork_depth),
        "nested_fork_selector": str(resolved_fork_selector),
        "nested_live_child_threshold": float(resolved_live_child_threshold),
        "nested_case_limit": nested_case_limit,
        "nested_fallback_case_limit": nested_fallback_case_limit,
        "nested_device": str(nested_device_name or device_name),
        "nested_fail_hard": bool(nested_fail_hard),
        "checkpoint": str(config_path),
        "train_summary": str(out_dir / "train_summary.json"),
        "heldout_summary": str(out_dir / "heldout_eval" / "heldout_summary.json"),
        "benchmark_summary": str(out_dir / "benchmark_expanded" / "benchmark_summary.json"),
        "continuity_circleworld": str(out_dir / "benchmark_expanded" / "continuity_circleworld.json"),
        "continuity_reference": str(out_dir / "benchmark_expanded" / "continuity_reference.json"),
        "nested_summary": str(out_dir / "nested_commitment" / "nested_commitment_report.json"),
        "law_token_library": str(out_dir / "law_token_library" / "law_token_library.json"),
        "selection": {
            "source": train_summary.get("selection_source"),
            "fallback_reasons": (
                train_summary.get("selection_fallback_diagnostics", {}).get("candidate_unavailable_reasons", [])
                if isinstance(train_summary.get("selection_fallback_diagnostics"), dict)
                else []
            ),
            "fallback_failure_counts": (
                train_summary.get("selection_fallback_diagnostics", {}).get("failure_counts", {})
                if isinstance(train_summary.get("selection_fallback_diagnostics"), dict)
                else {}
            ),
        },
        "heldout": {
            "mean_real_branch_fraction": heldout.get("mean_real_branch_fraction", 0.0),
            "mean_parent_real_branch_fraction": heldout.get("mean_parent_real_branch_fraction", 0.0),
            "mean_child_real_branch_fraction": heldout.get("mean_child_real_branch_fraction", 0.0),
            "phase_only_real_branch_fraction": heldout.get("phase_only_real_branch_fraction", 0.0),
            "phase_only_excess_branch_fraction": heldout.get("phase_only_excess_branch_fraction", 0.0),
            "decorative_slot2_low_phase_fraction": heldout.get("decorative_slot2_low_phase_fraction", 0.0),
            "mean_phase_only_branch_phase_wall": heldout.get("mean_phase_only_branch_phase_wall", 0.0),
            "mean_phase_only_branch_distinctness": heldout.get("mean_phase_only_branch_distinctness", 0.0),
            "final_phase_only_branch_live_fraction": heldout.get("final_phase_only_branch_live_fraction", 0.0),
            "final_phase_only_branch_phase_wall": heldout.get("final_phase_only_branch_phase_wall", 0.0),
            "final_phase_only_branch_distinctness": heldout.get("final_phase_only_branch_distinctness", 0.0),
            "mean_meso_branch_effect": heldout.get("mean_meso_branch_effect", 0.0),
            "mean_child_world_count": heldout.get("mean_child_world_count", 0.0),
            "mean_live_child_fraction": heldout.get("mean_live_child_fraction", 0.0),
            "mean_child_age": heldout.get("mean_child_age", 0.0),
            "mean_child_writeback_mass": heldout.get("mean_child_writeback_mass", 0.0),
            "mean_child_parent_divergence": heldout.get("mean_child_parent_divergence", 0.0),
            "mean_child_sibling_divergence": heldout.get("mean_child_sibling_divergence", 0.0),
            "mean_defect_without_branch_penalty": heldout.get("mean_defect_without_branch_penalty", 0.0),
            "mean_branch_resolution_delay": heldout.get("mean_branch_resolution_delay", 0.0),
            "mean_num_law_families": heldout.get("mean_num_law_families", 0.0),
            "mean_law_family_entropy": heldout.get("mean_law_family_entropy", 0.0),
            "mean_num_relational_signature_families": heldout.get("mean_num_relational_signature_families", 0.0),
            "mean_relational_signature_confidence": heldout.get("mean_relational_signature_confidence", 0.0),
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
            "mean_adjacent_chunk_similarity": continuity_circleworld["mean_adjacent_chunk_similarity"],
        },
        "nested": {
            "status": nested.get("status", "ok"),
            "overall_read": nested.get("overall_read"),
            "verdict_counts": nested.get("verdict_counts", {}),
            "attempts": nested.get("attempts", []),
            "num_live_child_start_cases": nested.get("num_live_child_start_cases", 0),
            "num_selected_fork_has_live_child_cases": nested.get("num_selected_fork_has_live_child_cases", 0),
            "mean_selected_fork_has_live_child": nested.get("mean_selected_fork_has_live_child", 0.0),
            "mean_selected_child_signal_found": nested.get("mean_selected_child_signal_found", 0.0),
            "mean_selected_live_child_eligible": nested.get("mean_selected_live_child_eligible", 0.0),
            "mean_selected_phase_only_eligible": nested.get("mean_selected_phase_only_eligible", 0.0),
            "mean_child_active_fraction": nested.get("mean_child_active_fraction", 0.0),
            "mean_child_meso_response": nested.get("mean_child_meso_response", 0.0),
            "mean_nested_sibling_fraction": nested.get("mean_nested_sibling_fraction", 0.0),
            "mean_branch_identity_carry": nested.get("mean_branch_identity_carry", 0.0),
            "mean_branch_identity_qualified_carry": nested.get("mean_branch_identity_qualified_carry", 0.0),
            "max_continuity_delta_magnitude": nested.get("max_continuity_delta_magnitude", 0.0),
            "branch_family_counts": nested.get("branch_family_counts", {}),
        },
        "law_tokens": {
            "mean_num_law_packets": law_tokens["mean_num_law_packets"],
            "mean_num_law_families": law_tokens["mean_num_law_families"],
            "aggregate_num_families": law_tokens["aggregate_library"]["num_families"],
        },
    }
    _write_json(out_dir / "childworld_experiment_summary.json", summary)
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description="Run the Circleworld child-world branching experiment with full sidecars.")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--checkpoint-dir", required=True)
    ap.add_argument("--init-config", default=str(DEFAULT_RAMANUJAN_INIT))
    ap.add_argument("--cases-json", default=str(DEFAULT_CASES))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--iterations", type=int, default=12)
    ap.add_argument("--population", type=int, default=8)
    ap.add_argument("--elite-count", type=int, default=3)
    ap.add_argument("--seed", type=int, default=20260421)
    ap.add_argument("--clip-seconds", type=int, default=10)
    ap.add_argument("--phase-blend", type=float, default=1.0)
    ap.add_argument("--skip-nested-commitment", action="store_true")
    ap.add_argument("--nested-case-json", default=None, help="Optional override for the nested sidecar case JSON. Branch profiles default to the seeded branch-active suite when unset.")
    ap.add_argument("--nested-depth", type=int, default=3)
    ap.add_argument("--nested-fork-depth", type=int, default=1)
    ap.add_argument("--nested-fork-selector", choices=["fixed", "first_live_child", "max_live_child"], default=None)
    ap.add_argument("--nested-live-child-threshold", type=float, default=0.05)
    ap.add_argument("--nested-case-limit", type=int, default=None)
    ap.add_argument("--nested-fallback-case-limit", type=int, default=4)
    ap.add_argument("--nested-device", default=None, help="Optional device override for fallback nested retry; defaults to cpu when the primary device is not cpu.")
    ap.add_argument("--nested-fail-hard", action="store_true", help="Propagate nested sidecar failures instead of writing a structured failure report.")
    ap.add_argument("--profile", choices=["childworld_v1", "writeback_v2", "writeback_v3_identity", "writeback_v4_real_anchor_transfer", "writeback_v5_thresholded_real_branch", "writeback_v6_naked_survival_curriculum", "writeback_v7_audio_recovery_with_branch_floor", "writeback_v8_recovery_with_law_floor", "writeback_v9_aligned_probe_recovery", "writeback_v11_hard_gate_seeded", "writeback_v12_verified_gate_seeded", "writeback_v13_targeted_replay_recovery", "writeback_v15_nested_divergence_parentmix", "writeback_v16_benchmark_guard_nested", "writeback_v17_nested_response_gate", "writeback_v18_nested_probe_objective", "writeback_v19_seeded_nested_substrate", "writeback_v20_child_record_perturb", "branch_identity_guard_v1", "branch_writeback_floor_v1", "branch_balance_guard_v1", "branch_recovery_soft_guard_v1", "naked_phase_nested_recovery_v1", "mode_replace_conversion_v1"], default="childworld_v1")
    args = ap.parse_args()

    summary = run_experiment(
        out_dir=Path(args.out_dir),
        checkpoint_dir=Path(args.checkpoint_dir),
        init_config_path=Path(args.init_config),
        cases_path=Path(args.cases_json),
        device_name=args.device,
        iterations=args.iterations,
        population=args.population,
        elite_count=args.elite_count,
        seed=args.seed,
        clip_seconds=args.clip_seconds,
        phase_blend=args.phase_blend,
        profile=args.profile,
        skip_nested_commitment=bool(args.skip_nested_commitment),
        nested_case_json=Path(args.nested_case_json) if args.nested_case_json else None,
        nested_depth=args.nested_depth,
        nested_fork_depth=args.nested_fork_depth,
        nested_fork_selector=args.nested_fork_selector,
        nested_live_child_threshold=args.nested_live_child_threshold,
        nested_case_limit=args.nested_case_limit,
        nested_fallback_case_limit=args.nested_fallback_case_limit,
        nested_device_name=args.nested_device,
        nested_fail_hard=bool(args.nested_fail_hard),
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
