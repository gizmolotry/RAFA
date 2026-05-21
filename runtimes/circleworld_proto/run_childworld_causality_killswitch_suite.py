from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
CORE = ROOT / "core"
LINEAGE = ROOT / "lineages" / "04_positive_replacement"
for path in (ROOT, CORE, LINEAGE):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from test_nested_commitment import _default_seeded_nested_cases, evaluate_nested_commitment_report


METRIC_KEYS: tuple[str, ...] = (
    "mean_nested_sibling_fraction",
    "mean_assay_readout_nested_sibling_fraction",
    "mean_assay_continuation_nested_sibling_fraction",
    "mean_assay_mode_replace_nested_sibling_fraction",
    "mean_assay_continuation_max_nested_sibling_readiness",
    "mean_assay_continuation_nested_sibling_readiness",
    "mean_assay_continuation_branch_identity_qualified_carry",
    "mean_assay_continuation_branch_identity_budget_retained",
    "mean_assay_continuation_readout_sibling_response",
    "mean_assay_continuation_fine_q_profile_corr",
    "mean_assay_continuation_continuation_write_enabled",
    "mean_assay_continuation_continuation_write_delta",
    "mean_assay_mode_replace_max_nested_sibling_readiness",
    "mean_assay_mode_replace_nested_sibling_readiness",
    "mean_assay_mode_replace_branch_identity_qualified_carry",
    "mean_assay_mode_replace_branch_identity_budget_retained",
    "mean_assay_mode_replace_readout_sibling_response",
    "mean_assay_mode_replace_fine_q_profile_corr",
    "mean_assay_mode_replace_mode1_replace_enabled",
    "mean_assay_mode_replace_mode1_replace_ratio",
    "mean_assay_mode_replace_mode1_replace_gate",
    "mean_assay_mode_replace_mode1_replace_static_record",
    "mean_assay_mode_replace_mode1_replace_gate_floor",
    "mean_assay_mode_replace_mode1_replace_ratio_scale",
    "mean_world_jump_penalty",
    "mean_over_rigid_fraction",
    "mean_direct_readout_dependency",
    "mean_final_direct_mix_shortcut_score",
    "mean_child_record_survival_score",
    "mean_parent_mode_conversion_sibling_fraction",
    "mean_mode_replace_conversion_score",
    "mean_child_volume_active_count",
    "mean_child_volume_effective_count",
    "mean_child_volume_top_score_share",
    "mean_assay_child_fragment_count",
    "max_assay_child_fragment_count",
    "mean_assay_grandchild_count",
    "max_assay_grandchild_count",
    "max_assay_child_generation",
    "mean_child_predictive_residual_reduction",
    "max_child_predictive_residual_reduction",
    "mean_child_predictive_positive_residual_reduction",
    "mean_child_predictive_boundary_match",
    "mean_child_predictive_explained_parent_defect",
    "mean_child_predictive_ontology_binding_mass",
    "mean_child_predictive_applied_fraction",
    "mean_branch_identity_qualified_carry",
    "mean_branch_identity_budget_retained",
)


def _load_payload(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise ValueError(f"Config payload must be a JSON object: {path}")
    return payload


def _config_dict(payload: dict[str, Any]) -> dict[str, Any]:
    cfg = payload.get("config", payload)
    if not isinstance(cfg, dict):
        raise ValueError("Config object missing or invalid")
    return dict(cfg)


def _write_variant_config(base_payload: dict[str, Any], out_path: Path, updates: dict[str, Any]) -> Path:
    cfg = _config_dict(base_payload)
    cfg.update(updates)
    payload = dict(base_payload)
    payload["config"] = cfg
    payload["causality_variant_updates"] = updates
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return out_path


def _metric_row(report: dict[str, Any]) -> dict[str, Any]:
    row = {key: report.get(key) for key in METRIC_KEYS}
    row["num_cases"] = report.get("num_cases")
    row["verdict_counts"] = report.get("verdict_counts", {})
    row["assay_family_nested_sibling_counts"] = report.get("assay_family_nested_sibling_counts", {})
    row["assay_family_branch_counts"] = report.get("assay_family_branch_counts", {})
    return row


def _decision_for_variant(name: str, row: dict[str, Any], positive: dict[str, Any]) -> dict[str, Any]:
    pos_cont = float(positive.get("mean_assay_continuation_nested_sibling_fraction", 0.0) or 0.0)
    cont = float(row.get("mean_assay_continuation_nested_sibling_fraction", 0.0) or 0.0)
    pos_mode = float(positive.get("mean_assay_mode_replace_nested_sibling_fraction", 0.0) or 0.0)
    mode_replace = float(row.get("mean_assay_mode_replace_nested_sibling_fraction", 0.0) or 0.0)
    readiness = float(row.get("mean_assay_continuation_max_nested_sibling_readiness", 0.0) or 0.0)
    mode_readiness = float(row.get("mean_assay_mode_replace_max_nested_sibling_readiness", 0.0) or 0.0)
    write_enabled = float(row.get("mean_assay_continuation_continuation_write_enabled", 0.0) or 0.0)
    write_delta = float(row.get("mean_assay_continuation_continuation_write_delta", 0.0) or 0.0)
    replace_enabled = float(row.get("mean_assay_mode_replace_mode1_replace_enabled", 0.0) or 0.0)
    replace_gate = float(row.get("mean_assay_mode_replace_mode1_replace_gate", 0.0) or 0.0)
    world_jump = float(row.get("mean_world_jump_penalty", 0.0) or 0.0)
    direct_readout_dependency = float(row.get("mean_direct_readout_dependency", 0.0) or 0.0)
    direct_mix_shortcut = float(row.get("mean_final_direct_mix_shortcut_score", 0.0) or 0.0)
    child_record_survival = float(row.get("mean_child_record_survival_score", 0.0) or 0.0)
    mode_conversion_score = float(row.get("mean_mode_replace_conversion_score", 0.0) or 0.0)
    drop = float(pos_cont - cont)
    mode_gain = float(mode_replace - pos_mode)
    conversion_supported = bool(mode_replace > max(pos_mode, 0.0) and world_jump <= 0.0)
    return {
        "variant": name,
        "continuation_fraction": cont,
        "positive_continuation_fraction": pos_cont,
        "continuation_fraction_drop": drop,
        "mode_replace_fraction": mode_replace,
        "positive_mode_replace_fraction": pos_mode,
        "mode_replace_fraction_gain": mode_gain,
        "continuation_readiness": readiness,
        "mode_replace_readiness": mode_readiness,
        "continuation_write_enabled": write_enabled,
        "continuation_write_delta": write_delta,
        "mode_replace_enabled": replace_enabled,
        "mode_replace_gate": replace_gate,
        "direct_readout_dependency": direct_readout_dependency,
        "final_direct_mix_shortcut_score": direct_mix_shortcut,
        "child_record_survival_score": child_record_survival,
        "mode_replace_conversion_score": mode_conversion_score,
        "world_jump_penalty": world_jump,
        "breaks_continuation_sibling": bool(pos_cont > 0.0 and cont <= 0.0),
        "preserves_continuation_sibling": bool(pos_cont > 0.0 and cont > 0.0),
        "creates_mode_replace_sibling": conversion_supported,
        "direct_mix_shortcut_present": bool(direct_mix_shortcut > 0.0 and mode_replace <= 0.0),
        "conversion_supported_without_direct_mix": bool(conversion_supported and direct_mix_shortcut <= 0.0),
        "read": (
            "positive_control"
            if name == "positive_control"
            else "mode_replace_signal_created"
            if conversion_supported
            else "kill_switch_breaks_signal"
            if pos_cont > 0.0 and cont <= 0.0
            else "signal_survives_kill_switch"
            if pos_cont > 0.0 and cont > 0.0
            else "no_positive_signal_to_compare"
        ),
    }


def _variant_specs() -> list[dict[str, Any]]:
    return [
        {
            "name": "positive_control",
            "description": "Selected checkpoint, no kill-switch.",
            "config_updates": {},
            "assay_killswitch": {},
        },
        {
            "name": "disable_continuation_write",
            "description": "Skip child-to-mode1 low-rank continuation write but keep direct continuation readout mix.",
            "config_updates": {},
            "assay_killswitch": {"disable_continuation_mode1_write": True},
        },
        {
            "name": "continuation_parent_only",
            "description": "Render parent phase at the final continuation override: no final child write and no final direct child mix.",
            "config_updates": {},
            "assay_killswitch": {"continuation_parent_only": True},
        },
        {
            "name": "continuation_write_only_readout",
            "description": "Apply child-to-mode1 write, then render from the parent mixture without direct child readout mix.",
            "config_updates": {},
            "assay_killswitch": {"continuation_write_only_readout": True},
        },
        {
            "name": "scramble_continuation_child_phase",
            "description": "Use a deterministic rolled child phase for continuation write/readout while leaving identity bookkeeping intact.",
            "config_updates": {},
            "assay_killswitch": {"scramble_continuation_child_phase": True},
        },
        {
            "name": "strict_parent_no_child_context",
            "description": "Strip child worlds before branch, skip branch pre-unroll, disable childworld runtime, and render mode-0 parent only.",
            "config_updates": {},
            "assay_killswitch": {
                "strip_child_worlds_before_branch": True,
                "disable_branch_pre_unroll": True,
                "disable_branch_childworld_runtime": True,
                "continuation_parent_only": True,
                "continuation_mode0_parent_only": True,
            },
        },
        {
            "name": "no_branch_pre_unroll",
            "description": "Skip branch-level child pre-unroll before the normal branch depth trace.",
            "config_updates": {},
            "assay_killswitch": {"disable_branch_pre_unroll": True},
        },
        {
            "name": "no_branch_childworld_runtime",
            "description": "Run branch continuation depth with native multimode runtime, carrying child records but disabling childworld evolution/spawn/writeback.",
            "config_updates": {},
            "assay_killswitch": {"disable_branch_childworld_runtime": True},
        },
        {
            "name": "no_continuation_pre_unroll",
            "description": "Skip the final continuation-specific child-only pre-unroll while keeping branch pre-unroll/runtime and final readout mix.",
            "config_updates": {},
            "assay_killswitch": {"disable_continuation_pre_unroll": True},
        },
        {
            "name": "strict_write_only_no_preunroll",
            "description": "Apply explicit continuation mode-1 write without final continuation pre-unroll, then render parent mixture without direct child mix.",
            "config_updates": {},
            "assay_killswitch": {
                "disable_continuation_pre_unroll": True,
                "disable_direct_child_continuation_mix": True,
            },
        },
        {
            "name": "strict_direct_mix_only",
            "description": "Disable explicit continuation mode-1 write and final continuation pre-unroll, leaving direct child continuation mix as the carrier.",
            "config_updates": {},
            "assay_killswitch": {
                "disable_continuation_pre_unroll": True,
                "disable_continuation_mode1_write": True,
            },
        },
        {
            "name": "static_child_record_direct_mix_only",
            "description": "Disable branch pre-unroll, branch childworld runtime, final continuation pre-unroll, and explicit write; test whether the forked child record alone can carry direct continuation mix.",
            "config_updates": {},
            "assay_killswitch": {
                "disable_branch_pre_unroll": True,
                "disable_branch_childworld_runtime": True,
                "disable_continuation_pre_unroll": True,
                "disable_continuation_mode1_write": True,
            },
        },
        {
            "name": "static_child_record_write_only",
            "description": "Disable branch pre-unroll, branch childworld runtime, final continuation pre-unroll, and direct mix; test whether explicit write from the static child record is sufficient.",
            "config_updates": {},
            "assay_killswitch": {
                "disable_branch_pre_unroll": True,
                "disable_branch_childworld_runtime": True,
                "disable_continuation_pre_unroll": True,
                "disable_direct_child_continuation_mix": True,
            },
        },
        {
            "name": "static_child_record_stripped_control",
            "description": "Strip child records while disabling branch pre-unroll/runtime and final pre-unroll/write; direct mix should have no child carrier.",
            "config_updates": {},
            "assay_killswitch": {
                "strip_child_worlds_before_branch": True,
                "disable_branch_pre_unroll": True,
                "disable_branch_childworld_runtime": True,
                "disable_continuation_pre_unroll": True,
                "disable_continuation_mode1_write": True,
            },
        },
        {
            "name": "static_child_record_mode1_replace",
            "description": "Disable branch pre-unroll/runtime and mode-replace pre-unroll; test whether the static child record can become parent mode1 via replacement branches.",
            "config_updates": {},
            "assay_killswitch": {
                "disable_branch_pre_unroll": True,
                "disable_branch_childworld_runtime": True,
                "disable_mode_replace_pre_unroll": True,
            },
        },
        {
            "name": "static_child_record_mode1_replace_stripped_control",
            "description": "Strip child records while disabling branch pre-unroll/runtime and mode-replace pre-unroll; mode replacement should have no child carrier.",
            "config_updates": {},
            "assay_killswitch": {
                "strip_child_worlds_before_branch": True,
                "disable_branch_pre_unroll": True,
                "disable_branch_childworld_runtime": True,
                "disable_mode_replace_pre_unroll": True,
            },
        },
        {
            "name": "static_child_record_mode1_replace_random_phase",
            "description": "Replace mode1 from a deterministic random child phase while clamping identity; tests phase specificity for mode replacement.",
            "config_updates": {},
            "assay_killswitch": {
                "disable_branch_pre_unroll": True,
                "disable_branch_childworld_runtime": True,
                "disable_mode_replace_pre_unroll": True,
                "mode_replace_child_phase_control": "random",
                "mode_replace_clamp_child_identity": True,
            },
        },
        {
            "name": "static_child_record_mode1_replace_clamped",
            "description": "Static child record mode replacement with support/coherence/budget clamped but true child phase preserved.",
            "config_updates": {},
            "assay_killswitch": {
                "disable_branch_pre_unroll": True,
                "disable_branch_childworld_runtime": True,
                "disable_mode_replace_pre_unroll": True,
                "mode_replace_clamp_child_identity": True,
            },
        },
        {
            "name": "static_child_record_mode1_replace_gate_floor",
            "description": "Static child record mode replacement with identity clamped and support gate floored to test whether weak masking is the failure.",
            "config_updates": {},
            "assay_killswitch": {
                "disable_branch_pre_unroll": True,
                "disable_branch_childworld_runtime": True,
                "disable_mode_replace_pre_unroll": True,
                "mode_replace_clamp_child_identity": True,
                "mode_replace_gate_floor": 0.85,
            },
        },
        {
            "name": "static_child_record_mode1_replace_overdrive",
            "description": "Static child record mode replacement with identity clamped, support gate floored, and replacement ratio scaled toward full mode takeover.",
            "config_updates": {},
            "assay_killswitch": {
                "disable_branch_pre_unroll": True,
                "disable_branch_childworld_runtime": True,
                "disable_mode_replace_pre_unroll": True,
                "mode_replace_clamp_child_identity": True,
                "mode_replace_gate_floor": 1.0,
                "mode_replace_ratio_scale": 1.25,
            },
        },
        {
            "name": "phase_roll_budget_clamped",
            "description": "Roll child phase while clamping support/coherence/budget so identity-budget loss cannot explain failure.",
            "config_updates": {},
            "assay_killswitch": {
                "continuation_child_phase_control": "roll",
                "continuation_clamp_child_identity": True,
            },
        },
        {
            "name": "phase_random_budget_clamped",
            "description": "Replace child phase with deterministic random phasors while clamping support/coherence/budget.",
            "config_updates": {},
            "assay_killswitch": {
                "continuation_child_phase_control": "random",
                "continuation_clamp_child_identity": True,
            },
        },
        {
            "name": "phase_zero_budget_clamped",
            "description": "Replace child phase with a zero-angle phasor while clamping support/coherence/budget.",
            "config_updates": {},
            "assay_killswitch": {
                "continuation_child_phase_control": "zero",
                "continuation_clamp_child_identity": True,
            },
        },
        {
            "name": "phase_cross_child_budget_clamped",
            "description": "Substitute another live child's phase when available while clamping support/coherence/budget.",
            "config_updates": {},
            "assay_killswitch": {
                "continuation_child_phase_control": "cross_child",
                "continuation_clamp_child_identity": True,
            },
        },
        {
            "name": "no_child_local_ifs",
            "description": "Disable child-local IFS recurrence in the runtime config.",
            "config_updates": {"child_local_ifs_enabled": False},
            "assay_killswitch": {},
        },
        {
            "name": "no_child_coherence_retention",
            "description": "Disable child-local coherence retention and floor support.",
            "config_updates": {
                "child_local_coherence_retention_enabled": False,
                "child_local_coherence_retention_mix": 0.0,
                "child_local_coherence_floor": 0.0,
            },
            "assay_killswitch": {},
        },
        {
            "name": "no_child_causal_gate",
            "description": "Disable causal gate on child-local coherence retention/floor.",
            "config_updates": {
                "child_local_coherence_causal_gate_enabled": False,
                "child_local_coherence_causal_gate_floor": 0.0,
            },
            "assay_killswitch": {},
        },
    ]


def main() -> None:
    ap = argparse.ArgumentParser(description="Run childworld nested-sibling causality kill-switch suite.")
    ap.add_argument("--config", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--depth", type=int, default=3)
    ap.add_argument("--fork-depth", type=int, default=1)
    ap.add_argument("--fork-selector", default="max_live_child", choices=["fixed", "first_live_child", "max_live_child"])
    ap.add_argument("--live-child-threshold", type=float, default=0.03)
    ap.add_argument("--mode", default="native_multimode_childworld", choices=["no_promotion", "passive_packets", "active_packets", "native_multimode", "native_multimode_childworld"])
    ap.add_argument("--case-json", default=None)
    ap.add_argument("--only", default="", help="Comma-separated variant names to run. Empty runs all.")
    args = ap.parse_args()

    config_path = Path(args.config)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    base_payload = _load_payload(config_path)
    if args.case_json:
        cases = json.loads(Path(args.case_json).read_text(encoding="utf-8-sig"))
    else:
        cases = _default_seeded_nested_cases()
    selected = {name.strip() for name in str(args.only).split(",") if name.strip()}
    specs = [spec for spec in _variant_specs() if not selected or spec["name"] in selected]
    if selected and len(specs) != len(selected):
        known = {spec["name"] for spec in _variant_specs()}
        missing = sorted(selected - known)
        raise ValueError(f"Unknown variant(s): {missing}")

    rows: dict[str, Any] = {}
    reports: dict[str, Any] = {}
    for spec in specs:
        name = str(spec["name"])
        variant_dir = out_dir / name
        cfg_path = _write_variant_config(
            base_payload,
            variant_dir / "variant_config.json",
            dict(spec.get("config_updates", {})),
        )
        killswitch = dict(spec.get("assay_killswitch", {}))
        (variant_dir / "assay_killswitch.json").write_text(json.dumps(killswitch, indent=2), encoding="utf-8")
        report = evaluate_nested_commitment_report(
            config_path=cfg_path,
            out_dir=variant_dir,
            device_name=str(args.device),
            depth=int(args.depth),
            fork_depth=int(args.fork_depth),
            fork_selector=str(args.fork_selector),
            live_child_threshold=float(args.live_child_threshold),
            mode=str(args.mode),
            cases=cases,
            assay_killswitch=killswitch,
        )
        reports[name] = report
        rows[name] = {
            "name": name,
            "description": spec.get("description", ""),
            "config_path": str(cfg_path),
            "out_dir": str(variant_dir),
            "config_updates": dict(spec.get("config_updates", {})),
            "assay_killswitch": killswitch,
            "metrics": _metric_row(report),
        }

    positive = rows.get("positive_control", {}).get("metrics", {})
    decisions = {
        name: _decision_for_variant(name, row["metrics"], positive)
        for name, row in rows.items()
    }
    break_variants = [
        name for name, decision in decisions.items()
        if name != "positive_control" and bool(decision.get("breaks_continuation_sibling", False))
    ]
    survive_variants = [
        name for name, decision in decisions.items()
        if name != "positive_control" and bool(decision.get("preserves_continuation_sibling", False))
    ]
    summary = {
        "schema": "circleworld_childworld_causality_killswitch_v1",
        "config": str(config_path),
        "out_dir": str(out_dir),
        "device": str(args.device),
        "depth": int(args.depth),
        "fork_depth": int(args.fork_depth),
        "fork_selector": str(args.fork_selector),
        "live_child_threshold": float(args.live_child_threshold),
        "mode": str(args.mode),
        "variant_order": [str(spec["name"]) for spec in specs],
        "rows": rows,
        "decisions": decisions,
        "break_variants": break_variants,
        "survive_variants": survive_variants,
        "overall_read": (
            "causality_supported_no_kill_survivors"
            if positive.get("mean_assay_continuation_nested_sibling_fraction", 0.0) > 0.0 and break_variants and not survive_variants
            else "mixed_mechanism_surviving_carrier_requires_inspection"
            if positive.get("mean_assay_continuation_nested_sibling_fraction", 0.0) > 0.0 and survive_variants
            else "no_positive_signal"
        ),
    }
    summary_path = out_dir / "childworld_causality_killswitch_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
