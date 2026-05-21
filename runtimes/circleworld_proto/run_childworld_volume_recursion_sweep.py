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
    "mean_assay_mode_replace_max_nested_sibling_readiness",
    "mean_child_record_survival_score",
    "mean_parent_mode_conversion_sibling_fraction",
    "mean_mode_replace_conversion_score",
    "mean_direct_readout_dependency",
    "mean_final_direct_mix_shortcut_score",
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
    "mean_child_predictive_target_residual_abs",
    "mean_child_predictive_raw_delta_abs",
    "mean_child_predictive_candidate_delta_abs",
    "mean_child_predictive_oracle_residual_reduction",
    "max_child_predictive_oracle_residual_reduction",
    "mean_child_predictive_learned_bridge_score",
    "mean_child_predictive_learned_bridge_acceptance",
    "mean_child_predictive_learned_bridge_authority_acceptance",
    "mean_child_predictive_learned_bridge_boundary_score",
    "mean_child_predictive_learned_bridge_volume_score",
    "mean_child_predictive_learned_bridge_support_score",
    "mean_child_predictive_learned_bridge_coherence_score",
    "mean_child_predictive_learned_bridge_delta_scale",
    "mean_child_predictive_learned_bridge_prewrite_gate",
    "mean_child_predictive_learned_bridge_effective_delta_scale",
    "mean_parent_ontology_charge",
    "max_parent_ontology_charge",
    "mean_parent_ontology_mode1_occupancy_after",
    "mean_assay_mode_replace_mode1_replace_parent_ontology_used",
    "mean_assay_mode_replace_mode1_replace_parent_ontology_charge",
    "mean_assay_mode_replace_mode1_replace_parent_ontology_carrier_used",
    "mean_assay_mode_replace_mode1_replace_parent_ontology_mode1_occupancy_after",
    "mean_world_jump_penalty",
    "mean_over_rigid_fraction",
    "mean_readout_sibling_response",
    "mean_branch_identity_qualified_carry",
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
    payload["childworld_volume_recursion_updates"] = updates
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return out_path


def _variant_specs() -> list[dict[str, Any]]:
    static_mode_replace = {
        "disable_branch_pre_unroll": True,
        "disable_branch_childworld_runtime": True,
        "disable_mode_replace_pre_unroll": True,
        "mode_replace_gate_floor": 0.20,
        "mode_replace_ratio_scale": 1.65,
    }
    return [
        {
            "name": "positive_control",
            "description": "Current childworld ontology assay without child-volume surgery.",
            "config_updates": {},
            "assay_killswitch": {},
        },
        {
            "name": "volume_x16_runtime",
            "description": "Runtime volume test: raise active child cap and reduce spawn threshold.",
            "config_updates": {
                "child_max_worlds": 16,
                "child_spawn_threshold": 0.12,
                "child_support_window": 8,
                "max_promotions": 24,
            },
            "assay_killswitch": {},
        },
        {
            "name": "volume_x16_same_window_runtime",
            "description": "Runtime volume test preserving the trained support-window geometry.",
            "config_updates": {
                "child_max_worlds": 16,
                "child_spawn_threshold": 0.10,
                "max_promotions": 24,
            },
            "assay_killswitch": {},
        },
        {
            "name": "volume_x64_runtime",
            "description": "Runtime volume test: much larger child cap and narrower child support windows.",
            "config_updates": {
                "child_max_worlds": 64,
                "child_spawn_threshold": 0.08,
                "child_support_window": 6,
                "max_promotions": 48,
            },
            "assay_killswitch": {},
        },
        {
            "name": "volume_x64_same_window_runtime",
            "description": "Large runtime volume while preserving the trained support-window geometry.",
            "config_updates": {
                "child_max_worlds": 64,
                "child_spawn_threshold": 0.06,
                "max_promotions": 48,
            },
            "assay_killswitch": {},
        },
        {
            "name": "volume_x256_runtime",
            "description": "Stress-volume runtime test. Intended for short seeded probes, not broad sweeps.",
            "config_updates": {
                "child_max_worlds": 256,
                "child_spawn_threshold": 0.05,
                "child_support_window": 4,
                "max_promotions": 96,
            },
            "assay_killswitch": {},
        },
        {
            "name": "child_volume_top1_static",
            "description": "Static control: keep only the strongest child record.",
            "config_updates": {},
            "assay_killswitch": {
                **static_mode_replace,
                "child_volume_control": "top1",
            },
        },
        {
            "name": "child_volume_drop_strongest",
            "description": "Remove the strongest child before branch/readout to test single-record dominance.",
            "config_updates": {},
            "assay_killswitch": {
                "child_volume_control": "drop_strongest",
            },
        },
        {
            "name": "fragment_x4_static_mode_replace",
            "description": "Split the strongest child into four time-local fragments, preserving phase.",
            "config_updates": {},
            "assay_killswitch": {
                **static_mode_replace,
                "fragment_child_records": True,
                "fragment_child_parts": 4,
                "fragment_child_axis": "support_window",
                "fragment_child_phase_mode": "preserve",
                "fragment_child_budget_scale": 1.0,
                "fragment_child_replace_source": True,
            },
        },
        {
            "name": "fragment_x16_static_mode_replace",
            "description": "Split the strongest child into up to sixteen time-local fragments.",
            "config_updates": {},
            "assay_killswitch": {
                **static_mode_replace,
                "fragment_child_records": True,
                "fragment_child_parts": 16,
                "fragment_child_axis": "support_window",
                "fragment_child_phase_mode": "preserve",
                "fragment_child_budget_scale": 1.0,
                "fragment_child_replace_source": True,
            },
        },
        {
            "name": "fragment_freq_x8_static_mode_replace",
            "description": "Split the strongest child into frequency-local fragments to test regime specificity by band.",
            "config_updates": {},
            "assay_killswitch": {
                **static_mode_replace,
                "fragment_child_records": True,
                "fragment_child_parts": 8,
                "fragment_child_axis": "freq",
                "fragment_child_phase_mode": "preserve",
                "fragment_child_budget_scale": 1.0,
                "fragment_child_replace_source": True,
            },
        },
        {
            "name": "fragment_random_phase_negative",
            "description": "Negative control: fragment support but randomize fragment phase.",
            "config_updates": {},
            "assay_killswitch": {
                **static_mode_replace,
                "fragment_child_records": True,
                "fragment_child_parts": 4,
                "fragment_child_axis": "support_window",
                "fragment_child_phase_mode": "random",
                "fragment_child_budget_scale": 1.0,
                "fragment_child_replace_source": True,
            },
        },
        {
            "name": "predictive_probe_only",
            "description": "Measure whether children reduce parent next-state residual, without applying binding.",
            "config_updates": {},
            "assay_killswitch": {
                "enable_child_predictive_probe": True,
            },
        },
        {
            "name": "predictive_bind_static_mode_replace",
            "description": "Allow child predictive residual reduction to bind into parent mode 1 before static mode replacement.",
            "config_updates": {},
            "assay_killswitch": {
                **static_mode_replace,
                "enable_child_predictive_probe": True,
                "enable_child_predictive_assimilation": True,
                "child_predictive_bind_gain": 1.20,
                "child_predictive_logit_gain": 0.60,
                "child_predictive_gate_floor": 0.05,
            },
        },
        {
            "name": "predictive_oracle_bind_static_mode_replace",
            "description": "Upper bound: bind the measured child-vs-parent future residual instead of raw child phase.",
            "config_updates": {},
            "assay_killswitch": {
                **static_mode_replace,
                "enable_child_predictive_probe": True,
                "enable_child_predictive_assimilation": True,
                "child_predictive_candidate": "oracle_parent_residual",
                "child_predictive_bind_gain": 1.00,
                "child_predictive_logit_gain": 0.60,
                "child_predictive_gate_floor": 0.05,
            },
        },
        {
            "name": "learned_parent_bridge_static_mode_replace",
            "description": "Non-oracle scout bridge: grant parent ontology authority from child compatibility, not measured residuals.",
            "config_updates": {},
            "assay_killswitch": {
                **static_mode_replace,
                "enable_child_predictive_probe": True,
                "enable_child_predictive_assimilation": True,
                "child_predictive_candidate": "learned_parent_bridge",
                "child_predictive_bind_gain": 1.00,
                "child_predictive_logit_gain": 0.60,
                "child_predictive_gate_floor": 0.05,
                "enable_parent_ontology_field": True,
                "parent_ontology_charge_gain": 7.0,
                "parent_ontology_use_in_mode_replace": True,
                "parent_ontology_mode_replace_source": "child_phase",
                "parent_ontology_mode1_logit_boost": 22.0,
                "parent_ontology_mode0_logit_suppress": 14.5,
                "parent_ontology_support_boost": 11.0,
                "learned_parent_bridge_score_threshold": 0.50,
                "learned_parent_bridge_acceptance_target": 0.13,
                "learned_parent_bridge_delta_scale": 0.0,
            },
        },
        {
            "name": "parent_ontology_learned_parent_bridge_gate54_static_mode_replace",
            "description": "Non-oracle learned bridge at the previous gate54 near-miss authority setting.",
            "config_updates": {},
            "assay_killswitch": {
                **static_mode_replace,
                "enable_child_predictive_probe": True,
                "enable_child_predictive_assimilation": True,
                "child_predictive_candidate": "learned_parent_bridge",
                "child_predictive_bind_gain": 1.00,
                "child_predictive_logit_gain": 0.60,
                "child_predictive_gate_floor": 0.05,
                "enable_parent_ontology_field": True,
                "parent_ontology_charge_gain": 6.9,
                "parent_ontology_use_in_mode_replace": True,
                "parent_ontology_mode_replace_source": "child_phase",
                "parent_ontology_mode1_logit_boost": 21.6,
                "parent_ontology_mode0_logit_suppress": 14.3,
                "parent_ontology_support_boost": 10.9,
                "parent_ontology_gate_floor": 0.54,
                "learned_parent_bridge_score_threshold": 0.50,
                "learned_parent_bridge_acceptance_target": 0.13,
                "learned_parent_bridge_delta_scale": 0.0,
            },
        },
        {
            "name": "parent_ontology_learned_parent_bridge_gate545_static_mode_replace",
            "description": "Non-oracle learned bridge at the scout-recommended midpoint between gate54 and gate55.",
            "config_updates": {},
            "assay_killswitch": {
                **static_mode_replace,
                "enable_child_predictive_probe": True,
                "enable_child_predictive_assimilation": True,
                "child_predictive_candidate": "learned_parent_bridge",
                "child_predictive_bind_gain": 1.00,
                "child_predictive_logit_gain": 0.60,
                "child_predictive_gate_floor": 0.05,
                "enable_parent_ontology_field": True,
                "parent_ontology_charge_gain": 6.95,
                "parent_ontology_use_in_mode_replace": True,
                "parent_ontology_mode_replace_source": "child_phase",
                "parent_ontology_mode1_logit_boost": 21.8,
                "parent_ontology_mode0_logit_suppress": 14.4,
                "parent_ontology_support_boost": 10.95,
                "parent_ontology_gate_floor": 0.545,
                "learned_parent_bridge_score_threshold": 0.50,
                "learned_parent_bridge_acceptance_target": 0.13,
                "learned_parent_bridge_delta_scale": 0.0,
            },
        },
        {
            "name": "parent_ontology_learned_parent_bridge_gate55_static_mode_replace",
            "description": "Non-oracle learned bridge at the known gate55 positive authority setting.",
            "config_updates": {},
            "assay_killswitch": {
                **static_mode_replace,
                "enable_child_predictive_probe": True,
                "enable_child_predictive_assimilation": True,
                "child_predictive_candidate": "learned_parent_bridge",
                "child_predictive_bind_gain": 1.00,
                "child_predictive_logit_gain": 0.60,
                "child_predictive_gate_floor": 0.05,
                "enable_parent_ontology_field": True,
                "parent_ontology_charge_gain": 7.0,
                "parent_ontology_use_in_mode_replace": True,
                "parent_ontology_mode_replace_source": "child_phase",
                "parent_ontology_mode1_logit_boost": 22.0,
                "parent_ontology_mode0_logit_suppress": 14.5,
                "parent_ontology_support_boost": 11.0,
                "parent_ontology_gate_floor": 0.55,
                "learned_parent_bridge_score_threshold": 0.50,
                "learned_parent_bridge_acceptance_target": 0.13,
                "learned_parent_bridge_delta_scale": 0.0,
            },
        },
        {
            "name": "parent_ontology_learned_parent_bridge_gate55_rawdelta001_static_mode_replace",
            "description": "Non-oracle learned bridge at gate55 with a tiny raw-child phase delta tail.",
            "config_updates": {},
            "assay_killswitch": {
                **static_mode_replace,
                "enable_child_predictive_probe": True,
                "enable_child_predictive_assimilation": True,
                "child_predictive_candidate": "learned_parent_bridge",
                "child_predictive_bind_gain": 1.00,
                "child_predictive_logit_gain": 0.60,
                "child_predictive_gate_floor": 0.05,
                "enable_parent_ontology_field": True,
                "parent_ontology_charge_gain": 7.0,
                "parent_ontology_use_in_mode_replace": True,
                "parent_ontology_mode_replace_source": "child_phase",
                "parent_ontology_mode1_logit_boost": 22.0,
                "parent_ontology_mode0_logit_suppress": 14.5,
                "parent_ontology_support_boost": 11.0,
                "parent_ontology_gate_floor": 0.55,
                "learned_parent_bridge_score_threshold": 0.50,
                "learned_parent_bridge_acceptance_target": 0.13,
                "learned_parent_bridge_delta_scale": 0.001,
            },
        },
        {
            "name": "parent_ontology_learned_parent_bridge_gate55_rawdelta002_static_mode_replace",
            "description": "Non-oracle learned bridge at gate55 with raw-child delta scaled to oracle-sized magnitude.",
            "config_updates": {},
            "assay_killswitch": {
                **static_mode_replace,
                "enable_child_predictive_probe": True,
                "enable_child_predictive_assimilation": True,
                "child_predictive_candidate": "learned_parent_bridge",
                "child_predictive_bind_gain": 1.00,
                "child_predictive_logit_gain": 0.60,
                "child_predictive_gate_floor": 0.05,
                "enable_parent_ontology_field": True,
                "parent_ontology_charge_gain": 7.0,
                "parent_ontology_use_in_mode_replace": True,
                "parent_ontology_mode_replace_source": "child_phase",
                "parent_ontology_mode1_logit_boost": 22.0,
                "parent_ontology_mode0_logit_suppress": 14.5,
                "parent_ontology_support_boost": 11.0,
                "parent_ontology_gate_floor": 0.55,
                "learned_parent_bridge_score_threshold": 0.50,
                "learned_parent_bridge_acceptance_target": 0.13,
                "learned_parent_bridge_delta_scale": 0.002,
            },
        },
        {
            "name": "parent_ontology_learned_parent_bridge_gate55_rawdelta005_static_mode_replace",
            "description": "Non-oracle learned bridge at gate55 with a larger tiny raw-child phase delta tail.",
            "config_updates": {},
            "assay_killswitch": {
                **static_mode_replace,
                "enable_child_predictive_probe": True,
                "enable_child_predictive_assimilation": True,
                "child_predictive_candidate": "learned_parent_bridge",
                "child_predictive_bind_gain": 1.00,
                "child_predictive_logit_gain": 0.60,
                "child_predictive_gate_floor": 0.05,
                "enable_parent_ontology_field": True,
                "parent_ontology_charge_gain": 7.0,
                "parent_ontology_use_in_mode_replace": True,
                "parent_ontology_mode_replace_source": "child_phase",
                "parent_ontology_mode1_logit_boost": 22.0,
                "parent_ontology_mode0_logit_suppress": 14.5,
                "parent_ontology_support_boost": 11.0,
                "parent_ontology_gate_floor": 0.55,
                "learned_parent_bridge_score_threshold": 0.50,
                "learned_parent_bridge_acceptance_target": 0.13,
                "learned_parent_bridge_delta_scale": 0.005,
            },
        },
        {
            "name": "parent_ontology_trained_projector_gate55_static_mode_replace",
            "description": "Non-oracle learned bridge at gate55 using offline-trained linear parent residual projector coefficients.",
            "config_updates": {},
            "assay_killswitch": {
                **static_mode_replace,
                "enable_child_predictive_probe": True,
                "enable_child_predictive_assimilation": True,
                "child_predictive_candidate": "learned_parent_bridge",
                "child_predictive_bind_gain": 1.00,
                "child_predictive_logit_gain": 0.60,
                "child_predictive_gate_floor": 0.05,
                "enable_parent_ontology_field": True,
                "parent_ontology_charge_gain": 7.0,
                "parent_ontology_use_in_mode_replace": True,
                "parent_ontology_mode_replace_source": "child_phase",
                "parent_ontology_mode1_logit_boost": 22.0,
                "parent_ontology_mode0_logit_suppress": 14.5,
                "parent_ontology_support_boost": 11.0,
                "parent_ontology_gate_floor": 0.55,
                "learned_parent_bridge_score_threshold": 0.50,
                "learned_parent_bridge_acceptance_target": 0.13,
                "learned_parent_bridge_delta_source": "trained_linear_projector",
                "learned_parent_bridge_delta_scale": 1.0,
            },
        },
        {
            "name": "parent_ontology_trained_projector_delta_gate55_static_mode_replace",
            "description": "Strict non-oracle projector replay: mode replacement consumes the trained ontology delta instead of direct child phase.",
            "config_updates": {},
            "assay_killswitch": {
                **static_mode_replace,
                "enable_child_predictive_probe": True,
                "enable_child_predictive_assimilation": True,
                "child_predictive_candidate": "learned_parent_bridge",
                "child_predictive_bind_gain": 1.00,
                "child_predictive_logit_gain": 0.60,
                "child_predictive_gate_floor": 0.05,
                "enable_parent_ontology_field": True,
                "parent_ontology_charge_gain": 7.0,
                "parent_ontology_use_in_mode_replace": True,
                "parent_ontology_mode_replace_source": "ontology_delta",
                "parent_ontology_mode_replace_delta_gain": 1.0,
                "parent_ontology_mode1_logit_boost": 22.0,
                "parent_ontology_mode0_logit_suppress": 14.5,
                "parent_ontology_support_boost": 11.0,
                "parent_ontology_gate_floor": 0.55,
                "learned_parent_bridge_score_threshold": 0.50,
                "learned_parent_bridge_acceptance_target": 0.13,
                "learned_parent_bridge_delta_source": "trained_linear_projector",
                "learned_parent_bridge_delta_scale": 1.0,
            },
        },
        {
            "name": "parent_ontology_trained_carrier_gate55_static_mode_replace",
            "description": "Non-oracle carrier replay: mode replacement consumes a trained carrier map instead of direct child phase.",
            "config_updates": {},
            "assay_killswitch": {
                **static_mode_replace,
                "enable_child_predictive_probe": True,
                "enable_child_predictive_assimilation": True,
                "child_predictive_candidate": "learned_parent_bridge",
                "child_predictive_bind_gain": 1.00,
                "child_predictive_logit_gain": 0.60,
                "child_predictive_gate_floor": 0.05,
                "enable_parent_ontology_field": True,
                "parent_ontology_charge_gain": 7.0,
                "parent_ontology_use_in_mode_replace": True,
                "parent_ontology_mode_replace_source": "trained_carrier_map",
                "parent_ontology_mode1_logit_boost": 22.0,
                "parent_ontology_mode0_logit_suppress": 14.5,
                "parent_ontology_support_boost": 11.0,
                "parent_ontology_gate_floor": 0.55,
                "learned_parent_bridge_score_threshold": 0.50,
                "learned_parent_bridge_acceptance_target": 0.13,
                "learned_parent_bridge_delta_source": "trained_linear_projector",
                "learned_parent_bridge_delta_scale": 1.0,
            },
        },
        {
            "name": "parent_ontology_trained_phasor_carrier_gate55_static_mode_replace",
            "description": "Non-oracle rich carrier replay: mode replacement consumes a trained unit-phasor carrier map.",
            "config_updates": {},
            "assay_killswitch": {
                **static_mode_replace,
                "enable_child_predictive_probe": True,
                "enable_child_predictive_assimilation": True,
                "child_predictive_candidate": "learned_parent_bridge",
                "child_predictive_bind_gain": 1.00,
                "child_predictive_logit_gain": 0.60,
                "child_predictive_gate_floor": 0.05,
                "enable_parent_ontology_field": True,
                "parent_ontology_charge_gain": 7.0,
                "parent_ontology_use_in_mode_replace": True,
                "parent_ontology_mode_replace_source": "trained_phasor_carrier_map",
                "parent_ontology_mode1_logit_boost": 22.0,
                "parent_ontology_mode0_logit_suppress": 14.5,
                "parent_ontology_support_boost": 11.0,
                "parent_ontology_gate_floor": 0.55,
                "learned_parent_bridge_score_threshold": 0.50,
                "learned_parent_bridge_acceptance_target": 0.13,
                "learned_parent_bridge_delta_source": "trained_linear_projector",
                "learned_parent_bridge_delta_scale": 1.0,
            },
        },
        {
            "name": "parent_ontology_blend_phasor_carrier_gate55_static_mode_replace",
            "description": "Non-oracle rich carrier replay: blend live child phase with trained unit-phasor carrier.",
            "config_updates": {},
            "assay_killswitch": {
                **static_mode_replace,
                "enable_child_predictive_probe": True,
                "enable_child_predictive_assimilation": True,
                "child_predictive_candidate": "learned_parent_bridge",
                "child_predictive_bind_gain": 1.00,
                "child_predictive_logit_gain": 0.60,
                "child_predictive_gate_floor": 0.05,
                "enable_parent_ontology_field": True,
                "parent_ontology_charge_gain": 7.0,
                "parent_ontology_use_in_mode_replace": True,
                "parent_ontology_mode_replace_source": "blend_child_trained_phasor_carrier",
                "parent_ontology_mode1_logit_boost": 22.0,
                "parent_ontology_mode0_logit_suppress": 14.5,
                "parent_ontology_support_boost": 11.0,
                "parent_ontology_gate_floor": 0.55,
                "learned_parent_bridge_score_threshold": 0.50,
                "learned_parent_bridge_acceptance_target": 0.13,
                "learned_parent_bridge_delta_source": "trained_linear_projector",
                "learned_parent_bridge_delta_scale": 1.0,
            },
        },
        {
            "name": "parent_ontology_trained_phasor_authority_gate55_static_mode_replace",
            "description": "Non-oracle rich carrier replay with trained scalar authority replacing the hand acceptance target.",
            "config_updates": {},
            "assay_killswitch": {
                **static_mode_replace,
                "enable_child_predictive_probe": True,
                "enable_child_predictive_assimilation": True,
                "child_predictive_candidate": "learned_parent_bridge",
                "child_predictive_bind_gain": 1.00,
                "child_predictive_logit_gain": 0.60,
                "child_predictive_gate_floor": 0.05,
                "enable_parent_ontology_field": True,
                "parent_ontology_charge_gain": 7.0,
                "parent_ontology_use_in_mode_replace": True,
                "parent_ontology_mode_replace_source": "trained_phasor_carrier_map",
                "parent_ontology_mode1_logit_boost": 22.0,
                "parent_ontology_mode0_logit_suppress": 14.5,
                "parent_ontology_support_boost": 11.0,
                "parent_ontology_gate_floor": 0.55,
                "learned_parent_bridge_score_threshold": 0.50,
                "learned_parent_bridge_authority_source": "trained_linear_authority",
                "learned_parent_bridge_delta_source": "trained_linear_projector",
                "learned_parent_bridge_delta_scale": 1.0,
            },
        },
        {
            "name": "parent_ontology_blend_phasor_authority_gate55_static_mode_replace",
            "description": "Non-oracle blend carrier with trained scalar authority replacing the hand acceptance target.",
            "config_updates": {},
            "assay_killswitch": {
                **static_mode_replace,
                "enable_child_predictive_probe": True,
                "enable_child_predictive_assimilation": True,
                "child_predictive_candidate": "learned_parent_bridge",
                "child_predictive_bind_gain": 1.00,
                "child_predictive_logit_gain": 0.60,
                "child_predictive_gate_floor": 0.05,
                "enable_parent_ontology_field": True,
                "parent_ontology_charge_gain": 7.0,
                "parent_ontology_use_in_mode_replace": True,
                "parent_ontology_mode_replace_source": "blend_child_trained_phasor_carrier",
                "parent_ontology_mode1_logit_boost": 22.0,
                "parent_ontology_mode0_logit_suppress": 14.5,
                "parent_ontology_support_boost": 11.0,
                "parent_ontology_gate_floor": 0.55,
                "learned_parent_bridge_score_threshold": 0.50,
                "learned_parent_bridge_authority_source": "trained_linear_authority",
                "learned_parent_bridge_delta_source": "trained_linear_projector",
                "learned_parent_bridge_delta_scale": 1.0,
            },
        },
        {
            "name": "parent_ontology_trained_phasor_authority_nodelta_gate55_static_mode_replace",
            "description": "Isolation: trained phasor carrier and trained authority, but no learned parent prewrite delta.",
            "config_updates": {},
            "assay_killswitch": {
                **static_mode_replace,
                "enable_child_predictive_probe": True,
                "enable_child_predictive_assimilation": True,
                "child_predictive_candidate": "learned_parent_bridge",
                "child_predictive_bind_gain": 1.00,
                "child_predictive_logit_gain": 0.60,
                "child_predictive_gate_floor": 0.05,
                "enable_parent_ontology_field": True,
                "parent_ontology_charge_gain": 7.0,
                "parent_ontology_use_in_mode_replace": True,
                "parent_ontology_mode_replace_source": "trained_phasor_carrier_map",
                "parent_ontology_mode1_logit_boost": 22.0,
                "parent_ontology_mode0_logit_suppress": 14.5,
                "parent_ontology_support_boost": 11.0,
                "parent_ontology_gate_floor": 0.55,
                "learned_parent_bridge_score_threshold": 0.50,
                "learned_parent_bridge_authority_source": "trained_linear_authority",
                "learned_parent_bridge_prewrite_policy": "disabled",
                "learned_parent_bridge_delta_source": "trained_linear_projector",
                "learned_parent_bridge_delta_scale": 1.0,
            },
        },
        {
            "name": "parent_ontology_trained_phasor_authority_carrier_only_gate55_static_mode_replace",
            "description": "Canonical learned bridge: trained phasor carrier and trained authority with parent prewrite disabled by explicit policy.",
            "config_updates": {},
            "assay_killswitch": {
                **static_mode_replace,
                "enable_child_predictive_probe": True,
                "enable_child_predictive_assimilation": True,
                "child_predictive_candidate": "learned_parent_bridge",
                "child_predictive_bind_gain": 1.00,
                "child_predictive_logit_gain": 0.60,
                "child_predictive_gate_floor": 0.05,
                "enable_parent_ontology_field": True,
                "parent_ontology_charge_gain": 7.0,
                "parent_ontology_use_in_mode_replace": True,
                "parent_ontology_mode_replace_source": "trained_phasor_carrier_map",
                "parent_ontology_mode1_logit_boost": 22.0,
                "parent_ontology_mode0_logit_suppress": 14.5,
                "parent_ontology_support_boost": 11.0,
                "parent_ontology_gate_floor": 0.55,
                "learned_parent_bridge_score_threshold": 0.50,
                "learned_parent_bridge_authority_source": "trained_linear_authority",
                "learned_parent_bridge_prewrite_policy": "carrier_only",
                "learned_parent_bridge_delta_source": "trained_linear_projector",
                "learned_parent_bridge_delta_scale": 1.0,
            },
        },
        {
            "name": "parent_ontology_oracle_residual_trained_phasor_gate55_static_mode_replace",
            "description": "Isolation: oracle parent prewrite residual, but mode replacement consumes the trained unit-phasor carrier.",
            "config_updates": {},
            "assay_killswitch": {
                **static_mode_replace,
                "enable_child_predictive_probe": True,
                "enable_child_predictive_assimilation": True,
                "child_predictive_candidate": "oracle_parent_residual",
                "child_predictive_bind_gain": 1.00,
                "child_predictive_logit_gain": 0.60,
                "child_predictive_gate_floor": 0.05,
                "enable_parent_ontology_field": True,
                "parent_ontology_charge_gain": 7.0,
                "parent_ontology_use_in_mode_replace": True,
                "parent_ontology_mode_replace_source": "trained_phasor_carrier_map",
                "parent_ontology_mode1_logit_boost": 22.0,
                "parent_ontology_mode0_logit_suppress": 14.5,
                "parent_ontology_support_boost": 11.0,
                "parent_ontology_gate_floor": 0.55,
            },
        },
        {
            "name": "parent_ontology_oracle_residual_blend_phasor_gate55_static_mode_replace",
            "description": "Isolation: oracle parent prewrite residual, blended live child phase and trained unit-phasor carrier.",
            "config_updates": {},
            "assay_killswitch": {
                **static_mode_replace,
                "enable_child_predictive_probe": True,
                "enable_child_predictive_assimilation": True,
                "child_predictive_candidate": "oracle_parent_residual",
                "child_predictive_bind_gain": 1.00,
                "child_predictive_logit_gain": 0.60,
                "child_predictive_gate_floor": 0.05,
                "enable_parent_ontology_field": True,
                "parent_ontology_charge_gain": 7.0,
                "parent_ontology_use_in_mode_replace": True,
                "parent_ontology_mode_replace_source": "blend_child_trained_phasor_carrier",
                "parent_ontology_mode1_logit_boost": 22.0,
                "parent_ontology_mode0_logit_suppress": 14.5,
                "parent_ontology_support_boost": 11.0,
                "parent_ontology_gate_floor": 0.55,
            },
        },
        {
            "name": "parent_ontology_oracle_delta_gate55_static_mode_replace",
            "description": "Strict oracle positive control: gate55 authority with mode replacement consuming oracle ontology delta.",
            "config_updates": {},
            "assay_killswitch": {
                **static_mode_replace,
                "enable_child_predictive_probe": True,
                "enable_child_predictive_assimilation": True,
                "child_predictive_candidate": "oracle_parent_residual",
                "child_predictive_bind_gain": 1.00,
                "child_predictive_logit_gain": 0.60,
                "child_predictive_gate_floor": 0.05,
                "enable_parent_ontology_field": True,
                "parent_ontology_charge_gain": 7.0,
                "parent_ontology_use_in_mode_replace": True,
                "parent_ontology_mode_replace_source": "ontology_delta",
                "parent_ontology_mode_replace_delta_gain": 1.0,
                "parent_ontology_mode1_logit_boost": 22.0,
                "parent_ontology_mode0_logit_suppress": 14.5,
                "parent_ontology_support_boost": 11.0,
                "parent_ontology_gate_floor": 0.55,
            },
        },
        {
            "name": "parent_ontology_phase_projector_gate55_tangent_static_mode_replace",
            "description": "Non-oracle phase-law projector: use parent-only tangent direction at gate55 authority.",
            "config_updates": {},
            "assay_killswitch": {
                **static_mode_replace,
                "enable_child_predictive_probe": True,
                "enable_child_predictive_assimilation": True,
                "child_predictive_candidate": "learned_parent_bridge",
                "child_predictive_bind_gain": 1.00,
                "child_predictive_logit_gain": 0.60,
                "child_predictive_gate_floor": 0.05,
                "enable_parent_ontology_field": True,
                "parent_ontology_charge_gain": 7.0,
                "parent_ontology_use_in_mode_replace": True,
                "parent_ontology_mode_replace_source": "child_phase",
                "parent_ontology_mode1_logit_boost": 22.0,
                "parent_ontology_mode0_logit_suppress": 14.5,
                "parent_ontology_support_boost": 11.0,
                "parent_ontology_gate_floor": 0.55,
                "learned_parent_bridge_score_threshold": 0.50,
                "learned_parent_bridge_acceptance_target": 0.13,
                "learned_parent_bridge_delta_source": "parent_tangent",
                "learned_parent_bridge_delta_scale": 1.0,
            },
        },
        {
            "name": "parent_ontology_phase_projector_gate55_anti_tangent_static_mode_replace",
            "description": "Non-oracle phase-law projector: use the opposite parent-only tangent direction.",
            "config_updates": {},
            "assay_killswitch": {
                **static_mode_replace,
                "enable_child_predictive_probe": True,
                "enable_child_predictive_assimilation": True,
                "child_predictive_candidate": "learned_parent_bridge",
                "child_predictive_bind_gain": 1.00,
                "child_predictive_logit_gain": 0.60,
                "child_predictive_gate_floor": 0.05,
                "enable_parent_ontology_field": True,
                "parent_ontology_charge_gain": 7.0,
                "parent_ontology_use_in_mode_replace": True,
                "parent_ontology_mode_replace_source": "child_phase",
                "parent_ontology_mode1_logit_boost": 22.0,
                "parent_ontology_mode0_logit_suppress": 14.5,
                "parent_ontology_support_boost": 11.0,
                "parent_ontology_gate_floor": 0.55,
                "learned_parent_bridge_score_threshold": 0.50,
                "learned_parent_bridge_acceptance_target": 0.13,
                "learned_parent_bridge_delta_source": "anti_parent_tangent",
                "learned_parent_bridge_delta_scale": 1.0,
            },
        },
        {
            "name": "parent_ontology_phase_projector_gate55_child_signed_tangent_static_mode_replace",
            "description": "Non-oracle phase-law projector: parent-only tangent magnitude with child-delta sign.",
            "config_updates": {},
            "assay_killswitch": {
                **static_mode_replace,
                "enable_child_predictive_probe": True,
                "enable_child_predictive_assimilation": True,
                "child_predictive_candidate": "learned_parent_bridge",
                "child_predictive_bind_gain": 1.00,
                "child_predictive_logit_gain": 0.60,
                "child_predictive_gate_floor": 0.05,
                "enable_parent_ontology_field": True,
                "parent_ontology_charge_gain": 7.0,
                "parent_ontology_use_in_mode_replace": True,
                "parent_ontology_mode_replace_source": "child_phase",
                "parent_ontology_mode1_logit_boost": 22.0,
                "parent_ontology_mode0_logit_suppress": 14.5,
                "parent_ontology_support_boost": 11.0,
                "parent_ontology_gate_floor": 0.55,
                "learned_parent_bridge_score_threshold": 0.50,
                "learned_parent_bridge_acceptance_target": 0.13,
                "learned_parent_bridge_delta_source": "child_signed_parent_tangent",
                "learned_parent_bridge_delta_scale": 1.0,
            },
        },
        {
            "name": "parent_ontology_oracle_static_mode_replace",
            "description": "Create parent ontology charge from oracle residual and let mode-replace readout consume that charge.",
            "config_updates": {},
            "assay_killswitch": {
                **static_mode_replace,
                "enable_child_predictive_probe": True,
                "enable_child_predictive_assimilation": True,
                "child_predictive_candidate": "oracle_parent_residual",
                "child_predictive_bind_gain": 1.00,
                "child_predictive_logit_gain": 0.60,
                "child_predictive_gate_floor": 0.05,
                "enable_parent_ontology_field": True,
                "parent_ontology_charge_gain": 2.0,
                "parent_ontology_use_in_mode_replace": True,
                "parent_ontology_mode_replace_source": "ontology_delta",
                "parent_ontology_mode_replace_delta_gain": 1.0,
                "parent_ontology_mode1_logit_boost": 4.0,
                "parent_ontology_mode0_logit_suppress": 2.0,
                "parent_ontology_support_boost": 3.0,
            },
        },
        {
            "name": "parent_ontology_childphase_static_mode_replace",
            "description": "Use ontology charge to expose child phase through parent mode-1 occupancy.",
            "config_updates": {},
            "assay_killswitch": {
                **static_mode_replace,
                "enable_child_predictive_probe": True,
                "enable_child_predictive_assimilation": True,
                "child_predictive_candidate": "oracle_parent_residual",
                "child_predictive_bind_gain": 1.00,
                "child_predictive_logit_gain": 0.60,
                "child_predictive_gate_floor": 0.05,
                "enable_parent_ontology_field": True,
                "parent_ontology_charge_gain": 2.0,
                "parent_ontology_use_in_mode_replace": True,
                "parent_ontology_mode_replace_source": "child_phase",
                "parent_ontology_mode1_logit_boost": 6.0,
                "parent_ontology_mode0_logit_suppress": 3.0,
                "parent_ontology_support_boost": 4.0,
            },
        },
        {
            "name": "parent_ontology_childphase_overdrive_static_mode_replace",
            "description": "Stress parent ontology readout enough to test whether mode-replace can cross the sibling-response baseline.",
            "config_updates": {},
            "assay_killswitch": {
                **static_mode_replace,
                "enable_child_predictive_probe": True,
                "enable_child_predictive_assimilation": True,
                "child_predictive_candidate": "oracle_parent_residual",
                "child_predictive_bind_gain": 1.00,
                "child_predictive_logit_gain": 0.60,
                "child_predictive_gate_floor": 0.05,
                "enable_parent_ontology_field": True,
                "parent_ontology_charge_gain": 4.0,
                "parent_ontology_use_in_mode_replace": True,
                "parent_ontology_mode_replace_source": "child_phase",
                "parent_ontology_mode1_logit_boost": 14.0,
                "parent_ontology_mode0_logit_suppress": 10.0,
                "parent_ontology_support_boost": 8.0,
                "parent_ontology_gate_floor": 0.35,
            },
        },
        {
            "name": "parent_ontology_childphase_maxreadout_static_mode_replace",
            "description": "Hard upper-bound parent ontology readout; allowed to reveal world-jump limits.",
            "config_updates": {},
            "assay_killswitch": {
                **static_mode_replace,
                "enable_child_predictive_probe": True,
                "enable_child_predictive_assimilation": True,
                "child_predictive_candidate": "oracle_parent_residual",
                "child_predictive_bind_gain": 1.00,
                "child_predictive_logit_gain": 0.60,
                "child_predictive_gate_floor": 0.05,
                "enable_parent_ontology_field": True,
                "parent_ontology_charge_gain": 8.0,
                "parent_ontology_use_in_mode_replace": True,
                "parent_ontology_mode_replace_source": "child_phase",
                "parent_ontology_mode1_logit_boost": 24.0,
                "parent_ontology_mode0_logit_suppress": 16.0,
                "parent_ontology_support_boost": 12.0,
                "parent_ontology_gate_floor": 0.60,
            },
        },
        {
            "name": "parent_ontology_childphase_gate45_static_mode_replace",
            "description": "Calibration rung between overdrive and maxreadout: ontology gate floor 0.45.",
            "config_updates": {},
            "assay_killswitch": {
                **static_mode_replace,
                "enable_child_predictive_probe": True,
                "enable_child_predictive_assimilation": True,
                "child_predictive_candidate": "oracle_parent_residual",
                "child_predictive_bind_gain": 1.00,
                "child_predictive_logit_gain": 0.60,
                "child_predictive_gate_floor": 0.05,
                "enable_parent_ontology_field": True,
                "parent_ontology_charge_gain": 6.0,
                "parent_ontology_use_in_mode_replace": True,
                "parent_ontology_mode_replace_source": "child_phase",
                "parent_ontology_mode1_logit_boost": 18.0,
                "parent_ontology_mode0_logit_suppress": 12.0,
                "parent_ontology_support_boost": 10.0,
                "parent_ontology_gate_floor": 0.45,
            },
        },
        {
            "name": "parent_ontology_childphase_gate50_static_mode_replace",
            "description": "Calibration rung: ontology gate floor 0.50.",
            "config_updates": {},
            "assay_killswitch": {
                **static_mode_replace,
                "enable_child_predictive_probe": True,
                "enable_child_predictive_assimilation": True,
                "child_predictive_candidate": "oracle_parent_residual",
                "child_predictive_bind_gain": 1.00,
                "child_predictive_logit_gain": 0.60,
                "child_predictive_gate_floor": 0.05,
                "enable_parent_ontology_field": True,
                "parent_ontology_charge_gain": 6.5,
                "parent_ontology_use_in_mode_replace": True,
                "parent_ontology_mode_replace_source": "child_phase",
                "parent_ontology_mode1_logit_boost": 20.0,
                "parent_ontology_mode0_logit_suppress": 13.5,
                "parent_ontology_support_boost": 10.5,
                "parent_ontology_gate_floor": 0.50,
            },
        },
        {
            "name": "parent_ontology_childphase_gate51_static_mode_replace",
            "description": "Fine calibration rung: ontology gate floor 0.51.",
            "config_updates": {},
            "assay_killswitch": {
                **static_mode_replace,
                "enable_child_predictive_probe": True,
                "enable_child_predictive_assimilation": True,
                "child_predictive_candidate": "oracle_parent_residual",
                "child_predictive_bind_gain": 1.00,
                "child_predictive_logit_gain": 0.60,
                "child_predictive_gate_floor": 0.05,
                "enable_parent_ontology_field": True,
                "parent_ontology_charge_gain": 6.6,
                "parent_ontology_use_in_mode_replace": True,
                "parent_ontology_mode_replace_source": "child_phase",
                "parent_ontology_mode1_logit_boost": 20.4,
                "parent_ontology_mode0_logit_suppress": 13.7,
                "parent_ontology_support_boost": 10.6,
                "parent_ontology_gate_floor": 0.51,
            },
        },
        {
            "name": "parent_ontology_childphase_gate52_static_mode_replace",
            "description": "Fine calibration rung: ontology gate floor 0.52.",
            "config_updates": {},
            "assay_killswitch": {
                **static_mode_replace,
                "enable_child_predictive_probe": True,
                "enable_child_predictive_assimilation": True,
                "child_predictive_candidate": "oracle_parent_residual",
                "child_predictive_bind_gain": 1.00,
                "child_predictive_logit_gain": 0.60,
                "child_predictive_gate_floor": 0.05,
                "enable_parent_ontology_field": True,
                "parent_ontology_charge_gain": 6.7,
                "parent_ontology_use_in_mode_replace": True,
                "parent_ontology_mode_replace_source": "child_phase",
                "parent_ontology_mode1_logit_boost": 20.8,
                "parent_ontology_mode0_logit_suppress": 13.9,
                "parent_ontology_support_boost": 10.7,
                "parent_ontology_gate_floor": 0.52,
            },
        },
        {
            "name": "parent_ontology_childphase_gate53_static_mode_replace",
            "description": "Fine calibration rung: ontology gate floor 0.53.",
            "config_updates": {},
            "assay_killswitch": {
                **static_mode_replace,
                "enable_child_predictive_probe": True,
                "enable_child_predictive_assimilation": True,
                "child_predictive_candidate": "oracle_parent_residual",
                "child_predictive_bind_gain": 1.00,
                "child_predictive_logit_gain": 0.60,
                "child_predictive_gate_floor": 0.05,
                "enable_parent_ontology_field": True,
                "parent_ontology_charge_gain": 6.8,
                "parent_ontology_use_in_mode_replace": True,
                "parent_ontology_mode_replace_source": "child_phase",
                "parent_ontology_mode1_logit_boost": 21.2,
                "parent_ontology_mode0_logit_suppress": 14.1,
                "parent_ontology_support_boost": 10.8,
                "parent_ontology_gate_floor": 0.53,
            },
        },
        {
            "name": "parent_ontology_childphase_gate54_static_mode_replace",
            "description": "Fine calibration rung: ontology gate floor 0.54.",
            "config_updates": {},
            "assay_killswitch": {
                **static_mode_replace,
                "enable_child_predictive_probe": True,
                "enable_child_predictive_assimilation": True,
                "child_predictive_candidate": "oracle_parent_residual",
                "child_predictive_bind_gain": 1.00,
                "child_predictive_logit_gain": 0.60,
                "child_predictive_gate_floor": 0.05,
                "enable_parent_ontology_field": True,
                "parent_ontology_charge_gain": 6.9,
                "parent_ontology_use_in_mode_replace": True,
                "parent_ontology_mode_replace_source": "child_phase",
                "parent_ontology_mode1_logit_boost": 21.6,
                "parent_ontology_mode0_logit_suppress": 14.3,
                "parent_ontology_support_boost": 10.9,
                "parent_ontology_gate_floor": 0.54,
            },
        },
        {
            "name": "parent_ontology_childphase_gate55_static_mode_replace",
            "description": "Calibration rung: ontology gate floor 0.55.",
            "config_updates": {},
            "assay_killswitch": {
                **static_mode_replace,
                "enable_child_predictive_probe": True,
                "enable_child_predictive_assimilation": True,
                "child_predictive_candidate": "oracle_parent_residual",
                "child_predictive_bind_gain": 1.00,
                "child_predictive_logit_gain": 0.60,
                "child_predictive_gate_floor": 0.05,
                "enable_parent_ontology_field": True,
                "parent_ontology_charge_gain": 7.0,
                "parent_ontology_use_in_mode_replace": True,
                "parent_ontology_mode_replace_source": "child_phase",
                "parent_ontology_mode1_logit_boost": 22.0,
                "parent_ontology_mode0_logit_suppress": 14.5,
                "parent_ontology_support_boost": 11.0,
                "parent_ontology_gate_floor": 0.55,
            },
        },
        {
            "name": "parent_ontology_childphase_gate55_no_carrier_static_mode_replace",
            "description": "Causality kill: gate55 authority with child-phase carrier disabled at mode replacement.",
            "config_updates": {},
            "assay_killswitch": {
                **static_mode_replace,
                "enable_child_predictive_probe": True,
                "enable_child_predictive_assimilation": True,
                "child_predictive_candidate": "oracle_parent_residual",
                "child_predictive_bind_gain": 1.00,
                "child_predictive_logit_gain": 0.60,
                "child_predictive_gate_floor": 0.05,
                "enable_parent_ontology_field": True,
                "disable_mode_replace_child_phase_carrier": True,
                "parent_ontology_charge_gain": 7.0,
                "parent_ontology_use_in_mode_replace": True,
                "parent_ontology_mode_replace_source": "child_phase",
                "parent_ontology_mode1_logit_boost": 22.0,
                "parent_ontology_mode0_logit_suppress": 14.5,
                "parent_ontology_support_boost": 11.0,
                "parent_ontology_gate_floor": 0.55,
            },
        },
        {
            "name": "parent_ontology_childphase_gate55_no_charge_static_mode_replace",
            "description": "Causality kill: gate55 child-phase carrier with parent ontology charge zeroed.",
            "config_updates": {},
            "assay_killswitch": {
                **static_mode_replace,
                "enable_child_predictive_probe": True,
                "enable_child_predictive_assimilation": True,
                "child_predictive_candidate": "oracle_parent_residual",
                "child_predictive_bind_gain": 1.00,
                "child_predictive_logit_gain": 0.60,
                "child_predictive_gate_floor": 0.05,
                "enable_parent_ontology_field": True,
                "disable_parent_ontology_charge": True,
                "parent_ontology_charge_gain": 7.0,
                "parent_ontology_use_in_mode_replace": True,
                "parent_ontology_mode_replace_source": "child_phase",
                "parent_ontology_mode1_logit_boost": 22.0,
                "parent_ontology_mode0_logit_suppress": 14.5,
                "parent_ontology_support_boost": 11.0,
                "parent_ontology_gate_floor": 0.55,
            },
        },
        {
            "name": "parent_ontology_childphase_gate55_no_support_gate_static_mode_replace",
            "description": "Causality kill: gate55 child-phase carrier with ontology support-gate authority disabled.",
            "config_updates": {},
            "assay_killswitch": {
                **static_mode_replace,
                "enable_child_predictive_probe": True,
                "enable_child_predictive_assimilation": True,
                "child_predictive_candidate": "oracle_parent_residual",
                "child_predictive_bind_gain": 1.00,
                "child_predictive_logit_gain": 0.60,
                "child_predictive_gate_floor": 0.05,
                "enable_parent_ontology_field": True,
                "disable_parent_ontology_support_gate": True,
                "parent_ontology_charge_gain": 7.0,
                "parent_ontology_use_in_mode_replace": True,
                "parent_ontology_mode_replace_source": "child_phase",
                "parent_ontology_mode1_logit_boost": 22.0,
                "parent_ontology_mode0_logit_suppress": 14.5,
                "parent_ontology_support_boost": 11.0,
                "parent_ontology_gate_floor": 0.55,
            },
        },
        {
            "name": "parent_ontology_childphase_gate55_no_logits_static_mode_replace",
            "description": "Causality kill: gate55 child-phase carrier with mode-logit authority disabled.",
            "config_updates": {},
            "assay_killswitch": {
                **static_mode_replace,
                "enable_child_predictive_probe": True,
                "enable_child_predictive_assimilation": True,
                "child_predictive_candidate": "oracle_parent_residual",
                "child_predictive_bind_gain": 1.00,
                "child_predictive_logit_gain": 0.60,
                "child_predictive_gate_floor": 0.05,
                "enable_parent_ontology_field": True,
                "disable_parent_ontology_mode_logits": True,
                "parent_ontology_charge_gain": 7.0,
                "parent_ontology_use_in_mode_replace": True,
                "parent_ontology_mode_replace_source": "child_phase",
                "parent_ontology_mode1_logit_boost": 22.0,
                "parent_ontology_mode0_logit_suppress": 14.5,
                "parent_ontology_support_boost": 11.0,
                "parent_ontology_gate_floor": 0.55,
            },
        },
        {
            "name": "parent_ontology_blend_static_mode_replace",
            "description": "Blend child phase with oracle residual while boosting parent ontology occupancy.",
            "config_updates": {},
            "assay_killswitch": {
                **static_mode_replace,
                "enable_child_predictive_probe": True,
                "enable_child_predictive_assimilation": True,
                "child_predictive_candidate": "oracle_parent_residual",
                "child_predictive_bind_gain": 1.00,
                "child_predictive_logit_gain": 0.60,
                "child_predictive_gate_floor": 0.05,
                "enable_parent_ontology_field": True,
                "parent_ontology_charge_gain": 2.0,
                "parent_ontology_use_in_mode_replace": True,
                "parent_ontology_mode_replace_source": "blend_child_ontology",
                "parent_ontology_mode_replace_delta_gain": 1.0,
                "parent_ontology_mode1_logit_boost": 5.0,
                "parent_ontology_mode0_logit_suppress": 2.5,
                "parent_ontology_support_boost": 4.0,
            },
        },
        {
            "name": "predictive_oracle_h3_bind_static_mode_replace",
            "description": "Upper bound at a longer horizon: bind measured residual after three parent steps.",
            "config_updates": {},
            "assay_killswitch": {
                **static_mode_replace,
                "enable_child_predictive_probe": True,
                "enable_child_predictive_assimilation": True,
                "child_predictive_candidate": "oracle_parent_residual",
                "child_predictive_target_horizon": 3,
                "child_predictive_bind_gain": 1.00,
                "child_predictive_logit_gain": 0.60,
                "child_predictive_gate_floor": 0.05,
            },
        },
        {
            "name": "predictive_oracle_overdrive10_static_mode_replace",
            "description": "Amplify the oracle parent residual 10x to test whether conversion is only below readout threshold.",
            "config_updates": {},
            "assay_killswitch": {
                **static_mode_replace,
                "enable_child_predictive_probe": True,
                "enable_child_predictive_assimilation": True,
                "child_predictive_candidate": "oracle_parent_residual",
                "child_predictive_bind_gain": 10.0,
                "child_predictive_logit_gain": 1.00,
                "child_predictive_gate_floor": 0.10,
            },
        },
        {
            "name": "predictive_oracle_overdrive50_static_mode_replace",
            "description": "Amplify the oracle parent residual 50x as a hard upper-bound readout stress test.",
            "config_updates": {},
            "assay_killswitch": {
                **static_mode_replace,
                "enable_child_predictive_probe": True,
                "enable_child_predictive_assimilation": True,
                "child_predictive_candidate": "oracle_parent_residual",
                "child_predictive_bind_gain": 50.0,
                "child_predictive_logit_gain": 1.50,
                "child_predictive_gate_floor": 0.10,
            },
        },
        {
            "name": "predictive_blend_oracle50_bind_static_mode_replace",
            "description": "Half raw child phase and half measured parent residual, testing whether child phase needs residual calibration.",
            "config_updates": {},
            "assay_killswitch": {
                **static_mode_replace,
                "enable_child_predictive_probe": True,
                "enable_child_predictive_assimilation": True,
                "child_predictive_candidate": "blend_raw_oracle",
                "child_predictive_oracle_mix": 0.50,
                "child_predictive_bind_gain": 1.00,
                "child_predictive_logit_gain": 0.60,
                "child_predictive_gate_floor": 0.05,
            },
        },
        {
            "name": "fragment_x4_predictive_bind_static_mode_replace",
            "description": "Fragment the strongest child, then bind only predictive fragments into mode 1.",
            "config_updates": {},
            "assay_killswitch": {
                **static_mode_replace,
                "fragment_child_records": True,
                "fragment_child_parts": 4,
                "fragment_child_axis": "support_window",
                "fragment_child_phase_mode": "preserve",
                "fragment_child_budget_scale": 1.0,
                "fragment_child_replace_source": True,
                "enable_child_predictive_probe": True,
                "enable_child_predictive_assimilation": True,
                "child_predictive_bind_gain": 1.20,
                "child_predictive_logit_gain": 0.60,
                "child_predictive_gate_floor": 0.05,
            },
        },
        {
            "name": "grandchild_x2_predictive_bind_static_mode_replace",
            "description": "Flatten proto-grandchildren, then bind predictive child/grandchild records into mode 1.",
            "config_updates": {},
            "assay_killswitch": {
                **static_mode_replace,
                "enable_grandchild_probe": True,
                "grandchild_probe_parts": 2,
                "grandchild_probe_depth": 1,
                "grandchild_probe_budget_scale": 0.70,
                "grandchild_probe_replace_parent": True,
                "grandchild_probe_flatten_to_child_worlds": True,
                "enable_child_predictive_probe": True,
                "enable_child_predictive_assimilation": True,
                "child_predictive_bind_gain": 1.20,
                "child_predictive_logit_gain": 0.60,
                "child_predictive_gate_floor": 0.05,
            },
        },
        {
            "name": "grandchild_x2_static_mode_replace",
            "description": "Flatten two proto-grandchildren into the child return path and replace the broad parent child.",
            "config_updates": {},
            "assay_killswitch": {
                **static_mode_replace,
                "enable_grandchild_probe": True,
                "grandchild_probe_parts": 2,
                "grandchild_probe_depth": 1,
                "grandchild_probe_budget_scale": 0.70,
                "grandchild_probe_replace_parent": True,
                "grandchild_probe_flatten_to_child_worlds": True,
            },
        },
        {
            "name": "grandchild_depth2_x2_static_mode_replace",
            "description": "Flatten depth-2 proto-grandchildren to test whether recursive narrowing helps.",
            "config_updates": {},
            "assay_killswitch": {
                **static_mode_replace,
                "enable_grandchild_probe": True,
                "grandchild_probe_parts": 2,
                "grandchild_probe_depth": 2,
                "grandchild_probe_budget_scale": 0.70,
                "grandchild_probe_replace_parent": True,
                "grandchild_probe_flatten_to_child_worlds": True,
            },
        },
        {
            "name": "fragment_x4_grandchild_x2_static_mode_replace",
            "description": "Combine first-order fragmentation with flattened proto-grandchildren.",
            "config_updates": {},
            "assay_killswitch": {
                **static_mode_replace,
                "fragment_child_records": True,
                "fragment_child_parts": 4,
                "fragment_child_axis": "support_window",
                "fragment_child_phase_mode": "preserve",
                "fragment_child_budget_scale": 1.0,
                "fragment_child_replace_source": True,
                "enable_grandchild_probe": True,
                "grandchild_probe_parts": 2,
                "grandchild_probe_depth": 1,
                "grandchild_probe_budget_scale": 0.70,
                "grandchild_probe_replace_parent": True,
                "grandchild_probe_flatten_to_child_worlds": True,
            },
        },
    ]


def _load_cases(case_json: str | None) -> list[dict[str, Any]]:
    if not case_json:
        return _default_seeded_nested_cases()
    raw = json.loads(Path(case_json).read_text(encoding="utf-8-sig"))
    if isinstance(raw, dict):
        return [{"name": str(name), "reference_wav": str(path)} for name, path in raw.items()]
    if not isinstance(raw, list):
        raise ValueError("--case-json must be a JSON list or object")
    return raw


def _load_json_object(path_or_json: str | None) -> dict[str, Any]:
    if not path_or_json:
        return {}
    raw = str(path_or_json)
    maybe_path = Path(raw)
    payload = json.loads(maybe_path.read_text(encoding="utf-8-sig")) if maybe_path.exists() else json.loads(raw)
    if not isinstance(payload, dict):
        raise ValueError("Expected a JSON object")
    return {str(k): v for k, v in payload.items()}


def _metric_row(report: dict[str, Any]) -> dict[str, Any]:
    row = {key: report.get(key) for key in METRIC_KEYS}
    row["num_cases"] = report.get("num_cases")
    row["verdict_counts"] = report.get("verdict_counts", {})
    row["assay_family_nested_sibling_counts"] = report.get("assay_family_nested_sibling_counts", {})
    row["assay_family_branch_counts"] = report.get("assay_family_branch_counts", {})
    return row


def _decision_for_variant(name: str, row: dict[str, Any], positive: dict[str, Any]) -> dict[str, Any]:
    pos_conversion = float(positive.get("mean_mode_replace_conversion_score", 0.0) or 0.0)
    conversion = float(row.get("mean_mode_replace_conversion_score", 0.0) or 0.0)
    pos_mode_fraction = float(positive.get("mean_assay_mode_replace_nested_sibling_fraction", 0.0) or 0.0)
    mode_fraction = float(row.get("mean_assay_mode_replace_nested_sibling_fraction", 0.0) or 0.0)
    pos_shortcut = float(positive.get("mean_final_direct_mix_shortcut_score", 0.0) or 0.0)
    shortcut = float(row.get("mean_final_direct_mix_shortcut_score", 0.0) or 0.0)
    world_jump = float(row.get("mean_world_jump_penalty", 0.0) or 0.0)
    fragments = float(row.get("mean_assay_child_fragment_count", 0.0) or 0.0)
    grandchildren = float(row.get("mean_assay_grandchild_count", 0.0) or 0.0)
    active_count = float(row.get("mean_child_volume_active_count", 0.0) or 0.0)
    effective_count = float(row.get("mean_child_volume_effective_count", 0.0) or 0.0)
    predictive_positive = float(row.get("mean_child_predictive_positive_residual_reduction", 0.0) or 0.0)
    predictive_binding = float(row.get("mean_child_predictive_ontology_binding_mass", 0.0) or 0.0)
    predictive_applied = float(row.get("mean_child_predictive_applied_fraction", 0.0) or 0.0)
    oracle_reduction = float(row.get("mean_child_predictive_oracle_residual_reduction", 0.0) or 0.0)
    learned_bridge_score = float(row.get("mean_child_predictive_learned_bridge_score", 0.0) or 0.0)
    learned_bridge_acceptance = float(row.get("mean_child_predictive_learned_bridge_acceptance", 0.0) or 0.0)
    learned_bridge_authority = float(row.get("mean_child_predictive_learned_bridge_authority_acceptance", 0.0) or 0.0)
    learned_bridge_prewrite_gate = float(row.get("mean_child_predictive_learned_bridge_prewrite_gate", 0.0) or 0.0)
    learned_bridge_effective_delta_scale = float(row.get("mean_child_predictive_learned_bridge_effective_delta_scale", 0.0) or 0.0)
    target_residual_abs = float(row.get("mean_child_predictive_target_residual_abs", 0.0) or 0.0)
    raw_delta_abs = float(row.get("mean_child_predictive_raw_delta_abs", 0.0) or 0.0)
    conversion_gain = conversion - pos_conversion
    mode_fraction_gain = mode_fraction - pos_mode_fraction
    shortcut_drop = pos_shortcut - shortcut
    if name == "positive_control":
        read = "positive_control"
    elif conversion_gain > 0.0 and world_jump <= 0.0:
        read = "conversion_improved"
    elif learned_bridge_score > 0.0 and predictive_binding > 0.0 and conversion <= pos_conversion:
        read = "learned_bridge_bound_without_conversion"
    elif predictive_binding > 0.0 and conversion <= pos_conversion:
        read = "predictive_binding_without_conversion"
    elif oracle_reduction > 0.0 and predictive_binding <= 0.0:
        read = "oracle_residual_signal_not_bound"
    elif predictive_positive > 0.0 and predictive_binding <= 0.0:
        read = "predictive_signal_not_bound"
    elif mode_fraction_gain > 0.0 and world_jump <= 0.0:
        read = "mode_replace_fraction_improved"
    elif (fragments > 0.0 or grandchildren > 0.0 or active_count > 1.0) and conversion <= pos_conversion:
        read = "volume_without_conversion"
    elif world_jump > 0.0:
        read = "world_jump_regression"
    else:
        read = "neutral_or_inconclusive"
    return {
        "variant": name,
        "read": read,
        "mode_replace_conversion_score": conversion,
        "positive_mode_replace_conversion_score": pos_conversion,
        "mode_replace_conversion_gain": conversion_gain,
        "mode_replace_nested_sibling_fraction": mode_fraction,
        "positive_mode_replace_nested_sibling_fraction": pos_mode_fraction,
        "mode_replace_fraction_gain": mode_fraction_gain,
        "final_direct_mix_shortcut_score": shortcut,
        "positive_final_direct_mix_shortcut_score": pos_shortcut,
        "final_direct_mix_shortcut_drop": shortcut_drop,
        "child_volume_active_count": active_count,
        "child_volume_effective_count": effective_count,
        "child_fragment_count": fragments,
        "grandchild_count": grandchildren,
        "max_child_generation": float(row.get("max_assay_child_generation", 0.0) or 0.0),
        "child_predictive_positive_residual_reduction": predictive_positive,
        "child_predictive_oracle_residual_reduction": oracle_reduction,
        "child_predictive_target_residual_abs": target_residual_abs,
        "child_predictive_raw_delta_abs": raw_delta_abs,
        "child_predictive_ontology_binding_mass": predictive_binding,
        "child_predictive_applied_fraction": predictive_applied,
        "child_predictive_learned_bridge_score": learned_bridge_score,
        "child_predictive_learned_bridge_acceptance": learned_bridge_acceptance,
        "child_predictive_learned_bridge_authority_acceptance": learned_bridge_authority,
        "child_predictive_learned_bridge_prewrite_gate": learned_bridge_prewrite_gate,
        "child_predictive_learned_bridge_effective_delta_scale": learned_bridge_effective_delta_scale,
        "child_predictive_learned_bridge_boundary_score": float(row.get("mean_child_predictive_learned_bridge_boundary_score", 0.0) or 0.0),
        "child_predictive_learned_bridge_volume_score": float(row.get("mean_child_predictive_learned_bridge_volume_score", 0.0) or 0.0),
        "child_predictive_learned_bridge_support_score": float(row.get("mean_child_predictive_learned_bridge_support_score", 0.0) or 0.0),
        "child_predictive_learned_bridge_coherence_score": float(row.get("mean_child_predictive_learned_bridge_coherence_score", 0.0) or 0.0),
        "child_predictive_learned_bridge_delta_scale": float(row.get("mean_child_predictive_learned_bridge_delta_scale", 0.0) or 0.0),
        "world_jump_penalty": world_jump,
        "supports_information_density_hypothesis": bool(
            (conversion_gain > 0.0 or mode_fraction_gain > 0.0)
            and world_jump <= 0.0
            and (fragments > 0.0 or grandchildren > 0.0 or effective_count > 1.0)
        ),
        "supports_predictive_binding_hypothesis": bool(
            (conversion_gain > 0.0 or mode_fraction_gain > 0.0)
            and world_jump <= 0.0
            and predictive_binding > 0.0
        ),
        "supports_learned_bridge_hypothesis": bool(
            (conversion_gain > 0.0 or mode_fraction_gain > 0.0)
            and world_jump <= 0.0
            and learned_bridge_score > 0.0
            and predictive_binding > 0.0
        ),
    }


def _write_markdown(summary: dict[str, Any], out_path: Path) -> None:
    lines: list[str] = [
        "# Childworld Volume / Recursion Sweep",
        "",
        "This assay tests whether narrower child regimes or flattened proto-grandchildren convert child records into parent-mode ontology.",
        "",
        "## Decisions",
        "",
        "| variant | read | conversion | mode-replace siblings | shortcut | active children | fragments | grandchildren | predictive binding |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, decision in summary.get("decisions", {}).items():
        row = summary.get("variants", {}).get(name, {}).get("metrics", {})
        lines.append(
            "| {name} | {read} | {conv:.4f} | {mode:.4f} | {shortcut:.4f} | {active:.2f} | {frags:.2f} | {grand:.2f} | {binding:.5f} |".format(
                name=name,
                read=decision.get("read", ""),
                conv=float(row.get("mean_mode_replace_conversion_score", 0.0) or 0.0),
                mode=float(row.get("mean_assay_mode_replace_nested_sibling_fraction", 0.0) or 0.0),
                shortcut=float(row.get("mean_final_direct_mix_shortcut_score", 0.0) or 0.0),
                active=float(row.get("mean_child_volume_active_count", 0.0) or 0.0),
                frags=float(row.get("mean_assay_child_fragment_count", 0.0) or 0.0),
                grand=float(row.get("mean_assay_grandchild_count", 0.0) or 0.0),
                binding=float(row.get("mean_child_predictive_ontology_binding_mass", 0.0) or 0.0),
            )
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            summary.get("interpretation", ""),
            "",
        ]
    )
    out_path.write_text("\n".join(lines), encoding="utf-8")


def _interpret(decisions: dict[str, Any]) -> str:
    positive = decisions.get("positive_control", {})
    winners = [
        name
        for name, decision in decisions.items()
        if name != "positive_control" and bool(decision.get("supports_information_density_hypothesis", False))
    ]
    predictive_winners = [
        name
        for name, decision in decisions.items()
        if name != "positive_control" and bool(decision.get("supports_predictive_binding_hypothesis", False))
    ]
    volume_no_conversion = [
        name for name, decision in decisions.items() if decision.get("read") == "volume_without_conversion"
    ]
    predictive_no_conversion = [
        name for name, decision in decisions.items() if decision.get("read") == "predictive_binding_without_conversion"
    ]
    if predictive_winners:
        return (
            "A predictive-binding variant improved parent-mode conversion without a world-jump. "
            "This supports the missing-ontology hypothesis: childworlds need to become accepted predictive commitments."
        )
    if winners:
        return (
            "At least one narrower/recursive variant improved parent-mode conversion without a world-jump. "
            "That supports the information-density hypothesis and should become the next training target."
        )
    if predictive_no_conversion:
        return (
            "Predictive binding found child residual-reduction signal and wrote some binding mass, but it still did not "
            "convert into mode-replace sibling ontology. The missing layer is probably not just acceptance; it needs a "
            "stronger parent assimilation state or a learned binding objective."
        )
    if volume_no_conversion:
        return (
            "The sweep increased child count or added fragments/grandchildren, but conversion did not improve. "
            "That argues volume alone is not sufficient; the missing piece is still a conversion/writeback law that lets "
            "narrow child regimes become mode-state rather than readout decoration."
        )
    if positive:
        return (
            "No volume or recursion variant beat the positive control. Treat this as evidence for a law bottleneck, "
            "not evidence against childworlds in general."
        )
    return "No positive-control row was available, so the sweep is structurally incomplete."


def main() -> None:
    ap = argparse.ArgumentParser(description="Run childworld volume/fragmentation/grandchild ontology sweeps.")
    ap.add_argument("--config", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--depth", type=int, default=3)
    ap.add_argument("--fork-depth", type=int, default=1)
    ap.add_argument("--fork-selector", default="max_live_child")
    ap.add_argument("--live-child-threshold", type=float, default=0.03)
    ap.add_argument("--mode", default="native_multimode_childworld")
    ap.add_argument("--case-json", default=None)
    ap.add_argument("--only", default=None, help="Comma-separated variant names to run.")
    ap.add_argument("--global-assay-killswitch-json", default=None, help="Optional JSON object/path merged over every variant assay killswitch.")
    ap.add_argument("--export-projector-dataset", action="store_true", help="Export parent phase projector NPZ rows for each predictive branch.")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    base_payload = _load_payload(Path(args.config))
    cases = _load_cases(args.case_json)
    global_assay = _load_json_object(args.global_assay_killswitch_json)
    wanted = {item.strip() for item in str(args.only or "").split(",") if item.strip()}

    variants = _variant_specs()
    if wanted:
        variants = [variant for variant in variants if str(variant["name"]) in wanted]
    if not variants:
        raise ValueError("--only did not match any volume/recursion variants")

    variant_rows: dict[str, Any] = {}
    for variant in variants:
        name = str(variant["name"])
        variant_dir = out_dir / name
        config_path = _write_variant_config(
            base_payload,
            variant_dir / "circleworld_config.json",
            dict(variant.get("config_updates", {})),
        )
        assay_killswitch = dict(variant.get("assay_killswitch", {}))
        assay_killswitch.update(global_assay)
        if bool(args.export_projector_dataset):
            assay_killswitch.update(
                {
                    "child_predictive_export_dataset": True,
                    "child_predictive_dataset_dir": str(variant_dir / "parent_phase_projector_dataset"),
                }
            )
        report = evaluate_nested_commitment_report(
            config_path=config_path,
            out_dir=variant_dir,
            device_name=args.device,
            depth=int(args.depth),
            fork_depth=int(args.fork_depth),
            fork_selector=str(args.fork_selector),
            live_child_threshold=float(args.live_child_threshold),
            mode=str(args.mode),
            cases=cases,
            assay_killswitch=assay_killswitch,
        )
        variant_rows[name] = {
            "description": variant.get("description", ""),
            "config_updates": variant.get("config_updates", {}),
            "assay_killswitch": assay_killswitch,
            "report_path": str(variant_dir / "nested_commitment_report.json"),
            "metrics": _metric_row(report),
        }

    positive = dict(variant_rows.get("positive_control", {}).get("metrics", {}))
    if not positive and variant_rows:
        positive = dict(next(iter(variant_rows.values())).get("metrics", {}))
    decisions = {
        name: _decision_for_variant(name, dict(row.get("metrics", {})), positive)
        for name, row in variant_rows.items()
    }
    summary = {
        "config": str(Path(args.config)),
        "out_dir": str(out_dir),
        "device": str(args.device),
        "depth": int(args.depth),
        "fork_depth": int(args.fork_depth),
        "fork_selector": str(args.fork_selector),
        "live_child_threshold": float(args.live_child_threshold),
        "mode": str(args.mode),
        "case_count": int(len(cases)),
        "global_assay_killswitch": global_assay,
        "export_projector_dataset": bool(args.export_projector_dataset),
        "variants": variant_rows,
        "decisions": decisions,
        "interpretation": _interpret(decisions),
    }
    summary_path = out_dir / "childworld_volume_recursion_sweep_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    _write_markdown(summary, out_dir / "CHILDWORLD_VOLUME_RECURSION_SWEEP.md")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
