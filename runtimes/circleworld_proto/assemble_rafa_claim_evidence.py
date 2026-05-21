from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


TOKENBURST_ROOT = Path(r"C:\Users\Andrew\AppData\Local\Temp\rafa_tokenburst")

DEFAULT_ARTIFACTS = {
    "child_writeback": TOKENBURST_ROOT
    / "child_writeback_ablation_v4_seed_fallback_cuda_2026_05_06"
    / "child_writeback_ablation_summary.json",
    "nested": TOKENBURST_ROOT
    / "nested_child_writeback_floor_high_2026_05_06"
    / "nested_commitment_report.json",
    "phase_gauge": TOKENBURST_ROOT
    / "phase_gauge_floor_high_cuda_2026_05_06"
    / "phase_gauge_invariance_report.json",
    "unit_phasor_contract": TOKENBURST_ROOT
    / "unit_phasor_contract_all_surfaces_cuda_2026_05_06"
    / "unit_phasor_contract_report.json",
    "unit_phasor_broad_direct_export": TOKENBURST_ROOT
    / "unit_phasor_broad_direct_export_cuda_2026_05_06"
    / "unit_phasor_contract_report.json",
    "unit_phasor_broad_audio_continuation": TOKENBURST_ROOT
    / "unit_phasor_broad_audio_continuation_cuda_2026_05_06"
    / "unit_phasor_contract_report.json",
    "unit_phasor_prefix_fail": TOKENBURST_ROOT
    / "unit_phasor_contract_prefix_fail_cuda_2026_05_06"
    / "unit_phasor_contract_report.json",
    "packet_alias_contract": TOKENBURST_ROOT
    / "packet_alias_contract_v2_cuda_2026_05_06"
    / "packet_alias_contract_report.json",
    "q_basis": TOKENBURST_ROOT
    / "q_basis_ablation_v3_seed_fallback_cuda_2026_05_06"
    / "q_basis_ablation_summary.json",
    "audio_prefix_hold": TOKENBURST_ROOT
    / "audio_continuation_broad18_prefix_hold_cuda_2026_05_06"
    / "audio_continuation_summary.json",
    "audio_flat": TOKENBURST_ROOT
    / "audio_continuation_broad18_flat_cuda_2026_05_06"
    / "audio_continuation_summary.json",
    "broad18_benchmark": TOKENBURST_ROOT
    / "benchmark_broad18_circleworld_4s_cuda_2026_05_06"
    / "benchmark_summary.json",
    "audio_method_compare": TOKENBURST_ROOT
    / "audio_continuation_method_compare_broad18_cuda_2026_05_06"
    / "audio_continuation_method_compare.json",
    "audio_copyphase_prefix_hold": TOKENBURST_ROOT
    / "audio_continuation_broad18_copyphase_prefix_hold_cuda_2026_05_06"
    / "audio_continuation_summary.json",
    "audio_copyphase_flat": TOKENBURST_ROOT
    / "audio_continuation_broad18_copyphase_flat_cuda_2026_05_06"
    / "audio_continuation_summary.json",
    "audio_method_compare_copyphase": TOKENBURST_ROOT
    / "audio_continuation_method_compare_copyphase_broad18_cuda_2026_05_06"
    / "audio_continuation_method_compare.json",
    "audio_phase_influence": TOKENBURST_ROOT
    / "audio_phase_influence_broad18_cuda_2026_05_06"
    / "audio_phase_influence_ablation.json",
    "audio_phase_seed": TOKENBURST_ROOT
    / "audio_phase_seed_broad18_cuda_2026_05_06"
    / "audio_phase_seed_ablation.json",
    "audio_phase_seed_weighted_fit": TOKENBURST_ROOT
    / "audio_phase_seed_weighted_fit_broad18_cuda_2026_05_06"
    / "audio_phase_seed_ablation.json",
    "audio_circle_delta_probe": TOKENBURST_ROOT
    / "audio_circle_delta_probe_broad18_cuda_2026_05_06"
    / "audio_circle_delta_probe.json",
    "audio_delta_objective_scout": TOKENBURST_ROOT
    / "audio_delta_objective_scout_broad18_cuda_2026_05_06"
    / "audio_delta_objective_scout.json",
    "audio_delta_objective_scout_relsig": TOKENBURST_ROOT
    / "audio_delta_objective_scout_relsig_broad18_cuda_2026_05_06"
    / "audio_delta_objective_scout.json",
    "audio_delta_mechanism_probe": TOKENBURST_ROOT
    / "audio_delta_mechanism_probe_broad18_cuda_2026_05_06"
    / "audio_delta_mechanism_probe.json",
    "audio_delta_mechanism_lockbox": TOKENBURST_ROOT
    / "audio_delta_mechanism_probe_lockbox18_cuda_2026_05_06"
    / "audio_delta_mechanism_probe.json",
    "audio_mechanism_family_sensitivity": TOKENBURST_ROOT
    / "audio_mechanism_family_sensitivity_broad_cuda_2026_05_06"
    / "audio_mechanism_family_sensitivity.json",
    "audio_electric_motor_holdout": TOKENBURST_ROOT
    / "audio_electric_motor_holdout_cuda_2026_05_06"
    / "audio_electric_motor_holdout.json",
    "audio_steady_phenotype_manifest": TOKENBURST_ROOT
    / "audio_steady_phenotype_manifest_cuda_2026_05_06"
    / "audio_steady_phenotype_manifest.json",
    "audio_phase_phenotype_diagnostics": TOKENBURST_ROOT
    / "audio_phase_phenotype_diagnostics_cuda_2026_05_06"
    / "audio_phase_phenotype_diagnostics.json",
    "audio_baseline_stratified_diagnostics": TOKENBURST_ROOT
    / "audio_baseline_stratified_diagnostics_cuda_2026_05_06"
    / "audio_baseline_stratified_diagnostics.json",
    "audio_predeclared_lockbox_manifest": TOKENBURST_ROOT
    / "audio_predeclared_lockbox_v1_2026_05_06"
    / "audio_predeclared_lockbox_manifest.json",
    "audio_predeclared_lockbox_probe": TOKENBURST_ROOT
    / "audio_predeclared_lockbox_probe_v1_cuda_2026_05_06"
    / "audio_delta_mechanism_probe.json",
    "audio_predeclared_lockbox_compare": TOKENBURST_ROOT
    / "audio_predeclared_lockbox_compare_v1_2026_05_06"
    / "audio_lockbox_result_compare.json",
    "audio_predeclared_lockbox_compare_no_single_source": TOKENBURST_ROOT
    / "audio_predeclared_lockbox_compare_v1_no_single_source_2026_05_06"
    / "audio_lockbox_result_compare.json",
    "audio_phase_law_scout": TOKENBURST_ROOT
    / "audio_phase_law_scout_v1_cuda_2026_05_07"
    / "audio_delta_mechanism_probe.json",
    "audio_phase_law_scout_compare_no_single_source": TOKENBURST_ROOT
    / "audio_phase_law_scout_compare_v1_no_single_source_2026_05_07"
    / "audio_lockbox_result_compare.json",
    "audio_internal_phase_law_raw_compare_no_single_source": TOKENBURST_ROOT
    / "internal_phase_law_objective_scout_auto_full_2026_05_07"
    / "absolute_compare"
    / "audio_lockbox_result_compare.json",
    "audio_internal_phase_law_variant_compare_no_single_source": TOKENBURST_ROOT
    / "internal_phase_law_objective_scout_auto_full_2026_05_07"
    / "variant_compare"
    / "internal_phase_law_variant_compare.json",
    "audio_internal_phase_law_objective_score": TOKENBURST_ROOT
    / "internal_phase_law_objective_scout_auto_full_2026_05_07"
    / "objective_score"
    / "internal_phase_law_objective_score.json",
    "audio_internal_phase_law_focused_raw_compare_no_single_source": TOKENBURST_ROOT
    / "internal_phase_law_objective_scout_focused_v1_2026_05_07"
    / "absolute_compare"
    / "audio_lockbox_result_compare.json",
    "audio_internal_phase_law_focused_variant_compare_no_single_source": TOKENBURST_ROOT
    / "internal_phase_law_objective_scout_focused_v1_2026_05_07"
    / "variant_compare"
    / "internal_phase_law_variant_compare.json",
    "audio_internal_phase_law_focused_objective_score": TOKENBURST_ROOT
    / "internal_phase_law_objective_scout_focused_v1_2026_05_07"
    / "objective_score"
    / "internal_phase_law_objective_score.json",
    "audio_internal_phase_law_focused_joint_rows": TOKENBURST_ROOT
    / "internal_phase_law_joint_rows_focused_v1_2026_05_07"
    / "internal_phase_law_joint_row_diagnostics.json",
    "audio_internal_phase_law_wide_raw_compare_no_single_source": TOKENBURST_ROOT
    / "internal_phase_law_objective_scout_wide_v1_2026_05_07"
    / "absolute_compare"
    / "audio_lockbox_result_compare.json",
    "audio_internal_phase_law_wide_variant_compare_no_single_source": TOKENBURST_ROOT
    / "internal_phase_law_objective_scout_wide_v1_2026_05_07"
    / "variant_compare"
    / "internal_phase_law_variant_compare.json",
    "audio_internal_phase_law_wide_objective_score": TOKENBURST_ROOT
    / "internal_phase_law_objective_scout_wide_v1_2026_05_07"
    / "objective_score"
    / "internal_phase_law_objective_score.json",
    "audio_internal_phase_law_wide_joint_rows": TOKENBURST_ROOT
    / "internal_phase_law_joint_rows_wide_v1_2026_05_07"
    / "internal_phase_law_joint_row_diagnostics.json",
    "audio_internal_phase_law_v2_guard_raw_compare_no_single_source": TOKENBURST_ROOT
    / "internal_phase_law_objective_scout_v2_guard_v1_2026_05_07"
    / "absolute_compare"
    / "audio_lockbox_result_compare.json",
    "audio_internal_phase_law_v2_guard_variant_compare_no_single_source": TOKENBURST_ROOT
    / "internal_phase_law_objective_scout_v2_guard_v1_2026_05_07"
    / "variant_compare"
    / "internal_phase_law_variant_compare.json",
    "audio_internal_phase_law_v2_guard_objective_score": TOKENBURST_ROOT
    / "internal_phase_law_objective_scout_v2_guard_v1_2026_05_07"
    / "objective_score"
    / "internal_phase_law_objective_score.json",
    "audio_internal_phase_law_v2_guard_joint_rows": TOKENBURST_ROOT
    / "internal_phase_law_joint_rows_v2_guard_v1_2026_05_07"
    / "internal_phase_law_joint_row_diagnostics.json",
    "audio_internal_phase_law_v2_guard_failure_atlas": TOKENBURST_ROOT
    / "internal_phase_law_failure_atlas_v2_guard_v1_2026_05_07"
    / "internal_phase_law_failure_atlas.json",
    "audio_internal_phase_law_v3_local_raw_compare_no_single_source": TOKENBURST_ROOT
    / "internal_phase_law_objective_scout_v3_local_v1_2026_05_07"
    / "absolute_compare"
    / "audio_lockbox_result_compare.json",
    "audio_internal_phase_law_v3_local_variant_compare_no_single_source": TOKENBURST_ROOT
    / "internal_phase_law_objective_scout_v3_local_v1_2026_05_07"
    / "variant_compare"
    / "internal_phase_law_variant_compare.json",
    "audio_internal_phase_law_v3_local_objective_score": TOKENBURST_ROOT
    / "internal_phase_law_objective_scout_v3_local_v1_2026_05_07"
    / "objective_score"
    / "internal_phase_law_objective_score.json",
    "audio_internal_phase_law_v3_local_joint_rows": TOKENBURST_ROOT
    / "internal_phase_law_joint_rows_v3_local_v1_2026_05_07"
    / "internal_phase_law_joint_row_diagnostics.json",
    "audio_internal_phase_law_v3_local_failure_atlas": TOKENBURST_ROOT
    / "internal_phase_law_failure_atlas_v3_local_v1_2026_05_07"
    / "internal_phase_law_failure_atlas.json",
    "audio_internal_phase_law_v3_local_strict_objective_score": TOKENBURST_ROOT
    / "internal_phase_law_objective_score_joint_strict_v3_local_v1_2026_05_07"
    / "internal_phase_law_objective_score.json",
    "audio_internal_phase_law_v3_local_strict_joint_rows": TOKENBURST_ROOT
    / "internal_phase_law_joint_rows_strict_v3_local_v1_2026_05_07"
    / "internal_phase_law_joint_row_diagnostics.json",
    "audio_internal_phase_law_v4_joint_nonbad_raw_compare_no_single_source": TOKENBURST_ROOT
    / "internal_phase_law_objective_scout_v4_joint_nonbad_v1_2026_05_07"
    / "absolute_compare"
    / "audio_lockbox_result_compare.json",
    "audio_internal_phase_law_v4_joint_nonbad_variant_compare_no_single_source": TOKENBURST_ROOT
    / "internal_phase_law_objective_scout_v4_joint_nonbad_v1_2026_05_07"
    / "variant_compare"
    / "internal_phase_law_variant_compare.json",
    "audio_internal_phase_law_v4_joint_nonbad_objective_score": TOKENBURST_ROOT
    / "internal_phase_law_objective_scout_v4_joint_nonbad_v1_2026_05_07"
    / "objective_score"
    / "internal_phase_law_objective_score.json",
    "audio_internal_phase_law_v4_joint_nonbad_joint_rows": TOKENBURST_ROOT
    / "internal_phase_law_objective_scout_v4_joint_nonbad_v1_2026_05_07"
    / "joint_diagnostics"
    / "internal_phase_law_joint_row_diagnostics.json",
    "audio_internal_phase_law_v4_joint_nonbad_failure_atlas": TOKENBURST_ROOT
    / "internal_phase_law_failure_atlas_v4_joint_nonbad_v1_2026_05_07"
    / "internal_phase_law_failure_atlas.json",
    "audio_internal_phase_law_v5_low_energy_row_raw_compare_no_single_source": TOKENBURST_ROOT
    / "internal_phase_law_objective_scout_v5_low_energy_row_v1_2026_05_07"
    / "absolute_compare"
    / "audio_lockbox_result_compare.json",
    "audio_internal_phase_law_v5_low_energy_row_variant_compare_no_single_source": TOKENBURST_ROOT
    / "internal_phase_law_objective_scout_v5_low_energy_row_v1_2026_05_07"
    / "variant_compare"
    / "internal_phase_law_variant_compare.json",
    "audio_internal_phase_law_v5_low_energy_row_objective_score": TOKENBURST_ROOT
    / "internal_phase_law_objective_scout_v5_low_energy_row_v1_2026_05_07"
    / "objective_score"
    / "internal_phase_law_objective_score.json",
    "audio_internal_phase_law_v5_low_energy_row_joint_rows": TOKENBURST_ROOT
    / "internal_phase_law_objective_scout_v5_low_energy_row_v1_2026_05_07"
    / "joint_diagnostics"
    / "internal_phase_law_joint_row_diagnostics.json",
    "audio_internal_phase_law_v5_low_energy_row_failure_atlas": TOKENBURST_ROOT
    / "internal_phase_law_failure_atlas_v5_low_energy_row_v1_2026_05_07"
    / "internal_phase_law_failure_atlas.json",
    "audio_internal_phase_law_v6_low_energy_win_raw_compare_no_single_source": TOKENBURST_ROOT
    / "internal_phase_law_objective_scout_v6_low_energy_win_v1_2026_05_07"
    / "absolute_compare"
    / "audio_lockbox_result_compare.json",
    "audio_internal_phase_law_v6_low_energy_win_variant_compare_no_single_source": TOKENBURST_ROOT
    / "internal_phase_law_objective_scout_v6_low_energy_win_v1_2026_05_07"
    / "variant_compare"
    / "internal_phase_law_variant_compare.json",
    "audio_internal_phase_law_v6_low_energy_win_objective_score": TOKENBURST_ROOT
    / "internal_phase_law_objective_scout_v6_low_energy_win_v1_2026_05_07"
    / "objective_score"
    / "internal_phase_law_objective_score.json",
    "audio_internal_phase_law_v6_low_energy_win_joint_rows": TOKENBURST_ROOT
    / "internal_phase_law_objective_scout_v6_low_energy_win_v1_2026_05_07"
    / "joint_diagnostics"
    / "internal_phase_law_joint_row_diagnostics.json",
    "audio_internal_phase_law_v6_low_energy_win_failure_atlas": TOKENBURST_ROOT
    / "internal_phase_law_failure_atlas_v6_low_energy_win_v1_2026_05_07"
    / "internal_phase_law_failure_atlas.json",
    "audio_internal_phase_law_v7_low_energy_gain_ladder_raw_compare_no_single_source": TOKENBURST_ROOT
    / "internal_phase_law_objective_scout_v7_low_energy_gain_ladder_v1_2026_05_07"
    / "absolute_compare"
    / "audio_lockbox_result_compare.json",
    "audio_internal_phase_law_v7_low_energy_gain_ladder_variant_compare_no_single_source": TOKENBURST_ROOT
    / "internal_phase_law_objective_scout_v7_low_energy_gain_ladder_v1_2026_05_07"
    / "variant_compare"
    / "internal_phase_law_variant_compare.json",
    "audio_internal_phase_law_v7_low_energy_gain_ladder_objective_score": TOKENBURST_ROOT
    / "internal_phase_law_objective_scout_v7_low_energy_gain_ladder_v1_2026_05_07"
    / "objective_score"
    / "internal_phase_law_objective_score.json",
    "audio_internal_phase_law_v7_low_energy_gain_ladder_joint_rows": TOKENBURST_ROOT
    / "internal_phase_law_objective_scout_v7_low_energy_gain_ladder_v1_2026_05_07"
    / "joint_diagnostics"
    / "internal_phase_law_joint_row_diagnostics.json",
    "audio_internal_phase_law_v7_low_energy_gain_ladder_failure_atlas": TOKENBURST_ROOT
    / "internal_phase_law_failure_atlas_v7_low_energy_gain_ladder_v1_2026_05_07"
    / "internal_phase_law_failure_atlas.json",
    "audio_internal_phase_law_v7_low_energy_gain_ladder_target_flips": TOKENBURST_ROOT
    / "internal_phase_law_target_flip_diagnostics_v7_low_energy_gain_ladder_v1_2026_05_07"
    / "internal_phase_law_target_flip_diagnostics.json",
    "audio_internal_phase_law_v8_family_support_mask_raw_compare_no_single_source": TOKENBURST_ROOT
    / "internal_phase_law_objective_scout_v8_family_support_mask_v1_2026_05_07"
    / "absolute_compare"
    / "audio_lockbox_result_compare.json",
    "audio_internal_phase_law_v8_family_support_mask_variant_compare_no_single_source": TOKENBURST_ROOT
    / "internal_phase_law_objective_scout_v8_family_support_mask_v1_2026_05_07"
    / "variant_compare"
    / "internal_phase_law_variant_compare.json",
    "audio_internal_phase_law_v8_family_support_mask_objective_score": TOKENBURST_ROOT
    / "internal_phase_law_objective_scout_v8_family_support_mask_v1_2026_05_07"
    / "objective_score"
    / "internal_phase_law_objective_score.json",
    "audio_internal_phase_law_v8_family_support_mask_joint_rows": TOKENBURST_ROOT
    / "internal_phase_law_objective_scout_v8_family_support_mask_v1_2026_05_07"
    / "joint_diagnostics"
    / "internal_phase_law_joint_row_diagnostics.json",
    "audio_internal_phase_law_v8_family_support_mask_failure_atlas": TOKENBURST_ROOT
    / "internal_phase_law_failure_atlas_v8_family_support_mask_v1_2026_05_07"
    / "internal_phase_law_failure_atlas.json",
    "audio_internal_phase_law_v8_family_support_mask_diagnostics": TOKENBURST_ROOT
    / "internal_phase_law_family_mask_diagnostics_v8_family_support_mask_v1_2026_05_07"
    / "internal_phase_law_family_mask_diagnostics.json",
    "audio_internal_phase_law_v8_support_router_oracle": TOKENBURST_ROOT
    / "internal_phase_law_support_router_oracle_v8_family_support_mask_v1_2026_05_07"
    / "internal_phase_law_support_router_oracle.json",
    "audio_internal_phase_law_v9_phase_masks_raw_compare_no_single_source": TOKENBURST_ROOT
    / "internal_phase_law_objective_scout_v9_phase_masks_v1_2026_05_07"
    / "absolute_compare"
    / "audio_lockbox_result_compare.json",
    "audio_internal_phase_law_v9_phase_masks_variant_compare_no_single_source": TOKENBURST_ROOT
    / "internal_phase_law_objective_scout_v9_phase_masks_v1_2026_05_07"
    / "variant_compare"
    / "internal_phase_law_variant_compare.json",
    "audio_internal_phase_law_v9_phase_masks_objective_score": TOKENBURST_ROOT
    / "internal_phase_law_objective_scout_v9_phase_masks_v1_2026_05_07"
    / "objective_score"
    / "internal_phase_law_objective_score.json",
    "audio_internal_phase_law_v9_phase_masks_joint_rows": TOKENBURST_ROOT
    / "internal_phase_law_objective_scout_v9_phase_masks_v1_2026_05_07"
    / "joint_diagnostics"
    / "internal_phase_law_joint_row_diagnostics.json",
    "audio_internal_phase_law_v9_phase_masks_phase_stable_target_score": TOKENBURST_ROOT
    / "internal_phase_law_objective_score_v9_phase_stable_target_v1_2026_05_07"
    / "internal_phase_law_objective_score.json",
    "audio_internal_phase_law_v9_phase_masks_phase_stable_target_joint_rows": TOKENBURST_ROOT
    / "internal_phase_law_joint_rows_v9_phase_stable_target_v1_2026_05_07"
    / "internal_phase_law_joint_row_diagnostics.json",
    "audio_internal_phase_law_v9_phase_masks_failure_atlas": TOKENBURST_ROOT
    / "internal_phase_law_failure_atlas_v9_phase_masks_v1_2026_05_07"
    / "internal_phase_law_failure_atlas.json",
    "audio_internal_phase_law_v10_phase_router_masks_raw_compare_no_single_source": TOKENBURST_ROOT
    / "internal_phase_law_objective_scout_v10_phase_router_masks_v1_2026_05_07"
    / "absolute_compare"
    / "audio_lockbox_result_compare.json",
    "audio_internal_phase_law_v10_phase_router_masks_variant_compare_no_single_source": TOKENBURST_ROOT
    / "internal_phase_law_objective_scout_v10_phase_router_masks_v1_2026_05_07"
    / "variant_compare"
    / "internal_phase_law_variant_compare.json",
    "audio_internal_phase_law_v10_phase_router_masks_objective_score": TOKENBURST_ROOT
    / "internal_phase_law_objective_scout_v10_phase_router_masks_v1_2026_05_07"
    / "objective_score"
    / "internal_phase_law_objective_score.json",
    "audio_internal_phase_law_v10_phase_router_masks_joint_rows": TOKENBURST_ROOT
    / "internal_phase_law_objective_scout_v10_phase_router_masks_v1_2026_05_07"
    / "joint_diagnostics"
    / "internal_phase_law_joint_row_diagnostics.json",
    "audio_internal_phase_law_v10_phase_stable_target_score": TOKENBURST_ROOT
    / "internal_phase_law_objective_score_v10_phase_stable_target_v1_2026_05_07"
    / "internal_phase_law_objective_score.json",
    "audio_internal_phase_law_v10_phase_stable_target_joint_rows": TOKENBURST_ROOT
    / "internal_phase_law_joint_rows_v10_phase_stable_target_v1_2026_05_07"
    / "internal_phase_law_joint_row_diagnostics.json",
    "audio_internal_phase_law_v10_phase_coherent_target_score": TOKENBURST_ROOT
    / "internal_phase_law_objective_score_v10_phase_coherent_target_v1_2026_05_07"
    / "internal_phase_law_objective_score.json",
    "audio_internal_phase_law_v10_phase_coherent_target_joint_rows": TOKENBURST_ROOT
    / "internal_phase_law_joint_rows_v10_phase_coherent_target_v1_2026_05_07"
    / "internal_phase_law_joint_row_diagnostics.json",
    "audio_internal_phase_law_v9_candidate_classes": TOKENBURST_ROOT
    / "internal_phase_law_candidate_classes_v9_phase_masks_v1_2026_05_07"
    / "internal_phase_law_candidate_classes.json",
    "audio_internal_phase_law_v10_candidate_classes": TOKENBURST_ROOT
    / "internal_phase_law_candidate_classes_v10_phase_router_v1_2026_05_07"
    / "internal_phase_law_candidate_classes.json",
    "audio_internal_phase_law_v11_phase_router_direct_raw_compare_no_single_source": TOKENBURST_ROOT
    / "internal_phase_law_objective_scout_v11_phase_router_direct_v1_2026_05_07"
    / "absolute_compare"
    / "audio_lockbox_result_compare.json",
    "audio_internal_phase_law_v11_phase_router_direct_variant_compare_no_single_source": TOKENBURST_ROOT
    / "internal_phase_law_objective_scout_v11_phase_router_direct_v1_2026_05_07"
    / "variant_compare"
    / "internal_phase_law_variant_compare.json",
    "audio_internal_phase_law_v11_phase_router_direct_objective_score": TOKENBURST_ROOT
    / "internal_phase_law_objective_scout_v11_phase_router_direct_v1_2026_05_07"
    / "objective_score"
    / "internal_phase_law_objective_score.json",
    "audio_internal_phase_law_v11_phase_router_direct_joint_rows": TOKENBURST_ROOT
    / "internal_phase_law_objective_scout_v11_phase_router_direct_v1_2026_05_07"
    / "joint_diagnostics"
    / "internal_phase_law_joint_row_diagnostics.json",
    "audio_internal_phase_law_v11_candidate_classes": TOKENBURST_ROOT
    / "internal_phase_law_candidate_classes_v11_phase_router_direct_v1_2026_05_07"
    / "internal_phase_law_candidate_classes.json",
    "audio_internal_phase_law_v12_phase_reentry_direct_raw_compare_no_single_source": TOKENBURST_ROOT
    / "internal_phase_law_objective_scout_v12_phase_reentry_direct_v1_2026_05_07"
    / "absolute_compare"
    / "audio_lockbox_result_compare.json",
    "audio_internal_phase_law_v12_phase_reentry_direct_variant_compare_no_single_source": TOKENBURST_ROOT
    / "internal_phase_law_objective_scout_v12_phase_reentry_direct_v1_2026_05_07"
    / "variant_compare"
    / "internal_phase_law_variant_compare.json",
    "audio_internal_phase_law_v12_phase_reentry_direct_objective_score": TOKENBURST_ROOT
    / "internal_phase_law_objective_scout_v12_phase_reentry_direct_v1_2026_05_07"
    / "objective_score"
    / "internal_phase_law_objective_score.json",
    "audio_internal_phase_law_v12_phase_reentry_direct_joint_rows": TOKENBURST_ROOT
    / "internal_phase_law_objective_scout_v12_phase_reentry_direct_v1_2026_05_07"
    / "joint_diagnostics"
    / "internal_phase_law_joint_row_diagnostics.json",
    "audio_internal_phase_law_v12_candidate_classes": TOKENBURST_ROOT
    / "internal_phase_law_candidate_classes_v12_phase_reentry_direct_v1_2026_05_07"
    / "internal_phase_law_candidate_classes.json",
    "audio_internal_phase_law_v13_causal_reentry_direct_raw_compare_no_single_source": TOKENBURST_ROOT
    / "internal_phase_law_objective_scout_v13_causal_reentry_direct_v1_2026_05_07"
    / "absolute_compare"
    / "audio_lockbox_result_compare.json",
    "audio_internal_phase_law_v13_causal_reentry_direct_variant_compare_no_single_source": TOKENBURST_ROOT
    / "internal_phase_law_objective_scout_v13_causal_reentry_direct_v1_2026_05_07"
    / "variant_compare"
    / "internal_phase_law_variant_compare.json",
    "audio_internal_phase_law_v13_causal_reentry_direct_objective_score": TOKENBURST_ROOT
    / "internal_phase_law_objective_scout_v13_causal_reentry_direct_v1_2026_05_07"
    / "objective_score"
    / "internal_phase_law_objective_score.json",
    "audio_internal_phase_law_v13_causal_reentry_direct_joint_rows": TOKENBURST_ROOT
    / "internal_phase_law_objective_scout_v13_causal_reentry_direct_v1_2026_05_07"
    / "joint_diagnostics"
    / "internal_phase_law_joint_row_diagnostics.json",
    "audio_internal_phase_law_v13_candidate_classes": TOKENBURST_ROOT
    / "internal_phase_law_candidate_classes_v13_causal_reentry_direct_v1_2026_05_07"
    / "internal_phase_law_candidate_classes.json",
    "audio_internal_phase_law_predeclared_diagnostic_class": TOKENBURST_ROOT
    / "internal_phase_law_predeclared_diagnostic_class_v10_v13_2026_05_07"
    / "internal_phase_law_predeclared_diagnostic_class.json",
    "dense_signature_claim": TOKENBURST_ROOT
    / "dense_signature_claim_broad18_cuda_2026_05_06"
    / "dense_signature_claim.json",
    "dense_signature_failure_modes": TOKENBURST_ROOT
    / "dense_signature_failure_modes_2026_05_07"
    / "dense_signature_failure_modes.json",
    "relational_signature_contract_audit": TOKENBURST_ROOT
    / "relational_signature_contract_audit_2026_05_07"
    / "relational_signature_contract_audit.json",
    "relational_signature_learning_contract": TOKENBURST_ROOT
    / "relational_signature_learning_contract_2026_05_07"
    / "relational_signature_learning_contract.json",
    "learned_signature_scout": TOKENBURST_ROOT
    / "learned_signature_scout_cuda_2026_05_07"
    / "learned_signature_scout.json",
    "learned_signature_scout_compare": TOKENBURST_ROOT
    / "learned_signature_scout_compare_2026_05_07"
    / "learned_signature_scout_compare.json",
    "learned_signature_split_suite": TOKENBURST_ROOT
    / "learned_signature_split_suite_v24_tf6_core_2026_05_07"
    / "learned_signature_split_suite.json",
    "learned_signature_split_suite_compare": TOKENBURST_ROOT
    / "learned_signature_split_suite_compare_2026_05_07"
    / "learned_signature_split_suite_compare.json",
    "learned_signature_case_failures": TOKENBURST_ROOT
    / "learned_signature_case_failures_2026_05_07"
    / "learned_signature_case_failures.json",
    "signature_operator_tail_probe_compare": TOKENBURST_ROOT
    / "signature_operator_tail_probe_compare_2026_05_07"
    / "signature_operator_tail_probe_compare.json",
    "signature_future_law_probe": TOKENBURST_ROOT / "signature_future_law_probe.json",
    "signature_tail_incremental_usefulness": TOKENBURST_ROOT
    / "signature_tail_incremental_usefulness.json",
    "semantic_projector_contract": TOKENBURST_ROOT
    / "semantic_projector_contract_2026_05_06"
    / "semantic_projector_contract.json",
    "substrate": TOKENBURST_ROOT
    / "substrate_ablation_cuda_2026_05_06"
    / "substrate_ablation_summary.json",
    "claim_isolation": TOKENBURST_ROOT
    / "claim_isolation_cuda_mixed_2026_05_06"
    / "claim_isolation_suite_summary.json",
    "qtrace_inheritance": TOKENBURST_ROOT
    / "claim_isolation_qtrace_cuda_mixed_2026_05_06"
    / "claim_isolation_suite_summary.json",
    "arc_lane": TOKENBURST_ROOT
    / "claim_isolation_arc_cuda_mixed_2026_05_06"
    / "claim_isolation_suite_summary.json",
    "arc_q_cross": TOKENBURST_ROOT
    / "claim_isolation_arc_q_cross_cuda_mixed_2026_05_06"
    / "claim_isolation_suite_summary.json",
    "arc_q_control_claim": TOKENBURST_ROOT
    / "arc_q_control_claim_2026_05_07"
    / "arc_q_control_claim.json",
}

OPTIONAL_TOKENBURST_ARTIFACT_FILENAMES = {
    "signature_future_law_probe": "signature_future_law_probe.json",
    "signature_tail_incremental_usefulness": "signature_tail_incremental_usefulness.json",
}


def _load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _resolve_optional_tokenburst_artifact(path: Path, filename: str) -> Path:
    if path.exists() or path.name != filename or path.parent != TOKENBURST_ROOT or not TOKENBURST_ROOT.exists():
        return path
    candidates = [candidate for candidate in TOKENBURST_ROOT.rglob(filename) if candidate.is_file()]
    if not candidates:
        return path
    return max(candidates, key=lambda candidate: (candidate.stat().st_mtime, str(candidate)))


def _resolve_artifacts(artifacts: dict[str, Path]) -> dict[str, Path]:
    resolved = dict(artifacts)
    for key, filename in OPTIONAL_TOKENBURST_ARTIFACT_FILENAMES.items():
        if key in resolved:
            resolved[key] = _resolve_optional_tokenburst_artifact(resolved[key], filename)
    return resolved


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _get(mapping: dict[str, Any] | None, *path: Any, default: Any = None) -> Any:
    cur: Any = mapping
    for key in path:
        if isinstance(cur, dict):
            cur = cur.get(key)
            continue
        if isinstance(cur, list) and isinstance(key, int) and 0 <= key < len(cur):
            cur = cur[key]
            continue
        else:
            return default
    return default if cur is None else cur


def _variant_map(payload: dict[str, Any] | None) -> dict[str, dict[str, Any]]:
    if not isinstance(payload, dict):
        return {}
    out: dict[str, dict[str, Any]] = {}
    for row in payload.get("variants", []) or []:
        if not isinstance(row, dict):
            continue
        name = row.get("variant") or row.get("name")
        if name:
            out[str(name)] = row
    return out


def _metrics(row: dict[str, Any] | None) -> dict[str, Any]:
    metrics = row.get("metrics") if isinstance(row, dict) else None
    return metrics if isinstance(metrics, dict) else {}


def _source(metrics: dict[str, Any], source: str) -> dict[str, Any]:
    by_source = metrics.get("by_source")
    if isinstance(by_source, dict) and isinstance(by_source.get(source), dict):
        return by_source[source]
    split = metrics.get("source_split")
    if isinstance(split, dict) and isinstance(split.get(source), dict):
        return split[source]
    return {}


def summarize_child_writeback(payload: dict[str, Any] | None) -> dict[str, Any]:
    variants = _variant_map(payload)
    rows: list[dict[str, Any]] = []
    for name, row in variants.items():
        m = _metrics(row)
        naked = _source(m, "naked_rafa")
        rows.append(
            {
                "variant": name,
                "phase_only_real_branch_fraction": m.get("phase_only_real_branch_fraction"),
                "mean_parent_real_branch_fraction": m.get("mean_parent_real_branch_fraction"),
                "mean_child_real_branch_fraction": m.get("mean_child_real_branch_fraction"),
                "mean_child_writeback_gate_mass": m.get("mean_child_writeback_gate_mass"),
                "mean_child_phase_writeback_delta_mass": m.get("mean_child_phase_writeback_delta_mass"),
                "mean_child_support_writeback_mass": m.get("mean_child_support_writeback_mass"),
                "mean_child_logit_writeback_mass": m.get("mean_child_logit_writeback_mass"),
                "mean_child_qtrace_writeback_mass": m.get("mean_child_qtrace_writeback_mass"),
                "mean_major_gain": m.get("mean_major_gain"),
                "naked_phase_only_real_branch_fraction": naked.get("phase_only_real_branch_fraction"),
                "naked_final_phase_only_branch_surface_peak": naked.get("final_phase_only_branch_surface_peak"),
                "naked_decorative_slot2_low_phase_fraction": naked.get("decorative_slot2_low_phase_fraction"),
            }
        )

    support_only = _metrics(variants.get("support_only_phase_operator_floor_off"))
    high = _metrics(variants.get("support_operator_floor_high"))
    support_phase = _as_float(support_only.get("phase_only_real_branch_fraction"))
    support_delta = _as_float(support_only.get("mean_child_phase_writeback_delta_mass"))
    high_naked = _as_float(_source(high, "naked_rafa").get("phase_only_real_branch_fraction"))
    high_decor = _as_float(_source(high, "naked_rafa").get("decorative_slot2_low_phase_fraction"))
    status = "missing"
    if rows:
        status = "partially_mapped"
        if support_phase == 0.0 and support_delta == 0.0 and high_naked > 0.0:
            status = "causal_path_isolated_but_decorative_risk"
    return {
        "status": status,
        "support_only_phase_delta_zero": support_delta == 0.0,
        "support_only_parent_visible_phase": support_phase,
        "high_floor_naked_phase": high_naked,
        "high_floor_naked_decorative": high_decor,
        "rows": rows,
    }


def summarize_nested(payload: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(payload, dict):
        return {"status": "missing"}
    sibling = _as_float(payload.get("mean_nested_sibling_fraction"))
    active = _as_float(payload.get("mean_child_active_fraction"))
    qualified = _as_float(payload.get("mean_branch_identity_qualified_carry"))
    readout = _as_float(payload.get("mean_readout_sibling_response"))
    raw_carry = _as_float(payload.get("mean_branch_identity_carry"))
    return {
        "status": "not_established" if sibling <= 0.0 else "established",
        "overall_read": payload.get("overall_read"),
        "verdict_counts": payload.get("verdict_counts"),
        "mean_nested_sibling_fraction": sibling,
        "mean_child_active_fraction": active,
        "mean_child_survival_signal": payload.get("mean_child_survival_signal"),
        "mean_readout_sibling_response": readout,
        "mean_branch_identity_carry": raw_carry,
        "mean_branch_identity_qualified_carry": qualified,
        "mean_world_jump_penalty": payload.get("mean_world_jump_penalty"),
    }


def summarize_phase_gauge(payload: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(payload, dict):
        return {"status": "missing"}
    max_abs = 0.0
    transform_rows: list[dict[str, Any]] = []
    for name, block in (payload.get("aggregate_differences") or {}).items():
        metrics = block.get("metrics") if isinstance(block, dict) else {}
        local_max = 0.0
        if isinstance(metrics, dict):
            for delta in metrics.values():
                if isinstance(delta, dict):
                    local_max = max(local_max, _as_float(delta.get("max_abs")))
        max_abs = max(max_abs, local_max)
        transform_rows.append({"transform": name, "max_abs_metric_delta": local_max})
    return {
        "status": "supported_for_metrics" if transform_rows and max_abs <= 1.0e-5 else "needs_review",
        "max_abs_metric_delta": max_abs,
        "num_transforms": len(transform_rows),
        "rows": transform_rows,
    }


def summarize_unit_phasor_contract(
    payload: dict[str, Any] | None,
    prefix_fail_payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if not isinstance(payload, dict):
        return {"status": "missing"}
    overall = payload.get("overall") if isinstance(payload.get("overall"), dict) else {}
    max_err = _as_float(overall.get("max_phasor_norm_error", overall.get("max_abs_norm_error")))
    violations = int(_as_float(overall.get("num_norm_violations")))
    min_finite = _as_float(overall.get("min_finite_fraction"))
    passed = bool(overall.get("phase_only_contract_passed", overall.get("passing", False)))
    hidden_mag = bool(overall.get("hidden_magnitude_channel_detected", violations > 0))
    prefix_overall = prefix_fail_payload.get("overall") if isinstance(prefix_fail_payload, dict) else {}
    prefix_max_err = _as_float(prefix_overall.get("max_phasor_norm_error", prefix_overall.get("max_abs_norm_error")))
    prefix_violations = int(_as_float(prefix_overall.get("num_norm_violations")))
    if isinstance(prefix_fail_payload, dict) and "num_norm_violations" not in prefix_overall:
        prefix_violations = 0
        for case in prefix_fail_payload.get("cases", []) or []:
            if not isinstance(case, dict):
                continue
            for row in case.get("rows", []) or []:
                if isinstance(row, dict) and _as_float(row.get("max_abs_norm_error")) > 1.0e-5:
                    prefix_violations += 1
    detected_prefix_failure = bool(prefix_max_err > max(1.0e-5, max_err) or prefix_violations > 0)
    negative = payload.get("negative_control") if isinstance(payload.get("negative_control"), dict) else {}
    negative_aggregate = negative.get("aggregate") if isinstance(negative.get("aggregate"), dict) else {}
    rows: list[dict[str, Any]] = []
    for name, block in (payload.get("group_summaries") or {}).items():
        if not isinstance(block, dict):
            continue
        rows.append(
            {
                "boundary": name,
                "checked_tensors": block.get("tensor_count"),
                "max_norm_error": block.get("max_abs_norm_error"),
                "mean_norm_error": block.get("mean_abs_norm_error"),
                "min_finite_fraction": block.get("min_finite_fraction"),
                "worst_path": block.get("worst_path"),
                "status": "pass" if _as_float(block.get("max_abs_norm_error")) <= 1.0e-5 else "needs_review",
            }
        )
    surface_rows: list[dict[str, Any]] = []
    for name, block in (payload.get("surface_summaries") or {}).items():
        if not isinstance(block, dict):
            continue
        surface_rows.append(
            {
                "surface": name,
                "num_cases": block.get("num_cases"),
                "passing_cases": block.get("passing_cases"),
                "max_norm_error": block.get("max_abs_norm_error"),
                "mean_norm_error": block.get("mean_abs_norm_error"),
                "min_finite_fraction": block.get("min_finite_fraction"),
                "min_raw_mixture_norm": block.get("min_raw_mixture_norm"),
                "status": "pass" if _as_float(block.get("max_abs_norm_error")) <= 1.0e-5 else "needs_review",
            }
        )
    return {
        "status": "supported_runtime_contract" if passed and not hidden_mag else "needs_review",
        "active_surfaces": payload.get("active_surfaces", []),
        "phase_only_contract_passed": passed,
        "hidden_magnitude_channel_detected": hidden_mag,
        "raw_mixture_cancellation_status": _get(payload, "mixture_cancellation", "status"),
        "min_raw_mixture_norm": _get(payload, "mixture_cancellation", "min_raw_mixture_norm"),
        "raw_mixture_state_count": _get(payload, "mixture_cancellation", "state_count"),
        "max_phasor_norm_error": max_err,
        "mean_abs_norm_error": overall.get("mean_abs_norm_error"),
        "min_finite_fraction": min_finite,
        "num_checked_phasors": overall.get("tensor_count"),
        "num_norm_violations": violations,
        "violating_boundaries": overall.get("violating_paths", []),
        "prefix_failure_detected_before_fix": detected_prefix_failure,
        "pre_fix_max_phasor_norm_error": prefix_max_err,
        "pre_fix_num_norm_violations": prefix_violations,
        "negative_control_enabled": bool(payload.get("negative_control_enabled")),
        "negative_control_detected": bool(negative.get("detected")),
        "negative_control_scale": negative.get("scale"),
        "negative_control_max_phasor_norm_error": _as_float(
            negative_aggregate.get("max_phasor_norm_error", negative_aggregate.get("max_abs_norm_error"))
        )
        if negative_aggregate
        else None,
        "negative_control_num_norm_violations": int(_as_float(negative_aggregate.get("num_norm_violations")))
        if negative_aggregate
        else None,
        "surface_rows": surface_rows,
        "rows": rows,
    }


def summarize_packet_alias_contract(payload: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(payload, dict):
        return {"status": "missing"}
    rows = payload.get("rows", []) if isinstance(payload.get("rows"), list) else []
    return {
        "status": "pass" if payload.get("status") == "pass" else "needs_review",
        "device": payload.get("device"),
        "num_rows": len(rows),
        "max_input_delta": max((_as_float(row.get("input_max_abs_delta")) for row in rows if isinstance(row, dict)), default=0.0),
        "max_output_delta": max((_as_float(row.get("output_max_abs_delta")) for row in rows if isinstance(row, dict)), default=0.0),
        "max_output_norm_error": max(
            (_as_float(row.get("output_max_norm_error")) for row in rows if isinstance(row, dict)),
            default=0.0,
        ),
        "any_shared_storage": any(bool(row.get("input_output_share_storage")) for row in rows if isinstance(row, dict)),
        "rows": rows,
    }


def summarize_q_basis(payload: dict[str, Any] | None) -> dict[str, Any]:
    variants = _variant_map(payload)
    rows: list[dict[str, Any]] = []
    for name, row in variants.items():
        m = _metrics(row)
        child = _get(m, "child_branch", default={})
        phase = _get(m, "phase_only_branch", default={})
        law = _get(m, "law_signature_family", default={})
        bench = _get(m, "benchmarkish_heldout", default={})
        naked = _get(m, "source_split", "naked_rafa", default={})
        rows.append(
            {
                "variant": name,
                "status": row.get("status"),
                "child_branch": child.get("mean_child_real_branch_fraction"),
                "phase_only_branch": phase.get("phase_only_real_branch_fraction"),
                "law_families": law.get("mean_num_law_families"),
                "law_entropy": law.get("mean_law_family_entropy"),
                "major_gain": bench.get("mean_major_gain"),
                "mean_loss": bench.get("mean_loss"),
                "naked_phase_only_branch": naked.get("phase_only_real_branch_fraction"),
                "naked_child_branch": naked.get("mean_child_real_branch_fraction"),
            }
        )
    ram = next((row for row in rows if row["variant"] == "ramanujan_baseline"), None)
    qtrace = next((row for row in rows if row["variant"] == "qtrace_only_control"), None)
    ram_qtrace_same = False
    if ram and qtrace:
        ram_qtrace_same = (
            abs(_as_float(ram.get("phase_only_branch")) - _as_float(qtrace.get("phase_only_branch"))) < 1.0e-8
            and abs(_as_float(ram.get("law_families")) - _as_float(qtrace.get("law_families"))) < 1.0e-8
            and abs(_as_float(ram.get("major_gain")) - _as_float(qtrace.get("major_gain"))) < 1.0e-5
        )
    return {
        "status": "ramanujan_specific_claim_not_proven" if ram_qtrace_same else "needs_review",
        "ramanujan_qtrace_numerically_same": ram_qtrace_same,
        "rows": rows,
    }


def summarize_audio(payloads: dict[str, dict[str, Any] | None]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    magnitude_leakage = False
    phase_leakage = False
    for mode, payload in payloads.items():
        if not isinstance(payload, dict):
            continue
        magnitude_leakage = magnitude_leakage or bool(payload.get("future_target_magnitude_reused"))
        magnitude_leakage = magnitude_leakage or bool(payload.get("target_future_stft_magnitude_accessed"))
        phase_leakage = phase_leakage or bool(payload.get("future_target_phase_reused"))
        phase_leakage = phase_leakage or bool(payload.get("target_future_stft_phase_accessed"))
        methods = payload.get("method_summary")
        if not isinstance(methods, dict):
            continue
        for method, metrics in methods.items():
            if not isinstance(metrics, dict):
                continue
            method_phase_seed = "waveform_copy_last_prefix" if method == "baseline_copy_last" else payload.get(
                "phase_seed_policy",
                "velocity",
            )
            rows.append(
                {
                    "magnitude_mode": payload.get("magnitude_mode") or mode,
                    "phase_seed_policy": method_phase_seed,
                    "method": method,
                    "case_count": metrics.get("case_count"),
                    "mean_corr": metrics.get("mean_corr"),
                    "mean_mae": metrics.get("mean_mae"),
                    "mean_mse": metrics.get("mean_mse"),
                    "mean_loop_autocorr_peak": metrics.get("mean_loop_autocorr_peak"),
                    "mean_first_chunk_reentry": metrics.get("mean_first_chunk_reentry"),
                    "any_future_target_magnitude_reused": metrics.get("any_future_target_magnitude_reused"),
                    "target_future_stft_magnitude_accessed": metrics.get(
                        "any_target_future_stft_magnitude_accessed",
                        payload.get("target_future_stft_magnitude_accessed", False),
                    ),
                    "future_target_phase_reused": metrics.get(
                        "any_future_target_phase_reused",
                        payload.get("future_target_phase_reused", False),
                    ),
                    "target_future_stft_phase_accessed": metrics.get(
                        "any_target_future_stft_phase_accessed",
                        payload.get("target_future_stft_phase_accessed", False),
                    ),
                }
            )
    circleworld_wins = 0
    for mode in sorted({str(row["magnitude_mode"]) for row in rows}):
        mode_rows = [row for row in rows if row["magnitude_mode"] == mode]
        cw = next((row for row in mode_rows if row["method"] == "circleworld"), None)
        if not cw:
            continue
        if _as_float(cw.get("mean_corr"), -1.0) >= max(_as_float(row.get("mean_corr"), -1.0) for row in mode_rows):
            circleworld_wins += 1
        if _as_float(cw.get("mean_mae"), 1e9) <= min(_as_float(row.get("mean_mae"), 1e9) for row in mode_rows):
            circleworld_wins += 1
        if _as_float(cw.get("mean_mse"), 1e9) <= min(_as_float(row.get("mean_mse"), 1e9) for row in mode_rows):
            circleworld_wins += 1
        if _as_float(cw.get("mean_loop_autocorr_peak"), 1e9) <= min(
            _as_float(row.get("mean_loop_autocorr_peak"), 1e9) for row in mode_rows
        ):
            circleworld_wins += 1
    status = "testable_weak_not_proven"
    if magnitude_leakage or phase_leakage:
        status = "invalid_future_leakage"
    return {
        "status": status,
        "future_target_magnitude_reused": magnitude_leakage,
        "future_target_phase_reused": phase_leakage,
        "circleworld_metric_wins": circleworld_wins,
        "rows": rows,
    }


def summarize_benchmark(payload: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(payload, dict):
        return {"status": "missing"}
    rows = payload.get("rows", []) if isinstance(payload.get("rows"), list) else []
    macro = payload.get("macro_time_diagnostics") if isinstance(payload.get("macro_time_diagnostics"), dict) else {}
    delta = macro.get("delta") if isinstance(macro.get("delta"), dict) else {}
    circleworld = macro.get("circleworld") if isinstance(macro.get("circleworld"), dict) else {}
    reference = macro.get("reference") if isinstance(macro.get("reference"), dict) else {}
    worst_rows = sorted(
        [
            {
                "name": row.get("name"),
                "corr": _as_float(row.get("corr")),
                "mae": _as_float(row.get("mae")),
                "mse": _as_float(row.get("mse")),
            }
            for row in rows
            if isinstance(row, dict)
        ],
        key=lambda row: row["corr"],
    )[:5]
    mean_corr = _as_float(payload.get("mean_corr"))
    return {
        "status": "benchmark_pass" if mean_corr >= 0.90 else "benchmark_needs_review",
        "case_count": len(rows),
        "clip_seconds": payload.get("clip_seconds"),
        "mean_corr": mean_corr,
        "mean_mae": _as_float(payload.get("mean_mae")),
        "mean_mse": _as_float(payload.get("mean_mse")),
        "continuity_delta_band": delta.get("delta_severity_band"),
        "continuity_max_abs_mean_delta": _as_float(delta.get("max_abs_mean_delta")),
        "circleworld_recurrence_score": _as_float(circleworld.get("mean_recurrence_score")),
        "circleworld_recurrence_band": circleworld.get("recurrence_severity_band"),
        "reference_recurrence_score": _as_float(reference.get("mean_recurrence_score")),
        "reference_recurrence_band": reference.get("recurrence_severity_band"),
        "recurrence_score_delta": _as_float(_get(delta, "delta", "mean_recurrence_score")),
        "worst_corr_rows": worst_rows,
    }


def summarize_audio_method_compare(payload: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(payload, dict):
        return {"status": "missing"}
    rows: list[dict[str, Any]] = []
    for run in payload.get("runs", []) or []:
        if not isinstance(run, dict):
            continue
        rows.append(
            {
                "label": run.get("label"),
                "magnitude_mode": run.get("magnitude_mode"),
                "phase_seed_policy": run.get("phase_seed_policy") or "velocity",
                "carrier_method": run.get("carrier_method"),
                "case_count": run.get("case_count"),
                "future_target_magnitude_reused": run.get("future_target_magnitude_reused", False),
                "target_future_stft_magnitude_accessed": run.get("target_future_stft_magnitude_accessed", False),
                "future_target_phase_reused": run.get("future_target_phase_reused", False),
                "target_future_stft_phase_accessed": run.get("target_future_stft_phase_accessed", False),
                "mean_cw_vs_carrier_corr": run.get("mean_cw_vs_carrier_corr"),
                "mean_cw_vs_carrier_mse": run.get("mean_cw_vs_carrier_mse"),
                "mean_cw_minus_best_baseline_corr": run.get("mean_cw_minus_best_baseline_corr"),
                "mean_cw_minus_best_baseline_mse": run.get("mean_cw_minus_best_baseline_mse"),
                "cw_win_fraction_corr": run.get("cw_win_fraction_corr"),
                "cw_win_fraction_mse": run.get("cw_win_fraction_mse"),
                "carrier_lock_score": run.get("carrier_lock_score"),
                "status": run.get("status"),
            }
        )
    return {
        "status": payload.get("status", "missing"),
        "rows": rows,
    }


def summarize_audio_phase_influence(payload: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(payload, dict):
        return {"status": "missing", "rows": []}
    rows: list[dict[str, Any]] = []
    for row in payload.get("aggregate", []) or []:
        if not isinstance(row, dict):
            continue
        rows.append(
            {
                "magnitude_mode": row.get("magnitude_mode"),
                "phase_gain": row.get("phase_gain"),
                "case_count": row.get("case_count"),
                "mean_target_corr": row.get("mean_target_corr"),
                "mean_target_mae": row.get("mean_target_mae"),
                "mean_target_mse": row.get("mean_target_mse"),
                "mean_vs_gain0_corr": row.get("mean_vs_gain0_corr"),
                "mean_vs_gain0_mse": row.get("mean_vs_gain0_mse"),
                "mean_target_corr_delta_vs_gain0": row.get("mean_target_corr_delta_vs_gain0"),
                "mean_target_mse_delta_vs_gain0": row.get("mean_target_mse_delta_vs_gain0"),
                "target_corr_win_fraction_vs_gain0": row.get("target_corr_win_fraction_vs_gain0"),
                "target_mse_win_fraction_vs_gain0": row.get("target_mse_win_fraction_vs_gain0"),
                "verdict": row.get("verdict"),
            }
        )
    return {
        "status": payload.get("status", "missing"),
        "case_count": payload.get("case_count"),
        "magnitude_modes": payload.get("magnitude_modes"),
        "phase_gains": payload.get("phase_gains"),
        "mean_future_abs_phase_delta": payload.get("mean_future_abs_phase_delta"),
        "mean_future_cos_phase_delta": payload.get("mean_future_cos_phase_delta"),
        "future_target_magnitude_reused": payload.get("future_target_magnitude_reused"),
        "target_future_stft_magnitude_accessed": payload.get("target_future_stft_magnitude_accessed"),
        "rows": rows,
    }


def summarize_audio_phase_seed(payload: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(payload, dict):
        return {"status": "missing", "rows": []}
    rows: list[dict[str, Any]] = []
    for row in payload.get("aggregate", []) or []:
        if not isinstance(row, dict):
            continue
        rows.append(
            {
                "magnitude_mode": row.get("magnitude_mode"),
                "phase_seed_policy": row.get("phase_seed_policy"),
                "case_count": row.get("case_count"),
                "mean_seed_target_corr": row.get("mean_seed_target_corr"),
                "mean_circle_target_corr": row.get("mean_circle_target_corr"),
                "mean_seed_target_mae": row.get("mean_seed_target_mae"),
                "mean_circle_target_mae": row.get("mean_circle_target_mae"),
                "mean_seed_target_mse": row.get("mean_seed_target_mse"),
                "mean_circle_target_mse": row.get("mean_circle_target_mse"),
                "mean_circle_minus_seed_corr": row.get("mean_circle_minus_seed_corr"),
                "mean_circle_minus_seed_mse": row.get("mean_circle_minus_seed_mse"),
                "circle_corr_win_fraction": row.get("circle_corr_win_fraction"),
                "circle_mse_win_fraction": row.get("circle_mse_win_fraction"),
                "mean_circle_vs_seed_corr": row.get("mean_circle_vs_seed_corr"),
                "mean_phase_abs_delta": row.get("mean_phase_abs_delta"),
                "verdict": row.get("verdict"),
            }
        )
    best = (
        payload.get("best_seed_policy_by_eval_target_corr")
        if isinstance(payload.get("best_seed_policy_by_eval_target_corr"), dict)
        else payload.get("best_seed_policy_by_corr")
        if isinstance(payload.get("best_seed_policy_by_corr"), dict)
        else {}
    )
    spread = payload.get("seed_policy_spread") if isinstance(payload.get("seed_policy_spread"), dict) else {}
    return {
        "status": payload.get("status", "missing"),
        "case_count": payload.get("case_count"),
        "phase_seed_policies": payload.get("phase_seed_policies"),
        "magnitude_modes": payload.get("magnitude_modes"),
        "max_seed_corr_spread": spread.get("max_seed_corr_spread"),
        "max_seed_mse_spread": spread.get("max_seed_mse_spread"),
        "seed_policy_matters": spread.get("seed_policy_matters"),
        "future_target_magnitude_reused": payload.get("future_target_magnitude_reused"),
        "target_future_stft_magnitude_accessed": payload.get("target_future_stft_magnitude_accessed"),
        "future_target_phase_reused": payload.get("future_target_phase_reused", False),
        "target_future_stft_phase_accessed": payload.get("target_future_stft_phase_accessed", False),
        "best_seed_policy": best.get("phase_seed_policy"),
        "best_seed_magnitude_mode": best.get("magnitude_mode"),
        "best_seed_target_corr": best.get("mean_seed_target_corr"),
        "best_circle_target_corr": best.get("mean_circle_target_corr"),
        "best_circle_minus_seed_corr": best.get("mean_circle_minus_seed_corr"),
        "best_seed_target_mse": best.get("mean_seed_target_mse"),
        "best_circle_target_mse": best.get("mean_circle_target_mse"),
        "rows": rows,
    }


def summarize_audio_circle_delta_probe(payload: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(payload, dict):
        return {"status": "missing", "rows": []}
    rows: list[dict[str, Any]] = []
    for row in payload.get("aggregate", []) or []:
        if not isinstance(row, dict):
            continue
        rows.append(
            {
                "magnitude_mode": row.get("magnitude_mode"),
                "mask_mode": row.get("mask_mode"),
                "gain": row.get("gain"),
                "case_count": row.get("case_count"),
                "mean_target_corr": row.get("mean_target_corr"),
                "mean_target_mae": row.get("mean_target_mae"),
                "mean_target_mse": row.get("mean_target_mse"),
                "mean_vs_gain0_corr": row.get("mean_vs_gain0_corr"),
                "mean_vs_gain0_mse": row.get("mean_vs_gain0_mse"),
                "mean_target_corr_delta_vs_gain0": row.get("mean_target_corr_delta_vs_gain0"),
                "mean_target_mse_delta_vs_gain0": row.get("mean_target_mse_delta_vs_gain0"),
                "target_corr_win_fraction_vs_gain0": row.get("target_corr_win_fraction_vs_gain0"),
                "target_mse_win_fraction_vs_gain0": row.get("target_mse_win_fraction_vs_gain0"),
                "mean_loop_autocorr_peak": row.get("mean_loop_autocorr_peak"),
                "mean_first_chunk_reentry": row.get("mean_first_chunk_reentry"),
                "mean_mask_bin_fraction": row.get("mean_mask_bin_fraction"),
                "mean_active_bin_abs_phase_delta": row.get("mean_active_bin_abs_phase_delta"),
                "verdict": row.get("verdict"),
            }
        )
    nonzero = [row for row in rows if abs(_as_float(row.get("gain"))) > 1.0e-12]
    best_corr = max(nonzero, key=lambda row: _as_float(row.get("mean_target_corr_delta_vs_gain0")), default={})
    best_mse = min(nonzero, key=lambda row: _as_float(row.get("mean_target_mse_delta_vs_gain0")), default={})
    leakage = bool(payload.get("future_target_magnitude_reused")) or bool(
        payload.get("target_future_stft_magnitude_accessed")
    )
    leakage = leakage or bool(payload.get("future_target_phase_reused")) or bool(
        payload.get("target_future_stft_phase_accessed")
    )
    verdicts = {str(row.get("verdict")) for row in nonzero}
    if leakage or payload.get("status") == "invalid_future_leakage":
        status = "invalid_future_leakage"
    elif "delta_helpful" in verdicts or payload.get("status") == "delta_helpful":
        status = "helpful_delta_evidence"
    elif verdicts & {"delta_mse_helpful_candidate", "delta_mse_corr_tradeoff"}:
        status = "delta_tradeoff_not_proof"
    elif payload.get("status") == "delta_decorative_or_harmful":
        status = "decorative_delta_not_useful"
    else:
        status = payload.get("status", "missing")
    return {
        "status": status,
        "raw_status": payload.get("status", "missing"),
        "case_count": payload.get("case_count"),
        "magnitude_modes": payload.get("magnitude_modes"),
        "mask_modes": payload.get("mask_modes"),
        "gains": payload.get("gains"),
        "phase_seed_policy": payload.get("phase_seed_policy"),
        "future_target_magnitude_reused": payload.get("future_target_magnitude_reused"),
        "target_future_stft_magnitude_accessed": payload.get("target_future_stft_magnitude_accessed"),
        "future_target_phase_reused": payload.get("future_target_phase_reused", False),
        "target_future_stft_phase_accessed": payload.get("target_future_stft_phase_accessed", False),
        "best_corr_delta": best_corr.get("mean_target_corr_delta_vs_gain0"),
        "best_corr_delta_row": best_corr,
        "best_mse_delta": best_mse.get("mean_target_mse_delta_vs_gain0"),
        "best_mse_delta_row": best_mse,
        "rows": rows,
    }


def summarize_audio_delta_mechanism_probe(payload: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(payload, dict):
        return {"status": "missing", "rows": []}
    rows: list[dict[str, Any]] = []
    for row in payload.get("aggregate", []) or []:
        if not isinstance(row, dict):
            continue
        rows.append(
            {
                "magnitude_mode": row.get("magnitude_mode"),
                "mask_mode": row.get("mask_mode"),
                "mechanism": row.get("mechanism"),
                "gain": row.get("gain"),
                "case_count": row.get("case_count"),
                "mean_target_corr": row.get("mean_target_corr"),
                "mean_target_mae": row.get("mean_target_mae"),
                "mean_target_mse": row.get("mean_target_mse"),
                "mean_vs_gain0_corr": row.get("mean_vs_gain0_corr"),
                "mean_vs_gain0_mse": row.get("mean_vs_gain0_mse"),
                "mean_target_corr_delta_vs_gain0": row.get("mean_target_corr_delta_vs_gain0"),
                "median_target_corr_delta_vs_gain0": row.get("median_target_corr_delta_vs_gain0"),
                "mean_target_mse_delta_vs_gain0": row.get("mean_target_mse_delta_vs_gain0"),
                "target_corr_win_fraction_vs_gain0": row.get("target_corr_win_fraction_vs_gain0"),
                "target_mse_win_fraction_vs_gain0": row.get("target_mse_win_fraction_vs_gain0"),
                "mean_loop_autocorr_peak": row.get("mean_loop_autocorr_peak"),
                "mean_first_chunk_reentry": row.get("mean_first_chunk_reentry"),
                "mean_mask_bin_fraction": row.get("mean_mask_bin_fraction"),
                "mean_active_bin_abs_phase_delta": row.get("mean_active_bin_abs_phase_delta"),
                "mean_abs_shaped_delta": row.get("mean_abs_shaped_delta"),
                "verdict": row.get("verdict"),
            }
        )
    nonzero = [row for row in rows if abs(_as_float(row.get("gain"))) > 1.0e-12]
    best_corr = max(nonzero, key=lambda row: _as_float(row.get("mean_target_corr_delta_vs_gain0")), default={})
    best_mse = min(nonzero, key=lambda row: _as_float(row.get("mean_target_mse_delta_vs_gain0")), default={})
    leakage = bool(payload.get("future_target_magnitude_reused")) or bool(
        payload.get("target_future_stft_magnitude_accessed")
    )
    leakage = leakage or bool(payload.get("future_target_phase_reused")) or bool(
        payload.get("target_future_stft_phase_accessed")
    )
    verdicts = {str(row.get("verdict")) for row in nonzero}
    raw_status = payload.get("status", "missing")
    if leakage or raw_status == "invalid_future_leakage":
        status = "invalid_future_leakage"
    elif raw_status == "mechanism_candidate_found" or (
        verdicts & {"mechanism_candidate", "mechanism_helpful_candidate"}
    ):
        status = "candidate_found"
    elif raw_status == "mechanism_tradeoff_signal_only" or "mechanism_tradeoff_signal" in verdicts:
        status = "mechanism_tradeoff_not_proof"
    elif raw_status == "mse_tradeoff_only" or "mse_only_or_corr_tradeoff" in verdicts:
        status = "mse_tradeoff_not_proof"
    elif raw_status == "no_mechanism_candidate":
        status = "no_candidate_found"
    else:
        status = raw_status
    return {
        "status": status,
        "raw_status": raw_status,
        "schema": payload.get("schema"),
        "case_count": payload.get("case_count"),
        "magnitude_modes": payload.get("magnitude_modes"),
        "mask_modes": payload.get("mask_modes"),
        "mechanisms": payload.get("mechanisms"),
        "gains": payload.get("gains"),
        "phase_seed_policy": payload.get("phase_seed_policy"),
        "future_target_magnitude_reused": payload.get("future_target_magnitude_reused"),
        "target_future_stft_magnitude_accessed": payload.get("target_future_stft_magnitude_accessed"),
        "future_target_phase_reused": payload.get("future_target_phase_reused", False),
        "target_future_stft_phase_accessed": payload.get("target_future_stft_phase_accessed", False),
        "best_corr_delta": best_corr.get("mean_target_corr_delta_vs_gain0"),
        "best_corr_delta_row": best_corr,
        "best_mse_delta": best_mse.get("mean_target_mse_delta_vs_gain0"),
        "best_mse_delta_row": best_mse,
        "rows": rows,
    }


def summarize_audio_predeclared_lockbox_manifest(payload: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(payload, dict):
        return {"status": "missing", "groups": []}
    groups: list[dict[str, Any]] = []
    raw_groups = payload.get("groups", {}) or {}
    group_iter = raw_groups.values() if isinstance(raw_groups, dict) else raw_groups
    for group in group_iter:
        if not isinstance(group, dict):
            continue
        groups.append(
            {
                "group": group.get("group_id") or group.get("group"),
                "role": group.get("role"),
                "selected_count": group.get("selected_count"),
                "min_cases": group.get("min_cases"),
                "max_cases": group.get("max_cases"),
                "met_min_cases": group.get("met_min_cases"),
                "candidate_count_before_disjoint_selection": group.get("candidate_count_before_disjoint_selection"),
            }
        )
    guardrails = payload.get("guardrails") if isinstance(payload.get("guardrails"), dict) else {}
    selected = int(_as_float(payload.get("selected_case_count")))
    unique_paths = int(_as_float(payload.get("selected_unique_paths")))
    unique_hashes = int(_as_float(payload.get("selected_unique_content_hashes")))
    unique_stems = int(_as_float(payload.get("selected_unique_provider_neutral_stems")))
    unmet = payload.get("unmet_min_case_groups", []) or []
    status = "lockbox_manifest_ready"
    if unmet or selected <= 0:
        status = "lockbox_manifest_needs_review"
    if selected != unique_paths or selected != unique_hashes or selected != unique_stems:
        status = "lockbox_manifest_uniqueness_failure"
    return {
        "status": status,
        "schema": payload.get("schema"),
        "selected_case_count": selected,
        "selected_unique_paths": unique_paths,
        "selected_unique_content_hashes": unique_hashes,
        "selected_unique_provider_neutral_stems": unique_stems,
        "unmet_min_case_groups": unmet,
        "cases_json_path": payload.get("cases_json_path"),
        "predeclared_row": payload.get("predeclared_row"),
        "timing": payload.get("timing"),
        "selection_uses_target_future_metrics": guardrails.get("selection_uses_target_future_metrics"),
        "selection_uses_baseline_metrics": guardrails.get("selection_uses_baseline_metrics"),
        "selection_uses_correlation_metrics": guardrails.get("selection_uses_correlation_metrics"),
        "selection_requires_riff_readable_wav": guardrails.get("selection_requires_riff_readable_wav"),
        "selection_enforces_content_hash_uniqueness": guardrails.get("selection_enforces_content_hash_uniqueness"),
        "selection_enforces_provider_neutral_stem_dedupe": guardrails.get(
            "selection_enforces_provider_neutral_stem_dedupe"
        ),
        "groups": groups,
    }


def summarize_audio_predeclared_lockbox_compare(payload: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(payload, dict):
        return {"status": "missing", "groups": [], "baseline_correlation_bins": []}
    overall = payload.get("overall_summary") if isinstance(payload.get("overall_summary"), dict) else {}
    groups = [
        row
        for row in payload.get("groups", []) or []
        if isinstance(row, dict)
    ]
    candidates = [
        row
        for row in payload.get("candidate_summaries", []) or []
        if isinstance(row, dict)
    ]
    bins = [
        row
        for row in payload.get("baseline_correlation_bins", []) or []
        if isinstance(row, dict)
    ]

    def fixed(name: str) -> dict[str, Any]:
        return next((row for row in bins if row.get("bin") == name), {})

    non_bad_bins = [
        row
        for row in bins
        if row.get("bin") != "bad_baseline" and int(_as_float(row.get("case_count"))) > 0
    ]
    non_bad_cases = sum(int(_as_float(row.get("case_count"))) for row in non_bad_bins)
    if non_bad_cases > 0:
        non_bad_mean_corr_delta = sum(
            _as_float(row.get("mean_corr_delta")) * int(_as_float(row.get("case_count")))
            for row in non_bad_bins
        ) / non_bad_cases
        non_bad_corr_win_fraction = sum(
            _as_float(row.get("corr_win_fraction")) * int(_as_float(row.get("case_count")))
            for row in non_bad_bins
        ) / non_bad_cases
    else:
        non_bad_mean_corr_delta = 0.0
        non_bad_corr_win_fraction = 0.0
    return {
        "status": payload.get("status", "present"),
        "schema": payload.get("schema"),
        "comparison_scope": payload.get("comparison_scope"),
        "excluded_roles": payload.get("excluded_roles"),
        "excluded_by_role_count": payload.get("excluded_by_role_count"),
        "manifest_path": payload.get("manifest_path"),
        "case_count": payload.get("case_count"),
        "analysis_row_count": payload.get("analysis_row_count"),
        "overall_mean_corr_delta": overall.get("mean_corr_delta"),
        "overall_median_corr_delta": overall.get("median_corr_delta"),
        "overall_corr_win_fraction": overall.get("corr_win_fraction"),
        "overall_mean_mse_delta": overall.get("mean_mse_delta"),
        "overall_positive_outlier_share": overall.get("positive_outlier_share"),
        "overall_min_leave_one_out_corr_delta": overall.get("min_leave_one_out_corr_delta"),
        "bad_baseline": fixed("bad_baseline"),
        "weak_baseline": fixed("weak_baseline"),
        "moderate_baseline": fixed("moderate_baseline"),
        "good_baseline": fixed("good_baseline"),
        "non_bad_case_count": non_bad_cases,
        "non_bad_mean_corr_delta": non_bad_mean_corr_delta,
        "non_bad_corr_win_fraction": non_bad_corr_win_fraction,
        "groups": groups,
        "candidate_summaries": candidates,
        "baseline_correlation_bins": bins,
    }


def summarize_internal_phase_law_variant_compare(payload: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(payload, dict):
        return {"status": "missing", "overall_summaries": [], "candidate_summaries": []}
    overall = [
        row
        for row in payload.get("overall_summaries", []) or []
        if isinstance(row, dict)
    ]
    candidates = [
        row
        for row in payload.get("candidate_summaries", []) or []
        if isinstance(row, dict)
    ]
    best_overall = overall[0] if overall else {}
    best_candidate = candidates[0] if candidates else {}
    return {
        "status": payload.get("status", "present"),
        "schema": payload.get("schema"),
        "baseline": payload.get("baseline"),
        "case_count": payload.get("case_count"),
        "delta_row_count": payload.get("delta_row_count"),
        "excluded_roles": payload.get("excluded_roles"),
        "missing_matches": payload.get("missing_matches"),
        "best_overall_run": best_overall.get("run_label"),
        "best_overall_mean_corr_delta": best_overall.get("mean_corr_delta_vs_baseline_config"),
        "best_overall_median_corr_delta": best_overall.get("median_corr_delta_vs_baseline_config"),
        "best_overall_corr_win_fraction": best_overall.get("corr_win_fraction_vs_baseline_config"),
        "best_overall_mean_mse_delta": best_overall.get("mean_mse_delta_vs_baseline_config"),
        "best_candidate_run": best_candidate.get("run_label"),
        "best_candidate_status": best_candidate.get("status"),
        "best_candidate_magnitude_mode": best_candidate.get("magnitude_mode"),
        "best_candidate_mask_mode": best_candidate.get("mask_mode"),
        "best_candidate_mechanism": best_candidate.get("mechanism"),
        "best_candidate_gain": best_candidate.get("gain"),
        "best_candidate_mean_corr_delta": best_candidate.get("mean_corr_delta_vs_baseline_config"),
        "best_candidate_median_corr_delta": best_candidate.get("median_corr_delta_vs_baseline_config"),
        "best_candidate_corr_win_fraction": best_candidate.get("corr_win_fraction_vs_baseline_config"),
        "best_candidate_mean_mse_delta": best_candidate.get("mean_mse_delta_vs_baseline_config"),
        "overall_summaries": overall,
        "candidate_summaries": candidates,
    }


def summarize_internal_phase_law_objective_score(payload: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(payload, dict):
        return {"status": "missing", "rows": []}
    rows = [
        row
        for row in payload.get("rows", []) or []
        if isinstance(row, dict)
    ]
    best_row = rows[0] if rows else {}
    target_row = payload.get("target_row", payload.get("target_low_energy_row"))
    target_row_count = payload.get("target_row_count", payload.get("target_low_energy_row_count"))
    target_candidate_count = payload.get(
        "target_row_candidate_count", payload.get("target_low_energy_candidate_count")
    )
    target_best_run = payload.get("target_row_best_run", payload.get("target_low_energy_best_run"))
    target_best_decision = payload.get(
        "target_row_best_decision", payload.get("target_low_energy_best_decision")
    )
    target_best_score = payload.get("target_row_best_score", payload.get("target_low_energy_best_score"))
    target_best_direct_mean = payload.get(
        "target_row_best_direct_mean_corr_delta",
        payload.get("target_low_energy_best_direct_mean_corr_delta"),
    )
    target_best_direct_median = payload.get(
        "target_row_best_direct_median_corr_delta",
        payload.get("target_low_energy_best_direct_median_corr_delta"),
    )
    target_best_direct_wins = payload.get(
        "target_row_best_direct_corr_win_fraction",
        payload.get("target_low_energy_best_direct_corr_win_fraction"),
    )
    target_best_abs_mean = payload.get(
        "target_row_best_absolute_mean_corr_delta",
        payload.get("target_low_energy_best_absolute_mean_corr_delta"),
    )
    target_best_abs_median = payload.get(
        "target_row_best_absolute_median_corr_delta",
        payload.get("target_low_energy_best_absolute_median_corr_delta"),
    )
    target_best_abs_wins = payload.get(
        "target_row_best_absolute_corr_win_fraction",
        payload.get("target_low_energy_best_absolute_corr_win_fraction"),
    )
    target_best_nonbad = payload.get(
        "target_row_best_nonbad_bin_mean_corr_delta",
        payload.get("target_low_energy_best_nonbad_bin_mean_corr_delta"),
    )
    target_best_nonbad_wins = payload.get(
        "target_row_best_nonbad_bin_corr_win_fraction",
        payload.get("target_low_energy_best_nonbad_bin_corr_win_fraction"),
    )
    target_best_good = payload.get(
        "target_row_best_good_bin_mean_corr_delta",
        payload.get("target_low_energy_best_good_bin_mean_corr_delta"),
    )
    return {
        "status": payload.get("status", "present"),
        "schema": payload.get("schema"),
        "run_count": payload.get("run_count"),
        "best_run": payload.get("best_run"),
        "best_decision": payload.get("best_decision"),
        "best_score": payload.get("best_score"),
        "mean_run_score": payload.get("mean_run_score"),
        "best_direct_overall_mean_corr_delta": best_row.get("direct_overall_mean_corr_delta"),
        "best_direct_overall_corr_win_fraction": best_row.get("direct_overall_corr_win_fraction"),
        "best_direct_candidate_mean_corr_delta": best_row.get("direct_best_mean_corr_delta"),
        "best_direct_candidate_corr_win_fraction": best_row.get("direct_best_corr_win_fraction"),
        "best_absolute_nonbad_mean_corr_delta": best_row.get("absolute_best_nonbad_mean_corr_delta"),
        "best_absolute_nonbad_corr_win_fraction": best_row.get("absolute_best_nonbad_corr_win_fraction"),
        "best_matched_absolute_mean_corr_delta": best_row.get("matched_absolute_mean_corr_delta"),
        "best_matched_absolute_corr_win_fraction": best_row.get("matched_absolute_corr_win_fraction"),
        "best_matched_absolute_nonbad_bin_mean_corr_delta": best_row.get(
            "matched_absolute_nonbad_bin_mean_corr_delta"
        ),
        "best_matched_absolute_nonbad_bin_corr_win_fraction": best_row.get(
            "matched_absolute_nonbad_bin_corr_win_fraction"
        ),
        "best_matched_absolute_good_bin_mean_corr_delta": best_row.get(
            "matched_absolute_good_bin_mean_corr_delta"
        ),
        "best_matched_absolute_status": best_row.get("matched_absolute_status"),
        "target_row": target_row,
        "target_row_count": target_row_count,
        "target_row_candidate_count": target_candidate_count,
        "target_row_best_run": target_best_run,
        "target_row_best_decision": target_best_decision,
        "target_row_best_score": target_best_score,
        "target_row_best_direct_mean_corr_delta": target_best_direct_mean,
        "target_row_best_direct_median_corr_delta": target_best_direct_median,
        "target_row_best_direct_corr_win_fraction": target_best_direct_wins,
        "target_row_best_absolute_mean_corr_delta": target_best_abs_mean,
        "target_row_best_absolute_median_corr_delta": target_best_abs_median,
        "target_row_best_absolute_corr_win_fraction": target_best_abs_wins,
        "target_row_best_nonbad_bin_mean_corr_delta": target_best_nonbad,
        "target_row_best_nonbad_bin_corr_win_fraction": target_best_nonbad_wins,
        "target_row_best_good_bin_mean_corr_delta": target_best_good,
        "target_low_energy_row": target_row,
        "target_low_energy_row_count": target_row_count,
        "target_low_energy_candidate_count": target_candidate_count,
        "target_low_energy_best_run": target_best_run,
        "target_low_energy_best_decision": target_best_decision,
        "target_low_energy_best_score": target_best_score,
        "target_low_energy_best_direct_mean_corr_delta": target_best_direct_mean,
        "target_low_energy_best_direct_median_corr_delta": target_best_direct_median,
        "target_low_energy_best_direct_corr_win_fraction": target_best_direct_wins,
        "target_low_energy_best_absolute_mean_corr_delta": target_best_abs_mean,
        "target_low_energy_best_absolute_median_corr_delta": target_best_abs_median,
        "target_low_energy_best_absolute_corr_win_fraction": target_best_abs_wins,
        "target_low_energy_best_nonbad_bin_mean_corr_delta": target_best_nonbad,
        "target_low_energy_best_nonbad_bin_corr_win_fraction": target_best_nonbad_wins,
        "target_low_energy_best_good_bin_mean_corr_delta": target_best_good,
        "rows": rows,
    }


def summarize_internal_phase_law_joint_rows(payload: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(payload, dict):
        return {"status": "missing", "rows": []}
    rows = [row for row in payload.get("rows", []) or [] if isinstance(row, dict)]
    best_row = payload.get("best_row") if isinstance(payload.get("best_row"), dict) else (rows[0] if rows else {})
    target_row = payload.get("target_row", payload.get("target_low_energy_row"))
    target_row_count = payload.get("target_row_count", payload.get("target_low_energy_row_count"))
    target_candidate_count = payload.get(
        "target_row_candidate_count", payload.get("target_low_energy_candidate_count")
    )
    target_best_row = payload.get("target_row_best_row", payload.get("target_low_energy_best_row"))
    target_best_run = payload.get("target_row_best_run", payload.get("target_low_energy_best_run"))
    target_best_decision = payload.get(
        "target_row_best_decision", payload.get("target_low_energy_best_decision")
    )
    target_best_score = payload.get("target_row_best_score", payload.get("target_low_energy_best_score"))
    target_best_direct_mean = payload.get(
        "target_row_best_direct_mean_corr_delta",
        payload.get("target_low_energy_best_direct_mean_corr_delta"),
    )
    target_best_direct_median = payload.get(
        "target_row_best_direct_median_corr_delta",
        payload.get("target_low_energy_best_direct_median_corr_delta"),
    )
    target_best_direct_wins = payload.get(
        "target_row_best_direct_corr_win_fraction",
        payload.get("target_low_energy_best_direct_corr_win_fraction"),
    )
    target_best_abs_mean = payload.get(
        "target_row_best_absolute_mean_corr_delta",
        payload.get("target_low_energy_best_absolute_mean_corr_delta"),
    )
    target_best_abs_median = payload.get(
        "target_row_best_absolute_median_corr_delta",
        payload.get("target_low_energy_best_absolute_median_corr_delta"),
    )
    target_best_abs_wins = payload.get(
        "target_row_best_absolute_corr_win_fraction",
        payload.get("target_low_energy_best_absolute_corr_win_fraction"),
    )
    target_best_nonbad = payload.get(
        "target_row_best_nonbad_bin_mean_corr_delta",
        payload.get("target_low_energy_best_nonbad_bin_mean_corr_delta"),
    )
    target_best_nonbad_wins = payload.get(
        "target_row_best_nonbad_bin_corr_win_fraction",
        payload.get("target_low_energy_best_nonbad_bin_corr_win_fraction"),
    )
    target_best_good = payload.get(
        "target_row_best_good_bin_mean_corr_delta",
        payload.get("target_low_energy_best_good_bin_mean_corr_delta"),
    )
    return {
        "status": payload.get("status", "present"),
        "schema": payload.get("schema"),
        "row_count": payload.get("row_count"),
        "candidate_count": payload.get("candidate_count"),
        "missing_absolute_count": payload.get("missing_absolute_count"),
        "decision_counts": payload.get("decision_counts"),
        "mean_absolute_nonbad_bin_corr_delta": payload.get("mean_absolute_nonbad_bin_corr_delta"),
        "best_run": best_row.get("run_label"),
        "best_magnitude_mode": best_row.get("magnitude_mode"),
        "best_mask_mode": best_row.get("mask_mode"),
        "best_gain": best_row.get("gain"),
        "best_decision": best_row.get("decision"),
        "best_score": best_row.get("score"),
        "best_direct_mean_corr_delta": best_row.get("direct_mean_corr_delta"),
        "best_direct_corr_win_fraction": best_row.get("direct_corr_win_fraction"),
        "best_absolute_mean_corr_delta": best_row.get("absolute_mean_corr_delta"),
        "best_absolute_nonbad_bin_mean_corr_delta": best_row.get("absolute_nonbad_bin_mean_corr_delta"),
        "best_absolute_weak_bin_mean_corr_delta": best_row.get("absolute_weak_bin_mean_corr_delta"),
        "best_absolute_moderate_bin_mean_corr_delta": best_row.get("absolute_moderate_bin_mean_corr_delta"),
        "best_absolute_good_bin_mean_corr_delta": best_row.get("absolute_good_bin_mean_corr_delta"),
        "best_absolute_median_corr_delta": best_row.get("absolute_median_corr_delta"),
        "best_absolute_corr_win_fraction": best_row.get("absolute_corr_win_fraction"),
        "best_absolute_status": best_row.get("absolute_status"),
        "target_row": target_row,
        "target_row_count": target_row_count,
        "target_row_candidate_count": target_candidate_count,
        "target_row_best_row": target_best_row,
        "target_row_best_run": target_best_run,
        "target_row_best_decision": target_best_decision,
        "target_row_best_score": target_best_score,
        "target_row_best_direct_mean_corr_delta": target_best_direct_mean,
        "target_row_best_direct_median_corr_delta": target_best_direct_median,
        "target_row_best_direct_corr_win_fraction": target_best_direct_wins,
        "target_row_best_absolute_mean_corr_delta": target_best_abs_mean,
        "target_row_best_absolute_median_corr_delta": target_best_abs_median,
        "target_row_best_absolute_corr_win_fraction": target_best_abs_wins,
        "target_row_best_nonbad_bin_mean_corr_delta": target_best_nonbad,
        "target_row_best_nonbad_bin_corr_win_fraction": target_best_nonbad_wins,
        "target_row_best_good_bin_mean_corr_delta": target_best_good,
        "target_low_energy_row": target_row,
        "target_low_energy_row_count": target_row_count,
        "target_low_energy_candidate_count": target_candidate_count,
        "target_low_energy_best_row": target_best_row,
        "target_low_energy_best_run": target_best_run,
        "target_low_energy_best_decision": target_best_decision,
        "target_low_energy_best_score": target_best_score,
        "target_low_energy_best_direct_mean_corr_delta": target_best_direct_mean,
        "target_low_energy_best_direct_median_corr_delta": target_best_direct_median,
        "target_low_energy_best_direct_corr_win_fraction": target_best_direct_wins,
        "target_low_energy_best_absolute_mean_corr_delta": target_best_abs_mean,
        "target_low_energy_best_absolute_median_corr_delta": target_best_abs_median,
        "target_low_energy_best_absolute_corr_win_fraction": target_best_abs_wins,
        "target_low_energy_best_nonbad_bin_mean_corr_delta": target_best_nonbad,
        "target_low_energy_best_nonbad_bin_corr_win_fraction": target_best_nonbad_wins,
        "target_low_energy_best_good_bin_mean_corr_delta": target_best_good,
        "rows": rows,
    }


def summarize_internal_phase_law_candidate_classes(payload: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(payload, dict):
        return {"status": "missing", "rows": []}
    rows = [row for row in payload.get("rows", []) or [] if isinstance(row, dict)]
    best = payload.get("best_row") if isinstance(payload.get("best_row"), dict) else (rows[0] if rows else {})
    return {
        "status": payload.get("status", "present"),
        "schema": payload.get("schema"),
        "row_count": payload.get("row_count"),
        "class_counts": payload.get("class_counts"),
        "strict_audio_candidate_count": payload.get("strict_audio_candidate_count"),
        "phase_support_candidate_count": payload.get("phase_support_candidate_count"),
        "best_class": best.get("candidate_class"),
        "best_run": best.get("run_label"),
        "best_mask_mode": best.get("mask_mode"),
        "best_gain": best.get("gain"),
        "best_class_score": best.get("class_score"),
        "best_direct_mean_corr_delta": best.get("direct_mean_corr_delta"),
        "best_absolute_mean_corr_delta": best.get("absolute_mean_corr_delta"),
        "best_absolute_nonbad_bin_mean_corr_delta": best.get("absolute_nonbad_bin_mean_corr_delta"),
        "best_absolute_good_bin_mean_corr_delta": best.get("absolute_good_bin_mean_corr_delta"),
        "rows": rows,
    }


def summarize_internal_phase_law_predeclared_diagnostic(payload: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(payload, dict):
        return {"status": "missing", "tracks": [], "top_rows": []}
    tracks = [row for row in payload.get("tracks", []) or [] if isinstance(row, dict)]
    top_rows = [row for row in payload.get("top_rows", []) or [] if isinstance(row, dict)]
    best = top_rows[0] if top_rows else {}
    return {
        "status": payload.get("status", "present"),
        "schema": payload.get("schema"),
        "promotion_effect": payload.get("promotion_effect"),
        "class_name": payload.get("class_name"),
        "track_count": payload.get("track_count"),
        "total_row_count": payload.get("total_row_count"),
        "diagnostic_count": payload.get("diagnostic_count"),
        "strict_audio_count": payload.get("strict_audio_count"),
        "best_track_by_count": payload.get("best_track_by_count"),
        "best_track": best.get("track"),
        "best_run": best.get("run_label"),
        "best_mask_mode": best.get("mask_mode"),
        "best_gain": best.get("gain"),
        "best_diagnostic_score": best.get("diagnostic_score"),
        "best_direct_mean_corr_delta": best.get("direct_mean_corr_delta"),
        "best_absolute_mean_corr_delta": best.get("absolute_mean_corr_delta"),
        "best_absolute_nonbad_bin_mean_corr_delta": best.get("absolute_nonbad_bin_mean_corr_delta"),
        "best_absolute_good_bin_mean_corr_delta": best.get("absolute_good_bin_mean_corr_delta"),
        "thresholds": (payload.get("definition") or {}).get("thresholds"),
        "tracks": tracks,
        "top_rows": top_rows,
    }


def summarize_internal_phase_law_failure_atlas(payload: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(payload, dict):
        return {"status": "missing", "baseline_bins": [], "top_runs": []}
    bins = [row for row in payload.get("baseline_bins", []) or [] if isinstance(row, dict)]
    top_runs = [row for row in payload.get("top_runs", []) or [] if isinstance(row, dict)]
    best_run = top_runs[0] if top_runs else {}
    return {
        "status": payload.get("status", "present"),
        "schema": payload.get("schema"),
        "absolute_status": payload.get("absolute_status"),
        "variant_status": payload.get("variant_status"),
        "joint_status": payload.get("joint_status"),
        "joint_candidate_count": payload.get("joint_candidate_count"),
        "joint_decision_counts": payload.get("joint_decision_counts"),
        "run_count": payload.get("run_count"),
        "family_row_count": payload.get("family_row_count"),
        "best_run": best_run.get("run_label"),
        "best_direct_mean_corr_delta": best_run.get("direct_mean_corr_delta"),
        "best_direct_corr_win_fraction": best_run.get("direct_corr_win_fraction"),
        "best_absolute_nonbad_mean_corr_delta": best_run.get("absolute_nonbad_best_mean_corr_delta"),
        "best_absolute_nonbad_corr_win_fraction": best_run.get("absolute_nonbad_best_corr_win_fraction"),
        "best_joint_decision": best_run.get("joint_best_decision"),
        "best_joint_absolute_median_corr_delta": best_run.get("joint_best_absolute_median_corr_delta"),
        "best_joint_absolute_corr_win_fraction": best_run.get("joint_best_absolute_corr_win_fraction"),
        "best_joint_absolute_status": best_run.get("joint_best_absolute_status"),
        "baseline_bins": bins,
        "top_runs": top_runs,
    }


def summarize_internal_phase_law_target_flips(payload: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(payload, dict):
        return {"status": "missing", "best_by_gain": [], "group_rows": [], "leave_one_out_rows": []}
    best_by_gain = [row for row in payload.get("best_by_gain", []) or [] if isinstance(row, dict)]
    group_rows = [row for row in payload.get("group_rows", []) or [] if isinstance(row, dict)]
    leave_one_out_rows = [
        row for row in payload.get("leave_one_out_rows", []) or [] if isinstance(row, dict)
    ]
    top_negative = [row for row in payload.get("top_negative_cases", []) or [] if isinstance(row, dict)]
    top_positive = [row for row in payload.get("top_positive_cases", []) or [] if isinstance(row, dict)]
    selected = payload.get("selected_target_gain_summary")
    selected = selected if isinstance(selected, dict) else {}
    return {
        "status": payload.get("status", "present"),
        "schema": payload.get("schema"),
        "selected_run": payload.get("selected_run"),
        "gain_count": payload.get("gain_count"),
        "candidate_row_count": payload.get("candidate_row_count"),
        "case_delta_row_count": payload.get("case_delta_row_count"),
        "selected_target_gain_mean_corr_delta": selected.get("mean_corr_delta"),
        "selected_target_gain_median_corr_delta": selected.get("median_corr_delta"),
        "selected_target_gain_corr_win_fraction": selected.get("corr_win_fraction"),
        "selected_target_gain_positive_count": selected.get("positive_count"),
        "selected_target_gain_negative_count": selected.get("negative_count"),
        "selected_target_gain_zero_count": selected.get("zero_count"),
        "category_counts": payload.get("category_counts"),
        "best_by_gain": best_by_gain,
        "group_rows": group_rows,
        "leave_one_out_rows": leave_one_out_rows,
        "top_negative_cases": top_negative,
        "top_positive_cases": top_positive,
    }


def summarize_internal_phase_law_family_masks(payload: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(payload, dict):
        return {
            "status": "missing",
            "selected_target_mask_rows": [],
            "selected_best_mask_by_family": [],
            "candidate_mask_gain_rows": [],
        }
    mask_rows = [
        row for row in payload.get("selected_target_mask_rows", []) or [] if isinstance(row, dict)
    ]
    family_rows = [
        row for row in payload.get("selected_best_mask_by_family", []) or [] if isinstance(row, dict)
    ]
    candidate_rows = [
        row for row in payload.get("candidate_mask_gain_rows", []) or [] if isinstance(row, dict)
    ]
    global_best = payload.get("global_best_direct_row")
    global_best = global_best if isinstance(global_best, dict) else {}
    low_row = next((row for row in mask_rows if row.get("mask_mode") == "low_energy_bins"), {})
    all_row = next((row for row in mask_rows if row.get("mask_mode") == "all_bins"), {})
    high_row = next((row for row in mask_rows if row.get("mask_mode") == "high_energy_bins"), {})
    return {
        "status": payload.get("status", "present"),
        "schema": payload.get("schema"),
        "selected_run": payload.get("selected_run"),
        "selected_run_basis": payload.get("selected_run_basis"),
        "objective_score": payload.get("objective_score"),
        "candidate_row_count": payload.get("candidate_row_count"),
        "case_delta_row_count": payload.get("case_delta_row_count"),
        "global_best_run": global_best.get("run_label"),
        "global_best_mask": global_best.get("mask_mode"),
        "global_best_gain": global_best.get("gain"),
        "global_best_mean_corr_delta": global_best.get("mean_corr_delta_vs_baseline_config"),
        "global_best_median_corr_delta": global_best.get("median_corr_delta_vs_baseline_config"),
        "global_best_corr_win_fraction": global_best.get("corr_win_fraction_vs_baseline_config"),
        "global_best_mean_mse_delta": global_best.get("mean_mse_delta_vs_baseline_config"),
        "selected_low_energy_mean_corr_delta": low_row.get("mean_corr_delta"),
        "selected_low_energy_median_corr_delta": low_row.get("median_corr_delta"),
        "selected_low_energy_corr_win_fraction": low_row.get("corr_win_fraction"),
        "selected_all_bins_mean_corr_delta": all_row.get("mean_corr_delta"),
        "selected_all_bins_median_corr_delta": all_row.get("median_corr_delta"),
        "selected_all_bins_corr_win_fraction": all_row.get("corr_win_fraction"),
        "selected_high_energy_mean_corr_delta": high_row.get("mean_corr_delta"),
        "selected_high_energy_median_corr_delta": high_row.get("median_corr_delta"),
        "selected_high_energy_corr_win_fraction": high_row.get("corr_win_fraction"),
        "selected_sign_flip_counts": payload.get("selected_sign_flip_counts"),
        "selected_best_mask_counts": payload.get("selected_best_mask_counts"),
        "selected_target_mask_rows": mask_rows,
        "selected_best_mask_by_family": family_rows,
        "candidate_mask_gain_rows": candidate_rows,
        "selected_leave_one_family_out_by_mask": payload.get("selected_leave_one_family_out_by_mask", []),
        "top_low_vs_all_cases": payload.get("top_low_vs_all_cases", []),
        "top_all_vs_low_cases": payload.get("top_all_vs_low_cases", []),
    }


def summarize_internal_phase_law_support_router_oracle(payload: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(payload, dict):
        return {"status": "missing", "summaries": {}}
    summaries = payload.get("summaries") if isinstance(payload.get("summaries"), dict) else {}
    deltas = payload.get("deltas_vs_global_fixed") if isinstance(payload.get("deltas_vs_global_fixed"), dict) else {}
    policy_counts = payload.get("policy_counts") if isinstance(payload.get("policy_counts"), dict) else {}
    return {
        "status": payload.get("status", "present"),
        "schema": payload.get("schema"),
        "selected_run": payload.get("selected_run"),
        "global_fixed_mask": payload.get("global_fixed_mask"),
        "family_policy": payload.get("family_policy"),
        "family_policy_counts": _get(policy_counts, "family_oracle_router", "overall"),
        "case_oracle_policy_counts": _get(policy_counts, "case_oracle_router", "overall"),
        "low_energy_fixed": summaries.get("low_energy_fixed"),
        "global_fixed_best": summaries.get("global_fixed_best"),
        "family_oracle_router": summaries.get("family_oracle_router"),
        "case_oracle_router": summaries.get("case_oracle_router"),
        "case_anti_oracle": summaries.get("case_anti_oracle"),
        "family_mean_corr_delta_gain": deltas.get("family_mean_corr_delta_gain"),
        "family_corr_win_fraction_gain": deltas.get("family_corr_win_fraction_gain"),
        "case_oracle_mean_corr_delta_gain": deltas.get("case_oracle_mean_corr_delta_gain"),
        "case_oracle_corr_win_fraction_gain": deltas.get("case_oracle_corr_win_fraction_gain"),
        "low_energy_mean_corr_delta_gap": deltas.get("low_energy_mean_corr_delta_gap"),
        "low_energy_corr_win_fraction_gap": deltas.get("low_energy_corr_win_fraction_gap"),
        "top_family_router_wins_vs_global": payload.get("top_family_router_wins_vs_global", []),
        "top_family_router_losses_vs_global": payload.get("top_family_router_losses_vs_global", []),
    }


def summarize_audio_mechanism_family_sensitivity(payload: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(payload, dict):
        return {"status": "missing", "families": []}
    families: list[dict[str, Any]] = []
    for row in payload.get("families", []) or []:
        if not isinstance(row, dict):
            continue
        top_positive = row.get("top_positive_case") if isinstance(row.get("top_positive_case"), dict) else {}
        top_negative = row.get("top_negative_case") if isinstance(row.get("top_negative_case"), dict) else {}
        families.append(
            {
                "family": row.get("family"),
                "status": row.get("status"),
                "case_count": row.get("case_count"),
                "mean_corr_delta": row.get("mean_corr_delta"),
                "median_corr_delta": row.get("median_corr_delta"),
                "min_leave_one_out_corr_delta": row.get("min_leave_one_out_corr_delta"),
                "corr_win_fraction": row.get("corr_win_fraction"),
                "mean_mse_delta": row.get("mean_mse_delta"),
                "mse_win_fraction": row.get("mse_win_fraction"),
                "positive_case_count": row.get("positive_case_count"),
                "positive_outlier_share": row.get("positive_outlier_share"),
                "top_positive_case": {
                    "name": top_positive.get("name"),
                    "source_wav": top_positive.get("source_wav"),
                    "corr_delta": top_positive.get("corr_delta"),
                    "mse_delta": top_positive.get("mse_delta"),
                },
                "top_negative_case": {
                    "name": top_negative.get("name"),
                    "source_wav": top_negative.get("source_wav"),
                    "corr_delta": top_negative.get("corr_delta"),
                    "mse_delta": top_negative.get("mse_delta"),
                },
                "case_json": row.get("case_json"),
                "probe_json": row.get("probe_json"),
            }
        )
    best = max(families, key=lambda row: _as_float(row.get("mean_corr_delta")), default={})
    return {
        "status": payload.get("status", "missing"),
        "schema": payload.get("schema"),
        "config_path": payload.get("config_path"),
        "wav_dir": payload.get("wav_dir"),
        "exclude_cases_json": payload.get("exclude_cases_json"),
        "valid_wav_count": payload.get("valid_wav_count"),
        "rejected_wav_count": payload.get("rejected_wav_count"),
        "deduped_duplicate_stem_count": payload.get("deduped_duplicate_stem_count"),
        "mechanism": payload.get("mechanism"),
        "target_gain": payload.get("target_gain"),
        "magnitude_mode": payload.get("magnitude_mode"),
        "mask_mode": payload.get("mask_mode"),
        "best_family": best.get("family"),
        "best_family_status": best.get("status"),
        "best_family_mean_corr_delta": best.get("mean_corr_delta"),
        "best_family_median_corr_delta": best.get("median_corr_delta"),
        "best_family_min_leave_one_out_corr_delta": best.get("min_leave_one_out_corr_delta"),
        "best_family_corr_win_fraction": best.get("corr_win_fraction"),
        "best_family_positive_outlier_share": best.get("positive_outlier_share"),
        "best_family_top_positive_case": best.get("top_positive_case"),
        "families": families,
    }


def _first_present(mapping: dict[str, Any] | None, *keys: str, default: Any = None) -> Any:
    if not isinstance(mapping, dict):
        return default
    for key in keys:
        if key in mapping and mapping[key] is not None:
            return mapping[key]
    return default


def _first_dict(payload: dict[str, Any] | None, paths: list[tuple[str, ...]]) -> dict[str, Any]:
    for path in paths:
        value = _get(payload, *path)
        if isinstance(value, dict):
            return value
    return {}


def _first_list(payload: dict[str, Any] | None, paths: list[tuple[str, ...]]) -> list[Any]:
    for path in paths:
        value = _get(payload, *path)
        if isinstance(value, list):
            return value
    return []


def _compact_audio_metric_row(row: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(row, dict):
        return {}
    return {
        "split": row.get("split"),
        "group": row.get("group"),
        "role": row.get("role"),
        "family": _first_present(row, "family", "holdout_family"),
        "subfamily": _first_present(row, "subfamily", "exploratory_subfamily", "name"),
        "status": row.get("status"),
        "verdict": row.get("verdict"),
        "case_count": _first_present(row, "case_count", "num_cases", "n"),
        "mechanism": row.get("mechanism"),
        "magnitude_mode": row.get("magnitude_mode"),
        "mask_mode": row.get("mask_mode"),
        "gain": _first_present(row, "gain", "target_gain"),
        "mean_corr_delta": _first_present(
            row,
            "mean_corr_delta",
            "primary_corr_delta",
            "mean_target_corr_delta_vs_gain0",
            "corr_delta",
        ),
        "median_corr_delta": _first_present(
            row,
            "median_corr_delta",
            "median_target_corr_delta_vs_gain0",
        ),
        "min_leave_one_out_corr_delta": row.get("min_leave_one_out_corr_delta"),
        "corr_win_fraction": _first_present(
            row,
            "corr_win_fraction",
            "primary_corr_win_fraction",
            "target_corr_win_fraction_vs_gain0",
        ),
        "mean_mse_delta": _first_present(
            row,
            "mean_mse_delta",
            "primary_mse_delta",
            "mean_target_mse_delta_vs_gain0",
            "mse_delta",
        ),
        "mse_win_fraction": _first_present(
            row,
            "mse_win_fraction",
            "primary_mse_win_fraction",
            "target_mse_win_fraction_vs_gain0",
        ),
        "positive_outlier_share": row.get("positive_outlier_share"),
        "top_positive_case": row.get("top_positive_case"),
        "top_negative_case": row.get("top_negative_case"),
        "mean_target_corr": row.get("mean_target_corr"),
        "mean_target_mse": row.get("mean_target_mse"),
    }


def _score_audio_metric_row(row: dict[str, Any]) -> float:
    for key in [
        "score",
        "mean_corr_delta",
        "primary_corr_delta",
        "mean_target_corr_delta_vs_gain0",
        "corr_delta",
        "median_corr_delta",
    ]:
        if key in row and row[key] is not None:
            return _as_float(row[key], float("-inf"))
    return float("-inf")


def _best_audio_metric_row(rows: list[Any]) -> dict[str, Any]:
    dict_rows = [row for row in rows if isinstance(row, dict)]
    if not dict_rows:
        return {}
    return max(dict_rows, key=_score_audio_metric_row)


def _collect_leakage_flags(payload: Any, depth: int = 0) -> dict[str, Any]:
    if depth > 4:
        return {}
    known_flags = {
        "future_target_magnitude_reused",
        "target_future_stft_magnitude_accessed",
        "future_target_phase_reused",
        "target_future_stft_phase_accessed",
        "future_magnitude_leakage",
        "future_phase_leakage",
        "leakage_detected",
        "uses_future_target_magnitude",
        "uses_future_target_phase",
    }
    flags: dict[str, Any] = {}
    if isinstance(payload, dict):
        nested = payload.get("leakage_flags")
        if isinstance(nested, dict):
            flags.update(nested)
        for key, value in payload.items():
            if key in known_flags:
                flags[key] = value
            elif isinstance(value, (dict, list)):
                for nested_key, nested_value in _collect_leakage_flags(value, depth + 1).items():
                    flags.setdefault(nested_key, nested_value)
    elif isinstance(payload, list):
        for row in payload:
            for key, value in _collect_leakage_flags(row, depth + 1).items():
                flags.setdefault(key, value)
    return flags


def summarize_audio_electric_motor_holdout(payload: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(payload, dict):
        return {"status": "missing"}
    groups = [row for row in payload.get("groups", []) or [] if isinstance(row, dict)]

    selected_fixed = _first_dict(
        payload,
        [
            ("selected_fixed_mechanism_row",),
            ("selected_fixed_row",),
            ("fixed_mechanism_row",),
            ("target_row",),
            ("selected_row",),
            ("selection", "fixed_mechanism_row"),
            ("selection", "selected_row"),
            ("fixed_mechanism", "row"),
            ("fixed", "row"),
        ],
    )
    if not selected_fixed:
        selected_fixed = next(
            (
                row
                for row in _first_list(
                    payload,
                    [
                        ("fixed_mechanism_rows",),
                        ("fixed_mechanism_results",),
                        ("rows",),
                        ("results",),
                    ],
                )
                if isinstance(row, dict) and bool(_first_present(row, "selected", "is_selected", "fixed_mechanism"))
            ),
            {},
        )
    if not selected_fixed:
        selected_fixed = {
            "status": "fixed_predeclared",
            "mechanism": payload.get("mechanism"),
            "magnitude_mode": payload.get("magnitude_mode"),
            "mask_mode": payload.get("mask_mode"),
            "target_gain": payload.get("target_gain"),
        }

    exploratory = _first_dict(
        payload,
        [
            ("best_exploratory_subfamily",),
            ("best_exploratory_subfamily_metrics",),
            ("exploratory", "best_subfamily"),
            ("exploratory", "best_subfamily_metrics"),
        ],
    )
    if not exploratory:
        exploratory = _best_audio_metric_row(
            _first_list(
                payload,
                [
                    ("exploratory_subfamilies",),
                    ("exploratory_families",),
                    ("exploratory", "subfamilies"),
                    ("exploratory", "families"),
                    ("subfamilies",),
                ],
            )
        )
    if not exploratory:
        exploratory = _best_audio_metric_row([row for row in groups if row.get("split") == "exploratory"])

    holdout = _first_dict(
        payload,
        [
            ("holdout_family_metrics",),
            ("holdout_family",),
            ("holdout", "family_metrics"),
            ("holdout", "metrics"),
            ("holdout_metrics",),
        ],
    )
    if not holdout:
        holdout_rows = _first_list(
            payload,
            [
                ("holdout_families",),
                ("holdout", "families"),
                ("holdout", "rows"),
                ("families",),
            ],
        )
        holdout = next(
            (
                row
                for row in holdout_rows
                if isinstance(row, dict)
                and "electric" in str(_first_present(row, "family", "subfamily", "name", default="")).lower()
                and "motor" in str(_first_present(row, "family", "subfamily", "name", default="")).lower()
            ),
            _best_audio_metric_row(holdout_rows),
        )
    if not holdout:
        holdout = _best_audio_metric_row([row for row in groups if row.get("split") == "holdout"])

    status = _first_present(payload, "status", default=_get(payload, "summary", "status", default="present"))
    verdict = payload.get("verdict")
    if isinstance(verdict, dict):
        status = _first_present(verdict, "status", "verdict", default=status)
    elif status == "present" and verdict is not None:
        status = verdict

    return {
        "status": status,
        "schema": payload.get("schema"),
        "config_path": payload.get("config_path"),
        "valid_wav_count": _first_present(
            payload,
            "valid_wav_count",
            "valid_count",
            "num_valid",
            default=_get(payload, "counts", "valid_wav_count"),
        ),
        "rejected_wav_count": _first_present(
            payload,
            "rejected_wav_count",
            "rejected_count",
            "num_rejected",
            default=_get(payload, "counts", "rejected_wav_count"),
        ),
        "deduped_duplicate_stem_count": _first_present(
            payload,
            "deduped_duplicate_stem_count",
            "dedupe_count",
            "deduped_count",
            "duplicate_stem_count",
            default=_get(payload, "counts", "deduped_duplicate_stem_count"),
        ),
        "withheld_razor_exploratory_count": payload.get("withheld_razor_exploratory_count"),
        "withheld_razor_exploratory_paths": payload.get("withheld_razor_exploratory_paths", []),
        "exclude_razor_from_exploratory": payload.get("exclude_razor_from_exploratory"),
        "best_nonrazor_motor_family": payload.get("best_nonrazor_motor_family"),
        "best_nonrazor_motor_status": payload.get("best_nonrazor_motor_status"),
        "best_nonrazor_motor_mean_corr_delta": payload.get("best_nonrazor_motor_mean_corr_delta"),
        "best_nonrazor_motor_median_corr_delta": payload.get("best_nonrazor_motor_median_corr_delta"),
        "best_nonrazor_motor_corr_win_fraction": payload.get("best_nonrazor_motor_corr_win_fraction"),
        "best_nonrazor_motor_positive_outlier_share": payload.get("best_nonrazor_motor_positive_outlier_share"),
        "best_razor_family": payload.get("best_razor_family"),
        "best_razor_status": payload.get("best_razor_status"),
        "best_razor_mean_corr_delta": payload.get("best_razor_mean_corr_delta"),
        "best_razor_median_corr_delta": payload.get("best_razor_median_corr_delta"),
        "best_razor_corr_win_fraction": payload.get("best_razor_corr_win_fraction"),
        "best_razor_positive_outlier_share": payload.get("best_razor_positive_outlier_share"),
        "selected_case_occurrences": payload.get("selected_case_occurrences"),
        "selected_unique_paths": payload.get("selected_unique_paths"),
        "selected_path_reuse_count": payload.get("selected_path_reuse_count"),
        "selected_unique_content_hashes": payload.get("selected_unique_content_hashes"),
        "selected_content_hash_reuse_count": payload.get("selected_content_hash_reuse_count"),
        "selected_fixed_mechanism_row": _compact_audio_metric_row(selected_fixed),
        "best_exploratory_subfamily": _compact_audio_metric_row(exploratory),
        "holdout_family_metrics": _compact_audio_metric_row(holdout),
        "groups": [_compact_audio_metric_row(row) for row in groups],
        "leakage_flags": _collect_leakage_flags(payload),
    }


def summarize_audio_steady_phenotype_manifest(payload: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(payload, dict):
        return {"status": "missing", "groups": []}
    groups = [row for row in payload.get("groups", []) or [] if isinstance(row, dict)]

    def by_role(role: str) -> list[dict[str, Any]]:
        return [row for row in groups if row.get("role") == role]

    best_overall = _best_audio_metric_row(groups)
    best_steady = _best_audio_metric_row(by_role("steady_buzz_control"))
    best_motor = _best_audio_metric_row(by_role("no_razor_motor"))
    best_synth = _best_audio_metric_row(by_role("synth_control"))
    best_razor = _best_audio_metric_row(by_role("single_source_probe"))
    return {
        "status": payload.get("status", "present"),
        "schema": payload.get("schema"),
        "config_path": payload.get("config_path"),
        "valid_wav_count": payload.get("valid_wav_count"),
        "rejected_wav_count": payload.get("rejected_wav_count"),
        "deduped_duplicate_stem_count": payload.get("deduped_duplicate_stem_count"),
        "selected_case_occurrences": payload.get("selected_case_occurrences"),
        "selected_unique_paths": payload.get("selected_unique_paths"),
        "selected_unique_content_hashes": payload.get("selected_unique_content_hashes"),
        "selection_reuse_rejection_count": payload.get("selection_reuse_rejection_count"),
        "target_leakage_detected": payload.get("target_leakage_detected"),
        "selected_fixed_mechanism_row": _compact_audio_metric_row(
            payload.get("selected_fixed_mechanism_row") if isinstance(payload.get("selected_fixed_mechanism_row"), dict) else {}
        ),
        "best_overall_group": _compact_audio_metric_row(best_overall),
        "best_steady_buzz_group": _compact_audio_metric_row(best_steady),
        "best_no_razor_motor_group": _compact_audio_metric_row(best_motor),
        "best_synth_control_group": _compact_audio_metric_row(best_synth),
        "best_razor_probe_group": _compact_audio_metric_row(best_razor),
        "groups": [_compact_audio_metric_row(row) for row in groups],
        "leakage_flags": _collect_leakage_flags(payload),
    }


def summarize_audio_phase_phenotype_diagnostics(payload: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(payload, dict):
        return {"status": "missing", "top_correlations": []}
    correlations = [
        row
        for row in payload.get("feature_correlations_no_single_source", []) or []
        if isinstance(row, dict)
    ]
    role_summary = [
        row
        for row in payload.get("role_summary", []) or []
        if isinstance(row, dict)
    ]
    top_cases = [
        row
        for row in payload.get("top_cases_by_corr_delta", []) or []
        if isinstance(row, dict)
    ]
    return {
        "status": payload.get("status", "present"),
        "schema": payload.get("schema"),
        "manifest_path": payload.get("manifest_path"),
        "case_count": payload.get("case_count"),
        "target_gain": payload.get("target_gain"),
        "strongest_non_single_feature": correlations[0].get("feature") if correlations else None,
        "strongest_non_single_corr": correlations[0].get("pearson_with_corr_delta") if correlations else None,
        "strongest_non_single_mse_corr": correlations[0].get("pearson_with_mse_delta") if correlations else None,
        "prefix_phase_velocity_corr": next(
            (
                row.get("pearson_with_corr_delta")
                for row in correlations
                if row.get("feature") == "prefix_phase_velocity_coherence"
            ),
            None,
        ),
        "prefix_spectral_flux_corr": next(
            (
                row.get("pearson_with_corr_delta")
                for row in correlations
                if row.get("feature") == "prefix_spectral_flux_mean"
            ),
            None,
        ),
        "mean_abs_shaped_delta_corr": next(
            (
                row.get("pearson_with_corr_delta")
                for row in correlations
                if row.get("feature") == "mean_abs_shaped_delta"
            ),
            None,
        ),
        "top_correlations": correlations[:20],
        "role_summary": role_summary,
        "top_cases": top_cases[:10],
    }


def summarize_audio_baseline_stratified_diagnostics(payload: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(payload, dict):
        return {"status": "missing", "fixed_bins": []}
    fixed_bins = [
        row
        for row in payload.get("fixed_bins", []) or []
        if isinstance(row, dict)
    ]
    quantile_bins = [
        row
        for row in payload.get("quantile_bins", []) or []
        if isinstance(row, dict)
    ]

    def fixed(name: str) -> dict[str, Any]:
        return next((row for row in fixed_bins if row.get("bin") == name), {})

    def weight_mean(rows: list[dict[str, Any]], key: str) -> float:
        total_cases = sum(int(_as_float(row.get("case_count"))) for row in rows)
        if total_cases <= 0:
            return 0.0
        return sum(_as_float(row.get(key)) * int(_as_float(row.get("case_count"))) for row in rows) / total_cases

    bad = fixed("bad_baseline")
    non_bad_bins = [
        row
        for row in (fixed("weak_baseline"), fixed("moderate_baseline"), fixed("good_baseline"))
        if int(_as_float(row.get("case_count"))) > 0
    ]
    non_bad_case_count = sum(int(_as_float(row.get("case_count"))) for row in non_bad_bins)
    non_bad_mean_corr_delta = weight_mean(non_bad_bins, "mean_corr_delta")
    non_bad_mean_mse_delta = weight_mean(non_bad_bins, "mean_mse_delta")
    non_bad_corr_win_fraction = weight_mean(non_bad_bins, "corr_win_fraction")
    bad_mean_corr_delta = _as_float(bad.get("mean_corr_delta"))
    bad_case_count = int(_as_float(bad.get("case_count")))
    rescue_ratio = bad_mean_corr_delta / max(abs(non_bad_mean_corr_delta), 1.0e-9)
    baseline_rescue_dominated = bool(
        payload.get("status") == "bad_baseline_rescue_dominated"
        or (
            bad_case_count > 0
            and non_bad_case_count > 0
            and bad_mean_corr_delta > 0.01
            and (non_bad_mean_corr_delta <= 0.001 or rescue_ratio >= 10.0)
        )
    )
    promotion_gate_status = "fail_baseline_rescue" if baseline_rescue_dominated else "needs_predeclared_lockbox"

    return {
        "status": payload.get("status", "present"),
        "schema": payload.get("schema"),
        "diagnostics_path": payload.get("diagnostics_path"),
        "case_count": payload.get("case_count"),
        "source_case_count": payload.get("source_case_count"),
        "exclude_single_source": payload.get("exclude_single_source"),
        "bad_baseline": bad,
        "weak_baseline": fixed("weak_baseline"),
        "moderate_baseline": fixed("moderate_baseline"),
        "good_baseline": fixed("good_baseline"),
        "non_bad_case_count": non_bad_case_count,
        "non_bad_mean_corr_delta": non_bad_mean_corr_delta,
        "non_bad_mean_mse_delta": non_bad_mean_mse_delta,
        "non_bad_corr_win_fraction": non_bad_corr_win_fraction,
        "bad_to_non_bad_corr_delta_ratio": rescue_ratio,
        "baseline_rescue_dominated": baseline_rescue_dominated,
        "promotion_gate_status": promotion_gate_status,
        "promotion_gate_pass": False,
        "promotion_gate_requirements": [
            "predeclared case manifest not selected by target or baseline metrics",
            "positive non-bad baseline mean corr delta",
            "positive non-bad baseline median corr delta",
            "non-bad baseline corr win fraction >= 0.60",
            "non-bad baseline mean MSE delta <= 0.0",
            "positive outlier share below 0.50",
        ],
        "fixed_bins": fixed_bins,
        "quantile_bins": quantile_bins,
    }


def summarize_audio_delta_objective_scout(payload: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(payload, dict):
        return {"status": "missing", "rows": []}
    rows: list[dict[str, Any]] = []
    for row in payload.get("ranked_candidates", []) or []:
        if not isinstance(row, dict):
            continue
        objective = row.get("objective") if isinstance(row.get("objective"), dict) else {}
        robustness = objective.get("case_robustness") if isinstance(objective.get("case_robustness"), dict) else {}
        rows.append(
            {
                "candidate": row.get("candidate"),
                "status": objective.get("status"),
                "score": objective.get("score"),
                "primary_corr_delta": objective.get("primary_corr_delta"),
                "primary_mse_delta": objective.get("primary_mse_delta"),
                "primary_corr_win_fraction": objective.get("primary_corr_win_fraction"),
                "primary_mse_win_fraction": objective.get("primary_mse_win_fraction"),
                "primary_vs_gain0_corr": objective.get("primary_vs_gain0_corr"),
                "primary_loop_delta": objective.get("primary_loop_delta"),
                "primary_reentry_delta": objective.get("primary_reentry_delta"),
                "median_corr_delta": robustness.get("median_corr_delta"),
                "min_leave_one_out_corr_delta": robustness.get("min_leave_one_out_corr_delta"),
                "max_positive_corr_gain_share": robustness.get("max_positive_corr_gain_share"),
                "best_corr_delta": objective.get("best_corr_delta"),
                "best_mse_delta": objective.get("best_mse_delta"),
                "probe_status": objective.get("probe_status"),
                "overrides": row.get("overrides", {}),
            }
        )
    best = rows[0] if rows else {}
    candidate_found = any(row.get("status") == "audio_delta_candidate" for row in rows)
    leakage = any(
        bool(_get(row, "objective", "future_target_magnitude_reused"))
        or bool(_get(row, "objective", "target_future_stft_magnitude_accessed"))
        or bool(_get(row, "objective", "future_target_phase_reused"))
        or bool(_get(row, "objective", "target_future_stft_phase_accessed"))
        for row in payload.get("ranked_candidates", []) or []
        if isinstance(row, dict)
    )
    status = payload.get("status", "missing")
    if leakage:
        status = "invalid_future_leakage"
    elif candidate_found:
        status = "candidate_found"
    return {
        "status": status,
        "raw_status": payload.get("status", "missing"),
        "base_config_path": payload.get("base_config_path"),
        "candidate_profile": payload.get("candidate_profile"),
        "candidate_count": payload.get("candidate_count"),
        "case_count": payload.get("case_count"),
        "objective_mode": payload.get("objective_mode"),
        "objective_mask": payload.get("objective_mask"),
        "objective_gain": payload.get("objective_gain"),
        "min_corr_delta": payload.get("min_corr_delta"),
        "min_corr_win_fraction": payload.get("min_corr_win_fraction"),
        "best_candidate": best.get("candidate"),
        "best_candidate_status": best.get("status"),
        "best_primary_corr_delta": best.get("primary_corr_delta"),
        "best_primary_mse_delta": best.get("primary_mse_delta"),
        "best_primary_corr_win_fraction": best.get("primary_corr_win_fraction"),
        "best_primary_mse_win_fraction": best.get("primary_mse_win_fraction"),
        "best_median_corr_delta": best.get("median_corr_delta"),
        "best_min_leave_one_out_corr_delta": best.get("min_leave_one_out_corr_delta"),
        "best_max_positive_corr_gain_share": best.get("max_positive_corr_gain_share"),
        "rows": rows,
    }


def summarize_dense_signature_claim(payload: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(payload, dict):
        return {"status": "missing"}
    verdict = payload.get("verdict") if isinstance(payload.get("verdict"), dict) else {}
    aggregate = payload.get("aggregate") if isinstance(payload.get("aggregate"), dict) else {}
    stability = aggregate.get("stability") if isinstance(aggregate.get("stability"), dict) else {}
    separation = aggregate.get("separation") if isinstance(aggregate.get("separation"), dict) else {}
    collapse = aggregate.get("collapse") if isinstance(aggregate.get("collapse"), dict) else {}
    dense_collapse = collapse.get("dense") if isinstance(collapse.get("dense"), dict) else {}
    metadata_collapse = collapse.get("metadata") if isinstance(collapse.get("metadata"), dict) else {}
    collapse_delta = (
        collapse.get("delta_dense_minus_metadata")
        if isinstance(collapse.get("delta_dense_minus_metadata"), dict)
        else {}
    )
    return {
        "status": verdict.get("status", "missing"),
        "dense_wins": bool(verdict.get("dense_wins")),
        "schema": payload.get("schema"),
        "num_cases": payload.get("num_cases"),
        "clip_seconds": payload.get("clip_seconds"),
        "dense_stability_score": stability.get("dense_score"),
        "metadata_stability_score": stability.get("metadata_score"),
        "stability_delta_dense_minus_metadata": stability.get("delta_dense_minus_metadata"),
        "dense_separation_score": separation.get("dense_score"),
        "metadata_separation_score": separation.get("metadata_score"),
        "separation_delta_dense_minus_metadata": separation.get("delta_dense_minus_metadata"),
        "dense_dominant_cluster_share": dense_collapse.get("dominant_cluster_share"),
        "metadata_dominant_cluster_share": metadata_collapse.get("dominant_cluster_share"),
        "dense_effective_rank": dense_collapse.get("effective_rank"),
        "metadata_effective_rank": metadata_collapse.get("effective_rank"),
        "dense_near_duplicate_pair_share": dense_collapse.get("near_duplicate_pair_share"),
        "metadata_near_duplicate_pair_share": metadata_collapse.get("near_duplicate_pair_share"),
        "collapse_delta_dense_minus_metadata": collapse_delta,
        "criteria": verdict.get("criteria", {}),
        "reasons": verdict.get("reasons", []),
    }


def summarize_dense_signature_failure_modes(payload: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(payload, dict):
        return {"status": "missing", "blocking_failures": [], "recommended_next_losses": []}
    axes = payload.get("required_axes") if isinstance(payload.get("required_axes"), dict) else {}
    deltas = axes.get("deltas") if isinstance(axes.get("deltas"), dict) else {}
    axis_checks = axes.get("axes") if isinstance(axes.get("axes"), dict) else {}
    prefix = payload.get("prefix_diagnostics") if isinstance(payload.get("prefix_diagnostics"), dict) else {}
    operator_tail = (
        payload.get("operator_tail_distinctness")
        if isinstance(payload.get("operator_tail_distinctness"), dict)
        else {}
    )
    return {
        "status": payload.get("claim_status", "present"),
        "schema": payload.get("schema"),
        "promotion_effect": payload.get("promotion_effect"),
        "dense_beats_metadata_on_all_required_axes": payload.get(
            "dense_beats_metadata_on_all_required_axes"
        ),
        "matryoshka_readiness_status": payload.get("matryoshka_readiness_status"),
        "operator_tail_distinctness_status": operator_tail.get("status"),
        "operator_tail_required_next_probe": operator_tail.get("required_next_probe"),
        "blocking_failures": payload.get("blocking_failures", []),
        "stability_delta": deltas.get("stability_delta"),
        "separation_delta": deltas.get("separation_delta"),
        "dominant_cluster_share_delta": deltas.get("dominant_cluster_share_delta"),
        "near_duplicate_pair_share_delta": deltas.get("near_duplicate_pair_share_delta"),
        "effective_rank_delta": deltas.get("effective_rank_delta"),
        "packet_pairwise_distance_delta": deltas.get("packet_pairwise_distance_delta"),
        "all_required_axes_pass": axis_checks.get("all_required_axes_pass"),
        "stability_beats_metadata": axis_checks.get("stability_beats_metadata"),
        "separation_beats_metadata": axis_checks.get("separation_beats_metadata"),
        "anti_collapse_not_worse": axis_checks.get("anti_collapse_not_worse"),
        "prefix_status": prefix.get("status"),
        "prefix_monotonicity_score": prefix.get("prefix_monotonicity_score"),
        "prefix_spread_mean": prefix.get("prefix_spread_mean"),
        "early_minus_full_mean": prefix.get("early_minus_full_mean"),
        "transform_family_stability": payload.get("transform_family_stability", {}),
        "recommended_next_losses": payload.get("recommended_next_losses", []),
    }


def summarize_relational_signature_contract_audit(payload: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(payload, dict):
        return {"status": "missing", "required_next_implementation": []}
    prefix_contract = (
        payload.get("prefix_contract")
        if isinstance(payload.get("prefix_contract"), dict)
        else {}
    )
    source_checks = (
        payload.get("source_checks")
        if isinstance(payload.get("source_checks"), dict)
        else {}
    )
    return {
        "status": payload.get("claim_status", "present"),
        "schema": payload.get("schema"),
        "schema_contract_passed": payload.get("schema_contract_passed"),
        "learned_body_status": payload.get("learned_body_status"),
        "signature_dim": prefix_contract.get("signature_dim"),
        "prefix_dims": prefix_contract.get("prefix_dims"),
        "operator_seed_dim": prefix_contract.get("operator_seed_dim"),
        "missing_expected_fields": payload.get("missing_expected_fields", []),
        "defines_nn_module": source_checks.get("defines_nn_module"),
        "learned_head_tokens_present": source_checks.get("learned_head_tokens_present"),
        "deterministic_assemblers": source_checks.get("deterministic_assemblers", []),
        "has_dense_assembly_from_factor_pack": source_checks.get(
            "has_dense_assembly_from_factor_pack"
        ),
        "interpretation": payload.get("interpretation"),
        "required_next_implementation": payload.get("required_next_implementation", []),
    }


def summarize_relational_signature_learning_contract(payload: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(payload, dict):
        return {"status": "missing", "checks": {}}
    contract = payload.get("contract") if isinstance(payload.get("contract"), dict) else {}
    return {
        "status": payload.get("status", "present"),
        "schema": payload.get("schema"),
        "claim_effect": payload.get("claim_effect"),
        "runtime_effect": payload.get("runtime_effect"),
        "device": payload.get("device"),
        "batch": payload.get("batch"),
        "hidden_dim": payload.get("hidden_dim"),
        "train_steps": payload.get("train_steps"),
        "learning_rate": payload.get("learning_rate"),
        "initial_total_loss": payload.get("initial_total_loss"),
        "final_total_loss": payload.get("final_total_loss"),
        "loss_reduction": payload.get("loss_reduction"),
        "gradient_norm": payload.get("gradient_norm"),
        "checks": payload.get("checks", {}),
        "h_shape": contract.get("h_shape"),
        "q_profile_shape": contract.get("q_profile_shape"),
        "branch_profile_shape": contract.get("branch_profile_shape"),
        "law_signature_shape": contract.get("law_signature_shape"),
        "operator_seed_shape": contract.get("operator_seed_shape"),
        "max_h_unit_norm_error": contract.get("max_h_unit_norm_error"),
        "max_law_unit_norm_error": contract.get("max_law_unit_norm_error"),
        "max_operator_unit_norm_error": contract.get("max_operator_unit_norm_error"),
        "max_q_sum_error": contract.get("max_q_sum_error"),
        "losses": contract.get("losses", {}),
        "interpretation": payload.get("interpretation"),
    }


def summarize_learned_signature_scout(payload: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(payload, dict):
        return {"status": "missing", "reasons": []}
    verdict = payload.get("verdict") if isinstance(payload.get("verdict"), dict) else {}
    aggregate = payload.get("aggregate") if isinstance(payload.get("aggregate"), dict) else {}
    stability = aggregate.get("stability") if isinstance(aggregate.get("stability"), dict) else {}
    separation = aggregate.get("separation") if isinstance(aggregate.get("separation"), dict) else {}
    collapse = aggregate.get("collapse") if isinstance(aggregate.get("collapse"), dict) else {}
    learned_collapse = collapse.get("learned") if isinstance(collapse.get("learned"), dict) else {}
    collapse_delta = (
        collapse.get("delta_learned_minus_metadata")
        if isinstance(collapse.get("delta_learned_minus_metadata"), dict)
        else {}
    )
    operator_tail = aggregate.get("operator_tail") if isinstance(aggregate.get("operator_tail"), dict) else {}
    train = payload.get("train_summary") if isinstance(payload.get("train_summary"), dict) else {}
    return {
        "status": verdict.get("status", payload.get("status", "present")),
        "schema": payload.get("schema"),
        "promotion_effect": verdict.get("promotion_effect"),
        "split_mode": verdict.get("split_mode", payload.get("split_mode")),
        "num_cases": payload.get("num_cases"),
        "num_transforms": payload.get("num_transforms"),
        "train_packet_count": train.get("train_packet_count"),
        "train_initial_batch_loss": train.get("initial_batch_loss"),
        "train_final_full_loss": train.get("final_full_loss"),
        "train_loss_reduction": train.get("loss_reduction"),
        "learned_stability_score": stability.get("learned_score"),
        "deterministic_stability_score": stability.get("deterministic_score"),
        "metadata_stability_score": stability.get("metadata_score"),
        "stability_delta_learned_minus_metadata": stability.get("delta_learned_minus_metadata"),
        "stability_delta_learned_minus_deterministic": stability.get("delta_learned_minus_deterministic"),
        "learned_separation_score": separation.get("learned_score"),
        "deterministic_separation_score": separation.get("deterministic_score"),
        "metadata_separation_score": separation.get("metadata_score"),
        "separation_delta_learned_minus_metadata": separation.get("delta_learned_minus_metadata"),
        "separation_delta_learned_minus_deterministic": separation.get("delta_learned_minus_deterministic"),
        "learned_dominant_cluster_share": learned_collapse.get("dominant_cluster_share"),
        "learned_near_duplicate_pair_share": learned_collapse.get("near_duplicate_pair_share"),
        "collapse_delta_learned_minus_metadata": collapse_delta,
        "operator_tail_effective_rank": operator_tail.get("effective_rank"),
        "operator_tail_effective_cluster_count": operator_tail.get("effective_cluster_count"),
        "operator_tail_dominant_cluster_share": operator_tail.get("dominant_cluster_share"),
        "operator_tail_near_duplicate_pair_share": operator_tail.get("near_duplicate_pair_share"),
        "criteria": verdict.get("criteria", {}),
        "reasons": verdict.get("reasons", []),
    }


def summarize_learned_signature_scout_compare(payload: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(payload, dict):
        return {"status": "missing", "rows": []}
    rows = [row for row in payload.get("rows", []) or [] if isinstance(row, dict)]
    pareto = [row for row in payload.get("pareto_rows", []) or [] if isinstance(row, dict)]
    candidates = [row for row in payload.get("candidate_rows", []) or [] if isinstance(row, dict)]
    seed_stability = payload.get("seed_stability") if isinstance(payload.get("seed_stability"), dict) else {}
    heldout_seed_stability = (
        payload.get("heldout_seed_stability")
        if isinstance(payload.get("heldout_seed_stability"), dict)
        else {}
    )
    return {
        "status": payload.get("status", "present"),
        "schema": payload.get("schema"),
        "promotion_effect": payload.get("promotion_effect"),
        "row_count": payload.get("row_count"),
        "candidate_count": payload.get("candidate_count"),
        "two_axis_candidate_count": payload.get("two_axis_candidate_count"),
        "stability_anticollapse_candidate_count": payload.get("stability_anticollapse_candidate_count"),
        "heldout_row_count": payload.get("heldout_row_count"),
        "heldout_candidate_count": payload.get("heldout_candidate_count"),
        "heldout_two_axis_candidate_count": payload.get("heldout_two_axis_candidate_count"),
        "heldout_stability_anticollapse_candidate_count": payload.get("heldout_stability_anticollapse_candidate_count"),
        "pareto_count": payload.get("pareto_count"),
        "best_separation_profile": payload.get("best_separation_profile"),
        "best_stability_profile": payload.get("best_stability_profile"),
        "best_separation_delta": payload.get("best_separation_delta"),
        "best_stability_delta": payload.get("best_stability_delta"),
        "seed_stability": seed_stability,
        "heldout_seed_stability": heldout_seed_stability,
        "interpretation": payload.get("interpretation"),
        "pareto_rows": pareto,
        "candidate_rows": candidates,
        "two_axis_rows": [row for row in payload.get("two_axis_rows", []) or [] if isinstance(row, dict)],
        "stability_anticollapse_rows": [
            row for row in payload.get("stability_anticollapse_rows", []) or [] if isinstance(row, dict)
        ],
        "rows": rows,
    }


def summarize_learned_signature_split_suite(payload: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(payload, dict):
        return {"status": "missing", "rows": []}
    rollup = payload.get("rollup") if isinstance(payload.get("rollup"), dict) else {}
    rows = [row for row in payload.get("rows", []) or [] if isinstance(row, dict)]
    failed_rows = [row for row in payload.get("failed_rows", []) or [] if isinstance(row, dict)]
    return {
        "status": rollup.get("status", payload.get("status", "present")),
        "schema": payload.get("schema"),
        "profile": payload.get("profile"),
        "split_set": payload.get("split_set"),
        "split_count": rollup.get("split_count"),
        "completed_count": rollup.get("completed_count"),
        "failed_count": rollup.get("failed_count"),
        "candidate_count": rollup.get("candidate_count"),
        "candidate_fraction": rollup.get("candidate_fraction"),
        "two_axis_candidate_count": rollup.get("two_axis_candidate_count"),
        "stability_anticollapse_candidate_count": rollup.get("stability_anticollapse_candidate_count"),
        "heldout_row_count": rollup.get("heldout_row_count"),
        "heldout_candidate_count": rollup.get("heldout_candidate_count"),
        "heldout_two_axis_candidate_count": rollup.get("heldout_two_axis_candidate_count"),
        "heldout_stability_anticollapse_candidate_count": rollup.get(
            "heldout_stability_anticollapse_candidate_count"
        ),
        "seed_split_count": rollup.get("seed_split_count"),
        "seed_candidate_count": rollup.get("seed_candidate_count"),
        "seed_candidate_fraction": rollup.get("seed_candidate_fraction"),
        "partition_split_count": rollup.get("partition_split_count"),
        "partition_candidate_count": rollup.get("partition_candidate_count"),
        "partition_candidate_fraction": rollup.get("partition_candidate_fraction"),
        "mean_stability_delta_learned_minus_metadata": rollup.get(
            "mean_stability_delta_learned_minus_metadata"
        ),
        "mean_separation_delta_learned_minus_metadata": rollup.get(
            "mean_separation_delta_learned_minus_metadata"
        ),
        "mean_collapse_effective_rank_delta": rollup.get("mean_collapse_effective_rank_delta"),
        "hard_stability_cases": [
            {
                "case": row.get("case"),
                "hit_count": row.get("hit_count"),
                "min_stability_delta_learned_minus_metadata": row.get(
                    "min_stability_delta_learned_minus_metadata"
                ),
                "split_ids": row.get("split_ids", []),
            }
            for row in (rollup.get("hard_stability_cases", []) or [])
            if isinstance(row, dict)
        ][:10],
        "rows": [
            {
                "split_id": row.get("split_id"),
                "kind": row.get("kind"),
                "status": row.get("status"),
                "candidate": row.get("candidate"),
                "two_axis_candidate": row.get("two_axis_candidate"),
                "stability_anticollapse_candidate": row.get("stability_anticollapse_candidate"),
                "heldout_evidence": row.get("heldout_evidence"),
                "heldout_candidate": row.get("heldout_candidate"),
                "seed": row.get("seed"),
                "case_offset": row.get("case_offset"),
                "case_shuffle_seed": row.get("case_shuffle_seed"),
                "stability_delta_learned_minus_metadata": row.get(
                    "stability_delta_learned_minus_metadata"
                ),
                "separation_delta_learned_minus_metadata": row.get(
                    "separation_delta_learned_minus_metadata"
                ),
                "collapse_effective_rank_delta": row.get("collapse_effective_rank_delta"),
                "collapse_mean_pairwise_distance_delta": row.get(
                    "collapse_mean_pairwise_distance_delta"
                ),
                "worst_stability_cases": row.get("worst_stability_cases", []),
            }
            for row in rows
        ],
        "failed_rows": failed_rows,
    }


def summarize_learned_signature_split_suite_compare(payload: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(payload, dict):
        return {"status": "missing", "rows": []}
    rows = [row for row in payload.get("rows", []) or [] if isinstance(row, dict)]
    return {
        "status": payload.get("status", "present"),
        "schema": payload.get("schema"),
        "promotion_effect": payload.get("promotion_effect"),
        "row_count": payload.get("row_count"),
        "robust_candidate_count": payload.get("robust_candidate_count"),
        "fragile_candidate_count": payload.get("fragile_candidate_count"),
        "partial_failure_count": payload.get("partial_failure_count"),
        "redacted_suite_count": payload.get("redacted_suite_count"),
        "redacted_candidate_count": payload.get("redacted_candidate_count"),
        "best_profile": payload.get("best_profile"),
        "best_status": payload.get("best_status"),
        "best_candidate_fraction": payload.get("best_candidate_fraction"),
        "best_seed_candidate_fraction": payload.get("best_seed_candidate_fraction"),
        "best_partition_candidate_fraction": payload.get("best_partition_candidate_fraction"),
        "best_mean_stability_delta": payload.get("best_mean_stability_delta"),
        "best_stability_profile": payload.get("best_stability_profile"),
        "best_stability_delta": payload.get("best_stability_delta"),
        "best_partition_profile": payload.get("best_partition_profile"),
        "best_partition_fraction": payload.get("best_partition_fraction"),
        "rows": rows,
    }


def summarize_learned_signature_case_failures(payload: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(payload, dict):
        return {"status": "missing", "worst_cases": []}
    worst_cases = [row for row in payload.get("worst_cases", []) or [] if isinstance(row, dict)]
    return {
        "status": payload.get("status", "present"),
        "schema": payload.get("schema"),
        "heldout_case_row_count": payload.get("heldout_case_row_count"),
        "case_count": payload.get("case_count"),
        "worst_cases": [
            {
                "case": row.get("case"),
                "row_count": row.get("row_count"),
                "stability_pass_fraction": row.get("stability_pass_fraction"),
                "mean_stability_delta": row.get("mean_stability_delta"),
                "worst_stability_delta": row.get("worst_stability_delta"),
                "worst_profile": row.get("worst_profile"),
            }
            for row in worst_cases[:10]
        ],
    }


def summarize_signature_operator_tail_probe_compare(payload: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(payload, dict):
        return {"status": "missing", "rows": []}
    rows = [row for row in payload.get("rows", []) or [] if isinstance(row, dict)]
    return {
        "status": payload.get("status", "present"),
        "schema": payload.get("schema"),
        "promotion_effect": payload.get("promotion_effect"),
        "row_count": payload.get("row_count"),
        "learned_win_count": payload.get("learned_win_count"),
        "learned_win_fraction": payload.get("learned_win_fraction"),
        "by_target": payload.get("by_target") if isinstance(payload.get("by_target"), dict) else {},
        "rows": rows,
        "interpretation": payload.get("interpretation"),
    }


def summarize_signature_future_law_probe(payload: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(payload, dict):
        return {
            "status": "missing",
            "evidence": "missing",
            "artifact_present": False,
            "target_rows": [],
        }
    verdict = payload.get("verdict") if isinstance(payload.get("verdict"), dict) else {}
    target_rows = [row for row in verdict.get("target_rows", []) or [] if isinstance(row, dict)]
    target_count = verdict.get("target_count", len(target_rows))
    learned_win_count = verdict.get("learned_win_count")
    learned_win_fraction = None
    if target_count:
        learned_win_fraction = _as_float(learned_win_count) / _as_float(target_count, 1.0)
    return {
        "status": verdict.get("status", payload.get("status", "present")),
        "evidence": "present",
        "artifact_present": True,
        "schema": payload.get("schema"),
        "promotion_effect": verdict.get("promotion_effect", payload.get("promotion_effect")),
        "num_cases": payload.get("num_cases"),
        "heldout_cases": payload.get("heldout_cases"),
          "num_transforms": payload.get("num_transforms"),
          "clip_seconds": payload.get("clip_seconds"),
          "ridge_alpha_selection_policy": payload.get("ridge_alpha_selection_policy"),
          "leakage_guard": payload.get("leakage_guard"),
          "train_example_count": payload.get("train_example_count"),
          "heldout_example_count": payload.get("heldout_example_count"),
        "target_count": target_count,
        "learned_win_count": learned_win_count,
        "learned_win_fraction": learned_win_fraction,
        "learned_beats_non_law_count": verdict.get("learned_beats_non_law_count"),
        "full_factor_win_count": verdict.get("full_factor_win_count"),
        "feature_view_count": len(payload.get("feature_views", {}) or {}),
        "targets": payload.get("targets", []),
        "target_rows": [
            {
                "target": row.get("target"),
                "best_view": row.get("best_view"),
                "best_heldout_mse": row.get("best_heldout_mse"),
                "best_learned_view": row.get("best_learned_view"),
                "best_learned_heldout_mse": row.get("best_learned_heldout_mse"),
                "explicit_non_law_heldout_mse": row.get("explicit_non_law_heldout_mse"),
                "explicit_full_factor_heldout_mse": row.get("explicit_full_factor_heldout_mse"),
                "learned_wins": row.get("learned_wins"),
                "learned_beats_non_law": row.get("learned_beats_non_law"),
                "full_factor_wins": row.get("full_factor_wins"),
            }
            for row in target_rows
        ],
        "interpretation": payload.get("interpretation"),
    }


def summarize_signature_tail_incremental_usefulness(payload: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(payload, dict):
        return {
            "status": "missing",
            "evidence": "missing",
            "artifact_present": False,
            "target_rows": [],
        }
    summary = (
        payload.get("incremental_summary")
        if isinstance(payload.get("incremental_summary"), dict)
        else {}
    )
    target_rows = [row for row in summary.get("target_rows", []) or [] if isinstance(row, dict)]
    target_count = summary.get("target_count", len(target_rows))
    tail_win_count = summary.get("tail_win_count")
    tail_win_fraction = None
    if target_count:
        tail_win_fraction = _as_float(tail_win_count) / _as_float(target_count, 1.0)
    return {
        "status": summary.get("status", payload.get("status", "present")),
        "evidence": "present",
        "artifact_present": True,
        "schema": payload.get("schema"),
        "promotion_effect": payload.get("promotion_effect"),
        "num_cases": payload.get("num_cases"),
        "heldout_cases": payload.get("heldout_cases"),
          "num_transforms": payload.get("num_transforms"),
          "clip_seconds": payload.get("clip_seconds"),
          "ridge_alpha_selection_policy": payload.get("ridge_alpha_selection_policy"),
          "leakage_guard": payload.get("leakage_guard"),
          "train_packet_count": payload.get("train_packet_count"),
          "heldout_packet_count": payload.get("heldout_packet_count"),
        "target_count": target_count,
        "tail_win_count": tail_win_count,
        "tail_win_fraction": tail_win_fraction,
        "tail_beats_prefix_count": summary.get("tail_beats_prefix_count"),
        "tail_beats_permuted_count": summary.get("tail_beats_permuted_count"),
        "tail_beats_cross_case_count": summary.get("tail_beats_cross_case_count"),
          "tail_beats_random_count": summary.get("tail_beats_random_count"),
          "shuffle_control_win_count": summary.get("shuffle_control_win_count"),
          "prefix_control_win_count": summary.get("prefix_control_win_count"),
          "cross_case_control_win_count": summary.get("cross_case_control_win_count"),
          "random_control_win_count": summary.get("random_control_win_count"),
          "control_win_count": summary.get("control_win_count"),
          "feature_view_count": len(payload.get("feature_views", {}) or {}),
        "targets": payload.get("targets", []),
        "target_rows": [
            {
                "target": row.get("target"),
                "metadata_heldout_mse": row.get("metadata_heldout_mse"),
                "tail_heldout_mse": row.get("tail_heldout_mse"),
                "prefix_heldout_mse": row.get("prefix_heldout_mse"),
                "permuted_tail_heldout_mse": row.get("permuted_tail_heldout_mse"),
                "cross_case_tail_heldout_mse": row.get("cross_case_tail_heldout_mse"),
                "random_tail_heldout_mse": row.get("random_tail_heldout_mse"),
                "full_h_heldout_mse": row.get("full_h_heldout_mse"),
                "delta_heldout_mse_vs_metadata": row.get("delta_heldout_mse_vs_metadata"),
                "delta_heldout_r2_vs_metadata": row.get("delta_heldout_r2_vs_metadata"),
                "tail_minus_prefix_delta_mse": row.get("tail_minus_prefix_delta_mse"),
                "tail_minus_permuted_delta_mse": row.get("tail_minus_permuted_delta_mse"),
                "tail_minus_cross_case_delta_mse": row.get("tail_minus_cross_case_delta_mse"),
                "tail_minus_random_delta_mse": row.get("tail_minus_random_delta_mse"),
                "tail_win": row.get("tail_win"),
                "tail_beats_prefix": row.get("tail_beats_prefix"),
                "tail_beats_permuted": row.get("tail_beats_permuted"),
                "tail_beats_cross_case": row.get("tail_beats_cross_case"),
                  "tail_beats_random": row.get("tail_beats_random"),
                  "shuffle_control_win": row.get("shuffle_control_win"),
                  "prefix_control_win": row.get("prefix_control_win"),
                  "cross_case_control_win": row.get("cross_case_control_win"),
                  "random_control_win": row.get("random_control_win"),
              }
            for row in target_rows
        ],
        "interpretation": payload.get("interpretation"),
    }


def dense_relational_signature_status(evidence: dict[str, Any]) -> str:
    compare_suite_status = evidence.get("learned_signature_split_suite_compare", {}).get("status")
    if compare_suite_status == "robust_split_suite_candidate_found":
        return "learned_dense_split_suite_candidate_needs_operator_usefulness"
    if compare_suite_status == "fragile_split_suite_candidate_only":
        return "learned_dense_split_suite_fragile_seed_partition_blocked"
    if compare_suite_status in {"no_split_suite_candidate", "split_suite_compare_partial_failures"}:
        return "learned_dense_split_suite_no_robust_candidate"
    split_status = evidence.get("learned_signature_split_suite", {}).get("status")
    if split_status in {"split_suite_all_passed", "split_suite_majority_passed"}:
        return "learned_dense_split_suite_candidate_needs_operator_usefulness"
    if split_status == "split_suite_fragile_candidate":
        return "learned_dense_split_suite_fragile_seed_partition_blocked"
    if split_status in {
        "split_suite_complementary_near_candidates",
        "split_suite_two_axis_collapse_blocked",
        "split_suite_no_candidate",
    }:
        return "learned_dense_split_suite_no_robust_candidate"
    if split_status and split_status not in {"missing", "split_suite_failed_to_run"}:
        return "learned_dense_split_suite_incomplete"
    compare_status = evidence["learned_signature_scout_compare"].get("status")
    if compare_status == "heldout_candidate_found_needs_seed":
        return "learned_dense_narrow_heldout_candidate_seed_partition_blocked"
    if compare_status == "in_sample_candidate_holdout_blocked":
        return "learned_dense_in_sample_candidate_holdout_blocked"
    if compare_status in {"candidate_tradeoff_found_needs_holdout"}:
        return "learned_dense_candidate_needs_holdout_seed_check"
    if evidence["dense_signature_failure_modes"].get("status") != "missing":
        return str(evidence["dense_signature_failure_modes"]["status"])
    return str(evidence["dense_signature_claim"]["status"])


def summarize_semantic_projector_contract(payload: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(payload, dict):
        return {"status": "missing"}
    projection_path = payload.get("projection_path") if isinstance(payload.get("projection_path"), dict) else {}
    checks = payload.get("checks") if isinstance(payload.get("checks"), dict) else {}
    schema_checks = checks.get("schema") if isinstance(checks.get("schema"), dict) else {}
    artifact_checks = checks.get("artifacts") if isinstance(checks.get("artifacts"), dict) else {}
    projector = payload.get("semantic_projector") if isinstance(payload.get("semantic_projector"), dict) else {}
    prompt_outputs = payload.get("outputs") if isinstance(payload.get("outputs"), dict) else {}
    diversity = checks.get("pairwise_diversity") if isinstance(checks.get("pairwise_diversity"), list) else []
    return {
        "status": payload.get("status", "missing"),
        "schema": payload.get("schema"),
        "import_ok": projector.get("import_ok"),
        "projection_path": projection_path.get("name"),
        "projection_detail": projection_path.get("detail"),
        "control_schema_exact": bool(
            schema_checks.get("control_dim_is_5")
            and schema_checks.get("control_names_exact_order")
            and schema_checks.get("control_names_exact_set")
        ),
        "artifact_readback_ok": artifact_checks.get("json_readback_ok"),
        "prompt_family_count": len(prompt_outputs),
        "pairwise_diversity_count": len(diversity),
        "all_pairs_diverse": bool(diversity) and all(bool(row.get("diverse")) for row in diversity if isinstance(row, dict)),
    }


def summarize_substrate(payload: dict[str, Any] | None) -> dict[str, Any]:
    variants = _variant_map(payload)
    rows: list[dict[str, Any]] = []
    for name, row in variants.items():
        m = _metrics(row)
        rows.append(
            {
                "variant": name,
                "status": row.get("status"),
                "phase_only_branch": _get(m, "branch", "phase_only_real_branch_fraction"),
                "parent_branch": _get(m, "branch", "mean_parent_real_branch_fraction"),
                "child_branch": _get(m, "branch", "mean_child_real_branch_fraction"),
                "major_gain": _get(m, "major_residue_loss", "mean_major_gain"),
                "mean_loss": _get(m, "major_residue_loss", "mean_loss"),
            }
        )
    if not rows:
        return {"status": "pending", "rows": rows}
    baseline = next((row for row in rows if row["variant"] == "lattice_baseline"), None)
    pointwise = next((row for row in rows if row["variant"] == "pointwise_no_neighborhood"), None)
    shuffled = next((row for row in rows if row["variant"] == "shuffled_neighborhood"), None)
    global_mean = next((row for row in rows if row["variant"] == "flat_global_mean"), None)
    status = "evaluated_needs_review"
    if baseline and pointwise:
        baseline_loss = _as_float(baseline.get("mean_loss"))
        pointwise_loss = _as_float(pointwise.get("mean_loss"))
        baseline_child = _as_float(baseline.get("child_branch"))
        pointwise_child = _as_float(pointwise.get("child_branch"))
        if pointwise_child >= baseline_child and pointwise_loss <= baseline_loss:
            status = "weak_regularizer_not_primary_causal"
    return {
        "status": status,
        "synthetic_only": bool(payload.get("synthetic_only")) if isinstance(payload, dict) else None,
        "baseline_vs_pointwise_loss_delta": (
            _as_float(pointwise.get("mean_loss")) - _as_float(baseline.get("mean_loss"))
            if baseline and pointwise
            else None
        ),
        "shuffled_loss_delta": (
            _as_float(shuffled.get("mean_loss")) - _as_float(baseline.get("mean_loss"))
            if baseline and shuffled
            else None
        ),
        "global_mean_loss_delta": (
            _as_float(global_mean.get("mean_loss")) - _as_float(baseline.get("mean_loss"))
            if baseline and global_mean
            else None
        ),
        "rows": rows,
    }


def summarize_claim_isolation(payload: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(payload, dict):
        return {"status": "missing", "tracks": []}
    tracks: list[dict[str, Any]] = []
    for track in payload.get("tracks", []) or []:
        if not isinstance(track, dict):
            continue
        rows: list[dict[str, Any]] = []
        for row in track.get("variants", []) or []:
            if not isinstance(row, dict):
                continue
            metrics = row.get("metrics") if isinstance(row.get("metrics"), dict) else {}
            top = metrics.get("top") if isinstance(metrics.get("top"), dict) else {}
            by_source = metrics.get("by_source") if isinstance(metrics.get("by_source"), dict) else {}
            naked = by_source.get("naked_rafa") if isinstance(by_source.get("naked_rafa"), dict) else {}
            delta = row.get("delta_vs_track_baseline") if isinstance(row.get("delta_vs_track_baseline"), dict) else {}
            top_delta = delta.get("top") if isinstance(delta.get("top"), dict) else {}
            rows.append(
                {
                    "variant": row.get("variant"),
                    "status": row.get("status"),
                    "parent_branch": top.get("mean_parent_real_branch_fraction"),
                    "child_branch": top.get("mean_child_real_branch_fraction"),
                    "phase_only_branch": top.get("phase_only_real_branch_fraction"),
                    "naked_phase_only_branch": naked.get("phase_only_real_branch_fraction"),
                    "gate_mass": top.get("mean_child_writeback_gate_mass"),
                    "phase_delta": top.get("mean_child_phase_writeback_delta_mass"),
                    "support_writeback": top.get("mean_child_support_writeback_mass"),
                    "law_families": top.get("mean_num_law_families"),
                    "major_gain": top.get("mean_major_gain"),
                    "loss": top.get("mean_loss"),
                    "delta_loss": top_delta.get("mean_loss"),
                }
            )
        tracks.append(
            {
                "track": track.get("track"),
                "baseline_variant": track.get("baseline_variant"),
                "rows": rows,
            }
        )
    return {
        "status": "evaluated",
        "time_steps": payload.get("time_steps"),
        "seed_plan": payload.get("seed_plan"),
        "tracks": tracks,
    }


def summarize_arc_q_control_claim(payload: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(payload, dict):
        return {"status": "missing"}
    return {
        "status": payload.get("status"),
        "schema": payload.get("schema"),
        "time_steps": payload.get("time_steps"),
        "q_variant_count": payload.get("q_variant_count"),
        "no_q_relation_arc_gate": payload.get("no_q_relation_arc_gate"),
        "ramanujan_qtrace_phase_parent_spread": payload.get("ramanujan_qtrace_phase_parent_spread"),
        "rows": payload.get("rows", []),
    }


def assemble(artifacts: dict[str, Path]) -> dict[str, Any]:
    artifacts = _resolve_artifacts(artifacts)
    loaded = {key: _load_json(path) for key, path in artifacts.items()}
    evidence = {
        "child_writeback": summarize_child_writeback(loaded["child_writeback"]),
        "nested_branch_ontology": summarize_nested(loaded["nested"]),
        "phase_gauge": summarize_phase_gauge(loaded["phase_gauge"]),
        "unit_phasor_contract": summarize_unit_phasor_contract(
            loaded["unit_phasor_contract"],
            loaded.get("unit_phasor_prefix_fail"),
        ),
        "unit_phasor_broad_direct_export": summarize_unit_phasor_contract(
            loaded.get("unit_phasor_broad_direct_export"),
        ),
        "unit_phasor_broad_audio_continuation": summarize_unit_phasor_contract(
            loaded.get("unit_phasor_broad_audio_continuation"),
        ),
        "packet_alias_contract": summarize_packet_alias_contract(loaded.get("packet_alias_contract")),
        "q_basis": summarize_q_basis(loaded["q_basis"]),
        "audio_continuation": summarize_audio(
            {
                "prefix_hold": loaded["audio_prefix_hold"],
                "flat": loaded["audio_flat"],
            }
        ),
        "audio_continuation_copyphase": summarize_audio(
            {
                "copyphase_prefix_hold": loaded.get("audio_copyphase_prefix_hold"),
                "copyphase_flat": loaded.get("audio_copyphase_flat"),
            }
        ),
        "broad18_benchmark": summarize_benchmark(loaded.get("broad18_benchmark")),
        "audio_method_compare": summarize_audio_method_compare(loaded.get("audio_method_compare")),
        "audio_method_compare_copyphase": summarize_audio_method_compare(loaded.get("audio_method_compare_copyphase")),
        "audio_phase_influence": summarize_audio_phase_influence(loaded.get("audio_phase_influence")),
        "audio_phase_seed": summarize_audio_phase_seed(loaded.get("audio_phase_seed")),
        "audio_phase_seed_weighted_fit": summarize_audio_phase_seed(loaded.get("audio_phase_seed_weighted_fit")),
        "audio_circle_delta_probe": summarize_audio_circle_delta_probe(loaded.get("audio_circle_delta_probe")),
        "audio_delta_objective_scout": summarize_audio_delta_objective_scout(
            loaded.get("audio_delta_objective_scout")
        ),
        "audio_delta_objective_scout_relsig": summarize_audio_delta_objective_scout(
            loaded.get("audio_delta_objective_scout_relsig")
        ),
        "audio_delta_mechanism_probe": summarize_audio_delta_mechanism_probe(
            loaded.get("audio_delta_mechanism_probe")
        ),
        "audio_delta_mechanism_lockbox": summarize_audio_delta_mechanism_probe(
            loaded.get("audio_delta_mechanism_lockbox")
        ),
        "audio_mechanism_family_sensitivity": summarize_audio_mechanism_family_sensitivity(
            loaded.get("audio_mechanism_family_sensitivity")
        ),
        "audio_electric_motor_holdout": summarize_audio_electric_motor_holdout(
            loaded.get("audio_electric_motor_holdout")
        ),
        "audio_steady_phenotype_manifest": summarize_audio_steady_phenotype_manifest(
            loaded.get("audio_steady_phenotype_manifest")
        ),
        "audio_phase_phenotype_diagnostics": summarize_audio_phase_phenotype_diagnostics(
            loaded.get("audio_phase_phenotype_diagnostics")
        ),
        "audio_baseline_stratified_diagnostics": summarize_audio_baseline_stratified_diagnostics(
            loaded.get("audio_baseline_stratified_diagnostics")
        ),
        "audio_predeclared_lockbox_manifest": summarize_audio_predeclared_lockbox_manifest(
            loaded.get("audio_predeclared_lockbox_manifest")
        ),
        "audio_predeclared_lockbox_probe": summarize_audio_delta_mechanism_probe(
            loaded.get("audio_predeclared_lockbox_probe")
        ),
        "audio_predeclared_lockbox_compare": summarize_audio_predeclared_lockbox_compare(
            loaded.get("audio_predeclared_lockbox_compare")
        ),
        "audio_predeclared_lockbox_compare_no_single_source": summarize_audio_predeclared_lockbox_compare(
            loaded.get("audio_predeclared_lockbox_compare_no_single_source")
        ),
        "audio_phase_law_scout": summarize_audio_delta_mechanism_probe(
            loaded.get("audio_phase_law_scout")
        ),
        "audio_phase_law_scout_compare_no_single_source": summarize_audio_predeclared_lockbox_compare(
            loaded.get("audio_phase_law_scout_compare_no_single_source")
        ),
        "audio_internal_phase_law_raw_compare_no_single_source": summarize_audio_predeclared_lockbox_compare(
            loaded.get("audio_internal_phase_law_raw_compare_no_single_source")
        ),
        "audio_internal_phase_law_variant_compare_no_single_source": summarize_internal_phase_law_variant_compare(
            loaded.get("audio_internal_phase_law_variant_compare_no_single_source")
        ),
        "audio_internal_phase_law_objective_score": summarize_internal_phase_law_objective_score(
            loaded.get("audio_internal_phase_law_objective_score")
        ),
        "audio_internal_phase_law_focused_raw_compare_no_single_source": summarize_audio_predeclared_lockbox_compare(
            loaded.get("audio_internal_phase_law_focused_raw_compare_no_single_source")
        ),
        "audio_internal_phase_law_focused_variant_compare_no_single_source": summarize_internal_phase_law_variant_compare(
            loaded.get("audio_internal_phase_law_focused_variant_compare_no_single_source")
        ),
        "audio_internal_phase_law_focused_objective_score": summarize_internal_phase_law_objective_score(
            loaded.get("audio_internal_phase_law_focused_objective_score")
        ),
        "audio_internal_phase_law_focused_joint_rows": summarize_internal_phase_law_joint_rows(
            loaded.get("audio_internal_phase_law_focused_joint_rows")
        ),
        "audio_internal_phase_law_wide_raw_compare_no_single_source": summarize_audio_predeclared_lockbox_compare(
            loaded.get("audio_internal_phase_law_wide_raw_compare_no_single_source")
        ),
        "audio_internal_phase_law_wide_variant_compare_no_single_source": summarize_internal_phase_law_variant_compare(
            loaded.get("audio_internal_phase_law_wide_variant_compare_no_single_source")
        ),
        "audio_internal_phase_law_wide_objective_score": summarize_internal_phase_law_objective_score(
            loaded.get("audio_internal_phase_law_wide_objective_score")
        ),
        "audio_internal_phase_law_wide_joint_rows": summarize_internal_phase_law_joint_rows(
            loaded.get("audio_internal_phase_law_wide_joint_rows")
        ),
        "audio_internal_phase_law_v2_guard_raw_compare_no_single_source": summarize_audio_predeclared_lockbox_compare(
            loaded.get("audio_internal_phase_law_v2_guard_raw_compare_no_single_source")
        ),
        "audio_internal_phase_law_v2_guard_variant_compare_no_single_source": summarize_internal_phase_law_variant_compare(
            loaded.get("audio_internal_phase_law_v2_guard_variant_compare_no_single_source")
        ),
        "audio_internal_phase_law_v2_guard_objective_score": summarize_internal_phase_law_objective_score(
            loaded.get("audio_internal_phase_law_v2_guard_objective_score")
        ),
        "audio_internal_phase_law_v2_guard_joint_rows": summarize_internal_phase_law_joint_rows(
            loaded.get("audio_internal_phase_law_v2_guard_joint_rows")
        ),
        "audio_internal_phase_law_v2_guard_failure_atlas": summarize_internal_phase_law_failure_atlas(
            loaded.get("audio_internal_phase_law_v2_guard_failure_atlas")
        ),
        "audio_internal_phase_law_v3_local_raw_compare_no_single_source": summarize_audio_predeclared_lockbox_compare(
            loaded.get("audio_internal_phase_law_v3_local_raw_compare_no_single_source")
        ),
        "audio_internal_phase_law_v3_local_variant_compare_no_single_source": summarize_internal_phase_law_variant_compare(
            loaded.get("audio_internal_phase_law_v3_local_variant_compare_no_single_source")
        ),
        "audio_internal_phase_law_v3_local_objective_score": summarize_internal_phase_law_objective_score(
            loaded.get("audio_internal_phase_law_v3_local_objective_score")
        ),
        "audio_internal_phase_law_v3_local_joint_rows": summarize_internal_phase_law_joint_rows(
            loaded.get("audio_internal_phase_law_v3_local_joint_rows")
        ),
        "audio_internal_phase_law_v3_local_failure_atlas": summarize_internal_phase_law_failure_atlas(
            loaded.get("audio_internal_phase_law_v3_local_failure_atlas")
        ),
        "audio_internal_phase_law_v3_local_strict_objective_score": summarize_internal_phase_law_objective_score(
            loaded.get("audio_internal_phase_law_v3_local_strict_objective_score")
        ),
        "audio_internal_phase_law_v3_local_strict_joint_rows": summarize_internal_phase_law_joint_rows(
            loaded.get("audio_internal_phase_law_v3_local_strict_joint_rows")
        ),
        "audio_internal_phase_law_v4_joint_nonbad_raw_compare_no_single_source": summarize_audio_predeclared_lockbox_compare(
            loaded.get("audio_internal_phase_law_v4_joint_nonbad_raw_compare_no_single_source")
        ),
        "audio_internal_phase_law_v4_joint_nonbad_variant_compare_no_single_source": summarize_internal_phase_law_variant_compare(
            loaded.get("audio_internal_phase_law_v4_joint_nonbad_variant_compare_no_single_source")
        ),
        "audio_internal_phase_law_v4_joint_nonbad_objective_score": summarize_internal_phase_law_objective_score(
            loaded.get("audio_internal_phase_law_v4_joint_nonbad_objective_score")
        ),
        "audio_internal_phase_law_v4_joint_nonbad_joint_rows": summarize_internal_phase_law_joint_rows(
            loaded.get("audio_internal_phase_law_v4_joint_nonbad_joint_rows")
        ),
        "audio_internal_phase_law_v4_joint_nonbad_failure_atlas": summarize_internal_phase_law_failure_atlas(
            loaded.get("audio_internal_phase_law_v4_joint_nonbad_failure_atlas")
        ),
        "audio_internal_phase_law_v5_low_energy_row_raw_compare_no_single_source": summarize_audio_predeclared_lockbox_compare(
            loaded.get("audio_internal_phase_law_v5_low_energy_row_raw_compare_no_single_source")
        ),
        "audio_internal_phase_law_v5_low_energy_row_variant_compare_no_single_source": summarize_internal_phase_law_variant_compare(
            loaded.get("audio_internal_phase_law_v5_low_energy_row_variant_compare_no_single_source")
        ),
        "audio_internal_phase_law_v5_low_energy_row_objective_score": summarize_internal_phase_law_objective_score(
            loaded.get("audio_internal_phase_law_v5_low_energy_row_objective_score")
        ),
        "audio_internal_phase_law_v5_low_energy_row_joint_rows": summarize_internal_phase_law_joint_rows(
            loaded.get("audio_internal_phase_law_v5_low_energy_row_joint_rows")
        ),
        "audio_internal_phase_law_v5_low_energy_row_failure_atlas": summarize_internal_phase_law_failure_atlas(
            loaded.get("audio_internal_phase_law_v5_low_energy_row_failure_atlas")
        ),
        "audio_internal_phase_law_v6_low_energy_win_raw_compare_no_single_source": summarize_audio_predeclared_lockbox_compare(
            loaded.get("audio_internal_phase_law_v6_low_energy_win_raw_compare_no_single_source")
        ),
        "audio_internal_phase_law_v6_low_energy_win_variant_compare_no_single_source": summarize_internal_phase_law_variant_compare(
            loaded.get("audio_internal_phase_law_v6_low_energy_win_variant_compare_no_single_source")
        ),
        "audio_internal_phase_law_v6_low_energy_win_objective_score": summarize_internal_phase_law_objective_score(
            loaded.get("audio_internal_phase_law_v6_low_energy_win_objective_score")
        ),
        "audio_internal_phase_law_v6_low_energy_win_joint_rows": summarize_internal_phase_law_joint_rows(
            loaded.get("audio_internal_phase_law_v6_low_energy_win_joint_rows")
        ),
        "audio_internal_phase_law_v6_low_energy_win_failure_atlas": summarize_internal_phase_law_failure_atlas(
            loaded.get("audio_internal_phase_law_v6_low_energy_win_failure_atlas")
        ),
        "audio_internal_phase_law_v7_low_energy_gain_ladder_raw_compare_no_single_source": summarize_audio_predeclared_lockbox_compare(
            loaded.get("audio_internal_phase_law_v7_low_energy_gain_ladder_raw_compare_no_single_source")
        ),
        "audio_internal_phase_law_v7_low_energy_gain_ladder_variant_compare_no_single_source": summarize_internal_phase_law_variant_compare(
            loaded.get("audio_internal_phase_law_v7_low_energy_gain_ladder_variant_compare_no_single_source")
        ),
        "audio_internal_phase_law_v7_low_energy_gain_ladder_objective_score": summarize_internal_phase_law_objective_score(
            loaded.get("audio_internal_phase_law_v7_low_energy_gain_ladder_objective_score")
        ),
        "audio_internal_phase_law_v7_low_energy_gain_ladder_joint_rows": summarize_internal_phase_law_joint_rows(
            loaded.get("audio_internal_phase_law_v7_low_energy_gain_ladder_joint_rows")
        ),
        "audio_internal_phase_law_v7_low_energy_gain_ladder_failure_atlas": summarize_internal_phase_law_failure_atlas(
            loaded.get("audio_internal_phase_law_v7_low_energy_gain_ladder_failure_atlas")
        ),
        "audio_internal_phase_law_v7_low_energy_gain_ladder_target_flips": summarize_internal_phase_law_target_flips(
            loaded.get("audio_internal_phase_law_v7_low_energy_gain_ladder_target_flips")
        ),
        "audio_internal_phase_law_v8_family_support_mask_raw_compare_no_single_source": summarize_audio_predeclared_lockbox_compare(
            loaded.get("audio_internal_phase_law_v8_family_support_mask_raw_compare_no_single_source")
        ),
        "audio_internal_phase_law_v8_family_support_mask_variant_compare_no_single_source": summarize_internal_phase_law_variant_compare(
            loaded.get("audio_internal_phase_law_v8_family_support_mask_variant_compare_no_single_source")
        ),
        "audio_internal_phase_law_v8_family_support_mask_objective_score": summarize_internal_phase_law_objective_score(
            loaded.get("audio_internal_phase_law_v8_family_support_mask_objective_score")
        ),
        "audio_internal_phase_law_v8_family_support_mask_joint_rows": summarize_internal_phase_law_joint_rows(
            loaded.get("audio_internal_phase_law_v8_family_support_mask_joint_rows")
        ),
        "audio_internal_phase_law_v8_family_support_mask_failure_atlas": summarize_internal_phase_law_failure_atlas(
            loaded.get("audio_internal_phase_law_v8_family_support_mask_failure_atlas")
        ),
        "audio_internal_phase_law_v8_family_support_mask_diagnostics": summarize_internal_phase_law_family_masks(
            loaded.get("audio_internal_phase_law_v8_family_support_mask_diagnostics")
        ),
        "audio_internal_phase_law_v8_support_router_oracle": summarize_internal_phase_law_support_router_oracle(
            loaded.get("audio_internal_phase_law_v8_support_router_oracle")
        ),
        "audio_internal_phase_law_v9_phase_masks_raw_compare_no_single_source": summarize_audio_predeclared_lockbox_compare(
            loaded.get("audio_internal_phase_law_v9_phase_masks_raw_compare_no_single_source")
        ),
        "audio_internal_phase_law_v9_phase_masks_variant_compare_no_single_source": summarize_internal_phase_law_variant_compare(
            loaded.get("audio_internal_phase_law_v9_phase_masks_variant_compare_no_single_source")
        ),
        "audio_internal_phase_law_v9_phase_masks_objective_score": summarize_internal_phase_law_objective_score(
            loaded.get("audio_internal_phase_law_v9_phase_masks_objective_score")
        ),
        "audio_internal_phase_law_v9_phase_masks_joint_rows": summarize_internal_phase_law_joint_rows(
            loaded.get("audio_internal_phase_law_v9_phase_masks_joint_rows")
        ),
        "audio_internal_phase_law_v9_phase_masks_phase_stable_target_score": (
            summarize_internal_phase_law_objective_score(
                loaded.get("audio_internal_phase_law_v9_phase_masks_phase_stable_target_score")
            )
        ),
        "audio_internal_phase_law_v9_phase_masks_phase_stable_target_joint_rows": (
            summarize_internal_phase_law_joint_rows(
                loaded.get("audio_internal_phase_law_v9_phase_masks_phase_stable_target_joint_rows")
            )
        ),
        "audio_internal_phase_law_v9_phase_masks_failure_atlas": summarize_internal_phase_law_failure_atlas(
            loaded.get("audio_internal_phase_law_v9_phase_masks_failure_atlas")
        ),
        "audio_internal_phase_law_v10_phase_router_masks_raw_compare_no_single_source": (
            summarize_audio_predeclared_lockbox_compare(
                loaded.get("audio_internal_phase_law_v10_phase_router_masks_raw_compare_no_single_source")
            )
        ),
        "audio_internal_phase_law_v10_phase_router_masks_variant_compare_no_single_source": (
            summarize_internal_phase_law_variant_compare(
                loaded.get("audio_internal_phase_law_v10_phase_router_masks_variant_compare_no_single_source")
            )
        ),
        "audio_internal_phase_law_v10_phase_router_masks_objective_score": (
            summarize_internal_phase_law_objective_score(
                loaded.get("audio_internal_phase_law_v10_phase_router_masks_objective_score")
            )
        ),
        "audio_internal_phase_law_v10_phase_router_masks_joint_rows": summarize_internal_phase_law_joint_rows(
            loaded.get("audio_internal_phase_law_v10_phase_router_masks_joint_rows")
        ),
        "audio_internal_phase_law_v10_phase_stable_target_score": summarize_internal_phase_law_objective_score(
            loaded.get("audio_internal_phase_law_v10_phase_stable_target_score")
        ),
        "audio_internal_phase_law_v10_phase_stable_target_joint_rows": summarize_internal_phase_law_joint_rows(
            loaded.get("audio_internal_phase_law_v10_phase_stable_target_joint_rows")
        ),
        "audio_internal_phase_law_v10_phase_coherent_target_score": summarize_internal_phase_law_objective_score(
            loaded.get("audio_internal_phase_law_v10_phase_coherent_target_score")
        ),
        "audio_internal_phase_law_v10_phase_coherent_target_joint_rows": summarize_internal_phase_law_joint_rows(
            loaded.get("audio_internal_phase_law_v10_phase_coherent_target_joint_rows")
        ),
        "audio_internal_phase_law_v9_candidate_classes": summarize_internal_phase_law_candidate_classes(
            loaded.get("audio_internal_phase_law_v9_candidate_classes")
        ),
        "audio_internal_phase_law_v10_candidate_classes": summarize_internal_phase_law_candidate_classes(
            loaded.get("audio_internal_phase_law_v10_candidate_classes")
        ),
        "audio_internal_phase_law_v11_phase_router_direct_raw_compare_no_single_source": (
            summarize_audio_predeclared_lockbox_compare(
                loaded.get("audio_internal_phase_law_v11_phase_router_direct_raw_compare_no_single_source")
            )
        ),
        "audio_internal_phase_law_v11_phase_router_direct_variant_compare_no_single_source": (
            summarize_internal_phase_law_variant_compare(
                loaded.get("audio_internal_phase_law_v11_phase_router_direct_variant_compare_no_single_source")
            )
        ),
        "audio_internal_phase_law_v11_phase_router_direct_objective_score": (
            summarize_internal_phase_law_objective_score(
                loaded.get("audio_internal_phase_law_v11_phase_router_direct_objective_score")
            )
        ),
        "audio_internal_phase_law_v11_phase_router_direct_joint_rows": summarize_internal_phase_law_joint_rows(
            loaded.get("audio_internal_phase_law_v11_phase_router_direct_joint_rows")
        ),
        "audio_internal_phase_law_v11_candidate_classes": summarize_internal_phase_law_candidate_classes(
            loaded.get("audio_internal_phase_law_v11_candidate_classes")
        ),
        "audio_internal_phase_law_v12_phase_reentry_direct_raw_compare_no_single_source": (
            summarize_audio_predeclared_lockbox_compare(
                loaded.get("audio_internal_phase_law_v12_phase_reentry_direct_raw_compare_no_single_source")
            )
        ),
        "audio_internal_phase_law_v12_phase_reentry_direct_variant_compare_no_single_source": (
            summarize_internal_phase_law_variant_compare(
                loaded.get("audio_internal_phase_law_v12_phase_reentry_direct_variant_compare_no_single_source")
            )
        ),
        "audio_internal_phase_law_v12_phase_reentry_direct_objective_score": (
            summarize_internal_phase_law_objective_score(
                loaded.get("audio_internal_phase_law_v12_phase_reentry_direct_objective_score")
            )
        ),
        "audio_internal_phase_law_v12_phase_reentry_direct_joint_rows": summarize_internal_phase_law_joint_rows(
            loaded.get("audio_internal_phase_law_v12_phase_reentry_direct_joint_rows")
        ),
        "audio_internal_phase_law_v12_candidate_classes": summarize_internal_phase_law_candidate_classes(
            loaded.get("audio_internal_phase_law_v12_candidate_classes")
        ),
        "audio_internal_phase_law_v13_causal_reentry_direct_raw_compare_no_single_source": (
            summarize_audio_predeclared_lockbox_compare(
                loaded.get("audio_internal_phase_law_v13_causal_reentry_direct_raw_compare_no_single_source")
            )
        ),
        "audio_internal_phase_law_v13_causal_reentry_direct_variant_compare_no_single_source": (
            summarize_internal_phase_law_variant_compare(
                loaded.get("audio_internal_phase_law_v13_causal_reentry_direct_variant_compare_no_single_source")
            )
        ),
        "audio_internal_phase_law_v13_causal_reentry_direct_objective_score": (
            summarize_internal_phase_law_objective_score(
                loaded.get("audio_internal_phase_law_v13_causal_reentry_direct_objective_score")
            )
        ),
        "audio_internal_phase_law_v13_causal_reentry_direct_joint_rows": summarize_internal_phase_law_joint_rows(
            loaded.get("audio_internal_phase_law_v13_causal_reentry_direct_joint_rows")
        ),
        "audio_internal_phase_law_v13_candidate_classes": summarize_internal_phase_law_candidate_classes(
            loaded.get("audio_internal_phase_law_v13_candidate_classes")
        ),
        "audio_internal_phase_law_predeclared_diagnostic_class": (
            summarize_internal_phase_law_predeclared_diagnostic(
                loaded.get("audio_internal_phase_law_predeclared_diagnostic_class")
            )
        ),
        "dense_signature_claim": summarize_dense_signature_claim(loaded.get("dense_signature_claim")),
        "dense_signature_failure_modes": summarize_dense_signature_failure_modes(
            loaded.get("dense_signature_failure_modes")
        ),
        "relational_signature_contract_audit": summarize_relational_signature_contract_audit(
            loaded.get("relational_signature_contract_audit")
        ),
        "relational_signature_learning_contract": summarize_relational_signature_learning_contract(
            loaded.get("relational_signature_learning_contract")
        ),
        "learned_signature_scout": summarize_learned_signature_scout(
            loaded.get("learned_signature_scout")
        ),
        "learned_signature_scout_compare": summarize_learned_signature_scout_compare(
            loaded.get("learned_signature_scout_compare")
        ),
        "learned_signature_split_suite": summarize_learned_signature_split_suite(
            loaded.get("learned_signature_split_suite")
        ),
        "learned_signature_split_suite_compare": summarize_learned_signature_split_suite_compare(
            loaded.get("learned_signature_split_suite_compare")
        ),
        "learned_signature_case_failures": summarize_learned_signature_case_failures(
            loaded.get("learned_signature_case_failures")
        ),
        "signature_operator_tail_probe_compare": summarize_signature_operator_tail_probe_compare(
            loaded.get("signature_operator_tail_probe_compare")
        ),
        "signature_future_law_probe": summarize_signature_future_law_probe(
            loaded.get("signature_future_law_probe")
        ),
        "signature_tail_incremental_usefulness": summarize_signature_tail_incremental_usefulness(
            loaded.get("signature_tail_incremental_usefulness")
        ),
        "semantic_projector_contract": summarize_semantic_projector_contract(
            loaded.get("semantic_projector_contract")
        ),
        "substrate_lattice": summarize_substrate(loaded["substrate"]),
        "claim_isolation": summarize_claim_isolation(loaded["claim_isolation"]),
        "qtrace_inheritance": summarize_claim_isolation(loaded["qtrace_inheritance"]),
        "arc_lane": summarize_claim_isolation(loaded["arc_lane"]),
        "arc_q_cross": summarize_claim_isolation(loaded["arc_q_cross"]),
        "arc_q_control_claim": summarize_arc_q_control_claim(loaded.get("arc_q_control_claim")),
    }
    claims = {
        "semi_tokenless_modeling": {
            "status": "open",
            "reason": "Audio/state continuation remains weak; dense signature and token baseline comparisons are not complete.",
        },
        "audio_without_future_magnitude": {
            "status": evidence["audio_continuation"]["status"],
            "reason": (
                "Harness prevents future magnitude leakage, but Circleworld does not decisively beat simple carriers. "
                f"Broad 18-case render benchmark mean corr is {evidence['broad18_benchmark'].get('mean_corr')} "
                f"with continuity delta band {evidence['broad18_benchmark'].get('continuity_delta_band')}; "
                f"method compare status is {evidence['audio_method_compare'].get('status')}; "
                f"copyphase continuation status is {evidence['audio_continuation_copyphase'].get('status')} "
                f"and copyphase compare status is {evidence['audio_method_compare_copyphase'].get('status')}; "
                f"phase influence status is {evidence['audio_phase_influence'].get('status')}; "
                f"phase seed status is {evidence['audio_phase_seed'].get('status')}; "
                f"weighted-fit phase seed status is {evidence['audio_phase_seed_weighted_fit'].get('status')}; "
                f"Circleworld delta-over-copyphase status is {evidence['audio_circle_delta_probe'].get('status')} "
                f"with best corr delta {evidence['audio_circle_delta_probe'].get('best_corr_delta')}; "
                f"fixed mechanism probe status is {evidence['audio_delta_mechanism_probe'].get('status')} "
                f"with best corr delta {evidence['audio_delta_mechanism_probe'].get('best_corr_delta')}; "
                f"predeclared mechanism lockbox status is {evidence['audio_delta_mechanism_lockbox'].get('status')} "
                f"with best corr delta {evidence['audio_delta_mechanism_lockbox'].get('best_corr_delta')} "
                f"and corr win fraction "
                f"{_get(evidence['audio_delta_mechanism_lockbox'], 'best_corr_delta_row', 'target_corr_win_fraction_vs_gain0')}; "
                f"family sensitivity status is {evidence['audio_mechanism_family_sensitivity'].get('status')} "
                f"with best family {evidence['audio_mechanism_family_sensitivity'].get('best_family')} "
                f"mean corr delta {evidence['audio_mechanism_family_sensitivity'].get('best_family_mean_corr_delta')}, "
                f"median {evidence['audio_mechanism_family_sensitivity'].get('best_family_median_corr_delta')}, "
                f"win fraction {evidence['audio_mechanism_family_sensitivity'].get('best_family_corr_win_fraction')}, "
                f"and outlier share {evidence['audio_mechanism_family_sensitivity'].get('best_family_positive_outlier_share')}; "
                f"electric-motor holdout status is {evidence['audio_electric_motor_holdout'].get('status')} "
                f"with best holdout family "
                f"{_get(evidence['audio_electric_motor_holdout'], 'holdout_family_metrics', 'family')} "
                f"over "
                f"{_get(evidence['audio_electric_motor_holdout'], 'holdout_family_metrics', 'case_count')} "
                f"cases; "
                f"mean corr delta "
                f"{_get(evidence['audio_electric_motor_holdout'], 'holdout_family_metrics', 'mean_corr_delta')}, "
                f"median "
                f"{_get(evidence['audio_electric_motor_holdout'], 'holdout_family_metrics', 'median_corr_delta')}, "
                f"win fraction "
                f"{_get(evidence['audio_electric_motor_holdout'], 'holdout_family_metrics', 'corr_win_fraction')}, "
                f"and outlier share "
                f"{_get(evidence['audio_electric_motor_holdout'], 'holdout_family_metrics', 'positive_outlier_share')}; "
                "treat this as phenotype-local until the no-razor structural motor rows clear robustness checks; "
                f"disjoint steady-phenotype manifest status is "
                f"{evidence['audio_steady_phenotype_manifest'].get('status')} "
                f"with steady-buzz mean corr delta "
                f"{_get(evidence['audio_steady_phenotype_manifest'], 'best_steady_buzz_group', 'mean_corr_delta')}, "
                f"no-razor motor mean corr delta "
                f"{_get(evidence['audio_steady_phenotype_manifest'], 'best_no_razor_motor_group', 'mean_corr_delta')}, "
                f"and selected unique paths "
                f"{evidence['audio_steady_phenotype_manifest'].get('selected_unique_paths')}; "
                f"phase-phenotype diagnostic status is "
                f"{evidence['audio_phase_phenotype_diagnostics'].get('status')} "
                f"with strongest non-single-source feature "
                f"{evidence['audio_phase_phenotype_diagnostics'].get('strongest_non_single_feature')} "
                f"corr "
                f"{evidence['audio_phase_phenotype_diagnostics'].get('strongest_non_single_corr')}, "
                f"prefix phase-velocity corr "
                f"{evidence['audio_phase_phenotype_diagnostics'].get('prefix_phase_velocity_corr')}, "
                f"and mean shaped-delta corr "
                f"{evidence['audio_phase_phenotype_diagnostics'].get('mean_abs_shaped_delta_corr')}; "
                f"baseline-stratified diagnostic status is "
                f"{evidence['audio_baseline_stratified_diagnostics'].get('status')} "
                f"with promotion gate "
                f"{evidence['audio_baseline_stratified_diagnostics'].get('promotion_gate_status')} "
                f"with bad-bin mean corr delta "
                f"{_get(evidence['audio_baseline_stratified_diagnostics'], 'bad_baseline', 'mean_corr_delta')}, "
                f"non-bad weighted mean "
                f"{evidence['audio_baseline_stratified_diagnostics'].get('non_bad_mean_corr_delta')}, "
                f"non-bad weighted win fraction "
                f"{evidence['audio_baseline_stratified_diagnostics'].get('non_bad_corr_win_fraction')}, "
                f"weak-bin mean "
                f"{_get(evidence['audio_baseline_stratified_diagnostics'], 'weak_baseline', 'mean_corr_delta')}, "
                f"moderate-bin mean "
                f"{_get(evidence['audio_baseline_stratified_diagnostics'], 'moderate_baseline', 'mean_corr_delta')}, "
                f"and good-bin mean "
                f"{_get(evidence['audio_baseline_stratified_diagnostics'], 'good_baseline', 'mean_corr_delta')}; "
                f"predeclared lockbox manifest status is "
                f"{evidence['audio_predeclared_lockbox_manifest'].get('status')} "
                f"with {evidence['audio_predeclared_lockbox_manifest'].get('selected_case_count')} selected cases; "
                f"predeclared lockbox compare status is "
                f"{evidence['audio_predeclared_lockbox_compare'].get('status')} "
                f"with overall mean corr delta "
                f"{evidence['audio_predeclared_lockbox_compare'].get('overall_mean_corr_delta')}, "
                f"median "
                f"{evidence['audio_predeclared_lockbox_compare'].get('overall_median_corr_delta')}, "
                f"win fraction "
                f"{evidence['audio_predeclared_lockbox_compare'].get('overall_corr_win_fraction')}, "
                f"non-bad mean corr delta "
                f"{evidence['audio_predeclared_lockbox_compare'].get('non_bad_mean_corr_delta')}, "
                f"and outlier share "
                f"{evidence['audio_predeclared_lockbox_compare'].get('overall_positive_outlier_share')}; "
                f"without the single-source probe, predeclared lockbox compare status is "
                f"{evidence['audio_predeclared_lockbox_compare_no_single_source'].get('status')} "
                f"with mean corr delta "
                f"{evidence['audio_predeclared_lockbox_compare_no_single_source'].get('overall_mean_corr_delta')}, "
                f"median "
                f"{evidence['audio_predeclared_lockbox_compare_no_single_source'].get('overall_median_corr_delta')}, "
                f"and mean MSE delta "
                f"{evidence['audio_predeclared_lockbox_compare_no_single_source'].get('overall_mean_mse_delta')}; "
                f"exploratory phase-law scout status is "
                f"{evidence['audio_phase_law_scout_compare_no_single_source'].get('status')} "
                f"with best candidate "
                f"{_get(evidence['audio_phase_law_scout_compare_no_single_source'], 'candidate_summaries', 0, 'mechanism')} "
                f"gain "
                f"{_get(evidence['audio_phase_law_scout_compare_no_single_source'], 'candidate_summaries', 0, 'gain')} "
                f"mean corr delta "
                f"{_get(evidence['audio_phase_law_scout_compare_no_single_source'], 'candidate_summaries', 0, 'mean_corr_delta')} "
                f"and median "
                f"{_get(evidence['audio_phase_law_scout_compare_no_single_source'], 'candidate_summaries', 0, 'median_corr_delta')}; "
                f"internal phase-law raw lockbox status is "
                f"{evidence['audio_internal_phase_law_raw_compare_no_single_source'].get('status')} "
                f"with best absolute candidate "
                f"{_get(evidence['audio_internal_phase_law_raw_compare_no_single_source'], 'candidate_summaries', 0, 'run_label')} "
                f"{_get(evidence['audio_internal_phase_law_raw_compare_no_single_source'], 'candidate_summaries', 0, 'magnitude_mode')} "
                f"{_get(evidence['audio_internal_phase_law_raw_compare_no_single_source'], 'candidate_summaries', 0, 'mask_mode')} "
                f"mean corr delta "
                f"{_get(evidence['audio_internal_phase_law_raw_compare_no_single_source'], 'candidate_summaries', 0, 'mean_corr_delta')}; "
                f"direct internal phase-law variant compare status is "
                f"{evidence['audio_internal_phase_law_variant_compare_no_single_source'].get('status')} "
                f"with best overall run "
                f"{evidence['audio_internal_phase_law_variant_compare_no_single_source'].get('best_overall_run')} "
                f"mean corr delta "
                f"{evidence['audio_internal_phase_law_variant_compare_no_single_source'].get('best_overall_mean_corr_delta')} "
                f"and best matched-row candidate "
                f"{evidence['audio_internal_phase_law_variant_compare_no_single_source'].get('best_candidate_run')} "
                f"{evidence['audio_internal_phase_law_variant_compare_no_single_source'].get('best_candidate_magnitude_mode')} "
                f"{evidence['audio_internal_phase_law_variant_compare_no_single_source'].get('best_candidate_mask_mode')} "
                f"mean corr delta "
                f"{evidence['audio_internal_phase_law_variant_compare_no_single_source'].get('best_candidate_mean_corr_delta')}; "
                f"locked internal phase-law objective status is "
                f"{evidence['audio_internal_phase_law_objective_score'].get('status')} "
                f"with best run {evidence['audio_internal_phase_law_objective_score'].get('best_run')} "
                f"and decision {evidence['audio_internal_phase_law_objective_score'].get('best_decision')}; "
                f"focused internal phase-law coefficient search status is "
                f"{evidence['audio_internal_phase_law_focused_objective_score'].get('status')} "
                f"with best run {evidence['audio_internal_phase_law_focused_objective_score'].get('best_run')} "
                f"and decision {evidence['audio_internal_phase_law_focused_objective_score'].get('best_decision')}; "
                f"wide internal phase-law coefficient search status is "
                f"{evidence['audio_internal_phase_law_wide_objective_score'].get('status')} "
                f"with best run {evidence['audio_internal_phase_law_wide_objective_score'].get('best_run')} "
                f"and decision {evidence['audio_internal_phase_law_wide_objective_score'].get('best_decision')}; "
                f"wide joint-row diagnostic status is "
                f"{evidence['audio_internal_phase_law_wide_joint_rows'].get('status')} "
                f"with candidate count {evidence['audio_internal_phase_law_wide_joint_rows'].get('candidate_count')}; "
                f"v2 guarded consensus/median internal phase-law status is "
                f"{evidence['audio_internal_phase_law_v2_guard_objective_score'].get('status')} "
                f"with best run {evidence['audio_internal_phase_law_v2_guard_objective_score'].get('best_run')} "
                f"and decision {evidence['audio_internal_phase_law_v2_guard_objective_score'].get('best_decision')}; "
                f"v2 guard joint-row diagnostic status is "
                f"{evidence['audio_internal_phase_law_v2_guard_joint_rows'].get('status')} "
                f"with candidate count {evidence['audio_internal_phase_law_v2_guard_joint_rows'].get('candidate_count')}; "
                f"v2 guard atlas joint status is "
                f"{evidence['audio_internal_phase_law_v2_guard_failure_atlas'].get('joint_status')} "
                f"with best direct mean corr delta "
                f"{evidence['audio_internal_phase_law_v2_guard_failure_atlas'].get('best_direct_mean_corr_delta')} "
                f"and best joint decision "
                f"{evidence['audio_internal_phase_law_v2_guard_failure_atlas'].get('best_joint_decision')}; "
                f"v3 local phase-law status is "
                f"{evidence['audio_internal_phase_law_v3_local_objective_score'].get('status')} "
                f"with best run {evidence['audio_internal_phase_law_v3_local_objective_score'].get('best_run')} "
                f"and decision {evidence['audio_internal_phase_law_v3_local_objective_score'].get('best_decision')}; "
                f"v3 local joint-row diagnostic status is "
                f"{evidence['audio_internal_phase_law_v3_local_joint_rows'].get('status')} "
                f"with candidate count {evidence['audio_internal_phase_law_v3_local_joint_rows'].get('candidate_count')}; "
                f"v3 local atlas joint status is "
                f"{evidence['audio_internal_phase_law_v3_local_failure_atlas'].get('joint_status')} "
                f"with best direct mean corr delta "
                f"{evidence['audio_internal_phase_law_v3_local_failure_atlas'].get('best_direct_mean_corr_delta')} "
                f"and best joint decision "
                f"{evidence['audio_internal_phase_law_v3_local_failure_atlas'].get('best_joint_decision')}; "
                f"v3 strict same-row objective status is "
                f"{evidence['audio_internal_phase_law_v3_local_strict_objective_score'].get('status')} "
                f"with best decision "
                f"{evidence['audio_internal_phase_law_v3_local_strict_objective_score'].get('best_decision')}; "
                f"v3 strict joint-row status is "
                f"{evidence['audio_internal_phase_law_v3_local_strict_joint_rows'].get('status')} "
                f"with candidate count "
                f"{evidence['audio_internal_phase_law_v3_local_strict_joint_rows'].get('candidate_count')} "
                f"and mean nonbad-bin corr delta "
                f"{evidence['audio_internal_phase_law_v3_local_strict_joint_rows'].get('mean_absolute_nonbad_bin_corr_delta')}; "
                f"v4 joint/nonbad objective status is "
                f"{evidence['audio_internal_phase_law_v4_joint_nonbad_objective_score'].get('status')} "
                f"with best run {evidence['audio_internal_phase_law_v4_joint_nonbad_objective_score'].get('best_run')} "
                f"and decision {evidence['audio_internal_phase_law_v4_joint_nonbad_objective_score'].get('best_decision')}; "
                f"v4 joint-row status is "
                f"{evidence['audio_internal_phase_law_v4_joint_nonbad_joint_rows'].get('status')} "
                f"with candidate count "
                f"{evidence['audio_internal_phase_law_v4_joint_nonbad_joint_rows'].get('candidate_count')} "
                f"and best direct mean corr delta "
                f"{evidence['audio_internal_phase_law_v4_joint_nonbad_joint_rows'].get('best_direct_mean_corr_delta')}; "
                f"v4 atlas joint status is "
                f"{evidence['audio_internal_phase_law_v4_joint_nonbad_failure_atlas'].get('joint_status')} "
                f"with best direct mean corr delta "
                f"{evidence['audio_internal_phase_law_v4_joint_nonbad_failure_atlas'].get('best_direct_mean_corr_delta')}; "
                f"v5 low-energy target-row objective status is "
                f"{evidence['audio_internal_phase_law_v5_low_energy_row_objective_score'].get('status')} "
                f"with target best run "
                f"{evidence['audio_internal_phase_law_v5_low_energy_row_objective_score'].get('target_low_energy_best_run')} "
                f"and target decision "
                f"{evidence['audio_internal_phase_law_v5_low_energy_row_objective_score'].get('target_low_energy_best_decision')}; "
                f"v5 target direct/nonbad corr deltas are "
                f"{evidence['audio_internal_phase_law_v5_low_energy_row_objective_score'].get('target_low_energy_best_direct_mean_corr_delta')} / "
                f"{evidence['audio_internal_phase_law_v5_low_energy_row_objective_score'].get('target_low_energy_best_nonbad_bin_mean_corr_delta')}; "
                f"v6 low-energy win objective status is "
                f"{evidence['audio_internal_phase_law_v6_low_energy_win_objective_score'].get('status')} "
                f"with target best run "
                f"{evidence['audio_internal_phase_law_v6_low_energy_win_objective_score'].get('target_low_energy_best_run')} "
                f"and target decision "
                f"{evidence['audio_internal_phase_law_v6_low_energy_win_objective_score'].get('target_low_energy_best_decision')}; "
                f"v6 target direct mean/median/wins are "
                f"{evidence['audio_internal_phase_law_v6_low_energy_win_objective_score'].get('target_low_energy_best_direct_mean_corr_delta')} / "
                f"{evidence['audio_internal_phase_law_v6_low_energy_win_objective_score'].get('target_low_energy_best_direct_median_corr_delta')} / "
                f"{evidence['audio_internal_phase_law_v6_low_energy_win_objective_score'].get('target_low_energy_best_direct_corr_win_fraction')}; "
                f"v6 target nonbad/good corr deltas are "
                f"{evidence['audio_internal_phase_law_v6_low_energy_win_objective_score'].get('target_low_energy_best_nonbad_bin_mean_corr_delta')} / "
                f"{evidence['audio_internal_phase_law_v6_low_energy_win_objective_score'].get('target_low_energy_best_good_bin_mean_corr_delta')}; "
                f"v7 low-energy gain ladder status is "
                f"{evidence['audio_internal_phase_law_v7_low_energy_gain_ladder_objective_score'].get('status')} "
                f"with target best run "
                f"{evidence['audio_internal_phase_law_v7_low_energy_gain_ladder_objective_score'].get('target_low_energy_best_run')} "
                f"and target decision "
                f"{evidence['audio_internal_phase_law_v7_low_energy_gain_ladder_objective_score'].get('target_low_energy_best_decision')}; "
                f"v7 target-flip selected run "
                f"{evidence['audio_internal_phase_law_v7_low_energy_gain_ladder_target_flips'].get('selected_run')} "
                f"has target mean/median/wins "
                f"{evidence['audio_internal_phase_law_v7_low_energy_gain_ladder_target_flips'].get('selected_target_gain_mean_corr_delta')} / "
                f"{evidence['audio_internal_phase_law_v7_low_energy_gain_ladder_target_flips'].get('selected_target_gain_median_corr_delta')} / "
                f"{evidence['audio_internal_phase_law_v7_low_energy_gain_ladder_target_flips'].get('selected_target_gain_corr_win_fraction')} "
                f"with sign categories "
                f"{evidence['audio_internal_phase_law_v7_low_energy_gain_ladder_target_flips'].get('category_counts')}; "
                f"v8 family/support-mask status is "
                f"{evidence['audio_internal_phase_law_v8_family_support_mask_objective_score'].get('status')} "
                f"with target best run "
                f"{evidence['audio_internal_phase_law_v8_family_support_mask_objective_score'].get('target_low_energy_best_run')} "
                f"and target decision "
                f"{evidence['audio_internal_phase_law_v8_family_support_mask_objective_score'].get('target_low_energy_best_decision')}; "
                f"v8 support-mask diagnostic selected run "
                f"{evidence['audio_internal_phase_law_v8_family_support_mask_diagnostics'].get('selected_run')} "
                f"has low/all/high target wins "
                f"{evidence['audio_internal_phase_law_v8_family_support_mask_diagnostics'].get('selected_low_energy_corr_win_fraction')} / "
                f"{evidence['audio_internal_phase_law_v8_family_support_mask_diagnostics'].get('selected_all_bins_corr_win_fraction')} / "
                f"{evidence['audio_internal_phase_law_v8_family_support_mask_diagnostics'].get('selected_high_energy_corr_win_fraction')}; "
                f"v8 global best direct row is "
                f"{evidence['audio_internal_phase_law_v8_family_support_mask_diagnostics'].get('global_best_run')} "
                f"{evidence['audio_internal_phase_law_v8_family_support_mask_diagnostics'].get('global_best_mask')} "
                f"gain "
                f"{evidence['audio_internal_phase_law_v8_family_support_mask_diagnostics'].get('global_best_gain')} "
                f"with mean/median/wins "
                f"{evidence['audio_internal_phase_law_v8_family_support_mask_diagnostics'].get('global_best_mean_corr_delta')} / "
                f"{evidence['audio_internal_phase_law_v8_family_support_mask_diagnostics'].get('global_best_median_corr_delta')} / "
                f"{evidence['audio_internal_phase_law_v8_family_support_mask_diagnostics'].get('global_best_corr_win_fraction')}; "
                f"v8 support-router oracle status is "
                f"{evidence['audio_internal_phase_law_v8_support_router_oracle'].get('status')} "
                f"with global/family/case oracle wins "
                f"{_get(evidence['audio_internal_phase_law_v8_support_router_oracle'], 'global_fixed_best', 'corr_win_fraction')} / "
                f"{_get(evidence['audio_internal_phase_law_v8_support_router_oracle'], 'family_oracle_router', 'corr_win_fraction')} / "
                f"{_get(evidence['audio_internal_phase_law_v8_support_router_oracle'], 'case_oracle_router', 'corr_win_fraction')}; "
                f"v9 phase-only mask scout joint status is "
                f"{evidence['audio_internal_phase_law_v9_phase_masks_joint_rows'].get('status')} "
                f"with candidates "
                f"{evidence['audio_internal_phase_law_v9_phase_masks_joint_rows'].get('candidate_count')} "
                f"and best row "
                f"{evidence['audio_internal_phase_law_v9_phase_masks_joint_rows'].get('best_run')} "
                f"{evidence['audio_internal_phase_law_v9_phase_masks_joint_rows'].get('best_mask_mode')} "
                f"gain "
                f"{evidence['audio_internal_phase_law_v9_phase_masks_joint_rows'].get('best_gain')} "
                f"direct/nonbad "
                f"{evidence['audio_internal_phase_law_v9_phase_masks_joint_rows'].get('best_direct_mean_corr_delta')} / "
                f"{evidence['audio_internal_phase_law_v9_phase_masks_joint_rows'].get('best_absolute_nonbad_bin_mean_corr_delta')}; "
                f"v9 explicit phase-stable target score has target candidates "
                f"{evidence['audio_internal_phase_law_v9_phase_masks_phase_stable_target_score'].get('target_row_candidate_count')} "
                f"of "
                f"{evidence['audio_internal_phase_law_v9_phase_masks_phase_stable_target_score'].get('target_row_count')} "
                f"with best run "
                f"{evidence['audio_internal_phase_law_v9_phase_masks_phase_stable_target_score'].get('target_row_best_run')} "
                f"and decision "
                f"{evidence['audio_internal_phase_law_v9_phase_masks_phase_stable_target_score'].get('target_row_best_decision')}; "
                f"v9 explicit phase-stable joint target row has candidates "
                f"{evidence['audio_internal_phase_law_v9_phase_masks_phase_stable_target_joint_rows'].get('target_row_candidate_count')} "
                f"and best direct/absolute/nonbad "
                f"{evidence['audio_internal_phase_law_v9_phase_masks_phase_stable_target_joint_rows'].get('target_row_best_direct_mean_corr_delta')} / "
                f"{evidence['audio_internal_phase_law_v9_phase_masks_phase_stable_target_joint_rows'].get('target_row_best_absolute_mean_corr_delta')} / "
                f"{evidence['audio_internal_phase_law_v9_phase_masks_phase_stable_target_joint_rows'].get('target_row_best_nonbad_bin_mean_corr_delta')}; "
                f"v10 phase-router mask scout has joint status "
                f"{evidence['audio_internal_phase_law_v10_phase_router_masks_joint_rows'].get('status')} "
                f"with candidates "
                f"{evidence['audio_internal_phase_law_v10_phase_router_masks_joint_rows'].get('candidate_count')} "
                f"and target router candidates "
                f"{evidence['audio_internal_phase_law_v10_phase_router_masks_joint_rows'].get('target_row_candidate_count')}; "
                f"v10 router target best direct/absolute/nonbad/good "
                f"{evidence['audio_internal_phase_law_v10_phase_router_masks_joint_rows'].get('target_row_best_direct_mean_corr_delta')} / "
                f"{evidence['audio_internal_phase_law_v10_phase_router_masks_joint_rows'].get('target_row_best_absolute_mean_corr_delta')} / "
                f"{evidence['audio_internal_phase_law_v10_phase_router_masks_joint_rows'].get('target_row_best_nonbad_bin_mean_corr_delta')} / "
                f"{evidence['audio_internal_phase_law_v10_phase_router_masks_joint_rows'].get('target_row_best_good_bin_mean_corr_delta')}; "
                f"candidate-class split is v9 phase-support "
                f"{evidence['audio_internal_phase_law_v9_candidate_classes'].get('phase_support_candidate_count')} "
                f"versus v10 phase-support "
                f"{evidence['audio_internal_phase_law_v10_candidate_classes'].get('phase_support_candidate_count')} "
                f"with strict audio candidates "
                f"{evidence['audio_internal_phase_law_v10_candidate_classes'].get('strict_audio_candidate_count')}; "
                f"v11 phase-router direct scout has objective status "
                f"{evidence['audio_internal_phase_law_v11_phase_router_direct_objective_score'].get('status')} "
                f"with best decision "
                f"{evidence['audio_internal_phase_law_v11_phase_router_direct_objective_score'].get('best_decision')}, "
                f"joint candidates "
                f"{evidence['audio_internal_phase_law_v11_phase_router_direct_joint_rows'].get('candidate_count')} "
                f"of "
                f"{evidence['audio_internal_phase_law_v11_phase_router_direct_joint_rows'].get('row_count')}, "
                f"target candidates "
                f"{evidence['audio_internal_phase_law_v11_phase_router_direct_joint_rows'].get('target_row_candidate_count')} "
                f"of "
                f"{evidence['audio_internal_phase_law_v11_phase_router_direct_joint_rows'].get('target_row_count')}, "
                f"and candidate-class phase-support "
                f"{evidence['audio_internal_phase_law_v11_candidate_classes'].get('phase_support_candidate_count')} "
                f"with strict audio candidates "
                f"{evidence['audio_internal_phase_law_v11_candidate_classes'].get('strict_audio_candidate_count')}; "
                f"v12 phase-reentry direct scout has objective status "
                f"{evidence['audio_internal_phase_law_v12_phase_reentry_direct_objective_score'].get('status')} "
                f"with best decision "
                f"{evidence['audio_internal_phase_law_v12_phase_reentry_direct_objective_score'].get('best_decision')}, "
                f"joint candidates "
                f"{evidence['audio_internal_phase_law_v12_phase_reentry_direct_joint_rows'].get('candidate_count')} "
                f"of "
                f"{evidence['audio_internal_phase_law_v12_phase_reentry_direct_joint_rows'].get('row_count')}, "
                f"target candidates "
                f"{evidence['audio_internal_phase_law_v12_phase_reentry_direct_joint_rows'].get('target_row_candidate_count')} "
                f"of "
                f"{evidence['audio_internal_phase_law_v12_phase_reentry_direct_joint_rows'].get('target_row_count')}, "
                f"and candidate-class phase-support "
                f"{evidence['audio_internal_phase_law_v12_candidate_classes'].get('phase_support_candidate_count')} "
                f"with strict audio candidates "
                f"{evidence['audio_internal_phase_law_v12_candidate_classes'].get('strict_audio_candidate_count')}; "
                f"v13 causal phase-reentry scout has objective status "
                f"{evidence['audio_internal_phase_law_v13_causal_reentry_direct_objective_score'].get('status')} "
                f"with best decision "
                f"{evidence['audio_internal_phase_law_v13_causal_reentry_direct_objective_score'].get('best_decision')}, "
                f"joint candidates "
                f"{evidence['audio_internal_phase_law_v13_causal_reentry_direct_joint_rows'].get('candidate_count')} "
                f"of "
                f"{evidence['audio_internal_phase_law_v13_causal_reentry_direct_joint_rows'].get('row_count')}, "
                f"target candidates "
                f"{evidence['audio_internal_phase_law_v13_causal_reentry_direct_joint_rows'].get('target_row_candidate_count')} "
                f"of "
                f"{evidence['audio_internal_phase_law_v13_causal_reentry_direct_joint_rows'].get('target_row_count')}, "
                f"and candidate-class phase-support "
                f"{evidence['audio_internal_phase_law_v13_candidate_classes'].get('phase_support_candidate_count')} "
                f"with strict audio candidates "
                f"{evidence['audio_internal_phase_law_v13_candidate_classes'].get('strict_audio_candidate_count')}; "
                f"the predeclared small-direct/strong-absolute diagnostic class has status "
                f"{evidence['audio_internal_phase_law_predeclared_diagnostic_class'].get('status')} "
                f"with diagnostic rows "
                f"{evidence['audio_internal_phase_law_predeclared_diagnostic_class'].get('diagnostic_count')} "
                f"across "
                f"{evidence['audio_internal_phase_law_predeclared_diagnostic_class'].get('total_row_count')} "
                f"rows, strict audio rows "
                f"{evidence['audio_internal_phase_law_predeclared_diagnostic_class'].get('strict_audio_count')}, "
                f"best track by count "
                f"{evidence['audio_internal_phase_law_predeclared_diagnostic_class'].get('best_track_by_count')}, "
                f"and best row "
                f"{evidence['audio_internal_phase_law_predeclared_diagnostic_class'].get('best_track')} / "
                f"{evidence['audio_internal_phase_law_predeclared_diagnostic_class'].get('best_run')}; "
                f"audio-delta objective scout status is {evidence['audio_delta_objective_scout'].get('status')} "
                f"with best candidate {evidence['audio_delta_objective_scout'].get('best_candidate')} "
                f"corr delta {evidence['audio_delta_objective_scout'].get('best_primary_corr_delta')}; "
                f"relsig-base scout status is {evidence['audio_delta_objective_scout_relsig'].get('status')}. "
                "This is preservation, seed-policy, and phase-change evidence, not autonomous continuation evidence."
            ),
        },
        "phase_only_gauge_invariance": {
            "status": evidence["phase_gauge"]["status"],
            "reason": f"Max internal metric drift is {evidence['phase_gauge'].get('max_abs_metric_delta')}.",
        },
        "unit_phasor_runtime_contract": {
            "status": evidence["unit_phasor_contract"]["status"],
            "reason": (
                f"Post-fix max phasor norm error is {evidence['unit_phasor_contract'].get('max_phasor_norm_error')} "
                f"across {evidence['unit_phasor_contract'].get('active_surfaces')}; "
                f"broad WAV direct/audio supplemental errors are "
                f"{evidence['unit_phasor_broad_direct_export'].get('max_phasor_norm_error')} / "
                f"{evidence['unit_phasor_broad_audio_continuation'].get('max_phasor_norm_error')}; "
                f"pre-fix packet-seed failure max error was {evidence['unit_phasor_contract'].get('pre_fix_max_phasor_norm_error')}; "
                f"negative control detected={evidence['unit_phasor_contract'].get('negative_control_detected')}."
            ),
        },
        "lattice_substrate": {
            "status": evidence["substrate_lattice"]["status"],
            "reason": "Synthetic-only substrate ablation suggests neighborhood topology is a weak regularizer, not the primary branch-survival cause.",
        },
        "branch_probe_hygiene": {
            "status": evidence["packet_alias_contract"]["status"],
            "reason": (
                f"Branch probe alias audit covers {evidence['packet_alias_contract'].get('num_rows')} packet/nested rows; "
                f"max input mutation {evidence['packet_alias_contract'].get('max_input_delta')} "
                f"and shared storage {evidence['packet_alias_contract'].get('any_shared_storage')}."
            ),
        },
        "ramanujan_prior": {
            "status": evidence["q_basis"]["status"],
            "reason": "qtrace-only matches the Ramanujan baseline in both standalone q-basis and unified mixed-seed comparisons.",
        },
        "q_trace_bookkeeping": {
            "status": "child_survival_qtrace_supported",
            "reason": "qtrace history/relation gain are inert here, but child qtrace survival weight materially affects branch/writeback/loss.",
        },
        "hardy_littlewood_arc": {
            "status": "strongly_causal_for_current_branch_path",
            "reason": (
                "Flattening/zeroing arc or promotability kills branch/writeback across q controls. "
                f"Extended control status is {evidence['arc_q_control_claim'].get('status')} with "
                f"no-q-relation arc gate {evidence['arc_q_control_claim'].get('no_q_relation_arc_gate')} "
                f"and Ramanujan/qtrace/phase parent spread "
                f"{evidence['arc_q_control_claim'].get('ramanujan_qtrace_phase_parent_spread')}."
            ),
        },
        "arc_q_interaction": {
            "status": "arc_necessary_q_basis_modulates_strength",
            "reason": "Arc/promotability is necessary across q families and relation kernels; q basis still modulates strength, law-family diversity, and compact/uniform control quality.",
        },
        "native_branch_ontology": {
            "status": evidence["nested_branch_ontology"]["status"],
            "reason": "Readout response and raw carry are nonzero, but nested sibling and active child survival are zero.",
        },
        "child_writeback_causal_path": {
            "status": evidence["child_writeback"]["status"],
            "reason": "Support-only is parent-phase inert; phase delta/floor creates small naked branch with decorative risk.",
        },
        "dense_relational_signature": {
            "status": dense_relational_signature_status(evidence),
            "reason": (
                "Dense-vs-explicit metadata harness is now separate. "
                f"Verdict is {evidence['dense_signature_claim'].get('status')}; "
                f"stability delta {evidence['dense_signature_claim'].get('stability_delta_dense_minus_metadata')}, "
                f"separation delta {evidence['dense_signature_claim'].get('separation_delta_dense_minus_metadata')}, "
                f"dense wins={evidence['dense_signature_claim'].get('dense_wins')}. "
                f"Failure-mode status is {evidence['dense_signature_failure_modes'].get('status')}; "
                f"matryoshka readiness is {evidence['dense_signature_failure_modes'].get('matryoshka_readiness_status')}. "
                f"Contract audit says learned body status is {evidence['relational_signature_contract_audit'].get('learned_body_status')}. "
                f"Learning module smoke is {evidence['relational_signature_learning_contract'].get('status')} "
                f"with runtime effect {evidence['relational_signature_learning_contract'].get('runtime_effect')}. "
                f"Learned scout status is {evidence['learned_signature_scout'].get('status')}; "
                f"learned separation delta vs metadata is {evidence['learned_signature_scout'].get('separation_delta_learned_minus_metadata')}, "
                f"stability delta is {evidence['learned_signature_scout'].get('stability_delta_learned_minus_metadata')}. "
                f"Scout compare status is {evidence['learned_signature_scout_compare'].get('status')} "
                f"with strict candidates {evidence['learned_signature_scout_compare'].get('candidate_count')}/"
                f"{evidence['learned_signature_scout_compare'].get('row_count')}, "
                f"two-axis {evidence['learned_signature_scout_compare'].get('two_axis_candidate_count')}/"
                f"{evidence['learned_signature_scout_compare'].get('row_count')}, and stability+anti-collapse "
                f"{evidence['learned_signature_scout_compare'].get('stability_anticollapse_candidate_count')}/"
                f"{evidence['learned_signature_scout_compare'].get('row_count')}. "
                f"Heldout candidates are {evidence['learned_signature_scout_compare'].get('heldout_candidate_count')}/"
                f"{evidence['learned_signature_scout_compare'].get('heldout_row_count')}; heldout two-axis "
                f"{evidence['learned_signature_scout_compare'].get('heldout_two_axis_candidate_count')}/"
                f"{evidence['learned_signature_scout_compare'].get('heldout_row_count')}. "
                f"Best in-sample seeded family candidate fraction is "
                f"{(evidence['learned_signature_scout_compare'].get('seed_stability') or {}).get('best_seeded_family', {}).get('candidate_seed_fraction')}; "
                f"best heldout seeded family candidate fraction is "
                f"{(evidence['learned_signature_scout_compare'].get('heldout_seed_stability') or {}).get('best_seeded_family', {}).get('candidate_seed_fraction')}. "
                f"Split-suite status is {evidence['learned_signature_split_suite'].get('status')} "
                f"with candidates {evidence['learned_signature_split_suite'].get('candidate_count')}/"
                f"{evidence['learned_signature_split_suite'].get('split_count')}, seed fraction "
                f"{evidence['learned_signature_split_suite'].get('seed_candidate_fraction')}, and partition fraction "
                f"{evidence['learned_signature_split_suite'].get('partition_candidate_fraction')}. "
                f"Split-suite compare status is {evidence['learned_signature_split_suite_compare'].get('status')} "
                f"with robust candidates {evidence['learned_signature_split_suite_compare'].get('robust_candidate_count')}/"
                f"{evidence['learned_signature_split_suite_compare'].get('row_count')}, redacted suites "
                f"{evidence['learned_signature_split_suite_compare'].get('redacted_suite_count')}, "
                f"redacted candidates {evidence['learned_signature_split_suite_compare'].get('redacted_candidate_count')}, "
                f"and best profile "
                f"{evidence['learned_signature_split_suite_compare'].get('best_profile')}. "
                f"Learned-signature case failure atlas status is "
                f"{evidence['learned_signature_case_failures'].get('status')} with worst cases "
                f"{[row.get('case') for row in evidence['learned_signature_case_failures'].get('worst_cases', [])[:5]]}. "
                f"Operator-tail predictive probe compare is "
                f"{evidence['signature_operator_tail_probe_compare'].get('status')} with learned wins "
                f"{evidence['signature_operator_tail_probe_compare'].get('learned_win_count')}/"
                f"{evidence['signature_operator_tail_probe_compare'].get('row_count')}. "
                f"Future-law usefulness probe is {evidence['signature_future_law_probe'].get('status')} "
                f"with learned wins {evidence['signature_future_law_probe'].get('learned_win_count')}/"
                f"{evidence['signature_future_law_probe'].get('target_count')} and learned-vs-non-law wins "
                f"{evidence['signature_future_law_probe'].get('learned_beats_non_law_count')}. "
                f"Tail incremental usefulness is "
                f"{evidence['signature_tail_incremental_usefulness'].get('status')} with tail wins "
                f"{evidence['signature_tail_incremental_usefulness'].get('tail_win_count')}/"
                f"{evidence['signature_tail_incremental_usefulness'].get('target_count')}, "
                f"tail-vs-prefix wins "
                f"{evidence['signature_tail_incremental_usefulness'].get('tail_beats_prefix_count')}, "
                f"tail-vs-permuted wins "
                f"{evidence['signature_tail_incremental_usefulness'].get('tail_beats_permuted_count')}, "
                f"tail-vs-cross-case wins "
                f"{evidence['signature_tail_incremental_usefulness'].get('tail_beats_cross_case_count')}, "
                f"and tail-vs-random wins "
                f"{evidence['signature_tail_incremental_usefulness'].get('tail_beats_random_count')}; "
                f"negative-control wins "
                f"{evidence['signature_tail_incremental_usefulness'].get('control_win_count')}."
            ),
        },
        "semantic_projector": {
            "status": evidence["semantic_projector_contract"]["status"],
            "reason": (
                "Five-control schema audit is available but not semantic proof. "
                f"Import ok={evidence['semantic_projector_contract'].get('import_ok')}; "
                f"projection path={evidence['semantic_projector_contract'].get('projection_path')}; "
                f"schema exact={evidence['semantic_projector_contract'].get('control_schema_exact')}."
            ),
        },
    }
    return {
        "runtime": "circleworld_proto",
        "schema": "rafa_claim_evidence_v0",
        "artifacts": {key: str(path) for key, path in artifacts.items()},
        "claims": claims,
        "evidence": evidence,
    }


def _fmt(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def write_markdown(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# RAFA Claim Evidence Summary",
        "",
        "Generated from current tokenburst artifacts. Statuses are claim-local, not a global promotion verdict.",
        "",
        "## Claim Verdicts",
        "",
        "| claim | status | reason |",
        "|---|---|---|",
    ]
    for claim, block in payload["claims"].items():
        lines.append(f"| `{claim}` | `{block.get('status')}` | {block.get('reason')} |")

    child = payload["evidence"]["child_writeback"]
    lines.extend(
        [
            "",
            "## Child Writeback",
            "",
            "| variant | phase-only | parent | child | phase delta | support wb | naked phase | naked decorative | major gain |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in child.get("rows", []):
        lines.append(
            "| {variant} | {phase} | {parent} | {child} | {delta} | {support} | {naked} | {decor} | {major} |".format(
                variant=row["variant"],
                phase=_fmt(row.get("phase_only_real_branch_fraction")),
                parent=_fmt(row.get("mean_parent_real_branch_fraction")),
                child=_fmt(row.get("mean_child_real_branch_fraction")),
                delta=_fmt(row.get("mean_child_phase_writeback_delta_mass")),
                support=_fmt(row.get("mean_child_support_writeback_mass")),
                naked=_fmt(row.get("naked_phase_only_real_branch_fraction")),
                decor=_fmt(row.get("naked_decorative_slot2_low_phase_fraction")),
                major=_fmt(row.get("mean_major_gain")),
            )
        )

    nested = payload["evidence"]["nested_branch_ontology"]
    lines.extend(
        [
            "",
            "## Nested Branch Ontology",
            "",
            f"- status: `{nested.get('status')}`",
            f"- readout sibling response: `{_fmt(nested.get('mean_readout_sibling_response'))}`",
            f"- raw same-child carry: `{_fmt(nested.get('mean_branch_identity_carry'))}`",
            f"- qualified carry: `{_fmt(nested.get('mean_branch_identity_qualified_carry'))}`",
            f"- nested sibling fraction: `{_fmt(nested.get('mean_nested_sibling_fraction'))}`",
            f"- child active fraction: `{_fmt(nested.get('mean_child_active_fraction'))}`",
        ]
    )

    gauge = payload["evidence"]["phase_gauge"]
    lines.extend(
        [
            "",
            "## Phase Gauge",
            "",
            f"- status: `{gauge.get('status')}`",
            f"- max absolute metric delta: `{_fmt(gauge.get('max_abs_metric_delta'))}`",
            f"- transform count: `{gauge.get('num_transforms')}`",
        ]
    )

    unit = payload["evidence"]["unit_phasor_contract"]
    lines.extend(
        [
            "",
            "## Unit-Phasor Contract",
            "",
            f"- status: `{unit.get('status')}`",
            f"- active surfaces: `{', '.join(unit.get('active_surfaces') or [])}`",
            f"- post-fix max phasor norm error: `{_fmt(unit.get('max_phasor_norm_error'))}`",
            f"- checked phasor tensors: `{unit.get('num_checked_phasors')}`",
            f"- norm violations: `{unit.get('num_norm_violations')}`",
            f"- hidden magnitude channel detected: `{unit.get('hidden_magnitude_channel_detected')}`",
            f"- pre-fix packet-seed failure detected: `{unit.get('prefix_failure_detected_before_fix')}`",
            f"- pre-fix max phasor norm error: `{_fmt(unit.get('pre_fix_max_phasor_norm_error'))}`",
            f"- negative control enabled: `{unit.get('negative_control_enabled')}`",
            f"- negative control detected: `{unit.get('negative_control_detected')}`",
            f"- negative control max phasor norm error: `{_fmt(unit.get('negative_control_max_phasor_norm_error'))}`",
            f"- raw mixture cancellation status: `{unit.get('raw_mixture_cancellation_status')}`",
            f"- minimum raw mixture norm: `{_fmt(unit.get('min_raw_mixture_norm'))}`",
            "",
            "| surface | cases | passing | max norm error | mean norm error | min finite | min raw mix | status |",
            "|---|---:|---:|---:|---:|---:|---:|---|",
        ]
    )
    for row in unit.get("surface_rows", []):
        lines.append(
            "| {surface} | {cases} | {passing} | {maxerr} | {meanerr} | {finite} | {rawmix} | {status} |".format(
                surface=row.get("surface"),
                cases=_fmt(row.get("num_cases")),
                passing=_fmt(row.get("passing_cases")),
                maxerr=_fmt(row.get("max_norm_error")),
                meanerr=_fmt(row.get("mean_norm_error")),
                finite=_fmt(row.get("min_finite_fraction")),
                rawmix=_fmt(row.get("min_raw_mixture_norm")),
                status=row.get("status"),
            )
        )
    broad_direct = payload["evidence"].get("unit_phasor_broad_direct_export", {})
    broad_audio = payload["evidence"].get("unit_phasor_broad_audio_continuation", {})
    lines.extend(
        [
            "",
            "Supplemental broad-WAV surfaces:",
            "",
            "| surface | status | checked tensors | max norm error | mean norm error | norm violations |",
            "|---|---|---:|---:|---:|---:|",
            "| broad_direct_export | {status} | {count} | {maxerr} | {meanerr} | {violations} |".format(
                status=broad_direct.get("status"),
                count=_fmt(broad_direct.get("num_checked_phasors")),
                maxerr=_fmt(broad_direct.get("max_phasor_norm_error")),
                meanerr=_fmt(broad_direct.get("mean_abs_norm_error")),
                violations=_fmt(broad_direct.get("num_norm_violations")),
            ),
            "| broad_audio_continuation | {status} | {count} | {maxerr} | {meanerr} | {violations} |".format(
                status=broad_audio.get("status"),
                count=_fmt(broad_audio.get("num_checked_phasors")),
                maxerr=_fmt(broad_audio.get("max_phasor_norm_error")),
                meanerr=_fmt(broad_audio.get("mean_abs_norm_error")),
                violations=_fmt(broad_audio.get("num_norm_violations")),
            ),
        ]
    )
    lines.extend(
        [
            "",
            "| boundary | checked tensors | max norm error | mean norm error | min finite | status | worst path |",
            "|---|---:|---:|---:|---:|---|---|",
        ]
    )
    for row in unit.get("rows", []):
        lines.append(
            "| {boundary} | {count} | {maxerr} | {meanerr} | {finite} | {status} | `{worst}` |".format(
                boundary=row.get("boundary"),
                count=_fmt(row.get("checked_tensors")),
                maxerr=_fmt(row.get("max_norm_error")),
                meanerr=_fmt(row.get("mean_norm_error")),
                finite=_fmt(row.get("min_finite_fraction")),
                status=row.get("status"),
                worst=row.get("worst_path"),
            )
        )

    alias = payload["evidence"]["packet_alias_contract"]
    lines.extend(
        [
            "",
            "## Packet Alias Contract",
            "",
            f"- status: `{alias.get('status')}`",
            f"- max input delta: `{_fmt(alias.get('max_input_delta'))}`",
            f"- max output delta: `{_fmt(alias.get('max_output_delta'))}`",
            f"- max output norm error: `{_fmt(alias.get('max_output_norm_error'))}`",
            f"- any shared storage: `{alias.get('any_shared_storage')}`",
            "",
            "| kind | input delta | sibling delta | output delta | norm error | shares storage | status |",
            "|---|---:|---:|---:|---:|---|---|",
        ]
    )
    for row in alias.get("rows", []):
        lines.append(
            "| {kind} | {input_delta} | {sibling_delta} | {output_delta} | {norm_error} | {shares} | {status} |".format(
                kind=row.get("kind"),
                input_delta=_fmt(row.get("input_max_abs_delta")),
                sibling_delta=_fmt(row.get("sibling_max_abs_delta")),
                output_delta=_fmt(row.get("output_max_abs_delta")),
                norm_error=_fmt(row.get("output_max_norm_error")),
                shares=row.get("input_output_share_storage"),
                status="pass" if row.get("passed") else "fail",
            )
        )

    q_basis = payload["evidence"]["q_basis"]
    lines.extend(
        [
            "",
            "## q-Basis",
            "",
            "| variant | child | phase-only | law families | major gain | naked phase |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in q_basis.get("rows", []):
        lines.append(
            "| {variant} | {child} | {phase} | {law} | {major} | {naked} |".format(
                variant=row["variant"],
                child=_fmt(row.get("child_branch")),
                phase=_fmt(row.get("phase_only_branch")),
                law=_fmt(row.get("law_families")),
                major=_fmt(row.get("major_gain")),
                naked=_fmt(row.get("naked_phase_only_branch")),
            )
        )

    audio = payload["evidence"]["audio_continuation"]
    lines.extend(
        [
            "",
            "## Audio Continuation",
            "",
            f"- status: `{audio.get('status')}`",
            f"- future target magnitude reused: `{audio.get('future_target_magnitude_reused')}`",
            f"- future target phase reused: `{audio.get('future_target_phase_reused')}`",
            "",
            "| mode | phase seed | method | corr | mae | mse | loop peak |",
            "|---|---|---|---:|---:|---:|---:|",
        ]
    )
    for row in audio.get("rows", []):
        lines.append(
            "| {mode} | {seed} | {method} | {corr} | {mae} | {mse} | {loop} |".format(
                mode=row.get("magnitude_mode"),
                seed=row.get("phase_seed_policy"),
                method=row.get("method"),
                corr=_fmt(row.get("mean_corr")),
                mae=_fmt(row.get("mean_mae")),
                mse=_fmt(row.get("mean_mse")),
                loop=_fmt(row.get("mean_loop_autocorr_peak")),
            )
        )

    audio_copyphase = payload["evidence"].get("audio_continuation_copyphase", {})
    lines.extend(
        [
            "",
            "## Audio Continuation Copyphase Seed",
            "",
            f"- status: `{audio_copyphase.get('status')}`",
            f"- future target magnitude reused: `{audio_copyphase.get('future_target_magnitude_reused')}`",
            f"- future target phase reused: `{audio_copyphase.get('future_target_phase_reused')}`",
            "",
            "| mode | phase seed | method | corr | mae | mse | loop peak |",
            "|---|---|---|---:|---:|---:|---:|",
        ]
    )
    for row in audio_copyphase.get("rows", []) or []:
        lines.append(
            "| {mode} | {seed} | {method} | {corr} | {mae} | {mse} | {loop} |".format(
                mode=row.get("magnitude_mode"),
                seed=row.get("phase_seed_policy"),
                method=row.get("method"),
                corr=_fmt(row.get("mean_corr")),
                mae=_fmt(row.get("mean_mae")),
                mse=_fmt(row.get("mean_mse")),
                loop=_fmt(row.get("mean_loop_autocorr_peak")),
            )
        )

    broad_benchmark = payload["evidence"].get("broad18_benchmark", {})
    lines.extend(
        [
            "",
            "## Broad 18-WAV Benchmark",
            "",
            f"- status: `{broad_benchmark.get('status')}`",
            f"- cases: `{_fmt(broad_benchmark.get('case_count'))}`",
            f"- clip seconds: `{_fmt(broad_benchmark.get('clip_seconds'))}`",
            f"- mean corr: `{_fmt(broad_benchmark.get('mean_corr'))}`",
            f"- mean MAE: `{_fmt(broad_benchmark.get('mean_mae'))}`",
            f"- continuity delta band: `{broad_benchmark.get('continuity_delta_band')}`",
            f"- continuity max abs mean delta: `{_fmt(broad_benchmark.get('continuity_max_abs_mean_delta'))}`",
            f"- Circleworld recurrence: `{_fmt(broad_benchmark.get('circleworld_recurrence_score'))}` / `{broad_benchmark.get('circleworld_recurrence_band')}`",
            f"- reference recurrence: `{_fmt(broad_benchmark.get('reference_recurrence_score'))}` / `{broad_benchmark.get('reference_recurrence_band')}`",
            f"- recurrence score delta: `{_fmt(broad_benchmark.get('recurrence_score_delta'))}`",
            "",
            "| worst case | corr | MAE | MSE |",
            "|---|---:|---:|---:|",
        ]
    )
    for row in broad_benchmark.get("worst_corr_rows", []) or []:
        lines.append(
            "| {name} | {corr} | {mae} | {mse} |".format(
                name=row.get("name"),
                corr=_fmt(row.get("corr")),
                mae=_fmt(row.get("mae")),
                mse=_fmt(row.get("mse")),
            )
        )

    method_compare = payload["evidence"].get("audio_method_compare", {})
    lines.extend(
        [
            "",
            "## Audio Carrier-Lock Compare",
            "",
            f"- status: `{method_compare.get('status')}`",
            "",
            "| run | mode | phase seed | carrier | cases | future mag | future phase | cw/carrier corr | cw/carrier MSE | cw-best corr | cw-best MSE | corr wins | mse wins | lock score | status |",
            "|---|---|---|---|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---|",
        ]
    )
    for row in method_compare.get("rows", []) or []:
        lines.append(
            "| {label} | {mode} | {seed} | {carrier} | {cases} | `{fmag}` | `{fphase}` | {ccorr} | {cmse} | {bcorr} | {bmse} | {wcorr} | {wmse} | {lock} | {status} |".format(
                label=row.get("label"),
                mode=row.get("magnitude_mode"),
                seed=row.get("phase_seed_policy"),
                carrier=row.get("carrier_method"),
                cases=_fmt(row.get("case_count")),
                fmag=row.get("future_target_magnitude_reused"),
                fphase=row.get("future_target_phase_reused"),
                ccorr=_fmt(row.get("mean_cw_vs_carrier_corr")),
                cmse=_fmt(row.get("mean_cw_vs_carrier_mse")),
                bcorr=_fmt(row.get("mean_cw_minus_best_baseline_corr")),
                bmse=_fmt(row.get("mean_cw_minus_best_baseline_mse")),
                wcorr=_fmt(row.get("cw_win_fraction_corr")),
                wmse=_fmt(row.get("cw_win_fraction_mse")),
                lock=_fmt(row.get("carrier_lock_score")),
                status=row.get("status"),
            )
        )

    method_compare_copyphase = payload["evidence"].get("audio_method_compare_copyphase", {})
    lines.extend(
        [
            "",
            "## Audio Copyphase Carrier-Lock Compare",
            "",
            f"- status: `{method_compare_copyphase.get('status')}`",
            "",
            "| run | mode | phase seed | carrier | cases | future mag | future phase | cw/carrier corr | cw/carrier MSE | cw-best corr | cw-best MSE | corr wins | mse wins | lock score | status |",
            "|---|---|---|---|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---|",
        ]
    )
    for row in method_compare_copyphase.get("rows", []) or []:
        lines.append(
            "| {label} | {mode} | {seed} | {carrier} | {cases} | `{fmag}` | `{fphase}` | {ccorr} | {cmse} | {bcorr} | {bmse} | {wcorr} | {wmse} | {lock} | {status} |".format(
                label=row.get("label"),
                mode=row.get("magnitude_mode"),
                seed=row.get("phase_seed_policy"),
                carrier=row.get("carrier_method"),
                cases=_fmt(row.get("case_count")),
                fmag=row.get("future_target_magnitude_reused"),
                fphase=row.get("future_target_phase_reused"),
                ccorr=_fmt(row.get("mean_cw_vs_carrier_corr")),
                cmse=_fmt(row.get("mean_cw_vs_carrier_mse")),
                bcorr=_fmt(row.get("mean_cw_minus_best_baseline_corr")),
                bmse=_fmt(row.get("mean_cw_minus_best_baseline_mse")),
                wcorr=_fmt(row.get("cw_win_fraction_corr")),
                wmse=_fmt(row.get("cw_win_fraction_mse")),
                lock=_fmt(row.get("carrier_lock_score")),
                status=row.get("status"),
            )
        )

    phase_influence = payload["evidence"].get("audio_phase_influence", {})
    lines.extend(
        [
            "",
            "## Audio Phase Influence",
            "",
            f"- status: `{phase_influence.get('status')}`",
            f"- cases: `{_fmt(phase_influence.get('case_count'))}`",
            f"- mean future abs phase delta: `{_fmt(phase_influence.get('mean_future_abs_phase_delta'))}`",
            f"- mean future cos phase delta: `{_fmt(phase_influence.get('mean_future_cos_phase_delta'))}`",
            f"- future target magnitude reused: `{phase_influence.get('future_target_magnitude_reused')}`",
            "",
            "| mode | gain | corr | MSE | vs gain0 corr | corr delta | MSE delta | corr wins | MSE wins | verdict |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
        ]
    )
    for row in phase_influence.get("rows", []) or []:
        lines.append(
            "| {mode} | {gain} | {corr} | {mse} | {vcorr} | {dcorr} | {dmse} | {wcorr} | {wmse} | {verdict} |".format(
                mode=row.get("magnitude_mode"),
                gain=_fmt(row.get("phase_gain")),
                corr=_fmt(row.get("mean_target_corr")),
                mse=_fmt(row.get("mean_target_mse")),
                vcorr=_fmt(row.get("mean_vs_gain0_corr")),
                dcorr=_fmt(row.get("mean_target_corr_delta_vs_gain0")),
                dmse=_fmt(row.get("mean_target_mse_delta_vs_gain0")),
                wcorr=_fmt(row.get("target_corr_win_fraction_vs_gain0")),
                wmse=_fmt(row.get("target_mse_win_fraction_vs_gain0")),
                verdict=row.get("verdict"),
            )
        )

    phase_seed = payload["evidence"].get("audio_phase_seed", {})
    lines.extend(
        [
            "",
            "## Audio Phase Seed Policy",
            "",
            f"- status: `{phase_seed.get('status')}`",
            f"- cases: `{_fmt(phase_seed.get('case_count'))}`",
            f"- best seed policy: `{phase_seed.get('best_seed_policy')}` / `{phase_seed.get('best_seed_magnitude_mode')}`",
            f"- best seed corr: `{_fmt(phase_seed.get('best_seed_target_corr'))}`",
            f"- best Circleworld corr: `{_fmt(phase_seed.get('best_circle_target_corr'))}`",
            f"- best Circleworld minus seed corr: `{_fmt(phase_seed.get('best_circle_minus_seed_corr'))}`",
            f"- max seed corr spread: `{_fmt(phase_seed.get('max_seed_corr_spread'))}`",
            f"- max seed MSE spread: `{_fmt(phase_seed.get('max_seed_mse_spread'))}`",
            f"- future target magnitude reused: `{phase_seed.get('future_target_magnitude_reused')}`",
            f"- future target phase reused: `{phase_seed.get('future_target_phase_reused')}`",
            "",
            "| mode | seed policy | seed corr | circle corr | delta corr | seed MSE | circle MSE | delta MSE | circle/seed corr | verdict |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|---|",
        ]
    )
    for row in phase_seed.get("rows", []) or []:
        lines.append(
            "| {mode} | {policy} | {seed_corr} | {circle_corr} | {dcorr} | {seed_mse} | {circle_mse} | {dmse} | {vseed} | {verdict} |".format(
                mode=row.get("magnitude_mode"),
                policy=row.get("phase_seed_policy"),
                seed_corr=_fmt(row.get("mean_seed_target_corr")),
                circle_corr=_fmt(row.get("mean_circle_target_corr")),
                dcorr=_fmt(row.get("mean_circle_minus_seed_corr")),
                seed_mse=_fmt(row.get("mean_seed_target_mse")),
                circle_mse=_fmt(row.get("mean_circle_target_mse")),
                dmse=_fmt(row.get("mean_circle_minus_seed_mse")),
                vseed=_fmt(row.get("mean_circle_vs_seed_corr")),
                verdict=row.get("verdict"),
            )
        )

    phase_seed_weighted = payload["evidence"].get("audio_phase_seed_weighted_fit", {})
    lines.extend(
        [
            "",
            "## Audio Phase Seed Policy Weighted/Fit Expansion",
            "",
            f"- status: `{phase_seed_weighted.get('status')}`",
            f"- cases: `{_fmt(phase_seed_weighted.get('case_count'))}`",
            f"- best seed policy: `{phase_seed_weighted.get('best_seed_policy')}` / `{phase_seed_weighted.get('best_seed_magnitude_mode')}`",
            f"- best seed corr: `{_fmt(phase_seed_weighted.get('best_seed_target_corr'))}`",
            f"- best Circleworld corr: `{_fmt(phase_seed_weighted.get('best_circle_target_corr'))}`",
            f"- best Circleworld minus seed corr: `{_fmt(phase_seed_weighted.get('best_circle_minus_seed_corr'))}`",
            f"- max seed corr spread: `{_fmt(phase_seed_weighted.get('max_seed_corr_spread'))}`",
            f"- max seed MSE spread: `{_fmt(phase_seed_weighted.get('max_seed_mse_spread'))}`",
            f"- future target magnitude reused: `{phase_seed_weighted.get('future_target_magnitude_reused')}`",
            f"- future target phase reused: `{phase_seed_weighted.get('future_target_phase_reused')}`",
            "",
            "| mode | seed policy | seed corr | circle corr | delta corr | seed MSE | circle MSE | delta MSE | circle/seed corr | verdict |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|---|",
        ]
    )
    for row in phase_seed_weighted.get("rows", []) or []:
        lines.append(
            "| {mode} | {policy} | {seed_corr} | {circle_corr} | {dcorr} | {seed_mse} | {circle_mse} | {dmse} | {vseed} | {verdict} |".format(
                mode=row.get("magnitude_mode"),
                policy=row.get("phase_seed_policy"),
                seed_corr=_fmt(row.get("mean_seed_target_corr")),
                circle_corr=_fmt(row.get("mean_circle_target_corr")),
                dcorr=_fmt(row.get("mean_circle_minus_seed_corr")),
                seed_mse=_fmt(row.get("mean_seed_target_mse")),
                circle_mse=_fmt(row.get("mean_circle_target_mse")),
                dmse=_fmt(row.get("mean_circle_minus_seed_mse")),
                vseed=_fmt(row.get("mean_circle_vs_seed_corr")),
                verdict=row.get("verdict"),
            )
        )

    delta_probe = payload["evidence"].get("audio_circle_delta_probe", {})
    best_corr = delta_probe.get("best_corr_delta_row") if isinstance(delta_probe.get("best_corr_delta_row"), dict) else {}
    best_mse = delta_probe.get("best_mse_delta_row") if isinstance(delta_probe.get("best_mse_delta_row"), dict) else {}
    lines.extend(
        [
            "",
            "## Audio Circle Delta Probe",
            "",
            f"- status: `{delta_probe.get('status')}`",
            f"- raw status: `{delta_probe.get('raw_status')}`",
            f"- cases: `{_fmt(delta_probe.get('case_count'))}`",
            f"- seed policy: `{delta_probe.get('phase_seed_policy')}`",
            f"- best corr delta: `{_fmt(delta_probe.get('best_corr_delta'))}` from `{best_corr.get('magnitude_mode')}` / `{best_corr.get('mask_mode')}` / gain `{_fmt(best_corr.get('gain'))}`",
            f"- best MSE delta: `{_fmt(delta_probe.get('best_mse_delta'))}` from `{best_mse.get('magnitude_mode')}` / `{best_mse.get('mask_mode')}` / gain `{_fmt(best_mse.get('gain'))}`",
            f"- future target magnitude reused: `{delta_probe.get('future_target_magnitude_reused')}`",
            f"- future target phase reused: `{delta_probe.get('future_target_phase_reused')}`",
            "",
            "| mode | mask | gain | corr | MSE | vs gain0 corr | corr delta | MSE delta | corr wins | MSE wins | active abs delta | verdict |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
        ]
    )
    for row in delta_probe.get("rows", []) or []:
        lines.append(
            "| {mode} | {mask} | {gain} | {corr} | {mse} | {vcorr} | {dcorr} | {dmse} | {wcorr} | {wmse} | {adelta} | {verdict} |".format(
                mode=row.get("magnitude_mode"),
                mask=row.get("mask_mode"),
                gain=_fmt(row.get("gain")),
                corr=_fmt(row.get("mean_target_corr")),
                mse=_fmt(row.get("mean_target_mse")),
                vcorr=_fmt(row.get("mean_vs_gain0_corr")),
                dcorr=_fmt(row.get("mean_target_corr_delta_vs_gain0")),
                dmse=_fmt(row.get("mean_target_mse_delta_vs_gain0")),
                wcorr=_fmt(row.get("target_corr_win_fraction_vs_gain0")),
                wmse=_fmt(row.get("target_mse_win_fraction_vs_gain0")),
                adelta=_fmt(row.get("mean_active_bin_abs_phase_delta")),
                verdict=row.get("verdict"),
            )
        )

    mechanism_probe = payload["evidence"].get("audio_delta_mechanism_probe", {})
    best_corr = (
        mechanism_probe.get("best_corr_delta_row")
        if isinstance(mechanism_probe.get("best_corr_delta_row"), dict)
        else {}
    )
    best_mse = (
        mechanism_probe.get("best_mse_delta_row")
        if isinstance(mechanism_probe.get("best_mse_delta_row"), dict)
        else {}
    )
    lines.extend(
        [
            "",
            "## Audio Delta Mechanism Probe",
            "",
            f"- status: `{mechanism_probe.get('status')}`",
            f"- raw status: `{mechanism_probe.get('raw_status')}`",
            f"- cases: `{_fmt(mechanism_probe.get('case_count'))}`",
            f"- seed policy: `{mechanism_probe.get('phase_seed_policy')}`",
            f"- best corr delta: `{_fmt(mechanism_probe.get('best_corr_delta'))}` from `{best_corr.get('magnitude_mode')}` / `{best_corr.get('mask_mode')}` / `{best_corr.get('mechanism')}` / gain `{_fmt(best_corr.get('gain'))}`",
            f"- best MSE delta: `{_fmt(mechanism_probe.get('best_mse_delta'))}` from `{best_mse.get('magnitude_mode')}` / `{best_mse.get('mask_mode')}` / `{best_mse.get('mechanism')}` / gain `{_fmt(best_mse.get('gain'))}`",
            f"- future target magnitude reused: `{mechanism_probe.get('future_target_magnitude_reused')}`",
            f"- future target phase reused: `{mechanism_probe.get('future_target_phase_reused')}`",
            "",
            "| mode | mask | mechanism | gain | corr | MSE | vs gain0 corr | corr delta | median corr delta | MSE delta | corr wins | MSE wins | mask frac | loop | reentry | shaped abs delta | verdict |",
            "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
        ]
    )
    for row in mechanism_probe.get("rows", []) or []:
        lines.append(
            "| {mode} | {mask} | {mechanism} | {gain} | {corr} | {mse} | {vcorr} | {dcorr} | {median} | {dmse} | {wcorr} | {wmse} | {maskfrac} | {loop} | {reentry} | {adelta} | {verdict} |".format(
                mode=row.get("magnitude_mode"),
                mask=row.get("mask_mode"),
                mechanism=row.get("mechanism"),
                gain=_fmt(row.get("gain")),
                corr=_fmt(row.get("mean_target_corr")),
                mse=_fmt(row.get("mean_target_mse")),
                vcorr=_fmt(row.get("mean_vs_gain0_corr")),
                dcorr=_fmt(row.get("mean_target_corr_delta_vs_gain0")),
                median=_fmt(row.get("median_target_corr_delta_vs_gain0")),
                dmse=_fmt(row.get("mean_target_mse_delta_vs_gain0")),
                wcorr=_fmt(row.get("target_corr_win_fraction_vs_gain0")),
                wmse=_fmt(row.get("target_mse_win_fraction_vs_gain0")),
                maskfrac=_fmt(row.get("mean_mask_bin_fraction")),
                loop=_fmt(row.get("mean_loop_autocorr_peak")),
                reentry=_fmt(row.get("mean_first_chunk_reentry")),
                adelta=_fmt(row.get("mean_abs_shaped_delta")),
                verdict=row.get("verdict"),
            )
        )

    mechanism_lockbox = payload["evidence"].get("audio_delta_mechanism_lockbox", {})
    best_corr = (
        mechanism_lockbox.get("best_corr_delta_row")
        if isinstance(mechanism_lockbox.get("best_corr_delta_row"), dict)
        else {}
    )
    best_mse = (
        mechanism_lockbox.get("best_mse_delta_row")
        if isinstance(mechanism_lockbox.get("best_mse_delta_row"), dict)
        else {}
    )
    lines.extend(
        [
            "",
            "## Audio Delta Mechanism Lockbox",
            "",
            f"- status: `{mechanism_lockbox.get('status')}`",
            f"- raw status: `{mechanism_lockbox.get('raw_status')}`",
            f"- cases: `{_fmt(mechanism_lockbox.get('case_count'))}`",
            f"- seed policy: `{mechanism_lockbox.get('phase_seed_policy')}`",
            f"- best corr delta: `{_fmt(mechanism_lockbox.get('best_corr_delta'))}` from `{best_corr.get('magnitude_mode')}` / `{best_corr.get('mask_mode')}` / `{best_corr.get('mechanism')}` / gain `{_fmt(best_corr.get('gain'))}`",
            f"- best corr win fraction: `{_fmt(best_corr.get('target_corr_win_fraction_vs_gain0'))}`",
            f"- best corr median delta: `{_fmt(best_corr.get('median_target_corr_delta_vs_gain0'))}`",
            f"- best MSE delta: `{_fmt(mechanism_lockbox.get('best_mse_delta'))}` from `{best_mse.get('magnitude_mode')}` / `{best_mse.get('mask_mode')}` / `{best_mse.get('mechanism')}` / gain `{_fmt(best_mse.get('gain'))}`",
            f"- future target magnitude reused: `{mechanism_lockbox.get('future_target_magnitude_reused')}`",
            f"- future target phase reused: `{mechanism_lockbox.get('future_target_phase_reused')}`",
            "",
            "| mode | mask | mechanism | gain | corr | MSE | vs gain0 corr | corr delta | median corr delta | MSE delta | corr wins | MSE wins | mask frac | loop | reentry | shaped abs delta | verdict |",
            "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
        ]
    )
    for row in mechanism_lockbox.get("rows", []) or []:
        lines.append(
            "| {mode} | {mask} | {mechanism} | {gain} | {corr} | {mse} | {vcorr} | {dcorr} | {median} | {dmse} | {wcorr} | {wmse} | {maskfrac} | {loop} | {reentry} | {adelta} | {verdict} |".format(
                mode=row.get("magnitude_mode"),
                mask=row.get("mask_mode"),
                mechanism=row.get("mechanism"),
                gain=_fmt(row.get("gain")),
                corr=_fmt(row.get("mean_target_corr")),
                mse=_fmt(row.get("mean_target_mse")),
                vcorr=_fmt(row.get("mean_vs_gain0_corr")),
                dcorr=_fmt(row.get("mean_target_corr_delta_vs_gain0")),
                median=_fmt(row.get("median_target_corr_delta_vs_gain0")),
                dmse=_fmt(row.get("mean_target_mse_delta_vs_gain0")),
                wcorr=_fmt(row.get("target_corr_win_fraction_vs_gain0")),
                wmse=_fmt(row.get("target_mse_win_fraction_vs_gain0")),
                maskfrac=_fmt(row.get("mean_mask_bin_fraction")),
                loop=_fmt(row.get("mean_loop_autocorr_peak")),
                reentry=_fmt(row.get("mean_first_chunk_reentry")),
                adelta=_fmt(row.get("mean_abs_shaped_delta")),
                verdict=row.get("verdict"),
            )
        )

    family_sensitivity = payload["evidence"].get("audio_mechanism_family_sensitivity", {})
    lines.extend(
        [
            "",
            "## Audio Mechanism Family Sensitivity",
            "",
            f"- status: `{family_sensitivity.get('status')}`",
            f"- mechanism: `{family_sensitivity.get('mechanism')}`",
            f"- target row: `{family_sensitivity.get('magnitude_mode')}` / `{family_sensitivity.get('mask_mode')}` / gain `{_fmt(family_sensitivity.get('target_gain'))}`",
            f"- valid WAVs after dedupe/exclusion: `{_fmt(family_sensitivity.get('valid_wav_count'))}`",
            f"- rejected WAVs: `{_fmt(family_sensitivity.get('rejected_wav_count'))}`",
            f"- deduped duplicate stems: `{_fmt(family_sensitivity.get('deduped_duplicate_stem_count'))}`",
            f"- best family: `{family_sensitivity.get('best_family')}` / `{family_sensitivity.get('best_family_status')}`",
            f"- best family mean corr delta: `{_fmt(family_sensitivity.get('best_family_mean_corr_delta'))}`",
            f"- best family median corr delta: `{_fmt(family_sensitivity.get('best_family_median_corr_delta'))}`",
            f"- best family min leave-one-out corr delta: `{_fmt(family_sensitivity.get('best_family_min_leave_one_out_corr_delta'))}`",
            f"- best family corr win fraction: `{_fmt(family_sensitivity.get('best_family_corr_win_fraction'))}`",
            f"- best family positive outlier share: `{_fmt(family_sensitivity.get('best_family_positive_outlier_share'))}`",
            "",
            "| family | status | cases | mean corr d | median corr d | min LOO d | corr wins | mean MSE d | outlier share | top positive | top positive d | top negative | top negative d |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|---|---:|---|---:|",
        ]
    )
    for row in family_sensitivity.get("families", []) or []:
        top_pos = row.get("top_positive_case") if isinstance(row.get("top_positive_case"), dict) else {}
        top_neg = row.get("top_negative_case") if isinstance(row.get("top_negative_case"), dict) else {}
        lines.append(
            "| {family} | {status} | {cases} | {mean} | {median} | {loo} | {wins} | {mse} | {share} | {pos} | {posd} | {neg} | {negd} |".format(
                family=row.get("family"),
                status=row.get("status"),
                cases=_fmt(row.get("case_count")),
                mean=_fmt(row.get("mean_corr_delta")),
                median=_fmt(row.get("median_corr_delta")),
                loo=_fmt(row.get("min_leave_one_out_corr_delta")),
                wins=_fmt(row.get("corr_win_fraction")),
                mse=_fmt(row.get("mean_mse_delta")),
                share=_fmt(row.get("positive_outlier_share")),
                pos=top_pos.get("name"),
                posd=_fmt(top_pos.get("corr_delta")),
                neg=top_neg.get("name"),
                negd=_fmt(top_neg.get("corr_delta")),
            )
        )

    motor_holdout = payload["evidence"].get("audio_electric_motor_holdout", {})
    lines.extend(
        [
            "",
            "## Audio Electric Motor Holdout",
            "",
            f"- status: `{motor_holdout.get('status')}`",
        ]
    )
    if motor_holdout.get("status") != "missing":
        leakage_flags = motor_holdout.get("leakage_flags") if isinstance(motor_holdout.get("leakage_flags"), dict) else {}
        leakage_text = ", ".join(f"{key}={value}" for key, value in sorted(leakage_flags.items())) or "none reported"
        lines.extend(
            [
                f"- valid WAVs after dedupe/exclusion: `{_fmt(motor_holdout.get('valid_wav_count'))}`",
                f"- rejected WAVs: `{_fmt(motor_holdout.get('rejected_wav_count'))}`",
                f"- deduped duplicate stems: `{_fmt(motor_holdout.get('deduped_duplicate_stem_count'))}`",
                f"- razor excluded from exploratory split: `{motor_holdout.get('exclude_razor_from_exploratory')}`",
                f"- withheld razor exploratory count: `{_fmt(motor_holdout.get('withheld_razor_exploratory_count'))}`",
                f"- selected unique paths / occurrences: `{_fmt(motor_holdout.get('selected_unique_paths'))}` / `{_fmt(motor_holdout.get('selected_case_occurrences'))}`",
                f"- selected content-hash reuse count: `{_fmt(motor_holdout.get('selected_content_hash_reuse_count'))}`",
                f"- best no-razor structural family: `{motor_holdout.get('best_nonrazor_motor_family')}` / `{motor_holdout.get('best_nonrazor_motor_status')}` / mean corr delta `{_fmt(motor_holdout.get('best_nonrazor_motor_mean_corr_delta'))}`",
                f"- best razor family: `{motor_holdout.get('best_razor_family')}` / `{motor_holdout.get('best_razor_status')}` / mean corr delta `{_fmt(motor_holdout.get('best_razor_mean_corr_delta'))}`",
                f"- leakage flags: `{leakage_text}`",
                "",
                "| row | status | family | subfamily | mechanism | mode | mask | gain | cases | mean corr d | median corr d | min LOO d | corr wins | mean MSE d | MSE wins | outlier share | top positive | top positive d | top negative | top negative d |",
                "|---|---|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---|---:|",
            ]
        )
        for label, row in [
            ("selected fixed", motor_holdout.get("selected_fixed_mechanism_row")),
            ("best exploratory", motor_holdout.get("best_exploratory_subfamily")),
            ("holdout family", motor_holdout.get("holdout_family_metrics")),
        ]:
            row = row if isinstance(row, dict) else {}
            top_pos = row.get("top_positive_case") if isinstance(row.get("top_positive_case"), dict) else {}
            top_neg = row.get("top_negative_case") if isinstance(row.get("top_negative_case"), dict) else {}
            lines.append(
                "| {label} | {status} | {family} | {subfamily} | {mechanism} | {mode} | {mask} | {gain} | {cases} | {mean} | {median} | {loo} | {wins} | {mse} | {msewins} | {share} | {pos} | {posd} | {neg} | {negd} |".format(
                    label=label,
                    status=row.get("status") or row.get("verdict"),
                    family=row.get("family"),
                    subfamily=row.get("subfamily"),
                    mechanism=row.get("mechanism"),
                    mode=row.get("magnitude_mode"),
                    mask=row.get("mask_mode"),
                    gain=_fmt(row.get("gain")),
                    cases=_fmt(row.get("case_count")),
                    mean=_fmt(row.get("mean_corr_delta")),
                    median=_fmt(row.get("median_corr_delta")),
                    loo=_fmt(row.get("min_leave_one_out_corr_delta")),
                    wins=_fmt(row.get("corr_win_fraction")),
                    mse=_fmt(row.get("mean_mse_delta")),
                    msewins=_fmt(row.get("mse_win_fraction")),
                    share=_fmt(row.get("positive_outlier_share")),
                    pos=top_pos.get("name"),
                    posd=_fmt(top_pos.get("corr_delta")),
                    neg=top_neg.get("name"),
                    negd=_fmt(top_neg.get("corr_delta")),
                )
            )
        if motor_holdout.get("groups"):
            lines.extend(
                [
                    "",
                    "| split | family | status | cases | mean corr d | median corr d | corr wins | mean MSE d | outlier share |",
                    "|---|---|---|---:|---:|---:|---:|---:|---:|",
                ]
            )
            for row in motor_holdout.get("groups", []) or []:
                if not isinstance(row, dict):
                    continue
                lines.append(
                    "| {split} | {family} | {status} | {cases} | {mean} | {median} | {wins} | {mse} | {share} |".format(
                        split=row.get("split"),
                        family=row.get("family"),
                        status=row.get("status"),
                        cases=_fmt(row.get("case_count")),
                        mean=_fmt(row.get("mean_corr_delta")),
                        median=_fmt(row.get("median_corr_delta")),
                        wins=_fmt(row.get("corr_win_fraction")),
                        mse=_fmt(row.get("mean_mse_delta")),
                        share=_fmt(row.get("positive_outlier_share")),
                    )
                )

    steady_manifest = payload["evidence"].get("audio_steady_phenotype_manifest", {})
    lines.extend(
        [
            "",
            "## Audio Steady Phenotype Manifest",
            "",
            f"- status: `{steady_manifest.get('status')}`",
        ]
    )
    if steady_manifest.get("status") != "missing":
        lines.extend(
            [
                f"- selected unique paths / occurrences: `{_fmt(steady_manifest.get('selected_unique_paths'))}` / `{_fmt(steady_manifest.get('selected_case_occurrences'))}`",
                f"- selected unique content hashes: `{_fmt(steady_manifest.get('selected_unique_content_hashes'))}`",
                f"- selection reuse rejection count: `{_fmt(steady_manifest.get('selection_reuse_rejection_count'))}`",
                f"- target leakage detected: `{steady_manifest.get('target_leakage_detected')}`",
                "",
                "| row | status | group | role | cases | mean corr d | median corr d | corr wins | mean MSE d | outlier share |",
                "|---|---|---|---|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for label, row in [
            ("best overall", steady_manifest.get("best_overall_group")),
            ("best steady buzz", steady_manifest.get("best_steady_buzz_group")),
            ("best no-razor motor", steady_manifest.get("best_no_razor_motor_group")),
            ("best synth control", steady_manifest.get("best_synth_control_group")),
            ("best razor probe", steady_manifest.get("best_razor_probe_group")),
        ]:
            row = row if isinstance(row, dict) else {}
            lines.append(
                "| {label} | {status} | {group} | {role} | {cases} | {mean} | {median} | {wins} | {mse} | {share} |".format(
                    label=label,
                    status=row.get("status"),
                    group=row.get("group"),
                    role=row.get("role"),
                    cases=_fmt(row.get("case_count")),
                    mean=_fmt(row.get("mean_corr_delta")),
                    median=_fmt(row.get("median_corr_delta")),
                    wins=_fmt(row.get("corr_win_fraction")),
                    mse=_fmt(row.get("mean_mse_delta")),
                    share=_fmt(row.get("positive_outlier_share")),
                )
            )
        if steady_manifest.get("groups"):
            lines.extend(
                [
                    "",
                    "| group | role | status | cases | mean corr d | median corr d | corr wins | mean MSE d | outlier share |",
                    "|---|---|---|---:|---:|---:|---:|---:|---:|",
                ]
            )
            for row in steady_manifest.get("groups", []) or []:
                if not isinstance(row, dict):
                    continue
                lines.append(
                    "| {group} | {role} | {status} | {cases} | {mean} | {median} | {wins} | {mse} | {share} |".format(
                        group=row.get("group"),
                        role=row.get("role"),
                        status=row.get("status"),
                        cases=_fmt(row.get("case_count")),
                        mean=_fmt(row.get("mean_corr_delta")),
                        median=_fmt(row.get("median_corr_delta")),
                        wins=_fmt(row.get("corr_win_fraction")),
                        mse=_fmt(row.get("mean_mse_delta")),
                        share=_fmt(row.get("positive_outlier_share")),
                    )
                )

    phase_diag = payload["evidence"].get("audio_phase_phenotype_diagnostics", {})
    lines.extend(
        [
            "",
            "## Audio Phase Phenotype Diagnostics",
            "",
            f"- status: `{phase_diag.get('status')}`",
        ]
    )
    if phase_diag.get("status") != "missing":
        lines.extend(
            [
                f"- case count: `{_fmt(phase_diag.get('case_count'))}`",
                f"- strongest non-single-source feature: `{phase_diag.get('strongest_non_single_feature')}` / corr `{_fmt(phase_diag.get('strongest_non_single_corr'))}`",
                f"- prefix phase-velocity coherence corr: `{_fmt(phase_diag.get('prefix_phase_velocity_corr'))}`",
                f"- prefix spectral flux corr: `{_fmt(phase_diag.get('prefix_spectral_flux_corr'))}`",
                f"- mean shaped-delta corr: `{_fmt(phase_diag.get('mean_abs_shaped_delta_corr'))}`",
                "",
                "| feature | corr with corr d | corr with MSE d | mean | std |",
                "|---|---:|---:|---:|---:|",
            ]
        )
        for row in phase_diag.get("top_correlations", []) or []:
            if not isinstance(row, dict):
                continue
            lines.append(
                "| {feature} | {ccorr} | {mcorr} | {mean} | {std} |".format(
                    feature=row.get("feature"),
                    ccorr=_fmt(row.get("pearson_with_corr_delta")),
                    mcorr=_fmt(row.get("pearson_with_mse_delta")),
                    mean=_fmt(row.get("feature_mean")),
                    std=_fmt(row.get("feature_std")),
                )
            )

    baseline_diag = payload["evidence"].get("audio_baseline_stratified_diagnostics", {})
    lines.extend(
        [
            "",
            "## Audio Baseline Stratified Diagnostics",
            "",
                f"- status: `{baseline_diag.get('status')}`",
        ]
    )
    if baseline_diag.get("status") != "missing":
        lines.extend(
            [
                f"- case count: `{_fmt(baseline_diag.get('case_count'))}`",
                f"- excluded single-source probes: `{baseline_diag.get('exclude_single_source')}`",
                f"- promotion gate: `{baseline_diag.get('promotion_gate_status')}`",
                f"- baseline rescue dominated: `{baseline_diag.get('baseline_rescue_dominated')}`",
                f"- non-bad weighted mean corr delta: `{_fmt(baseline_diag.get('non_bad_mean_corr_delta'))}`",
                f"- non-bad weighted corr win fraction: `{_fmt(baseline_diag.get('non_bad_corr_win_fraction'))}`",
                f"- bad/non-bad corr-delta ratio: `{_fmt(baseline_diag.get('bad_to_non_bad_corr_delta_ratio'))}`",
                "",
                "| bin | cases | mean gain0 corr | mean corr d | median corr d | corr wins | mean MSE d | outlier share | top positive | top d |",
                "|---|---:|---:|---:|---:|---:|---:|---:|---|---:|",
            ]
        )
        for row in baseline_diag.get("fixed_bins", []) or []:
            if not isinstance(row, dict):
                continue
            top = row.get("top_positive_case") if isinstance(row.get("top_positive_case"), dict) else {}
            lines.append(
                "| {bin} | {cases} | {base} | {mean} | {median} | {wins} | {mse} | {share} | {top} | {topd} |".format(
                    bin=row.get("bin"),
                    cases=_fmt(row.get("case_count")),
                    base=_fmt(row.get("mean_gain0_target_corr")),
                    mean=_fmt(row.get("mean_corr_delta")),
                    median=_fmt(row.get("median_corr_delta")),
                    wins=_fmt(row.get("corr_win_fraction")),
                    mse=_fmt(row.get("mean_mse_delta")),
                    share=_fmt(row.get("positive_outlier_share")),
                    top=top.get("name"),
                    topd=_fmt(top.get("corr_delta")),
                )
            )

    lockbox_manifest = payload["evidence"].get("audio_predeclared_lockbox_manifest", {})
    lockbox_compare = payload["evidence"].get("audio_predeclared_lockbox_compare", {})
    lockbox_no_single = payload["evidence"].get("audio_predeclared_lockbox_compare_no_single_source", {})
    lines.extend(
        [
            "",
            "## Audio Predeclared Lockbox v1",
            "",
            f"- manifest status: `{lockbox_manifest.get('status')}`",
            f"- selected cases: `{_fmt(lockbox_manifest.get('selected_case_count'))}`",
            f"- unique paths / hashes / stems: `{_fmt(lockbox_manifest.get('selected_unique_paths'))}` / `{_fmt(lockbox_manifest.get('selected_unique_content_hashes'))}` / `{_fmt(lockbox_manifest.get('selected_unique_provider_neutral_stems'))}`",
            f"- cases JSON: `{lockbox_manifest.get('cases_json_path')}`",
            f"- compare status: `{lockbox_compare.get('status')}`",
            f"- compare scope: `{lockbox_compare.get('comparison_scope')}`",
            f"- overall mean / median corr delta: `{_fmt(lockbox_compare.get('overall_mean_corr_delta'))}` / `{_fmt(lockbox_compare.get('overall_median_corr_delta'))}`",
            f"- overall corr wins: `{_fmt(lockbox_compare.get('overall_corr_win_fraction'))}`",
            f"- overall mean MSE delta: `{_fmt(lockbox_compare.get('overall_mean_mse_delta'))}`",
            f"- positive outlier share: `{_fmt(lockbox_compare.get('overall_positive_outlier_share'))}`",
            f"- non-bad weighted mean corr delta: `{_fmt(lockbox_compare.get('non_bad_mean_corr_delta'))}`",
            f"- no-single-source status: `{lockbox_no_single.get('status')}`",
            f"- no-single-source mean / median corr delta: `{_fmt(lockbox_no_single.get('overall_mean_corr_delta'))}` / `{_fmt(lockbox_no_single.get('overall_median_corr_delta'))}`",
            f"- no-single-source mean MSE delta: `{_fmt(lockbox_no_single.get('overall_mean_mse_delta'))}`",
            "",
            "| group | status | cases | mean corr d | median corr d | corr wins | mean MSE d | outlier share |",
            "|---|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in lockbox_compare.get("groups", []) or []:
        if not isinstance(row, dict):
            continue
        lines.append(
            "| {group} | {status} | {cases} | {mean} | {median} | {wins} | {mse} | {share} |".format(
                group=row.get("group"),
                status=row.get("status"),
                cases=_fmt(row.get("case_count")),
                mean=_fmt(row.get("mean_corr_delta")),
                median=_fmt(row.get("median_corr_delta")),
                wins=_fmt(row.get("corr_win_fraction")),
                mse=_fmt(row.get("mean_mse_delta")),
                share=_fmt(row.get("positive_outlier_share")),
            )
        )
    if lockbox_compare.get("baseline_correlation_bins"):
        lines.extend(
            [
                "",
                "| baseline bin | cases | mean base corr | mean corr d | median corr d | corr wins | mean MSE d | outlier share |",
                "|---|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for row in lockbox_compare.get("baseline_correlation_bins", []) or []:
            if not isinstance(row, dict):
                continue
            lines.append(
                "| {bin} | {cases} | {base} | {mean} | {median} | {wins} | {mse} | {share} |".format(
                    bin=row.get("bin"),
                    cases=_fmt(row.get("case_count")),
                    base=_fmt(row.get("mean_baseline_target_corr")),
                    mean=_fmt(row.get("mean_corr_delta")),
                    median=_fmt(row.get("median_corr_delta")),
                    wins=_fmt(row.get("corr_win_fraction")),
                    mse=_fmt(row.get("mean_mse_delta")),
                    share=_fmt(row.get("positive_outlier_share")),
                )
            )

    phase_scout = payload["evidence"].get("audio_phase_law_scout_compare_no_single_source", {})
    lines.extend(
        [
            "",
            "## Audio Phase-Law Scout v1",
            "",
            f"- status: `{phase_scout.get('status')}`",
            f"- comparison scope: `{phase_scout.get('comparison_scope')}`",
            f"- excluded roles: `{', '.join(phase_scout.get('excluded_roles') or [])}`",
            f"- cases: `{_fmt(phase_scout.get('case_count'))}`",
            f"- analysis rows: `{_fmt(phase_scout.get('analysis_row_count'))}`",
            f"- overall mean / median corr delta: `{_fmt(phase_scout.get('overall_mean_corr_delta'))}` / `{_fmt(phase_scout.get('overall_median_corr_delta'))}`",
            f"- overall corr wins: `{_fmt(phase_scout.get('overall_corr_win_fraction'))}`",
            f"- overall mean MSE delta: `{_fmt(phase_scout.get('overall_mean_mse_delta'))}`",
            "",
            "| mechanism | gain | status | cases | mean corr d | median corr d | corr wins | mean MSE d | outlier share |",
            "|---|---:|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in (phase_scout.get("candidate_summaries") or [])[:12]:
        if not isinstance(row, dict):
            continue
        lines.append(
            "| {mechanism} | {gain} | {status} | {cases} | {mean} | {median} | {wins} | {mse} | {share} |".format(
                mechanism=row.get("mechanism"),
                gain=_fmt(row.get("gain")),
                status=row.get("status"),
                cases=_fmt(row.get("case_count")),
                mean=_fmt(row.get("mean_corr_delta")),
                median=_fmt(row.get("median_corr_delta")),
                wins=_fmt(row.get("corr_win_fraction")),
                mse=_fmt(row.get("mean_mse_delta")),
                share=_fmt(row.get("positive_outlier_share")),
            )
        )

    internal_raw = payload["evidence"].get("audio_internal_phase_law_raw_compare_no_single_source", {})
    internal_variant = payload["evidence"].get("audio_internal_phase_law_variant_compare_no_single_source", {})
    internal_objective = payload["evidence"].get("audio_internal_phase_law_objective_score", {})
    lines.extend(
        [
            "",
            "## Internal Phase-Law Scout v1",
            "",
            f"- raw absolute status: `{internal_raw.get('status')}`",
            f"- raw absolute cases / rows: `{_fmt(internal_raw.get('case_count'))}` / `{_fmt(internal_raw.get('analysis_row_count'))}`",
            f"- raw absolute overall mean / median corr delta: `{_fmt(internal_raw.get('overall_mean_corr_delta'))}` / `{_fmt(internal_raw.get('overall_median_corr_delta'))}`",
            f"- direct variant status: `{internal_variant.get('status')}`",
            f"- direct variant cases / rows: `{_fmt(internal_variant.get('case_count'))}` / `{_fmt(internal_variant.get('delta_row_count'))}`",
            f"- best overall run: `{internal_variant.get('best_overall_run')}` mean corr d `{_fmt(internal_variant.get('best_overall_mean_corr_delta'))}` median `{_fmt(internal_variant.get('best_overall_median_corr_delta'))}` wins `{_fmt(internal_variant.get('best_overall_corr_win_fraction'))}`",
            f"- best matched-row candidate: `{internal_variant.get('best_candidate_run')}` / `{internal_variant.get('best_candidate_magnitude_mode')}` / `{internal_variant.get('best_candidate_mask_mode')}` / gain `{_fmt(internal_variant.get('best_candidate_gain'))}` mean corr d `{_fmt(internal_variant.get('best_candidate_mean_corr_delta'))}`",
            f"- locked objective status: `{internal_objective.get('status')}`",
            f"- locked objective best run / decision / score: `{internal_objective.get('best_run')}` / `{internal_objective.get('best_decision')}` / `{_fmt(internal_objective.get('best_score'))}`",
            f"- locked objective absolute nonbad mean / matched absolute mean: `{_fmt(internal_objective.get('best_absolute_nonbad_mean_corr_delta'))}` / `{_fmt(internal_objective.get('best_matched_absolute_mean_corr_delta'))}`",
            "",
            "| run | rows | cases | status | mean corr d | median corr d | corr wins | mean MSE d |",
            "|---|---:|---:|---|---:|---:|---:|---:|",
        ]
    )
    for row in (internal_variant.get("overall_summaries") or [])[:8]:
        if not isinstance(row, dict):
            continue
        lines.append(
            "| {run} | {rows} | {cases} | {status} | {mean} | {median} | {wins} | {mse} |".format(
                run=row.get("run_label"),
                rows=_fmt(row.get("row_count")),
                cases=_fmt(row.get("case_count")),
                status=row.get("status"),
                mean=_fmt(row.get("mean_corr_delta_vs_baseline_config")),
                median=_fmt(row.get("median_corr_delta_vs_baseline_config")),
                wins=_fmt(row.get("corr_win_fraction_vs_baseline_config")),
                mse=_fmt(row.get("mean_mse_delta_vs_baseline_config")),
            )
        )

    focused_raw = payload["evidence"].get("audio_internal_phase_law_focused_raw_compare_no_single_source", {})
    focused_variant = payload["evidence"].get("audio_internal_phase_law_focused_variant_compare_no_single_source", {})
    focused_objective = payload["evidence"].get("audio_internal_phase_law_focused_objective_score", {})
    lines.extend(
        [
            "",
            "## Internal Phase-Law Focused Coefficient Search v1",
            "",
            f"- raw absolute status: `{focused_raw.get('status')}`",
            f"- raw absolute cases / rows: `{_fmt(focused_raw.get('case_count'))}` / `{_fmt(focused_raw.get('analysis_row_count'))}`",
            f"- raw absolute overall mean / median corr delta: `{_fmt(focused_raw.get('overall_mean_corr_delta'))}` / `{_fmt(focused_raw.get('overall_median_corr_delta'))}`",
            f"- direct variant status: `{focused_variant.get('status')}`",
            f"- direct variant cases / rows: `{_fmt(focused_variant.get('case_count'))}` / `{_fmt(focused_variant.get('delta_row_count'))}`",
            f"- best overall run: `{focused_variant.get('best_overall_run')}` mean corr d `{_fmt(focused_variant.get('best_overall_mean_corr_delta'))}` median `{_fmt(focused_variant.get('best_overall_median_corr_delta'))}` wins `{_fmt(focused_variant.get('best_overall_corr_win_fraction'))}`",
            f"- best matched-row candidate: `{focused_variant.get('best_candidate_run')}` / `{focused_variant.get('best_candidate_magnitude_mode')}` / `{focused_variant.get('best_candidate_mask_mode')}` / gain `{_fmt(focused_variant.get('best_candidate_gain'))}` mean corr d `{_fmt(focused_variant.get('best_candidate_mean_corr_delta'))}`",
            f"- locked objective status: `{focused_objective.get('status')}`",
            f"- locked objective best run / decision / score: `{focused_objective.get('best_run')}` / `{focused_objective.get('best_decision')}` / `{_fmt(focused_objective.get('best_score'))}`",
            f"- locked objective absolute nonbad mean / matched absolute mean: `{_fmt(focused_objective.get('best_absolute_nonbad_mean_corr_delta'))}` / `{_fmt(focused_objective.get('best_matched_absolute_mean_corr_delta'))}`",
            "",
            "| run | decision | score | direct mean d | direct best d | abs nonbad mean d | abs nonbad wins | matched abs d | matched status |",
            "|---|---|---:|---:|---:|---:|---:|---:|---|",
        ]
    )
    for row in (focused_objective.get("rows") or [])[:12]:
        if not isinstance(row, dict):
            continue
        lines.append(
            "| {run} | {decision} | {score} | {direct} | {best} | {absmean} | {abswins} | {matched} | {matched_status} |".format(
                run=row.get("run_label"),
                decision=row.get("decision"),
                score=_fmt(row.get("score")),
                direct=_fmt(row.get("direct_overall_mean_corr_delta")),
                best=_fmt(row.get("direct_best_mean_corr_delta")),
                absmean=_fmt(row.get("absolute_best_nonbad_mean_corr_delta")),
                abswins=_fmt(row.get("absolute_best_nonbad_corr_win_fraction")),
                matched=_fmt(row.get("matched_absolute_mean_corr_delta")),
                matched_status=row.get("matched_absolute_status"),
            )
        )

    wide_raw = payload["evidence"].get("audio_internal_phase_law_wide_raw_compare_no_single_source", {})
    wide_variant = payload["evidence"].get("audio_internal_phase_law_wide_variant_compare_no_single_source", {})
    wide_objective = payload["evidence"].get("audio_internal_phase_law_wide_objective_score", {})
    wide_joint = payload["evidence"].get("audio_internal_phase_law_wide_joint_rows", {})
    lines.extend(
        [
            "",
            "## Internal Phase-Law Wide Coefficient Search v1",
            "",
            f"- raw absolute status: `{wide_raw.get('status')}`",
            f"- raw absolute cases / rows: `{_fmt(wide_raw.get('case_count'))}` / `{_fmt(wide_raw.get('analysis_row_count'))}`",
            f"- raw absolute overall mean / median corr delta: `{_fmt(wide_raw.get('overall_mean_corr_delta'))}` / `{_fmt(wide_raw.get('overall_median_corr_delta'))}`",
            f"- direct variant status: `{wide_variant.get('status')}`",
            f"- direct variant cases / rows: `{_fmt(wide_variant.get('case_count'))}` / `{_fmt(wide_variant.get('delta_row_count'))}`",
            f"- best overall run: `{wide_variant.get('best_overall_run')}` mean corr d `{_fmt(wide_variant.get('best_overall_mean_corr_delta'))}` median `{_fmt(wide_variant.get('best_overall_median_corr_delta'))}` wins `{_fmt(wide_variant.get('best_overall_corr_win_fraction'))}`",
            f"- best matched-row candidate: `{wide_variant.get('best_candidate_run')}` / `{wide_variant.get('best_candidate_magnitude_mode')}` / `{wide_variant.get('best_candidate_mask_mode')}` / gain `{_fmt(wide_variant.get('best_candidate_gain'))}` mean corr d `{_fmt(wide_variant.get('best_candidate_mean_corr_delta'))}`",
            f"- locked objective status: `{wide_objective.get('status')}`",
            f"- locked objective best run / decision / score: `{wide_objective.get('best_run')}` / `{wide_objective.get('best_decision')}` / `{_fmt(wide_objective.get('best_score'))}`",
            f"- joint-row status / candidates / rows: `{wide_joint.get('status')}` / `{_fmt(wide_joint.get('candidate_count'))}` / `{_fmt(wide_joint.get('row_count'))}`",
            f"- joint-row best: `{wide_joint.get('best_run')}` / `{wide_joint.get('best_magnitude_mode')}` / `{wide_joint.get('best_mask_mode')}` / gain `{_fmt(wide_joint.get('best_gain'))}` decision `{wide_joint.get('best_decision')}`",
            f"- joint-row best absolute median / wins / status: `{_fmt(wide_joint.get('best_absolute_median_corr_delta'))}` / `{_fmt(wide_joint.get('best_absolute_corr_win_fraction'))}` / `{wide_joint.get('best_absolute_status')}`",
            "",
            "| run | decision | score | direct mean d | direct best d | abs nonbad mean d | abs nonbad wins | matched abs d | matched status |",
            "|---|---|---:|---:|---:|---:|---:|---:|---|",
        ]
    )
    for row in (wide_objective.get("rows") or [])[:12]:
        if not isinstance(row, dict):
            continue
        lines.append(
            "| {run} | {decision} | {score} | {direct} | {best} | {absmean} | {abswins} | {matched} | {matched_status} |".format(
                run=row.get("run_label"),
                decision=row.get("decision"),
                score=_fmt(row.get("score")),
                direct=_fmt(row.get("direct_overall_mean_corr_delta")),
                best=_fmt(row.get("direct_best_mean_corr_delta")),
                absmean=_fmt(row.get("absolute_best_nonbad_mean_corr_delta")),
                abswins=_fmt(row.get("absolute_best_nonbad_corr_win_fraction")),
                matched=_fmt(row.get("matched_absolute_mean_corr_delta")),
                matched_status=row.get("matched_absolute_status"),
            )
        )

    v2_raw = payload["evidence"].get("audio_internal_phase_law_v2_guard_raw_compare_no_single_source", {})
    v2_variant = payload["evidence"].get("audio_internal_phase_law_v2_guard_variant_compare_no_single_source", {})
    v2_objective = payload["evidence"].get("audio_internal_phase_law_v2_guard_objective_score", {})
    v2_joint = payload["evidence"].get("audio_internal_phase_law_v2_guard_joint_rows", {})
    v2_atlas = payload["evidence"].get("audio_internal_phase_law_v2_guard_failure_atlas", {})
    lines.extend(
        [
            "",
            "## Internal Phase-Law v2 Guard Search v1",
            "",
            f"- raw absolute status: `{v2_raw.get('status')}`",
            f"- raw absolute cases / rows: `{_fmt(v2_raw.get('case_count'))}` / `{_fmt(v2_raw.get('analysis_row_count'))}`",
            f"- raw absolute overall mean / median corr delta: `{_fmt(v2_raw.get('overall_mean_corr_delta'))}` / `{_fmt(v2_raw.get('overall_median_corr_delta'))}`",
            f"- direct variant status: `{v2_variant.get('status')}`",
            f"- direct variant cases / rows: `{_fmt(v2_variant.get('case_count'))}` / `{_fmt(v2_variant.get('delta_row_count'))}`",
            f"- best overall run: `{v2_variant.get('best_overall_run')}` mean corr d `{_fmt(v2_variant.get('best_overall_mean_corr_delta'))}` median `{_fmt(v2_variant.get('best_overall_median_corr_delta'))}` wins `{_fmt(v2_variant.get('best_overall_corr_win_fraction'))}`",
            f"- best matched-row candidate: `{v2_variant.get('best_candidate_run')}` / `{v2_variant.get('best_candidate_magnitude_mode')}` / `{v2_variant.get('best_candidate_mask_mode')}` / gain `{_fmt(v2_variant.get('best_candidate_gain'))}` mean corr d `{_fmt(v2_variant.get('best_candidate_mean_corr_delta'))}`",
            f"- locked objective status: `{v2_objective.get('status')}`",
            f"- locked objective best run / decision / score: `{v2_objective.get('best_run')}` / `{v2_objective.get('best_decision')}` / `{_fmt(v2_objective.get('best_score'))}`",
            f"- joint-row status / candidates / rows: `{v2_joint.get('status')}` / `{_fmt(v2_joint.get('candidate_count'))}` / `{_fmt(v2_joint.get('row_count'))}`",
            f"- atlas status / joint status: `{v2_atlas.get('status')}` / `{v2_atlas.get('joint_status')}`",
            f"- atlas best run / direct mean / joint decision: `{v2_atlas.get('best_run')}` / `{_fmt(v2_atlas.get('best_direct_mean_corr_delta'))}` / `{v2_atlas.get('best_joint_decision')}`",
            f"- atlas best nonbad mean / wins: `{_fmt(v2_atlas.get('best_absolute_nonbad_mean_corr_delta'))}` / `{_fmt(v2_atlas.get('best_absolute_nonbad_corr_win_fraction'))}`",
            "",
            "| run | decision | score | direct mean d | direct best d | abs nonbad mean d | abs nonbad wins | matched abs d | matched status |",
            "|---|---|---:|---:|---:|---:|---:|---:|---|",
        ]
    )
    for row in (v2_objective.get("rows") or [])[:12]:
        if not isinstance(row, dict):
            continue
        lines.append(
            "| {run} | {decision} | {score} | {direct} | {best} | {absmean} | {abswins} | {matched} | {matched_status} |".format(
                run=row.get("run_label"),
                decision=row.get("decision"),
                score=_fmt(row.get("score")),
                direct=_fmt(row.get("direct_overall_mean_corr_delta")),
                best=_fmt(row.get("direct_best_mean_corr_delta")),
                absmean=_fmt(row.get("absolute_best_nonbad_mean_corr_delta")),
                abswins=_fmt(row.get("absolute_best_nonbad_corr_win_fraction")),
                matched=_fmt(row.get("matched_absolute_mean_corr_delta")),
                matched_status=row.get("matched_absolute_status"),
            )
        )

    v3_raw = payload["evidence"].get("audio_internal_phase_law_v3_local_raw_compare_no_single_source", {})
    v3_variant = payload["evidence"].get("audio_internal_phase_law_v3_local_variant_compare_no_single_source", {})
    v3_objective = payload["evidence"].get("audio_internal_phase_law_v3_local_objective_score", {})
    v3_joint = payload["evidence"].get("audio_internal_phase_law_v3_local_joint_rows", {})
    v3_atlas = payload["evidence"].get("audio_internal_phase_law_v3_local_failure_atlas", {})
    v3_strict_objective = payload["evidence"].get("audio_internal_phase_law_v3_local_strict_objective_score", {})
    v3_strict_joint = payload["evidence"].get("audio_internal_phase_law_v3_local_strict_joint_rows", {})
    lines.extend(
        [
            "",
            "## Internal Phase-Law v3 Local Search v1",
            "",
            f"- raw absolute status: `{v3_raw.get('status')}`",
            f"- raw absolute cases / rows: `{_fmt(v3_raw.get('case_count'))}` / `{_fmt(v3_raw.get('analysis_row_count'))}`",
            f"- raw absolute overall mean / median corr delta: `{_fmt(v3_raw.get('overall_mean_corr_delta'))}` / `{_fmt(v3_raw.get('overall_median_corr_delta'))}`",
            f"- direct variant status: `{v3_variant.get('status')}`",
            f"- direct variant cases / rows: `{_fmt(v3_variant.get('case_count'))}` / `{_fmt(v3_variant.get('delta_row_count'))}`",
            f"- best overall run: `{v3_variant.get('best_overall_run')}` mean corr d `{_fmt(v3_variant.get('best_overall_mean_corr_delta'))}` median `{_fmt(v3_variant.get('best_overall_median_corr_delta'))}` wins `{_fmt(v3_variant.get('best_overall_corr_win_fraction'))}`",
            f"- best matched-row candidate: `{v3_variant.get('best_candidate_run')}` / `{v3_variant.get('best_candidate_magnitude_mode')}` / `{v3_variant.get('best_candidate_mask_mode')}` / gain `{_fmt(v3_variant.get('best_candidate_gain'))}` mean corr d `{_fmt(v3_variant.get('best_candidate_mean_corr_delta'))}`",
            f"- locked objective status: `{v3_objective.get('status')}`",
            f"- locked objective best run / decision / score: `{v3_objective.get('best_run')}` / `{v3_objective.get('best_decision')}` / `{_fmt(v3_objective.get('best_score'))}`",
            f"- joint-row status / candidates / rows: `{v3_joint.get('status')}` / `{_fmt(v3_joint.get('candidate_count'))}` / `{_fmt(v3_joint.get('row_count'))}`",
            f"- atlas status / joint status: `{v3_atlas.get('status')}` / `{v3_atlas.get('joint_status')}`",
            f"- atlas best run / direct mean / joint decision: `{v3_atlas.get('best_run')}` / `{_fmt(v3_atlas.get('best_direct_mean_corr_delta'))}` / `{v3_atlas.get('best_joint_decision')}`",
            f"- atlas best nonbad mean / wins: `{_fmt(v3_atlas.get('best_absolute_nonbad_mean_corr_delta'))}` / `{_fmt(v3_atlas.get('best_absolute_nonbad_corr_win_fraction'))}`",
            f"- strict same-row objective status / best decision: `{v3_strict_objective.get('status')}` / `{v3_strict_objective.get('best_decision')}`",
            f"- strict joint-row status / candidates / mean nonbad-bin corr d: `{v3_strict_joint.get('status')}` / `{_fmt(v3_strict_joint.get('candidate_count'))}` / `{_fmt(v3_strict_joint.get('mean_absolute_nonbad_bin_corr_delta'))}`",
            "",
            "| run | decision | score | direct mean d | direct best d | abs nonbad mean d | abs nonbad wins | matched abs d | matched status |",
            "|---|---|---:|---:|---:|---:|---:|---:|---|",
        ]
    )
    for row in (v3_objective.get("rows") or [])[:12]:
        if not isinstance(row, dict):
            continue
        lines.append(
            "| {run} | {decision} | {score} | {direct} | {best} | {absmean} | {abswins} | {matched} | {matched_status} |".format(
                run=row.get("run_label"),
                decision=row.get("decision"),
                score=_fmt(row.get("score")),
                direct=_fmt(row.get("direct_overall_mean_corr_delta")),
                best=_fmt(row.get("direct_best_mean_corr_delta")),
                absmean=_fmt(row.get("absolute_best_nonbad_mean_corr_delta")),
                abswins=_fmt(row.get("absolute_best_nonbad_corr_win_fraction")),
                matched=_fmt(row.get("matched_absolute_mean_corr_delta")),
                matched_status=row.get("matched_absolute_status"),
            )
        )
    if v3_atlas.get("baseline_bins"):
        lines.extend(
            [
                "",
                "| atlas baseline bin | cases | mean corr d | median corr d | corr wins | mean MSE d |",
                "|---|---:|---:|---:|---:|---:|",
            ]
        )
        for row in v3_atlas["baseline_bins"]:
            if not isinstance(row, dict):
                continue
            lines.append(
                "| {bin} | {cases} | {mean_corr} | {median_corr} | {wins} | {mean_mse} |".format(
                    bin=row.get("bin"),
                    cases=_fmt(row.get("case_count")),
                    mean_corr=_fmt(row.get("mean_corr_delta")),
                    median_corr=_fmt(row.get("median_corr_delta")),
                    wins=_fmt(row.get("corr_win_fraction")),
                    mean_mse=_fmt(row.get("mean_mse_delta")),
                )
            )

    v4_raw = payload["evidence"].get(
        "audio_internal_phase_law_v4_joint_nonbad_raw_compare_no_single_source", {}
    )
    v4_variant = payload["evidence"].get(
        "audio_internal_phase_law_v4_joint_nonbad_variant_compare_no_single_source", {}
    )
    v4_objective = payload["evidence"].get("audio_internal_phase_law_v4_joint_nonbad_objective_score", {})
    v4_joint = payload["evidence"].get("audio_internal_phase_law_v4_joint_nonbad_joint_rows", {})
    v4_atlas = payload["evidence"].get("audio_internal_phase_law_v4_joint_nonbad_failure_atlas", {})
    lines.extend(
        [
            "",
            "## Internal Phase-Law v4 Joint/Nonbad Search v1",
            "",
            f"- raw absolute status: `{v4_raw.get('status')}`",
            f"- raw absolute cases / rows: `{_fmt(v4_raw.get('case_count'))}` / `{_fmt(v4_raw.get('analysis_row_count'))}`",
            f"- raw absolute overall mean / median corr delta: `{_fmt(v4_raw.get('overall_mean_corr_delta'))}` / `{_fmt(v4_raw.get('overall_median_corr_delta'))}`",
            f"- direct variant status: `{v4_variant.get('status')}`",
            f"- direct variant cases / rows: `{_fmt(v4_variant.get('case_count'))}` / `{_fmt(v4_variant.get('delta_row_count'))}`",
            f"- best overall run: `{v4_variant.get('best_overall_run')}` mean corr d `{_fmt(v4_variant.get('best_overall_mean_corr_delta'))}` median `{_fmt(v4_variant.get('best_overall_median_corr_delta'))}` wins `{_fmt(v4_variant.get('best_overall_corr_win_fraction'))}`",
            f"- best matched-row candidate: `{v4_variant.get('best_candidate_run')}` / `{v4_variant.get('best_candidate_magnitude_mode')}` / `{v4_variant.get('best_candidate_mask_mode')}` / gain `{_fmt(v4_variant.get('best_candidate_gain'))}` mean corr d `{_fmt(v4_variant.get('best_candidate_mean_corr_delta'))}`",
            f"- locked objective status: `{v4_objective.get('status')}`",
            f"- locked objective best run / decision / score: `{v4_objective.get('best_run')}` / `{v4_objective.get('best_decision')}` / `{_fmt(v4_objective.get('best_score'))}`",
            f"- joint-row status / candidates / rows: `{v4_joint.get('status')}` / `{_fmt(v4_joint.get('candidate_count'))}` / `{_fmt(v4_joint.get('row_count'))}`",
            f"- joint-row mean nonbad-bin corr d: `{_fmt(v4_joint.get('mean_absolute_nonbad_bin_corr_delta'))}`",
            f"- joint-row best run / mode / mask / gain: `{v4_joint.get('best_run')}` / `{v4_joint.get('best_magnitude_mode')}` / `{v4_joint.get('best_mask_mode')}` / `{_fmt(v4_joint.get('best_gain'))}`",
            f"- joint-row best direct / absolute / nonbad corr d: `{_fmt(v4_joint.get('best_direct_mean_corr_delta'))}` / `{_fmt(v4_joint.get('best_absolute_mean_corr_delta'))}` / `{_fmt(v4_joint.get('best_absolute_nonbad_bin_mean_corr_delta'))}`",
            f"- atlas status / joint status: `{v4_atlas.get('status')}` / `{v4_atlas.get('joint_status')}`",
            f"- atlas runs / family rows: `{_fmt(v4_atlas.get('run_count'))}` / `{_fmt(v4_atlas.get('family_row_count'))}`",
            f"- atlas best run / direct mean / joint decision: `{v4_atlas.get('best_run')}` / `{_fmt(v4_atlas.get('best_direct_mean_corr_delta'))}` / `{v4_atlas.get('best_joint_decision')}`",
            f"- atlas best nonbad mean / wins: `{_fmt(v4_atlas.get('best_absolute_nonbad_mean_corr_delta'))}` / `{_fmt(v4_atlas.get('best_absolute_nonbad_corr_win_fraction'))}`",
            "",
            "| run | decision | score | direct mean d | direct best d | abs nonbad mean d | matched abs d | matched nonbad bin d | matched good d | matched status |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|---|",
        ]
    )
    for row in (v4_objective.get("rows") or [])[:12]:
        if not isinstance(row, dict):
            continue
        lines.append(
            "| {run} | {decision} | {score} | {direct} | {best} | {absmean} | {matched} | {nonbad} | {good} | {matched_status} |".format(
                run=row.get("run_label"),
                decision=row.get("decision"),
                score=_fmt(row.get("score")),
                direct=_fmt(row.get("direct_overall_mean_corr_delta")),
                best=_fmt(row.get("direct_best_mean_corr_delta")),
                absmean=_fmt(row.get("absolute_best_nonbad_mean_corr_delta")),
                matched=_fmt(row.get("matched_absolute_mean_corr_delta")),
                nonbad=_fmt(row.get("matched_absolute_nonbad_bin_mean_corr_delta")),
                good=_fmt(row.get("matched_absolute_good_bin_mean_corr_delta")),
                matched_status=row.get("matched_absolute_status"),
            )
        )
    if v4_joint.get("rows"):
        lines.extend(
            [
                "",
                "| joint run | decision | mode | mask | gain | direct mean d | abs mean d | nonbad d | weak d | moderate d | good d | abs wins |",
                "|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for row in (v4_joint.get("rows") or [])[:12]:
            if not isinstance(row, dict):
                continue
            lines.append(
                "| {run} | {decision} | {mode} | {mask} | {gain} | {direct} | {absmean} | {nonbad} | {weak} | {moderate} | {good} | {wins} |".format(
                    run=row.get("run_label"),
                    decision=row.get("decision"),
                    mode=row.get("magnitude_mode"),
                    mask=row.get("mask_mode"),
                    gain=_fmt(row.get("gain")),
                    direct=_fmt(row.get("direct_mean_corr_delta")),
                    absmean=_fmt(row.get("absolute_mean_corr_delta")),
                    nonbad=_fmt(row.get("absolute_nonbad_bin_mean_corr_delta")),
                    weak=_fmt(row.get("absolute_weak_bin_mean_corr_delta")),
                    moderate=_fmt(row.get("absolute_moderate_bin_mean_corr_delta")),
                    good=_fmt(row.get("absolute_good_bin_mean_corr_delta")),
                    wins=_fmt(row.get("absolute_corr_win_fraction")),
                )
            )
    if v4_atlas.get("baseline_bins"):
        lines.extend(
            [
                "",
                "| atlas baseline bin | cases | mean corr d | median corr d | corr wins | mean MSE d |",
                "|---|---:|---:|---:|---:|---:|",
            ]
        )
        for row in v4_atlas["baseline_bins"]:
            if not isinstance(row, dict):
                continue
            lines.append(
                "| {bin} | {cases} | {mean_corr} | {median_corr} | {wins} | {mean_mse} |".format(
                    bin=row.get("bin"),
                    cases=_fmt(row.get("case_count")),
                    mean_corr=_fmt(row.get("mean_corr_delta")),
                    median_corr=_fmt(row.get("median_corr_delta")),
                    wins=_fmt(row.get("corr_win_fraction")),
                    mean_mse=_fmt(row.get("mean_mse_delta")),
                )
            )

    v5_raw = payload["evidence"].get(
        "audio_internal_phase_law_v5_low_energy_row_raw_compare_no_single_source", {}
    )
    v5_variant = payload["evidence"].get(
        "audio_internal_phase_law_v5_low_energy_row_variant_compare_no_single_source", {}
    )
    v5_objective = payload["evidence"].get("audio_internal_phase_law_v5_low_energy_row_objective_score", {})
    v5_joint = payload["evidence"].get("audio_internal_phase_law_v5_low_energy_row_joint_rows", {})
    v5_atlas = payload["evidence"].get("audio_internal_phase_law_v5_low_energy_row_failure_atlas", {})
    lines.extend(
        [
            "",
            "## Internal Phase-Law v5 Low-Energy Target Row v1",
            "",
            f"- raw absolute status: `{v5_raw.get('status')}`",
            f"- raw absolute cases / rows: `{_fmt(v5_raw.get('case_count'))}` / `{_fmt(v5_raw.get('analysis_row_count'))}`",
            f"- direct variant status: `{v5_variant.get('status')}`",
            f"- locked objective status: `{v5_objective.get('status')}`",
            f"- locked objective best run / decision / score: `{v5_objective.get('best_run')}` / `{v5_objective.get('best_decision')}` / `{_fmt(v5_objective.get('best_score'))}`",
            f"- target row: `flat` / `low_energy_bins` / `raw` / gain `2`",
            f"- target-row candidates / rows: `{_fmt(v5_objective.get('target_low_energy_candidate_count'))}` / `{_fmt(v5_objective.get('target_low_energy_row_count'))}`",
            f"- target-row best run / decision / score: `{v5_objective.get('target_low_energy_best_run')}` / `{v5_objective.get('target_low_energy_best_decision')}` / `{_fmt(v5_objective.get('target_low_energy_best_score'))}`",
            f"- target-row direct mean / median / wins: `{_fmt(v5_objective.get('target_low_energy_best_direct_mean_corr_delta'))}` / `{_fmt(v5_objective.get('target_low_energy_best_direct_median_corr_delta'))}` / `{_fmt(v5_objective.get('target_low_energy_best_direct_corr_win_fraction'))}`",
            f"- target-row absolute mean / wins / nonbad / good corr d: `{_fmt(v5_objective.get('target_low_energy_best_absolute_mean_corr_delta'))}` / `{_fmt(v5_objective.get('target_low_energy_best_absolute_corr_win_fraction'))}` / `{_fmt(v5_objective.get('target_low_energy_best_nonbad_bin_mean_corr_delta'))}` / `{_fmt(v5_objective.get('target_low_energy_best_good_bin_mean_corr_delta'))}`",
            f"- joint-row status / candidates / rows: `{v5_joint.get('status')}` / `{_fmt(v5_joint.get('candidate_count'))}` / `{_fmt(v5_joint.get('row_count'))}`",
            f"- joint-row mean nonbad-bin corr d: `{_fmt(v5_joint.get('mean_absolute_nonbad_bin_corr_delta'))}`",
            f"- joint target-row best run / decision / score: `{v5_joint.get('target_low_energy_best_run')}` / `{v5_joint.get('target_low_energy_best_decision')}` / `{_fmt(v5_joint.get('target_low_energy_best_score'))}`",
            f"- atlas status / joint status: `{v5_atlas.get('status')}` / `{v5_atlas.get('joint_status')}`",
            f"- atlas runs / family rows: `{_fmt(v5_atlas.get('run_count'))}` / `{_fmt(v5_atlas.get('family_row_count'))}`",
            "",
            "| run | decision | score | direct mean d | direct best d | matched nonbad bin d | matched good d | target decision | target direct d | target median | target wins | target nonbad d | matched status |",
            "|---|---|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---|",
        ]
    )
    for row in (v5_objective.get("rows") or [])[:12]:
        if not isinstance(row, dict):
            continue
        lines.append(
            "| {run} | {decision} | {score} | {direct} | {best} | {nonbad} | {good} | {target_decision} | {target_direct} | {target_median} | {target_wins} | {target_nonbad} | {matched_status} |".format(
                run=row.get("run_label"),
                decision=row.get("decision"),
                score=_fmt(row.get("score")),
                direct=_fmt(row.get("direct_overall_mean_corr_delta")),
                best=_fmt(row.get("direct_best_mean_corr_delta")),
                nonbad=_fmt(row.get("matched_absolute_nonbad_bin_mean_corr_delta")),
                good=_fmt(row.get("matched_absolute_good_bin_mean_corr_delta")),
                target_decision=row.get("target_low_energy_decision"),
                target_direct=_fmt(row.get("target_low_energy_direct_mean_corr_delta")),
                target_median=_fmt(row.get("target_low_energy_direct_median_corr_delta")),
                target_wins=_fmt(row.get("target_low_energy_direct_corr_win_fraction")),
                target_nonbad=_fmt(row.get("target_low_energy_nonbad_bin_mean_corr_delta")),
                matched_status=row.get("matched_absolute_status"),
            )
        )
    if v5_joint.get("rows"):
        lines.extend(
            [
                "",
                "| joint run | decision | mode | mask | gain | direct mean d | direct wins | abs mean d | nonbad d | good d | abs wins |",
                "|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for row in (v5_joint.get("rows") or [])[:12]:
            if not isinstance(row, dict):
                continue
            lines.append(
                "| {run} | {decision} | {mode} | {mask} | {gain} | {direct} | {dwins} | {absmean} | {nonbad} | {good} | {awins} |".format(
                    run=row.get("run_label"),
                    decision=row.get("decision"),
                    mode=row.get("magnitude_mode"),
                    mask=row.get("mask_mode"),
                    gain=_fmt(row.get("gain")),
                    direct=_fmt(row.get("direct_mean_corr_delta")),
                    dwins=_fmt(row.get("direct_corr_win_fraction")),
                    absmean=_fmt(row.get("absolute_mean_corr_delta")),
                    nonbad=_fmt(row.get("absolute_nonbad_bin_mean_corr_delta")),
                    good=_fmt(row.get("absolute_good_bin_mean_corr_delta")),
                    awins=_fmt(row.get("absolute_corr_win_fraction")),
                )
            )
    if v5_atlas.get("baseline_bins"):
        lines.extend(
            [
                "",
                "| atlas baseline bin | cases | mean corr d | median corr d | corr wins | mean MSE d |",
                "|---|---:|---:|---:|---:|---:|",
            ]
        )
        for row in v5_atlas["baseline_bins"]:
            if not isinstance(row, dict):
                continue
            lines.append(
                "| {bin} | {cases} | {mean_corr} | {median_corr} | {wins} | {mean_mse} |".format(
                    bin=row.get("bin"),
                    cases=_fmt(row.get("case_count")),
                    mean_corr=_fmt(row.get("mean_corr_delta")),
                    median_corr=_fmt(row.get("median_corr_delta")),
                    wins=_fmt(row.get("corr_win_fraction")),
                    mean_mse=_fmt(row.get("mean_mse_delta")),
                )
            )

    v6_raw = payload["evidence"].get(
        "audio_internal_phase_law_v6_low_energy_win_raw_compare_no_single_source", {}
    )
    v6_variant = payload["evidence"].get(
        "audio_internal_phase_law_v6_low_energy_win_variant_compare_no_single_source", {}
    )
    v6_objective = payload["evidence"].get("audio_internal_phase_law_v6_low_energy_win_objective_score", {})
    v6_joint = payload["evidence"].get("audio_internal_phase_law_v6_low_energy_win_joint_rows", {})
    v6_atlas = payload["evidence"].get("audio_internal_phase_law_v6_low_energy_win_failure_atlas", {})
    lines.extend(
        [
            "",
            "## Internal Phase-Law v6 Low-Energy Win Search v1",
            "",
            f"- raw absolute status: `{v6_raw.get('status')}`",
            f"- raw absolute cases / rows: `{_fmt(v6_raw.get('case_count'))}` / `{_fmt(v6_raw.get('analysis_row_count'))}`",
            f"- direct variant status: `{v6_variant.get('status')}`",
            f"- locked objective status: `{v6_objective.get('status')}`",
            f"- locked objective best run / decision / score: `{v6_objective.get('best_run')}` / `{v6_objective.get('best_decision')}` / `{_fmt(v6_objective.get('best_score'))}`",
            f"- target row: `flat` / `low_energy_bins` / `raw` / gain `2`",
            f"- target-row candidates / rows: `{_fmt(v6_objective.get('target_low_energy_candidate_count'))}` / `{_fmt(v6_objective.get('target_low_energy_row_count'))}`",
            f"- target-row best run / decision / score: `{v6_objective.get('target_low_energy_best_run')}` / `{v6_objective.get('target_low_energy_best_decision')}` / `{_fmt(v6_objective.get('target_low_energy_best_score'))}`",
            f"- target-row direct mean / median / wins: `{_fmt(v6_objective.get('target_low_energy_best_direct_mean_corr_delta'))}` / `{_fmt(v6_objective.get('target_low_energy_best_direct_median_corr_delta'))}` / `{_fmt(v6_objective.get('target_low_energy_best_direct_corr_win_fraction'))}`",
            f"- target-row absolute mean / wins / nonbad / good corr d: `{_fmt(v6_objective.get('target_low_energy_best_absolute_mean_corr_delta'))}` / `{_fmt(v6_objective.get('target_low_energy_best_absolute_corr_win_fraction'))}` / `{_fmt(v6_objective.get('target_low_energy_best_nonbad_bin_mean_corr_delta'))}` / `{_fmt(v6_objective.get('target_low_energy_best_good_bin_mean_corr_delta'))}`",
            f"- joint-row status / candidates / rows: `{v6_joint.get('status')}` / `{_fmt(v6_joint.get('candidate_count'))}` / `{_fmt(v6_joint.get('row_count'))}`",
            f"- joint-row mean nonbad-bin corr d: `{_fmt(v6_joint.get('mean_absolute_nonbad_bin_corr_delta'))}`",
            f"- joint target-row best run / decision / score: `{v6_joint.get('target_low_energy_best_run')}` / `{v6_joint.get('target_low_energy_best_decision')}` / `{_fmt(v6_joint.get('target_low_energy_best_score'))}`",
            f"- atlas status / joint status: `{v6_atlas.get('status')}` / `{v6_atlas.get('joint_status')}`",
            f"- atlas runs / family rows: `{_fmt(v6_atlas.get('run_count'))}` / `{_fmt(v6_atlas.get('family_row_count'))}`",
            "",
            "| run | decision | score | direct mean d | direct best d | matched nonbad bin d | matched good d | target decision | target direct d | target median | target wins | target nonbad d | matched status |",
            "|---|---|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---|",
        ]
    )
    for row in (v6_objective.get("rows") or [])[:12]:
        if not isinstance(row, dict):
            continue
        lines.append(
            "| {run} | {decision} | {score} | {direct} | {best} | {nonbad} | {good} | {target_decision} | {target_direct} | {target_median} | {target_wins} | {target_nonbad} | {matched_status} |".format(
                run=row.get("run_label"),
                decision=row.get("decision"),
                score=_fmt(row.get("score")),
                direct=_fmt(row.get("direct_overall_mean_corr_delta")),
                best=_fmt(row.get("direct_best_mean_corr_delta")),
                nonbad=_fmt(row.get("matched_absolute_nonbad_bin_mean_corr_delta")),
                good=_fmt(row.get("matched_absolute_good_bin_mean_corr_delta")),
                target_decision=row.get("target_low_energy_decision"),
                target_direct=_fmt(row.get("target_low_energy_direct_mean_corr_delta")),
                target_median=_fmt(row.get("target_low_energy_direct_median_corr_delta")),
                target_wins=_fmt(row.get("target_low_energy_direct_corr_win_fraction")),
                target_nonbad=_fmt(row.get("target_low_energy_nonbad_bin_mean_corr_delta")),
                matched_status=row.get("matched_absolute_status"),
            )
        )
    if v6_joint.get("rows"):
        lines.extend(
            [
                "",
                "| joint run | decision | mode | mask | gain | direct mean d | direct wins | abs mean d | nonbad d | good d | abs wins |",
                "|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for row in (v6_joint.get("rows") or [])[:12]:
            if not isinstance(row, dict):
                continue
            lines.append(
                "| {run} | {decision} | {mode} | {mask} | {gain} | {direct} | {dwins} | {absmean} | {nonbad} | {good} | {awins} |".format(
                    run=row.get("run_label"),
                    decision=row.get("decision"),
                    mode=row.get("magnitude_mode"),
                    mask=row.get("mask_mode"),
                    gain=_fmt(row.get("gain")),
                    direct=_fmt(row.get("direct_mean_corr_delta")),
                    dwins=_fmt(row.get("direct_corr_win_fraction")),
                    absmean=_fmt(row.get("absolute_mean_corr_delta")),
                    nonbad=_fmt(row.get("absolute_nonbad_bin_mean_corr_delta")),
                    good=_fmt(row.get("absolute_good_bin_mean_corr_delta")),
                    awins=_fmt(row.get("absolute_corr_win_fraction")),
                )
            )
    if v6_atlas.get("baseline_bins"):
        lines.extend(
            [
                "",
                "| atlas baseline bin | cases | mean corr d | median corr d | corr wins | mean MSE d |",
                "|---|---:|---:|---:|---:|---:|",
            ]
        )
        for row in v6_atlas["baseline_bins"]:
            if not isinstance(row, dict):
                continue
            lines.append(
                "| {bin} | {cases} | {mean_corr} | {median_corr} | {wins} | {mean_mse} |".format(
                    bin=row.get("bin"),
                    cases=_fmt(row.get("case_count")),
                    mean_corr=_fmt(row.get("mean_corr_delta")),
                    median_corr=_fmt(row.get("median_corr_delta")),
                    wins=_fmt(row.get("corr_win_fraction")),
                    mean_mse=_fmt(row.get("mean_mse_delta")),
                )
            )

    v7_raw = payload["evidence"].get(
        "audio_internal_phase_law_v7_low_energy_gain_ladder_raw_compare_no_single_source", {}
    )
    v7_variant = payload["evidence"].get(
        "audio_internal_phase_law_v7_low_energy_gain_ladder_variant_compare_no_single_source", {}
    )
    v7_objective = payload["evidence"].get(
        "audio_internal_phase_law_v7_low_energy_gain_ladder_objective_score", {}
    )
    v7_joint = payload["evidence"].get("audio_internal_phase_law_v7_low_energy_gain_ladder_joint_rows", {})
    v7_atlas = payload["evidence"].get("audio_internal_phase_law_v7_low_energy_gain_ladder_failure_atlas", {})
    v7_flips = payload["evidence"].get("audio_internal_phase_law_v7_low_energy_gain_ladder_target_flips", {})
    lines.extend(
        [
            "",
            "## Internal Phase-Law v7 Low-Energy Gain Ladder v1",
            "",
            f"- raw absolute status: `{v7_raw.get('status')}`",
            f"- raw absolute cases / rows: `{_fmt(v7_raw.get('case_count'))}` / `{_fmt(v7_raw.get('analysis_row_count'))}`",
            f"- direct variant status: `{v7_variant.get('status')}`",
            f"- locked objective status: `{v7_objective.get('status')}`",
            f"- locked objective best run / decision / score: `{v7_objective.get('best_run')}` / `{v7_objective.get('best_decision')}` / `{_fmt(v7_objective.get('best_score'))}`",
            f"- target-row best run / decision / score: `{v7_objective.get('target_low_energy_best_run')}` / `{v7_objective.get('target_low_energy_best_decision')}` / `{_fmt(v7_objective.get('target_low_energy_best_score'))}`",
            f"- target-row direct mean / median / wins: `{_fmt(v7_objective.get('target_low_energy_best_direct_mean_corr_delta'))}` / `{_fmt(v7_objective.get('target_low_energy_best_direct_median_corr_delta'))}` / `{_fmt(v7_objective.get('target_low_energy_best_direct_corr_win_fraction'))}`",
            f"- target-row absolute mean / wins / nonbad / good corr d: `{_fmt(v7_objective.get('target_low_energy_best_absolute_mean_corr_delta'))}` / `{_fmt(v7_objective.get('target_low_energy_best_absolute_corr_win_fraction'))}` / `{_fmt(v7_objective.get('target_low_energy_best_nonbad_bin_mean_corr_delta'))}` / `{_fmt(v7_objective.get('target_low_energy_best_good_bin_mean_corr_delta'))}`",
            f"- joint-row status / candidates / rows: `{v7_joint.get('status')}` / `{_fmt(v7_joint.get('candidate_count'))}` / `{_fmt(v7_joint.get('row_count'))}`",
            f"- atlas status / joint status: `{v7_atlas.get('status')}` / `{v7_atlas.get('joint_status')}`",
            f"- target flip selected run: `{v7_flips.get('selected_run')}`",
            f"- target flip mean / median / wins: `{_fmt(v7_flips.get('selected_target_gain_mean_corr_delta'))}` / `{_fmt(v7_flips.get('selected_target_gain_median_corr_delta'))}` / `{_fmt(v7_flips.get('selected_target_gain_corr_win_fraction'))}`",
            f"- target flip positives / negatives / zeros: `{_fmt(v7_flips.get('selected_target_gain_positive_count'))}` / `{_fmt(v7_flips.get('selected_target_gain_negative_count'))}` / `{_fmt(v7_flips.get('selected_target_gain_zero_count'))}`",
            f"- sign categories: `{v7_flips.get('category_counts')}`",
            "",
            "| gain | best run | mean corr d | median corr d | wins | mean MSE d |",
            "|---:|---|---:|---:|---:|---:|",
        ]
    )
    for row in v7_flips.get("best_by_gain", []) or []:
        if not isinstance(row, dict):
            continue
        lines.append(
            "| {gain} | {run} | {mean} | {median} | {wins} | {mse} |".format(
                gain=_fmt(row.get("gain")),
                run=row.get("run_label"),
                mean=_fmt(row.get("mean_corr_delta")),
                median=_fmt(row.get("median_corr_delta")),
                wins=_fmt(row.get("corr_win_fraction")),
                mse=_fmt(row.get("mean_mse_delta")),
            )
        )
    if v7_flips.get("group_rows"):
        lines.extend(
            [
                "",
                "| group | cases | mean corr d | median corr d | wins | positives | negatives | zeros |",
                "|---|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for row in v7_flips.get("group_rows", []) or []:
            if not isinstance(row, dict):
                continue
            lines.append(
                "| {group} | {cases} | {mean} | {median} | {wins} | {pos} | {neg} | {zero} |".format(
                    group=row.get("group"),
                    cases=_fmt(row.get("case_count")),
                    mean=_fmt(row.get("mean_corr_delta")),
                    median=_fmt(row.get("median_corr_delta")),
                    wins=_fmt(row.get("corr_win_fraction")),
                    pos=_fmt(row.get("positive_count")),
                    neg=_fmt(row.get("negative_count")),
                    zero=_fmt(row.get("zero_count")),
                )
            )

    v8_raw = payload["evidence"].get(
        "audio_internal_phase_law_v8_family_support_mask_raw_compare_no_single_source", {}
    )
    v8_variant = payload["evidence"].get(
        "audio_internal_phase_law_v8_family_support_mask_variant_compare_no_single_source", {}
    )
    v8_objective = payload["evidence"].get(
        "audio_internal_phase_law_v8_family_support_mask_objective_score", {}
    )
    v8_joint = payload["evidence"].get("audio_internal_phase_law_v8_family_support_mask_joint_rows", {})
    v8_atlas = payload["evidence"].get("audio_internal_phase_law_v8_family_support_mask_failure_atlas", {})
    v8_masks = payload["evidence"].get("audio_internal_phase_law_v8_family_support_mask_diagnostics", {})
    lines.extend(
        [
            "",
            "## Internal Phase-Law v8 Family/Support-Mask Scout v1",
            "",
            f"- raw absolute status: `{v8_raw.get('status')}`",
            f"- raw absolute cases / rows: `{_fmt(v8_raw.get('case_count'))}` / `{_fmt(v8_raw.get('analysis_row_count'))}`",
            f"- direct variant status: `{v8_variant.get('status')}`",
            f"- locked objective status: `{v8_objective.get('status')}`",
            f"- locked objective best run / decision / score: `{v8_objective.get('best_run')}` / `{v8_objective.get('best_decision')}` / `{_fmt(v8_objective.get('best_score'))}`",
            f"- target-row best run / decision / score: `{v8_objective.get('target_low_energy_best_run')}` / `{v8_objective.get('target_low_energy_best_decision')}` / `{_fmt(v8_objective.get('target_low_energy_best_score'))}`",
            f"- target-row direct mean / median / wins: `{_fmt(v8_objective.get('target_low_energy_best_direct_mean_corr_delta'))}` / `{_fmt(v8_objective.get('target_low_energy_best_direct_median_corr_delta'))}` / `{_fmt(v8_objective.get('target_low_energy_best_direct_corr_win_fraction'))}`",
            f"- target-row absolute mean / wins / nonbad / good corr d: `{_fmt(v8_objective.get('target_low_energy_best_absolute_mean_corr_delta'))}` / `{_fmt(v8_objective.get('target_low_energy_best_absolute_corr_win_fraction'))}` / `{_fmt(v8_objective.get('target_low_energy_best_nonbad_bin_mean_corr_delta'))}` / `{_fmt(v8_objective.get('target_low_energy_best_good_bin_mean_corr_delta'))}`",
            f"- joint-row status / candidates / rows: `{v8_joint.get('status')}` / `{_fmt(v8_joint.get('candidate_count'))}` / `{_fmt(v8_joint.get('row_count'))}`",
            f"- atlas status / joint status: `{v8_atlas.get('status')}` / `{v8_atlas.get('joint_status')}`",
            f"- support-mask diagnostic selected run / basis: `{v8_masks.get('selected_run')}` / `{v8_masks.get('selected_run_basis')}`",
            f"- global best direct row: `{v8_masks.get('global_best_run')}` / `{v8_masks.get('global_best_mask')}` / gain `{_fmt(v8_masks.get('global_best_gain'))}` / mean `{_fmt(v8_masks.get('global_best_mean_corr_delta'))}` / median `{_fmt(v8_masks.get('global_best_median_corr_delta'))}` / wins `{_fmt(v8_masks.get('global_best_corr_win_fraction'))}`",
            f"- sign-flip counts: `{v8_masks.get('selected_sign_flip_counts')}`",
            f"- best-mask counts: `{v8_masks.get('selected_best_mask_counts')}`",
            "",
            "| selected-run mask | cases | mean corr d | median corr d | wins | positives | negatives | mean MSE d |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in v8_masks.get("selected_target_mask_rows", []) or []:
        if not isinstance(row, dict):
            continue
        lines.append(
            "| {mask} | {cases} | {mean} | {median} | {wins} | {pos} | {neg} | {mse} |".format(
                mask=row.get("mask_mode"),
                cases=_fmt(row.get("case_count")),
                mean=_fmt(row.get("mean_corr_delta")),
                median=_fmt(row.get("median_corr_delta")),
                wins=_fmt(row.get("corr_win_fraction")),
                pos=_fmt(row.get("positive_count")),
                neg=_fmt(row.get("negative_count")),
                mse=_fmt(row.get("mean_mse_delta")),
            )
        )
    if v8_masks.get("selected_best_mask_by_family"):
        lines.extend(
            [
                "",
                "| family | best mask | cases | mean corr d | median corr d | wins | positives | negatives |",
                "|---|---|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for row in v8_masks.get("selected_best_mask_by_family", []) or []:
            if not isinstance(row, dict):
                continue
            lines.append(
                "| {family} | {mask} | {cases} | {mean} | {median} | {wins} | {pos} | {neg} |".format(
                    family=row.get("group"),
                    mask=row.get("mask_mode"),
                    cases=_fmt(row.get("case_count")),
                    mean=_fmt(row.get("mean_corr_delta")),
                    median=_fmt(row.get("median_corr_delta")),
                    wins=_fmt(row.get("corr_win_fraction")),
                    pos=_fmt(row.get("positive_count")),
                    neg=_fmt(row.get("negative_count")),
                )
            )
    v8_oracle = payload["evidence"].get("audio_internal_phase_law_v8_support_router_oracle", {})
    if v8_oracle:
        lines.extend(
            [
                "",
                "### v8 Support-Router Oracle",
                "",
                f"- status: `{v8_oracle.get('status')}`",
                f"- global fixed mask: `{v8_oracle.get('global_fixed_mask')}`",
                f"- family policy counts: `{v8_oracle.get('family_policy_counts')}`",
                f"- case-oracle policy counts: `{v8_oracle.get('case_oracle_policy_counts')}`",
                f"- family router win gain vs global: `{_fmt(v8_oracle.get('family_corr_win_fraction_gain'))}`",
                f"- case oracle win gain vs global: `{_fmt(v8_oracle.get('case_oracle_corr_win_fraction_gain'))}`",
                "",
                "| router | cases | mean corr d | median corr d | wins | positives | negatives |",
                "|---|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for name in [
            "low_energy_fixed",
            "global_fixed_best",
            "family_oracle_router",
            "case_oracle_router",
            "case_anti_oracle",
        ]:
            row = v8_oracle.get(name)
            if not isinstance(row, dict):
                continue
            lines.append(
                "| {name} | {cases} | {mean} | {median} | {wins} | {pos} | {neg} |".format(
                    name=name,
                    cases=_fmt(row.get("case_count")),
                    mean=_fmt(row.get("mean_corr_delta")),
                    median=_fmt(row.get("median_corr_delta")),
                    wins=_fmt(row.get("corr_win_fraction")),
                    pos=_fmt(row.get("positive_count")),
                    neg=_fmt(row.get("negative_count")),
                )
            )
    v9_raw = payload["evidence"].get("audio_internal_phase_law_v9_phase_masks_raw_compare_no_single_source", {})
    v9_variant = payload["evidence"].get(
        "audio_internal_phase_law_v9_phase_masks_variant_compare_no_single_source", {}
    )
    v9_objective = payload["evidence"].get("audio_internal_phase_law_v9_phase_masks_objective_score", {})
    v9_joint = payload["evidence"].get("audio_internal_phase_law_v9_phase_masks_joint_rows", {})
    v9_target_score = payload["evidence"].get(
        "audio_internal_phase_law_v9_phase_masks_phase_stable_target_score", {}
    )
    v9_target_joint = payload["evidence"].get(
        "audio_internal_phase_law_v9_phase_masks_phase_stable_target_joint_rows", {}
    )
    v9_atlas = payload["evidence"].get("audio_internal_phase_law_v9_phase_masks_failure_atlas", {})
    lines.extend(
        [
            "",
            "## Internal Phase-Law v9 Phase-Only Mask Scout v1",
            "",
            f"- raw absolute status: `{v9_raw.get('status')}`",
            f"- direct variant status: `{v9_variant.get('status')}`",
            f"- locked objective status: `{v9_objective.get('status')}`",
            f"- locked objective best run / decision / score: `{v9_objective.get('best_run')}` / `{v9_objective.get('best_decision')}` / `{_fmt(v9_objective.get('best_score'))}`",
            f"- joint-row status / candidates / rows: `{v9_joint.get('status')}` / `{_fmt(v9_joint.get('candidate_count'))}` / `{_fmt(v9_joint.get('row_count'))}`",
            f"- best joint row: `{v9_joint.get('best_run')}` / `{v9_joint.get('best_mask_mode')}` / gain `{_fmt(v9_joint.get('best_gain'))}` / decision `{v9_joint.get('best_decision')}`",
            f"- best joint direct mean / median / wins: `{_fmt(v9_joint.get('best_direct_mean_corr_delta'))}` / `{_fmt(v9_joint.get('best_direct_median_corr_delta'))}` / `{_fmt(v9_joint.get('best_direct_corr_win_fraction'))}`",
            f"- best joint absolute mean / median / wins: `{_fmt(v9_joint.get('best_absolute_mean_corr_delta'))}` / `{_fmt(v9_joint.get('best_absolute_median_corr_delta'))}` / `{_fmt(v9_joint.get('best_absolute_corr_win_fraction'))}`",
            f"- best joint nonbad / weak / moderate / good corr d: `{_fmt(v9_joint.get('best_absolute_nonbad_bin_mean_corr_delta'))}` / `{_fmt(v9_joint.get('best_absolute_weak_bin_mean_corr_delta'))}` / `{_fmt(v9_joint.get('best_absolute_moderate_bin_mean_corr_delta'))}` / `{_fmt(v9_joint.get('best_absolute_good_bin_mean_corr_delta'))}`",
            f"- explicit phase-stable target score candidates: `{_fmt(v9_target_score.get('target_row_candidate_count'))}` / `{_fmt(v9_target_score.get('target_row_count'))}`",
            f"- explicit phase-stable target score best run / decision / score: `{v9_target_score.get('target_row_best_run')}` / `{v9_target_score.get('target_row_best_decision')}` / `{_fmt(v9_target_score.get('target_row_best_score'))}`",
            f"- explicit phase-stable target score direct / absolute / nonbad / good corr d: `{_fmt(v9_target_score.get('target_row_best_direct_mean_corr_delta'))}` / `{_fmt(v9_target_score.get('target_row_best_absolute_mean_corr_delta'))}` / `{_fmt(v9_target_score.get('target_row_best_nonbad_bin_mean_corr_delta'))}` / `{_fmt(v9_target_score.get('target_row_best_good_bin_mean_corr_delta'))}`",
            f"- explicit phase-stable joint target candidates: `{_fmt(v9_target_joint.get('target_row_candidate_count'))}` / `{_fmt(v9_target_joint.get('target_row_count'))}`",
            f"- explicit phase-stable joint target best run / decision / score: `{v9_target_joint.get('target_row_best_run')}` / `{v9_target_joint.get('target_row_best_decision')}` / `{_fmt(v9_target_joint.get('target_row_best_score'))}`",
            f"- explicit phase-stable joint target direct / absolute / nonbad / good corr d: `{_fmt(v9_target_joint.get('target_row_best_direct_mean_corr_delta'))}` / `{_fmt(v9_target_joint.get('target_row_best_absolute_mean_corr_delta'))}` / `{_fmt(v9_target_joint.get('target_row_best_nonbad_bin_mean_corr_delta'))}` / `{_fmt(v9_target_joint.get('target_row_best_good_bin_mean_corr_delta'))}`",
            f"- atlas status / joint status: `{v9_atlas.get('status')}` / `{v9_atlas.get('joint_status')}`",
        ]
    )

    v10_raw = payload["evidence"].get(
        "audio_internal_phase_law_v10_phase_router_masks_raw_compare_no_single_source", {}
    )
    v10_variant = payload["evidence"].get(
        "audio_internal_phase_law_v10_phase_router_masks_variant_compare_no_single_source", {}
    )
    v10_objective = payload["evidence"].get("audio_internal_phase_law_v10_phase_router_masks_objective_score", {})
    v10_joint = payload["evidence"].get("audio_internal_phase_law_v10_phase_router_masks_joint_rows", {})
    v10_stable = payload["evidence"].get("audio_internal_phase_law_v10_phase_stable_target_joint_rows", {})
    v10_coherent = payload["evidence"].get("audio_internal_phase_law_v10_phase_coherent_target_joint_rows", {})
    v9_classes = payload["evidence"].get("audio_internal_phase_law_v9_candidate_classes", {})
    v10_classes = payload["evidence"].get("audio_internal_phase_law_v10_candidate_classes", {})
    v11_raw = payload["evidence"].get(
        "audio_internal_phase_law_v11_phase_router_direct_raw_compare_no_single_source", {}
    )
    v11_variant = payload["evidence"].get(
        "audio_internal_phase_law_v11_phase_router_direct_variant_compare_no_single_source", {}
    )
    v11_objective = payload["evidence"].get("audio_internal_phase_law_v11_phase_router_direct_objective_score", {})
    v11_joint = payload["evidence"].get("audio_internal_phase_law_v11_phase_router_direct_joint_rows", {})
    v11_classes = payload["evidence"].get("audio_internal_phase_law_v11_candidate_classes", {})
    v12_raw = payload["evidence"].get(
        "audio_internal_phase_law_v12_phase_reentry_direct_raw_compare_no_single_source", {}
    )
    v12_variant = payload["evidence"].get(
        "audio_internal_phase_law_v12_phase_reentry_direct_variant_compare_no_single_source", {}
    )
    v12_objective = payload["evidence"].get("audio_internal_phase_law_v12_phase_reentry_direct_objective_score", {})
    v12_joint = payload["evidence"].get("audio_internal_phase_law_v12_phase_reentry_direct_joint_rows", {})
    v12_classes = payload["evidence"].get("audio_internal_phase_law_v12_candidate_classes", {})
    v13_raw = payload["evidence"].get(
        "audio_internal_phase_law_v13_causal_reentry_direct_raw_compare_no_single_source", {}
    )
    v13_variant = payload["evidence"].get(
        "audio_internal_phase_law_v13_causal_reentry_direct_variant_compare_no_single_source", {}
    )
    v13_objective = payload["evidence"].get(
        "audio_internal_phase_law_v13_causal_reentry_direct_objective_score", {}
    )
    v13_joint = payload["evidence"].get("audio_internal_phase_law_v13_causal_reentry_direct_joint_rows", {})
    v13_classes = payload["evidence"].get("audio_internal_phase_law_v13_candidate_classes", {})
    predeclared_diag = payload["evidence"].get("audio_internal_phase_law_predeclared_diagnostic_class", {})
    lines.extend(
        [
            "",
            "## Internal Phase-Law v10 Phase-Router Mask Scout v1",
            "",
            f"- raw absolute status: `{v10_raw.get('status')}`",
            f"- direct variant status: `{v10_variant.get('status')}`",
            f"- objective status / best decision: `{v10_objective.get('status')}` / `{v10_objective.get('best_decision')}`",
            f"- joint-row status / candidates / rows: `{v10_joint.get('status')}` / `{_fmt(v10_joint.get('candidate_count'))}` / `{_fmt(v10_joint.get('row_count'))}`",
            f"- router target candidates: `{_fmt(v10_joint.get('target_row_candidate_count'))}` / `{_fmt(v10_joint.get('target_row_count'))}`",
            f"- router target best run / decision / score: `{v10_joint.get('target_row_best_run')}` / `{v10_joint.get('target_row_best_decision')}` / `{_fmt(v10_joint.get('target_row_best_score'))}`",
            f"- router target direct / absolute / nonbad / good corr d: `{_fmt(v10_joint.get('target_row_best_direct_mean_corr_delta'))}` / `{_fmt(v10_joint.get('target_row_best_absolute_mean_corr_delta'))}` / `{_fmt(v10_joint.get('target_row_best_nonbad_bin_mean_corr_delta'))}` / `{_fmt(v10_joint.get('target_row_best_good_bin_mean_corr_delta'))}`",
            f"- stable target candidates / best score: `{_fmt(v10_stable.get('target_row_candidate_count'))}` / `{_fmt(v10_stable.get('target_row_best_score'))}`",
            f"- coherent-motion target candidates / best score: `{_fmt(v10_coherent.get('target_row_candidate_count'))}` / `{_fmt(v10_coherent.get('target_row_best_score'))}`",
            f"- candidate-class split v9 strict / phase-support: `{_fmt(v9_classes.get('strict_audio_candidate_count'))}` / `{_fmt(v9_classes.get('phase_support_candidate_count'))}`",
            f"- candidate-class split v10 strict / phase-support: `{_fmt(v10_classes.get('strict_audio_candidate_count'))}` / `{_fmt(v10_classes.get('phase_support_candidate_count'))}`",
            f"- v10 best candidate class: `{v10_classes.get('best_class')}` / `{v10_classes.get('best_run')}` / `{v10_classes.get('best_mask_mode')}` / gain `{_fmt(v10_classes.get('best_gain'))}`",
        ]
    )

    lines.extend(
        [
            "",
            "## Internal Phase-Law v11 Phase-Router Direct Scout v1",
            "",
            f"- raw absolute status: `{v11_raw.get('status')}`",
            f"- direct variant status: `{v11_variant.get('status')}`",
            f"- objective status / best decision: `{v11_objective.get('status')}` / `{v11_objective.get('best_decision')}`",
            f"- objective target candidates: `{_fmt(v11_objective.get('target_row_candidate_count'))}` / `{_fmt(v11_objective.get('target_row_count'))}`",
            f"- objective target best run / decision: `{v11_objective.get('target_row_best_run')}` / `{v11_objective.get('target_row_best_decision')}`",
            f"- objective target direct / absolute / nonbad corr d: `{_fmt(v11_objective.get('target_row_best_direct_mean_corr_delta'))}` / `{_fmt(v11_objective.get('target_row_best_absolute_mean_corr_delta'))}` / `{_fmt(v11_objective.get('target_row_best_nonbad_bin_mean_corr_delta'))}`",
            f"- joint-row status / candidates / rows: `{v11_joint.get('status')}` / `{_fmt(v11_joint.get('candidate_count'))}` / `{_fmt(v11_joint.get('row_count'))}`",
            f"- joint target candidates: `{_fmt(v11_joint.get('target_row_candidate_count'))}` / `{_fmt(v11_joint.get('target_row_count'))}`",
            f"- joint best row / decision / score: `{v11_joint.get('best_run')}` / `{v11_joint.get('best_decision')}` / `{_fmt(v11_joint.get('best_score'))}`",
            f"- joint best direct / absolute / nonbad / good corr d: `{_fmt(v11_joint.get('best_direct_mean_corr_delta'))}` / `{_fmt(v11_joint.get('best_absolute_mean_corr_delta'))}` / `{_fmt(v11_joint.get('best_absolute_nonbad_bin_mean_corr_delta'))}` / `{_fmt(v11_joint.get('best_absolute_good_bin_mean_corr_delta'))}`",
            f"- candidate-class split v11 strict / phase-support: `{_fmt(v11_classes.get('strict_audio_candidate_count'))}` / `{_fmt(v11_classes.get('phase_support_candidate_count'))}`",
            f"- v11 best candidate class: `{v11_classes.get('best_class')}` / `{v11_classes.get('best_run')}` / `{v11_classes.get('best_mask_mode')}` / gain `{_fmt(v11_classes.get('best_gain'))}`",
        ]
    )

    lines.extend(
        [
            "",
            "## Internal Phase-Law v12 Phase-Reentry Direct Scout v1",
            "",
            f"- raw absolute status: `{v12_raw.get('status')}`",
            f"- direct variant status: `{v12_variant.get('status')}`",
            f"- objective status / best decision: `{v12_objective.get('status')}` / `{v12_objective.get('best_decision')}`",
            f"- objective target candidates: `{_fmt(v12_objective.get('target_row_candidate_count'))}` / `{_fmt(v12_objective.get('target_row_count'))}`",
            f"- objective target best run / decision: `{v12_objective.get('target_row_best_run')}` / `{v12_objective.get('target_row_best_decision')}`",
            f"- objective target direct / absolute / nonbad corr d: `{_fmt(v12_objective.get('target_row_best_direct_mean_corr_delta'))}` / `{_fmt(v12_objective.get('target_row_best_absolute_mean_corr_delta'))}` / `{_fmt(v12_objective.get('target_row_best_nonbad_bin_mean_corr_delta'))}`",
            f"- joint-row status / candidates / rows: `{v12_joint.get('status')}` / `{_fmt(v12_joint.get('candidate_count'))}` / `{_fmt(v12_joint.get('row_count'))}`",
            f"- joint target candidates: `{_fmt(v12_joint.get('target_row_candidate_count'))}` / `{_fmt(v12_joint.get('target_row_count'))}`",
            f"- joint best row / decision / score: `{v12_joint.get('best_run')}` / `{v12_joint.get('best_decision')}` / `{_fmt(v12_joint.get('best_score'))}`",
            f"- joint best direct / absolute / nonbad / good corr d: `{_fmt(v12_joint.get('best_direct_mean_corr_delta'))}` / `{_fmt(v12_joint.get('best_absolute_mean_corr_delta'))}` / `{_fmt(v12_joint.get('best_absolute_nonbad_bin_mean_corr_delta'))}` / `{_fmt(v12_joint.get('best_absolute_good_bin_mean_corr_delta'))}`",
            f"- candidate-class split v12 strict / phase-support: `{_fmt(v12_classes.get('strict_audio_candidate_count'))}` / `{_fmt(v12_classes.get('phase_support_candidate_count'))}`",
            f"- v12 best candidate class: `{v12_classes.get('best_class')}` / `{v12_classes.get('best_run')}` / `{v12_classes.get('best_mask_mode')}` / gain `{_fmt(v12_classes.get('best_gain'))}`",
        ]
    )

    lines.extend(
        [
            "",
            "## Internal Phase-Law v13 Causal Phase-Reentry Direct Scout v1",
            "",
            f"- raw absolute status: `{v13_raw.get('status')}`",
            f"- direct variant status: `{v13_variant.get('status')}`",
            f"- objective status / best decision: `{v13_objective.get('status')}` / `{v13_objective.get('best_decision')}`",
            f"- objective target candidates: `{_fmt(v13_objective.get('target_row_candidate_count'))}` / `{_fmt(v13_objective.get('target_row_count'))}`",
            f"- objective target best run / decision: `{v13_objective.get('target_row_best_run')}` / `{v13_objective.get('target_row_best_decision')}`",
            f"- objective target direct / absolute / nonbad corr d: `{_fmt(v13_objective.get('target_row_best_direct_mean_corr_delta'))}` / `{_fmt(v13_objective.get('target_row_best_absolute_mean_corr_delta'))}` / `{_fmt(v13_objective.get('target_row_best_nonbad_bin_mean_corr_delta'))}`",
            f"- joint-row status / candidates / rows: `{v13_joint.get('status')}` / `{_fmt(v13_joint.get('candidate_count'))}` / `{_fmt(v13_joint.get('row_count'))}`",
            f"- joint target candidates: `{_fmt(v13_joint.get('target_row_candidate_count'))}` / `{_fmt(v13_joint.get('target_row_count'))}`",
            f"- joint best row / decision / score: `{v13_joint.get('best_run')}` / `{v13_joint.get('best_decision')}` / `{_fmt(v13_joint.get('best_score'))}`",
            f"- joint best direct / absolute / nonbad / good corr d: `{_fmt(v13_joint.get('best_direct_mean_corr_delta'))}` / `{_fmt(v13_joint.get('best_absolute_mean_corr_delta'))}` / `{_fmt(v13_joint.get('best_absolute_nonbad_bin_mean_corr_delta'))}` / `{_fmt(v13_joint.get('best_absolute_good_bin_mean_corr_delta'))}`",
            f"- candidate-class split v13 strict / phase-support: `{_fmt(v13_classes.get('strict_audio_candidate_count'))}` / `{_fmt(v13_classes.get('phase_support_candidate_count'))}`",
            f"- v13 best candidate class: `{v13_classes.get('best_class')}` / `{v13_classes.get('best_run')}` / `{v13_classes.get('best_mask_mode')}` / gain `{_fmt(v13_classes.get('best_gain'))}`",
        ]
    )

    lines.extend(
        [
            "",
            "## Internal Phase-Law Predeclared Diagnostic Class",
            "",
            f"- status: `{predeclared_diag.get('status')}`",
            f"- class: `{predeclared_diag.get('class_name')}`",
            f"- promotion effect: `{predeclared_diag.get('promotion_effect')}`",
            f"- diagnostic / strict rows: `{_fmt(predeclared_diag.get('diagnostic_count'))}` / `{_fmt(predeclared_diag.get('strict_audio_count'))}`",
            f"- total rows / tracks: `{_fmt(predeclared_diag.get('total_row_count'))}` / `{_fmt(predeclared_diag.get('track_count'))}`",
            f"- best track by count: `{predeclared_diag.get('best_track_by_count')}`",
            f"- best row: `{predeclared_diag.get('best_track')}` / `{predeclared_diag.get('best_run')}` / `{predeclared_diag.get('best_mask_mode')}` / gain `{_fmt(predeclared_diag.get('best_gain'))}`",
            f"- best direct / absolute / nonbad / good corr d: `{_fmt(predeclared_diag.get('best_direct_mean_corr_delta'))}` / `{_fmt(predeclared_diag.get('best_absolute_mean_corr_delta'))}` / `{_fmt(predeclared_diag.get('best_absolute_nonbad_bin_mean_corr_delta'))}` / `{_fmt(predeclared_diag.get('best_absolute_good_bin_mean_corr_delta'))}`",
        ]
    )

    for section_title, scout_key in [
        ("Audio Delta Objective Scout", "audio_delta_objective_scout"),
        ("Audio Delta Objective Scout Relsig Base", "audio_delta_objective_scout_relsig"),
    ]:
        scout = payload["evidence"].get(scout_key, {})
        lines.extend(
            [
                "",
                f"## {section_title}",
                "",
                f"- status: `{scout.get('status')}`",
                f"- raw status: `{scout.get('raw_status')}`",
                f"- cases: `{_fmt(scout.get('case_count'))}`",
                f"- candidates: `{_fmt(scout.get('candidate_count'))}`",
                f"- objective: `{scout.get('objective_mode')}` / `{scout.get('objective_mask')}` / gain `{_fmt(scout.get('objective_gain'))}`",
                f"- best candidate: `{scout.get('best_candidate')}` / `{scout.get('best_candidate_status')}`",
                f"- best candidate corr delta: `{_fmt(scout.get('best_primary_corr_delta'))}`",
                f"- best candidate MSE delta: `{_fmt(scout.get('best_primary_mse_delta'))}`",
                f"- best candidate corr wins: `{_fmt(scout.get('best_primary_corr_win_fraction'))}`",
                f"- best candidate median corr delta: `{_fmt(scout.get('best_median_corr_delta'))}`",
                f"- best candidate min leave-one-out corr delta: `{_fmt(scout.get('best_min_leave_one_out_corr_delta'))}`",
                "",
                "| candidate | status | score | corr delta | MSE delta | corr wins | MSE wins | vs gain0 corr | median corr d | leave-one-out d | max gain share | best corr d | best MSE d |",
                "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for row in scout.get("rows", []) or []:
            lines.append(
                "| {candidate} | {status} | {score} | {corr} | {mse} | {cwins} | {mwins} | {carrier} | {median} | {loo} | {share} | {bestcorr} | {bestmse} |".format(
                    candidate=row.get("candidate"),
                    status=row.get("status"),
                    score=_fmt(row.get("score")),
                    corr=_fmt(row.get("primary_corr_delta")),
                    mse=_fmt(row.get("primary_mse_delta")),
                    cwins=_fmt(row.get("primary_corr_win_fraction")),
                    mwins=_fmt(row.get("primary_mse_win_fraction")),
                    carrier=_fmt(row.get("primary_vs_gain0_corr")),
                    median=_fmt(row.get("median_corr_delta")),
                    loo=_fmt(row.get("min_leave_one_out_corr_delta")),
                    share=_fmt(row.get("max_positive_corr_gain_share")),
                    bestcorr=_fmt(row.get("best_corr_delta")),
                    bestmse=_fmt(row.get("best_mse_delta")),
                )
            )

    dense_claim = payload["evidence"].get("dense_signature_claim", {})
    lines.extend(
        [
            "",
            "## Dense Signature Claim",
            "",
            f"- status: `{dense_claim.get('status')}`",
            f"- dense wins: `{dense_claim.get('dense_wins')}`",
            f"- cases: `{_fmt(dense_claim.get('num_cases'))}`",
            f"- clip seconds: `{_fmt(dense_claim.get('clip_seconds'))}`",
            f"- dense stability score: `{_fmt(dense_claim.get('dense_stability_score'))}`",
            f"- metadata stability score: `{_fmt(dense_claim.get('metadata_stability_score'))}`",
            f"- stability delta dense-minus-metadata: `{_fmt(dense_claim.get('stability_delta_dense_minus_metadata'))}`",
            f"- dense separation score: `{_fmt(dense_claim.get('dense_separation_score'))}`",
            f"- metadata separation score: `{_fmt(dense_claim.get('metadata_separation_score'))}`",
            f"- separation delta dense-minus-metadata: `{_fmt(dense_claim.get('separation_delta_dense_minus_metadata'))}`",
            f"- dense dominant cluster share: `{_fmt(dense_claim.get('dense_dominant_cluster_share'))}`",
            f"- metadata dominant cluster share: `{_fmt(dense_claim.get('metadata_dominant_cluster_share'))}`",
            f"- dense near-duplicate pair share: `{_fmt(dense_claim.get('dense_near_duplicate_pair_share'))}`",
            f"- metadata near-duplicate pair share: `{_fmt(dense_claim.get('metadata_near_duplicate_pair_share'))}`",
        ]
    )
    if dense_claim.get("reasons"):
        lines.extend(["", "Reasons:"])
        for reason in dense_claim.get("reasons", []) or []:
            lines.append(f"- {reason}")

    dense_fail = payload["evidence"].get("dense_signature_failure_modes", {})
    lines.extend(
        [
            "",
            "## Dense Signature Failure Modes",
            "",
            f"- status: `{dense_fail.get('status')}`",
            f"- promotion effect: `{dense_fail.get('promotion_effect')}`",
            f"- all required axes pass: `{dense_fail.get('all_required_axes_pass')}`",
            f"- matryoshka readiness: `{dense_fail.get('matryoshka_readiness_status')}`",
            f"- operator tail distinctness: `{dense_fail.get('operator_tail_distinctness_status')}`",
            f"- stability delta: `{_fmt(dense_fail.get('stability_delta'))}`",
            f"- separation delta: `{_fmt(dense_fail.get('separation_delta'))}`",
            f"- dominant cluster share delta: `{_fmt(dense_fail.get('dominant_cluster_share_delta'))}`",
            f"- near-duplicate pair share delta: `{_fmt(dense_fail.get('near_duplicate_pair_share_delta'))}`",
            f"- prefix status: `{dense_fail.get('prefix_status')}`",
            f"- prefix monotonicity score: `{_fmt(dense_fail.get('prefix_monotonicity_score'))}`",
            f"- prefix spread mean: `{_fmt(dense_fail.get('prefix_spread_mean'))}`",
            f"- early-minus-full mean: `{_fmt(dense_fail.get('early_minus_full_mean'))}`",
        ]
    )
    if dense_fail.get("blocking_failures"):
        lines.extend(["", "Blocking failures:"])
        for failure in dense_fail.get("blocking_failures", []) or []:
            lines.append(f"- `{failure}`")
    if dense_fail.get("recommended_next_losses"):
        lines.extend(["", "Recommended next losses:"])
        for loss in dense_fail.get("recommended_next_losses", []) or []:
            lines.append(f"- {loss}")

    sig_contract = payload["evidence"].get("relational_signature_contract_audit", {})
    lines.extend(
        [
            "",
            "## Relational Signature Contract Audit",
            "",
            f"- status: `{sig_contract.get('status')}`",
            f"- schema contract passed: `{sig_contract.get('schema_contract_passed')}`",
            f"- learned body status: `{sig_contract.get('learned_body_status')}`",
            f"- signature dim: `{_fmt(sig_contract.get('signature_dim'))}`",
            f"- prefix dims: `{sig_contract.get('prefix_dims')}`",
            f"- operator seed dim: `{_fmt(sig_contract.get('operator_seed_dim'))}`",
            f"- defines nn.Module: `{sig_contract.get('defines_nn_module')}`",
            f"- learned head tokens present: `{sig_contract.get('learned_head_tokens_present')}`",
            f"- deterministic assemblers: `{sig_contract.get('deterministic_assemblers')}`",
        ]
    )
    if sig_contract.get("required_next_implementation"):
        lines.extend(["", "Required next implementation:"])
        for item in sig_contract.get("required_next_implementation", []) or []:
            lines.append(f"- {item}")

    sig_learning = payload["evidence"].get("relational_signature_learning_contract", {})
    lines.extend(
        [
            "",
            "## Relational Signature Learning Contract",
            "",
            f"- status: `{sig_learning.get('status')}`",
            f"- claim effect: `{sig_learning.get('claim_effect')}`",
            f"- runtime effect: `{sig_learning.get('runtime_effect')}`",
            f"- batch: `{_fmt(sig_learning.get('batch'))}`",
            f"- hidden dim: `{_fmt(sig_learning.get('hidden_dim'))}`",
            f"- train steps: `{_fmt(sig_learning.get('train_steps'))}`",
            f"- initial total loss: `{_fmt(sig_learning.get('initial_total_loss'))}`",
            f"- final total loss: `{_fmt(sig_learning.get('final_total_loss'))}`",
            f"- loss reduction: `{_fmt(sig_learning.get('loss_reduction'))}`",
            f"- gradient norm: `{_fmt(sig_learning.get('gradient_norm'))}`",
            f"- h shape: `{sig_learning.get('h_shape')}`",
            f"- branch shape: `{sig_learning.get('branch_profile_shape')}`",
            f"- law signature shape: `{sig_learning.get('law_signature_shape')}`",
            f"- operator seed shape: `{sig_learning.get('operator_seed_shape')}`",
            f"- max h unit norm error: `{_fmt(sig_learning.get('max_h_unit_norm_error'))}`",
            f"- max law unit norm error: `{_fmt(sig_learning.get('max_law_unit_norm_error'))}`",
            f"- max q sum error: `{_fmt(sig_learning.get('max_q_sum_error'))}`",
        ]
    )
    if sig_learning.get("checks"):
        lines.extend(["", "Checks:"])
        for name, value in (sig_learning.get("checks") or {}).items():
            lines.append(f"- {name}: `{value}`")

    learned_scout = payload["evidence"].get("learned_signature_scout", {})
    lines.extend(
        [
            "",
            "## Learned Signature Scout",
            "",
            f"- status: `{learned_scout.get('status')}`",
            f"- promotion effect: `{learned_scout.get('promotion_effect')}`",
            f"- split mode: `{learned_scout.get('split_mode')}`",
            f"- cases: `{_fmt(learned_scout.get('num_cases'))}`",
            f"- transforms: `{_fmt(learned_scout.get('num_transforms'))}`",
            f"- train packets: `{_fmt(learned_scout.get('train_packet_count'))}`",
            f"- train loss reduction: `{_fmt(learned_scout.get('train_loss_reduction'))}`",
            f"- learned stability score: `{_fmt(learned_scout.get('learned_stability_score'))}`",
            f"- metadata stability score: `{_fmt(learned_scout.get('metadata_stability_score'))}`",
            f"- stability delta learned-minus-metadata: `{_fmt(learned_scout.get('stability_delta_learned_minus_metadata'))}`",
            f"- learned separation score: `{_fmt(learned_scout.get('learned_separation_score'))}`",
            f"- metadata separation score: `{_fmt(learned_scout.get('metadata_separation_score'))}`",
            f"- separation delta learned-minus-metadata: `{_fmt(learned_scout.get('separation_delta_learned_minus_metadata'))}`",
            f"- learned dominant cluster share: `{_fmt(learned_scout.get('learned_dominant_cluster_share'))}`",
            f"- learned near-duplicate pair share: `{_fmt(learned_scout.get('learned_near_duplicate_pair_share'))}`",
            f"- operator-tail effective rank: `{_fmt(learned_scout.get('operator_tail_effective_rank'))}`",
            f"- operator-tail effective cluster count: `{_fmt(learned_scout.get('operator_tail_effective_cluster_count'))}`",
        ]
    )
    if learned_scout.get("reasons"):
        lines.extend(["", "Reasons:"])
        for reason in learned_scout.get("reasons", []) or []:
            lines.append(f"- {reason}")

    learned_compare = payload["evidence"].get("learned_signature_scout_compare", {})
    learned_seed = (
        learned_compare.get("seed_stability", {}).get("best_seeded_family", {})
        if isinstance(learned_compare.get("seed_stability"), dict)
        else {}
    )
    learned_heldout_seed = (
        learned_compare.get("heldout_seed_stability", {}).get("best_seeded_family", {})
        if isinstance(learned_compare.get("heldout_seed_stability"), dict)
        else {}
    )
    lines.extend(
        [
            "",
            "## Learned Signature Scout Compare",
            "",
            f"- status: `{learned_compare.get('status')}`",
            f"- promotion effect: `{learned_compare.get('promotion_effect')}`",
            f"- rows: `{_fmt(learned_compare.get('row_count'))}`",
            f"- candidates: `{_fmt(learned_compare.get('candidate_count'))}`",
            f"- two-axis candidates: `{_fmt(learned_compare.get('two_axis_candidate_count'))}`",
            f"- stability+anti-collapse candidates: `{_fmt(learned_compare.get('stability_anticollapse_candidate_count'))}`",
            f"- heldout rows: `{_fmt(learned_compare.get('heldout_row_count'))}`",
            f"- heldout candidates: `{_fmt(learned_compare.get('heldout_candidate_count'))}`",
            f"- heldout two-axis candidates: `{_fmt(learned_compare.get('heldout_two_axis_candidate_count'))}`",
            f"- heldout stability+anti-collapse candidates: `{_fmt(learned_compare.get('heldout_stability_anticollapse_candidate_count'))}`",
            f"- pareto rows: `{_fmt(learned_compare.get('pareto_count'))}`",
            f"- best separation profile: `{learned_compare.get('best_separation_profile')}`",
            f"- best stability profile: `{learned_compare.get('best_stability_profile')}`",
            f"- best separation delta: `{_fmt(learned_compare.get('best_separation_delta'))}`",
            f"- best stability delta: `{_fmt(learned_compare.get('best_stability_delta'))}`",
            f"- best seeded family: `{learned_seed.get('family')}`",
            f"- seeded candidate fraction: `{_fmt(learned_seed.get('candidate_seed_fraction'))}`",
            f"- seeded candidate / failed seeds: `{learned_seed.get('candidate_seeds')}` / `{learned_seed.get('failed_seeds')}`",
            f"- best heldout seeded family: `{learned_heldout_seed.get('family')}`",
            f"- heldout seeded candidate fraction: `{_fmt(learned_heldout_seed.get('candidate_seed_fraction'))}`",
            f"- heldout seeded candidate / failed seeds: `{learned_heldout_seed.get('candidate_seeds')}` / `{learned_heldout_seed.get('failed_seeds')}`",
        ]
    )
    if learned_compare.get("rows"):
        lines.extend(
            [
                "",
                "| profile | seed | class | stability d | separation d | rank d | dist d | op clusters |",
                "|---|---:|---|---:|---:|---:|---:|---:|",
            ]
        )
        for row in learned_compare.get("rows", []) or []:
            lines.append(
                "| {profile} | {seed} | {klass} | {stable} | {sep} | {rank} | {dist} | {op} |".format(
                    profile=row.get("profile"),
                    seed=_fmt(row.get("seed")),
                    klass=row.get("profile_class"),
                    stable=_fmt(row.get("stability_delta_learned_minus_metadata")),
                    sep=_fmt(row.get("separation_delta_learned_minus_metadata")),
                    rank=_fmt(row.get("collapse_effective_rank_delta")),
                    dist=_fmt(row.get("collapse_mean_pairwise_distance_delta")),
                    op=_fmt(row.get("operator_tail_effective_cluster_count")),
                )
            )

    learned_split = payload["evidence"].get("learned_signature_split_suite", {})
    lines.extend(
        [
            "",
            "## Learned Signature Split Suite",
            "",
            f"- status: `{learned_split.get('status')}`",
            f"- profile: `{learned_split.get('profile')}`",
            f"- split set: `{learned_split.get('split_set')}`",
            f"- splits: `{_fmt(learned_split.get('split_count'))}`",
            f"- completed: `{_fmt(learned_split.get('completed_count'))}`",
            f"- failed: `{_fmt(learned_split.get('failed_count'))}`",
            f"- candidates: `{_fmt(learned_split.get('candidate_count'))}`",
            f"- heldout candidates: `{_fmt(learned_split.get('heldout_candidate_count'))}` / `{_fmt(learned_split.get('heldout_row_count'))}`",
            f"- candidate fraction: `{_fmt(learned_split.get('candidate_fraction'))}`",
            f"- seed candidate fraction: `{_fmt(learned_split.get('seed_candidate_fraction'))}`",
            f"- partition candidate fraction: `{_fmt(learned_split.get('partition_candidate_fraction'))}`",
            f"- mean stability delta: `{_fmt(learned_split.get('mean_stability_delta_learned_minus_metadata'))}`",
            f"- mean separation delta: `{_fmt(learned_split.get('mean_separation_delta_learned_minus_metadata'))}`",
            f"- mean effective-rank delta: `{_fmt(learned_split.get('mean_collapse_effective_rank_delta'))}`",
        ]
    )
    if learned_split.get("rows"):
        lines.extend(
            [
                "",
                "| split | kind | heldout | candidate | stability d | separation d | rank d | dist d |",
                "|---|---|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for row in learned_split.get("rows", []) or []:
            lines.append(
                "| {split} | {kind} | {heldout} | {candidate} | {stable} | {sep} | {rank} | {dist} |".format(
                    split=row.get("split_id"),
                    kind=row.get("kind"),
                    heldout="yes" if row.get("heldout_evidence") else "no",
                    candidate="yes" if row.get("heldout_candidate") else "no",
                    stable=_fmt(row.get("stability_delta_learned_minus_metadata")),
                    sep=_fmt(row.get("separation_delta_learned_minus_metadata")),
                    rank=_fmt(row.get("collapse_effective_rank_delta")),
                    dist=_fmt(row.get("collapse_mean_pairwise_distance_delta")),
                )
            )
    if learned_split.get("hard_stability_cases"):
        lines.extend(
            [
                "",
                "| hard case | hits | min stability d | splits |",
                "|---|---:|---:|---|",
            ]
        )
        for row in learned_split.get("hard_stability_cases", []) or []:
            lines.append(
                "| {case} | {hits} | {stable} | {splits} |".format(
                    case=row.get("case"),
                    hits=_fmt(row.get("hit_count")),
                    stable=_fmt(row.get("min_stability_delta_learned_minus_metadata")),
                    splits=", ".join(str(item) for item in row.get("split_ids", [])[:4]),
                )
            )

    learned_split_compare = payload["evidence"].get("learned_signature_split_suite_compare", {})
    lines.extend(
        [
            "",
            "## Learned Signature Split Suite Compare",
            "",
            f"- status: `{learned_split_compare.get('status')}`",
            f"- rows: `{_fmt(learned_split_compare.get('row_count'))}`",
            f"- robust candidates: `{_fmt(learned_split_compare.get('robust_candidate_count'))}`",
            f"- fragile candidates: `{_fmt(learned_split_compare.get('fragile_candidate_count'))}`",
            f"- best profile: `{learned_split_compare.get('best_profile')}`",
            f"- best status: `{learned_split_compare.get('best_status')}`",
            f"- best candidate fraction: `{_fmt(learned_split_compare.get('best_candidate_fraction'))}`",
            f"- best seed fraction: `{_fmt(learned_split_compare.get('best_seed_candidate_fraction'))}`",
            f"- best partition fraction: `{_fmt(learned_split_compare.get('best_partition_candidate_fraction'))}`",
            f"- best mean stability delta: `{_fmt(learned_split_compare.get('best_mean_stability_delta'))}`",
        ]
    )
    if learned_split_compare.get("rows"):
        lines.extend(
            [
                "",
                "| profile | status | cand frac | seed frac | partition frac | mean stability d | mean separation d |",
                "|---|---|---:|---:|---:|---:|---:|",
            ]
        )
        for row in learned_split_compare.get("rows", []) or []:
            lines.append(
                "| {profile} | {status} | {cand} | {seed} | {part} | {stable} | {sep} |".format(
                    profile=row.get("profile"),
                    status=row.get("status"),
                    cand=_fmt(row.get("candidate_fraction")),
                    seed=_fmt(row.get("seed_candidate_fraction")),
                    part=_fmt(row.get("partition_candidate_fraction")),
                    stable=_fmt(row.get("mean_stability_delta_learned_minus_metadata")),
                    sep=_fmt(row.get("mean_separation_delta_learned_minus_metadata")),
                )
            )

    learned_case_fail = payload["evidence"].get("learned_signature_case_failures", {})
    lines.extend(
        [
            "",
            "## Learned Signature Case Failures",
            "",
            f"- status: `{learned_case_fail.get('status')}`",
            f"- heldout case rows: `{_fmt(learned_case_fail.get('heldout_case_row_count'))}`",
            f"- cases: `{_fmt(learned_case_fail.get('case_count'))}`",
        ]
    )
    if learned_case_fail.get("worst_cases"):
        lines.extend(
            [
                "",
                "| case | rows | pass frac | mean stability d | worst stability d | worst profile |",
                "|---|---:|---:|---:|---:|---|",
            ]
        )
        for row in learned_case_fail.get("worst_cases", []) or []:
            lines.append(
                "| {case} | {rows} | {frac} | {mean} | {worst} | {profile} |".format(
                    case=row.get("case"),
                    rows=_fmt(row.get("row_count")),
                    frac=_fmt(row.get("stability_pass_fraction")),
                    mean=_fmt(row.get("mean_stability_delta")),
                    worst=_fmt(row.get("worst_stability_delta")),
                    profile=row.get("worst_profile"),
                )
            )

    op_tail = payload["evidence"].get("signature_operator_tail_probe_compare", {})
    lines.extend(
        [
            "",
            "## Signature Operator-Tail Probe Compare",
            "",
            f"- status: `{op_tail.get('status')}`",
            f"- promotion effect: `{op_tail.get('promotion_effect')}`",
            f"- rows: `{_fmt(op_tail.get('row_count'))}`",
            f"- learned wins: `{_fmt(op_tail.get('learned_win_count'))}`",
            f"- learned win fraction: `{_fmt(op_tail.get('learned_win_fraction'))}`",
        ]
    )
    if op_tail.get("rows"):
        lines.extend(
            [
                "",
                "| profile | offset | target | winner | learned beats explicit | learned mse | explicit mse |",
                "|---|---:|---|---|---|---:|---:|",
            ]
        )
        for row in op_tail.get("rows", []) or []:
            lines.append(
                "| {profile} | {offset} | {target} | {winner} | {beats} | {learned} | {explicit} |".format(
                    profile=row.get("profile"),
                    offset=_fmt(row.get("case_offset")),
                    target=row.get("target"),
                    winner=row.get("winning_view"),
                    beats=row.get("learned_beats_explicit"),
                    learned=_fmt(row.get("best_learned_heldout_mse")),
                    explicit=_fmt(row.get("explicit_heldout_mse")),
                )
            )

    future_law = payload["evidence"].get("signature_future_law_probe", {})
    lines.extend(
        [
            "",
            "## Signature Future-Law Probe",
            "",
            f"- status: `{future_law.get('status')}`",
              f"- evidence: `{future_law.get('evidence')}`",
              f"- promotion effect: `{future_law.get('promotion_effect')}`",
              f"- ridge alpha policy: `{future_law.get('ridge_alpha_selection_policy')}`",
              f"- leakage guard: `{future_law.get('leakage_guard')}`",
              f"- targets: `{_fmt(future_law.get('target_count'))}`",
            f"- learned wins: `{_fmt(future_law.get('learned_win_count'))}`",
            f"- learned win fraction: `{_fmt(future_law.get('learned_win_fraction'))}`",
            f"- learned beats non-law count: `{_fmt(future_law.get('learned_beats_non_law_count'))}`",
            f"- full-factor wins: `{_fmt(future_law.get('full_factor_win_count'))}`",
            f"- train / heldout examples: `{_fmt(future_law.get('train_example_count'))}` / `{_fmt(future_law.get('heldout_example_count'))}`",
        ]
    )
    if future_law.get("target_rows"):
        lines.extend(
            [
                "",
                "| target | best view | learned wins | learned beats non-law | learned mse | non-law mse | full-factor mse |",
                "|---|---|---|---|---:|---:|---:|",
            ]
        )
        for row in future_law.get("target_rows", []) or []:
            lines.append(
                "| {target} | {best} | {wins} | {beats} | {learned} | {nonlaw} | {full} |".format(
                    target=row.get("target"),
                    best=row.get("best_view"),
                    wins=row.get("learned_wins"),
                    beats=row.get("learned_beats_non_law"),
                    learned=_fmt(row.get("best_learned_heldout_mse")),
                    nonlaw=_fmt(row.get("explicit_non_law_heldout_mse")),
                    full=_fmt(row.get("explicit_full_factor_heldout_mse")),
                )
            )

    tail_use = payload["evidence"].get("signature_tail_incremental_usefulness", {})
    lines.extend(
        [
            "",
            "## Signature Tail Incremental Usefulness",
            "",
            f"- status: `{tail_use.get('status')}`",
              f"- evidence: `{tail_use.get('evidence')}`",
              f"- promotion effect: `{tail_use.get('promotion_effect')}`",
              f"- ridge alpha policy: `{tail_use.get('ridge_alpha_selection_policy')}`",
              f"- leakage guard: `{tail_use.get('leakage_guard')}`",
              f"- targets: `{_fmt(tail_use.get('target_count'))}`",
            f"- tail wins: `{_fmt(tail_use.get('tail_win_count'))}`",
            f"- tail win fraction: `{_fmt(tail_use.get('tail_win_fraction'))}`",
            f"- tail beats prefix / permuted / cross-case / random: `{_fmt(tail_use.get('tail_beats_prefix_count'))}` / `{_fmt(tail_use.get('tail_beats_permuted_count'))}` / `{_fmt(tail_use.get('tail_beats_cross_case_count'))}` / `{_fmt(tail_use.get('tail_beats_random_count'))}`",
            f"- shuffle-control wins: `{_fmt(tail_use.get('shuffle_control_win_count'))}`",
            f"- prefix-control wins: `{_fmt(tail_use.get('prefix_control_win_count'))}`",
            f"- cross-case-control wins: `{_fmt(tail_use.get('cross_case_control_win_count'))}`",
            f"- random-control wins: `{_fmt(tail_use.get('random_control_win_count'))}`",
            f"- total control wins: `{_fmt(tail_use.get('control_win_count'))}`",
            f"- train / heldout packets: `{_fmt(tail_use.get('train_packet_count'))}` / `{_fmt(tail_use.get('heldout_packet_count'))}`",
        ]
    )
    if tail_use.get("target_rows"):
        lines.extend(
            [
                "",
                "| target | tail win | beats prefix | beats permuted | beats cross-case | beats random | metadata mse | tail mse | prefix mse | random mse |",
                "|---|---|---|---|---|---|---:|---:|---:|---:|",
            ]
        )
        for row in tail_use.get("target_rows", []) or []:
            lines.append(
                "| {target} | {win} | {prefix} | {perm} | {cross} | {random_win} | {metadata} | {tail} | {prefix_mse} | {random_mse} |".format(
                    target=row.get("target"),
                    win=row.get("tail_win"),
                    prefix=row.get("tail_beats_prefix"),
                    perm=row.get("tail_beats_permuted"),
                    cross=row.get("tail_beats_cross_case"),
                    random_win=row.get("tail_beats_random"),
                    metadata=_fmt(row.get("metadata_heldout_mse")),
                    tail=_fmt(row.get("tail_heldout_mse")),
                    prefix_mse=_fmt(row.get("prefix_heldout_mse")),
                    random_mse=_fmt(row.get("random_tail_heldout_mse")),
                )
            )

    semantic = payload["evidence"].get("semantic_projector_contract", {})
    lines.extend(
        [
            "",
            "## Semantic Projector Contract",
            "",
            f"- status: `{semantic.get('status')}`",
            f"- import ok: `{semantic.get('import_ok')}`",
            f"- projection path: `{semantic.get('projection_path')}`",
            f"- control schema exact: `{semantic.get('control_schema_exact')}`",
            f"- prompt family count: `{_fmt(semantic.get('prompt_family_count'))}`",
            f"- pairwise diversity count: `{_fmt(semantic.get('pairwise_diversity_count'))}`",
            f"- all pairs diverse: `{semantic.get('all_pairs_diverse')}`",
        ]
    )

    substrate = payload["evidence"]["substrate_lattice"]
    lines.extend(["", "## Substrate", "", f"- status: `{substrate.get('status')}`"])
    if substrate.get("rows"):
        lines.extend(
            [
                "",
                "| variant | status | parent | child | phase-only | major gain | loss |",
                "|---|---|---:|---:|---:|---:|---:|",
            ]
        )
        for row in substrate["rows"]:
            lines.append(
                "| {variant} | {status} | {parent} | {child} | {phase} | {major} | {loss} |".format(
                    variant=row.get("variant"),
                    status=row.get("status"),
                    parent=_fmt(row.get("parent_branch")),
                    child=_fmt(row.get("child_branch")),
                    phase=_fmt(row.get("phase_only_branch")),
                    major=_fmt(row.get("major_gain")),
                    loss=_fmt(row.get("mean_loss")),
                )
            )

    claim_iso = payload["evidence"].get("claim_isolation", {})
    lines.extend(
        [
            "",
            "## Unified Claim Isolation",
            "",
            f"- status: `{claim_iso.get('status')}`",
            f"- time steps: `{claim_iso.get('time_steps')}`",
            f"- seed plan: `{claim_iso.get('seed_plan')}`",
        ]
    )
    for track in claim_iso.get("tracks", []) or []:
        lines.extend(
            [
                "",
                f"### {track.get('track')}",
                "",
                "| variant | parent | child | phase-only | naked phase | gate | phase delta | support | law families | major gain | loss | d loss |",
                "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for row in track.get("rows", []) or []:
            lines.append(
                "| {variant} | {parent} | {child} | {phase} | {naked_phase} | {gate} | {phase_delta} | {support} | {law} | {major} | {loss} | {dloss} |".format(
                    variant=row.get("variant"),
                    parent=_fmt(row.get("parent_branch")),
                    child=_fmt(row.get("child_branch")),
                    phase=_fmt(row.get("phase_only_branch")),
                    naked_phase=_fmt(row.get("naked_phase_only_branch")),
                    gate=_fmt(row.get("gate_mass")),
                    phase_delta=_fmt(row.get("phase_delta")),
                    support=_fmt(row.get("support_writeback")),
                    law=_fmt(row.get("law_families")),
                    major=_fmt(row.get("major_gain")),
                    loss=_fmt(row.get("loss")),
                    dloss=_fmt(row.get("delta_loss")),
                )
            )

    qtrace_iso = payload["evidence"].get("qtrace_inheritance", {})
    lines.extend(
        [
            "",
            "## qtrace Inheritance",
            "",
            f"- status: `{qtrace_iso.get('status')}`",
            f"- time steps: `{qtrace_iso.get('time_steps')}`",
            f"- seed plan: `{qtrace_iso.get('seed_plan')}`",
        ]
    )
    for track in qtrace_iso.get("tracks", []) or []:
        lines.extend(
            [
                "",
                f"### {track.get('track')}",
                "",
                "| variant | parent | child | phase-only | naked phase | gate | phase delta | support | major gain | loss | d loss |",
                "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for row in track.get("rows", []) or []:
            lines.append(
                "| {variant} | {parent} | {child} | {phase} | {naked_phase} | {gate} | {phase_delta} | {support} | {major} | {loss} | {dloss} |".format(
                    variant=row.get("variant"),
                    parent=_fmt(row.get("parent_branch")),
                    child=_fmt(row.get("child_branch")),
                    phase=_fmt(row.get("phase_only_branch")),
                    naked_phase=_fmt(row.get("naked_phase_only_branch")),
                    gate=_fmt(row.get("gate_mass")),
                    phase_delta=_fmt(row.get("phase_delta")),
                    support=_fmt(row.get("support_writeback")),
                    major=_fmt(row.get("major_gain")),
                    loss=_fmt(row.get("loss")),
                    dloss=_fmt(row.get("delta_loss")),
                )
            )

    arc_iso = payload["evidence"].get("arc_lane", {})
    lines.extend(
        [
            "",
            "## Arc Lane",
            "",
            f"- status: `{arc_iso.get('status')}`",
            f"- time steps: `{arc_iso.get('time_steps')}`",
            f"- seed plan: `{arc_iso.get('seed_plan')}`",
        ]
    )
    for track in arc_iso.get("tracks", []) or []:
        lines.extend(
            [
                "",
                f"### {track.get('track')}",
                "",
                "| variant | parent | child | phase-only | naked phase | gate | phase delta | support | law families | major gain | loss | d loss |",
                "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for row in track.get("rows", []) or []:
            lines.append(
                "| {variant} | {parent} | {child} | {phase} | {naked_phase} | {gate} | {phase_delta} | {support} | {law} | {major} | {loss} | {dloss} |".format(
                    variant=row.get("variant"),
                    parent=_fmt(row.get("parent_branch")),
                    child=_fmt(row.get("child_branch")),
                    phase=_fmt(row.get("phase_only_branch")),
                    naked_phase=_fmt(row.get("naked_phase_only_branch")),
                    gate=_fmt(row.get("gate_mass")),
                    phase_delta=_fmt(row.get("phase_delta")),
                    support=_fmt(row.get("support_writeback")),
                    law=_fmt(row.get("law_families")),
                    major=_fmt(row.get("major_gain")),
                    loss=_fmt(row.get("loss")),
                    dloss=_fmt(row.get("delta_loss")),
                )
            )

    arc_q = payload["evidence"].get("arc_q_cross", {})
    lines.extend(
        [
            "",
            "## Arc-Q Cross",
            "",
            f"- status: `{arc_q.get('status')}`",
            f"- time steps: `{arc_q.get('time_steps')}`",
            f"- seed plan: `{arc_q.get('seed_plan')}`",
        ]
    )
    for track in arc_q.get("tracks", []) or []:
        lines.extend(
            [
                "",
                f"### {track.get('track')}",
                "",
                "| variant | parent | child | phase-only | naked phase | gate | support | law families | major gain | loss | d loss |",
                "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for row in track.get("rows", []) or []:
            lines.append(
                "| {variant} | {parent} | {child} | {phase} | {naked_phase} | {gate} | {support} | {law} | {major} | {loss} | {dloss} |".format(
                    variant=row.get("variant"),
                    parent=_fmt(row.get("parent_branch")),
                    child=_fmt(row.get("child_branch")),
                    phase=_fmt(row.get("phase_only_branch")),
                    naked_phase=_fmt(row.get("naked_phase_only_branch")),
                    gate=_fmt(row.get("gate_mass")),
                    support=_fmt(row.get("support_writeback")),
                    law=_fmt(row.get("law_families")),
                    major=_fmt(row.get("major_gain")),
                    loss=_fmt(row.get("loss")),
                    dloss=_fmt(row.get("delta_loss")),
                )
            )
    arc_q_control = payload["evidence"].get("arc_q_control_claim", {})
    lines.extend(
        [
            "",
            "## Arc-Q Extended Control Claim",
            "",
            f"- status: `{arc_q_control.get('status')}`",
            f"- q variants: `{arc_q_control.get('q_variant_count')}`",
            f"- no-q-relation arc gate: `{arc_q_control.get('no_q_relation_arc_gate')}`",
            f"- Ramanujan/qtrace/phase parent spread: `{arc_q_control.get('ramanujan_qtrace_phase_parent_spread')}`",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description="Assemble claim-local RAFA evidence from tokenburst artifacts.")
    ap.add_argument("--out-dir", default=str(TOKENBURST_ROOT / "claim_evidence_2026_05_06"))
    for key, default in DEFAULT_ARTIFACTS.items():
        ap.add_argument(f"--{key.replace('_', '-')}", default=str(default))
    args = ap.parse_args()

    artifacts = {key: Path(getattr(args, key)) for key in DEFAULT_ARTIFACTS}
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = assemble(artifacts)
    json_path = out_dir / "rafa_claim_evidence_summary.json"
    md_path = out_dir / "RAFA_CLAIM_EVIDENCE_SUMMARY.md"
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    write_markdown(md_path, payload)
    print(json.dumps({"json": str(json_path), "markdown": str(md_path)}, indent=2))


if __name__ == "__main__":
    main()
