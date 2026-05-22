# Circleworld Experiment Ledger - 2026-05-13

- Schema: `circleworld_experiment_ledger_v1`
- Generated at: `2026-05-22T09:16:47Z`
- Report scan root: `docs\reports`
- Output scan root: `D:\RAFA\outputs\circleworld_proto`
- JSONL sidecar: `D:\RAFA_worktrees\circleworld\docs\reports\CIRCLEWORLD_EXPERIMENT_LEDGER_2026-05-13.jsonl`
- Since: `2026-05-17`
- Entries: `23` total, `3` report-derived, `20` output-only

## Scope

This ledger reconstructs recent Circleworld experiment claims from Markdown reports and summary JSON artifacts. It is an index and interpretation aid, not a new evaluator result.

## Summary Table

| date | source | claim/test | key metrics | artifacts |
| --- | --- | --- | --- | ---: |
| 2026-05-22 | report | Two non-KNN GPU-backed Circleworld attempts were tested: | `training_pair_accuracy`=0.77504; `training_top1_accuracy`=0.737179; `reentry_objective_score_mlp_v1`=[0.0637735, -0.0267602, -0.0683646]; `training_diagnostics.route_label_accuracy`=0.737179 | 7 |
| 2026-05-22 | report | Adding explicit no-future reentry/query features to the Circleworld phase-native route feature surface improves non-KNN learned/resonant... | `old_objective_score_mlp_v1`=[0.0534426, -0.0238738, -0.0706171]; `reentry_objective_score_mlp_v1`=[0.0637735, -0.0267602, -0.0683646]; `training_diagnostics.training_accuracy`=1; `training_diagnostics.route_label_accuracy`=1 | 7 |
| 2026-05-21 | output_json | Output artifact scan: phase_native_audio_objective_policy_audit_v1 | `training_diagnostics.route_label_accuracy`=0.102151 | 1 |
| 2026-05-21 | output_json | Output artifact scan: phase_native_audio_objective_policy_audit_v1 | `training_diagnostics.training_mse`=0.00457759; `training_diagnostics.route_label_accuracy`=0 | 1 |
| 2026-05-21 | output_json | Output artifact scan: phase_native_audio_objective_policy_audit_v1 | `training_diagnostics.training_accuracy`=0.994624; `training_diagnostics.route_label_accuracy`=0.994624 | 1 |
| 2026-05-21 | output_json | Output artifact scan: phase_native_audio_route_selector_contract_audit_v1 | `selected_route_audits[0].mean_mse_delta_vs_copy_last`=-0.0256499; `selected_route_audits[1].mean_mse_delta_vs_copy_last`=-0.0306256; `selected_route_audits[0].mean_corr_delta_vs_copy_last`=0.0892385; `selected_route_audits[1].mean_corr_delta_vs_copy_last`=0.0992766 | 1 |
| 2026-05-21 | output_json | Output artifact scan: phase_native_audio_route_selector_contract_audit_v1 | `selected_route_audits[0].mean_mse_delta_vs_copy_last`=-0.0256499; `selected_route_audits[1].mean_mse_delta_vs_copy_last`=-0.0306256; `selected_route_audits[0].mean_corr_delta_vs_copy_last`=0.0892385; `selected_route_audits[1].mean_corr_delta_vs_copy_last`=0.0992766 | 1 |
| 2026-05-21 | output_json | Output artifact scan: phase_native_audio_route_selector_contract_audit_v1 | `selected_route_audits[0].mean_mse_delta_vs_copy_last`=-0.0256499; `selected_route_audits[1].mean_mse_delta_vs_copy_last`=-0.0306256; `selected_route_audits[0].mean_corr_delta_vs_copy_last`=0.0892385; `selected_route_audits[1].mean_corr_delta_vs_copy_last`=0.0992766 | 1 |
| 2026-05-20 | output_json | Output artifact scan: rafa_cpu_phase_heat_worker_v1 | none reconstructed | 1 |
| 2026-05-20 | output_json | Output artifact scan: circleworld_internal_phase_law_variant_compare_v0 | `case_deltas[0].target_mse`=0.398; `case_deltas[1].target_mse`=0.384; `case_deltas[2].target_mse`=0.374; `case_deltas[3].target_mse`=0.391 | 1 |
| 2026-05-18 | output_json | Output artifact scan: phase_native_audio_route_selector_contract_audit_v1 | `selected_route_audits[0].mean_mse_delta_vs_copy_last`=-0.0256499; `selected_route_audits[1].mean_mse_delta_vs_copy_last`=-0.0306256; `selected_route_audits[0].mean_corr_delta_vs_copy_last`=0.0892385; `selected_route_audits[1].mean_corr_delta_vs_copy_last`=0.0992766 | 1 |
| 2026-05-17 | report | The rebuilt 2026-05-17 lockbox selected 62 cases from 10 predeclared groups with no unmet minimum groups: | `scorecard.task_scores[0].circleworld.mean_corr`=-0.0265849; `scorecard.task_scores[0].circleworld.mean_mae`=0.0227582; `scorecard.task_scores[0].circleworld.mean_mse`=0.00083282; `MSE`=-0.0149108 | 12 |
| 2026-05-17 | output_json | Output artifact scan: circleworld_audio_continuation_v0 | `method_summary.circleworld.mean_corr`=0.00103891; `method_summary.circleworld.mean_mae`=0.116003; `method_summary.circleworld.mean_mse`=0.0365265; `method_summary.circleworld.mean_loop_autocorr_peak`=0.117745 | 2 |
| 2026-05-17 | output_json | Output artifact scan: circleworld_audio_continuation_v0 | `method_summary.circleworld.mean_corr`=0.00430237; `method_summary.circleworld.mean_mae`=0.111791; `method_summary.circleworld.mean_mse`=0.0343516; `method_summary.circleworld.mean_loop_autocorr_peak`=0.106451 | 2 |
| 2026-05-17 | output_json | Output artifact scan: phase_native_audio_prefix_router_scout_v1 | `policies.global_route.mean_corr_delta_vs_gain0`=0.0360571; `policies.global_route.mean_mse_delta_vs_copy_last`=-0.0243259; `policies.global_route.mean_corr_delta_vs_copy_last`=0.0350646; `policies.global_route.mse_win_fraction_vs_copy_last`=0.741935 | 1 |
| 2026-05-17 | output_json | Output artifact scan: circleworld_audio_continuation_v0 | `method_summary.circleworld.mean_corr`=0.00189141; `method_summary.circleworld.mean_mae`=0.116952; `method_summary.circleworld.mean_mse`=0.0373879; `method_summary.circleworld.mean_loop_autocorr_peak`=0.109553 | 2 |
| 2026-05-17 | output_json | Output artifact scan: phase_native_audio_prefix_router_scout_v1 | `policies.global_route.mean_corr_delta_vs_gain0`=0.264452; `policies.global_route.mean_mse_delta_vs_copy_last`=-0.0737306; `policies.global_route.mean_corr_delta_vs_copy_last`=0.252792; `policies.global_route.mse_win_fraction_vs_copy_last`=0.666667 | 1 |
| 2026-05-17 | output_json | Output artifact scan: circleworld_audio_continuation_v0 | `rows[0].circleworld_meta.mean_child_writeback_mass`=0.035903; `rows[0].circleworld_meta.mean_child_parent_divergence`=0.323345; `method_summary.circleworld.mean_corr`=-9.261e-05; `method_summary.circleworld.mean_mae`=0.213522 | 2 |
| 2026-05-17 | output_json | Output artifact scan: phase_native_audio_prefix_router_scout_v1 | `policies.global_route.mean_corr_delta_vs_gain0`=0.00626739; `policies.global_route.mean_mse_delta_vs_copy_last`=-0.028956; `policies.global_route.mean_corr_delta_vs_copy_last`=0.0176702; `policies.global_route.mse_win_fraction_vs_copy_last`=0.983871 | 1 |
| 2026-05-17 | output_json | Output artifact scan: phase_native_audio_reentry_metric_audit_v1 | `method_aggregate[0].mean_corr`=-0.0116328; `method_aggregate[0].mean_mse`=0.0615568; `method_aggregate[1].mean_mse`=0.0337223; `method_aggregate[2].mean_mse`=0.0463441 | 1 |
| 2026-05-17 | output_json | Output artifact scan: circleworld_audio_continuation_v0 | `method_summary.circleworld.mean_corr`=0.00109841; `method_summary.circleworld.mean_mae`=0.110562; `method_summary.circleworld.mean_mse`=0.0336047; `method_summary.circleworld.mean_loop_autocorr_peak`=0.100567 | 2 |
| 2026-05-17 | output_json | Output artifact scan: circleworld_audio_continuation_v0 | `rows[0].circleworld_meta.mean_child_writeback_mass`=0.0173464; `rows[0].circleworld_meta.mean_child_parent_divergence`=0.156513; `method_summary.circleworld.mean_corr`=0.00524501; `method_summary.circleworld.mean_mae`=0.0204609 | 2 |
| 2026-04-28 | output_json | Output artifact scan: dag_health_audit_2026-04-28 | none reconstructed | 1 |

## Ledger Entries

### 1. 2026-05-22 - Two non-KNN GPU-backed Circleworld attempts were tested:

- Source kind: `report`
- Source path: `docs\reports\CIRCLEWORLD_PHASE_NATIVE_RANK_ROUTER_AND_GPU_CEM_2026-05-22.md`
- Claim/test: Two non-KNN GPU-backed Circleworld attempts were tested:
- Interpretation: objective_rank_mlp_v1 is a valid non-KNN learned route head. It passes the selected-route and operator-block strict replay guards, remains future-clean, and beats raw Circleworld.
- Limitations: Not explicitly reconstructed from this report.
- Next action: Review source report and linked artifacts before promotion.
- Metrics:
  - `training_pair_accuracy`: `0.77504`
  - `training_top1_accuracy`: `0.737179`
  - `reentry_objective_score_mlp_v1`: `[0.0637735, -0.0267602, -0.0683646]`
  - `training_diagnostics.route_label_accuracy`: `0.737179`
  - `training_diagnostics.training_pair_accuracy`: `0.77504`
  - `training_diagnostics.training_top1_accuracy`: `0.737179`
  - `training_route_counts.flat|||all_bins|||anti_reentry_late_decorrelator|||1`: `156`
  - `training_route_counts.flat|||all_bins|||anti_reentry_late_decorrelator|||2`: `156`
  - `training_route_counts.flat|||all_bins|||anti_reentry_late_decorrelator|||3`: `156`
  - `training_route_counts.flat|||all_bins|||anti_reentry_late_decorrelator|||4`: `156`
  - `training_route_counts.flat|||all_bins|||anti_reentry_late_decorrelator|||6`: `156`
  - `training_route_counts.flat|||all_bins|||anti_reentry_late_decorrelator|||8`: `156`
  - `training_route_counts.flat|||all_bins|||anti_reentry_late_decorrelator|||0.5`: `156`
  - `route_centroids.flat|||all_bins|||anti_reentry_late_decorrelator|||1.route.gain`: `1`
  - `route_centroids.flat|||all_bins|||anti_reentry_late_decorrelator|||2.route.gain`: `2`
  - `route_centroids.flat|||all_bins|||anti_reentry_late_decorrelator|||3.route.gain`: `3`
  - `route_centroids.flat|||all_bins|||anti_reentry_late_decorrelator|||4.route.gain`: `4`
  - `route_centroids.flat|||all_bins|||anti_reentry_late_decorrelator|||6.route.gain`: `6`
- Artifact paths:
  - `D:\RAFA\outputs\circleworld_proto\phase_native_audio_objective_route_policy_2026-05-22_objective_rank_mlp_v1_cuda_reentry_features_original_fresh_third_to_fourth\phase_native_audio_objective_route_policy.json`
  - `D:\RAFA\outputs\circleworld_proto\phase_native_audio_selected_route_2026-05-22_objective_rank_mlp_v1_cuda_reentry_features_fourth_full\phase_native_audio_selected_route.json`
  - `D:\RAFA\outputs\circleworld_proto\circleworld_operator_block_v1_2026-05-22_objective_rank_mlp_v1_cuda_reentry_features_fourth_full\circleworld_operator_block_v1.json`
  - `D:\RAFA\outputs\circleworld_proto\phase_native_audio_route_policy_comparison_2026-05-22_rank_mlp_vs_reentry_attempts_fourth\phase_native_audio_route_policy_comparison.json`
  - `D:\RAFA\outputs\circleworld_proto\real_anchor_cem_2026-05-22_gpu_rank_followup\train_summary.json`
  - `D:\RAFA\outputs\circleworld_proto\real_anchor_cem_2026-05-22_gpu_rank_followup\search_history.json`
  - `D:\RAFA\checkpoints_circleworld_proto\real_anchor_cem_2026-05-22_gpu_rank_followup\circleworld_real_anchor_config_cem_v1.json`

### 2. 2026-05-22 - Adding explicit no-future reentry/query features to the Circleworld phase-native route feature surface improves non-K...

- Source kind: `report`
- Source path: `docs\reports\CIRCLEWORLD_PHASE_NATIVE_REENTRY_QUERY_ROUTER_2026-05-22.md`
- Claim/test: Adding explicit no-future reentry/query features to the Circleworld phase-native route feature surface improves non-KNN learned/resonant route heads on the fourth lockbox.
- Interpretation: The missing reentry/query surface was a real bottleneck. With the new features, the score MLP and resonant-memory route heads both flip from strict-failing to strict-clean in the canonical circleworld_operator_block_v1 path.
- Limitations: The reentry-query features are still shallow scalar summaries, not a full; The route policy is still emitted as a case table for the lockbox run.; The fourth lockbox is one held-out battlefield; broader source families and
- Next action: Review source report and linked artifacts before promotion.
- Metrics:
  - `old_objective_score_mlp_v1`: `[0.0534426, -0.0238738, -0.0706171]`
  - `reentry_objective_score_mlp_v1`: `[0.0637735, -0.0267602, -0.0683646]`
  - `training_diagnostics.training_accuracy`: `1`
  - `training_diagnostics.route_label_accuracy`: `1`
  - `enriched_resonant_leave_one_route_label_accuracy`: `0.11828`
  - `training_route_counts.flat|||all_bins|||anti_reentry_delta_decorrelator_mix|||4`: `2`
  - `training_route_counts.flat|||all_bins|||anti_reentry_delta_decorrelator_mix|||6`: `3`
  - `training_route_counts.flat|||all_bins|||anti_reentry_staggered_decorrelator|||6`: `2`
  - `predicted_route_counts.flat|||all_bins|||anti_reentry_delta_decorrelator_mix|||4`: `1`
  - `predicted_route_counts.flat|||all_bins|||anti_reentry_staggered_decorrelator|||6`: `1`
  - `training_route_counts.prefix_hold|||all_bins|||anti_reentry_late_decorrelator|||4`: `1`
  - `training_route_counts.prefix_hold|||all_bins|||anti_reentry_late_decorrelator|||6`: `3`
  - `training_route_counts.prefix_hold|||all_bins|||anti_reentry_late_decorrelator|||8`: `4`
  - `predicted_route_counts.prefix_hold|||all_bins|||anti_reentry_late_decorrelator|||8`: `2`
  - `training_route_counts.flat|||phase_router_bins|||anti_reentry_late_decorrelator|||8`: `2`
  - `predicted_route_counts.flat|||phase_router_bins|||anti_reentry_late_decorrelator|||8`: `1`
  - `route_centroids.flat|||all_bins|||anti_reentry_delta_decorrelator_mix|||4.route.gain`: `4`
  - `route_centroids.flat|||all_bins|||anti_reentry_delta_decorrelator_mix|||6.route.gain`: `6`
- Artifact paths:
  - `D:\RAFA\outputs\circleworld_proto\phase_native_audio_objective_route_policy_2026-05-22_objective_mlp_v1_reentry_features_original_fresh_third_to_fourth\phase_native_audio_objective_route_policy.json`
  - `D:\RAFA\outputs\circleworld_proto\circleworld_operator_block_v1_2026-05-22_objective_mlp_v1_reentry_features_fourth_full\circleworld_operator_block_v1.json`
  - `D:\RAFA\outputs\circleworld_proto\phase_native_audio_objective_route_policy_2026-05-22_objective_score_mlp_v1_reentry_features_original_fresh_third_to_fourth\phase_native_audio_objective_route_policy.json`
  - `D:\RAFA\outputs\circleworld_proto\circleworld_operator_block_v1_2026-05-22_objective_score_mlp_v1_reentry_features_fourth_full\circleworld_operator_block_v1.json`
  - `D:\RAFA\outputs\circleworld_proto\phase_native_audio_objective_route_policy_2026-05-22_objective_resonant_memory_v1_reentry_features_original_fresh_third_to_fourth\phase_native_audio_objective_route_policy.json`
  - `D:\RAFA\outputs\circleworld_proto\circleworld_operator_block_v1_2026-05-22_objective_resonant_memory_v1_reentry_features_fourth_full\circleworld_operator_block_v1.json`
  - `D:\RAFA\outputs\circleworld_proto\phase_native_audio_route_policy_comparison_2026-05-22_reentry_feature_attempts_fourth\phase_native_audio_route_policy_comparison.json`

### 3. 2026-05-21 - Output artifact scan: phase_native_audio_objective_policy_audit_v1

- Source kind: `output_json`
- Source path: `D:\RAFA\outputs\circleworld_proto\phase_native_audio_objective_route_policy_2026-05-21_objective_resonant_memory_v1_original_fresh_third_to_fourth\phase_native_audio_objective_resonant_memory_v1_policy_audit.json`
- Claim/test: Output artifact scan: phase_native_audio_objective_policy_audit_v1
- Interpretation: Output JSON was found without a paired report entry in this ledger window.
- Limitations: Metrics are mechanically extracted; interpretation must be confirmed from reports or manual artifact review.
- Next action: Pair this artifact with a narrative report before using it as claim evidence.
- Metrics:
  - `training_diagnostics.route_label_accuracy`: `0.102151`
- Artifact paths:
  - `D:\RAFA\outputs\circleworld_proto\phase_native_audio_objective_route_policy_2026-05-21_objective_resonant_memory_v1_original_fresh_third_to_fourth\phase_native_audio_objective_resonant_memory_v1_policy_audit.json`

### 4. 2026-05-21 - Output artifact scan: phase_native_audio_objective_policy_audit_v1

- Source kind: `output_json`
- Source path: `D:\RAFA\outputs\circleworld_proto\phase_native_audio_objective_route_policy_2026-05-21_objective_score_mlp_v1_original_fresh_third_to_fourth\phase_native_audio_objective_score_mlp_v1_policy_audit.json`
- Claim/test: Output artifact scan: phase_native_audio_objective_policy_audit_v1
- Interpretation: Output JSON was found without a paired report entry in this ledger window.
- Limitations: Metrics are mechanically extracted; interpretation must be confirmed from reports or manual artifact review.
- Next action: Pair this artifact with a narrative report before using it as claim evidence.
- Metrics:
  - `training_diagnostics.training_mse`: `0.00457759`
  - `training_diagnostics.route_label_accuracy`: `0`
- Artifact paths:
  - `D:\RAFA\outputs\circleworld_proto\phase_native_audio_objective_route_policy_2026-05-21_objective_score_mlp_v1_original_fresh_third_to_fourth\phase_native_audio_objective_score_mlp_v1_policy_audit.json`

### 5. 2026-05-21 - Output artifact scan: phase_native_audio_objective_policy_audit_v1

- Source kind: `output_json`
- Source path: `D:\RAFA\outputs\circleworld_proto\phase_native_audio_objective_route_policy_2026-05-21_objective_mlp_v1_original_fresh_third_to_fourth\phase_native_audio_objective_mlp_v1_policy_audit.json`
- Claim/test: Output artifact scan: phase_native_audio_objective_policy_audit_v1
- Interpretation: Output JSON was found without a paired report entry in this ledger window.
- Limitations: Metrics are mechanically extracted; interpretation must be confirmed from reports or manual artifact review.
- Next action: Pair this artifact with a narrative report before using it as claim evidence.
- Metrics:
  - `training_diagnostics.training_accuracy`: `0.994624`
  - `training_diagnostics.route_label_accuracy`: `0.994624`
- Artifact paths:
  - `D:\RAFA\outputs\circleworld_proto\phase_native_audio_objective_route_policy_2026-05-21_objective_mlp_v1_original_fresh_third_to_fourth\phase_native_audio_objective_mlp_v1_policy_audit.json`

### 6. 2026-05-21 - Output artifact scan: phase_native_audio_route_selector_contract_audit_v1

- Source kind: `output_json`
- Source path: `D:\RAFA\outputs\circleworld_proto\phase_native_audio_objective_knn5_contract_audit_2026-05-21_final_check\phase_native_audio_route_selector_contract_audit.json`
- Claim/test: Output artifact scan: phase_native_audio_route_selector_contract_audit_v1
- Interpretation: Output JSON was found without a paired report entry in this ledger window.
- Limitations: Metrics are mechanically extracted; interpretation must be confirmed from reports or manual artifact review.
- Next action: Pair this artifact with a narrative report before using it as claim evidence.
- Metrics:
  - `selected_route_audits[0].mean_mse_delta_vs_copy_last`: `-0.0256499`
  - `selected_route_audits[1].mean_mse_delta_vs_copy_last`: `-0.0306256`
  - `selected_route_audits[0].mean_corr_delta_vs_copy_last`: `0.0892385`
  - `selected_route_audits[1].mean_corr_delta_vs_copy_last`: `0.0992766`
  - `raw_suite_audits[0].mean_circleworld_corr_delta_vs_copy_last`: `0.0110475`
  - `raw_suite_audits[1].mean_circleworld_corr_delta_vs_copy_last`: `0.00939052`
  - `raw_suite_audits[2].mean_circleworld_corr_delta_vs_copy_last`: `0.011266`
  - `raw_suite_audits[3].mean_circleworld_corr_delta_vs_copy_last`: `0.00940828`
- Artifact paths:
  - `D:\RAFA\outputs\circleworld_proto\phase_native_audio_objective_knn5_contract_audit_2026-05-21_final_check\phase_native_audio_route_selector_contract_audit.json`

### 7. 2026-05-21 - Output artifact scan: phase_native_audio_route_selector_contract_audit_v1

- Source kind: `output_json`
- Source path: `D:\RAFA\outputs\circleworld_proto\phase_native_audio_objective_knn5_contract_audit_2026-05-21_reverify_after_patch\phase_native_audio_route_selector_contract_audit.json`
- Claim/test: Output artifact scan: phase_native_audio_route_selector_contract_audit_v1
- Interpretation: Output JSON was found without a paired report entry in this ledger window.
- Limitations: Metrics are mechanically extracted; interpretation must be confirmed from reports or manual artifact review.
- Next action: Pair this artifact with a narrative report before using it as claim evidence.
- Metrics:
  - `selected_route_audits[0].mean_mse_delta_vs_copy_last`: `-0.0256499`
  - `selected_route_audits[1].mean_mse_delta_vs_copy_last`: `-0.0306256`
  - `selected_route_audits[0].mean_corr_delta_vs_copy_last`: `0.0892385`
  - `selected_route_audits[1].mean_corr_delta_vs_copy_last`: `0.0992766`
  - `raw_suite_audits[0].mean_circleworld_corr_delta_vs_copy_last`: `0.0110475`
  - `raw_suite_audits[1].mean_circleworld_corr_delta_vs_copy_last`: `0.00939052`
  - `raw_suite_audits[2].mean_circleworld_corr_delta_vs_copy_last`: `0.011266`
  - `raw_suite_audits[3].mean_circleworld_corr_delta_vs_copy_last`: `0.00940828`
- Artifact paths:
  - `D:\RAFA\outputs\circleworld_proto\phase_native_audio_objective_knn5_contract_audit_2026-05-21_reverify_after_patch\phase_native_audio_route_selector_contract_audit.json`

### 8. 2026-05-21 - Output artifact scan: phase_native_audio_route_selector_contract_audit_v1

- Source kind: `output_json`
- Source path: `D:\RAFA\outputs\circleworld_proto\phase_native_audio_objective_knn5_contract_audit_2026-05-21_reverify\phase_native_audio_route_selector_contract_audit.json`
- Claim/test: Output artifact scan: phase_native_audio_route_selector_contract_audit_v1
- Interpretation: Output JSON was found without a paired report entry in this ledger window.
- Limitations: Metrics are mechanically extracted; interpretation must be confirmed from reports or manual artifact review.
- Next action: Pair this artifact with a narrative report before using it as claim evidence.
- Metrics:
  - `selected_route_audits[0].mean_mse_delta_vs_copy_last`: `-0.0256499`
  - `selected_route_audits[1].mean_mse_delta_vs_copy_last`: `-0.0306256`
  - `selected_route_audits[0].mean_corr_delta_vs_copy_last`: `0.0892385`
  - `selected_route_audits[1].mean_corr_delta_vs_copy_last`: `0.0992766`
  - `raw_suite_audits[0].mean_circleworld_corr_delta_vs_copy_last`: `0.0110475`
  - `raw_suite_audits[1].mean_circleworld_corr_delta_vs_copy_last`: `0.00939052`
  - `raw_suite_audits[2].mean_circleworld_corr_delta_vs_copy_last`: `0.011266`
  - `raw_suite_audits[3].mean_circleworld_corr_delta_vs_copy_last`: `0.00940828`
- Artifact paths:
  - `D:\RAFA\outputs\circleworld_proto\phase_native_audio_objective_knn5_contract_audit_2026-05-21_reverify\phase_native_audio_route_selector_contract_audit.json`

### 9. 2026-05-20 - Output artifact scan: rafa_cpu_phase_heat_worker_v1

- Source kind: `output_json`
- Source path: `D:\RAFA\outputs\circleworld_proto\heat_training_2026-05-20\cpu_phase_heat_worker_hot\rafa_cpu_phase_heat_summary.json`
- Claim/test: Output artifact scan: rafa_cpu_phase_heat_worker_v1
- Interpretation: Output JSON was found without a paired report entry in this ledger window.
- Limitations: Metrics are mechanically extracted; interpretation must be confirmed from reports or manual artifact review.
- Next action: Pair this artifact with a narrative report before using it as claim evidence.
- Metrics:
  - none reconstructed
- Artifact paths:
  - `D:\RAFA\outputs\circleworld_proto\heat_training_2026-05-20\cpu_phase_heat_worker_hot\rafa_cpu_phase_heat_summary.json`

### 10. 2026-05-20 - Output artifact scan: circleworld_internal_phase_law_variant_compare_v0

- Source kind: `output_json`
- Source path: `D:\RAFA\outputs\circleworld_proto\codebase_cleanup_2026-05-20_common_io_smoke\variant_compare\internal_phase_law_variant_compare.json`
- Claim/test: Output artifact scan: circleworld_internal_phase_law_variant_compare_v0
- Interpretation: Output JSON was found without a paired report entry in this ledger window.
- Limitations: Metrics are mechanically extracted; interpretation must be confirmed from reports or manual artifact review.
- Next action: Pair this artifact with a narrative report before using it as claim evidence.
- Metrics:
  - `case_deltas[0].target_mse`: `0.398`
  - `case_deltas[1].target_mse`: `0.384`
  - `case_deltas[2].target_mse`: `0.374`
  - `case_deltas[3].target_mse`: `0.391`
  - `case_deltas[4].target_mse`: `0.481`
  - `case_deltas[5].target_mse`: `0.468`
  - `case_deltas[6].target_mse`: `0.462`
  - `case_deltas[7].target_mse`: `0.484`
  - `case_deltas[0].target_corr`: `0.602`
  - `case_deltas[1].target_corr`: `0.616`
  - `case_deltas[2].target_corr`: `0.626`
  - `case_deltas[3].target_corr`: `0.609`
  - `case_deltas[4].target_corr`: `0.519`
  - `case_deltas[5].target_corr`: `0.532`
  - `case_deltas[6].target_corr`: `0.538`
  - `case_deltas[7].target_corr`: `0.516`
- Artifact paths:
  - `D:\RAFA\outputs\circleworld_proto\codebase_cleanup_2026-05-20_common_io_smoke\variant_compare\internal_phase_law_variant_compare.json`

### 11. 2026-05-18 - Output artifact scan: phase_native_audio_route_selector_contract_audit_v1

- Source kind: `output_json`
- Source path: `D:\RAFA\outputs\circleworld_proto\phase_native_audio_objective_knn5_contract_audit_2026-05-18\phase_native_audio_route_selector_contract_audit.json`
- Claim/test: Output artifact scan: phase_native_audio_route_selector_contract_audit_v1
- Interpretation: Output JSON was found without a paired report entry in this ledger window.
- Limitations: Metrics are mechanically extracted; interpretation must be confirmed from reports or manual artifact review.
- Next action: Pair this artifact with a narrative report before using it as claim evidence.
- Metrics:
  - `selected_route_audits[0].mean_mse_delta_vs_copy_last`: `-0.0256499`
  - `selected_route_audits[1].mean_mse_delta_vs_copy_last`: `-0.0306256`
  - `selected_route_audits[0].mean_corr_delta_vs_copy_last`: `0.0892385`
  - `selected_route_audits[1].mean_corr_delta_vs_copy_last`: `0.0992766`
  - `raw_suite_audits[0].mean_circleworld_corr_delta_vs_copy_last`: `0.0110475`
  - `raw_suite_audits[1].mean_circleworld_corr_delta_vs_copy_last`: `0.00939052`
  - `raw_suite_audits[2].mean_circleworld_corr_delta_vs_copy_last`: `0.011266`
  - `raw_suite_audits[3].mean_circleworld_corr_delta_vs_copy_last`: `0.00940828`
- Artifact paths:
  - `D:\RAFA\outputs\circleworld_proto\phase_native_audio_objective_knn5_contract_audit_2026-05-18\phase_native_audio_route_selector_contract_audit.json`

### 12. 2026-05-17 - The rebuilt 2026-05-17 lockbox selected 62 cases from 10 predeclared groups with no unmet minimum groups:

- Source kind: `report`
- Source path: `docs\reports\RAFA_PHASE_NATIVE_AUDIO_RESET_SUITE_2026-05-17.md`
- Claim/test: The rebuilt 2026-05-17 lockbox selected 62 cases from 10 predeclared groups with no unmet minimum groups:
- Interpretation: The 2026-05-17 dry-run artifact is a registration record:
- Limitations: Not explicitly reconstructed from this report.
- Next action: The reentry guard scorer was added after the first full lockbox result. It joins delta-probe rows back to the copy-last continuation baseline and evaluates three views:
- Metrics:
  - `scorecard.task_scores[0].circleworld.mean_corr`: `-0.0265849`
  - `scorecard.task_scores[0].circleworld.mean_mae`: `0.0227582`
  - `scorecard.task_scores[0].circleworld.mean_mse`: `0.00083282`
  - `MSE`: `-0.0149108`
  - `Corr`: `0.00940828`
  - `corr`: `-0.00020972`
  - `Training MSE`: `0.00457759`
  - `corr vs gain-0`: `0.0933132`
  - `MSE vs copy-last`: `-0.0301895`
  - `corr vs copy-last`: `0.0948905`
  - `objective_score_mlp_v1`: `[0.0534426, -0.0238738, -0.0706171]`
  - `old_objective_score_mlp_v1`: `[0.0534426, -0.0238738, -0.0706171]`
  - `Route-label training accuracy`: `0.994624`
  - `Leave-one route-label accuracy`: `0.102151`
  - `reentry_objective_score_mlp_v1`: `[0.0637735, -0.0267602, -0.0683646]`
  - `scorecard.task_scores[0].case_count`: `1`
  - `scorecard.task_scores[1].case_count`: `1`
  - `scorecard.task_scores[2].case_count`: `1`
- Artifact paths:
  - `D:\RAFA\runtimes\circleworld_proto\run_phase_native_audio_reset_suite.py`
  - `D:\RAFA\docs\architecture\INTERLINEAGE_DAG.md`
  - `D:\RAFA\outputs\circleworld_proto\phase_native_audio_reset_2026-05-17_dryrun\phase_native_audio_reset_suite.json`
  - `D:\RAFA\outputs\circleworld_proto\phase_native_audio_reset_2026-05-17_dryrun\PHASE_NATIVE_AUDIO_RESET_SUITE.md`
  - `D:\RAFA\outputs\circleworld_proto\phase_native_audio_reset_2026-05-17_smoke_cpu\phase_native_audio_reset_suite.json`
  - `D:\RAFA\outputs\circleworld_proto\phase_native_audio_reset_2026-05-17_smoke_cpu\PHASE_NATIVE_AUDIO_RESET_SUITE.md`
  - `D:\RAFA\outputs\circleworld_proto\phase_native_audio_reset_lockbox_2026-05-17\audio_predeclared_lockbox_manifest.json`
  - `D:\RAFA\outputs\circleworld_proto\phase_native_audio_reset_lockbox_2026-05-17\audio_predeclared_lockbox_cases.json`
  - `D:\RAFA\outputs\circleworld_proto\phase_native_audio_reset_2026-05-17_full_lockbox\phase_native_audio_reset_suite.json`
  - `D:\RAFA\outputs\circleworld_proto\phase_native_audio_reset_2026-05-17_full_lockbox\PHASE_NATIVE_AUDIO_RESET_SUITE.md`
  - `D:\RAFA\outputs\circleworld_proto\phase_native_audio_reentry_guard_2026-05-17_full_lockbox\phase_native_audio_reentry_guard_score.json`
  - `D:\RAFA\outputs\circleworld_proto\phase_native_audio_seed_reentry_ablation_2026-05-17_full_lockbox\audio_phase_seed_ablation.json`

### 13. 2026-05-17 - Output artifact scan: circleworld_audio_continuation_v0

- Source kind: `output_json`
- Source path: `D:\RAFA\outputs\circleworld_proto\phase_native_audio_reset_2026-05-17_fourth_seed20260520_full\continuation_flat_copyphase\audio_continuation_summary.json`
- Claim/test: Output artifact scan: circleworld_audio_continuation_v0
- Interpretation: Output JSON was found without a paired report entry in this ledger window.
- Limitations: Metrics are mechanically extracted; interpretation must be confirmed from reports or manual artifact review.
- Next action: Pair this artifact with a narrative report before using it as claim evidence.
- Metrics:
  - `method_summary.circleworld.mean_corr`: `0.00103891`
  - `method_summary.circleworld.mean_mae`: `0.116003`
  - `method_summary.circleworld.mean_mse`: `0.0365265`
  - `method_summary.circleworld.mean_loop_autocorr_peak`: `0.117745`
  - `method_summary.baseline_copy_last.mean_mae`: `0.155724`
  - `method_summary.baseline_copy_last.mean_mse`: `0.0668578`
  - `method_summary.baseline_copy_last.mean_corr`: `-0.0100351`
  - `method_summary.baseline_flat_magnitude.mean_mae`: `0.116132`
  - `method_summary.baseline_flat_magnitude.mean_mse`: `0.036638`
  - `method_summary.baseline_flat_magnitude.mean_corr`: `-4.823e-05`
  - `method_summary.baseline_prefix_magnitude_hold.mean_mae`: `0.135963`
  - `method_summary.baseline_prefix_magnitude_hold.mean_mse`: `0.0502484`
  - `method_summary.baseline_prefix_magnitude_hold.mean_corr`: `-0.00835833`
  - `method_summary.baseline_copy_last.mean_loop_autocorr_peak`: `0.147968`
  - `method_summary.baseline_flat_magnitude.mean_loop_autocorr_peak`: `0.119188`
  - `method_summary.baseline_prefix_magnitude_hold.mean_loop_autocorr_peak`: `0.1732`
- Artifact paths:
  - `D:\RAFA\outputs\circleworld_proto\phase_native_audio_reset_2026-05-17_fourth_seed20260520_full\continuation_flat_copyphase\audio_continuation_summary.json`
  - `D:\RAFA\outputs\circleworld_proto\phase_native_audio_reset_2026-05-17_fourth_seed20260520_full\continuation_flat_copyphase`

### 14. 2026-05-17 - Output artifact scan: circleworld_audio_continuation_v0

- Source kind: `output_json`
- Source path: `D:\RAFA\outputs\circleworld_proto\phase_native_audio_reset_2026-05-17_third_seed20260519_full\continuation_flat_copyphase\audio_continuation_summary.json`
- Claim/test: Output artifact scan: circleworld_audio_continuation_v0
- Interpretation: Output JSON was found without a paired report entry in this ledger window.
- Limitations: Metrics are mechanically extracted; interpretation must be confirmed from reports or manual artifact review.
- Next action: Pair this artifact with a narrative report before using it as claim evidence.
- Metrics:
  - `method_summary.circleworld.mean_corr`: `0.00430237`
  - `method_summary.circleworld.mean_mae`: `0.111791`
  - `method_summary.circleworld.mean_mse`: `0.0343516`
  - `method_summary.circleworld.mean_loop_autocorr_peak`: `0.106451`
  - `method_summary.baseline_copy_last.mean_mae`: `0.150854`
  - `method_summary.baseline_copy_last.mean_mse`: `0.0631846`
  - `method_summary.baseline_copy_last.mean_corr`: `-0.00944165`
  - `method_summary.baseline_flat_magnitude.mean_mae`: `0.111947`
  - `method_summary.baseline_flat_magnitude.mean_mse`: `0.0344689`
  - `method_summary.baseline_flat_magnitude.mean_corr`: `0.00301188`
  - `method_summary.baseline_prefix_magnitude_hold.mean_mae`: `0.133467`
  - `method_summary.baseline_prefix_magnitude_hold.mean_mse`: `0.0499072`
  - `method_summary.baseline_prefix_magnitude_hold.mean_corr`: `-0.00776848`
  - `method_summary.baseline_copy_last.mean_loop_autocorr_peak`: `0.14114`
  - `method_summary.baseline_flat_magnitude.mean_loop_autocorr_peak`: `0.107865`
  - `method_summary.baseline_prefix_magnitude_hold.mean_loop_autocorr_peak`: `0.192945`
- Artifact paths:
  - `D:\RAFA\outputs\circleworld_proto\phase_native_audio_reset_2026-05-17_third_seed20260519_full\continuation_flat_copyphase\audio_continuation_summary.json`
  - `D:\RAFA\outputs\circleworld_proto\phase_native_audio_reset_2026-05-17_third_seed20260519_full\continuation_flat_copyphase`

### 15. 2026-05-17 - Output artifact scan: phase_native_audio_prefix_router_scout_v1

- Source kind: `output_json`
- Source path: `D:\RAFA\outputs\circleworld_proto\phase_native_audio_prefix_router_scout_2026-05-17_fresh_seed20260518_full\phase_native_audio_prefix_router_scout.json`
- Claim/test: Output artifact scan: phase_native_audio_prefix_router_scout_v1
- Interpretation: Output JSON was found without a paired report entry in this ledger window.
- Limitations: Metrics are mechanically extracted; interpretation must be confirmed from reports or manual artifact review.
- Next action: Pair this artifact with a narrative report before using it as claim evidence.
- Metrics:
  - `policies.global_route.mean_corr_delta_vs_gain0`: `0.0360571`
  - `policies.global_route.mean_mse_delta_vs_copy_last`: `-0.0243259`
  - `policies.global_route.mean_corr_delta_vs_copy_last`: `0.0350646`
  - `policies.global_route.mse_win_fraction_vs_copy_last`: `0.741935`
  - `policies.global_route.route_case_key_match_fraction`: `0.016129`
  - `policies.global_route.corr_win_fraction_vs_copy_last`: `0.370968`
  - `policies.global_route.loop_win_fraction_vs_copy_last`: `0.580645`
  - `policies.global_route.median_corr_delta_vs_copy_last`: `-0.00622883`
  - `policies.known_family_table.mean_corr_delta_vs_gain0`: `0.0446878`
  - `policies.global_route.route_family_key_match_fraction`: `0`
  - `policies.global_route.reentry_win_fraction_vs_copy_last`: `0.306452`
  - `policies.known_family_table.mean_mse_delta_vs_copy_last`: `-0.0317254`
  - `policies.known_family_table.mean_corr_delta_vs_copy_last`: `0.0523501`
  - `policies.known_family_table.mse_win_fraction_vs_copy_last`: `0.935484`
  - `policies.known_family_table.route_case_key_match_fraction`: `0.0483871`
  - `policies.known_family_table.corr_win_fraction_vs_copy_last`: `0.532258`
- Artifact paths:
  - `D:\RAFA\outputs\circleworld_proto\phase_native_audio_prefix_router_scout_2026-05-17_fresh_seed20260518_full\phase_native_audio_prefix_router_scout.json`

### 16. 2026-05-17 - Output artifact scan: circleworld_audio_continuation_v0

- Source kind: `output_json`
- Source path: `D:\RAFA\outputs\circleworld_proto\phase_native_audio_reset_2026-05-17_fresh_seed20260518_full\continuation_flat_copyphase\audio_continuation_summary.json`
- Claim/test: Output artifact scan: circleworld_audio_continuation_v0
- Interpretation: Output JSON was found without a paired report entry in this ledger window.
- Limitations: Metrics are mechanically extracted; interpretation must be confirmed from reports or manual artifact review.
- Next action: Pair this artifact with a narrative report before using it as claim evidence.
- Metrics:
  - `method_summary.circleworld.mean_corr`: `0.00189141`
  - `method_summary.circleworld.mean_mae`: `0.116952`
  - `method_summary.circleworld.mean_mse`: `0.0373879`
  - `method_summary.circleworld.mean_loop_autocorr_peak`: `0.109553`
  - `method_summary.baseline_copy_last.mean_mae`: `0.157392`
  - `method_summary.baseline_copy_last.mean_mse`: `0.0684706`
  - `method_summary.baseline_copy_last.mean_corr`: `-0.00969213`
  - `method_summary.baseline_flat_magnitude.mean_mae`: `0.1171`
  - `method_summary.baseline_flat_magnitude.mean_mse`: `0.0375141`
  - `method_summary.baseline_flat_magnitude.mean_corr`: `0.00026161`
  - `method_summary.baseline_prefix_magnitude_hold.mean_mae`: `0.137841`
  - `method_summary.baseline_prefix_magnitude_hold.mean_mse`: `0.052596`
  - `method_summary.baseline_prefix_magnitude_hold.mean_corr`: `-0.0106846`
  - `method_summary.baseline_copy_last.mean_loop_autocorr_peak`: `0.148145`
  - `method_summary.baseline_flat_magnitude.mean_loop_autocorr_peak`: `0.111232`
  - `method_summary.baseline_prefix_magnitude_hold.mean_loop_autocorr_peak`: `0.169321`
- Artifact paths:
  - `D:\RAFA\outputs\circleworld_proto\phase_native_audio_reset_2026-05-17_fresh_seed20260518_full\continuation_flat_copyphase\audio_continuation_summary.json`
  - `D:\RAFA\outputs\circleworld_proto\phase_native_audio_reset_2026-05-17_fresh_seed20260518_full\continuation_flat_copyphase`

### 17. 2026-05-17 - Output artifact scan: phase_native_audio_prefix_router_scout_v1

- Source kind: `output_json`
- Source path: `D:\RAFA\outputs\circleworld_proto\phase_native_audio_prefix_router_scout_2026-05-17_fresh_seed20260518\phase_native_audio_prefix_router_scout.json`
- Claim/test: Output artifact scan: phase_native_audio_prefix_router_scout_v1
- Interpretation: Output JSON was found without a paired report entry in this ledger window.
- Limitations: Metrics are mechanically extracted; interpretation must be confirmed from reports or manual artifact review.
- Next action: Pair this artifact with a narrative report before using it as claim evidence.
- Metrics:
  - `policies.global_route.mean_corr_delta_vs_gain0`: `0.264452`
  - `policies.global_route.mean_mse_delta_vs_copy_last`: `-0.0737306`
  - `policies.global_route.mean_corr_delta_vs_copy_last`: `0.252792`
  - `policies.global_route.mse_win_fraction_vs_copy_last`: `0.666667`
  - `policies.global_route.route_case_key_match_fraction`: `0.333333`
  - `policies.global_route.corr_win_fraction_vs_copy_last`: `0.666667`
  - `policies.global_route.loop_win_fraction_vs_copy_last`: `1`
  - `policies.global_route.median_corr_delta_vs_copy_last`: `0.0348501`
  - `policies.known_family_table.mean_corr_delta_vs_gain0`: `0.264452`
  - `policies.global_route.route_family_key_match_fraction`: `1`
  - `policies.global_route.reentry_win_fraction_vs_copy_last`: `0.333333`
  - `policies.known_family_table.mean_mse_delta_vs_copy_last`: `-0.0737306`
  - `policies.known_family_table.mean_corr_delta_vs_copy_last`: `0.252792`
  - `policies.known_family_table.mse_win_fraction_vs_copy_last`: `0.666667`
  - `policies.known_family_table.route_case_key_match_fraction`: `0.333333`
  - `policies.known_family_table.corr_win_fraction_vs_copy_last`: `0.666667`
- Artifact paths:
  - `D:\RAFA\outputs\circleworld_proto\phase_native_audio_prefix_router_scout_2026-05-17_fresh_seed20260518\phase_native_audio_prefix_router_scout.json`

### 18. 2026-05-17 - Output artifact scan: circleworld_audio_continuation_v0

- Source kind: `output_json`
- Source path: `D:\RAFA\outputs\circleworld_proto\phase_native_audio_reset_2026-05-17_fresh_seed20260518\continuation_flat_copyphase\audio_continuation_summary.json`
- Claim/test: Output artifact scan: circleworld_audio_continuation_v0
- Interpretation: Output JSON was found without a paired report entry in this ledger window.
- Limitations: Metrics are mechanically extracted; interpretation must be confirmed from reports or manual artifact review.
- Next action: Pair this artifact with a narrative report before using it as claim evidence.
- Metrics:
  - `rows[0].circleworld_meta.mean_child_writeback_mass`: `0.035903`
  - `rows[0].circleworld_meta.mean_child_parent_divergence`: `0.323345`
  - `method_summary.circleworld.mean_corr`: `-9.261e-05`
  - `method_summary.circleworld.mean_mae`: `0.213522`
  - `method_summary.circleworld.mean_mse`: `0.0717573`
  - `rows[0].methods.circleworld.metrics.mae`: `0.176007`
  - `rows[0].methods.circleworld.metrics.mse`: `0.0466577`
  - `rows[1].methods.circleworld.metrics.mae`: `0.205676`
  - `rows[1].methods.circleworld.metrics.mse`: `0.0691163`
  - `rows[2].methods.circleworld.metrics.mae`: `0.258884`
  - `rows[2].methods.circleworld.metrics.mse`: `0.0994979`
  - `rows[0].methods.circleworld.metrics.corr`: `-0.0345152`
  - `rows[1].methods.circleworld.metrics.corr`: `0.0144208`
  - `rows[2].methods.circleworld.metrics.corr`: `0.0198166`
  - `rows[0].circleworld_meta.mean_mode_divergence`: `0.0063583`
  - `rows[0].circleworld_meta.real_branch_fraction`: `0.466667`
- Artifact paths:
  - `D:\RAFA\outputs\circleworld_proto\phase_native_audio_reset_2026-05-17_fresh_seed20260518\continuation_flat_copyphase\audio_continuation_summary.json`
  - `D:\RAFA\outputs\circleworld_proto\phase_native_audio_reset_2026-05-17_fresh_seed20260518\continuation_flat_copyphase`

### 19. 2026-05-17 - Output artifact scan: phase_native_audio_prefix_router_scout_v1

- Source kind: `output_json`
- Source path: `D:\RAFA\outputs\circleworld_proto\phase_native_audio_prefix_router_scout_2026-05-17_full_lockbox\phase_native_audio_prefix_router_scout.json`
- Claim/test: Output artifact scan: phase_native_audio_prefix_router_scout_v1
- Interpretation: Output JSON was found without a paired report entry in this ledger window.
- Limitations: Metrics are mechanically extracted; interpretation must be confirmed from reports or manual artifact review.
- Next action: Pair this artifact with a narrative report before using it as claim evidence.
- Metrics:
  - `policies.global_route.mean_corr_delta_vs_gain0`: `0.00626739`
  - `policies.global_route.mean_mse_delta_vs_copy_last`: `-0.028956`
  - `policies.global_route.mean_corr_delta_vs_copy_last`: `0.0176702`
  - `policies.global_route.mse_win_fraction_vs_copy_last`: `0.983871`
  - `policies.global_route.route_case_key_match_fraction`: `0`
  - `policies.global_route.corr_win_fraction_vs_copy_last`: `0.467742`
  - `policies.global_route.loop_win_fraction_vs_copy_last`: `0.870968`
  - `policies.global_route.median_corr_delta_vs_copy_last`: `-0.00047826`
  - `policies.known_family_table.mean_corr_delta_vs_gain0`: `0.0463403`
  - `policies.global_route.route_family_key_match_fraction`: `0`
  - `policies.global_route.reentry_win_fraction_vs_copy_last`: `0.5`
  - `policies.known_family_table.mean_mse_delta_vs_copy_last`: `-0.0283788`
  - `policies.known_family_table.mean_corr_delta_vs_copy_last`: `0.0556299`
  - `policies.known_family_table.mse_win_fraction_vs_copy_last`: `0.935484`
  - `policies.known_family_table.route_case_key_match_fraction`: `0.0483871`
  - `policies.known_family_table.corr_win_fraction_vs_copy_last`: `0.516129`
- Artifact paths:
  - `D:\RAFA\outputs\circleworld_proto\phase_native_audio_prefix_router_scout_2026-05-17_full_lockbox\phase_native_audio_prefix_router_scout.json`

### 20. 2026-05-17 - Output artifact scan: phase_native_audio_reentry_metric_audit_v1

- Source kind: `output_json`
- Source path: `D:\RAFA\outputs\circleworld_proto\phase_native_audio_reentry_metric_audit_2026-05-17_full_lockbox\phase_native_audio_reentry_metric_audit.json`
- Claim/test: Output artifact scan: phase_native_audio_reentry_metric_audit_v1
- Interpretation: Output JSON was found without a paired report entry in this ledger window.
- Limitations: Metrics are mechanically extracted; interpretation must be confirmed from reports or manual artifact review.
- Next action: Pair this artifact with a narrative report before using it as claim evidence.
- Metrics:
  - `method_aggregate[0].mean_corr`: `-0.0116328`
  - `method_aggregate[0].mean_mse`: `0.0615568`
  - `method_aggregate[1].mean_mse`: `0.0337223`
  - `method_aggregate[2].mean_mse`: `0.0463441`
  - `method_aggregate[3].mean_mse`: `0.0395128`
  - `method_aggregate[1].mean_corr`: `-0.00022995`
  - `method_aggregate[2].mean_corr`: `-0.0094392`
  - `method_aggregate[3].mean_corr`: `-0.00058525`
  - `source_method_aggregate[0].mean_mse`: `0.0615568`
  - `source_method_aggregate[1].mean_mse`: `0.0337223`
  - `source_method_aggregate[2].mean_mse`: `0.0463441`
  - `source_method_aggregate[3].mean_mse`: `0.0336047`
  - `source_method_aggregate[4].mean_mse`: `0.0615568`
  - `source_method_aggregate[5].mean_mse`: `0.0337223`
  - `source_method_aggregate[6].mean_mse`: `0.0463441`
  - `source_method_aggregate[7].mean_mse`: `0.0454209`
- Artifact paths:
  - `D:\RAFA\outputs\circleworld_proto\phase_native_audio_reentry_metric_audit_2026-05-17_full_lockbox\phase_native_audio_reentry_metric_audit.json`

### 21. 2026-05-17 - Output artifact scan: circleworld_audio_continuation_v0

- Source kind: `output_json`
- Source path: `D:\RAFA\outputs\circleworld_proto\phase_native_audio_reset_2026-05-17_full_lockbox\continuation_flat_copyphase\audio_continuation_summary.json`
- Claim/test: Output artifact scan: circleworld_audio_continuation_v0
- Interpretation: Output JSON was found without a paired report entry in this ledger window.
- Limitations: Metrics are mechanically extracted; interpretation must be confirmed from reports or manual artifact review.
- Next action: Pair this artifact with a narrative report before using it as claim evidence.
- Metrics:
  - `method_summary.circleworld.mean_corr`: `0.00109841`
  - `method_summary.circleworld.mean_mae`: `0.110562`
  - `method_summary.circleworld.mean_mse`: `0.0336047`
  - `method_summary.circleworld.mean_loop_autocorr_peak`: `0.100567`
  - `method_summary.baseline_copy_last.mean_mae`: `0.148466`
  - `method_summary.baseline_copy_last.mean_mse`: `0.0615568`
  - `method_summary.baseline_copy_last.mean_corr`: `-0.0116328`
  - `method_summary.baseline_flat_magnitude.mean_mae`: `0.110719`
  - `method_summary.baseline_flat_magnitude.mean_mse`: `0.0337223`
  - `method_summary.baseline_flat_magnitude.mean_corr`: `-0.00022995`
  - `method_summary.baseline_prefix_magnitude_hold.mean_mae`: `0.129385`
  - `method_summary.baseline_prefix_magnitude_hold.mean_mse`: `0.0463441`
  - `method_summary.baseline_prefix_magnitude_hold.mean_corr`: `-0.0094392`
  - `method_summary.baseline_copy_last.mean_loop_autocorr_peak`: `0.149974`
  - `method_summary.baseline_flat_magnitude.mean_loop_autocorr_peak`: `0.102261`
  - `method_summary.baseline_prefix_magnitude_hold.mean_loop_autocorr_peak`: `0.172614`
- Artifact paths:
  - `D:\RAFA\outputs\circleworld_proto\phase_native_audio_reset_2026-05-17_full_lockbox\continuation_flat_copyphase\audio_continuation_summary.json`
  - `D:\RAFA\outputs\circleworld_proto\phase_native_audio_reset_2026-05-17_full_lockbox\continuation_flat_copyphase`

### 22. 2026-05-17 - Output artifact scan: circleworld_audio_continuation_v0

- Source kind: `output_json`
- Source path: `D:\RAFA\outputs\circleworld_proto\phase_native_audio_reset_2026-05-17_smoke_cpu\continuation_flat_copyphase\audio_continuation_summary.json`
- Claim/test: Output artifact scan: circleworld_audio_continuation_v0
- Interpretation: Output JSON was found without a paired report entry in this ledger window.
- Limitations: Metrics are mechanically extracted; interpretation must be confirmed from reports or manual artifact review.
- Next action: Pair this artifact with a narrative report before using it as claim evidence.
- Metrics:
  - `rows[0].circleworld_meta.mean_child_writeback_mass`: `0.0173464`
  - `rows[0].circleworld_meta.mean_child_parent_divergence`: `0.156513`
  - `method_summary.circleworld.mean_corr`: `0.00524501`
  - `method_summary.circleworld.mean_mae`: `0.0204609`
  - `method_summary.circleworld.mean_mse`: `0.00068254`
  - `rows[0].methods.circleworld.metrics.mae`: `0.0204609`
  - `rows[0].methods.circleworld.metrics.mse`: `0.00068254`
  - `rows[0].methods.circleworld.metrics.corr`: `0.00524501`
  - `rows[0].circleworld_meta.mean_mode_divergence`: `0.0001675`
  - `rows[0].circleworld_meta.real_branch_fraction`: `0`
  - `rows[0].circleworld_meta.child_writeback_count`: `1`
  - `rows[0].circleworld_meta.mean_live_child_fraction`: `0.0666667`
  - `rows[0].circleworld_meta.mean_slot2_live_fraction`: `0.103586`
  - `method_summary.circleworld.mean_loop_autocorr_peak`: `0.00286558`
  - `rows[0].circleworld_meta.child_real_branch_fraction`: `0`
  - `rows[0].circleworld_meta.silent_singlepath_fraction`: `0.896414`
- Artifact paths:
  - `D:\RAFA\outputs\circleworld_proto\phase_native_audio_reset_2026-05-17_smoke_cpu\continuation_flat_copyphase\audio_continuation_summary.json`
  - `D:\RAFA\outputs\circleworld_proto\phase_native_audio_reset_2026-05-17_smoke_cpu\continuation_flat_copyphase`

### 23. 2026-04-28 - Output artifact scan: dag_health_audit_2026-04-28

- Source kind: `output_json`
- Source path: `D:\RAFA\outputs\circleworld_proto\dag_health_audit_2026-04-28.json`
- Claim/test: Output artifact scan: dag_health_audit_2026-04-28
- Interpretation: Output JSON was found without a paired report entry in this ledger window.
- Limitations: Metrics are mechanically extracted; interpretation must be confirmed from reports or manual artifact review.
- Next action: Pair this artifact with a narrative report before using it as claim evidence.
- Metrics:
  - none reconstructed
- Artifact paths:
  - `D:\RAFA\outputs\circleworld_proto\dag_health_audit_2026-04-28.json`

## JSONL Reconstruction

Each line below is a compact JSON object with the required ledger fields.

```jsonl
{"artifact_paths": ["D:\\RAFA\\outputs\\circleworld_proto\\phase_native_audio_objective_route_policy_2026-05-22_objective_rank_mlp_v1_cuda_reentry_features_original_fresh_third_to_fourth\\phase_native_audio_objective_route_policy.json", "D:\\RAFA\\outputs\\circleworld_proto\\phase_native_audio_selected_route_2026-05-22_objective_rank_mlp_v1_cuda_reentry_features_fourth_full\\phase_native_audio_selected_route.json", "D:\\RAFA\\outputs\\circleworld_proto\\circleworld_operator_block_v1_2026-05-22_objective_rank_mlp_v1_cuda_reentry_features_fourth_full\\circleworld_operator_block_v1.json", "D:\\RAFA\\outputs\\circleworld_proto\\phase_native_audio_route_policy_comparison_2026-05-22_rank_mlp_vs_reentry_attempts_fourth\\phase_native_audio_route_policy_comparison.json", "D:\\RAFA\\outputs\\circleworld_proto\\real_anchor_cem_2026-05-22_gpu_rank_followup\\train_summary.json", "D:\\RAFA\\outputs\\circleworld_proto\\real_anchor_cem_2026-05-22_gpu_rank_followup\\search_history.json", "D:\\RAFA\\checkpoints_circleworld_proto\\real_anchor_cem_2026-05-22_gpu_rank_followup\\circleworld_real_anchor_config_cem_v1.json"], "claim_test": "Two non-KNN GPU-backed Circleworld attempts were tested:", "date": "2026-05-22", "interpretation": "objective_rank_mlp_v1 is a valid non-KNN learned route head. It passes the selected-route and operator-block strict replay guards, remains future-clean, and beats raw Circleworld.", "limitations": "Not explicitly reconstructed from this report.", "metrics": {"reentry_objective_score_mlp_v1": [0.06377346, -0.02676018, -0.06836457], "route_centroids.flat|||all_bins|||anti_reentry_late_decorrelator|||1.route.gain": 1.0, "route_centroids.flat|||all_bins|||anti_reentry_late_decorrelator|||2.route.gain": 2.0, "route_centroids.flat|||all_bins|||anti_reentry_late_decorrelator|||3.route.gain": 3.0, "route_centroids.flat|||all_bins|||anti_reentry_late_decorrelator|||4.route.gain": 4.0, "route_centroids.flat|||all_bins|||anti_reentry_late_decorrelator|||6.route.gain": 6.0, "training_diagnostics.route_label_accuracy": 0.73717949, "training_diagnostics.training_pair_accuracy": 0.77504006, "training_diagnostics.training_top1_accuracy": 0.73717949, "training_pair_accuracy": 0.77504006, "training_route_counts.flat|||all_bins|||anti_reentry_late_decorrelator|||0.5": 156.0, "training_route_counts.flat|||all_bins|||anti_reentry_late_decorrelator|||1": 156.0, "training_route_counts.flat|||all_bins|||anti_reentry_late_decorrelator|||2": 156.0, "training_route_counts.flat|||all_bins|||anti_reentry_late_decorrelator|||3": 156.0, "training_route_counts.flat|||all_bins|||anti_reentry_late_decorrelator|||4": 156.0, "training_route_counts.flat|||all_bins|||anti_reentry_late_decorrelator|||6": 156.0, "training_route_counts.flat|||all_bins|||anti_reentry_late_decorrelator|||8": 156.0, "training_top1_accuracy": 0.73717949}, "next_action": "Review source report and linked artifacts before promotion.", "source_kind": "report", "source_path": "docs\\reports\\CIRCLEWORLD_PHASE_NATIVE_RANK_ROUTER_AND_GPU_CEM_2026-05-22.md"}
{"artifact_paths": ["D:\\RAFA\\outputs\\circleworld_proto\\phase_native_audio_objective_route_policy_2026-05-22_objective_mlp_v1_reentry_features_original_fresh_third_to_fourth\\phase_native_audio_objective_route_policy.json", "D:\\RAFA\\outputs\\circleworld_proto\\circleworld_operator_block_v1_2026-05-22_objective_mlp_v1_reentry_features_fourth_full\\circleworld_operator_block_v1.json", "D:\\RAFA\\outputs\\circleworld_proto\\phase_native_audio_objective_route_policy_2026-05-22_objective_score_mlp_v1_reentry_features_original_fresh_third_to_fourth\\phase_native_audio_objective_route_policy.json", "D:\\RAFA\\outputs\\circleworld_proto\\circleworld_operator_block_v1_2026-05-22_objective_score_mlp_v1_reentry_features_fourth_full\\circleworld_operator_block_v1.json", "D:\\RAFA\\outputs\\circleworld_proto\\phase_native_audio_objective_route_policy_2026-05-22_objective_resonant_memory_v1_reentry_features_original_fresh_third_to_fourth\\phase_native_audio_objective_route_policy.json", "D:\\RAFA\\outputs\\circleworld_proto\\circleworld_operator_block_v1_2026-05-22_objective_resonant_memory_v1_reentry_features_fourth_full\\circleworld_operator_block_v1.json", "D:\\RAFA\\outputs\\circleworld_proto\\phase_native_audio_route_policy_comparison_2026-05-22_reentry_feature_attempts_fourth\\phase_native_audio_route_policy_comparison.json"], "claim_test": "Adding explicit no-future reentry/query features to the Circleworld phase-native route feature surface improves non-KNN learned/resonant route heads on the fourth lockbox.", "date": "2026-05-22", "interpretation": "The missing reentry/query surface was a real bottleneck. With the new features, the score MLP and resonant-memory route heads both flip from strict-failing to strict-clean in the canonical circleworld_operator_block_v1 path.", "limitations": "The reentry-query features are still shallow scalar summaries, not a full; The route policy is still emitted as a case table for the lockbox run.; The fourth lockbox is one held-out battlefield; broader source families and", "metrics": {"enriched_resonant_leave_one_route_label_accuracy": 0.11827957, "old_objective_score_mlp_v1": [0.0534426, -0.0238738, -0.07061707], "predicted_route_counts.flat|||all_bins|||anti_reentry_delta_decorrelator_mix|||4": 1.0, "predicted_route_counts.flat|||all_bins|||anti_reentry_staggered_decorrelator|||6": 1.0, "predicted_route_counts.flat|||phase_router_bins|||anti_reentry_late_decorrelator|||8": 1.0, "predicted_route_counts.prefix_hold|||all_bins|||anti_reentry_late_decorrelator|||8": 2.0, "reentry_objective_score_mlp_v1": [0.06377346, -0.02676018, -0.06836457], "route_centroids.flat|||all_bins|||anti_reentry_delta_decorrelator_mix|||4.route.gain": 4.0, "route_centroids.flat|||all_bins|||anti_reentry_delta_decorrelator_mix|||6.route.gain": 6.0, "training_diagnostics.route_label_accuracy": 1.0, "training_diagnostics.training_accuracy": 1.0, "training_route_counts.flat|||all_bins|||anti_reentry_delta_decorrelator_mix|||4": 2.0, "training_route_counts.flat|||all_bins|||anti_reentry_delta_decorrelator_mix|||6": 3.0, "training_route_counts.flat|||all_bins|||anti_reentry_staggered_decorrelator|||6": 2.0, "training_route_counts.flat|||phase_router_bins|||anti_reentry_late_decorrelator|||8": 2.0, "training_route_counts.prefix_hold|||all_bins|||anti_reentry_late_decorrelator|||4": 1.0, "training_route_counts.prefix_hold|||all_bins|||anti_reentry_late_decorrelator|||6": 3.0, "training_route_counts.prefix_hold|||all_bins|||anti_reentry_late_decorrelator|||8": 4.0}, "next_action": "Review source report and linked artifacts before promotion.", "source_kind": "report", "source_path": "docs\\reports\\CIRCLEWORLD_PHASE_NATIVE_REENTRY_QUERY_ROUTER_2026-05-22.md"}
{"artifact_paths": ["D:\\RAFA\\outputs\\circleworld_proto\\phase_native_audio_objective_route_policy_2026-05-21_objective_resonant_memory_v1_original_fresh_third_to_fourth\\phase_native_audio_objective_resonant_memory_v1_policy_audit.json"], "claim_test": "Output artifact scan: phase_native_audio_objective_policy_audit_v1", "date": "2026-05-21", "interpretation": "Output JSON was found without a paired report entry in this ledger window.", "limitations": "Metrics are mechanically extracted; interpretation must be confirmed from reports or manual artifact review.", "metrics": {"training_diagnostics.route_label_accuracy": 0.10215054}, "next_action": "Pair this artifact with a narrative report before using it as claim evidence.", "source_kind": "output_json", "source_path": "D:\\RAFA\\outputs\\circleworld_proto\\phase_native_audio_objective_route_policy_2026-05-21_objective_resonant_memory_v1_original_fresh_third_to_fourth\\phase_native_audio_objective_resonant_memory_v1_policy_audit.json"}
{"artifact_paths": ["D:\\RAFA\\outputs\\circleworld_proto\\phase_native_audio_objective_route_policy_2026-05-21_objective_score_mlp_v1_original_fresh_third_to_fourth\\phase_native_audio_objective_score_mlp_v1_policy_audit.json"], "claim_test": "Output artifact scan: phase_native_audio_objective_policy_audit_v1", "date": "2026-05-21", "interpretation": "Output JSON was found without a paired report entry in this ledger window.", "limitations": "Metrics are mechanically extracted; interpretation must be confirmed from reports or manual artifact review.", "metrics": {"training_diagnostics.route_label_accuracy": 0.0, "training_diagnostics.training_mse": 0.00457759}, "next_action": "Pair this artifact with a narrative report before using it as claim evidence.", "source_kind": "output_json", "source_path": "D:\\RAFA\\outputs\\circleworld_proto\\phase_native_audio_objective_route_policy_2026-05-21_objective_score_mlp_v1_original_fresh_third_to_fourth\\phase_native_audio_objective_score_mlp_v1_policy_audit.json"}
{"artifact_paths": ["D:\\RAFA\\outputs\\circleworld_proto\\phase_native_audio_objective_route_policy_2026-05-21_objective_mlp_v1_original_fresh_third_to_fourth\\phase_native_audio_objective_mlp_v1_policy_audit.json"], "claim_test": "Output artifact scan: phase_native_audio_objective_policy_audit_v1", "date": "2026-05-21", "interpretation": "Output JSON was found without a paired report entry in this ledger window.", "limitations": "Metrics are mechanically extracted; interpretation must be confirmed from reports or manual artifact review.", "metrics": {"training_diagnostics.route_label_accuracy": 0.99462366, "training_diagnostics.training_accuracy": 0.99462366}, "next_action": "Pair this artifact with a narrative report before using it as claim evidence.", "source_kind": "output_json", "source_path": "D:\\RAFA\\outputs\\circleworld_proto\\phase_native_audio_objective_route_policy_2026-05-21_objective_mlp_v1_original_fresh_third_to_fourth\\phase_native_audio_objective_mlp_v1_policy_audit.json"}
{"artifact_paths": ["D:\\RAFA\\outputs\\circleworld_proto\\phase_native_audio_objective_knn5_contract_audit_2026-05-21_final_check\\phase_native_audio_route_selector_contract_audit.json"], "claim_test": "Output artifact scan: phase_native_audio_route_selector_contract_audit_v1", "date": "2026-05-21", "interpretation": "Output JSON was found without a paired report entry in this ledger window.", "limitations": "Metrics are mechanically extracted; interpretation must be confirmed from reports or manual artifact review.", "metrics": {"raw_suite_audits[0].mean_circleworld_corr_delta_vs_copy_last": 0.01104751, "raw_suite_audits[1].mean_circleworld_corr_delta_vs_copy_last": 0.00939052, "raw_suite_audits[2].mean_circleworld_corr_delta_vs_copy_last": 0.01126603, "raw_suite_audits[3].mean_circleworld_corr_delta_vs_copy_last": 0.00940828, "selected_route_audits[0].mean_corr_delta_vs_copy_last": 0.08923855, "selected_route_audits[0].mean_mse_delta_vs_copy_last": -0.02564993, "selected_route_audits[1].mean_corr_delta_vs_copy_last": 0.09927656, "selected_route_audits[1].mean_mse_delta_vs_copy_last": -0.03062558}, "next_action": "Pair this artifact with a narrative report before using it as claim evidence.", "source_kind": "output_json", "source_path": "D:\\RAFA\\outputs\\circleworld_proto\\phase_native_audio_objective_knn5_contract_audit_2026-05-21_final_check\\phase_native_audio_route_selector_contract_audit.json"}
{"artifact_paths": ["D:\\RAFA\\outputs\\circleworld_proto\\phase_native_audio_objective_knn5_contract_audit_2026-05-21_reverify_after_patch\\phase_native_audio_route_selector_contract_audit.json"], "claim_test": "Output artifact scan: phase_native_audio_route_selector_contract_audit_v1", "date": "2026-05-21", "interpretation": "Output JSON was found without a paired report entry in this ledger window.", "limitations": "Metrics are mechanically extracted; interpretation must be confirmed from reports or manual artifact review.", "metrics": {"raw_suite_audits[0].mean_circleworld_corr_delta_vs_copy_last": 0.01104751, "raw_suite_audits[1].mean_circleworld_corr_delta_vs_copy_last": 0.00939052, "raw_suite_audits[2].mean_circleworld_corr_delta_vs_copy_last": 0.01126603, "raw_suite_audits[3].mean_circleworld_corr_delta_vs_copy_last": 0.00940828, "selected_route_audits[0].mean_corr_delta_vs_copy_last": 0.08923855, "selected_route_audits[0].mean_mse_delta_vs_copy_last": -0.02564993, "selected_route_audits[1].mean_corr_delta_vs_copy_last": 0.09927656, "selected_route_audits[1].mean_mse_delta_vs_copy_last": -0.03062558}, "next_action": "Pair this artifact with a narrative report before using it as claim evidence.", "source_kind": "output_json", "source_path": "D:\\RAFA\\outputs\\circleworld_proto\\phase_native_audio_objective_knn5_contract_audit_2026-05-21_reverify_after_patch\\phase_native_audio_route_selector_contract_audit.json"}
{"artifact_paths": ["D:\\RAFA\\outputs\\circleworld_proto\\phase_native_audio_objective_knn5_contract_audit_2026-05-21_reverify\\phase_native_audio_route_selector_contract_audit.json"], "claim_test": "Output artifact scan: phase_native_audio_route_selector_contract_audit_v1", "date": "2026-05-21", "interpretation": "Output JSON was found without a paired report entry in this ledger window.", "limitations": "Metrics are mechanically extracted; interpretation must be confirmed from reports or manual artifact review.", "metrics": {"raw_suite_audits[0].mean_circleworld_corr_delta_vs_copy_last": 0.01104751, "raw_suite_audits[1].mean_circleworld_corr_delta_vs_copy_last": 0.00939052, "raw_suite_audits[2].mean_circleworld_corr_delta_vs_copy_last": 0.01126603, "raw_suite_audits[3].mean_circleworld_corr_delta_vs_copy_last": 0.00940828, "selected_route_audits[0].mean_corr_delta_vs_copy_last": 0.08923855, "selected_route_audits[0].mean_mse_delta_vs_copy_last": -0.02564993, "selected_route_audits[1].mean_corr_delta_vs_copy_last": 0.09927656, "selected_route_audits[1].mean_mse_delta_vs_copy_last": -0.03062558}, "next_action": "Pair this artifact with a narrative report before using it as claim evidence.", "source_kind": "output_json", "source_path": "D:\\RAFA\\outputs\\circleworld_proto\\phase_native_audio_objective_knn5_contract_audit_2026-05-21_reverify\\phase_native_audio_route_selector_contract_audit.json"}
{"artifact_paths": ["D:\\RAFA\\outputs\\circleworld_proto\\heat_training_2026-05-20\\cpu_phase_heat_worker_hot\\rafa_cpu_phase_heat_summary.json"], "claim_test": "Output artifact scan: rafa_cpu_phase_heat_worker_v1", "date": "2026-05-20", "interpretation": "Output JSON was found without a paired report entry in this ledger window.", "limitations": "Metrics are mechanically extracted; interpretation must be confirmed from reports or manual artifact review.", "metrics": {}, "next_action": "Pair this artifact with a narrative report before using it as claim evidence.", "source_kind": "output_json", "source_path": "D:\\RAFA\\outputs\\circleworld_proto\\heat_training_2026-05-20\\cpu_phase_heat_worker_hot\\rafa_cpu_phase_heat_summary.json"}
{"artifact_paths": ["D:\\RAFA\\outputs\\circleworld_proto\\codebase_cleanup_2026-05-20_common_io_smoke\\variant_compare\\internal_phase_law_variant_compare.json"], "claim_test": "Output artifact scan: circleworld_internal_phase_law_variant_compare_v0", "date": "2026-05-20", "interpretation": "Output JSON was found without a paired report entry in this ledger window.", "limitations": "Metrics are mechanically extracted; interpretation must be confirmed from reports or manual artifact review.", "metrics": {"case_deltas[0].target_corr": 0.602, "case_deltas[0].target_mse": 0.398, "case_deltas[1].target_corr": 0.616, "case_deltas[1].target_mse": 0.384, "case_deltas[2].target_corr": 0.626, "case_deltas[2].target_mse": 0.374, "case_deltas[3].target_corr": 0.609, "case_deltas[3].target_mse": 0.391, "case_deltas[4].target_corr": 0.519, "case_deltas[4].target_mse": 0.481, "case_deltas[5].target_corr": 0.532, "case_deltas[5].target_mse": 0.468, "case_deltas[6].target_corr": 0.538, "case_deltas[6].target_mse": 0.462, "case_deltas[7].target_corr": 0.516, "case_deltas[7].target_mse": 0.484}, "next_action": "Pair this artifact with a narrative report before using it as claim evidence.", "source_kind": "output_json", "source_path": "D:\\RAFA\\outputs\\circleworld_proto\\codebase_cleanup_2026-05-20_common_io_smoke\\variant_compare\\internal_phase_law_variant_compare.json"}
{"artifact_paths": ["D:\\RAFA\\outputs\\circleworld_proto\\phase_native_audio_objective_knn5_contract_audit_2026-05-18\\phase_native_audio_route_selector_contract_audit.json"], "claim_test": "Output artifact scan: phase_native_audio_route_selector_contract_audit_v1", "date": "2026-05-18", "interpretation": "Output JSON was found without a paired report entry in this ledger window.", "limitations": "Metrics are mechanically extracted; interpretation must be confirmed from reports or manual artifact review.", "metrics": {"raw_suite_audits[0].mean_circleworld_corr_delta_vs_copy_last": 0.01104751, "raw_suite_audits[1].mean_circleworld_corr_delta_vs_copy_last": 0.00939052, "raw_suite_audits[2].mean_circleworld_corr_delta_vs_copy_last": 0.01126603, "raw_suite_audits[3].mean_circleworld_corr_delta_vs_copy_last": 0.00940828, "selected_route_audits[0].mean_corr_delta_vs_copy_last": 0.08923855, "selected_route_audits[0].mean_mse_delta_vs_copy_last": -0.02564993, "selected_route_audits[1].mean_corr_delta_vs_copy_last": 0.09927656, "selected_route_audits[1].mean_mse_delta_vs_copy_last": -0.03062558}, "next_action": "Pair this artifact with a narrative report before using it as claim evidence.", "source_kind": "output_json", "source_path": "D:\\RAFA\\outputs\\circleworld_proto\\phase_native_audio_objective_knn5_contract_audit_2026-05-18\\phase_native_audio_route_selector_contract_audit.json"}
{"artifact_paths": ["D:\\RAFA\\runtimes\\circleworld_proto\\run_phase_native_audio_reset_suite.py", "D:\\RAFA\\docs\\architecture\\INTERLINEAGE_DAG.md", "D:\\RAFA\\outputs\\circleworld_proto\\phase_native_audio_reset_2026-05-17_dryrun\\phase_native_audio_reset_suite.json", "D:\\RAFA\\outputs\\circleworld_proto\\phase_native_audio_reset_2026-05-17_dryrun\\PHASE_NATIVE_AUDIO_RESET_SUITE.md", "D:\\RAFA\\outputs\\circleworld_proto\\phase_native_audio_reset_2026-05-17_smoke_cpu\\phase_native_audio_reset_suite.json", "D:\\RAFA\\outputs\\circleworld_proto\\phase_native_audio_reset_2026-05-17_smoke_cpu\\PHASE_NATIVE_AUDIO_RESET_SUITE.md", "D:\\RAFA\\outputs\\circleworld_proto\\phase_native_audio_reset_lockbox_2026-05-17\\audio_predeclared_lockbox_manifest.json", "D:\\RAFA\\outputs\\circleworld_proto\\phase_native_audio_reset_lockbox_2026-05-17\\audio_predeclared_lockbox_cases.json", "D:\\RAFA\\outputs\\circleworld_proto\\phase_native_audio_reset_2026-05-17_full_lockbox\\phase_native_audio_reset_suite.json", "D:\\RAFA\\outputs\\circleworld_proto\\phase_native_audio_reset_2026-05-17_full_lockbox\\PHASE_NATIVE_AUDIO_RESET_SUITE.md", "D:\\RAFA\\outputs\\circleworld_proto\\phase_native_audio_reentry_guard_2026-05-17_full_lockbox\\phase_native_audio_reentry_guard_score.json", "D:\\RAFA\\outputs\\circleworld_proto\\phase_native_audio_seed_reentry_ablation_2026-05-17_full_lockbox\\audio_phase_seed_ablation.json"], "claim_test": "The rebuilt 2026-05-17 lockbox selected 62 cases from 10 predeclared groups with no unmet minimum groups:", "date": "2026-05-17", "interpretation": "The 2026-05-17 dry-run artifact is a registration record:", "limitations": "Not explicitly reconstructed from this report.", "metrics": {"Corr": 0.00940828, "Leave-one route-label accuracy": 0.10215054, "MSE": -0.01491078, "MSE vs copy-last": -0.03018954, "Route-label training accuracy": 0.99462366, "Training MSE": 0.00457759, "corr": -0.00020972, "corr vs copy-last": 0.09489052, "corr vs gain-0": 0.09331319, "objective_score_mlp_v1": [0.0534426, -0.0238738, -0.07061707], "old_objective_score_mlp_v1": [0.0534426, -0.0238738, -0.07061707], "reentry_objective_score_mlp_v1": [0.06377346, -0.02676018, -0.06836457], "scorecard.task_scores[0].case_count": 1.0, "scorecard.task_scores[0].circleworld.mean_corr": -0.02658485, "scorecard.task_scores[0].circleworld.mean_mae": 0.02275824, "scorecard.task_scores[0].circleworld.mean_mse": 0.00083282, "scorecard.task_scores[1].case_count": 1.0, "scorecard.task_scores[2].case_count": 1.0}, "next_action": "The reentry guard scorer was added after the first full lockbox result. It joins delta-probe rows back to the copy-last continuation baseline and evaluates three views:", "source_kind": "report", "source_path": "docs\\reports\\RAFA_PHASE_NATIVE_AUDIO_RESET_SUITE_2026-05-17.md"}
{"artifact_paths": ["D:\\RAFA\\outputs\\circleworld_proto\\phase_native_audio_reset_2026-05-17_fourth_seed20260520_full\\continuation_flat_copyphase\\audio_continuation_summary.json", "D:\\RAFA\\outputs\\circleworld_proto\\phase_native_audio_reset_2026-05-17_fourth_seed20260520_full\\continuation_flat_copyphase"], "claim_test": "Output artifact scan: circleworld_audio_continuation_v0", "date": "2026-05-17", "interpretation": "Output JSON was found without a paired report entry in this ledger window.", "limitations": "Metrics are mechanically extracted; interpretation must be confirmed from reports or manual artifact review.", "metrics": {"method_summary.baseline_copy_last.mean_corr": -0.01003514, "method_summary.baseline_copy_last.mean_loop_autocorr_peak": 0.14796829, "method_summary.baseline_copy_last.mean_mae": 0.15572425, "method_summary.baseline_copy_last.mean_mse": 0.06685779, "method_summary.baseline_flat_magnitude.mean_corr": -4.823e-05, "method_summary.baseline_flat_magnitude.mean_loop_autocorr_peak": 0.11918789, "method_summary.baseline_flat_magnitude.mean_mae": 0.11613206, "method_summary.baseline_flat_magnitude.mean_mse": 0.03663799, "method_summary.baseline_prefix_magnitude_hold.mean_corr": -0.00835833, "method_summary.baseline_prefix_magnitude_hold.mean_loop_autocorr_peak": 0.17320035, "method_summary.baseline_prefix_magnitude_hold.mean_mae": 0.13596254, "method_summary.baseline_prefix_magnitude_hold.mean_mse": 0.05024843, "method_summary.circleworld.mean_corr": 0.00103891, "method_summary.circleworld.mean_loop_autocorr_peak": 0.11774532, "method_summary.circleworld.mean_mae": 0.1160031, "method_summary.circleworld.mean_mse": 0.03652647}, "next_action": "Pair this artifact with a narrative report before using it as claim evidence.", "source_kind": "output_json", "source_path": "D:\\RAFA\\outputs\\circleworld_proto\\phase_native_audio_reset_2026-05-17_fourth_seed20260520_full\\continuation_flat_copyphase\\audio_continuation_summary.json"}
{"artifact_paths": ["D:\\RAFA\\outputs\\circleworld_proto\\phase_native_audio_reset_2026-05-17_third_seed20260519_full\\continuation_flat_copyphase\\audio_continuation_summary.json", "D:\\RAFA\\outputs\\circleworld_proto\\phase_native_audio_reset_2026-05-17_third_seed20260519_full\\continuation_flat_copyphase"], "claim_test": "Output artifact scan: circleworld_audio_continuation_v0", "date": "2026-05-17", "interpretation": "Output JSON was found without a paired report entry in this ledger window.", "limitations": "Metrics are mechanically extracted; interpretation must be confirmed from reports or manual artifact review.", "metrics": {"method_summary.baseline_copy_last.mean_corr": -0.00944165, "method_summary.baseline_copy_last.mean_loop_autocorr_peak": 0.14113976, "method_summary.baseline_copy_last.mean_mae": 0.15085417, "method_summary.baseline_copy_last.mean_mse": 0.06318461, "method_summary.baseline_flat_magnitude.mean_corr": 0.00301188, "method_summary.baseline_flat_magnitude.mean_loop_autocorr_peak": 0.10786454, "method_summary.baseline_flat_magnitude.mean_mae": 0.11194708, "method_summary.baseline_flat_magnitude.mean_mse": 0.03446885, "method_summary.baseline_prefix_magnitude_hold.mean_corr": -0.00776848, "method_summary.baseline_prefix_magnitude_hold.mean_loop_autocorr_peak": 0.19294547, "method_summary.baseline_prefix_magnitude_hold.mean_mae": 0.13346729, "method_summary.baseline_prefix_magnitude_hold.mean_mse": 0.04990719, "method_summary.circleworld.mean_corr": 0.00430237, "method_summary.circleworld.mean_loop_autocorr_peak": 0.10645101, "method_summary.circleworld.mean_mae": 0.11179138, "method_summary.circleworld.mean_mse": 0.03435159}, "next_action": "Pair this artifact with a narrative report before using it as claim evidence.", "source_kind": "output_json", "source_path": "D:\\RAFA\\outputs\\circleworld_proto\\phase_native_audio_reset_2026-05-17_third_seed20260519_full\\continuation_flat_copyphase\\audio_continuation_summary.json"}
{"artifact_paths": ["D:\\RAFA\\outputs\\circleworld_proto\\phase_native_audio_prefix_router_scout_2026-05-17_fresh_seed20260518_full\\phase_native_audio_prefix_router_scout.json"], "claim_test": "Output artifact scan: phase_native_audio_prefix_router_scout_v1", "date": "2026-05-17", "interpretation": "Output JSON was found without a paired report entry in this ledger window.", "limitations": "Metrics are mechanically extracted; interpretation must be confirmed from reports or manual artifact review.", "metrics": {"policies.global_route.corr_win_fraction_vs_copy_last": 0.37096774, "policies.global_route.loop_win_fraction_vs_copy_last": 0.58064516, "policies.global_route.mean_corr_delta_vs_copy_last": 0.03506464, "policies.global_route.mean_corr_delta_vs_gain0": 0.03605708, "policies.global_route.mean_mse_delta_vs_copy_last": -0.02432588, "policies.global_route.median_corr_delta_vs_copy_last": -0.00622883, "policies.global_route.mse_win_fraction_vs_copy_last": 0.74193548, "policies.global_route.reentry_win_fraction_vs_copy_last": 0.30645161, "policies.global_route.route_case_key_match_fraction": 0.01612903, "policies.global_route.route_family_key_match_fraction": 0.0, "policies.known_family_table.corr_win_fraction_vs_copy_last": 0.53225806, "policies.known_family_table.mean_corr_delta_vs_copy_last": 0.05235009, "policies.known_family_table.mean_corr_delta_vs_gain0": 0.04468784, "policies.known_family_table.mean_mse_delta_vs_copy_last": -0.03172538, "policies.known_family_table.mse_win_fraction_vs_copy_last": 0.93548387, "policies.known_family_table.route_case_key_match_fraction": 0.0483871}, "next_action": "Pair this artifact with a narrative report before using it as claim evidence.", "source_kind": "output_json", "source_path": "D:\\RAFA\\outputs\\circleworld_proto\\phase_native_audio_prefix_router_scout_2026-05-17_fresh_seed20260518_full\\phase_native_audio_prefix_router_scout.json"}
{"artifact_paths": ["D:\\RAFA\\outputs\\circleworld_proto\\phase_native_audio_reset_2026-05-17_fresh_seed20260518_full\\continuation_flat_copyphase\\audio_continuation_summary.json", "D:\\RAFA\\outputs\\circleworld_proto\\phase_native_audio_reset_2026-05-17_fresh_seed20260518_full\\continuation_flat_copyphase"], "claim_test": "Output artifact scan: circleworld_audio_continuation_v0", "date": "2026-05-17", "interpretation": "Output JSON was found without a paired report entry in this ledger window.", "limitations": "Metrics are mechanically extracted; interpretation must be confirmed from reports or manual artifact review.", "metrics": {"method_summary.baseline_copy_last.mean_corr": -0.00969213, "method_summary.baseline_copy_last.mean_loop_autocorr_peak": 0.14814543, "method_summary.baseline_copy_last.mean_mae": 0.15739172, "method_summary.baseline_copy_last.mean_mse": 0.06847055, "method_summary.baseline_flat_magnitude.mean_corr": 0.00026161, "method_summary.baseline_flat_magnitude.mean_loop_autocorr_peak": 0.11123242, "method_summary.baseline_flat_magnitude.mean_mae": 0.11709981, "method_summary.baseline_flat_magnitude.mean_mse": 0.03751407, "method_summary.baseline_prefix_magnitude_hold.mean_corr": -0.01068456, "method_summary.baseline_prefix_magnitude_hold.mean_loop_autocorr_peak": 0.16932057, "method_summary.baseline_prefix_magnitude_hold.mean_mae": 0.1378408, "method_summary.baseline_prefix_magnitude_hold.mean_mse": 0.05259598, "method_summary.circleworld.mean_corr": 0.00189141, "method_summary.circleworld.mean_loop_autocorr_peak": 0.1095528, "method_summary.circleworld.mean_mae": 0.11695178, "method_summary.circleworld.mean_mse": 0.03738789}, "next_action": "Pair this artifact with a narrative report before using it as claim evidence.", "source_kind": "output_json", "source_path": "D:\\RAFA\\outputs\\circleworld_proto\\phase_native_audio_reset_2026-05-17_fresh_seed20260518_full\\continuation_flat_copyphase\\audio_continuation_summary.json"}
{"artifact_paths": ["D:\\RAFA\\outputs\\circleworld_proto\\phase_native_audio_prefix_router_scout_2026-05-17_fresh_seed20260518\\phase_native_audio_prefix_router_scout.json"], "claim_test": "Output artifact scan: phase_native_audio_prefix_router_scout_v1", "date": "2026-05-17", "interpretation": "Output JSON was found without a paired report entry in this ledger window.", "limitations": "Metrics are mechanically extracted; interpretation must be confirmed from reports or manual artifact review.", "metrics": {"policies.global_route.corr_win_fraction_vs_copy_last": 0.66666667, "policies.global_route.loop_win_fraction_vs_copy_last": 1.0, "policies.global_route.mean_corr_delta_vs_copy_last": 0.25279202, "policies.global_route.mean_corr_delta_vs_gain0": 0.26445235, "policies.global_route.mean_mse_delta_vs_copy_last": -0.07373059, "policies.global_route.median_corr_delta_vs_copy_last": 0.03485006, "policies.global_route.mse_win_fraction_vs_copy_last": 0.66666667, "policies.global_route.reentry_win_fraction_vs_copy_last": 0.33333333, "policies.global_route.route_case_key_match_fraction": 0.33333333, "policies.global_route.route_family_key_match_fraction": 1.0, "policies.known_family_table.corr_win_fraction_vs_copy_last": 0.66666667, "policies.known_family_table.mean_corr_delta_vs_copy_last": 0.25279202, "policies.known_family_table.mean_corr_delta_vs_gain0": 0.26445235, "policies.known_family_table.mean_mse_delta_vs_copy_last": -0.07373059, "policies.known_family_table.mse_win_fraction_vs_copy_last": 0.66666667, "policies.known_family_table.route_case_key_match_fraction": 0.33333333}, "next_action": "Pair this artifact with a narrative report before using it as claim evidence.", "source_kind": "output_json", "source_path": "D:\\RAFA\\outputs\\circleworld_proto\\phase_native_audio_prefix_router_scout_2026-05-17_fresh_seed20260518\\phase_native_audio_prefix_router_scout.json"}
{"artifact_paths": ["D:\\RAFA\\outputs\\circleworld_proto\\phase_native_audio_reset_2026-05-17_fresh_seed20260518\\continuation_flat_copyphase\\audio_continuation_summary.json", "D:\\RAFA\\outputs\\circleworld_proto\\phase_native_audio_reset_2026-05-17_fresh_seed20260518\\continuation_flat_copyphase"], "claim_test": "Output artifact scan: circleworld_audio_continuation_v0", "date": "2026-05-17", "interpretation": "Output JSON was found without a paired report entry in this ledger window.", "limitations": "Metrics are mechanically extracted; interpretation must be confirmed from reports or manual artifact review.", "metrics": {"method_summary.circleworld.mean_corr": -9.261e-05, "method_summary.circleworld.mean_mae": 0.21352225, "method_summary.circleworld.mean_mse": 0.07175729, "rows[0].circleworld_meta.mean_child_parent_divergence": 0.32334483, "rows[0].circleworld_meta.mean_child_writeback_mass": 0.03590295, "rows[0].circleworld_meta.mean_mode_divergence": 0.0063583, "rows[0].circleworld_meta.real_branch_fraction": 0.46666667, "rows[0].methods.circleworld.metrics.corr": -0.03451523, "rows[0].methods.circleworld.metrics.mae": 0.17600687, "rows[0].methods.circleworld.metrics.mse": 0.0466577, "rows[1].methods.circleworld.metrics.corr": 0.01442083, "rows[1].methods.circleworld.metrics.mae": 0.20567583, "rows[1].methods.circleworld.metrics.mse": 0.06911627, "rows[2].methods.circleworld.metrics.corr": 0.01981657, "rows[2].methods.circleworld.metrics.mae": 0.25888404, "rows[2].methods.circleworld.metrics.mse": 0.09949791}, "next_action": "Pair this artifact with a narrative report before using it as claim evidence.", "source_kind": "output_json", "source_path": "D:\\RAFA\\outputs\\circleworld_proto\\phase_native_audio_reset_2026-05-17_fresh_seed20260518\\continuation_flat_copyphase\\audio_continuation_summary.json"}
{"artifact_paths": ["D:\\RAFA\\outputs\\circleworld_proto\\phase_native_audio_prefix_router_scout_2026-05-17_full_lockbox\\phase_native_audio_prefix_router_scout.json"], "claim_test": "Output artifact scan: phase_native_audio_prefix_router_scout_v1", "date": "2026-05-17", "interpretation": "Output JSON was found without a paired report entry in this ledger window.", "limitations": "Metrics are mechanically extracted; interpretation must be confirmed from reports or manual artifact review.", "metrics": {"policies.global_route.corr_win_fraction_vs_copy_last": 0.46774194, "policies.global_route.loop_win_fraction_vs_copy_last": 0.87096774, "policies.global_route.mean_corr_delta_vs_copy_last": 0.0176702, "policies.global_route.mean_corr_delta_vs_gain0": 0.00626739, "policies.global_route.mean_mse_delta_vs_copy_last": -0.02895604, "policies.global_route.median_corr_delta_vs_copy_last": -0.00047826, "policies.global_route.mse_win_fraction_vs_copy_last": 0.98387097, "policies.global_route.reentry_win_fraction_vs_copy_last": 0.5, "policies.global_route.route_case_key_match_fraction": 0.0, "policies.global_route.route_family_key_match_fraction": 0.0, "policies.known_family_table.corr_win_fraction_vs_copy_last": 0.51612903, "policies.known_family_table.mean_corr_delta_vs_copy_last": 0.05562995, "policies.known_family_table.mean_corr_delta_vs_gain0": 0.04634032, "policies.known_family_table.mean_mse_delta_vs_copy_last": -0.02837877, "policies.known_family_table.mse_win_fraction_vs_copy_last": 0.93548387, "policies.known_family_table.route_case_key_match_fraction": 0.0483871}, "next_action": "Pair this artifact with a narrative report before using it as claim evidence.", "source_kind": "output_json", "source_path": "D:\\RAFA\\outputs\\circleworld_proto\\phase_native_audio_prefix_router_scout_2026-05-17_full_lockbox\\phase_native_audio_prefix_router_scout.json"}
{"artifact_paths": ["D:\\RAFA\\outputs\\circleworld_proto\\phase_native_audio_reentry_metric_audit_2026-05-17_full_lockbox\\phase_native_audio_reentry_metric_audit.json"], "claim_test": "Output artifact scan: phase_native_audio_reentry_metric_audit_v1", "date": "2026-05-17", "interpretation": "Output JSON was found without a paired report entry in this ledger window.", "limitations": "Metrics are mechanically extracted; interpretation must be confirmed from reports or manual artifact review.", "metrics": {"method_aggregate[0].mean_corr": -0.01163276, "method_aggregate[0].mean_mse": 0.06155681, "method_aggregate[1].mean_corr": -0.00022995, "method_aggregate[1].mean_mse": 0.03372228, "method_aggregate[2].mean_corr": -0.0094392, "method_aggregate[2].mean_mse": 0.0463441, "method_aggregate[3].mean_corr": -0.00058525, "method_aggregate[3].mean_mse": 0.03951281, "source_method_aggregate[0].mean_mse": 0.06155681, "source_method_aggregate[1].mean_mse": 0.03372228, "source_method_aggregate[2].mean_mse": 0.0463441, "source_method_aggregate[3].mean_mse": 0.03360472, "source_method_aggregate[4].mean_mse": 0.06155681, "source_method_aggregate[5].mean_mse": 0.03372228, "source_method_aggregate[6].mean_mse": 0.0463441, "source_method_aggregate[7].mean_mse": 0.0454209}, "next_action": "Pair this artifact with a narrative report before using it as claim evidence.", "source_kind": "output_json", "source_path": "D:\\RAFA\\outputs\\circleworld_proto\\phase_native_audio_reentry_metric_audit_2026-05-17_full_lockbox\\phase_native_audio_reentry_metric_audit.json"}
{"artifact_paths": ["D:\\RAFA\\outputs\\circleworld_proto\\phase_native_audio_reset_2026-05-17_full_lockbox\\continuation_flat_copyphase\\audio_continuation_summary.json", "D:\\RAFA\\outputs\\circleworld_proto\\phase_native_audio_reset_2026-05-17_full_lockbox\\continuation_flat_copyphase"], "claim_test": "Output artifact scan: circleworld_audio_continuation_v0", "date": "2026-05-17", "interpretation": "Output JSON was found without a paired report entry in this ledger window.", "limitations": "Metrics are mechanically extracted; interpretation must be confirmed from reports or manual artifact review.", "metrics": {"method_summary.baseline_copy_last.mean_corr": -0.01163276, "method_summary.baseline_copy_last.mean_loop_autocorr_peak": 0.14997365, "method_summary.baseline_copy_last.mean_mae": 0.14846631, "method_summary.baseline_copy_last.mean_mse": 0.06155681, "method_summary.baseline_flat_magnitude.mean_corr": -0.00022995, "method_summary.baseline_flat_magnitude.mean_loop_autocorr_peak": 0.102261, "method_summary.baseline_flat_magnitude.mean_mae": 0.11071944, "method_summary.baseline_flat_magnitude.mean_mse": 0.03372228, "method_summary.baseline_prefix_magnitude_hold.mean_corr": -0.0094392, "method_summary.baseline_prefix_magnitude_hold.mean_loop_autocorr_peak": 0.17261368, "method_summary.baseline_prefix_magnitude_hold.mean_mae": 0.12938546, "method_summary.baseline_prefix_magnitude_hold.mean_mse": 0.0463441, "method_summary.circleworld.mean_corr": 0.00109841, "method_summary.circleworld.mean_loop_autocorr_peak": 0.1005667, "method_summary.circleworld.mean_mae": 0.11056223, "method_summary.circleworld.mean_mse": 0.03360472}, "next_action": "Pair this artifact with a narrative report before using it as claim evidence.", "source_kind": "output_json", "source_path": "D:\\RAFA\\outputs\\circleworld_proto\\phase_native_audio_reset_2026-05-17_full_lockbox\\continuation_flat_copyphase\\audio_continuation_summary.json"}
{"artifact_paths": ["D:\\RAFA\\outputs\\circleworld_proto\\phase_native_audio_reset_2026-05-17_smoke_cpu\\continuation_flat_copyphase\\audio_continuation_summary.json", "D:\\RAFA\\outputs\\circleworld_proto\\phase_native_audio_reset_2026-05-17_smoke_cpu\\continuation_flat_copyphase"], "claim_test": "Output artifact scan: circleworld_audio_continuation_v0", "date": "2026-05-17", "interpretation": "Output JSON was found without a paired report entry in this ledger window.", "limitations": "Metrics are mechanically extracted; interpretation must be confirmed from reports or manual artifact review.", "metrics": {"method_summary.circleworld.mean_corr": 0.00524501, "method_summary.circleworld.mean_loop_autocorr_peak": 0.00286558, "method_summary.circleworld.mean_mae": 0.02046089, "method_summary.circleworld.mean_mse": 0.00068254, "rows[0].circleworld_meta.child_real_branch_fraction": 0.0, "rows[0].circleworld_meta.child_writeback_count": 1.0, "rows[0].circleworld_meta.mean_child_parent_divergence": 0.15651312, "rows[0].circleworld_meta.mean_child_writeback_mass": 0.01734644, "rows[0].circleworld_meta.mean_live_child_fraction": 0.06666667, "rows[0].circleworld_meta.mean_mode_divergence": 0.0001675, "rows[0].circleworld_meta.mean_slot2_live_fraction": 0.10358566, "rows[0].circleworld_meta.real_branch_fraction": 0.0, "rows[0].circleworld_meta.silent_singlepath_fraction": 0.89641434, "rows[0].methods.circleworld.metrics.corr": 0.00524501, "rows[0].methods.circleworld.metrics.mae": 0.02046089, "rows[0].methods.circleworld.metrics.mse": 0.00068254}, "next_action": "Pair this artifact with a narrative report before using it as claim evidence.", "source_kind": "output_json", "source_path": "D:\\RAFA\\outputs\\circleworld_proto\\phase_native_audio_reset_2026-05-17_smoke_cpu\\continuation_flat_copyphase\\audio_continuation_summary.json"}
{"artifact_paths": ["D:\\RAFA\\outputs\\circleworld_proto\\dag_health_audit_2026-04-28.json"], "claim_test": "Output artifact scan: dag_health_audit_2026-04-28", "date": "2026-04-28", "interpretation": "Output JSON was found without a paired report entry in this ledger window.", "limitations": "Metrics are mechanically extracted; interpretation must be confirmed from reports or manual artifact review.", "metrics": {}, "next_action": "Pair this artifact with a narrative report before using it as claim evidence.", "source_kind": "output_json", "source_path": "D:\\RAFA\\outputs\\circleworld_proto\\dag_health_audit_2026-04-28.json"}
```

## Verification Notes

- The utility reads existing Markdown reports and JSON artifacts only.
- The utility writes this Markdown file and a JSONL sidecar by default.
- Metrics are best-effort extractions from report tables, inline metric lines, and linked JSON artifacts.
- Output-only entries are intentionally marked as mechanically extracted when no paired report narrative is found.

## Risks

- Heuristic parsing can miss metrics embedded in prose, unusual table shapes, or large JSON files above the configured byte limit.
- Report-derived interpretation may lag later artifacts if a report was not updated after follow-up runs.
- Output-only entries should not be treated as supported claims until reviewed against source reports and code provenance.
