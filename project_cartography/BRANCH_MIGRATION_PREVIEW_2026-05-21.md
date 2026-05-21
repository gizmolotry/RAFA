# Branch Migration Preview - 2026-05-21

This is a non-destructive allocation of the current dirty worktree into durable branch lanes. It is a preview only: no files were moved, staged, stashed, or committed by this document.

Source branch at generation time: codex/circleworld-cleanup-cartography-2026-05-21.

## Counts By Lane

| Lane | Files |
| --- | ---: |
| circleworld-runtime-or-spike | 116 |
| circleworld-signature-spike | 3 |
| repo-migration / project-cartography | 18 |
| research-track | 113 |
| runtime-contracts | 16 |

## Recommended Order

1. Move repo-migration / project-cartography first because it documents the cleanup and has the lowest runtime risk.
2. Move runtime-contracts second because manifests/tests should validate branch health.
3. Move research-track third because reports and constitutions can be preserved independently from behavior changes.
4. Move circleworld-runtime-or-spike last, split into promoted runtime changes versus speculative spike scripts.
5. Review needs-human-review manually before any migration.

## circleworld-runtime-or-spike

- [M] lineages/04_positive_replacement/ablate_formalization.py
- [M] lineages/04_positive_replacement/circleworld.py
- [??] runtimes/circleworld_proto/analyze_arc_q_control_claim.py
- [??] runtimes/circleworld_proto/analyze_dense_signature_failure_modes.py
- [??] runtimes/circleworld_proto/analyze_internal_phase_law_family_masks.py
- [??] runtimes/circleworld_proto/analyze_internal_phase_law_support_router_oracle.py
- [??] runtimes/circleworld_proto/analyze_internal_phase_law_target_flips.py
- [??] runtimes/circleworld_proto/analyze_learned_signature_case_failures.py
- [??] runtimes/circleworld_proto/assemble_childworld_cycle_report.py
- [??] runtimes/circleworld_proto/assemble_childworld_mechanism_audit.py
- [??] runtimes/circleworld_proto/assemble_circleworld_experiment_ledger.py
- [??] runtimes/circleworld_proto/assemble_circleworld_scoreboard.py
- [??] runtimes/circleworld_proto/assemble_phase_native_audio_route_policy_comparison.py
- [??] runtimes/circleworld_proto/assemble_phase_native_audio_shared_battlefield.py
- [??] runtimes/circleworld_proto/assemble_phase_token_repair_report.py
- [??] runtimes/circleworld_proto/assemble_rafa_claim_evidence.py
- [??] runtimes/circleworld_proto/assemble_static_child_conversion_report.py
- [??] runtimes/circleworld_proto/benchmark_audio_continuation.py
- [M] runtimes/circleworld_proto/benchmark_audio_continuity.py
- [M] runtimes/circleworld_proto/benchmark_circleworld_real_anchor.py
- [??] runtimes/circleworld_proto/build_audio_predeclared_lockbox.py
- [??] runtimes/circleworld_proto/build_childworld_mechanism_dataset.py
- [??] runtimes/circleworld_proto/build_internal_phase_law_failure_atlas.py
- [??] runtimes/circleworld_proto/build_internal_phase_law_variant_spec.py
- [??] runtimes/circleworld_proto/build_law_token_library.py
- [??] runtimes/circleworld_proto/build_relational_signature_library.py
- [??] runtimes/circleworld_proto/build_shadow_branch_law_table.py
- [??] runtimes/circleworld_proto/causal_operator_selector.py
- [??] runtimes/circleworld_proto/classify_internal_phase_law_candidate_classes.py
- [??] runtimes/circleworld_proto/common_io.py
- [??] runtimes/circleworld_proto/compare_audio_continuation_methods.py
- [??] runtimes/circleworld_proto/compare_audio_lockbox_results.py
- [??] runtimes/circleworld_proto/compare_internal_phase_law_variants.py
- [??] runtimes/circleworld_proto/compare_learned_signature_scouts.py
- [??] runtimes/circleworld_proto/compare_learned_signature_split_suites.py
- [??] runtimes/circleworld_proto/compare_relational_signature_runs.py
- [??] runtimes/circleworld_proto/compare_signature_operator_tail_probes.py
- [??] runtimes/circleworld_proto/diagnose_internal_phase_law_joint_rows.py
- [M] runtimes/circleworld_proto/evaluate_circleworld.py
- [??] runtimes/circleworld_proto/evaluate_dense_signature_claim.py
- [??] runtimes/circleworld_proto/evaluate_relational_metamers.py
- [??] runtimes/circleworld_proto/evaluate_shadow_branch_law_calibration.py
- [??] runtimes/circleworld_proto/evaluate_signature_future_law_probe.py
- [??] runtimes/circleworld_proto/evaluate_signature_operator_tail_probe.py
- [??] runtimes/circleworld_proto/evaluate_signature_tail_incremental_usefulness.py
- [??] runtimes/circleworld_proto/experiment_reports.py
- [M] runtimes/circleworld_proto/export_circleworld_audio.py
- [??] runtimes/circleworld_proto/package_phase_native_audio_route_selector_profile.py
- [??] runtimes/circleworld_proto/phase_audio_metrics.py
- [??] runtimes/circleworld_proto/phase_native_audio_operators.py
- [??] runtimes/circleworld_proto/predeclare_internal_phase_law_diagnostic_class.py
- [??] runtimes/circleworld_proto/profile_registry.py
- [M] runtimes/circleworld_proto/README.md
- [??] runtimes/circleworld_proto/resonant_law_objects.py
- [??] runtimes/circleworld_proto/retrospective_childworld_eval.py
- [??] runtimes/circleworld_proto/run_audio_baseline_stratified_diagnostics.py
- [??] runtimes/circleworld_proto/run_audio_circle_delta_probe.py
- [??] runtimes/circleworld_proto/run_audio_delta_mechanism_probe.py
- [??] runtimes/circleworld_proto/run_audio_delta_objective_scout.py
- [??] runtimes/circleworld_proto/run_audio_electric_motor_holdout.py
- [??] runtimes/circleworld_proto/run_audio_mechanism_family_sensitivity.py
- [??] runtimes/circleworld_proto/run_audio_phase_influence_ablation.py
- [??] runtimes/circleworld_proto/run_audio_phase_phenotype_diagnostics.py
- [??] runtimes/circleworld_proto/run_audio_phase_seed_ablation.py
- [??] runtimes/circleworld_proto/run_audio_steady_phenotype_manifest.py
- [??] runtimes/circleworld_proto/run_branch_pressure_experiment.py
- [??] runtimes/circleworld_proto/run_branchlaw_ablation_series.py
- [??] runtimes/circleworld_proto/run_causal_operator_selector_probe.py
- [??] runtimes/circleworld_proto/run_child_basis_coverage_assay.py
- [??] runtimes/circleworld_proto/run_child_writeback_ablation.py
- [??] runtimes/circleworld_proto/run_childworld_causality_killswitch_suite.py
- [??] runtimes/circleworld_proto/run_childworld_experiment.py
- [??] runtimes/circleworld_proto/run_childworld_volume_recursion_sweep.py
- [??] runtimes/circleworld_proto/run_circleworld_operator_block.py
- [??] runtimes/circleworld_proto/run_claim_isolation_suite.py
- [??] runtimes/circleworld_proto/run_contrastive_structural_embedding_probe.py
- [??] runtimes/circleworld_proto/run_graduation_phase_native_bridge.py
- [??] runtimes/circleworld_proto/run_internal_phase_law_objective_scout.py
- [??] runtimes/circleworld_proto/run_learned_branch_law_assay.py
- [??] runtimes/circleworld_proto/run_learned_gated_manychild_runtime_sandbox.py
- [??] runtimes/circleworld_proto/run_learned_gated_multistep_operator_sandbox.py
- [??] runtimes/circleworld_proto/run_learned_gated_runtime_sandbox.py
- [??] runtimes/circleworld_proto/run_learned_gated_writeback_sandbox.py
- [??] runtimes/circleworld_proto/run_learned_signature_split_suite.py
- [??] runtimes/circleworld_proto/run_learned_structural_family_probe.py
- [??] runtimes/circleworld_proto/run_parent_ontology_bridge_scout.py
- [??] runtimes/circleworld_proto/run_phase_native_audio_reset_suite.py
- [??] runtimes/circleworld_proto/run_phase_native_audio_selected_route.py
- [??] runtimes/circleworld_proto/run_q_basis_ablation.py
- [??] runtimes/circleworld_proto/run_relational_signature_experiment.py
- [??] runtimes/circleworld_proto/run_resonant_child_retrieval_assay.py
- [??] runtimes/circleworld_proto/run_resonant_operator_causality_assay.py
- [??] runtimes/circleworld_proto/run_substrate_ablation.py
- [??] runtimes/circleworld_proto/run_token_diversity_experiment.py
- [??] runtimes/circleworld_proto/score_internal_phase_law_objective.py
- [??] runtimes/circleworld_proto/score_phase_native_audio_prefix_router_scout.py
- [??] runtimes/circleworld_proto/score_phase_native_audio_reentry_guard.py
- [??] runtimes/circleworld_proto/score_phase_native_audio_reentry_metric_audit.py
- [??] runtimes/circleworld_proto/score_phase_native_audio_reentry_oracle.py
- [??] runtimes/circleworld_proto/score_phase_native_audio_route_transfer.py
- [??] runtimes/circleworld_proto/score_phase_native_audio_target_replay_oracle.py
- [??] runtimes/circleworld_proto/seed_substrate.py
- [??] runtimes/circleworld_proto/summarize_learned_signature_frontier.py
- [??] runtimes/circleworld_proto/sweep_childsurvival_replay.py
- [??] runtimes/circleworld_proto/sweep_parentmix_replay.py
- [M] runtimes/circleworld_proto/test_nested_commitment.py
- [??] runtimes/circleworld_proto/train_childworld_mechanism_classifier.py
- [M] runtimes/circleworld_proto/train_circleworld.py
- [M] runtimes/circleworld_proto/train_circleworld_real_anchor.py
- [??] runtimes/circleworld_proto/train_learned_signature_scout.py
- [??] runtimes/circleworld_proto/train_parent_phase_projector.py
- [??] runtimes/circleworld_proto/train_phase_native_audio_family_route_policy.py
- [??] runtimes/circleworld_proto/train_phase_native_audio_objective_route_policy.py
- [??] runtimes/circleworld_proto/train_phase_native_audio_route_policy.py
- [??] runtimes/circleworld_proto/train_shadow_branch_law_from_table.py
- [??] runtimes/circleworld_proto/validate_tokenburst_tracks.py

## circleworld-signature-spike

- [??] lineages/04_positive_replacement/rafa_relational_signature.py
- [??] lineages/04_positive_replacement/rafa_relational_signature_learning.py
- [??] lineages/04_positive_replacement/semantic_projector.py

## repo-migration / project-cartography

- [??] docs/architecture/BRANCH_DELEGATION_MAP.md
- [??] docs/architecture/BRANCH_MIGRATION_PREVIEW_2026-05-21.md
- [??] docs/architecture/PROJECT_ARTIFACT_INVENTORY.md
- [??] docs/architecture/PROJECT_REDUNDANCY_HOTSPOTS.md
- [??] docs/architecture/PROJECT_RETIREMENT_CANDIDATES.md
- [??] docs/architecture/PROJECT_SCAFFOLDING_GUIDE.md
- [??] docs/architecture/PROJECT_SCRIPT_INVENTORY.json
- [??] docs/architecture/PROJECT_SCRIPT_INVENTORY.md
- [??] docs/architecture/PROJECT_USAGE_MAP.json
- [??] docs/architecture/PROJECT_USAGE_MAP.md
- [M] docs/indexes/REPO_MAP.md
- [??] docs/reports/RAFA_CODEBASE_CLEANUP_ORCHESTRATION_2026-05-20.md
- [??] docs/reports/RAFA_CODEBASE_CLEANUP_PASS_2026-05-20.md
- [??] docs/reports/RAFA_PROJECT_CARTOGRAPHY_2026-05-19.md
- [??] docs/reports/RAFA_PROJECT_USAGE_AUDIT_2026-05-19.md
- [??] project_cartography/
- [??] tools/audit_project_scripts.py
- [??] tools/audit_project_usage.py

## research-track

- [??] docs/architecture/constitutions/10_resonant_attention_memory_lane.md
- [M] docs/architecture/constitutions/README.md
- [M] docs/architecture/INTERLINEAGE_DAG.md
- [??] docs/architecture/RAFA_ATTENTION_QKV_TOKEN_CONSTITUTION.md
- [M] docs/architecture/RAFA_BRANCH_STOCKTAKE.md
- [M] docs/architecture/RAFA_COGNITIVE_STACK.md
- [M] docs/architecture/RAFA_LINEAGE_LEDGER.md
- [??] docs/reports/CIRCLEWORLD_AGREEMENT_SCOUT_V1_2026-04-24.md
- [??] docs/reports/CIRCLEWORLD_BRANCH_PRESSURE_EXPERIMENT_2026-04-21.md
- [??] docs/reports/CIRCLEWORLD_BRANCHLAW_ABLATION_SERIES_2026-04-17.md
- [??] docs/reports/CIRCLEWORLD_CAUSAL_RETENTION_CHILDIFS_2026-05-10.md
- [??] docs/reports/CIRCLEWORLD_CHILD_LOCAL_IFS_TRAINING_MOVEMENT_2026-05-10.md
- [??] docs/reports/CIRCLEWORLD_CHILDSURVIVAL_SWEEP_2026-04-24.md
- [??] docs/reports/CIRCLEWORLD_CHILDWORLD_CYCLE_REPORT_2026-05-11.md
- [??] docs/reports/CIRCLEWORLD_CHILDWORLD_EXPERIMENT_2026-04-21.md
- [??] docs/reports/CIRCLEWORLD_CHILDWORLD_MECHANISM_AUDIT_2026-05-11.md
- [??] docs/reports/CIRCLEWORLD_CHILDWORLD_MECHANISM_AUDIT_COMBINED_2026-05-11.md
- [??] docs/reports/CIRCLEWORLD_CHILDWORLD_MECHANISM_CLASSIFIER_V1_2026-05-11.md
- [??] docs/reports/CIRCLEWORLD_CHILDWORLD_MECHANISM_DATASET_V1_2026-05-11.md
- [??] docs/reports/CIRCLEWORLD_CHILDWORLD_MECHANISM_DATASET_V2_2026-05-11.md
- [??] docs/reports/CIRCLEWORLD_CHILDWORLD_STRICT_MECHANISM_AUDIT_SINGLECASE_2026-05-11.md
- [??] docs/reports/CIRCLEWORLD_COHERENCE_RETENTION_CHILDIFS_2026-05-10.md
- [??] docs/reports/CIRCLEWORLD_CONTINUATION_CAUSAL_CHILDIFS_2026-05-11.md
- [??] docs/reports/CIRCLEWORLD_CONTINUATION_CAUSAL_KILLSWITCH_CHILDIFS_2026-05-11.md
- [??] docs/reports/CIRCLEWORLD_DAG_HEALTH_2026-04-28.md
- [??] docs/reports/CIRCLEWORLD_EXPERIMENT_LEDGER_2026-05-13.jsonl
- [??] docs/reports/CIRCLEWORLD_EXPERIMENT_LEDGER_2026-05-13.md
- [??] docs/reports/CIRCLEWORLD_EXPERIMENT_LEDGER_2026-05-14.jsonl
- [??] docs/reports/CIRCLEWORLD_EXPERIMENT_LEDGER_2026-05-14.md
- [??] docs/reports/CIRCLEWORLD_FIXED_EVALUATOR_RETROSPECTIVE_2026-04-24.md
- [??] docs/reports/CIRCLEWORLD_FIXED_SUBSTRATE_GUARD_2026-05-10.md
- [??] docs/reports/CIRCLEWORLD_GEOMETRIC_MASKING_PASS_2026-04-17.md
- [??] docs/reports/CIRCLEWORLD_INTERNAL_PHASE_LAW_SCOUT_2026-05-07.md
- [??] docs/reports/CIRCLEWORLD_LEARNED_PARENT_BRIDGE_VALIDATION_2026-05-13.md
- [??] docs/reports/CIRCLEWORLD_LIFECYCLE_COHERENCE_PUSH_2026-05-10.md
- [??] docs/reports/CIRCLEWORLD_NAKED_ONTOLOGY_CHILDIFS_2026-05-10.md
- [??] docs/reports/CIRCLEWORLD_NATIVE_MULTIMODE_V1_2026-04-16.md
- [??] docs/reports/CIRCLEWORLD_NESTED_IDENTITY_CHILDIFS_2026-05-10.md
- [??] docs/reports/CIRCLEWORLD_ONTOLOGY_V8_RECALIBRATION_2026-05-04.md
- [??] docs/reports/CIRCLEWORLD_PARENT_AUTHORITY_BRIDGE_REPLAY_2026-05-13.md
- [??] docs/reports/CIRCLEWORLD_PARENT_ONTOLOGY_AUTHORITY_2026-05-13.md
- [??] docs/reports/CIRCLEWORLD_PARENT_ONTOLOGY_BRIDGE_SCOUT_2026-05-13.md
- [??] docs/reports/CIRCLEWORLD_PARENT_PHASE_CARRIER_AUTHORITY_2026-05-13.md
- [??] docs/reports/CIRCLEWORLD_PARENT_PHASE_PROJECTOR_REPLAY_2026-05-13.md
- [??] docs/reports/CIRCLEWORLD_PARENT_PHASE_PROJECTOR_VALIDATION_2026-05-13.md
- [??] docs/reports/CIRCLEWORLD_PARENT_PREWRITE_CARRIER_ISOLATION_2026-05-13.md
- [??] docs/reports/CIRCLEWORLD_PARENT_PREWRITE_POLICY_2026-05-13.md
- [??] docs/reports/CIRCLEWORLD_PARENTMIX_SWEEP_2026-04-24.md
- [??] docs/reports/CIRCLEWORLD_PHASE_TOKEN_REPAIR_2026-05-05.md
- [??] docs/reports/CIRCLEWORLD_PHASE_WRITEBACK_RECALIBRATION_2026-05-06.md
- [??] docs/reports/CIRCLEWORLD_POSTLONG_BRANCH_SIGNATURE_DIAGNOSIS_2026-05-05.md
- [??] docs/reports/CIRCLEWORLD_POSTPATCH_BRANCH_SIGNATURE_SCOUTS_2026-05-04.md
- [??] docs/reports/CIRCLEWORLD_QUALIFIED_PERSISTENCE_PUSH_2026-05-10.md
- [??] docs/reports/CIRCLEWORLD_RELATION_TOKEN_BUILD_2026-04-20.md
- [??] docs/reports/CIRCLEWORLD_RELATIONAL_SIGNATURE_SCOUTS_2026-05-04.md
- [??] docs/reports/CIRCLEWORLD_RELATIONAL_SIGNATURE_V0_2026-05-04.md
- [??] docs/reports/CIRCLEWORLD_RELATIONAL_SIGNATURE_VERIFICATION_2026-05-04.md
- [??] docs/reports/CIRCLEWORLD_RETROSPECTIVE_V8_CANDIDATES_2026-04-23.md
- [??] docs/reports/CIRCLEWORLD_RICH_PHASOR_CARRIER_REPLAY_2026-05-13.md
- [??] docs/reports/CIRCLEWORLD_SELECTION_AGREEMENT_GATE_2026-04-24.md
- [??] docs/reports/CIRCLEWORLD_SOFT_MATRYOSHKA_FULL_REPORT_2026-04-13.md
- [??] docs/reports/CIRCLEWORLD_STATIC_CHILD_MODE_REPLACE_CONVERSION_2026-05-11.md
- [??] docs/reports/CIRCLEWORLD_STATIC_CHILD_MODE_REPLACE_OVERDRIVE_2026-05-11.md
- [??] docs/reports/CIRCLEWORLD_STATIC_CHILD_RECORD_MECHANISM_2026-05-11.md
- [??] docs/reports/CIRCLEWORLD_STATIC_CHILD_RECORD_MECHANISM_AUDIT_FULLSEED_2026-05-11.md
- [??] docs/reports/CIRCLEWORLD_STATIC_CHILD_RECORD_MECHANISM_AUDIT_SINGLECASE_2026-05-11.md
- [??] docs/reports/CIRCLEWORLD_TOKEN_DIVERSITY_EXPERIMENT_2026-04-21.md
- [??] docs/reports/CIRCLEWORLD_V1_V4_TRANSFER_RETROSPECTIVE_2026-04-22.md
- [??] docs/reports/CIRCLEWORLD_V12_VERIFIED_GATE_2026-04-24.md
- [??] docs/reports/CIRCLEWORLD_V13_TARGETED_REPLAY_RECOVERY_2026-04-24.md
- [??] docs/reports/CIRCLEWORLD_V17_NESTED_GATE_2026-04-24.md
- [??] docs/reports/CIRCLEWORLD_V18_NESTED_PROBE_OBJECTIVE_2026-04-28.md
- [??] docs/reports/CIRCLEWORLD_V19_SEEDED_NESTED_SUBSTRATE_2026-04-28.md
- [??] docs/reports/CIRCLEWORLD_V20_CHILD_RECORD_PERTURB_2026-04-29.md
- [??] docs/reports/CIRCLEWORLD_V5_PARENT_PROTECTION_DELAYED_REMIX_2026-04-29.md
- [??] docs/reports/CIRCLEWORLD_V5_THRESHOLDED_REAL_BRANCH_2026-04-23.md
- [??] docs/reports/CIRCLEWORLD_V6_CHILD_LOCAL_READOUT_2026-04-29.md
- [??] docs/reports/CIRCLEWORLD_V6_NAKED_SURVIVAL_CURRICULUM_2026-04-23.md
- [??] docs/reports/CIRCLEWORLD_V7_AUDIO_RECOVERY_WITH_BRANCH_FLOOR_2026-04-23.md
- [??] docs/reports/CIRCLEWORLD_V7_CHILD_ONLY_CONTINUATION_2026-04-29.md
- [??] docs/reports/CIRCLEWORLD_WRITEBACK_V2_2026-04-22.md
- [??] docs/reports/CIRCLEWORLD_WRITEBACK_V3_2026-04-22.md
- [??] docs/reports/RAFA_ATTENTION_QKV_TOKEN_CONSTITUTION_2026-05-13.md
- [??] docs/reports/RAFA_CAUSAL_OPERATOR_ROUTING_2026-05-15.md
- [??] docs/reports/RAFA_CAUSAL_OPERATOR_ROUTING_CROSSCONFIG_BRANCHLAW_2026-05-15.md
- [??] docs/reports/RAFA_CAUSAL_OPERATOR_ROUTING_MULTIHEAD_DEPTH_2026-05-15.md
- [??] docs/reports/RAFA_CAUSAL_OPERATOR_ROUTING_SCALE_2026-05-15.md
- [??] docs/reports/RAFA_CHILD_BASIS_COVERAGE_ASSAY_2026-05-15.md
- [??] docs/reports/RAFA_CLAIM_TAXONOMY_2026-05-06.md
- [??] docs/reports/RAFA_CLAIM_TEST_MATRIX_2026-05-06.md
- [??] docs/reports/RAFA_CONTRASTIVE_SIBLING_TARGET_ALLOCATOR_2026-05-15.md
- [??] docs/reports/RAFA_DEVELOPMENTAL_AUTHORITY_ALLOCATOR_2026-05-15.md
- [??] docs/reports/RAFA_DIRECTIONAL_CHILD_COALITION_ALLOCATOR_2026-05-15.md
- [??] docs/reports/RAFA_ECOLOGY_WRITEBACK_ALLOCATOR_2026-05-15.md
- [??] docs/reports/RAFA_LEARNED_GATED_MANYCHILD_RUNTIME_SANDBOX_2026-05-15.md
- [??] docs/reports/RAFA_LEARNED_GATED_MULTISTEP_OPERATOR_SANDBOX_2026-05-15.md
- [??] docs/reports/RAFA_LEARNED_GATED_RUNTIME_SANDBOX_2026-05-15.md
- [??] docs/reports/RAFA_LEARNED_GATED_WRITEBACK_SANDBOX_2026-05-15.md
- [??] docs/reports/RAFA_LEARNED_SOFT_GATED_WRITEBACK_SANDBOX_2026-05-15.md
- [??] docs/reports/RAFA_PHASE_NATIVE_AUDIO_RESET_SUITE_2026-05-17.md
- [??] docs/reports/RAFA_RESONANT_ATTENTION_REFACTOR_2026-05-13.md
- [??] docs/reports/RAFA_RESONANT_MEMORY_CONTRASTIVE_EMBEDDING_2026-05-14.md
- [??] docs/reports/RAFA_RESONANT_MEMORY_LEARNED_STRUCTURAL_PROBE_2026-05-14.md
- [??] docs/reports/RAFA_RESONANT_MEMORY_NEXT4_2026-05-14.md
- [??] docs/reports/RAFA_RESONANT_MEMORY_STRUCTURAL_SCOPE_2026-05-14.md
- [??] docs/reports/RAFA_RESONANT_OPERATOR_LEARNED_SELECTOR_2026-05-14.md
- [??] docs/reports/RAFA_RETROSPECTIVE_AND_RECOMMENDATIONS_2026-04-13.md
- [??] docs/reports/RAFA_SEPARABLE_CLAIMS_CURRENT_2026-05-07.md
- [??] docs/reports/RAFA_SHADOW_BRANCH_LAW_CALIBRATION_2026-05-15.md
- [??] docs/reports/RAFA_SHADOW_LEARNED_BRANCH_LAW_EXPANDED_2026-05-15.md
- [??] docs/reports/RAFA_TARGET_AUTHORITY_SCOUT_2026-05-16.md
- [??] docs/reports/RAFA_TARGET_CONDITIONED_CHILD_COALITION_2026-05-15.md
- [??] docs/reports/RAFA_VARIABLE_CLAIMS_2026-05-06.md

## runtime-contracts

- [M] docs/architecture/RUNTIME_CONTRACTS.md
- [??] runtimes/circleworld_proto/audit_circleworld_dag.py
- [??] runtimes/circleworld_proto/audit_circleworld_seed_determinism.py
- [??] runtimes/circleworld_proto/audit_phase_native_audio_route_selector_contract.py
- [??] runtimes/circleworld_proto/audit_relational_signature_contract.py
- [??] runtimes/circleworld_proto/audit_seeded_child_substrate.py
- [M] runtimes/circleworld_proto/runtime_manifest.json
- [??] runtimes/circleworld_proto/test_circleworld_operator_block_contract.py
- [??] runtimes/circleworld_proto/test_internal_phase_law_reentry_contract.py
- [??] runtimes/circleworld_proto/test_learned_branch_law_child_ifs_contract.py
- [??] runtimes/circleworld_proto/test_packet_alias_contract.py
- [??] runtimes/circleworld_proto/test_phase_gauge_invariance.py
- [??] runtimes/circleworld_proto/test_relational_signature_learning_contract.py
- [??] runtimes/circleworld_proto/test_resonant_attention_contract.py
- [??] runtimes/circleworld_proto/test_semantic_projector_contract.py
- [??] runtimes/circleworld_proto/test_unit_phasor_contract.py

## Safe Mechanical Next Step

Generate lane patch bundles from this preview, inspect each bundle, then apply each bundle onto its durable target branch in a separate worktree. Do not use repeated dirty-tree branch switching as the migration mechanism.
