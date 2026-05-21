# Project Script Inventory

Generated: `2026-05-20T17:29:07+00:00`

This is a machine-generated map of source-like scripts that are visible to git, including untracked files, while honoring `.gitignore`.

## Summary

| Metric                        | Value |
| ----------------------------- | ----- |
| Scripts                       | 239   |
| Tracked                       | 114   |
| Untracked                     | 125   |
| Total nonblank LOC            | 91671 |
| Python parse errors           | 0     |
| Scripts with CLI main guard   | 198   |
| Manifested/registered scripts | 47    |
| Documented/reported scripts   | 238   |

## By Lane

| Lane                 | Scripts |
| -------------------- | ------- |
| circleworld_proto    | 129     |
| tools                | 49      |
| research_track       | 19      |
| root_legacy          | 14      |
| core_shared          | 10      |
| positive_replacement | 5       |
| inference_package    | 4       |
| subtractive_fork     | 3       |
| parent_graduation    | 1       |
| local_ablations      | 1       |
| diffusion_parent_v3  | 1       |
| stage4_blackwell_14  | 1       |
| stage4_blackwell_16  | 1       |
| repo_tests           | 1       |

## By Role

| Role                     | Scripts |
| ------------------------ | ------- |
| script_or_module         | 62      |
| experiment_runner        | 51      |
| trainer                  | 17      |
| contract_test            | 17      |
| auditor                  | 13      |
| runtime_or_shared_module | 12      |
| exporter                 | 11      |
| report_assembler         | 10      |
| analysis                 | 9       |
| builder                  | 7       |
| comparator               | 7       |
| evaluator                | 7       |
| scorer                   | 7       |
| launcher                 | 3       |
| benchmark                | 3       |
| sweep                    | 2       |
| packager                 | 1       |

## By Lifecycle

| Lifecycle                    | Scripts |
| ---------------------------- | ------- |
| documented_or_reported       | 175     |
| canonical_or_manifested      | 47      |
| contract_or_test             | 16      |
| active_unregistered_research | 1       |

## Largest Files

| Path                                                                           | Lane                 | Role                     | Nonblank LOC | Lifecycle               |
| ------------------------------------------------------------------------------ | -------------------- | ------------------------ | ------------ | ----------------------- |
| `runtimes/circleworld_proto/assemble_rafa_claim_evidence.py`                   | circleworld_proto    | report_assembler         | 6420         | documented_or_reported  |
| `runtimes/circleworld_proto/train_circleworld_real_anchor.py`                  | circleworld_proto    | trainer                  | 3632         | canonical_or_manifested |
| `runtimes/circleworld_proto/test_nested_commitment.py`                         | circleworld_proto    | contract_test            | 3627         | canonical_or_manifested |
| `lineages/04_positive_replacement/circleworld.py`                              | positive_replacement | runtime_or_shared_module | 3140         | canonical_or_manifested |
| `runtimes/circleworld_proto/run_childworld_experiment.py`                      | circleworld_proto    | experiment_runner        | 3112         | documented_or_reported  |
| `runtimes/circleworld_proto/run_childworld_volume_recursion_sweep.py`          | circleworld_proto    | experiment_runner        | 1614         | documented_or_reported  |
| `runtimes/circleworld_proto/train_circleworld.py`                              | circleworld_proto    | trainer                  | 1535         | canonical_or_manifested |
| `runtimes/circleworld_proto/train_learned_signature_scout.py`                  | circleworld_proto    | trainer                  | 1516         | documented_or_reported  |
| `runtimes/circleworld_proto/build_internal_phase_law_variant_spec.py`          | circleworld_proto    | builder                  | 1309         | documented_or_reported  |
| `runtimes/circleworld_proto/run_learned_gated_manychild_runtime_sandbox.py`    | circleworld_proto    | experiment_runner        | 1303         | documented_or_reported  |
| `runtimes/circleworld_proto/test_unit_phasor_contract.py`                      | circleworld_proto    | contract_test            | 1031         | contract_or_test        |
| `runtimes/circleworld_proto/run_resonant_operator_causality_assay.py`          | circleworld_proto    | experiment_runner        | 1005         | documented_or_reported  |
| `tools/audit_project_scripts.py`                                               | tools                | auditor                  | 990          | documented_or_reported  |
| `runtimes/circleworld_proto/evaluate_signature_tail_incremental_usefulness.py` | circleworld_proto    | evaluator                | 907          | documented_or_reported  |
| `runtimes/circleworld_proto/run_audio_electric_motor_holdout.py`               | circleworld_proto    | experiment_runner        | 903          | documented_or_reported  |
| `runtimes/circleworld_proto/compare_audio_lockbox_results.py`                  | circleworld_proto    | comparator               | 893          | documented_or_reported  |
| `tools/audit_project_usage.py`                                                 | tools                | auditor                  | 881          | documented_or_reported  |
| `runtimes/circleworld_proto/build_childworld_mechanism_dataset.py`             | circleworld_proto    | builder                  | 852          | documented_or_reported  |
| `runtimes/circleworld_proto/benchmark_audio_continuation.py`                   | circleworld_proto    | benchmark                | 831          | documented_or_reported  |
| `runtimes/circleworld_proto/run_relational_signature_experiment.py`            | circleworld_proto    | experiment_runner        | 817          | documented_or_reported  |

## Canonical Or Manifested Scripts

| Path                                                                                | Lane                 | Role                     | Docs refs |
| ----------------------------------------------------------------------------------- | -------------------- | ------------------------ | --------- |
| `core/dataset.py`                                                                   | core_shared          | script_or_module         | 14        |
| `core/diffusion_models.py`                                                          | core_shared          | runtime_or_shared_module | 15        |
| `core/lib_blackwell.py`                                                             | core_shared          | script_or_module         | 18        |
| `core/triton_kernels.py`                                                            | core_shared          | script_or_module         | 11        |
| `diffusion_models.py`                                                               | root_legacy          | runtime_or_shared_module | 39        |
| `diffusion_utils.py`                                                                | root_legacy          | runtime_or_shared_module | 27        |
| `hf_local.py`                                                                       | root_legacy          | script_or_module         | 6         |
| `inference_package/diffusion_utils.py`                                              | inference_package    | runtime_or_shared_module | 11        |
| `inference_package/lib_blackwell.py`                                                | inference_package    | script_or_module         | 13        |
| `inference_package/triton_kernels.py`                                               | inference_package    | script_or_module         | 7         |
| `lineages/04_positive_replacement/ablate_formalization.py`                          | positive_replacement | script_or_module         | 10        |
| `lineages/04_positive_replacement/circleworld.py`                                   | positive_replacement | runtime_or_shared_module | 247       |
| `model.py`                                                                          | root_legacy          | runtime_or_shared_module | 45        |
| `phase_native_ifs.py`                                                               | root_legacy          | runtime_or_shared_module | 27        |
| `rafa_clutch_transformer_upgraded.py`                                               | root_legacy          | script_or_module         | 12        |
| `rafa_math_tools.py`                                                                | root_legacy          | runtime_or_shared_module | 21        |
| `runtimes/circleworld_proto/assemble_phase_native_audio_route_policy_comparison.py` | circleworld_proto    | report_assembler         | 6         |
| `runtimes/circleworld_proto/assemble_phase_native_audio_shared_battlefield.py`      | circleworld_proto    | report_assembler         | 6         |
| `runtimes/circleworld_proto/audit_phase_native_audio_route_selector_contract.py`    | circleworld_proto    | auditor                  | 4         |
| `runtimes/circleworld_proto/benchmark_audio_continuity.py`                          | circleworld_proto    | benchmark                | 11        |
| `runtimes/circleworld_proto/benchmark_circleworld_real_anchor.py`                   | circleworld_proto    | benchmark                | 13        |
| `runtimes/circleworld_proto/package_phase_native_audio_route_selector_profile.py`   | circleworld_proto    | packager                 | 4         |
| `runtimes/circleworld_proto/run_ablation.py`                                        | circleworld_proto    | experiment_runner        | 7         |
| `runtimes/circleworld_proto/run_circleworld_operator_block.py`                      | circleworld_proto    | experiment_runner        | 9         |
| `runtimes/circleworld_proto/run_graduation_phase_native_bridge.py`                  | circleworld_proto    | experiment_runner        | 9         |
| `runtimes/circleworld_proto/run_phase_native_audio_reset_suite.py`                  | circleworld_proto    | experiment_runner        | 13        |
| `runtimes/circleworld_proto/run_phase_native_audio_selected_route.py`               | circleworld_proto    | experiment_runner        | 9         |
| `runtimes/circleworld_proto/score_phase_native_audio_prefix_router_scout.py`        | circleworld_proto    | scorer                   | 7         |
| `runtimes/circleworld_proto/score_phase_native_audio_reentry_guard.py`              | circleworld_proto    | scorer                   | 9         |
| `runtimes/circleworld_proto/score_phase_native_audio_reentry_metric_audit.py`       | circleworld_proto    | scorer                   | 4         |
| `runtimes/circleworld_proto/score_phase_native_audio_reentry_oracle.py`             | circleworld_proto    | scorer                   | 7         |
| `runtimes/circleworld_proto/score_phase_native_audio_route_transfer.py`             | circleworld_proto    | scorer                   | 4         |
| `runtimes/circleworld_proto/score_phase_native_audio_target_replay_oracle.py`       | circleworld_proto    | scorer                   | 7         |
| `runtimes/circleworld_proto/test_nested_commitment.py`                              | circleworld_proto    | contract_test            | 100       |
| `runtimes/circleworld_proto/train_circleworld.py`                                   | circleworld_proto    | trainer                  | 76        |
| `runtimes/circleworld_proto/train_circleworld_real_anchor.py`                       | circleworld_proto    | trainer                  | 122       |
| `runtimes/circleworld_proto/train_phase_native_audio_family_route_policy.py`        | circleworld_proto    | trainer                  | 9         |
| `runtimes/circleworld_proto/train_phase_native_audio_objective_route_policy.py`     | circleworld_proto    | trainer                  | 9         |
| `runtimes/circleworld_proto/train_phase_native_audio_route_policy.py`               | circleworld_proto    | trainer                  | 9         |
| `runtimes/diffusion_parent_v3/sample_prompt.py`                                     | diffusion_parent_v3  | exporter                 | 7         |
| `runtimes/stage4_blackwell_14/export_graduation.py`                                 | stage4_blackwell_14  | exporter                 | 23        |
| `runtimes/stage4_blackwell_16/export_graduation.py`                                 | stage4_blackwell_16  | exporter                 | 20        |
| `sample_diffusion.py`                                                               | root_legacy          | exporter                 | 39        |
| `stft_utils.py`                                                                     | root_legacy          | runtime_or_shared_module | 21        |
| `tools/export_audio.py`                                                             | tools                | exporter                 | 18        |
| `tools/export_audio_compat14.py`                                                    | tools                | exporter                 | 16        |
| `tools/export_graduation_pack_restore.py`                                           | tools                | exporter                 | 11        |

## Active Unregistered Research Scripts

These are not necessarily bad. They are the scripts most likely to become confusing if they are not either documented, folded into a shared helper, or explicitly retired.

| Path                                           | Role             | Nonblank LOC | CLI? | Docs refs |
| ---------------------------------------------- | ---------------- | ------------ | ---- | --------- |
| `runtimes/circleworld_proto/seed_substrate.py` | script_or_module | 339          | no   | 0         |

## Full Script Index

| Path                                                                                | Lane                 | Role                     | Lifecycle                    | LOC  | Funcs | Classes |
| ----------------------------------------------------------------------------------- | -------------------- | ------------------------ | ---------------------------- | ---- | ----- | ------- |
| `clip_audio_loader.py`                                                              | root_legacy          | script_or_module         | documented_or_reported       | 47   | 0     | 1       |
| `core/blackwell_launch.py`                                                          | core_shared          | script_or_module         | documented_or_reported       | 66   | 1     | 0       |
| `core/blackwell_model.py`                                                           | core_shared          | script_or_module         | documented_or_reported       | 95   | 1     | 4       |
| `core/config.py`                                                                    | core_shared          | script_or_module         | documented_or_reported       | 11   | 1     | 0       |
| `core/dataset.py`                                                                   | core_shared          | script_or_module         | canonical_or_manifested      | 269  | 1     | 1       |
| `core/diffusion_models.py`                                                          | core_shared          | runtime_or_shared_module | canonical_or_manifested      | 58   | 0     | 3       |
| `core/infer.py`                                                                     | core_shared          | script_or_module         | documented_or_reported       | 162  | 5     | 0       |
| `core/lib_blackwell.py`                                                             | core_shared          | script_or_module         | canonical_or_manifested      | 124  | 2     | 2       |
| `core/triton_deq.py`                                                                | core_shared          | script_or_module         | documented_or_reported       | 164  | 5     | 2       |
| `core/triton_kernels.py`                                                            | core_shared          | script_or_module         | canonical_or_manifested      | 148  | 4     | 0       |
| `core/validate.py`                                                                  | core_shared          | script_or_module         | documented_or_reported       | 43   | 1     | 0       |
| `diffusion_models.py`                                                               | root_legacy          | runtime_or_shared_module | canonical_or_manifested      | 249  | 0     | 3       |
| `diffusion_utils.py`                                                                | root_legacy          | runtime_or_shared_module | canonical_or_manifested      | 33   | 7     | 0       |
| `hf_local.py`                                                                       | root_legacy          | script_or_module         | canonical_or_manifested      | 65   | 5     | 0       |
| `inference_package/diffusion_utils.py`                                              | inference_package    | runtime_or_shared_module | canonical_or_manifested      | 33   | 7     | 0       |
| `inference_package/infer_stage4.py`                                                 | inference_package    | exporter                 | documented_or_reported       | 113  | 2     | 0       |
| `inference_package/lib_blackwell.py`                                                | inference_package    | script_or_module         | canonical_or_manifested      | 124  | 2     | 2       |
| `inference_package/triton_kernels.py`                                               | inference_package    | script_or_module         | canonical_or_manifested      | 148  | 4     | 0       |
| `launch_lineage.ps1`                                                                | root_legacy          | launcher                 | documented_or_reported       | 38   | 0     | 0       |
| `lineages/01_parent_graduation/train.py`                                            | parent_graduation    | script_or_module         | documented_or_reported       | 303  | 8     | 0       |
| `lineages/02_local_ablations/train_diffusion.py`                                    | local_ablations      | trainer                  | documented_or_reported       | 162  | 5     | 0       |
| `lineages/03_subtractive_fork/eval_stage4.py`                                       | subtractive_fork     | script_or_module         | documented_or_reported       | 157  | 2     | 0       |
| `lineages/03_subtractive_fork/sample_stage4.py`                                     | subtractive_fork     | exporter                 | documented_or_reported       | 109  | 2     | 0       |
| `lineages/03_subtractive_fork/train_stage4_joint.py`                                | subtractive_fork     | trainer                  | documented_or_reported       | 110  | 3     | 0       |
| `lineages/04_positive_replacement/ablate_formalization.py`                          | positive_replacement | script_or_module         | canonical_or_manifested      | 272  | 9     | 0       |
| `lineages/04_positive_replacement/circleworld.py`                                   | positive_replacement | runtime_or_shared_module | canonical_or_manifested      | 3140 | 73    | 2       |
| `lineages/04_positive_replacement/rafa_relational_signature.py`                     | positive_replacement | script_or_module         | documented_or_reported       | 814  | 23    | 3       |
| `lineages/04_positive_replacement/rafa_relational_signature_learning.py`            | positive_replacement | script_or_module         | documented_or_reported       | 291  | 6     | 5       |
| `lineages/04_positive_replacement/semantic_projector.py`                            | positive_replacement | script_or_module         | documented_or_reported       | 278  | 13    | 3       |
| `model.py`                                                                          | root_legacy          | runtime_or_shared_module | canonical_or_manifested      | 568  | 5     | 5       |
| `phase_native_ifs.py`                                                               | root_legacy          | runtime_or_shared_module | canonical_or_manifested      | 248  | 6     | 5       |
| `rafa_clutch_transformer_upgraded.py`                                               | root_legacy          | script_or_module         | canonical_or_manifested      | 99   | 0     | 3       |
| `rafa_math_tools.py`                                                                | root_legacy          | runtime_or_shared_module | canonical_or_manifested      | 350  | 28    | 0       |
| `research_track/airflow/dags/rafa_learning_curve_resume.py`                         | research_track       | script_or_module         | documented_or_reported       | 65   | 1     | 0       |
| `research_track/airflow/dags/rafa_pathb_hypercube.py`                               | research_track       | script_or_module         | documented_or_reported       | 84   | 0     | 0       |
| `research_track/infra/aggregate_hypercube_results.py`                               | research_track       | script_or_module         | documented_or_reported       | 111  | 4     | 0       |
| `research_track/infra/aggregate_logic_diag_results.py`                              | research_track       | script_or_module         | documented_or_reported       | 93   | 4     | 0       |
| `research_track/infra/generate_conditioned_depth_jobs.py`                           | research_track       | script_or_module         | documented_or_reported       | 87   | 3     | 0       |
| `research_track/infra/generate_hypercube_jobs.py`                                   | research_track       | script_or_module         | documented_or_reported       | 187  | 4     | 0       |
| `research_track/infra/generate_logic_diag_jobs.py`                                  | research_track       | script_or_module         | documented_or_reported       | 120  | 4     | 0       |
| `research_track/infra/launch_conditioned_depth_job.ps1`                             | research_track       | launcher                 | documented_or_reported       | 61   | 0     | 0       |
| `research_track/infra/launch_conditioned_depth_job.py`                              | research_track       | script_or_module         | documented_or_reported       | 72   | 3     | 0       |
| `research_track/infra/run_conditioned_depth_batch.py`                               | research_track       | experiment_runner        | documented_or_reported       | 65   | 3     | 0       |
| `research_track/infra/run_conditioned_depth_job.py`                                 | research_track       | experiment_runner        | documented_or_reported       | 249  | 7     | 0       |
| `research_track/infra/run_hypercube_batch.py`                                       | research_track       | experiment_runner        | documented_or_reported       | 55   | 1     | 0       |
| `research_track/infra/run_hypercube_job.py`                                         | research_track       | experiment_runner        | documented_or_reported       | 337  | 9     | 0       |
| `research_track/infra/run_logic_diag_batch.py`                                      | research_track       | experiment_runner        | documented_or_reported       | 55   | 1     | 0       |
| `research_track/infra/run_logic_diag_job.py`                                        | research_track       | experiment_runner        | documented_or_reported       | 154  | 6     | 0       |
| `research_track/infra/schedule_conditioned_depth_job.ps1`                           | research_track       | launcher                 | documented_or_reported       | 62   | 0     | 0       |
| `research_track/infra/tests/test_generate_conditioned_depth_jobs.py`                | research_track       | contract_test            | contract_or_test             | 34   | 0     | 1       |
| `research_track/infra/tests/test_generate_hypercube_jobs.py`                        | research_track       | contract_test            | contract_or_test             | 49   | 0     | 1       |
| `research_track/infra/tests/test_generate_logic_diag_jobs.py`                       | research_track       | contract_test            | contract_or_test             | 29   | 0     | 1       |
| `runtimes/circleworld_proto/analyze_arc_q_control_claim.py`                         | circleworld_proto    | analysis                 | documented_or_reported       | 157  | 9     | 0       |
| `runtimes/circleworld_proto/analyze_dense_signature_failure_modes.py`               | circleworld_proto    | analysis                 | documented_or_reported       | 415  | 15    | 0       |
| `runtimes/circleworld_proto/analyze_internal_phase_law_family_masks.py`             | circleworld_proto    | analysis                 | documented_or_reported       | 435  | 21    | 0       |
| `runtimes/circleworld_proto/analyze_internal_phase_law_support_router_oracle.py`    | circleworld_proto    | analysis                 | documented_or_reported       | 320  | 15    | 0       |
| `runtimes/circleworld_proto/analyze_internal_phase_law_target_flips.py`             | circleworld_proto    | analysis                 | documented_or_reported       | 353  | 9     | 0       |
| `runtimes/circleworld_proto/analyze_learned_signature_case_failures.py`             | circleworld_proto    | analysis                 | documented_or_reported       | 157  | 11    | 0       |
| `runtimes/circleworld_proto/assemble_childworld_cycle_report.py`                    | circleworld_proto    | report_assembler         | documented_or_reported       | 616  | 22    | 0       |
| `runtimes/circleworld_proto/assemble_childworld_mechanism_audit.py`                 | circleworld_proto    | report_assembler         | documented_or_reported       | 622  | 24    | 0       |
| `runtimes/circleworld_proto/assemble_circleworld_experiment_ledger.py`              | circleworld_proto    | report_assembler         | documented_or_reported       | 556  | 32    | 1       |
| `runtimes/circleworld_proto/assemble_circleworld_scoreboard.py`                     | circleworld_proto    | report_assembler         | documented_or_reported       | 322  | 11    | 0       |
| `runtimes/circleworld_proto/assemble_phase_native_audio_route_policy_comparison.py` | circleworld_proto    | report_assembler         | canonical_or_manifested      | 133  | 7     | 0       |
| `runtimes/circleworld_proto/assemble_phase_native_audio_shared_battlefield.py`      | circleworld_proto    | report_assembler         | canonical_or_manifested      | 203  | 11    | 0       |
| `runtimes/circleworld_proto/assemble_phase_token_repair_report.py`                  | circleworld_proto    | report_assembler         | documented_or_reported       | 438  | 21    | 0       |
| `runtimes/circleworld_proto/assemble_rafa_claim_evidence.py`                        | circleworld_proto    | report_assembler         | documented_or_reported       | 6420 | 66    | 0       |
| `runtimes/circleworld_proto/assemble_static_child_conversion_report.py`             | circleworld_proto    | report_assembler         | documented_or_reported       | 404  | 24    | 0       |
| `runtimes/circleworld_proto/audit_circleworld_dag.py`                               | circleworld_proto    | auditor                  | documented_or_reported       | 191  | 11    | 0       |
| `runtimes/circleworld_proto/audit_circleworld_seed_determinism.py`                  | circleworld_proto    | auditor                  | documented_or_reported       | 579  | 28    | 0       |
| `runtimes/circleworld_proto/audit_phase_native_audio_route_selector_contract.py`    | circleworld_proto    | auditor                  | canonical_or_manifested      | 332  | 13    | 0       |
| `runtimes/circleworld_proto/audit_relational_signature_contract.py`                 | circleworld_proto    | auditor                  | documented_or_reported       | 188  | 5     | 0       |
| `runtimes/circleworld_proto/audit_seeded_child_substrate.py`                        | circleworld_proto    | auditor                  | documented_or_reported       | 215  | 7     | 0       |
| `runtimes/circleworld_proto/benchmark_audio_continuation.py`                        | circleworld_proto    | benchmark                | documented_or_reported       | 831  | 31    | 0       |
| `runtimes/circleworld_proto/benchmark_audio_continuity.py`                          | circleworld_proto    | benchmark                | canonical_or_manifested      | 589  | 25    | 0       |
| `runtimes/circleworld_proto/benchmark_circleworld_real_anchor.py`                   | circleworld_proto    | benchmark                | canonical_or_manifested      | 274  | 8     | 0       |
| `runtimes/circleworld_proto/build_audio_predeclared_lockbox.py`                     | circleworld_proto    | builder                  | documented_or_reported       | 792  | 27    | 0       |
| `runtimes/circleworld_proto/build_childworld_mechanism_dataset.py`                  | circleworld_proto    | builder                  | documented_or_reported       | 852  | 39    | 0       |
| `runtimes/circleworld_proto/build_internal_phase_law_failure_atlas.py`              | circleworld_proto    | builder                  | documented_or_reported       | 336  | 10    | 0       |
| `runtimes/circleworld_proto/build_internal_phase_law_variant_spec.py`               | circleworld_proto    | builder                  | documented_or_reported       | 1309 | 21    | 0       |
| `runtimes/circleworld_proto/build_law_token_library.py`                             | circleworld_proto    | builder                  | documented_or_reported       | 306  | 8     | 0       |
| `runtimes/circleworld_proto/build_relational_signature_library.py`                  | circleworld_proto    | builder                  | documented_or_reported       | 40   | 1     | 0       |
| `runtimes/circleworld_proto/build_shadow_branch_law_table.py`                       | circleworld_proto    | builder                  | documented_or_reported       | 191  | 8     | 0       |
| `runtimes/circleworld_proto/causal_operator_selector.py`                            | circleworld_proto    | runtime_or_shared_module | documented_or_reported       | 66   | 4     | 2       |
| `runtimes/circleworld_proto/classify_internal_phase_law_candidate_classes.py`       | circleworld_proto    | analysis                 | documented_or_reported       | 185  | 11    | 0       |
| `runtimes/circleworld_proto/common_io.py`                                           | circleworld_proto    | script_or_module         | documented_or_reported       | 112  | 14    | 0       |
| `runtimes/circleworld_proto/compare_audio_continuation_methods.py`                  | circleworld_proto    | comparator               | documented_or_reported       | 253  | 9     | 0       |
| `runtimes/circleworld_proto/compare_audio_lockbox_results.py`                       | circleworld_proto    | comparator               | documented_or_reported       | 893  | 36    | 0       |
| `runtimes/circleworld_proto/compare_internal_phase_law_variants.py`                 | circleworld_proto    | comparator               | documented_or_reported       | 319  | 12    | 0       |
| `runtimes/circleworld_proto/compare_learned_signature_scouts.py`                    | circleworld_proto    | comparator               | documented_or_reported       | 354  | 11    | 0       |
| `runtimes/circleworld_proto/compare_learned_signature_split_suites.py`              | circleworld_proto    | comparator               | documented_or_reported       | 254  | 11    | 0       |
| `runtimes/circleworld_proto/compare_relational_signature_runs.py`                   | circleworld_proto    | comparator               | documented_or_reported       | 473  | 15    | 0       |
| `runtimes/circleworld_proto/compare_signature_operator_tail_probes.py`              | circleworld_proto    | comparator               | documented_or_reported       | 166  | 7     | 0       |
| `runtimes/circleworld_proto/diagnose_internal_phase_law_joint_rows.py`              | circleworld_proto    | analysis                 | documented_or_reported       | 318  | 16    | 0       |
| `runtimes/circleworld_proto/evaluate_circleworld.py`                                | circleworld_proto    | evaluator                | documented_or_reported       | 421  | 7     | 0       |
| `runtimes/circleworld_proto/evaluate_dense_signature_claim.py`                      | circleworld_proto    | evaluator                | documented_or_reported       | 551  | 25    | 0       |
| `runtimes/circleworld_proto/evaluate_relational_metamers.py`                        | circleworld_proto    | evaluator                | documented_or_reported       | 452  | 20    | 0       |
| `runtimes/circleworld_proto/evaluate_shadow_branch_law_calibration.py`              | circleworld_proto    | evaluator                | documented_or_reported       | 367  | 10    | 0       |
| `runtimes/circleworld_proto/evaluate_signature_future_law_probe.py`                 | circleworld_proto    | evaluator                | documented_or_reported       | 674  | 17    | 0       |
| `runtimes/circleworld_proto/evaluate_signature_operator_tail_probe.py`              | circleworld_proto    | evaluator                | documented_or_reported       | 637  | 23    | 0       |
| `runtimes/circleworld_proto/evaluate_signature_tail_incremental_usefulness.py`      | circleworld_proto    | evaluator                | documented_or_reported       | 907  | 20    | 0       |
| `runtimes/circleworld_proto/experiment_reports.py`                                  | circleworld_proto    | script_or_module         | documented_or_reported       | 94   | 10    | 0       |
| `runtimes/circleworld_proto/export_circleworld_audio.py`                            | circleworld_proto    | exporter                 | documented_or_reported       | 321  | 12    | 0       |
| `runtimes/circleworld_proto/package_phase_native_audio_route_selector_profile.py`   | circleworld_proto    | packager                 | canonical_or_manifested      | 284  | 13    | 0       |
| `runtimes/circleworld_proto/phase_audio_metrics.py`                                 | circleworld_proto    | script_or_module         | documented_or_reported       | 106  | 7     | 0       |
| `runtimes/circleworld_proto/phase_native_audio_operators.py`                        | circleworld_proto    | runtime_or_shared_module | documented_or_reported       | 259  | 20    | 0       |
| `runtimes/circleworld_proto/predeclare_internal_phase_law_diagnostic_class.py`      | circleworld_proto    | analysis                 | documented_or_reported       | 262  | 11    | 0       |
| `runtimes/circleworld_proto/profile_registry.py`                                    | circleworld_proto    | script_or_module         | documented_or_reported       | 340  | 18    | 2       |
| `runtimes/circleworld_proto/resonant_law_objects.py`                                | circleworld_proto    | runtime_or_shared_module | documented_or_reported       | 781  | 30    | 0       |
| `runtimes/circleworld_proto/retrospective_childworld_eval.py`                       | circleworld_proto    | script_or_module         | documented_or_reported       | 370  | 12    | 0       |
| `runtimes/circleworld_proto/run_ablation.py`                                        | circleworld_proto    | experiment_runner        | canonical_or_manifested      | 26   | 1     | 0       |
| `runtimes/circleworld_proto/run_audio_baseline_stratified_diagnostics.py`           | circleworld_proto    | experiment_runner        | documented_or_reported       | 250  | 11    | 0       |
| `runtimes/circleworld_proto/run_audio_circle_delta_probe.py`                        | circleworld_proto    | experiment_runner        | documented_or_reported       | 642  | 15    | 0       |
| `runtimes/circleworld_proto/run_audio_delta_mechanism_probe.py`                     | circleworld_proto    | experiment_runner        | documented_or_reported       | 725  | 25    | 0       |
| `runtimes/circleworld_proto/run_audio_delta_objective_scout.py`                     | circleworld_proto    | experiment_runner        | documented_or_reported       | 520  | 15    | 0       |
| `runtimes/circleworld_proto/run_audio_electric_motor_holdout.py`                    | circleworld_proto    | experiment_runner        | documented_or_reported       | 903  | 25    | 0       |
| `runtimes/circleworld_proto/run_audio_mechanism_family_sensitivity.py`              | circleworld_proto    | experiment_runner        | documented_or_reported       | 560  | 18    | 0       |
| `runtimes/circleworld_proto/run_audio_phase_influence_ablation.py`                  | circleworld_proto    | experiment_runner        | documented_or_reported       | 396  | 9     | 0       |
| `runtimes/circleworld_proto/run_audio_phase_phenotype_diagnostics.py`               | circleworld_proto    | experiment_runner        | documented_or_reported       | 426  | 18    | 0       |
| `runtimes/circleworld_proto/run_audio_phase_seed_ablation.py`                       | circleworld_proto    | experiment_runner        | documented_or_reported       | 477  | 15    | 0       |
| `runtimes/circleworld_proto/run_audio_steady_phenotype_manifest.py`                 | circleworld_proto    | experiment_runner        | documented_or_reported       | 570  | 10    | 0       |
| `runtimes/circleworld_proto/run_branch_pressure_experiment.py`                      | circleworld_proto    | experiment_runner        | documented_or_reported       | 292  | 8     | 0       |
| `runtimes/circleworld_proto/run_branchlaw_ablation_series.py`                       | circleworld_proto    | experiment_runner        | documented_or_reported       | 339  | 12    | 0       |
| `runtimes/circleworld_proto/run_causal_operator_selector_probe.py`                  | circleworld_proto    | experiment_runner        | documented_or_reported       | 738  | 15    | 0       |
| `runtimes/circleworld_proto/run_child_basis_coverage_assay.py`                      | circleworld_proto    | experiment_runner        | documented_or_reported       | 515  | 8     | 0       |
| `runtimes/circleworld_proto/run_child_writeback_ablation.py`                        | circleworld_proto    | experiment_runner        | documented_or_reported       | 202  | 5     | 0       |
| `runtimes/circleworld_proto/run_childworld_causality_killswitch_suite.py`           | circleworld_proto    | experiment_runner        | documented_or_reported       | 491  | 7     | 0       |
| `runtimes/circleworld_proto/run_childworld_experiment.py`                           | circleworld_proto    | experiment_runner        | documented_or_reported       | 3112 | 18    | 0       |
| `runtimes/circleworld_proto/run_childworld_volume_recursion_sweep.py`               | circleworld_proto    | experiment_runner        | documented_or_reported       | 1614 | 11    | 0       |
| `runtimes/circleworld_proto/run_circleworld_operator_block.py`                      | circleworld_proto    | experiment_runner        | canonical_or_manifested      | 356  | 11    | 0       |
| `runtimes/circleworld_proto/run_claim_isolation_suite.py`                           | circleworld_proto    | experiment_runner        | documented_or_reported       | 621  | 25    | 0       |
| `runtimes/circleworld_proto/run_contrastive_structural_embedding_probe.py`          | circleworld_proto    | experiment_runner        | documented_or_reported       | 601  | 20    | 1       |
| `runtimes/circleworld_proto/run_graduation_phase_native_bridge.py`                  | circleworld_proto    | experiment_runner        | canonical_or_manifested      | 475  | 9     | 0       |
| `runtimes/circleworld_proto/run_internal_phase_law_objective_scout.py`              | circleworld_proto    | experiment_runner        | documented_or_reported       | 791  | 28    | 0       |
| `runtimes/circleworld_proto/run_learned_branch_law_assay.py`                        | circleworld_proto    | experiment_runner        | documented_or_reported       | 571  | 15    | 0       |
| `runtimes/circleworld_proto/run_learned_gated_manychild_runtime_sandbox.py`         | circleworld_proto    | experiment_runner        | documented_or_reported       | 1303 | 19    | 0       |
| `runtimes/circleworld_proto/run_learned_gated_multistep_operator_sandbox.py`        | circleworld_proto    | experiment_runner        | documented_or_reported       | 489  | 13    | 0       |
| `runtimes/circleworld_proto/run_learned_gated_runtime_sandbox.py`                   | circleworld_proto    | experiment_runner        | documented_or_reported       | 393  | 8     | 0       |
| `runtimes/circleworld_proto/run_learned_gated_writeback_sandbox.py`                 | circleworld_proto    | experiment_runner        | documented_or_reported       | 327  | 10    | 0       |
| `runtimes/circleworld_proto/run_learned_signature_split_suite.py`                   | circleworld_proto    | experiment_runner        | documented_or_reported       | 672  | 11    | 0       |
| `runtimes/circleworld_proto/run_learned_structural_family_probe.py`                 | circleworld_proto    | experiment_runner        | documented_or_reported       | 443  | 13    | 1       |
| `runtimes/circleworld_proto/run_parent_ontology_bridge_scout.py`                    | circleworld_proto    | experiment_runner        | documented_or_reported       | 783  | 30    | 1       |
| `runtimes/circleworld_proto/run_phase_native_audio_reset_suite.py`                  | circleworld_proto    | experiment_runner        | canonical_or_manifested      | 496  | 17    | 1       |
| `runtimes/circleworld_proto/run_phase_native_audio_selected_route.py`               | circleworld_proto    | experiment_runner        | canonical_or_manifested      | 288  | 8     | 0       |
| `runtimes/circleworld_proto/run_q_basis_ablation.py`                                | circleworld_proto    | experiment_runner        | documented_or_reported       | 449  | 17    | 0       |
| `runtimes/circleworld_proto/run_relational_signature_experiment.py`                 | circleworld_proto    | experiment_runner        | documented_or_reported       | 817  | 22    | 0       |
| `runtimes/circleworld_proto/run_resonant_child_retrieval_assay.py`                  | circleworld_proto    | experiment_runner        | documented_or_reported       | 189  | 7     | 0       |
| `runtimes/circleworld_proto/run_resonant_operator_causality_assay.py`               | circleworld_proto    | experiment_runner        | documented_or_reported       | 1005 | 19    | 0       |
| `runtimes/circleworld_proto/run_retrospective_grid.py`                              | circleworld_proto    | experiment_runner        | documented_or_reported       | 251  | 12    | 0       |
| `runtimes/circleworld_proto/run_substrate_ablation.py`                              | circleworld_proto    | experiment_runner        | documented_or_reported       | 525  | 20    | 0       |
| `runtimes/circleworld_proto/run_token_diversity_experiment.py`                      | circleworld_proto    | experiment_runner        | documented_or_reported       | 276  | 8     | 0       |
| `runtimes/circleworld_proto/score_internal_phase_law_objective.py`                  | circleworld_proto    | scorer                   | documented_or_reported       | 410  | 15    | 0       |
| `runtimes/circleworld_proto/score_phase_native_audio_prefix_router_scout.py`        | circleworld_proto    | scorer                   | canonical_or_manifested      | 414  | 20    | 0       |
| `runtimes/circleworld_proto/score_phase_native_audio_reentry_guard.py`              | circleworld_proto    | scorer                   | canonical_or_manifested      | 372  | 18    | 0       |
| `runtimes/circleworld_proto/score_phase_native_audio_reentry_metric_audit.py`       | circleworld_proto    | scorer                   | canonical_or_manifested      | 325  | 16    | 0       |
| `runtimes/circleworld_proto/score_phase_native_audio_reentry_oracle.py`             | circleworld_proto    | scorer                   | canonical_or_manifested      | 281  | 12    | 0       |
| `runtimes/circleworld_proto/score_phase_native_audio_route_transfer.py`             | circleworld_proto    | scorer                   | canonical_or_manifested      | 163  | 7     | 0       |
| `runtimes/circleworld_proto/score_phase_native_audio_target_replay_oracle.py`       | circleworld_proto    | scorer                   | canonical_or_manifested      | 366  | 16    | 0       |
| `runtimes/circleworld_proto/seed_substrate.py`                                      | circleworld_proto    | script_or_module         | active_unregistered_research | 339  | 18    | 2       |
| `runtimes/circleworld_proto/summarize_learned_signature_frontier.py`                | circleworld_proto    | report_assembler         | documented_or_reported       | 419  | 24    | 0       |
| `runtimes/circleworld_proto/sweep_childsurvival_replay.py`                          | circleworld_proto    | sweep                    | documented_or_reported       | 192  | 8     | 0       |
| `runtimes/circleworld_proto/sweep_parentmix_replay.py`                              | circleworld_proto    | sweep                    | documented_or_reported       | 161  | 7     | 0       |
| `runtimes/circleworld_proto/test_circleworld_operator_block_contract.py`            | circleworld_proto    | contract_test            | contract_or_test             | 62   | 3     | 0       |
| `runtimes/circleworld_proto/test_internal_phase_law_reentry_contract.py`            | circleworld_proto    | contract_test            | contract_or_test             | 113  | 4     | 0       |
| `runtimes/circleworld_proto/test_learned_branch_law_child_ifs_contract.py`          | circleworld_proto    | contract_test            | contract_or_test             | 353  | 11    | 0       |
| `runtimes/circleworld_proto/test_nested_commitment.py`                              | circleworld_proto    | contract_test            | canonical_or_manifested      | 3627 | 92    | 0       |
| `runtimes/circleworld_proto/test_packet_alias_contract.py`                          | circleworld_proto    | contract_test            | contract_or_test             | 303  | 18    | 0       |
| `runtimes/circleworld_proto/test_phase_gauge_invariance.py`                         | circleworld_proto    | contract_test            | contract_or_test             | 372  | 18    | 0       |
| `runtimes/circleworld_proto/test_relational_signature_learning_contract.py`         | circleworld_proto    | contract_test            | contract_or_test             | 236  | 7     | 0       |
| `runtimes/circleworld_proto/test_resonant_attention_contract.py`                    | circleworld_proto    | contract_test            | contract_or_test             | 227  | 14    | 0       |
| `runtimes/circleworld_proto/test_semantic_projector_contract.py`                    | circleworld_proto    | contract_test            | contract_or_test             | 391  | 23    | 2       |
| `runtimes/circleworld_proto/test_unit_phasor_contract.py`                           | circleworld_proto    | contract_test            | contract_or_test             | 1031 | 28    | 0       |
| `runtimes/circleworld_proto/train_childworld_mechanism_classifier.py`               | circleworld_proto    | trainer                  | documented_or_reported       | 261  | 12    | 0       |
| `runtimes/circleworld_proto/train_circleworld.py`                                   | circleworld_proto    | trainer                  | canonical_or_manifested      | 1535 | 27    | 0       |
| `runtimes/circleworld_proto/train_circleworld_real_anchor.py`                       | circleworld_proto    | trainer                  | canonical_or_manifested      | 3632 | 43    | 0       |
| `runtimes/circleworld_proto/train_learned_signature_scout.py`                       | circleworld_proto    | trainer                  | documented_or_reported       | 1516 | 34    | 0       |
| `runtimes/circleworld_proto/train_parent_phase_projector.py`                        | circleworld_proto    | trainer                  | documented_or_reported       | 743  | 22    | 0       |
| `runtimes/circleworld_proto/train_phase_native_audio_family_route_policy.py`        | circleworld_proto    | trainer                  | canonical_or_manifested      | 308  | 17    | 0       |
| `runtimes/circleworld_proto/train_phase_native_audio_objective_route_policy.py`     | circleworld_proto    | trainer                  | canonical_or_manifested      | 335  | 17    | 0       |
| `runtimes/circleworld_proto/train_phase_native_audio_route_policy.py`               | circleworld_proto    | trainer                  | canonical_or_manifested      | 300  | 16    | 0       |
| `runtimes/circleworld_proto/train_shadow_branch_law_from_table.py`                  | circleworld_proto    | trainer                  | documented_or_reported       | 340  | 15    | 0       |
| `runtimes/circleworld_proto/validate_tokenburst_tracks.py`                          | circleworld_proto    | auditor                  | documented_or_reported       | 178  | 6     | 0       |
| `runtimes/diffusion_parent_v3/sample_prompt.py`                                     | diffusion_parent_v3  | exporter                 | canonical_or_manifested      | 27   | 1     | 0       |
| `runtimes/stage4_blackwell_14/export_graduation.py`                                 | stage4_blackwell_14  | exporter                 | canonical_or_manifested      | 29   | 1     | 0       |
| `runtimes/stage4_blackwell_16/export_graduation.py`                                 | stage4_blackwell_16  | exporter                 | canonical_or_manifested      | 18   | 1     | 0       |
| `sample_diffusion.py`                                                               | root_legacy          | exporter                 | canonical_or_manifested      | 277  | 7     | 0       |
| `stft_utils.py`                                                                     | root_legacy          | runtime_or_shared_module | canonical_or_manifested      | 55   | 3     | 0       |
| `tests/test_registry_integrity.py`                                                  | repo_tests           | contract_test            | contract_or_test             | 99   | 1     | 1       |
| `tools/ablate_contract.py`                                                          | tools                | script_or_module         | documented_or_reported       | 110  | 3     | 0       |
| `tools/acid_test.py`                                                                | tools                | script_or_module         | documented_or_reported       | 102  | 2     | 0       |
| `tools/audio_debug.py`                                                              | tools                | script_or_module         | documented_or_reported       | 101  | 5     | 0       |
| `tools/audit_clap.py`                                                               | tools                | auditor                  | documented_or_reported       | 20   | 1     | 0       |
| `tools/audit_hierarchy.py`                                                          | tools                | auditor                  | documented_or_reported       | 19   | 1     | 0       |
| `tools/audit_params.py`                                                             | tools                | auditor                  | documented_or_reported       | 34   | 1     | 0       |
| `tools/audit_project_scripts.py`                                                    | tools                | auditor                  | documented_or_reported       | 990  | 34    | 2       |
| `tools/audit_project_usage.py`                                                      | tools                | auditor                  | documented_or_reported       | 881  | 28    | 4       |
| `tools/binding_stats.py`                                                            | tools                | script_or_module         | documented_or_reported       | 46   | 1     | 0       |
| `tools/causal_audit.py`                                                             | tools                | script_or_module         | documented_or_reported       | 64   | 1     | 0       |
| `tools/debug_nan.py`                                                                | tools                | script_or_module         | documented_or_reported       | 57   | 3     | 0       |
| `tools/dsp_labeler.py`                                                              | tools                | script_or_module         | documented_or_reported       | 96   | 3     | 0       |
| `tools/eval_semantic_tension.py`                                                    | tools                | script_or_module         | documented_or_reported       | 242  | 5     | 0       |
| `tools/export_audio.py`                                                             | tools                | exporter                 | canonical_or_manifested      | 68   | 2     | 0       |
| `tools/export_audio_compat14.py`                                                    | tools                | exporter                 | canonical_or_manifested      | 215  | 6     | 1       |
| `tools/export_graduation_pack_restore.py`                                           | tools                | exporter                 | canonical_or_manifested      | 68   | 3     | 0       |
| `tools/extract_tension_targets.py`                                                  | tools                | script_or_module         | documented_or_reported       | 96   | 3     | 0       |
| `tools/fault_localization.py`                                                       | tools                | script_or_module         | documented_or_reported       | 36   | 2     | 0       |
| `tools/hard_ablation_audit.py`                                                      | tools                | script_or_module         | documented_or_reported       | 83   | 2     | 0       |
| `tools/infer_rafa.py`                                                               | tools                | exporter                 | documented_or_reported       | 79   | 3     | 0       |
| `tools/invariance_utils.py`                                                         | tools                | script_or_module         | documented_or_reported       | 33   | 2     | 0       |
| `tools/manifest_gen.py`                                                             | tools                | script_or_module         | documented_or_reported       | 389  | 9     | 1       |
| `tools/path_b_eval.py`                                                              | tools                | script_or_module         | documented_or_reported       | 157  | 2     | 0       |
| `tools/profile_rafa.py`                                                             | tools                | script_or_module         | documented_or_reported       | 33   | 1     | 0       |
| `tools/profile_step.py`                                                             | tools                | script_or_module         | documented_or_reported       | 49   | 1     | 0       |
| `tools/rafa_leaderboard.py`                                                         | tools                | script_or_module         | documented_or_reported       | 62   | 1     | 0       |
| `tools/relational_probe.py`                                                         | tools                | script_or_module         | documented_or_reported       | 468  | 16    | 0       |
| `tools/render_binding_comparison.py`                                                | tools                | script_or_module         | documented_or_reported       | 78   | 2     | 0       |
| `tools/run_ablations.py`                                                            | tools                | experiment_runner        | documented_or_reported       | 114  | 3     | 0       |
| `tools/run_hypercube.py`                                                            | tools                | experiment_runner        | documented_or_reported       | 120  | 3     | 0       |
| `tools/run_learning_curve_resume.py`                                                | tools                | experiment_runner        | documented_or_reported       | 152  | 8     | 0       |
| `tools/run_validation.py`                                                           | tools                | experiment_runner        | documented_or_reported       | 143  | 3     | 0       |
| `tools/scrape_core.py`                                                              | tools                | script_or_module         | documented_or_reported       | 283  | 5     | 3       |
| `tools/scrape_freewavesamples.py`                                                   | tools                | script_or_module         | documented_or_reported       | 85   | 4     | 0       |
| `tools/scrape_high_volume.py`                                                       | tools                | script_or_module         | documented_or_reported       | 99   | 4     | 0       |
| `tools/scrape_mixkit.py`                                                            | tools                | script_or_module         | documented_or_reported       | 94   | 4     | 0       |
| `tools/scrape_soundbible.py`                                                        | tools                | script_or_module         | documented_or_reported       | 78   | 3     | 0       |
| `tools/scrape_wav.py`                                                               | tools                | script_or_module         | documented_or_reported       | 54   | 2     | 0       |
| `tools/scrape_wavsource.py`                                                         | tools                | script_or_module         | documented_or_reported       | 79   | 4     | 0       |
| `tools/test_relational_logic.py`                                                    | tools                | contract_test            | contract_or_test             | 207  | 2     | 0       |
| `tools/test_triton.py`                                                              | tools                | contract_test            | contract_or_test             | 97   | 4     | 0       |
| `tools/test_triton_deq.py`                                                          | tools                | contract_test            | contract_or_test             | 42   | 1     | 0       |
| `tools/tests_diffusion_shell.py`                                                    | tools                | script_or_module         | documented_or_reported       | 38   | 1     | 0       |
| `tools/total_recall.py`                                                             | tools                | script_or_module         | documented_or_reported       | 38   | 1     | 0       |
| `tools/train_stage1_voice.py`                                                       | tools                | trainer                  | documented_or_reported       | 47   | 1     | 0       |
| `tools/train_stage2_spine.py`                                                       | tools                | trainer                  | documented_or_reported       | 56   | 1     | 0       |
| `tools/train_stage3_brain.py`                                                       | tools                | trainer                  | documented_or_reported       | 83   | 3     | 0       |
| `tools/verify.py`                                                                   | tools                | auditor                  | documented_or_reported       | 24   | 1     | 0       |
| `tools/verify_physics.py`                                                           | tools                | auditor                  | documented_or_reported       | 443  | 11    | 0       |
| `train_functional.py`                                                               | root_legacy          | trainer                  | documented_or_reported       | 118  | 2     | 1       |
| `train_raw_tensors.py`                                                              | root_legacy          | trainer                  | documented_or_reported       | 57   | 2     | 0       |
| `train_sterile.py`                                                                  | root_legacy          | trainer                  | documented_or_reported       | 60   | 1     | 0       |

## Maintenance Rule

Regenerate this file after any large experiment burst or before a cleanup/refactor pass:

```powershell
python tools/audit_project_scripts.py
```
