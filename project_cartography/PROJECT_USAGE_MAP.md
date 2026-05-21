# Project Usage Map

Generated: `2026-05-20T17:36:23+00:00`

This is a conservative static usage map. It records imports, text references, manifest/registry visibility, artifact groups, and cold candidates. It does not prove deletion safety.

Top-level artifact aliases such as `outputs/`, `eval/`, `logs/`, and many `checkpoints_*` roots are normalized to their consolidated physical groups under `artifacts/` when possible.

## Summary

| Metric                 | Value       |
| ---------------------- | ----------- |
| Scripts                | 239         |
| Artifact files indexed | 54243       |
| Artifact groups        | 979         |
| Artifact bytes         | 93894554467 |

## Script Use Status

| Status                            | Scripts |
| --------------------------------- | ------- |
| documented_reference              | 143     |
| imported_dependency               | 48      |
| canonical_or_manifested           | 47      |
| untracked_low_reference_candidate | 1       |

## Artifact Group Status

| Status                       | Groups |
| ---------------------------- | ------ |
| cold_unreferenced_candidate  | 543    |
| referenced                   | 377    |
| producer_name_candidate_only | 59     |

## Most Imported Scripts

| Script                                                                        | Imported by | Lane                 | Role                     |
| ----------------------------------------------------------------------------- | ----------- | -------------------- | ------------------------ |
| `core/config.py`                                                              | 55          | core_shared          | script_or_module         |
| `rafa_math_tools.py`                                                          | 36          | root_legacy          | runtime_or_shared_module |
| `lineages/04_positive_replacement/circleworld.py`                             | 29          | positive_replacement | runtime_or_shared_module |
| `stft_utils.py`                                                               | 27          | root_legacy          | runtime_or_shared_module |
| `runtimes/circleworld_proto/evaluate_circleworld.py`                          | 24          | circleworld_proto    | evaluator                |
| `diffusion_utils.py`                                                          | 23          | root_legacy          | runtime_or_shared_module |
| `inference_package/diffusion_utils.py`                                        | 23          | inference_package    | runtime_or_shared_module |
| `core/dataset.py`                                                             | 21          | core_shared          | script_or_module         |
| `core/lib_blackwell.py`                                                       | 19          | core_shared          | script_or_module         |
| `inference_package/lib_blackwell.py`                                          | 19          | inference_package    | script_or_module         |
| `runtimes/circleworld_proto/export_circleworld_audio.py`                      | 15          | circleworld_proto    | exporter                 |
| `lineages/04_positive_replacement/ablate_formalization.py`                    | 14          | positive_replacement | script_or_module         |
| `runtimes/circleworld_proto/test_nested_commitment.py`                        | 13          | circleworld_proto    | contract_test            |
| `model.py`                                                                    | 12          | root_legacy          | runtime_or_shared_module |
| `runtimes/circleworld_proto/benchmark_audio_continuation.py`                  | 11          | circleworld_proto    | benchmark                |
| `core/diffusion_models.py`                                                    | 10          | core_shared          | runtime_or_shared_module |
| `diffusion_models.py`                                                         | 10          | root_legacy          | runtime_or_shared_module |
| `runtimes/circleworld_proto/common_io.py`                                     | 10          | circleworld_proto    | script_or_module         |
| `runtimes/circleworld_proto/resonant_law_objects.py`                          | 10          | circleworld_proto    | runtime_or_shared_module |
| `runtimes/circleworld_proto/benchmark_circleworld_real_anchor.py`             | 9           | circleworld_proto    | benchmark                |
| `lineages/04_positive_replacement/rafa_relational_signature.py`               | 8           | positive_replacement | script_or_module         |
| `runtimes/circleworld_proto/benchmark_audio_continuity.py`                    | 8           | circleworld_proto    | benchmark                |
| `runtimes/circleworld_proto/build_law_token_library.py`                       | 7           | circleworld_proto    | builder                  |
| `runtimes/circleworld_proto/score_phase_native_audio_reentry_guard.py`        | 7           | circleworld_proto    | scorer                   |
| `runtimes/circleworld_proto/train_circleworld_real_anchor.py`                 | 7           | circleworld_proto    | trainer                  |
| `phase_native_ifs.py`                                                         | 6           | root_legacy          | runtime_or_shared_module |
| `runtimes/circleworld_proto/build_shadow_branch_law_table.py`                 | 6           | circleworld_proto    | builder                  |
| `runtimes/circleworld_proto/evaluate_shadow_branch_law_calibration.py`        | 6           | circleworld_proto    | evaluator                |
| `runtimes/circleworld_proto/run_audio_delta_mechanism_probe.py`               | 6           | circleworld_proto    | experiment_runner        |
| `hf_local.py`                                                                 | 5           | root_legacy          | script_or_module         |
| `runtimes/circleworld_proto/run_resonant_operator_causality_assay.py`         | 5           | circleworld_proto    | experiment_runner        |
| `runtimes/circleworld_proto/causal_operator_selector.py`                      | 4           | circleworld_proto    | runtime_or_shared_module |
| `runtimes/circleworld_proto/profile_registry.py`                              | 4           | circleworld_proto    | script_or_module         |
| `runtimes/circleworld_proto/run_contrastive_structural_embedding_probe.py`    | 4           | circleworld_proto    | experiment_runner        |
| `runtimes/circleworld_proto/run_learned_gated_multistep_operator_sandbox.py`  | 4           | circleworld_proto    | experiment_runner        |
| `runtimes/circleworld_proto/run_learned_gated_writeback_sandbox.py`           | 4           | circleworld_proto    | experiment_runner        |
| `runtimes/circleworld_proto/score_phase_native_audio_prefix_router_scout.py`  | 4           | circleworld_proto    | scorer                   |
| `runtimes/circleworld_proto/score_phase_native_audio_target_replay_oracle.py` | 4           | circleworld_proto    | scorer                   |
| `runtimes/circleworld_proto/train_learned_signature_scout.py`                 | 4           | circleworld_proto    | trainer                  |
| `runtimes/circleworld_proto/train_shadow_branch_law_from_table.py`            | 4           | circleworld_proto    | trainer                  |

## Low-Reference Script Candidates

These are not deletion instructions. They are the scripts to inspect first during cleanup.

| Script                                         | Status                            | Lane              | Role             | LOC | Tracked |
| ---------------------------------------------- | --------------------------------- | ----------------- | ---------------- | --- | ------- |
| `runtimes/circleworld_proto/seed_substrate.py` | untracked_low_reference_candidate | circleworld_proto | script_or_module | 339 | False   |

## Most Referenced Artifact Groups

| Group                                                                                                                             | Refs | Files | Bytes       | Kinds                                  |
| --------------------------------------------------------------------------------------------------------------------------------- | ---- | ----- | ----------- | -------------------------------------- |
| `docs/reports`                                                                                                                    | 223  | 121   | 2019588     | report:114, json:6, log:1              |
| `artifacts/runtime/outputs/circleworld_proto/continuation_causal_childifs_2026-05-11`                                             | 61   | 5855  | 1461062991  | audio:5414, json:414, log:26, table:1  |
| `artifacts/checkpoints/stages/checkpoints_stage4`                                                                                 | 25   | 836   | 41089472021 | checkpoint:836                         |
| `checkpoints_circleworld_proto/training_run_2026-04-24_agreement_scout_v1`                                                        | 41   | 1     | 11233       | json:1                                 |
| `artifacts/runtime/outputs/circleworld_proto/coherence_retention_childifs_2026-05-10`                                             | 18   | 850   | 388113221   | audio:786, json:55, log:6, table:2     |
| `artifacts/runtime/outputs/circleworld_proto/training_run_2026-04-21_token_diverse_ramanujan`                                     | 17   | 192   | 58080987    | audio:156, json:34, table:1, report:1  |
| `artifacts/runtime/outputs/circleworld_proto/nested_identity_childifs_2026-05-10`                                                 | 16   | 841   | 292032682   | audio:771, json:59, log:6, report:3    |
| `artifacts/runtime/outputs/circleworld_proto/training_run_2026-04-24_agreement_scout_v1`                                          | 15   | 64    | 17590752    | audio:41, json:22, report:1            |
| `artifacts/runtime/outputs/circleworld_proto/causal_retention_childifs_2026-05-10`                                                | 14   | 740   | 439784673   | audio:687, json:48, log:2, table:1     |
| `artifacts/runtime/outputs/circleworld_proto/fixed_substrate_guard_childifs_2026-05-10`                                           | 14   | 855   | 429209877   | audio:786, json:57, log:8, table:2     |
| `artifacts/runtime/outputs/circleworld_proto/training_run_2026-04-22_ramanujan_childworld_writeback_v3_identity`                  | 14   | 195   | 59872249    | audio:156, json:35, log:2, table:1     |
| `artifacts/runtime/outputs/circleworld_proto/continuation_causal_childifs_2026-05-13`                                             | 13   | 1867  | 321279649   | audio:1708, json:150, report:9         |
| `checkpoints_circleworld_proto/continuation_causal_childifs_2026-05-11_guarded`                                                   | 24   | 1     | 24323       | json:1                                 |
| `artifacts/runtime/outputs/circleworld_proto/qualified_persistence_push_2026-05-10`                                               | 12   | 1277  | 664051711   | audio:1191, json:76, log:8, report:1   |
| `artifacts/runtime/outputs/circleworld_proto/relsig_scout_balance_2026-05-04`                                                     | 12   | 56    | 30159733    | json:36, audio:15, report:4, table:1   |
| `artifacts/runtime/outputs/circleworld_proto/training_run_2026-04-22_ramanujan_childworld_writeback_v2`                           | 12   | 195   | 60576090    | audio:156, json:35, log:2, table:1     |
| `artifacts/runtime/outputs/circleworld_proto/training_run_2026-04-21_ramanujan_childworld_v1`                                     | 11   | 195   | 60434449    | audio:156, json:35, log:2, table:1     |
| `artifacts/runtime/outputs/circleworld_proto/branchlaw_ablation_series_2026-04-17`                                                | 9    | 889   | 262168331   | audio:780, json:103, table:5, report:1 |
| `artifacts/runtime/outputs/circleworld_proto/parentmix_sweep_2026-04-24_v1`                                                       | 9    | 200   | 52495351    | audio:146, json:44, table:8, report:2  |
| `checkpoints_circleworld_proto/training_run_2026-04-24_ramanujan_childworld_writeback_v11_hard_gate_seeded`                       | 16   | 1     | 11042       | json:1                                 |
| `artifacts/runtime/outputs/circleworld_proto/relational_signature_full_2026-05-04`                                                | 8    | 30    | 20320477    | json:28, report:2                      |
| `artifacts/runtime/outputs/circleworld_proto/relational_signature_smoke_2026-05-04`                                               | 8    | 10    | 1754076     | json:8, report:2                       |
| `artifacts/runtime/outputs/circleworld_proto/retrospective_v8_candidates_2026-04-23`                                              | 8    | 683   | 191012345   | audio:576, json:101, table:3, report:3 |
| `checkpoints_circleworld_proto/qualified_persistence_push_2026-05-10`                                                             | 14   | 1     | 21948       | json:1                                 |
| `artifacts/runtime/outputs/circleworld_proto/relational_signature_full_parentmix_2026-05-04`                                      | 7    | 30    | 21605293    | json:28, report:2                      |
| `artifacts/runtime/outputs/circleworld_proto/training_run_2026-04-21_qtrace_only_childworld_v1`                                   | 7    | 195   | 60499866    | audio:156, json:35, log:2, table:1     |
| `artifacts/runtime/outputs/circleworld_proto/training_run_2026-04-24_nested_gate_scout_v17`                                       | 7    | 289   | 83698845    | audio:244, json:43, table:1, report:1  |
| `checkpoints_circleworld_proto/coherence_retention_childifs_2026-05-10`                                                           | 12   | 2     | 28640       | json:2                                 |
| `checkpoints_circleworld_proto/manual_candidates`                                                                                 | 12   | 1     | 11109       | json:1                                 |
| `artifacts/runtime/outputs/circleworld_proto/childsurvival_sweep_2026-04-24_v1`                                                   | 6    | 285   | 66982940    | audio:157, json:98, table:27, report:3 |
| `artifacts/runtime/outputs/circleworld_proto/law_token_library_2026-04-20_ramanujan`                                              | 6    | 15    | 2762138     | json:14, report:1                      |
| `artifacts/runtime/outputs/circleworld_proto/proper_attempt_2026-04-06`                                                           | 6    | 4     | 926909      | json:4                                 |
| `artifacts/runtime/outputs/circleworld_proto/relational_eval_active_2026-05-04`                                                   | 6    | 2     | 433317      | json:1, table:1                        |
| `artifacts/runtime/outputs/circleworld_proto/training_run_2026-04-24_ramanujan_childworld_writeback_v13_targeted_replay_recovery` | 6    | 229   | 68399628    | audio:192, json:35, table:1, report:1  |
| `checkpoints_circleworld_proto/fixed_substrate_guard_candidate00_2026-05-10`                                                      | 10   | 2     | 6251        | json:2                                 |
| `artifacts/runtime/outputs/circleworld_proto/benchmark_2026-04-09_realanchor_constrained_expanded`                                | 5    | 39    | 11567281    | audio:36, json:3                       |
| `artifacts/runtime/outputs/circleworld_proto/inference_2026-04-09_realanchor_constrained_ab`                                      | 5    | 28    | 7701570     | audio:24, json:3, report:1             |
| `artifacts/runtime/outputs/circleworld_proto/parent_phase_carrier_replay_2026-05-13_heldout_9110_9112`                            | 5    | 714   | 127105901   | audio:672, json:41, report:1           |
| `artifacts/runtime/outputs/circleworld_proto/parent_phase_carrier_replay_2026-05-13_seeded_suite`                                 | 5    | 714   | 127078394   | audio:672, json:41, report:1           |
| `artifacts/runtime/outputs/circleworld_proto/parent_phase_projector_dataset_2026-05-13_seeded_suite`                              | 5    | 658   | 379761840   | audio:336, data:300, json:21, report:1 |
| `artifacts/runtime/outputs/circleworld_proto/parent_phase_projector_delta_strict_2026-05-13_seeded_suite`                         | 5    | 358   | 63551596    | audio:336, json:21, report:1           |
| `artifacts/runtime/outputs/circleworld_proto/parent_phase_projector_replay_2026-05-13_seeded_suite`                               | 5    | 358   | 63590981    | audio:336, json:21, report:1           |
| `artifacts/runtime/outputs/circleworld_proto/parent_phasor_carrier_replay_2026-05-13_heldout_9110_9112`                           | 5    | 714   | 127175783   | audio:672, json:41, report:1           |
| `artifacts/runtime/outputs/circleworld_proto/parent_phasor_carrier_replay_2026-05-13_seeded_suite`                                | 5    | 714   | 127140322   | audio:672, json:41, report:1           |
| `artifacts/runtime/outputs/circleworld_proto/relational_metamer_active_2026-05-04`                                                | 5    | 2     | 71840       | report:1, json:1                       |
| `artifacts/runtime/outputs/circleworld_proto/relational_metamer_parentmix_2026-05-04`                                             | 5    | 2     | 72058       | report:1, json:1                       |
| `checkpoints_circleworld_proto/causal_retention_childifs_2026-05-10`                                                              | 8    | 1     | 22605       | json:1                                 |
| `artifacts/checkpoints/runs/checkpoints_diffusion_prompt_conditioned_clap_smoke_real_quick`                                       | 4    | 2     | 574471092   | checkpoint:1, json:1                   |
| `artifacts/checkpoints/runs/checkpoints_diffusion_rafa_full`                                                                      | 4    | 4     | 93986987    | checkpoint:2, json:2                   |
| `artifacts/runtime/logs/ablations`                                                                                                | 4    | 12    | 37126       | log:12                                 |

## Largest Cold Artifact Groups

Cold means this static scan did not find a source/doc/registry/report reference to the group. It may still be historically important.

| Group                                                                                                                          | Files | Bytes      | Newest                    | Kinds                                |
| ------------------------------------------------------------------------------------------------------------------------------ | ----- | ---------- | ------------------------- | ------------------------------------ |
| `artifacts/checkpoints/stages/checkpoints_stage3`                                                                              | 500   | 8331583599 | 2026-03-20T00:11:23+00:00 | checkpoint:500                       |
| `artifacts/runtime/tmp/torch-2.12.0.dev20260315+cu128-cp310-cp310-win_amd64.whl`                                               | 1     | 2761018141 | 2026-03-15T14:25:16+00:00 | other:1                              |
| `artifacts/checkpoints/runs/checkpoints_diffusion_prompt_conditioned_anchor_consistency_smoke`                                 | 4     | 1723418322 | 2026-03-10T22:55:55+00:00 | checkpoint:3, json:1                 |
| `artifacts/checkpoints/runs/checkpoints_diffusion_prompt_conditioned_hybrid_smoke`                                             | 2     | 1194652155 | 2026-03-13T01:25:29+00:00 | checkpoint:1, json:1                 |
| `artifacts/checkpoints/runs/checkpoints_diffusion_prompt_conditioned_ifs_control_smoke`                                        | 2     | 574537972  | 2026-03-12T13:09:37+00:00 | checkpoint:1, json:1                 |
| `artifacts/checkpoints/runs/checkpoints_diffusion_prompt_conditioned_ifs_control_probe21`                                      | 2     | 574537963  | 2026-03-12T23:34:06+00:00 | checkpoint:1, json:1                 |
| `artifacts/checkpoints/runs/checkpoints_diffusion_prompt_conditioned_ifs_control_smoke_to120`                                  | 2     | 574537963  | 2026-03-13T00:06:12+00:00 | checkpoint:1, json:1                 |
| `artifacts/runtime/outputs/circleworld_proto/training_run_2026-04-07_full`                                                     | 3     | 518450175  | 2026-04-07T17:24:33+00:00 | json:3                               |
| `artifacts/checkpoints/runs/checkpoints_diffusion_bridge_lc_fast_s30_v3`                                                       | 12    | 249325740  | 2026-03-05T01:35:47+00:00 | checkpoint:6, json:6                 |
| `artifacts/checkpoints/runs/checkpoints_diffusion_pathb_hypercube_mem1_ram1_slow1_cond1_meta1`                                 | 3     | 139995550  | 2026-03-10T03:32:44+00:00 | checkpoint:2, json:1                 |
| `artifacts/runtime/outputs/circleworld_proto/heat_training_2026-05-20`                                                         | 51    | 100976259  | 2026-05-20T17:35:27+00:00 | json:26, log:13, report:7, other:3   |
| `artifacts/bundles/extra_results_bundle_2026-03-04`                                                                            | 931   | 81714254   | 2026-03-05T00:39:04+00:00 | json:854, audio:72, report:3, log:1  |
| `artifacts/checkpoints/runs/checkpoints_diffusion_semantic_tension_steering_smoke`                                             | 2     | 70039163   | 2026-03-13T13:42:44+00:00 | checkpoint:1, json:1                 |
| `artifacts/checkpoints/runs/checkpoints_diffusion_prompt_conditioned_clap_smoke_quick`                                         | 2     | 69996340   | 2026-03-10T21:25:06+00:00 | checkpoint:1, json:1                 |
| `artifacts/runtime/outputs/circleworld_proto/naked_ontology_childifs_2026-05-10`                                               | 248   | 67051688   | 2026-05-10T23:49:34+00:00 | audio:198, json:28, log:11, report:4 |
| `artifacts/runtime/outputs/circleworld_proto/phase_native_audio_reentry_shear_probe_2026-05-17_third_seed20260519_full`        | 2     | 57982555   | 2026-05-17T19:17:27+00:00 | report:1, json:1                     |
| `artifacts/runtime/outputs/circleworld_proto/phase_native_audio_reentry_shear_probe_2026-05-17_fresh_seed20260518_full`        | 2     | 57979699   | 2026-05-17T18:36:24+00:00 | report:1, json:1                     |
| `artifacts/runtime/outputs/circleworld_proto/phase_native_audio_reentry_decorrelator_probe_2026-05-17_third_seed20260519_full` | 2     | 51801661   | 2026-05-17T19:19:26+00:00 | report:1, json:1                     |
| `artifacts/runtime/outputs/circleworld_proto/phase_native_audio_reentry_decorrelator_probe_2026-05-17_fresh_seed20260518_full` | 2     | 51795367   | 2026-05-17T18:38:26+00:00 | report:1, json:1                     |
| `artifacts/runtime/outputs/circleworld_proto/phase_native_audio_reset_2026-05-17_third_seed20260519_full`                      | 631   | 49046474   | 2026-05-17T18:50:25+00:00 | audio:620, report:4, json:4, log:3   |
| `artifacts/runtime/outputs/circleworld_proto/phase_native_audio_reset_2026-05-17_fresh_seed20260518_full`                      | 631   | 49040947   | 2026-05-17T18:35:09+00:00 | audio:620, report:4, json:4, log:3   |
| `artifacts/runtime/outputs/circleworld_proto/phase_native_audio_reset_2026-05-17_fourth_seed20260520_full`                     | 631   | 49031652   | 2026-05-17T19:21:14+00:00 | audio:620, report:4, json:4, log:3   |
| `artifacts/runtime/validation_results/run_old_s1337_20260328_050513`                                                           | 4     | 43192964   | 2026-03-28T12:19:22+00:00 | checkpoint:2, table:1, json:1        |
| `artifacts/runtime/validation_results/run_mode_only_s42_20260328_051936`                                                       | 4     | 43192945   | 2026-03-28T12:33:38+00:00 | checkpoint:2, table:1, json:1        |
| `artifacts/runtime/validation_results/run_old_s42_20260328_043633`                                                             | 4     | 43192945   | 2026-03-28T11:50:33+00:00 | checkpoint:2, table:1, json:1        |
| `artifacts/runtime/validation_results/run_full_s1337_20260328_045047`                                                          | 4     | 43192935   | 2026-03-28T12:05:10+00:00 | checkpoint:2, table:1, json:1        |
| `artifacts/runtime/validation_results/run_full_s42_20260328_042207`                                                            | 4     | 43192934   | 2026-03-28T11:36:30+00:00 | checkpoint:2, table:1, json:1        |
| `artifacts/checkpoints/runs/checkpoints_diffusion_bridge_lc_vram_smoke`                                                        | 2     | 42034091   | 2026-03-05T03:38:04+00:00 | checkpoint:1, json:1                 |
| `artifacts/checkpoints/runs/checkpoints_diffusion_bridge_e2e_smoke`                                                            | 2     | 42033963   | 2026-03-04T20:04:05+00:00 | checkpoint:1, json:1                 |
| `artifacts/checkpoints/runs/checkpoints_diffusion_bridge_lc_fast_s30_v2`                                                       | 2     | 41554291   | 2026-03-05T01:25:45+00:00 | checkpoint:1, json:1                 |
| `artifacts/checkpoints/runs/checkpoints_diffusion_bridge_lc_fast_s30`                                                          | 2     | 41554227   | 2026-03-05T01:11:18+00:00 | checkpoint:1, json:1                 |
| `artifacts/runtime/outputs/circleworld_proto/resonant_grandchild_composition_2026-05-14_naked_gpu_9100_9102_postfix_run.log`   | 1     | 30681222   | 2026-05-14T15:46:03+00:00 | log:1                                |
| `artifacts/runtime/outputs/circleworld_proto/resonant_grandchild_composition_2026-05-14_naked_gpu_9100_9102_run.log`           | 1     | 30049950   | 2026-05-14T15:41:04+00:00 | log:1                                |
| `artifacts/runtime/outputs/circleworld_proto/resonant_grandchild_composition_2026-05-14_naked_gpu_9100_9102`                   | 88    | 28056311   | 2026-05-14T15:40:56+00:00 | audio:84, json:4                     |
| `artifacts/runtime/tmp/gradpack_grid`                                                                                          | 432   | 27667008   | 2026-04-06T02:50:07+00:00 | audio:432                            |
| `artifacts/runtime/outputs/circleworld_proto/training_run_2026-04-09_middle`                                                   | 3     | 25823120   | 2026-04-09T11:37:32+00:00 | json:3                               |
| `artifacts/runtime/outputs/circleworld_proto/training_run_2026-04-09_gradmix_full`                                             | 3     | 21604474   | 2026-04-09T20:15:31+00:00 | json:3                               |
| `artifacts/runtime/tmp/gradpack_param_sweep_ep10_1200`                                                                         | 301   | 15461966   | 2026-04-06T13:06:58+00:00 | audio:240, json:61                   |
| `artifacts/runtime/outputs/circleworld_proto/phase_native_audio_reentry_shear_highgain_2026-05-17_third_seed20260519_full`     | 2     | 12037697   | 2026-05-17T19:18:10+00:00 | report:1, json:1                     |
| `artifacts/runtime/outputs/circleworld_proto/phase_native_audio_reentry_shear_highgain_2026-05-17_fresh_seed20260518_full`     | 2     | 12036352   | 2026-05-17T18:37:04+00:00 | report:1, json:1                     |
| `artifacts/runtime/outputs/circleworld_proto/benchmark_2026-04-09_realanchor_focus_expanded`                                   | 37    | 11553355   | 2026-04-09T22:36:00+00:00 | audio:36, json:1                     |
| `artifacts/runtime/outputs/circleworld_proto/benchmark_2026-04-09_realanchor_band_expanded`                                    | 37    | 11553305   | 2026-04-09T22:36:00+00:00 | audio:36, json:1                     |
| `artifacts/runtime/outputs/circleworld_proto/child_local_ifs_training_2026-05-10`                                              | 71    | 11128542   | 2026-05-10T14:53:09+00:00 | audio:30, json:22, log:8, table:5    |
| `artifacts/runtime/outputs/circleworld_proto/resonant_grandchild_composition_2026-05-14_synthetic_postfix_run.log`             | 1     | 10260944   | 2026-05-14T15:43:46+00:00 | log:1                                |
| `artifacts/runtime/outputs/circleworld_proto/resonant_grandchild_composition_2026-05-14_synthetic`                             | 30    | 9384240    | 2026-05-14T15:30:20+00:00 | audio:28, json:2                     |
| `artifacts/runtime/outputs/circleworld_proto/resonant_grandchild_composition_2026-05-14_naked_gpu_smoke`                       | 30    | 9347508    | 2026-05-14T15:36:07+00:00 | audio:28, json:2                     |
| `artifacts/runtime/outputs/circleworld_proto/training_run_2026-04-09_anchorheavy`                                              | 3     | 8997848    | 2026-04-09T21:35:19+00:00 | json:3                               |
| `artifacts/runtime/outputs/circleworld_proto/graduation_phase_native_bridge_2026-05-18_fourth_full_group_peak`                 | 250   | 8311647    | 2026-05-18T09:05:34+00:00 | audio:248, report:1, json:1          |
| `artifacts/runtime/outputs/circleworld_proto/training_run_2026-04-09_realanchor_focus`                                         | 2     | 7941982    | 2026-04-09T22:34:18+00:00 | json:2                               |
| `artifacts/runtime/outputs/circleworld_proto/nested_commitment_2026-04-11_full`                                                | 23    | 6453668    | 2026-04-11T11:10:25+00:00 | audio:20, json:3                     |
| `artifacts/runtime/outputs/circleworld_proto/nested_commitment_2026-04-11_penalized`                                           | 23    | 6452627    | 2026-04-11T11:10:44+00:00 | audio:20, json:3                     |
| `artifacts/runtime/outputs/circleworld_proto/resonant_child_retrieval_2026-05-14_naked_gpu_9100_9102_postfix`                  | 2     | 6103609    | 2026-05-14T15:48:21+00:00 | report:1, json:1                     |
| `artifacts/runtime/outputs/circleworld_proto/training_run_2026-04-09_realanchor_constrained`                                   | 2     | 5416276    | 2026-04-09T22:42:28+00:00 | json:2                               |
| `artifacts/runtime/outputs/circleworld_proto/training_run_2026-04-06_v2`                                                       | 3     | 5372853    | 2026-04-06T23:53:49+00:00 | json:3                               |
| `artifacts/runtime/outputs/circleworld_proto/benchmark_2026-04-09_realanchor_obj`                                              | 17    | 4814372    | 2026-04-09T22:09:47+00:00 | audio:15, report:1, json:1           |
| `artifacts/runtime/outputs/circleworld_proto/benchmark_2026-04-09_realanchor_constrained`                                      | 16    | 4814293    | 2026-04-09T22:43:54+00:00 | audio:15, json:1                     |
| `artifacts/runtime/outputs/circleworld_proto/benchmark_2026-04-09_realanchor_focus`                                            | 16    | 4813647    | 2026-04-09T22:34:42+00:00 | audio:15, json:1                     |
| `artifacts/runtime/outputs/circleworld_proto/benchmark_2026-04-09_realanchor_band`                                             | 16    | 4813623    | 2026-04-09T22:17:37+00:00 | audio:15, json:1                     |
| `artifacts/runtime/outputs/circleworld_proto/benchmark_2026-04-09_anchorheavy`                                                 | 16    | 4813449    | 2026-04-09T22:01:30+00:00 | audio:15, json:1                     |
| `artifacts/runtime/tmp/gradpack_local_sweep_ep8_12`                                                                            | 90    | 4627908    | 2026-04-06T13:01:41+00:00 | audio:72, json:18                    |
| `artifacts/runtime/outputs/circleworld_proto/training_run_2026-04-09_realanchor_band`                                          | 2     | 4599971    | 2026-04-09T22:17:11+00:00 | json:2                               |
| `artifacts/runtime/outputs/circleworld_proto/training_run_2026-04-09_realanchor_obj`                                           | 2     | 4218340    | 2026-04-09T22:09:02+00:00 | json:2                               |
| `artifacts/runtime/outputs/circleworld_proto/childworld_causality_killswitch_2026-05-11`                                       | 42    | 3232644    | 2026-05-11T20:56:46+00:00 | audio:37, json:3, log:2              |
| `artifacts/runtime/outputs/circleworld_proto/smoke_nested_native_multimode_2026-04-16`                                         | 12    | 3229039    | 2026-04-16T08:58:45+00:00 | audio:10, json:2                     |
| `artifacts/runtime/outputs/circleworld_proto/inference_2026-04-09_anchorheavy_ab`                                              | 19    | 3213838    | 2026-04-10T22:08:07+00:00 | audio:10, json:8, report:1           |
| `artifacts/runtime/outputs/circleworld_proto/inference_2026-04-09_realanchor_band_ab`                                          | 14    | 3210708    | 2026-04-10T22:08:07+00:00 | audio:10, json:3, report:1           |
| `artifacts/runtime/outputs/circleworld_proto/phase_native_audio_reentry_shear_probe_2026-05-17_fresh_seed20260518`             | 2     | 3148613    | 2026-05-17T18:30:56+00:00 | report:1, json:1                     |
| `artifacts/runtime/outputs/circleworld_proto/phase_native_audio_reentry_decorrelator_probe_2026-05-17_fresh_seed20260518`      | 2     | 2809016    | 2026-05-17T18:32:00+00:00 | report:1, json:1                     |
| `artifacts/runtime/logs/baseline_20k_steps.log`                                                                                | 1     | 2625722    | 2026-02-19T21:58:42+00:00 | log:1                                |
| `artifacts/runtime/tmp/gradpack_seed_sweep_ep10_1200_qkv_p035_m005`                                                            | 51    | 2576848    | 2026-04-06T13:12:47+00:00 | audio:40, json:11                    |
| `artifacts/bundles/RAFA_SOUND_PACK_V1`                                                                                         | 40    | 2561760    | 2026-04-04T17:00:22+00:00 | audio:40                             |
| `artifacts/runtime/outputs/circleworld_proto/phase_native_audio_reset_2026-05-17_fresh_seed20260518`                           | 41    | 2556414    | 2026-05-17T18:29:51+00:00 | audio:30, report:4, json:4, log:3    |
| `artifacts/runtime/tmp/gradpack_candidate_sweep`                                                                               | 46    | 2319658    | 2026-04-06T12:59:45+00:00 | audio:36, json:10                    |
| `artifacts/runtime/tmp/gradpack_g1_donor_sweep`                                                                                | 33    | 2054475    | 2026-04-06T13:13:49+00:00 | audio:32, json:1                     |
| `artifacts/runtime/outputs/circleworld_proto/resonant_child_retrieval_2026-05-14_grandchild_synthetic_postfix`                 | 2     | 2034642    | 2026-05-14T15:46:32+00:00 | report:1, json:1                     |
| `artifacts/runtime/outputs/circleworld_proto/resonant_child_retrieval_2026-05-14_grandchild_synthetic`                         | 2     | 1974641    | 2026-05-14T15:34:31+00:00 | report:1, json:1                     |
| `artifacts/runtime/outputs/circleworld_proto/resonant_child_retrieval_2026-05-14_naked_gpu_smoke`                              | 2     | 1961327    | 2026-05-14T15:36:47+00:00 | report:1, json:1                     |
| `artifacts/runtime/outputs/circleworld_proto/inference_2026-04-09_gradmix_10s`                                                 | 14    | 1931251    | 2026-04-09T20:18:02+00:00 | json:7, audio:6, report:1            |
| `artifacts/runtime/outputs/circleworld_proto/resonant_child_retrieval_2026-05-13_nested_smoke`                                 | 2     | 1854552    | 2026-05-14T03:58:06+00:00 | report:1, json:1                     |
| `artifacts/runtime/outputs/circleworld_proto/learned_structural_probe_2026-05-14_naked_gpu_holdout_9100`                       | 2     | 1657034    | 2026-05-14T17:21:47+00:00 | report:1, json:1                     |

## Regeneration

```powershell
python tools/audit_project_scripts.py
python tools/audit_project_usage.py
```
