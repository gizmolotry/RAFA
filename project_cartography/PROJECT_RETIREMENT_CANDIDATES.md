# Project Retirement Candidates

Generated: `2026-05-20`

## Scope

This artifact classifies source-like scripts for retirement triage only. It does not authorize deletion, renaming, moving, or source edits.

Ownership constraint honored: this pass creates/edits only `docs/architecture/PROJECT_RETIREMENT_CANDIDATES.md`.

## Evidence Read

- `docs/architecture/PROJECT_SCRIPT_INVENTORY.json`
- `docs/architecture/PROJECT_SCRIPT_INVENTORY.md`
- `docs/architecture/PROJECT_USAGE_MAP.json`
- `docs/architecture/PROJECT_USAGE_MAP.md`
- `docs/architecture/PROJECT_REDUNDANCY_HOTSPOTS.md`
- `docs/reports/RAFA_CODEBASE_CLEANUP_PASS_2026-05-20.md`
- `project_cartography/README.md`
- `docs/reports/RAFA_PROJECT_USAGE_AUDIT_2026-05-19.md`

## Classification Rules

| Category | Rule | Retirement posture |
| --- | --- | --- |
| `canonical` | Manifested/registered, imported dependency, or contract/test according to the usage map. | Do not delete. Treat as protected unless replaced through an explicit migration. |
| `active research` | Documented, report-linked, CLI-documented, or active unregistered research. | Do not delete during cleanup. Label lifecycle or consolidate helpers first. |
| `compatibility wrapper` | Launcher, scheduler, legacy entrypoint, or old-path adapter. | Do not delete until callers, docs, jobs, and path contracts are checked. |
| `superseded-review-needed` | Tracked low-reference/legacy script with weak static use evidence. | Candidate for manual review, not automatic deletion. |
| `retire-candidate-watchlist` | Untracked low-reference candidate with no imports/text refs in static map. | First inspection set for eventual retirement, after artifact/report provenance review. |

## Headline Counts

| Metric | Count |
| --- | ---: |
| Scripts classified | 235 |
| canonical | 101 |
| active research | 70 |
| compatibility wrapper | 12 |
| superseded-review-needed | 28 |
| retire-candidate-watchlist | 24 |

## Source Usage Counts

| Usage status from `PROJECT_USAGE_MAP.json` | Count |
| --- | ---: |
| `canonical_or_manifested` | 47 |
| `documented_cli` | 12 |
| `documented_reference` | 59 |
| `imported_dependency` | 46 |
| `low_reference_or_legacy` | 39 |
| `test_or_contract` | 8 |
| `untracked_low_reference_candidate` | 24 |

| Lifecycle from `PROJECT_SCRIPT_INVENTORY.json` | Count |
| --- | ---: |
| `active_unregistered_research` | 30 |
| `canonical_or_manifested` | 47 |
| `contract_or_test` | 16 |
| `documented_or_reported` | 90 |
| `legacy_or_tooling` | 26 |
| `unclassified` | 26 |

## Top Risky Deletions To Avoid

These scripts have the strongest static evidence of use or audit importance. They should not be deleted during a retirement pass.

| Script | Category | Evidence |
| --- | --- | --- |
| `lineages/04_positive_replacement/circleworld.py` | `canonical` | manifested; imported_by=29; docs_refs=239; text_refs=30; use_status=canonical_or_manifested; role=runtime_or_shared_module |
| `core/config.py` | `canonical` | imported_by=55; text_refs=1; use_status=imported_dependency; role=script_or_module |
| `rafa_math_tools.py` | `canonical` | manifested; imported_by=36; docs_refs=15; text_refs=7; use_status=canonical_or_manifested; role=runtime_or_shared_module |
| `runtimes/circleworld_proto/test_nested_commitment.py` | `canonical` | manifested; imported_by=13; docs_refs=96; text_refs=29; use_status=canonical_or_manifested; role=contract_test |
| `runtimes/circleworld_proto/train_circleworld_real_anchor.py` | `canonical` | manifested; imported_by=7; docs_refs=118; text_refs=22; use_status=canonical_or_manifested; role=trainer |
| `stft_utils.py` | `canonical` | manifested; imported_by=27; docs_refs=15; text_refs=6; use_status=canonical_or_manifested; role=runtime_or_shared_module |
| `runtimes/circleworld_proto/evaluate_circleworld.py` | `canonical` | imported_by=24; docs_refs=69; text_refs=15; use_status=imported_dependency; role=evaluator |
| `diffusion_utils.py` | `canonical` | manifested; imported_by=23; docs_refs=15; text_refs=7; use_status=canonical_or_manifested; role=runtime_or_shared_module |
| `inference_package/diffusion_utils.py` | `canonical` | manifested; imported_by=23; docs_refs=5; text_refs=7; use_status=canonical_or_manifested; role=runtime_or_shared_module |
| `core/dataset.py` | `canonical` | manifested; imported_by=21; docs_refs=8; text_refs=7; use_status=canonical_or_manifested; role=script_or_module |
| `model.py` | `canonical` | manifested; imported_by=12; docs_refs=36; text_refs=6; use_status=canonical_or_manifested; role=runtime_or_shared_module |
| `core/lib_blackwell.py` | `canonical` | manifested; imported_by=19; docs_refs=12; text_refs=4; use_status=canonical_or_manifested; role=script_or_module |
| `runtimes/circleworld_proto/train_circleworld.py` | `canonical` | manifested; imported_by=1; docs_refs=72; text_refs=17; use_status=canonical_or_manifested; role=trainer |
| `inference_package/lib_blackwell.py` | `canonical` | manifested; imported_by=19; docs_refs=7; text_refs=4; use_status=canonical_or_manifested; role=script_or_module |
| `diffusion_models.py` | `canonical` | manifested; imported_by=10; docs_refs=27; text_refs=5; use_status=canonical_or_manifested; role=runtime_or_shared_module |
| `runtimes/circleworld_proto/export_circleworld_audio.py` | `canonical` | imported_by=15; docs_refs=49; text_refs=11; use_status=imported_dependency; role=exporter |
| `lineages/04_positive_replacement/ablate_formalization.py` | `canonical` | manifested; imported_by=14; docs_refs=6; text_refs=5; use_status=canonical_or_manifested; role=script_or_module |
| `core/diffusion_models.py` | `canonical` | manifested; imported_by=10; docs_refs=9; text_refs=5; use_status=canonical_or_manifested; role=runtime_or_shared_module |
| `phase_native_ifs.py` | `canonical` | manifested; imported_by=6; docs_refs=21; text_refs=7; use_status=canonical_or_manifested; role=runtime_or_shared_module |
| `runtimes/circleworld_proto/benchmark_circleworld_real_anchor.py` | `canonical` | manifested; imported_by=9; docs_refs=9; text_refs=4; use_status=canonical_or_manifested; role=benchmark |
| `runtimes/circleworld_proto/benchmark_audio_continuity.py` | `canonical` | manifested; imported_by=8; docs_refs=7; text_refs=3; use_status=canonical_or_manifested; role=benchmark |
| `runtimes/circleworld_proto/test_resonant_attention_contract.py` | `active research` | docs_refs=67; text_refs=20; use_status=documented_reference; role=contract_test |
| `sample_diffusion.py` | `canonical` | manifested; imported_by=1; docs_refs=33; text_refs=8; use_status=canonical_or_manifested; role=exporter |
| `runtimes/circleworld_proto/score_phase_native_audio_reentry_guard.py` | `canonical` | manifested; imported_by=7; docs_refs=5; text_refs=3; use_status=canonical_or_manifested; role=scorer |
| `runtimes/circleworld_proto/test_learned_branch_law_child_ifs_contract.py` | `active research` | docs_refs=56; text_refs=21; use_status=documented_reference; role=contract_test |

## Immediate Review Sets

### Superseded Review Needed

Tracked scripts with low static usage. Review provenance, docs, jobs, and artifacts before any retirement decision.

| Script | Lane | Role | LOC | Evidence |
| --- | --- | --- | ---: | --- |
| `clip_audio_loader.py` | `root_legacy` | `script_or_module` | 47 | tracked=True; use_status=low_reference_or_legacy; imports=0; text_refs=0; docs_refs=0 |
| `research_track/infra/run_conditioned_depth_batch.py` | `research_track` | `experiment_runner` | 65 | tracked=True; use_status=low_reference_or_legacy; imports=0; text_refs=0; docs_refs=0 |
| `research_track/infra/run_conditioned_depth_job.py` | `research_track` | `experiment_runner` | 249 | tracked=True; use_status=low_reference_or_legacy; imports=0; text_refs=3; docs_refs=0 |
| `research_track/infra/run_hypercube_batch.py` | `research_track` | `experiment_runner` | 55 | tracked=True; use_status=low_reference_or_legacy; imports=0; text_refs=0; docs_refs=0 |
| `research_track/infra/run_logic_diag_batch.py` | `research_track` | `experiment_runner` | 55 | tracked=True; use_status=low_reference_or_legacy; imports=0; text_refs=0; docs_refs=0 |
| `runtimes/circleworld_proto/run_retrospective_grid.py` | `circleworld_proto` | `experiment_runner` | 251 | tracked=True; use_status=low_reference_or_legacy; imports=0; text_refs=0; docs_refs=0 |
| `tools/ablate_contract.py` | `tools` | `script_or_module` | 110 | tracked=True; use_status=low_reference_or_legacy; imports=0; text_refs=0; docs_refs=0 |
| `tools/acid_test.py` | `tools` | `script_or_module` | 102 | tracked=True; use_status=low_reference_or_legacy; imports=0; text_refs=0; docs_refs=0 |
| `tools/audit_clap.py` | `tools` | `auditor` | 20 | tracked=True; use_status=low_reference_or_legacy; imports=0; text_refs=0; docs_refs=0 |
| `tools/audit_hierarchy.py` | `tools` | `auditor` | 19 | tracked=True; use_status=low_reference_or_legacy; imports=0; text_refs=0; docs_refs=0 |
| `tools/audit_params.py` | `tools` | `auditor` | 34 | tracked=True; use_status=low_reference_or_legacy; imports=0; text_refs=0; docs_refs=0 |
| `tools/causal_audit.py` | `tools` | `script_or_module` | 64 | tracked=True; use_status=low_reference_or_legacy; imports=0; text_refs=0; docs_refs=0 |
| `tools/debug_nan.py` | `tools` | `script_or_module` | 57 | tracked=True; use_status=low_reference_or_legacy; imports=0; text_refs=0; docs_refs=0 |
| `tools/hard_ablation_audit.py` | `tools` | `script_or_module` | 83 | tracked=True; use_status=low_reference_or_legacy; imports=0; text_refs=0; docs_refs=0 |
| `tools/profile_rafa.py` | `tools` | `script_or_module` | 33 | tracked=True; use_status=low_reference_or_legacy; imports=0; text_refs=0; docs_refs=0 |
| `tools/profile_step.py` | `tools` | `script_or_module` | 49 | tracked=True; use_status=low_reference_or_legacy; imports=0; text_refs=0; docs_refs=0 |
| `tools/rafa_leaderboard.py` | `tools` | `script_or_module` | 62 | tracked=True; use_status=low_reference_or_legacy; imports=0; text_refs=0; docs_refs=0 |
| `tools/run_ablations.py` | `tools` | `experiment_runner` | 114 | tracked=True; use_status=low_reference_or_legacy; imports=0; text_refs=0; docs_refs=0 |
| `tools/run_hypercube.py` | `tools` | `experiment_runner` | 120 | tracked=True; use_status=low_reference_or_legacy; imports=0; text_refs=0; docs_refs=0 |
| `tools/run_learning_curve_resume.py` | `tools` | `experiment_runner` | 152 | tracked=True; use_status=low_reference_or_legacy; imports=0; text_refs=1; docs_refs=0 |
| `tools/scrape_freewavesamples.py` | `tools` | `script_or_module` | 85 | tracked=True; use_status=low_reference_or_legacy; imports=0; text_refs=1; docs_refs=0 |
| `tools/scrape_high_volume.py` | `tools` | `script_or_module` | 99 | tracked=True; use_status=low_reference_or_legacy; imports=0; text_refs=0; docs_refs=0 |
| `tools/scrape_mixkit.py` | `tools` | `script_or_module` | 94 | tracked=True; use_status=low_reference_or_legacy; imports=0; text_refs=1; docs_refs=0 |
| `tools/scrape_soundbible.py` | `tools` | `script_or_module` | 78 | tracked=True; use_status=low_reference_or_legacy; imports=0; text_refs=1; docs_refs=0 |
| `tools/scrape_wav.py` | `tools` | `script_or_module` | 54 | tracked=True; use_status=low_reference_or_legacy; imports=0; text_refs=0; docs_refs=0 |
| `tools/scrape_wavsource.py` | `tools` | `script_or_module` | 79 | tracked=True; use_status=low_reference_or_legacy; imports=0; text_refs=1; docs_refs=0 |
| `tools/tests_diffusion_shell.py` | `tools` | `script_or_module` | 38 | tracked=True; use_status=low_reference_or_legacy; imports=0; text_refs=0; docs_refs=0 |
| `tools/total_recall.py` | `tools` | `script_or_module` | 38 | tracked=True; use_status=low_reference_or_legacy; imports=0; text_refs=0; docs_refs=0 |

### Retire Candidate Watchlist

Untracked low-reference candidates. These are the first scripts to inspect, but still not deletion-authorized.

| Script | Lane | Role | LOC | Evidence |
| --- | --- | --- | ---: | --- |
| `runtimes/circleworld_proto/analyze_arc_q_control_claim.py` | `circleworld_proto` | `analysis` | 157 | tracked=False; use_status=untracked_low_reference_candidate; imports=0; text_refs=0; docs_refs=0 |
| `runtimes/circleworld_proto/analyze_dense_signature_failure_modes.py` | `circleworld_proto` | `analysis` | 415 | tracked=False; use_status=untracked_low_reference_candidate; imports=0; text_refs=0; docs_refs=0 |
| `runtimes/circleworld_proto/analyze_learned_signature_case_failures.py` | `circleworld_proto` | `analysis` | 157 | tracked=False; use_status=untracked_low_reference_candidate; imports=0; text_refs=0; docs_refs=0 |
| `runtimes/circleworld_proto/assemble_childworld_cycle_report.py` | `circleworld_proto` | `report_assembler` | 616 | tracked=False; use_status=untracked_low_reference_candidate; imports=0; text_refs=0; docs_refs=0 |
| `runtimes/circleworld_proto/assemble_childworld_mechanism_audit.py` | `circleworld_proto` | `report_assembler` | 622 | tracked=False; use_status=untracked_low_reference_candidate; imports=0; text_refs=0; docs_refs=0 |
| `runtimes/circleworld_proto/assemble_circleworld_experiment_ledger.py` | `circleworld_proto` | `report_assembler` | 556 | tracked=False; use_status=untracked_low_reference_candidate; imports=0; text_refs=0; docs_refs=0 |
| `runtimes/circleworld_proto/assemble_circleworld_scoreboard.py` | `circleworld_proto` | `report_assembler` | 322 | tracked=False; use_status=untracked_low_reference_candidate; imports=0; text_refs=0; docs_refs=0 |
| `runtimes/circleworld_proto/assemble_phase_token_repair_report.py` | `circleworld_proto` | `report_assembler` | 438 | tracked=False; use_status=untracked_low_reference_candidate; imports=0; text_refs=0; docs_refs=0 |
| `runtimes/circleworld_proto/assemble_static_child_conversion_report.py` | `circleworld_proto` | `report_assembler` | 404 | tracked=False; use_status=untracked_low_reference_candidate; imports=0; text_refs=0; docs_refs=0 |
| `runtimes/circleworld_proto/audit_circleworld_seed_determinism.py` | `circleworld_proto` | `auditor` | 579 | tracked=False; use_status=untracked_low_reference_candidate; imports=0; text_refs=0; docs_refs=0 |
| `runtimes/circleworld_proto/audit_relational_signature_contract.py` | `circleworld_proto` | `auditor` | 188 | tracked=False; use_status=untracked_low_reference_candidate; imports=0; text_refs=0; docs_refs=0 |
| `runtimes/circleworld_proto/build_childworld_mechanism_dataset.py` | `circleworld_proto` | `builder` | 852 | tracked=False; use_status=untracked_low_reference_candidate; imports=0; text_refs=0; docs_refs=0 |
| `runtimes/circleworld_proto/classify_internal_phase_law_candidate_classes.py` | `circleworld_proto` | `analysis` | 185 | tracked=False; use_status=untracked_low_reference_candidate; imports=0; text_refs=0; docs_refs=0 |
| `runtimes/circleworld_proto/compare_audio_continuation_methods.py` | `circleworld_proto` | `comparator` | 253 | tracked=False; use_status=untracked_low_reference_candidate; imports=0; text_refs=0; docs_refs=0 |
| `runtimes/circleworld_proto/compare_learned_signature_scouts.py` | `circleworld_proto` | `comparator` | 354 | tracked=False; use_status=untracked_low_reference_candidate; imports=0; text_refs=0; docs_refs=0 |
| `runtimes/circleworld_proto/compare_signature_operator_tail_probes.py` | `circleworld_proto` | `comparator` | 166 | tracked=False; use_status=untracked_low_reference_candidate; imports=0; text_refs=0; docs_refs=0 |
| `runtimes/circleworld_proto/evaluate_signature_future_law_probe.py` | `circleworld_proto` | `evaluator` | 674 | tracked=False; use_status=untracked_low_reference_candidate; imports=0; text_refs=0; docs_refs=0 |
| `runtimes/circleworld_proto/run_claim_isolation_suite.py` | `circleworld_proto` | `experiment_runner` | 621 | tracked=False; use_status=untracked_low_reference_candidate; imports=0; text_refs=0; docs_refs=0 |
| `runtimes/circleworld_proto/run_learned_branch_law_assay.py` | `circleworld_proto` | `experiment_runner` | 571 | tracked=False; use_status=untracked_low_reference_candidate; imports=0; text_refs=0; docs_refs=0 |
| `runtimes/circleworld_proto/summarize_learned_signature_frontier.py` | `circleworld_proto` | `report_assembler` | 419 | tracked=False; use_status=untracked_low_reference_candidate; imports=0; text_refs=0; docs_refs=0 |
| `runtimes/circleworld_proto/sweep_childsurvival_replay.py` | `circleworld_proto` | `sweep` | 192 | tracked=False; use_status=untracked_low_reference_candidate; imports=0; text_refs=0; docs_refs=0 |
| `runtimes/circleworld_proto/sweep_parentmix_replay.py` | `circleworld_proto` | `sweep` | 161 | tracked=False; use_status=untracked_low_reference_candidate; imports=0; text_refs=0; docs_refs=0 |
| `runtimes/circleworld_proto/train_childworld_mechanism_classifier.py` | `circleworld_proto` | `trainer` | 261 | tracked=False; use_status=untracked_low_reference_candidate; imports=0; text_refs=0; docs_refs=0 |
| `runtimes/circleworld_proto/validate_tokenburst_tracks.py` | `circleworld_proto` | `auditor` | 178 | tracked=False; use_status=untracked_low_reference_candidate; imports=0; text_refs=0; docs_refs=0 |

## Full Script Classification

### canonical

| Script | Lane | Role | Use status | Lifecycle | Evidence |
| --- | --- | --- | --- | --- | --- |
| `core/blackwell_model.py` | `core_shared` | `script_or_module` | `imported_dependency` | `unclassified` | tracked=yes; imported_by=2; text_refs=2; loc=95 |
| `core/config.py` | `core_shared` | `script_or_module` | `imported_dependency` | `unclassified` | tracked=yes; imported_by=55; text_refs=1; loc=11 |
| `core/dataset.py` | `core_shared` | `script_or_module` | `canonical_or_manifested` | `canonical_or_manifested` | tracked=yes; manifested=yes; imported_by=21; text_refs=7; docs_refs=8; loc=269 |
| `core/diffusion_models.py` | `core_shared` | `runtime_or_shared_module` | `canonical_or_manifested` | `canonical_or_manifested` | tracked=yes; manifested=yes; imported_by=10; text_refs=5; docs_refs=9; loc=58 |
| `core/lib_blackwell.py` | `core_shared` | `script_or_module` | `canonical_or_manifested` | `canonical_or_manifested` | tracked=yes; manifested=yes; imported_by=19; text_refs=4; docs_refs=12; loc=124 |
| `core/triton_deq.py` | `core_shared` | `script_or_module` | `imported_dependency` | `documented_or_reported` | tracked=yes; imported_by=1; text_refs=4; docs_refs=3; loc=164 |
| `core/triton_kernels.py` | `core_shared` | `script_or_module` | `canonical_or_manifested` | `canonical_or_manifested` | tracked=yes; manifested=yes; imported_by=2; text_refs=6; docs_refs=8; loc=148 |
| `diffusion_models.py` | `root_legacy` | `runtime_or_shared_module` | `canonical_or_manifested` | `canonical_or_manifested` | tracked=yes; manifested=yes; imported_by=10; text_refs=5; docs_refs=27; loc=249 |
| `diffusion_utils.py` | `root_legacy` | `runtime_or_shared_module` | `canonical_or_manifested` | `canonical_or_manifested` | tracked=yes; manifested=yes; imported_by=23; text_refs=7; docs_refs=15; loc=33 |
| `hf_local.py` | `root_legacy` | `script_or_module` | `canonical_or_manifested` | `canonical_or_manifested` | tracked=yes; manifested=yes; imported_by=5; text_refs=1; docs_refs=3; loc=65 |
| `inference_package/diffusion_utils.py` | `inference_package` | `runtime_or_shared_module` | `canonical_or_manifested` | `canonical_or_manifested` | tracked=yes; manifested=yes; imported_by=23; text_refs=7; docs_refs=5; loc=33 |
| `inference_package/lib_blackwell.py` | `inference_package` | `script_or_module` | `canonical_or_manifested` | `canonical_or_manifested` | tracked=yes; manifested=yes; imported_by=19; text_refs=4; docs_refs=7; loc=124 |
| `inference_package/triton_kernels.py` | `inference_package` | `script_or_module` | `canonical_or_manifested` | `canonical_or_manifested` | tracked=yes; manifested=yes; imported_by=2; text_refs=6; docs_refs=4; loc=148 |
| `lineages/01_parent_graduation/train.py` | `parent_graduation` | `script_or_module` | `imported_dependency` | `unclassified` | tracked=yes; imported_by=1; loc=303 |
| `lineages/04_positive_replacement/ablate_formalization.py` | `positive_replacement` | `script_or_module` | `canonical_or_manifested` | `canonical_or_manifested` | tracked=yes; manifested=yes; imported_by=14; text_refs=5; docs_refs=6; loc=272 |
| `lineages/04_positive_replacement/circleworld.py` | `positive_replacement` | `runtime_or_shared_module` | `canonical_or_manifested` | `canonical_or_manifested` | tracked=yes; manifested=yes; imported_by=29; text_refs=30; docs_refs=239; loc=3140 |
| `lineages/04_positive_replacement/rafa_relational_signature.py` | `positive_replacement` | `script_or_module` | `imported_dependency` | `documented_or_reported` | tracked=no; imported_by=8; text_refs=5; docs_refs=13; loc=814 |
| `lineages/04_positive_replacement/rafa_relational_signature_learning.py` | `positive_replacement` | `script_or_module` | `imported_dependency` | `unclassified` | tracked=no; imported_by=3; loc=291 |
| `model.py` | `root_legacy` | `runtime_or_shared_module` | `canonical_or_manifested` | `canonical_or_manifested` | tracked=yes; manifested=yes; imported_by=12; text_refs=6; docs_refs=36; loc=568 |
| `phase_native_ifs.py` | `root_legacy` | `runtime_or_shared_module` | `canonical_or_manifested` | `canonical_or_manifested` | tracked=yes; manifested=yes; imported_by=6; text_refs=7; docs_refs=21; loc=248 |
| `rafa_clutch_transformer_upgraded.py` | `root_legacy` | `script_or_module` | `canonical_or_manifested` | `canonical_or_manifested` | tracked=yes; manifested=yes; imported_by=3; text_refs=4; docs_refs=9; loc=99 |
| `rafa_math_tools.py` | `root_legacy` | `runtime_or_shared_module` | `canonical_or_manifested` | `canonical_or_manifested` | tracked=yes; manifested=yes; imported_by=36; text_refs=7; docs_refs=15; loc=350 |
| `research_track/infra/aggregate_hypercube_results.py` | `research_track` | `script_or_module` | `imported_dependency` | `unclassified` | tracked=yes; imported_by=2; loc=111 |
| `research_track/infra/aggregate_logic_diag_results.py` | `research_track` | `script_or_module` | `imported_dependency` | `unclassified` | tracked=yes; imported_by=1; loc=93 |
| `research_track/infra/generate_conditioned_depth_jobs.py` | `research_track` | `script_or_module` | `imported_dependency` | `unclassified` | tracked=yes; imported_by=1; loc=87 |
| `research_track/infra/generate_hypercube_jobs.py` | `research_track` | `script_or_module` | `imported_dependency` | `unclassified` | tracked=yes; imported_by=2; text_refs=1; loc=187 |
| `research_track/infra/generate_logic_diag_jobs.py` | `research_track` | `script_or_module` | `imported_dependency` | `unclassified` | tracked=yes; imported_by=1; loc=120 |
| `research_track/infra/run_hypercube_job.py` | `research_track` | `experiment_runner` | `imported_dependency` | `unclassified` | tracked=yes; imported_by=2; text_refs=1; loc=337 |
| `research_track/infra/run_logic_diag_job.py` | `research_track` | `experiment_runner` | `imported_dependency` | `unclassified` | tracked=yes; imported_by=1; text_refs=1; loc=154 |
| `research_track/infra/tests/test_generate_conditioned_depth_jobs.py` | `research_track` | `contract_test` | `test_or_contract` | `contract_or_test` | tracked=yes; loc=34 |
| `research_track/infra/tests/test_generate_hypercube_jobs.py` | `research_track` | `contract_test` | `test_or_contract` | `contract_or_test` | tracked=yes; loc=49 |
| `research_track/infra/tests/test_generate_logic_diag_jobs.py` | `research_track` | `contract_test` | `test_or_contract` | `contract_or_test` | tracked=yes; loc=29 |
| `runtimes/circleworld_proto/assemble_phase_native_audio_route_policy_comparison.py` | `circleworld_proto` | `report_assembler` | `canonical_or_manifested` | `canonical_or_manifested` | tracked=no; manifested=yes; text_refs=2; docs_refs=4; loc=133 |
| `runtimes/circleworld_proto/assemble_phase_native_audio_shared_battlefield.py` | `circleworld_proto` | `report_assembler` | `canonical_or_manifested` | `canonical_or_manifested` | tracked=no; manifested=yes; text_refs=2; docs_refs=4; loc=203 |
| `runtimes/circleworld_proto/audit_phase_native_audio_route_selector_contract.py` | `circleworld_proto` | `auditor` | `canonical_or_manifested` | `canonical_or_manifested` | tracked=no; manifested=yes; text_refs=1; docs_refs=2; loc=332 |
| `runtimes/circleworld_proto/benchmark_audio_continuation.py` | `circleworld_proto` | `benchmark` | `imported_dependency` | `active_unregistered_research` | tracked=no; imported_by=11; text_refs=1; loc=918 |
| `runtimes/circleworld_proto/benchmark_audio_continuity.py` | `circleworld_proto` | `benchmark` | `canonical_or_manifested` | `canonical_or_manifested` | tracked=yes; manifested=yes; imported_by=8; text_refs=3; docs_refs=7; loc=589 |
| `runtimes/circleworld_proto/benchmark_circleworld_real_anchor.py` | `circleworld_proto` | `benchmark` | `canonical_or_manifested` | `canonical_or_manifested` | tracked=yes; manifested=yes; imported_by=9; text_refs=4; docs_refs=9; loc=274 |
| `runtimes/circleworld_proto/build_law_token_library.py` | `circleworld_proto` | `builder` | `imported_dependency` | `documented_or_reported` | tracked=no; imported_by=7; text_refs=7; docs_refs=26; loc=306 |
| `runtimes/circleworld_proto/build_shadow_branch_law_table.py` | `circleworld_proto` | `builder` | `imported_dependency` | `documented_or_reported` | tracked=no; imported_by=6; text_refs=2; docs_refs=2; loc=191 |
| `runtimes/circleworld_proto/causal_operator_selector.py` | `circleworld_proto` | `runtime_or_shared_module` | `imported_dependency` | `documented_or_reported` | tracked=no; imported_by=4; text_refs=3; docs_refs=7; loc=66 |
| `runtimes/circleworld_proto/common_io.py` | `circleworld_proto` | `script_or_module` | `imported_dependency` | `documented_or_reported` | tracked=no; imported_by=10; text_refs=3; docs_refs=9; loc=112 |
| `runtimes/circleworld_proto/evaluate_circleworld.py` | `circleworld_proto` | `evaluator` | `imported_dependency` | `documented_or_reported` | tracked=yes; imported_by=24; text_refs=15; docs_refs=69; loc=421 |
| `runtimes/circleworld_proto/evaluate_dense_signature_claim.py` | `circleworld_proto` | `evaluator` | `imported_dependency` | `documented_or_reported` | tracked=no; imported_by=1; text_refs=1; docs_refs=4; loc=551 |
| `runtimes/circleworld_proto/evaluate_relational_metamers.py` | `circleworld_proto` | `evaluator` | `imported_dependency` | `active_unregistered_research` | tracked=no; imported_by=3; loc=452 |
| `runtimes/circleworld_proto/evaluate_shadow_branch_law_calibration.py` | `circleworld_proto` | `evaluator` | `imported_dependency` | `documented_or_reported` | tracked=no; imported_by=6; text_refs=2; docs_refs=5; loc=367 |
| `runtimes/circleworld_proto/evaluate_signature_operator_tail_probe.py` | `circleworld_proto` | `evaluator` | `imported_dependency` | `active_unregistered_research` | tracked=no; imported_by=2; loc=637 |
| `runtimes/circleworld_proto/evaluate_signature_tail_incremental_usefulness.py` | `circleworld_proto` | `evaluator` | `imported_dependency` | `active_unregistered_research` | tracked=no; imported_by=1; loc=907 |
| `runtimes/circleworld_proto/export_circleworld_audio.py` | `circleworld_proto` | `exporter` | `imported_dependency` | `documented_or_reported` | tracked=yes; imported_by=15; text_refs=11; docs_refs=49; loc=321 |
| `runtimes/circleworld_proto/package_phase_native_audio_route_selector_profile.py` | `circleworld_proto` | `packager` | `canonical_or_manifested` | `canonical_or_manifested` | tracked=no; manifested=yes; text_refs=1; docs_refs=2; loc=284 |
| `runtimes/circleworld_proto/phase_native_audio_operators.py` | `circleworld_proto` | `runtime_or_shared_module` | `imported_dependency` | `documented_or_reported` | tracked=no; imported_by=3; text_refs=3; docs_refs=5; loc=259 |
| `runtimes/circleworld_proto/resonant_law_objects.py` | `circleworld_proto` | `runtime_or_shared_module` | `imported_dependency` | `documented_or_reported` | tracked=no; imported_by=10; text_refs=4; docs_refs=16; loc=781 |
| `runtimes/circleworld_proto/run_ablation.py` | `circleworld_proto` | `experiment_runner` | `canonical_or_manifested` | `canonical_or_manifested` | tracked=yes; manifested=yes; text_refs=3; docs_refs=5; loc=26 |
| `runtimes/circleworld_proto/run_audio_circle_delta_probe.py` | `circleworld_proto` | `experiment_runner` | `imported_dependency` | `documented_or_reported` | tracked=no; imported_by=2; text_refs=3; docs_refs=6; loc=642 |
| `runtimes/circleworld_proto/run_audio_delta_mechanism_probe.py` | `circleworld_proto` | `experiment_runner` | `imported_dependency` | `documented_or_reported` | tracked=no; imported_by=6; text_refs=6; docs_refs=9; loc=725 |
| `runtimes/circleworld_proto/run_audio_electric_motor_holdout.py` | `circleworld_proto` | `experiment_runner` | `imported_dependency` | `documented_or_reported` | tracked=no; imported_by=1; text_refs=1; docs_refs=2; loc=903 |
| `runtimes/circleworld_proto/run_causal_operator_selector_probe.py` | `circleworld_proto` | `experiment_runner` | `imported_dependency` | `documented_or_reported` | tracked=no; imported_by=1; text_refs=4; docs_refs=9; loc=738 |
| `runtimes/circleworld_proto/run_child_writeback_ablation.py` | `circleworld_proto` | `experiment_runner` | `imported_dependency` | `documented_or_reported` | tracked=no; imported_by=1; text_refs=4; docs_refs=8; loc=202 |
| `runtimes/circleworld_proto/run_childworld_experiment.py` | `circleworld_proto` | `experiment_runner` | `imported_dependency` | `documented_or_reported` | tracked=no; imported_by=1; text_refs=11; docs_refs=47; loc=3112 |
| `runtimes/circleworld_proto/run_circleworld_operator_block.py` | `circleworld_proto` | `experiment_runner` | `canonical_or_manifested` | `canonical_or_manifested` | tracked=no; manifested=yes; imported_by=1; text_refs=4; docs_refs=7; loc=356 |
| `runtimes/circleworld_proto/run_contrastive_structural_embedding_probe.py` | `circleworld_proto` | `experiment_runner` | `imported_dependency` | `documented_or_reported` | tracked=no; imported_by=4; text_refs=1; docs_refs=2; loc=601 |
| `runtimes/circleworld_proto/run_graduation_phase_native_bridge.py` | `circleworld_proto` | `experiment_runner` | `canonical_or_manifested` | `canonical_or_manifested` | tracked=no; manifested=yes; text_refs=4; docs_refs=7; loc=475 |
| `runtimes/circleworld_proto/run_learned_gated_manychild_runtime_sandbox.py` | `circleworld_proto` | `experiment_runner` | `imported_dependency` | `documented_or_reported` | tracked=no; imported_by=1; text_refs=9; docs_refs=42; loc=1303 |
| `runtimes/circleworld_proto/run_learned_gated_multistep_operator_sandbox.py` | `circleworld_proto` | `experiment_runner` | `imported_dependency` | `documented_or_reported` | tracked=no; imported_by=4; text_refs=2; docs_refs=5; loc=489 |
| `runtimes/circleworld_proto/run_learned_gated_writeback_sandbox.py` | `circleworld_proto` | `experiment_runner` | `imported_dependency` | `documented_or_reported` | tracked=no; imported_by=4; text_refs=3; docs_refs=10; loc=327 |
| `runtimes/circleworld_proto/run_phase_native_audio_reset_suite.py` | `circleworld_proto` | `experiment_runner` | `canonical_or_manifested` | `canonical_or_manifested` | tracked=no; manifested=yes; text_refs=5; docs_refs=11; loc=496 |
| `runtimes/circleworld_proto/run_phase_native_audio_selected_route.py` | `circleworld_proto` | `experiment_runner` | `canonical_or_manifested` | `canonical_or_manifested` | tracked=no; manifested=yes; imported_by=1; text_refs=4; docs_refs=7; loc=301 |
| `runtimes/circleworld_proto/run_q_basis_ablation.py` | `circleworld_proto` | `experiment_runner` | `imported_dependency` | `active_unregistered_research` | tracked=no; imported_by=1; loc=449 |
| `runtimes/circleworld_proto/run_resonant_operator_causality_assay.py` | `circleworld_proto` | `experiment_runner` | `imported_dependency` | `documented_or_reported` | tracked=no; imported_by=5; text_refs=8; docs_refs=25; loc=1005 |
| `runtimes/circleworld_proto/run_substrate_ablation.py` | `circleworld_proto` | `experiment_runner` | `imported_dependency` | `documented_or_reported` | tracked=no; imported_by=1; text_refs=1; docs_refs=2; loc=525 |
| `runtimes/circleworld_proto/score_phase_native_audio_prefix_router_scout.py` | `circleworld_proto` | `scorer` | `canonical_or_manifested` | `canonical_or_manifested` | tracked=no; manifested=yes; imported_by=4; text_refs=3; docs_refs=5; loc=414 |
| `runtimes/circleworld_proto/score_phase_native_audio_reentry_guard.py` | `circleworld_proto` | `scorer` | `canonical_or_manifested` | `canonical_or_manifested` | tracked=no; manifested=yes; imported_by=7; text_refs=3; docs_refs=5; loc=372 |
| `runtimes/circleworld_proto/score_phase_native_audio_reentry_metric_audit.py` | `circleworld_proto` | `scorer` | `canonical_or_manifested` | `canonical_or_manifested` | tracked=no; manifested=yes; imported_by=1; text_refs=1; docs_refs=2; loc=325 |
| `runtimes/circleworld_proto/score_phase_native_audio_reentry_oracle.py` | `circleworld_proto` | `scorer` | `canonical_or_manifested` | `canonical_or_manifested` | tracked=no; manifested=yes; text_refs=3; docs_refs=5; loc=281 |
| `runtimes/circleworld_proto/score_phase_native_audio_route_transfer.py` | `circleworld_proto` | `scorer` | `canonical_or_manifested` | `canonical_or_manifested` | tracked=no; manifested=yes; text_refs=1; docs_refs=2; loc=163 |
| `runtimes/circleworld_proto/score_phase_native_audio_target_replay_oracle.py` | `circleworld_proto` | `scorer` | `canonical_or_manifested` | `canonical_or_manifested` | tracked=no; manifested=yes; imported_by=4; text_refs=3; docs_refs=5; loc=366 |
| `runtimes/circleworld_proto/test_circleworld_operator_block_contract.py` | `circleworld_proto` | `contract_test` | `test_or_contract` | `contract_or_test` | tracked=no; loc=62 |
| `runtimes/circleworld_proto/test_nested_commitment.py` | `circleworld_proto` | `contract_test` | `canonical_or_manifested` | `canonical_or_manifested` | tracked=yes; manifested=yes; imported_by=13; text_refs=29; docs_refs=96; loc=3627 |
| `runtimes/circleworld_proto/test_relational_signature_learning_contract.py` | `circleworld_proto` | `contract_test` | `test_or_contract` | `contract_or_test` | tracked=no; loc=236 |
| `runtimes/circleworld_proto/train_circleworld.py` | `circleworld_proto` | `trainer` | `canonical_or_manifested` | `canonical_or_manifested` | tracked=yes; manifested=yes; imported_by=1; text_refs=17; docs_refs=72; loc=1535 |
| `runtimes/circleworld_proto/train_circleworld_real_anchor.py` | `circleworld_proto` | `trainer` | `canonical_or_manifested` | `canonical_or_manifested` | tracked=yes; manifested=yes; imported_by=7; text_refs=22; docs_refs=118; loc=3632 |
| `runtimes/circleworld_proto/train_learned_signature_scout.py` | `circleworld_proto` | `trainer` | `imported_dependency` | `documented_or_reported` | tracked=no; imported_by=4; text_refs=1; docs_refs=2; loc=1516 |
| `runtimes/circleworld_proto/train_phase_native_audio_family_route_policy.py` | `circleworld_proto` | `trainer` | `canonical_or_manifested` | `canonical_or_manifested` | tracked=no; manifested=yes; text_refs=4; docs_refs=7; loc=321 |
| `runtimes/circleworld_proto/train_phase_native_audio_objective_route_policy.py` | `circleworld_proto` | `trainer` | `canonical_or_manifested` | `canonical_or_manifested` | tracked=no; manifested=yes; text_refs=4; docs_refs=7; loc=345 |
| `runtimes/circleworld_proto/train_phase_native_audio_route_policy.py` | `circleworld_proto` | `trainer` | `canonical_or_manifested` | `canonical_or_manifested` | tracked=no; manifested=yes; text_refs=4; docs_refs=7; loc=316 |
| `runtimes/circleworld_proto/train_shadow_branch_law_from_table.py` | `circleworld_proto` | `trainer` | `imported_dependency` | `documented_or_reported` | tracked=no; imported_by=4; text_refs=2; docs_refs=3; loc=340 |
| `runtimes/diffusion_parent_v3/sample_prompt.py` | `diffusion_parent_v3` | `exporter` | `canonical_or_manifested` | `canonical_or_manifested` | tracked=yes; manifested=yes; text_refs=3; docs_refs=5; loc=27 |
| `runtimes/stage4_blackwell_14/export_graduation.py` | `stage4_blackwell_14` | `exporter` | `canonical_or_manifested` | `canonical_or_manifested` | tracked=yes; manifested=yes; text_refs=7; docs_refs=20; loc=29 |
| `runtimes/stage4_blackwell_16/export_graduation.py` | `stage4_blackwell_16` | `exporter` | `canonical_or_manifested` | `canonical_or_manifested` | tracked=yes; manifested=yes; text_refs=7; docs_refs=17; loc=18 |
| `sample_diffusion.py` | `root_legacy` | `exporter` | `canonical_or_manifested` | `canonical_or_manifested` | tracked=yes; manifested=yes; imported_by=1; text_refs=8; docs_refs=33; loc=277 |
| `stft_utils.py` | `root_legacy` | `runtime_or_shared_module` | `canonical_or_manifested` | `canonical_or_manifested` | tracked=yes; manifested=yes; imported_by=27; text_refs=6; docs_refs=15; loc=55 |
| `tools/audio_debug.py` | `tools` | `script_or_module` | `imported_dependency` | `legacy_or_tooling` | tracked=yes; imported_by=1; loc=101 |
| `tools/audit_project_scripts.py` | `tools` | `auditor` | `imported_dependency` | `documented_or_reported` | tracked=no; imported_by=1; text_refs=5; docs_refs=12; loc=990 |
| `tools/export_audio.py` | `tools` | `exporter` | `canonical_or_manifested` | `canonical_or_manifested` | tracked=yes; manifested=yes; imported_by=1; text_refs=3; docs_refs=16; loc=68 |
| `tools/export_audio_compat14.py` | `tools` | `exporter` | `canonical_or_manifested` | `canonical_or_manifested` | tracked=yes; manifested=yes; imported_by=3; text_refs=4; docs_refs=14; loc=215 |
| `tools/export_graduation_pack_restore.py` | `tools` | `exporter` | `canonical_or_manifested` | `canonical_or_manifested` | tracked=yes; manifested=yes; text_refs=4; docs_refs=9; loc=68 |
| `tools/invariance_utils.py` | `tools` | `script_or_module` | `imported_dependency` | `legacy_or_tooling` | tracked=yes; imported_by=1; loc=33 |
| `tools/scrape_core.py` | `tools` | `script_or_module` | `imported_dependency` | `legacy_or_tooling` | tracked=yes; imported_by=4; loc=283 |
| `tools/test_relational_logic.py` | `tools` | `contract_test` | `test_or_contract` | `contract_or_test` | tracked=yes; text_refs=2; loc=207 |
| `tools/test_triton.py` | `tools` | `contract_test` | `test_or_contract` | `contract_or_test` | tracked=yes; text_refs=1; docs_refs=1; loc=97 |
| `tools/test_triton_deq.py` | `tools` | `contract_test` | `test_or_contract` | `contract_or_test` | tracked=yes; text_refs=1; docs_refs=1; loc=42 |

### active research

| Script | Lane | Role | Use status | Lifecycle | Evidence |
| --- | --- | --- | --- | --- | --- |
| `lineages/02_local_ablations/train_diffusion.py` | `local_ablations` | `trainer` | `documented_reference` | `documented_or_reported` | tracked=yes; text_refs=7; docs_refs=2; loc=162 |
| `lineages/03_subtractive_fork/train_stage4_joint.py` | `subtractive_fork` | `trainer` | `documented_reference` | `documented_or_reported` | tracked=yes; text_refs=2; docs_refs=3; loc=110 |
| `lineages/04_positive_replacement/semantic_projector.py` | `positive_replacement` | `script_or_module` | `documented_reference` | `documented_or_reported` | tracked=no; text_refs=6; docs_refs=16; loc=278 |
| `runtimes/circleworld_proto/analyze_internal_phase_law_family_masks.py` | `circleworld_proto` | `analysis` | `documented_reference` | `documented_or_reported` | tracked=no; text_refs=1; docs_refs=5; loc=435 |
| `runtimes/circleworld_proto/analyze_internal_phase_law_support_router_oracle.py` | `circleworld_proto` | `analysis` | `documented_reference` | `documented_or_reported` | tracked=no; text_refs=1; docs_refs=5; loc=320 |
| `runtimes/circleworld_proto/analyze_internal_phase_law_target_flips.py` | `circleworld_proto` | `analysis` | `documented_reference` | `documented_or_reported` | tracked=no; text_refs=1; docs_refs=5; loc=353 |
| `runtimes/circleworld_proto/assemble_rafa_claim_evidence.py` | `circleworld_proto` | `report_assembler` | `documented_reference` | `documented_or_reported` | tracked=no; text_refs=4; docs_refs=8; loc=6420 |
| `runtimes/circleworld_proto/audit_circleworld_dag.py` | `circleworld_proto` | `auditor` | `documented_reference` | `documented_or_reported` | tracked=no; text_refs=2; docs_refs=3; loc=191 |
| `runtimes/circleworld_proto/audit_seeded_child_substrate.py` | `circleworld_proto` | `auditor` | `documented_reference` | `documented_or_reported` | tracked=no; text_refs=1; docs_refs=1; loc=215 |
| `runtimes/circleworld_proto/build_audio_predeclared_lockbox.py` | `circleworld_proto` | `builder` | `documented_reference` | `documented_or_reported` | tracked=no; text_refs=1; docs_refs=2; loc=792 |
| `runtimes/circleworld_proto/build_internal_phase_law_failure_atlas.py` | `circleworld_proto` | `builder` | `documented_reference` | `documented_or_reported` | tracked=no; text_refs=4; docs_refs=12; loc=336 |
| `runtimes/circleworld_proto/build_internal_phase_law_variant_spec.py` | `circleworld_proto` | `builder` | `documented_reference` | `documented_or_reported` | tracked=no; text_refs=5; docs_refs=11; loc=1309 |
| `runtimes/circleworld_proto/build_relational_signature_library.py` | `circleworld_proto` | `builder` | `documented_reference` | `documented_or_reported` | tracked=no; text_refs=4; docs_refs=8; loc=40 |
| `runtimes/circleworld_proto/compare_audio_lockbox_results.py` | `circleworld_proto` | `comparator` | `documented_reference` | `documented_or_reported` | tracked=no; text_refs=2; docs_refs=2; loc=893 |
| `runtimes/circleworld_proto/compare_internal_phase_law_variants.py` | `circleworld_proto` | `comparator` | `documented_reference` | `documented_or_reported` | tracked=no; text_refs=6; docs_refs=14; loc=319 |
| `runtimes/circleworld_proto/compare_learned_signature_split_suites.py` | `circleworld_proto` | `comparator` | `documented_reference` | `documented_or_reported` | tracked=no; text_refs=1; docs_refs=2; loc=254 |
| `runtimes/circleworld_proto/compare_relational_signature_runs.py` | `circleworld_proto` | `comparator` | `documented_cli` | `documented_or_reported` | tracked=no; text_refs=1; docs_refs=1; loc=473 |
| `runtimes/circleworld_proto/diagnose_internal_phase_law_joint_rows.py` | `circleworld_proto` | `analysis` | `documented_reference` | `documented_or_reported` | tracked=no; text_refs=4; docs_refs=8; loc=318 |
| `runtimes/circleworld_proto/predeclare_internal_phase_law_diagnostic_class.py` | `circleworld_proto` | `analysis` | `documented_reference` | `documented_or_reported` | tracked=no; text_refs=1; docs_refs=2; loc=262 |
| `runtimes/circleworld_proto/retrospective_childworld_eval.py` | `circleworld_proto` | `script_or_module` | `documented_reference` | `documented_or_reported` | tracked=no; text_refs=4; docs_refs=6; loc=370 |
| `runtimes/circleworld_proto/run_audio_baseline_stratified_diagnostics.py` | `circleworld_proto` | `experiment_runner` | `documented_reference` | `documented_or_reported` | tracked=no; text_refs=1; docs_refs=2; loc=250 |
| `runtimes/circleworld_proto/run_audio_delta_objective_scout.py` | `circleworld_proto` | `experiment_runner` | `documented_reference` | `documented_or_reported` | tracked=no; text_refs=1; docs_refs=2; loc=520 |
| `runtimes/circleworld_proto/run_audio_mechanism_family_sensitivity.py` | `circleworld_proto` | `experiment_runner` | `documented_reference` | `documented_or_reported` | tracked=no; text_refs=3; docs_refs=6; loc=560 |
| `runtimes/circleworld_proto/run_audio_phase_influence_ablation.py` | `circleworld_proto` | `experiment_runner` | `documented_reference` | `documented_or_reported` | tracked=no; text_refs=3; docs_refs=6; loc=396 |
| `runtimes/circleworld_proto/run_audio_phase_phenotype_diagnostics.py` | `circleworld_proto` | `experiment_runner` | `documented_reference` | `documented_or_reported` | tracked=no; text_refs=1; docs_refs=2; loc=426 |
| `runtimes/circleworld_proto/run_audio_phase_seed_ablation.py` | `circleworld_proto` | `experiment_runner` | `documented_reference` | `documented_or_reported` | tracked=no; text_refs=3; docs_refs=6; loc=477 |
| `runtimes/circleworld_proto/run_audio_steady_phenotype_manifest.py` | `circleworld_proto` | `experiment_runner` | `documented_reference` | `documented_or_reported` | tracked=no; text_refs=1; docs_refs=2; loc=570 |
| `runtimes/circleworld_proto/run_branch_pressure_experiment.py` | `circleworld_proto` | `experiment_runner` | `documented_reference` | `documented_or_reported` | tracked=no; text_refs=5; docs_refs=7; loc=292 |
| `runtimes/circleworld_proto/run_branchlaw_ablation_series.py` | `circleworld_proto` | `experiment_runner` | `documented_reference` | `documented_or_reported` | tracked=no; text_refs=3; docs_refs=3; loc=339 |
| `runtimes/circleworld_proto/run_child_basis_coverage_assay.py` | `circleworld_proto` | `experiment_runner` | `documented_reference` | `documented_or_reported` | tracked=no; text_refs=3; docs_refs=10; loc=515 |
| `runtimes/circleworld_proto/run_childworld_causality_killswitch_suite.py` | `circleworld_proto` | `experiment_runner` | `documented_reference` | `documented_or_reported` | tracked=no; text_refs=1; docs_refs=1; loc=491 |
| `runtimes/circleworld_proto/run_childworld_volume_recursion_sweep.py` | `circleworld_proto` | `experiment_runner` | `documented_reference` | `documented_or_reported` | tracked=no; text_refs=6; docs_refs=20; loc=1614 |
| `runtimes/circleworld_proto/run_internal_phase_law_objective_scout.py` | `circleworld_proto` | `experiment_runner` | `documented_reference` | `documented_or_reported` | tracked=no; text_refs=6; docs_refs=20; loc=791 |
| `runtimes/circleworld_proto/run_learned_gated_runtime_sandbox.py` | `circleworld_proto` | `experiment_runner` | `documented_reference` | `documented_or_reported` | tracked=no; text_refs=2; docs_refs=3; loc=393 |
| `runtimes/circleworld_proto/run_learned_signature_split_suite.py` | `circleworld_proto` | `experiment_runner` | `documented_reference` | `documented_or_reported` | tracked=no; text_refs=2; docs_refs=4; loc=672 |
| `runtimes/circleworld_proto/run_learned_structural_family_probe.py` | `circleworld_proto` | `experiment_runner` | `documented_reference` | `documented_or_reported` | tracked=no; text_refs=1; docs_refs=4; loc=443 |
| `runtimes/circleworld_proto/run_parent_ontology_bridge_scout.py` | `circleworld_proto` | `experiment_runner` | `documented_reference` | `documented_or_reported` | tracked=no; text_refs=3; docs_refs=6; loc=783 |
| `runtimes/circleworld_proto/run_relational_signature_experiment.py` | `circleworld_proto` | `experiment_runner` | `documented_cli` | `documented_or_reported` | tracked=no; text_refs=1; docs_refs=1; loc=817 |
| `runtimes/circleworld_proto/run_resonant_child_retrieval_assay.py` | `circleworld_proto` | `experiment_runner` | `documented_reference` | `documented_or_reported` | tracked=no; text_refs=3; docs_refs=13; loc=189 |
| `runtimes/circleworld_proto/run_token_diversity_experiment.py` | `circleworld_proto` | `experiment_runner` | `documented_reference` | `documented_or_reported` | tracked=no; text_refs=5; docs_refs=9; loc=276 |
| `runtimes/circleworld_proto/score_internal_phase_law_objective.py` | `circleworld_proto` | `scorer` | `documented_reference` | `documented_or_reported` | tracked=no; text_refs=5; docs_refs=13; loc=410 |
| `runtimes/circleworld_proto/test_internal_phase_law_reentry_contract.py` | `circleworld_proto` | `contract_test` | `documented_reference` | `contract_or_test` | tracked=no; text_refs=2; docs_refs=4; loc=113 |
| `runtimes/circleworld_proto/test_learned_branch_law_child_ifs_contract.py` | `circleworld_proto` | `contract_test` | `documented_reference` | `contract_or_test` | tracked=no; text_refs=21; docs_refs=56; loc=353 |
| `runtimes/circleworld_proto/test_packet_alias_contract.py` | `circleworld_proto` | `contract_test` | `documented_reference` | `contract_or_test` | tracked=no; text_refs=2; docs_refs=5; loc=303 |
| `runtimes/circleworld_proto/test_phase_gauge_invariance.py` | `circleworld_proto` | `contract_test` | `documented_reference` | `contract_or_test` | tracked=no; text_refs=5; docs_refs=18; loc=372 |
| `runtimes/circleworld_proto/test_resonant_attention_contract.py` | `circleworld_proto` | `contract_test` | `documented_reference` | `contract_or_test` | tracked=no; text_refs=20; docs_refs=67; loc=227 |
| `runtimes/circleworld_proto/test_semantic_projector_contract.py` | `circleworld_proto` | `contract_test` | `documented_reference` | `contract_or_test` | tracked=no; text_refs=1; docs_refs=4; loc=391 |
| `runtimes/circleworld_proto/test_unit_phasor_contract.py` | `circleworld_proto` | `contract_test` | `documented_reference` | `contract_or_test` | tracked=no; text_refs=7; docs_refs=23; loc=1031 |
| `runtimes/circleworld_proto/train_parent_phase_projector.py` | `circleworld_proto` | `trainer` | `documented_reference` | `documented_or_reported` | tracked=no; text_refs=2; docs_refs=2; loc=743 |
| `tests/test_registry_integrity.py` | `repo_tests` | `contract_test` | `documented_reference` | `contract_or_test` | tracked=yes; text_refs=2; docs_refs=6; loc=99 |
| `tools/audit_project_usage.py` | `tools` | `auditor` | `documented_reference` | `documented_or_reported` | tracked=no; text_refs=5; docs_refs=12; loc=881 |
| `tools/binding_stats.py` | `tools` | `script_or_module` | `documented_cli` | `documented_or_reported` | tracked=yes; text_refs=1; docs_refs=1; loc=46 |
| `tools/dsp_labeler.py` | `tools` | `script_or_module` | `documented_cli` | `documented_or_reported` | tracked=yes; text_refs=1; docs_refs=1; loc=96 |
| `tools/eval_semantic_tension.py` | `tools` | `script_or_module` | `documented_reference` | `documented_or_reported` | tracked=yes; text_refs=3; docs_refs=4; loc=242 |
| `tools/extract_tension_targets.py` | `tools` | `script_or_module` | `documented_cli` | `documented_or_reported` | tracked=yes; text_refs=1; docs_refs=1; loc=96 |
| `tools/fault_localization.py` | `tools` | `script_or_module` | `documented_cli` | `documented_or_reported` | tracked=yes; text_refs=1; docs_refs=1; loc=36 |
| `tools/infer_rafa.py` | `tools` | `exporter` | `documented_reference` | `documented_or_reported` | tracked=yes; text_refs=1; docs_refs=2; loc=79 |
| `tools/manifest_gen.py` | `tools` | `script_or_module` | `documented_cli` | `documented_or_reported` | tracked=yes; text_refs=1; docs_refs=1; loc=389 |
| `tools/path_b_eval.py` | `tools` | `script_or_module` | `documented_reference` | `documented_or_reported` | tracked=yes; text_refs=8; docs_refs=4; loc=157 |
| `tools/relational_probe.py` | `tools` | `script_or_module` | `documented_cli` | `documented_or_reported` | tracked=yes; text_refs=2; docs_refs=1; loc=468 |
| `tools/render_binding_comparison.py` | `tools` | `script_or_module` | `documented_cli` | `documented_or_reported` | tracked=yes; text_refs=1; docs_refs=1; loc=78 |
| `tools/run_validation.py` | `tools` | `experiment_runner` | `documented_cli` | `documented_or_reported` | tracked=yes; text_refs=1; docs_refs=1; loc=143 |
| `tools/train_stage1_voice.py` | `tools` | `trainer` | `documented_reference` | `documented_or_reported` | tracked=yes; text_refs=2; docs_refs=3; loc=47 |
| `tools/train_stage2_spine.py` | `tools` | `trainer` | `documented_reference` | `documented_or_reported` | tracked=yes; text_refs=2; docs_refs=3; loc=56 |
| `tools/train_stage3_brain.py` | `tools` | `trainer` | `documented_reference` | `documented_or_reported` | tracked=yes; text_refs=2; docs_refs=3; loc=83 |
| `tools/verify.py` | `tools` | `auditor` | `documented_cli` | `documented_or_reported` | tracked=yes; text_refs=1; docs_refs=1; loc=24 |
| `tools/verify_physics.py` | `tools` | `auditor` | `documented_cli` | `documented_or_reported` | tracked=yes; text_refs=1; docs_refs=1; loc=443 |
| `train_functional.py` | `root_legacy` | `trainer` | `documented_reference` | `documented_or_reported` | tracked=yes; text_refs=2; docs_refs=6; loc=118 |
| `train_raw_tensors.py` | `root_legacy` | `trainer` | `documented_reference` | `documented_or_reported` | tracked=yes; text_refs=1; docs_refs=3; loc=57 |
| `train_sterile.py` | `root_legacy` | `trainer` | `documented_reference` | `documented_or_reported` | tracked=yes; text_refs=1; docs_refs=3; loc=60 |

### compatibility wrapper

| Script | Lane | Role | Use status | Lifecycle | Evidence |
| --- | --- | --- | --- | --- | --- |
| `core/blackwell_launch.py` | `core_shared` | `script_or_module` | `low_reference_or_legacy` | `unclassified` | tracked=yes; text_refs=1; loc=66 |
| `core/infer.py` | `core_shared` | `script_or_module` | `low_reference_or_legacy` | `unclassified` | tracked=yes; text_refs=1; loc=162 |
| `core/validate.py` | `core_shared` | `script_or_module` | `low_reference_or_legacy` | `unclassified` | tracked=yes; text_refs=1; loc=43 |
| `inference_package/infer_stage4.py` | `inference_package` | `exporter` | `low_reference_or_legacy` | `unclassified` | tracked=yes; loc=113 |
| `launch_lineage.ps1` | `root_legacy` | `launcher` | `documented_reference` | `documented_or_reported` | tracked=yes; text_refs=2; docs_refs=12; loc=38 |
| `lineages/03_subtractive_fork/eval_stage4.py` | `subtractive_fork` | `script_or_module` | `low_reference_or_legacy` | `unclassified` | tracked=yes; loc=157 |
| `lineages/03_subtractive_fork/sample_stage4.py` | `subtractive_fork` | `exporter` | `low_reference_or_legacy` | `unclassified` | tracked=yes; text_refs=1; loc=109 |
| `research_track/airflow/dags/rafa_learning_curve_resume.py` | `research_track` | `script_or_module` | `low_reference_or_legacy` | `unclassified` | tracked=yes; text_refs=1; loc=65 |
| `research_track/airflow/dags/rafa_pathb_hypercube.py` | `research_track` | `script_or_module` | `low_reference_or_legacy` | `unclassified` | tracked=yes; text_refs=2; loc=84 |
| `research_track/infra/launch_conditioned_depth_job.ps1` | `research_track` | `launcher` | `low_reference_or_legacy` | `unclassified` | tracked=yes; text_refs=1; loc=61 |
| `research_track/infra/launch_conditioned_depth_job.py` | `research_track` | `script_or_module` | `low_reference_or_legacy` | `unclassified` | tracked=yes; loc=72 |
| `research_track/infra/schedule_conditioned_depth_job.ps1` | `research_track` | `launcher` | `low_reference_or_legacy` | `unclassified` | tracked=yes; loc=62 |

### superseded-review-needed

| Script | Lane | Role | Use status | Lifecycle | Evidence |
| --- | --- | --- | --- | --- | --- |
| `clip_audio_loader.py` | `root_legacy` | `script_or_module` | `low_reference_or_legacy` | `legacy_or_tooling` | tracked=yes; loc=47 |
| `research_track/infra/run_conditioned_depth_batch.py` | `research_track` | `experiment_runner` | `low_reference_or_legacy` | `unclassified` | tracked=yes; loc=65 |
| `research_track/infra/run_conditioned_depth_job.py` | `research_track` | `experiment_runner` | `low_reference_or_legacy` | `unclassified` | tracked=yes; text_refs=3; loc=249 |
| `research_track/infra/run_hypercube_batch.py` | `research_track` | `experiment_runner` | `low_reference_or_legacy` | `unclassified` | tracked=yes; loc=55 |
| `research_track/infra/run_logic_diag_batch.py` | `research_track` | `experiment_runner` | `low_reference_or_legacy` | `unclassified` | tracked=yes; loc=55 |
| `runtimes/circleworld_proto/run_retrospective_grid.py` | `circleworld_proto` | `experiment_runner` | `low_reference_or_legacy` | `active_unregistered_research` | tracked=yes; loc=251 |
| `tools/ablate_contract.py` | `tools` | `script_or_module` | `low_reference_or_legacy` | `legacy_or_tooling` | tracked=yes; loc=110 |
| `tools/acid_test.py` | `tools` | `script_or_module` | `low_reference_or_legacy` | `legacy_or_tooling` | tracked=yes; loc=102 |
| `tools/audit_clap.py` | `tools` | `auditor` | `low_reference_or_legacy` | `legacy_or_tooling` | tracked=yes; loc=20 |
| `tools/audit_hierarchy.py` | `tools` | `auditor` | `low_reference_or_legacy` | `legacy_or_tooling` | tracked=yes; loc=19 |
| `tools/audit_params.py` | `tools` | `auditor` | `low_reference_or_legacy` | `legacy_or_tooling` | tracked=yes; loc=34 |
| `tools/causal_audit.py` | `tools` | `script_or_module` | `low_reference_or_legacy` | `legacy_or_tooling` | tracked=yes; loc=64 |
| `tools/debug_nan.py` | `tools` | `script_or_module` | `low_reference_or_legacy` | `legacy_or_tooling` | tracked=yes; loc=57 |
| `tools/hard_ablation_audit.py` | `tools` | `script_or_module` | `low_reference_or_legacy` | `legacy_or_tooling` | tracked=yes; loc=83 |
| `tools/profile_rafa.py` | `tools` | `script_or_module` | `low_reference_or_legacy` | `legacy_or_tooling` | tracked=yes; loc=33 |
| `tools/profile_step.py` | `tools` | `script_or_module` | `low_reference_or_legacy` | `legacy_or_tooling` | tracked=yes; loc=49 |
| `tools/rafa_leaderboard.py` | `tools` | `script_or_module` | `low_reference_or_legacy` | `legacy_or_tooling` | tracked=yes; loc=62 |
| `tools/run_ablations.py` | `tools` | `experiment_runner` | `low_reference_or_legacy` | `legacy_or_tooling` | tracked=yes; loc=114 |
| `tools/run_hypercube.py` | `tools` | `experiment_runner` | `low_reference_or_legacy` | `legacy_or_tooling` | tracked=yes; loc=120 |
| `tools/run_learning_curve_resume.py` | `tools` | `experiment_runner` | `low_reference_or_legacy` | `legacy_or_tooling` | tracked=yes; text_refs=1; loc=152 |
| `tools/scrape_freewavesamples.py` | `tools` | `script_or_module` | `low_reference_or_legacy` | `legacy_or_tooling` | tracked=yes; text_refs=1; loc=85 |
| `tools/scrape_high_volume.py` | `tools` | `script_or_module` | `low_reference_or_legacy` | `legacy_or_tooling` | tracked=yes; loc=99 |
| `tools/scrape_mixkit.py` | `tools` | `script_or_module` | `low_reference_or_legacy` | `legacy_or_tooling` | tracked=yes; text_refs=1; loc=94 |
| `tools/scrape_soundbible.py` | `tools` | `script_or_module` | `low_reference_or_legacy` | `legacy_or_tooling` | tracked=yes; text_refs=1; loc=78 |
| `tools/scrape_wav.py` | `tools` | `script_or_module` | `low_reference_or_legacy` | `legacy_or_tooling` | tracked=yes; loc=54 |
| `tools/scrape_wavsource.py` | `tools` | `script_or_module` | `low_reference_or_legacy` | `legacy_or_tooling` | tracked=yes; text_refs=1; loc=79 |
| `tools/tests_diffusion_shell.py` | `tools` | `script_or_module` | `low_reference_or_legacy` | `legacy_or_tooling` | tracked=yes; loc=38 |
| `tools/total_recall.py` | `tools` | `script_or_module` | `low_reference_or_legacy` | `legacy_or_tooling` | tracked=yes; loc=38 |

### retire-candidate-watchlist

| Script | Lane | Role | Use status | Lifecycle | Evidence |
| --- | --- | --- | --- | --- | --- |
| `runtimes/circleworld_proto/analyze_arc_q_control_claim.py` | `circleworld_proto` | `analysis` | `untracked_low_reference_candidate` | `active_unregistered_research` | tracked=no; loc=157 |
| `runtimes/circleworld_proto/analyze_dense_signature_failure_modes.py` | `circleworld_proto` | `analysis` | `untracked_low_reference_candidate` | `active_unregistered_research` | tracked=no; loc=415 |
| `runtimes/circleworld_proto/analyze_learned_signature_case_failures.py` | `circleworld_proto` | `analysis` | `untracked_low_reference_candidate` | `active_unregistered_research` | tracked=no; loc=157 |
| `runtimes/circleworld_proto/assemble_childworld_cycle_report.py` | `circleworld_proto` | `report_assembler` | `untracked_low_reference_candidate` | `active_unregistered_research` | tracked=no; loc=616 |
| `runtimes/circleworld_proto/assemble_childworld_mechanism_audit.py` | `circleworld_proto` | `report_assembler` | `untracked_low_reference_candidate` | `active_unregistered_research` | tracked=no; loc=622 |
| `runtimes/circleworld_proto/assemble_circleworld_experiment_ledger.py` | `circleworld_proto` | `report_assembler` | `untracked_low_reference_candidate` | `active_unregistered_research` | tracked=no; loc=556 |
| `runtimes/circleworld_proto/assemble_circleworld_scoreboard.py` | `circleworld_proto` | `report_assembler` | `untracked_low_reference_candidate` | `active_unregistered_research` | tracked=no; loc=322 |
| `runtimes/circleworld_proto/assemble_phase_token_repair_report.py` | `circleworld_proto` | `report_assembler` | `untracked_low_reference_candidate` | `active_unregistered_research` | tracked=no; loc=438 |
| `runtimes/circleworld_proto/assemble_static_child_conversion_report.py` | `circleworld_proto` | `report_assembler` | `untracked_low_reference_candidate` | `active_unregistered_research` | tracked=no; loc=404 |
| `runtimes/circleworld_proto/audit_circleworld_seed_determinism.py` | `circleworld_proto` | `auditor` | `untracked_low_reference_candidate` | `active_unregistered_research` | tracked=no; loc=579 |
| `runtimes/circleworld_proto/audit_relational_signature_contract.py` | `circleworld_proto` | `auditor` | `untracked_low_reference_candidate` | `active_unregistered_research` | tracked=no; loc=188 |
| `runtimes/circleworld_proto/build_childworld_mechanism_dataset.py` | `circleworld_proto` | `builder` | `untracked_low_reference_candidate` | `active_unregistered_research` | tracked=no; loc=852 |
| `runtimes/circleworld_proto/classify_internal_phase_law_candidate_classes.py` | `circleworld_proto` | `analysis` | `untracked_low_reference_candidate` | `active_unregistered_research` | tracked=no; loc=185 |
| `runtimes/circleworld_proto/compare_audio_continuation_methods.py` | `circleworld_proto` | `comparator` | `untracked_low_reference_candidate` | `active_unregistered_research` | tracked=no; loc=253 |
| `runtimes/circleworld_proto/compare_learned_signature_scouts.py` | `circleworld_proto` | `comparator` | `untracked_low_reference_candidate` | `active_unregistered_research` | tracked=no; loc=354 |
| `runtimes/circleworld_proto/compare_signature_operator_tail_probes.py` | `circleworld_proto` | `comparator` | `untracked_low_reference_candidate` | `active_unregistered_research` | tracked=no; loc=166 |
| `runtimes/circleworld_proto/evaluate_signature_future_law_probe.py` | `circleworld_proto` | `evaluator` | `untracked_low_reference_candidate` | `active_unregistered_research` | tracked=no; loc=674 |
| `runtimes/circleworld_proto/run_claim_isolation_suite.py` | `circleworld_proto` | `experiment_runner` | `untracked_low_reference_candidate` | `active_unregistered_research` | tracked=no; loc=621 |
| `runtimes/circleworld_proto/run_learned_branch_law_assay.py` | `circleworld_proto` | `experiment_runner` | `untracked_low_reference_candidate` | `active_unregistered_research` | tracked=no; loc=571 |
| `runtimes/circleworld_proto/summarize_learned_signature_frontier.py` | `circleworld_proto` | `report_assembler` | `untracked_low_reference_candidate` | `active_unregistered_research` | tracked=no; loc=419 |
| `runtimes/circleworld_proto/sweep_childsurvival_replay.py` | `circleworld_proto` | `sweep` | `untracked_low_reference_candidate` | `active_unregistered_research` | tracked=no; loc=192 |
| `runtimes/circleworld_proto/sweep_parentmix_replay.py` | `circleworld_proto` | `sweep` | `untracked_low_reference_candidate` | `active_unregistered_research` | tracked=no; loc=161 |
| `runtimes/circleworld_proto/train_childworld_mechanism_classifier.py` | `circleworld_proto` | `trainer` | `untracked_low_reference_candidate` | `active_unregistered_research` | tracked=no; loc=261 |
| `runtimes/circleworld_proto/validate_tokenburst_tracks.py` | `circleworld_proto` | `auditor` | `untracked_low_reference_candidate` | `active_unregistered_research` | tracked=no; loc=178 |

## Verification Notes

- Source of truth was static cartography only; no source scripts were edited, moved, renamed, or deleted.
- `PROJECT_USAGE_MAP.md` explicitly states that low-reference and cold labels do not prove deletion safety; this artifact preserves that constraint.
- The redundancy report recommends helper extraction before deletion; this classification therefore separates low-reference review from retirement authorization.
- Compatibility wrappers are conservative manual labels for launchers, schedulers, and legacy entrypoints; they require downstream caller checks before removal.
- Artifact groups were not reclassified here. The usage audit says artifact cleanup needs separate archival decisions after checkpoint/output groups are tied to reports or marked as expendable caches.

## Recommended Retirement Workflow

1. Start with `retire-candidate-watchlist`, because those scripts are untracked and have no imports/text references in the static usage map.
2. For each candidate, search reports, ledgers, shell history, job configs, and artifact producer names before marking retired.
3. Move tracked `superseded-review-needed` scripts only after replacing callers or preserving an explicit compatibility wrapper.
4. Do not delete `canonical`, `active research`, or `compatibility wrapper` scripts in the same pass as classification.
5. After any future lifecycle changes, regenerate `PROJECT_SCRIPT_INVENTORY` and `PROJECT_USAGE_MAP` and refresh this document.


## Post-Integration Addendum - 2026-05-20

The cleanup orchestration added four new Circleworld shared-scaffolding scripts after the original classification snapshot. They are not deletion candidates.

| Script | Category | Rationale |
| --- | --- | --- |
| `runtimes/circleworld_proto/experiment_reports.py` | `active research` | New additive report-helper infrastructure. Not fully consumed yet, but intended to reduce `_write_markdown`/table/front-matter duplication. |
| `runtimes/circleworld_proto/phase_audio_metrics.py` | `canonical` | New imported audio metric home used by `benchmark_audio_continuation.py`; preserves legacy `_compare_waveforms`/`_loop_reentry_metrics` aliases through that script. |
| `runtimes/circleworld_proto/profile_registry.py` | `canonical` | New imported profile/route registry used by phase-native route-policy scripts. |
| `runtimes/circleworld_proto/seed_substrate.py` | `active research` | New additive seed/substrate helper scaffold. Not fully consumed yet, but intended to replace duplicated seed parsing in assays. |

Measurement caveat: because this document references every classified script, later usage-map scans will report many more `documented_reference` scripts. For deletion triage, use this document's category tables rather than treating `documented_reference` as independent proof of current runtime use.
