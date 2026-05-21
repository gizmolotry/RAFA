# Project Redundancy Hotspots

Generated: `2026-05-20T17:29:07+00:00`

This report identifies overlap candidates. It does not prove code is safe to delete.

## Headline

- Source-like scripts: `239`
- Total nonblank LOC: `91671`
- Largest lane: `circleworld_proto` with `129` scripts
- Largest prefix cluster: `run_learned_gated`

## Largest Prefix Clusters

| Prefix             | Count | Nonblank LOC | Lanes                                                 | Roles               |
| ------------------ | ----- | ------------ | ----------------------------------------------------- | ------------------- |
| run_learned_gated  | 4     | 2512         | circleworld_proto:4                                   | experiment_runner:4 |
| evaluate_signature | 3     | 2218         | circleworld_proto:3                                   | evaluator:3         |
| score_phase_native | 6     | 1921         | circleworld_proto:6                                   | scorer:6            |
| run_audio_phase    | 3     | 1299         | circleworld_proto:3                                   | experiment_runner:3 |
| analyze_internal   | 3     | 1108         | circleworld_proto:3                                   | analysis:3          |
| train_phase_native | 3     | 943          | circleworld_proto:3                                   | trainer:3           |
| export_graduation  | 3     | 115          | stage4_blackwell_14:1, stage4_blackwell_16:1, tools:1 | exporter:3          |
| test_generate      | 3     | 112          | research_track:3                                      | contract_test:3     |

## Helper Candidates

Repeated helper names are the first obvious extraction surface. `main` and `parse_args` are intentionally excluded from this section.

| Function                 | Count | Files |
| ------------------------ | ----- | ----- |
| _write_markdown          | 47    | 47    |
| _as_float                | 39    | 39    |
| _mean                    | 29    | 29    |
| _fmt                     | 25    | 25    |
| _markdown                | 23    | 23    |
| _write_json              | 19    | 19    |
| _load_json               | 18    | 18    |
| _utc_timestamp           | 16    | 16    |
| _json_load               | 14    | 14    |
| _aggregate               | 12    | 12    |
| _safe_device             | 11    | 11    |
| _overall_status          | 8     | 8     |
| _run_case                | 8     | 8     |
| _clamp01                 | 6     | 6     |
| _median                  | 6     | 6     |
| _num                     | 6     | 6     |
| _parse_seeds             | 6     | 6     |
| assemble                 | 6     | 6     |
| render_markdown          | 6     | 6     |
| run_audit                | 6     | 6     |
| _case_group              | 5     | 5     |
| _markdown_summary        | 5     | 5     |
| _metric                  | 5     | 5     |
| _read_json               | 5     | 5     |
| _renorm                  | 5     | 5     |
| _row                     | 5     | 5     |
| _run                     | 5     | 5     |
| _save_wav                | 5     | 5     |
| _variant_specs           | 5     | 5     |
| pearson_correlation_loss | 5     | 5     |
| run_experiment           | 5     | 5     |
| run_probe                | 5     | 5     |
| write_markdown           | 5     | 5     |
| _case_key                | 4     | 4     |
| _centroid                | 4     | 4     |
| _decision                | 4     | 4     |
| _dict                    | 4     | 4     |
| _fit_scaler              | 4     | 4     |
| _get_soup                | 4     | 4     |
| _git_sha                 | 4     | 4     |

## Exact Duplicate Function Bodies

These are AST-identical top-level functions with at least four lines. Start here before doing semantic refactors.

| Names                          | Count | LOC | Files |
| ------------------------------ | ----- | --- | ----- |
| _as_float                      | 13    | 5   | 13    |
| _safe_device                   | 10    | 4   | 10    |
| _as_float                      | 8     | 5   | 8     |
| pearson_correlation_loss       | 5     | 8   | 5     |
| _as_float                      | 5     | 7   | 5     |
| _load_json                     | 5     | 5   | 5     |
| _parse_csv                     | 4     | 8   | 4     |
| _fmt                           | 4     | 6   | 4     |
| _centroid                      | 4     | 5   | 4     |
| _get_soup                      | 4     | 4   | 4     |
| _fit_scaler                    | 3     | 17  | 3     |
| _expand_inputs                 | 3     | 14  | 3     |
| _to_bool                       | 3     | 12  | 3     |
| _save_wav                      | 3     | 11  | 3     |
| compile_buckets                | 3     | 10  | 3     |
| _case_weights                  | 3     | 8   | 3     |
| _fmt                           | 3     | 8   | 3     |
| _median                        | 3     | 8   | 3     |
| _median                        | 3     | 8   | 3     |
| _parse_csv_names               | 3     | 8   | 3     |
| _leave_one_out_mean            | 3     | 7   | 3     |
| _parse_csv_floats              | 3     | 7   | 3     |
| _extra_val_wavs                | 3     | 6   | 3     |
| _feature_keys                  | 3     | 6   | 3     |
| _finite_float                  | 3     | 6   | 3     |
| _git_sha                       | 3     | 6   | 3     |
| _load                          | 3     | 5   | 3     |
| _extra_train_wavs              | 3     | 4   | 3     |
| _mean                          | 3     | 4   | 3     |
| _load_circle_cfg               | 2     | 135 | 2     |
| _fused_ramanujan_summary_logic | 2     | 94  | 2     |
| log_to_ledger                  | 2     | 50  | 2     |
| _ramanujan_score_logic         | 2     | 46  | 2     |
| _case_deltas                   | 2     | 32  | 2     |
| _load_local_pcm_wav            | 2     | 31  | 2     |
| _iter_valid_wavs               | 2     | 26  | 2     |
| _probe_matrix                  | 2     | 24  | 2     |
| _best_per_group                | 2     | 22  | 2     |
| _run_nested_commitment         | 2     | 21  | 2     |
| _candidate_assay_paths         | 2     | 20  | 2     |

## Similar Import Pairs

High import overlap is not a bug, but it is a good smell for repeated script skeletons.

| Jaccard | Left                                                                 | Right                                                                               | Shared imports                                                         |
| ------- | -------------------------------------------------------------------- | ----------------------------------------------------------------------------------- | ---------------------------------------------------------------------- |
| 1.0     | `lineages/03_subtractive_fork/eval_stage4.py`                        | `tools/path_b_eval.py`                                                              | argparse, datetime, json, os, pathlib, re, subprocess, sys, time       |
| 1.0     | `research_track/infra/aggregate_hypercube_results.py`                | `research_track/infra/aggregate_logic_diag_results.py`                              | __future__, argparse, json, pathlib, typing                            |
| 1.0     | `research_track/infra/aggregate_hypercube_results.py`                | `runtimes/circleworld_proto/analyze_arc_q_control_claim.py`                         | __future__, argparse, json, pathlib, typing                            |
| 1.0     | `research_track/infra/aggregate_hypercube_results.py`                | `runtimes/circleworld_proto/assemble_phase_native_audio_route_policy_comparison.py` | __future__, argparse, json, pathlib, typing                            |
| 1.0     | `research_track/infra/aggregate_hypercube_results.py`                | `runtimes/circleworld_proto/assemble_phase_native_audio_shared_battlefield.py`      | __future__, argparse, json, pathlib, typing                            |
| 1.0     | `research_track/infra/aggregate_hypercube_results.py`                | `runtimes/circleworld_proto/assemble_rafa_claim_evidence.py`                        | __future__, argparse, json, pathlib, typing                            |
| 1.0     | `research_track/infra/aggregate_hypercube_results.py`                | `runtimes/circleworld_proto/audit_phase_native_audio_route_selector_contract.py`    | __future__, argparse, json, pathlib, typing                            |
| 1.0     | `research_track/infra/aggregate_hypercube_results.py`                | `runtimes/circleworld_proto/build_internal_phase_law_variant_spec.py`               | __future__, argparse, json, pathlib, typing                            |
| 1.0     | `research_track/infra/aggregate_hypercube_results.py`                | `runtimes/circleworld_proto/classify_internal_phase_law_candidate_classes.py`       | __future__, argparse, json, pathlib, typing                            |
| 1.0     | `research_track/infra/aggregate_hypercube_results.py`                | `runtimes/circleworld_proto/compare_relational_signature_runs.py`                   | __future__, argparse, json, pathlib, typing                            |
| 1.0     | `research_track/infra/aggregate_hypercube_results.py`                | `runtimes/circleworld_proto/diagnose_internal_phase_law_joint_rows.py`              | __future__, argparse, json, pathlib, typing                            |
| 1.0     | `research_track/infra/aggregate_hypercube_results.py`                | `runtimes/circleworld_proto/predeclare_internal_phase_law_diagnostic_class.py`      | __future__, argparse, json, pathlib, typing                            |
| 1.0     | `research_track/infra/aggregate_hypercube_results.py`                | `runtimes/circleworld_proto/run_audio_baseline_stratified_diagnostics.py`           | __future__, argparse, json, pathlib, typing                            |
| 1.0     | `research_track/infra/aggregate_logic_diag_results.py`               | `runtimes/circleworld_proto/analyze_arc_q_control_claim.py`                         | __future__, argparse, json, pathlib, typing                            |
| 1.0     | `research_track/infra/aggregate_logic_diag_results.py`               | `runtimes/circleworld_proto/assemble_phase_native_audio_route_policy_comparison.py` | __future__, argparse, json, pathlib, typing                            |
| 1.0     | `research_track/infra/aggregate_logic_diag_results.py`               | `runtimes/circleworld_proto/assemble_phase_native_audio_shared_battlefield.py`      | __future__, argparse, json, pathlib, typing                            |
| 1.0     | `research_track/infra/aggregate_logic_diag_results.py`               | `runtimes/circleworld_proto/assemble_rafa_claim_evidence.py`                        | __future__, argparse, json, pathlib, typing                            |
| 1.0     | `research_track/infra/aggregate_logic_diag_results.py`               | `runtimes/circleworld_proto/audit_phase_native_audio_route_selector_contract.py`    | __future__, argparse, json, pathlib, typing                            |
| 1.0     | `research_track/infra/aggregate_logic_diag_results.py`               | `runtimes/circleworld_proto/build_internal_phase_law_variant_spec.py`               | __future__, argparse, json, pathlib, typing                            |
| 1.0     | `research_track/infra/aggregate_logic_diag_results.py`               | `runtimes/circleworld_proto/classify_internal_phase_law_candidate_classes.py`       | __future__, argparse, json, pathlib, typing                            |
| 1.0     | `research_track/infra/aggregate_logic_diag_results.py`               | `runtimes/circleworld_proto/compare_relational_signature_runs.py`                   | __future__, argparse, json, pathlib, typing                            |
| 1.0     | `research_track/infra/aggregate_logic_diag_results.py`               | `runtimes/circleworld_proto/diagnose_internal_phase_law_joint_rows.py`              | __future__, argparse, json, pathlib, typing                            |
| 1.0     | `research_track/infra/aggregate_logic_diag_results.py`               | `runtimes/circleworld_proto/predeclare_internal_phase_law_diagnostic_class.py`      | __future__, argparse, json, pathlib, typing                            |
| 1.0     | `research_track/infra/aggregate_logic_diag_results.py`               | `runtimes/circleworld_proto/run_audio_baseline_stratified_diagnostics.py`           | __future__, argparse, json, pathlib, typing                            |
| 1.0     | `research_track/infra/generate_hypercube_jobs.py`                    | `research_track/infra/generate_logic_diag_jobs.py`                                  | __future__, argparse, itertools, json, pathlib, typing, yaml           |
| 1.0     | `research_track/infra/run_conditioned_depth_batch.py`                | `research_track/infra/run_logic_diag_job.py`                                        | __future__, argparse, datetime, json, pathlib, subprocess, sys, typing |
| 1.0     | `research_track/infra/tests/test_generate_conditioned_depth_jobs.py` | `research_track/infra/tests/test_generate_hypercube_jobs.py`                        | __future__, pathlib, research_track, unittest, yaml                    |
| 1.0     | `research_track/infra/tests/test_generate_conditioned_depth_jobs.py` | `research_track/infra/tests/test_generate_logic_diag_jobs.py`                       | __future__, pathlib, research_track, unittest, yaml                    |
| 1.0     | `research_track/infra/tests/test_generate_hypercube_jobs.py`         | `research_track/infra/tests/test_generate_logic_diag_jobs.py`                       | __future__, pathlib, research_track, unittest, yaml                    |
| 1.0     | `runtimes/circleworld_proto/analyze_arc_q_control_claim.py`          | `runtimes/circleworld_proto/assemble_phase_native_audio_route_policy_comparison.py` | __future__, argparse, json, pathlib, typing                            |

## Recommended Extraction Targets

1. `runtimes/circleworld_proto/common_io.py`: JSON loading/writing, markdown table rendering, run directory creation.
2. `runtimes/circleworld_proto/phase_audio_metrics.py`: correlation, MSE, loop/reentry metrics, target-normalized replay summaries.
3. `runtimes/circleworld_proto/experiment_reports.py`: report headers, summary tables, comparison JSON scaffolds.
4. `runtimes/circleworld_proto/childworld_metrics.py`: child survival/carry/writeback/divergence aggregation.
5. `tools/project_cartography.py` or this script's helpers if more repo-wide audits appear.

## Non-Deletion Rule

Do not delete or rename any script from this report until a canonical artifact, ledger entry, or manifest path has been checked. This report is a triage map, not a cleanup patch.
