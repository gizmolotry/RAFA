# Circleworld Childworld Cycle Report - 2026-05-11

- Schema: `circleworld_childworld_cycle_report_v1`
- Cycle: `continuation_causal_childifs_2026-05-11`
- Mechanism read: `child_record_presence_required+coherence_retention_required+direct_child_phase_carrier+phase_specific_under_budget_clamp+readout_only+static_child_record_direct_carrier+strict_direct_child_mix_carrier`
- Mode-replacement result: `not_supported`
- Classifier readiness: `partial_artifact_classifier_ready_not_branch_law`

## Key Claims

- `supported`: Guarded childworld cycle produced a continuation-family nested sibling signal.
- `supported`: The best current mechanism is narrow: static/forked child phase record as a final direct continuation carrier.
- `not_supported`: Explicit low-rank continuation writeback is not established as necessary or sufficient.
- `not_supported`: Mode-replacement sibling behavior remains negative in this cycle.
- `direct_mix_shortcut_not_parent_mode_conversion`: The new conversion metrics separate child-record survival from parent/mode conversion.
- `partial_artifact_classifier_ready_not_branch_law`: Learned-law/classifier readiness is partial and artifact-level only.

## Evidence Metrics

### Nested Compare

| metric | previous | selected | delta |
| --- | ---: | ---: | ---: |
| `mean_nested_sibling_fraction` | 0.0416667 | 0.0833333 | 0.0416667 |
| `mean_assay_readout_nested_sibling_fraction` | 0.5 | 0.5 | 0 |
| `mean_assay_continuation_nested_sibling_fraction` | 0 | 0.5 | 0.5 |
| `mean_assay_mode_replace_nested_sibling_fraction` | 0 | 0 | 0 |
| `mean_assay_continuation_max_nested_sibling_readiness` | 0.966226 | 1 | 0.0337738 |
| `mean_assay_continuation_nested_sibling_readiness` | 0.952694 | 0.75 | -0.202694 |
| `mean_assay_continuation_continuation_write_delta` | 0.10936 | 0.118563 | 0.00920323 |
| `mean_world_jump_penalty` | 0 | 0 | 0 |
| `mean_over_rigid_fraction` | 0 | 0 | 0 |

### Held-Out Compare

| metric | previous | selected | delta |
| --- | ---: | ---: | ---: |
| `mean_real_branch_fraction` | 0.577778 | 0.577778 | 0 |
| `mean_child_writeback_mass` | 0.107311 | 0.119215 | 0.0119042 |
| `mean_child_parent_divergence` | 0.22057 | 0.239219 | 0.0186491 |
| `mean_child_sibling_divergence` | 0.155394 | 0.223374 | 0.0679804 |
| `mean_meso_branch_effect` | 0.0252147 | 0.0342448 | 0.0090301 |
| `mean_child_local_ifs_phase_delta` | 0.0155846 | 0.000743826 | -0.0148408 |
| `mean_child_local_ifs_coherence` | 0.394524 | 0.40814 | 0.0136161 |
| `mean_child_local_ifs_causal_gate` | 0.571226 | 0.588762 | 0.0175358 |

### Benchmark Compare

| metric | previous | selected | delta |
| --- | ---: | ---: | ---: |
| `mean_corr` | 0.988422 | 0.999996 | 0.0115742 |
| `mean_mae` | 0.0169982 | 0.00038027 | -0.0166179 |
| `mean_mse` | 0.00133093 | 3.57429e-07 | -0.00133058 |

- Selected recurrence score: `0.201897`
- Selected recurrence band: `low`
- Benchmark file count: `5`

## Mechanism Audit

- Suites: `5` total, `3` with positive signal

- `child_record_presence_required`: 1
- `coherence_retention_required`: 1
- `direct_child_phase_carrier`: 1
- `phase_specific_under_budget_clamp`: 1
- `readout_only`: 1
- `static_child_record_direct_carrier`: 1
- `strict_direct_child_mix_carrier`: 1

## Negative Mode-Replacement Result

- Status: `not_supported`
- Selected mode-replace nested sibling fraction: `0`
- Static mode-replace variants with signal: `0` of `8`
- Dataset target counts: `{"false": 12, "null": 161, "true": 0}`

## Conversion Metric Smoke

- Status: `direct_mix_shortcut_not_parent_mode_conversion`
- Read: single-seed smoke validates the new metric split: child record survives and direct continuation succeeds, but parent/mode conversion remains zero

| metric | value |
| --- | ---: |
| `mean_assay_readout_nested_sibling_fraction` | 0.5 |
| `mean_assay_continuation_nested_sibling_fraction` | 0.5 |
| `mean_assay_mode_replace_nested_sibling_fraction` | 0 |
| `mean_assay_mode_replace_max_nested_sibling_readiness` | 0.572817 |
| `mean_direct_readout_dependency` | 0 |
| `mean_final_direct_mix_shortcut_score` | 0.5 |
| `mean_child_record_survival_score` | 1 |
| `mean_parent_mode_conversion_sibling_fraction` | 0 |
| `mean_mode_replace_conversion_score` | 0 |
| `mean_world_jump_penalty` | 0 |

## Learned-Law / Classifier Readiness

- Readiness: `partial_artifact_classifier_ready_not_branch_law`
- Case count: `173`
- Trained targets: `target_direct_carrier`
- Skipped targets: `target_writeback_sufficient, target_phase_specific, target_child_record_required, target_mode_replace_supported`

| target | status | n | positive | negative | acc | precision | recall |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `target_direct_carrier` | `trained` | 20 | 12 | 8 | 1 | 1 | 1 |
| `target_writeback_sufficient` | `skipped_single_class_or_empty` | 6 | 0 |  |  |  |  |
| `target_phase_specific` | `skipped_single_class_or_empty` | 4 | 4 |  |  |  |  |
| `target_child_record_required` | `skipped_single_class_or_empty` | 1 | 1 |  |  |  |  |
| `target_mode_replace_supported` | `skipped_single_class_or_empty` | 12 | 0 |  |  |  |  |

## Next-Step Gates

- `open` `External replay gate`: Reproduce continuation direct-carrier signal on additional seeds and non-selected anchors.
- `blocked` `Mode-replacement gate`: Require nonzero mode-replacement nested sibling fraction under static child record variants.
- `blocked` `Classifier target-balance gate`: Collect both positive and negative examples for skipped classifier targets before claiming learned-law readiness.
- `blocked` `Runtime promotion gate`: Do not promote to broad active default until wider regression and heldout suites clear.

## Source Artifacts

- `cycle_compare`: `loaded` - `D:\RAFA\outputs\circleworld_proto\continuation_causal_childifs_2026-05-11\continuation_causal_compare.json`
- `mechanism_audit`: `loaded` - `D:\RAFA\outputs\circleworld_proto\continuation_causal_childifs_2026-05-11\childworld_mechanism_audit_combined.json`
- `dataset_v2`: `loaded` - `D:\RAFA\outputs\circleworld_proto\continuation_causal_childifs_2026-05-11\childworld_mechanism_dataset_v2.json`
- `classifier_v1`: `loaded` - `D:\RAFA\outputs\circleworld_proto\continuation_causal_childifs_2026-05-11\childworld_mechanism_classifier_v1.json`
- `static_conversion`: `loaded` - `D:\RAFA\outputs\circleworld_proto\continuation_causal_childifs_2026-05-11\static_child_mode_replace_conversion_combined.json`
- `heldout_summary`: `loaded` - `D:\RAFA\outputs\circleworld_proto\continuation_causal_childifs_2026-05-11\heldout_guarded_selected\heldout_summary.json`
- `benchmark_summary`: `loaded` - `D:\RAFA\outputs\circleworld_proto\continuation_causal_childifs_2026-05-11\benchmark_guarded_selected\benchmark_summary.json`
- `nested_readiness`: `loaded` - `D:\RAFA\outputs\circleworld_proto\continuation_causal_childifs_2026-05-11\nested_causal_selected_v4_readiness\nested_commitment_report.json`
- `conversion_metric_smoke`: `loaded` - `D:\RAFA\outputs\circleworld_proto\continuation_causal_childifs_2026-05-11\conversion_metric_smoke_seed9100_cuda\nested_commitment_report.json`
