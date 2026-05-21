# Childworld Mechanism Dataset

- schema: `circleworld_childworld_mechanism_dataset_v1`
- suite_count: `6`
- case_count: `138`

| suite | label_class | variant | pos | cont | readout | write | delta | carry | budget | q_corr | response | jump | direct | writeback | phase | child_record |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| positive_guarded | no_signal | positive_control | True | 0.5 | 0.5 | 1 | 0.118563 | 0.5 | 0.918532 | 0.998358 | 0.285716 | 0 |  |  |  |  |
| positive_guarded | direct_child_phase_carrier | scramble_continuation_child_phase | False | 0 | 0.5 | 1 | 0.302679 | 0.5 | 0.918532 | 0.998855 | 0.5038 | 0 | True |  |  |  |
| positive_guarded | direct_child_phase_carrier | continuation_parent_only | False | 0 | 0.5 | 0 | 0 | 0.5 | 0.918532 | 1 | 0.150069 | 0 | True |  |  |  |
| positive_guarded | gated_writeback_supported | disable_continuation_write | True | 0.5 | 0.5 | 0 | 0 | 0.5 | 0.918532 | 0.998427 | 0.284446 | 0 |  | False |  |  |
| positive_guarded | gated_writeback_supported | continuation_write_only_readout | False | 0 | 0.5 | 1 | 0.118563 | 0.5 | 0.918532 | 1 | 0.15007 | 0 |  | False |  |  |
| positive_guarded | child_local_ifs_supported | no_child_local_ifs | True | 0.5 | 0.5 | 1 | 0.165922 | 1 | 0.70688 | 0.995488 | 0.312465 | 0 |  |  |  |  |
| positive_guarded | coherence_retention_required | no_child_coherence_retention | False | 0 | 0 | 1 | 0.118345 | 0.25 | 0.475375 | 0.998246 | 0.289079 | 0 |  |  |  |  |
| positive_guarded | readout_only | disable_continuation_write | True | 0.5 | 0.5 | 0 | 0 | 0.5 | 0.918532 | 0.998427 | 0.284446 | 0 | True |  |  |  |
| positive_guarded | readout_only | continuation_write_only_readout | False | 0 | 0.5 | 1 | 0.118563 | 0.5 | 0.918532 | 1 | 0.15007 | 0 | True |  |  |  |
| positive_guarded | strict_direct_child_mix_carrier | strict_direct_mix_only |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| positive_guarded | strict_direct_child_mix_carrier | strict_write_only_no_preunroll |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| positive_guarded | phase_specific_under_budget_clamp | phase_roll_budget_clamped |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| positive_guarded | phase_specific_under_budget_clamp | phase_random_budget_clamped |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| positive_guarded | phase_specific_under_budget_clamp | phase_zero_budget_clamped |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| positive_guarded | phase_specific_under_budget_clamp | phase_cross_child_budget_clamped |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| positive_guarded | static_child_record_direct_carrier | static_child_record_direct_mix_only |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| positive_guarded | static_child_record_direct_carrier | static_child_record_write_only |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| positive_guarded | static_child_record_direct_carrier | static_child_record_stripped_control |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| positive_guarded | child_record_presence_required | static_child_record_stripped_control |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| positive_guarded | branch_preunroll_required | no_branch_pre_unroll |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| positive_guarded | branch_childworld_runtime_required | no_branch_childworld_runtime |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| positive_guarded | continuation_preunroll_required | no_continuation_pre_unroll |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| positive_guarded | strict_parent_no_child_context_breaks | strict_parent_no_child_context |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| previous_causal_retention | no_signal | positive_control | True | 0 | 0.5 | 1 | 0.10936 | 0.916667 | 0.712973 | 0.994949 | 0.326844 | 0 |  |  |  |  |
| previous_causal_retention | direct_child_phase_carrier | scramble_continuation_child_phase | False | 0 | 0.5 | 1 | 0.268397 | 0.916667 | 0.712973 | 0.994891 | 0.496538 | 0 | False |  |  |  |
| previous_causal_retention | direct_child_phase_carrier | continuation_parent_only | False | 0 | 0.5 | 0 | 0 | 1 | 0.755802 | 0.999963 | 0.164167 | 0 | False |  |  |  |
| previous_causal_retention | gated_writeback_supported | disable_continuation_write | False | 0 | 0.5 | 0 | 0 | 0.916667 | 0.712973 | 0.995143 | 0.326212 | 0 |  | False |  |  |
| previous_causal_retention | gated_writeback_supported | continuation_write_only_readout | False | 0 | 0.5 | 1 | 0.10936 | 0.916667 | 0.712973 | 0.999963 | 0.16417 | 0 |  | False |  |  |
| previous_causal_retention | child_local_ifs_supported | no_child_local_ifs | False | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |  |  |  |  |
| previous_causal_retention | coherence_retention_required | no_child_coherence_retention | False | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |  |  |  |  |
| previous_causal_retention | readout_only | disable_continuation_write | False | 0 | 0.5 | 0 | 0 | 0.916667 | 0.712973 | 0.995143 | 0.326212 | 0 | False |  |  |  |
| previous_causal_retention | readout_only | continuation_write_only_readout | False | 0 | 0.5 | 1 | 0.10936 | 0.916667 | 0.712973 | 0.999963 | 0.16417 | 0 | False |  |  |  |
| previous_causal_retention | strict_direct_child_mix_carrier | strict_direct_mix_only |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| previous_causal_retention | strict_direct_child_mix_carrier | strict_write_only_no_preunroll |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| previous_causal_retention | phase_specific_under_budget_clamp | phase_roll_budget_clamped |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| previous_causal_retention | phase_specific_under_budget_clamp | phase_random_budget_clamped |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| previous_causal_retention | phase_specific_under_budget_clamp | phase_zero_budget_clamped |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| previous_causal_retention | phase_specific_under_budget_clamp | phase_cross_child_budget_clamped |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| previous_causal_retention | static_child_record_direct_carrier | static_child_record_direct_mix_only |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| previous_causal_retention | static_child_record_direct_carrier | static_child_record_write_only |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| previous_causal_retention | static_child_record_direct_carrier | static_child_record_stripped_control |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| previous_causal_retention | child_record_presence_required | static_child_record_stripped_control |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| previous_causal_retention | branch_preunroll_required | no_branch_pre_unroll |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| previous_causal_retention | branch_childworld_runtime_required | no_branch_childworld_runtime |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| previous_causal_retention | continuation_preunroll_required | no_continuation_pre_unroll |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| previous_causal_retention | strict_parent_no_child_context_breaks | strict_parent_no_child_context |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| mutation_only_current | no_signal | positive_control | True | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |  |  |  |  |
| mutation_only_current | direct_child_phase_carrier | scramble_continuation_child_phase | False | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | False |  |  |  |
| mutation_only_current | direct_child_phase_carrier | continuation_parent_only | False | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | False |  |  |  |
| mutation_only_current | gated_writeback_supported | disable_continuation_write | False | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |  | False |  |  |
| mutation_only_current | gated_writeback_supported | continuation_write_only_readout | False | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |  | False |  |  |
| mutation_only_current | child_local_ifs_supported | no_child_local_ifs | False | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |  |  |  |  |
| mutation_only_current | coherence_retention_required | no_child_coherence_retention | False | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |  |  |  |  |
| mutation_only_current | readout_only | disable_continuation_write | False | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | False |  |  |  |
| mutation_only_current | readout_only | continuation_write_only_readout | False | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | False |  |  |  |
| mutation_only_current | strict_direct_child_mix_carrier | strict_direct_mix_only |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| mutation_only_current | strict_direct_child_mix_carrier | strict_write_only_no_preunroll |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| mutation_only_current | phase_specific_under_budget_clamp | phase_roll_budget_clamped |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| mutation_only_current | phase_specific_under_budget_clamp | phase_random_budget_clamped |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| mutation_only_current | phase_specific_under_budget_clamp | phase_zero_budget_clamped |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| mutation_only_current | phase_specific_under_budget_clamp | phase_cross_child_budget_clamped |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| mutation_only_current | static_child_record_direct_carrier | static_child_record_direct_mix_only |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| mutation_only_current | static_child_record_direct_carrier | static_child_record_write_only |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| mutation_only_current | static_child_record_direct_carrier | static_child_record_stripped_control |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| mutation_only_current | child_record_presence_required | static_child_record_stripped_control |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| mutation_only_current | branch_preunroll_required | no_branch_pre_unroll |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| mutation_only_current | branch_childworld_runtime_required | no_branch_childworld_runtime |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| mutation_only_current | continuation_preunroll_required | no_continuation_pre_unroll |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| mutation_only_current | strict_parent_no_child_context_breaks | strict_parent_no_child_context |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| strict_mechanism_suite_v1_singlecase | no_signal | positive_control | True | 0.5 | 0.5 | 1 | 0.118552 | 0.5 | 0.918532 | 0.99836 | 0.285536 | 0 |  |  |  |  |
| strict_mechanism_suite_v1_singlecase | direct_child_phase_carrier | scramble_continuation_child_phase |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| strict_mechanism_suite_v1_singlecase | direct_child_phase_carrier | continuation_parent_only |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| strict_mechanism_suite_v1_singlecase | gated_writeback_supported | disable_continuation_write |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| strict_mechanism_suite_v1_singlecase | gated_writeback_supported | continuation_write_only_readout |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| strict_mechanism_suite_v1_singlecase | child_local_ifs_supported | no_child_local_ifs |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| strict_mechanism_suite_v1_singlecase | coherence_retention_required | no_child_coherence_retention |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| strict_mechanism_suite_v1_singlecase | readout_only | disable_continuation_write |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| strict_mechanism_suite_v1_singlecase | readout_only | continuation_write_only_readout |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| strict_mechanism_suite_v1_singlecase | strict_direct_child_mix_carrier | strict_direct_mix_only | True | 0.5 | 0.5 | 0 | 0 | 0.5 | 0.918532 | 0.997942 | 0.265891 | 0 | True |  |  |  |
| strict_mechanism_suite_v1_singlecase | strict_direct_child_mix_carrier | strict_write_only_no_preunroll | False | 0 | 0.5 | 1 | 0.112901 | 0.5 | 0.918532 | 1 | 0.15007 | 0 | True |  |  |  |
| strict_mechanism_suite_v1_singlecase | phase_specific_under_budget_clamp | phase_roll_budget_clamped |  |  |  |  |  |  |  |  |  |  |  |  | True |  |
| strict_mechanism_suite_v1_singlecase | phase_specific_under_budget_clamp | phase_random_budget_clamped | False | 0 | 0.5 | 1 | 0.0461732 | 0.5 | 0.918532 | 0.908598 | 1.781 | 0.0416667 |  |  | True |  |
| strict_mechanism_suite_v1_singlecase | phase_specific_under_budget_clamp | phase_zero_budget_clamped |  |  |  |  |  |  |  |  |  |  |  |  | True |  |
| strict_mechanism_suite_v1_singlecase | phase_specific_under_budget_clamp | phase_cross_child_budget_clamped |  |  |  |  |  |  |  |  |  |  |  |  | True |  |
| strict_mechanism_suite_v1_singlecase | static_child_record_direct_carrier | static_child_record_direct_mix_only |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| strict_mechanism_suite_v1_singlecase | static_child_record_direct_carrier | static_child_record_write_only |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| strict_mechanism_suite_v1_singlecase | static_child_record_direct_carrier | static_child_record_stripped_control |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| strict_mechanism_suite_v1_singlecase | child_record_presence_required | static_child_record_stripped_control |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| strict_mechanism_suite_v1_singlecase | branch_preunroll_required | no_branch_pre_unroll |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| strict_mechanism_suite_v1_singlecase | branch_childworld_runtime_required | no_branch_childworld_runtime |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| strict_mechanism_suite_v1_singlecase | continuation_preunroll_required | no_continuation_pre_unroll |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| strict_mechanism_suite_v1_singlecase | strict_parent_no_child_context_breaks | strict_parent_no_child_context |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| static_child_record_suite_v1_fullseed | no_signal | positive_control | True | 0.5 | 0.5 | 1 | 0.118563 | 0.5 | 0.918532 | 0.998358 | 0.285716 | 0 |  |  |  |  |
| static_child_record_suite_v1_fullseed | direct_child_phase_carrier | scramble_continuation_child_phase |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| static_child_record_suite_v1_fullseed | direct_child_phase_carrier | continuation_parent_only |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| static_child_record_suite_v1_fullseed | gated_writeback_supported | disable_continuation_write |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| static_child_record_suite_v1_fullseed | gated_writeback_supported | continuation_write_only_readout |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| static_child_record_suite_v1_fullseed | child_local_ifs_supported | no_child_local_ifs |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| static_child_record_suite_v1_fullseed | coherence_retention_required | no_child_coherence_retention |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| static_child_record_suite_v1_fullseed | readout_only | disable_continuation_write |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| static_child_record_suite_v1_fullseed | readout_only | continuation_write_only_readout |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| static_child_record_suite_v1_fullseed | strict_direct_child_mix_carrier | strict_direct_mix_only |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| static_child_record_suite_v1_fullseed | strict_direct_child_mix_carrier | strict_write_only_no_preunroll |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| static_child_record_suite_v1_fullseed | phase_specific_under_budget_clamp | phase_roll_budget_clamped |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| static_child_record_suite_v1_fullseed | phase_specific_under_budget_clamp | phase_random_budget_clamped |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| static_child_record_suite_v1_fullseed | phase_specific_under_budget_clamp | phase_zero_budget_clamped |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| static_child_record_suite_v1_fullseed | phase_specific_under_budget_clamp | phase_cross_child_budget_clamped |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| static_child_record_suite_v1_fullseed | static_child_record_direct_carrier | static_child_record_direct_mix_only | True | 0.5 | 0.5 | 0 | 0 | 1 | 1 | 0.997092 | 0.343967 | 0 | True |  |  |  |
| static_child_record_suite_v1_fullseed | static_child_record_direct_carrier | static_child_record_write_only | False | 0 | 0.5 | 1 | 0.263051 | 1 | 1 | 1 | 0.150053 | 0 | True |  |  |  |
| static_child_record_suite_v1_fullseed | static_child_record_direct_carrier | static_child_record_stripped_control | False | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0.150051 | 0 | True |  |  |  |
| static_child_record_suite_v1_fullseed | child_record_presence_required | static_child_record_stripped_control | False | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0.150051 | 0 |  |  |  | True |
| static_child_record_suite_v1_fullseed | branch_preunroll_required | no_branch_pre_unroll |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| static_child_record_suite_v1_fullseed | branch_childworld_runtime_required | no_branch_childworld_runtime |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| static_child_record_suite_v1_fullseed | continuation_preunroll_required | no_continuation_pre_unroll |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| static_child_record_suite_v1_fullseed | strict_parent_no_child_context_breaks | strict_parent_no_child_context |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| static_child_mode_replace_suite_v1_fullseed | no_signal | positive_control | True | 0.5 | 0.5 | 1 | 0.118563 | 0.5 | 0.918532 | 0.998358 | 0.285716 | 0 |  |  |  |  |
| static_child_mode_replace_suite_v1_fullseed | direct_child_phase_carrier | scramble_continuation_child_phase |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| static_child_mode_replace_suite_v1_fullseed | direct_child_phase_carrier | continuation_parent_only |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| static_child_mode_replace_suite_v1_fullseed | gated_writeback_supported | disable_continuation_write |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| static_child_mode_replace_suite_v1_fullseed | gated_writeback_supported | continuation_write_only_readout |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| static_child_mode_replace_suite_v1_fullseed | child_local_ifs_supported | no_child_local_ifs |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| static_child_mode_replace_suite_v1_fullseed | coherence_retention_required | no_child_coherence_retention |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| static_child_mode_replace_suite_v1_fullseed | readout_only | disable_continuation_write |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| static_child_mode_replace_suite_v1_fullseed | readout_only | continuation_write_only_readout |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| static_child_mode_replace_suite_v1_fullseed | strict_direct_child_mix_carrier | strict_direct_mix_only |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| static_child_mode_replace_suite_v1_fullseed | strict_direct_child_mix_carrier | strict_write_only_no_preunroll |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| static_child_mode_replace_suite_v1_fullseed | phase_specific_under_budget_clamp | phase_roll_budget_clamped |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| static_child_mode_replace_suite_v1_fullseed | phase_specific_under_budget_clamp | phase_random_budget_clamped |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| static_child_mode_replace_suite_v1_fullseed | phase_specific_under_budget_clamp | phase_zero_budget_clamped |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| static_child_mode_replace_suite_v1_fullseed | phase_specific_under_budget_clamp | phase_cross_child_budget_clamped |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| static_child_mode_replace_suite_v1_fullseed | static_child_record_direct_carrier | static_child_record_direct_mix_only | True | 0.5 | 0.5 | 0 | 0 | 1 | 1 | 0.997092 | 0.343967 | 0 | True |  |  |  |
| static_child_mode_replace_suite_v1_fullseed | static_child_record_direct_carrier | static_child_record_write_only |  |  |  |  |  |  |  |  |  |  | True |  |  |  |
| static_child_mode_replace_suite_v1_fullseed | static_child_record_direct_carrier | static_child_record_stripped_control |  |  |  |  |  |  |  |  |  |  | True |  |  |  |
| static_child_mode_replace_suite_v1_fullseed | child_record_presence_required | static_child_record_stripped_control |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| static_child_mode_replace_suite_v1_fullseed | branch_preunroll_required | no_branch_pre_unroll |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| static_child_mode_replace_suite_v1_fullseed | branch_childworld_runtime_required | no_branch_childworld_runtime |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| static_child_mode_replace_suite_v1_fullseed | continuation_preunroll_required | no_continuation_pre_unroll |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| static_child_mode_replace_suite_v1_fullseed | strict_parent_no_child_context_breaks | strict_parent_no_child_context |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
