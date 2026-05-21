# Childworld Mechanism Audit

- Schema: `circleworld_childworld_mechanism_audit_v1`
- Overall read: `child_record_presence_required+coherence_retention_required+direct_child_phase_carrier+phase_specific_under_budget_clamp+readout_only+static_child_record_direct_carrier+strict_direct_child_mix_carrier`
- Suites: 5 total, 3 with positive signal

## Supported Class Counts

- `child_record_presence_required`: 1
- `coherence_retention_required`: 1
- `direct_child_phase_carrier`: 1
- `phase_specific_under_budget_clamp`: 1
- `readout_only`: 1
- `static_child_record_direct_carrier`: 1
- `strict_direct_child_mix_carrier`: 1

## Suite Decisions

| suite | mechanism_read | positive_cont | supported_classes |
| --- | --- | ---: | --- |
| positive_guarded | `direct_child_phase_carrier+coherence_retention_required+readout_only` | 0.5 | direct_child_phase_carrier, coherence_retention_required, readout_only |
| previous_causal_retention | `no_signal` | 0 | none |
| mutation_only_current | `no_signal` | 0 | none |
| strict_mechanism_suite_v1_singlecase | `strict_direct_child_mix_carrier+phase_specific_under_budget_clamp` | 0.5 | strict_direct_child_mix_carrier, phase_specific_under_budget_clamp |
| static_child_record_suite_v1_fullseed | `static_child_record_direct_carrier+child_record_presence_required` | 0.5 | static_child_record_direct_carrier, child_record_presence_required |

## Evidence

### positive_guarded

- `no_signal`: `not_supported` - positive control has a continuation-family nested sibling signal
- `direct_child_phase_carrier`: `supported` - scrambling child phase and/or parent-only rendering breaks the positive signal
- `gated_writeback_supported`: `not_supported` - signal does not require continuation writeback under this assay
- `child_local_ifs_supported`: `not_supported` - child-local IFS removal does not break the established signal
- `coherence_retention_required`: `supported` - disabling child-local coherence retention breaks the positive signal
- `readout_only`: `supported` - signal survives disabled writeback but breaks when direct child readout mix is removed
- `strict_direct_child_mix_carrier`: `not_evaluated` - strict direct-mix/write-only controls do not isolate direct child mix as sufficient
- `phase_specific_under_budget_clamp`: `not_evaluated` - budget-clamped child phase substitution does not break an established signal
- `static_child_record_direct_carrier`: `not_evaluated` - static forked child record does not carry the continuation signal by itself
- `child_record_presence_required`: `not_evaluated` - stripping child records does not break the signal or was not sufficient to test it
- `branch_preunroll_required`: `not_evaluated` - branch-level child pre-unroll is not required by this assay
- `branch_childworld_runtime_required`: `not_evaluated` - branch childworld runtime is not required by this assay
- `continuation_preunroll_required`: `not_evaluated` - final continuation pre-unroll is not required by this assay
- `strict_parent_no_child_context_breaks`: `not_evaluated` - strict parent/no-child context does not break the signal

### previous_causal_retention

- `no_signal`: `supported` - positive control has no continuation-family nested sibling signal
- `direct_child_phase_carrier`: `blocked_no_signal` - direct child phase/readout kill-switch did not break an established signal
- `gated_writeback_supported`: `blocked_no_signal` - signal does not require continuation writeback under this assay
- `child_local_ifs_supported`: `blocked_no_signal` - child-local IFS removal does not break the established signal
- `coherence_retention_required`: `blocked_no_signal` - coherence-retention removal does not break the established signal
- `readout_only`: `blocked_no_signal` - readout-only carrier is not isolated by these kill-switches
- `strict_direct_child_mix_carrier`: `not_evaluated` - strict direct-mix/write-only controls do not isolate direct child mix as sufficient
- `phase_specific_under_budget_clamp`: `not_evaluated` - budget-clamped child phase substitution does not break an established signal
- `static_child_record_direct_carrier`: `not_evaluated` - static forked child record does not carry the continuation signal by itself
- `child_record_presence_required`: `not_evaluated` - stripping child records does not break the signal or was not sufficient to test it
- `branch_preunroll_required`: `not_evaluated` - branch-level child pre-unroll is not required by this assay
- `branch_childworld_runtime_required`: `not_evaluated` - branch childworld runtime is not required by this assay
- `continuation_preunroll_required`: `not_evaluated` - final continuation pre-unroll is not required by this assay
- `strict_parent_no_child_context_breaks`: `not_evaluated` - strict parent/no-child context does not break the signal

### mutation_only_current

- `no_signal`: `supported` - positive control has no continuation-family nested sibling signal
- `direct_child_phase_carrier`: `blocked_no_signal` - direct child phase/readout kill-switch did not break an established signal
- `gated_writeback_supported`: `blocked_no_signal` - signal does not require continuation writeback under this assay
- `child_local_ifs_supported`: `blocked_no_signal` - child-local IFS removal does not break the established signal
- `coherence_retention_required`: `blocked_no_signal` - coherence-retention removal does not break the established signal
- `readout_only`: `blocked_no_signal` - readout-only carrier is not isolated by these kill-switches
- `strict_direct_child_mix_carrier`: `not_evaluated` - strict direct-mix/write-only controls do not isolate direct child mix as sufficient
- `phase_specific_under_budget_clamp`: `not_evaluated` - budget-clamped child phase substitution does not break an established signal
- `static_child_record_direct_carrier`: `not_evaluated` - static forked child record does not carry the continuation signal by itself
- `child_record_presence_required`: `not_evaluated` - stripping child records does not break the signal or was not sufficient to test it
- `branch_preunroll_required`: `not_evaluated` - branch-level child pre-unroll is not required by this assay
- `branch_childworld_runtime_required`: `not_evaluated` - branch childworld runtime is not required by this assay
- `continuation_preunroll_required`: `not_evaluated` - final continuation pre-unroll is not required by this assay
- `strict_parent_no_child_context_breaks`: `not_evaluated` - strict parent/no-child context does not break the signal

### strict_mechanism_suite_v1_singlecase

- `no_signal`: `not_supported` - positive control has a continuation-family nested sibling signal
- `direct_child_phase_carrier`: `not_evaluated` - direct child phase/readout kill-switch did not break an established signal
- `gated_writeback_supported`: `not_evaluated` - signal does not require continuation writeback under this assay
- `child_local_ifs_supported`: `not_evaluated` - child-local IFS removal does not break the established signal
- `coherence_retention_required`: `not_evaluated` - coherence-retention removal does not break the established signal
- `readout_only`: `not_evaluated` - readout-only carrier is not isolated by these kill-switches
- `strict_direct_child_mix_carrier`: `supported` - direct child continuation mix preserves the signal while strict write-only/no-preunroll fails
- `phase_specific_under_budget_clamp`: `supported` - budget-clamped child phase substitution breaks the continuation signal
- `static_child_record_direct_carrier`: `not_evaluated` - static forked child record does not carry the continuation signal by itself
- `child_record_presence_required`: `not_evaluated` - stripping child records does not break the signal or was not sufficient to test it
- `branch_preunroll_required`: `not_evaluated` - branch-level child pre-unroll is not required by this assay
- `branch_childworld_runtime_required`: `not_evaluated` - branch childworld runtime is not required by this assay
- `continuation_preunroll_required`: `not_evaluated` - final continuation pre-unroll is not required by this assay
- `strict_parent_no_child_context_breaks`: `not_evaluated` - strict parent/no-child context does not break the signal

### static_child_record_suite_v1_fullseed

- `no_signal`: `not_supported` - positive control has a continuation-family nested sibling signal
- `direct_child_phase_carrier`: `not_evaluated` - direct child phase/readout kill-switch did not break an established signal
- `gated_writeback_supported`: `not_evaluated` - signal does not require continuation writeback under this assay
- `child_local_ifs_supported`: `not_evaluated` - child-local IFS removal does not break the established signal
- `coherence_retention_required`: `not_evaluated` - coherence-retention removal does not break the established signal
- `readout_only`: `not_evaluated` - readout-only carrier is not isolated by these kill-switches
- `strict_direct_child_mix_carrier`: `not_evaluated` - strict direct-mix/write-only controls do not isolate direct child mix as sufficient
- `phase_specific_under_budget_clamp`: `not_evaluated` - budget-clamped child phase substitution does not break an established signal
- `static_child_record_direct_carrier`: `supported` - static forked child record can carry direct continuation mix without branch childworld runtime/pre-unroll
- `child_record_presence_required`: `supported` - stripping child records breaks the static direct-carrier control
- `branch_preunroll_required`: `not_evaluated` - branch-level child pre-unroll is not required by this assay
- `branch_childworld_runtime_required`: `not_evaluated` - branch childworld runtime is not required by this assay
- `continuation_preunroll_required`: `not_evaluated` - final continuation pre-unroll is not required by this assay
- `strict_parent_no_child_context_breaks`: `not_evaluated` - strict parent/no-child context does not break the signal
