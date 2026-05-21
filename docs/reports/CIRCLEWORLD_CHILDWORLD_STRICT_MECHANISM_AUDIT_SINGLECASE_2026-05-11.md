# Childworld Mechanism Audit

- Schema: `circleworld_childworld_mechanism_audit_v1`
- Overall read: `phase_specific_under_budget_clamp+strict_direct_child_mix_carrier`
- Suites: 1 total, 1 with positive signal

## Supported Class Counts

- `phase_specific_under_budget_clamp`: 1
- `strict_direct_child_mix_carrier`: 1

## Suite Decisions

| suite | mechanism_read | positive_cont | supported_classes |
| --- | --- | ---: | --- |
| strict_mechanism_suite_v1_singlecase | `strict_direct_child_mix_carrier+phase_specific_under_budget_clamp` | 0.5 | strict_direct_child_mix_carrier, phase_specific_under_budget_clamp |

## Evidence

### strict_mechanism_suite_v1_singlecase

- `no_signal`: `not_supported` - positive control has a continuation-family nested sibling signal
- `direct_child_phase_carrier`: `not_evaluated` - direct child phase/readout kill-switch did not break an established signal
- `gated_writeback_supported`: `not_evaluated` - signal does not require continuation writeback under this assay
- `child_local_ifs_supported`: `not_evaluated` - child-local IFS removal does not break the established signal
- `coherence_retention_required`: `not_evaluated` - coherence-retention removal does not break the established signal
- `readout_only`: `not_evaluated` - readout-only carrier is not isolated by these kill-switches
- `strict_direct_child_mix_carrier`: `supported` - direct child continuation mix preserves the signal while strict write-only/no-preunroll fails
- `phase_specific_under_budget_clamp`: `supported` - budget-clamped child phase substitution breaks the continuation signal
- `branch_preunroll_required`: `not_evaluated` - branch-level child pre-unroll is not required by this assay
- `branch_childworld_runtime_required`: `not_evaluated` - branch childworld runtime is not required by this assay
- `continuation_preunroll_required`: `not_evaluated` - final continuation pre-unroll is not required by this assay
- `strict_parent_no_child_context_breaks`: `not_evaluated` - strict parent/no-child context does not break the signal
