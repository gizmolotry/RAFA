# Childworld Mechanism Audit

- Schema: `circleworld_childworld_mechanism_audit_v1`
- Overall read: `child_record_presence_required+static_child_record_direct_carrier`
- Suites: 1 total, 1 with positive signal

## Supported Class Counts

- `child_record_presence_required`: 1
- `static_child_record_direct_carrier`: 1

## Suite Decisions

| suite | mechanism_read | positive_cont | supported_classes |
| --- | --- | ---: | --- |
| static_child_record_suite_v1_singlecase | `static_child_record_direct_carrier+child_record_presence_required` | 0.5 | static_child_record_direct_carrier, child_record_presence_required |

## Evidence

### static_child_record_suite_v1_singlecase

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
