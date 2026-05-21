# Childworld Mechanism Audit

- Schema: `circleworld_childworld_mechanism_audit_v1`
- Overall read: `coherence_retention_required+direct_child_phase_carrier+readout_only`
- Suites: 3 total, 1 with positive signal

## Supported Class Counts

- `coherence_retention_required`: 1
- `direct_child_phase_carrier`: 1
- `readout_only`: 1

## Suite Decisions

| suite | mechanism_read | positive_cont | supported_classes |
| --- | --- | ---: | --- |
| positive_guarded | `direct_child_phase_carrier+coherence_retention_required+readout_only` | 0.5 | direct_child_phase_carrier, coherence_retention_required, readout_only |
| previous_causal_retention | `no_signal` | 0 | none |
| mutation_only_current | `no_signal` | 0 | none |

## Evidence

### positive_guarded

- `no_signal`: `not_supported` - positive control has a continuation-family nested sibling signal
- `direct_child_phase_carrier`: `supported` - scrambling child phase and/or parent-only rendering breaks the positive signal
- `gated_writeback_supported`: `not_supported` - signal does not require continuation writeback under this assay
- `child_local_ifs_supported`: `not_supported` - child-local IFS removal does not break the established signal
- `coherence_retention_required`: `supported` - disabling child-local coherence retention breaks the positive signal
- `readout_only`: `supported` - signal survives disabled writeback but breaks when direct child readout mix is removed

### previous_causal_retention

- `no_signal`: `supported` - positive control has no continuation-family nested sibling signal
- `direct_child_phase_carrier`: `blocked_no_signal` - direct child phase/readout kill-switch did not break an established signal
- `gated_writeback_supported`: `blocked_no_signal` - signal does not require continuation writeback under this assay
- `child_local_ifs_supported`: `blocked_no_signal` - child-local IFS removal does not break the established signal
- `coherence_retention_required`: `blocked_no_signal` - coherence-retention removal does not break the established signal
- `readout_only`: `blocked_no_signal` - readout-only carrier is not isolated by these kill-switches

### mutation_only_current

- `no_signal`: `supported` - positive control has no continuation-family nested sibling signal
- `direct_child_phase_carrier`: `blocked_no_signal` - direct child phase/readout kill-switch did not break an established signal
- `gated_writeback_supported`: `blocked_no_signal` - signal does not require continuation writeback under this assay
- `child_local_ifs_supported`: `blocked_no_signal` - child-local IFS removal does not break the established signal
- `coherence_retention_required`: `blocked_no_signal` - coherence-retention removal does not break the established signal
- `readout_only`: `blocked_no_signal` - readout-only carrier is not isolated by these kill-switches
