# Static Child Conversion Report

- Overall classification: `mixed`
- Suites: 2 total, 2 with positive signal

## Classification Counts

- `direct_only`: 1
- `mixed`: 1

## Suite Summary

| suite | class | positive_cont | direct | write | stripped_signal | mode_replace |
| --- | --- | ---: | --- | --- | --- | --- |
| static_child_mode_replace_suite_v1_fullseed | `direct_only` | 0.5 | True | False | False | False |
| static_child_mode_replace_overdrive_suite_v1_fullseed | `mixed` | 0.5 | False | False | False | False |

## Focused Variants

### static_child_mode_replace_suite_v1_fullseed

- Classification: `direct_only`
- Rationale: static direct mix preserves the signal while static write does not
- `direct` `static_child_record_direct_mix_only`: cont=0.5, mode_replace=0, mode_ready=0.453503, mode_gate=0.228753, mode_ratio=0.675, read=`signal_survives_kill_switch`, has_signal=True
- `write`: missing
- `stripped`: missing
- `mode_replace` `static_child_record_mode1_replace`: cont=0, mode_replace=0, mode_ready=0.453502, mode_gate=0.231044, mode_ratio=0.675, read=`kill_switch_breaks_signal`, has_signal=False
- `mode_replace` `static_child_record_mode1_replace_stripped_control`: cont=0, mode_replace=0, mode_ready=0, mode_gate=0, mode_ratio=0.675, read=`kill_switch_breaks_signal`, has_signal=False
- `mode_replace` `static_child_record_mode1_replace_random_phase`: cont=0, mode_replace=0, mode_ready=0.453504, mode_gate=0.506669, mode_ratio=0.675, read=`kill_switch_breaks_signal`, has_signal=False

### static_child_mode_replace_overdrive_suite_v1_fullseed

- Classification: `mixed`
- Rationale: positive signal is present but static-child variants are missing, conflicting, or inconclusive
- `direct`: missing
- `write`: missing
- `stripped`: missing
- `mode_replace` `static_child_record_mode1_replace`: cont=0, mode_replace=0, mode_ready=0.453502, mode_gate=0.231044, mode_ratio=0.675, read=`kill_switch_breaks_signal`, has_signal=False
- `mode_replace` `static_child_record_mode1_replace_stripped_control`: cont=0, mode_replace=0, mode_ready=0, mode_gate=0, mode_ratio=0.675, read=`kill_switch_breaks_signal`, has_signal=False
- `mode_replace` `static_child_record_mode1_replace_clamped`: cont=0, mode_replace=0, mode_ready=0.453502, mode_gate=0.506669, mode_ratio=0.675, read=`kill_switch_breaks_signal`, has_signal=False
- `mode_replace` `static_child_record_mode1_replace_gate_floor`: cont=0, mode_replace=0, mode_ready=0.453502, mode_gate=0.881825, mode_ratio=0.675, read=`kill_switch_breaks_signal`, has_signal=False
- `mode_replace` `static_child_record_mode1_replace_overdrive`: cont=0, mode_replace=0, mode_ready=0.453502, mode_gate=1, mode_ratio=0.8125, read=`kill_switch_breaks_signal`, has_signal=False
