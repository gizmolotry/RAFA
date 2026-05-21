# Circleworld Continuation-Causal Kill-Switch Suite - 2026-05-11

## Result
The kill-switch suite gives a mixed but much sharper read.

The guarded checkpoint's continuation sibling is **not** dependent on the explicit low-rank `continuation_mode1_write` path. It is dependent on the child phase/direct continuation carrier and on coherence-retention/budget preservation.

Therefore the strong phrase "gated writeback causality" is too strong for this result. The better phrase is:

> child-phase continuation carrier with coherence-retained identity, under seeded branch-active nested assay.

## Positive Guarded Checkpoint
Checkpoint:
`D:\RAFA\checkpoints_circleworld_proto\continuation_causal_childifs_2026-05-11_guarded\circleworld_real_anchor_config_cem_v1.json`

| variant | readout sibling | continuation sibling | continuation readiness max | continuation budget | write enabled | write delta | world jump |
|---|---:|---:|---:|---:|---:|---:|---:|
| positive_control | 0.5 | 0.5 | 1.0 | 0.9185317848375707 | 1.0 | 0.1185634754394866 | 0.0 |
| disable_continuation_write | 0.5 | 0.5 | 1.0 | 0.9185317848375707 | 0.0 | 0.0 | 0.0 |
| continuation_parent_only | 0.5 | 0.0 | 0.5721695715297598 | 0.9185317848375707 | 0.0 | 0.0 | 0.0 |
| continuation_write_only_readout | 0.5 | 0.0 | 0.5721798768796496 | 0.9185317848375707 | 1.0 | 0.1185634754394866 | 0.0 |
| scramble_continuation_child_phase | 0.5 | 0.0 | 0.9630921498436829 | 0.9185317848375707 | 1.0 | 0.3026786611407033 | 0.0 |
| no_child_local_ifs | 0.5 | 0.5 | 1.0 | 0.7068803436799236 | 1.0 | 0.16592241964690047 | 0.0 |
| no_child_coherence_retention | 0.0 | 0.0 | 1.0 | 0.47537528179843314 | 1.0 | 0.11834526264674729 | 0.0 |
| no_child_causal_gate | 0.5 | 0.5 | 1.0 | 0.8735403725562718 | 1.0 | 0.11822456314301084 | 0.0 |

## Previous Causal-Retention Comparator
Checkpoint:
`D:\RAFA\checkpoints_circleworld_proto\causal_retention_childifs_2026-05-10\circleworld_real_anchor_config_cem_v1.json`

| variant | readout sibling | continuation sibling | continuation readiness max | continuation budget | write enabled | write delta | world jump |
|---|---:|---:|---:|---:|---:|---:|---:|
| positive_control | 0.5 | 0.0 | 0.9662262424196698 | 0.7129725796724659 | 1.0 | 0.10936024340334054 | 0.0 |
| disable_continuation_write | 0.5 | 0.0 | 0.9662262424196698 | 0.7129725796724659 | 0.0 | 0.0 | 0.0 |
| continuation_parent_only | 0.5 | 0.0 | 0.509837810788966 | 0.7558017454382084 | 0.0 | 0.0 | 0.0 |
| continuation_write_only_readout | 0.5 | 0.0 | 0.5098504830966809 | 0.7129725796724659 | 1.0 | 0.10936024340334054 | 0.0 |
| scramble_continuation_child_phase | 0.5 | 0.0 | 0.9662262424196698 | 0.7129725796724659 | 1.0 | 0.26839722973147556 | 0.0 |
| no_child_local_ifs | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| no_child_coherence_retention | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |

## Mutation-Only Comparator
Checkpoint:
`D:\RAFA\checkpoints_circleworld_proto\continuation_causal_childifs_2026-05-11\circleworld_real_anchor_config_cem_v1.json`

| variant | readout sibling | continuation sibling | continuation readiness max | continuation budget | write enabled | write delta | world jump |
|---|---:|---:|---:|---:|---:|---:|---:|
| positive_control | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| disable_continuation_write | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| continuation_parent_only | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| continuation_write_only_readout | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| scramble_continuation_child_phase | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| no_child_local_ifs | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| no_child_coherence_retention | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |

## What The Kill Switches Say

1. Disabling explicit continuation mode-1 write does not break the continuation sibling.
   - Positive control continuation sibling: `0.5`
   - Disable-write continuation sibling: `0.5`
   - Write enabled falls to `0.0`.

2. Parent-only continuation breaks the continuation sibling.
   - Parent-only continuation sibling: `0.0`

3. Write-only parent readout also breaks the continuation sibling.
   - Write-only continuation sibling: `0.0`
   - This is the key falsifier for "low-rank writeback alone carries the branch."

4. Scrambling the child phase breaks the continuation sibling.
   - Scrambled-child continuation sibling: `0.0`
   - Response can remain high, but the ontology label does not survive.

5. Disabling child-local IFS does not break the continuation sibling in this assay.
   - No-child-local-IFS continuation sibling: `0.5`

6. Disabling child coherence retention breaks the label.
   - No-retention continuation sibling: `0.0`
   - Budget falls to `0.47537528179843314`.

7. Disabling the causal coherence gate does not break the continuation sibling.
   - No-causal-gate continuation sibling: `0.5`

## Scientific Interpretation
The result is not leakage in the trivial sense: parent-only and scrambled-child controls kill continuation labels, while readout labels remain separable. The guarded checkpoint is also not just the previous checkpoint: previous causal-retention remains readout-only under the same kill-switch instrumentation.

But the result is not the strongest version of the May 11 hypothesis either. The explicit low-rank mode-1 write path is not necessary. Child-local IFS is also not necessary under this assay. The branch carrier is better understood as the child phase being directly mixed into the continuation branch while coherence retention preserves identity enough to satisfy the ontology label.

So the honest claim is narrower:

- Supported: a child-derived phase carrier can produce continuation-family `nested_sibling` under seeded branch-active assay.
- Supported: the label depends on child phase and coherence-retained identity.
- Not supported: explicit low-rank gated writeback is necessary.
- Not supported: child-local IFS recurrence is necessary for this label.
- Still absent: mode-replacement siblings.

## Decision
Status: `mixed_child_phase_continuation_supported_writeback_not_necessary`.

Do not promote this as a full child-local IFS/gated-writeback causal proof. Do preserve it as a major branch-ontology scout because it cleanly distinguishes readout-only previous baseline, no-signal mutation-only checkpoint, and child-phase-dependent continuation siblings.

## Strict Static-Record Follow-Up
A stricter follow-up isolated the final continuation carrier from branch pre-unroll, branch childworld runtime, final continuation pre-unroll, and explicit continuation writeback.

Full seeded substrate: `naked_seed_9100`, `naked_seed_9101`, `naked_seed_9102`.

| variant | continuation sibling | continuation readiness max | qualified carry | budget retained | write enabled | write delta | world jump |
|---|---:|---:|---:|---:|---:|---:|---:|
| positive_control | 0.5 | 1.0 | 0.5 | 0.9185317848375707 | 1.0 | 0.1185634754394866 | 0.0 |
| static_child_record_direct_mix_only | 0.5 | 1.0 | 1.0 | 1.0 | 0.0 | 0.0 | 0.0 |
| static_child_record_write_only | 0.0 | 0.4535136164393152 | 1.0 | 1.0 | 1.0 | 0.2630509799514513 | 0.0 |
| static_child_record_stripped_control | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |

This tightens the mechanism again:

- Supported: the forked child record can carry the continuation-family label through final direct child continuation mix even when branch childworld runtime, branch pre-unroll, final continuation pre-unroll, and explicit writeback are disabled.
- Supported: the child record itself is required under this static-carrier control, because stripping child records kills both readout and continuation labels.
- Not supported: explicit write from the same static child record is sufficient. It writes a larger delta than positive control but still produces no continuation sibling.

The current best phrase is therefore:

> static/forked child phase record as a final direct continuation carrier, under seeded branch-active nested assay.

That is sharper than the earlier "direct readout or leakage" phrase, but still narrower than true recursive childworld ontology.

## Artifacts
- Compare JSON: `D:\RAFA\outputs\circleworld_proto\continuation_causal_childifs_2026-05-11\causality_killswitch_suite_v1\continuation_causal_killswitch_compare.json`
- Positive suite root: `D:\RAFA\outputs\circleworld_proto\continuation_causal_childifs_2026-05-11\causality_killswitch_suite_v1\positive_guarded_full`
- Previous comparator root: `D:\RAFA\outputs\circleworld_proto\continuation_causal_childifs_2026-05-11\causality_killswitch_suite_v1\previous_causal_retention_focused`
- Mutation-only comparator root: `D:\RAFA\outputs\circleworld_proto\continuation_causal_childifs_2026-05-11\causality_killswitch_suite_v1\mutation_only_current_focused`
- Static-record full seeded suite: `D:\RAFA\outputs\circleworld_proto\continuation_causal_childifs_2026-05-11\static_child_record_suite_v1_fullseed`
- Combined mechanism audit: `D:\RAFA\outputs\circleworld_proto\continuation_causal_childifs_2026-05-11\childworld_mechanism_audit_combined.json`
