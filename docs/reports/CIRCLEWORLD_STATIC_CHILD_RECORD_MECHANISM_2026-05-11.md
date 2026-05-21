# Circleworld Static Child-Record Mechanism - 2026-05-11

## Result
The strongest current mechanism is not gated low-rank writeback and not child-local IFS recurrence.

Across the full seeded branch-active substrate (`9100/9101/9102`), the continuation-family `nested_sibling` signal is carried by the forked child phase record through final direct child continuation mix.

Best phrase:

> static/forked child phase record as a final direct continuation carrier, under seeded branch-active nested assay.

## Full Seeded Static-Record Suite
Checkpoint:
`D:\RAFA\checkpoints_circleworld_proto\continuation_causal_childifs_2026-05-11_guarded\circleworld_real_anchor_config_cem_v1.json`

Output root:
`D:\RAFA\outputs\circleworld_proto\continuation_causal_childifs_2026-05-11\static_child_record_suite_v1_fullseed`

| variant | continuation sibling | readout sibling | readiness max | qualified carry | budget retained | write enabled | write delta | world jump |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| positive_control | 0.5 | 0.5 | 1.0 | 0.5 | 0.9185317848375707 | 1.0 | 0.1185634754394866 | 0.0 |
| static_child_record_direct_mix_only | 0.5 | 0.5 | 1.0 | 1.0 | 1.0 | 0.0 | 0.0 | 0.0 |
| static_child_record_write_only | 0.0 | 0.5 | 0.4535136164393152 | 1.0 | 1.0 | 1.0 | 0.2630509799514513 | 0.0 |
| static_child_record_stripped_control | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |

## What This Proves
- The forked child record is sufficient for the current continuation-family label when used as a final direct continuation mix.
- Explicit low-rank writeback is not sufficient: the static write-only control writes a larger delta than positive control but loses the continuation label.
- Child record presence is required for this mechanism: stripping child records kills both readout and continuation labels.
- The result is stable across the current seeded cases, not only `9100`.

## What This Does Not Prove
- It does not prove recursive child-local IFS.
- It does not prove child worlds are self-evolving branch worlds.
- It does not prove parent-evolving writeback causality.
- It does not solve mode replacement: mode-replacement sibling fraction remains zero, including clamped and overdriven mode-1 replacement probes.
- It does not generalize beyond seeded branch-active starts yet.

## Mode-Replacement Follow-Up
The static child record was also tested as a mode-1 replacement source. The result stayed negative across the full seeded substrate.

Even with identity clamped, replacement gate floored to `1.0`, and replacement ratio scaled to `1.25`, `mode_replace_nested_sibling_fraction` stayed `0.0`.

Report:
`D:\RAFA\docs\reports\CIRCLEWORLD_STATIC_CHILD_MODE_REPLACE_OVERDRIVE_2026-05-11.md`

## Scientific Read
The previous question was:

Can branch-local packets become self-evolving child systems whose gated return changes the parent without destroying coarse identity?

Current answer:

Not yet. We have instead isolated a weaker but real substrate: the forked child record can serve as a phase-bearing continuation token inside the nested assay. It behaves like a semi-tokenless local phase token: it is not a symbolic token, and it is not magnitude-driven, but it is also not yet a self-recursing child world.

That distinction matters. The next training target should not reward final direct readout shortcuts as if they were childworld ontology. It should reward conversion of the static child phase record into parent mode state or mode-replacement sibling state.

## Next Tests
1. Static-record-to-parent conversion:
   require continuation sibling to survive with final direct child mix disabled.

2. Mode-replacement ladder:
   use the static child record as the source, but require mode 1 replacement to earn `nested_sibling`.

3. Learned branch/readout law:
   train the tiny learned law against positives from static direct carrier and negatives from static write-only, stripped-child, random-phase, and mutation-only controls.

4. Generic anchor reproduction:
   rerun the same static-record suite on non-seeded real-anchor starts.

5. Recursive childworld proof:
   isolated child continuation and parent-coupled child continuation must survive without final direct mix.

## Artifacts
- Full seeded summary: `D:\RAFA\outputs\circleworld_proto\continuation_causal_childifs_2026-05-11\static_child_record_suite_v1_fullseed\childworld_causality_killswitch_summary.json`
- Full seeded mechanism audit: `D:\RAFA\outputs\circleworld_proto\continuation_causal_childifs_2026-05-11\static_child_record_suite_v1_fullseed\childworld_mechanism_audit.json`
- Combined mechanism audit: `D:\RAFA\outputs\circleworld_proto\continuation_causal_childifs_2026-05-11\childworld_mechanism_audit_combined.json`
