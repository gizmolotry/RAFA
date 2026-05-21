# Circleworld Static Child Mode-Replacement Overdrive - 2026-05-11

## Result
Static child-record mode replacement is still negative.

The child record can carry final direct continuation mix, but it does not become a mode-replacement sibling under the current assay, even when support gates are clamped or overdriven.

## Full Seeded Overdrive Suite
Checkpoint:
`D:\RAFA\checkpoints_circleworld_proto\continuation_causal_childifs_2026-05-11_guarded\circleworld_real_anchor_config_cem_v1.json`

Output root:
`D:\RAFA\outputs\circleworld_proto\continuation_causal_childifs_2026-05-11\static_child_mode_replace_overdrive_suite_v1_fullseed`

| variant | continuation sibling | mode-replace sibling | mode readiness max | replace gate | replace ratio | static record | world jump |
|---|---:|---:|---:|---:|---:|---:|---:|
| positive_control | 0.5 | 0.0 | 0.5721780390895064 | 0.23047144214312235 | 0.6750000000000002 | 0.0 | 0.0 |
| static_child_record_mode1_replace | 0.0 | 0.0 | 0.45350163718909114 | 0.2310436616341273 | 0.6750000000000002 | 1.0 | 0.0 |
| static_child_record_mode1_replace_clamped | 0.0 | 0.0 | 0.4535016372255427 | 0.5066686272621155 | 0.6750000000000002 | 1.0 | 0.0 |
| static_child_record_mode1_replace_gate_floor | 0.0 | 0.0 | 0.45350163699626217 | 0.8818248510360718 | 0.6750000000000002 | 1.0 | 0.0 |
| static_child_record_mode1_replace_overdrive | 0.0 | 0.0 | 0.4535016426804712 | 1.0 | 0.8125 | 1.0 | 0.0 |
| static_child_record_mode1_replace_stripped_control | 0.0 | 0.0 | 0.0 | 0.0 | 0.6750000000000002 | 1.0 | 0.0 |

## Interpretation
This rules out the easy explanation that mode replacement failed only because the support mask was too weak. Raising the mean replace gate from about `0.23` to `1.0` does not move `mode_replace_nested_sibling_fraction` above `0.0`.

The current carrier is therefore not simply "child phase in any parent channel." It is specifically effective as final direct continuation mix. When pushed into parent mode 1 and rendered through the weighted parent readout, it becomes too parent-like: q correlation stays almost exactly `1.0`, readout response stays near `0.15005`, and the branch does not earn `nested_sibling`.

## Scientific Consequence
The next objective should not merely strengthen mode replacement. It should create a new conversion loss:

- preserve the static child record's child-phase distinctiveness;
- inject it into parent mode state;
- keep coarse world/q identity;
- increase response above readout-only baseline without final direct child mix.

That is the missing bridge from "phase-bearing child token" to "real child world or sibling mode."

## Artifacts
- Overdrive summary: `D:\RAFA\outputs\circleworld_proto\continuation_causal_childifs_2026-05-11\static_child_mode_replace_overdrive_suite_v1_fullseed\childworld_causality_killswitch_summary.json`
- Combined conversion report: `D:\RAFA\outputs\circleworld_proto\continuation_causal_childifs_2026-05-11\static_child_mode_replace_conversion_combined.json`
- Dataset v2: `D:\RAFA\outputs\circleworld_proto\continuation_causal_childifs_2026-05-11\childworld_mechanism_dataset_v2.json`
