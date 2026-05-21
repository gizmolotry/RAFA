# Circleworld Parent Phase Projector Replay

Date: 2026-05-13

## Question

Can the child-to-parent bridge become non-oracle by learning a small parent phase residual projector from seeded childworld assay tensors?

## Implementation

- Added assay-only parent phase projector tensor export in `test_nested_commitment.py`.
- Added `trained_linear_projector` replay source for `learned_parent_bridge`.
- Added `train_parent_phase_projector.py`, a tiny ridge scout that fits parent residual deltas from exported tensor rows.
- Added sweep variants for trained projector replay and strict `ontology_delta` controls.

Default Circleworld runtime and checkpoints remain unchanged.

## Artifacts

- Dataset sweep: `D:\RAFA\outputs\circleworld_proto\parent_phase_projector_dataset_2026-05-13_seeded_suite`
- Trained projector: `D:\RAFA\outputs\circleworld_proto\parent_phase_projector_trained_2026-05-13\parent_phase_projector_model.json`
- Replay sweep: `D:\RAFA\outputs\circleworld_proto\parent_phase_projector_replay_2026-05-13_seeded_suite`
- Strict delta sweep: `D:\RAFA\outputs\circleworld_proto\parent_phase_projector_delta_strict_2026-05-13_seeded_suite`

## Dataset

The exporter wrote 300 NPZ sidecars from seeded `naked_rafa` cases `9100/9101/9102`.

Each sidecar stores compact features plus full context tensors:

- child phase
- parent mode 0 / mode 1
- baseline and target mode 1
- raw child delta
- parent tangent delta
- oracle residual target
- support gate and boundary mask

## Projector Fit

The ridge projector is not decorative. It learns a stable residual direction:

| split | projector MAE | zero MAE | raw MAE | parent tangent MAE | reduction vs zero | reduction vs raw | reduction vs tangent |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| all | 0.181655 | 0.256645 | 0.331507 | 0.278928 | 0.292194 | 0.452033 | 0.348738 |
| leave-one 9100 | 0.181889 | 0.257372 | 0.331499 | 0.279612 | 0.293283 | 0.451313 | 0.349494 |
| leave-one 9101 | 0.181580 | 0.256243 | 0.331730 | 0.278555 | 0.291374 | 0.452627 | 0.348135 |
| leave-one 9102 | 0.181498 | 0.256320 | 0.331293 | 0.278616 | 0.291910 | 0.452153 | 0.348574 |

Interpretation: the parent residual is learnable from local child/parent phase features. This is the first concrete bridge from parametric/oracle child authority toward a learned non-oracle bridge.

## Replay Result

`parent_ontology_trained_projector_gate55_static_mode_replace`:

- mode-replace nested sibling fraction: `0.5`
- mode-replace conversion score: `0.499460`
- child predictive positive residual reduction: `0.157150`
- oracle residual reduction ceiling: `0.365416`
- world-jump penalty: `0.0`
- qualified identity carry: `1.0`

Branch-level read:

- `child_mode1_replace_50_shift` becomes `nested_sibling` in all three seeded cases.
- `child_mode1_replace_85_shift` remains `ambiguous_middle`.

This is a real non-oracle replay improvement over the old zero-delta learned bridge, but it is not yet a clean proof that the learned residual itself creates parent ontology.

## Critical Control

Strict `ontology_delta` replay:

| variant | mode-replace sibling fraction | conversion score | positive residual reduction |
| --- | ---: | ---: | ---: |
| trained projector, child-phase mode replace | 0.5 | 0.499460 | 0.157150 |
| trained projector, ontology-delta mode replace | 0.0 | 0.0 | 0.157150 |
| oracle delta, ontology-delta mode replace | 0.0 | 0.0 | 0.366930 |
| oracle child-phase gate55 | 1.0 | 0.999276 | 0.366930 |

This isolates the current mechanism:

- Learned projector residuals help predict parent phase targets.
- Residual prediction alone does not make the parent mode ontology convert.
- The successful route is parent-granted authority over child phase under the support/ontology gate.
- The `ontology_delta` path is too weak or geometrically mismatched for the current nested label/readout system, even when the delta is oracle.

## Scientific Read

The bottleneck is no longer simply volume, recursion, or scalar branch pressure.

The new claim is narrower:

> A child-local system can earn parent authority when the parent grants child-phase carrier status under a strong ontology gate. A learned low-rank residual projector can approximate the oracle parent residual, but the current parent-mode ontology path does not yet know how to turn that residual alone into sibling continuation.

That matters because it prevents a false victory. The learned projector is useful, but the parent ontology converter is still using child phase as the carrier.

## Next Work

1. Train/evaluate a projector whose output is a child-phase carrier blend, not only a residual delta.
2. Add a strict causality ladder that separately disables child-phase carrier, ontology charge, support gate, and mode-logit authority.
3. Replace `ontology_delta` mode replacement with a learned carrier map: `(child_phase, residual_delta, support_gate, parent_mode1) -> replacement_phase`.
