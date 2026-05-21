# Circleworld Parent Prewrite Policy - 2026-05-13

## Question

Can the parent-ontology bridge encode the prewrite discovery as an explicit policy rather than relying on an accidental `delta_scale=0` setting?

## Implementation

Added assay-only prewrite policy controls:

- `learned_parent_bridge_prewrite_policy="scale"` preserves prior behavior.
- `learned_parent_bridge_prewrite_policy="disabled"` blocks the parent prewrite residual.
- `learned_parent_bridge_prewrite_policy="carrier_only"` names the intended carrier/authority bridge path.

Added metrics:

- `learned_parent_bridge_prewrite_gate`
- `learned_parent_bridge_effective_delta_scale`
- aggregate/family rollups for both metrics

This makes the result auditable: the same learned bridge can now expose whether it actually wrote a parent prewrite residual before mode replacement.

## Seeded Replay

| variant | sibling | conversion | prewrite gate | effective delta | binding | charge | mode1 occ | jump |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| trained phasor + authority + prewrite on | 0.5000 | 0.499636 | 1.0 | 1.0 | 0.074486 | 0.257743 | 0.655421 | 0.0 |
| trained phasor + authority + prewrite disabled | 1.0000 | 0.999275 | 0.0 | 0.0 | 0.074486 | 0.257743 | 0.655394 | 0.0 |
| trained phasor + authority + carrier_only policy | 1.0000 | 0.999275 | 0.0 | 0.0 | 0.074486 | 0.257743 | 0.655394 | 0.0 |
| oracle child phase | 1.0000 | 0.999276 | 0.0 | 0.0 | 0.074451 | 0.257410 | 0.654142 | 0.0 |

## Held-Out Replay

| variant | sibling | conversion | prewrite gate | effective delta | binding | charge | mode1 occ | jump |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| trained phasor + authority + prewrite on | 0.5000 | 0.499636 | 1.0 | 1.0 | 0.074485 | 0.257737 | 0.655423 | 0.0 |
| trained phasor + authority + prewrite disabled | 1.0000 | 0.999275 | 0.0 | 0.0 | 0.074485 | 0.257737 | 0.655396 | 0.0 |
| trained phasor + authority + carrier_only policy | 1.0000 | 0.999275 | 0.0 | 0.0 | 0.074485 | 0.257737 | 0.655396 | 0.0 |
| oracle child phase | 1.0000 | 0.999276 | 0.0 | 0.0 | 0.074452 | 0.257416 | 0.654136 | 0.0 |

## Interpretation

The explicit policy reproduces the earlier isolation exactly.

The learned carrier/authority bridge reaches oracle-scale parent-mode ontology conversion when parent prewrite is disabled by policy. The prewrite-enabled path still collapses to half conversion despite matching binding, charge, and mode occupancy.

This is now a clean mechanistic separation:

- carrier/authority law: useful for parent ontology
- scalar parent prewrite law: currently harmful to ontology when applied before mode replacement

The important conceptual point is that one-step residual prediction is not the same as branch ontology compatibility. The residual can lower a local parent error while damaging the downstream sibling continuation operation.

## Recommended Next Baseline

Use this as the next assay baseline:

`parent_ontology_trained_phasor_authority_carrier_only_gate55_static_mode_replace`

It is non-oracle in the child-to-parent bridge, reproduces on held-out seeded cases, has no world-jump, and matches the oracle child-phase conversion score within numerical noise.

## Artifacts

- Seeded replay: `D:\RAFA\outputs\circleworld_proto\parent_prewrite_policy_replay_2026-05-13_seeded_suite\childworld_volume_recursion_sweep_summary.json`
- Held-out replay: `D:\RAFA\outputs\circleworld_proto\parent_prewrite_policy_replay_2026-05-13_heldout_9110_9112\childworld_volume_recursion_sweep_summary.json`
