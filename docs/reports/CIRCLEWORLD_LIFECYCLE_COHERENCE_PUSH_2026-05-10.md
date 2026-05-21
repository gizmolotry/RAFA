# Circleworld Lifecycle Coherence Push - 2026-05-10

## Decision

Hold. No new scout was better than `qualified_persistence_selected`.

The lifecycle-coherence run correctly fell back to the init checkpoint because no searched candidate preserved identity carry, divergence, and best-branch coherence/support at the same time.

Fallback checkpoint:

`D:\RAFA\checkpoints_circleworld_proto\lifecycle_coherence_push_2026-05-10\circleworld_real_anchor_config_cem_v1.json`

This is effectively the init reference from:

`D:\RAFA\checkpoints_circleworld_proto\qualified_persistence_push_2026-05-10\circleworld_real_anchor_config_cem_v1.json`

## What Changed

Added lifecycle metrics for the best identity branch:

- `best_identity_support_mean`
- `best_identity_coherence_mean`
- `best_identity_min_support`
- `best_identity_min_coherence`
- aggregate `mean_best_identity_*` fields
- trainer score/gate hooks for those fields

## Starting Point

The qualified-persistence scout had:

- best identity support mean: `0.232653`
- best identity coherence mean: `0.092736`
- best identity min support: `0.230919`
- best identity min coherence: `0.076996`
- max qualified carry: `0.25`
- max qualified steps: `0.0`

The child is close to the coherence threshold, but not across enough states.

## Training Outcome

- `selection_source=init_config_fallback_nested_or_agreement_required`
- `num_nested_gate_pass=0`
- all held-out gates passed for the evaluated candidates
- no candidate passed nested lifecycle gates

Top failed candidate:

| metric | value | target/meaning |
| --- | ---: | --- |
| live child start fraction | 1.000000 | preserved |
| identity carry | 0.583333 | failed `0.80` floor |
| max qualified carry | 0.250000 | preserved |
| max qualified steps | 0.000000 | still no persistence |
| best identity support | 0.222833 | barely preserved |
| best identity coherence | 0.081730 | below starting point and below target |
| best identity min coherence | 0.071813 | below starting point and below target |
| budget retained | 0.796776 | good |
| parent divergence | 0.063595 | failed |
| sibling divergence | 0.046170 | failed |

Interpretation: directly pushing coherence with the current CEM neighborhood did not solve qualified persistence. It tended to trade away carry/divergence rather than lift best-child coherence.

## Current Best State

Keep these as the useful scouts:

- branch-first scout: `D:\RAFA\checkpoints_circleworld_proto\fixed_substrate_guard_candidate00_2026-05-10\circleworld_real_anchor_config_cem_v1.json`
- ontology carry scout: `D:\RAFA\checkpoints_circleworld_proto\qualified_persistence_push_2026-05-10\circleworld_real_anchor_config_cem_v1.json`

Neither is a nested-sibling promotion.

## What We Learned

1. Fixed child substrate and same-child carry can be strengthened.
2. The current parametric/runtime search can raise carry, writeback, and branch-child count.
3. It does not easily raise best-child coherence above the qualification threshold.
4. The blocker is not just selection pressure; it is likely mechanistic in child update/coherence retention.
5. The next code change should modify child-local coherence lifecycle, not merely add more scoring weights.

## Next Technical Move

Patch the child-local IFS runtime to retain coherence more explicitly:

- add a config-gated child coherence retention term
- update child `mode_coherence` using a blend of previous coherence and child-local arc/promotion quality
- preserve support only where local packets remain active
- expose child-local coherence retention metrics in history
- keep default runtime behavior unchanged unless the config gate is enabled

This is more promising than another CEM-only pass because all candidate searches so far hit the same `max_qualified_steps=0` wall.

## Artifacts

- Compare JSON: `D:\RAFA\outputs\circleworld_proto\lifecycle_coherence_push_2026-05-10\lifecycle_coherence_compare.json`
- Training summary: `D:\RAFA\outputs\circleworld_proto\lifecycle_coherence_push_2026-05-10\training\train_summary.json`
- Init reference: `D:\RAFA\checkpoints_circleworld_proto\qualified_persistence_push_2026-05-10\circleworld_real_anchor_config_cem_v1.json`
