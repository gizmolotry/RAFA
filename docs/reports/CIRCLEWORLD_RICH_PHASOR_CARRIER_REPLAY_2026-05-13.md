# Circleworld Rich Phasor Carrier Replay

Date: 2026-05-13

## Question

Does parent-mode ontology need the live child phase itself, or only a learned carrier that preserves the same local phasor geometry?

The prior carrier map learned a scalar phase-delta carrier and fit offline, but failed to convert. This run adds a richer phasor carrier that predicts a unit phasor directly from local parent/child phase geometry.

## Implementation

Added assay-only support for:

- `learned_parent_bridge_phasor_carrier_cos_coeffs`
- `learned_parent_bridge_phasor_carrier_sin_coeffs`
- `learned_parent_bridge_phasor_carrier_feature_names`
- `parent_ontology_mode_replace_source="trained_phasor_carrier_map"`
- `parent_ontology_mode_replace_source="blend_child_trained_phasor_carrier"`

The phasor carrier uses local features including:

- parent mode-1 phasor
- full raw child-to-parent phase delta
- sine/cosine of raw delta
- low-rank raw delta
- parent tangent
- support gate
- boundary mask
- trigonometric interaction terms needed to reconstruct a unit carrier

Default Circleworld runtime remains unchanged.

## Artifacts

- Model: `D:\RAFA\outputs\circleworld_proto\parent_phasor_carrier_trained_2026-05-13\parent_phase_projector_model.json`
- Seeded replay: `D:\RAFA\outputs\circleworld_proto\parent_phasor_carrier_replay_2026-05-13_seeded_suite`
- Held-out replay: `D:\RAFA\outputs\circleworld_proto\parent_phasor_carrier_replay_2026-05-13_heldout_9110_9112`

## Offline Fit

| model | target | headline |
| --- | --- | ---: |
| residual projector | oracle parent residual | `29.2%` MAE reduction vs zero |
| scalar carrier map | child phase delta | `53.3%` MAE reduction vs zero |
| rich phasor carrier | child unit phasor | `99.9936%` angular reduction vs parent |

The phasor carrier is effectively exact:

- angular MAE: `0.0000213` radians
- parent angular MAE: `0.332107` radians
- phasor L1: `4.77e-08`

This is expected because the feature basis now preserves the local trigonometric structure that the low-rank carrier destroyed.

## Seeded Replay

| variant | mode-replace sibling fraction | conversion score | binding | charge | carrier used | mode1 occupancy | world jump |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| trained projector delta | 0.0 | 0.0 | 0.019925 | 0.139476 | 0.0 | 0.320988 | 0.0 |
| scalar carrier map | 0.0 | 0.0 | 0.019925 | 0.139476 | 1.0 | 0.320988 | 0.0 |
| rich phasor carrier | 0.5 | 0.499460 | 0.019925 | 0.139476 | 1.0 | 0.320988 | 0.0 |
| blend child + rich phasor carrier | 0.5 | 0.499460 | 0.019925 | 0.139476 | 1.0 | 0.320988 | 0.0 |
| oracle child-phase gate55 | 1.0 | 0.999276 | 0.074451 | 0.257410 | 0.0 | 0.654142 | 0.0 |
| no child-phase carrier | 0.0 | 0.0 | 0.074451 | 0.257410 | 0.0 | 0.654142 | 0.0 |
| no charge | 0.0 | 0.0 | 0.074451 | 0.0 | 0.0 | 0.003515 | 0.0 |
| no logits | 0.0 | 0.0 | 0.074451 | 0.257410 | 0.0 | 0.014629 | 0.0 |

## Held-Out Replay

Held-out seeds `9110/9111/9112` replicated the seeded pattern:

| variant | mode-replace sibling fraction | conversion score | binding | charge | carrier used | mode1 occupancy | world jump |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| trained projector delta | 0.0 | 0.0 | 0.019925 | 0.139476 | 0.0 | 0.320988 | 0.0 |
| scalar carrier map | 0.0 | 0.0 | 0.019925 | 0.139476 | 1.0 | 0.320988 | 0.0 |
| rich phasor carrier | 0.5 | 0.499460 | 0.019925 | 0.139476 | 1.0 | 0.320988 | 0.0 |
| blend child + rich phasor carrier | 0.5 | 0.499460 | 0.019925 | 0.139476 | 1.0 | 0.320988 | 0.0 |
| oracle child-phase gate55 | 1.0 | 0.999276 | 0.074452 | 0.257416 | 0.0 | 0.654136 | 0.0 |
| no child-phase carrier | 0.0 | 0.0 | 0.074452 | 0.257416 | 0.0 | 0.654136 | 0.0 |
| no charge | 0.0 | 0.0 | 0.074452 | 0.0 | 0.0 | 0.003515 | 0.0 |
| no logits | 0.0 | 0.0 | 0.074452 | 0.257416 | 0.0 | 0.014629 | 0.0 |

## Interpretation

This isolates a sharper claim:

> The parent bridge does not merely require a learned scalar phase correction. It requires preservation of high-order local phasor geometry from the child carrier.

Evidence:

- scalar residual delta fails
- scalar carrier delta fails
- rich phasor carrier succeeds at the same partial level as the non-oracle direct child-phase carrier
- rich phasor carrier replicates on held-out seeded states

But the full oracle child-phase gate55 still wins:

- rich phasor carrier: `0.5`
- oracle child-phase gate55: `1.0`

The remaining gap is authority amplitude, not carrier geometry alone:

- non-oracle learned bridge binding: about `0.019925`
- oracle child-phase binding: about `0.07445`
- non-oracle charge: about `0.13948`
- oracle charge: about `0.25741`
- non-oracle mode1 occupancy: about `0.32099`
- oracle mode1 occupancy: about `0.65414`

## Updated Bottleneck

The missing piece is now more specific:

1. Carrier geometry: solved partially by rich phasor features.
2. Parent authority: still underpowered in non-oracle learned bridge.
3. Full conversion likely requires learned authority calibration, not just carrier reconstruction.

## Next Test

Train or fit a non-oracle authority head that predicts:

- binding mass
- ontology charge
- logit boost
- support gate floor

Use oracle child-phase gate55 as a calibration target, but evaluate with no oracle residual access at replay.
