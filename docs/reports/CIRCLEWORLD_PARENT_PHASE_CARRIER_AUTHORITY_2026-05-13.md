# Circleworld Parent Phase Carrier Authority

Date: 2026-05-13

## Question

Can a learned non-oracle phase carrier replace direct child-phase authority in the parent-mode ontology path?

This follows the previous parent phase projector replay. That run showed a learned residual projector can predict part of the parent residual, but strict `ontology_delta` replay does not convert into parent-mode sibling ontology.

## Implementation

Added assay-only mechanisms:

- `learned_parent_bridge_carrier_coeffs`
- `learned_parent_bridge_carrier_feature_names`
- `parent_ontology_mode_replace_source="trained_carrier_map"`
- `disable_mode_replace_child_phase_carrier`
- `disable_parent_ontology_charge`
- `disable_parent_ontology_support_gate`
- `disable_parent_ontology_mode_logits`
- `disable_parent_ontology_mode_support`

Default runtime behavior is unchanged.

## Artifacts

- Carrier/projector model: `D:\RAFA\outputs\circleworld_proto\parent_phase_carrier_trained_2026-05-13\parent_phase_projector_model.json`
- Seeded replay: `D:\RAFA\outputs\circleworld_proto\parent_phase_carrier_replay_2026-05-13_seeded_suite`
- Held-out replay: `D:\RAFA\outputs\circleworld_proto\parent_phase_carrier_replay_2026-05-13_heldout_9110_9112`
- Held-out cases: `D:\RAFA\outputs\circleworld_proto\heldout_seeded_cases_9110_9112.json`

## Offline Fit

The residual projector remained useful:

| model | target | MAE | zero MAE | raw/low-rank MAE | reduction vs zero | reduction vs raw |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| residual projector | oracle parent residual | 0.181655 | 0.256645 | 0.331507 | 0.292194 | 0.452033 |
| carrier map | child-phase carrier delta | 0.155331 | 0.332349 | 0.202925 | 0.532627 | 0.234537 |

The carrier map is not a weak fit. It predicts the child-phase carrier delta better than the low-rank raw child delta baseline.

## Seeded Results

| variant | mode-replace sibling fraction | conversion score | positive residual reduction | ontology charge | carrier used | mode1 occupancy | world jump |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| trained projector, direct child-phase carrier | 0.5 | 0.499460 | 0.157150 | 0.139476 | 0.0 | 0.320988 | 0.0 |
| trained projector, ontology delta carrier | 0.0 | 0.0 | 0.157150 | 0.139476 | 0.0 | 0.320988 | 0.0 |
| trained carrier map | 0.0 | 0.0 | 0.157150 | 0.139476 | 1.0 | 0.320988 | 0.0 |
| oracle child-phase gate55 | 1.0 | 0.999276 | 0.366930 | 0.257410 | 0.0 | 0.654142 | 0.0 |
| no child-phase carrier | 0.0 | 0.0 | 0.366930 | 0.257410 | 0.0 | 0.654142 | 0.0 |
| no ontology charge | 0.0 | 0.0 | 0.366930 | 0.0 | 0.0 | 0.003515 | 0.0 |
| no ontology support gate | 0.5 | 0.499537 | 0.366930 | 0.257410 | 0.0 | 0.787749 | 0.0 |
| no mode logits | 0.0 | 0.0 | 0.366930 | 0.257410 | 0.0 | 0.014629 | 0.0 |

## Held-Out Results

Held-out seeds `9110/9111/9112` replicated the same pattern:

| variant | mode-replace sibling fraction | conversion score | positive residual reduction | ontology charge | carrier used | mode1 occupancy | world jump |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| trained projector, direct child-phase carrier | 0.5 | 0.499460 | 0.157151 | 0.139476 | 0.0 | 0.320988 | 0.0 |
| trained projector, ontology delta carrier | 0.0 | 0.0 | 0.157151 | 0.139476 | 0.0 | 0.320988 | 0.0 |
| trained carrier map | 0.0 | 0.0 | 0.157151 | 0.139476 | 1.0 | 0.320988 | 0.0 |
| oracle child-phase gate55 | 1.0 | 0.999276 | 0.366938 | 0.257416 | 0.0 | 0.654136 | 0.0 |
| no child-phase carrier | 0.0 | 0.0 | 0.366938 | 0.257416 | 0.0 | 0.654136 | 0.0 |
| no ontology charge | 0.0 | 0.0 | 0.366938 | 0.0 | 0.0 | 0.003515 | 0.0 |
| no ontology support gate | 0.5 | 0.499537 | 0.366938 | 0.257416 | 0.0 | 0.787752 | 0.0 |
| no mode logits | 0.0 | 0.0 | 0.366938 | 0.257416 | 0.0 | 0.014629 | 0.0 |

## Interpretation

The parent ontology pathway has four separable parts:

- predictive residual fit
- ontology charge
- child-phase carrier
- mode-logit authority

The decisive parts are not symmetric.

What is necessary:

- child-phase carrier
- ontology charge
- mode-logit authority

What is helpful but not solely necessary:

- ontology support-gate boost

What is currently insufficient:

- learned residual delta alone
- learned carrier map alone

This is the updated mechanistic claim:

> Parent-mode ontology conversion is not just a low-rank phase correction problem. It requires the parent to admit the live child phase as a carrier while simultaneously granting enough charge/logit authority for that carrier to become mode-1 ontology.

The trained carrier map failing despite strong offline fit is useful. It says the carrier is not merely a denoised version of low-rank child delta. The live child phase likely carries higher-order structure lost by the current low-rank feature basis.

## Next Question

The next target is not another scalar gain sweep.

The next target is a richer carrier learner that preserves more child-local geometry:

- train from phasor components, not scalar phase deltas only
- include local child arc/q/support features
- predict a unit-phasor carrier directly
- test blend modes between live child phase and learned carrier
- keep killswitch ladder active

This keeps the RAFA claim disciplined: no dense signature lane yet, no semantic projector, no audio-first promotion.
