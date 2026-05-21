# Circleworld Parent Authority Bridge Replay - 2026-05-13

## Addendum

This report's immediate "carrier expressivity/branch selection" interpretation was superseded by the follow-up prewrite isolation run in `CIRCLEWORLD_PARENT_PREWRITE_CARRIER_ISOLATION_2026-05-13.md`.

The follow-up showed that trained phasor carrier + trained authority reaches full parent-mode conversion when the learned parent prewrite delta is disabled. The remaining gap was therefore the learned prewrite residual corrupting downstream mode replacement, not insufficient carrier expressivity.

## Question

Can a learned, non-oracle authority head convert the rich phasor carrier into parent-mode ontology, or was the remaining gap after the phasor carrier replay just insufficient permission/binding?

## Implementation

- Added an assay-only trained authority path in `test_nested_commitment.py`.
- Added `learned_parent_bridge_authority_source="trained_linear_authority"`.
- Added authority feature/coeff keys without changing default runtime behavior.
- Extended `train_parent_phase_projector.py` to fit a scalar authority head against `max(0, oracle_residual_reduction) * boundary_match`.
- Added explicit replay variants in `run_childworld_volume_recursion_sweep.py`:
  - `parent_ontology_trained_phasor_authority_gate55_static_mode_replace`
  - `parent_ontology_blend_phasor_authority_gate55_static_mode_replace`

## Authority Fit

The authority head is narrow and assay-only. It does not learn the phasor carrier; it learns how much parent authority to grant from local bridge features.

- target acceptance mean: `0.374510`
- authority MAE: `0.014886`
- leave-one-case authority MAE: `~0.01587`
- reduction vs static bridge acceptance: `~0.95`

## Seeded Replay

| variant | sibling | conversion | binding | accepted | auth_pred | charge | mode1_occ | jump |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| phasor carrier, hand authority | 0.5000 | 0.499460 | 0.019925 | 0.102391 | 0.350458 | 0.139476 | 0.320988 | 0.0 |
| phasor carrier, trained authority | 0.5000 | 0.499636 | 0.074486 | 0.350458 | 0.350458 | 0.257743 | 0.655421 | 0.0 |
| blend phasor, trained authority | 0.5000 | 0.499636 | 0.074486 | 0.350458 | 0.350458 | 0.257743 | 0.655421 | 0.0 |
| oracle child phase + oracle authority | 1.0000 | 0.999276 | 0.074451 | 0.000000 | 0.000000 | 0.257410 | 0.654142 | 0.0 |
| oracle authority, no carrier | 0.0000 | 0.000000 | 0.074451 | 0.000000 | 0.000000 | 0.257410 | 0.654142 | 0.0 |
| oracle carrier, no charge | 0.0000 | 0.000000 | 0.074451 | 0.000000 | 0.000000 | 0.000000 | 0.003515 | 0.0 |
| oracle carrier, no logits | 0.0000 | 0.000000 | 0.074451 | 0.000000 | 0.000000 | 0.257410 | 0.014629 | 0.0 |

## Held-Out Replay

| variant | sibling | conversion | binding | accepted | auth_pred | charge | mode1_occ | jump |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| phasor carrier, hand authority | 0.5000 | 0.499460 | 0.019925 | 0.102391 | 0.350449 | 0.139476 | 0.320988 | 0.0 |
| phasor carrier, trained authority | 0.5000 | 0.499636 | 0.074485 | 0.350449 | 0.350449 | 0.257737 | 0.655423 | 0.0 |
| blend phasor, trained authority | 0.5000 | 0.499636 | 0.074485 | 0.350449 | 0.350449 | 0.257737 | 0.655423 | 0.0 |
| oracle child phase + oracle authority | 1.0000 | 0.999276 | 0.074452 | 0.000000 | 0.000000 | 0.257416 | 0.654136 | 0.0 |
| oracle authority, no carrier | 0.0000 | 0.000000 | 0.074452 | 0.000000 | 0.000000 | 0.257416 | 0.654136 | 0.0 |
| oracle carrier, no charge | 0.0000 | 0.000000 | 0.074452 | 0.000000 | 0.000000 | 0.000000 | 0.003515 | 0.0 |
| oracle carrier, no logits | 0.0000 | 0.000000 | 0.074452 | 0.000000 | 0.000000 | 0.257416 | 0.014629 | 0.0 |

## Interpretation

The authority hypothesis is partly confirmed but not sufficient.

Trained authority raises learned-bridge binding from `~0.0199` to `~0.0745`, matching the oracle child-phase path. It also raises parent ontology charge and mode-1 occupancy to the oracle scale, and it replicates on held-out seeds without producing world jumps.

But parent-mode sibling fraction stays at `0.5`, while direct oracle child-phase authority remains `1.0`. That means the remaining failure is not mainly authority magnitude anymore. The bottleneck has moved to carrier expressivity/branch selection: the learned phasor carrier can replay enough child geometry for half the cases, but it still does not reproduce the full child-phase carrier needed for the other half.

The kill switches remain decisive:

- no carrier: no conversion
- no charge: no conversion
- no logits: no conversion

So the parent ontology route needs all three pieces: carrier geometry, ontology charge, and mode-logit authority.

## Next Claim

The next falsifiable claim is:

> Parent-mode ontology conversion requires a child-phase carrier that is both locally phasor-expressive and branch-selective; scalar low-rank residuals and scalar authority are insufficient even when binding/charge/occupancy match oracle levels.

## Artifacts

- Model: `D:\RAFA\outputs\circleworld_proto\parent_authority_bridge_trained_2026-05-13\parent_phase_projector_model.json`
- Seeded replay: `D:\RAFA\outputs\circleworld_proto\parent_authority_bridge_replay_2026-05-13_seeded_suite\childworld_volume_recursion_sweep_summary.json`
- Held-out replay: `D:\RAFA\outputs\circleworld_proto\parent_authority_bridge_replay_2026-05-13_heldout_9110_9112\childworld_volume_recursion_sweep_summary.json`
