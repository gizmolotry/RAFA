# RAFA Shadow Branch-Law Calibration Pass - 2026-05-15

## Summary

This pass evaluated the learned shadow branch-law checkpoint on fresh raw causality assay rows before table aggregation.

The checkpoint correctly separated fresh positive depth-2 writeback rows from fresh negative depth-3 writeback-denial rows. This supports the next narrow claim: the learned shadow branch law can reproduce the current safe/unsafe writeback boundary on unseen seeds within the same nearby ontology families.

It does not yet prove distant generalization or runtime utility.

## Code Changes

- Added `evaluate_shadow_branch_law_calibration.py`.
- Added a calibration contract for converting learned branch-law outputs back into survival/collapse/writeback fields.
- Added threshold sweep reporting so writeback classification is separated from writeback-score calibration.
- No Circleworld runtime behavior changed.

## Fresh Causality Inputs

### Positive slice: carrier config, depth 2, seeds 9114-9116

Artifact:

- `D:\RAFA\outputs\circleworld_proto\resonant_operator_causality_2026-05-15_calib_carrier_9114_9116_d2\resonant_operator_causality_assay.json`

Metrics:

- cases: `3`
- law objects per case: `11`
- causal parent divergence: `0.18161292640342638`
- decoy parent divergence: `0.19017280141026904`
- causal world jump: `0.016863476222536183`
- decoy world jump: `0.021716969685621906`
- hard decoy against causal: `1.0`
- causal safety win over decoy: `1.0`
- causal authority win over decoy: `1.0`

Interpretation: depth 2 reproduces the positive safe-writeback regime. The decoy moves more but jumps worse, so branch-authorized writeback remains permitted.

### Negative slice: authority config, depth 3, seeds 9114-9116

Artifact:

- `D:\RAFA\outputs\circleworld_proto\resonant_operator_causality_2026-05-15_calib_authority_9114_9116_d3\resonant_operator_causality_assay.json`

Metrics:

- cases: `3`
- law objects per case: `15`
- causal parent divergence: `0.236907064944013`
- decoy parent divergence: `0.19059862746102227`
- causal world jump: `0.03070985751231899`
- decoy world jump: `0.022781218769796625`
- hard decoy against causal: `0.0`
- causal safety win over decoy: `0.0`
- causal authority win over decoy: `0.0`

Interpretation: depth 3 reproduces the negative unsafe-writeback regime. The causal route moves more but also jumps more, so writeback is denied.

## Calibration Result

Artifact:

- `D:\RAFA\outputs\circleworld_proto\shadow_branch_law_calibration_2026-05-15_fresh_d2_d3\shadow_branch_law_calibration.json`

Checkpoint:

- `D:\RAFA\outputs\circleworld_proto\shadow_learned_branch_law_2026-05-15_expanded_d2_d3\shadow_rafa_learned_branch_law_v0.pt`

Aggregate:

- rows: `6`
- sources: `2`
- writeback threshold: `0.78`
- writeback accuracy: `1.0`
- false permit rate: `0.0`
- false deny rate: `0.0`
- mean positive writeback score: `0.877267994483312`
- mean negative writeback score: `0.7057099990546704`
- writeback score margin: `0.1715579954286417`
- best threshold: `0.78`
- best threshold accuracy: `1.0`
- mean output absolute error: `0.00026478942329088897`
- max output absolute error: `0.0016405280709692605`
- mean survival error: `0.0004090376251792797`
- mean collapse error: `0.0004011721821179833`
- mean support-delta error: `0.000026362016797065735`

Row behavior:

- depth-2 rows target writeback `1.0`, predicted writeback `1.0`, writeback score about `0.8773`
- depth-3 rows target writeback `0.0`, predicted writeback `0.0`, writeback score about `0.7057`

## Interpretation

This is a useful calibration result. The learned shadow branch law can now classify fresh safe and unsafe writeback regimes using raw causality assay rows before table aggregation.

The important caveat: the writeback score is a decision score, not a calibrated probability. The negative rows score around `0.706`, which is not close to zero. The threshold boundary works here, but we should not interpret the score as confidence yet.

## Claim Status

Supported:

- Learned shadow branch-law fields reproduce fresh measured survival/collapse/writeback surfaces on unseen seeds.
- Depth-2 and depth-3 regimes remain separable by survival/collapse/writeback criteria.
- The current threshold `0.78` cleanly separates positive and negative fresh rows.

Not yet supported:

- Distant config generalization.
- Runtime branch-law improvement.
- Audio improvement.
- Calibrated probabilistic writeback confidence.

## Verification

- `python -m pytest -q D:\RAFA\runtimes\circleworld_proto\test_learned_branch_law_child_ifs_contract.py` -> `6 passed`
- `python -m py_compile D:\RAFA\runtimes\circleworld_proto\evaluate_shadow_branch_law_calibration.py` -> pass

## Next Move

The next useful move is a sandbox runtime intervention: use the learned branch-law checkpoint only as a gate on child writeback permission, compare learned-gated versus ungated child-local IFS on parent divergence, world-jump, and nested ontology labels. Keep it assay-only and reversible.
