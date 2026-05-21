# RAFA Causal Operator Routing Scale - 2026-05-15

## Scope

This cycle executed the next three steps after the first causal-router correction:

1. Scale causal operator label generation beyond the original three seeds.
2. Train a larger causal selector and compare it against membrane/recurrence routing.
3. Add an assay-only `geometric_score + causal_selector_residual` route in the operator causality harness.

All changes are assay-only. Circleworld default runtime is unchanged.

## Code Changes

- `D:\RAFA\runtimes\circleworld_proto\causal_operator_selector.py`
  - New shared causal selector module.
  - Defines `CausalOperatorSelector`, `PAIR_SCALAR_NAMES`, `causal_pair_feature_vector()`, and `causal_pair_feature_dim()`.

- `D:\RAFA\runtimes\circleworld_proto\run_resonant_operator_causality_assay.py`
  - Added optional `--causal-selector-checkpoint`.
  - Added `geometric_score + causal_selector_residual` shadow route.
  - Added causal route metrics and operator application side-by-side with geometric, recurrence, learned, and decoy routes.
  - Causal residual remains gated by identity/recurrence membrane; it is default-off unless a checkpoint is provided.

- `D:\RAFA\runtimes\circleworld_proto\test_resonant_attention_contract.py`
  - Added causal pair-feature contract test.

## Run 1: Larger Causal Label Bank

Artifact:

- `D:\RAFA\outputs\circleworld_proto\causal_operator_selector_2026-05-15_naked_gpu_9100_9111\CAUSAL_OPERATOR_SELECTOR_PROBE.md`
- `D:\RAFA\outputs\circleworld_proto\causal_operator_selector_2026-05-15_naked_gpu_9100_9111\causal_operator_selector_probe.json`
- `D:\RAFA\outputs\circleworld_proto\causal_operator_selector_2026-05-15_naked_gpu_9100_9111\causal_operator_selector_probe.pt`

Setup:

- Seeds: `9100-9111`
- Cases: `12`
- Candidate rows: `504`
- Feature dim: `116`
- Epochs: `300`
- Hidden dim: `128`

Metrics:

- Status: `pass_causal_operator_selector_probe`
- Final loss: `0.000042970536014763638`
- Mean holdout membrane target score: `0.7317542731262628`
- Mean holdout target-top score: `0.7377234402428642`
- Mean holdout membrane identity pass: `0.8571428571428571`
- Mean holdout same-local family: `0.2857142857142857`
- Mean holdout agreement with recurrence: `0.5595238095238094`

Interpretation:

Scaling from `126` to `504` candidate rows preserved the same target behavior. The causal selector remains identity-compatible on most held-out queries and stays close to the best target score. Agreement with recurrence dropped from the three-seed miniature, which is not necessarily bad: the causal target is not identical to recurrence. It is a parent-writeback target with identity and world-jump costs.

## Run 2: Causal Selector On Unseen Seeds With Membrane Preference

Artifact:

- `D:\RAFA\outputs\circleworld_proto\resonant_operator_causality_2026-05-15_naked_gpu_9112_9114_causal_selector\RESONANT_OPERATOR_CAUSALITY_ASSAY.md`
- `D:\RAFA\outputs\circleworld_proto\resonant_operator_causality_2026-05-15_naked_gpu_9112_9114_causal_selector\resonant_operator_causality_assay.json`

Setup:

- Causal selector trained on seeds `9100-9111`.
- Assay held out seeds: `9112, 9113, 9114`.
- Compared routes: geometric selected, recurrence selected, retrieval-learned selected, causal selected, wrong-family decoy.

Metrics:

- Status: `pass_operator_causality_present`
- Geometric/selected parent divergence: `0.2048563735110689`
- Recurrence parent divergence: `0.2048563735110689`
- Retrieval-learned parent divergence: `0.2048563735110689`
- Causal parent divergence: `0.2048563735110689`
- Decoy parent divergence: `0.19059907663367048`
- Causal model score: `0.7877008318901062`
- Causal hybrid score: `0.9464955430639637`
- Causal identity membrane pass: `1.0`
- Causal same-local family: `1.0`
- Causal agreement with recurrence: `1.0`

Interpretation:

On unseen seeds, the causal checkpoint chose the same identity-compatible operator as the recurrence/geometric route. This is safe but not yet a novel improvement. It shows the causal selector generalizes the membrane-compatible route.

## Run 3: Causal Residual-Only Control

Artifact:

- `D:\RAFA\outputs\circleworld_proto\resonant_operator_causality_2026-05-15_naked_gpu_9112_9114_causal_selector_residual_only\RESONANT_OPERATOR_CAUSALITY_ASSAY.md`
- `D:\RAFA\outputs\circleworld_proto\resonant_operator_causality_2026-05-15_naked_gpu_9112_9114_causal_selector_residual_only\resonant_operator_causality_assay.json`

Setup:

- Same held-out seeds `9112-9114`.
- Causal membrane preference disabled with `--causal-membrane-mode off`.
- Residual is still only applied to membrane-passing candidates.

Metrics:

- Causal parent divergence: `0.2048563735110689`
- Decoy parent divergence: `0.19059907663367048`
- Causal world-jump proxy: `0.02294657989758971`
- Decoy world-jump proxy: `0.022781314368852112`
- Causal geometric score: `0.8964955430639637`
- Causal residual: `0.05`
- Causal hybrid score: `0.9464955430639637`
- Causal identity membrane pass: `1.0`
- Causal same-local family: `1.0`
- Causal agreement with recurrence: `1.0`

Interpretation:

This is the important positive result of the cycle.

The wrong-family decoy has a higher raw geometric score than the identity-compatible causal candidate, but the learned causal residual raises the identity-compatible candidate above the decoy. That means the causal selector is doing useful work beyond the membrane preference. It is not merely being forced by the ranking guard.

The gain is still modest:

- Causal minus decoy parent divergence: `0.014257296877398407`
- Causal minus decoy world-jump proxy: `0.00016526552873759778`

But the direction is correct: causal learned residual favors the branch-authorized operator over the wrong-family decoy.

## Current Scientific Status

Supported now:

- Causal operator labels scale from `126` to `504` candidate rows without breaking.
- A causal selector trained on seeds `9100-9111` generalizes to unseen seeds `9112-9114`.
- `geometric_score + causal_selector_residual` can promote the identity-compatible operator above a higher-geometry wrong-family decoy.
- Learned routing remains safe only because residuals are membrane-gated.

Still not supported:

- Runtime promotion.
- Audio-facing improvement.
- Nested sibling emergence.
- Learned branch law replacing parametric branch law.
- Pure unsupervised ontology discovery.

## Interpretation Compared To Prior Retrieval Embedding

The retrieval embedding was a good address space but unsafe as an operator router. The causal selector is different: it is trained on what happens when a candidate writes into the parent.

That is the conceptual movement:

- Retrieval embedding: learns resonant similarity.
- Causal selector: learns writeback authority.

RAFA needs both, but they cannot be the same head.

## Next Step

The next cycle should scale difficulty, not just seed count:

1. Generate causal labels over multiple configs and grandchild depths.
2. Include stronger negative controls where decoys have higher movement but worse identity/world-jump behavior.
3. Train a causal selector that predicts a full vector: movement, world-jump risk, identity carry, recurrence compatibility, and final route score.
