# RAFA Causal Operator Routing Review - 2026-05-15

## Scope

This cycle executed the next three steps after the learned-selector negative result:

1. Add a recurrence/identity membrane before learned residual can influence operator routing.
2. Train a causal operator selector on actual candidate writeback outcomes.
3. Rerun learned shadow selection and review the last five testing runs.

Everything remains assay-only. Circleworld runtime defaults and checkpoint promotion are unchanged.

## Code Changes

- `D:\RAFA\runtimes\circleworld_proto\run_resonant_operator_causality_assay.py`
  - Added learned selector membrane fields:
    - `--learned-membrane-mode`
    - `--learned-strong-recurrence-threshold`
    - `--learned-strong-geometric-threshold`
  - Learned residual is now applied only after identity/recurrence membrane pass.
  - Learned selector ranking can prefer membrane-passing candidates.
  - Reports membrane pass, strong recurrence pass, same-local, same-structural, learned-selected agreement, and learned-decoy gaps.

- `D:\RAFA\runtimes\circleworld_proto\run_causal_operator_selector_probe.py`
  - New assay-only causal router probe.
  - Sweeps every non-self candidate operator for each law-object query.
  - Applies candidate writeback and records parent phase divergence, world-jump proxy, recurrence, geometry, and identity.
  - Trains a small MLP on causal operator targets rather than retrieval targets.
  - Exports JSON, Markdown, and `.pt` checkpoint.

- `D:\RAFA\runtimes\circleworld_proto\test_resonant_attention_contract.py`
  - Added contract test ensuring non-membrane decoys are capped in causal operator targets.

## New Run 1: Causal Operator Selector Probe

Artifact:

- `D:\RAFA\outputs\circleworld_proto\causal_operator_selector_2026-05-15_naked_gpu_9100_9102\CAUSAL_OPERATOR_SELECTOR_PROBE.md`
- `D:\RAFA\outputs\circleworld_proto\causal_operator_selector_2026-05-15_naked_gpu_9100_9102\causal_operator_selector_probe.json`
- `D:\RAFA\outputs\circleworld_proto\causal_operator_selector_2026-05-15_naked_gpu_9100_9102\causal_operator_selector_probe.pt`

Metrics:

- Status: `pass_causal_operator_selector_probe`
- Cases: `3`
- Candidate rows: `126`
- Feature dim: `116`
- Final training loss: `0.00009776327715371735`
- Mean holdout membrane target score: `0.7307437432665386`
- Mean holdout target-top score: `0.7377234059748957`
- Mean holdout membrane identity pass: `0.8571428571428571`
- Mean holdout membrane same-local family: `0.2857142857142857`
- Mean holdout membrane agreement with recurrence: `0.7619047619047619`

Interpretation:

Training on actual writeback outcomes works better conceptually than retrieval-only training. The causal selector does not just ask “which law object is close?” It asks whether candidate writeback produces safe movement under a branch identity membrane.

This is not promotion-ready. The dataset is tiny: `126` candidate rows from three seeds. But it is the correct target shape.

## New Run 2: Retrieval-Learned Selector With Identity Membrane

Artifact:

- `D:\RAFA\outputs\circleworld_proto\resonant_operator_causality_2026-05-15_naked_gpu_9100_9102_learned_selector_membrane\RESONANT_OPERATOR_CAUSALITY_ASSAY.md`
- `D:\RAFA\outputs\circleworld_proto\resonant_operator_causality_2026-05-15_naked_gpu_9100_9102_learned_selector_membrane\resonant_operator_causality_assay.json`

Metrics:

- Status: `pass_operator_causality_present`
- Mean selected/geometric parent divergence: `0.20485464095324304`
- Mean recurrence parent divergence: `0.20485464095324304`
- Mean learned parent divergence: `0.20485464095324304`
- Mean decoy parent divergence: `0.19059757054429358`
- Mean selected/geometric world-jump proxy: `0.022946314769144178`
- Mean learned world-jump proxy: `0.022946314769144178`
- Mean decoy world-jump proxy: `0.02278108191261352`
- Learned selector identity membrane pass: `1.0`
- Learned selector same-local family: `1.0`
- Learned selector agrees with selected: `1.0`
- Learned selector agrees with recurrence: `1.0`

Interpretation:

The retrieval-trained learned selector becomes safe only when a membrane is imposed. The membrane prevents the learned residual from routing to a wrong-family decoy. That validates the RAFA attention constitution: learned residual can steer, but geometry and identity must retain authority.

## Full Review: Last Five Testing Runs

### 1. Contrastive Structural Embedding Probe

Artifact:

- `D:\RAFA\outputs\circleworld_proto\contrastive_structural_embedding_2026-05-14_naked_gpu_9100_9102_holdout_9102\CONTRASTIVE_STRUCTURAL_EMBEDDING_PROBE.md`

Result:

- Holdout recurrence top1: `1.0`
- Post-hoc structural-family top1: `0.989547`
- Local leakage: `0.0`

What we learned:

The law-object bank has a learnable recurrence structure. A small embedding trained without `structural_family_key` can recover cross-seed structural relation almost perfectly. This supports resonant memory retrieval.

Limit:

Retrieval closeness is not the same as operator authority.

### 2. Learned Selector As Operator Router, No Effective Membrane

Artifact:

- `D:\RAFA\outputs\circleworld_proto\resonant_operator_causality_2026-05-14_naked_gpu_9100_9102_learned_selector\RESONANT_OPERATOR_CAUSALITY_ASSAY.md`

Result:

- Learned selector picked the same object as wrong-family decoy in all three seeds.
- Learned same-local family: `0.0`
- Learned same-structural family: `0.0`
- Learned agreement with recurrence: `0.0`

What we learned:

A retrieval embedding can be actively unsafe for parent writeback. It knows something about recurrence geometry, but not enough about branch authority.

Limit:

This run also revealed learned residual saturation: learned residual could swamp the geometric score.

### 3. Corrected Non-Self Operator Assay, Still Without Membrane Success

Artifact:

- `D:\RAFA\outputs\circleworld_proto\resonant_operator_causality_2026-05-14_naked_gpu_9100_9102_learned_selector_nonself_floor\RESONANT_OPERATOR_CAUSALITY_ASSAY.md`

Result:

- Geometric selected and recurrence selected agreed.
- Learned selector still chose the wrong-family decoy.
- Selected parent divergence: `0.20485464095324304`
- Decoy parent divergence: `0.19059757054429358`
- Selected-decoy phase gap: `0.014257070408949487`

What we learned:

The earlier selected-vs-decoy result was partially muddied by self-selection. After fixing non-self selection, operator causality remains nonzero but more modest. This made the assay more honest.

Limit:

The learned retrieval selector still failed as a router.

### 4. Causal Operator Selector Probe

Artifact:

- `D:\RAFA\outputs\circleworld_proto\causal_operator_selector_2026-05-15_naked_gpu_9100_9102\CAUSAL_OPERATOR_SELECTOR_PROBE.md`

Result:

- Candidate rows: `126`
- Mean holdout membrane target score: `0.7307437432665386`
- Mean holdout target-top score: `0.7377234059748957`
- Mean holdout membrane identity pass: `0.8571428571428571`
- Mean holdout agreement with recurrence: `0.7619047619047619`

What we learned:

Causal labels are the right supervision target for operator routing. The selector can learn to prefer candidates that move parent state while respecting an identity membrane.

Limit:

Tiny dataset. This is a proof of target shape, not a robust learned branch law.

### 5. Learned Selector With Identity/Recurrence Membrane

Artifact:

- `D:\RAFA\outputs\circleworld_proto\resonant_operator_causality_2026-05-15_naked_gpu_9100_9102_learned_selector_membrane\RESONANT_OPERATOR_CAUSALITY_ASSAY.md`

Result:

- Learned selector agreement with selected: `1.0`
- Learned selector agreement with recurrence: `1.0`
- Learned identity membrane pass: `1.0`
- Learned same-local family: `1.0`
- Learned parent divergence: `0.20485464095324304`
- Decoy parent divergence: `0.19059757054429358`

What we learned:

A membrane can make retrieval-trained learned residual safe as a shadow steering term. Without the membrane, learned selection is unsafe. With the membrane, it becomes subordinate to identity-compatible routing.

Limit:

This does not prove the learned residual adds useful new choices yet. It mostly proves the guard works.

## Current Scientific Status

Supported:

- Cross-seed law-object retrieval is learnable from RAFA recurrence geometry.
- Retrieval embeddings are not automatically safe operator routers.
- Operator causality is nonzero beyond inert control.
- Non-self selected/recurrence operators modestly beat wrong-family decoys on parent divergence.
- A recurrence/identity membrane prevents learned routing from collapsing into wrong-family decoy selection.
- Causal operator targets are a better next training objective than retrieval-only targets.

Not yet supported:

- Learned branch law promotion.
- Runtime operator routing promotion.
- Nested sibling emergence.
- Audio-facing continuity improvement.
- Pure unsupervised ontology discovery.

## Next Move

The next cycle should scale causal operator labels:

1. Generate causal candidate rows from more seeds, depths, and configs.
2. Train the causal selector on those labels, not just the three-seed miniature set.
3. Add an assay-only `geometric_score + causal_selector_residual` router and compare it against recurrence, membrane-only, and wrong-family decoy controls.
