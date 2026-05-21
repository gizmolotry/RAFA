# RAFA Resonant Memory Contrastive Structural Embedding - 2026-05-14

## Scope

This cycle implements the next step after the learned structural-family pseudo-label probe.

The previous learned probe used `structural_family_key` as the teacher. That was useful but still partly hand-bucketed. This probe removes that teacher from training and instead learns an embedding from cross-case recurrence compatibility and an operator-safety proxy.

This is assay-only. Circleworld runtime behavior, branch law behavior, and checkpoint promotion are unchanged.

## Code Changes

- `D:\RAFA\runtimes\circleworld_proto\run_contrastive_structural_embedding_probe.py`
  - New recurrence-trained contrastive embedding probe.
  - Builds positives and decoys from continuous recurrence compatibility, not `structural_family_key`.
  - Trains a small normalized embedding model plus safety head.
  - Exports JSON, Markdown, and `.pt` checkpoint.

- `D:\RAFA\runtimes\circleworld_proto\test_resonant_attention_contract.py`
  - Added a bounded-surface contract test for `recurrence_compatibility()` and `operator_safety_proxy()`.

## Teacher Definition

Training teacher:

- positive relation: cross-case objects with high recurrence compatibility.
- negative relation: cross-case objects with lower recurrence compatibility or hard near misses.
- safety target: continuous operator-safety proxy from support, coherence, boundary, residue, writeback budget, and causal-use readiness.

Not used for training:

- `structural_family_key`.
- `local_family_key`.
- human labels.

Post-hoc evaluation still reports structural-family agreement as a yardstick, because it tells us whether recurrence-derived training recovers the relation previously found by deterministic structural scoping.

## Synthetic Smoke

- Output: `D:\RAFA\outputs\circleworld_proto\contrastive_structural_embedding_2026-05-14_synthetic_smoke\CONTRASTIVE_STRUCTURAL_EMBEDDING_PROBE.md`
- Checkpoint: `D:\RAFA\outputs\circleworld_proto\contrastive_structural_embedding_2026-05-14_synthetic_smoke\contrastive_structural_embedding_probe.pt`
- Objects: `18`
- Pair count: `60`
- Positive pairs: `24`
- Negative pairs: `36`
- Holdout top1 recurrence accuracy: `1.0`
- Holdout post-hoc structural-family accuracy: `1.0`
- Holdout local leakage: `0.0`
- Mean embedding margin: `1.3829874843358994`
- Mean absolute safety error: `0.0670991837978363`

Interpretation:

The training/evaluation machinery works on controlled synthetic law objects. This is a smoke test only; it does not establish real RAFA ontology.

## Naked RAFA Leave-One-Seed Results

Input substrate:

- `D:\RAFA\outputs\circleworld_proto\resonant_grandchild_composition_2026-05-14_naked_gpu_9100_9102_postfix`

Each fold trains on two naked_rafa seeded cases and holds out the third.

| Holdout | Status | Objects | Pairs | Positives | Negatives | Holdout Queries | Top1 Recurrence | Post-hoc Structural Top1 | Local Leakage | Mean Margin | Safety Error |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `naked_seed_9100` | `pass_contrastive_recurrence_signal` | `1722` | `6854` | `3428` | `3426` | `574` | `1.0` | `0.989547` | `0.0` | `0.166874` | `0.025478` |
| `naked_seed_9101` | `pass_contrastive_recurrence_signal` | `1722` | `6854` | `3428` | `3426` | `574` | `1.0` | `0.989547` | `0.0` | `0.153077` | `0.023220` |
| `naked_seed_9102` | `pass_contrastive_recurrence_signal` | `1722` | `6854` | `3428` | `3426` | `574` | `1.0` | `0.989547` | `0.0` | `0.137548` | `0.019386` |

Artifacts:

- `D:\RAFA\outputs\circleworld_proto\contrastive_structural_embedding_2026-05-14_naked_gpu_9100_9102_holdout_9100\CONTRASTIVE_STRUCTURAL_EMBEDDING_PROBE.md`
- `D:\RAFA\outputs\circleworld_proto\contrastive_structural_embedding_2026-05-14_naked_gpu_9100_9102_holdout_9101\CONTRASTIVE_STRUCTURAL_EMBEDDING_PROBE.md`
- `D:\RAFA\outputs\circleworld_proto\contrastive_structural_embedding_2026-05-14_naked_gpu_9100_9102_holdout_9102\CONTRASTIVE_STRUCTURAL_EMBEDDING_PROBE.md`

## Interpretation

This is the strongest memory-substrate result so far, with an important caveat.

The result supports this narrower claim:

A reusable cross-seed structural relation can be learned from recurrence-compatible q/arc/support/lifecycle/operator geometry without training directly on the deterministic `structural_family_key` teacher.

Why this matters:

- Local child identity remained case-bound in earlier retrieval.
- Deterministic structural scope recovered a reusable relation.
- The pseudo-label learned probe showed that relation was learnable, but from a hand-bucket teacher.
- This contrastive probe shows the relation can also be recovered from continuous recurrence compatibility.

This moves the substrate closer to RAFA resonance memory: law objects are being addressed by compatible recurrence geometry, not by explicit local IDs or supervised labels.

## What This Does Not Prove

- It is not pure unsupervised ontology discovery. The recurrence compatibility function is still engineered RAFA geometry.
- It is not runtime attention yet. The learned embedding is not choosing child writeback in Circleworld production.
- It does not prove audio improvement or loop reduction.
- It does not prove nested sibling emergence.
- The structural-family metric remains post-hoc; it is a yardstick, not the training objective.

## Comparison To Prior Rungs

Prior deterministic structural scope:

- Structural top1: `0.9599303135888502`.
- Local scope top1: `0.0`.
- Structural margin: `0.040194472441424815`.

Prior pseudo-label learned probe:

- Holdout top1 structural accuracy: `0.9930-0.9948`.
- Holdout local leakage: `0.0`.
- Teacher: deterministic structural pseudo-label.

Current contrastive recurrence probe:

- Holdout top1 recurrence accuracy: `1.0` in all folds.
- Post-hoc structural-family top1: `0.989547` in all folds.
- Holdout local leakage: `0.0` in all folds.
- Teacher: cross-case recurrence compatibility, not structural pseudo-label.

Interpretation update:

The hand-bucket teacher was not the only thing carrying the signal. Continuous RAFA recurrence geometry is sufficient to train a reusable embedding that almost perfectly recovers the same structural relation on held-out seeds.

## Next Step

The next proof target should connect the learned embedding to causal operators:

1. Use the contrastive embedding to select child/grandchild law objects for operator causality assays.
2. Compare embedding-selected operators against recurrence-selected operators and wrong-family decoys.
3. If safe, run a shadow RAFA attention assay where `geometric_score + learned_embedding_residual` selects operator payloads without changing production Circleworld defaults.
