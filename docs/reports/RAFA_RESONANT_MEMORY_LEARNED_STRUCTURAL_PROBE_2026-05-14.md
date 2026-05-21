# RAFA Resonant Memory Learned Structural Probe - 2026-05-14

## Scope

This cycle implements the next three steps after the structural-scope split:

1. Add a law-object feature surface for learned probes.
2. Train a tiny pairwise learned structural-family probe.
3. Compare learned cross-seed structural retrieval against local identity and decoy behavior.

This is still assay-only. Runtime Circleworld behavior and checkpoint selection are unchanged.

## Code Changes

- `D:\RAFA\runtimes\circleworld_proto\resonant_law_objects.py`
  - Added `LAW_OBJECT_FEATURE_NAMES`.
  - Added `law_object_feature_vector()`.
  - Feature surface includes q profile, q entropy, arc/residue/support/coherence/boundary/temporal geometry, generation, operator budget, and object-kind indicators.

- `D:\RAFA\runtimes\circleworld_proto\run_learned_structural_family_probe.py`
  - New assay-only trainer/evaluator.
  - Trains a pairwise MLP on law-object feature pairs.
  - Structural head predicts same `structural_family_key`.
  - Local auxiliary head predicts same `local_family_key`.
  - Evaluation uses leave-one-case cross-seed retrieval.
  - Exports JSON, Markdown, and `.pt` checkpoint.

- `D:\RAFA\runtimes\circleworld_proto\test_resonant_attention_contract.py`
  - Added law-object feature-vector contract test.

## Main Naked RAFA Result

Canonical holdout:

- Input: `D:\RAFA\outputs\circleworld_proto\resonant_grandchild_composition_2026-05-14_naked_gpu_9100_9102_postfix`
- Output: `D:\RAFA\outputs\circleworld_proto\learned_structural_probe_2026-05-14_naked_gpu_9100_9102\LEARNED_STRUCTURAL_FAMILY_PROBE.md`
- Checkpoint: `D:\RAFA\outputs\circleworld_proto\learned_structural_probe_2026-05-14_naked_gpu_9100_9102\learned_structural_family_probe.pt`
- Holdout case: `naked_seed_9102`
- Objects: `1722`
- Train objects: `1148`
- Holdout objects: `574`
- Pair count: `50000`
- Final structural loss: `0.0011973219054420858`
- Final local loss: `0.4280124622948316`
- Holdout cross-case top1 structural accuracy: `0.9930313588850174`
- Holdout local leakage: `0.0`
- Holdout same-case fraction: `0.0`
- Mean learned margin: `0.945215328049638`
- Mean top structural probability: `0.9999843910596097`
- Mean top local probability: `0.316073032236329`

Interpretation:

A small learned probe can recover the structural family relation across unseen naked_rafa seed objects from q/arc/support/lifecycle/operator features. This upgrades the prior deterministic structural-bucket result: the relation is learnable from the law-object feature surface, not only hand-countable after the fact.

## Leave-One-Seed Robustness

| Holdout | Status | Holdout Top1 Structural | Holdout Local Leakage | Mean Learned Margin | Structural Loss | Local Loss |
|---|---|---:|---:|---:|---:|---:|
| `naked_seed_9100` | `pass_learned_structural_signal` | `0.9947735191637631` | `0.0` | `0.9392433641260964` | `0.00126003387687925` | `0.43323895487250114` |
| `naked_seed_9101` | `pass_learned_structural_signal` | `0.9947735191637631` | `0.0` | `0.9392038330835074` | `0.0012655218109158546` | `0.43322899998450765` |
| `naked_seed_9102` | `pass_learned_structural_signal` | `0.9930313588850174` | `0.0` | `0.945215328049638` | `0.0011973219054420858` | `0.4280124622948316` |

Interpretation:

The learned structural relation is stable across all three leave-one-seed folds. Local leakage is zero in the holdout cross-case retrieval because the selected structural matches do not collapse into local case/root-child identity.

## Comparison To Deterministic Structural Scope

Prior deterministic structural-scope assay on the same naked_rafa bank:

- Local scope top1: `0.0`.
- Structural bucket top1: `0.9599303135888502`.
- Structural margin: `0.040194472441424815`.

Learned probe:

- Holdout top1: `0.9930313588850174` to `0.9947735191637631` across folds.
- Holdout margin: `0.9392038330835074` to `0.945215328049638`.

Interpretation:

The learned probe sharpens the structural relation substantially relative to direct geometric retrieval under the hand bucket labels. However, the training target is still the deterministic structural pseudo-label, so this is not yet independent ontology discovery.

## What This Proves

Supported:

- Law objects contain enough q/arc/support/lifecycle/operator information for a tiny neural probe to learn structural cross-seed agreement.
- Structural identity can be learned separately from local identity.
- Cross-seed reusable relation is stronger than the earlier local-family score suggested.

Not yet supported:

- Unsupervised discovery of structural families.
- Runtime use of the learned probe inside Circleworld evolution.
- Nested sibling emergence.
- Audio improvement.
- A promoted checkpoint.

## Verification

```powershell
python -m pytest -q D:\RAFA\runtimes\circleworld_proto\test_resonant_attention_contract.py
python -m py_compile D:\RAFA\runtimes\circleworld_proto\resonant_law_objects.py D:\RAFA\runtimes\circleworld_proto\run_learned_structural_family_probe.py D:\RAFA\runtimes\circleworld_proto\test_resonant_attention_contract.py
```

Results:

- `8 passed`.
- Py-compile passed.

## Next Move

The next useful move is to remove the hand-bucket teacher and train a contrastive structural embedding from cross-seed recurrence itself:

- positives: cross-seed child/grandchild law objects that match by retrieval geometry and cross-depth role;
- negatives: high-scoring decoys with incompatible support/lifecycle or higher world-jump proxy;
- objective: retrieve cross-seed structural law while predicting safe operator causality.
