# RAFA Resonant Memory Structural Scope Push - 2026-05-14

## Scope

This follow-on implements the next three steps after the Next-4 evidence cycle:

1. Add dual family scopes for law-object retrieval.
2. Rerun retrieval with local versus structural family semantics separated.
3. Extend operator causality with selected-vs-decoy operator controls.

This remains Circleworld-memory evidence. It is not checkpoint promotion and not an audio-quality claim.

## Code Changes

- `D:\RAFA\runtimes\circleworld_proto\resonant_law_objects.py`
  - Added `local_family_key` and `structural_family_key`.
  - Kept `family_key` backward-compatible as local case/root-child identity.
  - Added `object_family_value()` and `structural_family_key_from_key()`.
  - Extended `score_query_to_law_object()` and `run_retrieval_assay()` with `family_key_field`.

- `D:\RAFA\runtimes\circleworld_proto\run_resonant_child_retrieval_assay.py`
  - Added retrieval scope summaries for `local_family_key` and `structural_family_key`.
  - Added top-candidate case summary to expose cross-case decoy pressure.

- `D:\RAFA\runtimes\circleworld_proto\run_resonant_operator_causality_assay.py`
  - Added best wrong-family decoy operator selection.
  - Reports selected parent divergence, decoy parent divergence, selected/decoy world-jump proxies, and selected-decoy gaps.

- `D:\RAFA\runtimes\circleworld_proto\test_resonant_attention_contract.py`
  - Added structural-scope regression coverage.

## Retrieval Results

### Naked RAFA CUDA, 3 Seeds

Artifact:

- `D:\RAFA\outputs\circleworld_proto\resonant_child_retrieval_2026-05-14_naked_gpu_9100_9102_structural_scope\RESONANT_CHILD_RETRIEVAL_ASSAY.md`

Local scope:

- Family field: `local_family_key`.
- Families: `30`.
- Status: `fail_no_retrieval_signal`.
- Top1 excluding self: `0.0`.
- Mean resonance margin: `-0.008260262737295857`.
- Top cross-case fraction: `0.9988385598141696`.
- Mean case-local top1: `0.6550522648083623`.

Structural scope:

- Family field: `structural_family_key`.
- Families: `43`.
- Status: `pass_nonself_retrieval_signal`.
- Top1 excluding self: `0.9599303135888502`.
- Top-k: `0.9599303135888502`.
- Mean resonance margin: `0.040194472441424815`.
- Top cross-case fraction: `0.44773519163763065`.
- Top same-structural fraction: `0.9599303135888502`.
- Mean case-local structural top1: `0.9683098591549296`.
- Cross-depth pairs: `900`.
- Mean cross-depth geometric score: `0.9039802902665957`.

Interpretation:

The previous aggregate cross-seed failure was partly a labeling problem. Local family identity asks whether the same case/root child wins; structural family identity asks whether the same kind of q/arc/support/lifecycle law object wins independent of seed. Under the structural question, reusable cross-seed resonant law identity is present in this assay.

Caution:

This is still deterministic geometric bucketing, not a learned structural ontology. It proves the current features contain cross-seed reusable structure; it does not yet prove the system learned that structure or uses it in runtime.

### Synthetic Grandchild Sanity Check

Artifact:

- `D:\RAFA\outputs\circleworld_proto\resonant_child_retrieval_2026-05-14_grandchild_synthetic_structural_scope\RESONANT_CHILD_RETRIEVAL_ASSAY.md`

Metrics:

- Local top1 excluding self: `0.7508710801393729`.
- Structural top1 excluding self: `0.9930313588850174`.
- Structural families: `23`.
- Structural margin: `0.020946292121033848`.
- Cross-depth pairs: `300`.
- Mean cross-depth geometric score: `0.9020264651206847`.

Interpretation:

Structural scope also improves the synthetic grandchild bank, which confirms the scope split is not only a naked_rafa artifact.

## Operator Causality With Decoy Control

Artifact:

- `D:\RAFA\outputs\circleworld_proto\resonant_operator_causality_2026-05-14_naked_gpu_9100_9102_decoy_control\RESONANT_OPERATOR_CAUSALITY_ASSAY.md`

Metrics:

- Status: `pass_operator_causality_present`.
- Cases: `3`.
- Mean selected parent phase divergence: `0.13983320365727522`.
- Mean inert parent phase divergence: `0.0`.
- Mean decoy parent phase divergence: `0.19017178083959577`.
- Mean selected world-jump proxy: `0.008904605384181771`.
- Mean decoy world-jump proxy: `0.02171684591925238`.
- Mean selected-decoy phase divergence gap: `-0.05033857718232057`.
- Mean selected-decoy world-jump gap: `-0.01281224053507061`.
- Selected top family hit excluding self: `1.0` for all three cases.

Interpretation:

Selected operators are not stronger than the best wrong-family decoy; they are safer and more compatible. The decoy pushes harder and causes more world-jump proxy. That is an important correction: operator-value success should not be measured as maximum parent movement. It should be measured as causal movement under compatibility constraints.

## Claim Status

Supported now:

- Grandchild law objects can be retrieved under both local and structural scopes.
- Cross-seed structural law identity is present geometrically in the naked_rafa CUDA assay.
- Selected operator payloads causally move parent state beyond inert control.
- Decoy operator controls show stronger but less safe movement.

Still not supported:

- Learned structural ontology.
- Nested sibling emergence.
- Runtime use of structural family keys.
- Audio improvement.
- Checkpoint promotion.

## Verification

```powershell
python -m pytest -q D:\RAFA\runtimes\circleworld_proto\test_resonant_attention_contract.py
python -m py_compile D:\RAFA\runtimes\circleworld_proto\resonant_law_objects.py D:\RAFA\runtimes\circleworld_proto\run_resonant_child_retrieval_assay.py D:\RAFA\runtimes\circleworld_proto\run_resonant_operator_causality_assay.py D:\RAFA\runtimes\circleworld_proto\test_resonant_attention_contract.py
```

Results:

- `7 passed`.
- Py-compile passed.

## Next Move

The next strong step is to learn the structural family key instead of hand-bucketing it. Build a tiny structural-family probe from q/arc/support/lifecycle features and train it to predict cross-seed family agreement while preserving local identity as a separate target.
