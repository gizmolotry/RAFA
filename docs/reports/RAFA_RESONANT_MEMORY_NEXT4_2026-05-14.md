# RAFA Resonant Memory Next-4 Evidence - 2026-05-14

## Scope

This report records the four-front continuation after the RAFA attention/QKV/token constitution work:

1. Grandchild composition in the Circleworld ontology substrate.
2. Resonant child/grandchild retrieval, including case-local versus cross-seed behavior.
3. Operator-valued law-object causality.
4. Ledger/report integration for claim separation.

This is Circleworld-memory evidence, not a checkpoint promotion and not an audio-quality claim.

## Code Changes

- Added collision-resistant child law object IDs and explicit `child_instance_key` in `D:\RAFA\runtimes\circleworld_proto\resonant_law_objects.py`.
- Changed child/grandchild family identity to use root `origin_child_id`, while keeping `parent_child_id` as the immediate cross-depth link.
- Added `composition_summary()` into nested reports through `resonant_composition`.
- Hardened composition against false positives by skipping self-parent fragments and requiring child generation to exceed parent generation.
- Added case-local retrieval summaries to `D:\RAFA\runtimes\circleworld_proto\run_resonant_child_retrieval_assay.py`.
- Added `D:\RAFA\runtimes\circleworld_proto\run_resonant_operator_causality_assay.py` for operator-valued child law payload testing.
- Added regression tests in `D:\RAFA\runtimes\circleworld_proto\test_resonant_attention_contract.py` for root-family continuity and self-parent composition filtering.

## Front 1 - Grandchild Composition

Synthetic grandchild rerun:

- Nested report: `D:\RAFA\outputs\circleworld_proto\resonant_grandchild_composition_2026-05-14_synthetic_postfix\nested_commitment_report.json`
- Standalone retrieval report: `D:\RAFA\outputs\circleworld_proto\resonant_child_retrieval_2026-05-14_grandchild_synthetic_casebreakdown\RESONANT_CHILD_RETRIEVAL_ASSAY.md`
- Law objects: `550` embedded, `574` standalone including branch summaries.
- Top-1 family accuracy excluding self: `0.7836363636363637` embedded, `0.7508710801393729` standalone.
- Mean resonance margin: `0.020018684939402728` embedded, `0.018805991394518964` standalone.
- Cross-depth pairs: `300`.
- Mean cross-depth geometric score: `0.9020264651206847`.
- Mean cross-depth q compatibility: `1.0`.
- Mean cross-depth support overlap: `0.4662117792167746`.
- Mean assay grandchildren: `6.0`.
- Max child generation: `3.0`.

Interpretation: grandchild records are not decorative bookkeeping anymore; they form queryable cross-depth law objects under the geometric retrieval assay. This supports the narrow claim that Circleworld can build a hierarchical resonant object bank inside one seeded substrate.

Limitation: nested sibling remains `0.0`; this is retrieval/composition evidence, not proof of child ontology converting into parent-mode ontology.

## Front 2 - Naked RAFA GPU Replay

Three-seed CUDA naked_rafa rerun:

- Nested report: `D:\RAFA\outputs\circleworld_proto\resonant_grandchild_composition_2026-05-14_naked_gpu_9100_9102_postfix\nested_commitment_report.json`
- Standalone retrieval report: `D:\RAFA\outputs\circleworld_proto\resonant_child_retrieval_2026-05-14_naked_gpu_9100_9102_casebreakdown\RESONANT_CHILD_RETRIEVAL_ASSAY.md`
- Device: `cuda`.
- Cases: `3` seeded naked_rafa cases, `9100`, `9101`, `9102`.
- Embedded law objects: `1650`.
- Standalone objects: `1722` including branch summaries.
- Cross-depth pairs: `900`.
- Mean cross-depth geometric score: `0.9039802902665957`.
- Mean assay grandchildren: `6.0`.
- Max child generation: `3.0`.
- Mean branch identity carry: `1.0`.
- Mean readout sibling response: `0.503833415688749`.
- Mean nested sibling fraction: `0.0`.

Case-local retrieval:

| Case | Objects | Status | Top1 Excl Self | Margin | Cross-Depth Pairs |
|---|---:|---|---:|---:|---:|
| naked_seed_9100 | 574 | pass_nonself_retrieval_signal | 0.6550522648083623 | 0.0892135002897013 | 300 |
| naked_seed_9101 | 574 | pass_nonself_retrieval_signal | 0.6550522648083623 | 0.08921276701006313 | 300 |
| naked_seed_9102 | 574 | pass_nonself_retrieval_signal | 0.6550522648083623 | 0.08921338599878711 | 300 |

Aggregate cross-seed retrieval:

- Status: `fail_no_retrieval_signal`.
- Aggregate top-1 family accuracy excluding self: `0.0`.
- Aggregate top-k family accuracy: `0.0`.
- Aggregate mean resonance margin: `-0.008260262737295857`.

Interpretation: the same machinery works inside each naked_rafa seed, but it does not yet produce reusable cross-seed law identity. Cross-seed decoys are geometrically too similar and outrank same-case family positives. This cleanly separates two claims: case-local resonant retrieval is present; cross-anchor reusable relational memory is not yet established.

## Front 3 - Operator Causality

Synthetic operator-causality smoke:

- Report: `D:\RAFA\outputs\circleworld_proto\resonant_operator_causality_2026-05-14_synthetic\RESONANT_OPERATOR_CAUSALITY_ASSAY.md`
- Status: `pass_operator_causality_present`.
- Cases: `1`.
- Mean parent phase divergence: `0.4674022032858465`.
- Mean inert parent phase divergence: `0.0`.
- Mean support shift: `0.09007472544908524`.
- Mean qtrace shift proxy: `0.0`.
- Mean world jump proxy: `0.15006260220821033`.

Naked_rafa CUDA operator-causality run:

- Report: `D:\RAFA\outputs\circleworld_proto\resonant_operator_causality_2026-05-14_naked_gpu_9100_9102\RESONANT_OPERATOR_CAUSALITY_ASSAY.md`
- Status: `pass_operator_causality_present`.
- Cases: `3`.
- Mean law object count: `11.0`.
- Mean parent phase divergence: `0.13983320365727522`.
- Mean inert parent phase divergence: `0.0`.
- Mean support shift: `0.04347693423430125`.
- Mean qtrace shift proxy: `0.03626033103645235`.
- Mean world jump proxy: `0.008904605384181771`.
- Mean selected resonance margin: `0.11626613324525943`.
- Selected top family hit excluding self: `1.0` for all three cases.

Interpretation: operator-valued law objects can causally change the parent phase state beyond an inert control, and the naked_rafa CUDA version is much safer than the synthetic operator smoke by the world-jump proxy. This is the strongest new evidence for RAFA `V` as an operator payload rather than a copied content vector.

Limitation: this still does not prove nested sibling conversion. It proves selected operator payloads can move parent state without a large world jump in this short naked_rafa assay.

## Front 4 - Claim Ledger Update

Ledger sidecar:

- `D:\RAFA\docs\reports\CIRCLEWORLD_EXPERIMENT_LEDGER_2026-05-14.jsonl`

Claim separation from this cycle:

- Supported: grandchild-expanded law objects can be composed by geometric resonance inside a seeded substrate.
- Supported: case-local naked_rafa retrieval works on CUDA live-child/grandchild objects.
- Supported: selected child law operators can causally perturb parent phase beyond inert controls.
- Not supported yet: cross-seed reusable law identity.
- Not supported yet: nested sibling emergence.
- Not claimed: audio improvement, checkpoint promotion, dense RAFA signatures, CLAP/text auto-formalization.

## Verification

Commands run:

```powershell
python -m pytest -q D:\RAFA\runtimes\circleworld_proto\test_resonant_attention_contract.py
python -m py_compile D:\RAFA\runtimes\circleworld_proto\resonant_law_objects.py D:\RAFA\runtimes\circleworld_proto\test_nested_commitment.py D:\RAFA\runtimes\circleworld_proto\run_resonant_operator_causality_assay.py D:\RAFA\runtimes\circleworld_proto\run_resonant_child_retrieval_assay.py
```

Results:

- `6 passed` for resonant attention contract tests.
- Py-compile passed.

## Next Move

The next discriminating experiment should target cross-seed law identity. The substrate can retrieve within a seed and compose across depth, but it does not yet know that structurally corresponding law objects across seeds should be related. The likely next patch is a two-level family key/reporting scheme:

- `local_family_key`: case + root child.
- `structural_family_key`: q/arc/support/lifecycle cluster independent of case.

Then rerun retrieval with both scopes. If structural retrieval improves without destroying local specificity, Circleworld-memory starts to look like a reusable RAFA resonant memory substrate rather than a per-seed ontology gadget.
