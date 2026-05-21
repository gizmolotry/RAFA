# RAFA Resonant Attention Refactor - 2026-05-13

## Summary

This pass implemented the first concrete assay layer for the RAFA Attention /
QKV / Token Constitution.

The branch hierarchy now treats Resonant Attention / Post-Token Memory as an
L4a research lane over Circleworld, not as a new production runtime and not as a
renaming of `relational_qkv_v2`.

The Circleworld ontology branch now emits compact law objects from live child
states during nested assays. These objects expose:

- query fields: q profile, support, boundary, temporal window, residue proxy
- key fields: advertised relational compatibility fields
- value fields: operator kind, operator norm, budget, support/coherence mass
- explicit geometric score components

## Implementation Artifacts

- `D:\RAFA\runtimes\circleworld_proto\resonant_law_objects.py`
- `D:\RAFA\runtimes\circleworld_proto\run_resonant_child_retrieval_assay.py`
- `D:\RAFA\runtimes\circleworld_proto\test_resonant_attention_contract.py`
- `D:\RAFA\runtimes\circleworld_proto\test_nested_commitment.py`

Docs updated:

- `D:\RAFA\docs\architecture\RAFA_COGNITIVE_STACK.md`
- `D:\RAFA\docs\architecture\RAFA_LINEAGE_LEDGER.md`
- `D:\RAFA\docs\architecture\INTERLINEAGE_DAG.md`
- `D:\RAFA\docs\architecture\RUNTIME_CONTRACTS.md`
- `D:\RAFA\docs\reports\RAFA_CLAIM_TEST_MATRIX_2026-05-06.md`
- `D:\RAFA\docs\reports\RAFA_SEPARABLE_CLAIMS_CURRENT_2026-05-07.md`
- `D:\RAFA\docs\reports\RAFA_VARIABLE_CLAIMS_2026-05-06.md`

## What Changed

### Branch hierarchy

The lane is now represented as an L4a/docs-evaluator lane:

`Resonant Attention / Post-Token Memory`

It sits over Circleworld artifacts and tests whether law objects can be
retrieved, composed, and causally used by relational resonance.

It does not replace:

- Graduation audio baseline
- Circleworld branch ontology
- dense signature claim family
- semantic projector lane

### Circleworld ontology branch

`test_nested_commitment.py` now exports:

- `resonant_law_objects`
- `resonant_retrieval`
- `resonant_retrieval_rows`
- per-branch `resonant_law_object_count`
- per-branch `resonant_law_object_ids`

These are additive JSON fields. Existing nested metrics and labels are not
renamed or removed.

### Standalone assay

`run_resonant_child_retrieval_assay.py` can read:

- per-case `nested_commitment_summary.json`
- aggregate `nested_commitment_report.json`
- directories containing either

It deduplicates law objects by `object_id` and emits:

- `resonant_child_retrieval_assay.json`
- `RESONANT_CHILD_RETRIEVAL_ASSAY.md`

## Results

### Synthetic smoke

Output:

`D:\RAFA\outputs\circleworld_proto\resonant_child_retrieval_2026-05-13_synthetic_smoke`

Result:

- status: `pass_nonself_retrieval_signal`
- object count: `6`

Interpretation:

The scoring/reporting contract can detect family retrieval when the law-object
bank has separable families.

### Old branch-summary fallback

Output:

`D:\RAFA\outputs\circleworld_proto\resonant_child_retrieval_2026-05-13_carrier_only_dedup`

Result:

- status: `fail_no_retrieval_signal`
- object count: `72`
- top1 family accuracy excluding self: `0.0`
- mean resonance margin: `-0.00939138502558581`
- mean top learned residual: `0.0`

Interpretation:

Old nested branch summaries are too saturated and similar to support resonant
retrieval. This is a useful negative result: the assay does not rubber-stamp old
child ontology metrics.

### Fresh live-child nested smoke

Output:

`D:\RAFA\outputs\circleworld_proto\resonant_nested_smoke_2026-05-13`

Embedded nested report result:

- `resonant_law_object_count`: `250`
- family count: `5`
- top1 family accuracy excluding self: `0.524`
- top-k family accuracy: `0.544`
- mean resonance margin: `0.004928029954680164`
- mean top learned residual: `0.0`
- status: `pass_nonself_retrieval_signal`

Standalone dedup result:

`D:\RAFA\outputs\circleworld_proto\resonant_child_retrieval_2026-05-13_nested_smoke_dedup`

- object count: `274`
- childworld objects: `250`
- top1 family accuracy excluding self: `0.4781021897810219`
- top-k family accuracy: `0.5109489051094891`
- mean resonance margin: `0.0036981202022341856`
- status: `pass_nonself_retrieval_signal`

Interpretation:

Live child-state law objects contain more retrievable relational structure than
old branch-summary proxies. The margin is small and entropy is high, so this is
not a promotion result. It is a first positive assay signal that the live child
ontology branch can produce queryable law objects.

## Verification

Commands run:

```powershell
python -m pytest -q D:\RAFA\runtimes\circleworld_proto\test_resonant_attention_contract.py
python -m pytest -q D:\RAFA\runtimes\circleworld_proto\test_learned_branch_law_child_ifs_contract.py D:\RAFA\runtimes\circleworld_proto\test_resonant_attention_contract.py
python -m py_compile D:\RAFA\runtimes\circleworld_proto\resonant_law_objects.py D:\RAFA\runtimes\circleworld_proto\run_resonant_child_retrieval_assay.py D:\RAFA\runtimes\circleworld_proto\test_resonant_attention_contract.py D:\RAFA\runtimes\circleworld_proto\test_nested_commitment.py
python D:\RAFA\runtimes\circleworld_proto\test_packet_alias_contract.py --out-dir D:\RAFA\outputs\circleworld_proto\resonant_packet_alias_contract_2026-05-13 --device cpu
git -C D:\RAFA diff --check
```

Observed:

- resonant attention tests: `4 passed`
- combined child-IFS/resonant tests: `8 passed`
- packet alias audit: `pass`
- compile: pass
- diff check: no whitespace errors; Git emitted line-ending normalization warnings only

## Current Verdict

The lane is now implemented as an assay substrate.

Promoted claim:

`Circleworld can export live child-state law objects and test resonant retrieval
over them with explicit geometric components.`

Not promoted:

- full RAFA attention
- RAFA post-tokens
- cross-depth composition
- operator-causal reasoning memory
- audio-facing benefit

## Next Required Step

The immediate next step is a real seeded/naked-RAFA retrieval run on the live
child-state export path using GPU or a CPU-safe seed source. The prior CPU attempt
with `naked_rafa` hit an existing Triton CPU limitation:

`Pointer argument cannot be accessed from Triton (cpu tensor?)`

Use either:

- GPU `naked_rafa` seeds, or
- synthetic/anchor seeds for CPU validation.

After that, add a cross-depth composition run with grandchild records enabled so
`composition_summary.cross_depth_pair_count` becomes nonzero.
