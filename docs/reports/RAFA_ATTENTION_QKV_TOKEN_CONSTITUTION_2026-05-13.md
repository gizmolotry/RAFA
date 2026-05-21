# RAFA Attention / QKV / Token Constitution - 2026-05-13

## Summary

This pass formalized the RAFA meaning of `token`, `attention`, `query`, `key`,
and `value`.

The key decision:

- `relational_qkv_v2` is a local Circleworld branch-QKV precursor.
- It is not the canonical RAFA attention or post-token memory substrate.
- Full RAFA attention requires resonance-mediated law selection, operator-valued
  values, explicit geometric score components, and retrieval/composition assays.

## Artifacts

- `D:\RAFA\docs\architecture\RAFA_ATTENTION_QKV_TOKEN_CONSTITUTION.md`
- `D:\RAFA\docs\architecture\constitutions\10_resonant_attention_memory_lane.md`
- `D:\RAFA\docs\architecture\RAFA_BRANCH_STOCKTAKE.md`
- `D:\RAFA\docs\architecture\constitutions\README.md`
- `D:\RAFA\docs\reports\RAFA_CLAIM_TAXONOMY_2026-05-06.md`

## Constitution Decisions

RAFA units are now separated as:

- raw lattice site: phase coordinate, not a token
- packet: promoted local phase-law candidate
- childworld: recursive local law carrier
- relational signature: compressed addressable law object
- RAFA post-token: earned relational object that passes retrieval, composition,
  and causal-use tests

RAFA QKV is now defined as:

- `Q`: unresolved phase/rational/support tension
- `K`: advertised relational law and compatibility signature
- `V`: phase or branch operator payload

RAFA attention is now defined as:

`score = geometric_score + learned_residual`

The geometric spine must remain reportable. Learned residuals may improve routing
but cannot silently replace q/arc/support/phase compatibility.

## Current Implementation Mapping

Current Circleworld code has:

- `_relational_branch_attention(...)`
- `branch_law_version = "relational_qkv_v2"`
- `relation_attention_gain`
- `relation_attention_sharpness`
- `relation_value_gain`

Interpretation:

This is local 2-mode branch coupling over phase-native features. It is a useful
precursor and should remain available, but it does not retrieve reusable
packet/child/grandchild law objects and therefore should not be promoted to full
RAFA attention.

## Term Audit

Non-mutating audit commands used:

```powershell
rg -n -i "\b(token|tokens|attention|query|queries|key|keys|value|values|qkv)\b" D:\RAFA\docs\architecture D:\RAFA\docs\reports -g "*.md"
rg -n -i "relational_qkv_v2|QKV|attention" D:\RAFA\docs\architecture D:\RAFA\docs\reports -g "*.md"
```

Audit result:

- Broad term hits across architecture/reports: 276 lines.
- Focused QKV/attention hits include Graduation restore reports, Circleworld
  geometric masking, the branch-law ablation report, the relation-token build
  report, and the new constitution.
- The most important historical Circleworld report already says the intent of
  `relational_qkv_v2` was not full learned token attention, but branch-to-branch
  relational coupling over phase-native state.

Policy:

Historical reports are left intact. From this point forward, new reports must
use the constitution vocabulary and must label `relational_qkv_v2` as a local
branch-QKV precursor unless retrieval/composition assays over reusable law
objects are run.

## Required Future Assays

The constitution defines four assay families:

- resonant child retrieval
- cross-depth composition
- interference selectivity
- audio bridge

No dense signature, childworld, or packet should be promoted to RAFA post-token
status until retrieval, composition, and operator-causality evidence is present.

## Interpretation

This reframes the broad Circleworld ambition without dissolving it into audio
utility alone.

Circleworld can still be tested as an audio continuity tool, but the larger
thesis is now separable:

`Can phase-native law objects become a resonant memory substrate that replaces
token-position attention?`

The next engineering move should be a resonant retrieval/composition assay using
existing packet, child, and branch artifacts as candidate law objects.
