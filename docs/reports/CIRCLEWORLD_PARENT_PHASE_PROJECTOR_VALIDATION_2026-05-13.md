# Circleworld Parent Phase Projector Validation - 2026-05-13

## Executive Claim

A second non-oracle bridge cycle tested whether simple parent-compatible phase directions could replace the oracle residual correction that enables `mode_replace_nested_sibling`.

Result:

> Parent-only tangent, anti-tangent, and child-signed tangent projectors do not convert into sibling ontology.

This strengthens the current conclusion: the missing module is not a scalar authority gate and not a simple hand-authored tangent. It is a learned child-conditioned parent residual law.

## Implementation

Files changed:

- `D:\RAFA\runtimes\circleworld_proto\test_nested_commitment.py`
- `D:\RAFA\runtimes\circleworld_proto\run_childworld_volume_recursion_sweep.py`

New assay config:

- `learned_parent_bridge_delta_source`

Supported non-oracle projector sources added:

- `raw_child_delta`
- `parent_tangent`
- `anti_parent_tangent`
- `child_signed_parent_tangent`
- `anti_child_signed_parent_tangent`

The parent tangent is computed from parent-only evolution, not from the child-enabled oracle residual.

## Validation Artifact

Primary sweep:

- `D:\RAFA\outputs\circleworld_proto\parent_phase_projector_validation_2026-05-13_seeded_suite\childworld_volume_recursion_sweep_summary.json`

Compared variants:

- `parent_ontology_learned_parent_bridge_gate55_static_mode_replace`
- `parent_ontology_learned_parent_bridge_gate55_rawdelta002_static_mode_replace`
- `parent_ontology_phase_projector_gate55_tangent_static_mode_replace`
- `parent_ontology_phase_projector_gate55_anti_tangent_static_mode_replace`
- `parent_ontology_phase_projector_gate55_child_signed_tangent_static_mode_replace`
- `parent_ontology_childphase_gate55_static_mode_replace`

## Aggregate Results

| Variant | Mode-Replace Sibling Fraction | Conversion Score | Readiness | Positive Residual Reduction | Candidate Delta Abs | Binding Mass | Mode1 Occupancy After | World Jump |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| learned bridge gate55 | `0.0` | `0.0` | `0.8947068938927772` | `0.0000007696272108141788` | `0.0` | `0.019499879265721474` | `0.3108166678005503` | `0.0` |
| rawdelta002 | `0.0` | `0.0` | `0.8947062277529026` | `0.1044994630552738` | `0.000651061358893363` | `0.019499879265721474` | `0.310818863384227` | `0.0` |
| parent tangent | `0.0` | `0.0` | `0.8947068711518398` | `0.0010886837734574463` | `0.0000715161973083369` | `0.019499879265721474` | `0.310816697877039` | `0.0` |
| anti parent tangent | `0.0` | `0.0` | `0.8947068735303705` | `0.00615560468508668` | `0.0000715161973083369` | `0.019499879265721474` | `0.31081663772406154` | `0.0` |
| child-signed tangent | `0.0` | `0.0` | `0.8947068164108699` | `0.0065751056558703115` | `0.0000715161973083369` | `0.019499879265721474` | `0.3108168595381659` | `0.0` |
| oracle gate55 | `0.5` | `0.49854242988999936` | `1.0` | `0.26338768174506794` | `0.0008050848772536231` | `0.05997418847659395` | `0.2931645728077845` | `0.0` |

## Branch-Level Comparison

On `child_mode1_replace_50_shift`:

| Variant | Label | Readout Minus Baseline | Coarse Corr | Q Corr | Candidate Delta Abs | Positive Residual Reduction | Mode1 Occupancy After |
|---|---|---:|---:|---:|---:|---:|---:|
| parent tangent | `ambiguous_middle` | about `-0.040` | about `0.9988` | about `0.9991` | about `0.0000599` | about `0.0013` | about `0.2641` |
| anti parent tangent | `ambiguous_middle` | about `-0.040` | about `0.9988` | about `0.9991` | about `0.0000599` | `0.0` on the displayed branch rows | about `0.2641` |
| child-signed tangent | `ambiguous_middle` | about `-0.040` | about `0.9988` | about `0.9991` | about `0.0000599` | about `0.00187` | about `0.2641` |
| oracle gate55 | `nested_sibling` | `0.000208` to `0.002201` | about `0.9980` | about `0.9982` | about `0.000626` | about `0.1045` | about `0.33048` |

## Interpretation

The non-oracle tangent projectors fail in two ways:

1. Their candidate delta is too small: about `0.0000715` versus oracle branch delta around `0.000626`.
2. Their direction is weakly predictive: positive residual reduction remains below `0.007` on aggregate.

The earlier rawdelta002 test matched oracle-scale magnitude better and reached positive residual reduction around `0.1045`, but still failed sibling conversion. That means magnitude alone is not enough either.

The oracle path wins because it supplies a tiny correction with the right child-conditioned direction and enough ontology binding/charge.

## Current Scientific State

The child framework now has a clearer decomposition:

- child survival: mechanically possible
- parent authority bridge: assay-positive when oracle-calibrated
- non-oracle authority bridge: binds but does not convert
- raw child delta: too broad/wrong
- parent tangent: too weak/wrong
- required next module: learned child-conditioned low-rank parent residual projector

This is closer to the original RAFA ambition: not symbolic branching, and not magnitude modeling, but a learned phase-law operator that decides how a child re-enters the parent lattice.

## Non-Promotion Decision

Do not promote any parent phase projector variant.

Reasons:

- `mean_assay_mode_replace_nested_sibling_fraction` remains `0.0`.
- `mean_mode_replace_conversion_score` remains `0.0`.
- `mean_world_jump_penalty` remains controlled, but no sibling ontology emerges.
- Oracle gate55 remains the only positive branch in this comparison.

## Next Action

Build a tensor-level projector dataset and train a tiny projector head.

Dataset target:

- input: child phase, parent mode 1, parent-only tangent, child support, child coherence, q/arc compatibility, boundary fields
- target: oracle residual delta `target_residual_delta` from the assay
- output: low-rank tiny parent-compatible phase correction

Acceptance criteria for the trained projector:

- candidate delta magnitude near oracle branch scale, around `0.0006`
- positive residual reduction meaningfully above raw/tangent baselines
- nonzero mode-replace sibling conversion
- no world-jump
- no use of oracle residual at runtime after training/evaluation split

## Verification

Contract tests passed after implementation and validation:

```text
pytest -q D:\RAFA\runtimes\circleworld_proto\test_learned_branch_law_child_ifs_contract.py D:\RAFA\runtimes\circleworld_proto\test_unit_phasor_contract.py D:\RAFA\runtimes\circleworld_proto\test_phase_gauge_invariance.py
....                                                                     [100%]
4 passed in 3.23s
```
