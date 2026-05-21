# Circleworld Learned Parent Bridge Validation - 2026-05-13

## Executive Claim

An assay-only `child_predictive_candidate="learned_parent_bridge"` path was implemented and tested against the seeded childworld ontology substrate.

The result is a useful negative:

> A non-oracle learned parent-authority bridge can bind, charge, and apply, but it does not yet convert into `mode_replace_nested_sibling`.

This separates three mechanisms that were previously tangled:

1. parent ontology authority,
2. small parent-mode phase correction,
3. correct residual-law direction.

## Implementation

Files changed:

- `D:\RAFA\runtimes\circleworld_proto\test_nested_commitment.py`
- `D:\RAFA\runtimes\circleworld_proto\run_childworld_volume_recursion_sweep.py`

New assay-only candidate:

- `child_predictive_candidate="learned_parent_bridge"`

New learned bridge metrics:

- `mean_child_predictive_learned_bridge_score`
- `mean_child_predictive_learned_bridge_acceptance`
- `mean_child_predictive_learned_bridge_boundary_score`
- `mean_child_predictive_learned_bridge_volume_score`
- `mean_child_predictive_learned_bridge_support_score`
- `mean_child_predictive_learned_bridge_coherence_score`
- `mean_child_predictive_learned_bridge_delta_scale`

Important constraint:

The learned bridge action path does not use `oracle_parent_residual` as its candidate delta. It uses child/parent compatibility features and optional tiny raw-child delta scale. Oracle residuals are still computed in the assay for evaluation only.

## Main Validation Sweep

Artifact:

- `D:\RAFA\outputs\circleworld_proto\learned_parent_bridge_validation_2026-05-13_seeded_suite\childworld_volume_recursion_sweep_summary.json`

Compared variants:

- `positive_control`
- `predictive_bind_static_mode_replace`
- `predictive_oracle_bind_static_mode_replace`
- `parent_ontology_childphase_gate54_static_mode_replace`
- `parent_ontology_childphase_gate55_static_mode_replace`
- `learned_parent_bridge_static_mode_replace`
- `parent_ontology_learned_parent_bridge_gate54_static_mode_replace`
- `parent_ontology_learned_parent_bridge_gate545_static_mode_replace`
- `parent_ontology_learned_parent_bridge_gate55_static_mode_replace`

Key learned bridge result:

| Variant | Mode-Replace Sibling Fraction | Conversion Score | Learned Score | Learned Acceptance | Binding Mass | Candidate Delta Abs | Parent Ontology Charge | Mode1 Occupancy After | World Jump |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| learned bridge static | `0.0` | `0.0` | `0.7063113831162983` | `0.0918204798051188` | `0.019499879265721474` | `0.0` | `0.1364991564510597` | `0.43597374520780163` | `0.0` |
| learned bridge gate54 | `0.0` | `0.0` | `0.7063113831162983` | `0.0918204798051188` | `0.019499879265721474` | `0.0` | `0.1345491689733333` | `0.3033944673191277` | `0.0` |
| learned bridge gate545 | `0.0` | `0.0` | `0.7063113831162983` | `0.0918204798051188` | `0.019499879265721474` | `0.0` | `0.13552416147043309` | `0.3072483895025846` | `0.0` |
| learned bridge gate55 | `0.0` | `0.0` | `0.7063113831162983` | `0.0918204798051188` | `0.019499879265721474` | `0.0` | `0.1364991564510597` | `0.3108166678005503` | `0.0` |
| oracle gate55 | `0.5` | `0.49854242988999936` | `0.0` | `0.0` | `0.05997418847659395` | `0.0008050848772536231` | `0.2191393373327123` | `0.2931645728077845` | `0.0` |

Interpretation:

- The learned bridge is active: score, acceptance, binding, and parent ontology charge are nonzero.
- It is not a raw fallback: candidate delta is `0.0` when `learned_parent_bridge_delta_scale=0.0`.
- It is not oracle leakage: candidate configs use `learned_parent_bridge`, not `oracle_parent_residual`.
- It still fails to produce `mode_replace_nested_sibling`.

## Raw-Delta Tail Sweep

Artifact:

- `D:\RAFA\outputs\circleworld_proto\learned_parent_bridge_rawdelta_tail_2026-05-13_seeded_suite\childworld_volume_recursion_sweep_summary.json`

Purpose:

Test whether the missing part is merely a tiny non-oracle phase tail. Since oracle gate55 has candidate delta around `0.000626` on the successful 50% mode-replace branch, we tested scaled raw child deltas:

- `0.001 * raw_child_delta`
- `0.002 * raw_child_delta`
- `0.005 * raw_child_delta`

Aggregate result:

| Variant | Candidate Delta Abs | Positive Residual Reduction | Mode-Replace Sibling Fraction | Conversion Score | World Jump |
|---|---:|---:|---:|---:|---:|
| rawdelta001 | `0.0003255306794466815` | `0.0573896981523594` | `0.0` | `0.0` | `0.0` |
| rawdelta002 | `0.000651061358893363` | `0.1044994630552738` | `0.0` | `0.0` | `0.0` |
| rawdelta005 | `0.0016276532809265023` | `0.04014796056248452` | `0.0` | `0.0` | `0.0` |
| oracle gate55 | `0.0008050848772536231` | `0.26338768174506794` | `0.5` | `0.49854242988999936` | `0.0` |

Branch-level result for `child_mode1_replace_50_shift`:

| Variant | Label | Readout Minus Baseline | Candidate Delta Abs | Positive Residual Reduction | Mode1 Occupancy After |
|---|---|---:|---:|---:|---:|
| learned gate55 | `ambiguous_middle` | about `-0.040` | `0.0` | about `0.0` | about `0.264` |
| rawdelta001 | `ambiguous_middle` | about `-0.040` | `0.000295` | about `0.017` | about `0.264` |
| rawdelta002 | `ambiguous_middle` | about `-0.040` | `0.000591` | about `0.034` | about `0.264` |
| rawdelta005 | `ambiguous_middle` | about `-0.040` | `0.001477` | about `0.063` | about `0.264` |
| oracle gate55 | `nested_sibling` | `0.000208` to `0.002201` | about `0.000626` | about `0.1045` | about `0.33048` |

Interpretation:

- Tiny raw-child phase tails can improve residual prediction somewhat.
- Even an oracle-sized raw delta tail does not convert to sibling ontology.
- Therefore the missing ingredient is not just phase-delta magnitude.
- The residual-law direction matters.
- The learned bridge also undershoots the successful 50% branch occupancy: around `0.264` versus oracle gate55 around `0.33048`.

## What We Learned

The prior positive oracle result had two ingredients:

1. parent ontology authority/readout access,
2. a tiny but correctly directed parent-mode phase correction.

The non-oracle learned bridge implemented here supplies ingredient 1 and optionally supplies tiny raw-child phase motion, but does not yet supply ingredient 2.

This is the new bottleneck:

> Learn the direction of the parent residual law, not merely the amount of child authority.

In RAFA terms: the child needs a parent-compatible phase-law projection. It cannot just be granted readout authority as raw child phase.

## Non-Promotion Decision

Do not promote the learned parent bridge candidate.

Reasons:

- `mean_assay_mode_replace_nested_sibling_fraction` remains `0.0` for all learned bridge variants.
- `mean_mode_replace_conversion_score` remains `0.0`.
- World-jump is controlled, but conversion is absent.
- The oracle upper bound still dominates.

## Next Action

The next bridge module should learn a non-oracle low-rank phase-law projector:

- input: child phase, parent mode 1, child support, child coherence, q/arc compatibility, boundary match
- output: a tiny parent-compatible phase correction, not a raw child delta
- target scale: around `0.0006` on the successful branch
- target behavior: preserve coarse/q identity while moving readout response over baseline

This is closer to a learned local operator than a threshold gate.

## Verification

Contract tests passed after implementation and sweeps:

```text
pytest -q D:\RAFA\runtimes\circleworld_proto\test_learned_branch_law_child_ifs_contract.py D:\RAFA\runtimes\circleworld_proto\test_unit_phasor_contract.py D:\RAFA\runtimes\circleworld_proto\test_phase_gauge_invariance.py
....                                                                     [100%]
4 passed in 3.46s
```
