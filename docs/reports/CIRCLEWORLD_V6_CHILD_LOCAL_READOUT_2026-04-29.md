# Circleworld V6 Child-Local Readout (2026-04-29)

## Scope
Circleworld only. Graduation untouched.

## What changed
We added a temporary child-local readout path inside the nested assay in `D:\RAFA\runtimes\circleworld_proto\test_nested_commitment.py`.

New branches:
- `child_local_readout_shift`
- `child_dominant_readout_shift`

These branches:
- perturb a live childworld record
- apply the existing lifecycle / pre-unroll path
- override final readout so the strongest live child can speak directly under its support mask before normal parent mixture dominates

## Run
- Config: `D:\RAFA\checkpoints_circleworld_proto\training_run_2026-04-24_agreement_scout_v1\circleworld_real_anchor_config_cem_v1.json`
- Output: `D:\RAFA\outputs\circleworld_proto\nested_seeded_refresh_agreement_scout_v1_child_readout_v6_2026-04-29`
- Compare JSON: `D:\RAFA\outputs\circleworld_proto\nested_seeded_compare_v6_2026-04-29.json`

## Main result
The child-local readout hypothesis is partially true and partially false.

True:
- child-local readout changes the rendered continuation much more than the previous gate/remix perturbations
- coarse identity stays very high
- texture and structural gains move sharply upward on the readout branches

False:
- this still does not create live child-active continuation under the current assay
- `mean_child_active_fraction` stayed `0.0`
- `mean_child_meso_response` stayed `0.0`
- `mean_nested_sibling_fraction` stayed `0.0`
- overall read stayed `nested_commitment_not_yet_established`

## Aggregate comparison vs v5
- `v5 mean_child_response_score = 0.8784418513`
- `v6 mean_child_response_score = 0.8781791075`
- delta: `-0.0002627439`
- `v5 mean_coarse_preservation = 0.9998278259`
- `v6 mean_coarse_preservation = 0.9994962444`
- both runs: `mean_child_active_fraction = 0.0`
- both runs: `mean_child_meso_response = 0.0`
- both runs: `overall_read = nested_commitment_not_yet_established`

So the readout branch made the probe more behaviorally expressive, but it did not improve the current scalar nested score because that score still rewards near-frozen coarse preservation more than readout divergence.

## Important branch-level observation
The new readout branches are the first ones in this lane to cause a visibly larger child-shaped deviation while preserving the broad world:

Typical `child_local_readout_shift`:
- `coarse_env_corr ~= 0.99825`
- `fine_texture_distance ~= 0.0099`
- `major_gain ~= 0.13619`
- `promotability_gain ~= 0.12414`

Typical `child_dominant_readout_shift`:
- `coarse_env_corr ~= 0.99481`
- `fine_texture_distance ~= 0.01279`
- `major_gain ~= 0.15918`
- `promotability_gain ~= 0.13927`

But both still land as `ambiguous_middle`, not `nested_sibling`.

## Diagnosis
This narrows the bottleneck again.

We now have evidence that:
- the child law can alter readout if we explicitly let it speak
- immediate parent mixture is part of the suppression problem
- but even with child-local readout, the child does not become an independently surviving continuation channel in the current ontology

So the failure is no longer just lifecycle gates, writeback timing, or parent protection.
It is now much closer to:
- missing child-local continuation semantics
- or a mismatch between what the current nested score rewards and what child-local branching actually looks like

## Recommendation
Next three steps after this run should be:
1. add a child-only continuation comparator in the nested assay, not just child-local final readout
2. score readout-driven sibling divergence separately from child-active survival so we stop collapsing both questions into one metric
3. test a 1-2 step child-dominant continuation branch before parent remix, then compare that against the current child-local readout branch
