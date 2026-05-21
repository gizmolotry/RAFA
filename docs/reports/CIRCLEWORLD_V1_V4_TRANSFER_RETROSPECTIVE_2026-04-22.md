# Circleworld V1-V4 Transfer Retrospective (2026-04-22)

## Bottom line

- `v3_ramanujan_identity` remains the best overall constitutional checkpoint if the goal is strongest live branching plus identity-preserving writeback.
- `v4_ramanujan_transfer` did not produce nonzero `naked_rafa` real-branch fraction, but it did move the harder substrate off the old dead-zero regime by sustaining child worlds and writeback there.
- `v4_qtrace_transfer` behaves similarly on the harder substrate, but with worse audio fidelity and weaker ontology, so Ramanujan remains the better default lane.
- Nested commitment is still not established in all runs.

## Key comparisons

### v1_ramanujan_childworld
- heldout real_branch_fraction: 0.444444
- heldout meso_branch_effect: 0.007118
- heldout child_writeback_mass: 0.000000
- heldout child_parent_divergence: 0.142351
- benchmark mean_corr: 0.982516
- benchmark mean_mae: 0.010777

### v2_ramanujan_writeback
- heldout real_branch_fraction: 0.000000
- heldout meso_branch_effect: 0.003511
- heldout child_writeback_mass: 0.025896
- heldout child_parent_divergence: 0.070214
- benchmark mean_corr: 0.969062
- benchmark mean_mae: 0.020106

### v3_ramanujan_identity
- heldout real_branch_fraction: 0.444444
- heldout meso_branch_effect: 0.015219
- heldout child_writeback_mass: 0.065297
- heldout child_parent_divergence: 0.155381
- benchmark mean_corr: 0.953772
- benchmark mean_mae: 0.018769

### v4_ramanujan_transfer
- heldout real_branch_fraction: 0.444444
- heldout meso_branch_effect: 0.007347
- heldout child_writeback_mass: 0.072627
- heldout child_parent_divergence: 0.097194
- benchmark mean_corr: 0.953606
- benchmark mean_mae: 0.019821
- synthetic: branch=0.666667, meso=0.009797, writeback=0.077572, parent_div=0.126292
- naked_rafa: branch=0.000000, meso=0.002447, writeback=0.062738, parent_div=0.038999

### v4_qtrace_transfer
- heldout real_branch_fraction: 0.444444
- heldout meso_branch_effect: 0.010054
- heldout child_writeback_mass: 0.081341
- heldout child_parent_divergence: 0.118105
- benchmark mean_corr: 0.854499
- benchmark mean_mae: 0.047918
- synthetic: branch=0.666667, meso=0.013648, writeback=0.087149, parent_div=0.156607
- naked_rafa: branch=0.000000, meso=0.002866, writeback=0.069724, parent_div=0.041102

## Read

- `v1` proved branch survival without causality.
- `v2` proved causality without preserving branch identity.
- `v3` was the first regime with nonzero held-out branching and nonzero held-out writeback at the same time.
- `v4_ramanujan_transfer` improved transfer-side child-world survival and writeback on `naked_rafa`, but the branch fraction threshold still does not trip there.
- `v4_qtrace_transfer` gives slightly stronger aggregate held-out meso/writeback than Ramanujan in some synthetic rows, but worse audio and fewer aggregate law families. It is not a better replacement lane.

## Recommendation

- Keep `v3_ramanujan_identity` as the current best Circleworld constitutional checkpoint.
- Treat `v4_ramanujan_transfer` as the best transfer experiment, not the new default checkpoint.
- The next step should be `v5_thresholded_real_branch`: lower or reformulate the real-branch criterion on `naked_rafa` so child worlds with nonzero writeback/divergence can become counted siblings when they are genuinely distinct at meso scale.
- In parallel, the nested-commitment assay needs to inspect child-history divergence earlier, because the current fork perturbations still miss the live branch window on real anchors.
