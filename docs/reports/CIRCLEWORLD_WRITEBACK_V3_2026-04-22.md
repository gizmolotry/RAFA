# Circleworld Writeback V3 2026-04-22

## Scope

This report covers the `writeback_v3_identity` Circleworld run.
Graduation was not touched.

Compared runs:
- `v1_branch_survival`
- `v2_writeback`
- `v3_identity_writeback`

Machine comparison:
- `D:\RAFA\outputs\circleworld_proto\childworld_progression_compare_2026-04-22.json`

## Why this run existed

Childworld v1 gave us branch survival without causality.
Childworld v2 gave us causality without enough branch identity.

So v3 changed the constitution to preserve both:
- weaker parent mode-0 writeback
- stronger child self-retention after writeback
- a search profile that rewards branch identity and writeback at the same time

## Main result

This is the strongest Circleworld branch result so far.

### v1
- `mean_real_branch_fraction = 0.4444`
- `mean_meso_branch_effect = 0.00712`
- `mean_child_writeback_mass = 0.0`
- `mean_child_parent_divergence = 0.14235`
- `mean_corr = 0.9825`
- `mean_mae = 0.01078`

### v2
- `mean_real_branch_fraction = 0.0`
- `mean_meso_branch_effect = 0.00351`
- `mean_child_writeback_mass = 0.02590`
- `mean_child_parent_divergence = 0.07021`
- `mean_corr = 0.9691`
- `mean_mae = 0.02011`

### v3
- `mean_real_branch_fraction = 0.4444`
- `mean_meso_branch_effect = 0.01522`
- `mean_child_writeback_mass = 0.06530`
- `mean_child_parent_divergence = 0.15538`
- `mean_child_sibling_divergence = 0.13144`
- `mean_num_law_families = 1.78`
- `mean_law_family_entropy = 0.1544`
- `mean_corr = 0.9538`
- `mean_mae = 0.01877`

## Clean read

V3 is the first run that keeps both:
- nonzero held-out branch fraction
- nonzero held-out writeback

That is the key result.

Relative to v2:
- branch fraction recovered: `0.0 -> 0.4444`
- meso branch effect increased: `0.00351 -> 0.01522`
- writeback increased: `0.02590 -> 0.06530`
- child-parent divergence increased: `0.07021 -> 0.15538`
- law-family diversity improved:
  - `1.67 -> 1.78`
  - entropy `0.108 -> 0.154`
- aggregate families improved:
  - `8 -> 9`

Relative to v1:
- writeback improved: `0.0 -> 0.06530`
- meso branch effect improved strongly: `0.00712 -> 0.01522`
- child-parent divergence improved: `0.14235 -> 0.15538`
- law-family diversity improved:
  - `1.0 -> 1.78`
  - entropy `0.0 -> 0.154`

The cost:
- benchmark fidelity regressed relative to v1
  - corr `0.9825 -> 0.9538`
  - mae `0.01078 -> 0.01877`

So the current state is:
- better Circleworld branching constitution
- worse audio preservation than the best earlier branch-survival run

## What still failed

Nested commitment still did not flip:
- `overall_read = nested_commitment_not_yet_established`
- `over_rigid_attractor = 12`

`naked_rafa` still did not branch:
- childworld activity remains concentrated in synthetic rows

So v3 is not the end state.
But it is the first run where branching, writeback, and identity are all alive together in one regime.

## Interpretation

The project has moved through three distinct Circleworld constitutions:

### v1
- child worlds can survive
- child worlds do not matter

### v2
- child worlds matter
- child worlds collapse into the parent too hard

### v3
- child worlds matter
- child worlds stay distinct enough to remain branches

That is real progress.

The next bottleneck is now much narrower:
- transfer branch behavior from synthetic seeds to real/naked_rafa seeds
- improve nested commitment without losing v3 writeback/identity

## Recommendation

Next run should be `writeback_v4_real_anchor_transfer`.

Priority:
1. keep v3 branch fraction and writeback alive
2. increase weighting on real-anchor and naked_rafa rows
3. penalize synthetic-only success
4. preserve child-parent divergence after writeback
5. keep Ramanujan as the default lane

The immediate question is no longer whether the childworld constitution works in principle.
It does.

The question is whether it can transfer to the real substrate.

## Artifacts

V3 run:
- `D:\RAFA\outputs\circleworld_proto\training_run_2026-04-22_ramanujan_childworld_writeback_v3_identity\childworld_experiment_summary.json`
- `D:\RAFA\outputs\circleworld_proto\training_run_2026-04-22_ramanujan_childworld_writeback_v3_identity\heldout_eval\heldout_summary.json`
- `D:\RAFA\outputs\circleworld_proto\training_run_2026-04-22_ramanujan_childworld_writeback_v3_identity\benchmark_expanded\benchmark_summary.json`
- `D:\RAFA\outputs\circleworld_proto\training_run_2026-04-22_ramanujan_childworld_writeback_v3_identity\nested_commitment\nested_commitment_report.json`
- `D:\RAFA\outputs\circleworld_proto\training_run_2026-04-22_ramanujan_childworld_writeback_v3_identity\law_token_library\law_token_library.json`
- `D:\RAFA\checkpoints_circleworld_proto\training_run_2026-04-22_ramanujan_childworld_writeback_v3_identity\circleworld_real_anchor_config_cem_v1.json`

Progression compare:
- `D:\RAFA\outputs\circleworld_proto\childworld_progression_compare_2026-04-22.json`
