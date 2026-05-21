# Circleworld Writeback V2 2026-04-22

## Scope

This run continues the childworld branch-reset line inside Circleworld only.
Graduation was not touched.

Compared against:
- `ramanujan_childworld_v1`
- `writeback_smoke`

Machine comparison:
- `D:\RAFA\outputs\circleworld_proto\childworld_writeback_compare_2026-04-22.json`

## Why this run existed

Childworld v1 proved that child worlds could exist and survive.
It failed because they never wrote back.

So v2 changed two things:
- the runtime now allows early writeback under strong local divergence/coherence conditions
- the experiment profile explicitly rewards writeback and penalizes delay

## Implementation changes

Updated:
- `D:\RAFA\lineages\04_positive_replacement\circleworld.py`
- `D:\RAFA\runtimes\circleworld_proto\run_childworld_experiment.py`

Main constitutional change in runtime:
- early writeback can occur at age 1 when parent divergence, sibling divergence, coherence, and local defect are all high enough

Main experiment change:
- new `writeback_v2` profile in the childworld runner
- stronger reward on `child_writeback_mass`
- lower reward on just keeping children alive
- negative weight on branch delay
- more permissive childworld init defaults for writeback

## Main result

Writeback is now real in the full Ramanujan run.

### Childworld v1
- `mean_child_writeback_mass = 0.0`
- `mean_real_branch_fraction = 0.4444`
- `mean_meso_branch_effect = 0.00712`
- `mean_corr = 0.9825`
- `mean_mae = 0.01078`

### Writeback v2
- `mean_child_writeback_mass = 0.02590`
- `mean_real_branch_fraction = 0.0`
- `mean_meso_branch_effect = 0.00351`
- `mean_corr = 0.9691`
- `mean_mae = 0.02011`

## Interpretation

This is a real trade.

What improved:
- child worlds now write back during the full run
- synthetic rows show repeated child writeback events
- held-out law-family diversity improved slightly over v1
  - `mean_num_law_families: 1.0 -> 1.67`
  - `mean_law_family_entropy: 0.0 -> 0.108`
- aggregate token library count improved
  - aggregate families: `7 -> 8`

What regressed:
- held-out real branch fraction fell back to `0.0`
- meso branch effect dropped
- audio fidelity dropped materially
- nested commitment still failed on all 12 cases
- naked_rafa still produced no childworld branching or writeback

So v2 solved the *causality* bottleneck but weakened the *branch identity* signal.

## Most important detail

The trained config drifted to:
- `child_min_age_for_writeback = 2`

But writeback still happened because the runtime's early-writeback path is now active.
That means the runtime fix was necessary, not cosmetic.

The held-out summary confirms that:
- writeback survived optimization
- it did not vanish back to zero

## What this means scientifically

We now have two distinct regimes.

### v1: surviving but non-causal child worlds
- branch-looking
- not causally active

### v2: causally active child worlds
- writeback exists
- but the system no longer preserves enough parent/child divergence to count as real branching under the current threshold

That is progress.
It means the bottleneck has moved again.

The new problem is no longer:
- can child worlds survive?
- can child worlds write back?

The new problem is:
- can child worlds write back **without collapsing branch identity**?

## Recommendation

Next run should be `writeback_v3_identity_preserving`.

Priority changes:
1. keep nonzero writeback as a hard target
2. raise reward on child-parent divergence after writeback
3. penalize writeback that reduces branch identity below threshold
4. add a post-writeback sibling-preservation term
5. keep Ramanujan as the default lane

The right next question is now:
- can a child write back into the parent while still remaining a sibling world rather than collapsing into it?

## Artifacts

Writeback v2:
- `D:\RAFA\outputs\circleworld_proto\training_run_2026-04-22_ramanujan_childworld_writeback_v2\childworld_experiment_summary.json`
- `D:\RAFA\outputs\circleworld_proto\training_run_2026-04-22_ramanujan_childworld_writeback_v2\heldout_eval\heldout_summary.json`
- `D:\RAFA\outputs\circleworld_proto\training_run_2026-04-22_ramanujan_childworld_writeback_v2\benchmark_expanded\benchmark_summary.json`
- `D:\RAFA\outputs\circleworld_proto\training_run_2026-04-22_ramanujan_childworld_writeback_v2\nested_commitment\nested_commitment_report.json`
- `D:\RAFA\outputs\circleworld_proto\training_run_2026-04-22_ramanujan_childworld_writeback_v2\law_token_library\law_token_library.json`
- `D:\RAFA\checkpoints_circleworld_proto\training_run_2026-04-22_ramanujan_childworld_writeback_v2\circleworld_real_anchor_config_cem_v1.json`

Comparison:
- `D:\RAFA\outputs\circleworld_proto\childworld_writeback_compare_2026-04-22.json`
