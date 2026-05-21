# Circleworld V7 Child-Only Continuation (2026-04-29)

## Scope
Circleworld only. Graduation untouched.

## What changed
We extended the nested assay in `D:\RAFA\runtimes\circleworld_proto\test_nested_commitment.py` with a true child-only continuation path.

New branches:
- `child_local_continuation_shift`
- `child_dominant_continuation_shift`

These branches:
- start from live-child seeded fork states
- give the child 1-2 extra child-only continuation steps before normal parent readout
- then render from a child-led continuation phase instead of only overriding the final readout instant

We also split the nested metrics so we now separately track:
- `mean_child_survival_signal`
- `mean_readout_sibling_response`

## Run
- Config: `D:\RAFA\checkpoints_circleworld_proto\training_run_2026-04-24_agreement_scout_v1\circleworld_real_anchor_config_cem_v1.json`
- Output: `D:\RAFA\outputs\circleworld_proto\nested_seeded_refresh_agreement_scout_v1_child_continuation_v7_2026-04-29`
- Compare JSON: `D:\RAFA\outputs\circleworld_proto\nested_seeded_compare_v7_2026-04-29.json`

## Main result
This is the cleanest separation we have so far.

What moved:
- `mean_readout_sibling_response` is now explicitly measurable
- `v7 mean_readout_sibling_response = 0.3141068449`
- the new continuation branches are slightly stronger than the earlier readout-only branches on major gain and promotability gain

What did not move:
- `mean_child_survival_signal = 0.0`
- `mean_child_active_fraction = 0.0`
- `mean_child_meso_response = 0.0`
- `mean_nested_sibling_fraction = 0.0`
- overall read remains `nested_commitment_not_yet_established`

## Comparison vs v6
- `v6 mean_child_response_score = 0.8781791075`
- `v7 mean_child_response_score = 0.8779457701`
- delta: `-0.0002333374`
- `v6 mean_readout_sibling_response = 0.0`
- `v7 mean_readout_sibling_response = 0.3141068449`
- `v6 mean_coarse_preservation = 0.9994962444`
- `v7 mean_coarse_preservation = 0.9992096461`
- both runs: `mean_child_survival_signal = 0.0`
- both runs: `mean_child_active_fraction = 0.0`

So the continuation path made the readout-side sibling signal visible, but it still did not create a live surviving child branch.

## Branch-level read
Typical readout-only branch:
- `child_local_readout_shift`
  - coarse corr about `0.99825`
  - texture about `0.0099`
  - major gain about `0.13619`
- `child_dominant_readout_shift`
  - coarse corr about `0.99481`
  - texture about `0.01279`
  - major gain about `0.15918`

Typical continuation branch:
- `child_local_continuation_shift`
  - coarse corr about `0.99800`
  - texture about `0.01051`
  - major gain about `0.14633`
- `child_dominant_continuation_shift`
  - coarse corr about `0.99474`
  - texture about `0.01292`
  - major gain about `0.16056`

So the continuation branches are real. They are not redundant with the readout-only branches.

## Diagnosis
We now have a sharper constitution-level answer:
- the child law can influence a child-led continuation path
- extending the child by 1-2 steps does increase child-shaped readout deviation
- but the child still does not register as an independently surviving continuation channel in the current branch ontology

That means the next blocker is no longer “can the child speak at all?”
It is:
- how to represent child survival / sibling continuation as a real stateful branch, rather than only as a readout-side deformation

## Recommendation
Next three steps after this run should be:
1. add a child-history carry metric that tracks whether the same child identity persists through the continuation branch
2. add a branch where parent mode 1 is partially replaced by the child state for the continuation interval, not just read out through it
3. score child-led continuation against the base branch on a dedicated sibling-divergence metric instead of relying on the legacy combined response score
