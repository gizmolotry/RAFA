# Circleworld Follow-Up

## What was done
Three follow-up tasks were executed on the Circleworld lane:

1. Added a held-out evaluator and trajectory export path
2. Compared the `v2` config against the 2000-iteration `full` config on held-out seeds
3. Added anti-saturation penalties and ran a new penalized training search

## New evaluator
- script: [D:\RAFA\runtimes\circleworld_proto\evaluate_circleworld.py](D:\RAFA\runtimes\circleworld_proto\evaluate_circleworld.py)

Outputs now include:
- held-out summary JSON
- per-time trajectory CSV

Artifacts:
- `v2`
  - [D:\RAFA\outputs\circleworld_proto\heldout_eval_v2_2026-04-08\heldout_summary.json](D:\RAFA\outputs\circleworld_proto\heldout_eval_v2_2026-04-08\heldout_summary.json)
  - [D:\RAFA\outputs\circleworld_proto\heldout_eval_v2_2026-04-08\heldout_trajectory.csv](D:\RAFA\outputs\circleworld_proto\heldout_eval_v2_2026-04-08\heldout_trajectory.csv)
- `full`
  - [D:\RAFA\outputs\circleworld_proto\heldout_eval_full_2026-04-08\heldout_summary.json](D:\RAFA\outputs\circleworld_proto\heldout_eval_full_2026-04-08\heldout_summary.json)
  - [D:\RAFA\outputs\circleworld_proto\heldout_eval_full_2026-04-08\heldout_trajectory.csv](D:\RAFA\outputs\circleworld_proto\heldout_eval_full_2026-04-08\heldout_trajectory.csv)
- `penalized`
  - [D:\RAFA\outputs\circleworld_proto\heldout_eval_penalized_2026-04-08\heldout_summary.json](D:\RAFA\outputs\circleworld_proto\heldout_eval_penalized_2026-04-08\heldout_summary.json)
  - [D:\RAFA\outputs\circleworld_proto\heldout_eval_penalized_2026-04-08\heldout_trajectory.csv](D:\RAFA\outputs\circleworld_proto\heldout_eval_penalized_2026-04-08\heldout_trajectory.csv)

## Held-out comparison
Machine-readable compare:
- [D:\RAFA\docs\reports\CIRCLEWORLD_HELDOUT_COMPARE_2026-04-08.json](D:\RAFA\docs\reports\CIRCLEWORLD_HELDOUT_COMPARE_2026-04-08.json)
- [D:\RAFA\docs\reports\CIRCLEWORLD_HELDOUT_TRIPLE_COMPARE_2026-04-08.json](D:\RAFA\docs\reports\CIRCLEWORLD_HELDOUT_TRIPLE_COMPARE_2026-04-08.json)

### v2
- `mean_major_gain = 0.00022`
- `mean_residue_drop = 0.00567`
- `mean_dominant_q_share = 0.64588`
- `mean_q_entropy = 0.58801`

### full
- `mean_major_gain = 0.30877`
- `mean_residue_drop = 0.14958`
- `mean_dominant_q_share = 0.99461`
- `mean_q_entropy = 0.02151`
- `mean_major_saturation = 0.05897`

### penalized
- `mean_major_gain = 0.0`
- `mean_residue_drop = 0.0`
- `mean_dominant_q_share = 0.22910`
- `mean_q_entropy = 0.90059`

## Read
The three configs now form a clear triangle:

- `full`: wins the old structural objective but does so by almost total q-collapse
- `penalized`: avoids collapse, but becomes effectively inert
- `v2`: weak but still the most balanced of the three

## Anti-saturation changes
The Circleworld objective now tracks:
- `dominant_q_share`
- `q_entropy`
- `major_saturation`

Penalty terms added:
- `l_q_dom`
- `l_q_entropy`
- `l_major_sat`

Relevant files:
- [D:\RAFA\lineages\04_positive_replacement\circleworld.py](D:\RAFA\lineages\04_positive_replacement\circleworld.py)
- [D:\RAFA\runtimes\circleworld_proto\train_circleworld.py](D:\RAFA\runtimes\circleworld_proto\train_circleworld.py)

## Penalized training run
- summary: [D:\RAFA\outputs\circleworld_proto\training_run_2026-04-08_penalized\train_summary.json](D:\RAFA\outputs\circleworld_proto\training_run_2026-04-08_penalized\train_summary.json)
- checkpoint: [D:\RAFA\checkpoints_circleworld_proto\training_run_2026-04-08_penalized\circleworld_config_cem_v1.json](D:\RAFA\checkpoints_circleworld_proto\training_run_2026-04-08_penalized\circleworld_config_cem_v1.json)

## Verdict
The anti-saturation penalty worked mechanically, but too strongly.

We now know:
- the unconstrained objective over-rewards q-collapse
- the current penalty can overcorrect into a no-op regime

That means the next version should not choose between:
- pathological domination
- total passivity

It needs a middle-band target:
- enough structural intervention to create real gains
- enough diversity to avoid single-q collapse
