# Circleworld v12 Verified Gate Report

Date: 2026-04-24

## What changed

We added replay-aware transfer-probe verification to Circleworld candidate selection.

Instead of trusting a single transfer probe per candidate, the trainer now:
- replays the heldout-style probe 3 times
- aggregates branch metrics across repeats
- applies hard gate checks to the conservative minima for branch-critical terms
- sorts candidates using worst-case transfer score before blended train/val score

This targets the specific issue found in v11: the same config could replay with materially different naked law / parent-divergence measurements.

## Main result

The pipeline improvement is real, but the checkpoint improvement is not.

v12 is more honest than v11, but weaker as a model:
- v11 overall heldout branch: 0.577778
- v12 overall heldout branch: 0.518519
- v11 naked branch: 0.400000
- v12 naked branch: 0.222222
- v11 benchmark corr: 0.863543
- v12 benchmark corr: 0.849705

## Important diagnostic

Under repeated replay, v11 is not actually as safe as its one-shot gate looked.

Repeated v11 probe snapshot:
- naked_branch_min = 0.000000
- naked_writeback_min = 0.000000
- naked_parent_div_min = 0.000000

That means the original v11 winner benefited from one-shot probe optimism.

## Current recommendation

Keep two conclusions separate:
- best current Circleworld branch-first checkpoint: D:\RAFA\checkpoints_circleworld_proto\training_run_2026-04-24_ramanujan_childworld_writeback_v11_hard_gate_seeded\circleworld_real_anchor_config_cem_v1.json
- best current Circleworld selection constitution: replay-verified gate from v12

So the next run should start from the v11 checkpoint but use the v12 verified-gate trainer.

## Files

- Compare JSON: D:\RAFA\outputs\circleworld_proto\v11_vs_v12_verified_gate_compare_2026-04-24.json
- v11 summary: D:\RAFA\outputs\circleworld_proto\training_run_2026-04-24_ramanujan_childworld_writeback_v11_hard_gate_seeded\childworld_experiment_summary.json
- v12 summary: D:\RAFA\outputs\circleworld_proto\training_run_2026-04-24_ramanujan_childworld_writeback_v12_verified_gate_seeded\childworld_experiment_summary.json
