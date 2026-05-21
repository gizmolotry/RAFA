# Circleworld v13 Targeted Replay Recovery

Date: 2026-04-24

## Purpose

v13 was a narrow follow-up to v12.

Goal:
- start from the stronger v11 checkpoint
- keep replay-verified hard gating
- push only the weak replay terms: naked writeback and naked parent divergence

## Result

v13 did not produce a replay-passing new candidate.

Important trainer outcome:
- `selected_candidate = null`
- `selection_source = init_config_fallback`

That means the v13 checkpoint is not a newly validated search winner. It is the prepared-init fallback constitution.

## What improved anyway

The fallback constitution is still informative.

Compared with v11 heldout:
- overall writeback: 0.096726 -> 0.098863
- overall parent divergence: 0.156758 -> 0.140859
- benchmark corr: 0.863543 -> 0.863628

But the actual branch-first lane did not improve:
- naked branch: 0.400000 -> 0.333333
- v13 replay min naked parent div: 0.035375

So the bottleneck remains the same:
replay-stable naked parent divergence.

## Current recommendation

Keep the distinction sharp:
- best current Circleworld model checkpoint: D:\RAFA\checkpoints_circleworld_proto\training_run_2026-04-24_ramanujan_childworld_writeback_v11_hard_gate_seeded\circleworld_real_anchor_config_cem_v1.json
- best current safe trainer behavior: verified-gate + fallback constitution from v12/v13

The next move should not be a broad search. It should be a direct parent-divergence intervention while keeping replay verification on.

## Files

- Compare JSON: D:\RAFA\outputs\circleworld_proto\v11_v12_v13_replay_recovery_compare_2026-04-24.json
- v13 summary: D:\RAFA\outputs\circleworld_proto\training_run_2026-04-24_ramanujan_childworld_writeback_v13_targeted_replay_recovery\childworld_experiment_summary.json
- v13 train summary: D:\RAFA\outputs\circleworld_proto\training_run_2026-04-24_ramanujan_childworld_writeback_v13_targeted_replay_recovery\train_summary.json
