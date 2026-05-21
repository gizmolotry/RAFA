# Circleworld V17 Nested Gate - 2026-04-24

## What changed

We integrated nested-commitment response into Circleworld selection, not just post-hoc reporting.

Files changed:
- `D:\RAFA\runtimes\circleworld_proto\test_nested_commitment.py`
- `D:\RAFA\runtimes\circleworld_proto\train_circleworld_real_anchor.py`
- `D:\RAFA\runtimes\circleworld_proto\run_childworld_experiment.py`

The trainer now:
- refreshes nested commitment on top heldout candidates
- records `selection_nested` and `selection_nested_gate`
- can require nested response before promoting a checkpoint

New profile:
- `writeback_v17_nested_response_gate`

## Nested refresh baselines

`agreement_scout_v1`
- mean child response: -1.882186
- child active fraction: 0.000000
- child meso response: 0.000000
- nested sibling fraction: 0.000000
- over rigid fraction: 1.000000

`v16_benchmark_guard_nested`
- mean child response: -1.881911
- child active fraction: 0.000000
- child meso response: 0.000000
- nested sibling fraction: 0.000000
- over rigid fraction: 1.000000

These refreshed numbers show the same thing the ear has been saying: child worlds survive in heldout summaries, but direct child-history perturbation still produces no meaningful sibling continuation.

## V17 scout result

Run:
- `D:\RAFA\outputs\circleworld_proto\training_run_2026-04-24_nested_gate_scout_v17`

Checkpoint:
- `D:\RAFA\checkpoints_circleworld_proto\training_run_2026-04-24_nested_gate_scout_v17\circleworld_real_anchor_config_cem_v1.json`

Selection outcome:
- selection source: `init_config_fallback_nested_or_agreement_required`

Heldout summary:
- branch fraction: 0.577778
- meso branch effect: 0.014349
- child writeback: 0.097273
- child parent divergence: 0.140268
- naked branch: 0.400000
- naked meso: 0.003550
- naked writeback: 0.082799
- naked parent divergence: 0.042873

Benchmark:
- mean corr: 0.890472
- mean mae: 0.030270

Nested commitment:
- mean child response: -1.882364
- child active fraction: 0.000000
- child meso response: 0.000000
- nested sibling fraction: 0.000000
- over rigid fraction: 1.000000
- overall read: `nested_commitment_not_yet_established`

## Interpretation

This run is useful because it failed for the right reason.

- The heldout branch/writeback metrics are still real.
- The benchmark is still decent.
- The nested gate refused promotion anyway.
- The fallback was correct, because the nested assay stayed fully over-rigid.

So the current blocker is now sharper than before:

`Circleworld can preserve branch-local survival statistics without producing child-history-sensitive sibling continuation.`

## Recommendation

Do not spend the next cycle on broader search.

Do this next instead:
1. Add direct child-history perturbation terms to the optimization target, not only to late-stage selection.
2. Target nonzero `mean_child_active_fraction` and `mean_child_meso_response` under fork/resume while preserving coarse identity.
3. Keep `agreement_scout_v1` as the active checkpoint until a run clears both heldout branching and nested response.
