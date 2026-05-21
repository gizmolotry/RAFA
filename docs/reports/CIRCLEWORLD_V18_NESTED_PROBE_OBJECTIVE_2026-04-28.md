# Circleworld V18 Nested Probe Objective - 2026-04-28

## What changed

We moved nested child-history response into the training loop itself.

Files changed:
- `D:\RAFA\runtimes\circleworld_proto\train_circleworld_real_anchor.py`
- `D:\RAFA\runtimes\circleworld_proto\run_childworld_experiment.py`

New behavior:
- top candidates in each iteration get a nested probe run
- the probe writes `nested_probe_score` into candidate rows
- elite selection can now feel nested-response pressure during search
- late-stage nested gate remains in place

New profile:
- `writeback_v18_nested_probe_objective`

## Run

Output:
- `D:\RAFA\outputs\circleworld_proto\training_run_2026-04-28_nested_probe_objective_v18`

Checkpoint:
- `D:\RAFA\checkpoints_circleworld_proto\training_run_2026-04-28_nested_probe_objective_v18\circleworld_real_anchor_config_cem_v1.json`

## Result

Selection outcome:
- `v17`: `init_config_fallback_nested_or_agreement_required`
- `v18`: `init_config_fallback_nested_or_agreement_required`

So `v18` still fell back. No promotable checkpoint cleared the combined heldout + nested bar.

## Important finding

The in-loop nested probe is now active, but on the current real-anchor nested cases it behaves almost like a constant penalty.

Nested probe stats across all 16 candidate evaluations:
- count: 16
- max probe score: -170.155301
- min probe score: -170.157925
- mean probe score: -170.157036
- max child response: -1.881911
- min child response: -1.882240
- mean child response: -1.882129
- max child active fraction: 0.000000
- max child meso response: 0.000000

This means the nested probe is no longer missing from optimization, but it still provides almost no discriminative signal in this case family.

## Final v18 metrics

Heldout:
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

Nested final:
- mean child response: -1.882364
- child active fraction: 0.000000
- child meso response: 0.000000
- nested sibling fraction: 0.000000
- over rigid fraction: 1.000000
- overall read: `nested_commitment_not_yet_established`

## Interpretation

This run sharpened the diagnosis again.

The problem is no longer:
- missing evaluator wiring
- missing trainer selection wiring
- missing in-loop nested objective

The problem is now:

`the current real-anchor nested cases almost never reach a live child-history regime, so the nested probe sees nearly the same dead response for every candidate.`

In other words, the optimization target exists now, but the probe substrate is too flat to guide search.

## Recommendation

Next step should not be a broader resweep.

Next step should be a new nested probe substrate:
1. use seeded `naked_rafa` / branch-active states for the nested probe, not only real-anchor fork points
2. probe from states where live child worlds already exist
3. then rerun the same in-loop nested objective

That is the shortest path to turning nested response from a constant penalty into a real optimization signal.
