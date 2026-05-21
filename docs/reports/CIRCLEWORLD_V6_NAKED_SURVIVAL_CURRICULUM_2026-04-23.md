# Circleworld V6 Naked Survival Curriculum Report - 2026-04-23

## Scope
Circleworld only. Graduation remained untouched. This stage implemented a source-aware branch survival curriculum with explicit `naked_rafa` penalties for defect-without-branch, zero child worlds, zero writeback, and branch shortfall.

## What Changed
- Added `naked_rafa`-specific transfer scoring in [train_circleworld_real_anchor.py](D:/RAFA/runtimes/circleworld_proto/train_circleworld_real_anchor.py).
- Added the `writeback_v6_naked_survival_curriculum` profile in [run_childworld_experiment.py](D:/RAFA/runtimes/circleworld_proto/run_childworld_experiment.py).
- Tightened childworld init defaults for the curriculum run: lower spawn threshold, longer support window, slower decay, stronger writeback gate, and stricter child branch thresholds.
- Ran both Ramanujan and qtrace control experiments from their respective branch-law baselines.

## Comparison

| run | heldout branch | naked heldout branch | naked writeback | naked parent div | synth heldout branch | law fams | bench corr | bench mae |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| v3_ramanujan_identity | 0.4444 | n/a | n/a | n/a | n/a | 1.7778 | 0.9538 | 0.0188 |
| v5_smoke_ramanujan | 0.5556 | 0.3333 | 0.0566 | 0.0376 | 0.6667 | 2.0000 | 0.9353 | 0.0359 |
| v6_ramanujan_curriculum | 0.5556 | 0.3333 | 0.0803 | 0.0402 | 0.6667 | 2.0000 | 0.8599 | 0.0336 |
| v6_qtrace_curriculum | 0.4444 | 0.0000 | 0.0340 | 0.0168 | 0.6667 | 1.0000 | 0.9173 | 0.0270 |

## Main Findings
- `v6_ramanujan_curriculum` is the first full Circleworld run in this sequence that keeps `naked_rafa` branching alive on heldout, not just on the training probe. Heldout `naked_rafa` branch fraction is `0.3333`, with writeback `0.0803` and parent divergence `0.0402`.
- `v6_qtrace_curriculum` does not hold that gain. It shows some naked branching in the training probe, but heldout `naked_rafa` collapses back to `0.0000`. That means the curriculum is not sufficient by itself; Ramanujan is doing real constitutional work here.
- Relative to `v3_ramanujan_identity`, the new Ramanujan curriculum run improves heldout branch mass and child writeback, and now exposes a nonzero naked branch lane. But it pays for that with worse benchmark audio preservation (`corr 0.8599` vs `0.9538`).
- `v5_smoke_ramanujan` already hinted that naked heldout branching was reachable. `v6_ramanujan_curriculum` turns that from a smoke-pocket into a repeatable full-run result.
- Nested commitment is still unresolved. All compared runs remain `over_rigid_attractor`, so we have branch emergence without Matryoshka-like sibling continuation yet.

## Diagnosis
The curriculum fix worked in the specific place we needed it to work: the search can no longer satisfy itself entirely through synthetic childworlds when using the Ramanujan branch law. In qtrace-only mode, the same curriculum is not enough to keep real-substrate child worlds alive on heldout. That is strong evidence that Ramanujan is not ornamental in this regime; it is part of the survival law.

The remaining problem is tradeoff. The branch constitution is now stronger, but the audio benchmark regressed. So the next Circleworld move is not ?more branch pressure.? It is controlled recovery of audio/continuity while preserving the newly recovered naked branch lane.

## Recommendation
- Keep `v6_ramanujan_curriculum` as the new best branch-first research checkpoint.
- Keep `v3_ramanujan_identity` as the better sound-preserving checkpoint.
- Do not advance `v6_qtrace_curriculum` as a primary lane; it failed heldout naked branching.
- Next implementation should be a branch-preserving audio recovery pass: start from `v6_ramanujan_curriculum`, restore some benchmark weighting, and constrain the search so heldout `naked_rafa` branch fraction cannot fall below `0.3333`.
- The nested assay should now fork from known live childworld states taken from the `v6` heldout rollouts, not generic anchor states. That is the right test for whether these sidecar worlds can become sibling continuations rather than just survive briefly.

## Artifacts
- Comparison JSON: `D:\RAFA\outputs\circleworld_proto\childworld_v3_v6_curriculum_compare_2026-04-23.json`
- Ramanujan curriculum summary: `D:\RAFA\outputs\circleworld_proto	raining_run_2026-04-23_ramanujan_childworld_writeback_v6_naked_survival_curriculum\childworld_experiment_summary.json`
- Qtrace curriculum summary: `D:\RAFA\outputs\circleworld_proto	raining_run_2026-04-23_qtrace_only_childworld_writeback_v6_naked_survival_curriculum\childworld_experiment_summary.json`

## Bottom Line
We forced real-substrate branching back into existence in the Ramanujan lane. That is a real constitutional result. The next job is not to rediscover branch pressure; it is to keep this branch lane alive while regaining audio quality and testing true sibling continuation.
