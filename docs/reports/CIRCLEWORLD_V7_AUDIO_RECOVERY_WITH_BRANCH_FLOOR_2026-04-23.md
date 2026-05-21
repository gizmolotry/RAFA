# Circleworld V7 Audio Recovery With Branch Floor - 2026-04-23

## Scope
Circleworld only. This pass started from the `v6_ramanujan_curriculum` checkpoint and attempted a partial audio recovery without allowing heldout `naked_rafa` branching to collapse back to zero.

## What Changed
- Added the `writeback_v7_audio_recovery_with_branch_floor` profile in [run_childworld_experiment.py](D:/RAFA/runtimes/circleworld_proto/run_childworld_experiment.py).
- Rebalanced the search toward benchmark preservation with stronger audio/baseline weights while keeping the source-aware naked branch penalties from v6.
- Raised the heldout probe floor so `naked_rafa` must keep roughly `0.333` branch fraction, plus nontrivial writeback and parent divergence.
- Updated [test_nested_commitment.py](D:/RAFA/runtimes/circleworld_proto/test_nested_commitment.py) to support `fork_selector=first_live_child`, so the assay can fork from states where child worlds are actually present.

## V6 vs V7

| run | heldout branch | naked branch | naked writeback | naked parent div | bench corr | bench mae | law fams |
|---|---:|---:|---:|---:|---:|---:|---:|
| v6_ramanujan_curriculum | 0.5556 | 0.3333 | 0.0803 | 0.0402 | 0.8599 | 0.0336 | 2.0000 |
| v7_ramanujan_recovery | 0.5556 | 0.3333 | 0.0773 | 0.0361 | 0.8617 | 0.0278 | 1.0000 |

## Main Findings
- The branch floor held. `v7_ramanujan_recovery` kept heldout `naked_rafa` branch fraction at `0.3333`, with child writeback `0.0773` and parent divergence `0.0361`.
- Audio recovered only slightly. Benchmark correlation moved from `0.8599` in v6 to `0.8617` in v7, while MAE improved from `0.0336` to `0.0278`. This is a real but modest recovery, not a restoration.
- The cost of recovery was law-family collapse. Heldout law families fell from `2.0` in v6 to `1.0` in v7, so the recovery profile is simplifying the law-token regime even while preserving branching.
- The nested assay is now using `fork_selector=first_live_child`, and at least some cases do fork later than the fixed depth. That means the assay is constitutionally better than before. But the outcome is still `nested_commitment_not_yet_established`.

## Diagnosis
V7 succeeded at the narrow objective: recover some audio without losing the real-substrate branch lane. It did not solve the deeper problem. The system can keep child worlds alive and write them back, but it still does not turn those sidecar worlds into durable sibling continuations. On top of that, the recovery profile is flattening law diversity.

## Recommendation
- Keep `v6_ramanujan_curriculum` as the best branch-first checkpoint.
- Treat `v7_ramanujan_recovery` as a partial recovery checkpoint, not the new default.
- Next pass should explicitly preserve law-family diversity while recovering audio. Right now the optimizer is regaining sound partly by collapsing the law-token structure.
- For nested commitment, stop perturbing only the parent state. Add branch-local perturbations directly to live childworld records before resume, then compare whether sibling continuation appears.

## Artifacts
- Comparison JSON: `D:\RAFA\outputs\circleworld_proto\childworld_v6_v7_recovery_compare_2026-04-23.json`
- V7 summary: `D:\RAFA\outputs\circleworld_proto	raining_run_2026-04-23_ramanujan_childworld_writeback_v7_audio_recovery_with_branch_floor\childworld_experiment_summary.json`
- V7 nested report: `D:\RAFA\outputs\circleworld_proto	raining_run_2026-04-23_ramanujan_childworld_writeback_v7_audio_recovery_with_branch_floor
ested_commitment
ested_commitment_report.json`

## Bottom Line
We can now recover some audio without erasing real-substrate branching. The remaining gap is qualitative: the branch lane survives, but it still does not behave like a Matryoshka of sibling lawful continuations.
