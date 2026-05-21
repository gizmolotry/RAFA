# Circleworld V5 Thresholded Real Branch Report - 2026-04-23

## Scope
Circleworld only. Graduation/runtime restore was not touched. This stage implemented child-aware branch accounting, upgraded nested-commitment probes to read childworld evidence, and ran Ramanujan/qtrace sidecar comparisons.

## What changed
- Added child-specific branch thresholds to `CircleworldConfig`: parent divergence, writeback mass, meso effect, and live support.
- Changed childworld branch counting so a live child counts as branch evidence only when it survives with support, diverges from parent, and writes back through the gated path.
- Added held-out `by_source` summaries for synthetic vs `naked_rafa`, because aggregate branch scores were hiding transfer failure.
- Added child-aware nested-commitment labels and per-branch child probes, so `nested_sibling` now requires child real-branch evidence rather than only parent-output differences.
- Added the `writeback_v5_thresholded_real_branch` profile and ran Ramanujan/qtrace variants with the full sidecar stack.

## Run comparison

| run | heldout branch | naked branch | synthetic branch | meso effect | writeback | parent div | law families | bench corr | bench mae |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| v3_ramanujan_identity | 0.4444 | n/a | n/a | 0.0152 | 0.0653 | 0.1554 | 1.7778 | 0.9538 | 0.0188 |
| v4_ramanujan_transfer | 0.4444 | 0.0000 | 0.6667 | 0.0073 | 0.0726 | 0.0972 | 1.0000 | 0.9536 | 0.0198 |
| v4_qtrace_transfer | 0.4444 | 0.0000 | 0.6667 | 0.0101 | 0.0813 | 0.1181 | 1.0000 | 0.8545 | 0.0479 |
| v5_smoke_ramanujan | 0.5556 | 0.3333 | 0.6667 | 0.0082 | 0.0680 | 0.1145 | 2.0000 | 0.9353 | 0.0359 |
| v5_ramanujan | 0.4444 | 0.0000 | 0.6667 | 0.0051 | 0.0459 | 0.0746 | 1.8889 | 0.9494 | 0.0248 |
| v5b_ramanujan | 0.4444 | 0.0000 | 0.6667 | 0.0072 | 0.0491 | 0.0977 | 2.3333 | 0.9639 | 0.0228 |
| v5_qtrace | 0.4444 | 0.0000 | 0.6667 | 0.0067 | 0.0493 | 0.0905 | 1.0000 | 0.9608 | 0.0233 |

## Main findings
- `v5_smoke_ramanujan` is the only run in this set with nonzero `naked_rafa` branch fraction: `0.3333`. It also had naked child writeback `0.0566` and parent divergence `0.0376`. This proves the child-aware criterion can detect a real naked branch pocket when it exists.
- Full v5 Ramanujan runs did not stabilize that pocket. Both `v5_ramanujan` and `v5b_ramanujan` returned to zero naked child worlds on standard heldout. That means branch survival is not yet robust; the search still finds synthetic-friendly childworlds and lets real-substrate branches die.
- `v5b_ramanujan` has the best benchmark preservation in this set (`corr=0.9639`, `mae=0.0228`) and better law-family diversity than qtrace, but it is not a better branch runtime because naked childworlds are zero.
- `v5_qtrace` behaves like a narrower kernel: synthetic branches exist, naked branches remain zero, and law-family diversity collapses to `1.0` in heldout. This supports keeping Ramanujan as the default branch-law lane.
- `v3_ramanujan_identity` remains the best stable constitutional checkpoint for now. It has stronger overall meso effect and parent divergence than the full v5 runs, even though it lacks the newer child-aware branch accounting.
- Nested commitment is still not established. Every compared run reports `over_rigid_attractor` across the nested assay. Childworlds can exist in heldout rollouts, but the fork/resume test is not yet producing durable `nested_sibling` cases.

## Diagnosis
The v5 implementation did not fail because childworlds are impossible. It failed because childworld survival is source-fragile. Synthetic states have enough defect/coherence geometry to spawn and preserve sidecar worlds. `naked_rafa` states often have weak or differently shaped branch evidence, so full training learns to satisfy aggregate objectives through synthetic branches while leaving real-substrate branch capacity dormant.

The short smoke run caught the thing we want: real naked branch support, writeback, and parent divergence. The full search then optimized away from it. That is the signal: threshold tuning alone is not enough. The next intervention needs to make real-substrate branch survival non-optional when defect pressure is present.

## Recommendation
- Keep `v3_ramanujan_identity` as the current best stable Circleworld checkpoint.
- Treat `v5_smoke_ramanujan` as evidence, not as a default runtime.
- Do not promote `v5b` to default despite its better benchmark preservation; it has zero naked childworlds.
- Next implementation should add a branch-local survival curriculum: source-balanced minibatches, a naked_rafa spawn/writeback floor under high defect, and explicit penalty for `defect > threshold` with `child_world_count == 0`.
- Nested commitment should be rerun from states where childworlds are known to be live, not only from generic real-anchor fork points. Otherwise the test mostly measures parent rigidity before branching has had a chance to exist.

## Artifacts
- Comparison JSON: `D:\RAFA\outputs\circleworld_proto\childworld_v3_v5_threshold_compare_2026-04-23.json`
- v3_ramanujan_identity: `D:\RAFA\outputs\circleworld_proto\training_run_2026-04-22_ramanujan_childworld_writeback_v3_identity\childworld_experiment_summary.json`
- v4_ramanujan_transfer: `D:\RAFA\outputs\circleworld_proto\training_run_2026-04-22_ramanujan_childworld_writeback_v4_real_anchor_transfer\childworld_experiment_summary.json`
- v4_qtrace_transfer: `D:\RAFA\outputs\circleworld_proto\training_run_2026-04-22_qtrace_only_childworld_writeback_v4_real_anchor_transfer\childworld_experiment_summary.json`
- v5_smoke_ramanujan: `D:\RAFA\outputs\circleworld_proto\training_run_2026-04-23_ramanujan_childworld_writeback_v5_threshold_smoke\childworld_experiment_summary.json`
- v5_ramanujan: `D:\RAFA\outputs\circleworld_proto\training_run_2026-04-23_ramanujan_childworld_writeback_v5_thresholded_real_branch\childworld_experiment_summary.json`
- v5b_ramanujan: `D:\RAFA\outputs\circleworld_proto\training_run_2026-04-23_ramanujan_childworld_writeback_v5b_thresholded_real_branch\childworld_experiment_summary.json`
- v5_qtrace: `D:\RAFA\outputs\circleworld_proto\training_run_2026-04-23_qtrace_only_childworld_writeback_v5_thresholded_real_branch\childworld_experiment_summary.json`

## Bottom line
Sidecar child worlds work as a mechanism, but not yet as a robust ontology. We now know the failure is not merely accounting: real-substrate branch survival needs to be forced by the training constitution, not hoped for through aggregate branch pressure.
