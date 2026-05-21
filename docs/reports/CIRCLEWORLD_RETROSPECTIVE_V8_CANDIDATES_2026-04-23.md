# Circleworld Retrospective V8 Candidate Recovery

Date: 2026-04-23

## What We Did

We added a retrospective selector and evaluator for Circleworld childworld search histories:

- runtime: `D:\RAFA\runtimes\circleworld_proto\retrospective_childworld_eval.py`
- input history: `D:\RAFA\outputs\circleworld_proto\training_run_2026-04-23_ramanujan_childworld_writeback_v8_recovery_with_law_floor\search_history.json`
- output root: `D:\RAFA\outputs\circleworld_proto\retrospective_v8_candidates_2026-04-23`

This recovered alternate candidates from the `v8` search rather than trusting the original scalar winner. Each selected candidate was materialized into its own config and run through the full sidecar stack:

- heldout evaluation
- 10-second real-anchor benchmark
- continuity benchmark
- live-child nested commitment
- law-token library build

## Candidates Evaluated

1. `branchfloor_audio_idx04_iter0`
2. `branchfloor_law_idx44_iter5`
3. `lawfirst_relaxed_idx16_iter2`

Master report:

- `D:\RAFA\outputs\circleworld_proto\retrospective_v8_candidates_2026-04-23\retrospective_report.json`
- `D:\RAFA\outputs\circleworld_proto\retrospective_v8_candidates_2026-04-23\retrospective_compare_2026-04-23.json`

## Main Result

Retrospective selection did recover candidates with materially better law diversity and, in one case, materially better audio correlation than the selected `v7`/`v8` basin.

But it also exposed a more important problem:

**the training-time transfer probe is not aligned with the heldout evaluator on `naked_rafa`.**

The recovered candidates looked valid in the search history because the probe reported nonzero naked branching. But on heldout evaluation, all three retrospective candidates fell to:

- `heldout naked_rafa mean_real_branch_fraction = 0.0`

So the issue is no longer just "pick a better candidate."
The issue is:

**the probe used during search can reward branch-looking candidates that do not survive the real heldout evaluator.**

## Best Candidate By Role

### Best audio retrospective candidate

`branchfloor_audio_idx04_iter0`

- benchmark corr: `0.8841`
- benchmark mae: `0.0316`
- heldout branch: `0.4444`
- heldout law families: `2.0`
- heldout naked branch: `0.0`
- law-token aggregate families: `10`

This is the strongest retrospective audio result.
It also keeps decent law diversity.
But it does **not** preserve naked heldout branching.

### Best law-diverse retrospective candidate

`branchfloor_law_idx44_iter5`

- benchmark corr: `0.8647`
- benchmark mae: `0.0327`
- heldout branch: `0.4444`
- heldout law families: `2.2222`
- heldout law entropy: `0.4612`
- heldout naked branch: `0.0`
- law-token aggregate families: `14`

This is the best retrospective law-family result.
Again, the failure is the same: no naked heldout branch survival.

### Best existing branch-first checkpoint that still holds the real-substrate lane

`v6_ramanujan_curriculum`

- checkpoint: `D:\RAFA\checkpoints_circleworld_proto\training_run_2026-04-23_ramanujan_childworld_writeback_v6_naked_survival_curriculum\circleworld_real_anchor_config_cem_v1.json`
- benchmark corr: `0.8599`
- benchmark mae: `0.0336`
- heldout branch: `0.5556`
- heldout law families: `2.0`

This remains the best real branch-first checkpoint because it is still the cleanest proof that the Ramanujan childworld lane can keep real-substrate child survival alive in the main heldout run.

## Nested Commitment

No retrospective candidate fixed nested commitment.
All three remain:

- `overall_read = nested_commitment_not_yet_established`
- `verdict_counts = {"over_rigid_attractor": 12}`

Child-directed perturbations are wired and running, but they still do not open sibling lawful continuation.

## Interpretation

The retrospective results split the problem into two layers:

### 1. Candidate selection was partly the problem

True.
The original `v8` scalar chooser did leave better law-diverse candidates on the table.

### 2. Probe/heldout mismatch is now the bigger problem

Also true, and more important.

The retrospective candidates demonstrate that:

- search-time probe success does not guarantee heldout naked branching
- some candidates can improve benchmark audio and law diversity while still failing the real-substrate heldout branch test
- more blind search or more retrospective picking will not solve this by itself

## Recommendations

### Immediate next step

Align the search-time transfer probe with the heldout evaluator.

Concretely:

1. make the probe use the same `naked_rafa` seed family or partially overlap with heldout-style seeds
2. increase the number of naked probe seeds during search
3. add a hard penalty for `probe naked branch > 0` but `probe naked live child / parent-div / writeback` failing to cohere
4. prefer a two-stage acceptance gate over a single scalar rank:
   - stage A: must satisfy naked branch survival on a stricter probe
   - stage B: among survivors, optimize audio/law tradeoff

### Do not do next

Do not launch another long `v9` search yet.
The selection layer is now good enough to show that the main bottleneck is evaluator alignment.

### Keep these checkpoints

- Keep `v6_ramanujan_curriculum` as the branch-first constitutional baseline.
- Keep `branchfloor_audio_idx04_iter0` as the best retrospective audio candidate.
- Keep `branchfloor_law_idx44_iter5` as the best retrospective law-diversity candidate.

They answer different questions and should not be collapsed into one "best" model.
