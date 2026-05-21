# Circleworld Childworld Experiment 2026-04-21

## Scope

This report covers the first full implementation of `native_multimode_childworld` in Circleworld only.
Graduation was not touched.

Compared runs:
- `token_diverse_ramanujan`
- `branch_pressure_ramanujan`
- `branch_pressure_qtrace_only`
- `ramanujan_childworld_v1`
- `qtrace_only_childworld_v1`

Machine-readable comparison:
- `D:\RAFA\outputs\circleworld_proto\childworld_compare_2026-04-21.json`

## What Was Implemented

The childworld reset is now real in code:
- explicit sidecar child-world registry inside Circleworld
- `native_multimode_childworld` runtime path
- delayed writeback path with budget/gating
- child-world metrics threaded through training, held-out eval, benchmark, and nested commitment
- full experiment runner for Ramanujan and q-trace control lanes

Primary implementation files:
- `D:\RAFA\lineages\04_positive_replacement\circleworld.py`
- `D:\RAFA\runtimes\circleworld_proto\train_circleworld_real_anchor.py`
- `D:\RAFA\runtimes\circleworld_proto\evaluate_circleworld.py`
- `D:\RAFA\runtimes\circleworld_proto\test_nested_commitment.py`
- `D:\RAFA\runtimes\circleworld_proto\run_childworld_experiment.py`
- `D:\RAFA\runtimes\circleworld_proto\export_circleworld_audio.py`
- `D:\RAFA\runtimes\circleworld_proto\build_law_token_library.py`

## Main Result

The childworld constitution succeeded at one important thing:
- held-out real branching became nonzero for the first time in this lane

But it failed at the deeper bar:
- no nested sibling verdicts
- no child writeback
- no branch carry-through on `naked_rafa`
- law-token ontology stayed collapsed

So this is the first run with real branch-local survival, but not yet real branch-local continuation.

## Headline Numbers

### Ramanujan childworld v1
- held-out `mean_real_branch_fraction = 0.4444`
- held-out `mean_meso_branch_effect = 0.00712`
- held-out `mean_child_world_count = 2.67`
- held-out `mean_child_parent_divergence = 0.14235`
- held-out `mean_child_writeback_mass = 0.0`
- benchmark `mean_corr = 0.9825`
- benchmark `mean_mae = 0.01078`

### Q-trace-only childworld v1
- held-out `mean_real_branch_fraction = 0.4444`
- held-out `mean_meso_branch_effect = 0.00618`
- held-out `mean_child_world_count = 2.67`
- held-out `mean_child_parent_divergence = 0.12354`
- held-out `mean_child_writeback_mass = 0.0`
- benchmark `mean_corr = 0.9706`
- benchmark `mean_mae = 0.01975`

### Best earlier Ramanujan baselines
Token-diverse Ramanujan:
- held-out `mean_real_branch_fraction = 0.0`
- held-out `mean_meso_branch_effect ~= 6.76e-08`
- benchmark `mean_corr = 0.9711`
- benchmark `mean_mae = 0.01100`

Branch-pressure Ramanujan:
- held-out `mean_real_branch_fraction = 0.0`
- held-out `mean_meso_branch_effect ~= 2.05e-06`
- benchmark `mean_corr = 0.9755`
- benchmark `mean_mae = 0.01042`

## What Improved

### 1. Branching is no longer zero
Compared with both earlier Ramanujan baselines, childworld turned branch activity from effectively zero into a measurable held-out effect.

Delta vs token-diverse Ramanujan:
- `real_branch_fraction +0.4444`
- `meso_branch_effect +0.00712`

Delta vs branch-pressure Ramanujan:
- `real_branch_fraction +0.4444`
- `meso_branch_effect +0.00712`

### 2. Ramanujan is the better childworld substrate
Ramanujan childworld and q-trace childworld reached the same coarse branch fraction, but Ramanujan kept better fidelity and stronger branch divergence.

Ramanujan childworld vs q-trace childworld:
- `mean_meso_branch_effect +0.00094`
- `mean_child_parent_divergence +0.01881`
- `mean_corr +0.01193`
- `mean_mae -0.00898`

So the current read is:
- childworld constitution can force survival in either lane
- Ramanujan makes those child worlds more lawful and less damaging

### 3. Synthetic seeds now branch materially
By source split:
- Ramanujan childworld synthetic `mean_real_branch_fraction = 0.6667`
- Q-trace childworld synthetic `mean_real_branch_fraction = 0.6667`

This is the first real sign that Circleworld can sustain branch-local multiplicity instead of only registering defect pressure.

## What Failed

### 1. No writeback
Both childworld runs produced:
- `mean_child_writeback_mass = 0.0`

That means child worlds are being born and surviving briefly, but never becoming causal parents again.
This is the biggest current bottleneck.

### 2. No nested commitment
All runs still ended with:
- `overall_read = nested_commitment_not_yet_established`
- `verdict_counts = { over_rigid_attractor: 12 }`

So the child worlds exist, but they do not yet create sibling continuations that preserve a shared coarse world.

### 3. No naked_rafa branching
By source split:
- Ramanujan childworld naked_rafa `mean_real_branch_fraction = 0.0`
- Q-trace childworld naked_rafa `mean_real_branch_fraction = 0.0`

That means the constitution is currently exploiting synthetic ambiguity much more than real restored-RAFA structure.

### 4. Ontology collapsed again
Both childworld runs landed at:
- held-out `mean_num_law_families = 1.0`
- held-out `mean_law_family_entropy = 0.0`

So branching improved, but relation-token diversity got worse.
Right now the system is finding one dominant child law, not a real family society.

## Diagnosis

The childworld reset solved the wrong half of the problem first.

It solved:
- branch-local survival
- branch-local divergence
- branch-first emergence on synthetic seeds

It did not solve:
- branch-to-parent causality
- branch persistence on real anchors / naked_rafa
- branch-dependent nested continuation
- ontology diversity

The present system is therefore:
- no longer single-path in the weak sense
- still single-destiny in the strong sense

That is why the correct summary is:
- child worlds now exist
- child worlds still do not matter enough

## Recommendation

### Immediate next step
Do **writeback-forcing childworld v2**.

Priority order:
1. Lower the writeback gate and make nonzero writeback a scored target.
2. Penalize `child_world_count > 0` with `child_writeback_mass == 0`.
3. Require at least one child to survive past current age and spend writeback budget.
4. Keep Ramanujan as the default lane.

### Specific constitution changes
- reduce writeback conservatism on parent divergence threshold
- raise reward on `child_writeback_mass`
- add penalty for decorative non-writing children
- allow child writeback to modify parent mode 1 more aggressively before weakly touching mode 0
- add explicit synthetic-vs-naked weighting so search cannot hide inside synthetic-only wins

### What not to do next
Do not claim branching is solved.
Do not move this into Graduation.
Do not drop Ramanujan.
Do not optimize law-token diversity first again.

The next bottleneck is now much clearer than before:
- not spawn
- not survival
- writeback

## Artifacts

Ramanujan childworld v1:
- `D:\RAFA\outputs\circleworld_proto\training_run_2026-04-21_ramanujan_childworld_v1\childworld_experiment_summary.json`
- `D:\RAFA\outputs\circleworld_proto\training_run_2026-04-21_ramanujan_childworld_v1\heldout_eval\heldout_summary.json`
- `D:\RAFA\outputs\circleworld_proto\training_run_2026-04-21_ramanujan_childworld_v1\benchmark_expanded\benchmark_summary.json`
- `D:\RAFA\outputs\circleworld_proto\training_run_2026-04-21_ramanujan_childworld_v1\nested_commitment\nested_commitment_report.json`
- `D:\RAFA\outputs\circleworld_proto\training_run_2026-04-21_ramanujan_childworld_v1\law_token_library\law_token_library.json`

Q-trace-only childworld v1:
- `D:\RAFA\outputs\circleworld_proto\training_run_2026-04-21_qtrace_only_childworld_v1\childworld_experiment_summary.json`
- `D:\RAFA\outputs\circleworld_proto\training_run_2026-04-21_qtrace_only_childworld_v1\heldout_eval\heldout_summary.json`
- `D:\RAFA\outputs\circleworld_proto\training_run_2026-04-21_qtrace_only_childworld_v1\benchmark_expanded\benchmark_summary.json`
- `D:\RAFA\outputs\circleworld_proto\training_run_2026-04-21_qtrace_only_childworld_v1\nested_commitment\nested_commitment_report.json`
- `D:\RAFA\outputs\circleworld_proto\training_run_2026-04-21_qtrace_only_childworld_v1\law_token_library\law_token_library.json`

Retrospective comparison:
- `D:\RAFA\outputs\circleworld_proto\childworld_compare_2026-04-21.json`
