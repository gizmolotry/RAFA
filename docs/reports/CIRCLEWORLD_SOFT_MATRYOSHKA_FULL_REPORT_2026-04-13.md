# Circleworld Soft Matryoshka Full Report

Date: 2026-04-13

Scope: Circleworld lane only. Graduation / restored Graduation runtime was not modified.

## Objective

Apply the "soft Matryoshka IFS" idea to Circleworld only:

- keep one recursive state
- avoid a hard world/development/detail tensor split
- add learned persistence structure inside a single state
- make smaller projections of the state matter
- bottleneck writes from local packet interventions

Then run a full real-anchor training cycle and all available sidecars on the result.

## What Changed

Core implementation was added in [circleworld.py](D:\RAFA\lineages\04_positive_replacement\circleworld.py).

The change was not a hand-authored ontology split. Instead, Circleworld now supports:

- `soft_matryoshka_enabled`
- learned persistence spectrum from slow to fast channels
- low-rank packet write path
- prefix-usefulness pressure
- prefix-alignment and prefix-delta reporting

Mechanically, that means:

- active packet deltas can be compressed into a low-rank basis before they write back
- slower directions get smaller write privilege and longer persistence
- faster directions can move more, but do not get unrestricted overwrite power
- the loss now includes a prefix term so coarse identity cannot hide only in the full state

Supporting runtime updates were made in:

- [train_circleworld.py](D:\RAFA\runtimes\circleworld_proto\train_circleworld.py)
- [train_circleworld_real_anchor.py](D:\RAFA\runtimes\circleworld_proto\train_circleworld_real_anchor.py)
- [export_circleworld_audio.py](D:\RAFA\runtimes\circleworld_proto\export_circleworld_audio.py)

## Training Run

Initialization config:

- [circleworld_real_anchor_config_cem_v1.json](D:\RAFA\checkpoints_circleworld_proto\training_run_2026-04-13_soft_matryoshka_init\circleworld_real_anchor_config_cem_v1.json)

Full trained config:

- [circleworld_real_anchor_config_cem_v1.json](D:\RAFA\checkpoints_circleworld_proto\training_run_2026-04-13_soft_matryoshka_full\circleworld_real_anchor_config_cem_v1.json)

Training summary:

- [train_summary.json](D:\RAFA\outputs\circleworld_proto\training_run_2026-04-13_soft_matryoshka_full\train_summary.json)

Run shape:

- real-anchor CEM trainer
- 64 iterations
- population 8
- elite count 3
- 10 second clips
- CUDA
- extra weighting on clarinet and metal-clang cases
- prefix-alignment and prefix-delta included in scoring

Best validation result from the full run:

- baseline validation score: `-0.9934`
- best validation score: `-0.0800`
- baseline validation correlation: `0.2710`
- best validation correlation: `0.8561`
- baseline validation MAE: `0.0710`
- best validation MAE: `0.0354`

Best config settled into:

- recursion depth `3`
- soft Matryoshka enabled
- matryoshka rank `37`
- slow persistence `0.9552`
- fast persistence `0.3141`
- slow write scale `0.0370`
- fast write scale `0.4948`
- prefix coarse weight `0.3116`
- prefix mid weight `0.3615`

## Sidecar Results

### 1. 5-case real-anchor benchmark

Artifacts:

- [benchmark_summary.json](D:\RAFA\outputs\circleworld_proto\benchmark_2026-04-13_soft_matryoshka_5case\benchmark_summary.json)

Result:

- mean correlation: `0.9683`
- mean MAE: `0.0260`
- mean MSE: `0.00288`
- bitwise stable rerenders: `true`

This is a strong result for anchor-conditioned fidelity.

### 2. Expanded 12-case benchmark

Artifacts:

- [benchmark_summary.json](D:\RAFA\outputs\circleworld_proto\benchmark_2026-04-13_soft_matryoshka_expanded\benchmark_summary.json)

Old constrained baseline for comparison:

- [benchmark_summary.json](D:\RAFA\outputs\circleworld_proto\benchmark_2026-04-09_realanchor_constrained_expanded\benchmark_summary.json)

Old vs new:

- old mean correlation: `0.5469`
- new mean correlation: `0.9249`
- old mean MAE: `0.0822`
- new mean MAE: `0.0273`
- old mean MSE: `0.0240`
- new mean MSE: `0.00371`

Hard cases improved substantially:

- `metal_clang`: correlation `-0.7620 -> 0.7590`
- `voice_alt`: correlation `-0.0389 -> 0.5657`
- `sax_like_clarinet`: correlation `0.2177 -> 0.9041`

Interpretation:

This is the clearest win of the whole run. The new Circleworld configuration is much better at preserving useful real-anchor identity across a broad benchmark set.

### 3. Continuity / loop sidecar

New artifacts:

- [continuity_circleworld.json](D:\RAFA\outputs\circleworld_proto\benchmark_2026-04-13_soft_matryoshka_expanded\continuity_circleworld.json)
- [continuity_reference.json](D:\RAFA\outputs\circleworld_proto\benchmark_2026-04-13_soft_matryoshka_expanded\continuity_reference.json)

Old constrained comparison:

- [continuity_circleworld.json](D:\RAFA\outputs\circleworld_proto\benchmark_2026-04-09_realanchor_constrained_expanded\continuity_circleworld.json)

Key means:

- old loop autocorr peak: `0.62645`
- new loop autocorr peak: `0.62662`
- reference loop autocorr peak: `0.62659`

- old adjacent chunk similarity: `0.90492`
- new adjacent chunk similarity: `0.91357`
- reference adjacent chunk similarity: `0.90551`

- old nonlocal chunk repeat: `0.96107`
- new nonlocal chunk repeat: `0.95919`
- reference nonlocal chunk repeat: `0.96145`

- old first chunk reentry: `0.90365`
- new first chunk reentry: `0.91232`
- reference first chunk reentry: `0.90418`

Interpretation:

This is not evidence that Circleworld learned a stronger forward-moving time-world.

It is evidence that the new model preserved the reference continuity profile very closely. That is useful, but it is still mostly preservation, not genuinely new macro-time governance.

### 4. Held-out structural evaluator

Artifacts:

- [heldout_summary.json](D:\RAFA\outputs\circleworld_proto\heldout_eval_2026-04-13_soft_matryoshka\heldout_summary.json)
- [heldout_trajectory.csv](D:\RAFA\outputs\circleworld_proto\heldout_eval_2026-04-13_soft_matryoshka\heldout_trajectory.csv)

Held-out summary:

- mean major gain: `0.0720`
- mean residue drop: `0.1039`
- mean promotability gain: `0.0780`
- mean dominant q share: `0.7297`
- mean q entropy: `0.3767`

Interpretation:

- synthetic seeds improve clearly
- naked_rafa seeds still collapse into a bad regime
- the evaluator still shows a substrate split: Circleworld is stronger on synthetic structure than on naked RAFA internal seeds

This means the method is not yet a clean general recursive runtime. It remains a mixed prototype.

### 5. Nested commitment / fork-resume sidecar

Artifacts:

- [nested_commitment_report.json](D:\RAFA\outputs\circleworld_proto\nested_commitment_2026-04-13_soft_matryoshka\nested_commitment_report.json)

Old constrained comparison:

- [nested_commitment_report.json](D:\RAFA\outputs\circleworld_proto\nested_commitment_2026-04-11_constrained\nested_commitment_report.json)

Result:

- overall read: `nested_commitment_not_yet_established`
- verdict pattern: `over_rigid_attractor`

Observed behavior:

- fine noise perturbations cause only tiny texture changes
- packet perturbations are effectively no-ops
- promotion perturbations barely move the later continuation
- coarse and meso trajectories remain almost identical after branching

Interpretation:

This is the main thing the new architecture did not solve.

The run improved real-anchor fidelity a lot, but it still did not produce true Matryoshka-like nested commitment where:

- coarse identity stays fixed
- meso development remains related but branch-sensitive
- fine texture varies independently

Right now the system still looks more frozen than hierarchically alive.

## What We Learned

### Circleworld did improve in a real way

The soft Matryoshka version is not a fake win.

It materially improved:

- real-anchor preservation quality
- stability on difficult cases
- benchmark breadth
- deterministic rerender consistency

That makes it a better research baseline than the prior constrained configuration.

### Circleworld did not yet become a recursive world-builder

The same run did not establish:

- nested commitment
- branch-sensitive continuation
- strong packet-level causal writing
- robust behavior on naked_rafa internal seeds

So the right reading is:

- good progress on representation and preservation
- weak progress on actual recursive world-law

### The continuity problem is still basically Step 2, not Step 4

The continuity sidecar reinforces the same diagnosis:

- we are still closer to preserving an anchor's existing world than building a new evolving one
- that is why the model sounds better without yet becoming truly developmental

This is still the "macro-time continuity" stage, not the "auto-formalized meaning" stage.

## Recommendation

### For Circleworld

Keep this soft Matryoshka configuration as the new Circleworld baseline.

Reason:

- it is clearly better than the prior constrained baseline on real-anchor testing
- it preserves the one-state ontology instead of hardcoding a semantic trisection
- it gives us a better platform for the next failure-focused iteration

But do not present it as solved recursive object formation yet.

The next Circleworld work should focus on:

1. branch-sensitive continuation pressure
2. stronger packet write efficacy without collapsing the whole state
3. making naked_rafa held-out rows stop degenerating
4. debugging why prefix metrics report strongly in real-anchor training and export but collapse to zero in held-out and nested sidecars

### For Graduation

Leave Graduation separate.

Graduation should remain:

- the certified sound engine
- the anchor for stable generation
- the benchmark reference for continuity and non-collapse

Circleworld should remain:

- the research lane for promotion, persistence geometry, packet law, and recursive world-building

Do not merge Circleworld into Graduation based on this run alone.

## Final Verdict

Soft Matryoshka Circleworld is a successful architectural upgrade for the Circleworld lane.

It is the best Circleworld real-anchor baseline so far.

But it is not yet the system you wanted philosophically.

The honest one-line summary is:

> We improved Circleworld from "fragile intervention layer" to "strong anchor-preserving recursive modifier," but we have not yet made it into a branch-sensitive recursive world-law engine.
