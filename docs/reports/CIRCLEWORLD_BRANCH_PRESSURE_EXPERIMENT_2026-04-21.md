# Circleworld Branch Pressure Experiment (2026-04-21)

## Scope

This report covers the first Circleworld-only branch-pressure pass after the token-diversity run.
The goal was to stop treating branching as a passive metric and instead make the runtime and trainer push on local lattice instability directly.

Nothing in the Graduation lane was touched.

## What Changed

Code changes:
- `D:\RAFA\lineages\04_positive_replacement\circleworld.py`
- `D:\RAFA\runtimes\circleworld_proto\train_circleworld_real_anchor.py`
- `D:\RAFA\runtimes\circleworld_proto\train_circleworld.py`
- `D:\RAFA\runtimes\circleworld_proto\evaluate_circleworld.py`
- `D:\RAFA\runtimes\circleworld_proto\export_circleworld_audio.py`
- `D:\RAFA\runtimes\circleworld_proto\run_branch_pressure_experiment.py`

Runtime additions:
- local defect mask derived from world-gradient, q disagreement, phase walls, residue, and sharpness
- instability seed path injected into branch seed generation
- instability support/logit gains so live multiplicity can be earned from unresolved lattice structure
- new diagnostics: `mean_branch_defect`, `mean_branch_world_grad`, `mean_branch_q_disagreement`, `mean_branch_phase_wall`, `mean_branch_seed_energy`
- real-anchor trainer scoring terms for those instability metrics
- one-click branch-pressure runner with full sidecars enabled

## Runs

### Ramanujan branch-pressure run
- output: `D:\RAFA\outputs\circleworld_proto\training_run_2026-04-21_branch_pressure_ramanujan`
- checkpoint: `D:\RAFA\checkpoints_circleworld_proto\training_run_2026-04-21_branch_pressure_ramanujan\circleworld_real_anchor_config_cem_v1.json`

### Q-trace-only branch-pressure run
- output: `D:\RAFA\outputs\circleworld_proto\training_run_2026-04-21_branch_pressure_qtrace_only`
- checkpoint: `D:\RAFA\checkpoints_circleworld_proto\training_run_2026-04-21_branch_pressure_qtrace_only\circleworld_real_anchor_config_cem_v1.json`

### Comparison bundle
- `D:\RAFA\outputs\circleworld_proto\branch_pressure_compare_2026-04-21.json`

## Main Results

### 1. Branch pressure entered the lattice, but did not become branching

On held-out evaluation:

Ramanujan branch-pressure:
- `mean_branch_defect = 0.1389`
- `mean_branch_seed_energy = 0.01694`
- `mean_slot2_live_fraction = 0.08627`
- `mean_real_branch_fraction = 0.0`
- `mean_meso_branch_effect = 2.05e-06`

Q-trace-only branch-pressure:
- `mean_branch_defect = 0.1522`
- `mean_branch_seed_energy = 0.01664`
- `mean_slot2_live_fraction = 0.14655`
- `mean_real_branch_fraction = 0.0`
- `mean_meso_branch_effect = 2.08e-05`

Read:
- the instability path is active
- slot 2 is more alive than before
- but the system still collapses before that local multiplicity becomes lawful sibling continuation

### 2. Nested commitment still fails completely

Both new runs ended with:
- `overall_read = nested_commitment_not_yet_established`
- `verdict_counts = { over_rigid_attractor: 12 }`

So the current system is still preserving one coarse path too rigidly rather than carrying multiple local futures.

### 3. Ramanujan remains the better real-audio branch-pressure substrate

Benchmark comparison:

Token-diverse Ramanujan:
- `mean_corr = 0.9711`
- `mean_mae = 0.01100`

Branch-pressure Ramanujan:
- `mean_corr = 0.9755`
- `mean_mae = 0.01042`

Branch-pressure q-trace-only:
- `mean_corr = 0.9679`
- `mean_mae = 0.01401`

So when we push on instability, Ramanujan still preserves the anchor world better.

### 4. Q-trace-only yields more live multiplicity, but poorer ontology and poorer audio

Held-out:
- q-trace-only has more `mean_slot2_live_fraction`
- q-trace-only has slightly higher `mean_meso_branch_effect`
- but q-trace-only collapses to `mean_num_law_families = 1.0` and `mean_law_family_entropy = 0.0`

Ramanujan branch-pressure held-out:
- `mean_num_law_families = 1.33`
- `mean_law_family_entropy = 0.306`

Q-trace-only branch-pressure held-out:
- `mean_num_law_families = 1.0`
- `mean_law_family_entropy = 0.0`

Read:
- removing Ramanujan makes the second mode easier to activate
- but the resulting multiplicity is worse structured and less reusable

### 5. Continuity did not move

Token-diverse Ramanujan continuity:
- `mean_loop_autocorr_peak = 0.62660`
- `mean_nonlocal_chunk_repeat = 0.96032`
- `mean_adjacent_chunk_similarity = 0.91047`

Branch-pressure Ramanujan continuity:
- `mean_loop_autocorr_peak = 0.62660`
- `mean_nonlocal_chunk_repeat = 0.96032`
- `mean_adjacent_chunk_similarity = 0.91043`

Branch-pressure q-trace-only continuity:
- `mean_loop_autocorr_peak = 0.62662`
- `mean_nonlocal_chunk_repeat = 0.96043`
- `mean_adjacent_chunk_similarity = 0.90908`

So branch pressure alone is not breaking the macro-time loop tendency.

## Best Current Interpretation

The failure mode is now sharper:

- the system can represent local instability
- it can open slot 2 more often
- it can produce law packets from that pressure
- but the slow world-law is still too globally fused, so branch-local alternatives never survive long enough to become meso-scale siblings

In plain terms: the IFS is feeling the defect, but the world state still resolves it too early.

## Recommendation

Next Circleworld step should not be ?more raw branch pressure.?
It should be branch-local write isolation.

Recommended next implementation:
1. Make slot-2 writes more local in time/support, not just stronger.
2. Add an explicit penalty when branch defect is high but readout stays effectively single-path.
3. Add branch-local summary packets so the second mode can carry its own local law instead of immediately mixing back into the dominant path.
4. Re-run the same Ramanujan vs q-trace-only pair after that change.

## Bottom Line

This experiment was worth doing.
It did not solve branching, but it removed a vague excuse.

We now know the problem is not just ?no branch pressure.?
The problem is that pressure is being generated, then erased before it becomes a stable local continuation law.
