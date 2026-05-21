# Circleworld v19 Seeded Nested Substrate 2026-04-28

## Goal

Replace the weak real-anchor nested probe substrate with a seeded branch-active substrate:

- probe from seeded `naked_rafa` states
- select fork points from states where live child worlds already exist when possible
- rerun the same in-loop nested objective on that substrate

## Implementation

Updated files:

- `D:\RAFA\runtimes\circleworld_proto\test_nested_commitment.py`
- `D:\RAFA\runtimes\circleworld_proto\train_circleworld_real_anchor.py`
- `D:\RAFA\runtimes\circleworld_proto\run_childworld_experiment.py`

Main changes:

1. `test_nested_commitment.py`
   - supports seeded nested cases in addition to reference-audio cases
   - generates seeded phase states via `naked_rafa`
   - generates deterministic synthetic magnitude for seeded renders
   - warmup trace can search for branch-active states before fork/resume

2. `train_circleworld_real_anchor.py`
   - nested case loading now supports:
     - `selection_nested_case_source = "reference_audio"`
     - `selection_nested_case_source = "seeded_branch_active"`
     - inline `selection_nested_case_plan`
   - both the in-loop nested probe and late-stage nested gate use the same resolved case set

3. `run_childworld_experiment.py`
   - added profile `writeback_v19_seeded_nested_substrate`
   - keeps the v18 nested objective weights
   - switches only the nested probe substrate to seeded branch-active cases

## Artifacts

Seeded nested refresh on current active checkpoint:

- `D:\RAFA\outputs\circleworld_proto\nested_seeded_refresh_agreement_scout_v1_2026-04-28\nested_commitment_report.json`

New scout run:

- `D:\RAFA\outputs\circleworld_proto\training_run_2026-04-28_seeded_nested_probe_v19\train_summary.json`
- `D:\RAFA\checkpoints_circleworld_proto\training_run_2026-04-28_seeded_nested_probe_v19\circleworld_real_anchor_config_cem_v1.json`

Seeded nested refresh on v19:

- `D:\RAFA\outputs\circleworld_proto\nested_seeded_refresh_v19_2026-04-28\nested_commitment_report.json`

Direct seeded comparison:

- `D:\RAFA\outputs\circleworld_proto\nested_seeded_compare_v19_2026-04-28.json`

## Result

### Constitutionally improved

The seeded substrate now does the intended thing:

- fork points are selected from branch-active seeded states
- `live_child_start_found = true` on the seeded naked cases
- the nested probe is no longer the old flat `-170` anchor failure mode

### But still not enough

Observed on seeded naked cases:

- `mean_child_response_score` is positive (`~0.8776`)
- `mean_child_active_fraction = 0.0`
- `mean_child_meso_response = 0.0`
- verdicts remain `mixed_or_inconclusive`, not `nested_sibling`

Observed in the v19 trainer run:

- `selection_source = init_config_fallback_nested_or_agreement_required`
- no candidate cleared both heldout and nested gates

### v19 vs seeded baseline

The seeded standalone assay on v19 is essentially flat relative to the seeded baseline:

- response delta: `+0.0000014`
- active fraction delta: `0.0`
- meso response delta: `0.0`

So the substrate is better, but the model still is not using child-history perturbation as a real continuation degree of freedom.

## Diagnosis

This narrows the failure further:

- previously: nested probe substrate was too weak and anchor-bound
- now: substrate is branch-active and seeded, but the actual branch perturbations still do not activate live child continuation response

That means the next bottleneck is no longer case selection.

It is the perturbation constitution itself.

## Recommendation

Next step should target perturbation mechanics directly:

1. perturb childworld records more aggressively than `child_phase_shift` / `child_qtrace_shift`
2. add explicit writeback-budget and support-window perturbations
3. score response only on cases with `live_child_start_found = true`
4. keep the seeded branch-active substrate as the default nested-probe lane

Graduation was not touched.
