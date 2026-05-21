# Circleworld Native Multimode V1 Report (2026-04-16)

## Scope

This report covers the first full `native_multimode` Circleworld run using the new local two-mode branching constitution on the `04_positive_replacement` lane only.

The goal of this run was not just higher Circleworld score. It was to test whether a native two-mode state could:

- keep deterministic real-anchor rendering intact
- produce measurable live multiplicity without external rerun heuristics
- improve meso divergence without destroying coarse identity
- survive the held-out and nested-commitment sidecars

## Artifacts

- Training summary: `D:\RAFA\outputs\circleworld_proto\training_run_2026-04-16_native_multimode_v1_full\train_summary.json`
- Search history: `D:\RAFA\outputs\circleworld_proto\training_run_2026-04-16_native_multimode_v1_full\search_history.json`
- Trained config: `D:\RAFA\checkpoints_circleworld_proto\training_run_2026-04-16_native_multimode_v1_full\circleworld_real_anchor_config_cem_v1.json`
- Expanded benchmark: `D:\RAFA\outputs\circleworld_proto\benchmark_2026-04-16_native_multimode_v1_full_expanded\benchmark_summary.json`
- Continuity sidecar (Circleworld outputs): `D:\RAFA\outputs\circleworld_proto\benchmark_2026-04-16_native_multimode_v1_full_expanded\continuity_circleworld.json`
- Continuity sidecar (references): `D:\RAFA\outputs\circleworld_proto\benchmark_2026-04-16_native_multimode_v1_full_expanded\continuity_reference.json`
- Held-out eval: `D:\RAFA\outputs\circleworld_proto\heldout_eval_2026-04-16_native_multimode_v1_full\heldout_summary.json`
- Nested commitment: `D:\RAFA\outputs\circleworld_proto\nested_commitment_2026-04-16_native_multimode_v1_full\nested_commitment_report.json`

Comparison baselines used in this report:

- Soft Matryoshka benchmark: `D:\RAFA\outputs\circleworld_proto\benchmark_2026-04-13_soft_matryoshka_expanded\benchmark_summary.json`
- Soft Matryoshka held-out eval: `D:\RAFA\outputs\circleworld_proto\heldout_eval_2026-04-13_soft_matryoshka\heldout_summary.json`
- Soft Matryoshka nested commitment: `D:\RAFA\outputs\circleworld_proto\nested_commitment_2026-04-13_soft_matryoshka\nested_commitment_report.json`

## What Was Implemented

The runtime used the new native multimode path in `D:\RAFA\lineages\04_positive_replacement\circleworld.py` with these properties:

- `branching_mode = native_multimode`
- `num_modes = 2`
- `readout_mode = weighted_mixture`
- dormant second-slot default
- branch-local support dynamics
- split / merge / survival / collapse law terms
- branch-aware summary metrics
- multimode perturbations in nested commitment

The benchmark and sidecar runners were also updated so the experiment could be exercised through the existing Circleworld filing pattern rather than a one-off path.

## Training Result

Real-anchor training completed successfully.

Key training deltas on validation:

- baseline val score: `-0.2598`
- best val score: `7.7414`
- baseline val meso branch effect: `0.00000547`
- best val meso branch effect: `0.00034607`
- baseline silent single-path fraction: `0.7433`
- best silent single-path fraction: `0.7211`
- baseline real branch fraction: `0.0`
- best real branch fraction: `0.0`

Interpretation:

- The search did find a way to increase internal mode divergence.
- The search did reduce silent single-path occupancy slightly.
- The search did not produce any true live branch survival under the current `real_branch_fraction` criterion.

The resulting config was aggressive in support spread and persistence:

- `support_spread = 15`
- `split_pressure = 0.2170`
- `split_support_gain = 0.5764`
- `merge_pressure = 0.1850`
- `slow_persistence = 0.9636`
- `fast_persistence = 0.3007`

## Benchmark Result

Expanded 12-case real-anchor benchmark:

- mean MSE: `0.00901`
- mean MAE: `0.04554`
- mean correlation: `0.86058`
- deterministic rerender: `true` across all cases

Compared with the 2026-04-13 soft-matryoshka expanded benchmark:

- prior mean MSE: `0.00371`
- prior mean MAE: `0.02729`
- prior mean correlation: `0.92488`

Interpretation:

- Native multimode stayed bitwise stable, which is good.
- Audio similarity regressed materially versus the soft-matryoshka baseline.
- So this v1 branch constitution is not yet a better audio runtime, even though it is a more explicit branching experiment.

Notable cases:

- `voice_alt` remained the hardest failure case with correlation near `0.1881`.
- `sax_like_clarinet` also degraded strongly, with major-mass and promotability losses and correlation near `0.7462`.
- Mechanical anchors like `airplane_takeoff`, `airplane_landing`, and `anvil_impact` remained comparatively intact.

Listen folder:

- `D:\RAFA\outputs\circleworld_proto\benchmark_2026-04-16_native_multimode_v1_full_expanded`

## Continuity Sidecars

Circleworld output continuity:

- mean loop autocorr peak: `0.62628`
- mean loop period: `1.4560 s`
- mean adjacent chunk similarity: `0.91555`
- mean nonlocal chunk repeat: `0.95831`
- mean first chunk reentry: `0.91443`

Reference continuity:

- mean loop autocorr peak: `0.62659`
- mean loop period: `1.4587 s`
- mean adjacent chunk similarity: `0.90551`
- mean nonlocal chunk repeat: `0.96145`
- mean first chunk reentry: `0.90418`

Interpretation:

- The continuity sidecar does not show a clear macro-time win.
- It also does not show catastrophic new looping relative to the reference set.
- The multimode system is therefore mostly preserving the anchor’s continuity profile rather than creating a new, better time-world.

## Held-Out Evaluator

Held-out native-multimode summary:

- mean major gain: `-0.000528`
- mean residue drop: `-0.000583`
- mean promotability gain: `-0.000476`
- mean slot-2 live fraction: `0.05175`
- mean real branch fraction: `0.0`
- mean meso branch effect: `0.0000192`
- mean silent single-path fraction: `0.94825`

Compared with soft-matryoshka held-out:

- prior mean major gain: `0.07196`
- prior mean residue drop: `0.10387`
- prior mean promotability gain: `0.07797`

Interpretation:

- Held-out behavior is still effectively single-path.
- Slot 2 exists, but it is mostly dormant or decorative.
- The new branch law improved train-time objective fit but did not transfer as meaningful structural gain on held-out seeds.
- Against the prior soft-matryoshka config, held-out structure is worse.

## Nested Commitment Test

Nested commitment result:

- overall read: `nested_commitment_not_yet_established`
- verdict counts: `12 over_rigid_attractor`

That means every expanded case failed the core fork/resume criterion in the same direction:

- not world-jumping
- not sibling branching
- over-rigid continuation under perturbation

Important point:

- This is a cleaner failure than before because the system now actually carries a multimode state internally.
- The result says the current branch constitution is still not earning persistent lawful multiplicity.
- Tiny branch-local perturbations mostly preserve the same path instead of generating sibling meso continuations.

## Final Read

This run is a valid and useful experiment, but it is not a success in the strong sense.

What succeeded:

- native multimode runtime implemented
- branch-aware training path implemented
- benchmark / held-out / continuity / nested sidecars all run cleanly
- deterministic render stability preserved
- internal meso-divergence terms increased during training

What failed:

- `real_branch_fraction` stayed at `0.0`
- held-out structure regressed versus soft-matryoshka
- benchmark audio similarity regressed versus soft-matryoshka
- nested commitment remained fully over-rigid

Most honest diagnosis:

- we implemented branch capacity
- we did not yet implement branch survival

The second mode is present as internal pressure, not as a live sibling continuation law.

## Recommendations

### Graduation / stable audio lane

Do not replace the stronger Graduation or softer Circleworld audio-facing baselines with this native multimode config.

Use this run as a research artifact, not a certified runtime.

### Circleworld research lane

Next changes should target branch survival specifically, not just branch occupancy.

Recommended order:

1. Tighten the acceptance metric for branch reality.
   Require nonzero support persistence plus measurable fork/resume meso divergence before a second mode counts as real.

2. Penalize decorative slot-2 occupancy.
   The current trainer can still win by creating tiny internal mode separation without world-level consequences.

3. Add support handoff and local-time branching pressure.
   Right now `mean_support_handoff_count` stayed at `0.0` across the run. That is a strong sign the second mode is not becoming a real temporal alternative.

4. Reduce reliance on long support smear.
   `support_spread = 15` likely helped produce gentle internal separation, but it also looks consistent with diffuse, non-committal mode support.

5. Gate split pressure on unresolved incompatibility rather than generic ambiguity.
   The next branch law should branch only when rational concentration or compatibility conflict persists, not merely when local evidence is broad.

6. Re-run the same sidecar stack after every branch-law revision.
   The current artifact set is now the right template for future branching tests.

## Recommendation For Naming / Filing

This run should be treated as the canonical first native branching packet:

- training: `training_run_2026-04-16_native_multimode_v1_full`
- benchmark: `benchmark_2026-04-16_native_multimode_v1_full_expanded`
- held-out: `heldout_eval_2026-04-16_native_multimode_v1_full`
- nested commitment: `nested_commitment_2026-04-16_native_multimode_v1_full`

That naming is consistent with the existing Circleworld experiment history and makes the sidecars discoverable by date and constitution.
