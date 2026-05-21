# Circleworld Geometric Masking Pass (2026-04-17)

## Scope

This report covers the Circleworld-only geometric masking pass performed on 2026-04-17.

Nothing in the Graduation restore/runtime lane was changed on purpose. The work stayed inside the `04_positive_replacement` Circleworld experiment family and its sidecars.

The goal was simple:

- add stronger masking so branch capacity is guided by geometry rather than generic ambiguity
- borrow masking ideas from recent geometric-reasoning literature rather than inventing another ad hoc threshold stack
- rerun training, held-out evaluation, benchmark exports, continuity sidecars, and nested commitment so the result is filed consistently

## Academic Inspirations

The masking changes were informed by recent geometry-heavy masking papers from the last year. These were used as design inspirations, not copied literally.

1. Topological masking
   Paper: *Linear Transformer Topological Masking* ([OpenReview](https://openreview.net/forum?id=6MBqQLp17E))

   Relevant idea: attention or interaction should be biased by structural neighborhood, not just raw token order or dense all-to-all mixing. In Circleworld terms, that maps cleanly to a topology-sensitive mask over local phase/evidence neighborhoods.

2. Geometry-adaptive plus instruction/context-aware masking
   Paper: *Masking Matters: Unlocking the Spatial Reasoning Capabilities of LLMs for 3D Scene-Language Understanding* ([arXiv:2512.02487](https://arxiv.org/abs/2512.02487))

   Relevant idea: replace generic causal or position-biased masking with a geometry-adaptive mask, and allow the right global context to cut through local bias. In Circleworld terms, that suggested separating local topology masking from a world-context mask.

3. Geometry-unleashing masking
   Paper: *Make Geometry Matter for Spatial Reasoning* ([arXiv:2603.26639](https://arxiv.org/abs/2603.26639))

   Relevant idea: suppress shortcut channels so the model is forced to rely on geometric evidence where geometry should matter. In Circleworld terms, that motivated explicit negative masking and auxiliary suppression against decorative branch occupancy.

4. Target-focused structured masking under occlusion
   Paper: *Occlusion-Aware 3D Hand-Object Pose Estimation with Masked AutoEncoders* ([arXiv:2506.10816](https://arxiv.org/abs/2506.10816))

   Relevant idea: structured masking can force the system to reason about the right missing or conflicted geometry instead of coasting on easy local cues. In Circleworld terms, that argued for masks that amplify unresolved local structure rather than merely rewarding more mode occupancy.

## What Was Implemented

### Core runtime changes

Primary implementation file:

- `D:\RAFA\lineages\04_positive_replacement\circleworld.py`

New Circleworld configuration fields were added:

- `mask_neighborhood`
- `topology_mask_gain`
- `complexity_mask_gain`
- `context_mask_gain`
- `contrastive_mask_gain`
- `aux_mask_suppression`

These now live in the config surface at the top of the file and are used directly inside the native multimode update path.

Relevant locations:

- `D:\RAFA\lineages\04_positive_replacement\circleworld.py:67`
- `D:\RAFA\lineages\04_positive_replacement\circleworld.py:68`
- `D:\RAFA\lineages\04_positive_replacement\circleworld.py:69`
- `D:\RAFA\lineages\04_positive_replacement\circleworld.py:70`
- `D:\RAFA\lineages\04_positive_replacement\circleworld.py:71`
- `D:\RAFA\lineages\04_positive_replacement\circleworld.py:72`

Inside the multimode step, the runtime now builds four masking components:

- topology mask from smoothed local structural neighborhood
- complexity mask from local evidence density / unresolved sharpness
- context mask from world-level arc/promotability context
- contrastive positive-vs-negative branch masks to separate lawful coexistence from decorative auxiliary activity

Relevant runtime locations:

- `D:\RAFA\lineages\04_positive_replacement\circleworld.py:593`
- `D:\RAFA\lineages\04_positive_replacement\circleworld.py:595`
- `D:\RAFA\lineages\04_positive_replacement\circleworld.py:603`
- `D:\RAFA\lineages\04_positive_replacement\circleworld.py:609`
- `D:\RAFA\lineages\04_positive_replacement\circleworld.py:612`
- `D:\RAFA\lineages\04_positive_replacement\circleworld.py:613`
- `D:\RAFA\lineages\04_positive_replacement\circleworld.py:634`
- `D:\RAFA\lineages\04_positive_replacement\circleworld.py:658`

The runtime also now records branch-hygiene metrics per step and in the final run summary:

- `branch_positive_mask`
- `branch_negative_mask`
- `decorative_slot2_fraction`

Relevant summary locations:

- `D:\RAFA\lineages\04_positive_replacement\circleworld.py:702`
- `D:\RAFA\lineages\04_positive_replacement\circleworld.py:703`
- `D:\RAFA\lineages\04_positive_replacement\circleworld.py:704`
- `D:\RAFA\lineages\04_positive_replacement\circleworld.py:958`
- `D:\RAFA\lineages\04_positive_replacement\circleworld.py:959`
- `D:\RAFA\lineages\04_positive_replacement\circleworld.py:960`
- `D:\RAFA\lineages\04_positive_replacement\circleworld.py:1035`
- `D:\RAFA\lineages\04_positive_replacement\circleworld.py:1036`

### Training / eval plumbing changes

The new masking parameters and branch-hygiene metrics were carried through all Circleworld tooling so the run could be trained, evaluated, and exported without one-off code.

Changed files:

- `D:\RAFA\runtimes\circleworld_proto\train_circleworld.py`
- `D:\RAFA\runtimes\circleworld_proto\train_circleworld_real_anchor.py`
- `D:\RAFA\runtimes\circleworld_proto\evaluate_circleworld.py`
- `D:\RAFA\runtimes\circleworld_proto\export_circleworld_audio.py`
- `D:\RAFA\runtimes\circleworld_proto\test_nested_commitment.py`

Important trainer locations:

- `D:\RAFA\runtimes\circleworld_proto\train_circleworld_real_anchor.py:175`
- `D:\RAFA\runtimes\circleworld_proto\train_circleworld_real_anchor.py:176`
- `D:\RAFA\runtimes\circleworld_proto\train_circleworld_real_anchor.py:177`
- `D:\RAFA\runtimes\circleworld_proto\train_circleworld_real_anchor.py:297`
- `D:\RAFA\runtimes\circleworld_proto\train_circleworld_real_anchor.py:410`
- `D:\RAFA\runtimes\circleworld_proto\train_circleworld_real_anchor.py:613`

Important runtime-loader locations:

- `D:\RAFA\runtimes\circleworld_proto\train_circleworld.py:96`
- `D:\RAFA\runtimes\circleworld_proto\train_circleworld.py:157`
- `D:\RAFA\runtimes\circleworld_proto\train_circleworld.py:222`
- `D:\RAFA\runtimes\circleworld_proto\evaluate_circleworld.py:82`
- `D:\RAFA\runtimes\circleworld_proto\export_circleworld_audio.py:85`

## Smoke Result Before Retraining

A smoke eval was run first using the older native-multimode config under the new masking rules.

Artifacts:

- `D:\RAFA\outputs\circleworld_proto\smoke_eval_2026-04-17_masked_native_base`

Result:

- masking worked mechanically
- slot 2 was almost fully suppressed
- no live branching appeared

Key means from that smoke pass:

- `mean_slot2_live_fraction = 0.01046`
- `mean_real_branch_fraction = 0.0`
- `mean_meso_branch_effect = 2.24e-08`
- `mean_branch_positive_mask = 0.000822`
- `mean_branch_negative_mask = 0.67244`
- `mean_decorative_slot2_fraction = 0.00825`

Interpretation:

The new mask stack was doing what it was told, but the old branch law could only respond by collapsing harder into single-path behavior.

## Full Masked Training Run

Training artifacts:

- training summary: `D:\RAFA\outputs\circleworld_proto\training_run_2026-04-17_native_multimode_masked_v1\train_summary.json`
- search history: `D:\RAFA\outputs\circleworld_proto\training_run_2026-04-17_native_multimode_masked_v1\search_history.json`
- trained config: `D:\RAFA\checkpoints_circleworld_proto\training_run_2026-04-17_native_multimode_masked_v1\circleworld_real_anchor_config_cem_v1.json`

Run constitution:

- trainer: real-anchor CEM
- seed: `20260417`
- iterations: `24`
- population: `8`
- elite count: `3`
- clip length: `10 s`
- init config: `D:\RAFA\checkpoints_circleworld_proto\training_run_2026-04-16_native_multimode_v1_full\circleworld_real_anchor_config_cem_v1.json`

Extra score terms used for this masking pass:

- reward `branch_positive_mask`
- penalize `branch_negative_mask`
- penalize `decorative_slot2_fraction`
- still require audio similarity and baseline-regret constraints

Best learned mask-heavy configuration:

- `mask_neighborhood = 7`
- `topology_mask_gain = 0.5907`
- `complexity_mask_gain = 0.6133`
- `context_mask_gain = 0.4124`
- `contrastive_mask_gain = 0.8702`
- `aux_mask_suppression = 0.3420`

### Training read

The masked search did **not** produce true branching.

What it did produce was a much cleaner branch-hygiene regime.

Key validation means from the trained run summary:

- `mean_score = -1.1517`
- `mean_slot2_live_fraction = 0.000316`
- `mean_real_branch_fraction = 0.0`
- `mean_meso_branch_effect = 1.67e-10`
- `mean_branch_positive_mask = 0.001894`
- `mean_branch_negative_mask = 0.71033`
- `mean_decorative_slot2_fraction = 0.000232`

Interpretation:

The masking pass improved slot-2 hygiene by teaching the search to keep slot 2 quiet unless strongly justified. But under the current branch law, that mostly meant suppressing slot 2 rather than making it survive lawfully.

## Held-Out Eval

Artifacts:

- `D:\RAFA\outputs\circleworld_proto\heldout_eval_2026-04-17_native_multimode_masked_v1\heldout_summary.json`

Held-out means:

- `mean_major_gain = 0.000724`
- `mean_residue_drop = 0.001325`
- `mean_promotability_gain = 0.000876`
- `mean_dominant_q_share = 0.70348`
- `mean_q_entropy = 0.45038`
- `mean_slot2_live_fraction = 0.01293`
- `mean_real_branch_fraction = 0.0`
- `mean_meso_branch_effect = 3.57e-08`
- `mean_silent_singlepath_fraction = 0.98707`
- `mean_branch_positive_mask = 0.000695`
- `mean_branch_negative_mask = 0.61927`
- `mean_decorative_slot2_fraction = 0.01050`

Interpretation:

This is slightly better structurally than the masked-base smoke pass, but it is still overwhelmingly single-path. The second mode remains mostly dormant, decorative, or immediately suppressed.

## Expanded 12-Case Benchmark

Artifacts:

- `D:\RAFA\outputs\circleworld_proto\benchmark_2026-04-17_native_multimode_masked_v1_expanded\benchmark_summary.json`

Benchmark means:

- `mean_mse = 0.002625`
- `mean_mae = 0.023420`
- `mean_corr = 0.952003`
- `all_circleworld_bitwise_stable = true`

Comparison against the prior native multimode report from 2026-04-16:

- prior native multimode mean MSE: `0.00901`
- prior native multimode mean MAE: `0.04554`
- prior native multimode mean correlation: `0.86058`

Interpretation:

This masking pass is materially better as an audio runtime than the earlier native multimode v1 full run. It preserved deterministic rendering and brought the outputs much closer to the real anchors.

That matters. It means the mask stack is not just aesthetic theory. It improved benchmark fidelity in a measurable way.

But this benchmark improvement should not be misread as branch success. It is a runtime-fidelity win, not a branching win.

## Continuity Sidecars

Artifacts:

- Circleworld outputs: `D:\RAFA\outputs\circleworld_proto\benchmark_2026-04-17_native_multimode_masked_v1_expanded\continuity_circleworld.json`
- Reference outputs: `D:\RAFA\outputs\circleworld_proto\benchmark_2026-04-17_native_multimode_masked_v1_expanded\continuity_reference.json`

Circleworld means:

- `mean_loop_autocorr_peak = 0.62662`
- `mean_loop_period_seconds = 1.45867`
- `mean_adjacent_chunk_similarity = 0.91159`
- `mean_nonlocal_chunk_repeat = 0.95974`
- `mean_first_chunk_reentry = 0.91035`

Reference means:

- `mean_loop_autocorr_peak = 0.62659`
- `mean_loop_period_seconds = 1.45867`
- `mean_adjacent_chunk_similarity = 0.90551`
- `mean_nonlocal_chunk_repeat = 0.96145`
- `mean_first_chunk_reentry = 0.90418`

Interpretation:

The masked Circleworld run tracks the reference continuity profile very closely. That is good in the narrow sense that it did not introduce a continuity catastrophe.

It is not yet a macro-time victory. The model is still mostly preserving the anchor’s time-world rather than discovering a stronger one.

## Nested Commitment Test

Artifacts:

- `D:\RAFA\outputs\circleworld_proto\nested_commitment_2026-04-17_native_multimode_masked_v1\nested_commitment_report.json`

Result:

- `overall_read = nested_commitment_not_yet_established`
- `verdict_counts = {"over_rigid_attractor": 12}`

Interpretation:

Every tested case still failed in the same direction:

- coarse identity stays fixed
- meso continuation does not split into lawful siblings
- perturbations mostly do nothing or produce negligible texture changes

So the masking pass did **not** create Matryoshka-like nested continuation. The system remains over-rigid under fork/resume.

## Most Honest Diagnosis

This pass improved Circleworld in one important way and failed in another.

What improved:

- branch hygiene improved
- decorative slot-2 occupancy was reduced
- benchmark audio fidelity improved strongly versus the previous native multimode run
- bitwise stability remained intact
- all sidecars ran cleanly and are now filed consistently

What did not improve:

- `real_branch_fraction` stayed at `0.0`
- meso branch effect remained effectively zero in practical terms
- silent single-path occupancy remained dominant
- nested commitment still failed as `over_rigid_attractor`

The shortest accurate summary is:

**the masking pass improved branch hygiene and restored runtime fidelity, but it did so mostly by suppressing slot 2 rather than by teaching slot 2 to survive lawfully.**

## Recommendations

### Immediate Circleworld next step

Do **not** respond to this result by adding even stronger global suppression.

The problem is no longer “branch slot 2 is too noisy.”
The problem is “branch slot 2 has almost no lawful path to survive.”

Next work should target **masked branch survival**, not stronger masked branch denial.

Recommended order:

1. Add support-handoff incentives.
   Right now `mean_support_handoff_count` stayed at `0.0`. That is a strong sign the second mode is not becoming a real temporal alternative.

2. Make negative masking more local in time.
   The current branch-negative pressure likely crushes slot 2 too broadly. It should suppress decorative support, not eliminate bounded local alternatives everywhere.

3. Gate split survival on unresolved incompatibility, not just ambiguity.
   Slot 2 should survive when local rational concentration, residue, or compatibility conflict remains genuinely unresolved.

4. Reward bounded branch windows explicitly.
   A healthy branch should appear over a finite support region, persist for a while, and either hand off, merge, or collapse. Right now the system has suppression and dormancy, but not enough survival constitution.

5. Keep the same full sidecar stack for every future branch-law revision.
   The current training/eval/benchmark/continuity/nested-commitment filing pattern is now the right standard.

### Graduation lane recommendation

Leave Graduation alone.

This masking pass was Circleworld-only and should stay that way until Circleworld can demonstrate nonzero real branch survival without hurting runtime fidelity.

## Filing

This should be treated as the canonical masking follow-up to the first native multimode run:

- training: `training_run_2026-04-17_native_multimode_masked_v1`
- held-out: `heldout_eval_2026-04-17_native_multimode_masked_v1`
- benchmark: `benchmark_2026-04-17_native_multimode_masked_v1_expanded`
- nested commitment: `nested_commitment_2026-04-17_native_multimode_masked_v1`

That naming stays consistent with the rest of the Circleworld retrospective and makes this experiment easy to compare against:

- `training_run_2026-04-16_native_multimode_v1_full`
- `benchmark_2026-04-16_native_multimode_v1_full_expanded`
- `heldout_eval_2026-04-16_native_multimode_v1_full`
- `nested_commitment_2026-04-16_native_multimode_v1_full`
