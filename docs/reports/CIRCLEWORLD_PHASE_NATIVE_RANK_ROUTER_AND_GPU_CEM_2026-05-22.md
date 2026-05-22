# Circleworld Phase-Native Rank Router And GPU CEM - 2026-05-22

## Claim

Two non-KNN GPU-backed Circleworld attempts were tested:

1. `objective_rank_mlp_v1`, a CUDA-trained candidate-route ranker for
   `circleworld_operator_block_v1`.
2. A bounded CUDA real-anchor CEM follow-up from the existing heat-training
   checkpoint.

The ranker is viable and strict-clean, but does not beat the best reentry
classifier MLP. The CEM run improves branch/child prevalence while regressing
audio fidelity and increasing decorative slot-2 pressure, so it is not a
promotion candidate.

## Implementation

- `objective_rank_mlp_v1` was added to
  `runtimes/circleworld_proto/train_phase_native_audio_objective_route_policy.py`.
- The route-policy trainer now accepts `--device` for Torch-backed MLP models.
- The ranker uses `_candidate_training_rows()` and trains a tiny route scorer
  over no-future prefix/Circleworld features plus route identity.
- Ranking loss is listwise plus capped within-case hard pair comparisons.
- KNN remains only a historical/reference baseline; this run does not abstain
  or fall back to KNN.

## Ranker Artifacts

- Policy:
  `D:\RAFA\outputs\circleworld_proto\phase_native_audio_objective_route_policy_2026-05-22_objective_rank_mlp_v1_cuda_reentry_features_original_fresh_third_to_fourth\phase_native_audio_objective_route_policy.json`
- Selected-route render:
  `D:\RAFA\outputs\circleworld_proto\phase_native_audio_selected_route_2026-05-22_objective_rank_mlp_v1_cuda_reentry_features_fourth_full\phase_native_audio_selected_route.json`
- Operator block:
  `D:\RAFA\outputs\circleworld_proto\circleworld_operator_block_v1_2026-05-22_objective_rank_mlp_v1_cuda_reentry_features_fourth_full\circleworld_operator_block_v1.json`
- Comparison:
  `D:\RAFA\outputs\circleworld_proto\phase_native_audio_route_policy_comparison_2026-05-22_rank_mlp_vs_reentry_attempts_fourth\phase_native_audio_route_policy_comparison.json`

## Ranker Training Metrics

| Metric | Value |
| --- | ---: |
| `device` | `cuda` |
| `candidate_row_count` | `89232` |
| `candidate_case_count` | `156` |
| `candidate_route_count` | `572` |
| `candidate_pair_count` | `9984` |
| `max_pairs_per_case` | `64` |
| `epochs` | `128` |
| `training_pair_accuracy` | `0.775040064` |
| `training_top1_accuracy` | `0.737179487` |
| `fallback_prediction_count` | `0` |

## Fourth Lockbox Operator Blocks

| Router | Strict | Future clean | Beats raw | Corr-copy | MSE-copy | Loop-copy | Reentry-copy | Harm-delta | Corr-gain0 | Row objective |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `reentry_objective_mlp_v1` | `true` | `true` | `true` | `+0.093308172` | `-0.028538295` | `-0.083435935` | `+0.027583936` | `-0.010720389` | `+0.092690453` | `+0.083630860` |
| `reentry_objective_score_mlp_v1` | `true` | `true` | `true` | `+0.063773463` | `-0.026760185` | `-0.068364567` | `+0.038347875` | `-0.001291646` | `+0.062323908` | `+0.032522740` |
| `reentry_objective_resonant_memory_v1` | `true` | `true` | `false` | `+0.028939929` | `-0.023032581` | `-0.040554935` | `+0.040479539` | `-0.000899973` | `+0.027263112` | `-0.004733058` |
| `cuda_objective_rank_mlp_v1` | `true` | `true` | `true` | `+0.083439596` | `-0.028021033` | `-0.072354774` | `+0.033721016` | `-0.006013128` | `+0.082098667` | `+0.064888374` |

## Ranker Interpretation

`objective_rank_mlp_v1` is a valid non-KNN learned route head. It passes the
selected-route and operator-block strict replay guards, remains future-clean,
and beats raw Circleworld.

It is not the best current selector. The reentry classifier MLP still has
better correlation, loop reduction, harmful replay reduction, and row objective.
The ranker is more faithful to the route-ordering objective than naive scalar
score regression, but it is currently underpowered.

## CEM Artifacts

- Summary:
  `D:\RAFA\outputs\circleworld_proto\real_anchor_cem_2026-05-22_gpu_rank_followup\train_summary.json`
- Search history:
  `D:\RAFA\outputs\circleworld_proto\real_anchor_cem_2026-05-22_gpu_rank_followup\search_history.json`
- Checkpoint:
  `D:\RAFA\checkpoints_circleworld_proto\real_anchor_cem_2026-05-22_gpu_rank_followup\circleworld_real_anchor_config_cem_v1.json`

## CEM Metrics

| Section | Mean score | Mean corr | Mean MSE | Real branch | Live child | Child worlds | Writeback | Parent div | Sibling div | Meso branch | Decorative slot2 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `baseline_train` | `2.107062342` | `0.937615941` | `0.002890573` | `0.574074094` | `0.574074094` | `1.722222249` | `0.042123002` | `0.275117747` | `0.272802236` | `0.013887908` | `0.131340270` |
| `baseline_val` | `2.302873160` | `0.952158070` | `0.002637940` | `0.622222245` | `0.622222245` | `1.866666670` | `0.039694065` | `0.313111019` | `0.309717554` | `0.015655551` | `0.184792608` |
| `best_train` | `2.387495800` | `0.902295431` | `0.004036949` | `0.652777801` | `0.652777801` | `2.611111200` | `0.041267281` | `0.272742252` | `0.270662114` | `0.014019215` | `0.220786107` |
| `best_val` | `2.360550300` | `0.934367979` | `0.003552201` | `0.633333349` | `0.633333349` | `2.533333400` | `0.039517457` | `0.313620913` | `0.310636121` | `0.015681046` | `0.243627036` |

## CEM Interpretation

The CEM follow-up buys more branch/child activity:

- Validation real branch increases from `0.622222245` to `0.633333349`.
- Validation child worlds increase from `1.866666670` to `2.533333400`.
- Validation meso branch effect is roughly flat-to-slightly-up.

But this comes with regressions:

- Validation correlation drops from `0.952158070` to `0.934367979`.
- Validation MSE worsens from `0.002637940` to `0.003552201`.
- Decorative slot-2 rises from `0.184792608` to `0.243627036`.
- `child_local_ifs_enabled` remains `false`, so this does not test the
  self-evolving child-local IFS question.

This checkpoint should not be promoted. It is useful evidence that branch
pressure and child volume can be increased, but not yet converted into cleaner
phase-native audio behavior.

## Next Action

- Keep `reentry_objective_mlp_v1` as the best current non-KNN learned route
  head.
- Improve `objective_rank_mlp_v1` with stronger route-order supervision or a
  larger route embedding before testing it again.
- Do not promote the CEM checkpoint; use it only as a diagnostic seed for
  child-volume and decorative-slot2 analysis.
- If CEM continues, explicitly enable child-local IFS in the init/config path
  so the run targets self-evolving child systems rather than more sidecar
  children.
