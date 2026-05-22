# Circleworld Phase-Native Reentry Query Router - 2026-05-22

## Claim

Adding explicit no-future reentry/query features to the Circleworld phase-native
route feature surface improves non-KNN learned/resonant route heads on the fourth
lockbox.

This is a Circleworld-internal operator-block test, not a Graduation RAFA
comparison and not a KNN fallback path.

## Implementation

- `runtimes/circleworld_proto/score_phase_native_audio_prefix_router_scout.py`
  extracts no-future reentry-query features from `mechanism_flags`.
- Rows that declare future/target leakage are ignored for these features.
- `runtimes/circleworld_proto/train_phase_native_audio_objective_route_policy.py`
  now groups `reentry`/`loop`/`temporal` keys before generic `phase` keys.

## Artifacts

- Enriched classifier MLP policy:
  `D:\RAFA\outputs\circleworld_proto\phase_native_audio_objective_route_policy_2026-05-22_objective_mlp_v1_reentry_features_original_fresh_third_to_fourth\phase_native_audio_objective_route_policy.json`
- Enriched classifier MLP operator block:
  `D:\RAFA\outputs\circleworld_proto\circleworld_operator_block_v1_2026-05-22_objective_mlp_v1_reentry_features_fourth_full\circleworld_operator_block_v1.json`
- Enriched score MLP policy:
  `D:\RAFA\outputs\circleworld_proto\phase_native_audio_objective_route_policy_2026-05-22_objective_score_mlp_v1_reentry_features_original_fresh_third_to_fourth\phase_native_audio_objective_route_policy.json`
- Enriched score MLP operator block:
  `D:\RAFA\outputs\circleworld_proto\circleworld_operator_block_v1_2026-05-22_objective_score_mlp_v1_reentry_features_fourth_full\circleworld_operator_block_v1.json`
- Enriched resonant-memory policy:
  `D:\RAFA\outputs\circleworld_proto\phase_native_audio_objective_route_policy_2026-05-22_objective_resonant_memory_v1_reentry_features_original_fresh_third_to_fourth\phase_native_audio_objective_route_policy.json`
- Enriched resonant-memory operator block:
  `D:\RAFA\outputs\circleworld_proto\circleworld_operator_block_v1_2026-05-22_objective_resonant_memory_v1_reentry_features_fourth_full\circleworld_operator_block_v1.json`
- Six-run comparison:
  `D:\RAFA\outputs\circleworld_proto\phase_native_audio_route_policy_comparison_2026-05-22_reentry_feature_attempts_fourth\phase_native_audio_route_policy_comparison.json`

## Metrics

| Metric | Value |
| --- | ---: |
| `feature_count_after_extraction` | `124` |
| `reentry_feature_count_before` | `0` |
| `reentry_feature_count_after` | `6` |
| `enriched_resonant_leave_one_route_label_accuracy` | `0.118279570` |
| `operator_block_case_count` | `62` |

## Fourth Lockbox Operator Blocks

| Router | Strict | Future clean | Beats raw | Corr-copy | MSE-copy | Loop-copy | Reentry-copy | Harm-delta | Corr-gain0 | Row objective |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `old_objective_mlp_v1` | `true` | `true` | `true` | `+0.092800381` | `-0.030052639` | `-0.078464776` | `+0.026755963` | `-0.010494122` | `+0.092788538` | `+0.082267108` |
| `old_objective_score_mlp_v1` | `false` | `true` | `false` | `+0.053442600` | `-0.023873799` | `-0.070617068` | `+0.044122135` | `+0.006819712` | `+0.054797230` | `+0.022061384` |
| `old_objective_resonant_memory_v1` | `false` | `true` | `false` | `+0.036257724` | `-0.022778305` | `-0.043177264` | `+0.042260223` | `+0.000573753` | `+0.034580908` | `+0.001308486` |
| `reentry_objective_mlp_v1` | `true` | `true` | `true` | `+0.093308172` | `-0.028538295` | `-0.083435935` | `+0.027583936` | `-0.010720389` | `+0.092690453` | `+0.083630860` |
| `reentry_objective_score_mlp_v1` | `true` | `true` | `true` | `+0.063773463` | `-0.026760185` | `-0.068364567` | `+0.038347875` | `-0.001291646` | `+0.062323908` | `+0.032522740` |
| `reentry_objective_resonant_memory_v1` | `true` | `true` | `false` | `+0.028939929` | `-0.023032581` | `-0.040554935` | `+0.040479539` | `-0.000899973` | `+0.027263112` | `-0.004733058` |

## Interpretation

The missing reentry/query surface was a real bottleneck. With the new features,
the score MLP and resonant-memory route heads both flip from strict-failing to
strict-clean in the canonical `circleworld_operator_block_v1` path.

The best current non-KNN learned route head is `reentry_objective_mlp_v1`.
It slightly improves correlation, loop reduction, harmful replay excess, and
row objective over the previous classifier MLP, with a small MSE regression.

The resonant-memory route is philosophically closer to RAFA attention/memory,
but remains weaker as an audio selector. The strict-clean flip supports
continuing non-KNN resonant routing work, but does not promote this v1 memory.

## Limitations

- The reentry-query features are still shallow scalar summaries, not a full
  operator-valued RAFA attention layer.
- The route policy is still emitted as a case table for the lockbox run.
- The fourth lockbox is one held-out battlefield; broader source families and
  longer horizons still need testing.

## Next Action

Test non-KNN margin/coverage-aware learned routing and pairwise/ranking losses
against the balanced row objective, without abstaining back to KNN.
