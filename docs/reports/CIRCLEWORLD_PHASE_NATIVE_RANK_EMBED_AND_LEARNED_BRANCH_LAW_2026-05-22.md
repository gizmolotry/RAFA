# Circleworld Phase-Native Rank-Embedding Router And Learned Branch Law - 2026-05-22

## Claim

This pass tested two learned Circleworld components without returning to KNN:

1. `objective_rank_embed_mlp_v1`, a CUDA-trained no-future route ranker that
   scores each candidate route from prefix/Circleworld features plus a learned
   route embedding.
2. A higher-resolution learned branch-law / child-local IFS assay over the
   guarded causal child-IFS checkpoint, plus a sandbox gain sweep.

The route-embedding head is contract-clean and beats raw Circleworld, but it
does not beat the earlier reentry-feature MLP on the fourth lockbox. The
learned branch-law lane shows real child-local IFS signal and scalable sandbox
actuation, but still weak causal authority over live-child ontology.

## Implementation

- Added `objective_rank_embed_mlp_v1` to
  `runtimes/circleworld_proto/train_phase_native_audio_objective_route_policy.py`.
- Registered the route policy in
  `runtimes/circleworld_proto/profile_registry.py`.
- Added synthetic contract tests in
  `runtimes/circleworld_proto/test_circleworld_operator_block_contract.py`.
- Patched
  `runtimes/circleworld_proto/score_phase_native_audio_target_replay_oracle.py`
  so selected-route aggregates use `"<unspecified>"` rather than an empty JSON
  property name for missing `delta_source`.

The new ranker uses no KNN fallback. It preserves the same no-future route
selection contract as the previous objective route heads.

## Route-Embedding Artifacts

- Policy:
  `D:\RAFA\outputs\circleworld_proto\phase_native_audio_objective_route_policy_2026-05-22_objective_rank_embed_mlp_v1_cuda_reentry_features_original_fresh_third_to_fourth\phase_native_audio_objective_route_policy.json`
- Selected-route render:
  `D:\RAFA\outputs\circleworld_proto\phase_native_audio_selected_route_2026-05-22_objective_rank_embed_mlp_v1_cuda_reentry_features_fourth_full\phase_native_audio_selected_route.json`
- Operator block:
  `D:\RAFA\outputs\circleworld_proto\circleworld_operator_block_v1_2026-05-22_objective_rank_embed_mlp_v1_cuda_reentry_features_fourth_full\circleworld_operator_block_v1.json`
- Comparison:
  `D:\RAFA\outputs\circleworld_proto\phase_native_audio_route_policy_comparison_2026-05-22_rank_embed_vs_reentry_attempts_fourth\phase_native_audio_route_policy_comparison.json`

## Route-Embedding Training Metrics

| Metric | Value |
| --- | ---: |
| `device` | `cuda` |
| `candidate_row_count` | `89232` |
| `candidate_case_count` | `156` |
| `candidate_route_count` | `572` |
| `candidate_pair_count` | `9984` |
| `epochs` | `160` |
| `hidden_dim` | `48` |
| `route_embedding_dim` | `16` |
| `training_pair_accuracy` | `0.750400641` |
| `training_top1_accuracy` | `0.743589744` |
| `fallback_prediction_count` | `0` |
| `source_future_access_clean` | `true` |

## Fourth Lockbox Comparison

| Router | Strict | Corr-copy | MSE-copy | Loop-copy | Reentry-copy | Harm-delta | Harm win frac | Corr-gain0 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `reentry_objective_mlp_v1` | `true` | `+0.093308172` | `-0.028538295` | `-0.083435935` | `+0.027583936` | `-0.010720389` | `0.693548387` | `+0.092690453` |
| `objective_score_mlp_v1` | `true` | `+0.063773463` | `-0.026760185` | `-0.068364567` | `+0.038347875` | `-0.001291646` | `0.596774194` | `+0.062323908` |
| `objective_rank_mlp_v1` | `true` | `+0.083439596` | `-0.028021033` | `-0.072354774` | `+0.033721016` | `-0.006013128` | `0.661290323` | `+0.082098667` |
| `objective_rank_embed_mlp_v1` | `true` | `+0.083608555` | `-0.028142850` | `-0.066815291` | `+0.033022939` | `-0.005827460` | `0.629032258` | `+0.081866005` |

The route-embedding head is slightly better than `objective_rank_mlp_v1` on
mean correlation and MSE, but worse on loop reduction and harmful replay. It
does not beat `reentry_objective_mlp_v1`.

The associated `circleworld_operator_block_v1` artifact passed:

- `status = operator_block_target_replay_pass`
- `strict = true`
- `future_access_clean = true`
- `beats_raw_circleworld = true`
- empty JSON property audit: `0` empty keys after regeneration

## Learned Branch-Law Artifacts

- Shadow table fit:
  `D:\RAFA\outputs\circleworld_proto\shadow_learned_branch_law_2026-05-22_cuda_h1024_e30000\shadow_branch_law_training.json`
- High-resolution child-IFS assay:
  `D:\RAFA\outputs\circleworld_proto\learned_branch_law_child_ifs_assay_2026-05-22_causal_guarded_hi\learned_branch_law_assay.json`
- Gain sweep:
  `D:\RAFA\outputs\circleworld_proto\learned_branch_law_child_ifs_assay_2026-05-22_causal_guarded_gain0p5\learned_branch_law_assay.json`
  `D:\RAFA\outputs\circleworld_proto\learned_branch_law_child_ifs_assay_2026-05-22_causal_guarded_gain3p0\learned_branch_law_assay.json`
  `D:\RAFA\outputs\circleworld_proto\learned_branch_law_child_ifs_assay_2026-05-22_causal_guarded_gain6p0\learned_branch_law_assay.json`

## Shadow Branch-Law Fit

The standalone shadow fit over the curated May 15 branch-law table passed:

| Metric | Value |
| --- | ---: |
| `row_count` | `17` |
| `source_count` | `6` |
| `device` | `cuda` |
| `epochs` | `30000` |
| `hidden_dim` | `1024` |
| `final_train_loss` | `0.0000000147` |
| `final_mean_mae` | `0.000065843` |
| `mean_holdout_mae` | `0.000378024` |
| `max_holdout_mae` | `0.001756468` |

This proves the tiny learned branch-law module can imitate the curated shadow
law table. It does not prove broad branch-law generalization because the table
is still very small.

## Child-Local IFS Assay

The high-resolution guarded assay produced:

| Metric | Value |
| --- | ---: |
| `status` | `child_local_ifs_signal_present` |
| `feature_rows` | `147456` |
| `live_child_case_count` | `12 / 12` |
| `sibling_case_count` | `12` |
| `mean_isolated_coherence` | `0.676260387` |
| `mean_coupled_vs_isolated_phase_delta` | `0.017590978` |
| `mean_parent_writeback_divergence` | `0.000001161` |
| `mean_sibling_phase_delta` | `0.451747961` |
| `shadow_law.train_loss_final` | `0.000418396` |
| `sandbox_mean_phase_delta` | `0.000003307` |
| `sandbox_mean_live_child_delta` | `0.000000000` |
| `sandbox_mean_writeback_delta` | `0.000000030` |

This supports the claim that child-local IFS signal exists and that learned
branch-law fitting is feasible on runtime-derived features. It does not show
strong causal authority yet: sandbox intervention moves phase only slightly and
does not change live-child counts.

## Sandbox Gain Sweep

| Gain | Live child cases | Sandbox phase delta | Live-child delta | Writeback delta | Parent divergence delta | Sibling divergence delta |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `0.5` | `12` | `0.000001387` | `0.000000000` | `0.000000009` | `-0.000000016` | `0.000000702` |
| `1.5` | `12` | `0.000003307` | `0.000000000` | `0.000000030` | `-0.000000036` | `0.000002063` |
| `3.0` | `12` | `0.000004933` | `0.000000000` | `0.000000052` | `-0.000000075` | `0.000004093` |
| `6.0` | `12` | `0.000006040` | `0.000000000` | `0.000000109` | `-0.000000132` | `0.000008019` |

The actuator path scales with sandbox gain, but remains too weak to alter
child ontology. The next branch-law step should not just increase gain; it
should train against causal targets that reward parent/writeback divergence
and live-child changes directly.

## Interpretation

The route-embedding head is a valid non-KNN learned route candidate, but not a
promotion over the reentry-feature MLP. It should stay as an architectural
candidate because it is closer to "score candidate law objects" than a plain
classifier, but the current metric leader remains `reentry_objective_mlp_v1`.

The learned branch-law lane made a more important conceptual step: learned law
fitting and child-local IFS signal are both real under the guarded checkpoint.
However, the learned law is currently mostly descriptive. Even when sandbox
gain increases, it produces only tiny writeback/parent-divergence changes and
no live-child count changes.

## Next Action

- Keep `reentry_objective_mlp_v1` as the current fourth-lockbox metric leader.
- Keep `objective_rank_embed_mlp_v1` as a valid non-KNN candidate for future
  route-object scoring, but do not promote it.
- Build the next learned branch-law assay around causal targets, not only
  parametric imitation: parent divergence, writeback mass, live-child survival,
  and sibling divergence should be explicit supervised or reinforcement-style
  targets.
- Continue requiring `circleworld_operator_block_v1` replay and no-future
  audits for every route-head change.
