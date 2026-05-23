# Circleworld Non-KNN LOOCV And Deep Child-IFS Assay - 2026-05-22

## Claim

This pass extended the phase-native audio route-head evidence from a single fourth-lockbox replay into a four-fold leave-one-lockbox-out suite, then wrapped every held-out selected-route run in `circleworld_operator_block_v1`.

It also ran a deeper learned branch-law / child-local IFS assay with 24 seeds, deeper recurrence, larger local state, and sandbox gain 12.

The route-head result is now stable enough to separate two objectives:

- `objective_rank_mlp_v1` is the best correlation/MSE head.
- `objective_mlp_v1` remains the best practical anti-loop / harmful-replay head.

The child-IFS result is also clearer:

- child-local IFS signal is robust across more seeds and larger state,
- learned branch-law fitting improves,
- causal sandbox authority is still too weak to change live-child ontology or parent writeback materially.

## Artifacts

Root:

`D:\RAFA\outputs\circleworld_proto\phase_native_audio_loocv_2026-05-22_gpu_nonknn`

Deep child-IFS assay:

`D:\RAFA\outputs\circleworld_proto\learned_branch_law_child_ifs_assay_2026-05-22_causal_guarded_deep_gain12\learned_branch_law_assay.json`

## LOOCV Setup

Four held-out folds:

- `original`
- `fresh`
- `third`
- `fourth`

Four learned non-KNN route heads:

- `objective_mlp_v1`
- `objective_score_mlp_v1`
- `objective_rank_mlp_v1`
- `objective_rank_embed_mlp_v1`

Each fold trained on the other three lockboxes, rendered selected routes on the held-out lockbox, and was then wrapped by `circleworld_operator_block_v1`.

A useful artifact-gap was found: the fourth lockbox has the core delta probe but not the auxiliary reentry shear/decorrelator probe set. The corrected LOOCV run used only existing delta probes per source fold.

## Operator-Block Aggregate Results

| Model | Strict folds | Future-clean folds | Beats raw folds | Mean corr | Mean MSE | Mean loop | Mean reentry | Mean harmful replay delta | Harm win frac |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `objective_mlp_v1` | `4` | `4` | `4` | `0.073426126` | `-0.026879779` | `-0.075143140` | `0.028737029` | `-0.010176270` | `0.689516129` |
| `objective_rank_mlp_v1` | `4` | `4` | `4` | `0.081385730` | `-0.026922088` | `-0.073207317` | `0.033150707` | `-0.007663913` | `0.653225806` |
| `objective_rank_embed_mlp_v1` | `4` | `4` | `4` | `0.077321092` | `-0.026611408` | `-0.069791109` | `0.034138302` | `-0.007134661` | `0.629032258` |
| `objective_score_mlp_v1` | `4` | `4` | `4` | `0.057663513` | `-0.025861879` | `-0.071426981` | `0.035363901` | `-0.003698755` | `0.629032258` |

## Fold-Level Notes

All learned route heads strict-passed every held-out fold. This is stronger than the previous single fourth-lockbox replay.

The metric ordering is not one-dimensional:

- `objective_rank_mlp_v1` is best on mean correlation and mean MSE.
- `objective_mlp_v1` is best on loop reduction, harmful replay reduction, and harmful replay win fraction.
- `objective_rank_embed_mlp_v1` remains useful but not leading; it trails `objective_rank_mlp_v1` on correlation and trails `objective_mlp_v1` on replay control.
- `objective_score_mlp_v1` strict-passes but is the weakest of the four heads on the branch-relevant anti-replay target.

## Training Diagnostics

`objective_mlp_v1` uses source-derived best labels and trained on 186 case-level rows per fold.

The rank-style heads trained on tens of thousands of candidate rows:

- `objective_rank_mlp_v1`: route-label/top-1 accuracy ranged from `0.713450292` to `0.829268293`.
- `objective_rank_embed_mlp_v1`: route-label/top-1 accuracy ranged from `0.634146341` to `0.743589744`.
- all rank-style runs had `fallback_prediction_count = 0` and `source_future_access_clean = true`.

`objective_score_mlp_v1` reports `route_label_accuracy = 0.0` because it is score-regression oriented rather than a route-label classifier. Its selected-route artifacts still strict-pass, but its replay metrics lag.

## Deep Child-IFS Assay

Configuration highlights:

- checkpoint: `continuation_causal_childifs_2026-05-11_guarded`
- seeds: `24`
- depth: `8`
- bins: `96`
- time steps: `192`
- child-local steps: `8`
- max points per state: `4096`
- epochs: `3000`
- sandbox gain: `12.0`

Results:

| Metric | Value |
| --- | ---: |
| `status` | `child_local_ifs_signal_present` |
| `feature_rows` | `786432` |
| `live_child_case_count` | `24 / 24` |
| `sibling_case_count` | `24` |
| `mean_isolated_coherence` | `0.680792071` |
| `mean_coupled_vs_isolated_phase_delta` | `0.034777730` |
| `mean_parent_writeback_divergence` | `0.000000674` |
| `mean_sibling_phase_delta` | `0.360528478` |
| `shadow_law.train_loss_final` | `0.000323318` |
| `sandbox_mean_phase_delta` | `0.000004217` |
| `sandbox_mean_live_child_delta` | `0.000000000` |
| `sandbox_mean_writeback_delta` | `0.000000049` |
| `sandbox_mean_parent_divergence_delta` | `-0.000000362` |
| `sandbox_mean_sibling_divergence_delta` | `0.000005273` |

## Interpretation

The route-head lane now has a clean internal hierarchy:

1. If the objective is audio-facing anti-loop / harmful replay suppression, keep `objective_mlp_v1` as the current practical leader.
2. If the objective is correlation/MSE, `objective_rank_mlp_v1` is the best learned route scorer.
3. Route embeddings are conceptually attractive but not yet better than the simpler rank MLP.
4. Score regression alone is not enough for the current replay objective.

The child-IFS lane shows robust child-local survival but weak causal writeback. The bottleneck is no longer whether child-local IFS signal exists; it does. The bottleneck is authority: learned branch-law outputs do not yet move parent mode ontology or live-child survival meaningfully.

## Next Action

- Build a hybrid route head that combines `objective_mlp_v1` replay features with rank-style candidate scoring.
- Do not promote `objective_rank_embed_mlp_v1` yet.
- For child-IFS, stop training only against parametric imitation targets. Add causal targets for parent divergence, writeback mass, live-child survival delta, and sibling divergence delta.
- Keep all future route-head work wrapped in `circleworld_operator_block_v1`, because the block evidence now reproduces selected-route evidence cleanly across all held-out folds.
