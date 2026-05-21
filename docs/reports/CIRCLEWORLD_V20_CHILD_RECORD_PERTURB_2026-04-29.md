# Circleworld v20: Child Record Perturb Retrospective

Date: 2026-04-29
Lane: Circleworld only
Status: Filed

## What changed

This cycle pushed the nested assay one step closer to the actual failure surface.

Implemented:
- stronger direct child-record perturbations in `circleworld.py`
  - `child_support_window_shift`
  - `child_writeback_budget_shift`
  - `child_parent_mix_shift`
- live-child-only nested scoring in `test_nested_commitment.py`
- seeded branch-active scout profile `writeback_v20_child_record_perturb`

## What we tested

1. Seeded nested refresh on the active checkpoint using branch-active `naked_rafa` starts with the stronger child-record perturbations.
2. A new `v20` scout run using the same seeded nested substrate in trainer selection.
3. Search-history inspection to see whether the in-loop nested objective became more informative.

## Main result

The assay improved. The model did not.

The important gain is that the nested probe is no longer as flat as the earlier branch-active refreshes.

Search-history read for `v20`:
- `nested_probe_score_min = -32.64`
- `nested_probe_score_max = -29.8566591997161`
- `nested_probe_score_avg = -30.668522345238689`
- history rows = `24`

That means candidate perturbation response is now measurably variable under the seeded live-child probe.

But selection still failed cleanly:
- `selection_source = init_config_fallback_nested_or_agreement_required`
- nested gate pass count = `0`
- agreement gate pass count = `0`

So the stronger perturbations gave the optimizer more signal, but not enough signal to promote a real successor.

## Seeded nested refresh comparison

Baseline seeded refresh:
- `mean_child_response_score = 0.8775544813013066`
- `mean_child_active_fraction = 0.0`
- `mean_child_meso_response = 0.0`
- `mean_coarse_preservation = 0.9999999579964691`

Perturb-v2 seeded refresh:
- `num_live_child_start_cases = 3`
- `num_eligible_nested_cases = 3`
- `mean_child_response_score = 0.8777531048497166`
- `mean_child_active_fraction = 0.0`
- `mean_child_meso_response = 0.0`
- `mean_coarse_preservation = 0.999986110835382`

Delta:
- `mean_child_response_score = +0.0001986235484100`
- `mean_child_active_fraction = 0.0`
- `mean_child_meso_response = 0.0`
- `mean_coarse_preservation = -0.0000138471610871`

Interpretation:
- the new perturbations do move the seeded live-child cases a bit more
- they still do not produce actual child-history-sensitive sibling continuation
- the response remains below the level where nested child activity becomes measurable in the current summary metrics

## Best candidate read inside v20

`v20` did not promote a candidate, but the best internal candidate was not completely flat.

Best-candidate nested probe:
- iteration = `1`
- `mean_child_response_score = 0.8777669133356238`
- `mean_child_active_fraction = 0.0`
- `mean_child_meso_response = 0.0`

That is consistent with the new diagnosis:
- direct perturbations can alter continuation slightly
- the continuation still does not branch lawfully in a way the child-response metrics recognize

## Diagnosis

The bottleneck is now narrower than before.

It is not:
- evaluator drift
- dead seeded substrate
- dead nested probe wiring
- dead perturbation lane

It is now specifically:
- child-history perturbations do not persist into a child-active continuation regime strongly enough to register as `mean_child_active_fraction > 0` or `mean_child_meso_response > 0`

In other words:
- the assay can now see a little movement
- the childworld constitution still collapses too quickly, or never hands its perturbation back into a live continuation law

## Recommendation for the next cycle

Do not broaden search yet.

The next implementation step should target childworld lifecycle persistence directly:
1. perturb child `survival_age` and child readiness for writeback, not just support/phase/mix
2. perturb the childworld continuation window after fork so the child gets at least one extra local unroll before parent remix dominates
3. keep live-child-only nested scoring in place and keep the seeded branch-active probe as the default nested substrate

## Artifacts

- comparison JSON: `D:\RAFA\outputs\circleworld_proto\v20_child_record_perturb_compare_2026-04-29.json`
- seeded nested refresh: `D:\RAFA\outputs\circleworld_proto\nested_seeded_refresh_agreement_scout_v1_perturb_v2_2026-04-29\nested_commitment_report.json`
- v20 run summary: `D:\RAFA\outputs\circleworld_proto\training_run_2026-04-29_child_record_perturb_v20\train_summary.json`
