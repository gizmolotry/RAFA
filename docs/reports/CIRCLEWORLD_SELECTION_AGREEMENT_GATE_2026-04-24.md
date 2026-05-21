# Circleworld Selection Agreement Gate

## What changed

- Added a late-stage heldout evaluator inside `train_circleworld_real_anchor.py` so top candidates are checked with the true heldout logic before promotion.
- Added a heldout gate mirroring the branch-first naked/global thresholds.
- Added probe-vs-heldout agreement scoring so replay probe drift is visible at selection time.
- Updated childworld profiles so future runs use `selection_heldout_top_k=4` and `selection_require_heldout_agreement=true` by default.

## Smoketest

- summary: `D:\RAFA\outputs\circleworld_proto\selection_agreement_smoketest_2026-04-24_b\train_summary.json`
- selection_source: `init_config_fallback_agreement_required`
- agreement candidates evaluated: `2`
- best agreement candidate present: `False`

Read:
- The trainer now refuses to promote a candidate when no top candidate clears the heldout-agreement bar.
- In this smoketest it fell back to the init config instead of silently selecting a replay-fragile candidate.

## Candidate sanity check

- saved: `D:\RAFA\outputs\circleworld_proto\selection_agreement_sanity_2026-04-24.json`

### baseline_parentmix_220_100
- probe: passed=False margin=-0.333000
- heldout: passed=False margin=-0.001670
- agreement weighted abs delta: 1.714993
- heldout naked branch: 0.400000
- heldout naked parent div: 0.038330

### probe_pass_survival_w28_d014_g600
- probe: passed=False margin=-0.002122
- heldout: passed=False margin=-1.000000
- agreement weighted abs delta: 0.546619
- heldout naked branch: 0.400000
- heldout naked parent div: 0.039393

### heldout_best_survival_w28_d018_g520
- probe: passed=False margin=-0.008099
- heldout: passed=False margin=-1.000000
- agreement weighted abs delta: 0.647297
- heldout naked branch: 0.400000
- heldout naked parent div: 0.041117

## Recommendation

- Keep the agreement gate enabled for future Circleworld childworld searches.
- Treat `init_config_fallback_agreement_required` as a correct safety outcome, not as training failure.
- Only promote future candidates when the trainer reports `selection_source = best_agreement_candidate`.
