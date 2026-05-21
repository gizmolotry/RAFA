# Circleworld Fixed Evaluator Retrospective (2026-04-24)

After fixing deterministic `naked_rafa` substrate construction and collapsing trainer/external heldout evaluation onto one shared implementation, these are the current checkpoints under the corrected evaluator.

| candidate | benchmark_corr | benchmark_mae | naked_branch | naked_parent_div | heldout_branch | heldout_writeback | heldout_law_families | nested |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| agreement_long_v1_fixed | 0.928721 | 0.025142 | 0.333333 | 0.042821 | 0.518519 | 0.114621 | 2.333 | nested_commitment_not_yet_established |
| agreement_scout_v1_fixed | 0.913847 | 0.045055 | 0.400000 | 0.042895 | 0.577778 | 0.095291 | 1.667 | nested_commitment_not_yet_established |
| baseline_parentmix_220_100 | 0.896369 | 0.049241 | 0.400000 | 0.043180 | 0.577778 | 0.095472 | 1.667 | nested_commitment_not_yet_established |
| agreement_fixed_scout_v2 | 0.725511 | 0.066133 | 0.333333 | 0.043879 | 0.555556 | 0.106035 | 2.667 | nested_commitment_not_yet_established |

## Read

- `agreement_scout_v1_fixed` is still the best balanced checkpoint.
- `agreement_long_v1_fixed` wins on benchmark fidelity, but gives up real-substrate branching.
- `agreement_fixed_scout_v2` increased law diversity and sibling-divergence pressure, but benchmark fidelity collapsed and nested commitment still failed.
- `baseline_parentmix_220_100` remains a strong conservative baseline with the best naked branch floor among the stable candidates.

## Recommendation

- Keep `agreement_scout_v1_fixed` as the active Circleworld checkpoint.
- Keep `baseline_parentmix_220_100` as the conservative fallback / manual baseline.
- Do not promote `agreement_fixed_scout_v2`. It is useful as evidence about the tradeoff surface, not as a new default.
- Next training should explicitly protect benchmark fidelity while targeting nested commitment, rather than broad branch-pressure expansion.
