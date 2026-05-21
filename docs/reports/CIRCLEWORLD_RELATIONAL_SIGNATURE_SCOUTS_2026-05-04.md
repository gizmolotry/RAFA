# Circleworld Relational Signature Scouts (2026-05-04)

## Recommendation

- branch-first checkpoint stays `agreement_scout_v1`
- signature-lane next init should be `relsig_balance_guard_v1`
- diversity/stability reference stays `parentmix_220_100`
- do not promote: `relsig_diversity_guard_v1, relsig_confidence_guard_v1`
- read: balance_guard_v1 is the only new scout that materially improves benchmark and metamer stability while preserving nontrivial held-out signature-family diversity; the other scouts satisfy the new objective mainly by over-compressing the signature family structure.

## Table

| run | type | corr | mae | heldout branch | heldout sig fam | heldout sig conf | export agg fam | export conf | metamer prefix | metamer fam drift |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| agreement_scout_v1 | reference | 0.9138 | 0.0451 | 0.5778 | 1.6667 | 0.8592 | 8 | 0.6943 | 0.9922 | 0.3125 |
| parentmix_220_100 | reference | 0.8964 | 0.0492 | 0.5778 | 1.6667 | 0.8485 | 12 | 0.6777 | 0.9918 | 0.2292 |
| relsig_diversity_guard_v1 | scout | 0.9956 | 0.0127 | 0.3556 | 1.0000 | 0.8123 | 5 | 0.5747 | 0.9954 | 0.2292 |
| relsig_confidence_guard_v1 | scout | 0.9792 | 0.0246 | 0.4444 | 1.0000 | 0.8104 | 3 | 0.6242 | 0.9949 | 0.2083 |
| relsig_balance_guard_v1 | scout | 0.9801 | 0.0255 | 0.4444 | 2.3333 | 0.8264 | 8 | 0.6255 | 0.9948 | 0.2708 |

## Metric Leaders

- benchmark_corr: `relsig_diversity_guard_v1`
- heldout_branch: `agreement_scout_v1`
- heldout_sig_families: `relsig_balance_guard_v1`
- heldout_sig_conf: `agreement_scout_v1`
- heldout_sig_entropy: `agreement_scout_v1`
- heldout_branch_mass: `relsig_confidence_guard_v1`
- export_mean_sig_families: `parentmix_220_100`
- export_agg_sig_families: `parentmix_220_100`
- export_sig_conf: `agreement_scout_v1`
- metamer_prefix_cos: `relsig_diversity_guard_v1`
- benchmark_mae: `relsig_diversity_guard_v1`
- metamer_family_delta: `relsig_confidence_guard_v1`
- metamer_conf_delta: `relsig_diversity_guard_v1`

## Notes

- `relsig_diversity_guard_v1` over-optimized benchmark / stability and collapsed export family diversity to 5 aggregate families.
- `relsig_confidence_guard_v1` stayed cleaner than the diversity profile but collapsed export family diversity even harder to 3 aggregate families.
- `relsig_balance_guard_v1` is the useful middle result: held-out signature families rose to 2.3333, benchmark stayed strong, and metamer family drift stayed better than the active checkpoint, but it still does not match `parentmix_220_100` on export-side diversity or `agreement_scout_v1` on held-out branch fraction/confidence.