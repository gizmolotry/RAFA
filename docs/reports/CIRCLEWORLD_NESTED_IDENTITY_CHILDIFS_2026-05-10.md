# Circleworld Nested Identity Child-IFS Run - 2026-05-10

## Decision

Hold. Do not promote `nested_identity_childifs`.

This run improved some generic held-out child-local metrics, but it failed the fixed seeded nested substrate that matters for the ontology cycle. The checkpoint has no live child signal on `naked_rafa` seeds `9100/9101/9102` at `time_steps=128`, so the nested assay cannot test same-child identity carry. That is a hard fail for branch ontology, even though benchmark/audio stayed healthy.

## Checkpoint Under Test

`D:\RAFA\checkpoints_circleworld_proto\nested_identity_childifs_2026-05-10\circleworld_real_anchor_config_cem_v1.json`

Training selected `selection_source=best_agreement_candidate`. The training-time nested gate did not pass: identity carry, qualified carry, budget retained, and identity sibling divergence all remained below gate.

## What Changed In This Cycle

- Extended `train_circleworld_real_anchor.py` selection scoring with nested identity carry, qualified carry, budget retained, identity parent divergence, and identity sibling divergence.
- Ran a focused CEM pass initialized from `naked_ontology_childifs` with child-local IFS enabled.
- Added `audit_seeded_child_substrate.py` to detect whether the fixed seeded nested substrate actually contains live child worlds across recurrence depths.
- Ran external held-out, nested commitment, benchmark, learned-law sandbox, and seeded-substrate audits.

## Held-Out Evaluation

| config | real branch | naked branch | synthetic branch | writeback | naked writeback | parent div | naked parent div | sibling div | local steps | naked local delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| agreement_baseline | 0.577778 | 0.400000 | 0.666667 | 0.124880 | 0.102351 | 0.225840 | 0.050027 | 0.169703 | 0.000000 | 0.000000 |
| child_local_v2_weighted | 0.444444 | 0.000000 | 0.666667 | 0.091739 | 0.000000 | 0.210885 | 0.000000 | 0.197331 | 0.888889 | 0.000000 |
| naked_ontology_childifs | 0.577778 | 0.400000 | 0.666667 | 0.125679 | 0.102178 | 0.231855 | 0.056503 | 0.171607 | 1.333333 | 0.067336 |
| nested_identity_childifs | 0.555556 | 0.333333 | 0.666667 | 0.126495 | 0.104701 | 0.233410 | 0.056914 | 0.193508 | 1.333333 | 0.055244 |

Interpretation: generic held-out behavior is not catastrophic. The new checkpoint keeps child-local IFS active and improves sibling divergence over `naked_ontology_childifs`, but naked branch fraction drops from `0.400000` to `0.333333`, below the branch-first reference.

## Nested Commitment

| config | live child start cases | nested sibling | identity carry | qualified carry | budget retained | readout sibling response | parent div | sibling div | verdict |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| nested_agreement_baseline | 3 | 0.000000 | 0.937500 | 0.045833 | n/a | 8.584920 | 0.083977 | 0.069969 | no sibling |
| nested_naked_ontology_childifs | 3 | 0.000000 | 0.424306 | 0.045833 | n/a | 6.579102 | 0.081968 | 0.082436 | no sibling |
| nested_identity_childifs | 0 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | mixed_or_inconclusive |

Interpretation: this is the primary failure. The new checkpoint does not merely fail the stricter `nested_sibling` label; it has no live child start cases on the fixed nested substrate, so identity carry cannot start.

## Seeded Child-Substrate Audit

Fixed nested substrate: `naked_rafa`, seeds `9100/9101/9102`, `time_steps=128`, depth `5`.

| checkpoint | live seed fraction | signal seed fraction | mean best child score | mean best live child | mean child worlds | live depth fraction |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| agreement_scout_v1 | 1.0000 | 1.0000 | 2.346621 | 1.0000 | 5.0000 | 0.8000 |
| child_local_v2_weighted | 0.0000 | 0.0000 | 0.000000 | 0.0000 | 0.0000 | 0.0000 |
| naked_ontology_childifs | 1.0000 | 1.0000 | 1.453187 | 0.6000 | 3.0000 | 0.8000 |
| nested_identity_childifs | 0.0000 | 0.0000 | 0.000000 | 0.0000 | 0.0000 | 0.0000 |

Compact assay substrate: same seeds, `time_steps=32`, depth `4`.

| checkpoint | live seed fraction | signal seed fraction | mean best child score | mean best live child | mean child worlds | live depth fraction |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| agreement_scout_v1 | 1.0000 | 1.0000 | 2.520144 | 1.0000 | 5.0000 | 0.7500 |
| naked_ontology_childifs | 1.0000 | 1.0000 | 2.528228 | 1.0000 | 5.0000 | 0.7500 |
| nested_identity_childifs | 1.0000 | 1.0000 | 2.777968 | 1.0000 | 6.0000 | 0.7500 |

Interpretation: the child framework is substrate-scale sensitive. The new checkpoint can spawn children on the compact learned-law assay substrate, but fails on the fixed `128`-step nested substrate. This explains why learned-law child probes can look alive while nested commitment reads zero.

## Learned Branch-Law Sandbox

| config | shadow loss | isolated coherence | writeback div | sibling delta | sandbox phase delta | sandbox live-child delta | sandbox writeback delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| agreement_baseline | 0.037223 | 0.763071 | 0.079642 | 0.421111 | 0.000645 | 0.000000 | 0.000000 |
| child_local_v2_weighted | 0.034012 | 0.701301 | 0.054408 | 0.461545 | 0.000043 | 0.000000 | 0.000000 |
| naked_ontology_childifs | 0.035378 | 0.669767 | 0.043253 | 0.523814 | 0.000258 | 0.000000 | 0.000000 |
| nested_identity_childifs | 0.037002 | 0.631511 | 0.049127 | 0.547658 | 0.000174 | 0.000000 | 0.000000 |

Interpretation: the tiny learned law still fits the shadow target but does not control live-child survival or writeback. It is still an assay-readout module, not a behavioral branch law.

## Benchmark Smoke

| config | mean corr | mean mse | mean mae | bitwise stable |
| --- | ---: | ---: | ---: | --- |
| agreement_baseline | 0.970819 | 0.003224 | 0.026708 | true |
| naked_ontology_childifs | 0.976594 | 0.002535 | 0.024810 | true |
| nested_identity_childifs | 0.987291 | 0.001367 | 0.020150 | true |

Interpretation: parent audio continuity is not the limiting factor. The new checkpoint is benchmark-safe, but ontology-failing.

## What We Learned

1. Branch survival, child-local recurrence, and nested identity are now clearly separable claims.
2. Child-local IFS can be active without producing nested sibling identity.
3. The fixed seeded nested substrate must be a hard training/evaluation gate. Generic held-out `naked_rafa` seeds are not enough.
4. The learned branch law is still not a learned controller. It currently learns a shadow surface whose sandbox outputs barely perturb parent/child behavior.
5. The child framework is scale-sensitive: `time_steps=32` can look healthy while `time_steps=128` has zero live child signal for the same seeds and checkpoint.

## Next Technical Move

Do not promote `nested_identity_childifs`. Use `naked_ontology_childifs` as the best child-local ontology checkpoint for now, because it preserves fixed-seed live child starts.

The next training run should optimize the fixed seeded substrate directly:

- Gate candidate selection with `audit_seeded_child_substrate` style metrics on seeds `9100/9101/9102` at `time_steps=128`.
- Require nonzero live child start before nested identity terms can score.
- Preserve the `naked_ontology_childifs` floor: `live_seed_fraction=1.0`, `mean_best_child_score>=1.45`, and nonzero writeback.
- Only then score same-child carry, qualified carry, and nested sibling.
- Keep learned branch law in shadow mode until sandbox live-child/writeback deltas are nonzero.

## Artifacts

- Held-out summary: `D:\RAFA\outputs\circleworld_proto\nested_identity_childifs_2026-05-10\heldout_nested_identity\heldout_summary.json`
- Held-out comparison: `D:\RAFA\outputs\circleworld_proto\nested_identity_childifs_2026-05-10\heldout_compare.json`
- Nested report: `D:\RAFA\outputs\circleworld_proto\nested_identity_childifs_2026-05-10\nested_nested_identity\nested_commitment_report.json`
- Nested comparison: `D:\RAFA\outputs\circleworld_proto\nested_identity_childifs_2026-05-10\nested_compare.json`
- Benchmark summary: `D:\RAFA\outputs\circleworld_proto\nested_identity_childifs_2026-05-10\benchmark_nested_identity\benchmark_summary.json`
- Learned-law assay: `D:\RAFA\outputs\circleworld_proto\nested_identity_childifs_2026-05-10\assay_nested_identity_sandbox\learned_branch_law_assay.json`
- Fixed substrate audit: `D:\RAFA\outputs\circleworld_proto\nested_identity_childifs_2026-05-10\seeded_child_substrate_audit\seeded_child_substrate_audit.json`
- Compact substrate audit: `D:\RAFA\outputs\circleworld_proto\nested_identity_childifs_2026-05-10\compact_child_substrate_audit\seeded_child_substrate_audit.json`
