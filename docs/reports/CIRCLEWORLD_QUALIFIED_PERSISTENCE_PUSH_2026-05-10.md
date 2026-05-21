# Circleworld Qualified Persistence Push - 2026-05-10

## Decision

Do not declare nested ontology promotion. Preserve the selected checkpoint as the current best ontology scout for fixed-substrate child identity carry.

Selected checkpoint:

`D:\RAFA\checkpoints_circleworld_proto\qualified_persistence_push_2026-05-10\circleworld_real_anchor_config_cem_v1.json`

Status: scout candidate, not active branch-first checkpoint.

Reason: it improves fixed seeded child substrate strength and same-child carry, but `nested_sibling` remains `0.0` because branch-level qualified carry still lasts for only one continuation state.

## What Changed

- Added max branch identity metrics to `test_nested_commitment.py`:
  - `max_branch_identity_carry`
  - `max_branch_identity_qualified_carry`
  - `max_branch_identity_qualified_steps`
  - `max_branch_identity_budget_retained`
  - `max_branch_identity_parent_div`
  - `max_branch_identity_sibling_div`
  - `best_identity_branch`
- Added trainer scoring/gating hooks in `train_circleworld_real_anchor.py`:
  - `selection_nested_min_max_qualified_carry`
  - `selection_nested_min_max_qualified_steps`
  - `w_nested_probe_max_qualified_carry`
  - `w_nested_probe_max_qualified_steps`
  - corresponding shortfall weights
- Ran a focused CEM pass from `sibling_divergence_candidate00`.

## Training Outcome

The run selected a candidate instead of falling back:

- `selection_source=best_agreement_candidate`
- nested gate passed
- live child start fraction: `1.0`
- selected live child fraction: `1.0`
- identity carry: `0.865278`
- max qualified carry: `0.25`
- max qualified steps: `0.0`
- budget retained: `0.746571`
- parent divergence: `0.092039`
- sibling divergence: `0.076973`

Interpretation: this is a real stage movement on child identity carry, but not yet a real nested sibling.

## Held-Out Comparison

| config | real branch | naked branch | writeback | naked writeback | parent div | sibling div | local steps |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| naked_ontology_childifs | 0.577778 | 0.400000 | 0.125679 | 0.102178 | 0.231855 | 0.171607 | 1.333333 |
| near_miss_candidate00 | 0.611111 | 0.500000 | 0.130911 | 0.106781 | 0.240199 | 0.174903 | 1.333333 |
| sibling_divergence_candidate00 | 0.577778 | 0.400000 | 0.145490 | 0.118950 | 0.242085 | 0.167171 | 1.333333 |
| qualified_persistence_selected | 0.555556 | 0.333333 | 0.139396 | 0.114294 | 0.235624 | 0.151345 | 1.333333 |

Interpretation: the selected checkpoint regresses held-out naked branch fraction compared with the best branch-first scout. It should not replace `near_miss_candidate00` as the branch-first candidate. It is useful specifically as an ontology scout.

## Nested Ontology Comparison

| config | live start | nested sibling | carry | qualified carry | max qualified carry | max qualified steps | budget retained | parent div | sibling div | max sibling div | world jump |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| naked_ontology_childifs | 1.000000 | 0.000000 | 0.424306 | 0.045833 | 0.250000 | 0.000000 | 0.793438 | 0.081968 | 0.082436 | 0.222377 | 0.000000 |
| near_miss_candidate00 | 1.000000 | 0.000000 | 0.576389 | 0.045833 | 0.250000 | 0.000000 | 0.735082 | 0.090172 | 0.065301 | 0.091271 | 0.000000 |
| sibling_divergence_candidate00 | 1.000000 | 0.000000 | 0.838889 | 0.045833 | 0.250000 | 0.000000 | 0.643842 | 0.097056 | 0.078735 | 0.104974 | 0.000000 |
| qualified_persistence_selected | 1.000000 | 0.000000 | 0.865278 | 0.045833 | 0.250000 | 0.000000 | 0.746571 | 0.092039 | 0.076973 | 0.091368 | 0.000000 |

Interpretation: the selected checkpoint has the strongest mean identity carry so far while retaining budget and divergence. The hard blocker is unchanged: `max_qualified_steps=0.0`, so no branch keeps the same child qualified for more than one continuation state.

## Fixed Seed Substrate Audit

Fixed substrate: `naked_rafa`, seeds `9100/9101/9102`, `time_steps=128`, depth `5`.

| config | live seed fraction | signal seed fraction | mean best child score | mean best live child | mean child worlds | live depth fraction |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| near_miss_candidate00 | 1.000000 | 1.000000 | 2.109719 | 1.000000 | 4.000000 | 0.800000 |
| sibling_divergence_candidate00 | 1.000000 | 1.000000 | 2.365795 | 1.000000 | 5.000000 | 0.800000 |
| qualified_persistence_selected | 1.000000 | 1.000000 | 2.608020 | 1.000000 | 6.000000 | 0.800000 |

Interpretation: fixed child-substrate strength is improving monotonically across these scouts.

## Benchmark Smoke

| config | mean corr | mean mse | mean mae | bitwise stable |
| --- | ---: | ---: | ---: | --- |
| naked_ontology_childifs | 0.976594 | 0.002535 | 0.024810 | true |
| near_miss_candidate00 | 0.977907 | 0.002518 | 0.022798 | true |
| sibling_divergence_candidate00 | 0.974498 | 0.002737 | 0.026702 | true |
| qualified_persistence_selected | 0.978767 | 0.002253 | 0.025832 | true |

Benchmark remains acceptable for scout purposes.

## Learned-Law Status

No promotion of learned branch law. The sandbox remains non-behavioral: learned outputs still do not materially change live child or writeback outcomes. The next learned-law step should wait until qualified persistence has a stronger target signal.

## What We Learned

1. Fixed seeded child substrate can be strengthened without world jump or over-rigid collapse.
2. Same-child identity carry can be driven high: `0.424306 -> 0.865278`.
3. Budget retained can be recovered after the sibling-divergence scout: `0.643842 -> 0.746571`.
4. The bottleneck is no longer live child start, carry, budget, parent divergence, or coarse preservation.
5. The bottleneck is qualified persistence over time: `max_qualified_carry=0.25`, `max_qualified_steps=0.0`.
6. Current child identity exists, but it does not remain above support/coherence/budget thresholds for multiple continuation states.

## Next Move

Target the child coherence/support lifecycle directly.

Recommended next run:

- Start from `qualified_persistence_selected`.
- Keep fixed live seed fraction at `1.0`.
- Keep identity carry floor around `0.80`.
- Keep max qualified carry floor at `0.25`.
- Add direct pressure on child coherence/support retention, not just carry.
- Make `selection_nested_min_max_qualified_steps` a soft objective first, then hard gate only after any candidate reaches `>0`.

The important shift: stop optimizing more branches or more divergence; optimize the same child staying qualified for at least two states.

## Artifacts

- Comparison JSON: `D:\RAFA\outputs\circleworld_proto\qualified_persistence_push_2026-05-10\qualified_persistence_compare.json`
- Selected checkpoint: `D:\RAFA\checkpoints_circleworld_proto\qualified_persistence_push_2026-05-10\circleworld_real_anchor_config_cem_v1.json`
- Training summary: `D:\RAFA\outputs\circleworld_proto\qualified_persistence_push_2026-05-10\training\train_summary.json`
- Held-out summary: `D:\RAFA\outputs\circleworld_proto\qualified_persistence_push_2026-05-10\heldout_selected\heldout_summary.json`
- Nested report: `D:\RAFA\outputs\circleworld_proto\qualified_persistence_push_2026-05-10\nested_selected\nested_commitment_report.json`
- Benchmark summary: `D:\RAFA\outputs\circleworld_proto\qualified_persistence_push_2026-05-10\benchmark_selected\benchmark_summary.json`
- Seed audit: `D:\RAFA\outputs\circleworld_proto\qualified_persistence_push_2026-05-10\audit_selected\seeded_child_substrate_audit.json`
