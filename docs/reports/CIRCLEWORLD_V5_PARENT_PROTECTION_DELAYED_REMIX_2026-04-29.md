# Circleworld v5: Parent Protection And Delayed Remix

Date: 2026-04-29
Lane: Circleworld only
Status: Filed

## What changed

This cycle tested the coupling hypothesis directly.

Implemented in the nested assay:
- parent-protected pre-unroll
- delayed parent remix
- branch-local lifecycle config overrides

New assay branches:
- `child_parent_protect_shift`
- `child_delayed_remix_shift`

These were added on top of the existing seeded live-child substrate.

## Main result

The coupling variants increased probe sensitivity again.

Response score progression:
- baseline: `0.8775544813013066`
- v4: `0.8783087735915659`
- v5: `0.8784418513112421`

Delta:
- `v5 - v4 = +0.0001330777196762`
- `v5 - baseline = +0.0008873700099355`

So parent protection and delayed remix are not no-ops. They do make the child perturbation matter a little more.

## What did not change

The hard nested metrics are still frozen:
- `mean_child_active_fraction = 0.0`
- `mean_child_meso_response = 0.0`
- `mean_nested_sibling_fraction = 0.0`

So even when we protect the parent and delay remix, the child law still does not become a live sibling continuation.

## Diagnosis

This narrows the failure again.

It is now unlikely that the whole problem is just:
- parent mode 0 being overwritten too fast
- child writeback happening too early
- lack of a child-only pre-unroll

Those matter, but only weakly.

The stronger conclusion is:
- the childworld perturbation can influence the continuation metric landscape
- but the system still lacks a constitution where the child becomes an independently surviving continuation channel

## Recommendation

The next implementation step should stop treating the child as only a sidecar perturbation target.

Next move:
1. give the child a temporary branch-local readout path in the assay
2. measure whether the child can drive its own continuation before weighted parent mixture collapses it
3. if that still fails, the bottleneck is likely in the branch ontology itself, not just lifecycle/coupling

## Artifacts

- compare JSON: `D:\RAFA\outputs\circleworld_proto\nested_seeded_compare_v5_2026-04-29.json`
- nested report: `D:\RAFA\outputs\circleworld_proto\nested_seeded_refresh_agreement_scout_v1_perturb_v5_2026-04-29\nested_commitment_report.json`
