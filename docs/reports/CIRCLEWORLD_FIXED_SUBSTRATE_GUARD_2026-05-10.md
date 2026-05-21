# Circleworld Fixed Substrate Guard Run - 2026-05-10

## Decision

Do not declare a nested ontology promotion. Preserve `near_miss_candidate00` as a named branch-first scout checkpoint.

The fixed-substrate guard worked: it refused to promote candidates that erased live child worlds on seeded `naked_rafa` cases `9100/9101/9102` at `time_steps=128`. The saved checkpoint for `fixed_substrate_guard_childifs` is the conservative init fallback, equivalent to `naked_ontology_childifs`.

A near-miss candidate emerged that is scientifically useful: it preserves fixed seeded live-child starts, improves held-out branch survival, improves same-child identity carry, and keeps benchmark quality intact. It still produces `nested_sibling=0`, so it is not the ontology winner.

## Named Scout Checkpoint

`D:\RAFA\checkpoints_circleworld_proto\fixed_substrate_guard_candidate00_2026-05-10\circleworld_real_anchor_config_cem_v1.json`

Status: scout candidate, not active.

Reason: preserves fixed seeded child substrate and improves branch/carry metrics, but misses the nested sibling objective.

## Training Selection Outcome

`fixed_substrate_guard_childifs` selected `init_config_fallback_nested_or_agreement_required`.

That is the intended behavior under the new guard. No searched candidate passed all nested gates, and missing-nested fallback was disabled.

Failure summary from the selection pass:

- `num_nested_gate_pass=0`
- top candidate only failed `nested_identity_sibling_div`
- most other candidates failed live-child substrate checks outright
- fallback checkpoint kept the known-good fixed seeded child substrate

## Held-Out Comparison

| config | real branch | naked branch | writeback | naked writeback | parent div | naked parent div | sibling div | local steps |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| naked_ontology_childifs | 0.577778 | 0.400000 | 0.125679 | 0.102178 | 0.231855 | 0.056503 | 0.171607 | 1.333333 |
| nested_identity_childifs | 0.555556 | 0.333333 | 0.126495 | 0.104701 | 0.233410 | 0.056914 | 0.193508 | 1.333333 |
| fixed_substrate_guard_fallback | 0.577778 | 0.400000 | 0.125679 | 0.102178 | 0.231855 | 0.056503 | 0.171607 | 1.333333 |
| near_miss_candidate00 | 0.611111 | 0.500000 | 0.130911 | 0.106781 | 0.240199 | 0.055556 | 0.174903 | 1.333333 |

Interpretation: `near_miss_candidate00` is the best branch-first candidate in this mini-cycle. It beats `naked_ontology_childifs` on held-out real branch fraction, naked branch fraction, writeback, and benchmark MAE, while keeping child-local IFS active.

## Nested Ontology Comparison

| config | live start fraction | nested sibling | identity carry | qualified carry | budget retained | parent div | sibling div | readout response | world jump | over rigid |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| naked_ontology_childifs | 1.000000 | 0.000000 | 0.424306 | 0.045833 | 0.793438 | 0.081968 | 0.082436 | 0.235043 | 0.000000 | 0.000000 |
| nested_identity_childifs | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| fixed_substrate_guard_fallback | 1.000000 | 0.000000 | 0.424306 | 0.045833 | 0.793438 | 0.081968 | 0.082436 | 0.235043 | 0.000000 | 0.000000 |
| near_miss_candidate00 | 1.000000 | 0.000000 | 0.576389 | 0.045833 | 0.735082 | 0.090172 | 0.065301 | 0.274977 | 0.000000 | 0.000000 |

Interpretation: `near_miss_candidate00` improves identity carry and readout response without causing world jump or over-rigid collapse. It loses some identity sibling divergence relative to `naked_ontology_childifs`, which is why the strict nested gate rejected it.

## Fixed Seed Substrate Audit

Fixed substrate: `naked_rafa`, seeds `9100/9101/9102`, `time_steps=128`, depth `5`.

| config | live seed fraction | signal seed fraction | mean best child score | mean best live child | mean child worlds | live depth fraction |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| naked_ontology_childifs | 1.000000 | 1.000000 | 1.453187 | 0.600000 | 3.000000 | 0.800000 |
| fixed_substrate_guard_fallback | 1.000000 | 1.000000 | 1.453187 | 0.600000 | 3.000000 | 0.800000 |
| near_miss_candidate00 | 1.000000 | 1.000000 | 2.109719 | 1.000000 | 4.000000 | 0.800000 |

Interpretation: the near-miss is not a decorative metric artifact. It strengthens the fixed seed substrate: best child score rises from `1.453187` to `2.109719`, best live child fraction rises from `0.600000` to `1.000000`, and child worlds rise from `3` to `4`.

## Benchmark Smoke

| config | mean corr | mean mse | mean mae | bitwise stable |
| --- | ---: | ---: | ---: | --- |
| naked_ontology_childifs | 0.976594 | 0.002535 | 0.024810 | true |
| nested_identity_childifs | 0.987291 | 0.001367 | 0.020150 | true |
| fixed_substrate_guard_fallback | 0.976594 | 0.002535 | 0.024810 | true |
| near_miss_candidate00 | 0.977907 | 0.002518 | 0.022798 | true |

Interpretation: the near-miss is benchmark-safe for this level of testing.

## What We Learned

1. The previous nested-identity failure was real: it erased the fixed seeded child substrate despite decent generic held-out metrics.
2. Explicit fixed-substrate gates prevent that failure mode.
3. Same-child identity carry can be improved without getting `nested_sibling` yet.
4. The current `nested_sibling` bottleneck is now narrower: not live child start, not writeback, not coarse collapse, but sibling divergence / branch label formation under identity carry.
5. The strict sibling-divergence threshold may be mixing two stages: first preserve/carry the child, then encourage sibling divergence. Treating both as a single hard gate rejected a useful intermediate candidate.

## Next Move

Use `near_miss_candidate00` as the next init for an identity-carry-plus-divergence run.

Recommended changes:

- Keep fixed seeded live-child start as a hard gate at `1.0`.
- Keep identity carry floor near `0.55`.
- Lower the first-pass identity sibling divergence floor to `0.06` so near-miss-like candidates can survive.
- Add a second-stage objective that raises sibling divergence without sacrificing carry.
- Keep `nested_sibling` as the final label, but do not require it during the preservation stage.

## Artifacts

- Comparison JSON: `D:\RAFA\outputs\circleworld_proto\fixed_substrate_guard_childifs_2026-05-10\fixed_substrate_guard_compare.json`
- Training summary: `D:\RAFA\outputs\circleworld_proto\fixed_substrate_guard_childifs_2026-05-10\training\train_summary.json`
- Fixed guard checkpoint: `D:\RAFA\checkpoints_circleworld_proto\fixed_substrate_guard_childifs_2026-05-10\circleworld_real_anchor_config_cem_v1.json`
- Near-miss checkpoint: `D:\RAFA\checkpoints_circleworld_proto\fixed_substrate_guard_candidate00_2026-05-10\circleworld_real_anchor_config_cem_v1.json`
- Near-miss nested report: `D:\RAFA\outputs\circleworld_proto\fixed_substrate_guard_childifs_2026-05-10\training\_selection_nested\candidate_00\nested_commitment_report.json`
- Near-miss held-out summary: `D:\RAFA\outputs\circleworld_proto\fixed_substrate_guard_childifs_2026-05-10\heldout_near_miss_candidate00\heldout_summary.json`
- Near-miss benchmark summary: `D:\RAFA\outputs\circleworld_proto\fixed_substrate_guard_childifs_2026-05-10\benchmark_near_miss_candidate00\benchmark_summary.json`
- Near-miss substrate audit: `D:\RAFA\outputs\circleworld_proto\fixed_substrate_guard_childifs_2026-05-10\near_miss_candidate00_audit\seeded_child_substrate_audit.json`
