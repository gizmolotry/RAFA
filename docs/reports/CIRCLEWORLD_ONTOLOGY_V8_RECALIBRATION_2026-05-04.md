# Circleworld Ontology V8 Recalibration (2026-05-04)

## Scope
Circleworld only. Graduation untouched.

## What changed
We implemented the ontology-first recalibration inside `D:\RAFA\runtimes\circleworld_proto\test_nested_commitment.py`.

Added:
- child identity carry tracking
- new branch identity aggregate metrics
- new parent mode replacement assay branches
- new `nested_sibling` definition based on carry + readout sibling response instead of the legacy probe shortcut

New branch families:
- `child_mode1_replace_50_shift`
- `child_mode1_replace_85_shift`

## Runs
Primary checkpoint:
- `D:\RAFA\checkpoints_circleworld_proto\training_run_2026-04-24_agreement_scout_v1\circleworld_real_anchor_config_cem_v1.json`
- output: `D:\RAFA\outputs\circleworld_proto\nested_seeded_refresh_agreement_scout_v1_ontology_v8_2026-05-04`

Fallback baseline:
- `D:\RAFA\checkpoints_circleworld_proto\manual_candidates\circleworld_parentmix_220_100_2026-04-24.json`
- output: `D:\RAFA\outputs\circleworld_proto\nested_seeded_refresh_parentmix_220_100_ontology_v8_2026-05-04`

Compare JSON:
- `D:\RAFA\outputs\circleworld_proto\ontology_v8_compare_2026-05-04.json`

## Main result
By the new ontology-cycle criteria, this run technically succeeds.

Observed on both checkpoints:
- `mean_nested_sibling_fraction = 0.0833333333`
- `mean_branch_identity_carry = 1.0`
- `mean_readout_sibling_response > 0`
- no winning branch is labeled overwrite/world-jump

Nested sibling branches appeared in the seeded suite.

For the active checkpoint, the winning ontology branch is:
- `child_dominant_continuation_shift`

For the fallback baseline, the winning ontology branch is also:
- `child_dominant_continuation_shift`

## Important caution
This success is real in the sense that the new assay now detects branch-identity-carry plus sibling-like child-led continuation.

But the carry signal is currently saturated:
- `mean_branch_identity_carry = 1.0`
- `same_child_carry_fraction = 1.0` on essentially all seeded branches

So the new identity metric is too permissive in its current form.
It proves that the same child record remains present across the measured interval, but it does not yet sharply distinguish meaningful branch identity from trivial persistence.

## Read vs previous rung
Compared with `v7` on the active checkpoint:
- `mean_nested_sibling_fraction` moved from `0.0` to `0.0833333333`
- `mean_readout_sibling_response` stayed nonzero and slightly improved in the ontology run
- `mean_child_survival_signal` stayed `0.0`

So we crossed the formal success bar, but not because the old survival probe woke up. We crossed it because the new carry-and-readout ontology made sibling-like branches detectable.

## Comparative read: active vs fallback
Active checkpoint:
- `mean_readout_sibling_response = 0.2714875949`
- `mean_branch_identity_parent_div = 0.0710941644`
- `mean_branch_identity_sibling_div = 0.0772453087`

Fallback baseline:
- `mean_readout_sibling_response = 0.2752790369`
- `mean_branch_identity_parent_div = 0.0722051789`
- `mean_branch_identity_sibling_div = 0.0703979316`

So neither checkpoint clearly dominates the other on ontology-only terms.
The fallback is slightly stronger on readout sibling response; the active checkpoint is slightly stronger on sibling divergence.
That means the active checkpoint should remain the default, but not because `v8` proved it decisively better than the fallback.

## Interpretation
What we now know:
- child-led continuation can satisfy the new sibling criterion
- parent mode replacement branches were useful but did not beat the dominant continuation branch
- the ontology failure has narrowed from “no sibling signal at all” to “identity carry is too easy and survival remains underdefined”

What we do not know yet:
- whether branch identity is genuinely robust, or just trivially carried by a persistent child record
- whether this survives a stricter carry definition

## Recommendation
Next step should not be more branch families yet.
The next step should be to tighten the ontology metric itself:
1. make `same_child_carry_fraction` require nontrivial support/coherence over threshold, not mere child presence
2. make carry fail if the child becomes support-negligible or budget-empty through the interval
3. rerun the same seeded suite unchanged so we can see whether the current `nested_sibling` survives a stricter carry constitution

## Monitoring note
No new benchmark/audio export was run here because this was an assay-only recalibration on existing checkpoints.
