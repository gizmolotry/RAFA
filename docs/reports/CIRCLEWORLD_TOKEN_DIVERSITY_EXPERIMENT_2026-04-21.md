# Circleworld Token-Diversity Experiment (2026-04-21)

## Scope

This run continued the new relation-token direction inside Circleworld only.

The immediate goal was:

- keep the Ramanujan branch-law lane
- add an explicit token-diversity pressure to training
- rerun the full sidecar stack
- compare against the prior `2026-04-17` Ramanujan branch-law checkpoint

Graduation runtime was not modified.

## What Changed

Updated files:

- `D:\RAFA\lineages\04_positive_replacement\circleworld.py`
- `D:\RAFA\runtimes\circleworld_proto\evaluate_circleworld.py`
- `D:\RAFA\runtimes\circleworld_proto\train_circleworld.py`
- `D:\RAFA\runtimes\circleworld_proto\train_circleworld_real_anchor.py`
- `D:\RAFA\runtimes\circleworld_proto\run_token_diversity_experiment.py`
- `D:\RAFA\runtimes\circleworld_proto\README.md`

Main implementation changes:

1. Core summaries now expose law-family collapse metrics:

- `dominant_law_family_share`
- `law_family_entropy`
- `dominant_law_q_share`
- `law_top_q_entropy`
- `num_law_top_q_unique`

2. The real-anchor trainer now scores aggregate law-token diversity, not just audio and branch metrics.

New training pressure includes:

- per-case law-family shortfall
- per-case law-family entropy shortfall
- per-case dominant-family penalty
- per-case dominant-q penalty
- aggregate law-family count shortfall
- aggregate family entropy shortfall
- aggregate top-q entropy shortfall
- aggregate top-q uniqueness shortfall
- aggregate dominant-family penalty
- aggregate dominant-q penalty

3. The search can now mutate the law-packet constitution itself:

- `law_packet_merge_threshold`
- `law_packet_min_score`
- `law_packet_topk_families`

4. A reproducible experiment runner now exists:

- `D:\RAFA\runtimes\circleworld_proto\run_token_diversity_experiment.py`

## Experiment Run

Run output root:

- `D:\RAFA\outputs\circleworld_proto\training_run_2026-04-21_token_diverse_ramanujan`

Checkpoint:

- `D:\RAFA\checkpoints_circleworld_proto\training_run_2026-04-21_token_diverse_ramanujan\circleworld_real_anchor_config_cem_v1.json`

Artifacts:

- train summary:
  - `D:\RAFA\outputs\circleworld_proto\training_run_2026-04-21_token_diverse_ramanujan\train_summary.json`
- held-out eval:
  - `D:\RAFA\outputs\circleworld_proto\training_run_2026-04-21_token_diverse_ramanujan\heldout_eval\heldout_summary.json`
- expanded benchmark:
  - `D:\RAFA\outputs\circleworld_proto\training_run_2026-04-21_token_diverse_ramanujan\benchmark_expanded\benchmark_summary.json`
- continuity:
  - `D:\RAFA\outputs\circleworld_proto\training_run_2026-04-21_token_diverse_ramanujan\benchmark_expanded\continuity_circleworld.json`
  - `D:\RAFA\outputs\circleworld_proto\training_run_2026-04-21_token_diverse_ramanujan\benchmark_expanded\continuity_reference.json`
- nested commitment:
  - `D:\RAFA\outputs\circleworld_proto\training_run_2026-04-21_token_diverse_ramanujan\nested_commitment\nested_commitment_report.json`
- rebuilt law-token library:
  - `D:\RAFA\outputs\circleworld_proto\training_run_2026-04-21_token_diverse_ramanujan\law_token_library\law_token_library.json`
  - `D:\RAFA\outputs\circleworld_proto\training_run_2026-04-21_token_diverse_ramanujan\law_token_library\LAW_TOKEN_LIBRARY.md`
- machine-readable comparison to the older Ramanujan run:
  - `D:\RAFA\outputs\circleworld_proto\training_run_2026-04-21_token_diverse_ramanujan\compare_to_2026-04-17_ramanujan.json`

## Headline Result

This run **did succeed** at broadening the law-token library.

It did **not** succeed at making branching or nested commitment real.

That is the cleanest summary.

## Old vs New

Baseline for comparison:

- `D:\RAFA\outputs\circleworld_proto\branchlaw_ablation_series_2026-04-17\relational_qkv_v2_ramanujan`

### 1. Audio preservation improved

Old Ramanujan benchmark:

- mean corr: `0.95976`
- mean mae: `0.02231`
- mean mse: `0.002303`
- derived audio score: `0.94803`

New token-diversity run:

- mean corr: `0.97108`
- mean mae: `0.01100`
- mean mse: `0.001435`
- derived audio score: `0.96522`

So the token-diversity run was actually **more anchor-faithful** on the expanded 12-case benchmark.

### 2. Law-token diversity improved meaningfully

Old Ramanujan law-token library:

- mean law packets per case: `48.0`
- mean law families per case: `2.17`
- aggregate families: `6`
- aggregate family counts: `[426, 60, 42, 30, 12, 6]`

New token-diversity law-token library:

- mean law packets per case: `21.17`
- mean law families per case: `3.0`
- aggregate families: `9`
- aggregate family counts: `[60, 46, 34, 24, 17, 17, 10, 8, 7]`

Interpretation:

- fewer packets
- more families
- much flatter family distribution

This is the strongest positive result of the run.

The old library was dominated by one massive family.
The new library is still imperfect, but it is far less monopolized.

### 3. The q-collapse weakened, but did not disappear

Important new signal from the rebuilt library:

- one family now has top q = `3`

That matters because the old library was effectively all top-q = `2`.

So the token space is no longer perfectly single-q collapsed.

But the collapse is not solved:

- most families are still top-q = `2`
- held-out law-top-q entropy is still `0.0`
- held-out mean number of unique top-q families is still `1.0`

So the new run broadened family structure faster than it broadened q-structure.

### 4. Branching did not improve

Old held-out branch corridor:

- mean slot2 live fraction: `0.01709`
- mean real branch fraction: `0.0`
- mean silent singlepath fraction: `0.98291`

New held-out branch corridor:

- mean slot2 live fraction: `0.01629`
- mean real branch fraction: `0.0`
- mean silent singlepath fraction: `0.98371`

That is effectively unchanged, maybe slightly worse.

So the new objective did **not** open a true branch survival corridor.

### 5. Nested commitment still failed

Old:

- `nested_commitment_not_yet_established`
- verdict counts: `{"over_rigid_attractor": 12}`

New:

- `nested_commitment_not_yet_established`
- verdict counts: `{"over_rigid_attractor": 12}`

No real movement here.

### 6. Continuity did not materially improve

Old Circleworld continuity:

- mean loop autocorr peak: `0.62662`
- mean nonlocal chunk repeat: `0.95970`
- mean adjacent chunk similarity: `0.91184`
- mean first chunk reentry: `0.91060`

New Circleworld continuity:

- mean loop autocorr peak: `0.62660`
- mean nonlocal chunk repeat: `0.96032`
- mean adjacent chunk similarity: `0.91047`
- mean first chunk reentry: `0.90928`

This is basically flat.

So the token-diversity objective did not yet solve the macro-time repetition problem.

## Best Reading

The experiment produced a real and useful split:

### What improved

- audio preservation
- family-count diversity
- family-balance
- a small amount of top-q broadening inside the aggregate library

### What did not improve

- live branching
- nested commitment
- continuity / repetition
- held-out token diversity on synthetic seeds

This is important because it tells us the token layer is not fake, but also not yet enough.

## What The New Checkpoint Really Is

The new checkpoint is best understood as:

- a better **relation-token extraction** checkpoint

not:

- a better **branching** checkpoint

That is still a valid win.

It means Circleworld can now produce a less collapsed law-token library without sacrificing anchor fidelity.

## Recommendation

The next Circleworld move should be:

1. Keep this new checkpoint as the current law-token checkpoint.

2. Do **not** present it as a branching breakthrough.

3. Run the next experiment as a two-objective continuation:

- preserve this broader law-family distribution
- explicitly pressure top-q diversity and sibling continuation

The most honest next target is:

- keep aggregate family count around `8-10`
- keep family dominance low
- force more than one top-q family to survive across held-out seeds
- then retest nested commitment

## Bottom Line

This experiment worked in one important sense:

- Circleworld now has a trained Ramanujan checkpoint that builds a broader law-token library than the earlier branch-law run.

But it also clarified the remaining wall:

- token diversity can be improved without causing true branching to emerge.

So the next frontier is now sharper:

- not "do tokens exist?"
- but "how do tokens become live sibling continuations instead of just a nicer retrospective clustering?"
