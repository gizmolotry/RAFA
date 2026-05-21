# Circleworld Branch-Law Ablation Series (2026-04-17)

## Scope

This report covers a Circleworld-only ablation series for a new `relational_qkv_v2` branch law.

The goal was to move branching slightly away from pure scalar inference heuristics and toward a relation-mediated branch coupling mechanism, while keeping:

- the same multimode state
- the same real-anchor training path
- the same held-out evaluator
- the same benchmark and continuity sidecars
- the same nested-commitment test

Graduation runtime was not modified.

## What Was Implemented

Primary runtime file:

- `D:\RAFA\lineages\04_positive_replacement\circleworld.py`

New branch-law surface:

- `branch_law_version`
- `branch_kernel_version`
- `relation_attention_gain`
- `relation_attention_sharpness`
- `relation_value_gain`
- `relation_support_gain`
- `relation_logit_gain`
- `relation_qtrace_gain`
- `relation_residual_mix`

The new path keeps the existing multimode state:

- `phase_modes`
- `mode_logits`
- `mode_support`
- `mode_q_trace`

but adds a second branch constitution:

- build per-mode relational features from evidence, coherence, q-trace strength, residue, support, occupancy, and world context
- compute a tiny 2x2 QKV-style attention over the two local modes
- gate that attention with a configurable branch kernel
- use the resulting cross-mode message to bias:
  - support growth
  - logit growth
  - q-trace inheritance

The intent was not full learned token attention. It was branch-to-branch relational coupling over phase-native state.

Supporting files updated:

- `D:\RAFA\runtimes\circleworld_proto\train_circleworld.py`
- `D:\RAFA\runtimes\circleworld_proto\train_circleworld_real_anchor.py`
- `D:\RAFA\runtimes\circleworld_proto\evaluate_circleworld.py`
- `D:\RAFA\runtimes\circleworld_proto\export_circleworld_audio.py`

New ablation runner:

- `D:\RAFA\runtimes\circleworld_proto\run_branchlaw_ablation_series.py`

## Variants Tested

Output root:

- `D:\RAFA\outputs\circleworld_proto\branchlaw_ablation_series_2026-04-17`

Base config:

- `D:\RAFA\checkpoints_circleworld_proto\training_run_2026-04-17_native_multimode_masked_v1\circleworld_real_anchor_config_cem_v1.json`

Variants:

1. `parametric_masked_control`
2. `relational_qkv_v2_ramanujan`
3. `relational_qkv_v2_phase_only`
4. `relational_qkv_v2_qtrace_only`
5. `relational_qkv_v2_uniform`

Training budget for relational variants:

- iterations: `8`
- population: `6`
- elite count: `2`
- clip length: `10 s`

Shared sidecars per variant:

- real-anchor training summary
- held-out phase eval
- expanded 12-case benchmark
- continuity sidecars
- nested-commitment test

## Aggregate Ranking

Aggregate summary:

- `D:\RAFA\outputs\circleworld_proto\branchlaw_ablation_series_2026-04-17\branchlaw_ablation_summary.json`
- `D:\RAFA\outputs\circleworld_proto\branchlaw_ablation_series_2026-04-17\BRANCHLAW_ABLATION_SUMMARY.md`

Top-to-bottom ranking by the combined score used in the runner:

1. `relational_qkv_v2_qtrace_only`
2. `relational_qkv_v2_uniform`
3. `relational_qkv_v2_ramanujan`
4. `relational_qkv_v2_phase_only`
5. `parametric_masked_control`

## Main Results

### 1. All relational variants beat the masked parametric control on audio fidelity

Control:

- benchmark corr: `0.95200`
- benchmark mae: `0.02342`
- benchmark mse: `0.002625`

Best variant, `relational_qkv_v2_qtrace_only`:

- benchmark corr: `0.96800`
- benchmark mae: `0.01528`
- benchmark mse: `0.001597`

Interpretation:

The relational branch law materially improved runtime fidelity on the expanded benchmark. It did not destabilize rendering. In fact it made the outputs more anchor-faithful.

### 2. The relational law improved the branch corridor, but only slightly

Control:

- `mean_slot2_live_fraction = 0.01289`
- `mean_relation_handoff_drive = 0.0`
- `mean_relation_attn_10 = 0.0`
- `mean_silent_singlepath_fraction = 0.98711`

Best variant, `relational_qkv_v2_qtrace_only`:

- `mean_slot2_live_fraction = 0.02167`
- `mean_relation_handoff_drive = 5.9999e-05`
- `mean_relation_attn_10 = 0.37063`
- `mean_silent_singlepath_fraction = 0.97833`

Interpretation:

The new law did create a measurable branch-to-branch interaction channel. That is real progress.

But the magnitude is still tiny. Handoff exists as a metric now, not yet as a strong dynamical event.

### 3. No variant achieved real branching

Across every variant:

- `mean_real_branch_fraction = 0.0`

That means none of the kernels crossed the line into the current definition of live branch survival.

### 4. No variant achieved nested commitment

Across every variant:

- nested overall read: `nested_commitment_not_yet_established`
- verdict counts: `{"over_rigid_attractor": 12}`

Interpretation:

The system is still not producing sibling meso continuations under fork/resume.

### 5. Ramanujan did not win this ablation

`relational_qkv_v2_ramanujan` ranked below both:

- `relational_qkv_v2_qtrace_only`
- `relational_qkv_v2_uniform`

Key held-out comparison:

`qtrace_only`

- branch score: `-0.58327`
- slot2 live: `0.02167`
- branch negative mask: `0.59226`
- benchmark corr: `0.96800`

`ramanujan`

- branch score: `-0.60523`
- slot2 live: `0.01709`
- branch negative mask: `0.61923`
- benchmark corr: `0.95976`

Interpretation:

In this v2 implementation, the Ramanujan kernel did not open a better survival corridor. It remained a valid kernel interpretation, but it was not the best one under the current law.

That does **not** mean Ramanujan is useless in principle. It means that under this particular branch-coupling constitution, it did not outperform simpler kernels.

## Best Variant

Best config:

- `D:\RAFA\outputs\circleworld_proto\branchlaw_ablation_series_2026-04-17\relational_qkv_v2_qtrace_only\checkpoint\circleworld_real_anchor_config_cem_v1.json`

Notable learned values:

- `branch_law_version = relational_qkv_v2`
- `branch_kernel_version = qtrace_only`
- `aux_mask_suppression = 0.15967`
- `split_support_gain = 0.71124`
- `relation_attention_gain = 0.73019`
- `relation_attention_sharpness = 1.28255`
- `relation_value_gain = 0.37621`
- `relation_support_gain = 0.30021`
- `relation_logit_gain = 0.28747`
- `relation_qtrace_gain = 0.16911`
- `relation_residual_mix = 0.70204`

Interpretation:

The search consistently moved the best relational variant toward:

- lower broad suppression
- stronger branch-to-branch message gains
- moderate cross-branch attention
- stronger support transfer into slot 2

That is exactly the direction we would expect if the previous failure was “no survival corridor.”

## Honest Diagnosis

This series produced a meaningful result:

- relation-mediated branch coupling is better than the current scalar-only masked control
- that improvement is visible both in audio fidelity and in branch-corridor metrics

But the stronger claim is still false:

- the system does not yet branch in the strong sense
- the system does not yet exhibit nested commitment

So the cleanest read is:

**relational coupling improved branch hygiene and branch transport, but it still did not produce lawful sibling continuation.**

## Recommendations

### 1. Keep `relational_qkv_v2` and drop `parametric_v1` as the main Circleworld research baseline

Not because branching is solved.
Because the relational family is now clearly the better place to keep iterating.

### 2. Use `qtrace_only` as the working baseline for the next pass

It won this series.

That makes it the best current scaffold for the next branch-survival experiments.

### 3. Do not over-interpret the Ramanujan result

The Ramanujan-backed kernel lost here, but only inside one particular branch-law form.

Next time, Ramanujan should be tested in a more selective role, for example:

- only as a survival gate
- only as a merge-prevention prior
- only on promoted packet coupling

not necessarily as the main branch-attention kernel.

### 4. The next missing piece is still support handoff

Even the winning variant had:

- `mean_support_handoff_count = 0.0`

That is the biggest remaining failure signal.

The next implementation pass should explicitly model:

- bounded local branch windows
- local takeover
- local release / merge-back

### 5. Keep the same sidecar constitution

This series used the right retrospective stack:

- held-out eval
- expanded benchmark
- continuity sidecars
- nested commitment

That should remain the standard for every future Circleworld branch-law revision.

## Bottom Line

This was not a null result.

What we learned:

- branch-to-branch relation coupling is a better constitution than scalar split/merge/collapse alone
- simpler kernels beat the first Ramanujan-kernel interpretation
- the project is still failing at the true target: live sibling continuation and nested commitment

The next Circleworld move should be:

**keep the relational branch law, keep the sidecars, and implement explicit local handoff survival instead of adding more global pressure.**
