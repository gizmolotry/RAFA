# Circleworld Phase-Writeback Recalibration - 2026-05-06

## Status

This is a research/runtime recalibration patch, not a promotion.

The patch moves Circleworld from "child worlds exist as sidecar records" toward "child worlds can write parent-visible phase/support evidence back into mode 1." It produces the first measurable naked-RAFA phase-only branch signal in the strongest local run, but it does not establish nested commitment or `nested_sibling`.

## Code Changes

Primary runtime:

- `D:\RAFA\lineages\04_positive_replacement\circleworld.py`
- `D:\RAFA\runtimes\circleworld_proto\test_nested_commitment.py`
- `D:\RAFA\runtimes\circleworld_proto\evaluate_circleworld.py`
- `D:\RAFA\runtimes\circleworld_proto\train_circleworld_real_anchor.py`
- `D:\RAFA\runtimes\circleworld_proto\train_circleworld.py`
- `D:\RAFA\runtimes\circleworld_proto\export_circleworld_audio.py`

Implemented:

- Added `child_support_writeback_mass` to child metrics, runtime summaries, train aggregation, and heldout evaluation.
- Changed `_evolve_child_worlds` to return explicit child identity/support writeback for parent mode 1.
- Parent mode 1 now receives child support, logit, and q-trace writeback when child worlds remain active.
- Added a phase-only child operator seed `_child_operator_phase_bias(...)` derived from the child law signature and support mask.
- Added a gated phase-floor return path so child identity can produce parent-visible phase separation instead of only child-record survival.
- Exposed phase-writeback controls as `CircleworldConfig` fields:
  - `child_operator_seed_gain`
  - `child_operator_promotability_gain`
  - `child_writeback_operator_mix`
  - `child_writeback_phase_floor_target`
  - `child_writeback_phase_floor_threshold_mult`
  - `child_writeback_phase_floor_gain`
  - `child_writeback_parent_mix_cap`
  - `child_support_writeback_gain`
  - `child_support_writeback_floor_gain`
- Plumbed those controls through config loading, export/nested loading, CEM materialization, real-anchor CEM mean/std search, and saved config payloads.
- Added `D:\RAFA\runtimes\circleworld_proto\run_child_writeback_ablation.py` to generate/run fixed support/operator/floor isolation variants.
- Added `D:\RAFA\runtimes\circleworld_proto\run_audio_phase_influence_ablation.py` to scale Circleworld phase deltas against the no-future-magnitude gain-0 carrier.
- Added `D:\RAFA\runtimes\circleworld_proto\run_audio_phase_seed_ablation.py` to compare legal no-future phase seed policies before and after Circleworld rollout.
- Added `D:\RAFA\runtimes\circleworld_proto\run_audio_circle_delta_probe.py` to test Circleworld future-frame phase deltas over the strongest legal copyphase seed.
- Added `D:\RAFA\runtimes\circleworld_proto\run_audio_delta_mechanism_probe.py` to test fixed prefix-only shaping mechanisms over Circleworld future-frame phase deltas.
- Added `D:\RAFA\runtimes\circleworld_proto\run_audio_mechanism_family_sensitivity.py` to run a frozen mechanism row across deduped filename-derived sound-family lockboxes.
- Added `D:\RAFA\runtimes\circleworld_proto\run_audio_electric_motor_holdout.py` to split the strongest fixed-mechanism audio clue into no-razor structural motor, razor, mixed razor diagnostic, and non-motor buzz controls with provider-neutral dedupe.
- Added `D:\RAFA\runtimes\circleworld_proto\run_audio_steady_phenotype_manifest.py` to allocate steady-buzz, no-razor motor, tool, combustion, synth, and razor-probe cases into disjoint content-hash-unique manifests.
- Added `D:\RAFA\runtimes\circleworld_proto\run_audio_phase_phenotype_diagnostics.py` to compute post-hoc feature correlations for the frozen mechanism row over the disjoint manifest.
- Added `D:\RAFA\runtimes\circleworld_proto\run_audio_baseline_stratified_diagnostics.py` to separate fixed-mechanism gains by gain-0 baseline quality instead of letting bad-baseline rescue masquerade as a phenotype law.
- Added `D:\RAFA\runtimes\circleworld_proto\build_audio_predeclared_lockbox.py` to freeze filename-derived, RIFF-readable, disjoint audio lockbox manifests before running any target/baseline metrics.
- Added `D:\RAFA\runtimes\circleworld_proto\compare_audio_lockbox_results.py` to summarize already-run lockbox probes by manifest group and gain-0 baseline-correlation bin without selecting cases by target metrics.
- Extended `D:\RAFA\runtimes\circleworld_proto\run_audio_delta_mechanism_probe.py` with experimental prefix-only stability/energy/velocity/softclip mechanisms while preserving the old core mechanism set as the CLI default.
- Added default-off internal phase-law controls in `CircleworldConfig`: `phase_law_precondition_gain`, `phase_law_velocity_mix`, `phase_law_stability_gain`, `phase_law_softclip`, and `phase_law_low_rank`.
- Added a phase-only temporal stability/velocity preconditioner inside Circleworld packet transport and child writeback, so the next audio intervention changes recurrence rather than only export-time delta shaping.
- Added `D:\RAFA\runtimes\circleworld_proto\compare_internal_phase_law_variants.py` to compare matched lockbox rows between the no-op config and internal phase-law variants.
- Added `D:\RAFA\runtimes\circleworld_proto\run_audio_delta_objective_scout.py` to scout fixed config candidates against a frozen delta-over-copyphase objective row.
- Added `D:\RAFA\runtimes\circleworld_proto\evaluate_dense_signature_claim.py` to compare dense relational signatures against explicit q/law metadata on the same packet stream.
- Added `D:\RAFA\runtimes\circleworld_proto\test_semantic_projector_contract.py` to audit the fixed five-control semantic projector interface as contract smoke only.

## Audio Phenotype Holdout - 2026-05-06

Run:

`C:\Users\Andrew\AppData\Local\Temp\rafa_tokenburst\audio_electric_motor_holdout_cuda_2026_05_06\audio_electric_motor_holdout.json`

This run freezes the previously selected prefix-only delta mechanism:

- Magnitude mode: `prefix_hold`
- Mask mode: `all_bins`
- Mechanism: `time_smooth_3`
- Target gain: `2.0`
- Baseline gain: `0.0`
- Target future audio: metrics only; leakage flags false

The important correction is provider-neutral dedupe: `soundbible_` / `freewavesamples_` prefixes and trailing 10-hex hashes are stripped before case selection. That reduces the apparent electric-razor family to one canonical source, not a real multi-source holdout.

Top-level result:

- Status: `nonrazor_motor_tradeoff_with_razor_diagnostic_signal`
- Valid WAVs after broad exclusion/dedupe: `3677`
- Rejected WAVs: `51`
- Deduped duplicate stems: `2079`
- Withheld razor exploratory count: `1`
- Selected case occurrences / unique paths: `143` / `66`
- Selected content-hash reuse count: `77`

| Split | Family | Cases | Status | Mean corr delta | Median corr delta | Corr wins | Mean MSE delta | Outlier share |
|---|---|---:|---|---:|---:|---:|---:|---:|
| exploratory | `primary_no_razor_motor_union` | 31 | `exploratory_tradeoff_signal` | 0.000823 | -0.000074 | 0.419355 | -0.0000158 | 0.471040 |
| exploratory | `expanded_no_razor_motor_union` | 34 | `exploratory_tradeoff_signal` | 0.000877 | -0.000037 | 0.441176 | -0.0000145 | 0.435586 |
| holdout | `nonrazor_appliance_holdout` | 7 | `holdout_tradeoff_signal` | 0.003838 | 0.000221 | 0.714286 | -0.0000533 | 0.882072 |
| holdout | `razor_only_holdout` | 1 | `insufficient_valid_cases` | n/a | n/a | n/a | n/a | n/a |
| holdout | `appliance_with_razor_diagnostic` | 5 | `holdout_holdout_like_signal` | 0.186098 | 0.000221 | 0.800000 | -0.027144 | 0.998450 |
| exploratory control | `electrical_nonmotor_hum_buzz_control` | 10 | `exploratory_robust_signal` | 0.034002 | 0.000557 | 0.600000 | -0.003945 | 0.653707 |

Interpretation:

- The fixed mechanism does not currently support a broad electric-motor claim.
- The strongest non-razor structural motor row is a tiny tradeoff, not a candidate.
- The mixed razor diagnostic is dominated by one `Electric Razor.wav` case with corr delta `+0.929525`.
- The non-motor hum/buzz control is stronger than the structural motor unions, which points to a steady buzz/phase-continuity phenotype rather than motor ontology.
- Cross-group reuse is high because union, diagnostic, and control groups intentionally overlap; promotion-grade follow-up needs disjoint curated manifests, not keyword unions alone.
- This is useful as a next-manifest clue, not promotion evidence.

## Disjoint Steady-Phenotype Manifest - 2026-05-06

Run:

`C:\Users\Andrew\AppData\Local\Temp\rafa_tokenburst\audio_steady_phenotype_manifest_cuda_2026_05_06\audio_steady_phenotype_manifest.json`

This rerun keeps the same frozen mechanism row but allocates each selected WAV to one group only, with provider-neutral stem dedupe and selected content-hash uniqueness.

Top-level result:

- Status: `disjoint_phenotype_tradeoff_only`
- Selected case occurrences / unique paths / unique hashes: `60` / `60` / `60`
- Selection reuse rejections: `1`
- Target leakage detected: `False`

| Group | Role | Cases | Status | Mean corr delta | Median corr delta | Corr wins | Mean MSE delta | Outlier share |
|---|---|---:|---|---:|---:|---:|---:|---:|
| `steady_buzz_nonmotor` | steady buzz control | 12 | `steady_buzz_control_tradeoff_signal` | 0.026831 | -0.000049 | 0.416667 | -0.003206 | 0.670547 |
| `small_appliance_motor_nonrazor` | no-razor motor | 7 | `no_razor_motor_tradeoff_signal` | 0.004305 | 0.000221 | 0.714286 | -0.0000535 | 0.792661 |
| `rotary_tool_motor_nonrazor` | no-razor motor | 11 | `no_razor_motor_tradeoff_signal` | 0.000488 | -0.000393 | 0.363636 | -0.0000214 | 0.475392 |
| `saw_chain_tool_motor` | tool/saw control | 8 | `tool_motor_or_saw_control_not_improved` | -0.000931 | -0.000907 | 0.375000 | 0.00000198 | 0.612854 |
| `combustion_engine_control` | combustion control | 12 | `combustion_control_tradeoff_signal` | 0.000810 | 0.001025 | 0.666667 | -0.0000369 | 0.310704 |
| `buzzy_synth_control` | synth control | 5 | `synth_control_not_improved` | -0.066417 | -0.123877 | 0.200000 | 0.012287 | 1.000000 |
| `razor_single_source_probe` | single-source probe | 1 | `single_source_probe_signal` | 0.929525 | 0.929525 | 1.000000 | -0.135716 | 1.000000 |

Interpretation:

- The disjoint manifest weakens the earlier audio clue rather than promoting it.
- Steady buzz remains the strongest multi-case non-razor phenotype by mean corr delta, but the negative median, low win fraction, and high outlier share keep it tradeoff-only.
- No-razor motors are weaker than steady buzz and do not currently support a motor-ontology claim.
- The synth control failing is useful: the mechanism is not merely helping every buzzy filename, but the current positive signal is still too sparse and outlier-driven.

## Phase Phenotype Diagnostics - 2026-05-06

Run:

`C:\Users\Andrew\AppData\Local\Temp\rafa_tokenburst\audio_phase_phenotype_diagnostics_cuda_2026_05_06\audio_phase_phenotype_diagnostics.json`

This is a post-hoc diagnostic over the frozen disjoint manifest. It uses target metrics only to correlate already-observed gains with prefix/mechanism features; it does not select a new mechanism.

Top-level result:

- Status: `posthoc_diagnostic_only`
- Cases: `60`
- Single-source razor excluded from the main feature-correlation read: yes

Strongest non-single-source correlations with corr delta:

| Feature | Pearson with corr delta | Pearson with MSE delta | Interpretation |
|---|---:|---:|---|
| `gain0_target_corr` | -0.686859 | 0.607852 | Biggest gains happen where the gain-0 carrier already matches target badly. |
| `vs_gain0_mse` | -0.498847 | 0.683344 | Larger deviation from the gain-0 carrier is tied to MSE worsening risk. |
| `reentry_delta` | -0.451581 | 0.549363 | Reentry changes are not cleanly beneficial. |
| `prefix_phase_velocity_coherence` | 0.265985 | -0.103703 | Weak positive hint, not enough for a phenotype claim. |
| `prefix_spectral_flux_mean` | -0.258010 | 0.163485 | Slight preference for lower-flux/stationary prefixes, still weak. |
| `mean_abs_shaped_delta` | -0.223002 | 0.300470 | Bigger shaped deltas do not explain the wins; they lean negative. |

Interpretation:

- Current fixed-mechanism wins are best explained as rescue attempts when the baseline seed is bad, not as evidence that RAFA has isolated a robust steady-motor/steady-buzz law.
- The weak positive phase-velocity/stationarity hints are useful only as predeclared follow-up variables.
- The next audio test must stratify by gain-0 baseline correlation before claiming a mechanism or phenotype effect.

## Internal Phase-Law Scout - 2026-05-07

Report:

`D:\RAFA\docs\reports\CIRCLEWORLD_INTERNAL_PHASE_LAW_SCOUT_2026-05-07.md`

Key artifacts:

- Config variants: `C:\Users\Andrew\AppData\Local\Temp\rafa_tokenburst\internal_phase_law_configs_2026_05_07\internal_phase_law_config_manifest.json`
- Full raw scout root: `C:\Users\Andrew\AppData\Local\Temp\rafa_tokenburst\internal_phase_law_raw_scout_full_2026_05_07\`
- Absolute compare without single-source probe: `C:\Users\Andrew\AppData\Local\Temp\rafa_tokenburst\internal_phase_law_raw_scout_full_compare_no_single_source_2026_05_07\audio_lockbox_result_compare.json`
- Direct variant compare without single-source probe: `C:\Users\Andrew\AppData\Local\Temp\rafa_tokenburst\internal_phase_law_variant_compare_no_single_source_2026_05_07\internal_phase_law_variant_compare.json`

Result:

- Direct variant-vs-no-op effect is real but small: best overall variant `velocity025` has mean corr delta `+0.001202` against no-op, and best matched row `velocity025/prefix_hold/all_bins/raw/gain=2.0` has mean corr delta `+0.003734`.
- Absolute lockbox read is still weak: no-single-source overall mean corr delta is `-0.000796`, and the best absolute row is only `+0.000194` with `bad_baseline_rescue_dominated` status.

Interpretation:

- Internal phase-law coefficients are now isolated as causal recurrence variables.
- Current variants are not audio promotion evidence.
- The next audio objective should require both direct variant improvement over no-op and absolute improvement over gain-0/copyphase carrier by baseline-correlation bin.

## Baseline-Stratified Audio Diagnostics - 2026-05-06

Run:

`C:\Users\Andrew\AppData\Local\Temp\rafa_tokenburst\audio_baseline_stratified_diagnostics_cuda_2026_05_06\audio_baseline_stratified_diagnostics.json`

This diagnostic bins the frozen disjoint-manifest cases by the gain-0 carrier's correlation with the target future. The single-source razor probe is excluded so it cannot dominate the read.

Top-level result:

- Status: `bad_baseline_rescue_dominated`
- Cases: `59`
- Single-source razor excluded: yes

| Bin | Cases | Mean gain-0 corr | Mean corr delta | Median corr delta | Corr wins | Mean MSE delta | Outlier share |
|---|---:|---:|---:|---:|---:|---:|---:|
| `bad_baseline` | 9 | -0.205587 | 0.096760 | 0.000188 | 0.555556 | -0.007099 | 0.603996 |
| `weak_baseline` | 40 | 0.002520 | 0.000396 | 0.000170 | 0.525000 | 0.000002 | 0.185901 |
| `moderate_baseline` | 8 | 0.092916 | -0.036125 | -0.003205 | 0.250000 | -0.000054 | 0.553115 |
| `good_baseline` | 2 | 0.328848 | -0.283350 | -0.283350 | 0.500000 | 0.043072 | 1.000000 |

Interpretation:

- The fixed RAFA phase-writeback mechanism is currently a bad-baseline rescue effect, not a robust continuation law.
- The mechanism's apparent gains shrink to near zero in weak-baseline cases and turn negative once the legal phase carrier is already moderately or strongly aligned.
- This downgrades the audio claim to `testable_weak_not_proven` and makes any next audio claim require a baseline-balanced, predeclared holdout.

## Predeclared Audio Lockbox v1 - 2026-05-06

Artifacts:

- Manifest: `C:\Users\Andrew\AppData\Local\Temp\rafa_tokenburst\audio_predeclared_lockbox_v1_2026_05_06\audio_predeclared_lockbox_manifest.json`
- Cases JSON: `C:\Users\Andrew\AppData\Local\Temp\rafa_tokenburst\audio_predeclared_lockbox_v1_2026_05_06\audio_predeclared_lockbox_cases.json`
- Probe: `C:\Users\Andrew\AppData\Local\Temp\rafa_tokenburst\audio_predeclared_lockbox_probe_v1_cuda_2026_05_06\audio_delta_mechanism_probe.json`
- Compare: `C:\Users\Andrew\AppData\Local\Temp\rafa_tokenburst\audio_predeclared_lockbox_compare_v1_2026_05_06\audio_lockbox_result_compare.json`
- Compare without single-source probe: `C:\Users\Andrew\AppData\Local\Temp\rafa_tokenburst\audio_predeclared_lockbox_compare_v1_no_single_source_2026_05_06\audio_lockbox_result_compare.json`

This lockbox freezes case selection before target metrics:

- Filename-derived group selection only.
- RIFF-readable `.wav` validation during scan.
- Disjoint selected paths, content hashes, and provider-neutral stem keys.
- No target future metrics, baseline metrics, correlation metrics, or outcome metrics in selection.
- Frozen row: `prefix_hold` / `all_bins` / `time_smooth_3`, baseline gain `0.0`, target gain `2.0`.

Top-level result:

- Manifest status: `lockbox_manifest_ready`
- Selected cases: `62`
- Probe status: `mechanism_tradeoff_signal_only`
- Compare status: `tradeoff_signal_only`
- Future target magnitude/phase reused: `False` / `False`

| Read | Value |
|---|---:|
| Overall mean corr delta | 0.011134 |
| Overall median corr delta | -0.000039 |
| Corr win fraction | 0.451613 |
| Mean MSE delta | -0.001240 |
| Positive outlier share | 0.561650 |
| Min leave-one-out corr delta | -0.004088 |
| Non-bad weighted mean corr delta | -0.015601 |

Removing the deliberate single-source razor probe makes the result negative:

- Cases: `61`
- Mean corr delta: `-0.004088`
- Median corr delta: `-0.000074`
- Corr win fraction: `0.442623`
- Mean MSE delta: `+0.001024`

Baseline-correlation bins:

| Bin | Cases | Mean gain-0 corr | Mean corr delta | Median corr delta | Corr wins | Mean MSE delta | Outlier share |
|---|---:|---:|---:|---:|---:|---:|---:|
| `bad_baseline` | 8 | -0.269750 | 0.191589 | 0.006620 | 0.750000 | -0.022472 | 0.585687 |
| `weak_baseline` | 43 | -0.001973 | -0.000123 | -0.000004 | 0.441860 | 0.000037 | 0.200121 |
| `moderate_baseline` | 9 | 0.105957 | -0.030612 | -0.003092 | 0.222222 | 0.001692 | 0.553115 |
| `good_baseline` | 2 | 0.357550 | -0.280812 | -0.280812 | 0.500000 | 0.043044 | 1.000000 |

Interpretation:

- The cleaned predeclared lockbox confirms the earlier baseline-stratified diagnostic instead of rescuing the audio claim.
- The fixed mechanism has a positive mean only because the bad-baseline bin is large; weak/moderate/good baseline bins are flat or negative.
- No current audio result proves semi-tokenless phase-law continuation. The next useful work is mechanism design, not more post-hoc slicing of this row.

## Exploratory Phase-Law Scout v1 - 2026-05-07

Artifacts:

- Probe: `C:\Users\Andrew\AppData\Local\Temp\rafa_tokenburst\audio_phase_law_scout_v1_cuda_2026_05_07\audio_delta_mechanism_probe.json`
- No-single-source compare: `C:\Users\Andrew\AppData\Local\Temp\rafa_tokenburst\audio_phase_law_scout_compare_v1_no_single_source_2026_05_07\audio_lockbox_result_compare.json`

This scout adds prefix-only candidate mechanisms:

- `stable_time_smooth_3`
- `stable_energy_time_smooth_3`
- `stable_velocity_time_smooth_3`
- `stable_energy_velocity_time_smooth_3`
- `stable_softclip_time_smooth_3`
- `stable_energy_velocity_softclip_time_smooth_3`

The mechanism family uses prefix magnitude stationarity, prefix energy, prefix phase-velocity coherence, and optional phase-delta soft clipping. It does not read future target magnitude or future target phase.

No-single-source comparison:

- Status: `tradeoff_signal_only`
- Cases: `61`
- Analysis rows: `1098`
- Overall mean corr delta: `+0.001334`
- Overall median corr delta: `-0.000014`
- Overall corr win fraction: `0.410747`
- Overall mean MSE delta: `+0.000006`

Best candidate row:

| Mechanism | Gain | Status | Mean corr delta | Median corr delta | Corr wins | Mean MSE delta | Outlier share |
|---|---:|---|---:|---:|---:|---:|---:|
| `stable_velocity_time_smooth_3` | 2.0 | `bad_baseline_rescue_dominated` | 0.004642 | -0.000162 | 0.393443 | -0.000193 | 0.790519 |

Interpretation:

- Prefix stability/energy/velocity gating improves the failure mode only superficially.
- The best row remains bad-baseline-rescue dominated, with negative median, low win fraction, and high outlier pressure.
- The next real audio intervention must alter the Circleworld phase law itself, not only post-shape its deltas with prefix gates.

## Heldout Runtime Evidence

All rows below use the same seeded branch-active/naked recovery path under:

`C:\Users\Andrew\AppData\Local\Temp\rafa_tokenburst\naked_phase_nested_recovery_v1_smoke`

| Run | Overall phase-only branch | Overall parent branch | Naked phase-only branch | Naked final phase peak | Naked live slot-2 fraction | Naked support writeback | Mean major gain |
|---|---:|---:|---:|---:|---:|---:|---:|
| support writeback only | 0.030019 | 0.009629 | 0.000000 | 0.024017 | 0.006238 | 0.003521 | 0.062138 |
| spawn phase bias | 0.030005 | 0.010356 | 0.000000 | 0.072809 | 0.218750 | 0.016560 | 0.062136 |
| dominant-q operator | 0.029918 | 0.009973 | 0.000000 | 0.072861 | 0.218750 | 0.016504 | 0.062132 |
| return operator | 0.030597 | 0.011022 | 0.000000 | 0.073267 | 0.218750 | 0.016070 | 0.062159 |
| phase floor | 0.046438 | 0.019171 | 0.000000 | 0.089580 | 0.218750 | 0.016102 | 0.041961 |
| final floor | 0.046135 | 0.021735 | 0.003089 | 0.146823 | 0.218750 | 0.016103 | 0.022065 |

Interpretation:

- Support writeback makes child identity parent-visible but is not enough for naked phase-only branching.
- Spawn/operator bias makes slot 2 live in naked states but remains mostly decorative.
- The phase-floor return law is the first intervention that makes naked `phase_only_real_branch_fraction` nonzero.
- The gain is not clean: final floor substantially reduces `mean_major_gain`, so this is not promotion-grade.

Final floor heldout details:

- Overall `mean_real_branch_fraction`: `0.555556`
- Overall `mean_parent_real_branch_fraction`: `0.021735`
- Overall `phase_only_real_branch_fraction`: `0.046135`
- Overall `mean_child_support_writeback_mass`: `0.032617`
- Overall `mean_child_parent_divergence`: `0.194458`
- Naked `mean_child_real_branch_fraction`: `0.333333`
- Naked `phase_only_real_branch_fraction`: `0.003089`
- Naked `decorative_slot2_low_phase_fraction`: `0.215661`

The headline branch fraction remains child-readiness dominated. Parent-visible phase branch is real but small.

## Child Writeback Ablation v1 - Superseded/Confounded

Run:

`C:\Users\Andrew\AppData\Local\Temp\rafa_tokenburst\child_writeback_ablation_cuda_2026_05_06`

Status: superseded by the corrected ablations below. In this older support-only isolation, raw child-to-parent phase delta was not fully disabled, so the support-only row is confounded and must not be used as the current-cycle causal read on support/logit/q-trace writeback alone.

| Variant | Overall phase-only | Parent branch | Child branch | Support writeback | Major gain | Naked phase-only | Naked peak | Naked decorative |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `baseline_current` | 0.046135 | 0.021735 | 0.555556 | 0.032617 | 0.022065 | 0.003089 | 0.146823 | 0.215661 |
| `support_only_operator_floor_off` | 0.052292 | 0.024070 | 0.555556 | 0.033701 | 0.025455 | 0.000000 | 0.094464 | 0.218750 |
| `operator_floor_only_support_off` | 0.043914 | 0.007543 | 0.555556 | 0.000000 | 0.022044 | 0.000000 | 0.013605 | 0.006056 |
| `support_operator_no_floor` | 0.052474 | 0.022556 | 0.555556 | 0.032634 | 0.021695 | 0.000000 | 0.096977 | 0.218750 |
| `support_operator_floor_low` | 0.048376 | 0.021008 | 0.555556 | 0.032562 | 0.021814 | 0.000000 | 0.104669 | 0.218750 |
| `support_operator_floor_default` | 0.046135 | 0.021735 | 0.555556 | 0.032617 | 0.022065 | 0.003089 | 0.146823 | 0.215661 |
| `support_operator_floor_high` | 0.049365 | 0.025954 | 0.555556 | 0.032780 | 0.022309 | 0.006541 | 0.199482 | 0.212209 |

Superseded interpretation:

- The v1 support-only row is confounded because raw phase delta was still present.
- The v1 table remains useful only as historical context for the floor sweep shape.
- Do not infer that support/logit/q-trace writeback alone produces phase-only branching from this v1 table.

## Corrected Child Writeback Ablation v4 - Seed-Fallback Fixed - 2026-05-06

Run:

`C:\Users\Andrew\AppData\Local\Temp\rafa_tokenburst\child_writeback_ablation_v4_seed_fallback_cuda_2026_05_06`

This v4 run repeats the child-writeback split after repairing both packet-seed unit-phasor failures: temporal mean normalization and near-cancellation fallback to the local center phasor. It is the current-cycle ablation to use.

| Variant | Overall phase-only | Parent branch | Child branch | Gate mass | Phase delta | Support writeback | Logit writeback | Major gain | Naked phase-only | Naked peak | Naked decorative |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `baseline_current` | 0.068960 | 0.037728 | 0.555556 | 0.136087 | 0.088659 | 0.046359 | 0.021077 | 0.030880 | 0.007187 | 0.196094 | 0.284480 |
| `support_only_phase_operator_floor_off` | 0.000000 | 0.000000 | 0.555556 | 0.138234 | 0.000000 | 0.045840 | 0.021663 | 0.005761 | 0.000000 | 0.059368 | 0.291667 |
| `raw_phase_only_support_operator_floor_off` | 0.069821 | 0.013440 | 0.555556 | 0.140959 | 0.080881 | 0.000000 | 0.000000 | 0.036319 | 0.000000 | 0.036815 | 0.008398 |
| `operator_floor_only_support_off` | 0.004908 | 0.000924 | 0.555556 | 0.133274 | 0.012773 | 0.000000 | 0.000000 | 0.005709 | 0.000000 | 0.073430 | 0.007752 |
| `support_operator_no_floor` | 0.072656 | 0.035512 | 0.555556 | 0.136021 | 0.080971 | 0.046380 | 0.021069 | 0.030087 | 0.000000 | 0.095469 | 0.291667 |
| `support_operator_floor_low` | 0.069238 | 0.033520 | 0.555556 | 0.135971 | 0.082158 | 0.046266 | 0.021020 | 0.030254 | 0.000000 | 0.103890 | 0.291667 |
| `support_operator_floor_fixed_2_10` | 0.066493 | 0.034507 | 0.555556 | 0.135968 | 0.085435 | 0.046209 | 0.021001 | 0.030576 | 0.002907 | 0.148426 | 0.288760 |
| `support_operator_floor_high` | 0.068960 | 0.037728 | 0.555556 | 0.136087 | 0.088659 | 0.046359 | 0.021077 | 0.030880 | 0.007187 | 0.196094 | 0.284480 |

Current-cycle interpretation:

- Pure support/logit/q-trace writeback with phase/operator/floor off does not produce overall phase-only branch or parent branch in v4.
- Raw phase-delta writeback can raise overall phase-only branch, but it does not solve naked phase-only branch.
- Operator/floor without support writeback is also insufficient.
- The best current heldout variant remains `support_operator_floor_high`, with naked phase-only `0.007187`, but naked decorative pressure remains high at `0.284480`.
- The v4 seed-fallback result moves the bottleneck away from simple writeback amplitude and toward child coherence and qualified carry.

## Nested Evidence - v1 Final-Floor

Nested run:

`C:\Users\Andrew\AppData\Local\Temp\rafa_tokenburst\naked_phase_nested_recovery_v1_smoke\nested_runtime_patch_finalfloor_2026_05_06\nested_commitment_report.json`

Top-level result:

- `overall_read`: `nested_commitment_not_yet_established`
- `verdict_counts`: `{"mixed_or_inconclusive": 3}`
- `num_cases`: `3`
- `num_eligible_nested_cases`: `3`
- `num_live_child_start_cases`: `3`
- `num_selected_fork_has_live_child_cases`: `3`
- `mean_selected_fork_has_live_child`: `1.0`
- `mean_branch_identity_carry`: `1.0`
- `mean_branch_identity_qualified_carry`: `0.045833`
- `mean_readout_sibling_response`: `0.296093`
- `mean_nested_sibling_fraction`: `0.0`
- `mean_child_active_fraction`: `0.0`
- `mean_child_survival_signal`: `0.0`
- `mean_child_meso_response`: `0.0`
- `mean_world_jump_penalty`: `0.0`

Representative child branches:

| Branch | Label | Readout response | Coarse env corr | Fine q corr | Same-child carry | Support mean | Coherence mean | Qualified carry |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `child_dominant_continuation_shift` | `ambiguous_middle` | 0.376691 to 0.378493 | about 0.9916 to 0.9918 | about 0.999120 | 1.0 | 0.218750 | about 0.096980 | 0.0 |
| `child_mode1_replace_85_shift` | `ambiguous_middle` | 0.199057 to 0.199650 | about 0.99964 | about 0.999923 | 1.0 | 0.218750 | about 0.096980 | 0.0 |
| `child_writeback_gate_shift` | `ambiguous_middle` | 0.192433 to 0.192656 | about 0.99965 | about 0.999950 | 1.0 | 0.218750 | about 0.096980 | 0.0 |

Interpretation:

- Same child IDs persist through the nested branch interval.
- Readout sibling response is nonzero.
- Coarse world and q profile remain preserved.
- The active-child survival/writeback signal during continuation is zero.
- Coherence is just under the current qualified threshold in key branches.
- Therefore this is not `nested_sibling`; it is a measurable pre-sibling state.

## Nested Evidence - High-Floor Current Cycle

Nested run:

`C:\Users\Andrew\AppData\Local\Temp\rafa_tokenburst\nested_child_writeback_floor_high_2026_05_06\nested_commitment_report.json`

Top-level result:

- `overall_read`: `nested_commitment_not_yet_established`
- `verdict_counts`: `{"mixed_or_inconclusive": 3}`
- `num_cases`: `3`
- `num_eligible_nested_cases`: `3`
- `num_live_child_start_cases`: `3`
- `num_selected_fork_has_live_child_cases`: `3`
- `num_selected_live_child_eligible_cases`: `3`
- `num_selected_phase_only_eligible_cases`: `0`
- `mean_selected_fork_has_live_child`: `1.0`
- `mean_selected_live_child_eligible`: `1.0`
- `mean_selected_phase_only_eligible`: `0.0`
- `mean_child_response_score`: `8.981092`
- `max_child_response_score`: `13.529089`
- `mean_readout_sibling_response`: `0.303228`
- `mean_nested_sibling_fraction`: `0.0`
- `mean_child_active_fraction`: `0.0`
- `mean_child_survival_signal`: `0.0`
- `mean_branch_identity_carry`: `1.0`
- `mean_branch_identity_qualified_carry`: `0.045833`
- `mean_branch_identity_budget_retained`: `0.727110`
- `mean_branch_identity_parent_div`: `0.086083`
- `mean_branch_identity_sibling_div`: `0.163704`
- `mean_coarse_preservation`: `0.998874`
- `mean_continuity_delta_magnitude`: `0.002807`
- `max_continuity_delta_magnitude`: `0.022330`

Representative high-floor child branches:

| Branch | Label | Readout response | Coarse env corr | Fine q corr | Same-child carry | Support mean | Coherence mean | Budget retained | Qualified carry |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `child_dominant_continuation_shift` | `ambiguous_middle` | 0.389167 | 0.990772 | 0.999094 | 1.000000 | 0.218750 | 0.096980 | 0.850081 | 0.000000 |
| `child_mode1_replace_85_shift` | `ambiguous_middle` | 0.204465 | 0.999569 | 0.999917 | 1.000000 | 0.218750 | 0.096980 | 0.850081 | 0.000000 |
| `child_writeback_gate_shift` | `ambiguous_middle` | 0.201710 | 0.999555 | 0.999933 | 1.000000 | 0.218750 | 0.096980 | 0.895939 | 0.000000 |

Interpretation:

- High floor keeps child IDs carried across the branch interval and produces nonzero readout response.
- The assay still has zero nested sibling fraction, zero active-child survival signal, and zero qualified carry in representative branches.
- The coherence mean remains just under the `0.1` qualification threshold in these branches, so the bottleneck is child coherence/qualified carry rather than writeback amplitude alone.

## Phase Gauge Invariance - High-Floor Current Cycle

Run:

`C:\Users\Andrew\AppData\Local\Temp\rafa_tokenburst\phase_gauge_floor_high_cuda_2026_05_06\phase_gauge_invariance_report.json`

Configuration:

- Device: `cuda`
- Time steps: `128`
- Seed plan: 3 synthetic seeds plus 3 naked-RAFA seeds
- Config: `support_operator_floor_high`

| Transform | Cases | Max q diff | Max arc diff | Max branch diff | Max q-mass L1 |
|---|---:|---:|---:|---:|---:|
| `global_rotation_pi_over_7` | 6 | 1.788e-07 | 1.788e-07 | 5.364e-07 | 2.407e-07 |
| `global_rotation_pi_over_2` | 6 | 0.000e+00 | 0.000e+00 | 0.000e+00 | 0.000e+00 |
| `global_rotation_minus_2pi_over_3` | 6 | 2.384e-07 | 1.192e-07 | 3.576e-07 | 1.602e-07 |
| `phase_preserving_rescale_half` | 6 | 0.000e+00 | 0.000e+00 | 0.000e+00 | 0.000e+00 |
| `phase_preserving_rescale_double` | 6 | 0.000e+00 | 0.000e+00 | 0.000e+00 | 0.000e+00 |
| `phase_preserving_freq_time_taper` | 6 | 1.192e-07 | 1.192e-07 | 2.980e-07 | 1.700e-07 |

Interpretation:

- The high-floor current-cycle metrics are invariant to global phase rotations to numerical precision.
- The rescale/taper rows are normalized phase-preserving identity controls; they are useful regression checks, but they do not prove general amplitude-gauge invariance.
- This supports treating the reported phase/writeback changes as relative-phase behavior, not absolute gauge artifacts, within the measured internal metrics.

## Unit-Phasor / Phase-Only Contract Audit - Current Cycle

Runs:

- Pre-fix failure: `C:\Users\Andrew\AppData\Local\Temp\rafa_tokenburst\unit_phasor_contract_prefix_fail_cuda_2026_05_06\unit_phasor_contract_report.json`
- Post-fix heldout pass: `C:\Users\Andrew\AppData\Local\Temp\rafa_tokenburst\unit_phasor_contract_cuda_2026_05_06\unit_phasor_contract_report.json`
- Post-fix all-surfaces pass: `C:\Users\Andrew\AppData\Local\Temp\rafa_tokenburst\unit_phasor_contract_all_surfaces_cuda_2026_05_06\unit_phasor_contract_report.json`

Code:

- Audit harness: `D:\RAFA\runtimes\circleworld_proto\test_unit_phasor_contract.py`
- Runtime repair: promoted packet `seed_state` is now normalized after temporal phasor averaging in `D:\RAFA\lineages\04_positive_replacement\circleworld.py`.

All-surfaces summary:

| Surface | Cases | Passing | Max norm error | Mean norm error | Min finite | Min raw mix | Status |
|---|---:|---:|---:|---:|---:|---:|---|
| `heldout` | 6 | 6 | 1.788e-07 | 9.317e-09 | 1.0 | 0.998114 | pass |
| `nested` | 3 | 3 | 1.788e-07 | 4.551e-09 | 1.0 | 0.997107 | pass |
| `export` | 6 | 6 | 1.788e-07 | 9.185e-09 | 1.0 | 0.998114 | pass |
| `direct_export` | 5 | 5 | 1.192e-07 | 8.453e-09 | 1.0 | 0.999489 | pass |
| `audio_continuation` | 5 | 5 | 1.192e-07 | 7.803e-09 | 1.0 | 0.999105 | pass |

All-surfaces boundary summary:

| Boundary | Checked tensors | Max norm error | Mean norm error | Min finite | Status |
|---|---:|---:|---:|---:|---|
| `input` | 6 | 1.788e-07 | 1.549e-08 | 1.0 | pass |
| `phase_state` | 6 | 1.192e-07 | 5.748e-09 | 1.0 | pass |
| `mixed_phase_state` | 6 | 1.192e-07 | 5.748e-09 | 1.0 | pass |
| `multimode_state` | 12 | 1.192e-07 | 3.114e-09 | 1.0 | pass |
| `multimode_history` | 36 | 1.192e-07 | 2.830e-09 | 1.0 | pass |
| `child_worlds` | 162 | 5.960e-08 | 1.648e-10 | 1.0 | pass |
| `packets` | 288 | 1.192e-07 | 1.634e-08 | 1.0 | pass |
| `nested` | 1521 | 1.788e-07 | 4.927e-09 | 1.0 | pass |
| `nested_branch_start` | 144 | 1.192e-07 | 2.157e-09 | 1.0 | pass |
| `nested_pre_unroll` | 36 | 1.192e-07 | 1.940e-09 | 1.0 | pass |
| `nested_identity_probe` | 48 | 1.192e-07 | 1.766e-09 | 1.0 | pass |
| `export` | 522 | 1.788e-07 | 9.725e-09 | 1.0 | pass |
| `export_blend` | 6 | 5.960e-08 | 2.567e-11 | 1.0 | pass |
| `direct_export` | 279 | 1.192e-07 | 9.136e-09 | 1.0 | pass |
| `direct_export_blend` | 5 | 1.192e-07 | 7.360e-09 | 1.0 | pass |
| `audio_continuation` | 270 | 1.192e-07 | 8.412e-09 | 1.0 | pass |
| `audio_continuation_final` | 5 | 1.192e-07 | 7.148e-09 | 1.0 | pass |

Raw multimode mixture cancellation check:

- States checked: `304`
- Minimum raw mixture norm before readout normalization: `0.997107`
- Max fraction below `1e-4`: `0.0`
- Status: `pass`

Negative-control detection:

- The all-surfaces run also injected a deliberate phasor scale of `0.83` on `negative_control.synthetic_5100.scaled_phase_state.phase_state`.
- Expected status: `fail`
- Detected: `true`
- Max norm error: `0.170000076`
- Norm violations: `1`

Failure caught before repair:

- Promoted packet `seed_state` was built by averaging phasors over a support window without renormalization.
- The pre-fix audit found max norm error `0.180798`, with worst paths under packet/law-packet `seed_state`.
- This was a real hidden magnitude channel because packet seeds were used in `_phase_alignment` during child-spawn divergence checks.

Post-repair interpretation:

- The current childworld runtime preserves unit phasors across input, mode update, child spawn/evolve, child writeback, mixed readout, history snapshots, and promoted packet seeds to numerical precision.
- The all-surfaces audit extends this to seeded nested branch starts, child pre-unroll states, identity-probe readouts, and the export-like phasor blend path.
- Direct WAV/STFT export instrumentation passes on the five standard real-anchor WAVs, including final blend phasors and finite inverse-STFT output.
- Audio-continuation tensor instrumentation passes on the same five anchors, covering prefix phase extrapolation, Circleworld continuation, final phase canvas, and finite inverse-STFT future output.
- The negative control confirms the audit would fail if a hidden magnitude channel re-entered a phasor tensor.
- This strengthens the phase-only runtime contract, but it does not by itself prove audio viability, nested sibling ontology, or Ramanujan-specific causality.
- Supplemental broad-WAV direct export pass: `C:\Users\Andrew\AppData\Local\Temp\rafa_tokenburst\unit_phasor_broad_direct_export_cuda_2026_05_06\unit_phasor_contract_report.json`
- Supplemental broad-WAV audio-continuation pass: `C:\Users\Andrew\AppData\Local\Temp\rafa_tokenburst\unit_phasor_broad_audio_continuation_cuda_2026_05_06\unit_phasor_contract_report.json`
- The 18-WAV direct export audit passes 739 tensors with max norm error `1.192e-07`; the 18-WAV audio-continuation audit passes 737 tensors with max norm error `1.192e-07`.
- Broad 18-WAV 4s benchmark: `C:\Users\Andrew\AppData\Local\Temp\rafa_tokenburst\benchmark_broad18_circleworld_4s_cuda_2026_05_06\benchmark_summary.json`
- Broad 18-WAV 4s benchmark result: mean corr `0.963021`, mean MAE `0.014660`, continuity delta band `near_identity`, continuity max abs mean delta `0.003492`, recurrence score delta `0.000406`.
- Broad no-future continuation against flat/prefix/copy-last baselines is now available under the current audio continuation artifacts below.
- Broad carrier-lock compare: `C:\Users\Andrew\AppData\Local\Temp\rafa_tokenburst\audio_continuation_method_compare_broad18_cuda_2026_05_06\audio_continuation_method_compare.json`
- Broad carrier-lock result: status `carrier_locked`; prefix-hold Circleworld vs carrier corr `0.986092`; flat Circleworld vs carrier corr `0.985675`; Circleworld loses to best simple baseline on mean corr by `0.109320` and `0.099988` respectively.
- Broad copyphase prefix-hold continuation: `C:\Users\Andrew\AppData\Local\Temp\rafa_tokenburst\audio_continuation_broad18_copyphase_prefix_hold_cuda_2026_05_06\audio_continuation_summary.json`
- Broad copyphase flat continuation: `C:\Users\Andrew\AppData\Local\Temp\rafa_tokenburst\audio_continuation_broad18_copyphase_flat_cuda_2026_05_06\audio_continuation_summary.json`
- Broad copyphase carrier compare: `C:\Users\Andrew\AppData\Local\Temp\rafa_tokenburst\audio_continuation_method_compare_copyphase_broad18_cuda_2026_05_06\audio_continuation_method_compare.json`
- Broad copyphase result: no future target magnitude or phase reused; prefix-hold Circleworld corr `0.088204` versus copy-last waveform baseline corr `0.090227`; copyphase method compare status `needs_review`; Circleworld remains below the best simple baseline by mean corr `-0.015584` on prefix-hold and `-0.084159` on flat.
- Broad phase-influence ablation: `C:\Users\Andrew\AppData\Local\Temp\rafa_tokenburst\audio_phase_influence_broad18_cuda_2026_05_06\audio_phase_influence_ablation.json`
- Broad phase-influence result: status `phase_changes_without_target_gain`; 18 cases, 12 aggregate gain/mode rows, 216 case-level rows; mean future abs phase delta `0.077543`; gain `2.0` moves away from the gain-0 carrier but does not produce material target improvement.
- Broad phase-seed ablation: `C:\Users\Andrew\AppData\Local\Temp\rafa_tokenburst\audio_phase_seed_broad18_cuda_2026_05_06\audio_phase_seed_ablation.json`
- Broad phase-seed result: status `seed_policy_matters_circleworld_not_helpful`; best seed is `copy_last_waveform_phase` with `prefix_hold`, seed corr `0.087144`, Circleworld corr `0.088204`, Circleworld-minus-seed corr `0.001059`, max seed corr spread `0.097656`, max seed MSE spread `0.015938`.
- Broad weighted/fit phase-seed expansion: `C:\Users\Andrew\AppData\Local\Temp\rafa_tokenburst\audio_phase_seed_weighted_fit_broad18_cuda_2026_05_06\audio_phase_seed_ablation.json`
- Broad weighted/fit phase-seed result: status `seed_policy_matters_circleworld_not_helpful`; `copy_last_waveform_phase` + `prefix_hold` remains best, seed corr `0.087144`, Circleworld corr `0.088204`, Circleworld-minus-seed corr `0.001059`, max seed corr spread `0.093912`, max seed MSE spread `0.017936`.
- Broad Circleworld delta-over-copyphase probe: `C:\Users\Andrew\AppData\Local\Temp\rafa_tokenburst\audio_circle_delta_probe_broad18_cuda_2026_05_06\audio_circle_delta_probe.json`
- Broad Circleworld delta-over-copyphase result: status `needs_review` in the raw probe and `delta_tradeoff_not_proof` in the claim evidence rollup. Best corr delta is `+0.002819` at `prefix_hold/all_bins/gain=2.0`; best MSE delta is `-0.000115` at `prefix_hold/high_energy_bins/gain=2.0`; no future target magnitude or phase is reused.
- Broad fixed mechanism delta probe: `C:\Users\Andrew\AppData\Local\Temp\rafa_tokenburst\audio_delta_mechanism_probe_broad18_cuda_2026_05_06\audio_delta_mechanism_probe.json`
- Broad fixed mechanism delta result: raw status `mechanism_tradeoff_signal_only`, rollup status `mechanism_tradeoff_not_proof`. Best fixed prefix-only mechanism is `time_smooth_3` on `prefix_hold/all_bins` at gain `2.0`, with corr delta `+0.005640`, MSE delta `-0.000160`, corr win fraction `0.666667`, and no future target magnitude or phase reuse. This remains below the strict `+0.02` mechanism-candidate bar.
- Predeclared fixed mechanism lockbox cases: `C:\Users\Andrew\AppData\Local\Temp\rafa_tokenburst\audio_delta_mechanism_lockbox_cases_18_2026_05_06.json`
- Predeclared fixed mechanism lockbox probe: `C:\Users\Andrew\AppData\Local\Temp\rafa_tokenburst\audio_delta_mechanism_probe_lockbox18_cuda_2026_05_06\audio_delta_mechanism_probe.json`
- Predeclared fixed mechanism lockbox result: status `mechanism_tradeoff_not_proof`. The broad best mechanism (`time_smooth_3`, `prefix_hold/all_bins`, gain `2.0`) gives mean corr delta `+0.056605` and MSE delta `-0.007520` on 18 unused PCM WAV cases, but median corr delta is only `+0.000411`, corr win fraction is `0.611111`, and the mean is dominated by one `electric_razor` outlier.
- Predeclared audio lockbox v1 manifest: `C:\Users\Andrew\AppData\Local\Temp\rafa_tokenburst\audio_predeclared_lockbox_v1_2026_05_06\audio_predeclared_lockbox_manifest.json`
- Predeclared audio lockbox v1 probe: `C:\Users\Andrew\AppData\Local\Temp\rafa_tokenburst\audio_predeclared_lockbox_probe_v1_cuda_2026_05_06\audio_delta_mechanism_probe.json`
- Predeclared audio lockbox v1 compare: `C:\Users\Andrew\AppData\Local\Temp\rafa_tokenburst\audio_predeclared_lockbox_compare_v1_2026_05_06\audio_lockbox_result_compare.json`
- Predeclared audio lockbox v1 no-single-source compare: `C:\Users\Andrew\AppData\Local\Temp\rafa_tokenburst\audio_predeclared_lockbox_compare_v1_no_single_source_2026_05_06\audio_lockbox_result_compare.json`
- Predeclared audio lockbox v1 result: `62` unique, RIFF-readable cases; compare status `tradeoff_signal_only`; mean corr delta `+0.011134`, median `-0.000039`, corr win fraction `0.451613`, non-bad weighted mean corr delta `-0.015601`, and positive outlier share `0.561650`. Removing the single-source probe gives mean corr delta `-0.004088` and mean MSE delta `+0.001024`.
- Exploratory phase-law scout v1 probe: `C:\Users\Andrew\AppData\Local\Temp\rafa_tokenburst\audio_phase_law_scout_v1_cuda_2026_05_07\audio_delta_mechanism_probe.json`
- Exploratory phase-law scout v1 no-single-source compare: `C:\Users\Andrew\AppData\Local\Temp\rafa_tokenburst\audio_phase_law_scout_compare_v1_no_single_source_2026_05_07\audio_lockbox_result_compare.json`
- Exploratory phase-law scout v1 result: best no-single-source candidate `stable_velocity_time_smooth_3` at gain `2.0` is still `bad_baseline_rescue_dominated`, with mean corr delta `+0.004642`, median `-0.000162`, corr wins `0.393443`, and outlier share `0.790519`.
- Internal phase-law scout report: `D:\RAFA\docs\reports\CIRCLEWORLD_INTERNAL_PHASE_LAW_SCOUT_2026-05-07.md`
- Internal phase-law raw absolute compare: `C:\Users\Andrew\AppData\Local\Temp\rafa_tokenburst\internal_phase_law_raw_scout_full_compare_no_single_source_2026_05_07\audio_lockbox_result_compare.json`
- Internal phase-law direct variant compare: `C:\Users\Andrew\AppData\Local\Temp\rafa_tokenburst\internal_phase_law_variant_compare_no_single_source_2026_05_07\internal_phase_law_variant_compare.json`
- Internal phase-law result: direct recurrence effect is positive but small (`velocity025` overall mean corr delta `+0.001202` vs no-op), while absolute lockbox performance remains `tradeoff_signal_only` with no-single-source mean corr delta `-0.000796`.
- Sound-family mechanism sensitivity: `C:\Users\Andrew\AppData\Local\Temp\rafa_tokenburst\audio_mechanism_family_sensitivity_broad_cuda_2026_05_06\audio_mechanism_family_sensitivity.json`
- Sound-family mechanism sensitivity result: status `family_tradeoff_signal_only`. The strongest deduped filename-derived family is `electric_motor_hum`, with mean corr delta `+0.105432`, median corr delta `+0.000110`, min leave-one-out corr delta `+0.030514`, corr win fraction `0.500000`, MSE delta `-0.014666`, and positive outlier share `0.720103`. This localizes the clue but does not create a promotion-grade audio result.
- Audio delta objective scout, floor-high base: `C:\Users\Andrew\AppData\Local\Temp\rafa_tokenburst\audio_delta_objective_scout_broad18_cuda_2026_05_06\audio_delta_objective_scout.json`
- Audio delta objective scout, floor-high result: status `tradeoff_signal_only`; best candidate `branch_aggressive_delta_high` has objective corr delta `+0.002833`, MSE delta `-0.000111`, corr win fraction `0.555556`, and does not clear the frozen `+0.01` corr / `0.60` win bar.
- Audio delta objective scout, relsig base: `C:\Users\Andrew\AppData\Local\Temp\rafa_tokenburst\audio_delta_objective_scout_relsig_broad18_cuda_2026_05_06\audio_delta_objective_scout.json`
- Audio delta objective scout, relsig result: status `no_candidate_found`; best candidate `support_operator_floor_high` has objective corr delta `+0.002037`, MSE delta `+0.000043`, and corr win fraction `0.500000`.

Branch probe alias contract:

- Artifact: `C:\Users\Andrew\AppData\Local\Temp\rafa_tokenburst\packet_alias_contract_v2_cuda_2026_05_06\packet_alias_contract_report.json`
- Passive packet applicator: input max delta `0.0`, output max delta `0.582707`, output norm error `1.192e-07`, no shared storage.
- Active packet applicator: input max delta `0.0`, output max delta `0.264845`, output norm error `5.960e-08`, no shared storage.
- Empty passive and active packet paths now clone instead of returning shared storage.
- Nested branch clone, child perturbation, and child pre-unroll rows also pass: source input delta `0.0`, no sibling mutation, no shared storage, and unit phasor output.
- This closes the nested explorer's alias warning for packet perturbation and pre-unroll paths: branch probes can now change their own output without mutating a shared baseline tensor.

## Q-Basis Ablation - Current Cycle

Run:

`C:\Users\Andrew\AppData\Local\Temp\rafa_tokenburst\q_basis_ablation_v3_seed_fallback_cuda_2026_05_06\q_basis_ablation_summary.json`

| Variant | Parent branch | Child branch | Phase-only | Naked phase-only | Support writeback | Major gain | Dominant law-q share | Relational confidence |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `ramanujan_baseline` | 0.037728 | 0.555556 | 0.068960 | 0.007187 | 0.046358 | 0.030880 | 0.655394 | 0.850303 |
| `qtrace_only_control` | 0.037800 | 0.555556 | 0.068960 | 0.007187 | 0.046358 | 0.030881 | 0.655394 | 0.850303 |
| `uniform_q_weights` | 0.032264 | 0.407407 | 0.059109 | 0.000000 | 0.032645 | 0.002962 | 0.314163 | 0.485117 |
| `shuffled_qset` | 0.033152 | 0.407407 | 0.060912 | 0.000000 | 0.036555 | 0.002288 | 0.457918 | 0.578829 |
| `compact_fourierish_control` | 0.035593 | 0.444444 | 0.065488 | 0.000000 | 0.033803 | 0.001488 | 0.279369 | 0.503628 |

Interpretation:

- `qtrace_only_control` is effectively identical to the Ramanujan baseline on the current metrics.
- Flattening weights, shuffling q labels, or compressing to the compact control still removes naked phase-only branch after the unit-phasor repair.
- q bookkeeping/basis choices matter for naked branch and benchmark structure, but this run does not prove Ramanujan-specific arithmetic because qtrace-only matches the Ramanujan baseline.

## Audio Continuation - No Future Magnitude

Runs:

- `C:\Users\Andrew\AppData\Local\Temp\rafa_tokenburst\audio_continuation_broad18_prefix_hold_cuda_2026_05_06\audio_continuation_summary.json`
- `C:\Users\Andrew\AppData\Local\Temp\rafa_tokenburst\audio_continuation_broad18_flat_cuda_2026_05_06\audio_continuation_summary.json`
- `C:\Users\Andrew\AppData\Local\Temp\rafa_tokenburst\audio_continuation_broad18_copyphase_prefix_hold_cuda_2026_05_06\audio_continuation_summary.json`
- `C:\Users\Andrew\AppData\Local\Temp\rafa_tokenburst\audio_continuation_broad18_copyphase_flat_cuda_2026_05_06\audio_continuation_summary.json`

All runs explicitly set `Future target magnitude reused` to `False`; the copyphase runs also set `Future target phase reused` to `False`.

| Run | Circleworld magnitude mode | Phase seed | Cases | Method | Corr | MAE | MSE | Loop peak | First reentry | Future mag reused |
|---|---|---|---:|---|---:|---:|---:|---:|---:|---|
| `audio_continuation_broad18_prefix_hold_cuda_2026_05_06` | `prefix_hold` | `waveform_copy_last_prefix` | 18 | `baseline_copy_last` | 0.090227 | 0.092010 | 0.028418 | 0.149877 | 0.754496 | `False` |
| `audio_continuation_broad18_prefix_hold_cuda_2026_05_06` | `prefix_hold` | `velocity` | 18 | `baseline_flat_magnitude` | -0.001723 | 0.097568 | 0.034524 | 0.083495 | 0.977860 | `False` |
| `audio_continuation_broad18_prefix_hold_cuda_2026_05_06` | `prefix_hold` | `velocity` | 18 | `baseline_prefix_magnitude_hold` | -0.010512 | 0.112238 | 0.043660 | 0.325314 | 0.986702 | `False` |
| `audio_continuation_broad18_prefix_hold_cuda_2026_05_06` | `prefix_hold` | `velocity` | 18 | `circleworld` | -0.011036 | 0.112180 | 0.043648 | 0.326501 | 0.983857 | `False` |
| `audio_continuation_broad18_flat_cuda_2026_05_06` | `flat` | `waveform_copy_last_prefix` | 18 | `baseline_copy_last` | 0.090227 | 0.092010 | 0.028418 | 0.149877 | 0.754496 | `False` |
| `audio_continuation_broad18_flat_cuda_2026_05_06` | `flat` | `velocity` | 18 | `baseline_flat_magnitude` | -0.001723 | 0.097568 | 0.034524 | 0.083495 | 0.977860 | `False` |
| `audio_continuation_broad18_flat_cuda_2026_05_06` | `flat` | `velocity` | 18 | `baseline_prefix_magnitude_hold` | -0.010512 | 0.112238 | 0.043660 | 0.325314 | 0.986702 | `False` |
| `audio_continuation_broad18_flat_cuda_2026_05_06` | `flat` | `velocity` | 18 | `circleworld` | -0.001704 | 0.097558 | 0.034522 | 0.083305 | 0.975447 | `False` |
| `audio_continuation_broad18_copyphase_prefix_hold_cuda_2026_05_06` | `prefix_hold` | `waveform_copy_last_prefix` | 18 | `baseline_copy_last` | 0.090227 | 0.092010 | 0.028418 | 0.149877 | 0.754496 | `False` |
| `audio_continuation_broad18_copyphase_prefix_hold_cuda_2026_05_06` | `prefix_hold` | `copy_last_waveform_phase` | 18 | `baseline_prefix_magnitude_hold` | 0.087144 | 0.099280 | 0.030628 | 0.162093 | 0.866454 | `False` |
| `audio_continuation_broad18_copyphase_prefix_hold_cuda_2026_05_06` | `prefix_hold` | `copy_last_waveform_phase` | 18 | `circleworld` | 0.088204 | 0.099135 | 0.030608 | 0.170894 | 0.867351 | `False` |
| `audio_continuation_broad18_copyphase_flat_cuda_2026_05_06` | `flat` | `waveform_copy_last_prefix` | 18 | `baseline_copy_last` | 0.090227 | 0.092010 | 0.028418 | 0.149877 | 0.754496 | `False` |
| `audio_continuation_broad18_copyphase_flat_cuda_2026_05_06` | `flat` | `copy_last_waveform_phase` | 18 | `baseline_flat_magnitude` | 0.019146 | 0.098042 | 0.034158 | 0.062875 | 0.837481 | `False` |
| `audio_continuation_broad18_copyphase_flat_cuda_2026_05_06` | `flat` | `copy_last_waveform_phase` | 18 | `circleworld` | 0.019629 | 0.098004 | 0.034156 | 0.074570 | 0.837195 | `False` |

Interpretation:

- These are no-future-magnitude continuation checks, not target-magnitude leakage.
- Circleworld is extremely close to the matching magnitude-carrier baseline in each run, with method-compare status `carrier_locked`.
- The stronger legal `copy_last_waveform_phase` seed improves the STFT continuation baseline, but it does not rescue the Circleworld claim: copyphase prefix-hold Circleworld corr `0.088204` remains below the copy-last waveform baseline corr `0.090227`, and the copyphase method compare remains `needs_review` rather than promotion evidence.
- Phase-gain ablation shows Circleworld phase deltas are present but not useful under this assay: nonzero gains either remain carrier-locked or change output without improving held-out target metrics.
- Phase-seed ablation shows the seed policy matters far more than the current Circleworld rollout. `copy_last_waveform_phase` is a legal no-future seed and nearly recovers copy-last waveform corr under prefix magnitude, while Circleworld only adds a small `+0.001059` corr on top of that seed.
- The weighted/fit phase-seed expansion does not find a better legal seed than `copy_last_waveform_phase`, so the next audio bar is not "find a cleverer seed"; it is making Circleworld's own phase law add robust value over that seed.
- The delta-over-copyphase probe shows a small broad 18-case tradeoff signal, not a win. Gain `2.0` over all bins improves prefix-hold corr by only `+0.002819` and MSE by `-0.000111` while staying close to the carrier; this is `delta_tradeoff_not_proof`, not autonomous audio continuation evidence.
- The fixed mechanism probe strengthens the diagnostic slightly: prefix-only `time_smooth_3` shaping over `prefix_hold/all_bins` reaches corr delta `+0.005640` and MSE delta `-0.000160`, but it is still `mechanism_tradeoff_not_proof` below the strict candidate bar and must not be promoted as audio viability evidence.
- The original 18-case predeclared lockbox finds one dramatic case-level improvement but not a robust mechanism: mean corr delta is `+0.056605`, median corr delta is `+0.000411`, and corr win fraction is `0.611111`. Treat this as a clue about sound-family sensitivity, not proof of general audio continuation.
- The stronger 62-case predeclared lockbox v1 confirms the negative read: mean corr delta is `+0.011134`, median corr delta is `-0.000039`, corr win fraction is `0.451613`, and non-bad weighted mean corr delta is `-0.015601`. Removing the single-source probe makes the mean corr delta negative at `-0.004088`.
- The phase-law scout v1 shows that prefix-only stability/energy/velocity gates are not the missing mechanism. The best no-single-source candidate is still bad-baseline-rescue dominated and has negative median corr delta.
- The internal phase-law scout v1 shows that recurrence-side knobs are causal relative to the current no-op config, but not yet sufficient for absolute audio viability: `velocity025` improves matched rows by `+0.001202` mean corr overall versus no-op, while the no-single-source absolute lockbox mean remains `-0.000796`.
- The sound-family split makes that clue more specific: `electric_motor_hum` is the only strong family, but it still has only `0.500000` corr wins and `0.720103` positive outlier share. That is a candidate research substrate, not a solved audio model.
- The objective scout confirms this is not solved by simple config knobs. Branch-pressure candidates move the floor-high base only slightly, and qtrace/phase-only relation controls are numerically near the baseline. The relsig base is weaker and does not produce a tradeoff signal under the same frozen objective row.
- The continuation results are not promotion evidence; they show that current audio behavior is dominated by carrier choice plus low-yield phase changes rather than a useful Circleworld phase-law continuation advantage.


## Substrate / Lattice Ablation - Current Cycle

Run:

`C:\Users\Andrew\AppData\Local\Temp\rafa_tokenburst\substrate_ablation_cuda_2026_05_06\substrate_ablation_summary.json`

Configuration:

- Device: `cuda`
- Time steps: `128`
- Seed plan: synthetic-only seeds `5100` through `5105`
- Config: `support_operator_floor_high`

| Variant | Parent branch | Child branch | Phase-only | Decorative | Excess | Gate mass | Phase delta | Support writeback | Major gain | Loss |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `lattice_baseline` | 0.038931 | 0.666667 | 0.070777 | 0.019827 | 0.031846 | 0.110815 | 0.082367 | 0.041117 | 0.025438 | 1.154298 |
| `pointwise_no_neighborhood` | 0.045300 | 0.666667 | 0.070777 | 0.018811 | 0.025476 | 0.110886 | 0.082413 | 0.041210 | 0.025465 | 1.154013 |
| `shuffled_neighborhood` | 0.026536 | 0.666667 | 0.070555 | 0.065299 | 0.044018 | 0.107589 | 0.080031 | 0.037175 | 0.024465 | 1.159790 |
| `flat_global_mean` | 0.026980 | 0.666667 | 0.070474 | 0.050196 | 0.043494 | 0.107820 | 0.080205 | 0.037442 | 0.024539 | 1.158523 |

Interpretation:

- Under the high-floor synthetic substrate, neighborhood topology is not the primary causal source of child branch survival: `pointwise_no_neighborhood` preserves child branch and phase-only branch, and slightly improves loss/decorative/excess metrics.
- Topology is not completely inert: shuffled/global smoothing slightly worsens loss, lowers writeback mass, and increases decorative slot-2 pressure.
- This is synthetic-only. It must not be generalized to naked-RAFA or seeded branch-active cases until rerun with a mixed/naked plan and paired per-seed deltas.


## Unified Claim-Isolation Suite - Mixed Seeds

Run:

`C:\Users\Andrew\AppData\Local\Temp\rafa_tokenburst\claim_isolation_cuda_mixed_2026_05_06\claim_isolation_suite_summary.json`

Configuration:

- Device: `cuda`
- Time steps: `96`
- Seed plan: synthetic seeds `5100`-`5102`, naked-RAFA seeds `6100`-`6102`
- Tracks: `child_writeback`, `q_basis`, `substrate`
- Baselines: `baseline_current`, `ramanujan_baseline`, `lattice_baseline`

Key paired results:

| Track | Variant | Phase-only | Naked phase-only | Gate mass | Phase delta | Support writeback | Law families | Major gain | Loss delta |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `child_writeback` | `support_only_phase_operator_floor_off` | 0.000000 | 0.000000 | 0.133112 | 0.000000 | 0.039967 | 1.000000 | 0.008244 | +0.051332 |
| `child_writeback` | `raw_phase_only_support_operator_floor_off` | 0.052420 | 0.000000 | 0.135161 | 0.063143 | 0.000000 | 1.500000 | 0.031307 | -0.015323 |
| `child_writeback` | `support_operator_floor_high` | 0.053591 | 0.007187 | 0.130011 | 0.074915 | 0.040155 | 1.500000 | 0.027245 | +0.000000 |
| `q_basis` | `qtrace_only_control` | 0.053591 | 0.007187 | 0.130011 | 0.074915 | 0.040155 | 1.500000 | 0.027245 | -0.000018 |
| `q_basis` | `uniform_q_weights` | 0.044291 | 0.000000 | 0.064879 | 0.047370 | 0.024480 | 2.833333 | 0.002252 | +0.016229 |
| `q_basis` | `shuffled_qset` | 0.045879 | 0.000000 | 0.070561 | 0.052803 | 0.027398 | 3.666667 | 0.001819 | +0.018254 |
| `substrate` | `pointwise_no_neighborhood` | 0.053618 | 0.007187 | 0.130070 | 0.074950 | 0.040226 | 1.500000 | 0.027263 | +0.000012 |
| `substrate` | `shuffled_neighborhood` | 0.053658 | 0.007187 | 0.129353 | 0.074446 | 0.039269 | 1.500000 | 0.027076 | +0.001070 |
| `substrate` | `flat_global_mean` | 0.053739 | 0.007187 | 0.128268 | 0.073641 | 0.037852 | 1.500000 | 0.026781 | +0.003531 |

Interpretation:

- The unified mixed-seed suite confirms the corrected child-writeback reading: support/logit/qtrace without phase delta/floor does not create parent-visible phase branch; raw phase alone helps synthetic branch but not naked branch; high floor gives small naked phase branch with decorative risk.
- It strengthens the Ramanujan demotion: `qtrace_only_control` again matches `ramanujan_baseline`, while uniform/shuffled/compact controls lose naked phase-only branch and pay loss/major-gain costs.
- It weakens the lattice-causality claim: pointwise/no-neighborhood preserves naked phase-only branch and the high-floor mechanism; shuffled/global smoothing cause only small loss/writeback degradation.


## Q-Trace Inheritance Ablation - Mixed Seeds

Run:

`C:\Users\Andrew\AppData\Local\Temp\rafa_tokenburst\claim_isolation_qtrace_cuda_mixed_2026_05_06\claim_isolation_suite_summary.json`

Configuration:

- Device: `cuda`
- Time steps: `96`
- Seed plan: synthetic seeds `5100`-`5102`, naked-RAFA seeds `6100`-`6102`
- Track: `qtrace_inheritance`

| Variant | Phase-only | Naked phase-only | Gate mass | Phase delta | Support writeback | Qtrace writeback | Major gain | Loss delta |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `qtrace_baseline` | 0.053591 | 0.007187 | 0.130011 | 0.074915 | 0.040155 | 0.000820 | 0.027245 | 0.000000 |
| `qtrace_current_only_no_memory` | 0.053591 | 0.007187 | 0.130011 | 0.074915 | 0.040155 | 0.011878 | 0.027245 | 0.000000 |
| `qtrace_frozen_no_write_no_survival` | 0.052783 | 0.006056 | 0.120168 | 0.068299 | 0.036734 | 0.000000 | 0.024028 | +0.006836 |
| `qtrace_survival_weights_off` | 0.052783 | 0.006056 | 0.120168 | 0.068299 | 0.036734 | 0.000794 | 0.024028 | +0.006836 |
| `qtrace_relation_gain_off` | 0.053591 | 0.007187 | 0.130011 | 0.074915 | 0.040155 | 0.000820 | 0.027245 | 0.000000 |
| `qtrace_child_survival_off` | 0.052783 | 0.006056 | 0.120168 | 0.068299 | 0.036734 | 0.000794 | 0.024029 | +0.006834 |

Interpretation:

- q-trace history is not the active ingredient here: `qtrace_current_only_no_memory` is behaviorally identical to baseline, despite a much larger qtrace writeback mass.
- Relational qtrace residual drive is not the active ingredient either: `qtrace_relation_gain_off` is identical to baseline.
- Child-local qtrace survival is causal: disabling child qtrace survival reproduces the degradation seen when all qtrace survival is disabled, lowering naked phase, gate/writeback, and major gain while increasing loss.
- This splits the q result into a narrower claim: current evidence supports child qtrace survival weighting, not Ramanujan arithmetic and not qtrace history/momentum.


## Hardy-Littlewood Arc Lane Ablation - Mixed Seeds

Run:

`C:\Users\Andrew\AppData\Local\Temp\rafa_tokenburst\claim_isolation_arc_cuda_mixed_2026_05_06\claim_isolation_suite_summary.json`

Configuration:

- Device: `cuda`
- Time steps: `96`
- Seed plan: synthetic seeds `5100`-`5102`, naked-RAFA seeds `6100`-`6102`
- Track: `arc_lane`

| Variant | Parent branch | Child branch | Phase-only | Naked phase-only | Gate mass | Phase delta | Support writeback | Law families | Major gain | Loss delta |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `arc_baseline` | 0.027899 | 0.500000 | 0.053591 | 0.007187 | 0.130011 | 0.074915 | 0.040155 | 1.500000 | 0.027245 | 0.000000 |
| `arc_flat_major_residue` | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | -0.000000 | +0.257992 |
| `arc_time_shuffled` | 0.000431 | 0.000000 | 0.098097 | 0.000000 | 0.055862 | 0.035530 | 0.018121 | 2.500000 | 0.025458 | +0.072666 |
| `arc_major_off_residue_high` | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | +0.621487 |
| `promotability_flat` | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | -0.000001 | +0.271803 |
| `promotability_zero` | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | -0.000000 | +0.302875 |

Interpretation:

- The Hardy-Littlewood arc/promotability lane is causal in the current high-floor branch path.
- Flattening major/minor arc fields kills child branch, parent branch, phase-only branch, law families, and writeback.
- Zeroing or flattening promotability also kills branch/writeback, showing packet promotion remains the gateway for childworld evidence.
- Time-shuffling arc fields produces high decorative phase-only activity (`0.098097`) but no child branch and no naked phase branch, which is exactly the failure case the ontology split is meant to catch.
- This is the strongest positive component claim in this testing cycle: arc/promotability is doing real work; Ramanujan-specific arithmetic and lattice topology are not yet proven at the same level.


## Arc x Q Cross-Ablation - Mixed Seeds

Run:

`C:\Users\Andrew\AppData\Local\Temp\rafa_tokenburst\claim_isolation_arc_q_cross_cuda_mixed_2026_05_06\claim_isolation_suite_summary.json`

Configuration:

- Device: `cuda`
- Time steps: `96`
- Seed plan: synthetic seeds `5100`-`5102`, naked-RAFA seeds `6100`-`6102`
- Track: `arc_q_cross`
- Crossed q controls: `ramanujan_baseline`, `qtrace_only_control`, `uniform_q_weights`, `shuffled_qset`, `compact_fourierish_control`
- Crossed arc controls: `arc_baseline`, `arc_flat_major_residue`, `arc_time_shuffled`, `promotability_zero`

Baseline arc rows:

| Q variant | Child branch | Phase-only | Naked phase-only | Gate mass | Support writeback | Law families | Major gain | Loss |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `ramanujan_baseline` | 0.500000 | 0.053591 | 0.007187 | 0.130011 | 0.040155 | 1.500000 | 0.027245 | 1.529047 |
| `qtrace_only_control` | 0.500000 | 0.053591 | 0.007187 | 0.130011 | 0.040155 | 1.500000 | 0.027245 | 1.529029 |
| `uniform_q_weights` | 0.305556 | 0.044291 | 0.000000 | 0.064879 | 0.024480 | 2.833333 | 0.002252 | 1.545276 |
| `shuffled_qset` | 0.305556 | 0.045879 | 0.000000 | 0.070561 | 0.027398 | 3.666667 | 0.001819 | 1.547301 |
| `compact_fourierish_control` | 0.333333 | 0.049271 | 0.000000 | 0.066082 | 0.025339 | 2.333333 | 0.001149 | 1.560882 |

Arc-flatten rows:

| Q variant | Child branch | Phase-only | Naked phase-only | Gate mass | Support writeback | Law families | Loss |
|---|---:|---:|---:|---:|---:|---:|---:|
| `ramanujan_baseline` | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 1.787039 |
| `qtrace_only_control` | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 1.787036 |
| `uniform_q_weights` | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 1.657845 |
| `shuffled_qset` | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 1.654118 |
| `compact_fourierish_control` | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 1.676147 |

Interpretation:

- Arc/promotability remains necessary across every q family tested. No q control survives `arc_flat_major_residue` or `promotability_zero` with branch/writeback intact.
- Ramanujan and qtrace-only are still indistinguishable on the intact-arc row, including naked phase branch.
- Uniform, shuffled, and compact q controls retain some child branch with intact arcs, but they lose naked phase branch and most major gain.
- This splits the current best explanation: Hardy-Littlewood arc/promotability is the causal gateway; q structure modulates whether that gateway produces naked branch and benchmark structure; Ramanujan-specific arithmetic remains unproven because qtrace-only matches it.

## Dense Signature And Semantic Contract - Current Cycle

Dense signature claim:

- Harness: `D:\RAFA\runtimes\circleworld_proto\evaluate_dense_signature_claim.py`
- Artifact: `C:\Users\Andrew\AppData\Local\Temp\rafa_tokenburst\dense_signature_claim_broad18_cuda_2026_05_06\dense_signature_claim.json`
- Status: `needs_review`
- Cases: `18`
- Dense stability score: `0.995793`
- Metadata stability score: `0.993945`
- Stability delta dense-minus-metadata: `+0.001848`
- Dense separation score: `0.011154`
- Metadata separation score: `0.022951`
- Separation delta dense-minus-metadata: `-0.011797`
- Dense dominant cluster share: `0.625000`
- Metadata dominant cluster share: `0.583333`
- Dense near-duplicate pair share: `0.503885`
- Metadata near-duplicate pair share: `0.479021`

Interpretation:

- Dense signatures are now a separate testable RAFA-token claim, not just a sidecar export.
- This run does not prove dense signatures. They are slightly more stable under transforms, but worse at case separation and collapse diagnostics than explicit q/law metadata.
- The next dense-signature bar is not more export volume; it is Matryoshka/predictive training that improves stability, separation, and anti-collapse simultaneously.

Semantic projector contract:

- Harness: `D:\RAFA\runtimes\circleworld_proto\test_semantic_projector_contract.py`
- Artifact: `C:\Users\Andrew\AppData\Local\Temp\rafa_tokenburst\semantic_projector_contract_2026_05_06\semantic_projector_contract.json`
- Status: `contract_smoke_only`
- Import ok: `True`
- Projection path: `deterministic_local_mock`
- Control schema exact: `True`
- Prompt family count: `4`
- Pairwise diversity count: `6`
- All prompt pairs diverse: `True`

Interpretation:

- The five-control interface is contract-ready: `harmonic_coupling`, `decay_rate`, `noise_injection`, `branch_temperature`, and `support_spread`.
- This is not semantic proof. The current projector still requires explicit weights, so the audit uses deterministic mock vectors and only proves interface/readiness.

## Validation

Current report-only update:

- Report sections were updated from the listed JSON/Markdown summaries.
- After validator feedback, stale child-writeback/q-basis/substrate summaries were regenerated from raw heldout files with additive writeback schema fields.
- The old v1 support-only ablation is explicitly marked superseded/confounded, and current-cycle conclusions use v4 seed-fallback results after robust packet seed repair.
- Current compile check passed for the repaired harnesses, audio phase influence/seed/delta/mechanism/family/electric-holdout/steady-manifest/phenotype/baseline-stratification/predeclared-lockbox/scout harnesses, dense signature harness, semantic projector contract harness, and claim-evidence assembler; the predeclared mechanism lockboxes were run on 18 unused PCM WAV cases and then on a stronger 62-case RIFF-validated manifest.

Earlier runtime validation commands recorded in this report:

```powershell
python -m py_compile D:\RAFA\lineages\04_positive_replacement\circleworld.py D:\RAFA\runtimes\circleworld_proto\train_circleworld.py D:\RAFA\runtimes\circleworld_proto\train_circleworld_real_anchor.py D:\RAFA\runtimes\circleworld_proto\evaluate_circleworld.py D:\RAFA\runtimes\circleworld_proto\export_circleworld_audio.py D:\RAFA\runtimes\circleworld_proto\test_nested_commitment.py
git diff --check -- D:\RAFA\lineages\04_positive_replacement\circleworld.py D:\RAFA\runtimes\circleworld_proto\train_circleworld.py D:\RAFA\runtimes\circleworld_proto\train_circleworld_real_anchor.py D:\RAFA\runtimes\circleworld_proto\evaluate_circleworld.py D:\RAFA\runtimes\circleworld_proto\export_circleworld_audio.py D:\RAFA\runtimes\circleworld_proto\test_nested_commitment.py
python D:\RAFA\lineages\04_positive_replacement\ablate_formalization.py --mode native_multimode_childworld --seed-source synthetic --device cpu --batch-size 1 --time-steps 16 --depth 2
```

Earlier runtime validation results:

- Python compile passed.
- Diff check passed except line-ending warnings from Git.
- CPU smoke passed and emitted the additive `mean_child_support_writeback_mass` field.
- Config round-trip verified that export/evaluate loaders and CEM materializer preserve the new child writeback controls.
- Ablation harness write-only smoke generated seven fixed child-writeback variant configs.

## Decision

Do not promote this patch as a checkpoint improvement.

Keep the instrumentation and config plumbing. Treat the phase-floor runtime as an active research lever. The corrected current-cycle evidence does not support promotion: the next bottleneck is child coherence and qualified active carry, not just writeback amplitude. The immediate next scientific question is not "can we raise child branch fraction"; it is "can we convert raw child ID carry into qualified active carry without increasing decorative slot-2 pressure or destroying major/benchmark structure."

## Next Variables To Sweep

- Lower or retune child coherence qualification only as a diagnostic, not as a success shortcut.
- Sweep `child_writeback_phase_floor_gain` and `child_writeback_phase_floor_target` against naked phase-only branch and audio regression.
- Sweep `child_operator_seed_gain` independently from support writeback.
- Add an ablation where `child_support_writeback_mass` is forced to zero while operator phase writeback remains active.
- Add an ablation where operator/floor is zero while child support/logit/qtrace writeback remains active.
- Separate reports must continue to show parent branch, child branch, and phase-only branch as distinct metrics.
