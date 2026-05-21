# Circleworld Internal Phase-Law Scout - 2026-05-07

## Status

This is a runtime/instrumentation scout, not a checkpoint promotion.

The new evidence splits two questions that were previously tangled:

- Does changing Circleworld's internal phase law affect recurrence? Yes, weakly but measurably.
- Does the current internal-law variant make autonomous audio continuation promotion-grade? No.

## Code Changes

Primary files:

- `D:\RAFA\lineages\04_positive_replacement\circleworld.py`
- `D:\RAFA\runtimes\circleworld_proto\evaluate_circleworld.py`
- `D:\RAFA\runtimes\circleworld_proto\export_circleworld_audio.py`
- `D:\RAFA\runtimes\circleworld_proto\train_circleworld.py`
- `D:\RAFA\runtimes\circleworld_proto\train_circleworld_real_anchor.py`
- `D:\RAFA\runtimes\circleworld_proto\build_internal_phase_law_variant_spec.py`
- `D:\RAFA\runtimes\circleworld_proto\run_internal_phase_law_objective_scout.py`
- `D:\RAFA\runtimes\circleworld_proto\compare_internal_phase_law_variants.py`
- `D:\RAFA\runtimes\circleworld_proto\score_internal_phase_law_objective.py`
- `D:\RAFA\runtimes\circleworld_proto\diagnose_internal_phase_law_joint_rows.py`
- `D:\RAFA\runtimes\circleworld_proto\build_internal_phase_law_failure_atlas.py`
- `D:\RAFA\runtimes\circleworld_proto\assemble_rafa_claim_evidence.py`

Implemented:

- Added default-off Circleworld config fields:
  - `phase_law_precondition_gain`
  - `phase_law_velocity_mix`
  - `phase_law_stability_gain`
  - `phase_law_softclip`
  - `phase_law_low_rank`
  - `phase_law_consensus_mix`
  - `phase_law_consensus_damping`
  - `phase_law_median_guard`
  - `phase_law_local_velocity_mix`
  - `phase_law_local_coherence_damping`
  - `phase_law_curvature_guard`
- Added a phase-only internal delta preconditioner based on relative phase velocity and temporal stability.
- Added default-off v2 guard controls based on phase-velocity consensus, consensus damping, and median phase-delta guarding.
- Added default-off v3 local controls based on neighboring-frequency phase velocity, local coherence damping, and phase-curvature guarding.
- Applied the preconditioner at the `apply_active_packets(...)` seam, so it affects both single-path and native multimode packet transport.
- Applied the same default-off hook to child-world writeback deltas before parent mode phase update.
- Propagated the new fields through evaluation/export config loaders and CEM materialization/payload output.
- Added bounded CEM sampling for phase-law controls and repaired missing defect/instability/law-packet CEM mean/std/materialized fields exposed by a tiny CPU trainer smoke.
- Added `run_internal_phase_law_objective_scout.py` to generate internal-law configs, fan out lockbox probes, run absolute/direct compares, and score the locked objective in one command.
- Added `build_internal_phase_law_variant_spec.py` to generate reproducible coefficient-search specs.
- Added `compare_internal_phase_law_variants.py` to compare matched lockbox rows between a no-op control config and internal-law variants.
- Added `score_internal_phase_law_objective.py` to require both direct recurrence improvement and absolute carrier-beating improvement.
- Added `diagnose_internal_phase_law_joint_rows.py` to join direct rows to absolute rows and expose row-local failure reasons.
- Added `build_internal_phase_law_failure_atlas.py` to separate bad-baseline rescue, robust absolute bins, and direct recurrence effects.
- Upgraded the locked objective and joint-row diagnostics to expose non-bad baseline-bin regression directly, so bad-baseline rescue cannot masquerade as same-row success.

The new runtime path remains default-off. Existing configs without these fields preserve prior behavior.

## Scout Design

Base checkpoint:

`D:\RAFA\checkpoints_circleworld_proto\training_run_2026-04-24_agreement_scout_v1\circleworld_real_anchor_config_cem_v1.json`

Generated variants:

`C:\Users\Andrew\AppData\Local\Temp\rafa_tokenburst\internal_phase_law_configs_2026_05_07\internal_phase_law_config_manifest.json`

Variants:

- `agreement_scout_v1_phase_law_off`
- `internal_gate_025_lr12`
- `internal_gate_050_lr12`
- `internal_velocity_025_lr12`
- `internal_softclip_050_lr12`

Lockbox substrate:

`C:\Users\Andrew\AppData\Local\Temp\rafa_tokenburst\audio_predeclared_lockbox_v1_2026_05_06\audio_predeclared_lockbox_cases.json`

Probe row family:

- Mechanism: `raw`
- Gains: `0.0`, `1.0`, `2.0`
- Magnitude modes: `prefix_hold`, `flat`
- Masks: `all_bins`, `high_energy_bins`, `low_energy_bins`
- Full lockbox: `62` cases
- Main read excludes role `predeclared_single_source_probe`, leaving `61` cases.

## Artifacts

- Raw full scout root: `C:\Users\Andrew\AppData\Local\Temp\rafa_tokenburst\internal_phase_law_raw_scout_full_2026_05_07\`
- Absolute no-single-source compare: `C:\Users\Andrew\AppData\Local\Temp\rafa_tokenburst\internal_phase_law_raw_scout_full_compare_no_single_source_2026_05_07\audio_lockbox_result_compare.json`
- Direct variant no-single-source compare: `C:\Users\Andrew\AppData\Local\Temp\rafa_tokenburst\internal_phase_law_variant_compare_no_single_source_2026_05_07\internal_phase_law_variant_compare.json`
- Locked objective score: `C:\Users\Andrew\AppData\Local\Temp\rafa_tokenburst\internal_phase_law_objective_score_2026_05_07\internal_phase_law_objective_score.json`
- Automation dry-run verification: `C:\Users\Andrew\AppData\Local\Temp\rafa_tokenburst\internal_phase_law_objective_scout_auto_dryrun_verify_2026_05_07\internal_phase_law_objective_scout_manifest.json`
- Automation one-case smoke: `C:\Users\Andrew\AppData\Local\Temp\rafa_tokenburst\internal_phase_law_objective_scout_auto_smoke_2026_05_07\TRACK_REPORT.md`
- Automation full lockbox run: `C:\Users\Andrew\AppData\Local\Temp\rafa_tokenburst\internal_phase_law_objective_scout_auto_full_2026_05_07\TRACK_REPORT.md`
- Automation full objective score: `C:\Users\Andrew\AppData\Local\Temp\rafa_tokenburst\internal_phase_law_objective_scout_auto_full_2026_05_07\objective_score\internal_phase_law_objective_score.json`
- Focused coefficient spec: `C:\Users\Andrew\AppData\Local\Temp\rafa_tokenburst\internal_phase_law_variant_specs_2026_05_07\internal_phase_law_focused_v1.json`
- Focused coefficient run: `C:\Users\Andrew\AppData\Local\Temp\rafa_tokenburst\internal_phase_law_objective_scout_focused_v1_2026_05_07\TRACK_REPORT.md`
- Focused coefficient objective score: `C:\Users\Andrew\AppData\Local\Temp\rafa_tokenburst\internal_phase_law_objective_scout_focused_v1_2026_05_07\objective_score\internal_phase_law_objective_score.json`
- Focused joint row diagnostic: `C:\Users\Andrew\AppData\Local\Temp\rafa_tokenburst\internal_phase_law_joint_rows_focused_v1_2026_05_07\internal_phase_law_joint_row_diagnostics.json`
- Wide coefficient spec: `C:\Users\Andrew\AppData\Local\Temp\rafa_tokenburst\internal_phase_law_variant_specs_2026_05_07\internal_phase_law_wide_v1.json`
- Wide coefficient run: `C:\Users\Andrew\AppData\Local\Temp\rafa_tokenburst\internal_phase_law_objective_scout_wide_v1_2026_05_07\TRACK_REPORT.md`
- Wide coefficient objective score: `C:\Users\Andrew\AppData\Local\Temp\rafa_tokenburst\internal_phase_law_objective_scout_wide_v1_2026_05_07\objective_score\internal_phase_law_objective_score.json`
- Wide joint row diagnostic: `C:\Users\Andrew\AppData\Local\Temp\rafa_tokenburst\internal_phase_law_joint_rows_wide_v1_2026_05_07\internal_phase_law_joint_row_diagnostics.json`
- v2 guard coefficient spec: `C:\Users\Andrew\AppData\Local\Temp\rafa_tokenburst\internal_phase_law_variant_specs_2026_05_07\internal_phase_law_v2_guard_v1.json`
- v2 guard coefficient run: `C:\Users\Andrew\AppData\Local\Temp\rafa_tokenburst\internal_phase_law_objective_scout_v2_guard_v1_2026_05_07\TRACK_REPORT.md`
- v2 guard coefficient objective score: `C:\Users\Andrew\AppData\Local\Temp\rafa_tokenburst\internal_phase_law_objective_scout_v2_guard_v1_2026_05_07\objective_score\internal_phase_law_objective_score.json`
- v2 guard joint row diagnostic: `C:\Users\Andrew\AppData\Local\Temp\rafa_tokenburst\internal_phase_law_joint_rows_v2_guard_v1_2026_05_07\internal_phase_law_joint_row_diagnostics.json`
- v2 guard failure atlas: `C:\Users\Andrew\AppData\Local\Temp\rafa_tokenburst\internal_phase_law_failure_atlas_v2_guard_v1_2026_05_07\internal_phase_law_failure_atlas.json`
- v3 local coefficient spec: `C:\Users\Andrew\AppData\Local\Temp\rafa_tokenburst\internal_phase_law_variant_specs_2026_05_07\internal_phase_law_v3_local_v1.json`
- v3 local coefficient run: `C:\Users\Andrew\AppData\Local\Temp\rafa_tokenburst\internal_phase_law_objective_scout_v3_local_v1_2026_05_07\TRACK_REPORT.md`
- v3 local raw absolute compare: `C:\Users\Andrew\AppData\Local\Temp\rafa_tokenburst\internal_phase_law_objective_scout_v3_local_v1_2026_05_07\absolute_compare\audio_lockbox_result_compare.json`
- v3 local direct variant compare: `C:\Users\Andrew\AppData\Local\Temp\rafa_tokenburst\internal_phase_law_objective_scout_v3_local_v1_2026_05_07\variant_compare\internal_phase_law_variant_compare.json`
- v3 local coefficient objective score: `C:\Users\Andrew\AppData\Local\Temp\rafa_tokenburst\internal_phase_law_objective_scout_v3_local_v1_2026_05_07\objective_score\internal_phase_law_objective_score.json`
- v3 local joint row diagnostic: `C:\Users\Andrew\AppData\Local\Temp\rafa_tokenburst\internal_phase_law_joint_rows_v3_local_v1_2026_05_07\internal_phase_law_joint_row_diagnostics.json`
- v3 local failure atlas: `C:\Users\Andrew\AppData\Local\Temp\rafa_tokenburst\internal_phase_law_failure_atlas_v3_local_v1_2026_05_07\internal_phase_law_failure_atlas.json`
- v3 local strict joint objective score: `C:\Users\Andrew\AppData\Local\Temp\rafa_tokenburst\internal_phase_law_objective_score_joint_strict_v3_local_v1_2026_05_07\internal_phase_law_objective_score.json`
- v3 local strict joint row diagnostic: `C:\Users\Andrew\AppData\Local\Temp\rafa_tokenburst\internal_phase_law_joint_rows_strict_v3_local_v1_2026_05_07\internal_phase_law_joint_row_diagnostics.json`
- v4 joint/nonbad full run root: `C:\Users\Andrew\AppData\Local\Temp\rafa_tokenburst\internal_phase_law_objective_scout_v4_joint_nonbad_v1_2026_05_07\`
- v4 joint/nonbad objective score: `C:\Users\Andrew\AppData\Local\Temp\rafa_tokenburst\internal_phase_law_objective_scout_v4_joint_nonbad_v1_2026_05_07\objective_score\internal_phase_law_objective_score.json`
- v4 joint/nonbad row diagnostic: `C:\Users\Andrew\AppData\Local\Temp\rafa_tokenburst\internal_phase_law_objective_scout_v4_joint_nonbad_v1_2026_05_07\joint_diagnostics\internal_phase_law_joint_row_diagnostics.json`
- v4 joint/nonbad failure atlas: `C:\Users\Andrew\AppData\Local\Temp\rafa_tokenburst\internal_phase_law_failure_atlas_v4_joint_nonbad_v1_2026_05_07\internal_phase_law_failure_atlas.json`
- v5 low-energy target-row full run root: `C:\Users\Andrew\AppData\Local\Temp\rafa_tokenburst\internal_phase_law_objective_scout_v5_low_energy_row_v1_2026_05_07\`
- v5 low-energy target-row objective score: `C:\Users\Andrew\AppData\Local\Temp\rafa_tokenburst\internal_phase_law_objective_scout_v5_low_energy_row_v1_2026_05_07\objective_score\internal_phase_law_objective_score.json`
- v5 low-energy target-row row diagnostic: `C:\Users\Andrew\AppData\Local\Temp\rafa_tokenburst\internal_phase_law_objective_scout_v5_low_energy_row_v1_2026_05_07\joint_diagnostics\internal_phase_law_joint_row_diagnostics.json`
- v5 low-energy target-row failure atlas: `C:\Users\Andrew\AppData\Local\Temp\rafa_tokenburst\internal_phase_law_failure_atlas_v5_low_energy_row_v1_2026_05_07\internal_phase_law_failure_atlas.json`
- v6 low-energy win full run root: `C:\Users\Andrew\AppData\Local\Temp\rafa_tokenburst\internal_phase_law_objective_scout_v6_low_energy_win_v1_2026_05_07\`
- v6 low-energy win objective score: `C:\Users\Andrew\AppData\Local\Temp\rafa_tokenburst\internal_phase_law_objective_scout_v6_low_energy_win_v1_2026_05_07\objective_score\internal_phase_law_objective_score.json`
- v6 low-energy win row diagnostic: `C:\Users\Andrew\AppData\Local\Temp\rafa_tokenburst\internal_phase_law_objective_scout_v6_low_energy_win_v1_2026_05_07\joint_diagnostics\internal_phase_law_joint_row_diagnostics.json`
- v6 low-energy win failure atlas: `C:\Users\Andrew\AppData\Local\Temp\rafa_tokenburst\internal_phase_law_failure_atlas_v6_low_energy_win_v1_2026_05_07\internal_phase_law_failure_atlas.json`
- v7 low-energy gain-ladder full run root: `C:\Users\Andrew\AppData\Local\Temp\rafa_tokenburst\internal_phase_law_objective_scout_v7_low_energy_gain_ladder_v1_2026_05_07\`
- v7 low-energy gain-ladder objective score: `C:\Users\Andrew\AppData\Local\Temp\rafa_tokenburst\internal_phase_law_objective_scout_v7_low_energy_gain_ladder_v1_2026_05_07\objective_score\internal_phase_law_objective_score.json`
- v7 low-energy gain-ladder row diagnostic: `C:\Users\Andrew\AppData\Local\Temp\rafa_tokenburst\internal_phase_law_objective_scout_v7_low_energy_gain_ladder_v1_2026_05_07\joint_diagnostics\internal_phase_law_joint_row_diagnostics.json`
- v7 low-energy gain-ladder failure atlas: `C:\Users\Andrew\AppData\Local\Temp\rafa_tokenburst\internal_phase_law_failure_atlas_v7_low_energy_gain_ladder_v1_2026_05_07\internal_phase_law_failure_atlas.json`
- v7 target flip diagnostic: `C:\Users\Andrew\AppData\Local\Temp\rafa_tokenburst\internal_phase_law_target_flip_diagnostics_v7_low_energy_gain_ladder_v1_2026_05_07\internal_phase_law_target_flip_diagnostics.json`
- v9 explicit phase-stable target objective score: `C:\Users\Andrew\AppData\Local\Temp\rafa_tokenburst\internal_phase_law_objective_score_v9_phase_stable_target_v1_2026_05_07\internal_phase_law_objective_score.json`
- v9 explicit phase-stable target joint diagnostic: `C:\Users\Andrew\AppData\Local\Temp\rafa_tokenburst\internal_phase_law_joint_rows_v9_phase_stable_target_v1_2026_05_07\internal_phase_law_joint_row_diagnostics.json`
- v10 phase-router mask scout: `C:\Users\Andrew\AppData\Local\Temp\rafa_tokenburst\internal_phase_law_objective_scout_v10_phase_router_masks_v1_2026_05_07\`
- v10 phase-router target objective score: `C:\Users\Andrew\AppData\Local\Temp\rafa_tokenburst\internal_phase_law_objective_scout_v10_phase_router_masks_v1_2026_05_07\objective_score\internal_phase_law_objective_score.json`
- v10 phase-router target joint diagnostic: `C:\Users\Andrew\AppData\Local\Temp\rafa_tokenburst\internal_phase_law_objective_scout_v10_phase_router_masks_v1_2026_05_07\joint_diagnostics\internal_phase_law_joint_row_diagnostics.json`
- v9 candidate-class diagnostic: `C:\Users\Andrew\AppData\Local\Temp\rafa_tokenburst\internal_phase_law_candidate_classes_v9_phase_masks_v1_2026_05_07\internal_phase_law_candidate_classes.json`
- v10 candidate-class diagnostic: `C:\Users\Andrew\AppData\Local\Temp\rafa_tokenburst\internal_phase_law_candidate_classes_v10_phase_router_v1_2026_05_07\internal_phase_law_candidate_classes.json`
- v11 phase-router direct scout: `C:\Users\Andrew\AppData\Local\Temp\rafa_tokenburst\internal_phase_law_objective_scout_v11_phase_router_direct_v1_2026_05_07\`
- v11 candidate-class diagnostic: `C:\Users\Andrew\AppData\Local\Temp\rafa_tokenburst\internal_phase_law_candidate_classes_v11_phase_router_direct_v1_2026_05_07\internal_phase_law_candidate_classes.json`
- v12 phase-reentry direct scout: `C:\Users\Andrew\AppData\Local\Temp\rafa_tokenburst\internal_phase_law_objective_scout_v12_phase_reentry_direct_v1_2026_05_07\`
- v12 candidate-class diagnostic: `C:\Users\Andrew\AppData\Local\Temp\rafa_tokenburst\internal_phase_law_candidate_classes_v12_phase_reentry_direct_v1_2026_05_07\internal_phase_law_candidate_classes.json`
- v13 causal phase-reentry direct scout: `C:\Users\Andrew\AppData\Local\Temp\rafa_tokenburst\internal_phase_law_objective_scout_v13_causal_reentry_direct_v1_2026_05_07\`
- v13 candidate-class diagnostic: `C:\Users\Andrew\AppData\Local\Temp\rafa_tokenburst\internal_phase_law_candidate_classes_v13_causal_reentry_direct_v1_2026_05_07\internal_phase_law_candidate_classes.json`
- Reentry contract audit: `C:\Users\Andrew\AppData\Local\Temp\rafa_tokenburst\internal_phase_law_reentry_contract_2026_05_07\internal_phase_law_reentry_contract.json`
- Predeclared diagnostic class: `C:\Users\Andrew\AppData\Local\Temp\rafa_tokenburst\internal_phase_law_predeclared_diagnostic_class_v10_v13_2026_05_07\internal_phase_law_predeclared_diagnostic_class.json`
- Updated claim rollup: `C:\Users\Andrew\AppData\Local\Temp\rafa_tokenburst\claim_evidence_2026_05_06\rafa_claim_evidence_summary.json`

## Automation Utility

The rerunnable utility is:

`python D:\RAFA\runtimes\circleworld_proto\run_internal_phase_law_objective_scout.py`

Default behavior:

- Base config: active `agreement_scout_v1` checkpoint.
- Cases: predeclared audio lockbox v1.
- Device: `cuda`.
- Cases: all cases via `--num-cases 0`.
- Probe family: `raw`, gains `0.0,1.0,2.0`, modes `prefix_hold,flat`, masks `all_bins,high_energy_bins,low_energy_bins`.
- Compare hygiene: excludes `single_source` and any matching manifest role such as `predeclared_single_source_probe`.
- Outputs: variant configs, per-variant probe outputs, absolute compare, direct variant compare, locked objective score, same-row joint diagnostics, manifest, and `TRACK_REPORT.md`.

Verification:

- `python -m py_compile D:\RAFA\runtimes\circleworld_proto\run_internal_phase_law_objective_scout.py` passed.
- `python -m py_compile D:\RAFA\runtimes\circleworld_proto\train_circleworld.py` passed after adding phase-law CEM sampling.
- A one-iteration CPU CEM smoke completed under `C:\Users\Andrew\AppData\Local\Temp\rafa_tokenburst\train_phase_law_sampling_smoke_2026_05_07\`, confirming phase-law controls are sampled, bounded, materialized, and serialized.
- Dry-run wrote manifest/report successfully.
- One-case CPU smoke completed the full subprocess chain and ended with `no_locked_candidate`, as expected for a non-evidence wiring test.
- Full CUDA lockbox automation completed across `61` non-single-source cases / `3660` analysis rows and reproduced the hand-stitched locked objective rows exactly.
- Custom `--variant-spec-json` dry-run passed; candidate-only specs auto-insert a no-op baseline and reject unknown non-phase-law keys.

Coefficient-search use:

`python D:\RAFA\runtimes\circleworld_proto\run_internal_phase_law_objective_scout.py --variant-spec-json <candidate_spec.json> --out-dir <candidate_out_dir>`

The variant spec may be either a list or an object with a `variants` list. Each candidate may set only:

- `phase_law_precondition_gain`
- `phase_law_velocity_mix`
- `phase_law_stability_gain`
- `phase_law_softclip`
- `phase_law_low_rank`
- `phase_law_consensus_mix`
- `phase_law_consensus_damping`
- `phase_law_median_guard`
- `phase_law_local_velocity_mix`
- `phase_law_local_coherence_damping`
- `phase_law_curvature_guard`

## Absolute Audio Read

This compares each variant's raw Circleworld future delta against its own gain-0 carrier.

- Status: `tradeoff_signal_only`
- Cases: `61`
- Analysis rows: `3660`
- Overall mean corr delta: `-0.000795945`
- Overall median corr delta: `0.0`
- Overall corr win fraction: `0.440437`
- Overall mean MSE delta: `+0.000257976`
- Non-bad weighted mean corr delta: `-0.00401684`

Best absolute candidate row:

- Run: `velocity025`
- Mode: `flat`
- Mask: `high_energy_bins`
- Mechanism: `raw`
- Gain: `2.0`
- Status: `bad_baseline_rescue_dominated`
- Mean corr delta: `+0.000193858`
- Median corr delta: `+0.0000243263`
- Corr win fraction: `0.524590`
- Mean MSE delta: `-0.0000138633`

Interpretation: absolute audio continuation remains weak. The internal phase law does not beat the copyphase/gain-0 carrier at promotion scale.

## Direct Variant Read

This compares each internal-law variant against the no-op `agreement_scout_v1_phase_law_off` config on matched lockbox rows.

Overall direct compare:

| Variant | Rows | Cases | Status | Mean corr delta vs no-op | Median corr delta | Corr wins | Mean MSE delta |
|---|---:|---:|---|---:|---:|---:|---:|
| `velocity025` | 732 | 61 | `variant_tradeoff_or_tiny_signal` | 0.00120204 | 0.000000326 | 0.535519 | -0.0000443978 |
| `softclip050` | 732 | 61 | `variant_tradeoff_or_tiny_signal` | 0.000998695 | 0.0000000235 | 0.515027 | -0.0000914907 |
| `gate050` | 732 | 61 | `variant_tradeoff_or_tiny_signal` | 0.000878036 | 0.000000227 | 0.527322 | -0.0000395471 |
| `gate025` | 732 | 61 | `variant_tradeoff_or_tiny_signal` | 0.000216345 | 0.0000000104 | 0.516393 | 0.00000216239 |

Best matched row:

- Variant: `velocity025`
- Mode: `prefix_hold`
- Mask: `all_bins`
- Mechanism: `raw`
- Gain: `2.0`
- Status: `variant_improved`
- Mean corr delta vs no-op: `+0.00373389`
- Median corr delta: `+0.000129043`
- Corr win fraction: `0.573770`
- Mean MSE delta: `-0.000110839`

Interpretation: internal phase-law knobs are real recurrence variables. They improve matched raw Circleworld rows against the no-op config, but the improvement is not enough to make the absolute audio task succeed.

## Locked Objective Score

The locked objective requires both direct improvement over no-op and absolute improvement over the gain-0/copyphase carrier.

Result:

- Status: `no_locked_candidate`
- Best run: `velocity025`
- Best decision: `hold_absolute_win_fraction`
- Best score: `0.00513798`

| Variant | Decision | Score | Direct mean d | Direct best d | Abs nonbad mean d | Abs nonbad wins | Matched abs d | Matched status |
|---|---|---:|---:|---:|---:|---:|---:|---|
| `velocity025` | `hold_absolute_win_fraction` | 0.005138 | 0.001202 | 0.003734 | 0.000109 | 0.540984 | 0.00000143 | `bad_baseline_rescue_dominated` |
| `softclip050` | `hold_matched_absolute_negative` | 0.002227 | 0.000999 | 0.002652 | 0.000092 | 0.557377 | -0.001211 | `bad_baseline_rescue_dominated` |
| `gate050` | `hold_matched_absolute_negative` | 0.001606 | 0.000878 | 0.002438 | 0.000114 | 0.557377 | -0.001294 | `bad_baseline_rescue_dominated` |
| `gate025` | `hold_absolute_win_fraction` | -0.005582 | 0.000216 | 0.000571 | 0.000134 | 0.540984 | -0.003293 | `bad_baseline_rescue_dominated` |

Interpretation: the scorer prevents the exact failure mode we want to avoid. Direct recurrence influence alone is not enough; the matched absolute row also has to beat the carrier without bad-baseline domination.

## Claim Impact

Promote:

- `internal_phase_law_influence` as a testable causal variable.

Do not promote:

- `audio_without_future_magnitude`
- `semi_tokenless_audio_model`
- any checkpoint based on this scout alone.

Next decisive test:

Search internal phase-law coefficients against a locked objective that gates both:

- direct variant improvement versus no-op, and
- absolute improvement versus gain-0/copyphase carrier stratified by baseline quality.

The current result says: change the recurrence, not the export post-shaper, but keep the audio claim negative until absolute lockbox performance clears.

## Focused Coefficient Search v1

Spec:

`C:\Users\Andrew\AppData\Local\Temp\rafa_tokenburst\internal_phase_law_variant_specs_2026_05_07\internal_phase_law_focused_v1.json`

Run:

`C:\Users\Andrew\AppData\Local\Temp\rafa_tokenburst\internal_phase_law_objective_scout_focused_v1_2026_05_07\TRACK_REPORT.md`

This searched `16` candidate coefficient rows around the earlier `velocity025` signal, plus an inserted no-op baseline.

Result:

- Status: `no_locked_candidate`
- Best run: `v025_vel008_stab20_lr12`
- Best decision: `hold_absolute_win_fraction`
- Best score: `0.00723151`
- Direct overall mean corr delta: `+0.00160001`
- Direct best row corr delta: `+0.00503149`
- Absolute non-bad mean corr delta: `+0.0000913004`
- Absolute non-bad corr win fraction: `0.540984`
- Matched absolute corr delta: `+0.00129903`
- Matched absolute status: `bad_baseline_rescue_dominated`

Interpretation:

The focused search found a better direct recurrence coefficient than the fixed scout (`velocity_mix=0.08` rather than `0.05`), but it still fails the locked audio gate. This strengthens the variable isolation claim while keeping the audio/semi-tokenless claim negative.

Joint-row diagnostic:

- Status: `no_joint_row_candidate`
- Joined rows: `192`
- Candidate rows: `0`
- Main failure counts:
  - `hold_absolute_nonpositive`: `73`
  - `hold_direct_nonpositive`: `64`
  - `hold_absolute_median`: `22`
  - `hold_direct_win_fraction`: `22`
  - `hold_absolute_win_fraction`: `9`
- Top joined row: `v025_vel008_stab20_lr12` / `prefix_hold` / `high_energy_bins` / gain `2.0`
- Top row direct mean corr delta: `+0.00501554`
- Top row absolute mean corr delta: `+0.00130907`
- Top row absolute median corr delta: `-0.000150692`
- Top row absolute win fraction: `0.426230`
- Top row absolute status: `bad_baseline_rescue_dominated`

The practical next search should optimize robust absolute median/win behavior and avoid bad-baseline rescue, not merely maximize direct recurrence delta.

## Wide Coefficient Search v1

Spec:

`C:\Users\Andrew\AppData\Local\Temp\rafa_tokenburst\internal_phase_law_variant_specs_2026_05_07\internal_phase_law_wide_v1.json`

Run:

`C:\Users\Andrew\AppData\Local\Temp\rafa_tokenburst\internal_phase_law_objective_scout_wide_v1_2026_05_07\TRACK_REPORT.md`

This searched `32` candidate coefficient rows across precondition `0.10..0.40`, velocity `0.00..0.10`, and stability `1.25/2.00`, plus an inserted no-op baseline.

Result:

- Status: `no_locked_candidate`
- Best run: `wide_p040_v010_s125`
- Best decision: `hold_absolute_win_fraction`
- Best score: `0.00738541`
- Direct overall mean corr delta: `+0.00162075`
- Direct best row corr delta: `+0.00514391`
- Absolute non-bad mean corr delta: `+0.0000487168`
- Absolute non-bad corr win fraction: `0.540984`
- Matched absolute corr delta: `+0.00141144`
- Matched absolute status: `bad_baseline_rescue_dominated`

Wide joint-row diagnostic:

- Status: `no_joint_row_candidate`
- Joined rows: `384`
- Candidate rows: `0`
- Main failure counts:
  - `hold_absolute_nonpositive`: `146`
  - `hold_direct_nonpositive`: `128`
  - `hold_direct_win_fraction`: `51`
  - `hold_absolute_median`: `31`
  - `hold_absolute_win_fraction`: `28`
- Top joined row: `wide_p040_v010_s125` / `prefix_hold` / `all_bins` / gain `2.0`
- Top row direct mean corr delta: `+0.00514391`
- Top row absolute mean corr delta: `+0.00141144`
- Top row absolute median corr delta: `0.0`
- Top row absolute win fraction: `0.475410`
- Top row absolute status: `bad_baseline_rescue_dominated`

Interpretation:

The wider grid confirms the same shape at larger scale. More internal recurrence pressure increases direct variant-vs-noop metrics, but absolute audio remains non-robust. The next intervention should not simply increase `phase_law_precondition_gain` or `phase_law_velocity_mix`; it needs a different objective or law form that lifts median/win behavior outside bad-baseline bins.

## v2 Guard Search v1

Spec:

`C:\Users\Andrew\AppData\Local\Temp\rafa_tokenburst\internal_phase_law_variant_specs_2026_05_07\internal_phase_law_v2_guard_v1.json`

Run:

`C:\Users\Andrew\AppData\Local\Temp\rafa_tokenburst\internal_phase_law_objective_scout_v2_guard_v1_2026_05_07\TRACK_REPORT.md`

This searched `10` guarded candidates, plus an inserted no-op baseline. The new controls were still phase-only and default-off:

- `phase_law_consensus_mix`
- `phase_law_consensus_damping`
- `phase_law_median_guard`

Result:

- Status: `no_locked_candidate`
- Best run: `v2_cons006_damp050_guard020_p040_v010_s200`
- Best decision: `hold_absolute_win_fraction`
- Best score: `0.00566896`
- Direct overall mean corr delta: `+0.00130224`
- Absolute non-bad mean corr delta: approximately `+0.0000315`
- Absolute non-bad corr win fraction: `0.524590`
- Matched absolute status: `bad_baseline_rescue_dominated`

v2 joint-row diagnostic:

- Status: `no_joint_row_candidate`
- Joined rows: `120`
- Candidate rows: `0`
- Best atlas joint decision: `hold_absolute_nonpositive`

v2 failure atlas:

- Status: `atlas_built`
- Run count: `10`
- Family rows: `90`
- Joint status: `no_joint_row_candidate`
- Bad-baseline bin mean corr delta: `+0.0218711`
- Weak-baseline bin mean corr delta: `-0.000200563`
- Moderate-baseline bin mean corr delta: `-0.00814378`
- Good-baseline bin mean corr delta: `-0.0874728`

Interpretation:

The v2 guard reduced some absolute damage relative to the wide v1 grid, but it also reduced the direct recurrence effect and did not break the same failure mode. Consensus and median guards are useful diagnostics, not a promotion path yet. The negative result is important: the next law form should not merely damp the current phase-velocity pressure; it should change the recurrence objective or introduce a more local phase-law carrier that improves robust absolute median/win rows.

## v3 Local Search v1

Spec:

`C:\Users\Andrew\AppData\Local\Temp\rafa_tokenburst\internal_phase_law_variant_specs_2026_05_07\internal_phase_law_v3_local_v1.json`

Run:

`C:\Users\Andrew\AppData\Local\Temp\rafa_tokenburst\internal_phase_law_objective_scout_v3_local_v1_2026_05_07\TRACK_REPORT.md`

This searched `10` local carrier candidates, plus an inserted no-op baseline. The new controls were still phase-only and default-off:

- `phase_law_local_velocity_mix`
- `phase_law_local_coherence_damping`
- `phase_law_curvature_guard`

Result:

- Status: `no_locked_candidate`
- Best run: `v3_local008_raw004_damp050_curv035_p040_s200`
- Best decision: `hold_absolute_win_fraction`
- Best score: `0.0107100`
- Direct overall mean corr delta: `+0.00229787`
- Direct best-row mean corr delta: `+0.00711422`
- Absolute non-bad mean corr delta: approximately `+0.0001225`
- Absolute non-bad corr win fraction: `0.524590`
- Matched absolute status: `bad_baseline_rescue_dominated`

v3 joint-row diagnostic:

- Status: `no_joint_row_candidate`
- Joined rows: `120`
- Candidate rows: `0`
- Top-row decision: `hold_absolute_median`
- Top-row absolute win fraction: `0.475410`
- Top-row absolute status: `bad_baseline_rescue_dominated`

v3 failure atlas:

- Status: `atlas_built`
- Run count: `10`
- Family rows: `90`
- Joint status: `no_joint_row_candidate`
- Bad-baseline bin mean corr delta: `+0.0354147`
- Weak-baseline bin mean corr delta: `-0.000270799`
- Moderate-baseline bin mean corr delta: `-0.00950706`
- Good-baseline bin mean corr delta: `-0.0923390`

Interpretation:

The v3 local carrier is the strongest direct causal phase-law signal so far, but it is also a clearer non-promotion. It improves matched no-op recurrence more than v2, while concentrating the absolute gain in bad-baseline rescue and worsening weak/moderate/good baseline bins. This supports the claim that internal phase-law variables are causal, but it rejects the narrower hypothesis that simply adding local phase-velocity carrier pressure solves robust no-future audio continuation.

## Strict Same-Row Joint Objective

After the v3 result, the objective scripts were tightened to surface non-bad baseline-bin preservation directly:

- `score_internal_phase_law_objective.py` now adds matched-row non-bad and good-bin fields and penalizes those regressions.
- `diagnose_internal_phase_law_joint_rows.py` now reports bad/weak/moderate/good bin deltas per joined row.
- `run_internal_phase_law_objective_scout.py` now emits joint-row diagnostics automatically after absolute/direct compares.

Strict v3 rerun:

- Objective status: `no_locked_candidate`
- Objective best run: `v3_local010_damp050_curv050_p040_v000_s125`
- Objective best decision: `hold_absolute_median`
- Joint-row status: `no_joint_row_candidate`
- Joined rows: `120`
- Candidate rows: `0`
- Mean joined-row non-bad bin corr delta: `-0.00290468`
- Decision counts: `hold_absolute_nonbad_bin_regression=48`, `hold_absolute_nonpositive=26`, `hold_direct_nonpositive=40`, `hold_direct_win_fraction=6`

Interpretation:

The stricter joint objective makes the next step unambiguous. Some rows preserve non-bad bins but have negative direct recurrence. Rows with positive direct recurrence regress non-bad bins. The next experiment should search for a law that satisfies both on the same row, rather than accepting separate direct and absolute wins from different slices.

## v4 Joint/Nonbad Search v1

Run root:

`C:\Users\Andrew\AppData\Local\Temp\rafa_tokenburst\internal_phase_law_objective_scout_v4_joint_nonbad_v1_2026_05_07`

The v4 scout made the same-row gate explicit around joint candidate rows and nonbad baseline-bin preservation. It searched `12` runs and produced `144` joined rows.

Objective result:

- Status: `no_locked_candidate`
- Best run: `v4_local005_damp020_curv025_p025_s150_lr4`
- Best decision: `hold_absolute_win_fraction`
- Best score: `-0.4853986778133745`
- Direct overall mean corr delta: `+0.0014133001979746507`
- Direct best-row mean corr delta: `+0.00410316475423101`
- Matched nonbad bin corr delta: `-0.011955829830617834`
- Matched good bin corr delta: `-0.22175256182054173`

Joint-row diagnostic:

- Status: `no_joint_row_candidate`
- Candidate rows: `0`
- Joined rows: `144`
- Mean absolute nonbad-bin corr delta: `-0.0040797997254572544`

Best joint row by the diagnostic:

- Run: `v4_local004_damp010_curv010_p000_s125_lr0`
- Row: `flat` / `low_energy_bins` / gain `2.0`
- Direct mean corr delta: `-0.00000683905271107356`
- Absolute mean corr delta: `+0.0001417974014295602`
- Absolute nonbad bin corr delta: `+0.00009552880734839431`
- Weak bin corr delta: `+0.00004287588986119856`
- Moderate bin corr delta: `+0.000806343193425537`
- Good bin corr delta: `0.0`
- Absolute corr win fraction: `0.5573770491803278`

Failure atlas:

- Status: `atlas_built`
- Joint status: `no_joint_row_candidate`
- Run count: `12`
- Family row count: `108`
- Bad-baseline bin mean corr delta: `+0.040754753013395`
- Bad-baseline win fraction: `0.558974358974359`
- Weak-baseline bin mean corr delta: `-0.00036189144602898705`
- Weak-baseline win fraction: `0.4415807560137457`
- Moderate-baseline bin mean corr delta: `-0.012559782386438872`
- Moderate-baseline win fraction: `0.4477317554240631`
- Good-baseline bin mean corr delta: `-0.12593373735696445`
- Good-baseline win fraction: `0.5192307692307693`

Interpretation:

The v4 result completes the joint/nonbad scout as a negative promotion read. The best objective run keeps a positive direct recurrence effect, but the same matched row regresses nonbad and good baseline bins. The best diagnostic joint row goes the other way: it has small positive absolute/nonbad-bin deltas, but its direct recurrence delta is slightly negative. The failure atlas confirms the old pattern at higher specificity: the apparent absolute gain is concentrated in bad-baseline rows, while weak, moderate, and good baseline bins remain negative on mean. This strengthens `internal_phase_law_influence` as a causal-variable claim and rejects the current v4 law family as robust no-future audio continuation evidence.

## v5 Low-Energy Target Row v1

Run root:

`C:\Users\Andrew\AppData\Local\Temp\rafa_tokenburst\internal_phase_law_objective_scout_v5_low_energy_row_v1_2026_05_07`

The v5 scout narrowed the search to the exact v4 near-miss row family: `flat` / `low_energy_bins` / `raw` / gain `2.0`.

Objective result:

- Status: `no_locked_candidate`
- Best global run: `v5_lowrow_local004_damp010_curv010_p000_s125_lr2`
- Best global decision: `hold_absolute_win_fraction`
- Best global score: `-0.563689912926062`
- Target-row candidate count: `0` / `12`
- Target-row best run: `v5_lowrow_local004_damp005_curv005_p000_s125_lr0`
- Target-row best decision: `hold_target_direct_win_fraction`
- Target-row direct mean corr delta: `+0.0000010155743203960363`
- Target-row direct median corr delta: `0.0`
- Target-row direct win fraction: `0.47540983606557374`
- Target-row absolute mean corr delta: `+0.00014965202846102982`
- Target-row absolute win fraction: `0.5573770491803278`
- Target-row nonbad bin corr delta: `+0.00010164278219327388`
- Target-row good bin corr delta: `0.0`

Joint-row diagnostic:

- Status: `no_joint_row_candidate`
- Candidate rows: `0`
- Joined rows: `144`
- Mean absolute nonbad-bin corr delta: `-0.004646750836327035`
- Best target-row decision: `hold_direct_median`
- Best target-row direct win fraction: `0.47541`
- Best target-row absolute corr win fraction: `0.5573770491803278`

Failure atlas:

- Status: `atlas_built`
- Joint status: `no_joint_row_candidate`
- Run count: `12`
- Family row count: `108`
- Bad-baseline bin mean corr delta: `+0.04362277679952343`
- Weak-baseline bin mean corr delta: `-0.0004220056700717941`
- Moderate-baseline bin mean corr delta: `-0.014665772940473651`
- Good-baseline bin mean corr delta: `-0.1378854093109648`

Interpretation:

v5 crosses the target row from v4's direct mean `-6.839e-06` to `+1.016e-06` while preserving positive absolute/nonbad deltas. That means the low-energy ridge is real and controllable. It still does not promote: the row fails direct median/win robustness, and the global run remains bad-baseline-rescue shaped with weak/moderate/good baseline regressions. The next search should target target-row direct median/win fraction, not just direct mean.

## v6 Low-Energy Win Search v1

Run root:

`C:\Users\Andrew\AppData\Local\Temp\rafa_tokenburst\internal_phase_law_objective_scout_v6_low_energy_win_v1_2026_05_07`

The v6 scout narrowed around the v5 ridge with lighter guards and small raw local-velocity variants, explicitly trying to improve the exact `flat` / `low_energy_bins` / `raw` / gain `2.0` target-row win fraction.

Objective result:

- Status: `no_locked_candidate`
- Best global run: `v6_lowwin_local004_raw003_damp003_curv003_p000_s125_lr0`
- Best global decision: `hold_matched_bad_baseline_dominated`
- Best global score: `-0.565501948626719`
- Target-row candidate count: `0` / `14`
- Target-row best run: `v6_lowwin_local004_raw003_damp003_curv003_p000_s125_lr0`
- Target-row best decision: `hold_target_direct_win_fraction`
- Target-row direct mean corr delta: `+0.000008544895570530304`
- Target-row direct median corr delta: `+0.000000196153106327146`
- Target-row direct win fraction: `0.5081967213114754`
- Target-row absolute mean corr delta: `+0.00015718134971116408`
- Target-row absolute median corr delta: `+0.00001466761477929594`
- Target-row absolute win fraction: `0.5901639344262295`
- Target-row nonbad bin corr delta: `+0.0001230016167545306`
- Target-row nonbad bin win fraction: `0.5862068965517241`
- Target-row good bin corr delta: `0.0`

Joint-row diagnostic:

- Status: `no_joint_row_candidate`
- Candidate rows: `0`
- Joined rows: `168`
- Mean absolute nonbad-bin corr delta: `-0.0047463987725493975`
- Best target-row decision: `hold_direct_win_fraction`
- Best target-row direct win fraction: `0.5081967213114754`
- Best target-row absolute corr win fraction: `0.5901639344262295`

Failure atlas:

- Status: `atlas_built`
- Joint status: `no_joint_row_candidate`
- Run count: `14`
- Family row count: `126`
- Bad-baseline bin mean corr delta: `+0.04729170141675616`
- Weak-baseline bin mean corr delta: `-0.00043647997738324154`
- Moderate-baseline bin mean corr delta: `-0.015318471193859833`
- Good-baseline bin mean corr delta: `-0.1380008551313037`

Interpretation:

v6 is a better near-miss than v5, not a promotion. The exact target row now has positive direct mean, positive direct median, positive absolute mean, positive nonbad-bin mean, and zero good-bin damage. The remaining target failure is direct breadth: `31/61` cases win, short of the `>=0.55` gate, which would require about three more positive cases. The global objective still rejects the run because the best global row is matched-bad-baseline dominated and the atlas still shows bad-baseline rescue with weak/moderate/good regressions. The next scout should not broaden the mean search; it should run a target-row gain ladder plus sign-flip/family leave-one-out diagnostics around the `31/61` win surface.

## v7 Low-Energy Gain Ladder v1

Run root:

`C:\Users\Andrew\AppData\Local\Temp\rafa_tokenburst\internal_phase_law_objective_scout_v7_low_energy_gain_ladder_v1_2026_05_07`

The v7 scout kept only the target row family, expanded gains to `1.0`, `1.25`, `1.5`, `1.75`, and `2.0`, and added a reusable target-flip diagnostic.

Objective and joint result:

- Status: `no_locked_candidate`
- Best global run: `v7_gain_local004_raw003_damp002_curv002_s100`
- Best global decision: `hold_direct_too_small`
- Direct variant status: `variant_tradeoff_or_tiny_signal`
- Joint status: `no_joint_row_candidate`
- Candidate rows: `0`
- Joined rows: `45`
- Joint decision counts: `hold_direct_median=37`, `hold_direct_win_fraction=8`

Best target-row result:

- Target-row best run: `v7_gain_local0045_raw003_damp003_curv003_s125`
- Target-row best decision: `hold_target_direct_win_fraction`
- Target-row direct mean corr delta: `+0.000010019920859155829`
- Target-row direct median corr delta: `+0.00000019353332522242112`
- Target-row direct win fraction: `0.5081967213114754`
- Target-row absolute mean corr delta: `+0.0001586563749997896`
- Target-row absolute win fraction: `0.5901639344262295`
- Target-row nonbad bin corr delta: `+0.00012720936557853514`
- Target-row good bin corr delta: `0.0`

Target-flip diagnostic selected run:

- Selected run: `v7_gain_local0035_raw003_damp003_curv003_s125`
- Target gain mean / median / wins: `+0.000008438469877048604` / `+0.0000002237293352498962` / `0.5081967213114754`
- Target gain cases: `31` positive, `28` negative, `2` zero
- Sign categories across gains: `29` positive at all gains, `30` nonpositive at all gains, `2` rescued by gain

Gain-ladder winners:

| Gain | Best run | Mean corr d | Median corr d | Wins |
|---:|---|---:|---:|---:|
| `1.0` | `v7_gain_local004_raw0035_damp003_curv003_s125` | `+0.0000049400770134492145` | `0.0` | `0.4918032786885246` |
| `1.25` | `v7_gain_local004_raw002_damp003_curv003_s125` | `+0.000002227903213601102` | `+0.00000012257861355113864` | `0.5081967213114754` |
| `1.5` | `v7_gain_local004_raw002_damp003_curv003_s125` | `+0.000002526547967657036` | `+0.000000021853364617907545` | `0.5081967213114754` |
| `1.75` | `v7_gain_local0035_raw003_damp003_curv003_s125` | `+0.000006134412992520064` | `+0.0000001958828286458339` | `0.5081967213114754` |
| `2.0` | `v7_gain_local0035_raw003_damp003_curv003_s125` | `+0.000008438469877048604` | `+0.0000002237293352498962` | `0.5081967213114754` |

Family pressure at gain `2.0`:

- Strong positive families: `saw_chain_tool_motor` wins `0.7777777777777778`, `combustion_engine_control` wins `0.8333333333333334`
- Persistent drag families: `buzzy_synth_control` wins `0.2`, `household_appliance_motor` wins `0.2`, `steady_buzz_nonmotor_control` wins `0.3`
- Leave-one-out excluding `steady_buzz_nonmotor_control` reaches win fraction `0.5490196078431373`, just under the `0.55` target gate.

Interpretation:

v7 says the target-row problem is not mainly gain amplitude. Increasing gain improves direct mean and MSE but does not move breadth past `31/61`; only two cases flip from nonpositive to positive as gain rises. The decisive seam is family/case structure: buzz/synth and some steady-buzz cases pin the win fraction, while saw/chain and combustion engine cases already work. The next useful experiment should be a family-conditional phase-law or support-mask diagnostic, not another global gain ladder.

## v8 Family/Support-Mask Scout v1

Run root:

`C:\Users\Andrew\AppData\Local\Temp\rafa_tokenburst\internal_phase_law_objective_scout_v8_family_support_mask_v1_2026_05_07`

The v8 scout tested whether the v7 failure was really a global phase-law coefficient problem or a support-geometry problem. It reused the v7/v6 ridge variants, held the mechanism to `raw`, held mode to `flat`, and compared `all_bins`, `high_energy_bins`, and `low_energy_bins` across gains `1.25`, `1.5`, `1.75`, and `2.0`.

Objective and joint result:

- Status: `no_locked_candidate`
- Direct variant status: `variant_improved`
- Best global run: `v8_mask_local004_raw003_damp002_curv002_s100`
- Best global decision: `hold_matched_bad_baseline_dominated`
- Joint status: `no_joint_row_candidate`
- Candidate rows: `0`
- Joined rows: `60`
- Joint decision counts: `hold_direct_win_fraction=7`, `hold_direct_median=13`, `hold_absolute_nonbad_bin_regression=20`, `hold_absolute_nonpositive=20`

Best target-row result:

- Target-row best run: `v8_mask_local0045_raw003_damp003_curv003_s125`
- Target-row best decision: `hold_target_direct_win_fraction`
- Target-row direct mean corr delta: `+0.000010019920859155829`
- Target-row direct median corr delta: `+0.00000019353332522242112`
- Target-row direct win fraction: `0.5081967213114754`
- Target-row absolute mean corr delta: `+0.0001586563749997896`
- Target-row absolute win fraction: `0.5901639344262295`
- Target-row nonbad bin corr delta: `+0.00012720936557853514`
- Target-row good bin corr delta: `0.0`

Family/support-mask diagnostic:

- Diagnostic: `C:\Users\Andrew\AppData\Local\Temp\rafa_tokenburst\internal_phase_law_family_mask_diagnostics_v8_family_support_mask_v1_2026_05_07\internal_phase_law_family_mask_diagnostics.json`
- Selected run: `v8_mask_local0045_raw003_damp003_curv003_s125`
- Selected-run basis: `objective-score target_low_energy_best_run`
- Best direct row: `v8_mask_local0045_raw003_damp003_curv003_s125` / `high_energy_bins` / gain `1.75`
- Best direct row mean / median / wins: `+0.0008355768415039223` / `+0.00008620824849370776` / `0.6557377049180327`

Selected-run mask comparison at gain `2.0`:

| Mask | Mean corr d | Median corr d | Wins | Positives | Negatives |
|---|---:|---:|---:|---:|---:|
| `all_bins` | `+0.001042980210303603` | `+0.0001348094873312427` | `0.6229508196721312` | `38` | `21` |
| `high_energy_bins` | `+0.0009958627140724337` | `+0.00009632557031789531` | `0.6229508196721312` | `38` | `21` |
| `low_energy_bins` | `+0.000010019920859155829` | `+0.00000019353332522242112` | `0.5081967213114754` | `31` | `28` |

Best mask by family at gain `2.0`:

| Family | Best mask | Mean corr d | Median corr d | Wins |
|---|---|---:|---:|---:|
| `buzzy_synth_control` | `high_energy_bins` | `+0.0007855063383835646` | `+0.0008296637080098847` | `0.6` |
| `combustion_engine_control` | `low_energy_bins` | `+0.0000154264523253197` | `+0.0000012900653072489034` | `0.8333333333333334` |
| `generic_electric_motor` | `low_energy_bins` | `-0.00000571850433687569` | `-0.0000009346316547920175` | `0.25` |
| `household_appliance_motor` | `high_energy_bins` | `-0.00031332153088286477` | `-0.00000009547477701488116` | `0.4` |
| `hvac_fan_airflow_motor` | `high_energy_bins` | `+0.00007388162892371045` | `+0.00007388162892371045` | `1.0` |
| `rotary_tool_motor` | `high_energy_bins` | `-0.00022384268996643482` | `+0.00008868328559227833` | `0.6` |
| `saw_chain_tool_motor` | `low_energy_bins` | `+0.00008049787903602774` | `+0.00001812488389282569` | `0.7777777777777778` |
| `steady_buzz_nonmotor_control` | `all_bins` | `+0.006303019471503994` | `+0.00148073201327438` | `1.0` |
| `typewriter_nonmotor_control` | `all_bins` | `-0.0001335326319159793` | `+0.00007166490254153146` | `0.5` |

Interpretation:

v8 proves support geometry is a real separable lever. The low-energy target row preserves the v7 near-miss, but broader `all_bins`/`high_energy_bins` masks strongly improve direct recurrence breadth, reaching `0.62295` at gain `2.0` and `0.65574` at high-energy gain `1.75`. This is not a promotion: the locked objective still rejects all rows because global direct improvements are paired with absolute/nonbad-bin regression or bad-baseline dominance. The next step is not another global mask. It is either family-conditional support selection or a learned support head that can route saw/chain and combustion toward low-energy support while routing steady-buzz, buzzy-synth, HVAC, and rotary cases toward all/high-energy support without hand-labeling families.

Support-router oracle:

`C:\Users\Andrew\AppData\Local\Temp\rafa_tokenburst\internal_phase_law_support_router_oracle_v8_family_support_mask_v1_2026_05_07\internal_phase_law_support_router_oracle.json`

| Router | Mean corr d | Median corr d | Wins |
|---|---:|---:|---:|
| `low_energy_fixed` | `+0.000010019920859155829` | `+0.00000019353332522242112` | `0.5081967213114754` |
| `global_fixed_best` (`all_bins`) | `+0.001042980210303603` | `+0.0001348094873312427` | `0.6229508196721312` |
| `family_oracle_router` | `+0.0010434926049990685` | `+0.00002535188285181407` | `0.7049180327868853` |
| `case_oracle_router` | `+0.0016661391383341617` | `+0.00014526865731809366` | `0.8032786885245902` |
| `case_anti_oracle` | `-0.0006191477310165207` | `-0.000015438286774254947` | `0.3442622950819672` |

Oracle interpretation:

Family-labeled routing is not an acceptable runtime mechanism, but it is a useful ceiling diagnostic. It raises direct win breadth from `0.62295` to `0.70492`; per-case oracle routing raises it to `0.80328`. That justifies a v9 learned/local support-router experiment. The router should be conditioned on phase/signature/support observables, not filename family labels.

## v9 Phase-Only Mask Scout v1

Run root:

`C:\Users\Andrew\AppData\Local\Temp\rafa_tokenburst\internal_phase_law_objective_scout_v9_phase_masks_v1_2026_05_07`

This assay adds default-off prefix-phase-derived mask modes to the audio delta probe:

- `phase_stable_bins`
- `phase_dynamic_bins`
- `phase_low_motion_bins`

These masks use prefix STFT phase velocity only. They do not use future target phase or future target magnitude. Existing `all_bins`, `high_energy_bins`, and `low_energy_bins` defaults are unchanged.

Objective result:

- Status: `no_locked_candidate`
- Best global run: `v8_mask_local004_raw003_damp002_curv002_s100`
- Best global decision: `hold_absolute_win_fraction`
- Direct variant status: `variant_tradeoff_or_tiny_signal`
- Absolute compare status: `tradeoff_signal_only`

Joint-row diagnostic:

- Status: `candidate_joint_row_found`
- Candidate rows: `11`
- Joined rows: `60`
- Decision counts: `candidate_joint_row=11`, `hold_absolute_win_fraction=29`, `hold_direct_win_fraction=19`, `hold_absolute_nonpositive=1`
- Mean absolute nonbad-bin corr delta: `-0.0005610993530122514`

Best joint row:

- Run: `v8_mask_local004_raw003_damp002_curv002_s100`
- Mask/gain: `phase_stable_bins` / `2.0`
- Direct mean / median / wins: `+0.00031847714441993473` / `+0.000046031686952028444` / `0.639344262295082`
- Absolute mean / median / wins: `+0.002700202968841946` / `+0.00007090066385779387` / `0.5737704918032787`
- Nonbad / weak / moderate / good corr d: `+0.0008055606836901082` / `+0.00048060491060858017` / `+0.005192463620290737` / `0.0`

Explicit phase-stable target rerun:

- Objective artifact: `C:\Users\Andrew\AppData\Local\Temp\rafa_tokenburst\internal_phase_law_objective_score_v9_phase_stable_target_v1_2026_05_07\internal_phase_law_objective_score.json`
- Joint artifact: `C:\Users\Andrew\AppData\Local\Temp\rafa_tokenburst\internal_phase_law_joint_rows_v9_phase_stable_target_v1_2026_05_07\internal_phase_law_joint_row_diagnostics.json`
- Target row: `flat` / `phase_stable_bins` / `raw` / gain `2.0`
- Target objective candidates: `2` / `5`
- Target objective best run / decision / score: `v8_mask_local004_raw003_damp002_curv002_s100` / `target_row_candidate` / `0.009769155093884003`
- Target objective direct / absolute / nonbad / good corr d: `+0.00031847714441993473` / `+0.002700202968841946` / `+0.0008055606836901082` / `0.0`
- Target joint candidates: `2` / `5`
- Target joint best run / decision / score: `v8_mask_local004_raw003_damp002_curv002_s100` / `candidate_joint_row` / `0.009428266037978807`
- Target joint direct / absolute / nonbad / good corr d: `+0.00031847714441993473` / `+0.002700202968841946` / `+0.0008055606836901082` / `0.0`
- Candidate-class diagnostic: `0` strict audio candidates, `0` phase-support candidates.

Interpretation:

v9 is the strongest conceptual result in this internal phase-law sequence so far: a prefix-phase-only support mask creates same-row joint candidates with positive direct recurrence, positive absolute carrier-beating signal, positive nonbad bins, and no good-bin damage. The scorer and joint diagnostic now support an explicit generic target row; when locked to `phase_stable_bins/gain2`, the target row itself clears the row-level candidate criteria. This is still not a checkpoint promotion because the global run objective remains `no_locked_candidate` / `hold_absolute_win_fraction`, so promotion would overstate the evidence. The next code step is a learned/local phase-support router or phase-stable-focused coefficient scout that preserves this target-row win while improving global row selection.

## v10 Phase-Router Mask Scout v1

Run root:

`C:\Users\Andrew\AppData\Local\Temp\rafa_tokenburst\internal_phase_law_objective_scout_v10_phase_router_masks_v1_2026_05_07`

This assay keeps the v9 `phase_stable_bins` control and adds three runtime-legal prefix-phase masks:

- `phase_coherent_motion_bins`: stable phase drift with nonzero local motion
- `phase_curvature_bins`: high prefix phase-velocity curvature
- `phase_router_bins`: composite stable-motion-smoothness score

Objective and joint result:

- Objective status: `no_locked_candidate`
- Objective best run / decision / score: `v8_mask_local004_raw003_damp002_curv002_s100` / `hold_direct_too_small` / `0.0015328724081388613`
- Direct variant status: `variant_tradeoff_or_tiny_signal`
- Absolute compare status: `tradeoff_signal_only`
- Joint status: `candidate_joint_row_found`
- Joint candidates: `29` / `80`
- Decision counts: `candidate_joint_row=29`, `hold_absolute_win_fraction=30`, `hold_absolute_median=1`, `hold_direct_median=18`, `hold_direct_win_fraction=2`
- Candidate-class diagnostic: `0` strict audio candidates, `3` phase-support candidates, status `phase_support_candidate_found_not_promotion`

Best phase-router target row:

- Target row: `flat` / `phase_router_bins` / `raw` / gain `2.0`
- Target objective candidates: `3` / `5`
- Target joint candidates: `4` / `5`
- Best target run / decision / score: `v8_mask_local004_raw0035_damp003_curv003_s125` / `candidate_joint_row` / `0.012583116024110755`
- Direct mean / median / wins: `+0.0008050548815987055` / `+0.00004077381867189896` / `0.5901639344262295`
- Absolute mean / median / wins: `+0.0034083202721412876` / `+0.0001662880192602051` / `0.5901639344262295`
- Nonbad / good corr d: `+0.0011328893661807422` / `0.0`
- Top phase-support class row: `v8_mask_local004_raw003_damp002_curv002_s100` / `phase_router_bins` / gain `2.0`, direct `+0.0008209024010130947`, absolute `+0.003424167791555677`, nonbad `+0.001149618247286744`, good `0.0`

Mask-level joint summary:

| mask | rows | candidate rows | mean direct d | mean abs d | mean nonbad d | best score |
|---|---:|---:|---:|---:|---:|---:|
| `phase_router_bins` | `20` | `9` | `+0.0006856990808860795` | `+0.0027361828495532795` | `+0.0008377215672535404` | `0.012583116024110755` |
| `phase_coherent_motion_bins` | `20` | `9` | `+0.0002890909103806428` | `+0.0021835996276617885` | `+0.0006214598904435091` | `0.00961552098200868` |
| `phase_stable_bins` | `20` | `11` | `+0.00028422445491278054` | `+0.0021338690971788627` | `+0.0006191903027520661` | `0.009428266037978807` |
| `phase_curvature_bins` | `20` | `0` | `+0.00006954827739293064` | `-0.0006863002510858044` | `-0.0006608255056265911` | `-0.004187756264041254` |

Interpretation:

v10 confirms that local prefix-phase support routing is a real RAFA variable. `phase_router_bins` beats the prior stable mask on best row score, direct mean, absolute mean, and nonbad mean while preserving zero good-bin damage. The candidate-class split is intentionally diagnostic, not promotional: v9 has no phase-support candidates under the stricter class, while v10 has `3` phase-support candidates and `0` strict audio candidates. The failure mode has changed: the best global run now fails as `hold_direct_too_small`, not as a nonbad/good-bin regression. That means promotion should not be declared, but the next scout can be narrower: either lower the direct-min threshold only after an explicit predeclared rationale, or better, search phase-router coefficients for `direct_best_mean_corr_delta >= 0.001` while preserving the current positive absolute/nonbad/good profile.

## v11 Phase-Router Direct Scout v1

Run root:

`C:\Users\Andrew\AppData\Local\Temp\rafa_tokenburst\internal_phase_law_objective_scout_v11_phase_router_direct_v1_2026_05_07`

This assay holds the support substrate fixed to `flat` / `phase_router_bins` / `raw` and searches a bounded coefficient neighborhood around the v10 winning rows. It uses gains `1.75`, `2.0`, and `2.25` to test whether the remaining blocker is just insufficient phase-router strength.

Objective and joint result:

- Objective status: `no_locked_candidate`
- Objective best run / decision / score: `v11_router_local004_raw003_damp002_curv002_s100` / `hold_direct_too_small` / `0.0024321854820790446`
- Objective target candidates: `12` / `12`
- Objective target best run / decision: `v11_router_local004_raw0035_damp003_curv003_s125` / `target_row_candidate`
- Objective target direct / absolute / nonbad corr d: `+0.0008050548815987055` / `+0.0034083202721412876` / `+0.0011328893661807422`
- Joint status: `candidate_joint_row_found`
- Joint candidates: `36` / `36`
- Joint best row / decision / score: `v11_router_local004_raw0035_damp002_curv002_s100` / `candidate_joint_row` / `0.013589180728752261`
- Joint best direct / absolute / nonbad / good corr d: `+0.0007964143239529634` / `+0.0037224731069172433` / `+0.0013637684558537445` / `0.0`
- Candidate-class diagnostic: `0` strict audio candidates, `19` phase-support candidates, status `phase_support_candidate_found_not_promotion`
- Top phase-support class row: `v11_router_local005_raw0035_damp002_curv002_s100` / `phase_router_bins` / gain `2.25`, direct `+0.0008021568076402359`, absolute `+0.003728215590604516`, nonbad `+0.0014298780535432387`, good `0.0`

Interpretation:

v11 strengthens the phase-router claim but does not promote audio. Narrowing the substrate turns the router surface into `36/36` same-row joint candidates and raises diagnostic phase-support candidates from v10's `3` to `19`, while strict audio candidates remain `0`. The direct effect does not cross `0.001`; stronger local/raw pressure mostly increases absolute and nonbad carrier-beating support. This falsifies the simplest "push the same coefficients harder" path. The next experiment should either predeclare a separate small-direct/strong-absolute diagnostic acceptance class or introduce a new phase-only mechanism that changes direct recurrence without merely adding support mass.

## v12 Phase-Reentry Direct Scout v1

Run root:

`C:\Users\Andrew\AppData\Local\Temp\rafa_tokenburst\internal_phase_law_objective_scout_v12_phase_reentry_direct_v1_2026_05_07`

This assay adds a default-off phase-only temporal reentry carrier to Circleworld. The carrier is derived from relative phase velocity plus wrapped acceleration, with no future magnitude path and no export post-shaper. Two rows isolate reentry without local/router assist; the rest combine reentry with the v11 router seam.

Objective and joint result:

- Objective status: `no_locked_candidate`
- Objective best run / decision / score: `v12_reentry_r008_a075_local004_raw0035_damp002_curv002_s100` / `hold_direct_too_small` / `0.0025618165983370958`
- Objective target candidates: `11` / `12`
- Objective target best run / decision: `v12_reentry_r008_a075_local004_raw0035_damp002_curv002_s100` / `target_row_candidate`
- Objective target direct / absolute / nonbad corr d: `+0.0008534610651987389` / `+0.003456726455741321` / `+0.0014933570819019395`
- Joint status: `candidate_joint_row_found`
- Joint candidates: `35` / `36`
- Joint best row / decision / score: `v12_reentry_r006_a050_local004_raw003_damp002_curv002_s100` / `candidate_joint_row` / `0.014176713694300075`
- Joint best direct / absolute / nonbad / good corr d: `+0.000868096871660527` / `+0.003794155654624807` / `+0.0016471319853362948` / `0.0`
- Candidate-class diagnostic: `0` strict audio candidates, `16` phase-support candidates, status `phase_support_candidate_found_not_promotion`
- Top phase-support class row: `v12_reentry_r008_a075_local004_raw0035_damp002_curv002_s100` / `phase_router_bins` / gain `2.25`, direct `+0.000879655278537222`, absolute `+0.0038057140615015018`, nonbad `+0.001701117585137462`, good `0.0`

Interpretation:

v12 slightly improves the direct target row over v11 and improves nonbad carrier-beating support, but it does not break the direct-size ceiling. It also reduces diagnostic phase-support rows from `19` to `16`, so temporal reentry is not a simple monotonic improvement over router pressure. The result supports a separate RAFA variable claim, "phase-derived temporal reentry can modulate recurrence," but still rejects audio/checkpoint promotion. The next experiment should be either a causality-clean reentry variant without centered local smoothing, or a predeclared diagnostic class for small-direct / strong-absolute rows rather than another coefficient ladder.

## v13 Causal Phase-Reentry Direct Scout v1

Run root:

`C:\Users\Andrew\AppData\Local\Temp\rafa_tokenburst\internal_phase_law_objective_scout_v13_causal_reentry_direct_v1_2026_05_07`

This assay repeats the v12 reentry question with `phase_law_reentry_causal=True`, which uses a zero boundary and past/current relative phase velocity for the reentry carrier. It separates the temporal-reentry variable from the v12 boundary lookahead concern.

Contract audit:

- Script: `D:\RAFA\runtimes\circleworld_proto\test_internal_phase_law_reentry_contract.py`
- Artifact: `C:\Users\Andrew\AppData\Local\Temp\rafa_tokenburst\internal_phase_law_reentry_contract_2026_05_07\internal_phase_law_reentry_contract.json`
- Status: `pass`
- Checks: default no-op, nonzero reentry delta, unit-phasor preservation, causal first-bin zero boundary, and noncausal first-bin contrast.

Objective and joint result:

- Objective status: `no_locked_candidate`
- Objective best run / decision / score: `v13_causal_reentry_r008_a075_local004_raw0035_damp002_curv002_s100` / `hold_direct_too_small` / `0.002566547525920997`
- Objective target candidates: `5` / `10`
- Objective target best run / decision: `v13_causal_reentry_r008_a075_local004_raw0035_damp002_curv002_s100` / `target_row_candidate`
- Objective target direct / absolute / nonbad corr d: `+0.00085477891809921` / `+0.003458044308641792` / `+0.0014958846296200376`
- Joint status: `candidate_joint_row_found`
- Joint candidates: `24` / `30`
- Joint best row / decision / score: `v13_causal_reentry_r006_a050_local004_raw003_damp002_curv002_s100` / `candidate_joint_row` / `0.014182578728878074`
- Joint best direct / absolute / nonbad / good corr d: `+0.0008692742414680994` / `+0.003795333024432379` / `+0.0016491977007333555` / `0.0`
- Candidate-class diagnostic: `0` strict audio candidates, `12` phase-support candidates, status `phase_support_candidate_found_not_promotion`
- Top phase-support class row: `v13_causal_reentry_r008_a075_local004_raw0035_damp002_curv002_s100` / `phase_router_bins` / gain `2.25`, direct `+0.000881642325190853`, absolute `+0.0038077011081551326`, nonbad `+0.0017039658929137432`, good `0.0`

Interpretation:

v13 preserves the best-row v12 signal under a stricter causal reentry carrier, so the reentry effect is not merely a first-bin boundary artifact. It is weaker in breadth than v12: joint candidates drop from `35/36` to `24/30`, target objective candidates drop from `11/12` to `5/10`, and phase-support class rows drop from `16` to `12`. This is clean evidence for a causal phase-only temporal-reentry variable, but still not audio viability. The next decision is objective design: either create a predeclared small-direct / strong-absolute diagnostic class, or stop the internal audio-law ladder and move to learned dense signatures/operators.

## Predeclared Small-Direct / Strong-Absolute Diagnostic Class

Artifact:

`C:\Users\Andrew\AppData\Local\Temp\rafa_tokenburst\internal_phase_law_predeclared_diagnostic_class_v10_v13_2026_05_07\internal_phase_law_predeclared_diagnostic_class.json`

This report freezes the diagnostic class that the v10-v13 sequence had been pointing toward. It is explicitly not a checkpoint/audio promotion class.

Definition:

- Class: `small_direct_strong_absolute_nonbad_not_promotion`
- Direct mean corr delta `>= 0.00075`
- Direct median corr delta `> 0`
- Direct win fraction `>= 0.55`
- Absolute mean corr delta `>= 0.0025`
- Absolute median corr delta `> 0`
- Absolute win fraction `>= 0.55`
- Absolute nonbad-bin mean corr delta `>= 0.00075`
- Absolute good-bin mean corr delta `>= 0`
- Not bad-baseline dominated

Outcome:

- Status: `diagnostic_acceptance_found_not_promotion`
- Tracks: `4`
- Total rows: `182`
- Diagnostic rows: `50`
- Strict audio rows: `0`
- Best track by count: `v11_phase_router_direct`
- Track counts: v10 `3/80`, v11 `19/36`, v12 `16/36`, v13 `12/30`
- Best diagnostic row: `v13_causal_reentry_direct` / `v13_causal_reentry_r008_a075_local004_raw0035_damp002_curv002_s100` / gain `2.25`
- Best row direct / absolute / nonbad / good corr d: `+0.000881642325190853` / `+0.0038077011081551326` / `+0.0017039658929137432` / `0.0`

Interpretation:

This is the proper landing point for the internal audio-law ladder. We have a reproducible non-promotional diagnostic class with strong absolute/nonbad support and no good-bin damage, but strict audio remains zero. Further blind coefficient/reentry searches are low value unless the objective changes or the mechanism becomes learned/adaptive.
