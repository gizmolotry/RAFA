# RAFA Phase-Native Audio Reset Suite - 2026-05-17

## Status

Registered as a Circleworld testing-suite overlay, not a checkpoint promotion.

This reset suite exists to keep the phase-native audio claim honest: Circleworld
must improve future audio continuation from prefix-only phase/law state without
future target magnitude or future target phase leakage. Childworld ontology,
resonant memory, and dense-signature claims are not allowed to substitute for
audio-facing evidence in this lane.

## Registered Entry Point

- Suite runner: `D:\RAFA\runtimes\circleworld_proto\run_phase_native_audio_reset_suite.py`
- Runtime manifest key: `phase_native_audio_reset_suite_entrypoint`
- DAG overlay: `D:\RAFA\docs\architecture\INTERLINEAGE_DAG.md`
- Dry-run artifact: `D:\RAFA\outputs\circleworld_proto\phase_native_audio_reset_2026-05-17_dryrun\phase_native_audio_reset_suite.json`
- Dry-run report: `D:\RAFA\outputs\circleworld_proto\phase_native_audio_reset_2026-05-17_dryrun\PHASE_NATIVE_AUDIO_RESET_SUITE.md`
- Smoke artifact: `D:\RAFA\outputs\circleworld_proto\phase_native_audio_reset_2026-05-17_smoke_cpu\phase_native_audio_reset_suite.json`
- Smoke report: `D:\RAFA\outputs\circleworld_proto\phase_native_audio_reset_2026-05-17_smoke_cpu\PHASE_NATIVE_AUDIO_RESET_SUITE.md`
- Full lockbox manifest: `D:\RAFA\outputs\circleworld_proto\phase_native_audio_reset_lockbox_2026-05-17\audio_predeclared_lockbox_manifest.json`
- Full lockbox cases: `D:\RAFA\outputs\circleworld_proto\phase_native_audio_reset_lockbox_2026-05-17\audio_predeclared_lockbox_cases.json`
- Full lockbox suite artifact: `D:\RAFA\outputs\circleworld_proto\phase_native_audio_reset_2026-05-17_full_lockbox\phase_native_audio_reset_suite.json`
- Full lockbox suite report: `D:\RAFA\outputs\circleworld_proto\phase_native_audio_reset_2026-05-17_full_lockbox\PHASE_NATIVE_AUDIO_RESET_SUITE.md`
- Reentry guard score: `D:\RAFA\outputs\circleworld_proto\phase_native_audio_reentry_guard_2026-05-17_full_lockbox\phase_native_audio_reentry_guard_score.json`
- Phase-seed reentry ablation: `D:\RAFA\outputs\circleworld_proto\phase_native_audio_seed_reentry_ablation_2026-05-17_full_lockbox\audio_phase_seed_ablation.json`
- Anti-reentry shear probe: `D:\RAFA\outputs\circleworld_proto\phase_native_audio_reentry_shear_probe_2026-05-17_full_lockbox\audio_delta_mechanism_probe.json`
- Anti-reentry high-gain probe: `D:\RAFA\outputs\circleworld_proto\phase_native_audio_reentry_shear_highgain_2026-05-17_full_lockbox\audio_delta_mechanism_probe.json`
- Reentry oracle ceiling: `D:\RAFA\outputs\circleworld_proto\phase_native_audio_reentry_oracle_2026-05-17_full_lockbox\phase_native_audio_reentry_oracle_score.json`
- Target-normalized reentry audit: `D:\RAFA\outputs\circleworld_proto\phase_native_audio_reentry_metric_audit_2026-05-17_full_lockbox\phase_native_audio_reentry_metric_audit.json`
- Late-decorrelator probe: `D:\RAFA\outputs\circleworld_proto\phase_native_audio_reentry_decorrelator_probe_2026-05-17_full_lockbox\audio_delta_mechanism_probe.json`
- Oracle plus decorrelator: `D:\RAFA\outputs\circleworld_proto\phase_native_audio_reentry_oracle_plus_decorrelator_2026-05-17_full_lockbox\phase_native_audio_reentry_oracle_score.json`

## Suite Components

| Task | Purpose |
| --- | --- |
| `continuation_prefix_hold_copyphase` | No-future Circleworld continuation against copy-last waveform and prefix/copyphase carrier baselines using prefix-hold magnitude. |
| `continuation_flat_copyphase` | Same test under flat prefix-derived magnitude so wins cannot hide in prefix magnitude structure. |
| `delta_mechanism_probe_core` | Fixed prefix-only phase-delta mechanisms over all-bins, phase-stable, and phase-router masks, scored against gain-0 carriers. |

## Promotion Contract

A future executed suite is only a candidate if all are true:

- Future target magnitude is not reused or accessed.
- Future target phase is not reused or accessed.
- Circleworld beats copy-last waveform on mean correlation.
- Circleworld beats the prefix/copyphase carrier on mean correlation.
- Circleworld does not worsen mean MSE versus copy-last.
- Circleworld does not worsen loop autocorrelation or first-chunk reentry versus copy-last.
- Positive evidence is not dominated by a single outlier family or known bad-baseline rescue bin.

## Current Interpretation

The 2026-05-17 dry-run artifact is a registration record:

- It proves the commands, output namespace, and scorecard contract are wired.
- It does not execute the audio benchmarks.
- It does not alter the active Circleworld checkpoint.
- It does not reverse the prior finding that strict phase-native audio viability remains unproven.

The 2026-05-17 one-case CPU smoke run executed successfully and returned
`phase_native_audio_not_promotional`:

- All three child commands returned `ok`.
- Future-access flags were clean.
- Mean Circleworld correlation delta versus copy-last was positive on this
  single case: `+0.026863100`.
- Mean Circleworld correlation delta versus prefix/copyphase carrier was also
  positive: `+0.015549599`.
- Promotion still failed because mean MSE worsened versus copy-last
  (`+0.000024410`) and first-chunk reentry worsened (`+0.044303566`).

## Full Lockbox Result

The rebuilt 2026-05-17 lockbox selected `62` cases from `10` predeclared groups
with no unmet minimum groups:

- `generic_electric_motor`: `4`
- `hvac_fan_airflow_motor`: `2`
- `household_appliance_motor`: `5`
- `rotary_tool_motor`: `10`
- `saw_chain_tool_motor`: `9`
- `steady_buzz_nonmotor_control`: `10`
- `typewriter_nonmotor_control`: `4`
- `combustion_engine_control`: `12`
- `buzzy_synth_control`: `5`
- `razor_single_source_probe`: `1`

The full suite executed successfully and returned
`phase_native_audio_not_promotional`:

- Child commands: all `ok`
- Future-access clean: `true`
- Mean Circleworld corr delta versus copy-last: `+0.011047512`
- Mean Circleworld corr delta versus prefix/copyphase carrier: `+0.008853951`
- Mean Circleworld MSE delta versus copy-last: `-0.022043999`
- Mean Circleworld loop-peak delta versus copy-last: `-0.016873862`
- Mean Circleworld first-chunk reentry delta versus copy-last: `+0.046929410`

Interpretation: this is better than the prior pure-negative audio read on some
aggregate axes, but it still fails the predeclared promotion contract because
first-chunk reentry worsens. The result says Circleworld can move the waveform
metrics in a useful direction under this suite, but the current phase-native
law is still replay/reentry-prone.

The best fixed phase-delta mechanism row was:

- Magnitude mode: `prefix_hold`
- Mask: `phase_router_bins`
- Mechanism: `time_smooth_3`
- Gain: `2.0`
- Mean corr delta versus gain-0: `+0.024537976`
- Median corr delta versus gain-0: `+0.000027691`
- Corr win fraction: `0.516129032`
- Mean MSE delta versus gain-0: `-0.002924019`
- Verdict: `mechanism_tradeoff_signal`

The mechanism result remains a tradeoff signal, not a promotion signal. The
best-row mean is strongly helped by the single `razor_single_source_probe`
case and the `buzzy_synth_control` group:

- Best mechanism row excluding the single razor probe: mean corr delta
  `+0.010310973`
- Excluding both razor and buzzy-synth controls: mean corr delta `+0.001024004`

## Updated Next Step

The reentry guard scorer was added after the first full lockbox result. It
joins delta-probe rows back to the copy-last continuation baseline and evaluates
three views:

- `all`
- `no_razor`
- `no_razor_no_buzzy`

Current core-grid result:

- Status: `no_reentry_guard_candidate`
- Raw delta rows scored: `7440`
- Strict candidates: `0`
- Diagnostic candidates: `0`
- Best all-view row: `flat / phase_router_bins / time_smooth_3 / gain 2`
- Best all-view reentry delta versus copy-last: `+0.043523982`

This means the original mechanism grid does not contain a hidden row that
solves reentry while preserving the other gains.

## Phase-Seed Reentry Check

The phase-seed ablation was extended to report first-chunk reentry. The full
lockbox phase-seed run returned `circleworld_policy_helpful_candidate` under
its older seed-policy criteria, but no phase seed cleared the copy-last reentry
guard.

Best joined rows against copy-last:

- `flat / copy_last_waveform_phase`: corr `+0.012731`, MSE `-0.027952`,
  loop `-0.049407`, reentry `+0.045006`
- `prefix_hold / copy_last_waveform_phase`: corr `+0.009364`,
  MSE `-0.016136`, loop `+0.015659`, reentry `+0.048852`
- `prefix_hold / copy_last_waveform_phase_0_5s`: corr `+0.020873`,
  MSE `-0.016922`, loop `+0.099163`, reentry `+0.068152`

Interpretation: seed policy matters, but seed-policy selection does not remove
the first-chunk reentry failure. The bottleneck is now recurrence law, not only
initial phase carrier choice.

## Anti-Reentry Shear Probe

Three default-off phase-only mechanisms were added to
`run_audio_delta_mechanism_probe.py`:

- `anti_reentry_phase_shear`
- `anti_reentry_curvature_shear`
- `anti_reentry_delta_shear_mix`

These use prefix phase velocity/acceleration and centered frequency-relative
phase shear. They do not read future target magnitude or phase.

First shear probe:

- Native probe status: `mechanism_candidate_found`
- Best native row: `prefix_hold / phase_router_bins / anti_reentry_delta_shear_mix / gain 4`
- Mean corr delta versus gain-0: `+0.034494246`
- Median corr delta versus gain-0: `+0.001478723`
- Corr win fraction versus gain-0: `0.629032258`
- Mean MSE delta versus gain-0: `-0.004645073`
- Mean first-chunk reentry: `0.869580451`
- Copy-last reentry guard status: `no_reentry_guard_candidate`

High-gain shear follow-up:

- Native probe status: `mechanism_tradeoff_signal_only`
- Copy-last reentry guard status: `no_reentry_guard_candidate`
- Best all-view guard row: `prefix_hold / all_bins / anti_reentry_delta_shear_mix / gain 4`
- Corr delta versus copy-last: `+0.034595450`
- MSE delta versus copy-last: `-0.019902234`
- Loop delta versus copy-last: `-0.035967724`
- Reentry delta versus copy-last: `+0.042719970`

Interpretation: anti-reentry shear improves the shape of the result and reduces
reentry versus gain-0, but it does not beat copy-last on the actual reentry
contract. The next objective should not merely increase shear gain. It should
directly train or select against first-chunk reentry while preserving the
positive corr/MSE/loop movement found here.

## Reentry Oracle Ceiling

The oracle scorer combines the original mechanism grid, the anti-reentry shear
grid, and the high-gain shear grid. It then evaluates three ceilings:

- `global`: one row used for all cases
- `family_oracle`: one row selected per predeclared group
- `case_oracle`: one row selected per case

Combined pool:

- Delta rows scored: `27652`
- Future-access clean: `true`
- Global strict pass: `false`
- Family-oracle strict pass: `false`
- Case-oracle strict pass: `false`

Oracle summary:

| Selector | Corr-copy | MSE-copy | Loop-copy | Reentry-copy | Corr-gain0 | Reentry wins | Strict |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `global` | `+0.013784879` | `-0.028774500` | `-0.149937948` | `+0.031057793` | `+0.002382068` | `0.516129` | `false` |
| `family_oracle` | `+0.059819180` | `-0.026632472` | `-0.113919062` | `+0.029249033` | `+0.043565019` | `0.449438` | `false` |
| `case_oracle` | `+0.088663125` | `-0.029350330` | `-0.089331196` | `+0.019916199` | `+0.085040980` | `0.370968` | `false` |

Interpretation: selection/routing alone is not enough over the current
mechanism pool. The case oracle makes correlation much stronger while keeping
MSE and loop favorable, but first-chunk reentry remains positive on average.
This points to missing operator capacity or an overly strict copy-last
first-chunk floor, not merely poor row selection.

Group-level case-oracle failures are concentrated in:

- `combustion_engine_control`: reentry `+0.040864429`
- `household_appliance_motor`: reentry `+0.014988693`
- `hvac_fan_airflow_motor`: reentry `+0.022151619`
- `razor_single_source_probe`: reentry `+0.094772637`
- `saw_chain_tool_motor`: reentry `+0.080529393`
- `steady_buzz_nonmotor_control`: reentry `+0.091837488`

Group-level case-oracle passes:

- `buzzy_synth_control`
- `rotary_tool_motor`
- `typewriter_nonmotor_control`

Updated next step: do not spend the next cycle on a selector over the same
mechanism pool. Either introduce a genuinely new anti-reentry operator family,
or define a second metric that distinguishes harmful first-chunk replay from
lawful high-similarity continuation before treating copy-last as an absolute
reentry floor.

## Target-Normalized Reentry Metric Audit

The first-chunk reentry metric was audited against the target future's own
first-chunk reentry. This asks whether the generated future is more replay-like
than the real continuation, rather than merely self-similar.

Audit result:

- Status: `target_normalized_reentry_still_fails`
- Target margin: `0.01`
- Circleworld harmful replay excess: `0.088733691`
- Copy-last harmful replay excess: `0.081570061`
- Circleworld harmful excess delta versus copy-last: `+0.007163631`
- Circleworld target-normalized reentry delta versus copy-last:
  `+0.046573340`
- Circleworld first-reentry profile-error delta versus copy-last:
  `-0.029455195`
- Circleworld corr delta versus copy-last: `+0.011047512`
- Circleworld MSE delta versus copy-last: `-0.022044000`

Interpretation: the old self-only reentry metric is blunt, but the failure is
not just metric artifact. Circleworld actually matches the target reentry
profile better than copy-last on average, yet still has more target-normalized
harmful replay excess. So the correct objective is not simply "lower all
self-similarity"; it is "lower excess replay over target-like stationarity."

Highest Circleworld harmful replay groups:

- `generic_electric_motor`: `0.405575903`
- `hvac_fan_airflow_motor`: `0.258832415`
- `rotary_tool_motor`: `0.180664276`
- `typewriter_nonmotor_control`: `0.083405686`
- `steady_buzz_nonmotor_control`: `0.062459843`

## Late Decorrelator Operator Probe

A second anti-reentry operator family was added:

- `anti_reentry_late_decorrelator`
- `anti_reentry_staggered_decorrelator`
- `anti_reentry_delta_decorrelator_mix`

These are deterministic phase-only later-chunk decorrelators. They do not use
future target audio. They are designed to change later chunks more than the
first generated chunk.

Probe result:

- Native probe status: `mechanism_tradeoff_signal_only`
- Copy-last guard status: `no_reentry_guard_candidate`
- Oracle plus decorrelator status: `oracle_reentry_guard_not_possible`

Oracle plus decorrelator:

| Selector | Corr-copy | MSE-copy | Loop-copy | Reentry-copy | Corr-gain0 | Reentry wins | Strict |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `global` | `+0.013740669` | `-0.028926117` | `-0.120445500` | `+0.024479190` | `+0.002337858` | `0.564516` | `false` |
| `family_oracle` | `+0.075538419` | `-0.031631445` | `-0.126133131` | `+0.032483103` | `+0.054533595` | `0.452055` | `false` |
| `case_oracle` | `+0.097471967` | `-0.029267235` | `-0.110088807` | `+0.019799228` | `+0.095989805` | `0.387097` | `false` |

Interpretation: the late decorrelator improves the global oracle reentry delta
from about `+0.0311` to `+0.0245`, but it still does not cross the copy-last
guard. The case oracle remains positive on reentry. This keeps the next
intervention in "new objective/new operator" territory.

## Target Replay Oracle

The target-normalized replay oracle scores harmful replay excess relative to
the real future's own first-chunk reentry. This asks whether a continuation is
more replay-like than the target allows, instead of treating all self-similarity
as equally bad.

Combined pool:

- Delta rows scored: `42532`
- Future-access clean: `true`
- Target margin: `0.01`
- Global strict pass: `false`
- Family-oracle strict pass: `true`
- Case-oracle strict pass: `true`

Oracle summary:

| Selector | Corr-copy | MSE-copy | Loop-copy | Reentry-copy | Harm-delta | Harm wins | Corr-gain0 | Strict |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `global` | `+0.017670204` | `-0.028956036` | `-0.152669509` | `+0.034598151` | `+0.009799815` | `0.774194` | `+0.006267394` | `false` |
| `family_oracle` | `+0.055629949` | `-0.028378773` | `-0.123534194` | `+0.026987440` | `-0.002426999` | `0.790323` | `+0.046340320` | `true` |
| `case_oracle` | `+0.099344254` | `-0.027512528` | `-0.107029241` | `+0.026600461` | `-0.012536025` | `0.725806` | `+0.097727187` | `true` |

Interpretation: this is the first reentry-facing pass in the phase-native
audio reset lane, but it is an oracle pass, not a deployable runtime policy.
The old copy-last first-chunk guard remains unbeaten. Under a better
target-normalized harmful replay objective, however, the current phase-only
operator pool already contains enough capacity for family-level routing to
reduce harmful replay excess while preserving corr/MSE/loop gains.

Updated next step: train or predeclare a prefix-only family/router policy for
the target-normalized objective. The required policy should predict the
family-level mechanism choice without future target audio, then be rerun on the
same lockbox and on a fresh lockbox.

## Prefix Router Scout

The prefix-router scout evaluates whether the target-normalized route choices
can be approximated from no-future Circleworld/prefix metadata instead of being
selected directly from future metrics.

Artifact:

- `D:\RAFA\outputs\circleworld_proto\phase_native_audio_prefix_router_scout_2026-05-17_full_lockbox\phase_native_audio_prefix_router_scout.json`

Policy summary:

| Policy | Corr-copy | MSE-copy | Loop-copy | Harm-delta | Corr-gain0 | Family-key match | Group match | Strict |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `global_route` | `+0.017670204` | `-0.028956036` | `-0.152669509` | `+0.009799815` | `+0.006267394` | `0.000000` | n/a | `false` |
| `known_family_table` | `+0.055629949` | `-0.028378773` | `-0.123534194` | `-0.002426999` | `+0.046340320` | `1.000000` | n/a | `true` |
| `leave_one_case_nearest_case` | `+0.027219802` | `-0.023543982` | `-0.060132780` | `-0.006045332` | `+0.014257674` | `0.032258` | n/a | `true` |
| `leave_one_case_family_centroid` | `+0.025762483` | `-0.024899695` | `-0.141079893` | `-0.007402695` | `+0.013682832` | `0.209677` | `0.209677` | `true` |
| `leave_one_group_family_centroid` | `+0.004611739` | `-0.024811664` | `-0.118567405` | `-0.005991842` | `-0.012620061` | `0.000000` | `0.000000` | `false` |

Interpretation: this is stronger than the oracle-only result. A simple
prefix-feature nearest-case policy and a leave-one-case family-centroid policy
both pass the target-normalized replay guard without future target audio. The
known-family route table also passes. However, leave-one-group routing does not
pass because corr versus gain-0 is negative, so the result supports
within-family or nearby-case routing but not zero-shot routing to unseen
families.

Updated next step: freeze the target-normalized objective and rerun the
prefix-router scout on a fresh lockbox. If the leave-one-case policies hold,
promote a real runtime router experiment that selects a phase-only operator
from prefix/Circleworld metadata before rendering.

## Fresh Lockbox Validation

A second deterministic lockbox was built with seed `20260518` under:

- `D:\RAFA\artifacts\runtime\outputs\circleworld_proto\phase_native_audio_reset_lockbox_fresh_2026-05-17_seed20260518\audio_predeclared_lockbox_manifest.json`

The first fresh execution accidentally used the suite default of three cases;
that smoke-sized result is excluded from evidence. The full validation reran
with `--num-cases 0` over all `62` selected cases.

Fresh raw suite scorecard:

| Metric | Delta vs copy-last |
| --- | ---: |
| Corr | `+0.009390516` |
| MSE | `-0.024008087` |
| Loop peak | `-0.012624197` |
| First-chunk reentry | `+0.048045225` |

Fresh target replay oracle:

| Selector | Corr-copy | MSE-copy | Loop-copy | Reentry-copy | Harm-delta | Harm wins | Corr-gain0 | Strict |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `global` | `+0.035064643` | `-0.024325884` | `-0.014943761` | `+0.038904384` | `-0.002546027` | `0.548387` | `+0.036057077` | `true` |
| `family_oracle` | `+0.052350092` | `-0.031725382` | `-0.132691818` | `+0.026443936` | `-0.004493690` | `0.806452` | `+0.044687844` | `true` |
| `case_oracle` | `+0.104064632` | `-0.029962208` | `-0.095134848` | `+0.026016192` | `-0.014048789` | `0.725806` | `+0.106874552` | `true` |

Fresh prefix-router scout:

| Policy | Corr-copy | MSE-copy | Loop-copy | Harm-delta | Corr-gain0 | Strict |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `global_route` | `+0.035064643` | `-0.024325884` | `-0.014943761` | `-0.002546027` | `+0.036057077` | `true` |
| `known_family_table` | `+0.052350092` | `-0.031725382` | `-0.132691818` | `-0.004493690` | `+0.044687844` | `true` |
| `leave_one_case_nearest_case` | `+0.035841336` | `-0.026143814` | `-0.056563145` | `-0.006061551` | `+0.022911919` | `true` |
| `leave_one_case_family_centroid` | `+0.012611313` | `-0.028940959` | `-0.129697182` | `-0.011523022` | `+0.008595513` | `true` |
| `leave_one_group_family_centroid` | `-0.002372295` | `-0.028745080` | `-0.113574819` | `-0.008541711` | `-0.013298365` | `false` |

Interpretation: the target-normalized anti-replay result survives a fresh
62-case lockbox. The raw suite still fails the old first-chunk reentry guard,
but a fixed global phase-only route now passes target-normalized replay on the
fresh lockbox, and prefix-feature leave-one-case policies also pass. The
remaining failure is zero-shot unseen-family routing, not same-family or nearby
case routing.

## Route Transfer

Route transfer applies source-lockbox route choices to the opposite lockbox
without using target future metrics to choose the route.

Original-to-fresh transfer:

| Policy | Corr-copy | MSE-copy | Loop-copy | Harm-delta | Corr-gain0 | Strict |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `source_global_route` | `+0.015444601` | `-0.032189500` | `-0.146315001` | `+0.009288040` | `+0.005490868` | `false` |
| `source_family_table` | `+0.052371939` | `-0.031758902` | `-0.128221287` | `-0.004044815` | `+0.044709691` | `true` |

Fresh-to-original transfer:

| Policy | Corr-copy | MSE-copy | Loop-copy | Harm-delta | Corr-gain0 | Strict |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `source_global_route` | `+0.024191454` | `-0.021703119` | `-0.028892794` | `-0.001408176` | `+0.021997894` | `true` |
| `source_family_table` | `+0.054172195` | `-0.028264552` | `-0.123580557` | `-0.002011568` | `+0.044882567` | `true` |

Interpretation: bidirectional transfer is the strongest evidence so far in the
phase-native audio lane. The family route table transfers both directions, and
the fresh global route transfers back to the original lockbox. This still does
not promote Circleworld as an audio model: it is an evidence lane over rendered
mechanism rows, not yet a live runtime selector. But the separable claim now
has support:

> Prefix-only phase/Circleworld metadata can route phase-only continuation
> operators that reduce target-normalized harmful replay while preserving or
> improving corr/MSE/loop metrics against copy-last.

Updated next step: implement a live route-selected renderer that chooses the
global/family/prefix route before rendering, then run it as a first-class
continuation method against both lockboxes and a third fresh seed.

## Live Selected-Route Renderer

The route-selected renderer chooses a route before rendering and renders only
that selected phase-only continuation route per case. This is stronger than the
offline route-transfer scorer because the selected route is not filtered from a
pre-rendered grid after scoring.

Entrypoint:

- `D:\RAFA\runtimes\circleworld_proto\run_phase_native_audio_selected_route.py`

Selected-route runs:

| Run | Route source | Target lockbox | Policy | Corr-copy | MSE-copy | Loop-copy | Harm-delta | Corr-gain0 | Strict |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| `original_family_on_original` | original | original | `family_table` | `+0.055629949` | `-0.028378773` | `-0.123534194` | `-0.002426999` | `+0.046340320` | `true` |
| `fresh_global_on_fresh` | fresh | fresh | `global` | `+0.035064643` | `-0.024325884` | `-0.014943761` | `-0.002546027` | `+0.036057077` | `true` |
| `original_family_on_fresh` | original | fresh | `family_table` | `+0.052371939` | `-0.031758902` | `-0.128221287` | `-0.004044815` | `+0.044709691` | `true` |
| `fresh_global_on_original` | fresh | original | `global` | `+0.024191454` | `-0.021703119` | `-0.028892794` | `-0.001408176` | `+0.021997894` | `true` |
| `fresh_family_on_original` | fresh | original | `family_table` | `+0.054172195` | `-0.028264552` | `-0.123580557` | `-0.002011568` | `+0.044882567` | `true` |

Interpretation: the selected-route renderer confirms that the route-transfer
result is not merely a post-hoc scorer artifact. The selected route is chosen
before rendering, and the rendered outputs preserve the same target-normalized
harmful replay gains. The remaining caveat is that the route tables themselves
are still chosen from source-lockbox oracle summaries. This is enough to
promote a new controlled runtime experiment, but not enough to declare a fully
learned route policy.

Updated next step: freeze two candidate live policies for a third lockbox:

- `fresh_global_route_v1`: `prefix_hold / all_bins / anti_reentry_delta_shear_mix / gain=16`
- `family_transfer_route_v1`: source family table from the fresh lockbox

Then run both on a third predeclared seed before considering any branch/audio
promotion language.

## Third Lockbox Frozen-Policy Validation

A third deterministic lockbox was built with seed `20260519`. The raw reset
suite again remains non-promotional:

| Metric | Delta vs copy-last |
| --- | ---: |
| Corr | `+0.011266029` |
| MSE | `-0.021517834` |
| Loop peak | `+0.005860078` |
| First-chunk reentry | `+0.049837844` |

Frozen selected policies from the fresh lockbox source oracle:

| Policy | Status | Corr-copy | MSE-copy | Loop-copy | Reentry-copy | Harm-delta | Corr-gain0 | Strict |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `fresh_global_route_v1` | `selected_route_not_yet` | `+0.031369457` | `-0.021254905` | `+0.000883569` | `+0.039324156` | `-0.001083568` | `+0.029696283` | `false` |
| `family_transfer_route_v1` | `selected_route_target_replay_pass` | `+0.053683083` | `-0.029347210` | `-0.122391613` | `+0.029821102` | `-0.000650100` | `+0.042878462` | `true` |

Interpretation: the third lockbox sharpens the result. The frozen global route
still improves corr, MSE, harmful replay, and gain-0 correlation, but it fails
strict promotion because loop peak is slightly positive. The family-transfer
route passes again and preserves the broad pattern: positive correlation,
negative MSE, strongly negative loop peak, negative harmful replay excess, and
positive gain-0 correlation.

Current phase-native audio conclusion: family-conditioned phase-only routing is
the first robust Circleworld audio mechanism worth pursuing. It does not solve
raw self-reentry, but it does reduce target-normalized harmful replay while
improving ordinary waveform metrics across three predeclared lockboxes. The
next implementation should replace oracle-derived family tables with a learned
prefix-only route classifier/regressor and compare it against this frozen
family-transfer table.

## Learned Prefix-Only Route Policy

A low-capacity learned route-policy scout was added:

- `D:\RAFA\runtimes\circleworld_proto\train_phase_native_audio_route_policy.py`
- `D:\RAFA\runtimes\circleworld_proto\assemble_phase_native_audio_route_policy_comparison.py`

The learner uses source-lockbox family-oracle route labels, but predicts target
case routes from no-future Circleworld/prefix metadata. Three variants were
tested:

- `nearest_centroid_route_v1`
- `nearest_case_route_v1`
- `knn5_route_vote_v1`

Comparison artifact:

- `D:\RAFA\outputs\circleworld_proto\phase_native_audio_route_policy_comparison_2026-05-17\phase_native_audio_route_policy_comparison.json`

Summary:

| Policy | Target | Corr-copy | MSE-copy | Loop-copy | Harm-delta | Corr-gain0 | Strict |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| `centroid` | original-to-fresh | `+0.020875162` | `-0.029392042` | `-0.145548229` | `+0.006502477` | `+0.017488120` | `false` |
| `centroid` | fresh-to-original | `+0.030817598` | `-0.027671087` | `-0.135257362` | `-0.012670942` | `+0.024800261` | `true` |
| `centroid` | original+fresh-to-third | `+0.032495408` | `-0.028462442` | `-0.117508747` | `+0.012192046` | `+0.025226130` | `false` |
| `knn1` | original-to-fresh | `+0.040098573` | `-0.029805366` | `-0.117112479` | `-0.002424600` | `+0.032636162` | `true` |
| `knn1` | fresh-to-original | `+0.045135280` | `-0.027309710` | `-0.120571959` | `-0.001771148` | `+0.042333864` | `true` |
| `knn1` | original+fresh-to-third | `+0.047321097` | `-0.027827463` | `-0.113290591` | `+0.001174216` | `+0.038031896` | `false` |
| `knn5` | original-to-fresh | `+0.041278214` | `-0.029669492` | `-0.114237903` | `-0.002817824` | `+0.035436123` | `true` |
| `knn5` | fresh-to-original | `+0.044384340` | `-0.027249026` | `-0.123930160` | `-0.002157567` | `+0.041545709` | `true` |
| `knn5` | original+fresh-to-third | `+0.046431469` | `-0.026307447` | `-0.115702635` | `+0.001429030` | `+0.039411010` | `false` |
| frozen family table | original-to-fresh | `+0.052371939` | `-0.031758902` | `-0.128221287` | `-0.004044815` | `+0.044709691` | `true` |
| frozen family table | fresh-to-original | `+0.054172195` | `-0.028264552` | `-0.123580557` | `-0.002011568` | `+0.044882567` | `true` |
| frozen family table | fresh-to-third | `+0.053683083` | `-0.029347210` | `-0.122391613` | `-0.000650100` | `+0.042878462` | `true` |

Interpretation: low-capacity learned routers are not garbage. `knn1` and
`knn5` pass both original/fresh transfer directions and preserve the same
positive corr/MSE/loop pattern. They fail the third holdout only on
target-normalized harmful replay, with small positive misses. The frozen family
table remains stronger and passes all three lockboxes.

Updated conclusion: phase-native routing has crossed from "interesting
operator trick" to "separable testable component." The current robust component
is family-conditioned route selection. The pure learned prefix router is not
yet robust enough to replace the family table. Next work should either:

- add explicit no-future family identification as a supervised subtask, or
- train the route selector on more lockboxes before testing a fourth seed.

## No-Future Family Route Policy

A second learner made the family step explicit:

- `D:\RAFA\runtimes\circleworld_proto\train_phase_native_audio_family_route_policy.py`

This policy predicts a family from no-future Circleworld/prefix metadata, then
applies the frozen fresh-lockbox family route table. Three family predictors
were tested on the third lockbox:

- `group_knn1_v1`
- `group_knn5_v1`
- `group_centroid_v1`

Third-lockbox comparison:

| Policy | Group accuracy / note | Corr-copy | MSE-copy | Loop-copy | Harm-delta | Corr-gain0 | Strict |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| direct `knn1` route | direct route learner | `+0.047321097` | `-0.027827463` | `-0.113290591` | `+0.001174216` | `+0.038031896` | `false` |
| direct `knn5` route | direct route learner | `+0.046431469` | `-0.026307447` | `-0.115702635` | `+0.001429030` | `+0.039411010` | `false` |
| family `knn1` | train leave-one accuracy `0.645161` | `+0.046261102` | `-0.027823104` | `-0.117139295` | `+0.001378614` | `+0.036971900` | `false` |
| family `knn5` | train leave-one accuracy `0.629032` | `+0.044476714` | `-0.026807439` | `-0.117592664` | `+0.001039817` | `+0.036807871` | `false` |
| family `centroid` | train leave-one accuracy `0.217742` | `+0.034664080` | `-0.028444819` | `-0.107745053` | `-0.009128800` | `+0.026537374` | `true` |
| frozen fresh family table | oracle family table | `+0.053683083` | `-0.029347210` | `-0.122391613` | `-0.000650100` | `+0.042878462` | `true` |
| frozen fresh global route | single route | `+0.031369457` | `-0.021254905` | `+0.000883569` | `-0.001083568` | `+0.029696283` | `false` |

Interpretation: exact acoustic family identification is not the same as
operator-family routing. The kNN family classifiers have much better audit
accuracy, but still miss strict harmful replay on the third lockbox. The lower
accuracy centroid classifier passes because its "wrong" family assignments
route cases into a safer operator family basin. This is useful: the next
learned router should not optimize label accuracy alone. It should learn
operator-family assignment directly against target-normalized replay, while
retaining no-future feature constraints.

Updated next step: train a small objective-aware route classifier with labels
derived from source-lockbox route outcomes, not human/file family names. The
loss should include harmful replay excess, corr-gain0, MSE, and loop terms.

## Objective-Aware Route Policy

A third learner derived route labels from source-lockbox route outcomes rather
than human/file family names:

- `D:\RAFA\runtimes\circleworld_proto\train_phase_native_audio_objective_route_policy.py`

Training labels are selected from source lockboxes using the same
target-normalized objective family used by the oracle scorer. Target selection
still uses only no-future Circleworld/prefix metadata.

Third-lockbox comparison:

| Policy | Corr-copy | MSE-copy | Loop-copy | Harm-delta | Corr-gain0 | Strict |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| objective `centroid` | `+0.067868057` | `-0.024873474` | `-0.055687728` | `-0.007681783` | `+0.056762253` | `true` |
| objective `knn1` | `+0.087187174` | `-0.026378813` | `-0.079227675` | `-0.009779265` | `+0.085452992` | `true` |
| objective `knn5` | `+0.089238554` | `-0.025649934` | `-0.074221048` | `-0.008133271` | `+0.088318922` | `true` |
| family `centroid` | `+0.034664080` | `-0.028444819` | `-0.107745053` | `-0.009128800` | `+0.026537374` | `true` |
| direct `knn5` route | `+0.046431469` | `-0.026307447` | `-0.115702635` | `+0.001429030` | `+0.039411010` | `false` |
| frozen fresh family table | `+0.053683083` | `-0.029347210` | `-0.122391613` | `-0.000650100` | `+0.042878462` | `true` |
| frozen fresh global route | `+0.031369457` | `-0.021254905` | `+0.000883569` | `-0.001083568` | `+0.029696283` | `false` |

Interpretation: objective-aware route labeling is the first learned route
policy to pass the third lockbox cleanly. The kNN objective policies
substantially improve correlation and corr-gain0 over the frozen family table
and reduce harmful replay excess more strongly. They give back some MSE/loop
advantage compared with the frozen family table, so this is not yet a universal
winner, but it is the strongest learned phase-native routing result so far.

Current frontier:

- Robust frozen policy: `family_transfer_route_v1`
- Best learned policy: `objective_knn5_v1`
- Best harmful replay policy in this batch: `objective_knn1_v1`
- Best MSE/loop conservative policy: frozen fresh family table

Updated next step: freeze `objective_knn1_v1` and `objective_knn5_v1`, then
test them on a fourth predeclared lockbox against the frozen family table. If
one survives, this becomes a legitimate phase-native audio promotion candidate
for the route-selector component.

## Fourth Lockbox Frozen Objective-Policy Validation

A fourth deterministic lockbox was built with seed `20260520`. Before testing
the learned objective policies, the third lockbox's expanded route-outcome
pools were generated so the policies could be trained on three source
lockboxes: original, fresh, and third.

Raw fourth suite:

| Metric | Delta vs copy-last |
| --- | ---: |
| Corr | `+0.009408278` |
| MSE | `-0.023924697` |
| Loop peak | `-0.006556849` |
| First-chunk reentry | `+0.048144219` |

Fourth selected-route comparison:

| Policy | Corr-copy | MSE-copy | Loop-copy | Reentry-copy | Harm-delta | Corr-gain0 | Strict |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| objective `knn1` | `+0.099273337` | `-0.030187520` | `-0.088029211` | `+0.027277244` | `-0.010134023` | `+0.097430213` | `true` |
| objective `knn5` | `+0.099276564` | `-0.030625581` | `-0.090603731` | `+0.026583330` | `-0.010507811` | `+0.097699237` | `true` |
| frozen fresh family table | `+0.051447135` | `-0.030792081` | `-0.128362537` | `+0.027727049` | `-0.001616901` | `+0.042823363` | `true` |

Interpretation: objective-aware learned routing survives the fourth lockbox.
Both objective kNN policies pass strict target-normalized replay. They nearly
double the frozen family table's correlation lift and improve harmful replay
excess much more strongly, while giving back some loop reduction. MSE is
effectively comparable.

Current promotion candidate for the route-selector component:

- `objective_knn5_v1`

Promotion caveat: this is a route-selector component, not a full audio model
promotion. Raw Circleworld remains non-promotional because first-chunk reentry
still worsens. The component claim is narrower and now supported:

> A no-future objective-aware route selector over phase-only Circleworld
> continuation operators improves copy-last-relative correlation, MSE, loop, and
> target-normalized harmful replay on held-out predeclared lockboxes.

Updated next step: package `objective_knn5_v1` as a frozen route-selector
profile and run one final audit that checks future-access flags, route-training
source/target separation, and all four-lockbox score summaries.

## Frozen Route-Selector Profile And Contract Audit

`objective_knn5_v1` has now been frozen as a route-selector component profile:

- Profile JSON:
  `D:\RAFA\outputs\circleworld_proto\phase_native_audio_objective_knn5_profile_2026-05-18\phase_native_audio_route_selector_profile.json`
- Profile report:
  `D:\RAFA\outputs\circleworld_proto\phase_native_audio_objective_knn5_profile_2026-05-18\PHASE_NATIVE_AUDIO_ROUTE_SELECTOR_PROFILE.md`
- Contract audit JSON:
  `D:\RAFA\outputs\circleworld_proto\phase_native_audio_objective_knn5_contract_audit_2026-05-18\phase_native_audio_route_selector_contract_audit.json`
- Contract audit report:
  `D:\RAFA\outputs\circleworld_proto\phase_native_audio_objective_knn5_contract_audit_2026-05-18\PHASE_NATIVE_AUDIO_ROUTE_SELECTOR_CONTRACT_AUDIT.md`

Profile status:

- `route_selector_component_candidate_frozen`
- Scope: `route_selector_component_only_not_full_audio_model`
- Primary policy: `objective_knn5_v1`
- Primary policy feature count: `118`
- Primary policy source-training cases: `186`
- Primary policy target cases: `62`

Selected-route evidence included in the frozen profile:

| Validation | Corr-copy | MSE-copy | Loop-copy | Harm-delta | Corr-gain0 | Strict |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| third lockbox `objective_knn5` | `+0.089238554` | `-0.025649934` | `-0.074221048` | `-0.008133271` | `+0.088318922` | `true` |
| fourth lockbox `objective_knn5` | `+0.099276564` | `-0.030625581` | `-0.090603731` | `-0.010507811` | `+0.097699237` | `true` |

Raw Circleworld guard context across all four lockboxes remains
non-promotional:

| Lockbox | Raw status | Corr-copy | MSE-copy | Loop-copy | Reentry-copy |
| --- | --- | ---: | ---: | ---: | ---: |
| original | `phase_native_audio_not_promotional` | `+0.011047512` | `-0.022044000` | `-0.016873862` | `+0.046929410` |
| fresh seed `20260518` | `phase_native_audio_not_promotional` | `+0.009390516` | `-0.024008087` | `-0.012624197` | `+0.048045225` |
| third seed `20260519` | `phase_native_audio_not_promotional` | `+0.011266029` | `-0.021517834` | `+0.005860078` | `+0.049837844` |
| fourth seed `20260520` | `phase_native_audio_not_promotional` | `+0.009408278` | `-0.023924697` | `-0.006556849` | `+0.048144219` |

Contract audit result:

- Status: `route_selector_contract_pass`
- Checks: `37 / 37`
- Verified no-future target route selection.
- Verified source/target lockbox separation.
- Verified selected-route metric guard.
- Verified raw-Circleworld non-promotion guard.
- Verified feature-key hygiene: route selection uses case horizon metadata and
  Circleworld/prefix metadata, not target metrics or future audio features.

Interpretation:

`objective_knn5_v1` is now a legitimate phase-native route-selector component
candidate. This does **not** promote raw Circleworld and does **not** claim a
full audio-model win. It freezes the narrower claim:

> A no-future objective-aware route selector over phase-only Circleworld
> continuation operators improves copy-last-relative correlation, MSE, loop,
> and target-normalized harmful replay on held-out predeclared lockboxes.

Updated next step: run a true shared-battlefield comparison against Graduation
RAFA using the same prefix/future split and the same continuation metrics. That
is the bridge test needed to decide whether the route-selector component is
only a Circleworld-internal improvement or genuinely useful against the current
RAFA sound-engine baseline.

## Graduation Bridge Adapter And Shared Battlefield

A first Graduation bridge adapter was added:

- Entrypoint:
  `D:\RAFA\runtimes\circleworld_proto\run_graduation_phase_native_bridge.py`
- Fourth-lockbox bridge JSON:
  `D:\RAFA\outputs\circleworld_proto\graduation_phase_native_bridge_2026-05-18_fourth_full_group_prefixrms\graduation_phase_native_bridge.json`
- Fourth-lockbox bridge report:
  `D:\RAFA\outputs\circleworld_proto\graduation_phase_native_bridge_2026-05-18_fourth_full_group_prefixrms\GRADUATION_PHASE_NATIVE_BRIDGE.md`

This adapter drives the Graduation compat14 restore runtime with a prefix-only
STFT continuation canvas. It is future-access clean, but it is not a native
Graduation continuation baseline and must not be read as a failure of the
protected Graduation RAFA branch.

Fourth-lockbox Graduation bridge result versus copy-last:

| Metric | Delta |
| --- | ---: |
| corr | `-0.000209720` |
| MSE | `-0.014910783` |
| loop | `+0.028808543` |
| reentry | `+0.051972151` |

Result: `graduation_bridge_smoke_not_candidate`.

A shared-battlefield assembler was added:

- Entrypoint:
  `D:\RAFA\runtimes\circleworld_proto\assemble_phase_native_audio_shared_battlefield.py`
- Shared battlefield JSON:
  `D:\RAFA\outputs\circleworld_proto\phase_native_audio_shared_battlefield_2026-05-18_fourth\phase_native_audio_shared_battlefield.json`
- Shared battlefield report:
  `D:\RAFA\outputs\circleworld_proto\phase_native_audio_shared_battlefield_2026-05-18_fourth\PHASE_NATIVE_AUDIO_SHARED_BATTLEFIELD.md`

Fourth-lockbox shared battlefield:

| Method | Status | Corr-copy | MSE-copy | Loop-copy | Reentry-copy | Harm-delta |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| raw Circleworld | `phase_native_audio_not_promotional` | `+0.009408278` | `-0.023924697` | `-0.006556849` | `+0.048144219` | `NA` |
| routed Circleworld `objective_knn5` | `selected_route_target_replay_pass` | `+0.099276564` | `-0.030625581` | `-0.090603731` | `+0.026583330` | `-0.010507811` |
| Graduation compat14 bridge | `graduation_bridge_smoke_not_candidate` | `-0.000209720` | `-0.014910783` | `+0.028808543` | `+0.051972151` | `NA` |

Shared-battlefield status:

- `routed_circleworld_beats_raw_and_current_adapter_not_graduation_baseline`

Caveat:

Graduation here is the compat14 restore runtime forced into a prefix-only
continuation canvas. The result proves that the Circleworld route-selector
component is useful on this shared harness, but it does not prove that routed
Circleworld beats Graduation RAFA. It only says the current experimental
adapter is not yet calibrated as a fair sound-engine comparator.

Updated next step: if we want a fair Graduation comparison, do it from the
protected `codex/graduation-runtime` branch/worktree or an isolated worktree
created from it, then build a native continuation harness there. Do not treat
the Circleworld-side compat14 adapter as the Graduation baseline.

## Circleworld Operator Block V1 Addendum - 2026-05-19

`circleworld_operator_block_v1` was added as an internal Circleworld architecture consolidation layer, not as a Graduation comparison and not as a raw-Circleworld checkpoint promotion.

New runtime pieces:

- Operator bank: `D:\RAFA\runtimes\circleworld_proto\phase_native_audio_operators.py`
- Block runner: `D:\RAFA\runtimes\circleworld_proto\run_circleworld_operator_block.py`
- Full fourth-lockbox block artifact: `D:\RAFA\outputs\circleworld_proto\circleworld_operator_block_v1_2026-05-19_fourth_full\circleworld_operator_block_v1.json`
- Canonical selected-route rerender: `D:\RAFA\outputs\circleworld_proto\phase_native_audio_selected_route_2026-05-19_objective_knn5_canonical_operator_bank_fourth_full\phase_native_audio_selected_route.json`

Full fourth-lockbox block result:

| Metric | Delta |
| --- | ---: |
| corr vs copy-last | `+0.092920529` |
| MSE vs copy-last | `-0.030161893` |
| loop vs copy-last | `-0.082551264` |
| reentry vs copy-last | `+0.027050994` |
| harmful replay excess vs copy-last | `-0.010289780` |
| corr vs gain-0 | `+0.091343201` |

Status: `operator_block_target_replay_pass`.

The block beats raw Circleworld on corr, MSE, loop, and reentry deltas for the fourth lockbox. It exactly matches the regenerated canonical selected-route renderer after both paths use the extracted operator bank. The older 2026-05-17 selected-route artifact remains valid historical evidence, but the post-extraction canonical selected-route rerender is the correct equivalence reference for `circleworld_operator_block_v1`.

Interpretation update: `objective_knn5_v1` is now best understood as the first routing head inside a reusable Circleworld operator block. This is architectural consolidation of a supported component claim, not a new full audio model claim and not a Graduation-vs-Circleworld result.
