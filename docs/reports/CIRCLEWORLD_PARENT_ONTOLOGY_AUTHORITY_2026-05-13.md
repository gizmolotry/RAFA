# Circleworld Parent Ontology Authority Breakthrough - 2026-05-13

## Executive Claim

The latest seeded childworld ontology sweep produced the first replicated `mode_replace_nested_sibling` result under the new parent-ontology authority assay.

This is not yet production runtime behavior and not yet a trained law. It is an assay-only positive control showing that childworld continuation can convert into parent-mode ontology when the parent grants sufficient mode-1 authority to the child-derived field.

## What Changed

The new assay path adds parent-side ontology fields after child predictive assimilation:

- `ontology_charge`
- `mode_child_binding`
- `ontology_phase_delta`
- `ontology_child_id`

These fields are carried through the nested continuation bundle and used only in mode-replacement assays when `parent_ontology_use_in_mode_replace=True`.

The key conceptual shift is:

> Child survival/writeback alone is not enough. The parent lattice must expose an ontology authority path that lets a child-local structure become a competing parent mode.

## Evidence Artifacts

Primary seeded-suite summary:

- `D:\RAFA\outputs\circleworld_proto\continuation_causal_childifs_2026-05-13\parent_ontology_maxreadout_seeded_suite\childworld_volume_recursion_sweep_summary.json`

Calibration summary:

- `D:\RAFA\outputs\circleworld_proto\continuation_causal_childifs_2026-05-13\parent_ontology_calibration_seeded_suite\childworld_volume_recursion_sweep_summary.json`

Implemented assay code:

- `D:\RAFA\runtimes\circleworld_proto\test_nested_commitment.py`
- `D:\RAFA\runtimes\circleworld_proto\run_childworld_volume_recursion_sweep.py`

## Seeded Replication

The maxreadout parent-ontology assay was run against seeded branch-active cases `9100`, `9101`, and `9102`.

Aggregate result:

- `mean_assay_mode_replace_nested_sibling_fraction`: `0.5`
- `mean_mode_replace_conversion_score`: `0.4985446508561216`
- `mean_parent_mode_conversion_sibling_fraction`: `0.5`
- `mean_world_jump_penalty`: `0.0`
- `mean_final_direct_mix_shortcut_score`: `0.0`
- `mean_child_predictive_positive_residual_reduction`: `0.26338768174506794`
- `mean_parent_ontology_charge`: `0.2364940657797787`
- `mean_assay_mode_replace_mode1_replace_parent_ontology_mode1_occupancy_after`: `0.3584014981126253`

Interpretation:

- The result replicates across all seeded cases.
- The positive branch is the 50% mode-replacement branch.
- The 85% replacement branch remains `ambiguous_middle`, which is useful: too much child authority does not automatically improve ontology.
- No winning case is classified as overwrite/world-jump.
- The direct-mix shortcut score is zero in the positive maxreadout run.

## Calibration Result

A focused calibration sweep compared parent-ontology authority levels.

| Variant | Gate Floor | Mode-1 Occupancy After | Nested Sibling Fraction | Conversion Score | World Jump Penalty |
|---|---:|---:|---:|---:|---:|
| gate50 | `0.50` | `0.2506303694899127` | `0.0` | `0.0` | `0.0` |
| gate55 | `0.55` | `0.2931645728077845` | `0.5` | `0.49854242988999936` | `0.0` |
| maxreadout | `0.60` | `0.3584014981126253` | `0.5` | `0.4985446508561216` | `0.0` |

Case-level check:

- `gate50`: all mode-replacement branches remain `ambiguous_middle`.
- `gate55`: `child_mode1_replace_50_shift` is `nested_sibling` for all seeded cases.
- `maxreadout`: `child_mode1_replace_50_shift` is `nested_sibling` for all seeded cases.

The apparent threshold is between mode-1 occupancy around `0.2506` and `0.2932`, corresponding to `parent_ontology_gate_floor` moving from `0.50` to `0.55`.

## What We Learned

The prior failure was probably not just too little child volume, weak recursion, or insufficient local IFS survival.

The stronger hypothesis now is:

1. Childworlds can carry meaningful deformation, but the parent has been treating them as sidecar effects.
2. A child does not become a parent ontology unless the parent mode system allocates it explicit authority.
3. The missing bridge is a parent-side child-binding field: a way for child-local evidence to become mode occupancy, support, and readout authority.
4. Too little authority leaves the child decorative. Too much or wrong replacement can remain ambiguous. The successful zone is currently a calibrated, gated partial authority path.

This is the first evidence that the architecture can express:

> parent phase lattice -> child-local structure -> parent-recognized sibling mode

rather than only:

> parent phase lattice -> sidecar child perturbation -> decorative writeback

## What This Does Not Prove Yet

This does not prove that the current child field itself has learned a general law.

Known limitations:

- The positive path still uses assay-only parent ontology authority.
- The predictor uses oracle-informed residual machinery in this scout path.
- The branch law is not yet a trained production module.
- The result is seeded-substrate evidence, not broad held-out runtime evidence.
- This should not be promoted as default Circleworld behavior.

## Verification

Compilation passed for the touched assay/runnable files:

- `test_nested_commitment.py`
- `run_childworld_volume_recursion_sweep.py`
- `run_childworld_causality_killswitch_suite.py`

Contract tests passed:

```text
pytest -q D:\RAFA\runtimes\circleworld_proto\test_learned_branch_law_child_ifs_contract.py D:\RAFA\runtimes\circleworld_proto\test_unit_phasor_contract.py D:\RAFA\runtimes\circleworld_proto\test_phase_gauge_invariance.py
....                                                                     [100%]
4 passed in 2.53s
```

## Next Falsification Tests

The next cycle should answer whether this can become a real learned branch law rather than an assay trick.

1. Replace oracle-informed parent residual authority with a learned parent ontology head.
2. Train the head to predict where child-derived phase evidence should gain mode-1 authority.
3. Run kill-switches: remove `ontology_charge`, remove `mode_child_binding`, remove child phase source, remove readout boost independently.
4. Sweep authority around the discovered transition with finer gates: `0.51`, `0.52`, `0.53`, `0.54`, `0.55`.
5. Rerun held-out seeded substrates beyond `9100/9101/9102` to check whether the threshold generalizes.
6. Compare parent-coupled versus isolated child-local IFS under the same parent-ontology authority path.

## Current Recommendation

Do not move to dense RAFA signatures or semantic prompting yet.

The immediate research frontier is now:

> Learn the parent ontology authority bridge.

If this bridge can be learned from branch features without oracle residuals, the childworld framework becomes much more than a sidecar perturbation system.

## Fine Calibration Addendum

A follow-up sweep added fine parent-ontology authority rungs at `0.51`, `0.52`, `0.53`, and `0.54`.

Fine calibration summary:

| Variant | Gate Floor | Mode-1 Occupancy After | Max Mode-Replace Readiness | Mode-Replace Nested Sibling Fraction | Conversion Score |
|---|---:|---:|---:|---:|---:|
| gate51 | `0.51` | `0.2619385757961833` | `0.8961330052919592` | `0.0` | `0.0` |
| gate52 | `0.52` | `0.2718641122794487` | `0.9269611936200057` | `0.0` | `0.0` |
| gate53 | `0.53` | `0.2802876219481636` | `0.9592792911743225` | `0.0` | `0.0` |
| gate54 | `0.54` | `0.28729756568353865` | `0.9842584635889091` | `0.0` | `0.0` |
| gate55 | `0.55` | `0.2931645728077845` | `1.0` | `0.5` | `0.49854242988999936` |

The transition is a knife-edge between `gate54` and `gate55`.

The failing predicate at `gate54` is not coarse-world preservation or q-profile preservation. The 50% mode replacement branch has high coarse and q correlation, but the readout sibling response remains below the readout-only baseline:

- case 0: `readout_minus_baseline = -0.003995295257162801`
- case 1: `readout_minus_baseline = -0.005536430897533862`
- case 2: `readout_minus_baseline = -0.006091711720622395`

At `gate55`, the same branch crosses the baseline:

- case 0: `readout_minus_baseline = 0.0022008356447649202`
- case 1: `readout_minus_baseline = 0.0005760427323794337`
- case 2: `readout_minus_baseline = 0.0002083205342774952`

Interpretation:

- The conversion boundary is not arbitrary collapse into a world jump.
- It is a narrow parent-authority/readout-response threshold.
- The next learned bridge should target this boundary explicitly: predict enough child-derived parent-mode occupancy to cross sibling response while preserving coarse/q identity.

Fine calibration artifact:

- `D:\RAFA\outputs\circleworld_proto\continuation_causal_childifs_2026-05-13\parent_ontology_fine_calibration_seeded_suite\childworld_volume_recursion_sweep_summary.json`
