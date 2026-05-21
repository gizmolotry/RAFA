# Circleworld Parent Ontology Bridge Scout - 2026-05-13

## Executive Claim

A standalone bridge scout was added to learn the boundary between parent-ontology misses and `mode_replace_nested_sibling` conversions from existing nested-assay artifacts.

The scout does not change Circleworld runtime behavior. It reads existing JSON reports and fits a deterministic ridge/threshold surrogate over branch-level features.

## Test Artifacts

Full-root scout:

- `D:\RAFA\outputs\circleworld_proto\continuation_causal_childifs_2026-05-13\parent_ontology_bridge_scout.json`

Focused calibration scout:

- `D:\RAFA\outputs\circleworld_proto\continuation_causal_childifs_2026-05-13\parent_ontology_bridge_scout_focused.json`

Implementation:

- `D:\RAFA\runtimes\circleworld_proto\run_parent_ontology_bridge_scout.py`

## Full-Root Scout

The full-root scout scanned all May 13 child-ontology artifacts.

Key metrics:

- `row_count`: `1503`
- `positive_bridge_count`: `122`
- `nested_sibling_count`: `83`
- `readout_crossing_count`: `122`
- `oracle_tainted_count`: `1378`
- `train_accuracy_at_0_5`: `0.9953426480372588`

Best univariate boundary:

- `readout_minus_baseline >= 0.0002083205342774952`
- negative max: `-0.003995295257162801`
- positive min: `0.0002083205342774952`
- accuracy: `1.0`

Interpretation:

The full-root scout confirms that the current label boundary is dominated by readout crossing, not by a broad world-jump transition. However, the full-root recommendation is noisy because it mixes non-oracle smoke variants, perturb branches, readout branches, and parent-ontology calibration branches.

## Focused Calibration Scout

The focused scout used only:

- `parent_ontology_calibration_seeded_suite`
- `parent_ontology_fine_calibration_seeded_suite`
- `parent_ontology_maxreadout_seeded_suite`

Key metrics:

- `row_count`: `803`
- `positive_bridge_count`: `56`
- `nested_sibling_count`: `48`
- `readout_crossing_count`: `56`
- `oracle_tainted_count`: `803`
- `train_accuracy_at_0_5`: `0.9925280199252802`

Best univariate boundary:

- `readout_minus_baseline >= 0.0002083205342774952`
- negative max: `-0.003995295257162801`
- positive min: `0.0002083205342774952`
- accuracy: `1.0`

Mode-replace family boundary:

- parent ontology gate boundary: `parent_ontology_gate_floor >= 0.55`
- mode-1 occupancy boundary: `parent_ontology_mode1_occupancy_after >= 0.33048028513233146`

Recommended next target:

- variant: `parent_ontology_childphase_gate54_static_mode_replace`
- branch: `child_mode1_replace_50_shift`
- current parent ontology gate: `0.54`
- recommended next parent ontology gate: `0.545`
- current mode-1 occupancy after: `0.3234183449887706`
- recommended next mode-1 occupancy after: `0.326949315060551`

## What We Learned

The bridge scout supports the parent-authority interpretation:

1. `gate54` is a near miss, not a failed childworld.
2. The miss is specifically below the readout-crossing predicate.
3. Coarse-world and q-profile preservation remain intact near the boundary.
4. The next learned bridge should target parent-mode authority/occupancy around the narrow `gate54 -> gate55` transition.

The practical target is no longer vague:

> Learn a non-oracle head that moves a child-derived branch from gate `0.54`-like authority to approximately gate `0.545` / mode-1 occupancy `0.327`, while keeping coarse/q identity stable.

## Limitations

The focused scout is entirely oracle-tainted because the positive calibration evidence uses oracle-informed parent residual authority. Therefore the learned boundary should be treated as an assay target, not a deployable policy.

Do not copy oracle residuals into runtime. The next module must learn from non-oracle child/parent features and only aim for the discovered authority boundary.

## Next Action

Add an assay-only `child_predictive_candidate="learned_parent_bridge"` path that feeds the existing ontology contract without using `oracle_parent_residual` at runtime.

This path should output:

- parent-mode delta or child-phase low-rank delta
- ontology charge
- mode-1 occupancy target
- acceptance/confidence
- support/logit authority hints

It must be compared against:

- raw child delta
- oracle residual upper bound
- focused gate54/gate55 boundary

## Ledger Status

The experiment ledger utility now emits both Markdown and JSONL:

- `D:\RAFA\docs\reports\CIRCLEWORLD_EXPERIMENT_LEDGER_2026-05-13.md`
- `D:\RAFA\docs\reports\CIRCLEWORLD_EXPERIMENT_LEDGER_2026-05-13.jsonl`

The ledger should be regenerated after each new assay/report so interpretations and artifact paths remain synchronized.

## Leakage-Audited Scout Addendum

The scout now emits two model views:

1. `model`: all available features, including readout/label-adjacent fields.
2. `nonleaky_model`: excludes direct readout, baseline, post-replacement occupancy, world-jump, and shortcut fields.

Focused calibration nonleaky result:

- `nonleaky_train_accuracy_at_0_5`: `0.9439601494396015`
- strongest nonleaky saliency fields:
  - `parent_ontology_support_boost`: coefficient `-2.787708485257634`
  - `parent_ontology_charge_gain`: coefficient `1.843573442805342`
  - `child_predictive_boundary_match`: coefficient `1.040099107222621`
  - `child_volume_top_score_share`: coefficient `1.0049049314132485`
  - `child_volume_effective_count`: coefficient `0.8267918384686462`

Interpretation:

- The all-feature model mostly rediscovers the label boundary: `readout_minus_baseline > 0`.
- The nonleaky model is weaker but still informative, with accuracy around `0.944` on focused calibration rows.
- Nonleaky saliency points toward parent authority knobs plus child boundary compatibility, not just child volume.
- This strengthens the current next step: learn a parent authority head from child/parent compatibility and authority features, not from final readout labels.

Important caveat:

The current nonleaky model still operates over assay rows, not direct runtime tensors. It is a scout for choosing targets and features, not yet the learned bridge module itself.

The next implementation should therefore be explicit about status:

- `scout`: learned from saved assay JSON rows
- `assay-only candidate`: allowed to test authority hypotheses
- `runtime learned bridge`: not yet promoted
