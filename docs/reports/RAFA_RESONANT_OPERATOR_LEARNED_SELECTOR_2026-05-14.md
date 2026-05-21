# RAFA Resonant Operator Learned Selector Assay - 2026-05-14

## Scope

This cycle connects the contrastive recurrence embedding to operator causality as a shadow selector.

It does not change production Circleworld runtime behavior. The learned selector is assay-only and additive in the JSON contract.

## Code Changes

- `D:\RAFA\runtimes\circleworld_proto\run_resonant_operator_causality_assay.py`
  - Added optional `--learned-selector-checkpoint`.
  - Added learned embedding shadow selector from `contrastive_structural_embedding_probe.pt`.
  - Added continuous recurrence selector comparison.
  - Added learned-vs-geometric-vs-recurrence-vs-decoy metrics.
  - Added explicit learned residual cap so learned residual cannot silently replace the geometric score.
  - Fixed the selected operator path to prefer non-self same-family candidates instead of accidentally applying the query object itself when self-scoring wins.

## Run

Input config:

- `D:\RAFA\outputs\circleworld_proto\parent_prewrite_policy_replay_2026-05-13_seeded_suite\parent_ontology_trained_phasor_authority_carrier_only_gate55_static_mode_replace\circleworld_config.json`

Learned selector checkpoint:

- `D:\RAFA\outputs\circleworld_proto\contrastive_structural_embedding_2026-05-14_naked_gpu_9100_9102_holdout_9102\contrastive_structural_embedding_probe.pt`

Output:

- `D:\RAFA\outputs\circleworld_proto\resonant_operator_causality_2026-05-14_naked_gpu_9100_9102_learned_selector_nonself_floor\RESONANT_OPERATOR_CAUSALITY_ASSAY.md`
- `D:\RAFA\outputs\circleworld_proto\resonant_operator_causality_2026-05-14_naked_gpu_9100_9102_learned_selector_nonself_floor\resonant_operator_causality_assay.json`

## Metrics

| Selector | Mean Parent Phase Divergence | Mean World Jump Proxy | Mean Selector Score | Same Local Family | Same Structural Family |
|---|---:|---:|---:|---:|---:|
| non-self geometric/local selected | `0.20485464095324304` | `0.022946314769144178` | `0.8964956046935787` | `1.0` | not separately reported |
| recurrence selected | `0.20485464095324304` | `0.022946314769144178` | `0.8951612095661611` | `1.0` | `1.0` by selected object identity in this run |
| learned hybrid selected | `0.19059757054429358` | `0.02278108191261352` | `0.9537409108499312` | `0.0` | `0.0` |
| wrong-family decoy | `0.19059757054429358` | `0.02278108191261352` | n/a | `0.0` | `0.0` |

Additional learned selector metrics:

- Mean learned embedding score: `0.8858997325102488`
- Mean raw learned residual: `0.10609280616044998`
- Mean capped learned residual: `0.05`
- Mean learned selector recurrence compatibility: `0.8944027621067173`
- Learned selector agreement with non-self geometric selected: `0.0`
- Learned selector agreement with recurrence selected: `0.0`

## Interpretation

This is a useful negative result.

The contrastive recurrence embedding is strong as a retrieval embedding, but it is not yet safe as an operator router. When used as a learned residual on top of geometric scoring, it consistently selects the same object as the wrong-family decoy rather than the non-self same-origin/recurrence-selected child.

That tells us two things:

1. The learned embedding recovered cross-seed structural recurrence, but its notion of closeness is not aligned with parent-writeback ontology.
2. Operator routing needs a stricter compatibility membrane than retrieval routing.

The non-self fix also changes the interpretation of the earlier selected-vs-decoy causality result. The prior assay was partly inflated by self-selection. After correction, the selected/recurrence operator still causes nonzero parent movement, but its gap over decoy is smaller:

- selected minus decoy phase divergence: `0.014257070408949487`
- selected minus decoy world-jump proxy: `0.00016523285653065786`

So the honest current status is:

- operator causality exists beyond inert control.
- recurrence/local selected operators are modestly stronger than wrong-family decoys in this corrected assay.
- learned contrastive selector should not be promoted into runtime or branch ontology yet.

## Design Consequence

The RAFA attention constitution held up: learned residual must be bounded and separately reported. Without that guard, the learned selector saturated the score and hid the fact that it was selecting a wrong-family decoy.

Next operator-selection work should train against causal labels directly:

- selected operator changes parent without world jump.
- decoy operator may move parent but fails branch identity.
- learned selector should predict causal-safe branch identity, not just recurrence embedding similarity.
