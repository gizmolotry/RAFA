# Circleworld Parent Prewrite / Carrier Isolation - 2026-05-13

## Question

After trained authority matched oracle-scale binding but still produced only `0.5` parent-mode sibling conversion, was the remaining gap caused by:

- insufficient phasor carrier expressivity, or
- the learned parent prewrite residual corrupting the mode-replace state?

## New Isolation Variants

- `parent_ontology_trained_phasor_authority_gate55_static_mode_replace`
  - trained phasor carrier
  - trained authority
  - learned parent prewrite delta enabled

- `parent_ontology_trained_phasor_authority_nodelta_gate55_static_mode_replace`
  - trained phasor carrier
  - trained authority
  - learned parent prewrite delta disabled with `learned_parent_bridge_delta_scale=0.0`

- `parent_ontology_oracle_residual_trained_phasor_gate55_static_mode_replace`
  - oracle parent residual prewrite
  - trained phasor carrier at mode replacement

- `parent_ontology_oracle_residual_blend_phasor_gate55_static_mode_replace`
  - oracle parent residual prewrite
  - blended live child phase and trained phasor carrier

## Seeded Results

| variant | sibling | conversion | positive residual reduction | binding | acceptance | delta scale | charge | mode1 occ | jump |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| trained phasor + trained authority + learned delta | 0.5000 | 0.499636 | 0.157150 | 0.074486 | 0.350458 | 1.0 | 0.257743 | 0.655421 | 0.0 |
| trained phasor + trained authority + no delta | 1.0000 | 0.999275 | 0.000000 | 0.074486 | 0.350458 | 0.0 | 0.257743 | 0.655394 | 0.0 |
| oracle residual + trained phasor | 1.0000 | 0.999276 | 0.366930 | 0.074451 | 0.000000 | 0.0 | 0.257410 | 0.654142 | 0.0 |
| oracle residual + blend phasor | 1.0000 | 0.999276 | 0.366930 | 0.074451 | 0.000000 | 0.0 | 0.257410 | 0.654142 | 0.0 |
| oracle child phase | 1.0000 | 0.999276 | 0.366930 | 0.074451 | 0.000000 | 0.0 | 0.257410 | 0.654142 | 0.0 |

## Held-Out Results

| variant | sibling | conversion | positive residual reduction | binding | acceptance | delta scale | charge | mode1 occ | jump |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| trained phasor + trained authority + learned delta | 0.5000 | 0.499636 | 0.157151 | 0.074485 | 0.350449 | 1.0 | 0.257737 | 0.655423 | 0.0 |
| trained phasor + trained authority + no delta | 1.0000 | 0.999275 | 0.000000 | 0.074485 | 0.350449 | 0.0 | 0.257737 | 0.655396 | 0.0 |
| oracle residual + trained phasor | 1.0000 | 0.999276 | 0.366938 | 0.074452 | 0.000000 | 0.0 | 0.257416 | 0.654136 | 0.0 |
| oracle residual + blend phasor | 1.0000 | 0.999276 | 0.366938 | 0.074452 | 0.000000 | 0.0 | 0.257416 | 0.654136 | 0.0 |
| oracle child phase | 1.0000 | 0.999276 | 0.366938 | 0.074452 | 0.000000 | 0.0 | 0.257416 | 0.654136 | 0.0 |

## Interpretation

This is the cleanest result in the childworld parent-ontology lane so far.

The learned phasor carrier plus trained authority is sufficient to match oracle parent-mode conversion when the learned parent prewrite delta is disabled. The same result reproduces on held-out seeded cases.

That means the prior `0.5` ceiling was not caused by missing carrier expressivity or insufficient authority. It was caused by the learned parent prewrite projector perturbing mode 1 before the mode-replacement ontology step.

The surprising detail is that the no-delta variant has essentially zero positive residual reduction but achieves full parent-mode ontology conversion. In this assay, the correct path is:

1. preserve parent mode 1 before ontology replacement
2. grant child-derived authority/charge/logits
3. replace through the trained child phasor carrier

The learned parent prewrite residual is trying to be helpful locally, but it breaks the downstream ontology operation.

## Updated Claim

> Child-to-parent ontology conversion does not require a learned scalar parent prewrite residual in this assay. It requires a faithful child phasor carrier, learned authority, ontology charge, and mode-logit authority. The learned parent prewrite residual should be disabled or separately gated until it can prove downstream compatibility with mode replacement.

## Consequence

For the next runtime/probe design, split the bridge into two laws:

- carrier/authority law: allowed to drive parent ontology
- parent prewrite residual law: optional and must pass a downstream-compatibility gate

Do not merge these into one "predictive residual" score. The residual can improve one-step phase error while damaging branch ontology.

## Artifacts

- Seeded replay: `D:\RAFA\outputs\circleworld_proto\parent_prewrite_carrier_isolation_2026-05-13_seeded_suite\childworld_volume_recursion_sweep_summary.json`
- Held-out replay: `D:\RAFA\outputs\circleworld_proto\parent_prewrite_carrier_isolation_2026-05-13_heldout_9110_9112\childworld_volume_recursion_sweep_summary.json`
- Authority model: `D:\RAFA\outputs\circleworld_proto\parent_authority_bridge_trained_2026-05-13\parent_phase_projector_model.json`
