# Circleworld Coherence-Retention Child-IFS Scout

## Result

Child-local coherence retention fixed the immediate `max_qualified_steps=0` blocker, but did not yet produce `nested_sibling`.

The selected checkpoint is audio-stable and branch-active. The tradeoff checkpoint is more ontology-interesting because it keeps stronger carry/divergence, but it costs more benchmark fidelity.

## Checkpoints

- selected: `D:\RAFA\checkpoints_circleworld_proto\coherence_retention_childifs_2026-05-10\circleworld_real_anchor_config_cem_v1.json`
- tradeoff: `D:\RAFA\checkpoints_circleworld_proto\coherence_retention_childifs_2026-05-10\circleworld_real_anchor_config_scientific_tradeoff_v1.json`

## Nested Ontology

| config | nested sibling | qualified steps | max qualified carry | identity carry | best coherence | min coherence | parent div | sibling div | readout response | world jump |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| selected | 0 | 3 | 1 | 0.561806 | 0.242821 | 0.124018 | 0.0786518 | 0.0761558 | 0.236189 | 0 |
| tradeoff | 0 | 3 | 0.8 | 0.972222 | 0.144231 | 0.0954702 | 0.0909127 | 0.0938902 | 0.247263 | 0 |

## Held-Out Branch Metrics

| config | real branch | parent branch | meso effect | live child | writeback | local coherence | raw coherence | retention delta | child parent div | child sibling div | law families | phase-only branch |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| selected | 0.577778 | 0 | 0.0257002 | 0.577778 | 0.111548 | 0.428109 | 0.387808 | 0.0403015 | 0.217557 | 0.20577 | 1 | 0 |
| tradeoff | 0.577778 | 0.0789594 | 0.0254915 | 0.577778 | 0.10931 | 0.402809 | 0.356522 | 0.0462867 | 0.219665 | 0.144937 | 1.66667 | 0.114388 |

## Benchmark

| config | corr | mae | mse | bitwise stable |
| --- | ---: | ---: | ---: | --- |
| selected | 0.999051 | 0.00569159 | 0.000106832 | True |
| tradeoff | 0.991879 | 0.0147758 | 0.000928435 | True |

## Interpretation

- Retention is real: selected held-out local coherence is `0.428` versus raw `0.388`; nested qualified identity survives for `3` local steps.
- Too much retention is not enough: high coherence can reduce parent/sibling divergence, so the next law must retain child identity only when the child is causally moving the parent.
- The tradeoff candidate is scientifically more useful than the automated selected checkpoint for ontology analysis, but selected is safer for benchmark continuity.
- No promotion yet: `nested_sibling` remains zero.

## Artifacts

- selected_checkpoint: `D:\RAFA\checkpoints_circleworld_proto\coherence_retention_childifs_2026-05-10\circleworld_real_anchor_config_cem_v1.json`
- tradeoff_checkpoint: `D:\RAFA\checkpoints_circleworld_proto\coherence_retention_childifs_2026-05-10\circleworld_real_anchor_config_scientific_tradeoff_v1.json`
- train: `D:\RAFA\outputs\circleworld_proto\coherence_retention_childifs_2026-05-10\training\train_summary.json`
- heldout_selected: `D:\RAFA\outputs\circleworld_proto\coherence_retention_childifs_2026-05-10\heldout_selected\heldout_summary.json`
- heldout_tradeoff: `D:\RAFA\outputs\circleworld_proto\coherence_retention_childifs_2026-05-10\heldout_tradeoff\heldout_summary.json`
- benchmark_selected: `D:\RAFA\outputs\circleworld_proto\coherence_retention_childifs_2026-05-10\benchmark_selected\benchmark_summary.json`
- benchmark_tradeoff: `D:\RAFA\outputs\circleworld_proto\coherence_retention_childifs_2026-05-10\benchmark_tradeoff\benchmark_summary.json`
- nested_selected: `D:\RAFA\outputs\circleworld_proto\coherence_retention_childifs_2026-05-10\training\_selection_nested\candidate_00\nested_commitment_report.json`
- nested_tradeoff: `D:\RAFA\outputs\circleworld_proto\coherence_retention_childifs_2026-05-10\training\_selection_nested\candidate_02\nested_commitment_report.json`
- audit: `D:\RAFA\outputs\circleworld_proto\coherence_retention_childifs_2026-05-10\audit_selected_tradeoff\seeded_child_substrate_audit.json`
