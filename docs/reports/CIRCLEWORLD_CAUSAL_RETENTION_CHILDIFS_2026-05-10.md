# Circleworld Causal-Retention Child-IFS Scout

## Result

Causal-gated retention plus the corrected support-masked identity assay produced the first nonzero `nested_sibling` result in this child-IFS lane.

This is an ontology success, not a full checkpoint promotion: the successful sibling is `child_local_readout_shift` in each seeded case, while child-local continuation and mode-replacement branches remain `ambiguous_middle`. The learned branch-law module is still shadow/diagnostic; its sandbox intervention barely moves the parent.

## Checkpoint

- selected: `D:\RAFA\checkpoints_circleworld_proto\causal_retention_childifs_2026-05-10\circleworld_real_anchor_config_cem_v1.json`

## Assay Correction

`same_child_coherence_mean` is now support-masked. The old full-lattice mean wrongly graded empty space outside a local child as failed coherence. A regression test now locks this behavior.

## Nested Ontology Comparison

| config | nested sibling | qualified steps | qualified carry | identity carry | best coherence | min coherence | budget retained | parent div | sibling div | readout response | world jump |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| previous selected | 0 | 3 | 1 | 0.561806 | 0.416909 | 0.341589 | 1 | 0.0786518 | 0.0761558 | 0.236189 | 0 |
| previous tradeoff | 0 | 5 | 1 | 0.972222 | 0.280395 | 0.267604 | 0.886654 | 0.0909127 | 0.0938902 | 0.247263 | 0 |
| causal selected | 0.0416667 | 5 | 1 | 0.979167 | 0.303275 | 0.300154 | 0.894521 | 0.0980833 | 0.0978314 | 0.270796 | 0 |

## Held-Out Branch Metrics

| config | real branch | meso effect | local coh | raw coh | retention | parent phase delta | causal gate | gate loss | parent div | sibling div | law families | phase-only branch |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| previous selected | 0.577778 | 0.0257002 | 0.428109 | 0.387808 | 0.0403015 | 0 | 0 | 0 | 0.217557 | 0.20577 | 1 | 0 |
| previous tradeoff | 0.577778 | 0.0254915 | 0.402809 | 0.356522 | 0.0462867 | 0 | 0 | 0 | 0.219665 | 0.144937 | 1.66667 | 0.114388 |
| causal selected | 0.577778 | 0.0252147 | 0.394524 | 0.351849 | 0.0426746 | 0.575584 | 0.571226 | 0.00899166 | 0.22057 | 0.155394 | 1.66667 | 0.111071 |

## Benchmark

| config | corr | mae | mse | bitwise stable |
| --- | ---: | ---: | ---: | --- |
| previous selected | 0.999051 | 0.00569159 | 0.000106832 | True |
| previous tradeoff | 0.991879 | 0.0147758 | 0.000928435 | True |
| causal selected | 0.988422 | 0.0169982 | 0.00133093 | True |

## Nested Sibling Hits

- `naked_seed_9100` / `child_local_readout_shift`: readout `0.336107` vs baseline `0.321935`, carry `1.000`, qualified steps `5.0`, coherence `0.303281`, budget retained `0.613609`
- `naked_seed_9101` / `child_local_readout_shift`: readout `0.336082` vs baseline `0.323127`, carry `1.000`, qualified steps `5.0`, coherence `0.303281`, budget retained `0.613609`
- `naked_seed_9102` / `child_local_readout_shift`: readout `0.334396` vs baseline `0.320945`, carry `1.000`, qualified steps `5.0`, coherence `0.303262`, budget retained `0.613607`

## Learned Branch-Law Assay

- status: `child_local_ifs_signal_present`
- feature rows: `15360`
- shadow final loss: `0.010657`
- mean isolated child coherence: `0.662149`
- mean coupled-vs-isolated phase delta: `0.005903`
- sandbox mean phase delta: `0.000039`
- sandbox live-child/writeback delta: `0.000000` / `0.000000`

Interpretation: the learned module can fit the shadow branch-law targets, but it is not yet a causal controller.

## Interpretation

- The child framework is no longer only decorative: seeded branch-active cases now contain labeled nested siblings under the corrected ontology assay.
- The sibling is readout-first: child identity survives and changes readout, but stricter continuation/mode-replacement branches have not yet become sibling carriers.
- Causal retention preserved branch activity and improved nested response, with benchmark correlation still above `0.98`.
- No active checkpoint promotion yet; promote this as an ontology/instrumentation milestone and use it to drive the next continuation-causality objective.

## Artifacts

- train_summary: `D:\RAFA\outputs\circleworld_proto\causal_retention_childifs_2026-05-10\training\train_summary.json`
- heldout_summary: `D:\RAFA\outputs\circleworld_proto\causal_retention_childifs_2026-05-10\heldout_selected\heldout_summary.json`
- benchmark_summary: `D:\RAFA\outputs\circleworld_proto\causal_retention_childifs_2026-05-10\benchmark_selected\benchmark_summary.json`
- nested_report_masked: `D:\RAFA\outputs\circleworld_proto\causal_retention_childifs_2026-05-10\nested_selected_masked_coherence\nested_commitment_report.json`
- learned_branch_law_assay: `D:\RAFA\outputs\circleworld_proto\causal_retention_childifs_2026-05-10\learned_branch_law_assay\learned_branch_law_assay.json`
- learned_branch_law_markdown: `D:\RAFA\outputs\circleworld_proto\causal_retention_childifs_2026-05-10\learned_branch_law_assay\LEARNED_BRANCH_LAW_ASSAY.md`
- compare: `D:\RAFA\outputs\circleworld_proto\causal_retention_childifs_2026-05-10\causal_retention_compare.json`
