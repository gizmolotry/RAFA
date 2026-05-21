# Circleworld Continuation-Causal Child IFS Scout - 2026-05-11

## Result
The guarded isolated scout produced the first externally reproduced `continuation_nested_sibling` signal in this cycle.

Selected checkpoint:
`D:\RAFA\checkpoints_circleworld_proto\continuation_causal_childifs_2026-05-11_guarded\circleworld_real_anchor_config_cem_v1.json`

## What Changed
- Added family-specific nested assay metrics for readout, continuation, and mode-replacement branches.
- Added actual-execution provenance for continuation mode-1 child writeback.
- Added a nested-sibling readiness score so one branch must co-satisfy the ontology gates.
- Added trainer support for continuation/mode-replacement nested objectives.
- Added guarded CEM candidates so the parent checkpoint and CEM mean cannot be skipped by the nested probe pool.
- Added nested-preferred selection for ontology cycles.

## Key Nested Metrics
| Metric | Previous causal-retention | Guarded selected | Delta |
|---|---:|---:|---:|
| total nested sibling fraction | 0.041666666666666664 | 0.08333333333333333 | 0.041666666666666664 |
| readout sibling fraction | 0.5 | 0.5 | 0.0 |
| continuation sibling fraction | 0.0 | 0.5 | 0.5 |
| mode-replace sibling fraction | 0.0 | 0.0 | 0.0 |
| continuation readiness max | 0.9662262424196698 | 1.0 | 0.03377375758033019 |
| continuation write delta | 0.10936024340334054 | 0.1185634754394866 | 0.009203232036146064 |
| world-jump penalty | 0.0 | 0.0 | 0.0 |

## Held-Out Branch Metrics
| Metric | Previous | Guarded selected | Delta |
|---|---:|---:|---:|
| real branch fraction | 0.5777778029441833 | 0.5777778029441833 | 0.0 |
| child writeback mass | 0.10731078022056156 | 0.11921497020456526 | 0.011904189984003694 |
| child parent divergence | 0.22057032129830784 | 0.2392194705704848 | 0.01864914927217695 |
| child sibling divergence | 0.15539374450842539 | 0.22337413248088625 | 0.06798038797246087 |
| meso branch effect | 0.0252147042056455 | 0.03424480808455264 | 0.009030103878907136 |

## Benchmark / Continuity
| Metric | Previous | Guarded selected | Delta |
|---|---:|---:|---:|
| mean corr | 0.9884218145085317 | 0.9999959764865876 | 0.01157416197805583 |
| mean mae | 0.016998153738677503 | 0.0003802701336098835 | -0.01661788360506762 |
| mean mse | 0.0013309324749570806 | 3.57429360064998e-07 | -0.0013305750455970156 |
| recurrence score | 0.20183167832057727 | 0.20189656971562844 | 6.489139505116981e-05 |

## Interpretation
The previous checkpoint had readout-only siblings: child identity could affect readout, but continuation and mode-replacement branches did not earn `nested_sibling` labels. The guarded scout found a candidate where `child_dominant_continuation_shift` satisfies the ontology gates in all seeded cases: same-child carry, budget retention, coarse preservation, q-profile preservation, and stronger-than-readout-baseline response.

This supports a narrower claim: branch-local child state can become a causal continuation carrier under the current assay. It does not yet prove mode-replacement siblings or general audio-domain branch ontology.

## Kill-Switch Revision
Subsequent kill-switch testing narrowed the mechanism. The continuation-family `nested_sibling` label survives when the explicit low-rank continuation mode-1 write path is disabled, and it also survives when child-local IFS recurrence is disabled. It does not survive strict parent-only continuation, write-only parent readout, scrambled child phase, or removal of child coherence retention.

So this report should not be read as proof of strong gated-writeback causality or proof that child-local IFS recurrence is necessary. The current supported mechanism is narrower:

- child-derived phase carrier affects the continuation branch;
- same-child coherence and budget retention are necessary for the ontology label;
- explicit low-rank writeback is not necessary under this assay;
- child-local IFS recurrence is not necessary under this assay.

Updated kill-switch report:
`D:\RAFA\docs\reports\CIRCLEWORLD_CONTINUATION_CAUSAL_KILLSWITCH_CHILDIFS_2026-05-11.md`

Strict static-record follow-up pushed this narrower again. Across the full seeded substrate, `static_child_record_direct_mix_only` preserves continuation sibling fraction at `0.5` with branch pre-unroll, branch childworld runtime, final continuation pre-unroll, and explicit writeback all disabled. `static_child_record_write_only` drops continuation sibling to `0.0` despite active explicit write, and `static_child_record_stripped_control` drops continuation sibling to `0.0`.

Current best mechanism phrase:

> static/forked child phase record as a final direct continuation carrier, under seeded branch-active nested assay.

## Remaining Gaps
- Mode-replacement sibling fraction remains zero.
- The successful continuation branch is `child_dominant_continuation_shift`; local continuation still loses identity budget in this selected checkpoint.
- The result is seeded branch-active substrate evidence, not generic anchor-start evidence.
- Learned branch law remains assay/shadow-side; this run tunes parametric runtime coefficients plus causal assay selection, not a deployed neural branch law.

## Artifacts
- Compare JSON: `D:\RAFA\outputs\circleworld_proto\continuation_causal_childifs_2026-05-11\continuation_causal_compare.json`
- Training summary: `D:\RAFA\outputs\circleworld_proto\continuation_causal_childifs_2026-05-11\training_guarded\train_summary.json`
- External nested report: `D:\RAFA\outputs\circleworld_proto\continuation_causal_childifs_2026-05-11\nested_guarded_selected_external\nested_commitment_report.json`
- Held-out summary: `D:\RAFA\outputs\circleworld_proto\continuation_causal_childifs_2026-05-11\heldout_guarded_selected\heldout_summary.json`
- Benchmark summary: `D:\RAFA\outputs\circleworld_proto\continuation_causal_childifs_2026-05-11\benchmark_guarded_selected\benchmark_summary.json`
