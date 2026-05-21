# Circleworld Phase/Token Repair - 2026-05-05

## Status
- Diagnostic cycle completed; no checkpoint promotion.
- Primary post-repair diagnostic: `phase_token_repair_branch_diag_v2`.
- Latest pressure test: `naked_phase_nested_recovery_v1_smoke`.
- Active branch checkpoint remains `agreement_scout_v1`.

## What changed
- Added phase-only branch metrics and naked RAFA magnitude-invariance audit surfaces.
- Separated parent support-gated branch, child-world branch readiness, phase-only excess branch, and low-phase decorative slot-2 activity.
- Made branch recovery fallback diagnostics explicit across transfer, heldout, and nested gates.
- Added RAFA relational factor-pack seams, operator-seed construction, and five-control semantic projector checks.
- Expanded nested ontology reports with selected-fork live-child truth, branch family counts, carry metrics, and continuity deltas.
- Added `naked_phase_nested_recovery_v1` as an additive objective profile.

## Validator fixes
- Corrected ambiguous naked shortfall scoring through explicit shortfall(target, observed) helper.
- Added qset and residue_scale to final train checkpoint config payload.
- Changed parent branch metric to use per-location phase divergence instead of scalar mean divergence gate.
- Added parent_real_branch_fraction, child_real_branch_fraction, phase_only_excess_branch_fraction, and decorative_slot2_low_phase_fraction.
- Fixed continuity summary comparison aliases for single-WAV nested continuity payloads.
- Expanded childworld experiment summaries to include selection fallback, phase-only heldout, by_source, and nested selected-fork metrics.

## Postfix separated branch read
- Heldout real branch fraction: `0.5555555721124014`
- Heldout parent real branch fraction: `0.00473729536558191`
- Heldout child real branch fraction: `0.5555555721124014`
- Heldout phase-only real branch fraction: `0.026600183298190434`
- Heldout phase-only excess branch fraction: `0.021862888087828953`
- Heldout low-phase decorative slot-2 fraction: `0.12068663982467519`
- Naked RAFA parent real branch fraction: `0.0`
- Naked RAFA child real branch fraction: `0.3333333432674408`
- Naked RAFA phase-only branch fraction: `0.0`
- Naked RAFA phase-only distinctness: `0.002165326771015922`

## Naked phase profile smoke
- Selection source: `init_config_fallback_nested_or_agreement_required`
- Heldout parent real branch fraction: `0.005592`
- Heldout child real branch fraction: `0.555556`
- Heldout phase-only branch fraction: `0.029433`
- Heldout phase-only excess branch fraction: `0.023841`
- Naked RAFA phase-only branch fraction: `0.0`
- Naked RAFA phase-only excess branch fraction: `0.0`
- Naked RAFA child writeback mass: `0.088197`
- Nested child active fraction: `0.0`
- Nested child meso response: `0.0`
- Nested sibling fraction: `0.083333`
- Benchmark mean corr: `0.890557`

## Selection diagnosis
- Candidate unavailable reasons: `['no_transfer_gate_passing_candidate', 'no_candidate_passed_heldout_gate', 'no_candidate_passed_nested_gate', 'no_soft_nested_pending_candidate']`
- Heldout gate passes: `0`
- Nested gate passes: `0`
- Latest top failure families:
  - `selection_gate`: `{'min_phase_only_branch': 2, 'min_phase_only_distinctness': 2, 'naked_phase_only_branch': 2, 'naked_phase_only_distinctness': 2, 'naked_law_families': 2, 'naked_law_entropy': 2, 'min_branch': 1, 'min_meso': 1, 'min_writeback': 1, 'min_parent_div': 1, 'min_live_child': 1, 'naked_branch': 1, 'naked_meso': 1, 'naked_writeback': 1, 'naked_parent_div': 1, 'naked_live_child': 1}`
  - `heldout_gate`: `{'heldout_naked_phase_only_branch': 2, 'heldout_naked_phase_only_distinctness': 2, 'heldout_naked_law_families': 2, 'heldout_naked_law_entropy': 2, 'heldout_naked_branch': 1, 'heldout_naked_meso': 1, 'heldout_naked_writeback': 1, 'heldout_naked_parent_div': 1, 'heldout_naked_live_child': 1, 'heldout_branch_floor': 1, 'heldout_live_child_floor': 1, 'heldout_writeback_floor': 1, 'heldout_parent_div_floor': 1, 'heldout_naked_branch_floor': 1, 'heldout_naked_live_child_floor': 1, 'heldout_naked_writeback_floor': 1, 'heldout_naked_parent_div_floor': 1}`
  - `nested_gate`: `{'nested_active_fraction': 2, 'nested_meso_response': 2}`

## Scientific read
- The old support-gated branch number is mostly child-world readiness, not parent mode split: parent real branch is near zero while child real branch carries the headline fraction.
- Naked RAFA still has child survival/writeback, but no naked parent branch and no phase-only branch.
- Raising objective pressure increased overall phase-only/excess branch slightly and writeback slightly, but did not move the naked phase-only bottleneck or nested active/meso response.
- This says the next change must alter the law/runtime path that produces phase divergence, not just selection weights.

## Recommendation
- Keep agreement_scout_v1 as active branch checkpoint.
- Do not promote naked_phase_nested_recovery_v1_smoke.
- Next implementation should modify the actual child-to-parent phase write path or mode split law, not just objective weights.
- Target naked_rafa parent/phase branch emergence: naked parent_real_branch_fraction > 0 and naked phase_only_real_branch_fraction > 0 before chasing benchmark.
- Keep naked law-family diversity as a guardrail, not as the dominant objective.

## Artifacts
- Assembled Markdown report: `D:\RAFA\outputs\circleworld_proto\phase_token_repair_2026-05-05_assembled\PHASE_TOKEN_REPAIR_REPORT.md`
- Assembled JSON summary: `D:\RAFA\outputs\circleworld_proto\phase_token_repair_2026-05-05_assembled\phase_token_repair_report_summary.json`
- Persistent JSON summary: `D:\RAFA\outputs\circleworld_proto\phase_token_repair_2026-05-05\phase_token_repair_summary.json`
- Postfix separated heldout: `C:\Users\Andrew\AppData\Local\Temp\rafa_tokenburst\phase_token_repair_branch_diag_v2\heldout_eval_postfix_separated\heldout_summary.json`
- Naked phase profile summary: `C:\Users\Andrew\AppData\Local\Temp\rafa_tokenburst\naked_phase_nested_recovery_v1_smoke\childworld_experiment_summary.json`
