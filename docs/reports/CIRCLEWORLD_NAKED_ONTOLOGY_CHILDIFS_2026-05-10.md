# Circleworld Naked Ontology Child-IFS Run - 2026-05-10

## Decision

Hold. Do not declare a new active checkpoint yet.

This is a real positive movement over the previous child-local IFS checkpoint: seeded naked branch survival is restored and naked child-local IFS is active in external held-out evaluation. The remaining blocker is nested ontology: no
ested_sibling yet, and learned branch-law sandbox control is still too weak to count as a behavioral learned branch law.

## What Changed

- Added assay-only learned-law sandbox control to
un_learned_branch_law_assay.py.
- Ran a focused CEM pass with seeded
aked_rafa cases 9100/9101/9102 inside candidate selection.
- Used hard selection pressure against zero naked branch/writeback.
- Selected checkpoint source: $(@{runtime=circleworld_proto; schema=circleworld_v0_real_anchor; status=completed; trainer=cem_real_anchor; selection_source=best_agreement_candidate; seed=20260512; device=cuda; iterations=3; population=6; elite_count=2; clip_seconds=3; phase_blend=0.65; checkpoint=D:\RAFA\checkpoints_circleworld_proto\naked_ontology_childifs_2026-05-10\circleworld_real_anchor_config_cem_v1.json; baseline_train=; baseline_val=; best_train=; best_val=; best_transfer_probe=; best_candidate=; best_passing_candidate=; best_agreement_candidate=; best_nested_pending_candidate=; selection_fallback_diagnostics=; selection_agreement_candidates=System.Object[]; selected_candidate=; best_candidate_gate=; selected_candidate_gate=; selected_candidate_heldout_gate=; selected_candidate_nested_gate=; best_candidate_transfer_probe=; best_candidate_heldout_probe=; best_candidate_heldout_gate=; best_candidate_nested_gate=; best_candidate_nested_probe=; selected_candidate_transfer_probe=; selected_candidate_heldout_probe=; selected_candidate_nested_probe=; best_config=; artifact_paths=}.selection_source).

## New Checkpoint

D:\RAFA\checkpoints_circleworld_proto\naked_ontology_childifs_2026-05-10\circleworld_real_anchor_config_cem_v1.json

## Held-Out Evaluation

| config | real branch | naked branch | synthetic branch | naked writeback | naked parent div | naked local steps | naked local packets | naked local delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: || agreement_baseline | 0.577778 | 0.4 | 0.666667 | 0.102351 | 0.050027 | 0 | 0 | 0 |
| child_local_v2_weighted | 0.444444 | 0 | 0.666667 | 0 | 0 | 0 | 0 | 0 |
| naked_ontology_childifs | 0.577778 | 0.4 | 0.666667 | 0.102178 | 0.056503 | 1.333333 | 8 | 0.067336 |

Interpretation: the previous child-local v2 checkpoint had
aked_branch=0.0 and zero naked child-local recurrence. The targeted run restores
aked_branch=0.400000,
aked_writeback=0.102178,
aked_local_steps=1.333333, and
aked_local_packets=8.0.

## Learned-Law Sandbox

| config | loss | isolated coherence | writeback div | sibling delta | sandbox phase delta | sandbox live-child delta | sandbox writeback delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: || agreement_baseline | 0.037223 | 0.763071 | 0.079642 | 0.421111 | 0.000645 | 0 | 0 |
| child_local_v2_weighted | 0.034012 | 0.701301 | 0.054408 | 0.461545 | 4.3E-05 | 0 | 0 |
| naked_ontology_childifs | 0.035378 | 0.669767 | 0.043253 | 0.523814 | 0.000258 | 0 | 0 |

Interpretation: the learned branch-law network fits the shadow surface, but sandbox control does not yet move live-child or writeback outcomes. This is not yet a trained neural branch law in the strong sense.

## Benchmark Smoke

| config | mean corr | mean mse | mean mae | bitwise stable | recurrence delta | severity |
| --- | ---: | ---: | ---: | --- | ---: | --- || agreement_baseline | 0.970819 | 0.00322405823644658 | 0.026708 | True | -4.94652931733175E-05 | near_identity |
| naked_ontology_childifs | 0.976594 | 0.00253519775837958 | 0.02481 | True | -3.2037453117334E-05 | near_identity |

## Nested Commitment

| config | nested sibling | identity carry | qualified carry | parent div | sibling div | selected live fork |
| --- | ---: | ---: | ---: | ---: | ---: | ---: || nested_agreement_baseline | 0 | 0.9375 | 0.045833 | 0.083977 | 0.069969 | 1 |
| nested_naked_ontology_childifs | 0 | 0.424306 | 0.045833 | 0.081968 | 0.082436 | 1 |

Interpretation: nested ontology is still the blocker. The targeted checkpoint has live child forks and higher sibling divergence, but it does not produce
ested_sibling, and identity carry is lower than the agreement baseline.

## What We Learned

1. The next-step hypothesis was correct: putting seeded naked branch-active cases into selection fixes the synthetic-only failure.
2. Child-local IFS can be active on the naked substrate after targeted selection.
3. Branch survival/writeback and nested sibling identity are separate claims. This run improves the former, not the latter.
4. The learned branch-law module is still shadow/sandbox only. It is not yet an effective controller.

## Next Step

Train specifically against nested identity carry, not just naked branch survival. The next objective should include same_child_qualified_carry_fraction, same_child_budget_retained, and nested_sibling_fraction in candidate selection, while preserving the new naked child-local IFS floor.
