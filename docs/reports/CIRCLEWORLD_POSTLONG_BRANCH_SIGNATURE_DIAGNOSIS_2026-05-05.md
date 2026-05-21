# Circleworld Post-Long Diagnosis

- generated_utc: `2026-05-05T11:37:51.273489+00:00`

## Key Result

The branch runner is now structurally correct for ontology work, but the branch-recovery search is still falling back to the same seeded branch-balance basin.

## Long Recovery Run

- benchmark corr / mae: `0.8906` / `0.0302`
- heldout branch / naked branch: `0.5556` / `0.3333`
- naked parent div / writeback: `0.0428` / `0.0819`
- seeded nested sibling frac / qualified carry: `0.0833` / `0.8438`
- train selection source: `init_config_fallback_nested_or_agreement_required`

## Conclusions

- Repairing the childworld runner and switching branch profiles to seeded branch-active nested cases was necessary and successful.
- branch_balance_guard_v1 remains the strongest new branch-ontology probe because it preserves naked branching and nonzero seeded nested sibling signal.
- branch_recovery_soft_guard_v1 did not discover a better checkpoint even after seeded-selection alignment and a larger 2x4 search; the trainer still fell back to the init basin.
- export_diversity_floor_v1 and branchmass_plurality_v1 both weakened signature plurality relative to branchmass_balance_v1, so branchmass_balance_v1 remains the strongest new signature-lane candidate.
- The trainer summary now exports best-candidate nested and heldout probes even when selection falls back, which should make the next cycle easier to diagnose.

## Current Best

- branch-first active: `D:\RAFA\checkpoints_circleworld_proto\training_run_2026-04-24_agreement_scout_v1\circleworld_real_anchor_config_cem_v1.json`
- strongest new branch probe: `D:\RAFA\checkpoints_circleworld_proto\tokenburst_2026-05-04_branch_balance_guard_run_seeded\circleworld_real_anchor_config_cem_v1.json`
- strongest new signature candidate: `D:\RAFA\checkpoints_circleworld_proto\tokenburst_2026-05-04_branchmass_balance_v1_run_retry\circleworld_real_anchor_config_cem_v1.json`
## Signature Follow-up On C:

- profile: `balance_guard_v2`
- init config: `D:\RAFA\checkpoints_circleworld_proto\tokenburst_2026-05-04_branchmass_balance_v1_run_retry\circleworld_real_anchor_config_cem_v1.json`
- benchmark corr / mae: `0.9924` / `0.0172`
- heldout branch: `0.3889`
- heldout signature families / confidence: `1.7778` / `0.6272`
- export signature families: `4.0`
- metamer prefix cos / family drift: `0.9969` / `0.1750`

- Read: the init was not the main problem here. `balance_guard_v2` still re-collapses plurality when started from the stronger `branchmass_balance_v1` checkpoint.
