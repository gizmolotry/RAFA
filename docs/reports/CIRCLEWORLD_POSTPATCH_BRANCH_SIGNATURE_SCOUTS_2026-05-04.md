# Circleworld Post-Patch Scout Report

- generated_utc: `2026-05-04T22:25:49.111482Z`

## Branch References

- active `agreement_scout_v1`: bench corr `0.913847`, heldout branch `0.577778`, naked branch `0.4`, nested frac `0.083333`
- diversity `parentmix_220_100`: bench corr `0.896369`, heldout branch `0.577778`, nested frac `0.0`

## New Branch Scouts

### branch_identity_guard_safe
- benchmark corr / mae: `0.8906` / `0.0302`
- heldout branch / naked branch: `0.5556` / `0.3333`
- naked parent div / writeback: `0.0432` / `0.0896`
- nested overall: `nested_commitment_not_yet_established`
- nested sibling frac / qualified carry / readout sibling: `0.0000` / `0.0000` / `0.5305`
- nested live-child cases: `1` eligible: `1`
- nested case json: `None`

### branch_balance_guard_seeded
- benchmark corr / mae: `0.8906` / `0.0302`
- heldout branch / naked branch: `0.5556` / `0.3333`
- naked parent div / writeback: `0.0428` / `0.0819`
- nested overall: `nested_commitment_not_yet_established`
- nested sibling frac / qualified carry / readout sibling: `0.0833` / `0.8438` / `0.2867`
- nested live-child cases: `3` eligible: `3`
- nested case json: `D:\RAFA\outputs\circleworld_proto\tokenburst_2026-05-04_branch_balance_guard_run_seeded\nested_seeded_branch_active_cases.json`

## Signature Reference

- `relsig_balance_guard_v1`: bench corr `0.9801`, heldout signature families `2.3333`, export families `8.0`, metamer drift `0.2708`

## New Signature Scouts

### balance_guard_v2
- benchmark corr / mae: `0.9880` / `0.0176`
- heldout branch: `0.4074`
- heldout signature families / confidence: `1.1111` / `0.8236`
- export signature families: `11.0`
- metamer prefix cos / family drift: `0.9957` / `0.2667`

### export_diversity_floor_v1
- benchmark corr / mae: `0.9057` / `0.0470`
- heldout branch: `0.5778`
- heldout signature families / confidence: `1.0000` / `0.8446`
- export signature families: `5.0`
- metamer prefix cos / family drift: `0.9947` / `0.2167`

### branchmass_balance_v1
- benchmark corr / mae: `0.9829` / `0.0223`
- heldout branch: `0.4127`
- heldout signature families / confidence: `2.3333` / `0.8145`
- export signature families: `13.0`
- metamer prefix cos / family drift: `0.9952` / `0.2250`

## Recommendation

- Hold `agreement_scout_v1` as the branch-first active checkpoint.
- Treat `branch_balance_guard_seeded` as the strongest new branch-ontology probe, not a promotion yet.
- Hold `relsig_balance_guard_v1` as the signature-lane init for now.
- Treat `branchmass_balance_v1` as the strongest new signature candidate and the best follow-on init for the next signature-lane run if we choose to advance that lane.
