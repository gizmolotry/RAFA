# Circleworld Agreement Scout v1

## Result

- The new agreement-aware selector produced a real promoted candidate.
- selection source: `best_agreement_candidate`
- This candidate fails the replay probe gate but passes the true heldout gate.
- That is exactly the distinction we needed the selector to make.

## Agreement-selected candidate

- checkpoint: `D:\RAFA\checkpoints_circleworld_proto\training_run_2026-04-24_agreement_scout_v1\circleworld_real_anchor_config_cem_v1.json`
- heldout branch=0.577778 meso=0.014198 parent_div=0.141187
- heldout naked branch=0.400000 meso=0.003559 parent_div=0.042934 writeback=0.082903
- benchmark corr/mae=0.913847/0.045055
- continuity loop_peak=0.571026 nonlocal_repeat=0.954680
- nested=nested_commitment_not_yet_established {'over_rigid_attractor': 2}
- law mean families=2.250000 aggregate families=13

## Comparison to stable baseline

- baseline heldout naked branch=0.400000
- baseline heldout naked parent_div=0.036022
- baseline benchmark corr/mae=0.896369/0.049241

Read:
- Agreement scout v1 improves benchmark fidelity over the stable baseline while also clearing the heldout naked branch bar under the new selector.
- Nested commitment is still not established; this is a better candidate-selection constitution, not yet a Matryoshka win.

## Artifacts

- compare json: `D:\RAFA\outputs\circleworld_proto\agreement_scout_compare_2026-04-24.json`
- benchmark: `D:\RAFA\outputs\circleworld_proto\training_run_2026-04-24_agreement_scout_v1\benchmark_expanded\benchmark_summary.json`
- continuity: `D:\RAFA\outputs\circleworld_proto\training_run_2026-04-24_agreement_scout_v1\benchmark_expanded\continuity_circleworld.json`
- nested: `D:\RAFA\outputs\circleworld_proto\training_run_2026-04-24_agreement_scout_v1\nested_commitment\nested_commitment_report.json`
- law token library: `D:\RAFA\outputs\circleworld_proto\training_run_2026-04-24_agreement_scout_v1\law_token_library\law_token_library.json`
