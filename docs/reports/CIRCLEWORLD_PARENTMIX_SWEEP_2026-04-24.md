# Circleworld Parent-Mix Sweep Report

## What Changed
- Fixed a runtime mismatch: `evaluate_circleworld.py` and `export_circleworld_audio.py` were not loading `child_parent_mix` and `child_parent_mix_early`.
- Reran parent-mix evaluation under the corrected runtime and then swept a focused grid around the parent writeback constitution.

## Sweep Outcome
- Sweep artifact: `D:\RAFA\outputs\circleworld_proto\parentmix_sweep_2026-04-24_v1\parentmix_sweep_summary.json`
- Variants tested: `8`
- Replay-gate passes: `2`
- Pass: `parentmix_220_100` with `child_parent_mix=0.220`, `child_parent_mix_early=0.100`, `naked_parent_div_min=0.043614`, `bench_corr=0.896369`
- Pass: `parentmix_260_100` with `child_parent_mix=0.260`, `child_parent_mix_early=0.100`, `naked_parent_div_min=0.041658`, `bench_corr=0.896071`

## Best Candidate
- Variant: `parentmix_220_100`
- Config: `D:\RAFA\outputs\circleworld_proto\parentmix_sweep_2026-04-24_v1\configs\parentmix_220_100.json`
- Promoted checkpoint copy: `D:\RAFA\checkpoints_circleworld_proto\manual_candidates\circleworld_parentmix_220_100_2026-04-24.json`
- Replay gate: `pass`
- `naked_parent_div_min`: `0.043614`
- `naked_writeback_min`: `0.082669`
- `naked_branch_min`: `0.400000`
- Benchmark corr / mae: `0.896369` / `0.049241`

## Sidecars
- Law-token library: mean families `2.333`, aggregate families `12`
- Continuity: loop peak `0.571037`, mean loop period `1.824s`, nonlocal repeat `0.955163`
- Nested commitment: `nested_commitment_not_yet_established` with verdicts `{'over_rigid_attractor': 2}`

## Read
- The replay-safe branch gate is now satisfiable with a focused parent-writeback constitution.
- That is a real improvement over the previous near-miss regime.
- But the winning config still fails the stronger ontology test: nested commitment remains over-rigid rather than sibling-branching.
- So the next search should start from this config, but optimize for lawful divergence under fork/resume instead of replay recovery alone.
