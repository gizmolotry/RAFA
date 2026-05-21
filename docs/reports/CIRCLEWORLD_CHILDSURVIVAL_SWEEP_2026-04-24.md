# Circleworld Child-Survival Sweep Retrospective

## Compared candidates

### parentmix_220_100_baseline
- config: `D:\RAFA\checkpoints_circleworld_proto\manual_candidates\circleworld_parentmix_220_100_2026-04-24.json`
- heldout: branch=0.577778 meso=0.014442 writeback=0.096885 parent_div=0.141993
- heldout naked_rafa: branch=0.400000 meso=0.003024 writeback=0.083952 parent_div=0.036022
- benchmark: corr=0.896369 mae=0.049241
- continuity: loop_peak=0.571037 nonlocal_repeat=0.955163 adjacent_similarity=0.890594
- nested: nested_commitment_not_yet_established {'over_rigid_attractor': 2}
- law library: mean_families=2.333333 aggregate_families=0.0

### survival_w28_d014_g600_probe_pass
- config: `D:\RAFA\outputs\circleworld_proto\childsurvival_sweep_2026-04-24_v1\configs\survival_w28_d014_g600.json`
- heldout: branch=0.444444 meso=0.015878 writeback=0.089827 parent_div=0.146577
- heldout naked_rafa: branch=0.000000 meso=0.001327 writeback=0.045341 parent_div=0.026542
- benchmark: corr=0.896179 mae=0.049285
- continuity: loop_peak=0.571032 nonlocal_repeat=0.954490 adjacent_similarity=0.890535
- nested: nested_commitment_not_yet_established {'over_rigid_attractor': 2}
- law library: mean_families=2.333333 aggregate_families=0.0

### survival_w28_d018_g520_heldout_best
- config: `D:\RAFA\outputs\circleworld_proto\childsurvival_sweep_2026-04-24_v1\configs\survival_w28_d018_g520.json`
- heldout: branch=0.577778 meso=0.017528 writeback=0.103738 parent_div=0.160294
- heldout naked_rafa: branch=0.400000 meso=0.004109 writeback=0.087903 parent_div=0.046746
- benchmark: corr=0.896414 mae=0.049235
- continuity: loop_peak=0.571035 nonlocal_repeat=0.954908 adjacent_similarity=0.890493
- nested: nested_commitment_not_yet_established {'over_rigid_attractor': 2}
- law library: mean_families=2.333333 aggregate_families=0.0

## Read

- `survival_w28_d014_g600` is the best replay-gate pass from the sweep, but its heldout `naked_rafa` branch fraction collapses to 0.0.
- `survival_w28_d018_g520` is the strongest actual heldout naked-branch candidate from the sweep, with better heldout naked parent divergence than baseline, but it fails the replay gate.
- Both survival variants remain `nested_commitment_not_yet_established` and stay in `over_rigid_attractor` on the fork/resume assay.
- Benchmark and continuity remain nearly unchanged across all three; the current fight is inside branch constitution, not broad audio fidelity.

## Recommendation

- Keep `circleworld_parentmix_220_100_2026-04-24.json` as the default stable Circleworld baseline.
- Treat `survival_w28_d018_g520` as the best real heldout naked-branch side candidate.
- Do not promote `survival_w28_d014_g600` as default despite the replay-gate pass, because its heldout naked branch collapses.
- Next search should explicitly optimize for agreement between replay-probe naked metrics and heldout naked metrics, not just either one alone.
