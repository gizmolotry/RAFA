# Circleworld Child-Local IFS Training Movement - 2026-05-10

## Decision

Hold. Do not promote the new child-local IFS checkpoint as active.

The newer system has now been trained in an isolated branch, not just smoke-tested. The weighted child-local checkpoint is mechanically real and benchmark-stable, but it does not beat the active agreement baseline on branch ontology. The failure is specific: held-out naked_rafa branch survival is zero for the trained child-local checkpoints, while the active baseline remains nonzero.

## Exact Movement

1. Created isolated branch: codex/circleworld-child-ifs-training-2026-05-10.
2. Took the active agreement scout checkpoint as the base config.
3. Enabled the new runtime path for isolated training: ranching_mode=native_multimode_childworld, child_local_ifs_enabled=true, child_local_steps=2, child_local_support_only=true.
4. Ran first real-anchor CEM pass with downstream branch-child pressure.
5. Patched aggregate reporting so child-local recurrence metrics are surfaced in held-out and training summaries.
6. Added optional CEM score hooks for mean_child_local_ifs_step_count, mean_child_local_ifs_packet_count, mean_child_local_ifs_phase_delta, mean_child_local_ifs_support, and mean_child_local_ifs_coherence.
7. Ran second real-anchor CEM pass with direct child-local recurrence pressure.
8. Reran held-out evaluation on active baseline, child-local v1, and child-local v2 weighted.
9. Ran learned branch-law/child-local IFS assay on all three configs.
10. Ran real-anchor benchmark smoke on active baseline and child-local v2 weighted.

## Held-Out Comparison

| config | real branch | naked branch | synthetic branch | meso | live child | writeback | parent div | sibling div | local steps | local packets | local delta | local coherence |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: || agreement_baseline_patched | 0.577778 | 0.4 | 0.666667 | 0.030183 | 0.577778 | 0.12488 | 0.22584 | 0.169703 | 0 | 0 | 0 | 0 |
| child_local_v1_patched | 0.380952 | 0 | 0.571429 | 0.022155 | 0.380952 | 0.079324 | 0.186195 | 0.178024 | 0.888889 | 4.444445 | 0.001149 | 0.308969 |
| child_local_v2_weighted | 0.444444 | 0 | 0.666667 | 0.02902 | 0.444444 | 0.091739 | 0.210885 | 0.197331 | 0.888889 | 5.333333 | 0.002054 | 0.295587 |

## Learned Branch-Law / Child-Local IFS Assay

| config | live child cases | sibling cases | isolated coherence | coupled-isolated delta | writeback div | sibling delta | final shadow loss |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: || agreement_baseline | 3 | 3 | 0.763071 | 0.005362 | 0.079642 | 0.421111 | 0.037223 |
| child_local_v1 | 3 | 3 | 0.725778 | 0.009557 | 0.016418 | 0.351982 | 0.034164 |
| child_local_v2_weighted | 3 | 3 | 0.701301 | 0.005621 | 0.054408 | 0.461545 | 0.034012 |

## Benchmark Smoke

| config | mean corr | mean mse | mean mae | bitwise stable | recurrence delta | severity |
| --- | ---: | ---: | ---: | --- | ---: | --- || benchmark_agreement_baseline | 0.970819 | 0.00322405823644658 | 0.026708 | True | -4.94652931733175E-05 | near_identity |
| benchmark_child_local_v2_weighted | 0.999745 | 3.06746547266812E-05 | 0.002712 | True | -2.85110600398153E-06 | near_identity |

## Interpretation

The new system was trained for real. It is not merely an untrained runtime switch anymore.

The direct child-local recurrence pressure did what it was supposed to do locally: v2 increased held-out local packet activity, local phase delta, parent divergence, and sibling divergence relative to v1. It also preserved benchmark/audio stability.

The promotion blocker is branch ontology on the seeded/naked substrate. Both trained child-local checkpoints report
aked_branch=0.0, while the active baseline reports
aked_branch=0.400000. That means the new child-local mechanism is still learning a synthetic-favorable child survival strategy, not the seeded branch-active ontology we actually care about.

## Next Move

The next training run should include seeded/naked branch-active cases directly in the CEM objective, not only as post-hoc held-out evaluation. The child-local IFS mechanism is viable enough to keep, but it needs substrate-specific pressure before promotion.
