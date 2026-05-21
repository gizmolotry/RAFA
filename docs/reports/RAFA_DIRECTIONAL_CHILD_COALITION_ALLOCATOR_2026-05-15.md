# RAFA Directional Child Coalition Allocator - 2026-05-15

## Scope

This push tests whether the many-child failure is caused by anti-aligned child return vectors.

Previous result:

- scalar developmental allocation stayed safe but produced negative parent lift;
- support overlap and cancellation dominated;
- next hypothesis was that we need to select an aligned coalition of child returns.

This run adds directional coalition variants inside the assay-only many-child runtime sandbox.

## Code

Updated:

- `D:\RAFA\runtimes\circleworld_proto\run_learned_gated_manychild_runtime_sandbox.py`

New variants:

- `learned_soft_directional_coalition`
- `learned_soft_directional_coalition_boost`

New metrics:

- `mean_directional_coalition_size`
- `mean_directional_alignment`
- `mean_directional_opposition`

The allocator:

- computes each child return direction as a support/coherence-gated phase delta from parent mode 1 to child phase;
- scores child return vectors by learned writeback, jump safety, recurrence, and developmental drive;
- chooses a seed coalition by positive vector alignment;
- suppresses anti-aligned children;
- optionally boosts the aligned coalition to test under-allocation.

## Verification

```powershell
python -m py_compile D:\RAFA\runtimes\circleworld_proto\run_learned_gated_manychild_runtime_sandbox.py
python -m pytest D:\RAFA\runtimes\circleworld_proto\test_learned_branch_law_child_ifs_contract.py D:\RAFA\runtimes\circleworld_proto\test_resonant_attention_contract.py -q
```

Result:

```text
23 passed
```

## Outputs

Directional runs:

- `D:\RAFA\outputs\circleworld_proto\directional_child_coalition_allocator_2026-05-15_authority_d3_soft70\learned_gated_manychild_runtime_sandbox.json`
- `D:\RAFA\outputs\circleworld_proto\directional_child_coalition_allocator_2026-05-15_authority_d3_strict88\learned_gated_manychild_runtime_sandbox.json`
- `D:\RAFA\outputs\circleworld_proto\directional_child_coalition_allocator_2026-05-15_carrier_d2_strict88\learned_gated_manychild_runtime_sandbox.json`

Boost runs:

- `D:\RAFA\outputs\circleworld_proto\directional_child_coalition_boost_2026-05-15_carrier_d2_strict88\learned_gated_manychild_runtime_sandbox.json`
- `D:\RAFA\outputs\circleworld_proto\directional_child_coalition_boost_2026-05-15_authority_d3_soft70\learned_gated_manychild_runtime_sandbox.json`

Compare:

- `D:\RAFA\outputs\circleworld_proto\directional_child_coalition_allocator_compare_2026-05-15\directional_child_coalition_allocator_compare.json`
- `D:\RAFA\outputs\circleworld_proto\directional_child_coalition_allocator_compare_2026-05-15\DIRECTIONAL_CHILD_COALITION_ALLOCATOR_COMPARE.md`

## Key Results

### Depth-2 Strict88

Directional coalition:

- mean scale: `0.11297374748018668`
- child writeback mass: `0.02032736983709057`
- coalition size: `10.75`
- mean alignment: `0.7717080563306808`
- parent lift vs blocked: `-0.003532763570547104`
- final world jump: `0.0`

Boosted directional coalition:

- mean scale: `0.20900103449596342`
- child writeback mass: `0.03809397295117378`
- coalition size: `10.75`
- mean alignment: `0.7710791081190109`
- parent lift vs blocked: `-0.015024314323862391`
- final world jump: `0.0`

Interpretation: unboosted directional coalition gets closest to safe positive parent lift, but still stays slightly negative. Boosting the same aligned coalition increases writeback and worsens stabilization/cancellation.

### Depth-3 Soft70

Directional coalition:

- mean scale: `0.20772175800165817`
- child writeback mass: `0.038905318826436996`
- coalition size: `12.75`
- mean alignment: `0.7499839216470718`
- parent lift vs blocked: `-0.0167632345110178`
- final world jump: `0.0`

Boosted directional coalition:

- mean scale: `0.40057345828600246`
- child writeback mass: `0.07657126896063487`
- coalition size: `12.75`
- mean alignment: `0.750737763941288`
- parent lift vs blocked: `-0.02512061161319414`
- final world jump: `0.0009580949942270914`

Unallocated Soft70 remained the only strong positive movement case:

- parent lift vs blocked: `0.5043135386955135`
- jump lift vs blocked: `0.31183378228320974`

So it is not acceptable as branch ontology evidence.

## Interpretation

This is another useful negative, but sharper than the previous one.

The child returns are not random anti-aligned noise:

- directional coalition alignment is high, around `0.75-0.77`;
- directional opposition is `0.0` in these probes;
- coalition sizes are large, around `10.75-12.75` children.

The failure is subtler:

> the aligned coalition direction itself is stabilizing, not developmental.

So the next bottleneck is not just child count, not scalar score, not budget, and not simple vector alignment.

The parent needs a target-conditioned developmental direction: a direction of allowable sibling continuation. Then child coalitions should be selected by alignment to that target, not only by mutual alignment.

## Next Step

Implement a target-conditioned directional assay:

- derive a target direction from the safe/unallocated gap or a controlled sibling probe;
- score child return vectors by projection onto that target;
- select a coalition that is aligned to the target and mutually coherent;
- reject coalitions that are coherent but point back into stabilizing cancellation;
- report target projection, mutual alignment, parent lift, and jump.

Working label:

`target_conditioned_child_coalition_allocator_v0`
