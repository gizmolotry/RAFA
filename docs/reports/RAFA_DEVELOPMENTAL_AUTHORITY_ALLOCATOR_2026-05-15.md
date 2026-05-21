# RAFA Developmental Authority Allocator - 2026-05-15

## Scope

This push separates **stabilizing authority** from **developmental authority** in the many-child learned writeback sandbox.

Previous result:

- allocation solved many-child jump risk;
- allocated children still behaved mostly like stabilizers/cancelers;
- parent-development lift stayed negative relative to blocked baseline.

This run adds `learned_soft_develop_alloc`, an assay-only allocator profile that rewards safe parent movement directly.

## Code

Updated:

- `D:\RAFA\runtimes\circleworld_proto\run_learned_gated_manychild_runtime_sandbox.py`

New variant:

- `learned_soft_develop_alloc`

New allocator metric:

- `mean_developmental_drive`

Developmental drive uses:

- per-child parent phase divergence,
- predicted jump safety,
- recurrence compatibility,
- causal world-jump penalty,
- support-overlap and redundancy competition.

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

Runs:

- `D:\RAFA\outputs\circleworld_proto\developmental_authority_allocator_2026-05-15_authority_d3_soft70\learned_gated_manychild_runtime_sandbox.json`
- `D:\RAFA\outputs\circleworld_proto\developmental_authority_allocator_2026-05-15_authority_d3_strict88\learned_gated_manychild_runtime_sandbox.json`
- `D:\RAFA\outputs\circleworld_proto\developmental_authority_allocator_2026-05-15_carrier_d2_strict88\learned_gated_manychild_runtime_sandbox.json`

Compare:

- `D:\RAFA\outputs\circleworld_proto\developmental_authority_allocator_compare_2026-05-15\developmental_authority_allocator_compare.json`
- `D:\RAFA\outputs\circleworld_proto\developmental_authority_allocator_compare_2026-05-15\DEVELOPMENTAL_AUTHORITY_ALLOCATOR_COMPARE.md`

## Key Results

### Depth-3 Soft70

Unallocated soft:

- parent lift vs blocked: `0.5043135386955135`
- jump lift vs blocked: `0.31183378228320974`
- final world jump: `0.3155106890115032`

Loose stabilizing allocator:

- scale: `0.23759149703118562`
- writeback mass: `0.045360103249549866`
- parent lift vs blocked: `-0.02802545208970096`
- final world jump: `0.0`

Developmental allocator:

- scale: `0.2631924353365912`
- writeback mass: `0.04814476644061506`
- developmental drive: `0.24449328776166786`
- parent lift vs blocked: `-0.024908808370431263`
- final world jump: `0.0`

Interpretation: the developmental profile increases writeback versus the loose stabilizer and keeps jump at zero, but it still does not produce positive parent-development lift.

### Depth-3 Strict88

Developmental allocator:

- scale: `0.14640071797766648`
- writeback mass: `0.025355910261472065`
- developmental drive: `0.24356995032921718`
- parent lift vs blocked: `-0.02042111486196518`
- final world jump: `0.0`

### Depth-2 Strict88

Developmental allocator:

- scale: `0.147942061267377`
- writeback mass: `0.02599087357521057`
- developmental drive: `0.24147776156032085`
- parent lift vs blocked: `-0.019192957629760106`
- final world jump: `0.0`

## Interpretation

This is a useful negative result.

We now have three distinct regimes:

1. **Unallocated many-child writeback**: strong parent movement, but jumpy.
2. **Stabilizing allocated writeback**: nonzero child mass, safe, but parent movement is reduced below blocked baseline.
3. **Developmental allocated writeback**: slightly more child mass than stabilizer, still safe, but still no positive parent lift.

So the current bottleneck is not just authority budget. It is **directional coherence of child returns**.

The children may each carry a locally lawful return, but when many child returns coexist they cancel or stabilize instead of coherently pushing the parent into a sibling continuation.

## Next Step

The next objective should become directional, not scalar:

- estimate a child return direction in phase space;
- cluster compatible child return directions;
- allocate authority to a coherent directional coalition;
- suppress anti-aligned child returns even if individually lawful;
- measure coalition parent lift and jump separately.

Working label:

`directional_child_coalition_allocator_v0`

This is the first clear evidence that the child ecology problem is not simply volume, not simply threshold, and not simply soft vs hard gate. It is a vector/coalition problem.
