# RAFA Ecology Writeback Allocator - 2026-05-15

## Scope

This push extends the many-child learned runtime sandbox with ecology-level writeback allocation.

The previous result showed that independent per-child learned permission over-permits when many live child worlds coexist. The allocator tests a stronger rule:

> A child is not allowed to write back merely because it is locally lawful. It must share parent authority with overlapping siblings.

## Code

Updated:

- `D:\RAFA\runtimes\circleworld_proto\run_learned_gated_manychild_runtime_sandbox.py`

New variants:

- `learned_hard_alloc`
- `learned_soft_alloc`
- `learned_soft_alloc_loose`

The allocator uses:

- learned writeback score,
- predicted jump safety,
- support-overlap load,
- same-family redundancy,
- global mean-scale budget.

It remains assay-only and acts only by assigning each child's `assay_writeback_scale` before runtime writeback.

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

Allocator runs:

- `D:\RAFA\outputs\circleworld_proto\learned_gated_manychild_allocator_2026-05-15_carrier_d2_soft70\learned_gated_manychild_runtime_sandbox.json`
- `D:\RAFA\outputs\circleworld_proto\learned_gated_manychild_allocator_2026-05-15_authority_d3_soft70\learned_gated_manychild_runtime_sandbox.json`
- `D:\RAFA\outputs\circleworld_proto\learned_gated_manychild_allocator_2026-05-15_carrier_d2_strict88\learned_gated_manychild_runtime_sandbox.json`
- `D:\RAFA\outputs\circleworld_proto\learned_gated_manychild_allocator_2026-05-15_authority_d3_strict88\learned_gated_manychild_runtime_sandbox.json`

Allocator profile runs:

- `D:\RAFA\outputs\circleworld_proto\learned_gated_manychild_allocator_profiles_2026-05-15_authority_d3_soft70\learned_gated_manychild_runtime_sandbox.json`
- `D:\RAFA\outputs\circleworld_proto\learned_gated_manychild_allocator_profiles_2026-05-15_authority_d3_strict88\learned_gated_manychild_runtime_sandbox.json`
- `D:\RAFA\outputs\circleworld_proto\learned_gated_manychild_allocator_profiles_2026-05-15_carrier_d2_strict88\learned_gated_manychild_runtime_sandbox.json`

Combined compare:

- `D:\RAFA\outputs\circleworld_proto\ecology_writeback_allocator_compare_2026-05-15\ecology_writeback_allocator_compare.json`
- `D:\RAFA\outputs\circleworld_proto\ecology_writeback_allocator_compare_2026-05-15\ECOLOGY_WRITEBACK_ALLOCATOR_COMPARE.md`

## Key Results

### Depth-3 Soft70

Unallocated learned soft:

- mean scale: `0.8828476185020145`
- writeback mass: `0.166703167061011`
- final world jump: `0.3155106890115032`
- parent lift vs blocked: `0.5043135386955135`

Conservative allocator:

- mean scale: `0.05803588088857786`
- writeback mass: `0.011764404596139478`
- final world jump: `0.0`
- parent lift vs blocked: `-0.02179128676652908`

Loose allocator:

- mean scale: `0.23759075191617012`
- writeback mass: `0.04536004116106778`
- final world jump: `0.0`
- parent lift vs blocked: `-0.028025000045696894`

Interpretation: allocation fixes the Soft70 jump problem, and loose allocation preserves more child return than conservative allocation. But both behave as stabilizers, not sibling-development carriers.

### Depth-3 Strict88

Unallocated learned soft:

- mean scale: `0.5072239750914929`
- writeback mass: `0.09030687188108762`
- final world jump: `0.004862337754881096`
- parent lift vs blocked: `-0.013576438969279264`

Loose allocator:

- mean scale: `0.13244199047647253`
- writeback mass: `0.024074798604473275`
- final world jump: `0.0`
- parent lift vs blocked: `-0.01946932077407837`

Interpretation: strict unallocated soft was already near safe. Allocation makes it safer but reduces child authority further.

### Depth-2 Strict88

Unallocated learned soft:

- writeback mass: `0.08547393729289372`
- final world jump: `0.0018601853963004695`
- parent lift vs blocked: `-0.026189030122391543`

Loose allocator:

- writeback mass: `0.02506202335158984`
- final world jump: `0.0`
- parent lift vs blocked: `-0.02221079667409261`

Interpretation: in this branch substrate, allocated child return can be nonzero and safe, but it is not yet producing positive parent-development lift over blocked baseline.

## What We Learned

The allocator confirms the right architectural correction but exposes the next bottleneck.

Confirmed:

- many-child branch law needs authority allocation, not independent child permission;
- support-overlap pressure is real and huge, around `7.5-8.6` in these probes;
- allocated writeback can keep nonzero child mass while removing world-jump.

Not yet solved:

- allocated writeback is currently stabilizing/canceling the parent rather than creating positive sibling development;
- the allocator is overlap-pressure dominated, not global-budget dominated;
- selected-child causal usefulness does not yet translate into many-child ontology growth.

## Next Step

The next branch-law target should separate two authority channels:

1. stabilizing authority: child return that reduces jump/cancels turbulence;
2. developmental authority: child return that moves parent into a sibling continuation.

Right now the learned allocator finds stabilizers. We still need a developmental carrier objective.
