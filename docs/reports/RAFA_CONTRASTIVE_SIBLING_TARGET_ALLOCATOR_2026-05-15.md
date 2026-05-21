# RAFA Contrastive Sibling Target Allocator - 2026-05-15

## Scope

This pass tests whether many child worlds can be routed toward an **exogenous sibling-development direction** instead of a target derived from their own stabilizing return field.

The new variant is `learned_soft_contrastive_sibling_target` in `D:\RAFA\runtimes\circleworld_proto\run_learned_gated_manychild_runtime_sandbox.py`.

## What Changed

- Added a controlled tangent sibling target built from parent/mode geometry and the child union support mask.
- Projected each child return onto that external target after removing collapse-to-mode0 direction.
- Allocated learned-gated writeback authority only to positively projecting children.
- Added aggregate metrics: `mean_contrastive_target_projection` and `mean_contrastive_target_norm`.

## Verification

```powershell
python -m py_compile D:\RAFA\runtimes\circleworld_proto\run_learned_gated_manychild_runtime_sandbox.py
python -m pytest D:\RAFA\runtimes\circleworld_proto\test_learned_branch_law_child_ifs_contract.py D:\RAFA\runtimes\circleworld_proto\test_resonant_attention_contract.py -q
```

Result: `23 passed`.

## Outputs

- `D:\RAFA\outputs\circleworld_proto\contrastive_sibling_target_allocator_2026-05-15_carrier_d2_strict88\learned_gated_manychild_runtime_sandbox.json`
- `D:\RAFA\outputs\circleworld_proto\contrastive_sibling_target_allocator_2026-05-15_authority_d3_soft70\learned_gated_manychild_runtime_sandbox.json`
- `D:\RAFA\outputs\circleworld_proto\contrastive_sibling_target_allocator_2026-05-15_authority_d3_strict88\learned_gated_manychild_runtime_sandbox.json`
- `D:\RAFA\outputs\circleworld_proto\contrastive_sibling_target_allocator_compare_2026-05-15\contrastive_sibling_target_allocator_compare.json`
- `D:\RAFA\outputs\circleworld_proto\contrastive_sibling_target_allocator_compare_2026-05-15\CONTRASTIVE_SIBLING_TARGET_ALLOCATOR_COMPARE.md`

## Results

| substrate | variant | target proj | ctarget proj | coalition | writeback | parent lift | world jump |
|---|---|---:|---:|---:|---:|---:|---:|
| carrier_d2_strict88 | directional | 0.0 | 0.0 | 10.75 | 0.02032653180261453 | -0.0035328819914917514 | 0.0 |
| carrier_d2_strict88 | endogenous_target | 0.8007056265617861 | 0.0 | 10.75 | 0.026376886448512476 | -0.016380775896543395 | 0.0 |
| carrier_d2_strict88 | contrastive_sibling_target | 0.005702414401722225 | 0.005702414401722225 | 4.0 | 0.0009855892179378618 | -0.000415575091684911 | 0.0031114215035187667 |
| authority_d3_soft70 | directional | 0.0 | 0.0 | 12.75 | 0.03890549826125304 | -0.016763039558578524 | 0.0 |
| authority_d3_soft70 | endogenous_target | 0.7894824966551766 | 0.0 | 12.75 | 0.05136618080238501 | -0.020164635455283672 | 0.0 |
| authority_d3_soft70 | contrastive_sibling_target | 0.00396335497073504 | 0.00396335497073504 | 4.5 | 0.001806910508700336 | -6.672725653855682e-06 | 0.002702149397135608 |
| authority_d3_strict88 | directional | 0.0 | 0.0 | 12.75 | 0.02092209116866191 | -0.004421707317873155 | 0.0 |
| authority_d3_strict88 | endogenous_target | 0.7868977775058102 | 0.0 | 12.75 | 0.027093247200051945 | -0.018234978024178256 | 0.0 |
| authority_d3_strict88 | contrastive_sibling_target | 0.004569303280016083 | 0.004569303280016083 | 4.75 | 0.000962287721146519 | -0.0006293415563105864 | 0.0030543509946701817 |

## Key Interpretation

The contrastive allocator is safe but nearly inert:

- depth-2 strict88 contrastive target projection is about `0.0057`, with writeback mass about `0.00099` and parent lift about `-0.00042`.
- depth-3 soft70 contrastive target projection is about `0.0040`, with writeback mass about `0.00181` and parent lift about `-0.0000067`.
- depth-3 strict88 contrastive target projection is about `0.0046`, with writeback mass about `0.00096` and parent lift about `-0.00063`.

Compared with the endogenous target allocator, this is less stabilizing/canceling, but only because it cannot find children aligned with the external sibling target. It reduces writeback authority by roughly one to two orders of magnitude.

So the current child bank has a real basis problem: it contains coherent returns, but those returns mostly align with stabilization/collapse directions, not controlled sibling-development directions.

## Updated Claim

The bottleneck is no longer just learned gating or ecology allocation. The child spawning/training process is not generating enough diverse operator directions. More child volume may help only if spawning is explicitly pressured to cover contrastive tangent sibling directions. Blindly spawning more copies of the same stabilizing basis will mostly increase cancellation or jump risk.

## Next Move

Train or generate child worlds against contrastive sibling targets directly:

- create sibling-target supervision from controlled tangent perturbations;
- add a child-basis coverage metric before writeback allocation;
- spawn/split children to maximize target-direction coverage under low world-jump constraints;
- rerun the allocator only after the basis contains nontrivial positive target projection.
