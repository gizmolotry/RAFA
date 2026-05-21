# RAFA Target-Conditioned Child Coalition - 2026-05-15

## Scope

This push tests whether target conditioning can turn a mutually aligned but stabilizing child coalition into a sibling-development carrier.

Previous result:

- mutually aligned child return vectors were coherent, not noisy;
- boosting aligned coalition authority worsened cancellation;
- therefore the coherent direction itself was stabilizing.

This run adds `learned_soft_target_coalition`, an assay-only allocator that builds a target direction by removing collapse-back-to-mode0 components from child return vectors.

## Code

Updated:

- `D:\RAFA\runtimes\circleworld_proto\run_learned_gated_manychild_runtime_sandbox.py`

New variant:

- `learned_soft_target_coalition`

New metrics:

- `mean_target_projection`
- `mean_target_norm`
- `mean_collapse_component_abs`

The allocator:

- computes each child return vector;
- computes the collapse direction from parent mode 1 back toward parent mode 0;
- removes collapse-direction projection to form tangent child returns;
- builds a weighted target from tangent returns;
- allocates authority by projection onto that target.

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

- `D:\RAFA\outputs\circleworld_proto\target_conditioned_child_coalition_2026-05-15_authority_d3_soft70\learned_gated_manychild_runtime_sandbox.json`
- `D:\RAFA\outputs\circleworld_proto\target_conditioned_child_coalition_2026-05-15_authority_d3_strict88\learned_gated_manychild_runtime_sandbox.json`
- `D:\RAFA\outputs\circleworld_proto\target_conditioned_child_coalition_2026-05-15_carrier_d2_strict88\learned_gated_manychild_runtime_sandbox.json`

Compare:

- `D:\RAFA\outputs\circleworld_proto\target_conditioned_child_coalition_compare_2026-05-15\target_conditioned_child_coalition_compare.json`
- `D:\RAFA\outputs\circleworld_proto\target_conditioned_child_coalition_compare_2026-05-15\TARGET_CONDITIONED_CHILD_COALITION_COMPARE.md`

## Key Results

### Depth-2 Strict88

Plain directional coalition:

- parent lift vs blocked: `-0.0035328819914917514`
- final world jump: `0.0`
- writeback mass: `0.02032653180261453`

Target-conditioned coalition:

- target projection: `0.8007062077522278`
- collapse component abs: `0.4978642937541008`
- writeback mass: `0.026376938447356224`
- parent lift vs blocked: `-0.016380633115768433`
- final world jump: `0.0`

Interpretation: target conditioning increased projection and writeback, but worsened parent lift. The tangent target is still selecting a stabilizing/canceling direction.

### Depth-3 Soft70

Plain directional coalition:

- parent lift vs blocked: `-0.016763039558578524`
- final world jump: `0.0`
- writeback mass: `0.03890549826125304`

Target-conditioned coalition:

- target projection: `0.7894822359085083`
- collapse component abs: `0.4467148880163829`
- writeback mass: `0.051365673542022705`
- parent lift vs blocked: `-0.020164720714092255`
- final world jump: `0.0`

### Depth-3 Strict88

Target-conditioned coalition:

- target projection: `0.7868983944257101`
- writeback mass: `0.02709333971142769`
- parent lift vs blocked: `-0.01823532829713076`
- final world jump: `0.0`

## Interpretation

This is a decisive negative for **endogenous** target conditioning.

The target projection is high, so the math is not simply failing to find target-aligned children. Instead, the target itself is derived from the same stabilizing child-return field. Removing direct collapse-to-mode0 is insufficient because the remaining tangent direction still cancels or stabilizes parent evolution.

Current ladder:

1. independent child permission: too jumpy;
2. scalar allocation: safe but stabilizing;
3. developmental scalar allocation: safe, slightly more writeback, still stabilizing;
4. mutual directional coalition: coherent but stabilizing;
5. endogenous target-conditioned coalition: target-aligned but still stabilizing.

So the next target must be **exogenous or contrastive**.

## Next Step

Build a contrastive sibling target assay:

- generate blocked runtime trajectory;
- generate a controlled sibling trajectory with a known acceptable perturbation or readout-preserving mode shift;
- define target direction as blocked-to-sibling delta;
- allocate child coalitions by projection onto that external sibling target;
- report target projection, child writeback, parent lift, and world jump.

Working label:

`contrastive_sibling_target_allocator_v0`
