# RAFA Learned-Gated Writeback Sandbox Pass - 2026-05-15

## Summary

This pass tested the learned shadow branch-law checkpoint as a reversible gate over measured child writeback outcomes.

The sandbox is intentionally conservative: it does not modify `circleworld_step()` and does not promote the learned law into runtime. It asks one narrow question:

If the learned branch-law checkpoint decides whether a child return path is allowed, does it preserve safe parent movement and suppress jumpy unsafe returns?

On the fresh depth-2/depth-3 calibration slice, yes.

## Code Changes

- Added `run_learned_gated_writeback_sandbox.py`.
- Added a contract test proving denied writeback suppresses parent divergence/world-jump in the sandbox row model.
- No Circleworld runtime behavior changed.

## Inputs

Checkpoint:

- `D:\RAFA\outputs\circleworld_proto\shadow_learned_branch_law_2026-05-15_expanded_d2_d3\shadow_rafa_learned_branch_law_v0.pt`

Fresh causality assays:

- positive depth-2: `D:\RAFA\outputs\circleworld_proto\resonant_operator_causality_2026-05-15_calib_carrier_9114_9116_d2\resonant_operator_causality_assay.json`
- negative depth-3: `D:\RAFA\outputs\circleworld_proto\resonant_operator_causality_2026-05-15_calib_authority_9114_9116_d3\resonant_operator_causality_assay.json`

Output:

- `D:\RAFA\outputs\circleworld_proto\learned_gated_writeback_sandbox_2026-05-15_fresh_d2_d3\learned_gated_writeback_sandbox.json`

## Aggregate Result

- rows: `6`
- sources: `2`
- permit fraction: `0.5`
- target permit fraction: `0.5`
- writeback accuracy: `1.0`
- false permit rate: `0.0`
- false deny rate: `0.0`
- ungated parent divergence: `0.20925999567371967`
- gated parent divergence: `0.09080646320171319`
- parent divergence retention vs ungated: `0.43394086341901467`
- ungated world jump: `0.023786666867427586`
- gated world jump: `0.008431738111268091`
- world-jump retention vs ungated: `0.3544732920447186`
- world-jump reduction vs ungated: `0.015354928756159495`

## Per-Regime Behavior

### Depth-2 safe-writeback regime

- permit fraction: `1.0`
- writeback accuracy: `1.0`
- parent divergence retained: `1.0`
- world-jump retained: `1.0`
- ungated parent divergence: `0.18161292640342638`
- gated parent divergence: `0.18161292640342638`
- ungated world jump: `0.016863476222536183`
- gated world jump: `0.016863476222536183`
- mean writeback score: `0.877267994483312`

Interpretation: the learned gate preserves safe depth-2 child returns.

### Depth-3 unsafe-writeback regime

- permit fraction: `0.0`
- writeback accuracy: `1.0`
- parent divergence retained: `0.0`
- world-jump retained: `0.0`
- ungated parent divergence: `0.236907064944013`
- gated parent divergence: `0.0`
- ungated world jump: `0.03070985751231899`
- gated world jump: `0.0`
- mean writeback score: `0.7057099990546704`

Interpretation: the learned gate suppresses jumpy depth-3 child returns entirely.

## What This Means

This is the first reversible intervention where learned branch-law outputs directly change whether child writeback is allowed in an assay.

It supports a narrow claim:

The learned shadow branch law can act as a safety gate over child returns, preserving measured safe writeback and suppressing measured unsafe writeback on fresh seeds within nearby ontology configs.

It does not yet support:

- nuanced partial writeback modulation
- multi-step runtime improvement
- nested sibling emergence
- audio improvement
- distant config generalization

The current gate is binary. It prevents bad returns, but when it denies writeback it also discards all parent influence from that child. The next step should test soft gating, where writeback is attenuated rather than zeroed.

## Verification

- `python -m pytest -q D:\RAFA\runtimes\circleworld_proto\test_learned_branch_law_child_ifs_contract.py` -> `7 passed`
- `python -m py_compile D:\RAFA\runtimes\circleworld_proto\run_learned_gated_writeback_sandbox.py` -> pass

## Next Move

Add a soft learned-gate variant:

- hard gate: `permit -> full writeback`, `deny -> zero writeback`
- soft gate: scale writeback by normalized learned survival/writeback score
- compare parent divergence retained versus world-jump reduction

That is the path from safety filter toward actual learned branch-law control.
