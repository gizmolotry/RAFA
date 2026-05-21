# RAFA Learned-Gated Multi-Step Operator Sandbox - 2026-05-15

## Summary

This pass moved learned branch-law gating from one-shot operator scaling into a multi-step operator recurrence sandbox.

This is still not production `circleworld_step()` and still not a full nested assay. But it is stronger than the previous scalar proxy because the selected child operator is repeatedly applied to a regenerated branch state, and parent mode 1 naturally saturates as it approaches the child phase.

The key result: soft floor `0.70`, full `0.88` preserves almost all safe depth-2 influence and preserves a small amount of depth-3 influence while eliminating final world-jump in this recurrence model.

## Code Changes

- Added `run_learned_gated_multistep_operator_sandbox.py`.
- Added a contract test for scaled recurrent operator application.
- No Circleworld runtime behavior changed.

## Runs

Depth-2 safe regime:

- `D:\RAFA\outputs\circleworld_proto\learned_gated_multistep_operator_sandbox_2026-05-15_carrier_d2_soft70\learned_gated_multistep_operator_sandbox.json`

Depth-3 unsafe regime:

- `D:\RAFA\outputs\circleworld_proto\learned_gated_multistep_operator_sandbox_2026-05-15_authority_d3_soft70\learned_gated_multistep_operator_sandbox.json`

Comparison artifact:

- `D:\RAFA\outputs\circleworld_proto\learned_gated_multistep_operator_tradeoff_2026-05-15\multistep_operator_tradeoff_compare.json`

Common settings:

- recurrence steps: `4`
- soft floor: `0.70`
- soft full: `0.88`
- writeback threshold: `0.78`
- seeds: `9114,9115,9116`

## Depth-2 Safe Regime

| variant | scale | final parent divergence | final world-jump | parent retained | jump retained |
|---|---:|---:|---:|---:|---:|
| ungated | `1.0` | `0.3762523516055791` | `0.0841428343700406` | `1.0` | `1.0` |
| hard gate | `1.0` | `0.3762523516055791` | `0.0841428343700406` | `1.0` | `1.0` |
| soft gate | `0.9848221915739556` | `0.3741925906414239` | `0.08320105988159014` | `0.9945255864704483` | `0.98880743089413` |

Interpretation: safe depth-2 returns are preserved. Soft gating slightly attenuates them, but keeps nearly all parent influence.

## Depth-3 Unsafe Regime

| variant | scale | final parent divergence | final world-jump | parent retained | jump retained |
|---|---:|---:|---:|---:|---:|
| ungated | `1.0` | `0.43390669933830156` | `0.11008824432558217` | `1.0` | `1.0` |
| hard gate | `0.0` | `1.6031930388306319e-10` | `0.0` | `3.6947874768365344e-10` | `0.0` |
| soft gate | `0.0317222169703911` | `0.029330696161945936` | `0.0` | `0.06759678107453658` | `0.0` |

Interpretation: soft gate retains about `6.76%` of parent movement from an otherwise denied depth-3 child return while still ending at zero final world-jump in this recurrence model. Cumulative jump is not exactly zero (`0.000009613780444987322`), so the result should be read as tiny transient jump, zero final jump.

## What We Learned

1. Soft floor `0.70` remains the best immediate candidate.
It is almost transparent on safe depth-2 returns and highly suppressive on unsafe depth-3 returns.

2. Recurrent application changes the story slightly.
One-step soft gating showed residual jump. In multi-step operator recurrence, the depth-3 soft-gated path preserves a small phase movement while final parent/world alignment no longer records a jump penalty.

3. This is still operator recurrence, not full runtime branch law.
The sandbox repeatedly applies selected child operator pressure to parent mode 1. It does not yet let the full Circleworld runtime spawn, kill, merge, and re-score children after each learned-gated writeback.

## Claim Status

Supported:

- Learned soft gating can retain small unsafe-child influence while avoiding final world-jump in multi-step operator recurrence.
- Safe-child influence remains nearly intact.
- Soft gate is now more than a binary safety filter in this sandbox.

Not yet supported:

- Full `circleworld_step()` runtime improvement.
- Nested sibling emergence.
- Audio improvement.
- Broad config generalization.

## Verification

- `python -m pytest -q D:\RAFA\runtimes\circleworld_proto\test_learned_branch_law_child_ifs_contract.py` -> `9 passed`
- `python -m py_compile D:\RAFA\runtimes\circleworld_proto\run_learned_gated_multistep_operator_sandbox.py` -> pass

## Next Move

The next step is a true runtime sandbox hook, still assay-only:

- before `_evolve_child_worlds` applies child writeback, multiply child writeback budget or child writeback gain by learned soft score
- run full `circleworld_step()` for several steps
- compare ungated, hard-gated, and soft-gated runs on child survival, writeback mass, parent divergence, world-jump proxy, and nested labels

That is the bridge from operator recurrence to actual learned branch-law runtime control.
