# RAFA Learned Soft-Gated Writeback Sandbox Pass - 2026-05-15

## Summary

This pass extended the learned-gated writeback sandbox from binary hard gating to continuous soft attenuation.

The soft gate does not change Circleworld runtime. It scales measured child writeback by a learned writeback score window:

- below `soft_floor`: no writeback
- above `soft_full`: full writeback
- between them: linear attenuation

This gives us a tunable bridge between safety filtering and learned branch-law control.

## Code Changes

- Extended `run_learned_gated_writeback_sandbox.py` with `--gate-mode hard|soft`.
- Added `--soft-floor` and `--soft-full` controls.
- Added `writeback_scale` metrics.
- Added contract coverage for soft attenuation.

## Runs

Hard baseline, schema v2:

- `D:\RAFA\outputs\circleworld_proto\learned_hard_gated_writeback_sandbox_2026-05-15_fresh_d2_d3_schema_v2\learned_gated_writeback_sandbox.json`

Soft floor `0.60`, full `0.88`:

- `D:\RAFA\outputs\circleworld_proto\learned_soft_gated_writeback_sandbox_2026-05-15_fresh_d2_d3_floor60\learned_gated_writeback_sandbox.json`

Soft floor `0.65`, full `0.88`:

- `D:\RAFA\outputs\circleworld_proto\learned_soft_gated_writeback_sandbox_2026-05-15_fresh_d2_d3\learned_gated_writeback_sandbox.json`

Soft floor `0.70`, full `0.88`:

- `D:\RAFA\outputs\circleworld_proto\learned_soft_gated_writeback_sandbox_2026-05-15_fresh_d2_d3_floor70\learned_gated_writeback_sandbox.json`

Comparison artifact:

- `D:\RAFA\outputs\circleworld_proto\learned_soft_gate_tradeoff_2026-05-15\soft_gate_tradeoff_compare.json`

## Tradeoff Table

| gate | floor | mean scale | parent retained | jump retained | jump reduction | influence per jump retained |
|---|---:|---:|---:|---:|---:|---:|
| hard | `0.65` | `0.5` | `0.43394086341901467` | `0.3544732920447186` | `0.015354928756159495` | `1.224184933414591` |
| soft | `0.60` | `0.6838892741749684` | `0.6434143704164841` | `0.594724023098211` | `0.009640164651934131` | `1.0818704902227103` |
| soft | `0.65` | `0.6151695511695269` | `0.5658957552896325` | `0.5066205498586916` | `0.0117358526197459` | `1.1170011864845872` |
| soft | `0.70` | `0.5082722042721733` | `0.44531124287008605` | `0.36957070259721725` | `0.014995811680786425` | `1.2049419495122045` |

## Interpretation

Hard gating is still the safest setting. It gives the strongest jump suppression, but it completely removes denied child influence.

Soft floor `0.60` keeps much more parent influence, but lets more jump through.

Soft floor `0.65` is a middle tradeoff.

Soft floor `0.70` is the most interesting immediate setting: it stays close to hard-gate safety while allowing a small amount of denied-child influence. On this slice, it retains `0.4453` of parent divergence versus hard gate's `0.4339`, while retaining `0.3696` of world-jump versus hard gate's `0.3545`.

The improvement is small, but conceptually important: this is the first learned branch-law behavior that is not simply full allow/full deny.

## Claim Status

Supported:

- Soft learned gating creates a smooth safety-control tradeoff.
- The soft floor parameter controls parent influence retention versus jump suppression.
- Floor `0.70`, full `0.88` is near-hard but nonzero-soft, making it the best next sandbox setting.

Not yet supported:

- Multi-step runtime improvement.
- Nested sibling emergence.
- Audio continuity improvement.
- Broad config generalization.

## Verification

- `python -m pytest -q D:\RAFA\runtimes\circleworld_proto\test_learned_branch_law_child_ifs_contract.py` -> `8 passed`
- `python -m py_compile D:\RAFA\runtimes\circleworld_proto\run_learned_gated_writeback_sandbox.py` -> pass

## Next Move

Use soft floor `0.70`, full `0.88` inside a multi-step sandbox run:

- apply learned attenuation to child writeback over several recursive steps
- compare against hard gate and ungated child-local IFS
- report parent divergence, world-jump proxy, child survival, and nested branch labels

That is the first real bridge from operator-level gating toward runtime branch-law control.
