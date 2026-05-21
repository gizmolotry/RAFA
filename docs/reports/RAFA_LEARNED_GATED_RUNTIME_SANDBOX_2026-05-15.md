# RAFA Learned-Gated Runtime Sandbox - 2026-05-15

## Scope

This push moved the learned shadow branch-law gate from detached operator recurrence into an assay-only Circleworld runtime hook.

Runtime defaults remain stable:

- `child_assay_writeback_scale_enabled=False`
- `child_assay_writeback_scale_default=1.0`

When enabled, each child may carry `assay_writeback_scale`. Circleworld applies that scale to child phase writeback, support writeback, and writeback-budget consumption before the child return reaches parent mode 1 / mode 0.

## Code Changes

- Added assay-only child writeback scale fields in `D:\RAFA\lineages\04_positive_replacement\circleworld.py`.
- Added child writeback scale metrics:
  - `child_assay_writeback_scale_mean`
  - `child_assay_writeback_scale_min`
  - aggregate `mean_child_assay_writeback_scale`
  - aggregate `min_child_assay_writeback_scale`
- Added `D:\RAFA\runtimes\circleworld_proto\run_learned_gated_runtime_sandbox.py`.
- Extended `D:\RAFA\runtimes\circleworld_proto\test_learned_branch_law_child_ifs_contract.py` with a runtime suppression contract.

## Verification

Command:

```powershell
python -m pytest D:\RAFA\runtimes\circleworld_proto\test_learned_branch_law_child_ifs_contract.py D:\RAFA\runtimes\circleworld_proto\test_resonant_attention_contract.py -q
```

Result:

```text
23 passed
```

The new contract verifies that `assay_writeback_scale=0.0` suppresses runtime child writeback mass and phase writeback delta, while `assay_writeback_scale=1.0` produces nonzero return.

## Runtime Runs

Depth-2 carrier run:

- JSON: `D:\RAFA\outputs\circleworld_proto\learned_gated_runtime_sandbox_2026-05-15_carrier_d2_soft70_v2\learned_gated_runtime_sandbox.json`
- Markdown: `D:\RAFA\outputs\circleworld_proto\learned_gated_runtime_sandbox_2026-05-15_carrier_d2_soft70_v2\LEARNED_GATED_RUNTIME_SANDBOX.md`

Depth-3 authority/grandchild run:

- JSON: `D:\RAFA\outputs\circleworld_proto\learned_gated_runtime_sandbox_2026-05-15_authority_d3_soft70_v2\learned_gated_runtime_sandbox.json`
- Markdown: `D:\RAFA\outputs\circleworld_proto\learned_gated_runtime_sandbox_2026-05-15_authority_d3_soft70_v2\LEARNED_GATED_RUNTIME_SANDBOX.md`

Combined compare:

- JSON: `D:\RAFA\outputs\circleworld_proto\learned_gated_runtime_tradeoff_2026-05-15\learned_gated_runtime_tradeoff_compare.json`
- Markdown: `D:\RAFA\outputs\circleworld_proto\learned_gated_runtime_tradeoff_2026-05-15\LEARNED_GATED_RUNTIME_TRADEOFF_COMPARE.md`

## Key Results

### Depth-2 Carrier Condition

Blocked-zero baseline:

- parent divergence: `0.1202591536606848`
- world jump proxy: `0.0033604623867748242`
- child writeback mass: `0.0`

Ungated / hard / soft all matched because the learned gate permitted the depth-2 carrier return:

- scale: `1.0`
- parent divergence: `0.3326834181794982`
- world jump proxy: `0.046437528931260484`
- child writeback mass: `0.11391727502147357`
- parent lift vs blocked: `0.2124242645188134`
- jump lift vs blocked: `0.04307706654448566`

Interpretation: the learned branch law treats this as a safe carrier regime and allows full child return. That child return is causally active in the runtime, not decorative.

### Depth-3 Authority / Grandchild Condition

Blocked-zero and hard-gate matched because the learned gate denied the depth-3 authority return:

- hard scale: `0.0`
- hard parent divergence: `0.13050034368414246`
- hard world jump proxy: `0.003652911779429835`
- hard child writeback mass: `0.0`

Ungated return:

- scale: `1.0`
- parent divergence: `0.28664508402055844`
- world jump proxy: `0.031549850753845754`
- child writeback mass: `0.055931490225096546`
- parent lift vs blocked: `0.15614474033641598`
- jump lift vs blocked: `0.02789693897441592`

Soft-gated return:

- mean scale: `0.1884535465527466`
- parent divergence: `0.13883095780475352`
- world jump proxy: `0.0021340210796744565`
- child writeback mass: `0.010537743801251054`
- parent lift vs blocked: `0.008330614120611063`
- jump lift vs blocked: `-0.0015188906997553784`
- parent retained vs ungated: `0.48433050327420385`
- jump retained vs ungated: `0.06763965688219083`

Interpretation: in actual Circleworld recurrence, the soft gate preserves a small amount of depth-3 child return while greatly reducing jump relative to ungated. The negative jump lift vs blocked is possible because runtime evolution is nonlinear: attenuated child return can slightly stabilize the parent trajectory relative to the zero-return baseline.

## What We Learned

The learned branch-law path has crossed a meaningful boundary:

1. It is no longer only a table fit.
2. It is no longer only a detached operator toy.
3. It now controls actual child writeback inside `circleworld_step()` when explicitly enabled.

The learned law currently says:

- depth-2 child carriers are useful enough to permit fully;
- depth-3 authority/grandchild returns are risky enough to block hard;
- soft attenuation can preserve a small causal return from risky children without admitting the ungated jump.

This is still assay-only. It is not production branch-law promotion yet.

## Limitations

- Only the selected child is isolated per runtime case; this is controlled causality, not full many-child ecology.
- The learned target remains engineered from causality proxies, not trained from downstream audio or long-horizon ontology success.
- `writeback_event_count` includes zero-scaled writeback-ready events; use writeback mass and phase delta to judge actual return.
- Runtime metrics are over three naked_rafa seeds only: `9114`, `9115`, `9116`.

## Next Action

Run the same learned runtime gate in a many-child ecology instead of isolating one selected child. That will test whether the learned gate can arbitrate simultaneous carrier/authority/grandchild returns without central selection.
