# RAFA Learned-Gated Many-Child Runtime Sandbox - 2026-05-15

## Scope

This follow-up tests whether the selected-child learned writeback gate survives contact with a many-child ecology.

Instead of isolating one child, the new sandbox assigns an assay-only `assay_writeback_scale` to every live child before each real `circleworld_step()` call.

Variants:

- `blocked_zero`: all child returns scale `0.0`
- `ungated`: all child returns scale `1.0`
- `learned_hard`: per-child learned score threshold
- `learned_soft`: per-child learned soft attenuation

## Code

- `D:\RAFA\runtimes\circleworld_proto\run_learned_gated_manychild_runtime_sandbox.py`

The script reuses:

- the shadow learned branch-law checkpoint,
- the causal operator selector,
- law-object retrieval rows,
- per-child runtime writeback scale hook in `circleworld.py`.

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

Soft70 runs:

- `D:\RAFA\outputs\circleworld_proto\learned_gated_manychild_runtime_sandbox_2026-05-15_carrier_d2_soft70\learned_gated_manychild_runtime_sandbox.json`
- `D:\RAFA\outputs\circleworld_proto\learned_gated_manychild_runtime_sandbox_2026-05-15_authority_d3_soft70\learned_gated_manychild_runtime_sandbox.json`

Strict88 runs:

- `D:\RAFA\outputs\circleworld_proto\learned_gated_manychild_runtime_sandbox_2026-05-15_carrier_d2_strict88\learned_gated_manychild_runtime_sandbox.json`
- `D:\RAFA\outputs\circleworld_proto\learned_gated_manychild_runtime_sandbox_2026-05-15_authority_d3_strict88\learned_gated_manychild_runtime_sandbox.json`

Combined compare:

- `D:\RAFA\outputs\circleworld_proto\learned_gated_manychild_runtime_tradeoff_2026-05-15\learned_gated_manychild_runtime_tradeoff_compare.json`
- `D:\RAFA\outputs\circleworld_proto\learned_gated_manychild_runtime_tradeoff_2026-05-15\LEARNED_GATED_MANYCHILD_RUNTIME_TRADEOFF_COMPARE.md`

## Results

### Soft70 Many-Child

Depth-2:

- blocked world jump: `0.003676906728293461`
- ungated world jump: `0.27757941909531453`
- learned hard world jump: `0.05940194352219256`
- learned soft world jump: `0.05756455153041803`
- learned hard parent lift vs blocked: `0.15295290468539918`
- learned soft parent lift vs blocked: `0.08869501339076495`

Depth-3:

- blocked world jump: `0.003676906728293461`
- ungated world jump: `0.50685973596848`
- learned hard world jump: `0.12130260095697291`
- learned soft world jump: `0.3155106890115032`
- learned hard nonzero child fraction: `0.8125`
- learned soft mean scale: `0.8828476185020145`

Interpretation: soft70 is too permissive when every child is allowed to present itself as a candidate. The selected-child safety result does not automatically lift to an ecology.

### Strict88 Many-Child

Depth-2:

- blocked world jump: `0.003676906728293461`
- learned soft world jump: `0.0018601853963004695`
- learned soft child writeback mass: `0.08547393729289372`
- learned soft parent lift vs blocked: `-0.026189030122391543`
- learned hard world jump: `0.055442154985163916`

Depth-3:

- blocked world jump: `0.003676906728293461`
- learned soft world jump: `0.004862337754881096`
- learned soft child writeback mass: `0.09030687188108762`
- learned soft parent lift vs blocked: `-0.013576438969279264`
- learned hard world jump: `0.11248111844913644`

Interpretation: strict88 soft gating is the first useful many-child ecology setting. It keeps nonzero child writeback mass while holding world jump close to blocked baseline. But the negative parent lift means child writeback is acting more like a stabilizing/canceling field than a clear sibling-development carrier.

## What We Learned

The volume hypothesis is partially vindicated and partially falsified:

- More child worlds create much larger possible parent movement.
- More child worlds also create much larger jump risk.
- A learned selected-child gate is not sufficient for many-child ecology.
- Soft attenuation is better than hard thresholding in the ecology setting.
- The next law needs a global or local writeback budget, not independent per-child permission.

So the issue is not only training stability. It is authority allocation.

In RAFA terms: a child should not only ask, "am I lawful?" It must also ask, "am I the right law to spend parent authority on, given the other live laws?"

## Next Design Correction

Add an ecology-level writeback allocator:

- score each child locally,
- normalize child authority across siblings under support overlap,
- apply a global writeback budget per support region,
- penalize redundant same-family children,
- reward complementary low-jump children,
- report selected-child score separately from actual allocated authority.

That turns branch law from independent child permission into competitive resonant authority.
