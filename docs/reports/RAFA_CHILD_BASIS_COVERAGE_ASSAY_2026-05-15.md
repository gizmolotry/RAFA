# RAFA Child Basis Coverage Assay - 2026-05-15

## Scope

This pass moves upstream of contrastive allocation. It asks whether live child worlds span controlled sibling-development directions at all, and whether synthetic target-aligned children can make the contrastive runtime route less inert.

New utility: `D:\RAFA\runtimes\circleworld_proto\run_child_basis_coverage_assay.py`.

## Verification

```powershell
python -m py_compile D:\RAFA\runtimes\circleworld_proto\run_child_basis_coverage_assay.py D:\RAFA\runtimes\circleworld_proto\run_learned_gated_manychild_runtime_sandbox.py
python -m pytest D:\RAFA\runtimes\circleworld_proto\test_learned_branch_law_child_ifs_contract.py D:\RAFA\runtimes\circleworld_proto\test_resonant_attention_contract.py -q
```

Result: `23 passed`.

## Outputs

- `D:\RAFA\outputs\circleworld_proto\child_basis_coverage_2026-05-15_carrier_d2_strict88_v2\child_basis_coverage_assay.json`
- `D:\RAFA\outputs\circleworld_proto\child_basis_coverage_2026-05-15_authority_d3_soft70_v2\child_basis_coverage_assay.json`
- `D:\RAFA\outputs\circleworld_proto\child_basis_coverage_2026-05-15_authority_d3_strict88_v2\child_basis_coverage_assay.json`
- `D:\RAFA\outputs\circleworld_proto\child_basis_coverage_compare_2026-05-15\child_basis_coverage_compare.json`
- `D:\RAFA\outputs\circleworld_proto\child_basis_coverage_compare_2026-05-15\CHILD_BASIS_COVERAGE_COMPARE.md`

## Coverage Summary

| substrate | phase | children | rank | eff rank | pair align | target0 max | target0 top3 | target0 frac@.05 | runtime ctarget | runtime scale | runtime writeback | runtime jump |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| carrier_d2_strict88 | natural | 11.0 | 8.0 | 4.415234088897705 | 0.4595559239387512 | 0.11202688018480937 | 0.054911699559953474 | 0.0909090909090909 | 0.005702414401722225 | 0.006362131894470803 | 0.0009855892179378618 | 0.0031114215035187667 |
| carrier_d2_strict88 | target_augmented | 19.0 | 15.0 | 6.453569253285726 | 0.28390217820803326 | 0.9974270065625509 | 0.9313853051927355 | 0.47368421052631576 | 0.08942003125378724 | 0.010728102392212868 | 0.0009871196137585987 | 0.0030506970901408303 |
| authority_d3_soft70 | natural | 15.0 | 10.0 | 6.191741466522217 | 0.3498610357443492 | 0.11202688018480937 | 0.06232208924161064 | 0.06666666666666667 | 0.00396335497073504 | 0.01093630290301644 | 0.001806910508700336 | 0.002702149397135608 |
| authority_d3_soft70 | target_augmented | 23.0 | 17.0 | 8.206918398539225 | 0.23868036270141602 | 0.9974270065625509 | 0.9313853051927355 | 0.391304347826087 | 0.07413433262842288 | 0.020621060002164345 | 0.001806910508700336 | 0.002702149397135608 |
| authority_d3_strict88 | natural | 15.0 | 10.0 | 6.191741466522217 | 0.3498610357443492 | 0.11202688018480937 | 0.06232208924161064 | 0.06666666666666667 | 0.004569303280016083 | 0.006160615183303048 | 0.000962287721146519 | 0.0030543509946701817 |
| authority_d3_strict88 | target_augmented | 23.0 | 17.0 | 8.206918398539225 | 0.23868036270141602 | 0.9974270065625509 | 0.9313853051927355 | 0.391304347826087 | 0.0747402809377039 | 0.00997441724915917 | 0.000962287721146519 | 0.0030543509946701817 |

## Interpretation

Natural child banks are not totally empty. The allocator target has a best natural child around `0.112` projection, but only `~6.7-9.1%` of children clear `.05`, so coverage is sparse and fragile.

Target augmentation is a strong sanity check: target0 max projection jumps to about `0.997`, target0 top-3 projection to about `0.931`, basis effective rank rises, and pairwise child alignment drops. So the lattice can hold target-aligned child carriers when we explicitly create them.

But runtime writeback barely moves. The contrastive target projection reported by the allocator rises from `~0.004-0.006` to `~0.074-0.089`, while child writeback mass remains around `0.001-0.0018`. That means basis coverage alone is insufficient under the current learned gate and conservative allocator.

Updated diagnosis:

- The previous claim "children do not span the sibling direction" was too strong.
- The sharper claim is: natural children provide sparse target coverage, and target-aligned children do not yet receive enough runtime authority to become parent-visible sibling ontology.
- Therefore the next training objective must jointly optimize target-direction coverage and writeback authority under world-jump constraints.

## Next Step

Add a target-authority scout: supervised child spawning should generate children near contrastive sibling targets, and learned branch-law training should treat target projection plus low world-jump as a positive writeback-authority signal.
