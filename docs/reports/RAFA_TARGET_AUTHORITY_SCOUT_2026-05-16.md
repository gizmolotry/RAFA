# RAFA Target Authority Scout - 2026-05-16

## Scope

This pass tests whether target-aligned child carriers become parent-visible when the allocator grants them stronger assay-only writeback authority.

New allocator variant:

- `learned_soft_target_authority_scout` in `D:\RAFA\runtimes\circleworld_proto\run_learned_gated_manychild_runtime_sandbox.py`

Updated assay:

- `D:\RAFA\runtimes\circleworld_proto\run_child_basis_coverage_assay.py` now runs `blocked_zero`, conservative contrastive allocation, and target-authority allocation for natural and target-augmented child banks.

## Verification

```powershell
python -m py_compile D:\RAFA\runtimes\circleworld_proto\run_learned_gated_manychild_runtime_sandbox.py D:\RAFA\runtimes\circleworld_proto\run_child_basis_coverage_assay.py
python -m pytest D:\RAFA\runtimes\circleworld_proto\test_learned_branch_law_child_ifs_contract.py D:\RAFA\runtimes\circleworld_proto\test_resonant_attention_contract.py -q
```

Result: `23 passed`.

## Outputs

- `D:\RAFA\outputs\circleworld_proto\target_authority_scout_2026-05-16_carrier_d2_strict88_v2\child_basis_coverage_assay.json`
- `D:\RAFA\outputs\circleworld_proto\target_authority_scout_2026-05-16_authority_d3_soft70_v2\child_basis_coverage_assay.json`
- `D:\RAFA\outputs\circleworld_proto\target_authority_scout_2026-05-16_authority_d3_strict88_v2\child_basis_coverage_assay.json`
- `D:\RAFA\outputs\circleworld_proto\target_authority_scout_compare_2026-05-16\target_authority_scout_compare.json`
- `D:\RAFA\outputs\circleworld_proto\target_authority_scout_compare_2026-05-16\TARGET_AUTHORITY_SCOUT_COMPARE.md`

## Runtime Results

| substrate | phase | variant | scale | ctarget | writeback | parent div | world jump | parent lift | jump lift |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| carrier_d2_strict88 | natural | blocked_zero | 0.0 | 0.0 | 0.0 | 0.12430946859232882 | 0.003676906728293461 | 0.0 | 0.0 |
| carrier_d2_strict88 | natural | learned_soft_contrastive_sibling_target | 0.006362131894470803 | 0.005702414401722225 | 0.0009855892179378618 | 0.1238938935006439 | 0.0031114215035187667 | -0.000415575091684911 | -0.0005654852247746942 |
| carrier_d2_strict88 | natural | learned_soft_target_authority_scout | 0.03383623572832778 | 0.0039050608771062935 | 0.00310866444488056 | 0.13774554107929074 | 0.0039609369070308835 | 0.013436072486961922 | 0.00028403017873742265 |
| carrier_d2_strict88 | target_augmented | blocked_zero | 0.0 | 0.0 | 0.0 | 0.12430946859232882 | 0.003676906728293461 | 0.0 | 0.0 |
| carrier_d2_strict88 | target_augmented | learned_soft_contrastive_sibling_target | 0.010728102392212868 | 0.08942003125378724 | 0.0009871196137585987 | 0.12374418781221629 | 0.0030506970901408303 | -0.0005652807801125315 | -0.0006262096381526305 |
| carrier_d2_strict88 | target_augmented | learned_soft_target_authority_scout | 0.07137104943332952 | 0.08756211993571235 | 0.003271448687883094 | 0.14297376952404556 | 0.004528997264523864 | 0.01866430093171674 | 0.0008520905362304028 |
| authority_d3_soft70 | natural | blocked_zero | 0.0 | 0.0 | 0.0 | 0.12430946859232882 | 0.003676906728293461 | 0.0 | 0.0 |
| authority_d3_soft70 | natural | learned_soft_contrastive_sibling_target | 0.01093630290301644 | 0.00396335497073504 | 0.001806910508700336 | 0.12430279586667496 | 0.002702149397135608 | -6.672725653855682e-06 | -0.000974757331157853 |
| authority_d3_soft70 | natural | learned_soft_target_authority_scout | 0.04539011832806191 | 0.001424297671912918 | 0.004838730422003816 | 0.16055153409519266 | 0.006970779864641785 | 0.03624206550286384 | 0.0032938731363483242 |
| authority_d3_soft70 | target_augmented | blocked_zero | 0.0 | 0.0 | 0.0 | 0.12430946859232882 | 0.003676906728293461 | 0.0 | 0.0 |
| authority_d3_soft70 | target_augmented | learned_soft_contrastive_sibling_target | 0.020621060002164345 | 0.07413433262842288 | 0.001806910508700336 | 0.12430279586667496 | 0.002702149397135608 | -6.672725653855682e-06 | -0.000974757331157853 |
| authority_d3_soft70 | target_augmented | learned_soft_target_authority_scout | 0.08188121833010416 | 0.07159527532960075 | 0.004838730422003816 | 0.16055153409519266 | 0.006970779864641785 | 0.03624206550286384 | 0.0032938731363483242 |
| authority_d3_strict88 | natural | blocked_zero | 0.0 | 0.0 | 0.0 | 0.12430946859232882 | 0.003676906728293461 | 0.0 | 0.0 |
| authority_d3_strict88 | natural | learned_soft_contrastive_sibling_target | 0.006160615183303048 | 0.004569303280016083 | 0.000962287721146519 | 0.12368012703601823 | 0.0030543509946701817 | -0.0006293415563105864 | -0.0006225557336232792 |
| authority_d3_strict88 | natural | learned_soft_target_authority_scout | 0.029451544646812683 | 0.0026786760678367007 | 0.0032743684423621744 | 0.14292779992515547 | 0.004518522738205262 | 0.018618331332826657 | 0.0008416160099118013 |
| authority_d3_strict88 | target_augmented | blocked_zero | 0.0 | 0.0 | 0.0 | 0.12430946859232882 | 0.003676906728293461 | 0.0 | 0.0 |
| authority_d3_strict88 | target_augmented | learned_soft_contrastive_sibling_target | 0.00997441724915917 | 0.0747402809377039 | 0.000962287721146519 | 0.12368012703601823 | 0.0030543509946701817 | -0.0006293415563105864 | -0.0006225557336232792 |
| authority_d3_strict88 | target_augmented | learned_soft_target_authority_scout | 0.06227376850408337 | 0.07284965372552453 | 0.0032743684423621744 | 0.14292779992515547 | 0.004518522738205262 | 0.018618331332826657 | 0.0008416160099118013 |

## Interpretation

Target authority makes child writeback parent-visible:

- depth-2 strict88 natural parent lift rises to `0.0134`, with jump lift `0.00028`.
- depth-2 strict88 target-augmented parent lift rises to `0.0187`, with jump lift `0.00085`.
- depth-3 soft70 parent lift rises to `0.0362`, but jump lift rises to `0.00329`.
- depth-3 strict88 parent lift rises to `0.0186`, with jump lift `0.00084`.

This confirms that the previous inertness was partly authority starvation. But it is not a clean sibling result because stronger authority also increases world-jump. The target-augmented cases improve target projection and scale, but depth-3 writeback mass and final parent movement are unchanged relative to natural authority, which points to a writeback saturation or support-path bottleneck.

Updated diagnosis:

- Natural child banks have sparse sibling-target coverage.
- Synthetic target augmentation can insert high-projection child carriers.
- Stronger authority can make child return parent-visible.
- Current authority is not selective enough to preserve coarse world identity at higher movement.
- The next objective must jointly optimize target projection, parent movement, and low world-jump, not any one of those alone.

## Decision

Do not promote this allocator. It is an assay result, not a runtime policy. It proves the path can move the parent, but it does not yet prove sibling ontology.

## Next Step

Build a low-jump target-authority trainer/table: label child candidates positive only when they have high target projection, nonzero parent movement, and bounded world-jump. Then retrain the learned branch-law writeback head on those examples instead of hand-authoring authority floors.
