# Graduation RAFA Parallel Instructions

## Mission
Continue refining the Graduation RAFA restore/runtime while the Circleworld prototype is being trained in parallel.

Your job is to improve the Graduation family without contaminating the Circleworld lane or re-breaking the restore baseline.

## Current Ground Truth

### Graduation runtime family
- Runtime: `stage4_blackwell_14`
- Canonical runtime entrypoint: [D:\RAFA\runtimes\stage4_blackwell_14\export_graduation.py](D:\RAFA\runtimes\stage4_blackwell_14\export_graduation.py)
- Compatibility implementation: [D:\RAFA\tools\export_audio_compat14.py](D:\RAFA\tools\export_audio_compat14.py)
- Current restore wrapper: [D:\RAFA\tools\export_graduation_pack_restore.py](D:\RAFA\tools\export_graduation_pack_restore.py)
- Findings note: [D:\RAFA\docs\reports\GRADPACK_RESTORE_FINDINGS_2026-04-05.md](D:\RAFA\docs\reports\GRADPACK_RESTORE_FINDINGS_2026-04-05.md)

### Current best-known restore baseline
- Checkpoint: `D:\RAFA\checkpoints_stage4\bound_weights_ep10_step1200.pt`
- Source: `qkv`
- `phase_mix = 0.35`
- `mag_mix = 0.05`
- `seed = 101`

Baseline outputs:
- [D:\RAFA\outputs\graduation_pack_restore_candidate\grad_engine.wav](D:\RAFA\outputs\graduation_pack_restore_candidate\grad_engine.wav)
- [D:\RAFA\outputs\graduation_pack_restore_candidate\grad_voice.wav](D:\RAFA\outputs\graduation_pack_restore_candidate\grad_voice.wav)
- [D:\RAFA\outputs\graduation_pack_restore_candidate\grad_impact.wav](D:\RAFA\outputs\graduation_pack_restore_candidate\grad_impact.wav)
- [D:\RAFA\outputs\graduation_pack_restore_candidate\grad_drone.wav](D:\RAFA\outputs\graduation_pack_restore_candidate\grad_drone.wav)

Baseline comparison:
- [D:\RAFA\outputs\graduation_pack_restore_candidate\comparison_summary.json](D:\RAFA\outputs\graduation_pack_restore_candidate\comparison_summary.json)
- `mean_mse = 0.009564753388985991`
- `mean_corr = 0.8047516061188855`

## Hard Boundaries
Do not modify any of the following:
- `D:\RAFA\lineages\04_positive_replacement\*`
- `D:\RAFA\runtimes\circleworld_proto\*`
- `D:\RAFA\outputs\circleworld_proto\*`
- `D:\RAFA\checkpoints_circleworld_proto\*`

Do not “fix” Circleworld shared behavior by editing Graduation runtime files.

## Shared-Module Rule
If you think you need to change a shared module, pause and prefer one of these options first:
1. Add a Graduation-specific adapter under `D:\RAFA\runtimes\stage4_blackwell_14\`
2. Add a Graduation-specific helper under `D:\RAFA\tools\`
3. Copy a tiny compatibility shim locally to the Graduation runtime instead of editing a shared primitive

Only touch shared modules if the change is clearly backward compatible for:
- `stage4_blackwell_14`
- `stage4_blackwell_16`
- `diffusion_parent_v3`
- `circleworld_proto`

## Expected Working Surface
Use the registries before changing anything:
- [D:\RAFA\registries\RUNTIME_REGISTRY.json](D:\RAFA\registries\RUNTIME_REGISTRY.json)
- [D:\RAFA\registries\CHECKPOINT_REGISTRY.json](D:\RAFA\registries\CHECKPOINT_REGISTRY.json)
- [D:\RAFA\docs\architecture\RUNTIME_CONTRACTS.md](D:\RAFA\docs\architecture\RUNTIME_CONTRACTS.md)

Treat the Graduation problem as a runtime/checkpoint contract problem, not a generic audio problem.

## What To Optimize
Priority order:
1. Recover the exact pre-`g1_*` forward for the `bw14` family
2. Improve similarity to the original Graduation pack without regressing determinism
3. Keep the old-family restore path explicit and isolated

## What Not To Do
- Do not use `sample_diffusion.py` as the restore lane
- Do not patch the live root path just to make a checkpoint run
- Do not claim success from metadata-only matches like duration, mono, or file count
- Do not overwrite the canonical baseline outputs in `graduation_pack_restore_candidate`

## Required Reporting
For every meaningful attempt, save:
1. output WAVs to a new subfolder under `D:\RAFA\outputs\`
2. a comparison JSON against `D:\RAFA\outputs\graduation_pack`
3. a short note with:
   - runtime family
   - checkpoint
   - entrypoint
   - exact parameter deltas from the current baseline
   - whether `mean_corr` improved or regressed

## Success Gate
Only call a change an improvement if it beats the current restore baseline on actual comparison metrics or obvious perceptual review, without losing determinism.

## Relationship To Circleworld
Circleworld is currently a separate prototype lane. Assume:
- it is not production generation
- it is not the place to hide Graduation fixes
- it is allowed to evolve independently

Your job is to keep Graduation RAFA clean, measurable, and reproducible while Circleworld explores a different internal ontology.
