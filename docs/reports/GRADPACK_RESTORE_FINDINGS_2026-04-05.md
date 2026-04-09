# Graduation Pack Restore Findings

Date: 2026-04-05

## Summary

The attempted "Graduation Pack regeneration" through `sample_diffusion.py` and `checkpoints_diffusion_rafa_full/diff_step1000.pt` was chasing the wrong path.

The original `outputs/graduation_pack` artifacts are not produced by that diffusion sampler stack.

## What Is Now Confirmed

1. The current expanded pack is invalid as a graduation-pack restoration.
   - Files under `outputs/expanded_graduation_pack` are 1.0 second mono WAVs.
   - For a fixed seed, different prompts hash identically.
   - Prompt control is dead on that path.

2. The original graduation files are different in a fundamental way.
   - Files under `outputs/graduation_pack` are 2.0 second mono WAVs.
   - The four original files are distinct by hash.

3. The original graduation pack was produced by the stage-4 Blackwell path, not the diffusion sampler.
   - Source script: `tools/export_audio.py`
   - Runtime family: `NakedRAFA` + `NakedDenoiser`
   - Checkpoint family: `checkpoints_stage4/bound_weights_*.pt`

4. The repo's own lineage ledger identifies the graduation anchor checkpoint as:
   - `checkpoints_stage4/bound_weights_ep49_step100.pt`

5. The currently hardcoded checkpoint in `tools/export_audio.py` is not that anchor.
   - It points at `bound_weights_ep99_step300.pt`

6. The stage-4 runtime contract changed exactly at epoch 50.
   - `bound_weights_ep49_step100.pt`
     - 14-key core
     - no `g1_w_ph`
     - no `g1_w_mag`
   - `bound_weights_ep50_step0.pt`
     - 16-key core
     - adds `g1_w_ph`
     - adds `g1_w_mag`

This means the repo crossed from the old graduation runtime family into the newer projection-head family at the `ep49 -> ep50` seam.

## Key Evidence

### A. Wrong pipeline

The restored diffusion path used:
- `sample_diffusion.py`
- `model.py`
- `diffusion_models.py`
- `phase_native_ifs.py`
- `checkpoints_diffusion_rafa_full/diff_step1000.pt`

That path produces 1-second outputs and dead prompt conditioning. It does not match the original graduation artifacts.

### B. Right pipeline

`tools/export_audio.py` renders:
- dataset-conditioned stage-4 examples
- via `NakedRAFA` / `NakedDenoiser`
- to `outputs/graduation_pack`

This aligns with the original 2-second artifact family.

### C. Checkpoint/runtime schema split

The stage-4 checkpoint schemas split into two families:

- `bound_weights_ep49_step100.pt`
  - 14-key core
  - missing `g1_w_ph`
  - missing `g1_w_mag`

- `bound_weights_ep99_step200.pt`
  - 16-key core
  - includes `g1_w_ph`
  - includes `g1_w_mag`

Current `core/lib_blackwell.py` expects the 16-key family.

That is why trying to render the graduation suite with the anchor checkpoint currently fails:
- `KeyError: 'g1_w_ph'`

## What Was Tested

### 1. Clean legacy diffusion lane

An isolated restore lane was built under:
- `tmp/legacy_restore_gradpack`

It used:
- `model.py`, `diffusion_models.py`, `phase_native_ifs.py`,
  `rafa_clutch_transformer_upgraded.py`, `rafa_math_tools.py`
  restored from commit `e2d36ec`
- sampler copied from `lineages/01_parent_graduation/sample_diffusion.py`

Outcome:
- it runs
- it produces different audio from the patched expanded pack
- but prompts are still dead for fixed seed
- output duration remains 1 second

Conclusion:
- useful for proving the interpolation patch changed behavior
- not the original graduation-pack path

### 2. Direct stage-4 render path

Tried `tools/export_audio.py` through the real stage-4 route.

Outcome:
- import/runtime path is valid in `rafa-triton`
- fails against `bound_weights_ep49_step100.pt` because current `lib_blackwell.py` requires `g1_*` weights absent from the anchor checkpoint

### 3. Compatibility-shim experiments for the 14-key checkpoint

Using `bound_weights_ep49_step100.pt`, several fallback mappings were tested for the missing `g1_*` stage:

- `spine_split`
- `qkv_split`
- `phase_passthrough`
- `spine_mag_phase_input`

All of them can render 2-second audio through the stage-4 path.

This work was formalized into:
- `tools/export_audio_compat14.py`

The compatibility tool now:
- handles old 14-key checkpoints without touching live `core/lib_blackwell.py`
- skips `spectrum_id` routing when `spectrum_biases` are absent
- allows source and mixing sweeps for the missing `g1_*` projection stage

### 4. Broader 14-key checkpoint sweep

A broader compatibility sweep across representative 14-key checkpoints shows that the ledger anchor is not the best behavioral match under the current reconstructed runtime.

Representative results:
- `bound_weights_ep10_step1200.pt`: `mean_mse ~= 0.010140`
- `bound_weights_ep10_step1000.pt`: `mean_mse ~= 0.010289`
- `bound_weights_ep0_step1000.pt`: `mean_mse ~= 0.010639`
- `bound_weights_ep49_step100.pt`: `mean_mse ~= 0.010711`
- `bound_weights_ep99_step300.pt`: `mean_mse ~= 0.013357`

Artifacts:
- `tmp/gradpack_candidate_sweep/sweep_summary.json`
- `tmp/gradpack_local_sweep_ep8_12`

### 5. Current best restore candidate

The best current reconstruction found so far is:
- checkpoint: `bound_weights_ep10_step1200.pt`
- compatibility source: `qkv`
- `phase_mix = 0.35`
- `mag_mix = 0.05`
- `seed = 101`

Best sweep result:
- broad parameter sweep best: `mean_mse ~= 0.009442`
- deterministic seeded restore candidate: `mean_mse ~= 0.009565`

Current best render lane:
- `tmp/gradpack_best_repro_ep10_1200_qkv_p035_m005`
- canonical deterministic export: `outputs/graduation_pack_restore_candidate`

Artifacts:
- `tmp/gradpack_best_repro_ep10_1200_qkv_p035_m005/compat14_render_meta.json`
- `tmp/gradpack_best_repro_ep10_1200_qkv_p035_m005/comparison_summary.json`
- `tmp/gradpack_seed_sweep_ep10_1200_qkv_p035_m005/seed_sweep_summary.json`
- `outputs/graduation_pack_restore_candidate/comparison_summary.json`

### 6. Determinism fix

The stage-4 export path was not deterministic because `q_sample_x0(...)` draws fresh Gaussian noise and the export scripts did not seed the RNG state.

This is now fixed in the compatibility lane:
- `tools/export_audio_compat14.py`
  - adds explicit seeding
  - enables deterministic CuDNN mode

Confirmed:
- rerendering the same checkpoint/settings/seed now produces identical WAV hashes

Canonical restore wrapper:
- `tools/export_graduation_pack_restore.py`

Default best-known restore config:
- checkpoint: `bound_weights_ep10_step1200.pt`
- source: `qkv`
- `phase_mix = 0.35`
- `mag_mix = 0.05`
- `seed = 101`

This is better than the old `ep49`-based compatibility defaults, but it is still not a proof of exact historical restoration.

## Most Important Conclusion

The graduation-pack regression is not primarily "a bad interpolation patch" problem.

It is a path confusion problem:

- the wrong sampler stack was used
- the wrong checkpoint family was treated as the anchor
- the current `lib_blackwell.py` no longer matches the true graduation anchor checkpoint schema

## Current Best Model of the Truth

The original graduation pack was likely generated by:
- the stage-4 export path
- a 14-key Blackwell core runtime
- from the pre-`g1_*` family before the `ep50` runtime change
- with an older `lib_blackwell` forward variant that is no longer present in exact form

What is now less likely:
- that `sample_diffusion.py` was ever the right restore path
- that the current hardcoded `ep99_step300` export target is the graduation anchor
- that `ep49_step100` is necessarily the literal source checkpoint, even if it remains a plausible lineage anchor

## Recommended Next Moves

1. Stop using `sample_diffusion.py` and `diff_step1000.pt` for graduation-pack restoration.
2. Treat `tools/export_audio.py` + stage-4 checkpoints as the real restoration lane.
3. Recover the historical 14-key `lib_blackwell` forward path if exact reproduction is required.
4. Keep benchmarking 14-key checkpoints through `tools/export_audio_compat14.py` to narrow the most likely source checkpoint.
5. Investigate remaining nondeterminism or hidden config/data drift in the stage-4 path before claiming restoration.

## Useful Paths

- `outputs/graduation_pack`
- `outputs/expanded_graduation_pack`
- `tools/export_audio.py`
- `tools/infer_rafa.py`
- `lineages/03_subtractive_fork/train_stage4_joint.py`
- `core/lib_blackwell.py`
- `checkpoints_stage4/bound_weights_ep49_step100.pt`
- `checkpoints_stage4/bound_weights_ep10_step1200.pt`
- `checkpoints_stage4/bound_weights_ep99_step200.pt`
- `checkpoints_stage4/bound_weights_ep99_step300.pt`
- `tmp/legacy_restore_gradpack`
- `tmp/gradpack_variants`
- `tmp/gradpack_candidate_sweep/sweep_summary.json`
- `tmp/gradpack_best_repro_ep10_1200_qkv_p035_m005/comparison_summary.json`
