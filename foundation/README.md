# Foundation

`foundation/` is the shared-module map for RAFA.

This folder is intentionally light-touch: it organizes the modules that multiple runtime families share without physically moving the existing implementation files out of their current locations.

## Shared Areas

- `core/dataset.py`
  - dataset loading, manifest-backed local WAV access, tension-track loading
- `diffusion_utils.py`
  - diffusion schedules, q-sampling, phasor conversions
- `stft_utils.py`
  - forward/inverse STFT utilities
- `rafa_math_tools.py`
  - phasor math, Ramanujan helpers, harmonic/inharmonic metrics
- `core/triton_kernels.py`
  - Triton kernels used by Blackwell-family runtimes
- `core/triton_deq.py`
  - experimental DEQ/Triton dynamic-depth work
- `phase_native_ifs.py`
  - phase-native IFS chamber used by newer runtime families
- `rafa_clutch_transformer_upgraded.py`
  - clutch/multiscale routing used by current RAFA-family paths

## Rule

Shared modules may be used by multiple runtime families, but a runtime family should own its own:

- entrypoint
- checkpoint schema
- inference contract
- smoke test

That keeps checkpoint families from silently drifting onto each other again.
