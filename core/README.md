# Core

`core/` is the current implementation surface for RAFA.

## Main files
- `blackwell_model.py`: main RAFA model assembly and clutch/IFS wiring
- `phase_native_ifs.py`: phase-native recursive chamber and impedance plumbing
- `lib_blackwell.py`: lightweight Blackwell-compatible RAFA/NakedRAFA implementations
- `blackwell_diffusion.py`, `diffusion_utils.py`: diffusion-side helpers
- `dataset.py`: dataset loading
- `rafa_math_tools.py`: phasor, Ramanujan, harmonic/inharmonic, and loss utilities
- `triton_kernels.py`, `triton_deq.py`: Triton kernels and DEQ experiments
- `validate.py`, `infer.py`, `blackwell_launch.py`: runtime utilities

## Rule of thumb
If you are changing the actual current model/runtime, start here.
