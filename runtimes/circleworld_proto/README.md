# Runtime: Circleworld Proto

This is the positive-replacement formalization prototype.

## Status

- active prototype
- ablation-focused
- not yet integrated into the main generation stack

## Canonical Entry Point

- `run_ablation.py`
- `train_circleworld.py`
- `evaluate_circleworld.py`

## Scope

This runtime is for:

- coherent-structure detection
- packet promotion
- recursive packet feedback

It is not yet a production generation runtime.

## Training Lane

`train_circleworld.py` runs an isolated parameter-training attempt for Circleworld itself.

It is intentionally separate from:
- Graduation restore
- diffusion parent prompt-conditioning
- semantic steering / tension lane

Artifacts from this lane live under:
- `outputs/circleworld_proto/`
- `checkpoints_circleworld_proto/`

## Evaluation

`evaluate_circleworld.py` runs a held-out config evaluation and exports:
- per-sample summary JSON
- per-time trajectory CSV

This is the main inspection path for trained Circleworld configs.

`benchmark_circleworld_real_anchor.py` runs the bit-stable real-audio benchmark used for
reference-vs-output preservation checks.

`benchmark_audio_continuity.py` runs the macro-time continuity / loop benchmark used for
repetition, re-entry, and time-world diagnostics.

`test_nested_commitment.py` runs the fork/resume assay used to test whether recursion
preserves a coarse world-law while later recursion only refines it.

Current retrospective read:
- the best real-audio Circleworld runs are still preservation-heavy
- they do not yet create a new macro-time law
- the current fork/resume evidence points to over-rigid attractors rather than
  Matryoshka-like nested commitment
- the next frontier is reducing loop/re-entry without leaving the real-audio frontier
