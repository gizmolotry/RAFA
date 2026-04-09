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
