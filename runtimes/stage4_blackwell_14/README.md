# Runtime: Stage4 Blackwell 14

This is the old 14-key stage-4 Blackwell runtime family.

## Status

- canonical restore family for the graduation pack
- loaded through a compatibility runtime because the exact pre-`g1_*` forward is not preserved verbatim in the live repo

## Key Characteristics

- 14-key `core` checkpoint schema
- no `g1_w_ph`
- no `g1_w_mag`
- old-family stage-4 inference contract

## Canonical Entry Points

- `export_graduation.py`
- `D:\\RAFA\\tools\\export_graduation_pack_restore.py`

## Best-Known Restore Candidate

- checkpoint: `checkpoints_stage4/bound_weights_ep10_step1200.pt`
- source: `qkv`
- `phase_mix = 0.35`
- `mag_mix = 0.05`
- `seed = 101`
