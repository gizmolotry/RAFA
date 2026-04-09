# Runtimes

`runtimes/` defines the named runtime families in this repo.

Each runtime family gets:

- a stable folder
- a manifest
- one or more canonical entrypoints
- a checkpoint schema identifier

## Runtime Families

- `stage4_blackwell_14/`
  - old 14-key stage-4 Blackwell family
  - current best graduation-pack restore lane
- `stage4_blackwell_16/`
  - newer 16-key stage-4 Blackwell family with explicit `g1_*` projection heads
- `diffusion_parent_v3/`
  - diffusion parent / sample-diffusion family
  - historically important, but not the graduation-pack restore path
- `circleworld_proto/`
  - positive-replacement / Circleworld formalization prototype

## Policy

Experiment scripts may call into these runtimes, but they should not redefine the runtime contract in-place.

If a new checkpoint family needs a different load rule or forward path, it should get:

1. a new runtime family id
2. a new manifest entry
3. a dedicated entrypoint or adapter
