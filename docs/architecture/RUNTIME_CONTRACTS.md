# Runtime Contracts

This document defines how RAFA should be divided going forward.

## The Layering Rule

RAFA now has four layers of organization:

1. `foundation/`
   - shared modules and utilities
   - STFT, dataset IO, diffusion math, phasor math, kernels
2. `runtimes/`
   - named runtime families
   - each family owns its inference contract
3. `lineages/`
   - architectural branches and ablation families
4. `registries/`
   - provenance layer that says which runtime and schema each important checkpoint belongs to

## Why This Exists

The graduation-pack restore work showed that the repo had:

- multiple inference families
- multiple checkpoint schemas
- overlapping file names
- no hard provenance path from output back to runtime contract

That made restoration much harder than it should have been.

## Rules

### 1. Checkpoints are API contracts

A checkpoint is not just weights.

It implies:

- runtime family
- schema family
- entrypoint
- config regime
- data regime

Every important checkpoint should be represented in `registries/CHECKPOINT_REGISTRY.json`.

### 2. Runtime families do not silently absorb each other

If a checkpoint family needs:

- a different loader
- a different forward contract
- a compatibility adapter

then it gets a new runtime family or adapter entrypoint instead of mutating an existing family in-place.

### 3. Shared modules are shared, not sovereign

Modules in `foundation/` may be used by many runtimes.
They do not define the runtime contract by themselves.

The runtime contract lives in:

- runtime manifest
- canonical entrypoint
- checkpoint schema id

### 4. Lineages and runtimes are not the same thing

Lineages answer:
- what branch or philosophical family is this?

Runtimes answer:
- what exact code path loads and executes this checkpoint?

Both must be tracked.

## Current Runtime Families

- `stage4_blackwell_14`
- `stage4_blackwell_16`
- `diffusion_parent_v3`
- `circleworld_proto`

## Current Shared Foundation Modules

- `core/dataset.py`
- `diffusion_utils.py`
- `stft_utils.py`
- `rafa_math_tools.py`
- `core/triton_kernels.py`
- `core/triton_deq.py`
- `phase_native_ifs.py`
- `rafa_clutch_transformer_upgraded.py`

## Operational Standard

Before a new artifact is treated as canonical, we should be able to answer:

1. Which runtime family produced it?
2. Which checkpoint schema does it require?
3. Which entrypoint generates it?
4. Which seed/config pair was used?

If those answers are missing, the artifact is not canonical.
