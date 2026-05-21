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

### 5. Research lanes are not runtime families

An operating lane can define a claim, constitution, or evaluator without creating
a new runtime family.

The Resonant Attention / Post-Token Memory lane is currently such a lane:

- It may evaluate Circleworld packets, childworld records, dense relational
  signatures, and branch-law diagnostics.
- It does not currently define a new checkpoint schema or entrypoint.
- It does not rename `circleworld_proto` or `positive_replacement`.
- `relational_qkv_v2` remains a local Circleworld branch-QKV precursor, not the
  runtime contract for full RAFA attention.

If this lane later owns a distinct loader, checkpoint schema, or inference
entrypoint, it must be added as a runtime family or adapter in the registries.

## Current Runtime Families

- `stage4_blackwell_14`
- `stage4_blackwell_16`
- `diffusion_parent_v3`
- `circleworld_proto`

Current docs-only research lanes using existing runtime artifacts:

- `resonant_attention_post_token_memory_lane` over `circleworld_proto` and
  dense-signature artifacts; no separate runtime contract yet.

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

## Script Cartography Standard

RAFA now has a repo-wide script inventory and redundancy audit layer:

- [D:\RAFA\project_cartography\README.md](D:\RAFA\project_cartography\README.md)
- [D:\RAFA\tools\audit_project_scripts.py](D:\RAFA\tools\audit_project_scripts.py)
- [D:\RAFA\docs\architecture\PROJECT_SCRIPT_INVENTORY.md](D:\RAFA\docs\architecture\PROJECT_SCRIPT_INVENTORY.md)
- [D:\RAFA\docs\architecture\PROJECT_SCRIPT_INVENTORY.json](D:\RAFA\docs\architecture\PROJECT_SCRIPT_INVENTORY.json)
- [D:\RAFA\docs\architecture\PROJECT_REDUNDANCY_HOTSPOTS.md](D:\RAFA\docs\architecture\PROJECT_REDUNDANCY_HOTSPOTS.md)
- [D:\RAFA\docs\architecture\PROJECT_SCAFFOLDING_GUIDE.md](D:\RAFA\docs\architecture\PROJECT_SCAFFOLDING_GUIDE.md)
- [D:\RAFA\tools\audit_project_usage.py](D:\RAFA\tools\audit_project_usage.py)
- [D:\RAFA\docs\architecture\PROJECT_USAGE_MAP.md](D:\RAFA\docs\architecture\PROJECT_USAGE_MAP.md)
- [D:\RAFA\docs\architecture\PROJECT_USAGE_MAP.json](D:\RAFA\docs\architecture\PROJECT_USAGE_MAP.json)
- [D:\RAFA\docs\architecture\PROJECT_ARTIFACT_INVENTORY.md](D:\RAFA\docs\architecture\PROJECT_ARTIFACT_INVENTORY.md)

Every large experiment burst should regenerate the inventory before cleanup
or promotion decisions:

```powershell
python tools/audit_project_scripts.py
python tools/audit_project_usage.py
```

The inventory does not authorize deletion. It classifies scripts by lane, role,
lifecycle, manifest/documentation visibility, and redundancy hotspots so cleanup
can be staged without breaking historical artifacts.

The usage map adds static evidence for imports, manifest/report references,
artifact groups, and cold artifact candidates. It normalizes common top-level
artifact aliases such as `outputs/`, `eval/`, `logs/`, and `checkpoints_*`
into the consolidated `artifacts/` tree when possible, but it still remains a
static analysis layer rather than proof that a file is safe to remove.

The `project_cartography/` root folder is a convenience mirror for navigation.
The canonical generated outputs remain under `docs/architecture/` and
`docs/reports/`, and the runnable auditors remain under `tools/`.

Future scripts should follow the identity contract in
`PROJECT_SCAFFOLDING_GUIDE.md`: declare runtime/lane, role, outputs,
provenance inputs, and future-access stance for audio experiments. Helper
logic repeated across three or more scripts should be marked for extraction;
helper logic repeated across five or more scripts should generally be extracted
before adding another one-off variant.
