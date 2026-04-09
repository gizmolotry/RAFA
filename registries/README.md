# Registries

`registries/` is the provenance layer for RAFA.

These files are the source of truth for:

- named runtime families
- checkpoint schema families
- lineage-to-runtime mapping
- shared module ownership

## Files

- `RUNTIME_REGISTRY.json`
- `CHECKPOINT_REGISTRY.json`
- `LINEAGE_REGISTRY.json`
- `SHARED_MODULE_REGISTRY.json`

## Rule

No important checkpoint or artifact should exist without:

1. a runtime family id
2. a checkpoint schema id
3. a canonical entrypoint
