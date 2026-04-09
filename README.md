# RAFA

This repo is organized around four things:
- `foundation/`: shared modules and utilities used by multiple runtime families
- `runtimes/`: named runtime families with stable entrypoints and manifests
- `registries/`: provenance layer for checkpoint schemas, lineages, and runtime ownership
- `core/`: the current runtime and model substrate
- `lineages/`: the major architectural branches and ablation families
- `tools/`: runnable scripts for training, eval, audits, and diagnostics
- `research_track/`: orchestration, DAGs, manifests, and experiment summaries
- `artifacts/`: checkpoints, generated outputs, logs, bundles, and archived runs
- `data/`: corpora, scrapes, manifests, and raw inputs

## Start Here
- Repo map: [docs/indexes/REPO_MAP.md](docs/indexes/REPO_MAP.md)
- Architecture: [docs/architecture/RAFA_COGNITIVE_STACK.md](docs/architecture/RAFA_COGNITIVE_STACK.md)
- Runtime contracts: [docs/architecture/RUNTIME_CONTRACTS.md](docs/architecture/RUNTIME_CONTRACTS.md)
- Lineages: [docs/architecture/RAFA_LINEAGE_LEDGER.md](docs/architecture/RAFA_LINEAGE_LEDGER.md)
- Current eval ledger: [EVAL_LEDGER.md](EVAL_LEDGER.md)

## Main Working Areas
- Runtime families: [runtimes/README.md](runtimes/README.md)
- Shared modules: [foundation/README.md](foundation/README.md)
- Provenance registries: [registries/README.md](registries/README.md)
- Core runtime: [core/README.md](core/README.md)
- Experiment lineages: [lineages/README.md](lineages/README.md)
- Utility scripts: [tools/README.md](tools/README.md)
- Orchestration and reports: [research_track/README.md](research_track/README.md)
- Artifacts and bundles: [artifacts/README.md](artifacts/README.md)
- Data inputs: [data/README.md](data/README.md)
- Inference bundle: [inference_package/README.md](inference_package/README.md)

## Artifact Policy
Large experiment artifacts live in-place for now, but they are not the repo's conceptual entry points. Use the repo map first, then drop into artifact folders only when you need outputs.

## Provenance Policy
Important artifacts should now resolve through:

1. a runtime family in `registries/RUNTIME_REGISTRY.json`
2. a checkpoint schema in `registries/CHECKPOINT_REGISTRY.json`
3. a canonical entrypoint under `runtimes/`

