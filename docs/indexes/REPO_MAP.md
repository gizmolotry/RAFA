# Repo Map

## Code
- `foundation/`: shared module map for utilities used by multiple runtime families.
- `runtimes/`: stable runtime families and canonical entrypoints.
- `registries/`: provenance for runtime families, lineages, shared modules, and checkpoints.
- `core/`: current source of truth for the RAFA runtime, phase IFS, diffusion helpers, Triton kernels, dataset loader, and validation.
- `lineages/`: frozen lineage-specific entrypoints.
  - `01_parent_graduation/`: strongest older hybrid baseline.
  - `02_local_ablations/`: local knockouts from the parent lineage.
  - `03_subtractive_fork/`: stripped stage-4 fork.
  - `04_positive_replacement/`: Circleworld / Hardy-Littlewood positive replacement lane.
- `tools/`: top-level operational scripts.
- `research_track/`: DAGs, manifests, batch runners, infra reducers, and long-form experiment summaries.
- `inference_package/`: compact inference bundle with sample assets.

## Key Docs
- `docs/architecture/RAFA_COGNITIVE_STACK.md`
- `docs/architecture/RUNTIME_CONTRACTS.md`
- `docs/architecture/RAFA_LINEAGE_LEDGER.md`
- `docs/reports/ROOT_FAILURE_ANALYSIS_2026-03-12.md`
- `EVAL_LEDGER.md`

## Config + Entry Points
- Root config: `config.yaml`
- Lineage launcher: `launch_lineage.ps1`
- Functional training scripts: `train_functional.py`, `train_raw_tensors.py`, `train_sterile.py`

## High-Volume Artifact Areas
- `artifacts/`: consolidated historical checkpoints, bundles, and legacy exports
- `logs/`: detached logs and experiment logs
- `eval/`: evaluation outputs and scorecards
- `outputs/`: visualization and generated outputs
- `wav_files/`: local audio corpus
- `datasets/`: processed data helpers and manifests

## Suggested Navigation Pattern
1. Start at `README.md`
2. Identify the runtime family in `registries/`
3. Read the subsystem README in the area you need
4. Use lineage folders for historical architectural branches
5. Use artifact directories only after you know which run or lineage you want


