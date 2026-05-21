# Project Scaffolding Guide

This is the persistent rulebook for future RAFA scripts. It exists because the project now has enough exploratory code that script identity has become an engineering risk.

## Script Identity Contract

Every new script should be easy to classify without reading the whole file.

Required header-level answers:

- Runtime or lane: `circleworld_proto`, `stage4_blackwell_14`, `diffusion_parent_v3`, `positive_replacement`, `tools`, or explicit `research_track`.
- Role: `trainer`, `experiment_runner`, `evaluator`, `benchmark`, `scorer`, `comparator`, `report_assembler`, `auditor`, `builder`, `exporter`, `contract_test`, or `runtime_or_shared_module`.
- Artifact outputs: name every JSON/Markdown/checkpoint path pattern the script writes.
- Provenance inputs: checkpoint, config, seed, source lockbox, target lockbox, or manifest path.
- Future-access stance for audio work: declare whether target/future phase or magnitude is read, and why.

## Lifecycle Labels

- `canonical_or_manifested`: appears in a runtime manifest or registry.
- `documented_or_reported`: referenced in reports or architecture docs.
- `contract_or_test`: verifies an invariant or schema.
- `active_unregistered_research`: useful but not yet manifest/report canonical.
- `legacy_or_tooling`: old root/tooling script; preserve until explicitly retired.
- `retire_candidate`: only after an audit identifies replacement, artifacts, and no live references.

## Promotion Rules

A script can move from scout to canonical only if:

- It has a deterministic CLI with `argparse` or a clear launcher contract.
- It emits machine-readable JSON and, when useful, a Markdown report.
- It is referenced by a report, manifest, registry, DAG overlay, or ledger entry.
- It has a fresh smoke run or contract test.
- It does not silently compare lineages without naming the runtime/checkpoint boundary.

## Duplication Rules

- If a helper appears in 3 scripts, mark it for extraction.
- If a helper appears in 5 scripts, extract it before adding another variant unless there is a measured reason not to.
- For Circleworld report/scorer/probe scripts, use `runtimes/circleworld_proto/common_io.py` before copying JSON, scalar coercion, mean/median, formatting, CSV-float, gain parsing, or safe-device helpers.
- For audio continuation metrics, use `runtimes/circleworld_proto/phase_audio_metrics.py` before copying waveform correlation, MSE, loop/reentry, autocorrelation, chunk-vector, or cosine helpers.
- For report boilerplate, use `runtimes/circleworld_proto/experiment_reports.py` before copying Markdown write/table/front-matter helpers.
- For phase-native route/model/profile identifiers, use `runtimes/circleworld_proto/profile_registry.py` before hard-coding route IDs or route key conversion helpers.
- For seeded substrate logic, use `runtimes/circleworld_proto/seed_substrate.py` before copying seed-count/list/source/case-key parsing helpers.
- If a script exceeds 2,000 nonblank LOC, split report assembly, metric computation, and CLI orchestration unless the file is a frozen historical artifact.
- If two scripts differ only by objective weights or route tables, prefer one parameterized runner plus named profile JSON.
- If a script is superseded, keep a compatibility wrapper until reports and ledgers point to the replacement.

## Canonical Output Patterns

- Runtime outputs: `outputs/<runtime>/<run_id>/...`
- Checkpoints: `checkpoints_<runtime>/<run_id>/...`
- Architecture scaffolding: `docs/architecture/...`
- Experiment reports: `docs/reports/<LANE>_<TOPIC>_<YYYY-MM-DD>.md`
- Machine ledgers: JSONL next to the Markdown ledger when the lane already uses one.

## Required Cartography Loop

Run this after each large experiment burst:

```powershell
python tools/audit_project_scripts.py
python tools/audit_project_usage.py
python -m py_compile tools/audit_project_scripts.py
python -m py_compile tools/audit_project_usage.py
```

Then inspect:

- `docs/architecture/PROJECT_SCRIPT_INVENTORY.md`
- `docs/architecture/PROJECT_REDUNDANCY_HOTSPOTS.md`
- `docs/architecture/PROJECT_SCRIPT_INVENTORY.json`
- `docs/architecture/PROJECT_USAGE_MAP.md`
- `docs/architecture/PROJECT_ARTIFACT_INVENTORY.md`
- `docs/architecture/PROJECT_RETIREMENT_CANDIDATES.md`

Note: `PROJECT_RETIREMENT_CANDIDATES.md` intentionally references every classified script. After it exists, usage-map `documented_reference` counts are no longer a clean proxy for independent experiment/report provenance; use the retirement categories for deletion triage.

## Cleanup Sequence

1. Freeze current artifacts and ledger interpretation.
2. Extract shared helpers without changing outputs.
3. Add compatibility wrappers for renamed entrypoints.
4. Rerun contract tests and a representative smoke run.
5. Mark old scripts as `retire_candidate` in docs before deletion.
6. Delete only after a separate cleanup patch verifies no manifest/report/ledger reference remains.

## RAFA-Specific Guardrails

- Do not let Circleworld scaffold claims overwrite Graduation runtime claims.
- Do not compare a Circleworld adapter failure to Graduation itself unless the protected Graduation runtime was actually run.
- For phase-native audio work, no-future route selection must remain explicit.
- For resonant memory work, dense signatures are not RAFA post-tokens until retrieval, composition, and causal-use tests pass.
- For childworld ontology work, internal labels are evidence only if they connect to causal writeback, retrieval, composition, or audio-facing continuity.

Last generated from inventory summary: `2026-05-20T17:29:07+00:00`.
