# RAFA Branch Delegation Map

This document defines how repo work should be delegated across Git branches.

It exists because simply creating a new branch for every session creates clutter. RAFA needs branch lanes that correspond to scientific claims, runtime ownership, and artifact governance.

## Current Branch Topology

Snapshot from `2026-05-21`:

| Branch | Role | Current interpretation |
| --- | --- | --- |
| `master` | historical baseline | Original validation/protocol baseline. Do not use for active research work. |
| `codex/graduation-runtime` | Graduation audio baseline | Stable RAFA audio-engine lane. Circleworld cleanup must not land here unless it is explicitly a bridge or comparison wrapper. |
| `codex/circleworld` | Circleworld main research lane | Published Circleworld branch. Should stay focused on validated Circleworld runtime/assay changes. |
| `codex/circleworld-child-ifs-training-2026-05-10` | child-local IFS experiment lane | Specific childworld training/IFS branch. Should not accumulate project-wide cleanup or cartography work. |
| `codex/circleworld-cleanup-cartography-2026-05-21` | temporary triage shelf | Current dirty work landed here only to stop it sitting on the child-IFS branch. It should be split into durable lanes before promotion. |
| `codex/repo-migration` | structural repo migration | Runtime/lineage/registry organization. Use only for structural moves, wrappers, and compatibility migration. |
| `codex/research-track` | broad research documentation lane | Claim ledgers, reports, and conceptual synthesis that are not runtime-specific. |
| `codex/runtime-contracts` | registry/DAG/contract lane | Runtime manifests, contract tests, DAG health, schema guards. |

## Durable Branch Families

Use these lanes instead of minting one-off branches.

### 1. Graduation Runtime

Branch: `codex/graduation-runtime`

Owns:
- canonical Graduation audio runtime
- Graduation smoke/eval wrappers
- Graduation-vs-other comparison harnesses, only when read-only against Graduation internals

Does not own:
- Circleworld childworld experiments
- repository cartography
- speculative ontology docs

Promotion rule:
- only audio-facing changes with direct Graduation relevance

### 2. Circleworld Runtime

Branch: `codex/circleworld`

Owns:
- `lineages/04_positive_replacement/circleworld.py`
- Circleworld runtime paths
- Circleworld evaluator/trainer changes
- childworld/native-multimode/runtime assay code once validated

Does not own:
- broad repo inventory
- retirement candidate lists
- pure documentation synthesis unrelated to Circleworld execution

Promotion rule:
- must pass runtime contract tests, unit-phasor/gauge checks when relevant, DAG audit, and Circleworld smoke/eval scripts affected by the change

### 3. Circleworld Experimental Spikes

Branch naming:
- `codex/circleworld-<mechanism>-YYYY-MM-DD`

Owns:
- isolated speculative experiments
- non-promoted probes
- temporary assay runners

Exit paths:
- promote validated runtime pieces into `codex/circleworld`
- move docs/evidence into `codex/research-track`
- mark obsolete scripts as retirement candidates before deletion

### 4. Runtime Contracts

Branch: `codex/runtime-contracts`

Owns:
- runtime manifests
- DAG audits
- registry integrity tests
- schema/backward-compatibility tests
- shared runtime contract documentation

Does not own:
- experimental model behavior changes unless the change is a contract wrapper

### 5. Repo Migration And Cleanup

Branch: `codex/repo-migration`

Owns:
- mechanical file moves
- compatibility wrappers
- shared helper extraction when it is not Circleworld-specific
- cleanup scaffolding
- retirement-candidate process

Does not own:
- scientific result interpretation
- model behavior changes

Promotion rule:
- no behavior change unless explicitly documented
- py_compile and registry/DAG/contract checks must pass

### 6. Research Track

Branch: `codex/research-track`

Owns:
- claim taxonomy
- interpretation reports
- ledgers
- constitutions
- project-level scientific synthesis

Does not own:
- runtime code changes

### 7. Project Cartography

Preferred durable home: `codex/repo-migration`

Temporary branch allowed:
- `codex/project-cartography-YYYY-MM-DD`

Owns:
- `project_cartography/`
- `tools/audit_project_scripts.py`
- `tools/audit_project_usage.py`
- generated script/artifact inventory docs
- redundancy hotspot reports
- branch delegation maps

Promotion rule:
- generated inventories must match current repo state
- no deletion authorized by cartography alone

## Current Dirty Work Allocation

The current worktree has dirty/untracked material spread across:

| Area | Count class | Correct branch lane |
| --- | --- | --- |
| `docs/architecture/*` stocktake, constitutions, runtime contracts | modified/new docs | split between `codex/research-track`, `codex/runtime-contracts`, and `codex/repo-migration` |
| `docs/reports/*` Circleworld/RAFA reports | many untracked reports | `codex/research-track` unless tied to a runtime contract artifact |
| `lineages/04_positive_replacement/circleworld.py` | modified runtime core | `codex/circleworld` or a named Circleworld spike branch |
| `lineages/04_positive_replacement/rafa_relational_signature*.py`, `semantic_projector.py` | new signature/semantic code | named experimental branch until promoted; not Graduation |
| `runtimes/circleworld_proto/common_io.py`, `phase_audio_metrics.py`, `experiment_reports.py`, `profile_registry.py`, `seed_substrate.py` | shared Circleworld cleanup infrastructure | `codex/repo-migration` if no behavior change; `codex/circleworld` if runtime behavior depends on it |
| `runtimes/circleworld_proto/run_*`, `score_*`, `train_*`, `test_*` new scripts | active Circleworld research scripts | named Circleworld spike branches or `codex/circleworld` after validation |
| `runtimes/circleworld_proto/runtime_manifest.json` | modified registry/manifest | `codex/runtime-contracts` unless bundled with a Circleworld runtime promotion |
| `project_cartography/*` | root dashboard mirror | `codex/repo-migration` / project cartography lane |
| `tools/audit_project_scripts.py`, `tools/audit_project_usage.py` | audit tooling | `codex/repo-migration` / project cartography lane |

## Safe Migration Procedure

Do not keep creating branches with the whole dirty tree.

Use this sequence instead:

1. Freeze the current dirty state on a temporary triage branch.
2. Generate grouped patch bundles by lane.
3. Create or switch to the durable branch for one lane.
4. Apply only that lane's patch bundle.
5. Verify that lane independently.
6. Commit that lane if requested.
7. Repeat for the next lane.
8. Delete the temporary triage branch only after every wanted change exists in a durable branch or patch bundle.

Preferred patch-bundle names:

| Bundle | Target lane |
| --- | --- |
| `branch_bundle_project_cartography.patch` | `codex/repo-migration` |
| `branch_bundle_runtime_contracts.patch` | `codex/runtime-contracts` |
| `branch_bundle_circleworld_runtime.patch` | `codex/circleworld` |
| `branch_bundle_circleworld_spikes.patch` | named Circleworld spike branch |
| `branch_bundle_research_reports.patch` | `codex/research-track` |

## Branch Hygiene Rules

- One branch should answer one governance question.
- A branch should not mix runtime behavior changes with broad cartography unless the cartography is only documenting that runtime change.
- Reports can be large, but they must have a lane and a corresponding source script or ledger entry.
- Generated inventories belong to cartography, not to scientific-result branches.
- Graduation changes are isolated by default.
- Circleworld is allowed to be messy inside spike branches, but promoted Circleworld branches must pass contracts.
- Retirement candidates are labels, not deletion permission.
- Deleting or force-resetting branches requires explicit user approval.

## Immediate Recommendation

Treat `codex/circleworld-cleanup-cartography-2026-05-21` as a triage shelf, not a final branch.

Next safe action:

1. Generate patch bundles for the current dirty tree by target lane.
2. Review bundle membership before applying anything.
3. Move the cartography/helper-extraction bundle to `codex/repo-migration`.
4. Move runtime contract changes to `codex/runtime-contracts`.
5. Keep childworld/operator/phase-native experiments on `codex/circleworld` or named Circleworld spike branches.

This prevents branch multiplication while still rescuing the current overloaded worktree.
