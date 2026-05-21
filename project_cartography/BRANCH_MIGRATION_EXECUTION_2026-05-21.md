# Branch Migration Execution Report - 2026-05-21

This report records the first branch-delegation execution pass after the overloaded dirty worktree was moved onto `codex/circleworld-cleanup-cartography-2026-05-21`.

The goal was not to create more branch clutter. The goal was to preserve current work into durable lane worktrees and expose which lanes are actually independent.

## Source State

- Source checkout: `D:\RAFA`
- Source branch: `codex/circleworld-cleanup-cartography-2026-05-21`
- Source role: temporary triage shelf
- Initial dirty paths: `266`

No source files were staged, committed, deleted, or reset during this pass.

## Worktrees Created

Sibling worktree root: `D:\RAFA_worktrees`

| Worktree | Branch | Role |
| --- | --- | --- |
| `D:\RAFA_worktrees\repo-migration` | `codex/repo-migration` | repo cartography, cleanup scaffolding, audit tooling |
| `D:\RAFA_worktrees\runtime-contracts` | `codex/runtime-contracts` | parked contract extraction candidate |
| `D:\RAFA_worktrees\research-track` | `codex/research-track` | reports, constitutions, interpretation ledger |
| `D:\RAFA_worktrees\circleworld` | `codex/circleworld` | Circleworld runtime/integration lane |
| `D:\RAFA_worktrees\circleworld-signature-spike-2026-05-21` | `codex/circleworld-signature-spike-2026-05-21` | signature/semantic spike lane |

One new branch was created:

- `codex/circleworld-signature-spike-2026-05-21`

## Bundle Application

Manifests live under:

- `project_cartography/branch_bundles/2026-05-21/`

Initial path-preserving copy counts:

| Manifest | Target worktree | Files copied |
| --- | --- | ---: |
| `lane_repo_migration_project_cartography.txt` | `repo-migration` | 18 |
| `lane_runtime_contracts.txt` | `runtime-contracts` | 16 |
| `lane_research_track.txt` | `research-track` | 113 |
| `lane_circleworld_runtime_or_spike.txt` | `circleworld` | 116 |
| `lane_circleworld_signature_spike.txt` | `circleworld-signature-spike-2026-05-21` | 3 |

Two dependency corrections were necessary:

1. `project_cartography/` had to be copied recursively into `repo-migration`.
2. `runtime_contracts` files and signature files also had to be copied into `circleworld`.

The second correction is important: current Circleworld runtime code imports relational signature code, and Circleworld runtime verification needs the relevant contract files co-located with the runtime. Therefore the standalone `runtime-contracts` worktree is a parked extraction candidate, not independently promotable yet.

## Verification Results

### Diff Sanity

`git diff --check` passed in all five worktrees.

Only CRLF normalization warnings were reported.

### Syntax And JSON

| Worktree | Check | Result |
| --- | --- | --- |
| `repo-migration` | `py_compile` changed Python files | pass, 2 files |
| `repo-migration` | JSON parse over branch JSON files | pass |
| `runtime-contracts` | `py_compile` changed Python files | pass, 14 files |
| `circleworld` | `py_compile` changed Python files after dependency correction | pass, 132 files |
| `circleworld-signature-spike-2026-05-21` | `py_compile` changed Python files | pass, 3 files |

### Circleworld Runtime Contracts

Tiny CPU unit-phasor smoke:

- Command: `test_unit_phasor_contract.py`
- Result: pass
- Tensor count: `35`
- Max phasor norm error: `1.1920928955078125e-07`
- Hidden magnitude channel detected: `false`

Tiny CPU gauge-invariance smoke:

- Command: `test_phase_gauge_invariance.py`
- Result: pass
- Max q/arc/branch drift remained at float-noise scale for tested global rotations and phase-preserving rescalings.

Circleworld DAG audit:

- Result: `warn`
- Good news: manifest structure and entrypoint checks were ok, and compile status was ok.
- Warning source: recent-run artifacts are not present inside the isolated worktree's `outputs/` tree.

Registry integrity pytest:

- Result: failed in the isolated `circleworld` worktree.
- Failure reason: expected large external artifact/checkpoint paths are absent from the sibling worktree, for example `checkpoints_stage4/bound_weights_ep10_step1200.pt`, `outputs/graduation_pack_restore_candidate`, and `checkpoints_circleworld_proto`.
- Interpretation: this is an artifact-location problem for isolated worktrees, not evidence that the branch copy lost code.

## Validator Follow-Up

The first repo-migration validator pass found two scaffolding hygiene issues:

1. `project_cartography/BRANCH_DELEGATION_MAP.md` and `project_cartography/BRANCH_MIGRATION_EXECUTION_2026-05-21.md` were abbreviated summaries, while the manifest treated them as canonical mirrors.
2. Lane manifests used literal `` `t`` text between Git status and path, which made them poor machine-readable path manifests.

Both were corrected:

- Branch map, migration preview, and migration execution files in `project_cartography/` now literally mirror their canonical `docs/architecture/` files.
- `lane_*.txt` files are now clean path-only manifests.
- `lane_*_status.tsv` files preserve the original Git status plus path for auditability.
- Mirror/hash sanity checks now report no mirror problems.
- Path manifest checks now report zero malformed lines.

## What This Proved

The split is partially real:

- repo-migration/cartography can be isolated cleanly as code/docs scaffolding.
- research-track can be isolated as interpretation/report material.
- Circleworld runtime needs its contract files and relational signature dependency co-located.
- `runtime-contracts` is not independently promotable yet unless the relevant runtime dependencies are also present or the tests are rewritten to resolve artifact roots externally.

## Cleanup Decision

Do not delete `codex/circleworld-cleanup-cartography-2026-05-21` yet.

Reason:

- The durable worktrees preserve the current content by lane, but no lane commits have been made.
- Some branches are intentionally dirty and uncommitted.
- `runtime-contracts` is not independently verified as a standalone branch.
- Circleworld verification still depends on external artifact roots outside the isolated worktree.

The temporary branch remains the safety shelf until lane commits or explicit patch archives exist.

## Next Safe Step

1. Commit or archive the `repo-migration` cartography bundle first.
2. Fold Circleworld runtime contracts and signature dependency into the Circleworld integration lane.
3. Leave `runtime-contracts` as a future extraction branch only after contract tests can run against an external artifact root or a small fixture set.
4. Commit research-track reports separately from runtime code.
5. Only after those commits exist, re-evaluate whether the temporary triage branch can be deleted.

## Completion Pass

Additional cleanup completed after the initial execution report:

| Branch | Commit | Status |
| --- | --- | --- |
| `codex/repo-migration` | `2e4c6b8 docs: add RAFA project cartography branch split` | clean, ahead of origin by 1 |
| `codex/research-track` | `9d5b4e5 docs: preserve RAFA research track reports` | clean, ahead of origin by 1 |
| `codex/runtime-contracts` | `010c63c test: add Circleworld runtime contract audits` | clean, ahead of origin by 1 |
| `codex/circleworld-signature-spike-2026-05-21` | `7bf8f3e feat: add RAFA relational signature spike` | clean, local branch |
| `codex/circleworld` | `e54f21d feat: consolidate Circleworld phase-native integration lane` | clean, ahead of origin by 1 |

Circleworld cleanup actions during completion:

- Added the manifest-referenced `RAFA_PHASE_NATIVE_AUDIO_RESET_SUITE_2026-05-17.md` report to the Circleworld lane.
- Synchronized the `circleworld_proto` row in `registries/RUNTIME_REGISTRY.json` with the expanded runtime manifest entrypoint surface.
- Verified tracked manifest entrypoint/report paths exist, excluding large `outputs/` artifact roots.
- Verified Circleworld imports for the shared cleanup helpers, phase-native audio operators, relational signature files, and operator block runner.
- Verified cached Python compile for 132 staged Python files.
- Verified focused Circleworld contract subset: `26 passed`.

Remaining safety shelf:

- `codex/circleworld-cleanup-cartography-2026-05-21` is still dirty in `D:\RAFA` by design.
- Its contents are now preserved in durable lane commits, but it was not deleted/reset because that would be destructive.
- The next operator can retire it after reviewing the commits and explicitly approving destructive cleanup of the aggregate checkout.
