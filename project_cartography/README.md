# RAFA Project Cartography

This root folder is the quick entry point for understanding what exists in the RAFA repo, what uses what, and where redundancy/artifact sprawl is accumulating.

## Start Here

1. `PROJECT_USAGE_MAP.md` - script usage status plus artifact group references.
2. `PROJECT_SCRIPT_INVENTORY.md` - every visible source-like script, classified by lane/role/lifecycle.
3. `PROJECT_REDUNDANCY_HOTSPOTS.md` - repeated helpers, duplicate bodies, and extraction targets.
4. `PROJECT_ARTIFACT_INVENTORY.md` - output/checkpoint/log/report artifact groups.
5. `PROJECT_SCAFFOLDING_GUIDE.md` - rules for future scripts and cleanup.
6. `PROJECT_RETIREMENT_CANDIDATES.md` - retirement triage; not deletion authorization.
7. `BRANCH_DELEGATION_MAP.md` - branch ownership lanes and safe dirty-work migration policy.
8. `BRANCH_MIGRATION_PREVIEW_2026-05-21.md` - current dirty-tree allocation by durable branch lane.
9. `BRANCH_MIGRATION_EXECUTION_2026-05-21.md` - branch split execution report and preservation status.
10. `RAFA_CODEBASE_CLEANUP_PASS_2026-05-20.md` - first no-behavior-change helper extraction pass.
11. `RAFA_CODEBASE_CLEANUP_ORCHESTRATION_2026-05-20.md` - multi-agent cleanup orchestration pass.

## Canonical vs Mirror

This folder is a convenience mirror. The canonical generated files remain in `docs/architecture` and `docs/reports`.

The runnable auditors remain in:

- `tools/audit_project_scripts.py`
- `tools/audit_project_usage.py`

## Regenerate

Run from `D:\RAFA`:

```powershell
python tools\audit_project_scripts.py
python tools\audit_project_usage.py
Copy-Item docs\architecture\PROJECT_* project_cartography\ -Force
Copy-Item docs\reports\RAFA_PROJECT_CARTOGRAPHY_2026-05-19.md project_cartography\ -Force
Copy-Item docs\reports\RAFA_PROJECT_USAGE_AUDIT_2026-05-19.md project_cartography\ -Force
Copy-Item docs\reports\RAFA_CODEBASE_CLEANUP_PASS_2026-05-20.md project_cartography\ -Force
Copy-Item docs\reports\RAFA_CODEBASE_CLEANUP_ORCHESTRATION_2026-05-20.md project_cartography\ -Force
```

Then inspect `CARTOGRAPHY_MANIFEST.json` to confirm the mirrored files and byte sizes.

## Safety Rule

Nothing in this folder authorizes deletion. Low-reference, watchlist, and cold artifact labels mean inspect first, not remove.
