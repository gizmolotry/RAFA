# RAFA Project Usage Audit - 2026-05-19

## What This Adds

This pass extends script cartography into a static usage map over scripts and artifact groups.

It normalizes common reparse/alias roots into the consolidated artifact tree, so top-level `outputs`, `eval`, `logs`, and checkpoint aliases are tracked without double-counting the same physical files.

It answers:

- Which scripts are imported, manifested, documented, or low-reference?
- Which artifact groups are referenced by source/docs/registries/reports?
- Which artifact groups are large and cold according to static references?
- Where should cleanup start without touching model semantics?

## Headline Counts

| Metric                          | Value       |
| ------------------------------- | ----------- |
| Scripts mapped                  | 239         |
| Artifact files indexed          | 54243       |
| Artifact groups                 | 979         |
| Artifact bytes                  | 93894554467 |
| Low-reference script candidates | 1           |
| Cold artifact group candidates  | 543         |

## Interpretation

The project has two different cleanup problems:

1. Script sprawl: many one-off Circleworld scripts are real experimental history but need helper extraction and lifecycle labels.
2. Artifact sprawl: many output/checkpoint groups are large and not statically referenced; they need archival decisions, not blind deletion.

The safest next move remains no-behavior-change consolidation of shared script helpers. Artifact cleanup should come later, after checkpoint/output groups are tied to reports or marked as expendable caches.

## Files

- `docs/architecture/PROJECT_USAGE_MAP.md`
- `docs/architecture/PROJECT_USAGE_MAP.json`
- `docs/architecture/PROJECT_ARTIFACT_INVENTORY.md`
