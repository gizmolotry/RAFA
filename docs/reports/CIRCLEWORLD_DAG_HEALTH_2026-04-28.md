# Circleworld DAG Health 2026-04-28

## Result

Circleworld runtime DAG status is `ok`.

Audit artifact:

- `D:\RAFA\outputs\circleworld_proto\dag_health_audit_2026-04-28.json`

## What was fixed

1. `run_branchlaw_ablation_series.py` now emits the same core sidecars as the other Circleworld experiment runners:
   - `heldout_summary`
   - `benchmark_summary`
   - `nested_summary`
   - `law_token_library`

2. `audit_circleworld_dag.py` now treats the retrospective lane correctly:
   - imported `_run_nested_commitment(...)` reuse counts as a valid nested-assay path
   - reevaluated checkpoints can satisfy the heldout artifact check through an explicit external artifact path

## Health read

- runtime manifest: `ok`
- Circleworld runtime + lineage compile sweep: `ok`
- experiment runner sidecar uniformity: `ok`
- recent run artifact consistency: `ok`

## Current conclusion

The Circleworld DAG is now uniform in the places that matter for ongoing work:

- shared sidecars are consistent across the active experiment runners
- nested-assay routing is consistent across direct and retrospective paths
- recent benchmarked runs resolve to complete artifact sets

Graduation was not touched.
