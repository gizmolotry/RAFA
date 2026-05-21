# RAFA Codebase Cleanup Orchestration - 2026-05-20

## Purpose

This pass executed the requested three cleanup steps with multi-agent orchestration:

1. Extract more common infrastructure.
2. Begin unifying near-duplicate runner families through named profiles and shared route helpers.
3. Mark retirement candidates without deleting scripts.

No scripts were deleted or renamed.

## Multi-Agent Tracks

| Track | Output |
| --- | --- |
| Metrics explorer | Identified audio metric extraction surfaces and caught a compatibility break after extraction. |
| Report helper worker | Added `experiment_reports.py`. |
| Profile registry worker | Added `profile_registry.py`. |
| Seed/substrate worker | Added `seed_substrate.py`. |
| Retirement classifier worker | Added `PROJECT_RETIREMENT_CANDIDATES.md`. |
| Parent integration | Added `phase_audio_metrics.py`, migrated route helpers into consumers, fixed compatibility aliases, verified DAG/contracts, regenerated cartography. |

## Common Infrastructure Added

| Module | Role | Current Consumption |
| --- | --- | --- |
| `runtimes/circleworld_proto/common_io.py` | JSON/text IO, scalar coercion, simple stats, formatting, gain parsing. | Used by internal phase-law analyzers and audio probe/scout scripts. |
| `runtimes/circleworld_proto/phase_audio_metrics.py` | Waveform comparison, loop/reentry, autocorr, chunk vectors, cosine helpers. | Used by `benchmark_audio_continuation.py`; legacy private aliases remain available from that script. |
| `runtimes/circleworld_proto/experiment_reports.py` | Markdown write/table/front-matter/status helpers. | Additive scaffold; next pass should migrate report bodies gradually. |
| `runtimes/circleworld_proto/profile_registry.py` | Frozen route/model/profile/lockbox IDs plus route key helpers. | Used by four phase-native route-policy scripts. |
| `runtimes/circleworld_proto/seed_substrate.py` | Seed list/count/source parsing, case-key normalization, seed plan records. | Additive scaffold; next pass should migrate seed-heavy assays. |

## Runner Unification Started

The following phase-native route-policy scripts now import route identity helpers from `profile_registry.py` instead of carrying local copies:

- `runtimes/circleworld_proto/run_phase_native_audio_selected_route.py`
- `runtimes/circleworld_proto/train_phase_native_audio_route_policy.py`
- `runtimes/circleworld_proto/train_phase_native_audio_family_route_policy.py`
- `runtimes/circleworld_proto/train_phase_native_audio_objective_route_policy.py`

This is compatibility-first unification: old CLIs remain intact, old JSON keys remain intact, and old entrypoint names remain intact.

## Compatibility Repair

During extraction, the metrics explorer found that `score_phase_native_audio_reentry_metric_audit.py` imported private `_chunk_vectors` and `_cosine` helpers from `benchmark_audio_continuation.py`.

Fix:

- `benchmark_audio_continuation.py` now imports and re-exports underscore aliases from `phase_audio_metrics.py`.
- `score_phase_native_audio_reentry_metric_audit.py` and `score_phase_native_audio_target_replay_oracle.py` import successfully.

## Retirement Classification

Created:

- `docs/architecture/PROJECT_RETIREMENT_CANDIDATES.md`

Original classification snapshot:

| Category | Count |
| --- | ---: |
| canonical | 101 |
| active research | 70 |
| compatibility wrapper | 12 |
| superseded-review-needed | 28 |
| retire-candidate-watchlist | 24 |

Important caveat: the retirement document references every classified script, so future usage-map `documented_reference` counts are inflated by design. For deletion triage, use `PROJECT_RETIREMENT_CANDIDATES.md` rather than raw `documented_reference` counts.

## Measured Effects

Helper definitions after this pass:

| Helper | Remaining definitions |
| --- | ---: |
| `_as_float` | 39 |
| `_mean` | 28 |
| `_median` | 6 |
| `_parse_csv_floats` | 3 |
| `_case_group` | 5 |
| `_route_key_from_dict` | 0 |
| `_route_dict` | 0 |
| `_route_id` | 0 |
| `_route_from_id` | 0 |
| `_compare_waveforms` | 0 |
| `_loop_reentry_metrics` | 0 |

`_route_*` and waveform/loop helpers now live under shared modules instead of per-runner local definitions.

## Verification

Passed:

```powershell
python -m py_compile runtimes\circleworld_proto\common_io.py runtimes\circleworld_proto\experiment_reports.py runtimes\circleworld_proto\phase_audio_metrics.py runtimes\circleworld_proto\profile_registry.py runtimes\circleworld_proto\seed_substrate.py runtimes\circleworld_proto\benchmark_audio_continuation.py runtimes\circleworld_proto\run_audio_circle_delta_probe.py runtimes\circleworld_proto\run_audio_delta_mechanism_probe.py runtimes\circleworld_proto\run_audio_delta_objective_scout.py runtimes\circleworld_proto\run_audio_phase_influence_ablation.py runtimes\circleworld_proto\run_phase_native_audio_selected_route.py runtimes\circleworld_proto\train_phase_native_audio_route_policy.py runtimes\circleworld_proto\train_phase_native_audio_family_route_policy.py runtimes\circleworld_proto\train_phase_native_audio_objective_route_policy.py runtimes\circleworld_proto\score_phase_native_audio_reentry_metric_audit.py runtimes\circleworld_proto\score_phase_native_audio_target_replay_oracle.py
```

Passed:

```powershell
python -c "import sys; sys.path.insert(0, r'D:\RAFA\runtimes\circleworld_proto'); import score_phase_native_audio_reentry_metric_audit, score_phase_native_audio_target_replay_oracle; print('imports ok')"
```

Passed:

```powershell
python -m pytest tests\test_registry_integrity.py -q
# 7 passed
```

Passed:

```powershell
python runtimes\circleworld_proto\audit_circleworld_dag.py
# status: ok
```

Regenerated:

```powershell
python tools\audit_project_scripts.py
python tools\audit_project_usage.py
```

## Current Cartography Snapshot

After adding four shared infrastructure modules and the retirement artifact:

- Source-like scripts: `239`
- Python scripts: `236`
- Circleworld scripts: `129`
- Total nonblank LOC: `91,671`
- Python parse errors: `0`
- Manifested/registered scripts: `47`
- Artifact files indexed: `54,243`
- Artifact groups indexed: `979`

The LOC count rose because this pass added scaffolding and the retirement classifier artifact references the whole script surface. This was an infrastructure consolidation pass, not a deletion pass.

## Next Cleanup Cuts

Recommended next pass:

1. Migrate `score_phase_native_audio_reentry_metric_audit.py`, `score_phase_native_audio_target_replay_oracle.py`, and `benchmark_audio_continuity.py` directly onto `phase_audio_metrics.py` public helpers.
2. Migrate seed-heavy assays to `seed_substrate.py`, starting with `test_unit_phasor_contract.py`, `test_phase_gauge_invariance.py`, and `run_causal_operator_selector_probe.py`.
3. Migrate 3 to 5 report scripts to `experiment_reports.py` write/table helpers.
4. Review `PROJECT_RETIREMENT_CANDIDATES.md` watchlist manually before any deletion.
5. Only after compatibility wrappers and provenance checks, begin actual script retirement.
