# RAFA Codebase Cleanup Pass - 2026-05-20

## Purpose

This pass starts the codebase reduction effort without deleting or renaming research scripts.

The cleanup policy is no-behavior-change first: extract repeated helper logic, rerun affected scripts, regenerate cartography, and only then decide which scripts are safe retirement candidates.

## Change Summary

Added shared Circleworld helper module:

- `D:\RAFA\runtimes\circleworld_proto\common_io.py`

The module centralizes low-risk primitives that were repeatedly copied across Circleworld report/scorer utilities:

- JSON load/write helpers
- text writing with parent directory creation
- scalar coercion via `as_float`
- `mean`, `median`, and `win_fraction`
- Markdown numeric formatting helpers
- CSV float parsing
- required/gain CSV parsing
- safe device selection

Refactored six low-risk internal phase-law utilities to import shared helpers:

- `D:\RAFA\runtimes\circleworld_proto\analyze_internal_phase_law_family_masks.py`
- `D:\RAFA\runtimes\circleworld_proto\analyze_internal_phase_law_support_router_oracle.py`
- `D:\RAFA\runtimes\circleworld_proto\analyze_internal_phase_law_target_flips.py`
- `D:\RAFA\runtimes\circleworld_proto\build_internal_phase_law_failure_atlas.py`
- `D:\RAFA\runtimes\circleworld_proto\compare_internal_phase_law_variants.py`
- `D:\RAFA\runtimes\circleworld_proto\score_internal_phase_law_objective.py`

Refactored four phase-native audio probe/scout utilities to import shared scalar/stat/gain parsing helpers:

- `D:\RAFA\runtimes\circleworld_proto\run_audio_circle_delta_probe.py`
- `D:\RAFA\runtimes\circleworld_proto\run_audio_delta_mechanism_probe.py`
- `D:\RAFA\runtimes\circleworld_proto\run_audio_delta_objective_scout.py`
- `D:\RAFA\runtimes\circleworld_proto\run_audio_phase_influence_ablation.py`

No scripts were deleted, renamed, or moved.

## Verification

Syntax and direct CLI import checks passed:

```powershell
python -m py_compile runtimes\circleworld_proto\common_io.py runtimes\circleworld_proto\analyze_internal_phase_law_family_masks.py runtimes\circleworld_proto\analyze_internal_phase_law_support_router_oracle.py runtimes\circleworld_proto\compare_internal_phase_law_variants.py runtimes\circleworld_proto\build_internal_phase_law_failure_atlas.py runtimes\circleworld_proto\score_internal_phase_law_objective.py runtimes\circleworld_proto\analyze_internal_phase_law_target_flips.py
```

Additional phase-audio CLI load checks passed:

```powershell
python -m py_compile runtimes\circleworld_proto\common_io.py runtimes\circleworld_proto\run_audio_circle_delta_probe.py runtimes\circleworld_proto\run_audio_delta_mechanism_probe.py runtimes\circleworld_proto\run_audio_delta_objective_scout.py runtimes\circleworld_proto\run_audio_phase_influence_ablation.py
python runtimes\circleworld_proto\run_audio_circle_delta_probe.py --help
python runtimes\circleworld_proto\run_audio_delta_mechanism_probe.py --help
python runtimes\circleworld_proto\run_audio_delta_objective_scout.py --help
python runtimes\circleworld_proto\run_audio_phase_influence_ablation.py --help
```

Shared gain parsing contract check passed:

```powershell
@'
from common_io import parse_gain_csv
assert parse_gain_csv('1,2') == [0.0, 1.0, 2.0]
assert parse_gain_csv('0,2') == [0.0, 2.0]
'@ | python -
```

A deterministic synthetic smoke chain was run under:

`D:\RAFA\outputs\circleworld_proto\codebase_cleanup_2026-05-20_common_io_smoke\`

The smoke chain exercised all six refactored scripts end-to-end:

1. `compare_internal_phase_law_variants.py`
2. `score_internal_phase_law_objective.py`
3. `build_internal_phase_law_failure_atlas.py`
4. `analyze_internal_phase_law_family_masks.py`
5. `analyze_internal_phase_law_support_router_oracle.py`
6. `analyze_internal_phase_law_target_flips.py`

The first smoke attempt caught two stale `json.dumps` imports after helper extraction; those were fixed, and the rerun passed.

Repo-level checks passed:

```powershell
python -m pytest tests\test_registry_integrity.py -q
# 7 passed

python runtimes\circleworld_proto\audit_circleworld_dag.py
# status: ok
```

Cartography was regenerated:

```powershell
python tools\audit_project_scripts.py
python tools\audit_project_usage.py
```

## Measured Reduction

Pre-pass cartography headline:

| Metric | Before |
| --- | ---: |
| Source-like scripts | 234 |
| Python scripts | 231 |
| Total nonblank LOC | 90,975 |
| Circleworld scripts | 124 |
| Untracked scripts | 120 |

Post-pass cartography headline:

| Metric | After |
| --- | ---: |
| Source-like scripts | 235 |
| Python scripts | 232 |
| Total nonblank LOC | 90,931 |
| Circleworld scripts | 125 |
| Untracked scripts | 121 |

Interpretation: adding `common_io.py` increased the script count by one, but removing duplicated local helper bodies still reduced total nonblank LOC by forty-four lines. This is still a first incision, not the main surgery.

Helper duplication changed as follows:

| Helper | Before | After | Delta |
| --- | ---: | ---: | ---: |
| `_as_float` | 49 | 39 | -10 |
| `_mean` | 39 | 28 | -11 |
| `_fmt` | 31 | 25 | -6 |
| `_write_json` | 21 | 19 | -2 |
| `_load_json` | 19 | 17 | -2 |
| `_median` | 12 | 6 | -6 |
| `_load` | 8 | 3 | -5 |
| `_parse_csv_floats` | 7 | 3 | -4 |

The largest remaining helper target is now `_write_markdown` at 47 copies.

## Current Cartography Snapshot

- Source-like scripts: `235`
- Python parse errors: `0`
- Manifested or registered scripts: `47`
- Documented or reported scripts: `147`
- Artifact files indexed: `54,243`
- Artifact groups indexed: `979`
- Artifact bytes indexed: `93.89 GB`
- Cold/unreferenced artifact groups: `543`

## Interpretation

The user's redundancy suspicion is confirmed, but the first cleanup proved something useful: the worst visible duplication is mostly boring infrastructure, not RAFA theory.

That is good news. We can cut thousands of lines over multiple passes by extracting repeated IO, Markdown, metric, and route-profile scaffolding without touching the Circleworld runtime semantics.

The codebase is not broken. It is research-saturated and helper-duplicated.

## Next Cleanup Targets

Recommended next no-behavior-change extraction pass:

1. Extract `experiment_reports.py` for `_write_markdown`, Markdown tables, report front matter, and status blocks.
2. Extract `phase_audio_metrics.py` for repeated correlation/MSE/reentry/loop metric helpers.
3. Extract route/profile constants into `profile_registry.py` or an existing registry if compatible.
4. Target another 8 to 12 low-reference report/scorer scripts, not canonical trainers.
5. Mark genuinely superseded one-off scripts as `retire_candidate` only after reports/ledgers/manifests are checked.

Deletion remains a separate later patch. The next safe work is more extraction, then retirement marking, then deletion.
