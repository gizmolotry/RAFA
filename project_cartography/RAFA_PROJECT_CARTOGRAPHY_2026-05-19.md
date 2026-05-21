# RAFA Project Cartography Pass - 2026-05-19

## Purpose

This pass steps back from model changes and maps the codebase itself.

The immediate goal is not deletion. The goal is to make script identity,
runtime ownership, and redundancy visible enough that future cleanup can be
safe instead of vibes-based.

## New Scaffolding

- `tools/audit_project_scripts.py`
- `docs/architecture/PROJECT_SCRIPT_INVENTORY.json`
- `docs/architecture/PROJECT_SCRIPT_INVENTORY.md`
- `docs/architecture/PROJECT_REDUNDANCY_HOTSPOTS.md`
- `docs/architecture/PROJECT_SCAFFOLDING_GUIDE.md`
- `tools/audit_project_usage.py`
- `docs/architecture/PROJECT_USAGE_MAP.json`
- `docs/architecture/PROJECT_USAGE_MAP.md`
- `docs/architecture/PROJECT_ARTIFACT_INVENTORY.md`
- `docs/reports/RAFA_PROJECT_USAGE_AUDIT_2026-05-19.md`

The audit script scans source-like files visible to git, including untracked
files, while honoring `.gitignore`. That means generated outputs,
checkpoints, archived root dumps, and local datasets stay out of the count.

## Inventory Result

Current source-like script surface:

| Metric | Value |
| --- | ---: |
| Scripts | 234 |
| Python scripts | 231 |
| PowerShell launchers | 3 |
| Total nonblank LOC | 90,975 |
| Tracked scripts | 114 |
| Untracked scripts | 120 |
| Python parse errors | 0 |
| Scripts with CLI main guard | 198 |
| Manifested or registered scripts | 47 |
| Documented or reported scripts | 143 |

## Lane Distribution

| Lane | Scripts |
| --- | ---: |
| `circleworld_proto` | 124 |
| `tools` | 49 |
| `research_track` | 19 |
| `root_legacy` | 14 |
| `core_shared` | 10 |
| `positive_replacement` | 5 |
| `inference_package` | 4 |
| `subtractive_fork` | 3 |
| `parent_graduation` | 1 |
| `local_ablations` | 1 |
| `diffusion_parent_v3` | 1 |
| `stage4_blackwell_14` | 1 |
| `stage4_blackwell_16` | 1 |
| `repo_tests` | 1 |

Interpretation: the redundancy risk is overwhelmingly a Circleworld research
lane problem, not a Graduation runtime problem.

## Largest Files

| File | Nonblank LOC | Interpretation |
| --- | ---: | --- |
| `runtimes/circleworld_proto/assemble_rafa_claim_evidence.py` | 6,420 | Report assembly has become a monolith. Split only after preserving output schema. |
| `runtimes/circleworld_proto/train_circleworld_real_anchor.py` | 3,632 | Canonical trainer; do not casually refactor without regression runs. |
| `runtimes/circleworld_proto/test_nested_commitment.py` | 3,627 | Canonical ontology assay; likely needs helper extraction, not semantic rewrite. |
| `lineages/04_positive_replacement/circleworld.py` | 3,140 | Core runtime module; refactor only with phasor/gauge tests. |
| `runtimes/circleworld_proto/run_childworld_experiment.py` | 3,112 | Experiment-profile sprawl; candidate for profile JSON separation. |

## Redundancy Evidence

Top repeated helper names:

| Helper | Count |
| --- | ---: |
| `_as_float` | 49 |
| `_write_markdown` | 47 |
| `_mean` | 39 |
| `_fmt` | 31 |
| `_markdown` | 23 |
| `_write_json` | 21 |
| `_load_json` | 19 |
| `_utc_timestamp` | 16 |
| `_json_load` | 14 |
| `_aggregate` | 12 |
| `_median` | 12 |
| `_safe_device` | 11 |

Top exact duplicate function bodies:

| Function | Duplicate bodies |
| --- | ---: |
| `_as_float` | 22 |
| `_safe_device` | 10 |
| `_mean` | 6 |
| `_median` | 5 |
| `pearson_correlation_loss` | 5 |

Interpretation: this confirms the user's suspicion. There are likely thousands
of lines of redundant script skeleton and helper logic. The redundancy is mostly
boring IO/report/metric glue, which is good news: it can be extracted without
rewriting the research claims.

## First Extraction Targets

1. `runtimes/circleworld_proto/common_io.py`
   - JSON load/write
   - Markdown writing
   - timestamp/run-directory helpers
   - scalar coercion and formatting

2. `runtimes/circleworld_proto/phase_audio_metrics.py`
   - correlation
   - MSE
   - loop/reentry metrics
   - target-normalized replay summaries

3. `runtimes/circleworld_proto/experiment_reports.py`
   - report front matter
   - Markdown tables
   - compare JSON scaffolds
   - status/verdict helpers

4. `runtimes/circleworld_proto/childworld_metrics.py`
   - survival/carry/writeback aggregation
   - divergence summaries
   - parent/sibling/collapse status helpers

5. `runtimes/circleworld_proto/profile_registry.py`
   - route policy/profile identifiers
   - frozen objective names
   - source/target lockbox naming

## Cleanup Policy

No scripts were deleted or renamed in this pass.

Before deleting anything:

1. Check whether the script is in a runtime manifest or registry.
2. Check whether reports or ledgers reference it.
3. Preserve output schema with a compatibility wrapper if needed.
4. Run registry/DAG/contract tests.
5. Mark the old script as a retirement candidate before removing it.

## Project Interpretation

The project is not structurally broken. It is research-saturated.

The healthy parts:

- Runtime/lineage/registry separation exists.
- Circleworld has many reports and assays rather than silent undocumented code.
- Python syntax parses cleanly across the visible script surface.
- Canonical Graduation runtime paths remain tiny relative to Circleworld and are not the redundancy center.

The risky parts:

- Circleworld has enough one-off runners that names no longer communicate state.
- Helper duplication is severe.
- Several high-value assays are monolithic and hard to reason about.
- Active unregistered research scripts can become invisible assumptions if not linked to reports or manifests.

## Next Recommended Move

Do a no-behavior-change extraction pass:

1. Create `runtimes/circleworld_proto/common_io.py`.
2. Replace the easiest exact duplicate helpers in 3 to 5 low-risk report/scorer scripts.
3. Rerun the affected scripts or their contract tests.
4. Update the inventory and confirm helper counts fall without changing artifacts.

That is the right first cleanup bite: low conceptual risk, high codebase clarity.
