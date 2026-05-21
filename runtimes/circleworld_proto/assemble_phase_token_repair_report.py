import argparse
import json
from collections import OrderedDict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SUMMARY_JSON_NAME = "phase_token_repair_report_summary.json"
REPORT_MD_NAME = "PHASE_TOKEN_REPAIR_REPORT.md"


DIRECT_CANDIDATES = {
    "phase_summary": [
        "phase_token_repair_summary.json",
        "summary.json",
    ],
    "train": [
        "train_summary.json",
        "train/train_summary.json",
        "training/train_summary.json",
    ],
    "heldout": [
        "heldout_summary.json",
        "heldout_eval_phasekeys/heldout_summary.json",
        "heldout_eval/heldout_summary.json",
    ],
    "nested": [
        "nested_commitment_report.json",
        "nested_commitment_phasekeys/nested_commitment_report.json",
        "nested_commitment/nested_commitment_report.json",
    ],
    "benchmark": [
        "benchmark_summary.json",
        "benchmark/benchmark_summary.json",
        "benchmark_expanded/benchmark_summary.json",
    ],
    "continuity": [
        "continuity_circleworld.json",
        "benchmark/continuity_circleworld.json",
        "benchmark_expanded/continuity_circleworld.json",
    ],
}

RECURSIVE_PATTERNS = {
    "phase_summary": ["*phase*repair*summary.json"],
    "train": ["train_summary.json"],
    "heldout": ["heldout_summary.json"],
    "nested": ["nested_commitment_report.json"],
    "benchmark": ["benchmark_summary.json"],
    "continuity": ["continuity_circleworld.json"],
}

ARTIFACT_KEYS = {
    "train": ["train_summary"],
    "heldout": ["heldout_phase_summary", "heldout_summary", "heldout_eval"],
    "nested": ["nested_phase_report", "nested_report", "nested_commitment_report"],
    "benchmark": ["benchmark_summary", "benchmark"],
    "continuity": ["continuity_summary", "continuity_circleworld"],
}


MetricSpec = tuple[str, tuple[str, ...]]


TRAIN_METRICS: list[MetricSpec] = [
    ("mean_score", ("mean_score", "best_score", "selected_score", "baseline_train.mean_score")),
    ("mean_corr", ("mean_corr", "baseline_train.mean_corr")),
    ("mean_mae", ("mean_mae", "baseline_train.mean_mae")),
    ("mean_real_branch_fraction", ("mean_real_branch_fraction", "baseline_train.mean_real_branch_fraction")),
    ("phase_only_real_branch_fraction", ("phase_only_real_branch_fraction", "baseline_train.phase_only_real_branch_fraction")),
    ("mean_phase_only_branch_distinctness", ("mean_phase_only_branch_distinctness", "baseline_train.mean_phase_only_branch_distinctness")),
    ("mean_meso_branch_effect", ("mean_meso_branch_effect", "baseline_train.mean_meso_branch_effect")),
    ("mean_live_child_fraction", ("mean_live_child_fraction", "baseline_train.mean_live_child_fraction")),
    ("mean_child_writeback_mass", ("mean_child_writeback_mass", "baseline_train.mean_child_writeback_mass")),
    ("mean_child_parent_divergence", ("mean_child_parent_divergence", "baseline_train.mean_child_parent_divergence")),
    ("mean_num_law_families", ("mean_num_law_families", "baseline_train.mean_num_law_families")),
    ("mean_law_family_entropy", ("mean_law_family_entropy", "baseline_train.mean_law_family_entropy")),
    ("mean_num_relational_signature_families", ("mean_num_relational_signature_families", "baseline_train.mean_num_relational_signature_families")),
    ("mean_relational_signature_confidence", ("mean_relational_signature_confidence", "baseline_train.mean_relational_signature_confidence")),
    ("aggregate_num_relational_signature_families", ("aggregate_num_relational_signature_families", "baseline_train.aggregate_num_relational_signature_families")),
]

HELDOUT_METRICS: list[MetricSpec] = [
    ("num_samples", ("num_samples",)),
    ("mean_real_branch_fraction", ("mean_real_branch_fraction",)),
    ("mean_parent_real_branch_fraction", ("mean_parent_real_branch_fraction",)),
    ("mean_child_real_branch_fraction", ("mean_child_real_branch_fraction",)),
    ("phase_only_real_branch_fraction", ("phase_only_real_branch_fraction",)),
    ("phase_only_excess_branch_fraction", ("phase_only_excess_branch_fraction",)),
    ("decorative_slot2_low_phase_fraction", ("decorative_slot2_low_phase_fraction",)),
    ("mean_phase_only_branch_surface", ("mean_phase_only_branch_surface",)),
    ("mean_phase_only_branch_phase_wall", ("mean_phase_only_branch_phase_wall",)),
    ("mean_phase_only_branch_distinctness", ("mean_phase_only_branch_distinctness",)),
    ("mean_meso_branch_effect", ("mean_meso_branch_effect",)),
    ("mean_live_child_fraction", ("mean_live_child_fraction",)),
    ("mean_child_writeback_mass", ("mean_child_writeback_mass",)),
    ("mean_child_parent_divergence", ("mean_child_parent_divergence",)),
    ("mean_child_sibling_divergence", ("mean_child_sibling_divergence",)),
    ("mean_num_law_families", ("mean_num_law_families",)),
    ("mean_law_family_entropy", ("mean_law_family_entropy",)),
    ("mean_num_relational_signature_families", ("mean_num_relational_signature_families",)),
    ("mean_relational_signature_confidence", ("mean_relational_signature_confidence",)),
    ("mean_relational_branch_mass", ("mean_relational_branch_mass",)),
]

NAKED_HELDOUT_METRICS: list[MetricSpec] = [
    ("count", ("count",)),
    ("mean_real_branch_fraction", ("mean_real_branch_fraction",)),
    ("mean_parent_real_branch_fraction", ("mean_parent_real_branch_fraction",)),
    ("mean_child_real_branch_fraction", ("mean_child_real_branch_fraction",)),
    ("phase_only_real_branch_fraction", ("phase_only_real_branch_fraction",)),
    ("phase_only_excess_branch_fraction", ("phase_only_excess_branch_fraction",)),
    ("decorative_slot2_low_phase_fraction", ("decorative_slot2_low_phase_fraction",)),
    ("mean_phase_only_branch_phase_wall", ("mean_phase_only_branch_phase_wall",)),
    ("mean_phase_only_branch_distinctness", ("mean_phase_only_branch_distinctness",)),
    ("mean_meso_branch_effect", ("mean_meso_branch_effect",)),
    ("mean_live_child_fraction", ("mean_live_child_fraction",)),
    ("mean_child_writeback_mass", ("mean_child_writeback_mass",)),
    ("mean_child_parent_divergence", ("mean_child_parent_divergence",)),
    ("mean_num_law_families", ("mean_num_law_families",)),
    ("mean_law_family_entropy", ("mean_law_family_entropy",)),
    ("mean_num_relational_signature_families", ("mean_num_relational_signature_families",)),
    ("mean_relational_signature_confidence", ("mean_relational_signature_confidence",)),
]

NESTED_METRICS: list[MetricSpec] = [
    ("num_cases", ("num_cases",)),
    ("num_selected_fork_has_live_child_cases", ("num_selected_fork_has_live_child_cases",)),
    ("mean_selected_fork_has_live_child", ("mean_selected_fork_has_live_child",)),
    ("mean_child_response_score", ("mean_child_response_score",)),
    ("mean_child_active_fraction", ("mean_child_active_fraction",)),
    ("mean_child_meso_response", ("mean_child_meso_response",)),
    ("mean_nested_sibling_fraction", ("mean_nested_sibling_fraction",)),
    ("mean_readout_sibling_response", ("mean_readout_sibling_response",)),
    ("mean_branch_identity_qualified_carry", ("mean_branch_identity_qualified_carry",)),
]

BENCHMARK_METRICS: list[MetricSpec] = [
    ("num_cases", ("num_cases", "case_count")),
    ("mean_mse", ("mean_mse",)),
    ("mean_mae", ("mean_mae",)),
    ("mean_corr", ("mean_corr",)),
    ("all_circleworld_bitwise_stable", ("all_circleworld_bitwise_stable",)),
    ("mean_nonlocal_chunk_repeat", ("mean_nonlocal_chunk_repeat", "circleworld.mean_nonlocal_chunk_repeat")),
    ("mean_first_chunk_reentry", ("mean_first_chunk_reentry", "circleworld.mean_first_chunk_reentry")),
    ("mean_adjacent_chunk_similarity", ("mean_adjacent_chunk_similarity", "circleworld.mean_adjacent_chunk_similarity")),
    ("recurrence_severity_band", ("recurrence_severity_band", "circleworld.recurrence_severity_band")),
]


def _load_json(path: Path, warnings: list[str]) -> Any | None:
    try:
        with path.open("r", encoding="utf-8-sig") as fh:
            return json.load(fh)
    except FileNotFoundError:
        warnings.append(f"missing file: {path}")
    except json.JSONDecodeError as exc:
        warnings.append(f"invalid JSON in {path}: {exc}")
    except OSError as exc:
        warnings.append(f"could not read {path}: {exc}")
    return None


def _as_mapping(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _get_path(data: Any, dotted: str) -> Any:
    current = data
    for part in dotted.split("."):
        if not isinstance(current, dict) or part not in current:
            return None
        current = current[part]
    return current


def _first_value(data: Any, paths: tuple[str, ...]) -> Any:
    for path in paths:
        value = _get_path(data, path)
        if value is not None:
            return value
    return None


def _extract_metrics(data: Any, specs: list[MetricSpec]) -> OrderedDict[str, Any]:
    payload = _as_mapping(data)
    row: OrderedDict[str, Any] = OrderedDict()
    for out_key, paths in specs:
        row[out_key] = _normalize_scalar(_first_value(payload, paths))
    return row


def _normalize_scalar(value: Any) -> Any:
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return round(value, 6)
    return value


def _find_direct(run_dir: Path, section: str) -> Path | None:
    for rel in DIRECT_CANDIDATES.get(section, []):
        path = run_dir / rel
        if path.exists() and path.is_file():
            return path
    return None


def _find_recursive(run_dir: Path, section: str) -> Path | None:
    patterns = RECURSIVE_PATTERNS.get(section, [])
    matches: list[Path] = []
    for pattern in patterns:
        matches.extend(path for path in run_dir.rglob(pattern) if path.is_file())
    if not matches:
        return None
    return sorted(matches, key=lambda path: (len(path.relative_to(run_dir).parts), str(path).lower()))[0]


def _path_from_artifacts(run_dir: Path, phase_summary: dict[str, Any], section: str) -> Path | None:
    artifacts = _as_mapping(phase_summary.get("artifacts"))
    for key in ARTIFACT_KEYS.get(section, []):
        value = artifacts.get(key)
        if not isinstance(value, str) or not value:
            continue
        path = Path(value)
        if not path.is_absolute():
            path = run_dir / path
        if path.exists() and path.is_file():
            return path
    return None


def _load_section(run_dir: Path, section: str, phase_summary: dict[str, Any], warnings: list[str]) -> tuple[Any | None, str | None]:
    path = _find_direct(run_dir, section)
    if path is None:
        path = _path_from_artifacts(run_dir, phase_summary, section)
    if path is None:
        path = _find_recursive(run_dir, section)
    if path is not None:
        return _load_json(path, warnings), str(path)

    fallback = phase_summary.get(section)
    if fallback is not None:
        return fallback, "phase_token_repair_summary.json"
    warnings.append(f"missing {section} summary under {run_dir}")
    return None, None


def _extract_naked_heldout(heldout: Any) -> OrderedDict[str, Any]:
    payload = _as_mapping(heldout)
    naked = payload.get("naked_rafa")
    if not isinstance(naked, dict):
        naked = _get_path(payload, "by_source.naked_rafa")
    return _extract_metrics(naked, NAKED_HELDOUT_METRICS)


def _selection_highlights(phase_summary: dict[str, Any], train: Any) -> OrderedDict[str, Any]:
    selection = _as_mapping(phase_summary.get("selection"))
    train_payload = _as_mapping(train)
    fallback = _as_mapping(selection.get("fallback_diagnostics"))
    if not fallback:
        fallback = _as_mapping(train_payload.get("selection_fallback_diagnostics"))
    availability = _as_mapping(fallback.get("availability"))
    failure_counts = _as_mapping(fallback.get("failure_counts"))

    row: OrderedDict[str, Any] = OrderedDict()
    row["selection_source"] = (
        phase_summary.get("selection_source")
        or selection.get("selection_source")
        or selection.get("source")
        or fallback.get("selection_source")
        or train_payload.get("selection_source")
    )
    row["used_init_config_fallback"] = fallback.get("used_init_config_fallback")
    row["candidate_unavailable_reasons"] = fallback.get("candidate_unavailable_reasons")
    row["num_agreement_candidates"] = availability.get("num_agreement_candidates")
    row["num_heldout_gate_pass"] = availability.get("num_heldout_gate_pass")
    row["num_nested_gate_pass"] = availability.get("num_nested_gate_pass")
    row["failure_counts"] = failure_counts if failure_counts else None
    row["top_candidate_failures"] = fallback.get("top_candidate_failures")
    return row


def _top_failure_counts(failure_counts: Any, limit: int = 5) -> list[str]:
    if not isinstance(failure_counts, dict):
        return []
    flat: list[tuple[str, int]] = []
    for group_name, group_counts in failure_counts.items():
        if not isinstance(group_counts, dict):
            continue
        for name, count in group_counts.items():
            if isinstance(count, (int, float)):
                flat.append((f"{group_name}.{name}", int(count)))
    flat.sort(key=lambda item: (-item[1], item[0]))
    return [f"{name}={count}" for name, count in flat[:limit]]


def _summarize_run(run_dir: Path) -> OrderedDict[str, Any]:
    warnings: list[str] = []
    phase_path = _find_direct(run_dir, "phase_summary") or _find_recursive(run_dir, "phase_summary")
    phase_summary = _as_mapping(_load_json(phase_path, warnings) if phase_path else None)

    train, train_path = _load_section(run_dir, "train", phase_summary, warnings)
    heldout, heldout_path = _load_section(run_dir, "heldout", phase_summary, warnings)
    nested, nested_path = _load_section(run_dir, "nested", phase_summary, warnings)
    benchmark, benchmark_path = _load_section(run_dir, "benchmark", phase_summary, warnings)
    continuity, continuity_path = _load_section(run_dir, "continuity", phase_summary, [])

    if isinstance(continuity, dict) and isinstance(benchmark, dict):
        benchmark = dict(benchmark)
        benchmark.setdefault("circleworld", continuity)

    train_payload = _as_mapping(train)
    heldout_payload = _as_mapping(heldout)
    nested_payload = _as_mapping(nested)
    benchmark_payload = _as_mapping(benchmark)

    run: OrderedDict[str, Any] = OrderedDict()
    run["name"] = str(phase_summary.get("cycle") or phase_summary.get("name") or run_dir.name)
    run["run_dir"] = str(run_dir)
    run["status"] = phase_summary.get("status") or train_payload.get("status")
    run["runtime"] = phase_summary.get("runtime") or train_payload.get("runtime")
    run["generated_from"] = OrderedDict(
        phase_summary=str(phase_path) if phase_path else None,
        train=train_path,
        heldout=heldout_path,
        nested=nested_path,
        benchmark=benchmark_path,
        continuity=continuity_path,
    )
    run["train"] = _extract_metrics(train_payload, TRAIN_METRICS)
    run["train"]["iterations"] = _normalize_scalar(train_payload.get("iterations"))
    run["train"]["population"] = _normalize_scalar(train_payload.get("population"))
    run["train"]["seed"] = _normalize_scalar(train_payload.get("seed"))
    run["train"]["checkpoint"] = train_payload.get("checkpoint") or train_payload.get("config_path") or phase_summary.get("checkpoint")
    run["heldout"] = _extract_metrics(heldout_payload, HELDOUT_METRICS)
    run["heldout"]["naked_rafa"] = _extract_naked_heldout(heldout_payload)
    run["nested"] = _extract_metrics(nested_payload, NESTED_METRICS)
    run["nested"]["overall_read"] = nested_payload.get("overall_read")
    run["nested"]["verdict_counts"] = nested_payload.get("verdict_counts")
    run["nested"]["branch_family_counts"] = nested_payload.get("branch_family_counts")
    run["benchmark"] = _extract_metrics(benchmark_payload, BENCHMARK_METRICS)
    if run["benchmark"]["num_cases"] is None:
        cases = benchmark_payload.get("cases")
        rows = benchmark_payload.get("rows")
        if isinstance(cases, dict):
            run["benchmark"]["num_cases"] = len(cases)
        elif isinstance(rows, list):
            run["benchmark"]["num_cases"] = len(rows)
    run["selection"] = _selection_highlights(phase_summary, train_payload)
    run["read"] = phase_summary.get("read")
    run["recommendation"] = phase_summary.get("recommendation")
    run["changed_focus"] = phase_summary.get("changed_focus")
    run["warnings"] = warnings
    return run


def _fmt(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def _md_escape(value: Any) -> str:
    text = _fmt(value)
    return text.replace("|", "\\|").replace("\n", " ")


def _bullet(label: str, value: Any) -> str:
    return f"- {label}: `{_md_escape(value)}`"


def _make_markdown(title: str, runs: list[OrderedDict[str, Any]], generated_utc: str) -> str:
    lines: list[str] = [
        f"# {title}",
        "",
        f"- generated_utc: `{generated_utc}`",
        f"- run_count: `{len(runs)}`",
        "",
        "## Run Matrix",
        "",
        "| run | status | train branch | heldout branch | heldout phase-only | naked phase-only | nested live | nested active | nested meso | bench corr | bench mae | selection | promote |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|",
    ]
    for run in runs:
        heldout = run["heldout"]
        nested = run["nested"]
        benchmark = run["benchmark"]
        recommendation = _as_mapping(run.get("recommendation"))
        lines.append(
            "| {name} | {status} | {train_branch} | {heldout_branch} | {phase_branch} | {naked_phase_branch} | {nested_live} | {nested_active} | {nested_meso} | {bench_corr} | {bench_mae} | {selection} | {promote} |".format(
                name=_md_escape(run.get("name")),
                status=_md_escape(run.get("status")),
                train_branch=_md_escape(run["train"].get("mean_real_branch_fraction")),
                heldout_branch=_md_escape(heldout.get("mean_real_branch_fraction")),
                phase_branch=_md_escape(heldout.get("phase_only_real_branch_fraction")),
                naked_phase_branch=_md_escape(heldout.get("naked_rafa", {}).get("phase_only_real_branch_fraction")),
                nested_live=_md_escape(nested.get("num_selected_fork_has_live_child_cases")),
                nested_active=_md_escape(nested.get("mean_child_active_fraction")),
                nested_meso=_md_escape(nested.get("mean_child_meso_response")),
                bench_corr=_md_escape(benchmark.get("mean_corr")),
                bench_mae=_md_escape(benchmark.get("mean_mae")),
                selection=_md_escape(run["selection"].get("selection_source")),
                promote=_md_escape(recommendation.get("promote_checkpoint")),
            )
        )

    for run in runs:
        lines.extend(["", f"## {run['name']}", "", _bullet("run_dir", run.get("run_dir"))])
        if run.get("changed_focus"):
            lines.append("- changed_focus: " + "; ".join(_md_escape(item) for item in run["changed_focus"]))

        lines.extend(["", "### Train", ""])
        for key in ("mean_score", "mean_corr", "mean_mae", "mean_real_branch_fraction", "phase_only_real_branch_fraction", "mean_child_writeback_mass", "mean_child_parent_divergence", "mean_num_relational_signature_families", "mean_relational_signature_confidence", "aggregate_num_relational_signature_families"):
            lines.append(_bullet(key, run["train"].get(key)))

        lines.extend(["", "### Heldout", ""])
        for key in ("mean_real_branch_fraction", "mean_parent_real_branch_fraction", "mean_child_real_branch_fraction", "phase_only_real_branch_fraction", "phase_only_excess_branch_fraction", "decorative_slot2_low_phase_fraction", "mean_phase_only_branch_phase_wall", "mean_phase_only_branch_distinctness", "mean_child_writeback_mass", "mean_child_parent_divergence", "mean_num_relational_signature_families", "mean_relational_signature_confidence"):
            lines.append(_bullet(key, run["heldout"].get(key)))
        naked = run["heldout"].get("naked_rafa", {})
        lines.append(_bullet("naked_rafa.phase_only_real_branch_fraction", naked.get("phase_only_real_branch_fraction")))
        lines.append(_bullet("naked_rafa.phase_only_excess_branch_fraction", naked.get("phase_only_excess_branch_fraction")))
        lines.append(_bullet("naked_rafa.mean_real_branch_fraction", naked.get("mean_real_branch_fraction")))

        lines.extend(["", "### Nested", ""])
        for key in ("overall_read", "num_cases", "num_selected_fork_has_live_child_cases", "mean_selected_fork_has_live_child", "mean_child_response_score", "mean_child_active_fraction", "mean_child_meso_response", "mean_nested_sibling_fraction", "mean_branch_identity_qualified_carry"):
            lines.append(_bullet(key, run["nested"].get(key)))
        if run["nested"].get("branch_family_counts"):
            lines.append(_bullet("branch_family_counts", run["nested"].get("branch_family_counts")))

        lines.extend(["", "### Benchmark", ""])
        for key in ("num_cases", "mean_corr", "mean_mae", "mean_mse", "all_circleworld_bitwise_stable", "mean_nonlocal_chunk_repeat", "mean_first_chunk_reentry", "mean_adjacent_chunk_similarity", "recurrence_severity_band"):
            lines.append(_bullet(key, run["benchmark"].get(key)))

        selection = run["selection"]
        lines.extend(["", "### Selection", ""])
        for key in ("selection_source", "used_init_config_fallback", "candidate_unavailable_reasons", "num_agreement_candidates", "num_heldout_gate_pass", "num_nested_gate_pass"):
            lines.append(_bullet(key, selection.get(key)))
        top_failures = _top_failure_counts(selection.get("failure_counts"))
        if top_failures:
            lines.append(_bullet("top_failure_counts", ", ".join(top_failures)))

        read = _as_mapping(run.get("read"))
        if read:
            lines.extend(["", "### Read", ""])
            for key, value in read.items():
                lines.append(f"- {key}: {_md_escape(value)}")

        recommendation = _as_mapping(run.get("recommendation"))
        if recommendation:
            lines.extend(["", "### Recommendation", ""])
            for key, value in recommendation.items():
                lines.append(_bullet(key, value))

        if run.get("warnings"):
            lines.extend(["", "### Warnings", ""])
            for warning in run["warnings"]:
                lines.append(f"- {_md_escape(warning)}")

        sources = _as_mapping(run.get("generated_from"))
        lines.extend(["", "### Source Files", ""])
        for key, value in sources.items():
            lines.append(_bullet(key, value))

    lines.append("")
    return "\n".join(lines)


def assemble_report(run_dirs: list[Path], out_dir: Path, title: str) -> OrderedDict[str, Any]:
    generated_utc = datetime.now(timezone.utc).isoformat()
    runs = [_summarize_run(run_dir.resolve()) for run_dir in run_dirs]
    summary: OrderedDict[str, Any] = OrderedDict(
        title=title,
        generated_utc=generated_utc,
        run_count=len(runs),
        runs=runs,
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / SUMMARY_JSON_NAME
    md_path = out_dir / REPORT_MD_NAME
    summary["outputs"] = OrderedDict(json=str(json_path), markdown=str(md_path))
    with json_path.open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
        fh.write("\n")
    md_path.write_text(_make_markdown(title, runs, generated_utc), encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Assemble phase-token repair run summaries into JSON and Markdown reports.")
    parser.add_argument("--run-dir", action="append", required=True, type=Path, help="Phase-token repair run directory. Repeat for multiple runs.")
    parser.add_argument("--out-dir", required=True, type=Path, help="Directory where report JSON and Markdown should be written.")
    parser.add_argument("--title", default="Circleworld Phase-Token Repair Report", help="Markdown report title.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = assemble_report(args.run_dir, args.out_dir, args.title)
    outputs = summary["outputs"]
    print(f"wrote {outputs['json']}")
    print(f"wrote {outputs['markdown']}")


if __name__ == "__main__":
    main()
