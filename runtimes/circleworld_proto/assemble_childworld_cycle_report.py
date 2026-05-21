from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SCHEMA = "circleworld_childworld_cycle_report_v1"
DEFAULT_CYCLE = "continuation_causal_childifs_2026-05-11"
DEFAULT_CYCLE_ROOT = (
    Path(r"D:\RAFA\outputs\circleworld_proto") / DEFAULT_CYCLE
)
DEFAULT_DOCS_DIR = Path(r"D:\RAFA\docs\reports")
DEFAULT_OUTPUT_JSON = DEFAULT_CYCLE_ROOT / "childworld_cycle_report_2026-05-11.json"
DEFAULT_OUTPUT_MD = DEFAULT_DOCS_DIR / "CIRCLEWORLD_CHILDWORLD_CYCLE_REPORT_2026-05-11.md"


ARTIFACTS = {
    "cycle_compare": "continuation_causal_compare.json",
    "mechanism_audit": "childworld_mechanism_audit_combined.json",
    "dataset_v2": "childworld_mechanism_dataset_v2.json",
    "classifier_v1": "childworld_mechanism_classifier_v1.json",
    "static_conversion": "static_child_mode_replace_conversion_combined.json",
    "heldout_summary": "heldout_guarded_selected/heldout_summary.json",
    "benchmark_summary": "benchmark_guarded_selected/benchmark_summary.json",
    "nested_readiness": "nested_causal_selected_v4_readiness/nested_commitment_report.json",
    "conversion_metric_smoke": "conversion_metric_smoke_seed9100_cuda/nested_commitment_report.json",
}


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return payload


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _num(value: Any) -> float | None:
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return None
    return None


def _get(mapping: dict[str, Any], *keys: str) -> Any:
    value: Any = mapping
    for key in keys:
        value = _dict(value).get(key)
    return value


def _fmt(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def _artifact_paths(cycle_root: Path) -> dict[str, Path]:
    return {name: cycle_root / rel for name, rel in ARTIFACTS.items()}


def _load_artifacts(cycle_root: Path) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    paths = _artifact_paths(cycle_root)
    loaded: dict[str, dict[str, Any]] = {}
    status: dict[str, Any] = {}
    for name, path in paths.items():
        exists = path.exists()
        status[name] = {
            "path": str(path),
            "exists": exists,
            "loaded": False,
            "schema": None,
            "error": None,
        }
        if not exists:
            status[name]["error"] = "missing"
            continue
        try:
            payload = _load_json(path)
        except Exception as exc:  # pragma: no cover - defensive CLI reporting.
            status[name]["error"] = str(exc)
            continue
        loaded[name] = payload
        status[name]["loaded"] = True
        status[name]["schema"] = payload.get("schema")
    return loaded, status


def _selected_nested_metrics(compare: dict[str, Any]) -> dict[str, Any]:
    nested = _dict(compare.get("nested"))
    selected = _dict(nested.get("selected"))
    previous = _dict(nested.get("previous"))
    delta = _dict(nested.get("delta"))
    keys = [
        "mean_nested_sibling_fraction",
        "mean_assay_readout_nested_sibling_fraction",
        "mean_assay_continuation_nested_sibling_fraction",
        "mean_assay_mode_replace_nested_sibling_fraction",
        "mean_assay_continuation_max_nested_sibling_readiness",
        "mean_assay_continuation_nested_sibling_readiness",
        "mean_assay_continuation_continuation_write_delta",
        "mean_world_jump_penalty",
        "mean_over_rigid_fraction",
    ]
    return {
        key: {
            "previous": previous.get(key),
            "selected": selected.get(key),
            "delta": delta.get(key),
        }
        for key in keys
    }


def _selected_heldout_metrics(compare: dict[str, Any]) -> dict[str, Any]:
    heldout = _dict(compare.get("heldout"))
    selected = _dict(heldout.get("selected"))
    previous = _dict(heldout.get("previous"))
    delta = _dict(heldout.get("delta"))
    keys = [
        "mean_real_branch_fraction",
        "mean_child_writeback_mass",
        "mean_child_parent_divergence",
        "mean_child_sibling_divergence",
        "mean_meso_branch_effect",
        "mean_child_local_ifs_phase_delta",
        "mean_child_local_ifs_coherence",
        "mean_child_local_ifs_causal_gate",
    ]
    return {
        key: {
            "previous": previous.get(key),
            "selected": selected.get(key),
            "delta": delta.get(key),
        }
        for key in keys
    }


def _benchmark_metrics(compare: dict[str, Any]) -> dict[str, Any]:
    benchmark = _dict(compare.get("benchmark"))
    selected = _dict(benchmark.get("selected"))
    previous = _dict(benchmark.get("previous"))
    delta = _dict(benchmark.get("delta"))
    selected_macro = _dict(benchmark.get("selected_macro"))
    keys = ["mean_corr", "mean_mae", "mean_mse"]
    return {
        "audio_similarity": {
            key: {
                "previous": previous.get(key),
                "selected": selected.get(key),
                "delta": delta.get(key),
            }
            for key in keys
        },
        "continuity": {
            "mean_recurrence_score": selected_macro.get("mean_recurrence_score"),
            "dominant_recurrence_band": selected_macro.get("dominant_recurrence_band"),
            "file_count": selected_macro.get("file_count"),
        },
    }


def _mechanism_evidence(audit: dict[str, Any]) -> dict[str, Any]:
    aggregate = _dict(audit.get("aggregate_decision"))
    suite_rows: list[dict[str, Any]] = []
    for suite in _list(audit.get("suite_decisions")):
        suite_obj = _dict(suite)
        positive = _dict(suite_obj.get("positive_metrics"))
        suite_rows.append(
            {
                "suite_name": suite_obj.get("suite_name"),
                "mechanism_read": suite_obj.get("mechanism_read"),
                "positive_signal": suite_obj.get("positive_signal"),
                "positive_continuation_fraction": positive.get("continuation_fraction"),
                "supported_classes": suite_obj.get("supported_classes", []),
            }
        )
    return {
        "overall_read": aggregate.get("overall_read"),
        "suite_count": aggregate.get("suite_count"),
        "suites_with_positive_signal": aggregate.get("suites_with_positive_signal"),
        "suites_without_positive_signal": aggregate.get("suites_without_positive_signal"),
        "supported_class_counts": aggregate.get("supported_class_counts", {}),
        "suite_rows": suite_rows,
    }


def _mode_replace_result(compare: dict[str, Any], conversion: dict[str, Any], dataset: dict[str, Any]) -> dict[str, Any]:
    selected_mode = _get(
        compare,
        "nested",
        "selected",
        "mean_assay_mode_replace_nested_sibling_fraction",
    )
    previous_mode = _get(
        compare,
        "nested",
        "previous",
        "mean_assay_mode_replace_nested_sibling_fraction",
    )
    conversion_aggregate = _dict(conversion.get("aggregate"))
    suite_results: list[dict[str, Any]] = []
    mode_variant_count = 0
    mode_signal_count = 0
    for suite in _list(conversion.get("suite_reports")):
        suite_obj = _dict(suite)
        focused = _dict(suite_obj.get("focused_variants"))
        variants = [_dict(item) for item in _list(focused.get("mode_replace"))]
        mode_variant_count += len(variants)
        mode_signal_count += sum(1 for item in variants if item.get("has_signal") is True)
        suite_results.append(
            {
                "suite_name": suite_obj.get("suite_name"),
                "classification": suite_obj.get("classification"),
                "mode_replace_supported": _dict(suite_obj.get("signals")).get(
                    "mode_replace_supported"
                ),
                "mode_replace_variants": [
                    {
                        "variant": item.get("variant"),
                        "continuation_fraction": item.get("continuation_fraction"),
                        "mode_replace_fraction": item.get("mode_replace_fraction"),
                        "mode_replace_readiness": item.get("mode_replace_readiness"),
                        "mode_replace_gate": item.get("mode_replace_gate"),
                        "mode_replace_ratio": item.get("mode_replace_ratio"),
                        "read": item.get("read"),
                        "has_signal": item.get("has_signal"),
                    }
                    for item in variants
                ],
            }
        )

    target_counts = _dict(_dict(dataset.get("aggregate")).get("target_counts"))
    mode_target_counts = _dict(target_counts.get("target_mode_replace_supported"))
    status = "not_supported"
    if mode_signal_count > 0 or (_num(selected_mode) or 0.0) > 0.0:
        status = "supported_or_mixed"

    return {
        "status": status,
        "read": (
            "mode-replacement did not produce nested sibling signal in the selected nested compare "
            "or static child mode-replacement conversion suites"
        ),
        "selected_mode_replace_nested_sibling_fraction": selected_mode,
        "previous_mode_replace_nested_sibling_fraction": previous_mode,
        "conversion_overall_classification": conversion_aggregate.get("overall_classification"),
        "conversion_classification_counts": conversion_aggregate.get("classification_counts", {}),
        "mode_replace_variant_count": mode_variant_count,
        "mode_replace_signal_count": mode_signal_count,
        "dataset_target_counts": mode_target_counts,
        "suite_results": suite_results,
    }


def _classifier_readiness(classifier: dict[str, Any], dataset: dict[str, Any]) -> dict[str, Any]:
    targets = _dict(classifier.get("targets"))
    trained = [name for name, item in targets.items() if _dict(item).get("status") == "trained"]
    skipped = [
        name
        for name, item in targets.items()
        if _dict(item).get("status") != "trained"
    ]
    target_summaries: dict[str, Any] = {}
    for name, item in targets.items():
        item_obj = _dict(item)
        metrics = _dict(item_obj.get("metrics"))
        target_summaries[name] = {
            "status": item_obj.get("status"),
            "example_count": item_obj.get("example_count"),
            "positive_count": item_obj.get("positive_count", metrics.get("positive_count")),
            "negative_count": metrics.get("negative_count"),
            "accuracy": metrics.get("accuracy"),
            "precision": metrics.get("precision"),
            "recall": metrics.get("recall"),
            "top_weights": item_obj.get("top_weights", [])[:4],
        }

    dataset_aggregate = _dict(dataset.get("aggregate"))
    readiness = "partial_artifact_classifier_ready"
    if not trained:
        readiness = "not_ready"
    elif skipped:
        readiness = "partial_artifact_classifier_ready_not_branch_law"

    return {
        "readiness": readiness,
        "read": (
            "direct-carrier label is separable in the artifact-level baseline; other targets are "
            "not train-ready because they are single-class or sparse"
        ),
        "case_count": classifier.get("case_count"),
        "dataset_case_count": dataset_aggregate.get("case_count"),
        "dataset_evidence_status_counts": dataset_aggregate.get("evidence_status_counts", {}),
        "dataset_target_counts": dataset_aggregate.get("target_counts", {}),
        "feature_present_fraction": classifier.get("feature_present_fraction", {}),
        "trained_targets": trained,
        "skipped_targets": skipped,
        "target_summaries": target_summaries,
        "producer_notes": classifier.get("notes", []),
    }


def _conversion_metric_smoke(smoke: dict[str, Any]) -> dict[str, Any]:
    keys = [
        "mean_assay_readout_nested_sibling_fraction",
        "mean_assay_continuation_nested_sibling_fraction",
        "mean_assay_mode_replace_nested_sibling_fraction",
        "mean_assay_mode_replace_max_nested_sibling_readiness",
        "mean_direct_readout_dependency",
        "mean_final_direct_mix_shortcut_score",
        "mean_child_record_survival_score",
        "mean_parent_mode_conversion_sibling_fraction",
        "mean_mode_replace_conversion_score",
        "mean_world_jump_penalty",
    ]
    values = {key: smoke.get(key) for key in keys}
    return {
        "status": (
            "direct_mix_shortcut_not_parent_mode_conversion"
            if (_num(values.get("mean_final_direct_mix_shortcut_score")) or 0.0) > 0.0
            and (_num(values.get("mean_mode_replace_conversion_score")) or 0.0) <= 0.0
            else "conversion_signal_present"
            if (_num(values.get("mean_mode_replace_conversion_score")) or 0.0) > 0.0
            else "no_conversion_smoke_signal"
        ),
        "read": (
            "single-seed smoke validates the new metric split: child record survives and direct continuation succeeds, "
            "but parent/mode conversion remains zero"
        ),
        "metrics": values,
    }


def _key_claims(
    compare: dict[str, Any],
    audit_evidence: dict[str, Any],
    mode_replace: dict[str, Any],
    readiness: dict[str, Any],
    conversion_smoke: dict[str, Any],
) -> list[dict[str, Any]]:
    decision = _dict(compare.get("decision"))
    nested_selected = _dict(_dict(compare.get("nested")).get("selected"))
    return [
        {
            "claim": "Guarded childworld cycle produced a continuation-family nested sibling signal.",
            "status": "supported" if decision.get("primary_success") else "not_supported",
            "evidence": {
                "ontology_result": decision.get("ontology_result"),
                "selected_continuation_fraction": nested_selected.get(
                    "mean_assay_continuation_nested_sibling_fraction"
                ),
                "selected_continuation_readiness_max": nested_selected.get(
                    "mean_assay_continuation_max_nested_sibling_readiness"
                ),
            },
        },
        {
            "claim": "The best current mechanism is narrow: static/forked child phase record as a final direct continuation carrier.",
            "status": "supported",
            "evidence": {
                "mechanism_overall_read": audit_evidence.get("overall_read"),
                "supported_class_counts": audit_evidence.get("supported_class_counts"),
            },
        },
        {
            "claim": "Explicit low-rank continuation writeback is not established as necessary or sufficient.",
            "status": "not_supported",
            "evidence": {
                "audit_overall_read": audit_evidence.get("overall_read"),
                "reason": "kill-switch/static-record summaries preserve direct signal without explicit writeback and fail write-only controls",
            },
        },
        {
            "claim": "Mode-replacement sibling behavior remains negative in this cycle.",
            "status": mode_replace.get("status"),
            "evidence": {
                "selected_mode_replace_fraction": mode_replace.get(
                    "selected_mode_replace_nested_sibling_fraction"
                ),
                "mode_replace_signal_count": mode_replace.get("mode_replace_signal_count"),
                "mode_replace_variant_count": mode_replace.get("mode_replace_variant_count"),
            },
        },
        {
            "claim": "The new conversion metrics separate child-record survival from parent/mode conversion.",
            "status": conversion_smoke.get("status"),
            "evidence": conversion_smoke.get("metrics"),
        },
        {
            "claim": "Learned-law/classifier readiness is partial and artifact-level only.",
            "status": readiness.get("readiness"),
            "evidence": {
                "trained_targets": readiness.get("trained_targets"),
                "skipped_targets": readiness.get("skipped_targets"),
                "case_count": readiness.get("case_count"),
            },
        },
    ]


def _next_step_gates(mode_replace: dict[str, Any], readiness: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "gate": "External replay gate",
            "status": "open",
            "criterion": "Reproduce continuation direct-carrier signal on additional seeds and non-selected anchors.",
        },
        {
            "gate": "Mode-replacement gate",
            "status": "blocked",
            "criterion": "Require nonzero mode-replacement nested sibling fraction under static child record variants.",
            "current_value": {
                "status": mode_replace.get("status"),
                "mode_replace_signal_count": mode_replace.get("mode_replace_signal_count"),
            },
        },
        {
            "gate": "Classifier target-balance gate",
            "status": "blocked",
            "criterion": "Collect both positive and negative examples for skipped classifier targets before claiming learned-law readiness.",
            "current_value": {
                "trained_targets": readiness.get("trained_targets"),
                "skipped_targets": readiness.get("skipped_targets"),
            },
        },
        {
            "gate": "Runtime promotion gate",
            "status": "blocked",
            "criterion": "Do not promote to broad active default until wider regression and heldout suites clear.",
        },
    ]


def assemble(cycle_root: Path) -> dict[str, Any]:
    artifacts, artifact_status = _load_artifacts(cycle_root)
    compare = artifacts.get("cycle_compare", {})
    audit = artifacts.get("mechanism_audit", {})
    dataset = artifacts.get("dataset_v2", {})
    classifier = artifacts.get("classifier_v1", {})
    conversion = artifacts.get("static_conversion", {})

    audit_evidence = _mechanism_evidence(audit)
    mode_replace = _mode_replace_result(compare, conversion, dataset)
    readiness = _classifier_readiness(classifier, dataset)
    conversion_smoke = _conversion_metric_smoke(artifacts.get("conversion_metric_smoke", {}))

    return {
        "schema": SCHEMA,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "cycle": DEFAULT_CYCLE,
        "cycle_root": str(cycle_root),
        "source_artifacts": artifact_status,
        "key_claims": _key_claims(compare, audit_evidence, mode_replace, readiness, conversion_smoke),
        "evidence_metrics": {
            "training_selection": compare.get("training"),
            "nested_compare": _selected_nested_metrics(compare),
            "heldout_compare": _selected_heldout_metrics(compare),
            "benchmark_compare": _benchmark_metrics(compare),
            "mechanism_audit": audit_evidence,
            "conversion_metric_smoke": conversion_smoke,
        },
        "negative_mode_replacement_result": mode_replace,
        "learned_law_classifier_readiness": readiness,
        "next_step_gates": _next_step_gates(mode_replace, readiness),
        "notes": [
            "Narrow additive cycle report: source artifacts are read and summarized, not rewritten.",
            "This report does not promote runtime defaults, change training behavior, or alter artifact schemas.",
            "Nulls indicate missing or non-applicable source fields rather than inferred zeros.",
        ],
    }


def _metric_table(metrics: dict[str, Any]) -> list[str]:
    lines = [
        "| metric | previous | selected | delta |",
        "| --- | ---: | ---: | ---: |",
    ]
    for name, values in metrics.items():
        row = _dict(values)
        lines.append(
            f"| `{name}` | {_fmt(row.get('previous'))} | {_fmt(row.get('selected'))} | {_fmt(row.get('delta'))} |"
        )
    return lines


def render_markdown(report: dict[str, Any]) -> str:
    mode = _dict(report.get("negative_mode_replacement_result"))
    readiness = _dict(report.get("learned_law_classifier_readiness"))
    evidence = _dict(report.get("evidence_metrics"))
    mechanism = _dict(evidence.get("mechanism_audit"))
    benchmark = _dict(evidence.get("benchmark_compare"))
    conversion_smoke = _dict(evidence.get("conversion_metric_smoke"))

    lines = [
        "# Circleworld Childworld Cycle Report - 2026-05-11",
        "",
        f"- Schema: `{report.get('schema')}`",
        f"- Cycle: `{report.get('cycle')}`",
        f"- Mechanism read: `{mechanism.get('overall_read')}`",
        f"- Mode-replacement result: `{mode.get('status')}`",
        f"- Classifier readiness: `{readiness.get('readiness')}`",
        "",
        "## Key Claims",
        "",
    ]
    for claim in _list(report.get("key_claims")):
        claim_obj = _dict(claim)
        lines.append(f"- `{claim_obj.get('status')}`: {claim_obj.get('claim')}")

    lines.extend(["", "## Evidence Metrics", "", "### Nested Compare", ""])
    lines.extend(_metric_table(_dict(evidence.get("nested_compare"))))
    lines.extend(["", "### Held-Out Compare", ""])
    lines.extend(_metric_table(_dict(evidence.get("heldout_compare"))))
    lines.extend(["", "### Benchmark Compare", ""])
    audio = _dict(benchmark.get("audio_similarity"))
    lines.extend(_metric_table(audio))
    continuity = _dict(benchmark.get("continuity"))
    lines.extend(
        [
            "",
            f"- Selected recurrence score: `{_fmt(continuity.get('mean_recurrence_score'))}`",
            f"- Selected recurrence band: `{continuity.get('dominant_recurrence_band')}`",
            f"- Benchmark file count: `{continuity.get('file_count')}`",
            "",
            "## Mechanism Audit",
            "",
            f"- Suites: `{mechanism.get('suite_count')}` total, `{mechanism.get('suites_with_positive_signal')}` with positive signal",
            "",
        ]
    )
    counts = _dict(mechanism.get("supported_class_counts"))
    if counts:
        for name in sorted(counts):
            lines.append(f"- `{name}`: {counts[name]}")
    else:
        lines.append("- No supported classes")

    lines.extend(
        [
            "",
            "## Negative Mode-Replacement Result",
            "",
            f"- Status: `{mode.get('status')}`",
            f"- Selected mode-replace nested sibling fraction: `{_fmt(mode.get('selected_mode_replace_nested_sibling_fraction'))}`",
            f"- Static mode-replace variants with signal: `{mode.get('mode_replace_signal_count')}` of `{mode.get('mode_replace_variant_count')}`",
            f"- Dataset target counts: `{json.dumps(mode.get('dataset_target_counts', {}), sort_keys=True)}`",
            "",
            "## Conversion Metric Smoke",
            "",
            f"- Status: `{conversion_smoke.get('status')}`",
            f"- Read: {conversion_smoke.get('read')}",
            "",
            "| metric | value |",
            "| --- | ---: |",
        ]
    )
    for name, value in _dict(conversion_smoke.get("metrics")).items():
        lines.append(f"| `{name}` | {_fmt(value)} |")

    lines.extend(
        [
            "",
            "## Learned-Law / Classifier Readiness",
            "",
            f"- Readiness: `{readiness.get('readiness')}`",
            f"- Case count: `{readiness.get('case_count')}`",
            f"- Trained targets: `{', '.join(str(item) for item in readiness.get('trained_targets', [])) or 'none'}`",
            f"- Skipped targets: `{', '.join(str(item) for item in readiness.get('skipped_targets', [])) or 'none'}`",
            "",
            "| target | status | n | positive | negative | acc | precision | recall |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for name, summary in _dict(readiness.get("target_summaries")).items():
        row = _dict(summary)
        lines.append(
            "| "
            + " | ".join(
                [
                    f"`{name}`",
                    f"`{row.get('status')}`",
                    _fmt(row.get("example_count")),
                    _fmt(row.get("positive_count")),
                    _fmt(row.get("negative_count")),
                    _fmt(row.get("accuracy")),
                    _fmt(row.get("precision")),
                    _fmt(row.get("recall")),
                ]
            )
            + " |"
        )

    lines.extend(["", "## Next-Step Gates", ""])
    for gate in _list(report.get("next_step_gates")):
        gate_obj = _dict(gate)
        lines.append(
            f"- `{gate_obj.get('status')}` `{gate_obj.get('gate')}`: {gate_obj.get('criterion')}"
        )

    lines.extend(["", "## Source Artifacts", ""])
    for name, status in _dict(report.get("source_artifacts")).items():
        status_obj = _dict(status)
        loaded = "loaded" if status_obj.get("loaded") else "missing/error"
        lines.append(f"- `{name}`: `{loaded}` - `{status_obj.get('path')}`")

    return "\n".join(lines).rstrip() + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Assemble the narrow 2026-05-11 childworld cycle JSON and Markdown reports."
    )
    parser.add_argument(
        "--cycle-root",
        default=str(DEFAULT_CYCLE_ROOT),
        help="Root containing the childworld cycle artifacts.",
    )
    parser.add_argument(
        "--output-json",
        default=str(DEFAULT_OUTPUT_JSON),
        help="Path for the consolidated JSON report.",
    )
    parser.add_argument(
        "--output-md",
        default=str(DEFAULT_OUTPUT_MD),
        help="Path for the consolidated Markdown report.",
    )
    args = parser.parse_args()

    report = assemble(Path(args.cycle_root))
    _write_json(Path(args.output_json), report)
    md_path = Path(args.output_md)
    md_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text(render_markdown(report), encoding="utf-8")

    print(
        json.dumps(
            {
                "schema": report["schema"],
                "cycle": report["cycle"],
                "output_json": str(Path(args.output_json)),
                "output_md": str(md_path),
                "mode_replacement_status": _dict(
                    report.get("negative_mode_replacement_result")
                ).get("status"),
                "classifier_readiness": _dict(
                    report.get("learned_law_classifier_readiness")
                ).get("readiness"),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
