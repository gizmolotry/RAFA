from __future__ import annotations

import argparse
import glob
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SCHEMA = "circleworld_childworld_mechanism_audit_v1"
SUMMARY_SCHEMA = "circleworld_childworld_causality_killswitch_v1"
KNOWN_INPUT_FILENAMES = {
    "childworld_causality_killswitch_summary.json",
    "continuation_causal_killswitch_compare.json",
}
SIGNAL_EPS = 1.0e-9


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return payload


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _flatten_inputs(values: list[list[str]]) -> list[str]:
    return [item for group in values for item in group]


def _expand_input(value: str) -> list[Path]:
    matches = [Path(p) for p in glob.glob(value)]
    if not matches:
        matches = [Path(value)]
    expanded: list[Path] = []
    for path in matches:
        if path.is_dir():
            direct = [path / name for name in KNOWN_INPUT_FILENAMES if (path / name).exists()]
            if direct:
                expanded.extend(direct)
            else:
                for name in KNOWN_INPUT_FILENAMES:
                    expanded.extend(path.rglob(name))
        else:
            expanded.append(path)
    return expanded


def _discover_inputs(raw_inputs: list[str]) -> list[Path]:
    seen: set[str] = set()
    paths: list[Path] = []
    for value in raw_inputs:
        for path in _expand_input(value):
            resolved = str(path.resolve())
            if resolved in seen:
                continue
            if not path.exists():
                raise FileNotFoundError(f"Input does not exist: {path}")
            if path.name not in KNOWN_INPUT_FILENAMES:
                raise ValueError(
                    "Input must be childworld_causality_killswitch_summary.json "
                    f"or continuation_causal_killswitch_compare.json: {path}"
                )
            seen.add(resolved)
            paths.append(path)
    if not paths:
        raise ValueError("No input files found")
    return paths


def _num(value: Any, default: float | None = None) -> float | None:
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return default
    return default


def _get_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _metric(summary: dict[str, Any], variant: str, key: str) -> float | None:
    rows = _get_dict(summary.get("rows"))
    row = _get_dict(rows.get(variant))
    metrics = _get_dict(row.get("metrics"))
    return _num(metrics.get(key))


def _variant_present(summary: dict[str, Any], variant: str) -> bool:
    rows = _get_dict(summary.get("rows"))
    decisions = _get_dict(summary.get("decisions"))
    return variant in rows or variant in decisions


def _variant_read(summary: dict[str, Any], variant: str) -> str | None:
    decision = _get_dict(_get_dict(summary.get("decisions")).get(variant))
    read = decision.get("read")
    return str(read) if read is not None else None


def _positive_continuation(summary: dict[str, Any]) -> float:
    value = _metric(summary, "positive_control", "mean_assay_continuation_nested_sibling_fraction")
    if value is not None:
        return value
    decision = _get_dict(_get_dict(summary.get("decisions")).get("positive_control"))
    return _num(decision.get("continuation_fraction"), 0.0) or 0.0


def _positive_readout(summary: dict[str, Any]) -> float:
    return _metric(summary, "positive_control", "mean_assay_readout_nested_sibling_fraction") or 0.0


def _continuation(summary: dict[str, Any], variant: str) -> float | None:
    value = _metric(summary, variant, "mean_assay_continuation_nested_sibling_fraction")
    if value is not None:
        return value
    decision = _get_dict(_get_dict(summary.get("decisions")).get(variant))
    return _num(decision.get("continuation_fraction"))


def _breaks(summary: dict[str, Any], variant: str, positive_signal: bool) -> bool | None:
    if not _variant_present(summary, variant):
        return None
    decision = _get_dict(_get_dict(summary.get("decisions")).get(variant))
    if isinstance(decision.get("breaks_continuation_sibling"), bool):
        return bool(decision["breaks_continuation_sibling"])
    read = _variant_read(summary, variant)
    if read == "kill_switch_breaks_signal":
        return True
    if read in {"signal_survives_kill_switch", "no_positive_signal_to_compare"}:
        return False
    cont = _continuation(summary, variant)
    if cont is None:
        return None
    return bool(positive_signal and cont <= SIGNAL_EPS)


def _survives(summary: dict[str, Any], variant: str, positive_signal: bool) -> bool | None:
    if not _variant_present(summary, variant):
        return None
    decision = _get_dict(_get_dict(summary.get("decisions")).get(variant))
    if isinstance(decision.get("preserves_continuation_sibling"), bool):
        return bool(decision["preserves_continuation_sibling"])
    read = _variant_read(summary, variant)
    if read == "signal_survives_kill_switch":
        return True
    if read in {"kill_switch_breaks_signal", "no_positive_signal_to_compare"}:
        return False
    cont = _continuation(summary, variant)
    if cont is None:
        return None
    return bool(positive_signal and cont > SIGNAL_EPS)


def _variant_metrics(summary: dict[str, Any], variant: str) -> dict[str, Any]:
    return {
        "continuation_fraction": _continuation(summary, variant),
        "readout_fraction": _metric(summary, variant, "mean_assay_readout_nested_sibling_fraction"),
        "continuation_readiness": _metric(
            summary,
            variant,
            "mean_assay_continuation_max_nested_sibling_readiness",
        ),
        "continuation_write_enabled": _metric(
            summary,
            variant,
            "mean_assay_continuation_continuation_write_enabled",
        ),
        "continuation_write_delta": _metric(
            summary,
            variant,
            "mean_assay_continuation_continuation_write_delta",
        ),
        "branch_identity_qualified_carry": _metric(
            summary,
            variant,
            "mean_assay_continuation_branch_identity_qualified_carry",
        ),
        "branch_identity_budget_retained": _metric(
            summary,
            variant,
            "mean_assay_continuation_branch_identity_budget_retained",
        ),
    }


def _evidence(
    summary: dict[str, Any],
    evidence_class: str,
    status: str,
    variants: list[str],
    read: str,
) -> dict[str, Any]:
    positive_signal = _positive_continuation(summary) > SIGNAL_EPS
    any_variant_present = any(_variant_present(summary, variant) for variant in variants)
    final_status = (
        "not_evaluated"
        if evidence_class != "no_signal" and not any_variant_present
        else status
    )
    return {
        "evidence_class": evidence_class,
        "status": final_status,
        "read": read,
        "variants": {
            variant: {
                "present": _variant_present(summary, variant),
                "breaks_signal": _breaks(summary, variant, positive_signal),
                "preserves_signal": _survives(summary, variant, positive_signal),
                "producer_read": _variant_read(summary, variant),
                "metrics": _variant_metrics(summary, variant),
            }
            for variant in variants
        },
    }


def _classify_summary(summary: dict[str, Any], suite_name: str, source_path: Path) -> dict[str, Any]:
    positive_cont = _positive_continuation(summary)
    positive_readout = _positive_readout(summary)
    positive_signal = positive_cont > SIGNAL_EPS

    disable_write_breaks = _breaks(summary, "disable_continuation_write", positive_signal)
    write_only_survives = _survives(summary, "continuation_write_only_readout", positive_signal)
    write_only_breaks = _breaks(summary, "continuation_write_only_readout", positive_signal)
    parent_only_breaks = _breaks(summary, "continuation_parent_only", positive_signal)
    scramble_breaks = _breaks(summary, "scramble_continuation_child_phase", positive_signal)
    no_ifs_breaks = _breaks(summary, "no_child_local_ifs", positive_signal)
    no_retention_breaks = _breaks(summary, "no_child_coherence_retention", positive_signal)
    strict_direct_survives = _survives(summary, "strict_direct_mix_only", positive_signal)
    strict_write_breaks = _breaks(summary, "strict_write_only_no_preunroll", positive_signal)
    static_direct_survives = _survives(summary, "static_child_record_direct_mix_only", positive_signal)
    static_write_breaks = _breaks(summary, "static_child_record_write_only", positive_signal)
    static_stripped_breaks = _breaks(summary, "static_child_record_stripped_control", positive_signal)
    no_branch_pre_breaks = _breaks(summary, "no_branch_pre_unroll", positive_signal)
    no_branch_runtime_breaks = _breaks(summary, "no_branch_childworld_runtime", positive_signal)
    no_cont_pre_breaks = _breaks(summary, "no_continuation_pre_unroll", positive_signal)
    strict_parent_breaks = _breaks(summary, "strict_parent_no_child_context", positive_signal)
    budget_phase_variants = [
        "phase_roll_budget_clamped",
        "phase_random_budget_clamped",
        "phase_zero_budget_clamped",
        "phase_cross_child_budget_clamped",
    ]
    budget_phase_breaks = [
        variant
        for variant in budget_phase_variants
        if _breaks(summary, variant, positive_signal)
    ]

    evidence: list[dict[str, Any]] = []
    if not positive_signal:
        evidence.append(
            _evidence(
                summary,
                "no_signal",
                "supported",
                ["positive_control"],
                "positive control has no continuation-family nested sibling signal",
            )
        )
    else:
        evidence.append(
            _evidence(
                summary,
                "no_signal",
                "not_supported",
                ["positive_control"],
                "positive control has a continuation-family nested sibling signal",
            )
        )

    direct_supported = bool(scramble_breaks or parent_only_breaks)
    evidence.append(
        _evidence(
            summary,
            "direct_child_phase_carrier",
            "supported" if direct_supported else "not_supported" if positive_signal else "blocked_no_signal",
            ["scramble_continuation_child_phase", "continuation_parent_only"],
            (
                "scrambling child phase and/or parent-only rendering breaks the positive signal"
                if direct_supported
                else "direct child phase/readout kill-switch did not break an established signal"
            ),
        )
    )

    gated_supported = bool(disable_write_breaks or write_only_survives)
    evidence.append(
        _evidence(
            summary,
            "gated_writeback_supported",
            "supported" if gated_supported else "not_supported" if positive_signal else "blocked_no_signal",
            ["disable_continuation_write", "continuation_write_only_readout"],
            (
                "continuation write is necessary or write-only readout preserves the signal"
                if gated_supported
                else "signal does not require continuation writeback under this assay"
            ),
        )
    )

    evidence.append(
        _evidence(
            summary,
            "child_local_ifs_supported",
            "supported" if no_ifs_breaks else "not_supported" if positive_signal else "blocked_no_signal",
            ["no_child_local_ifs"],
            (
                "disabling child-local IFS breaks the positive signal"
                if no_ifs_breaks
                else "child-local IFS removal does not break the established signal"
            ),
        )
    )

    evidence.append(
        _evidence(
            summary,
            "coherence_retention_required",
            "supported" if no_retention_breaks else "not_supported" if positive_signal else "blocked_no_signal",
            ["no_child_coherence_retention"],
            (
                "disabling child-local coherence retention breaks the positive signal"
                if no_retention_breaks
                else "coherence-retention removal does not break the established signal"
            ),
        )
    )

    readout_supported = bool(positive_signal and disable_write_breaks is False and write_only_breaks)
    evidence.append(
        _evidence(
            summary,
            "readout_only",
            "supported" if readout_supported else "not_supported" if positive_signal else "blocked_no_signal",
            ["disable_continuation_write", "continuation_write_only_readout"],
            (
                "signal survives disabled writeback but breaks when direct child readout mix is removed"
                if readout_supported
                else "readout-only carrier is not isolated by these kill-switches"
            ),
        )
    )

    strict_direct_supported = bool(strict_direct_survives and strict_write_breaks)
    evidence.append(
        _evidence(
            summary,
            "strict_direct_child_mix_carrier",
            "supported" if strict_direct_supported else "not_supported" if positive_signal else "blocked_no_signal",
            ["strict_direct_mix_only", "strict_write_only_no_preunroll"],
            (
                "direct child continuation mix preserves the signal while strict write-only/no-preunroll fails"
                if strict_direct_supported
                else "strict direct-mix/write-only controls do not isolate direct child mix as sufficient"
            ),
        )
    )

    phase_budget_supported = bool(budget_phase_breaks and strict_direct_survives)
    evidence.append(
        _evidence(
            summary,
            "phase_specific_under_budget_clamp",
            "supported" if phase_budget_supported else "not_supported" if positive_signal else "blocked_no_signal",
            budget_phase_variants,
            (
                "budget-clamped child phase substitution breaks the continuation signal"
                if phase_budget_supported
                else "budget-clamped child phase substitution does not break an established signal"
            ),
        )
    )

    static_record_supported = bool(static_direct_survives and (static_write_breaks is not False))
    evidence.append(
        _evidence(
            summary,
            "static_child_record_direct_carrier",
            "supported" if static_record_supported else "not_supported" if positive_signal else "blocked_no_signal",
            [
                "static_child_record_direct_mix_only",
                "static_child_record_write_only",
                "static_child_record_stripped_control",
            ],
            (
                "static forked child record can carry direct continuation mix without branch childworld runtime/pre-unroll"
                if static_record_supported
                else "static forked child record does not carry the continuation signal by itself"
            ),
        )
    )

    evidence.append(
        _evidence(
            summary,
            "child_record_presence_required",
            "supported" if static_stripped_breaks else "not_supported" if positive_signal else "blocked_no_signal",
            ["static_child_record_stripped_control"],
            (
                "stripping child records breaks the static direct-carrier control"
                if static_stripped_breaks
                else "stripping child records does not break the signal or was not sufficient to test it"
            ),
        )
    )

    evidence.append(
        _evidence(
            summary,
            "branch_preunroll_required",
            "supported" if no_branch_pre_breaks else "not_supported" if positive_signal else "blocked_no_signal",
            ["no_branch_pre_unroll"],
            (
                "skipping branch-level child pre-unroll breaks the signal"
                if no_branch_pre_breaks
                else "branch-level child pre-unroll is not required by this assay"
            ),
        )
    )

    evidence.append(
        _evidence(
            summary,
            "branch_childworld_runtime_required",
            "supported" if no_branch_runtime_breaks else "not_supported" if positive_signal else "blocked_no_signal",
            ["no_branch_childworld_runtime"],
            (
                "disabling branch childworld runtime breaks the signal"
                if no_branch_runtime_breaks
                else "branch childworld runtime is not required by this assay"
            ),
        )
    )

    evidence.append(
        _evidence(
            summary,
            "continuation_preunroll_required",
            "supported" if no_cont_pre_breaks else "not_supported" if positive_signal else "blocked_no_signal",
            ["no_continuation_pre_unroll"],
            (
                "skipping final continuation pre-unroll breaks the signal"
                if no_cont_pre_breaks
                else "final continuation pre-unroll is not required by this assay"
            ),
        )
    )

    evidence.append(
        _evidence(
            summary,
            "strict_parent_no_child_context_breaks",
            "supported" if strict_parent_breaks else "not_supported" if positive_signal else "blocked_no_signal",
            ["strict_parent_no_child_context"],
            (
                "strict parent/no-child context breaks the signal"
                if strict_parent_breaks
                else "strict parent/no-child context does not break the signal"
            ),
        )
    )

    supported_classes = [
        item["evidence_class"]
        for item in evidence
        if item.get("status") == "supported" and item.get("evidence_class") != "no_signal"
    ]
    if not positive_signal:
        mechanism_read = "no_signal"
    elif supported_classes:
        mechanism_read = "+".join(supported_classes)
    else:
        mechanism_read = "signal_present_unclassified"

    rows = _get_dict(summary.get("rows"))
    return {
        "suite_name": suite_name,
        "source_path": str(source_path),
        "input_schema": summary.get("schema"),
        "config": summary.get("config"),
        "out_dir": summary.get("out_dir"),
        "producer_overall_read": summary.get("overall_read"),
        "variant_order": summary.get("variant_order", list(rows.keys())),
        "positive_signal": positive_signal,
        "positive_metrics": {
            "continuation_fraction": positive_cont,
            "readout_fraction": positive_readout,
            "num_cases": _metric(summary, "positive_control", "num_cases"),
        },
        "mechanism_read": mechanism_read,
        "supported_classes": supported_classes,
        "evidence": evidence,
    }


def _extract_summaries(path: Path) -> tuple[list[tuple[str, dict[str, Any]]], dict[str, Any]]:
    payload = _load_json(path)
    if payload.get("schema") == SUMMARY_SCHEMA or "rows" in payload:
        return [(path.parent.name or path.stem, payload)], {
            "input_type": "childworld_causality_killswitch_summary",
            "producer_decision": None,
        }

    suites = _get_dict(payload.get("suites"))
    if suites:
        extracted: list[tuple[str, dict[str, Any]]] = []
        for suite_name, suite_payload in suites.items():
            suite_obj = _get_dict(suite_payload)
            summary = _get_dict(suite_obj.get("summary"))
            if not summary and suite_obj.get("summary_path"):
                summary_path = Path(str(suite_obj["summary_path"]))
                if summary_path.exists():
                    summary = _load_json(summary_path)
            if summary:
                extracted.append((str(suite_name), summary))
        return extracted, {
            "input_type": "continuation_causal_killswitch_compare",
            "cycle": payload.get("cycle"),
            "suite": payload.get("suite"),
            "paths": payload.get("paths"),
            "selected_flags": payload.get("selected_flags"),
            "producer_decision": payload.get("decision"),
        }

    return [], {
        "input_type": "unknown",
        "producer_decision": payload.get("decision"),
        "top_level_keys": sorted(str(key) for key in payload.keys()),
    }


def _aggregate(suite_decisions: list[dict[str, Any]]) -> dict[str, Any]:
    class_counts: dict[str, int] = {}
    no_signal_count = 0
    for decision in suite_decisions:
        if not decision.get("positive_signal"):
            no_signal_count += 1
        for evidence_class in decision.get("supported_classes", []):
            class_counts[evidence_class] = class_counts.get(evidence_class, 0) + 1
    suites_with_signal = len(suite_decisions) - no_signal_count
    if suites_with_signal <= 0:
        overall_read = "no_signal"
    elif class_counts:
        ordered = sorted(class_counts.items(), key=lambda item: (-item[1], item[0]))
        overall_read = "+".join(name for name, _count in ordered)
    else:
        overall_read = "signal_present_unclassified"
    return {
        "overall_read": overall_read,
        "suite_count": len(suite_decisions),
        "suites_with_positive_signal": suites_with_signal,
        "suites_without_positive_signal": no_signal_count,
        "supported_class_counts": class_counts,
    }


def assemble(paths: list[Path]) -> dict[str, Any]:
    input_records: list[dict[str, Any]] = []
    suite_decisions: list[dict[str, Any]] = []
    for path in paths:
        summaries, metadata = _extract_summaries(path)
        record = {
            "path": str(path),
            "input_type": metadata.get("input_type"),
            "suite_names": [name for name, _summary in summaries],
            "suite_count": len(summaries),
            "producer_metadata": {key: value for key, value in metadata.items() if key != "input_type"},
        }
        input_records.append(record)
        for suite_name, summary in summaries:
            suite_decisions.append(_classify_summary(summary, suite_name, path))

    return {
        "schema": SCHEMA,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "thresholds": {
            "signal_eps": SIGNAL_EPS,
        },
        "input_records": input_records,
        "aggregate_decision": _aggregate(suite_decisions),
        "suite_decisions": suite_decisions,
        "notes": [
            "Additive audit schema: producer payloads are referenced and summarized, not rewritten.",
            "Missing variants or metrics yield null evidence fields and conservative missing/not-supported reads.",
        ],
    }


def _fmt(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def render_markdown(audit: dict[str, Any]) -> str:
    agg = _get_dict(audit.get("aggregate_decision"))
    lines = [
        "# Childworld Mechanism Audit",
        "",
        f"- Schema: `{audit.get('schema')}`",
        f"- Overall read: `{agg.get('overall_read')}`",
        f"- Suites: {agg.get('suite_count')} total, {agg.get('suites_with_positive_signal')} with positive signal",
        "",
        "## Supported Class Counts",
        "",
    ]
    counts = _get_dict(agg.get("supported_class_counts"))
    if counts:
        for name, count in sorted(counts.items()):
            lines.append(f"- `{name}`: {count}")
    else:
        lines.append("- none")

    lines.extend(
        [
            "",
            "## Suite Decisions",
            "",
            "| suite | mechanism_read | positive_cont | supported_classes |",
            "| --- | --- | ---: | --- |",
        ]
    )
    for decision in audit.get("suite_decisions", []):
        if not isinstance(decision, dict):
            continue
        positive = _get_dict(decision.get("positive_metrics"))
        supported = ", ".join(str(item) for item in decision.get("supported_classes", [])) or "none"
        lines.append(
            "| "
            + " | ".join(
                [
                    str(decision.get("suite_name")),
                    f"`{decision.get('mechanism_read')}`",
                    _fmt(positive.get("continuation_fraction")),
                    supported,
                ]
            )
            + " |"
        )

    lines.extend(["", "## Evidence", ""])
    for decision in audit.get("suite_decisions", []):
        if not isinstance(decision, dict):
            continue
        lines.append(f"### {decision.get('suite_name')}")
        lines.append("")
        for item in decision.get("evidence", []):
            if not isinstance(item, dict):
                continue
            lines.append(
                f"- `{item.get('evidence_class')}`: `{item.get('status')}` - {item.get('read')}"
            )
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Assemble childworld causality kill-switch summaries into mechanism-audit decisions."
    )
    parser.add_argument(
        "--input",
        required=True,
        nargs="+",
        action="append",
        help="One or more summary/compare JSON files, directories, or glob patterns.",
    )
    parser.add_argument("--output-json", required=True, help="Path for the assembled mechanism audit JSON.")
    parser.add_argument("--output-md", default=None, help="Optional path for a Markdown report.")
    args = parser.parse_args()

    inputs = _discover_inputs(_flatten_inputs(args.input))
    audit = assemble(inputs)
    _write_json(Path(args.output_json), audit)
    if args.output_md:
        md_path = Path(args.output_md)
        md_path.parent.mkdir(parents=True, exist_ok=True)
        md_path.write_text(render_markdown(audit), encoding="utf-8")
    print(json.dumps(audit["aggregate_decision"], indent=2))


if __name__ == "__main__":
    main()
