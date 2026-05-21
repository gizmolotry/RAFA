from __future__ import annotations

import argparse
import glob
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SCHEMA = "circleworld_childworld_mechanism_dataset_v1"
SUMMARY_SCHEMA = "circleworld_childworld_causality_killswitch_v1"
AUDIT_SCHEMA = "circleworld_childworld_mechanism_audit_v1"

KNOWN_INPUT_FILENAMES = {
    "childworld_causality_killswitch_summary.json",
    "childworld_mechanism_audit.json",
    "childworld_mechanism_audit_combined.json",
}

SIGNAL_EPS = 1.0e-9

TARGET_CLASSES = {
    "target_direct_carrier": {
        "direct_child_phase_carrier",
        "readout_only",
        "strict_direct_child_mix_carrier",
        "static_child_record_direct_carrier",
    },
    "target_writeback_sufficient": {
        "gated_writeback_supported",
    },
    "target_phase_specific": {
        "phase_specific_under_budget_clamp",
    },
    "target_child_record_required": {
        "child_record_presence_required",
    },
    "target_mode_replace_supported": {
        "static_child_record_mode_replace_supported",
    },
}

BASIC_VARIANT_CLASSES = {
    "positive_control": ["positive_control"],
    "disable_continuation_write": ["gated_writeback_supported", "readout_only"],
    "continuation_write_only_readout": ["gated_writeback_supported", "readout_only"],
    "continuation_parent_only": ["direct_child_phase_carrier"],
    "scramble_continuation_child_phase": ["direct_child_phase_carrier"],
    "no_child_local_ifs": ["child_local_ifs_supported"],
    "no_child_coherence_retention": ["coherence_retention_required"],
    "strict_direct_mix_only": ["strict_direct_child_mix_carrier"],
    "strict_write_only_no_preunroll": ["strict_direct_child_mix_carrier"],
    "phase_roll_budget_clamped": ["phase_specific_under_budget_clamp"],
    "phase_random_budget_clamped": ["phase_specific_under_budget_clamp"],
    "phase_zero_budget_clamped": ["phase_specific_under_budget_clamp"],
    "phase_cross_child_budget_clamped": ["phase_specific_under_budget_clamp"],
    "static_child_record_direct_mix_only": ["static_child_record_direct_carrier"],
    "static_child_record_write_only": ["static_child_record_direct_carrier"],
    "static_child_record_stripped_control": [
        "static_child_record_direct_carrier",
        "child_record_presence_required",
    ],
    "static_child_record_mode1_replace": ["static_child_record_mode_replace_supported"],
    "static_child_record_mode1_replace_stripped_control": [
        "static_child_record_mode_replace_supported",
        "child_record_presence_required",
    ],
    "static_child_record_mode1_replace_random_phase": ["static_child_record_mode_replace_supported"],
    "static_child_record_mode1_replace_clamped": ["static_child_record_mode_replace_supported"],
    "static_child_record_mode1_replace_gate_floor": ["static_child_record_mode_replace_supported"],
    "static_child_record_mode1_replace_overdrive": ["static_child_record_mode_replace_supported"],
    "no_branch_pre_unroll": ["branch_preunroll_required"],
    "no_branch_childworld_runtime": ["branch_childworld_runtime_required"],
    "no_continuation_pre_unroll": ["continuation_preunroll_required"],
    "strict_parent_no_child_context": ["strict_parent_no_child_context_breaks"],
}


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
                continue
            for name in KNOWN_INPUT_FILENAMES:
                expanded.extend(path.rglob(name))
            expanded.extend(path.rglob("childworld_mechanism_audit*.json"))
        else:
            expanded.append(path)
    return expanded


def _is_known_input(path: Path) -> bool:
    return path.name in KNOWN_INPUT_FILENAMES or (
        path.name.startswith("childworld_mechanism_audit") and path.suffix == ".json"
    )


def _discover_inputs(raw_inputs: list[str]) -> list[Path]:
    seen: set[str] = set()
    paths: list[Path] = []
    for value in raw_inputs:
        for path in _expand_input(value):
            if not path.exists():
                raise FileNotFoundError(f"Input does not exist: {path}")
            if not _is_known_input(path):
                raise ValueError(
                    "Input must be a childworld_causality_killswitch_summary.json "
                    "or childworld_mechanism_audit*.json file, or a directory containing them: "
                    f"{path}"
                )
            resolved = str(path.resolve())
            if resolved in seen:
                continue
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


def _bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        lower = value.strip().lower()
        if lower in {"true", "yes", "1"}:
            return True
        if lower in {"false", "no", "0"}:
            return False
    return None


def _get_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _get_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _metric(summary: dict[str, Any], variant: str, key: str) -> float | None:
    row = _get_dict(_get_dict(summary.get("rows")).get(variant))
    metrics = _get_dict(row.get("metrics"))
    return _num(metrics.get(key))


def _row_metrics(summary: dict[str, Any], variant: str) -> dict[str, Any]:
    row = _get_dict(_get_dict(summary.get("rows")).get(variant))
    return _get_dict(row.get("metrics"))


def _decision(summary: dict[str, Any], variant: str) -> dict[str, Any]:
    return _get_dict(_get_dict(summary.get("decisions")).get(variant))


def _variant_present(summary: dict[str, Any], variant: str) -> bool:
    return variant in _get_dict(summary.get("rows")) or variant in _get_dict(summary.get("decisions"))


def _positive_continuation(summary: dict[str, Any]) -> float:
    value = _metric(summary, "positive_control", "mean_assay_continuation_nested_sibling_fraction")
    if value is not None:
        return value
    return _num(_decision(summary, "positive_control").get("continuation_fraction"), 0.0) or 0.0


def _continuation(summary: dict[str, Any], variant: str) -> float | None:
    value = _metric(summary, variant, "mean_assay_continuation_nested_sibling_fraction")
    if value is not None:
        return value
    return _num(_decision(summary, variant).get("continuation_fraction"))


def _readout(summary: dict[str, Any], variant: str) -> float | None:
    return _metric(summary, variant, "mean_assay_readout_nested_sibling_fraction")


def _mode_replace(summary: dict[str, Any], variant: str) -> float | None:
    value = _metric(summary, variant, "mean_assay_mode_replace_nested_sibling_fraction")
    if value is not None:
        return value
    return _num(_decision(summary, variant).get("mode_replace_fraction"))


def _producer_read(summary: dict[str, Any], variant: str) -> str | None:
    read = _decision(summary, variant).get("read")
    return str(read) if read is not None else None


def _breaks(summary: dict[str, Any], variant: str, positive_signal: bool) -> bool | None:
    if not _variant_present(summary, variant):
        return None
    decision = _decision(summary, variant)
    breaks = _bool(decision.get("breaks_continuation_sibling"))
    if breaks is not None:
        return breaks
    read = _producer_read(summary, variant)
    if read == "kill_switch_breaks_signal":
        return True
    if read in {"signal_survives_kill_switch", "positive_control", "no_positive_signal_to_compare"}:
        return False
    cont = _continuation(summary, variant)
    if cont is None:
        return None
    return bool(positive_signal and cont <= SIGNAL_EPS)


def _survives(summary: dict[str, Any], variant: str, positive_signal: bool) -> bool | None:
    if not _variant_present(summary, variant):
        return None
    decision = _decision(summary, variant)
    survives = _bool(decision.get("preserves_continuation_sibling"))
    if survives is not None:
        return survives
    read = _producer_read(summary, variant)
    if read in {"signal_survives_kill_switch", "positive_control"}:
        return True
    if read in {"kill_switch_breaks_signal", "no_positive_signal_to_compare"}:
        return False
    cont = _continuation(summary, variant)
    if cont is None:
        return None
    return bool(positive_signal and cont > SIGNAL_EPS)


def _summary_variant_metrics(summary: dict[str, Any], variant: str) -> dict[str, Any]:
    metrics = _row_metrics(summary, variant)
    decision = _decision(summary, variant)
    return {
        "continuation_fraction": _continuation(summary, variant),
        "mode_replace_fraction": _mode_replace(summary, variant),
        "readout_fraction": _readout(summary, variant),
        "continuation_write_enabled": _num(
            metrics.get("mean_assay_continuation_continuation_write_enabled")
        ),
        "continuation_write_delta": _num(
            metrics.get("mean_assay_continuation_continuation_write_delta")
        ),
        "branch_identity_qualified_carry": _num(
            metrics.get("mean_assay_continuation_branch_identity_qualified_carry"),
            _num(metrics.get("mean_branch_identity_qualified_carry")),
        ),
        "branch_identity_budget_retained": _num(
            metrics.get("mean_assay_continuation_branch_identity_budget_retained"),
            _num(metrics.get("mean_branch_identity_budget_retained")),
        ),
        "fine_q_profile_corr": _num(metrics.get("mean_assay_continuation_fine_q_profile_corr")),
        "readout_sibling_response": _num(
            metrics.get("mean_assay_continuation_readout_sibling_response")
        ),
        "mode_replace_readiness": _num(
            metrics.get("mean_assay_mode_replace_max_nested_sibling_readiness")
        ),
        "mode_replace_enabled": _num(metrics.get("mean_assay_mode_replace_mode1_replace_enabled")),
        "mode_replace_gate": _num(metrics.get("mean_assay_mode_replace_mode1_replace_gate")),
        "mode_replace_ratio": _num(metrics.get("mean_assay_mode_replace_mode1_replace_ratio")),
        "mode_replace_static_record": _num(
            metrics.get("mean_assay_mode_replace_mode1_replace_static_record")
        ),
        "direct_readout_dependency": _num(
            metrics.get("mean_direct_readout_dependency"),
            _num(decision.get("direct_readout_dependency")),
        ),
        "final_direct_mix_shortcut_score": _num(
            metrics.get("mean_final_direct_mix_shortcut_score"),
            _num(decision.get("final_direct_mix_shortcut_score")),
        ),
        "child_record_survival_score": _num(
            metrics.get("mean_child_record_survival_score"),
            _num(decision.get("child_record_survival_score")),
        ),
        "parent_mode_conversion_sibling_fraction": _num(
            metrics.get("mean_parent_mode_conversion_sibling_fraction"),
        ),
        "mode_replace_conversion_score": _num(
            metrics.get("mean_mode_replace_conversion_score"),
            _num(decision.get("mode_replace_conversion_score")),
        ),
        "world_jump_penalty": _num(
            metrics.get("mean_world_jump_penalty"),
            _num(decision.get("world_jump_penalty")),
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
        if evidence_class not in {"no_signal", "positive_control"} and not any_variant_present
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
                "producer_read": _producer_read(summary, variant),
                "metrics": _summary_variant_metrics(summary, variant),
            }
            for variant in variants
        },
    }


def _classify_summary(summary: dict[str, Any], suite_name: str, source_path: Path) -> dict[str, Any]:
    positive_signal = _positive_continuation(summary) > SIGNAL_EPS

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
    mode_replace_variants = [
        "static_child_record_mode1_replace",
        "static_child_record_mode1_replace_clamped",
        "static_child_record_mode1_replace_gate_floor",
        "static_child_record_mode1_replace_overdrive",
    ]
    mode_replace_supported_variants = [
        variant
        for variant in mode_replace_variants
        if (_mode_replace(summary, variant) or 0.0) > SIGNAL_EPS
    ]
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
        variant for variant in budget_phase_variants if _breaks(summary, variant, positive_signal)
    ]

    evidence: list[dict[str, Any]] = [
        _evidence(
            summary,
            "no_signal",
            "not_supported" if positive_signal else "supported",
            ["positive_control"],
            (
                "positive control has a continuation-family nested sibling signal"
                if positive_signal
                else "positive control has no continuation-family nested sibling signal"
            ),
        )
    ]

    direct_supported = bool(scramble_breaks or parent_only_breaks)
    evidence.append(
        _evidence(
            summary,
            "direct_child_phase_carrier",
            "supported" if direct_supported else "not_supported" if positive_signal else "blocked_no_signal",
            ["scramble_continuation_child_phase", "continuation_parent_only"],
            "child phase/parent-only controls isolate whether direct child phase carries the signal",
        )
    )

    gated_supported = bool(disable_write_breaks or write_only_survives)
    evidence.append(
        _evidence(
            summary,
            "gated_writeback_supported",
            "supported" if gated_supported else "not_supported" if positive_signal else "blocked_no_signal",
            ["disable_continuation_write", "continuation_write_only_readout"],
            "writeback controls isolate whether continuation writeback is sufficient or necessary",
        )
    )

    evidence.append(
        _evidence(
            summary,
            "child_local_ifs_supported",
            "supported" if no_ifs_breaks else "not_supported" if positive_signal else "blocked_no_signal",
            ["no_child_local_ifs"],
            "child-local IFS removal control",
        )
    )
    evidence.append(
        _evidence(
            summary,
            "coherence_retention_required",
            "supported" if no_retention_breaks else "not_supported" if positive_signal else "blocked_no_signal",
            ["no_child_coherence_retention"],
            "child-local coherence retention control",
        )
    )

    readout_supported = bool(positive_signal and disable_write_breaks is False and write_only_breaks)
    evidence.append(
        _evidence(
            summary,
            "readout_only",
            "supported" if readout_supported else "not_supported" if positive_signal else "blocked_no_signal",
            ["disable_continuation_write", "continuation_write_only_readout"],
            "readout-only/direct-mix carrier control",
        )
    )

    strict_direct_supported = bool(strict_direct_survives and strict_write_breaks)
    evidence.append(
        _evidence(
            summary,
            "strict_direct_child_mix_carrier",
            "supported" if strict_direct_supported else "not_supported" if positive_signal else "blocked_no_signal",
            ["strict_direct_mix_only", "strict_write_only_no_preunroll"],
            "strict direct child mix sufficiency control",
        )
    )

    phase_budget_supported = bool(budget_phase_breaks and strict_direct_survives)
    evidence.append(
        _evidence(
            summary,
            "phase_specific_under_budget_clamp",
            "supported" if phase_budget_supported else "not_supported" if positive_signal else "blocked_no_signal",
            budget_phase_variants,
            "phase substitution under budget-clamp control",
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
            "static forked child-record direct carrier control",
        )
    )
    evidence.append(
        _evidence(
            summary,
            "child_record_presence_required",
            "supported" if static_stripped_breaks else "not_supported" if positive_signal else "blocked_no_signal",
            ["static_child_record_stripped_control"],
            "static child-record stripping control",
        )
    )
    evidence.append(
        _evidence(
            summary,
            "static_child_record_mode_replace_supported",
            "supported"
            if mode_replace_supported_variants
            else "not_supported"
            if positive_signal
            else "blocked_no_signal",
            [
                "static_child_record_mode1_replace",
                "static_child_record_mode1_replace_clamped",
                "static_child_record_mode1_replace_gate_floor",
                "static_child_record_mode1_replace_overdrive",
                "static_child_record_mode1_replace_stripped_control",
                "static_child_record_mode1_replace_random_phase",
            ],
            "static child-record mode replacement conversion control",
        )
    )
    evidence.append(
        _evidence(
            summary,
            "branch_preunroll_required",
            "supported" if no_branch_pre_breaks else "not_supported" if positive_signal else "blocked_no_signal",
            ["no_branch_pre_unroll"],
            "branch pre-unroll removal control",
        )
    )
    evidence.append(
        _evidence(
            summary,
            "branch_childworld_runtime_required",
            "supported" if no_branch_runtime_breaks else "not_supported" if positive_signal else "blocked_no_signal",
            ["no_branch_childworld_runtime"],
            "branch childworld runtime removal control",
        )
    )
    evidence.append(
        _evidence(
            summary,
            "continuation_preunroll_required",
            "supported" if no_cont_pre_breaks else "not_supported" if positive_signal else "blocked_no_signal",
            ["no_continuation_pre_unroll"],
            "continuation pre-unroll removal control",
        )
    )
    evidence.append(
        _evidence(
            summary,
            "strict_parent_no_child_context_breaks",
            "supported" if strict_parent_breaks else "not_supported" if positive_signal else "blocked_no_signal",
            ["strict_parent_no_child_context"],
            "strict parent/no-child context control",
        )
    )

    supported_classes = [
        item["evidence_class"]
        for item in evidence
        if item.get("status") == "supported" and item.get("evidence_class") != "no_signal"
    ]
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
            "continuation_fraction": _positive_continuation(summary),
            "readout_fraction": _readout(summary, "positive_control"),
            "num_cases": _metric(summary, "positive_control", "num_cases"),
        },
        "mechanism_read": "no_signal" if not positive_signal else "+".join(supported_classes) or "signal_present_unclassified",
        "supported_classes": supported_classes,
        "evidence": evidence,
    }


def _extract_summary_decisions(path: Path, payload: dict[str, Any]) -> list[dict[str, Any]]:
    suite_name = path.parent.name or path.stem
    return [_classify_summary(payload, suite_name, path)]


def _source_summary_for_decision(decision: dict[str, Any]) -> dict[str, Any] | None:
    source = decision.get("source_path")
    if not source:
        return None
    source_path = Path(str(source))
    if not source_path.exists():
        return None
    try:
        payload = _load_json(source_path)
    except Exception:
        return None

    if payload.get("schema") == SUMMARY_SCHEMA or "rows" in payload:
        return payload

    suites = _get_dict(payload.get("suites"))
    suite = _get_dict(suites.get(str(decision.get("suite_name"))))
    summary = _get_dict(suite.get("summary"))
    if summary:
        return summary
    summary_path_value = suite.get("summary_path")
    if summary_path_value:
        summary_path = Path(str(summary_path_value))
        if summary_path.exists():
            try:
                return _load_json(summary_path)
            except Exception:
                return None
    return None


def _enrich_decision_metrics(decision: dict[str, Any]) -> dict[str, Any]:
    summary = _source_summary_for_decision(decision)
    if not summary:
        return decision

    positive_signal = _positive_continuation(summary) > SIGNAL_EPS
    for evidence_item in _get_list(decision.get("evidence")):
        evidence = _get_dict(evidence_item)
        for variant_name, variant_payload in _get_dict(evidence.get("variants")).items():
            variant = _get_dict(variant_payload)
            metrics = _get_dict(variant.get("metrics"))
            summary_metrics = _summary_variant_metrics(summary, str(variant_name))
            for key, value in summary_metrics.items():
                if metrics.get(key) is None and value is not None:
                    metrics[key] = value
            variant["metrics"] = metrics
            if variant.get("present") is None:
                variant["present"] = _variant_present(summary, str(variant_name))
            if variant.get("breaks_signal") is None:
                variant["breaks_signal"] = _breaks(summary, str(variant_name), positive_signal)
            if variant.get("preserves_signal") is None:
                variant["preserves_signal"] = _survives(summary, str(variant_name), positive_signal)
            if variant.get("producer_read") is None:
                variant["producer_read"] = _producer_read(summary, str(variant_name))
    return decision


def _extract_audit_decisions(path: Path, payload: dict[str, Any]) -> list[dict[str, Any]]:
    decisions: list[dict[str, Any]] = []
    for item in _get_list(payload.get("suite_decisions")):
        decision = _get_dict(item)
        if decision:
            if not decision.get("source_path"):
                decision["source_path"] = str(path)
            decisions.append(_enrich_decision_metrics(decision))
    return decisions


def _extract_decisions(path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    payload = _load_json(path)
    schema = payload.get("schema")
    if schema == AUDIT_SCHEMA or "suite_decisions" in payload:
        decisions = _extract_audit_decisions(path, payload)
        return decisions, {
            "path": str(path),
            "input_type": "childworld_mechanism_audit",
            "schema": schema,
            "suite_count": len(decisions),
            "aggregate_decision": payload.get("aggregate_decision"),
        }
    if schema == SUMMARY_SCHEMA or "rows" in payload:
        decisions = _extract_summary_decisions(path, payload)
        return decisions, {
            "path": str(path),
            "input_type": "childworld_causality_killswitch_summary",
            "schema": schema,
            "suite_count": len(decisions),
            "aggregate_decision": None,
        }
    return [], {
        "path": str(path),
        "input_type": "unknown",
        "schema": schema,
        "suite_count": 0,
        "top_level_keys": sorted(str(key) for key in payload.keys()),
    }


def _target_value(label_class: str, status: str | None, target_name: str) -> bool | None:
    if label_class not in TARGET_CLASSES[target_name]:
        return None
    if status == "supported":
        return True
    if status in {"not_supported", "blocked_no_signal", "no_signal"}:
        return False
    return None


def _positive_case(breaks_signal: Any, preserves_signal: Any, variant_name: str) -> bool | None:
    breaks = _bool(breaks_signal)
    preserves = _bool(preserves_signal)
    if preserves is True:
        return True
    if breaks is True:
        return False
    if variant_name == "positive_control" and breaks is False:
        return True
    if preserves is False:
        return False
    return None


def _case_from_variant(
    decision: dict[str, Any],
    evidence: dict[str, Any],
    variant_name: str,
    variant: dict[str, Any],
) -> dict[str, Any]:
    metrics = _get_dict(variant.get("metrics"))
    label_class = str(evidence.get("evidence_class", "unknown"))
    status = str(evidence.get("status")) if evidence.get("status") is not None else None
    present = _bool(variant.get("present"))
    breaks_signal = _bool(variant.get("breaks_signal"))
    preserves_signal = _bool(variant.get("preserves_signal"))

    case = {
        "suite_name": decision.get("suite_name"),
        "source_path": decision.get("source_path"),
        "config": decision.get("config"),
        "out_dir": decision.get("out_dir"),
        "mechanism_read": decision.get("mechanism_read"),
        "positive_signal": _bool(decision.get("positive_signal")),
        "variant_name": variant_name,
        "label_class": label_class,
        "evidence_status": status,
        "evidence_read": evidence.get("read"),
        "present": present,
        "positive_case": _positive_case(breaks_signal, preserves_signal, variant_name),
        "negative_case": None,
        "breaks_signal": breaks_signal,
        "preserves_signal": preserves_signal,
        "producer_read": variant.get("producer_read"),
        "continuation_fraction": _num(metrics.get("continuation_fraction")),
        "mode_replace_fraction": _num(metrics.get("mode_replace_fraction")),
        "readout_fraction": _num(metrics.get("readout_fraction")),
        "continuation_write_enabled": _num(metrics.get("continuation_write_enabled")),
        "continuation_write_delta": _num(metrics.get("continuation_write_delta")),
        "branch_identity_qualified_carry": _num(metrics.get("branch_identity_qualified_carry")),
        "branch_identity_budget_retained": _num(metrics.get("branch_identity_budget_retained")),
        "fine_q_profile_corr": _num(
            metrics.get("fine_q_profile_corr"),
            _num(metrics.get("q_corr")),
        ),
        "readout_sibling_response": _num(
            metrics.get("readout_sibling_response"),
            _num(metrics.get("response")),
        ),
        "mode_replace_readiness": _num(metrics.get("mode_replace_readiness")),
        "mode_replace_enabled": _num(metrics.get("mode_replace_enabled")),
        "mode_replace_gate": _num(metrics.get("mode_replace_gate")),
        "mode_replace_ratio": _num(metrics.get("mode_replace_ratio")),
        "mode_replace_static_record": _num(metrics.get("mode_replace_static_record")),
        "world_jump_penalty": _num(metrics.get("world_jump_penalty")),
        "target_direct_carrier": _target_value(label_class, status, "target_direct_carrier"),
        "target_writeback_sufficient": _target_value(
            label_class, status, "target_writeback_sufficient"
        ),
        "target_phase_specific": _target_value(label_class, status, "target_phase_specific"),
        "target_child_record_required": _target_value(
            label_class, status, "target_child_record_required"
        ),
        "target_mode_replace_supported": _target_value(
            label_class, status, "target_mode_replace_supported"
        ),
    }
    positive = _bool(case["positive_case"])
    case["negative_case"] = None if positive is None else not positive
    return case


def _cases_from_decision(decision: dict[str, Any]) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for evidence_item in _get_list(decision.get("evidence")):
        evidence = _get_dict(evidence_item)
        variants = _get_dict(evidence.get("variants"))
        for variant_name, variant_payload in variants.items():
            variant = _get_dict(variant_payload)
            key = (str(evidence.get("evidence_class", "unknown")), str(variant_name))
            if key in seen:
                continue
            seen.add(key)
            cases.append(_case_from_variant(decision, evidence, str(variant_name), variant))

    if cases:
        return cases

    # Fallback for hand-written or partial records that have rows/decisions but no audit evidence.
    for variant_name in _get_list(decision.get("variant_order")):
        label_classes = BASIC_VARIANT_CLASSES.get(str(variant_name), ["unknown"])
        for label_class in label_classes:
            evidence = {
                "evidence_class": label_class,
                "status": "observed",
                "read": "variant observed without classified evidence",
                "variants": {},
            }
            variant = {
                "present": True,
                "breaks_signal": None,
                "preserves_signal": None,
                "producer_read": None,
                "metrics": {},
            }
            cases.append(_case_from_variant(decision, evidence, str(variant_name), variant))
    return cases


def _aggregate(cases: list[dict[str, Any]]) -> dict[str, Any]:
    label_counts: dict[str, int] = {}
    status_counts: dict[str, int] = {}
    target_counts: dict[str, dict[str, int]] = {
        name: {"true": 0, "false": 0, "null": 0} for name in TARGET_CLASSES
    }
    for case in cases:
        label = str(case.get("label_class", "unknown"))
        label_counts[label] = label_counts.get(label, 0) + 1
        status = str(case.get("evidence_status", "unknown"))
        status_counts[status] = status_counts.get(status, 0) + 1
        for target_name in TARGET_CLASSES:
            value = case.get(target_name)
            bucket = "null" if value is None else "true" if bool(value) else "false"
            target_counts[target_name][bucket] += 1
    return {
        "case_count": len(cases),
        "label_class_counts": dict(sorted(label_counts.items())),
        "evidence_status_counts": dict(sorted(status_counts.items())),
        "target_counts": target_counts,
    }


def build_dataset(paths: list[Path]) -> dict[str, Any]:
    input_records: list[dict[str, Any]] = []
    suite_decisions: list[dict[str, Any]] = []
    cases: list[dict[str, Any]] = []

    for path in paths:
        decisions, record = _extract_decisions(path)
        input_records.append(record)
        suite_decisions.extend(decisions)
        for decision in decisions:
            cases.extend(_cases_from_decision(decision))

    return {
        "schema": SCHEMA,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "input_records": input_records,
        "suite_count": len(suite_decisions),
        "aggregate": _aggregate(cases),
        "cases": cases,
    }


def _fmt(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def _write_markdown(path: Path, dataset: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Childworld Mechanism Dataset",
        "",
        f"- schema: `{dataset.get('schema')}`",
        f"- suite_count: `{dataset.get('suite_count')}`",
        f"- case_count: `{_get_dict(dataset.get('aggregate')).get('case_count', 0)}`",
        "",
        "| suite | label_class | variant | pos | cont | readout | write | delta | carry | budget | q_corr | response | jump | direct | writeback | phase | child_record |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for case in _get_list(dataset.get("cases")):
        row = [
            case.get("suite_name"),
            case.get("label_class"),
            case.get("variant_name"),
            case.get("positive_case"),
            case.get("continuation_fraction"),
            case.get("readout_fraction"),
            case.get("continuation_write_enabled"),
            case.get("continuation_write_delta"),
            case.get("branch_identity_qualified_carry"),
            case.get("branch_identity_budget_retained"),
            case.get("fine_q_profile_corr"),
            case.get("readout_sibling_response"),
            case.get("world_jump_penalty"),
            case.get("target_direct_carrier"),
            case.get("target_writeback_sufficient"),
            case.get("target_phase_specific"),
            case.get("target_child_record_required"),
        ]
        lines.append("| " + " | ".join(_fmt(value).replace("|", "\\|") for value in row) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a learned-law/control dataset from childworld kill-switch summaries "
            "and mechanism audit artifacts."
        )
    )
    parser.add_argument(
        "--input",
        action="append",
        nargs="+",
        required=True,
        help=(
            "One or more childworld summary/audit JSON files, directories, or globs. "
            "May be provided more than once."
        ),
    )
    parser.add_argument("--output-json", required=True, help="Path for the dataset JSON artifact.")
    parser.add_argument("--output-md", help="Optional Markdown table for inspection.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = _discover_inputs(_flatten_inputs(args.input))
    dataset = build_dataset(paths)
    _write_json(Path(args.output_json), dataset)
    if args.output_md:
        _write_markdown(Path(args.output_md), dataset)
    print(
        json.dumps(
            {
                "schema": dataset["schema"],
                "inputs": len(paths),
                "suite_count": dataset["suite_count"],
                "case_count": dataset["aggregate"]["case_count"],
                "output_json": str(Path(args.output_json)),
                "output_md": str(Path(args.output_md)) if args.output_md else None,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
