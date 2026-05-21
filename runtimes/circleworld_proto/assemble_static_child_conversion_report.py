from __future__ import annotations

import argparse
import glob
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SCHEMA = "circleworld_static_child_conversion_report_v1"
INPUT_SCHEMA = "circleworld_childworld_causality_killswitch_v1"
INPUT_FILENAME = "childworld_causality_killswitch_summary.json"
SIGNAL_EPS = 1.0e-9

FOCUS_VARIANT_GROUPS: dict[str, tuple[str, ...]] = {
    "direct": ("static_child_record_direct_mix_only",),
    "write": ("static_child_record_write_only",),
    "stripped": ("static_child_record_stripped_control",),
    "mode_replace": (
        "static_child_record_mode_replace_only",
        "static_child_record_mode_replace",
        "static_child_mode_replace_only",
        "static_child_mode_replace",
    ),
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
    matches = [Path(path) for path in glob.glob(value)]
    if not matches:
        matches = [Path(value)]

    expanded: list[Path] = []
    for path in matches:
        if path.is_dir():
            direct = path / INPUT_FILENAME
            if direct.exists():
                expanded.append(direct)
            else:
                expanded.extend(path.rglob(INPUT_FILENAME))
        else:
            expanded.append(path)
    return expanded


def _discover_inputs(raw_inputs: list[str]) -> list[Path]:
    seen: set[str] = set()
    paths: list[Path] = []
    for value in raw_inputs:
        for path in _expand_input(value):
            if not path.exists():
                raise FileNotFoundError(f"Input does not exist: {path}")
            if path.name != INPUT_FILENAME:
                raise ValueError(f"Input must be named {INPUT_FILENAME}: {path}")
            resolved = str(path.resolve())
            if resolved in seen:
                continue
            seen.add(resolved)
            paths.append(path)
    if not paths:
        raise ValueError("No input files found")
    return paths


def _dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


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


def _metric(summary: dict[str, Any], variant: str, key: str) -> float | None:
    rows = _dict(summary.get("rows"))
    row = _dict(rows.get(variant))
    metrics = _dict(row.get("metrics"))
    return _num(metrics.get(key))


def _decision(summary: dict[str, Any], variant: str) -> dict[str, Any]:
    return _dict(_dict(summary.get("decisions")).get(variant))


def _variant_names(summary: dict[str, Any]) -> list[str]:
    names: list[str] = []
    for source in (_dict(summary.get("rows")), _dict(summary.get("decisions"))):
        for name in source:
            text = str(name)
            if text not in names:
                names.append(text)
    order = summary.get("variant_order")
    if isinstance(order, list):
        ordered = [str(name) for name in order if str(name) in names]
        ordered.extend(name for name in names if name not in ordered)
        return ordered
    return names


def _variant_group(name: str) -> str | None:
    lowered = name.lower()
    for group, aliases in FOCUS_VARIANT_GROUPS.items():
        if name in aliases:
            return group
    if "static" not in lowered or "child" not in lowered:
        return None
    if "mode_replace" in lowered or "mode-replace" in lowered or "mode1_replace" in lowered:
        return "mode_replace"
    if "stripped" in lowered or "strip" in lowered:
        return "stripped"
    if "write" in lowered:
        return "write"
    if "direct" in lowered:
        return "direct"
    return None


def _continuation_fraction(summary: dict[str, Any], variant: str) -> float | None:
    decision = _decision(summary, variant)
    value = _num(decision.get("continuation_fraction"))
    if value is not None:
        return value
    return _metric(summary, variant, "mean_assay_continuation_nested_sibling_fraction")


def _mode_replace_fraction(summary: dict[str, Any], variant: str) -> float | None:
    return _metric(summary, variant, "mean_assay_mode_replace_nested_sibling_fraction")


def _preserves(summary: dict[str, Any], variant: str) -> bool | None:
    decision = _decision(summary, variant)
    if isinstance(decision.get("preserves_continuation_sibling"), bool):
        return bool(decision["preserves_continuation_sibling"])
    read = decision.get("read")
    if read == "signal_survives_kill_switch":
        return True
    if read in {"kill_switch_breaks_signal", "no_positive_signal_to_compare"}:
        return False
    cont = _continuation_fraction(summary, variant)
    if cont is None:
        return None
    return cont > SIGNAL_EPS


def _breaks(summary: dict[str, Any], variant: str) -> bool | None:
    decision = _decision(summary, variant)
    if isinstance(decision.get("breaks_continuation_sibling"), bool):
        return bool(decision["breaks_continuation_sibling"])
    read = decision.get("read")
    if read == "kill_switch_breaks_signal":
        return True
    if read in {"signal_survives_kill_switch", "no_positive_signal_to_compare"}:
        return False
    cont = _continuation_fraction(summary, variant)
    if cont is None:
        return None
    return cont <= SIGNAL_EPS


def _positive_continuation(summary: dict[str, Any]) -> float:
    value = _continuation_fraction(summary, "positive_control")
    return value if value is not None else 0.0


def _positive_mode_replace(summary: dict[str, Any]) -> float:
    value = _mode_replace_fraction(summary, "positive_control")
    return value if value is not None else 0.0


def _variant_record(summary: dict[str, Any], variant: str, group: str) -> dict[str, Any]:
    cont = _continuation_fraction(summary, variant)
    mode_replace = _mode_replace_fraction(summary, variant)
    preserves = _preserves(summary, variant)
    breaks = _breaks(summary, variant)
    return {
        "variant": variant,
        "group": group,
        "description": _dict(_dict(summary.get("rows")).get(variant)).get("description"),
        "read": _decision(summary, variant).get("read"),
        "preserves_continuation_signal": preserves,
        "breaks_continuation_signal": breaks,
        "continuation_fraction": cont,
        "mode_replace_fraction": mode_replace,
        "mode_replace_readiness": _metric(
            summary,
            variant,
            "mean_assay_mode_replace_max_nested_sibling_readiness",
        ),
        "mode_replace_gate": _metric(
            summary,
            variant,
            "mean_assay_mode_replace_mode1_replace_gate",
        ),
        "mode_replace_ratio": _metric(
            summary,
            variant,
            "mean_assay_mode_replace_mode1_replace_ratio",
        ),
        "mode_replace_static_record": _metric(
            summary,
            variant,
            "mean_assay_mode_replace_mode1_replace_static_record",
        ),
        "readout_fraction": _metric(summary, variant, "mean_assay_readout_nested_sibling_fraction"),
        "continuation_readiness": _metric(
            summary,
            variant,
            "mean_assay_continuation_max_nested_sibling_readiness",
        ),
        "write_enabled": _metric(
            summary,
            variant,
            "mean_assay_continuation_continuation_write_enabled",
        ),
        "write_delta": _metric(
            summary,
            variant,
            "mean_assay_continuation_continuation_write_delta",
        ),
        "has_signal": bool(preserves or (cont is not None and cont > SIGNAL_EPS) or (mode_replace is not None and mode_replace > SIGNAL_EPS)),
    }


def _classify_suite(summary: dict[str, Any], source_path: Path) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = {group: [] for group in FOCUS_VARIANT_GROUPS}
    for variant in _variant_names(summary):
        group = _variant_group(variant)
        if group is not None:
            grouped[group].append(_variant_record(summary, variant, group))

    positive_cont = _positive_continuation(summary)
    positive_mode = _positive_mode_replace(summary)
    positive_signal = positive_cont > SIGNAL_EPS
    direct_supported = any(item["has_signal"] for item in grouped["direct"])
    write_supported = any(item["has_signal"] for item in grouped["write"])
    mode_replace_supported = any(
        item["has_signal"] or (_num(item.get("mode_replace_fraction")) or 0.0) > SIGNAL_EPS
        for item in grouped["mode_replace"]
    )
    stripped_has_signal = any(item["has_signal"] for item in grouped["stripped"])
    stripped_breaks = any(item.get("breaks_continuation_signal") is True for item in grouped["stripped"])

    if not positive_signal:
        classification = "no_signal"
        rationale = "positive_control has no continuation-family nested sibling signal"
    elif mode_replace_supported and not (direct_supported or write_supported or stripped_has_signal):
        classification = "mode_replace_supported"
        rationale = "static child mode-replace variant preserves a signal without competing static direct/write signal"
    elif mode_replace_supported:
        classification = "mixed"
        rationale = "mode-replace signal coexists with direct/write/stripped static-child evidence"
    elif write_supported and not stripped_has_signal:
        classification = "parent_write_supported"
        rationale = "static child write-only variant preserves the continuation signal"
    elif direct_supported and not write_supported and not stripped_has_signal and (stripped_breaks or not grouped["stripped"]):
        classification = "direct_only"
        rationale = "static direct mix preserves the signal while static write does not"
    else:
        classification = "mixed"
        rationale = "positive signal is present but static-child variants are missing, conflicting, or inconclusive"

    return {
        "suite_name": source_path.parent.name or source_path.stem,
        "source_path": str(source_path),
        "input_schema": summary.get("schema"),
        "producer_overall_read": summary.get("overall_read"),
        "out_dir": summary.get("out_dir"),
        "config": summary.get("config"),
        "mode": summary.get("mode"),
        "positive_signal": positive_signal,
        "positive_metrics": {
            "continuation_fraction": positive_cont,
            "mode_replace_fraction": positive_mode,
        },
        "classification": classification,
        "rationale": rationale,
        "signals": {
            "direct_supported": direct_supported,
            "write_supported": write_supported,
            "stripped_has_signal": stripped_has_signal,
            "stripped_breaks": stripped_breaks,
            "mode_replace_supported": mode_replace_supported,
        },
        "focused_variants": grouped,
        "missing_focus_groups": [group for group, items in grouped.items() if not items],
    }


def _aggregate(suites: list[dict[str, Any]]) -> dict[str, Any]:
    counts: dict[str, int] = {}
    for suite in suites:
        classification = str(suite.get("classification"))
        counts[classification] = counts.get(classification, 0) + 1

    non_no_signal = {key for key in counts if key != "no_signal" and counts[key] > 0}
    if not suites:
        overall = "no_signal"
    elif not non_no_signal:
        overall = "no_signal"
    elif len(non_no_signal) == 1 and counts.get("no_signal", 0) == 0:
        overall = next(iter(non_no_signal))
    else:
        overall = "mixed"

    return {
        "overall_classification": overall,
        "suite_count": len(suites),
        "classification_counts": counts,
        "suites_with_positive_signal": sum(1 for suite in suites if suite.get("positive_signal")),
    }


def assemble(paths: list[Path]) -> dict[str, Any]:
    suites = [_classify_suite(_load_json(path), path) for path in paths]
    return {
        "schema": SCHEMA,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "thresholds": {"signal_eps": SIGNAL_EPS},
        "inputs": [str(path) for path in paths],
        "aggregate": _aggregate(suites),
        "suite_reports": suites,
        "notes": [
            "This report focuses on static-child direct/write/stripped/mode-replace variants when present.",
            "Inputs are summarized only; source kill-switch summaries are not modified.",
        ],
    }


def _fmt(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def render_markdown(report: dict[str, Any]) -> str:
    aggregate = _dict(report.get("aggregate"))
    lines = [
        "# Static Child Conversion Report",
        "",
        f"- Overall classification: `{aggregate.get('overall_classification')}`",
        f"- Suites: {aggregate.get('suite_count')} total, {aggregate.get('suites_with_positive_signal')} with positive signal",
        "",
        "## Classification Counts",
        "",
    ]
    counts = _dict(aggregate.get("classification_counts"))
    if counts:
        for name in sorted(counts):
            lines.append(f"- `{name}`: {counts[name]}")
    else:
        lines.append("- none")

    lines.extend(
        [
            "",
            "## Suite Summary",
            "",
            "| suite | class | positive_cont | direct | write | stripped_signal | mode_replace |",
            "| --- | --- | ---: | --- | --- | --- | --- |",
        ]
    )
    for suite in report.get("suite_reports", []):
        if not isinstance(suite, dict):
            continue
        metrics = _dict(suite.get("positive_metrics"))
        signals = _dict(suite.get("signals"))
        lines.append(
            "| "
            + " | ".join(
                [
                    str(suite.get("suite_name")),
                    f"`{suite.get('classification')}`",
                    _fmt(metrics.get("continuation_fraction")),
                    str(bool(signals.get("direct_supported"))),
                    str(bool(signals.get("write_supported"))),
                    str(bool(signals.get("stripped_has_signal"))),
                    str(bool(signals.get("mode_replace_supported"))),
                ]
            )
            + " |"
        )

    lines.extend(["", "## Focused Variants", ""])
    for suite in report.get("suite_reports", []):
        if not isinstance(suite, dict):
            continue
        lines.append(f"### {suite.get('suite_name')}")
        lines.append("")
        lines.append(f"- Classification: `{suite.get('classification')}`")
        lines.append(f"- Rationale: {suite.get('rationale')}")
        focused = _dict(suite.get("focused_variants"))
        for group in ("direct", "write", "stripped", "mode_replace"):
            variants = focused.get(group)
            if not isinstance(variants, list) or not variants:
                lines.append(f"- `{group}`: missing")
                continue
            for item in variants:
                if not isinstance(item, dict):
                    continue
                lines.append(
                    f"- `{group}` `{item.get('variant')}`: "
                    f"cont={_fmt(item.get('continuation_fraction'))}, "
                    f"mode_replace={_fmt(item.get('mode_replace_fraction'))}, "
                    f"mode_ready={_fmt(item.get('mode_replace_readiness'))}, "
                    f"mode_gate={_fmt(item.get('mode_replace_gate'))}, "
                    f"mode_ratio={_fmt(item.get('mode_replace_ratio'))}, "
                    f"read=`{item.get('read')}`, "
                    f"has_signal={bool(item.get('has_signal'))}"
                )
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Assemble static-child conversion reports from childworld causality kill-switch summaries."
    )
    parser.add_argument(
        "--input",
        required=True,
        nargs="+",
        action="append",
        help="One or more childworld_causality_killswitch_summary.json files, directories, or glob patterns.",
    )
    parser.add_argument("--output-json", required=True, help="Path for the assembled JSON report.")
    parser.add_argument("--output-md", default=None, help="Optional path for a Markdown report.")
    args = parser.parse_args()

    inputs = _discover_inputs(_flatten_inputs(args.input))
    report = assemble(inputs)
    _write_json(Path(args.output_json), report)
    if args.output_md:
        md_path = Path(args.output_md)
        md_path.parent.mkdir(parents=True, exist_ok=True)
        md_path.write_text(render_markdown(report), encoding="utf-8")
    print(json.dumps(report["aggregate"], indent=2))


if __name__ == "__main__":
    main()
