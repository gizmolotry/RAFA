from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SCHEMA = "circleworld_experiment_ledger_v1"
ROOT = Path(__file__).resolve().parents[2]
REPORTS_DIR = ROOT / "docs" / "reports"
OUTPUTS_DIR = ROOT / "outputs" / "circleworld_proto"
DEFAULT_OUTPUT_MD = REPORTS_DIR / "CIRCLEWORLD_EXPERIMENT_LEDGER_2026-05-13.md"
DEFAULT_OUTPUT_JSONL = REPORTS_DIR / "CIRCLEWORLD_EXPERIMENT_LEDGER_2026-05-13.jsonl"
DEFAULT_SINCE = "2026-04-20"
REPORT_GLOBS = ("CIRCLEWORLD*.md", "RAFA_PHASE_NATIVE*.md")

DATE_RE = re.compile(r"(20\d{2}-\d{2}-\d{2})")
WINDOWS_PATH_RE = re.compile(r"D:\\RAFA\\[^\s`\r\n,;)]+")
HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$", re.MULTILINE)
INLINE_METRIC_RE = re.compile(
    r"`(?P<key>[A-Za-z0-9_./ -]+)`\s*:\s*`?(?P<value>-?\d+(?:\.\d+)?(?:e[+-]?\d+)?)`?",
    re.IGNORECASE,
)
TABLE_ROW_RE = re.compile(r"^\|\s*`?(?P<key>[A-Za-z0-9_./ -]+)`?\s*\|(?P<cells>.+)\|$")

SUMMARY_NAME_HINTS = (
    "summary",
    "compare",
    "report",
    "classifier",
    "dataset",
    "audit",
    "scout",
)
METRIC_KEY_HINTS = (
    "fraction",
    "score",
    "readiness",
    "conversion",
    "sibling",
    "writeback",
    "divergence",
    "coherence",
    "corr",
    "mae",
    "mse",
    "recurrence",
    "penalty",
    "charge",
    "occupancy",
    "accuracy",
    "precision",
    "recall",
)


@dataclass
class LedgerEntry:
    date: str
    claim_test: str
    artifact_paths: list[str] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)
    interpretation: str = ""
    limitations: str = ""
    next_action: str = ""
    source_path: str = ""
    source_kind: str = "report"

    def as_jsonable(self) -> dict[str, Any]:
        return {
            "date": self.date,
            "claim_test": self.claim_test,
            "artifact_paths": self.artifact_paths,
            "metrics": self.metrics,
            "interpretation": self.interpretation,
            "limitations": self.limitations,
            "next_action": self.next_action,
            "source_path": self.source_path,
            "source_kind": self.source_kind,
        }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Assemble a Markdown/JSONL-style ledger of recent Circleworld experiments."
    )
    parser.add_argument("--reports-dir", type=Path, default=REPORTS_DIR)
    parser.add_argument("--outputs-dir", type=Path, default=OUTPUTS_DIR)
    parser.add_argument("--output-md", type=Path, default=DEFAULT_OUTPUT_MD)
    parser.add_argument("--output-jsonl", type=Path, default=DEFAULT_OUTPUT_JSONL)
    parser.add_argument("--since", default=DEFAULT_SINCE, help="Inclusive YYYY-MM-DD lower bound.")
    parser.add_argument("--max-reports", type=int, default=36)
    parser.add_argument("--max-output-only", type=int, default=14)
    parser.add_argument("--max-json-bytes", type=int, default=5_000_000)
    parser.add_argument(
        "--jsonl-stdout",
        action="store_true",
        help="Also print compact JSONL entries to stdout.",
    )
    return parser.parse_args()


def _date_from_path(path: Path) -> str:
    match = DATE_RE.search(str(path))
    if match:
        return match.group(1)
    return datetime.fromtimestamp(path.stat().st_mtime).strftime("%Y-%m-%d")


def _safe_read_text(path: Path) -> str:
    data = path.read_bytes()
    try:
        return data.decode("utf-8-sig")
    except UnicodeDecodeError:
        return data.decode("cp1252", errors="replace")


def _safe_load_json(path: Path, max_bytes: int) -> dict[str, Any] | None:
    if not path.exists() or path.stat().st_size > max_bytes:
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
    except Exception:
        return None
    return payload if isinstance(payload, dict) else None


def _clean_path(raw: str) -> str:
    value = raw.strip().rstrip(".,;)")
    if " - " in value:
        value = value.split(" - ", 1)[0].rstrip()
    return value


def _dedupe_preserve_order(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value not in seen:
            seen.add(value)
            result.append(value)
    return result


def _first_heading(text: str, fallback: str) -> str:
    for line in text.splitlines():
        if line.startswith("# "):
            return line[2:].strip()
    return fallback


def _section_map(text: str) -> dict[str, str]:
    matches = list(HEADING_RE.finditer(text))
    sections: dict[str, str] = {}
    for idx, match in enumerate(matches):
        title = match.group(2).strip()
        start = match.end()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)
        sections[title] = text[start:end].strip()
    return sections


def _find_section(sections: dict[str, str], *needles: str) -> str:
    lowered = [(title.lower(), body) for title, body in sections.items()]
    for needle in needles:
        needle_l = needle.lower()
        for title, body in lowered:
            if needle_l in title:
                return body
    return ""


def _strip_md(value: str) -> str:
    value = re.sub(r"`([^`]+)`", r"\1", value)
    value = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", value)
    value = re.sub(r"\s+", " ", value)
    return value.strip(" -*\t")


def _first_paragraph_or_bullets(section: str, max_items: int = 3) -> str:
    if not section:
        return ""
    bullets: list[str] = []
    paragraph: list[str] = []
    in_fence = False
    for raw in section.splitlines():
        line = raw.strip()
        if line.startswith("```"):
            in_fence = not in_fence
            continue
        if in_fence or not line or line.startswith("|") or set(line) <= {"-", ":", " "}:
            if paragraph:
                break
            continue
        if line.startswith(("-", "*", "1.", "2.", "3.", "4.", "5.", "6.")):
            bullets.append(_strip_md(re.sub(r"^\d+\.\s+", "", line)))
            if len(bullets) >= max_items:
                break
            continue
        if not bullets:
            paragraph.append(_strip_md(line))
    if bullets:
        return "; ".join(b for b in bullets if b)[:900]
    return " ".join(paragraph)[:900]


def _metric_value(value: str) -> int | float | None:
    try:
        number = float(value)
    except ValueError:
        return None
    if number.is_integer() and abs(number) < 10_000_000:
        return int(number)
    return round(number, 8)


def _metrics_from_markdown(text: str, limit: int = 14) -> dict[str, Any]:
    metrics: dict[str, Any] = {}
    for match in INLINE_METRIC_RE.finditer(text):
        key = _strip_md(match.group("key"))
        value = _metric_value(match.group("value"))
        if value is not None and _is_interesting_metric(key):
            metrics[key] = value
        if len(metrics) >= limit:
            return metrics
    for raw in text.splitlines():
        row = TABLE_ROW_RE.match(raw.strip())
        if not row:
            continue
        key = _strip_md(row.group("key"))
        if not _is_interesting_metric(key):
            continue
        cells = [cell.strip().strip("`") for cell in row.group("cells").split("|")]
        numeric = [_metric_value(cell) for cell in cells]
        numeric = [value for value in numeric if value is not None]
        if numeric:
            metrics[key] = numeric[-1] if len(numeric) == 1 else numeric[:3]
        if len(metrics) >= limit:
            break
    return metrics


def _is_interesting_metric(key: str) -> bool:
    key_l = key.lower()
    return any(hint in key_l for hint in METRIC_KEY_HINTS)


def _flatten_metrics(
    payload: Any,
    prefix: str = "",
    depth: int = 0,
    max_depth: int = 5,
) -> dict[str, Any]:
    if depth > max_depth:
        return {}
    metrics: dict[str, Any] = {}
    if isinstance(payload, dict):
        for key, value in payload.items():
            key_s = str(key)
            next_prefix = f"{prefix}.{key_s}" if prefix else key_s
            if isinstance(value, bool):
                continue
            if isinstance(value, (int, float)) and _is_interesting_metric(next_prefix):
                metrics[next_prefix] = round(float(value), 8)
            elif isinstance(value, dict):
                metrics.update(_flatten_metrics(value, next_prefix, depth + 1, max_depth))
            elif isinstance(value, list) and len(value) <= 8:
                for idx, item in enumerate(value):
                    if isinstance(item, dict):
                        metrics.update(
                            _flatten_metrics(item, f"{next_prefix}[{idx}]", depth + 1, max_depth)
                        )
    return metrics


def _select_metrics(metrics: dict[str, Any], limit: int = 16) -> dict[str, Any]:
    priority = (
        "mean_assay_mode_replace_nested_sibling_fraction",
        "mean_parent_mode_conversion_sibling_fraction",
        "mean_mode_replace_conversion_score",
        "mean_assay_continuation_nested_sibling_fraction",
        "mean_nested_sibling_fraction",
        "mean_real_branch_fraction",
        "mean_child_writeback_mass",
        "mean_child_parent_divergence",
        "mean_corr",
        "mean_mae",
        "mean_mse",
        "mean_world_jump_penalty",
        "mean_parent_ontology_charge",
        "mean_assay_mode_replace_mode1_replace_parent_ontology_mode1_occupancy_after",
    )
    selected: dict[str, Any] = {}
    for suffix in priority:
        candidates = [(key, value) for key, value in metrics.items() if key.endswith(suffix)]
        for key, value in sorted(candidates, key=lambda item: _metric_rank(item[0])):
            if key not in selected:
                selected[key] = value
                break
        if len(selected) >= limit:
            return selected
    for key, value in sorted(metrics.items(), key=lambda item: _metric_rank(item[0])):
        if key not in selected:
            selected[key] = value
        if len(selected) >= limit:
            break
    return selected


def _metric_rank(key: str) -> tuple[int, int, str]:
    key_l = key.lower()
    rank = 100
    for token in ("selected", "guarded", "maxreadout", "gate55", "combined", "decision"):
        if token in key_l:
            rank -= 12
    for token in ("parent_ontology", "mode_replace", "conversion", "continuation"):
        if token in key_l:
            rank -= 4
    for token in ("positive_control", "previous", "baseline", "base_"):
        if token in key_l:
            rank += 18
    return (rank, len(key), key)


def _artifact_paths_from_text(text: str) -> list[str]:
    return _dedupe_preserve_order([_clean_path(match.group(0)) for match in WINDOWS_PATH_RE.finditer(text)])


def _metrics_from_artifacts(paths: list[str], max_json_bytes: int, limit: int = 18) -> dict[str, Any]:
    metrics: dict[str, Any] = {}
    for raw in paths:
        path = Path(raw)
        if path.suffix.lower() != ".json":
            continue
        payload = _safe_load_json(path, max_json_bytes)
        if not payload:
            continue
        flattened = _flatten_metrics(payload)
        for key, value in _select_metrics(flattened, limit).items():
            compact_key = key
            if compact_key.startswith("variants."):
                parts = compact_key.split(".")
                if len(parts) >= 4:
                    compact_key = f"{parts[1]}.{parts[-1]}"
            metrics.setdefault(compact_key, value)
            if len(metrics) >= limit:
                return metrics
    return metrics


def _claim_from_sections(title: str, sections: dict[str, str]) -> str:
    for needle in ("Executive Claim", "Result", "Key Claims", "Claim", "What Changed"):
        body = _find_section(sections, needle)
        claim = _first_paragraph_or_bullets(body, max_items=2)
        if claim:
            return claim
    return title


def _entry_from_report(path: Path, max_json_bytes: int) -> LedgerEntry:
    text = _safe_read_text(path)
    title = _first_heading(text, path.stem)
    sections = _section_map(text)
    artifacts = _artifact_paths_from_text(text)
    md_metrics = _metrics_from_markdown(text)
    artifact_metrics = _metrics_from_artifacts(artifacts, max_json_bytes)
    metrics = {**artifact_metrics, **md_metrics}
    interpretation = _first_paragraph_or_bullets(
        _find_section(sections, "Interpretation", "What We Learned", "Current Recommendation"),
        max_items=3,
    )
    limitations = _first_paragraph_or_bullets(
        _find_section(
            sections,
            "What This Does Not Prove",
            "Remaining Gaps",
            "Known limitations",
            "Limitations",
        ),
        max_items=4,
    )
    next_action = _first_paragraph_or_bullets(
        _find_section(sections, "Next Falsification", "Next-Step", "Next Step", "Current Recommendation"),
        max_items=4,
    )
    return LedgerEntry(
        date=_date_from_path(path),
        claim_test=_claim_from_sections(title, sections),
        artifact_paths=artifacts[:12],
        metrics=_select_metrics(metrics, 18),
        interpretation=interpretation or "No explicit interpretation section found; see source report.",
        limitations=limitations or "Not explicitly reconstructed from this report.",
        next_action=next_action or "Review source report and linked artifacts before promotion.",
        source_path=str(path),
        source_kind="report",
    )


def _report_paths(reports_dir: Path, since: str, max_reports: int) -> list[Path]:
    if not reports_dir.exists():
        return []
    paths: list[Path] = []
    for pattern in REPORT_GLOBS:
        paths.extend(
            path
            for path in reports_dir.glob(pattern)
            if _date_from_path(path) >= since and path.name != DEFAULT_OUTPUT_MD.name
        )
    return sorted(paths, key=lambda p: (_date_from_path(p), p.stat().st_mtime), reverse=True)[:max_reports]


def _output_summary_paths(outputs_dir: Path, since: str, max_json_bytes: int) -> list[Path]:
    if not outputs_dir.exists():
        return []
    paths: list[Path] = []
    for path in outputs_dir.rglob("*.json"):
        if path.stat().st_size > max_json_bytes:
            continue
        name_l = path.name.lower()
        if name_l == "nested_commitment_summary.json":
            continue
        if not any(hint in name_l for hint in SUMMARY_NAME_HINTS):
            continue
        if _date_from_path(path) >= since or datetime.fromtimestamp(path.stat().st_mtime).strftime("%Y-%m-%d") >= since:
            paths.append(path)
    return sorted(paths, key=lambda p: p.stat().st_mtime, reverse=True)


def _entry_from_output_json(path: Path, max_json_bytes: int) -> LedgerEntry | None:
    payload = _safe_load_json(path, max_json_bytes)
    if not payload:
        return None
    metrics = _select_metrics(_flatten_metrics(payload), 16)
    artifacts = [str(path)]
    for key in ("report_path", "out_dir", "config"):
        value = payload.get(key)
        if isinstance(value, str) and value.startswith("D:\\RAFA\\"):
            artifacts.append(value)
    for value in payload.get("paths", {}).values() if isinstance(payload.get("paths"), dict) else []:
        if isinstance(value, str) and value.startswith("D:\\RAFA\\"):
            artifacts.append(value)
    rel = path.relative_to(ROOT) if path.is_relative_to(ROOT) else path
    schema = payload.get("schema") or payload.get("cycle") or payload.get("mode") or path.stem
    return LedgerEntry(
        date=_date_from_path(path),
        claim_test=f"Output artifact scan: {schema}",
        artifact_paths=_dedupe_preserve_order(artifacts)[:10],
        metrics=metrics,
        interpretation="Output JSON was found without a paired report entry in this ledger window.",
        limitations="Metrics are mechanically extracted; interpretation must be confirmed from reports or manual artifact review.",
        next_action="Pair this artifact with a narrative report before using it as claim evidence.",
        source_path=str(rel),
        source_kind="output_json",
    )


def _build_entries(args: argparse.Namespace) -> list[LedgerEntry]:
    report_entries = [
        _entry_from_report(path, args.max_json_bytes)
        for path in _report_paths(args.reports_dir, args.since, args.max_reports)
    ]
    referenced = {Path(path).resolve() for entry in report_entries for path in entry.artifact_paths if Path(path).exists()}
    output_entries: list[LedgerEntry] = []
    seen_output_dirs: set[Path] = set()
    for path in _output_summary_paths(args.outputs_dir, args.since, args.max_json_bytes):
        resolved = path.resolve()
        if resolved in referenced:
            continue
        top = _top_output_dir(path, args.outputs_dir)
        if top in seen_output_dirs:
            continue
        seen_output_dirs.add(top)
        entry = _entry_from_output_json(path, args.max_json_bytes)
        if entry:
            output_entries.append(entry)
        if len(output_entries) >= args.max_output_only:
            break
    return sorted(report_entries + output_entries, key=lambda e: (e.date, e.source_kind), reverse=True)


def _top_output_dir(path: Path, outputs_dir: Path) -> Path:
    try:
        rel = path.relative_to(outputs_dir)
    except ValueError:
        return path.parent
    return outputs_dir / rel.parts[0] if rel.parts else path.parent


def _fmt_metric_value(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.6g}"
    if isinstance(value, list):
        return "[" + ", ".join(_fmt_metric_value(v) for v in value[:3]) + "]"
    return str(value)


def _shorten(value: str, limit: int = 180) -> str:
    value = _strip_md(value)
    if len(value) <= limit:
        return value
    return value[: limit - 3].rstrip() + "..."


def _metrics_inline(metrics: dict[str, Any], limit: int = 5) -> str:
    if not metrics:
        return "none reconstructed"
    items = list(metrics.items())[:limit]
    return "; ".join(f"`{key}`={_fmt_metric_value(value)}" for key, value in items)


def _write_markdown(path: Path, entries: list[LedgerEntry], args: argparse.Namespace) -> None:
    generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    report_count = sum(1 for entry in entries if entry.source_kind == "report")
    output_count = sum(1 for entry in entries if entry.source_kind == "output_json")
    lines: list[str] = [
        "# Circleworld Experiment Ledger - 2026-05-13",
        "",
        f"- Schema: `{SCHEMA}`",
        f"- Generated at: `{generated_at}`",
        f"- Report scan root: `{args.reports_dir}`",
        f"- Output scan root: `{args.outputs_dir}`",
        f"- JSONL sidecar: `{args.output_jsonl}`",
        f"- Since: `{args.since}`",
        f"- Entries: `{len(entries)}` total, `{report_count}` report-derived, `{output_count}` output-only",
        "",
        "## Scope",
        "",
        "This ledger reconstructs recent Circleworld experiment claims from Markdown reports and summary JSON artifacts. It is an index and interpretation aid, not a new evaluator result.",
        "",
        "## Summary Table",
        "",
        "| date | source | claim/test | key metrics | artifacts |",
        "| --- | --- | --- | --- | ---: |",
    ]
    for entry in entries:
        lines.append(
            "| "
            + " | ".join(
                [
                    entry.date,
                    entry.source_kind,
                    _shorten(entry.claim_test, 140).replace("|", "\\|"),
                    _metrics_inline(entry.metrics, 4).replace("|", "\\|"),
                    str(len(entry.artifact_paths)),
                ]
            )
            + " |"
        )
    lines.extend(["", "## Ledger Entries", ""])
    for idx, entry in enumerate(entries, start=1):
        lines.extend(
            [
                f"### {idx}. {entry.date} - {_shorten(entry.claim_test, 120)}",
                "",
                f"- Source kind: `{entry.source_kind}`",
                f"- Source path: `{entry.source_path}`",
                f"- Claim/test: {_shorten(entry.claim_test, 900)}",
                f"- Interpretation: {_shorten(entry.interpretation, 900)}",
                f"- Limitations: {_shorten(entry.limitations, 900)}",
                f"- Next action: {_shorten(entry.next_action, 900)}",
                "- Metrics:",
            ]
        )
        if entry.metrics:
            for key, value in list(entry.metrics.items())[:18]:
                lines.append(f"  - `{key}`: `{_fmt_metric_value(value)}`")
        else:
            lines.append("  - none reconstructed")
        lines.append("- Artifact paths:")
        if entry.artifact_paths:
            for artifact in entry.artifact_paths[:12]:
                lines.append(f"  - `{artifact}`")
        else:
            lines.append("  - none reconstructed")
        lines.append("")
    lines.extend(
        [
            "## JSONL Reconstruction",
            "",
            "Each line below is a compact JSON object with the required ledger fields.",
            "",
            "```jsonl",
        ]
    )
    for entry in entries:
        lines.append(json.dumps(entry.as_jsonable(), ensure_ascii=False, sort_keys=True))
    lines.extend(
        [
            "```",
            "",
            "## Verification Notes",
            "",
            "- The utility reads existing Markdown reports and JSON artifacts only.",
            "- The utility writes this Markdown file and a JSONL sidecar by default.",
            "- Metrics are best-effort extractions from report tables, inline metric lines, and linked JSON artifacts.",
            "- Output-only entries are intentionally marked as mechanically extracted when no paired report narrative is found.",
            "",
            "## Risks",
            "",
            "- Heuristic parsing can miss metrics embedded in prose, unusual table shapes, or large JSON files above the configured byte limit.",
            "- Report-derived interpretation may lag later artifacts if a report was not updated after follow-up runs.",
            "- Output-only entries should not be treated as supported claims until reviewed against source reports and code provenance.",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, entries: list[LedgerEntry]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for entry in entries:
            handle.write(json.dumps(entry.as_jsonable(), ensure_ascii=False, sort_keys=True) + "\n")


def main() -> int:
    args = _parse_args()
    entries = _build_entries(args)
    _write_markdown(args.output_md, entries, args)
    _write_jsonl(args.output_jsonl, entries)
    if args.jsonl_stdout:
        for entry in entries:
            print(json.dumps(entry.as_jsonable(), ensure_ascii=False, sort_keys=True))
    print(f"Wrote {len(entries)} ledger entries to {args.output_md} and {args.output_jsonl}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
