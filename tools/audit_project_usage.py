from __future__ import annotations

import argparse
import ast
import json
import os
import re
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import audit_project_scripts as script_audit


ARTIFACT_ROOT_NAMES = {
    "archive",
    "archive_checkpoints",
    "artifacts",
    "eval",
    "logs",
    "outputs",
    "validation_results",
}
ARTIFACT_ROOT_PREFIXES = ("checkpoints",)
SOURCE_TEXT_ROOTS = {
    "core",
    "docs",
    "foundation",
    "lineages",
    "registries",
    "research_track",
    "runtimes",
    "tests",
    "tools",
}
TEXT_EXTENSIONS = {
    ".bat",
    ".cfg",
    ".csv",
    ".ini",
    ".json",
    ".jsonl",
    ".md",
    ".ps1",
    ".py",
    ".sh",
    ".toml",
    ".txt",
    ".yaml",
    ".yml",
}
ARTIFACT_TEXT_EXTENSIONS = {
    ".csv",
    ".html",
    ".json",
    ".jsonl",
    ".md",
    ".txt",
    ".yaml",
    ".yml",
}
SCRIPT_EXTENSIONS = script_audit.SCRIPT_EXTENSIONS
SELF_GENERATED_USAGE_FILES = {
    "docs/architecture/PROJECT_USAGE_MAP.json",
    "docs/architecture/PROJECT_USAGE_MAP.md",
    "docs/architecture/PROJECT_ARTIFACT_INVENTORY.md",
    "docs/reports/RAFA_PROJECT_USAGE_AUDIT_2026-05-19.md",
}
PATH_RE = re.compile(
    r"(?:(?:[A-Za-z]:)?[\\/])?(?:outputs|artifacts|archive_checkpoints|archive|eval|logs|validation_results|docs[\\/]reports|checkpoints[\w-]*)[\\/][A-Za-z0-9_.@=+\\/\-]+"
)
ALIAS_ROOT_TO_PHYSICAL_PREFIX = {
    "outputs": "artifacts/runtime/outputs",
    "eval": "artifacts/runtime/eval",
    "logs": "artifacts/runtime/logs",
    "validation_results": "artifacts/runtime/validation_results",
    "tmp": "artifacts/runtime/tmp",
    "reap": "artifacts/runtime/reap",
    "archive_checkpoints": "artifacts/checkpoints/archive/archive_checkpoints",
    "checkpoints": "artifacts/checkpoints/archive/checkpoints",
}


@dataclass
class TextReference:
    path: str
    kind: str
    mention_count: int


@dataclass
class ScriptUsage:
    path: str
    lane: str
    role: str
    lifecycle: str
    git_tracked: bool
    nonblank_loc: int
    registered_or_manifested: bool
    has_main_guard: bool
    import_roots: list[str]
    imported_by: list[str] = field(default_factory=list)
    text_referenced_by: list[TextReference] = field(default_factory=list)
    artifact_groups_mentioned: list[str] = field(default_factory=list)
    use_status: str = "unknown"
    use_confidence: str = "low"


@dataclass
class ArtifactRecord:
    path: str
    root: str
    group: str
    kind: str
    extension: str
    bytes: int
    mtime_utc: str | None
    direct_reference_count: int
    group_reference_count: int


@dataclass
class ArtifactGroup:
    group: str
    root: str
    kind_counts: dict[str, int]
    extension_counts: dict[str, int]
    file_count: int
    total_bytes: int
    newest_mtime_utc: str | None
    direct_reference_count: int
    group_reference_count: int
    referenced_by: list[TextReference] = field(default_factory=list)
    producer_script_candidates: list[str] = field(default_factory=list)
    status: str = "unknown"


def repo_root_from_script() -> Path:
    return Path(__file__).resolve().parents[1]


def normalize_rel(path: Path, root: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def path_kind(rel_path: str) -> str:
    if rel_path.startswith("docs/") and rel_path.endswith(".md"):
        return "docs"
    if rel_path.startswith("docs/reports/"):
        return "report"
    if rel_path.startswith("registries/") or rel_path.endswith("runtime_manifest.json"):
        return "registry_or_manifest"
    if Path(rel_path).suffix.lower() in SCRIPT_EXTENSIONS:
        return "code"
    return "text"


def is_artifact_root_name(name: str) -> bool:
    return name in ARTIFACT_ROOT_NAMES or any(name.startswith(prefix) for prefix in ARTIFACT_ROOT_PREFIXES)


def is_under_artifact_root(rel_path: str) -> bool:
    first = rel_path.split("/", 1)[0]
    if rel_path.startswith("docs/reports/"):
        return True
    return is_artifact_root_name(first)


def safe_read(path: Path, max_bytes: int = 2_000_000) -> str | None:
    try:
        if path.stat().st_size > max_bytes:
            return None
        return path.read_text(encoding="utf-8-sig", errors="replace")
    except OSError:
        return None


def iso_mtime(path: Path) -> str | None:
    try:
        return datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc).isoformat(timespec="seconds")
    except OSError:
        return None


def artifact_kind(path: Path) -> str:
    suffix = path.suffix.lower()
    name = path.name.lower()
    if suffix in {".pt", ".pth", ".ckpt", ".safetensors"}:
        return "checkpoint"
    if suffix in {".wav", ".flac", ".mp3", ".ogg"}:
        return "audio"
    if suffix in {".json", ".jsonl"}:
        return "json"
    if suffix in {".md", ".html"}:
        return "report"
    if suffix in {".png", ".jpg", ".jpeg", ".gif", ".svg", ".webp"}:
        return "image"
    if suffix in {".npy", ".npz", ".pkl", ".pickle", ".parquet"}:
        return "data"
    if suffix in {".csv", ".tsv"}:
        return "table"
    if suffix in {".log", ".txt"} or "log" in name:
        return "log"
    if suffix in {".yaml", ".yml", ".ini", ".cfg"}:
        return "config"
    return "other"


def artifact_group_for(rel_path: str) -> str:
    parts = rel_path.split("/")
    if rel_path.startswith("docs/reports/"):
        return "docs/reports"
    if not parts:
        return rel_path
    root = parts[0]
    if root in ALIAS_ROOT_TO_PHYSICAL_PREFIX:
        physical = ALIAS_ROOT_TO_PHYSICAL_PREFIX[root]
        rest = "/".join(parts[1:])
        return artifact_group_for(f"{physical}/{rest}" if rest else physical)
    if root.startswith("checkpoints_stage"):
        return f"artifacts/checkpoints/stages/{root}"
    if root.startswith("checkpoints_diffusion"):
        return f"artifacts/checkpoints/runs/{root}"
    if root == "outputs":
        return "/".join(parts[:3]) if len(parts) >= 3 else "/".join(parts)
    if root == "artifacts":
        if len(parts) >= 3 and parts[1] == "runtime" and parts[2] == "outputs":
            return "/".join(parts[:5]) if len(parts) >= 5 else "/".join(parts[:4])
        if len(parts) >= 3 and parts[1] == "runtime":
            return "/".join(parts[:4]) if len(parts) >= 4 else "/".join(parts)
        if len(parts) >= 3 and parts[1] == "checkpoints":
            return "/".join(parts[:4]) if len(parts) >= 4 else "/".join(parts)
        return "/".join(parts[:3]) if len(parts) >= 3 else "/".join(parts)
    if root == "checkpoints_circleworld_proto":
        return "/".join(parts[:2]) if len(parts) >= 2 else root
    if root.startswith("checkpoints"):
        return root
    if root == "archive_checkpoints":
        return "/".join(parts[:2]) if len(parts) >= 2 else root
    if root in {"eval", "logs", "validation_results", "archive"}:
        return "/".join(parts[:2]) if len(parts) >= 2 else root
    return "/".join(parts[:2]) if len(parts) >= 2 else root


def collect_visible_files(root: Path) -> list[Path]:
    files = script_audit.list_git_visible_files(root)
    # Add ignored artifact roots deliberately. Avoid input datasets and wav corpus.
    for child in root.iterdir():
        if not child.is_dir():
            continue
        if not is_artifact_root_name(child.name):
            continue
        for current_root, dirnames, filenames in os.walk(child, followlinks=False):
            dirnames[:] = [
                name
                for name in dirnames
                if name not in {".git", "__pycache__", ".pytest_cache"}
            ]
            current = Path(current_root)
            for filename in filenames:
                files.append(current / filename)
    # docs/reports is usually git-visible, but include it explicitly for ignored report bursts.
    reports = root / "docs" / "reports"
    if reports.exists():
        for current_root, _, filenames in os.walk(reports, followlinks=False):
            current = Path(current_root)
            for filename in filenames:
                files.append(current / filename)
    unique: dict[str, Path] = {}
    for path in files:
        if path.is_file():
            try:
                rel = normalize_rel(path, root)
            except ValueError:
                continue
            unique[rel] = path
    return [unique[key] for key in sorted(unique)]


def collect_text_sources(root: Path, visible_files: list[Path]) -> dict[str, str]:
    texts: dict[str, str] = {}
    for path in visible_files:
        rel = normalize_rel(path, root)
        if rel in script_audit.SELF_GENERATED_REFERENCE_FILES or rel in SELF_GENERATED_USAGE_FILES:
            continue
        first = rel.split("/", 1)[0]
        suffix = path.suffix.lower()
        if first not in SOURCE_TEXT_ROOTS and "/" in rel:
            continue
        if is_under_artifact_root(rel) and not rel.startswith("docs/reports/"):
            continue
        if suffix not in TEXT_EXTENSIONS:
            continue
        text = safe_read(path)
        if text is None:
            continue
        texts[rel] = text
    return texts


def normalize_mentioned_path(raw: str) -> str:
    value = raw.replace("\\", "/")
    value = re.sub(r"^[A-Za-z]:/RAFA/", "", value, flags=re.IGNORECASE)
    value = value.lstrip("/")
    value = value.rstrip(".,;:)'\"`]>}")
    return value


def collect_artifact_path_mentions(texts: dict[str, str]) -> dict[str, Counter[str]]:
    mentions: dict[str, Counter[str]] = {}
    for source, text in texts.items():
        counter: Counter[str] = Counter()
        for match in PATH_RE.finditer(text):
            value = normalize_mentioned_path(match.group(0))
            if value:
                counter[value] += 1
        if counter:
            mentions[source] = counter
    return mentions


def count_text_references(
    target_path: str,
    target_name: str,
    texts: dict[str, str],
    exclude_self: bool = True,
) -> list[TextReference]:
    refs: list[TextReference] = []
    backslash = target_path.replace("/", "\\")
    for source, text in texts.items():
        if exclude_self and source == target_path:
            continue
        count = text.count(target_path) + text.count(backslash)
        if count == 0 and target_name:
            # Basename-only hits are weak, but useful for legacy scripts and reports.
            count = text.count(target_name)
        if count:
            refs.append(TextReference(source, path_kind(source), count))
    refs.sort(key=lambda row: (row.kind, row.path))
    return refs


def module_name_for(rel_path: str) -> str | None:
    if not rel_path.endswith(".py"):
        return None
    no_suffix = rel_path[:-3]
    if no_suffix.endswith("/__init__"):
        no_suffix = no_suffix[: -len("/__init__")]
    return no_suffix.replace("/", ".")


def imported_modules_from_py(text: str) -> set[str]:
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return set()
    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                modules.add(alias.name)
                modules.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                modules.add(node.module)
                modules.add(node.module.split(".")[0])
    return modules


def build_import_graph(records: list[dict[str, Any]], texts: dict[str, str]) -> dict[str, list[str]]:
    module_to_path: dict[str, set[str]] = defaultdict(set)
    stem_to_path: dict[str, set[str]] = defaultdict(set)
    for row in records:
        path = row["path"]
        module = module_name_for(path)
        if module:
            module_to_path[module].add(path)
            stem_to_path[Path(path).stem].add(path)

    imported_by: dict[str, set[str]] = defaultdict(set)
    for source, text in texts.items():
        if not source.endswith(".py"):
            continue
        modules = imported_modules_from_py(text)
        for module in modules:
            for target in module_to_path.get(module, set()):
                if target != source:
                    imported_by[target].add(source)
            if "." not in module:
                for target in stem_to_path.get(module, set()):
                    if target != source:
                        imported_by[target].add(source)
    return {path: sorted(sources) for path, sources in imported_by.items()}


def status_for_script(
    row: dict[str, Any],
    imported_by: list[str],
    text_refs: list[TextReference],
) -> tuple[str, str]:
    if row["registered_or_manifested"]:
        return "canonical_or_manifested", "high"
    if imported_by:
        return "imported_dependency", "high"
    if any(ref.kind in {"registry_or_manifest", "report", "docs"} for ref in text_refs):
        return "documented_reference", "medium"
    if row["role"] == "contract_test":
        return "test_or_contract", "medium"
    if row["has_main_guard"] and row["docs_reference_count"] > 0:
        return "documented_cli", "medium"
    if not row["git_tracked"] and row["lifecycle"] in {"active_unregistered_research", "unclassified"}:
        return "untracked_low_reference_candidate", "low"
    return "low_reference_or_legacy", "low"


def build_script_usage(
    inventory: dict[str, Any],
    texts: dict[str, str],
    artifact_mentions: dict[str, Counter[str]],
) -> list[ScriptUsage]:
    records = inventory["scripts"]
    imported_by_map = build_import_graph(records, texts)
    usages: list[ScriptUsage] = []
    for row in records:
        path = row["path"]
        text_refs = count_text_references(path, Path(path).name, texts)
        imported_by = imported_by_map.get(path, [])
        mentioned_groups = sorted(
            {
                artifact_group_for(path_value)
                for counter in artifact_mentions.values()
                for path_value in counter
                if path_value in texts.get(path, "")
            }
        )
        status, confidence = status_for_script(row, imported_by, text_refs)
        usages.append(
            ScriptUsage(
                path=path,
                lane=row["lane"],
                role=row["role"],
                lifecycle=row["lifecycle"],
                git_tracked=row["git_tracked"],
                nonblank_loc=row["nonblank_loc"],
                registered_or_manifested=row["registered_or_manifested"],
                has_main_guard=row["has_main_guard"],
                import_roots=row["import_roots"],
                imported_by=imported_by,
                text_referenced_by=text_refs[:50],
                artifact_groups_mentioned=mentioned_groups,
                use_status=status,
                use_confidence=confidence,
            )
        )
    return usages


def build_artifact_inventory(
    root: Path,
    visible_files: list[Path],
    texts: dict[str, str],
    artifact_mentions: dict[str, Counter[str]],
    max_file_records: int,
) -> tuple[list[ArtifactRecord], list[ArtifactGroup]]:
    direct_counts: Counter[str] = Counter()
    source_refs_by_path: dict[str, list[TextReference]] = defaultdict(list)
    for source, counter in artifact_mentions.items():
        for mentioned, count in counter.items():
            direct_counts[mentioned] += count
            source_refs_by_path[mentioned].append(TextReference(source, path_kind(source), count))

    artifact_paths = []
    for path in visible_files:
        rel = normalize_rel(path, root)
        if is_under_artifact_root(rel):
            artifact_paths.append(path)

    group_reference_counts: Counter[str] = Counter()
    group_refs: dict[str, list[TextReference]] = defaultdict(list)
    mentioned_paths = list(direct_counts.keys())
    for mentioned in mentioned_paths:
        group = artifact_group_for(mentioned)
        group_reference_counts[group] += direct_counts[mentioned]
        group_refs[group].extend(source_refs_by_path.get(mentioned, []))

    group_acc: dict[str, dict[str, Any]] = {}
    records: list[ArtifactRecord] = []
    sorted_paths = sorted(artifact_paths, key=lambda value: normalize_rel(value, root))
    for idx, path in enumerate(sorted_paths):
        rel = normalize_rel(path, root)
        group = artifact_group_for(rel)
        root_name = rel.split("/", 1)[0]
        kind = artifact_kind(path)
        suffix = path.suffix.lower() or "<none>"
        try:
            stat = path.stat()
            size = int(stat.st_size)
            mtime = datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat(timespec="seconds")
        except OSError:
            size = 0
            mtime = None
        direct_count = direct_counts[rel] + direct_counts[rel.replace("/", "\\")]
        group_count = group_reference_counts[group]
        acc = group_acc.setdefault(
            group,
            {
                "root": root_name,
                "kind_counts": Counter(),
                "extension_counts": Counter(),
                "file_count": 0,
                "total_bytes": 0,
                "newest_mtime_utc": None,
                "direct_reference_count": 0,
                "group_reference_count": group_count,
            },
        )
        acc["kind_counts"][kind] += 1
        acc["extension_counts"][suffix] += 1
        acc["file_count"] += 1
        acc["total_bytes"] += size
        acc["direct_reference_count"] += direct_count
        if mtime and (acc["newest_mtime_utc"] is None or mtime > acc["newest_mtime_utc"]):
            acc["newest_mtime_utc"] = mtime
        if max_file_records <= 0 or idx < max_file_records:
            records.append(
                ArtifactRecord(
                    path=rel,
                    root=root_name,
                    group=group,
                    kind=kind,
                    extension=suffix,
                    bytes=size,
                    mtime_utc=mtime,
                    direct_reference_count=direct_count,
                    group_reference_count=group_count,
                )
            )

    script_texts = {path: text for path, text in texts.items() if path.endswith(".py")}
    groups: list[ArtifactGroup] = []
    for group, acc in group_acc.items():
        producer_candidates = []
        group_name = group.split("/")[-1]
        for script_path, text in script_texts.items():
            if group in text or group.replace("/", "\\") in text or group_name in text:
                producer_candidates.append(script_path)
        refs = group_refs.get(group, [])
        if acc["group_reference_count"] or acc["direct_reference_count"]:
            status = "referenced"
        elif producer_candidates:
            status = "producer_name_candidate_only"
        elif group.startswith("docs/reports"):
            status = "report_archive"
        else:
            status = "cold_unreferenced_candidate"
        groups.append(
            ArtifactGroup(
                group=group,
                root=acc["root"],
                kind_counts=dict(acc["kind_counts"].most_common()),
                extension_counts=dict(acc["extension_counts"].most_common()),
                file_count=acc["file_count"],
                total_bytes=acc["total_bytes"],
                newest_mtime_utc=acc["newest_mtime_utc"],
                direct_reference_count=acc["direct_reference_count"],
                group_reference_count=acc["group_reference_count"],
                referenced_by=refs[:50],
                producer_script_candidates=sorted(set(producer_candidates))[:50],
                status=status,
            )
        )
    groups.sort(key=lambda row: (-row.total_bytes, row.group))
    return records, groups


def compact_refs(refs: list[TextReference], limit: int = 5) -> str:
    if not refs:
        return ""
    parts = [f"{ref.path} ({ref.mention_count})" for ref in refs[:limit]]
    if len(refs) > limit:
        parts.append(f"+{len(refs) - limit} more")
    return "; ".join(parts)


def table_rows(rows: list[list[Any]]) -> str:
    if not rows:
        return ""
    widths = [0 for _ in rows[0]]
    for row in rows:
        for idx, value in enumerate(row):
            widths[idx] = max(widths[idx], len(str(value)))
    lines = []
    for row_idx, row in enumerate(rows):
        lines.append("| " + " | ".join(str(value).ljust(widths[idx]) for idx, value in enumerate(row)) + " |")
        if row_idx == 0:
            lines.append("| " + " | ".join("-" * widths[idx] for idx in range(len(widths))) + " |")
    return "\n".join(lines)


def summarize_usage(
    scripts: list[ScriptUsage],
    artifact_records: list[ArtifactRecord],
    artifact_groups: list[ArtifactGroup],
) -> dict[str, Any]:
    return {
        "script_count": len(scripts),
        "artifact_file_records_written": len(artifact_records),
        "artifact_group_count": len(artifact_groups),
        "artifact_total_files": sum(group.file_count for group in artifact_groups),
        "artifact_total_bytes": sum(group.total_bytes for group in artifact_groups),
        "script_use_status": dict(Counter(row.use_status for row in scripts).most_common()),
        "script_lanes": dict(Counter(row.lane for row in scripts).most_common()),
        "artifact_roots": dict(Counter(group.root for group in artifact_groups).most_common()),
        "artifact_group_status": dict(Counter(group.status for group in artifact_groups).most_common()),
        "artifact_kinds": dict(
            Counter(
                kind
                for group in artifact_groups
                for kind, count in group.kind_counts.items()
                for _ in range(count)
            ).most_common()
        ),
        "largest_artifact_groups": [
            {
                "group": group.group,
                "root": group.root,
                "file_count": group.file_count,
                "total_bytes": group.total_bytes,
                "status": group.status,
            }
            for group in artifact_groups[:25]
        ],
    }


def write_usage_markdown(path: Path, data: dict[str, Any]) -> None:
    summary = data["summary"]
    scripts = [ScriptUsage(**row) for row in data["scripts"]]
    groups = [ArtifactGroup(**row) for row in data["artifact_groups"]]
    for script in scripts:
        script.text_referenced_by = [
            TextReference(**ref) if isinstance(ref, dict) else ref for ref in script.text_referenced_by
        ]
    for group in groups:
        group.referenced_by = [
            TextReference(**ref) if isinstance(ref, dict) else ref for ref in group.referenced_by
        ]

    low_ref_scripts = [
        row
        for row in scripts
        if row.use_status in {"untracked_low_reference_candidate", "low_reference_or_legacy"}
    ]
    low_ref_scripts.sort(key=lambda row: (-row.nonblank_loc, row.path))
    imported = [row for row in scripts if row.imported_by]
    imported.sort(key=lambda row: (-len(row.imported_by), row.path))
    referenced_groups = [row for row in groups if row.status == "referenced"]
    referenced_groups.sort(key=lambda row: (-row.group_reference_count, -row.direct_reference_count, row.group))
    cold_groups = [row for row in groups if row.status == "cold_unreferenced_candidate"]
    cold_groups.sort(key=lambda row: (-row.total_bytes, row.group))

    lines = [
        "# Project Usage Map",
        "",
        f"Generated: `{data['generated_at']}`",
        "",
        "This is a conservative static usage map. It records imports, text references, manifest/registry visibility, artifact groups, and cold candidates. It does not prove deletion safety.",
        "",
        "Top-level artifact aliases such as `outputs/`, `eval/`, `logs/`, and many `checkpoints_*` roots are normalized to their consolidated physical groups under `artifacts/` when possible.",
        "",
        "## Summary",
        "",
        table_rows(
            [
                ["Metric", "Value"],
                ["Scripts", summary["script_count"]],
                ["Artifact files indexed", summary["artifact_total_files"]],
                ["Artifact groups", summary["artifact_group_count"]],
                ["Artifact bytes", summary["artifact_total_bytes"]],
            ]
        ),
        "",
        "## Script Use Status",
        "",
        table_rows([["Status", "Scripts"], *summary["script_use_status"].items()]),
        "",
        "## Artifact Group Status",
        "",
        table_rows([["Status", "Groups"], *summary["artifact_group_status"].items()]),
        "",
        "## Most Imported Scripts",
        "",
        table_rows(
            [
                ["Script", "Imported by", "Lane", "Role"],
                *[
                    [f"`{row.path}`", len(row.imported_by), row.lane, row.role]
                    for row in imported[:40]
                ],
            ]
        ),
        "",
        "## Low-Reference Script Candidates",
        "",
        "These are not deletion instructions. They are the scripts to inspect first during cleanup.",
        "",
        table_rows(
            [
                ["Script", "Status", "Lane", "Role", "LOC", "Tracked"],
                *[
                    [f"`{row.path}`", row.use_status, row.lane, row.role, row.nonblank_loc, row.git_tracked]
                    for row in low_ref_scripts[:80]
                ],
            ]
        ),
        "",
        "## Most Referenced Artifact Groups",
        "",
        table_rows(
            [
                ["Group", "Refs", "Files", "Bytes", "Kinds"],
                *[
                    [
                        f"`{row.group}`",
                        row.group_reference_count + row.direct_reference_count,
                        row.file_count,
                        row.total_bytes,
                        ", ".join(f"{k}:{v}" for k, v in list(row.kind_counts.items())[:4]),
                    ]
                    for row in referenced_groups[:50]
                ],
            ]
        ),
        "",
        "## Largest Cold Artifact Groups",
        "",
        "Cold means this static scan did not find a source/doc/registry/report reference to the group. It may still be historically important.",
        "",
        table_rows(
            [
                ["Group", "Files", "Bytes", "Newest", "Kinds"],
                *[
                    [
                        f"`{row.group}`",
                        row.file_count,
                        row.total_bytes,
                        row.newest_mtime_utc or "",
                        ", ".join(f"{k}:{v}" for k, v in list(row.kind_counts.items())[:4]),
                    ]
                    for row in cold_groups[:80]
                ],
            ]
        ),
        "",
        "## Regeneration",
        "",
        "```powershell",
        "python tools/audit_project_scripts.py",
        "python tools/audit_project_usage.py",
        "```",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def write_artifact_markdown(path: Path, data: dict[str, Any]) -> None:
    summary = data["summary"]
    groups = [ArtifactGroup(**row) for row in data["artifact_groups"]]
    groups.sort(key=lambda row: (row.root, row.group))
    lines = [
        "# Project Artifact Inventory",
        "",
        f"Generated: `{data['generated_at']}`",
        "",
        "Artifact files are grouped by run/checkpoint/report root so the map is readable. Per-file records are stored in `PROJECT_USAGE_MAP.json`.",
        "",
        "Alias roots are normalized conservatively: for example, `outputs/<family>/<run>` maps to `artifacts/runtime/outputs/<family>/<run>` and `checkpoints_stage4` maps to `artifacts/checkpoints/stages/checkpoints_stage4`.",
        "",
        "## Summary",
        "",
        table_rows(
            [
                ["Metric", "Value"],
                ["Artifact files", summary["artifact_total_files"]],
                ["Artifact groups", summary["artifact_group_count"]],
                ["Artifact bytes", summary["artifact_total_bytes"]],
            ]
        ),
        "",
        "## By Root",
        "",
        table_rows([["Root", "Groups"], *summary["artifact_roots"].items()]),
        "",
        "## Groups",
        "",
        table_rows(
            [
                ["Group", "Root", "Status", "Files", "Bytes", "Kinds"],
                *[
                    [
                        f"`{row.group}`",
                        row.root,
                        row.status,
                        row.file_count,
                        row.total_bytes,
                        ", ".join(f"{k}:{v}" for k, v in list(row.kind_counts.items())[:4]),
                    ]
                    for row in groups
                ],
            ]
        ),
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def write_report(path: Path, data: dict[str, Any]) -> None:
    summary = data["summary"]
    cold = [
        group
        for group in data["artifact_groups"]
        if group["status"] == "cold_unreferenced_candidate"
    ]
    low_scripts = [
        script
        for script in data["scripts"]
        if script["use_status"] in {"untracked_low_reference_candidate", "low_reference_or_legacy"}
    ]
    lines = [
        "# RAFA Project Usage Audit - 2026-05-19",
        "",
        "## What This Adds",
        "",
        "This pass extends script cartography into a static usage map over scripts and artifact groups.",
        "",
        "It normalizes common reparse/alias roots into the consolidated artifact tree, so top-level `outputs`, `eval`, `logs`, and checkpoint aliases are tracked without double-counting the same physical files.",
        "",
        "It answers:",
        "",
        "- Which scripts are imported, manifested, documented, or low-reference?",
        "- Which artifact groups are referenced by source/docs/registries/reports?",
        "- Which artifact groups are large and cold according to static references?",
        "- Where should cleanup start without touching model semantics?",
        "",
        "## Headline Counts",
        "",
        table_rows(
            [
                ["Metric", "Value"],
                ["Scripts mapped", summary["script_count"]],
                ["Artifact files indexed", summary["artifact_total_files"]],
                ["Artifact groups", summary["artifact_group_count"]],
                ["Artifact bytes", summary["artifact_total_bytes"]],
                ["Low-reference script candidates", len(low_scripts)],
                ["Cold artifact group candidates", len(cold)],
            ]
        ),
        "",
        "## Interpretation",
        "",
        "The project has two different cleanup problems:",
        "",
        "1. Script sprawl: many one-off Circleworld scripts are real experimental history but need helper extraction and lifecycle labels.",
        "2. Artifact sprawl: many output/checkpoint groups are large and not statically referenced; they need archival decisions, not blind deletion.",
        "",
        "The safest next move remains no-behavior-change consolidation of shared script helpers. Artifact cleanup should come later, after checkpoint/output groups are tied to reports or marked as expendable caches.",
        "",
        "## Files",
        "",
        "- `docs/architecture/PROJECT_USAGE_MAP.md`",
        "- `docs/architecture/PROJECT_USAGE_MAP.json`",
        "- `docs/architecture/PROJECT_ARTIFACT_INVENTORY.md`",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a static RAFA project usage map over scripts and artifact groups."
    )
    parser.add_argument("--root", type=Path, default=repo_root_from_script())
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--json", type=Path, default=None)
    parser.add_argument("--markdown", type=Path, default=None)
    parser.add_argument("--artifacts-md", type=Path, default=None)
    parser.add_argument("--report", type=Path, default=None)
    parser.add_argument(
        "--max-file-records",
        type=int,
        default=0,
        help="Maximum artifact file records to store in JSON; 0 means store all.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = args.root.resolve()
    out_dir = (args.out_dir or root / "docs" / "architecture").resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = (args.json or out_dir / "PROJECT_USAGE_MAP.json").resolve()
    markdown_path = (args.markdown or out_dir / "PROJECT_USAGE_MAP.md").resolve()
    artifacts_md_path = (args.artifacts_md or out_dir / "PROJECT_ARTIFACT_INVENTORY.md").resolve()
    report_path = (
        args.report
        or root / "docs" / "reports" / "RAFA_PROJECT_USAGE_AUDIT_2026-05-19.md"
    ).resolve()

    script_inventory_path = root / "docs" / "architecture" / "PROJECT_SCRIPT_INVENTORY.json"
    if not script_inventory_path.exists():
        raise FileNotFoundError(
            f"Missing {script_inventory_path}; run tools/audit_project_scripts.py first."
        )
    inventory = json.loads(script_inventory_path.read_text(encoding="utf-8"))
    visible_files = collect_visible_files(root)
    texts = collect_text_sources(root, visible_files)
    artifact_mentions = collect_artifact_path_mentions(texts)

    script_usage = build_script_usage(inventory, texts, artifact_mentions)
    artifact_records, artifact_groups = build_artifact_inventory(
        root,
        visible_files,
        texts,
        artifact_mentions,
        max_file_records=args.max_file_records,
    )
    summary = summarize_usage(script_usage, artifact_records, artifact_groups)
    data = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "root": str(root),
        "scope": {
            "artifact_roots": sorted(ARTIFACT_ROOT_NAMES),
            "artifact_root_prefixes": list(ARTIFACT_ROOT_PREFIXES),
            "text_reference_roots": sorted(SOURCE_TEXT_ROOTS),
            "max_file_records": args.max_file_records,
            "deletion_authorized": False,
            "static_analysis_only": True,
        },
        "summary": summary,
        "scripts": [asdict(row) for row in script_usage],
        "artifact_files": [asdict(row) for row in artifact_records],
        "artifact_groups": [asdict(row) for row in artifact_groups],
    }
    json_path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")
    write_usage_markdown(markdown_path, data)
    write_artifact_markdown(artifacts_md_path, data)
    write_report(report_path, data)
    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"Wrote {json_path}")
    print(f"Wrote {markdown_path}")
    print(f"Wrote {artifacts_md_path}")
    print(f"Wrote {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
