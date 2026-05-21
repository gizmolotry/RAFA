from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import os
import re
import subprocess
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


SCRIPT_EXTENSIONS = {".py", ".ps1", ".bat", ".sh", ".ipynb"}
REFERENCE_EXTENSIONS = {
    ".md",
    ".json",
    ".jsonl",
    ".yaml",
    ".yml",
    ".toml",
    ".txt",
}
EXCLUDED_DIRS = {
    ".git",
    ".pytest_cache",
    "__pycache__",
    "artifacts",
    "archive_checkpoints",
    "checkpoints",
    "checkpoints_circleworld_proto",
    "data",
    "datasets",
    "eval",
    "input",
    "logs",
    "outputs",
    "reap",
    "scrapes",
    "tmp",
    "validation_results",
    "wav_files",
}
CLI_FUNCTION_NAMES = {
    "main",
    "parse_args",
    "build_parser",
    "add_arguments",
    "cli",
}
METHOD_NOISE_NAMES = {
    "__init__",
    "__post_init__",
    "forward",
    "setup",
    "teardown",
}
TEXT_SCAN_ROOTS = {
    "docs",
    "registries",
    "runtimes",
    "lineages",
    "tests",
    "tools",
}
SELF_GENERATED_REFERENCE_FILES = {
    "docs/architecture/PROJECT_SCRIPT_INVENTORY.json",
    "docs/architecture/PROJECT_SCRIPT_INVENTORY.md",
    "docs/architecture/PROJECT_REDUNDANCY_HOTSPOTS.md",
    "docs/architecture/PROJECT_SCAFFOLDING_GUIDE.md",
    "docs/architecture/PROJECT_USAGE_MAP.json",
    "docs/architecture/PROJECT_USAGE_MAP.md",
    "docs/architecture/PROJECT_ARTIFACT_INVENTORY.md",
    "docs/reports/RAFA_PROJECT_USAGE_AUDIT_2026-05-19.md",
}


@dataclass
class FunctionInfo:
    name: str
    line: int
    end_line: int | None
    loc: int | None
    body_hash: str


@dataclass
class ScriptRecord:
    path: str
    extension: str
    lane: str
    role: str
    lifecycle: str
    git_tracked: bool
    registered_or_manifested: bool
    docs_reference_count: int
    bytes: int
    loc: int
    nonblank_loc: int
    sha1: str
    parse_error: str | None = None
    module_doc_first_line: str | None = None
    argparse_description: str | None = None
    has_main_guard: bool = False
    import_roots: list[str] = field(default_factory=list)
    top_level_functions: list[FunctionInfo] = field(default_factory=list)
    classes: list[str] = field(default_factory=list)
    top_level_constants: list[str] = field(default_factory=list)


def repo_root_from_script() -> Path:
    return Path(__file__).resolve().parents[1]


def normalize_rel(path: Path, root: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def run_git(root: Path, args: list[str]) -> list[str] | None:
    try:
        output = subprocess.check_output(
            ["git", *args],
            cwd=root,
            text=True,
            stderr=subprocess.DEVNULL,
        )
    except Exception:
        return None
    return [line.strip() for line in output.splitlines() if line.strip()]


def list_git_visible_files(root: Path) -> list[Path]:
    git_files = run_git(root, ["ls-files", "--cached", "--others", "--exclude-standard"])
    if git_files is not None:
        return [root / line for line in git_files if (root / line).is_file()]

    files: list[Path] = []
    for current_root, dirnames, filenames in os.walk(root):
        current = Path(current_root)
        dirnames[:] = [
            name
            for name in dirnames
            if name not in EXCLUDED_DIRS and not name.startswith("ROOT_")
        ]
        for filename in filenames:
            files.append(current / filename)
    return files


def list_tracked_files(root: Path) -> set[str]:
    tracked = run_git(root, ["ls-files"])
    if tracked is None:
        return set()
    return {line.replace("\\", "/") for line in tracked}


def safe_read(path: Path) -> str:
    with path.open("r", encoding="utf-8-sig", errors="replace") as handle:
        return handle.read()


def sha1_text(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8", errors="replace")).hexdigest()


def first_sentence(text: str | None, max_len: int = 160) -> str | None:
    if not text:
        return None
    line = " ".join(text.strip().split())
    if not line:
        return None
    if len(line) <= max_len:
        return line
    return line[: max_len - 3].rstrip() + "..."


def script_lane(rel_path: str) -> str:
    parts = rel_path.split("/")
    if rel_path.startswith("runtimes/circleworld_proto/"):
        return "circleworld_proto"
    if rel_path.startswith("runtimes/stage4_blackwell_14/"):
        return "stage4_blackwell_14"
    if rel_path.startswith("runtimes/stage4_blackwell_16/"):
        return "stage4_blackwell_16"
    if rel_path.startswith("runtimes/diffusion_parent_v3/"):
        return "diffusion_parent_v3"
    if rel_path.startswith("runtimes/"):
        return "runtime_other"
    if rel_path.startswith("lineages/04_positive_replacement/"):
        return "positive_replacement"
    if rel_path.startswith("lineages/01_parent_graduation/"):
        return "parent_graduation"
    if rel_path.startswith("lineages/02_local_ablations/"):
        return "local_ablations"
    if rel_path.startswith("lineages/03_subtractive_fork/"):
        return "subtractive_fork"
    if rel_path.startswith("core/"):
        return "core_shared"
    if rel_path.startswith("foundation/"):
        return "foundation"
    if rel_path.startswith("tools/"):
        return "tools"
    if rel_path.startswith("research_track/"):
        return "research_track"
    if rel_path.startswith("tests/"):
        return "repo_tests"
    if rel_path.startswith("inference_package/"):
        return "inference_package"
    if len(parts) == 1:
        return "root_legacy"
    return parts[0]


def script_role(rel_path: str, extension: str) -> str:
    stem = Path(rel_path).stem
    lower = stem.lower()
    if extension == ".ipynb":
        return "notebook"
    if extension in {".ps1", ".bat", ".sh"}:
        return "launcher"
    if lower.startswith("test_") or rel_path.startswith("tests/"):
        return "contract_test"
    if lower in {
        "circleworld",
        "model",
        "diffusion_models",
        "phase_native_ifs",
        "rafa_math_tools",
        "stft_utils",
        "diffusion_utils",
        "resonant_law_objects",
        "phase_native_audio_operators",
        "causal_operator_selector",
    }:
        return "runtime_or_shared_module"
    prefixes = [
        ("train_", "trainer"),
        ("run_", "experiment_runner"),
        ("evaluate_", "evaluator"),
        ("benchmark_", "benchmark"),
        ("score_", "scorer"),
        ("compare_", "comparator"),
        ("assemble_", "report_assembler"),
        ("summarize_", "report_assembler"),
        ("audit_", "auditor"),
        ("validate_", "auditor"),
        ("verify", "auditor"),
        ("build_", "builder"),
        ("export_", "exporter"),
        ("sample_", "exporter"),
        ("infer_", "exporter"),
        ("analyze_", "analysis"),
        ("diagnose_", "analysis"),
        ("classify_", "analysis"),
        ("predeclare_", "analysis"),
        ("sweep_", "sweep"),
        ("package_", "packager"),
    ]
    for prefix, role in prefixes:
        if lower.startswith(prefix):
            return role
    return "script_or_module"


def is_main_guard(node: ast.AST) -> bool:
    if not isinstance(node, ast.If):
        return False
    test = node.test
    if not isinstance(test, ast.Compare):
        return False
    left = test.left
    if not (isinstance(left, ast.Name) and left.id == "__name__"):
        return False
    if len(test.ops) != 1 or not isinstance(test.ops[0], ast.Eq):
        return False
    if len(test.comparators) != 1:
        return False
    comp = test.comparators[0]
    return isinstance(comp, ast.Constant) and comp.value == "__main__"


def call_name(node: ast.AST) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        prefix = call_name(node.value)
        if prefix:
            return f"{prefix}.{node.attr}"
        return node.attr
    return None


def literal_string(node: ast.AST) -> str | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    return None


def extract_argparse_description(tree: ast.Module) -> str | None:
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        name = call_name(node.func)
        if name not in {"ArgumentParser", "argparse.ArgumentParser"}:
            continue
        for keyword in node.keywords:
            if keyword.arg == "description":
                value = literal_string(keyword.value)
                if value:
                    return first_sentence(value, 240)
        if node.args:
            value = literal_string(node.args[0])
            if value:
                return first_sentence(value, 240)
    return None


def import_roots(tree: ast.Module) -> list[str]:
    roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                roots.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                roots.add(node.module.split(".")[0])
    return sorted(roots)


def top_level_constants(tree: ast.Module) -> list[str]:
    constants: list[str] = []
    for node in tree.body:
        targets: Iterable[ast.expr]
        if isinstance(node, ast.Assign):
            targets = node.targets
        elif isinstance(node, ast.AnnAssign):
            targets = [node.target]
        else:
            continue
        for target in targets:
            if isinstance(target, ast.Name) and target.id.isupper():
                constants.append(target.id)
    return sorted(set(constants))


def function_hash(node: ast.FunctionDef | ast.AsyncFunctionDef) -> str:
    clone = ast.dump(node, include_attributes=False)
    return hashlib.sha1(clone.encode("utf-8")).hexdigest()


def analyze_python(text: str) -> dict[str, Any]:
    tree = ast.parse(text)
    functions: list[FunctionInfo] = []
    classes: list[str] = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            end_line = getattr(node, "end_lineno", None)
            loc = end_line - node.lineno + 1 if end_line is not None else None
            functions.append(
                FunctionInfo(
                    name=node.name,
                    line=node.lineno,
                    end_line=end_line,
                    loc=loc,
                    body_hash=function_hash(node),
                )
            )
        elif isinstance(node, ast.ClassDef):
            classes.append(node.name)
    return {
        "module_doc_first_line": first_sentence(ast.get_docstring(tree), 220),
        "argparse_description": extract_argparse_description(tree),
        "has_main_guard": any(is_main_guard(node) for node in tree.body),
        "import_roots": import_roots(tree),
        "top_level_functions": functions,
        "classes": classes,
        "top_level_constants": top_level_constants(tree),
    }


def collect_reference_texts(root: Path, visible_files: list[Path]) -> tuple[str, str]:
    docs_parts: list[str] = []
    registry_parts: list[str] = []
    for path in visible_files:
        rel = normalize_rel(path, root)
        if path.suffix.lower() not in REFERENCE_EXTENSIONS:
            continue
        if rel in SELF_GENERATED_REFERENCE_FILES:
            continue
        first = rel.split("/", 1)[0]
        if first not in TEXT_SCAN_ROOTS:
            continue
        try:
            text = safe_read(path)
        except OSError:
            continue
        docs_parts.append(f"\n--- {rel} ---\n{text}")
        if rel.startswith("registries/") or rel.endswith("runtime_manifest.json"):
            registry_parts.append(f"\n--- {rel} ---\n{text}")
    return "\n".join(docs_parts), "\n".join(registry_parts)


def count_references(rel_path: str, basename: str, docs_text: str) -> int:
    slash_count = docs_text.count(rel_path)
    backslash_count = docs_text.count(rel_path.replace("/", "\\"))
    basename_count = docs_text.count(basename)
    return slash_count + backslash_count + basename_count


def lifecycle_for(
    rel_path: str,
    role: str,
    registered_or_manifested: bool,
    docs_reference_count: int,
) -> str:
    if registered_or_manifested:
        return "canonical_or_manifested"
    if role == "contract_test":
        return "contract_or_test"
    if docs_reference_count > 0:
        return "documented_or_reported"
    if rel_path.startswith("runtimes/circleworld_proto/"):
        return "active_unregistered_research"
    if rel_path.startswith("tools/") or "/" not in rel_path:
        return "legacy_or_tooling"
    return "unclassified"


def make_record(
    root: Path,
    path: Path,
    tracked: set[str],
    docs_text: str,
    registry_text: str,
) -> ScriptRecord:
    rel_path = normalize_rel(path, root)
    text = safe_read(path)
    lines = text.splitlines()
    docs_refs = count_references(rel_path, path.name, docs_text)
    registered = (
        rel_path in registry_text
        or rel_path.replace("/", "\\") in registry_text
        or path.name in registry_text
    )
    role = script_role(rel_path, path.suffix.lower())
    record = ScriptRecord(
        path=rel_path,
        extension=path.suffix.lower(),
        lane=script_lane(rel_path),
        role=role,
        lifecycle=lifecycle_for(rel_path, role, registered, docs_refs),
        git_tracked=rel_path in tracked,
        registered_or_manifested=registered,
        docs_reference_count=docs_refs,
        bytes=len(text.encode("utf-8", errors="replace")),
        loc=len(lines),
        nonblank_loc=sum(1 for line in lines if line.strip()),
        sha1=sha1_text(text),
    )
    if path.suffix.lower() == ".py":
        try:
            py_info = analyze_python(text)
        except SyntaxError as exc:
            record.parse_error = f"{exc.__class__.__name__}: {exc}"
        else:
            record.module_doc_first_line = py_info["module_doc_first_line"]
            record.argparse_description = py_info["argparse_description"]
            record.has_main_guard = py_info["has_main_guard"]
            record.import_roots = py_info["import_roots"]
            record.top_level_functions = py_info["top_level_functions"]
            record.classes = py_info["classes"]
            record.top_level_constants = py_info["top_level_constants"]
    return record


def prefix_key(path: str) -> str:
    stem = Path(path).stem.lower()
    parts = stem.split("_")
    if len(parts) >= 3 and parts[0] in {"run", "score", "train", "assemble", "compare"}:
        return "_".join(parts[:3])
    if len(parts) >= 2:
        return "_".join(parts[:2])
    return stem


def jaccard(left: set[str], right: set[str]) -> float:
    if not left and not right:
        return 1.0
    union = left | right
    if not union:
        return 0.0
    return len(left & right) / len(union)


def build_redundancy(records: list[ScriptRecord]) -> dict[str, Any]:
    function_by_name: dict[str, list[dict[str, Any]]] = defaultdict(list)
    function_by_hash: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        for fn in record.top_level_functions:
            row = {
                "path": record.path,
                "name": fn.name,
                "line": fn.line,
                "loc": fn.loc,
                "body_hash": fn.body_hash,
            }
            function_by_name[fn.name].append(row)
            if fn.loc is not None and fn.loc >= 4:
                function_by_hash[fn.body_hash].append(row)

    repeated_function_names = []
    for name, rows in function_by_name.items():
        if len(rows) < 3:
            continue
        repeated_function_names.append(
            {
                "name": name,
                "count": len(rows),
                "files": sorted({row["path"] for row in rows}),
                "role": (
                    "cli_boilerplate"
                    if name in CLI_FUNCTION_NAMES
                    else "method_noise"
                    if name in METHOD_NOISE_NAMES
                    else "helper_candidate"
                ),
            }
        )
    repeated_function_names.sort(key=lambda row: (-row["count"], row["name"]))

    exact_duplicate_functions = []
    for body_hash, rows in function_by_hash.items():
        files = sorted({row["path"] for row in rows})
        if len(files) < 2:
            continue
        exact_duplicate_functions.append(
            {
                "body_hash": body_hash,
                "count": len(rows),
                "names": sorted({row["name"] for row in rows}),
                "files": files,
                "loc": max((row["loc"] or 0) for row in rows),
            }
        )
    exact_duplicate_functions.sort(key=lambda row: (-row["count"], -row["loc"], row["names"]))

    prefix_clusters: dict[str, list[ScriptRecord]] = defaultdict(list)
    for record in records:
        prefix_clusters[prefix_key(record.path)].append(record)
    prefix_cluster_rows = []
    for key, rows in prefix_clusters.items():
        if len(rows) < 3:
            continue
        prefix_cluster_rows.append(
            {
                "prefix": key,
                "count": len(rows),
                "total_nonblank_loc": sum(row.nonblank_loc for row in rows),
                "lanes": dict(Counter(row.lane for row in rows)),
                "roles": dict(Counter(row.role for row in rows)),
                "files": [row.path for row in sorted(rows, key=lambda row: row.path)],
            }
        )
    prefix_cluster_rows.sort(
        key=lambda row: (-row["total_nonblank_loc"], -row["count"], row["prefix"])
    )

    import_pairs = []
    import_sets = [(record, set(record.import_roots)) for record in records if len(record.import_roots) >= 5]
    for idx, (left, left_imports) in enumerate(import_sets):
        for right, right_imports in import_sets[idx + 1 :]:
            score = jaccard(left_imports, right_imports)
            if score < 0.72:
                continue
            import_pairs.append(
                {
                    "left": left.path,
                    "right": right.path,
                    "jaccard": round(score, 4),
                    "shared_imports": sorted(left_imports & right_imports),
                }
            )
    import_pairs.sort(key=lambda row: (-row["jaccard"], row["left"], row["right"]))

    largest_files = [
        {
            "path": row.path,
            "lane": row.lane,
            "role": row.role,
            "nonblank_loc": row.nonblank_loc,
            "bytes": row.bytes,
            "lifecycle": row.lifecycle,
        }
        for row in sorted(records, key=lambda row: row.nonblank_loc, reverse=True)[:30]
    ]

    helper_candidates = [
        row
        for row in repeated_function_names
        if row["role"] == "helper_candidate"
    ][:40]

    return {
        "largest_files": largest_files,
        "prefix_clusters": prefix_cluster_rows[:80],
        "repeated_function_names": repeated_function_names[:120],
        "helper_candidates": helper_candidates,
        "exact_duplicate_functions": exact_duplicate_functions[:120],
        "similar_import_pairs": import_pairs[:80],
    }


def summarize(records: list[ScriptRecord], redundancy: dict[str, Any]) -> dict[str, Any]:
    by_lane = Counter(row.lane for row in records)
    by_role = Counter(row.role for row in records)
    by_lifecycle = Counter(row.lifecycle for row in records)
    by_extension = Counter(row.extension for row in records)
    return {
        "script_count": len(records),
        "tracked_count": sum(1 for row in records if row.git_tracked),
        "untracked_count": sum(1 for row in records if not row.git_tracked),
        "total_nonblank_loc": sum(row.nonblank_loc for row in records),
        "python_parse_errors": sum(1 for row in records if row.parse_error),
        "with_main_guard": sum(1 for row in records if row.has_main_guard),
        "manifested_or_registered_count": sum(1 for row in records if row.registered_or_manifested),
        "documented_or_reported_count": sum(1 for row in records if row.docs_reference_count > 0),
        "by_lane": dict(by_lane.most_common()),
        "by_role": dict(by_role.most_common()),
        "by_lifecycle": dict(by_lifecycle.most_common()),
        "by_extension": dict(by_extension.most_common()),
        "largest_file": redundancy["largest_files"][0] if redundancy["largest_files"] else None,
        "largest_prefix_cluster": (
            redundancy["prefix_clusters"][0] if redundancy["prefix_clusters"] else None
        ),
        "top_helper_candidate": (
            redundancy["helper_candidates"][0] if redundancy["helper_candidates"] else None
        ),
    }


def table_rows(rows: list[list[Any]]) -> str:
    if not rows:
        return ""
    widths = [0 for _ in rows[0]]
    for row in rows:
        for idx, value in enumerate(row):
            widths[idx] = max(widths[idx], len(str(value)))
    lines = []
    for row_idx, row in enumerate(rows):
        values = [str(value) for value in row]
        lines.append("| " + " | ".join(value.ljust(widths[idx]) for idx, value in enumerate(values)) + " |")
        if row_idx == 0:
            lines.append("| " + " | ".join("-" * widths[idx] for idx in range(len(widths))) + " |")
    return "\n".join(lines)


def md_link(path: str) -> str:
    return f"`{path}`"


def write_inventory_markdown(path: Path, data: dict[str, Any]) -> None:
    summary = data["summary"]
    records = [ScriptRecord(**row) for row in data["scripts"]]
    for record in records:
        record.top_level_functions = [
            FunctionInfo(**fn) if isinstance(fn, dict) else fn for fn in record.top_level_functions
        ]
    redundancy = data["redundancy"]
    generated_at = data["generated_at"]

    lines: list[str] = [
        "# Project Script Inventory",
        "",
        f"Generated: `{generated_at}`",
        "",
        "This is a machine-generated map of source-like scripts that are visible to git, including untracked files, while honoring `.gitignore`.",
        "",
        "## Summary",
        "",
        table_rows(
            [
                ["Metric", "Value"],
                ["Scripts", summary["script_count"]],
                ["Tracked", summary["tracked_count"]],
                ["Untracked", summary["untracked_count"]],
                ["Total nonblank LOC", summary["total_nonblank_loc"]],
                ["Python parse errors", summary["python_parse_errors"]],
                ["Scripts with CLI main guard", summary["with_main_guard"]],
                ["Manifested/registered scripts", summary["manifested_or_registered_count"]],
                ["Documented/reported scripts", summary["documented_or_reported_count"]],
            ]
        ),
        "",
        "## By Lane",
        "",
        table_rows([["Lane", "Scripts"], *summary["by_lane"].items()]),
        "",
        "## By Role",
        "",
        table_rows([["Role", "Scripts"], *summary["by_role"].items()]),
        "",
        "## By Lifecycle",
        "",
        table_rows([["Lifecycle", "Scripts"], *summary["by_lifecycle"].items()]),
        "",
        "## Largest Files",
        "",
        table_rows(
            [
                ["Path", "Lane", "Role", "Nonblank LOC", "Lifecycle"],
                *[
                    [
                        md_link(row["path"]),
                        row["lane"],
                        row["role"],
                        row["nonblank_loc"],
                        row["lifecycle"],
                    ]
                    for row in redundancy["largest_files"][:20]
                ],
            ]
        ),
        "",
        "## Canonical Or Manifested Scripts",
        "",
    ]

    canonical = [row for row in records if row.registered_or_manifested]
    lines.append(
        table_rows(
            [
                ["Path", "Lane", "Role", "Docs refs"],
                *[
                    [md_link(row.path), row.lane, row.role, row.docs_reference_count]
                    for row in sorted(canonical, key=lambda row: row.path)[:120]
                ],
            ]
        )
    )
    lines.extend(
        [
            "",
            "## Active Unregistered Research Scripts",
            "",
            "These are not necessarily bad. They are the scripts most likely to become confusing if they are not either documented, folded into a shared helper, or explicitly retired.",
            "",
        ]
    )
    active_unregistered = [
        row
        for row in records
        if row.lifecycle == "active_unregistered_research"
    ]
    lines.append(
        table_rows(
            [
                ["Path", "Role", "Nonblank LOC", "CLI?", "Docs refs"],
                *[
                    [
                        md_link(row.path),
                        row.role,
                        row.nonblank_loc,
                        "yes" if row.has_main_guard else "no",
                        row.docs_reference_count,
                    ]
                    for row in sorted(active_unregistered, key=lambda row: row.nonblank_loc, reverse=True)[:80]
                ],
            ]
        )
    )
    lines.extend(
        [
            "",
            "## Full Script Index",
            "",
            table_rows(
                [
                    ["Path", "Lane", "Role", "Lifecycle", "LOC", "Funcs", "Classes"],
                    *[
                        [
                            md_link(row.path),
                            row.lane,
                            row.role,
                            row.lifecycle,
                            row.nonblank_loc,
                            len(row.top_level_functions),
                            len(row.classes),
                        ]
                        for row in sorted(records, key=lambda row: row.path)
                    ],
                ]
            ),
            "",
            "## Maintenance Rule",
            "",
            "Regenerate this file after any large experiment burst or before a cleanup/refactor pass:",
            "",
            "```powershell",
            "python tools/audit_project_scripts.py",
            "```",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def write_redundancy_markdown(path: Path, data: dict[str, Any]) -> None:
    redundancy = data["redundancy"]
    summary = data["summary"]
    lines: list[str] = [
        "# Project Redundancy Hotspots",
        "",
        f"Generated: `{data['generated_at']}`",
        "",
        "This report identifies overlap candidates. It does not prove code is safe to delete.",
        "",
        "## Headline",
        "",
        f"- Source-like scripts: `{summary['script_count']}`",
        f"- Total nonblank LOC: `{summary['total_nonblank_loc']}`",
        f"- Largest lane: `{next(iter(summary['by_lane']))}` with `{next(iter(summary['by_lane'].values()))}` scripts",
        f"- Largest prefix cluster: `{summary['largest_prefix_cluster']['prefix'] if summary['largest_prefix_cluster'] else 'n/a'}`",
        "",
        "## Largest Prefix Clusters",
        "",
        table_rows(
            [
                ["Prefix", "Count", "Nonblank LOC", "Lanes", "Roles"],
                *[
                    [
                        row["prefix"],
                        row["count"],
                        row["total_nonblank_loc"],
                        ", ".join(f"{k}:{v}" for k, v in row["lanes"].items()),
                        ", ".join(f"{k}:{v}" for k, v in row["roles"].items()),
                    ]
                    for row in redundancy["prefix_clusters"][:30]
                ],
            ]
        ),
        "",
        "## Helper Candidates",
        "",
        "Repeated helper names are the first obvious extraction surface. `main` and `parse_args` are intentionally excluded from this section.",
        "",
        table_rows(
            [
                ["Function", "Count", "Files"],
                *[
                    [row["name"], row["count"], len(row["files"])]
                    for row in redundancy["helper_candidates"][:40]
                ],
            ]
        ),
        "",
        "## Exact Duplicate Function Bodies",
        "",
        "These are AST-identical top-level functions with at least four lines. Start here before doing semantic refactors.",
        "",
        table_rows(
            [
                ["Names", "Count", "LOC", "Files"],
                *[
                    [
                        ", ".join(row["names"]),
                        row["count"],
                        row["loc"],
                        len(row["files"]),
                    ]
                    for row in redundancy["exact_duplicate_functions"][:40]
                ],
            ]
        ),
        "",
        "## Similar Import Pairs",
        "",
        "High import overlap is not a bug, but it is a good smell for repeated script skeletons.",
        "",
        table_rows(
            [
                ["Jaccard", "Left", "Right", "Shared imports"],
                *[
                    [
                        row["jaccard"],
                        md_link(row["left"]),
                        md_link(row["right"]),
                        ", ".join(row["shared_imports"][:10]),
                    ]
                    for row in redundancy["similar_import_pairs"][:30]
                ],
            ]
        ),
        "",
        "## Recommended Extraction Targets",
        "",
        "1. `runtimes/circleworld_proto/common_io.py`: JSON loading/writing, markdown table rendering, run directory creation.",
        "2. `runtimes/circleworld_proto/phase_audio_metrics.py`: correlation, MSE, loop/reentry metrics, target-normalized replay summaries.",
        "3. `runtimes/circleworld_proto/experiment_reports.py`: report headers, summary tables, comparison JSON scaffolds.",
        "4. `runtimes/circleworld_proto/childworld_metrics.py`: child survival/carry/writeback/divergence aggregation.",
        "5. `tools/project_cartography.py` or this script's helpers if more repo-wide audits appear.",
        "",
        "## Non-Deletion Rule",
        "",
        "Do not delete or rename any script from this report until a canonical artifact, ledger entry, or manifest path has been checked. This report is a triage map, not a cleanup patch.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def write_scaffolding_guide(path: Path, data: dict[str, Any]) -> None:
    lines = [
        "# Project Scaffolding Guide",
        "",
        "This is the persistent rulebook for future RAFA scripts. It exists because the project now has enough exploratory code that script identity has become an engineering risk.",
        "",
        "## Script Identity Contract",
        "",
        "Every new script should be easy to classify without reading the whole file.",
        "",
        "Required header-level answers:",
        "",
        "- Runtime or lane: `circleworld_proto`, `stage4_blackwell_14`, `diffusion_parent_v3`, `positive_replacement`, `tools`, or explicit `research_track`.",
        "- Role: `trainer`, `experiment_runner`, `evaluator`, `benchmark`, `scorer`, `comparator`, `report_assembler`, `auditor`, `builder`, `exporter`, `contract_test`, or `runtime_or_shared_module`.",
        "- Artifact outputs: name every JSON/Markdown/checkpoint path pattern the script writes.",
        "- Provenance inputs: checkpoint, config, seed, source lockbox, target lockbox, or manifest path.",
        "- Future-access stance for audio work: declare whether target/future phase or magnitude is read, and why.",
        "",
        "## Lifecycle Labels",
        "",
        "- `canonical_or_manifested`: appears in a runtime manifest or registry.",
        "- `documented_or_reported`: referenced in reports or architecture docs.",
        "- `contract_or_test`: verifies an invariant or schema.",
        "- `active_unregistered_research`: useful but not yet manifest/report canonical.",
        "- `legacy_or_tooling`: old root/tooling script; preserve until explicitly retired.",
        "- `retire_candidate`: only after an audit identifies replacement, artifacts, and no live references.",
        "",
        "## Promotion Rules",
        "",
        "A script can move from scout to canonical only if:",
        "",
        "- It has a deterministic CLI with `argparse` or a clear launcher contract.",
        "- It emits machine-readable JSON and, when useful, a Markdown report.",
        "- It is referenced by a report, manifest, registry, DAG overlay, or ledger entry.",
        "- It has a fresh smoke run or contract test.",
        "- It does not silently compare lineages without naming the runtime/checkpoint boundary.",
        "",
        "## Duplication Rules",
        "",
        "- If a helper appears in 3 scripts, mark it for extraction.",
        "- If a helper appears in 5 scripts, extract it before adding another variant unless there is a measured reason not to.",
        "- If a script exceeds 2,000 nonblank LOC, split report assembly, metric computation, and CLI orchestration unless the file is a frozen historical artifact.",
        "- If two scripts differ only by objective weights or route tables, prefer one parameterized runner plus named profile JSON.",
        "- If a script is superseded, keep a compatibility wrapper until reports and ledgers point to the replacement.",
        "",
        "## Canonical Output Patterns",
        "",
        "- Runtime outputs: `outputs/<runtime>/<run_id>/...`",
        "- Checkpoints: `checkpoints_<runtime>/<run_id>/...`",
        "- Architecture scaffolding: `docs/architecture/...`",
        "- Experiment reports: `docs/reports/<LANE>_<TOPIC>_<YYYY-MM-DD>.md`",
        "- Machine ledgers: JSONL next to the Markdown ledger when the lane already uses one.",
        "",
        "## Required Cartography Loop",
        "",
        "Run this after each large experiment burst:",
        "",
        "```powershell",
        "python tools/audit_project_scripts.py",
        "python tools/audit_project_usage.py",
        "python -m py_compile tools/audit_project_scripts.py",
        "python -m py_compile tools/audit_project_usage.py",
        "```",
        "",
        "Then inspect:",
        "",
        "- `docs/architecture/PROJECT_SCRIPT_INVENTORY.md`",
        "- `docs/architecture/PROJECT_REDUNDANCY_HOTSPOTS.md`",
        "- `docs/architecture/PROJECT_SCRIPT_INVENTORY.json`",
        "- `docs/architecture/PROJECT_USAGE_MAP.md`",
        "- `docs/architecture/PROJECT_ARTIFACT_INVENTORY.md`",
        "",
        "## Cleanup Sequence",
        "",
        "1. Freeze current artifacts and ledger interpretation.",
        "2. Extract shared helpers without changing outputs.",
        "3. Add compatibility wrappers for renamed entrypoints.",
        "4. Rerun contract tests and a representative smoke run.",
        "5. Mark old scripts as `retire_candidate` in docs before deletion.",
        "6. Delete only after a separate cleanup patch verifies no manifest/report/ledger reference remains.",
        "",
        "## RAFA-Specific Guardrails",
        "",
        "- Do not let Circleworld scaffold claims overwrite Graduation runtime claims.",
        "- Do not compare a Circleworld adapter failure to Graduation itself unless the protected Graduation runtime was actually run.",
        "- For phase-native audio work, no-future route selection must remain explicit.",
        "- For resonant memory work, dense signatures are not RAFA post-tokens until retrieval, composition, and causal-use tests pass.",
        "- For childworld ontology work, internal labels are evidence only if they connect to causal writeback, retrieval, composition, or audio-facing continuity.",
        "",
        f"Last generated from inventory summary: `{data['generated_at']}`.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a repo-wide RAFA script inventory and redundancy hotspot report."
    )
    parser.add_argument("--root", type=Path, default=repo_root_from_script())
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--json", type=Path, default=None)
    parser.add_argument("--markdown", type=Path, default=None)
    parser.add_argument("--redundancy", type=Path, default=None)
    parser.add_argument("--guide", type=Path, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = args.root.resolve()
    out_dir = (args.out_dir or root / "docs" / "architecture").resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    json_path = (args.json or out_dir / "PROJECT_SCRIPT_INVENTORY.json").resolve()
    markdown_path = (args.markdown or out_dir / "PROJECT_SCRIPT_INVENTORY.md").resolve()
    redundancy_path = (
        args.redundancy or out_dir / "PROJECT_REDUNDANCY_HOTSPOTS.md"
    ).resolve()
    guide_path = (args.guide or out_dir / "PROJECT_SCAFFOLDING_GUIDE.md").resolve()

    visible_files = list_git_visible_files(root)
    tracked = list_tracked_files(root)
    docs_text, registry_text = collect_reference_texts(root, visible_files)

    script_paths = [
        path
        for path in visible_files
        if path.suffix.lower() in SCRIPT_EXTENSIONS
    ]
    records = [
        make_record(root, path, tracked, docs_text, registry_text)
        for path in sorted(script_paths, key=lambda value: normalize_rel(value, root))
    ]
    redundancy = build_redundancy(records)
    summary = summarize(records, redundancy)
    data = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "root": str(root),
        "scope": {
            "file_source": "git ls-files --cached --others --exclude-standard",
            "script_extensions": sorted(SCRIPT_EXTENSIONS),
            "reference_extensions": sorted(REFERENCE_EXTENSIONS),
            "ignored_by_gitignore": True,
        },
        "summary": summary,
        "scripts": [asdict(record) for record in records],
        "redundancy": redundancy,
    }

    write_json(json_path, data)
    write_inventory_markdown(markdown_path, data)
    write_redundancy_markdown(redundancy_path, data)
    write_scaffolding_guide(guide_path, data)

    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"Wrote {json_path}")
    print(f"Wrote {markdown_path}")
    print(f"Wrote {redundancy_path}")
    print(f"Wrote {guide_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
