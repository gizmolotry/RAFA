from __future__ import annotations

import sys
from pathlib import Path


def _exit_if_help_requested_without_runtime() -> None:
    if any(arg in ("-h", "--help") for arg in sys.argv[1:]):
        print(f"usage: {Path(__file__).name} [runtime-coupled contract options]")
        print()
        print("This contract is coupled to the Circleworld runtime/signature lane.")
        print("Run it from a Circleworld integration worktree for full argument parsing and execution.")
        raise SystemExit(0)


_exit_if_help_requested_without_runtime()

import ast
import json
import py_compile
import re
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
RUNTIME = ROOT / "runtimes" / "circleworld_proto"
LINEAGE = ROOT / "lineages" / "04_positive_replacement"
MANIFEST = RUNTIME / "runtime_manifest.json"


REQUIRED_MANIFEST_KEYS = [
    "id",
    "status",
    "checkpoint_schema",
    "canonical_entrypoint",
    "training_entrypoint",
    "specialized_training_entrypoint",
    "benchmark_entrypoint",
    "nested_commitment_entrypoint",
    "implementation_entrypoint",
]

EXPERIMENT_RUNNERS = [
    "run_branch_pressure_experiment.py",
    "run_token_diversity_experiment.py",
    "run_branchlaw_ablation_series.py",
    "run_childworld_experiment.py",
    "retrospective_childworld_eval.py",
]

REQUIRED_ARTIFACT_KEYS = [
    "heldout_summary",
    "benchmark_summary",
    "nested_summary",
    "law_token_library",
]

REQUIRED_RECENT_RUN_ARTIFACTS = {
    "train_summary": ["train_summary.json"],
    "heldout_summary": ["heldout_eval/heldout_summary.json"],
    "benchmark_summary": ["benchmark_expanded/benchmark_summary.json"],
    "nested_summary": ["nested_commitment/nested_commitment_report.json"],
    "law_token_library": ["law_token_library/law_token_library.json"],
}


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _compile_status(paths: list[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in paths:
        try:
            py_compile.compile(str(path), doraise=True)
            rows.append({"file": str(path), "status": "ok"})
        except Exception as exc:  # pragma: no cover
            rows.append({"file": str(path), "status": "error", "error": str(exc)})
    return rows


def _check_manifest() -> dict[str, Any]:
    payload = _read_json(MANIFEST)
    missing = [key for key in REQUIRED_MANIFEST_KEYS if key not in payload]
    path_checks: dict[str, bool] = {}
    for key in REQUIRED_MANIFEST_KEYS:
        value = payload.get(key)
        if isinstance(value, str) and ("/" in value or "\\" in value):
            path_checks[key] = (ROOT / value).exists()
    optional_entrypoint_checks: dict[str, bool] = {}
    for key, value in sorted(payload.items()):
        if key in REQUIRED_MANIFEST_KEYS:
            continue
        if key.endswith("_entrypoint") and isinstance(value, str):
            optional_entrypoint_checks[key] = (ROOT / value).exists()
    shared_modules = [str(ROOT / mod) for mod in payload.get("shared_modules", [])]
    shared_exist = {mod: Path(mod).exists() for mod in shared_modules}
    paths_ok = all(path_checks.values()) and all(optional_entrypoint_checks.values())
    return {
        "path": str(MANIFEST),
        "status": "ok" if not missing and paths_ok and all(shared_exist.values()) else "warn",
        "missing_keys": missing,
        "path_checks": path_checks,
        "optional_entrypoint_checks": optional_entrypoint_checks,
        "shared_modules_exist": shared_exist,
        "payload": payload,
    }


def _runner_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _extract_profile_choices(text: str) -> list[str]:
    match = re.search(r'--profile",\s*choices=\[(.*?)\],\s*default=', text, re.S)
    if not match:
        return []
    return re.findall(r'"([^"]+)"', match.group(1))


def _check_runner(path: Path) -> dict[str, Any]:
    text = _runner_text(path)
    artifact_keys = [key for key in REQUIRED_ARTIFACT_KEYS if f'"{key}"' in text]
    nested_targets_test = "test_nested_commitment.py" in text or "_run_nested_commitment" in text
    mode_match = re.findall(r'"--mode",\s*"([^"]+)"', text)
    nested_modes = sorted(set(mode_match))
    summary_json_names = re.findall(r'"([a-z_]+_summary\.json)"', text)
    profile_choices = _extract_profile_choices(text)
    return {
        "file": str(path),
        "status": "ok" if nested_targets_test and len(artifact_keys) == len(REQUIRED_ARTIFACT_KEYS) else "warn",
        "artifact_keys_present": artifact_keys,
        "nested_targets_test_nested_commitment": nested_targets_test,
        "nested_modes": nested_modes,
        "summary_json_names": sorted(set(summary_json_names)),
        "profile_choices": profile_choices,
    }


def _check_summary_consistency() -> dict[str, Any]:
    rows = [_check_runner(RUNTIME / name) for name in EXPERIMENT_RUNNERS]
    return {
        "status": "ok" if all(row["status"] == "ok" for row in rows) else "warn",
        "rows": rows,
    }


def _resolve_artifact(base_dir: Path, candidates: list[str]) -> dict[str, Any]:
    checked: list[str] = []
    for rel in candidates:
        if rel.startswith("ROOT::"):
            path = ROOT / rel.split("::", 1)[1]
            checked.append(str(path))
            if path.exists():
                return {"status": "ok", "path": str(path), "checked": checked, "source": "external_root"}
            continue
        if "*" in rel:
            matches = sorted(ROOT.glob(rel))
            checked.append(str(ROOT / rel))
            if matches:
                return {"status": "ok", "path": str(matches[0]), "checked": checked, "source": "external_glob"}
            continue
        path = base_dir / rel
        checked.append(str(path))
        if path.exists():
            return {"status": "ok", "path": str(path), "checked": checked, "source": "local"}
    return {"status": "warn", "path": None, "checked": checked, "source": None}


def _check_recent_runs() -> dict[str, Any]:
    runs = [
        {
            "name": "agreement_scout_v1",
            "path": ROOT / "outputs" / "circleworld_proto" / "training_run_2026-04-24_agreement_scout_v1",
            "alternates": {
                "heldout_summary": ["ROOT::outputs/circleworld_proto/reeval_agreement_scout_v1_fixed_2026-04-24/heldout_summary.json"],
            },
        },
        {
            "name": "v17_nested_gate",
            "path": ROOT / "outputs" / "circleworld_proto" / "training_run_2026-04-24_nested_gate_scout_v17",
            "alternates": {},
        },
        {
            "name": "v18_nested_probe",
            "path": ROOT / "outputs" / "circleworld_proto" / "training_run_2026-04-28_nested_probe_objective_v18",
            "alternates": {},
        },
    ]
    rows = []
    for spec in runs:
        run_dir = spec["path"]
        checks: dict[str, Any] = {}
        for key, local_candidates in REQUIRED_RECENT_RUN_ARTIFACTS.items():
            candidates = list(local_candidates) + list(spec.get("alternates", {}).get(key, []))
            checks[key] = _resolve_artifact(run_dir, candidates)
        rows.append(
            {
                "name": spec["name"],
                "path": str(run_dir),
                "status": "ok" if all(item["status"] == "ok" for item in checks.values()) else "warn",
                "checks": checks,
            }
        )
    return {"status": "ok" if all(row["status"] == "ok" for row in rows) else "warn", "rows": rows}


def run_audit() -> dict[str, Any]:
    runtime_py = sorted(RUNTIME.glob("*.py"))
    lineage_py = sorted(LINEAGE.glob("*.py"))
    compile_rows = _compile_status(runtime_py + lineage_py)
    compile_ok = all(row["status"] == "ok" for row in compile_rows)
    manifest = _check_manifest()
    runners = _check_summary_consistency()
    recent_runs = _check_recent_runs()
    status = "ok" if compile_ok and manifest["status"] == "ok" and runners["status"] == "ok" and recent_runs["status"] == "ok" else "warn"
    return {
        "runtime": "circleworld_proto",
        "audit": "dag_health",
        "status": status,
        "manifest": manifest,
        "compile": {"status": "ok" if compile_ok else "warn", "rows": compile_rows},
        "runners": runners,
        "recent_runs": recent_runs,
    }


def main() -> None:
    out = run_audit()
    out_path = ROOT / "outputs" / "circleworld_proto" / "dag_health_audit_2026-04-28.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps({"status": out["status"], "saved": str(out_path)}, indent=2))


if __name__ == "__main__":
    main()
