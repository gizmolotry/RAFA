from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _default_status_path(repo_root: Path, job_name: str) -> Path:
    return repo_root / "research_track" / "infra" / "summaries" / f"{job_name}.status.json"


def launch_job(job_path: Path, repo_root: Path, status_path: Path | None = None) -> dict[str, Any]:
    with job_path.open("r", encoding="utf-8") as f:
        job = json.load(f)
    job_name = str(job["name"])
    status_path = status_path or _default_status_path(repo_root, job_name)
    python_exe = str(job.get("python_executable") or os.environ.get("RAFA_PYTHON_EXE") or sys.executable)

    logs_dir = repo_root / "logs" / "conditioned_depth"
    logs_dir.mkdir(parents=True, exist_ok=True)
    launch_log = logs_dir / f"launch_{job_name}.log"
    launch_err = logs_dir / f"launch_{job_name}.err.log"

    cmd = [
        python_exe,
        "-u",
        str(repo_root / "research_track" / "infra" / "run_conditioned_depth_job.py"),
        "--job-json",
        str(job_path),
        "--status-json",
        str(status_path),
    ]
    env = os.environ.copy()
    creationflags = 0
    if os.name == "nt":
        creationflags = subprocess.CREATE_NEW_PROCESS_GROUP

    with launch_log.open("w", encoding="utf-8") as out_f, launch_err.open("w", encoding="utf-8") as err_f:
        proc = subprocess.Popen(
            cmd,
            cwd=str(repo_root),
            env=env,
            stdout=out_f,
            stderr=err_f,
            close_fds=False,
            creationflags=creationflags,
        )

    payload = {
        "name": job_name,
        "status": "launched",
        "phase": "launch",
        "pid": int(proc.pid),
        "job_json": str(job_path),
        "status_json": str(status_path),
        "python_executable": python_exe,
        "launch_log": str(launch_log),
        "launch_err": str(launch_err),
        "launched_utc": datetime.now(timezone.utc).isoformat(),
    }
    status_path.parent.mkdir(parents=True, exist_ok=True)
    with status_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(json.dumps(payload, indent=2))
    return payload


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--job-json", required=True, help="Path to JSON with a single conditioned depth job object.")
    ap.add_argument("--status-json", default=None, help="Optional output status JSON path.")
    args = ap.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    job_path = Path(args.job_json)
    status_path = Path(args.status_json) if args.status_json else None
    launch_job(job_path=job_path, repo_root=repo_root, status_path=status_path)


if __name__ == "__main__":
    main()
