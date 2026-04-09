from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _run(cmd: list[str], *, cwd: Path) -> None:
    proc = subprocess.run(cmd, cwd=str(cwd), check=False)
    if proc.returncode != 0:
        raise RuntimeError(f"Command failed ({proc.returncode}): {' '.join(cmd)}")


def run_batch(jobs: list[dict[str, Any]], repo_root: Path, start: int, count: int | None, stop_on_error: bool) -> dict[str, Any]:
    selected = jobs[start:] if count is None else jobs[start : start + count]
    tmp_dir = repo_root / "tmp" / "conditioned_depth_jobs"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, Any]] = []
    for idx, job in enumerate(selected, start=start):
        job_path = tmp_dir / f"{job['name']}.json"
        with job_path.open("w", encoding="utf-8") as f:
            json.dump(job, f, indent=2)
        try:
            _run(
                [
                    sys.executable,
                    "-u",
                    str(repo_root / "research_track" / "infra" / "run_conditioned_depth_job.py"),
                    "--job-json",
                    str(job_path),
                ],
                cwd=repo_root,
            )
            results.append({"index": idx, "name": job["name"], "status": "ok"})
        except Exception as e:
            results.append({"index": idx, "name": job["name"], "status": "error", "error": str(e)})
            if stop_on_error:
                break

    summary = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "start": start,
        "count": len(selected),
        "stop_on_error": stop_on_error,
        "results": results,
    }
    summary_dir = repo_root / "research_track" / "infra" / "summaries"
    summary_dir.mkdir(parents=True, exist_ok=True)
    out_path = summary_dir / "last_conditioned_depth_batch.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs-json", required=True, help="Path to generated conditioned depth jobs JSON.")
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--count", type=int, default=None)
    ap.add_argument("--stop-on-error", action="store_true")
    args = ap.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    with Path(args.jobs_json).open("r", encoding="utf-8") as f:
        payload = json.load(f)
    jobs = list(payload.get("jobs", []))
    run_batch(jobs, repo_root=repo_root, start=max(0, args.start), count=args.count, stop_on_error=bool(args.stop_on_error))


if __name__ == "__main__":
    main()
