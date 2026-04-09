from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from aggregate_hypercube_results import build_hypercube_report
from run_hypercube_job import run_job


def main() -> None:
    ap = argparse.ArgumentParser(description="Run a subset of hypercube jobs from a jobs JSON file.")
    ap.add_argument("--jobs-json", required=True, help="Path to JSON with {\"jobs\": [...]} payload.")
    ap.add_argument("--start", type=int, default=0, help="Start index (inclusive).")
    ap.add_argument("--count", type=int, default=1, help="How many jobs to run.")
    ap.add_argument("--stop-on-error", action="store_true", help="Stop on first failure.")
    args = ap.parse_args()

    jobs_path = Path(args.jobs_json)
    repo_root = Path(__file__).resolve().parents[2]
    with jobs_path.open("r", encoding="utf-8") as f:
        payload: dict[str, Any] = json.load(f)
    jobs = list(payload.get("jobs", []))

    start = max(0, int(args.start))
    end = min(len(jobs), start + max(0, int(args.count)))
    if start >= len(jobs):
        raise RuntimeError(f"start index {start} out of range for {len(jobs)} jobs")

    results: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for i in range(start, end):
        job = jobs[i]
        name = str(job.get("name", f"job_{i}"))
        print(f"[{i}] running {name}", flush=True)
        try:
            out = run_job(job, repo_root=repo_root)
            results.append({"index": i, "name": name, "status": "ok", "summary_path": out.get("summary_path", "")})
            print(f"[{i}] ok {name}", flush=True)
        except Exception as exc:  # noqa: BLE001
            failures.append({"index": i, "name": name, "status": "failed", "error": str(exc)})
            print(f"[{i}] failed {name}: {exc}", flush=True)
            if args.stop_on_error:
                break

    out = {
        "jobs_json": str(jobs_path),
        "start": start,
        "end_exclusive": end,
        "ok": results,
        "failed": failures,
    }
    leaderboard = build_hypercube_report(repo_root=repo_root, prefix="pathb_hypercube")
    out["leaderboard_json"] = leaderboard.get("json_path")
    out["leaderboard_md"] = leaderboard.get("md_path")
    out_path = repo_root / "research_track" / "infra" / "summaries" / "last_batch_run.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"Wrote batch summary to {out_path}", flush=True)


if __name__ == "__main__":
    main()
