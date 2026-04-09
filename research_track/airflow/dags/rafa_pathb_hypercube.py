from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml
from airflow import DAG
from airflow.decorators import get_current_context, task

import sys


REPO_ROOT = Path(__file__).resolve().parents[3]
INFRA_DIR = REPO_ROOT / "research_track" / "infra"
MANIFEST_PATH = INFRA_DIR / "hypercube_manifest.yaml"
JOBS_JSON_PATH = INFRA_DIR / "generated_jobs.json"
sys.path.append(str(INFRA_DIR))

from generate_hypercube_jobs import generate_jobs  # noqa: E402
from aggregate_hypercube_results import build_hypercube_report  # noqa: E402


default_args = {
    "owner": "rafa",
    "depends_on_past": False,
    "retries": 0,
}


with DAG(
    dag_id="rafa_pathb_hypercube",
    default_args=default_args,
    start_date=datetime(2026, 3, 3),
    schedule=None,
    catchup=False,
    max_active_runs=1,
    tags=["rafa", "pathb", "hypercube", "ablation"],
) as dag:
    @task
    def build_jobs() -> list[dict[str, Any]]:
        context = get_current_context()
        dag_run = context.get("dag_run")
        conf = (dag_run.conf if dag_run and dag_run.conf else {}) or {}
        start = max(0, int(conf.get("start", 0)))
        count_raw = conf.get("count")

        with MANIFEST_PATH.open("r", encoding="utf-8") as f:
            manifest = yaml.safe_load(f)
        jobs = generate_jobs(manifest)
        if count_raw is None:
            selected = jobs[start:]
        else:
            count = max(0, int(count_raw))
            selected = jobs[start : start + count]

        if not selected:
            raise ValueError(
                f"No jobs selected from hypercube set (total={len(jobs)} start={start} count={count_raw})."
            )

        with JOBS_JSON_PATH.open("w", encoding="utf-8") as f:
            json.dump({"jobs": selected, "selection": {"start": start, "count": count_raw}}, f, indent=2)
        return selected

    @task
    def run_single(job: dict[str, Any]) -> dict[str, Any]:
        from run_hypercube_job import run_job

        return run_job(job, repo_root=REPO_ROOT)

    @task
    def summarize(results: list[dict[str, Any]]) -> dict[str, Any]:
        resolved: list[dict[str, Any]] = []
        for r in results:
            sp = Path(str(r.get("summary_path", "")))
            if sp.exists():
                with sp.open("r", encoding="utf-8") as f:
                    resolved.append(json.load(f))
            else:
                resolved.append(r)
        summary = {
            "num_jobs": len(resolved),
            "jobs": [r["name"] for r in resolved if "name" in r],
            "checkpoints": [r["checkpoint"] for r in resolved if "checkpoint" in r],
            "s3_prefixes": [r.get("s3_prefix") for r in resolved if r.get("s3_prefix")],
            "text_condition_modes": sorted({str(r.get("text_condition_mode", "film")) for r in resolved}),
            "semantic_condition_modes": sorted({str(r.get("semantic_condition_mode", "none")) for r in resolved}),
        }
        leaderboard = build_hypercube_report(repo_root=REPO_ROOT, prefix="pathb_hypercube")
        summary["leaderboard_json"] = leaderboard.get("json_path")
        summary["leaderboard_md"] = leaderboard.get("md_path")
        out_path = INFRA_DIR / "last_airflow_summary.json"
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        return summary

    jobs = build_jobs()
    results = run_single.expand(job=jobs)
    summarize(results)
