from __future__ import annotations

import subprocess
import sys
from datetime import datetime
from pathlib import Path

from airflow import DAG
from airflow.decorators import task


REPO_ROOT = Path(__file__).resolve().parents[3]
RUNNER = REPO_ROOT / "tools" / "run_learning_curve_resume.py"
CFG = REPO_ROOT / "tmp" / "bridge_lc_vram_config.yaml"
CKPT_DIR = REPO_ROOT / "checkpoints_diffusion_bridge_lc_vram_progress"
BASE_RESUME = REPO_ROOT / "checkpoints_diffusion_bridge_lc_s30" / "diff_ep1.pt"


default_args = {
    "owner": "rafa",
    "depends_on_past": False,
    "retries": 0,
}


def _run_target(target_step: int) -> dict[str, str | int]:
    cmd = [
        sys.executable,
        "-u",
        str(RUNNER),
        "--config",
        str(CFG),
        "--ckpt_dir",
        str(CKPT_DIR),
        "--base_resume_ckpt",
        str(BASE_RESUME),
        "--base_resume_step",
        "30",
        "--targets",
        str(target_step),
        "--val_steps",
        "1",
    ]
    proc = subprocess.run(cmd, cwd=str(REPO_ROOT), check=False)
    if proc.returncode != 0:
        raise RuntimeError(f"Learning-curve target {target_step} failed with exit={proc.returncode}")
    return {"target": target_step, "status": "ok"}


with DAG(
    dag_id="rafa_learning_curve_resume",
    default_args=default_args,
    start_date=datetime(2026, 3, 6),
    schedule=None,
    catchup=False,
    max_active_runs=1,
    tags=["rafa", "learning_curve", "resume", "pathb", "probe"],
) as dag:
    @task
    def run_300() -> dict[str, str | int]:
        return _run_target(300)

    @task
    def run_3000() -> dict[str, str | int]:
        return _run_target(3000)

    @task
    def run_8000() -> dict[str, str | int]:
        return _run_target(8000)

    @task
    def summarize(a: dict[str, str | int], b: dict[str, str | int], c: dict[str, str | int]) -> dict[str, object]:
        return {"targets": [a, b, c]}

    a = run_300()
    b = run_3000()
    c = run_8000()
    a >> b >> c
    summarize(a, b, c)

