from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _run(cmd: list[str], *, cwd: Path, log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as f:
        f.write(" ".join(cmd) + "\n\n")
        f.flush()
        proc = subprocess.run(cmd, cwd=str(cwd), stdout=f, stderr=subprocess.STDOUT, check=False)
    if proc.returncode != 0:
        raise RuntimeError(f"Command failed ({proc.returncode}): {' '.join(cmd)}")


def _git_sha(repo_root: Path) -> str:
    try:
        out = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=str(repo_root), text=True).strip()
        return out or "unknown"
    except Exception:
        return "unknown"


def _mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return float(sum(values) / len(values))


def _logic_result_path(repo_root: Path, model_tag: str, q1: int, q2: int, target_q: int, seed: int) -> Path:
    return repo_root / "research_track" / "logic_results" / f"logic_{model_tag}_{q1}x{q2}_t{target_q}_s{seed}.json"


def run_job(job: dict[str, Any], repo_root: Path) -> dict[str, Any]:
    name = str(job["name"])
    checkpoint = str(job["checkpoint"])
    checkpoint_name = str(job.get("checkpoint_name", name))
    tasks = list(job.get("tasks", []))
    iterations = int(job.get("iterations", 25))
    inject_strength = float(job.get("inject_strength", 0.5))
    seed = int(job.get("seed", 1337))
    anchor_enabled = bool(job.get("anchor_enabled", False))
    anchor_lambda = float(job.get("anchor_lambda", 0.15))
    anchor_lambda_min = float(job.get("anchor_lambda_min", 0.05))
    anchor_decay = float(job.get("anchor_decay", 0.9))
    intermediate_consistency_enabled = bool(job.get("intermediate_consistency_enabled", False))
    intermediate_consistency_detach_target = bool(job.get("intermediate_consistency_detach_target", False))
    num_steps = int(job.get("num_steps", 2))
    notes = str(job.get("notes", ""))

    logs_dir = repo_root / "logs" / "logic_diag"
    logs_dir.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, Any]] = []

    for task in tasks:
        q1 = int(task["q1"])
        q2 = int(task["q2"])
        target_q = int(task.get("target_q") or __import__("math").lcm(q1, q2))
        cmd = [
            sys.executable,
            str(repo_root / "tools" / "test_relational_logic.py"),
            "--ckpt",
            checkpoint,
            "--model_tag",
            name,
            "--q1",
            str(q1),
            "--q2",
            str(q2),
            "--target_q",
            str(target_q),
            "--iterations",
            str(iterations),
            "--inject_strength",
            str(inject_strength),
            "--seed",
            str(seed),
            "--num_steps",
            str(num_steps),
        ]
        if anchor_enabled:
            cmd.extend(
                [
                    "--h0_anchor_enabled",
                    "--h0_anchor_lambda",
                    str(anchor_lambda),
                    "--h0_anchor_lambda_min",
                    str(anchor_lambda_min),
                    "--h0_anchor_decay",
                    str(anchor_decay),
                ]
            )
        else:
            cmd.append("--no_h0_anchor_enabled")
        if intermediate_consistency_enabled:
            cmd.append("--intermediate_consistency_enabled")
        else:
            cmd.append("--no_intermediate_consistency_enabled")
        if intermediate_consistency_detach_target:
            cmd.append("--intermediate_consistency_detach_target")
        else:
            cmd.append("--no_intermediate_consistency_detach_target")

        log_path = logs_dir / f"{name}_{q1}x{q2}_t{target_q}_s{seed}.log"
        _run(cmd, cwd=repo_root, log_path=log_path)
        artifact_path = _logic_result_path(repo_root, name, q1, q2, target_q, seed)
        if not artifact_path.exists():
            raise RuntimeError(f"Missing logic artifact after run: {artifact_path}")
        with artifact_path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        result = dict(payload.get("result", {}))
        result["artifact_path"] = str(artifact_path)
        result["log_path"] = str(log_path)
        results.append(result)

    success_rate = _mean([1.0 if bool(r.get("success", False)) else 0.0 for r in results])
    run_summary: dict[str, Any] = {
        "name": name,
        "checkpoint_name": checkpoint_name,
        "checkpoint": checkpoint,
        "tasks": tasks,
        "iterations": iterations,
        "inject_strength": inject_strength,
        "seed": seed,
        "anchor_enabled": anchor_enabled,
        "anchor_lambda": anchor_lambda,
        "anchor_lambda_min": anchor_lambda_min,
        "anchor_decay": anchor_decay,
        "intermediate_consistency_enabled": intermediate_consistency_enabled,
        "intermediate_consistency_detach_target": intermediate_consistency_detach_target,
        "ifs_num_steps": num_steps,
        "num_tasks": len(results),
        "num_success": sum(1 for r in results if bool(r.get("success", False))),
        "success_rate": success_rate,
        "mean_final_rank": _mean([float(r.get("final_rank", 999)) for r in results]),
        "mean_persistence": _mean([float(r.get("persistence_score", 0.0)) for r in results]),
        "mean_final_margin": _mean([float(r.get("final_margin", 0.0)) for r in results]),
        "mean_energy_gain": _mean([float(r.get("energy_gain", 0.0)) for r in results]),
        "mean_intermediate_consistency_reg": _mean([float(r.get("mean_intermediate_consistency_reg", 0.0)) for r in results]),
        "mean_anchor_lambda": _mean([float(r.get("mean_anchor_lambda", 0.0)) for r in results]),
        "results": results,
        "notes": notes,
        "git_sha": _git_sha(repo_root),
        "finished_utc": datetime.now(timezone.utc).isoformat(),
    }

    summary_dir = repo_root / "research_track" / "infra" / "summaries"
    summary_dir.mkdir(parents=True, exist_ok=True)
    out_summary = summary_dir / f"{name}.json"
    with out_summary.open("w", encoding="utf-8") as f:
        json.dump(run_summary, f, indent=2)
    run_summary["summary_path"] = str(out_summary)
    print(json.dumps(run_summary, indent=2))
    return run_summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--job-json", required=True, help="Path to JSON with a single job object.")
    args = ap.parse_args()
    repo_root = Path(__file__).resolve().parents[2]
    with Path(args.job_json).open("r", encoding="utf-8") as f:
        job = json.load(f)
    run_job(job, repo_root=repo_root)


if __name__ == "__main__":
    main()
