from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml


def _write_status(status_path: Path | None, payload: dict[str, Any]) -> None:
    if status_path is None:
        return
    status_path.parent.mkdir(parents=True, exist_ok=True)
    with status_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def _run(cmd: list[str], *, cwd: Path, log_path: Path, env: dict[str, str] | None = None) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as f:
        f.write(" ".join(cmd) + "\n\n")
        f.flush()
        proc = subprocess.run(cmd, cwd=str(cwd), env=env, stdout=f, stderr=subprocess.STDOUT, check=False)
    if proc.returncode != 0:
        raise RuntimeError(f"Command failed ({proc.returncode}): {' '.join(cmd)}")


def _git_sha(repo_root: Path) -> str:
    try:
        out = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=str(repo_root), text=True).strip()
        return out or "unknown"
    except Exception:
        return "unknown"


def _latest_checkpoint(ckpt_dir: Path) -> Path | None:
    candidates = sorted(
        [p for p in ckpt_dir.glob("diff_step*.pt")] + [p for p in ckpt_dir.glob("diff_ep*.pt")],
        key=lambda p: (p.stat().st_mtime_ns, p.name),
    )
    return candidates[-1] if candidates else None


def _build_logic_job(job: dict[str, Any], checkpoint_path: Path) -> dict[str, Any]:
    logic_cfg = dict(job.get("logic_diag", {}))
    enabled = bool(logic_cfg.get("enabled", False))
    if not enabled:
        return {}
    return {
        "name": str(job["name"]),
        "checkpoint_name": str(job.get("checkpoint_name", job["name"])),
        "checkpoint": str(checkpoint_path),
        "notes": str(job.get("notes", "")),
        "tasks": list(logic_cfg.get("tasks", [])),
        "iterations": int(logic_cfg.get("iterations", 25)),
        "inject_strength": float(logic_cfg.get("inject_strength", 0.5)),
        "seed": int(logic_cfg.get("seed", 1337)),
        "anchor_enabled": bool(job.get("anchor_enabled", False)),
        "anchor_lambda": float(job.get("anchor_lambda", 0.15)),
        "anchor_lambda_min": float(job.get("anchor_lambda_min", 0.05)),
        "anchor_decay": float(job.get("anchor_decay", 0.9)),
        "intermediate_consistency_enabled": bool(job.get("intermediate_consistency_enabled", False)),
        "intermediate_consistency_detach_target": bool(job.get("intermediate_consistency_detach_target", False)),
        "num_steps": int(job.get("num_steps", 2)),
    }


def run_job(job: dict[str, Any], repo_root: Path, *, status_path: Path | None = None) -> dict[str, Any]:
    name = str(job["name"])
    base_config = Path(str(job["base_config"]))
    resume_ckpt = Path(str(job["resume_ckpt"]))
    checkpoint_dir = Path(str(job["checkpoint_dir"]))
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    tmp_cfg_dir = repo_root / "tmp" / "conditioned_depth_configs"
    tmp_cfg_dir.mkdir(parents=True, exist_ok=True)
    tmp_cfg_path = tmp_cfg_dir / f"{name}.yaml"

    with base_config.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    cfg.setdefault("model", {})
    text_condition_mode = str(job.get("text_condition_mode", "film")).strip().lower()
    model_text_conditioned = bool(job.get("model_text_conditioned", text_condition_mode != "ifs_control"))
    cfg["model"]["text_conditioned"] = model_text_conditioned
    cfg.setdefault("diffusion", {})
    cfg["diffusion"]["use_clap_cond"] = True
    cfg["diffusion"]["require_clap_cond"] = True
    cfg["diffusion"]["text_condition_mode"] = text_condition_mode
    cfg["diffusion"]["text_ifs_control_strength"] = float(job.get("text_ifs_control_strength", 0.25))
    cfg["diffusion"]["train_epochs"] = int(max(2, int(job["resume_epoch"]) + 3))
    cfg.setdefault("phase_native_ifs", {})
    cfg["phase_native_ifs"]["num_steps"] = int(job["num_steps"])
    cfg["phase_native_ifs"]["h0_anchor_enabled"] = bool(job.get("anchor_enabled", False))
    cfg["phase_native_ifs"]["h0_anchor_lambda"] = float(job.get("anchor_lambda", 0.15))
    cfg["phase_native_ifs"]["h0_anchor_lambda_min"] = float(job.get("anchor_lambda_min", 0.05))
    cfg["phase_native_ifs"]["h0_anchor_decay"] = float(job.get("anchor_decay", 0.9))
    cfg["phase_native_ifs"]["intermediate_consistency_enabled"] = bool(job.get("intermediate_consistency_enabled", False))
    cfg["phase_native_ifs"]["intermediate_consistency_detach_target"] = bool(
        job.get("intermediate_consistency_detach_target", False)
    )
    cfg.setdefault("training", {})
    cfg.setdefault("training", {}).setdefault("losses", {})
    cfg["training"]["losses"]["use_ifs_intermediate_consistency"] = bool(
        job.get("intermediate_consistency_enabled", False)
    )

    with tmp_cfg_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)

    env = os.environ.copy()
    env["RAFA_CONFIG_PATH"] = str(tmp_cfg_path)

    logs_dir = repo_root / "logs" / "conditioned_depth"
    logs_dir.mkdir(parents=True, exist_ok=True)
    train_cmd = [
        sys.executable,
        "-u",
        str(repo_root / "train_diffusion.py"),
        "--model_type",
        str(job.get("model_type", "rafa")),
        "--max_steps",
        str(int(job["max_steps"])),
        "--ckpt_dir",
        str(checkpoint_dir),
        "--resume_ckpt",
        str(resume_ckpt),
        "--resume_global_step",
        str(int(job.get("resume_global_step", 0))),
        "--resume_epoch",
        str(int(job.get("resume_epoch", 1))),
        "--val_steps",
        str(int(job.get("val_steps", 1))),
    ]
    train_log = logs_dir / f"train_{name}.log"
    _write_status(
        status_path,
        {
            "name": name,
            "phase": "training",
            "status": "running",
            "train_log": str(train_log),
            "checkpoint_dir": str(checkpoint_dir),
            "updated_utc": datetime.now(timezone.utc).isoformat(),
        },
    )
    _run(train_cmd, cwd=repo_root, log_path=train_log, env=env)

    checkpoint_path = _latest_checkpoint(checkpoint_dir)
    if checkpoint_path is None:
        _write_status(
            status_path,
            {
                "name": name,
                "phase": "training",
                "status": "error",
                "error": f"No checkpoint produced for conditioned depth job {name}",
                "updated_utc": datetime.now(timezone.utc).isoformat(),
            },
        )
        raise RuntimeError(f"No checkpoint produced for conditioned depth job {name}")

    logic_summary_path = None
    logic_job = _build_logic_job(job, checkpoint_path)
    if logic_job:
        tmp_logic_dir = repo_root / "tmp" / "conditioned_depth_logic"
        tmp_logic_dir.mkdir(parents=True, exist_ok=True)
        logic_job_path = tmp_logic_dir / f"{name}.json"
        with logic_job_path.open("w", encoding="utf-8") as f:
            json.dump(logic_job, f, indent=2)
        logic_log = logs_dir / f"logic_{name}.log"
        _write_status(
            status_path,
            {
                "name": name,
                "phase": "logic_diag",
                "status": "running",
                "train_log": str(train_log),
                "logic_log": str(logic_log),
                "checkpoint": str(checkpoint_path),
                "updated_utc": datetime.now(timezone.utc).isoformat(),
            },
        )
        logic_cmd = [
            sys.executable,
            "-u",
            str(repo_root / "research_track" / "infra" / "run_logic_diag_job.py"),
            "--job-json",
            str(logic_job_path),
        ]
        _run(logic_cmd, cwd=repo_root, log_path=logic_log, env=env)
        logic_summary_path = repo_root / "research_track" / "infra" / "summaries" / f"{name}.json"

    summary = {
        "name": name,
        "checkpoint_name": str(job.get("checkpoint_name", name)),
        "base_config": str(base_config),
        "temp_config": str(tmp_cfg_path),
        "resume_ckpt": str(resume_ckpt),
        "resume_global_step": int(job.get("resume_global_step", 0)),
        "resume_epoch": int(job.get("resume_epoch", 1)),
        "checkpoint_dir": str(checkpoint_dir),
        "checkpoint": str(checkpoint_path),
        "max_steps": int(job["max_steps"]),
        "val_steps": int(job.get("val_steps", 1)),
        "text_condition_mode": text_condition_mode,
        "model_text_conditioned": model_text_conditioned,
        "text_ifs_control_strength": float(job.get("text_ifs_control_strength", 0.25)),
        "num_steps": int(job["num_steps"]),
        "anchor_enabled": bool(job.get("anchor_enabled", False)),
        "anchor_lambda": float(job.get("anchor_lambda", 0.15)),
        "anchor_lambda_min": float(job.get("anchor_lambda_min", 0.05)),
        "anchor_decay": float(job.get("anchor_decay", 0.9)),
        "intermediate_consistency_enabled": bool(job.get("intermediate_consistency_enabled", False)),
        "intermediate_consistency_detach_target": bool(job.get("intermediate_consistency_detach_target", False)),
        "train_log": str(train_log),
        "logic_summary_path": str(logic_summary_path) if logic_summary_path is not None else "",
        "notes": str(job.get("notes", "")),
        "git_sha": _git_sha(repo_root),
        "finished_utc": datetime.now(timezone.utc).isoformat(),
    }
    summary_dir = repo_root / "research_track" / "infra" / "summaries"
    summary_dir.mkdir(parents=True, exist_ok=True)
    summary_path = summary_dir / f"{name}.train.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    summary["summary_path"] = str(summary_path)
    _write_status(
        status_path,
        {
            "name": name,
            "phase": "done",
            "status": "completed",
            "checkpoint": str(checkpoint_path),
            "train_log": str(train_log),
            "logic_summary_path": str(logic_summary_path) if logic_summary_path is not None else "",
            "summary_path": str(summary_path),
            "updated_utc": datetime.now(timezone.utc).isoformat(),
        },
    )
    print(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--job-json", required=True, help="Path to JSON with a single conditioned depth job object.")
    ap.add_argument("--status-json", default=None, help="Optional status JSON path for live phase updates.")
    args = ap.parse_args()
    repo_root = Path(__file__).resolve().parents[2]
    with Path(args.job_json).open("r", encoding="utf-8") as f:
        job = json.load(f)
    status_path = Path(args.status_json) if args.status_json else None
    try:
        run_job(job, repo_root=repo_root, status_path=status_path)
    except Exception as exc:
        _write_status(
            status_path,
            {
                "name": str(job.get("name", "unknown")),
                "phase": "error",
                "status": "error",
                "error": str(exc),
                "updated_utc": datetime.now(timezone.utc).isoformat(),
            },
        )
        raise


if __name__ == "__main__":
    main()
