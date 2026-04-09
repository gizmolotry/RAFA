from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
import yaml


def _run(
    cmd: list[str],
    env: dict[str, str] | None = None,
    log_path: Path | None = None,
    cwd: Path | None = None,
) -> None:
    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("w", encoding="utf-8") as f:
            f.write(" ".join(cmd) + "\n\n")
            f.flush()
            proc = subprocess.run(cmd, env=env, cwd=str(cwd) if cwd is not None else None, stdout=f, stderr=subprocess.STDOUT, check=False)
    else:
        proc = subprocess.run(cmd, env=env, cwd=str(cwd) if cwd is not None else None, check=False)
    if proc.returncode != 0:
        raise RuntimeError(f"Command failed ({proc.returncode}): {' '.join(cmd)}")


def _latest_checkpoint(ckpt_dir: Path, preferred_step: int) -> Path | None:
    preferred = ckpt_dir / f"diff_step{preferred_step}.pt"
    if preferred.exists():
        return preferred
    files = sorted(ckpt_dir.glob("*.pt"), key=lambda p: p.stat().st_mtime)
    if not files:
        return None
    return files[-1]


def _checkpoint_compatible(
    checkpoint: Path,
    *,
    model_type: str,
    memory_enabled: bool,
    slow_clock_enabled: bool,
    hyena_conductor_enabled: bool,
    metamer_enabled: bool,
    text_prompt_conditioned: bool,
    text_condition_mode: str,
    semantic_condition_mode: str,
    model_text_conditioned: bool,
) -> bool:
    try:
        ckpt = torch.load(str(checkpoint), map_location="cpu")
    except Exception:
        return False

    ck_model_type = str(ckpt.get("diffusion", {}).get("model_type", "rafa"))
    if ck_model_type != str(model_type):
        return False

    ifs_cfg = ckpt.get("config", {}).get("phase_native_ifs", {})
    if "memory_enabled" in ifs_cfg and bool(ifs_cfg.get("memory_enabled")) != bool(memory_enabled):
        return False
    if "slow_clock_enabled" in ifs_cfg and bool(ifs_cfg.get("slow_clock_enabled")) != bool(slow_clock_enabled):
        return False
    mcfg = ckpt.get("config", {}).get("model", {})
    if "hyena_conductor_enabled" in mcfg and bool(mcfg.get("hyena_conductor_enabled")) != bool(hyena_conductor_enabled):
        return False
    tcfg = ckpt.get("config", {}).get("training", {}).get("metamers", {})
    if "enabled" in tcfg and bool(tcfg.get("enabled")) != bool(metamer_enabled):
        return False
    dcfg = ckpt.get("config", {}).get("diffusion", {})
    ck_text_cond = bool(dcfg.get("use_clap_cond", dcfg.get("use_clip_cond", False)))
    if ck_text_cond != bool(text_prompt_conditioned):
        return False
    ck_text_mode = str(dcfg.get("text_condition_mode", "film"))
    if ck_text_mode != str(text_condition_mode):
        return False
    ck_semantic_mode = str(dcfg.get("semantic_condition_mode", "none"))
    if ck_semantic_mode != str(semantic_condition_mode):
        return False
    ck_model_text_cond = bool(ckpt.get("config", {}).get("model", {}).get("text_conditioned", False))
    if ck_model_text_cond != bool(model_text_conditioned):
        return False
    return True


def _git_sha(repo_root: Path) -> str:
    try:
        out = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=str(repo_root), text=True).strip()
        return out or "unknown"
    except Exception:
        return "unknown"


def _upload_dir_to_s3(local_dir: Path, bucket: str, prefix: str) -> list[str]:
    try:
        import boto3  # type: ignore
    except Exception as exc:
        raise RuntimeError(f"boto3 is required for S3 upload: {exc}") from exc

    s3 = boto3.client("s3")
    uploaded: list[str] = []
    for p in local_dir.rglob("*"):
        if not p.is_file():
            continue
        key = f"{prefix.rstrip('/')}/{p.relative_to(local_dir).as_posix()}"
        s3.upload_file(str(p), bucket, key)
        uploaded.append(f"s3://{bucket}/{key}")
    return uploaded


def _load_scorecard(eval_dir: Path) -> dict[str, Any] | None:
    scorecard_path = eval_dir / "scorecard.json"
    if not scorecard_path.exists():
        return None
    with scorecard_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def run_job(job: dict[str, Any], repo_root: Path) -> dict[str, Any]:
    name = str(job["name"])
    model_type = str(job.get("model_type", "rafa"))
    max_steps = int(job.get("max_steps", 1000))
    memory_enabled = bool(job.get("memory_enabled", True))
    ablate_ramanujan = bool(job.get("ablate_ramanujan", False))
    ablate_slow_clock = bool(job.get("ablate_slow_clock", False))
    hyena_conductor_enabled = bool(job.get("hyena_conductor_enabled", True))
    metamer_enabled = bool(job.get("metamer_enabled", True))
    text_prompt_conditioned = bool(job.get("text_prompt_conditioned", False))
    text_condition_mode = str(job.get("text_condition_mode", "film")).strip().lower()
    semantic_condition_mode = str(job.get("semantic_condition_mode", "none")).strip().lower()
    model_text_conditioned = bool(job.get("model_text_conditioned", text_condition_mode != "ifs_control"))
    text_encoder = str(job.get("text_encoder", "laion/clap-htsat-unfused"))
    guidance_scale = float(job.get("guidance_scale", 3.0))
    p_uncond = float(job.get("p_uncond", 0.15))
    cond_mag_scale = float(job.get("cond_mag_scale", 0.1))
    text_ifs_control_strength = float(job.get("text_ifs_control_strength", 0.25))
    semantic_tension_strength = float(job.get("semantic_tension_strength", 0.75))
    semantic_tension_alpha_strength = float(job.get("semantic_tension_alpha_strength", 0.2))
    semantic_tension_temp_strength = float(job.get("semantic_tension_temp_strength", 0.35))
    semantic_tension_delta_strength = float(job.get("semantic_tension_delta_strength", 0.25))
    phase_qset = [int(q) for q in job.get("phase_qset", [])]
    val_steps = int(job.get("val_steps", 50))
    require_clap_cond = bool(job.get("require_clap_cond", text_prompt_conditioned))
    use_existing_checkpoint = bool(job.get("use_existing_checkpoint", True))
    notes = str(job.get("notes", ""))

    logs_dir = repo_root / "logs" / "hypercube"
    logs_dir.mkdir(parents=True, exist_ok=True)

    ckpt_dir = repo_root / f"checkpoints_diffusion_{name}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    eval_dir = repo_root / "eval" / name
    eval_dir.mkdir(parents=True, exist_ok=True)

    base_cfg_path = repo_root / "config.yaml"
    tmp_cfg_dir = repo_root / "tmp" / "hypercube_configs"
    tmp_cfg_dir.mkdir(parents=True, exist_ok=True)
    tmp_cfg_path = tmp_cfg_dir / f"{name}.yaml"

    with base_cfg_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg.setdefault("phase_native_ifs", {})
    cfg["phase_native_ifs"]["memory_enabled"] = memory_enabled
    if phase_qset:
        cfg["phase_native_ifs"]["qset"] = phase_qset
        cfg["phase_native_ifs"]["q_weights"] = [1.0] * len(phase_qset)
    cfg.setdefault("model", {})
    if phase_qset:
        cfg["model"]["hyena_qset"] = phase_qset
        for gear in cfg["model"].get("gears", []):
            if isinstance(gear, dict):
                gear["hyena_qset"] = phase_qset
    cfg["model"]["hyena_conductor_enabled"] = hyena_conductor_enabled
    cfg["model"]["text_conditioned"] = model_text_conditioned
    cfg["model"]["text_encoder"] = text_encoder
    cfg.setdefault("training", {})
    if phase_qset:
        cfg["training"]["crystal_qs"] = phase_qset
    cfg["training"].setdefault("metamers", {})
    cfg["training"]["metamers"]["enabled"] = metamer_enabled
    if phase_qset:
        cfg["training"]["metamers"]["qset"] = phase_qset
    cfg.setdefault("diffusion", {})
    cfg["diffusion"]["use_clap_cond"] = text_prompt_conditioned
    cfg["diffusion"]["use_cfg"] = text_prompt_conditioned
    cfg["diffusion"]["text_condition_mode"] = text_condition_mode
    cfg["diffusion"]["semantic_condition_mode"] = semantic_condition_mode
    cfg["diffusion"]["guidance_scale"] = guidance_scale
    cfg["diffusion"]["p_uncond"] = p_uncond
    cfg["diffusion"]["cond_mag_scale"] = cond_mag_scale
    cfg["diffusion"]["text_ifs_control_strength"] = text_ifs_control_strength
    cfg["diffusion"]["semantic_tension_strength"] = semantic_tension_strength
    cfg["diffusion"]["semantic_tension_alpha_strength"] = semantic_tension_alpha_strength
    cfg["diffusion"]["semantic_tension_temp_strength"] = semantic_tension_temp_strength
    cfg["diffusion"]["semantic_tension_delta_strength"] = semantic_tension_delta_strength
    cfg["diffusion"]["require_clap_cond"] = require_clap_cond
    with tmp_cfg_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)

    env = os.environ.copy()
    env["RAFA_CONFIG_PATH"] = str(tmp_cfg_path)

    checkpoint = _latest_checkpoint(ckpt_dir, max_steps)
    if use_existing_checkpoint and checkpoint is not None:
        ok = _checkpoint_compatible(
            checkpoint,
            model_type=model_type,
            memory_enabled=memory_enabled,
            slow_clock_enabled=(not ablate_slow_clock),
            hyena_conductor_enabled=hyena_conductor_enabled,
            metamer_enabled=metamer_enabled,
            text_prompt_conditioned=text_prompt_conditioned,
            text_condition_mode=text_condition_mode,
            semantic_condition_mode=semantic_condition_mode,
            model_text_conditioned=model_text_conditioned,
        )
        if not ok:
            checkpoint = None

    if checkpoint is None or not use_existing_checkpoint:
        train_cmd = [
            sys.executable,
            "-u",
            str(repo_root / "train_diffusion.py"),
            "--model_type",
            model_type,
            "--max_steps",
            str(max_steps),
            "--ckpt_dir",
            str(ckpt_dir),
            "--val_steps",
            str(val_steps),
        ]
        if ablate_ramanujan:
            train_cmd.append("--ablate_ramanujan")
        if ablate_slow_clock:
            train_cmd.append("--ablate_slow_clock")
        _run(train_cmd, env=env, log_path=logs_dir / f"train_{name}.log", cwd=repo_root)
        checkpoint = _latest_checkpoint(ckpt_dir, max_steps)

    if checkpoint is None or not checkpoint.exists():
        raise RuntimeError(f"No checkpoint found for job={name} in {ckpt_dir}")

    eval_cmd = [
        sys.executable,
        "-u",
        str(repo_root / "tools" / "path_b_eval.py"),
        "--name",
        name,
        "--ckpt",
        str(checkpoint),
        "--notes",
        notes,
    ]
    _run(eval_cmd, env=env, log_path=logs_dir / f"eval_{name}.log", cwd=repo_root)
    scorecard = _load_scorecard(eval_dir)
    semantic_summary = None
    if semantic_condition_mode != "none":
        semantic_cmd = [
            sys.executable,
            "-u",
            str(repo_root / "tools" / "eval_semantic_tension.py"),
            "--name",
            name,
            "--ckpt",
            str(checkpoint),
        ]
        _run(semantic_cmd, env=env, log_path=logs_dir / f"semantic_eval_{name}.log", cwd=repo_root)
        semantic_summary = _load_json(eval_dir / "semantic_tension_summary.json")

    run_summary: dict[str, Any] = {
        "name": name,
        "model_type": model_type,
        "max_steps": max_steps,
        "memory_enabled": memory_enabled,
        "ablate_ramanujan": ablate_ramanujan,
        "ablate_slow_clock": ablate_slow_clock,
        "hyena_conductor_enabled": hyena_conductor_enabled,
        "metamer_enabled": metamer_enabled,
        "text_prompt_conditioned": text_prompt_conditioned,
        "text_condition_mode": text_condition_mode,
        "semantic_condition_mode": semantic_condition_mode,
        "model_text_conditioned": model_text_conditioned,
        "text_encoder": text_encoder,
        "guidance_scale": guidance_scale,
        "text_ifs_control_strength": text_ifs_control_strength,
        "semantic_tension_strength": semantic_tension_strength,
        "semantic_tension_alpha_strength": semantic_tension_alpha_strength,
        "semantic_tension_temp_strength": semantic_tension_temp_strength,
        "semantic_tension_delta_strength": semantic_tension_delta_strength,
        "phase_qset": phase_qset,
        "checkpoint": str(checkpoint),
        "eval_dir": str(eval_dir),
        "val_steps": val_steps,
        "scorecard_path": str(eval_dir / "scorecard.json"),
        "path_b_scorecard": scorecard,
        "semantic_tension_summary_path": str(eval_dir / "semantic_tension_summary.json") if semantic_summary is not None else "",
        "semantic_tension_summary": semantic_summary,
        "notes": notes,
        "git_sha": _git_sha(repo_root),
        "finished_utc": datetime.now(timezone.utc).isoformat(),
    }
    run_summary_light: dict[str, Any] = dict(run_summary)

    s3_cfg = dict(job.get("s3", {}))
    if bool(s3_cfg.get("enabled", False)):
        bucket = str(s3_cfg.get("bucket", "")).strip()
        if not bucket:
            raise RuntimeError("S3 upload requested but no bucket was provided.")
        prefix_root = str(s3_cfg.get("prefix", "rafa/path_b_hypercube")).strip("/")
        run_prefix = f"{prefix_root}/{name}"
        staging_dir = repo_root / "tmp" / "hypercube_staging" / name
        if staging_dir.exists():
            shutil.rmtree(staging_dir)
        staging_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(checkpoint, staging_dir / checkpoint.name)
        for log_name in (f"train_{name}.log", f"eval_{name}.log"):
            src = logs_dir / log_name
            if src.exists():
                shutil.copy2(src, staging_dir / log_name)
        if eval_dir.exists():
            shutil.copytree(eval_dir, staging_dir / "eval", dirs_exist_ok=True)

        summary_path = staging_dir / "run_summary.json"
        with summary_path.open("w", encoding="utf-8") as f:
            json.dump(run_summary, f, indent=2)

        uploaded = _upload_dir_to_s3(staging_dir, bucket=bucket, prefix=run_prefix)
        run_summary["s3_uploaded"] = uploaded
        run_summary["s3_uploaded_count"] = len(uploaded)
        run_summary["s3_prefix"] = f"s3://{bucket}/{run_prefix}"
        run_summary_light["s3_prefix"] = run_summary["s3_prefix"]
        run_summary_light["s3_uploaded_count"] = run_summary["s3_uploaded_count"]

    summary_dir = repo_root / "research_track" / "infra" / "summaries"
    summary_dir.mkdir(parents=True, exist_ok=True)
    out_summary = summary_dir / f"{name}.json"
    with out_summary.open("w", encoding="utf-8") as f:
        json.dump(run_summary, f, indent=2)
    print(json.dumps(run_summary, indent=2))
    run_summary_light["summary_path"] = str(out_summary)
    return run_summary_light


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--job-json", required=True, help="Path to JSON with a single job object.")
    args = ap.parse_args()
    job_path = Path(args.job_json)
    repo_root = Path(__file__).resolve().parents[2]

    with job_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if "job" in payload:
        job = payload["job"]
    else:
        job = payload
    run_job(job, repo_root=repo_root)


if __name__ == "__main__":
    main()
