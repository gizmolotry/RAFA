from __future__ import annotations

import argparse
import glob
import os
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path


def _ts() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _log(msg: str) -> None:
    print(f"[{_ts()}] {msg}", flush=True)


def _parse_step_from_name(path: str) -> int | None:
    name = os.path.basename(path)
    m = re.search(r"diff_step(\d+)\.pt$", name)
    if m:
        return int(m.group(1))
    return None


def _latest_step_ckpt(ckpt_dir: str) -> tuple[str | None, int]:
    pats = glob.glob(os.path.join(ckpt_dir, "diff_step*.pt"))
    best_path = None
    best_step = -1
    for p in pats:
        s = _parse_step_from_name(p)
        if s is not None and s > best_step:
            best_step = s
            best_path = p
    if best_path is None:
        return None, -1
    return best_path, best_step


def _run(cmd: list[str], env: dict[str, str]) -> None:
    _log("RUN: " + " ".join(cmd))
    p = subprocess.Popen(cmd, env=env)
    rc = p.wait()
    if rc != 0:
        raise RuntimeError(f"Command failed (exit={rc}): {' '.join(cmd)}")


def _run_train_to_target(
    *,
    target_step: int,
    ckpt_dir: str,
    config_path: str,
    base_resume_ckpt: str,
    base_resume_step: int,
    val_steps: int,
) -> str:
    os.makedirs(ckpt_dir, exist_ok=True)
    latest_ckpt, latest_step = _latest_step_ckpt(ckpt_dir)
    if latest_ckpt is not None and latest_step >= target_step:
        _log(f"Checkpoint already exists for target={target_step}: {latest_ckpt}")
        return latest_ckpt

    resume_ckpt = latest_ckpt if latest_ckpt is not None else base_resume_ckpt
    resume_step = latest_step if latest_ckpt is not None else base_resume_step
    env = os.environ.copy()
    env["RAFA_CONFIG_PATH"] = config_path
    env["CUDA_VISIBLE_DEVICES"] = "0"

    cmd = [
        sys.executable,
        "-u",
        "train_diffusion.py",
        "--epochs",
        "999",
        "--max_steps",
        str(target_step),
        "--val_steps",
        str(val_steps),
        "--ckpt_dir",
        ckpt_dir,
        "--resume_ckpt",
        resume_ckpt,
        "--resume_global_step",
        str(resume_step),
        "--resume_epoch",
        "1",
    ]
    _run(cmd, env)

    out_ckpt = os.path.join(ckpt_dir, f"diff_step{target_step}.pt")
    if not os.path.exists(out_ckpt):
        raise RuntimeError(f"Target checkpoint missing after train: {out_ckpt}")
    return out_ckpt


def _run_evals(*, target_step: int, ckpt_path: str) -> None:
    eval_json = f"eval/relational_probe_controls_bridge_lc_vram_s{target_step}.json"
    eval_dir = f"eval/bridge_lc_vram_s{target_step}"

    if not os.path.exists(eval_dir):
        _run(
            [
                sys.executable,
                "tools/path_b_eval.py",
                "--name",
                f"bridge_lc_vram_s{target_step}",
                "--ckpt",
                ckpt_path,
                "--notes",
                f"bridge_learning_curve_vram_step_{target_step}",
            ],
            os.environ.copy(),
        )
    else:
        _log(f"Skipping path_b_eval for s{target_step}; exists: {eval_dir}")

    if not os.path.exists(eval_json):
        _run(
            [
                sys.executable,
                "tools/relational_probe.py",
                "--ckpt_full",
                ckpt_path,
                "--probe_steps",
                "200",
                "--eval_seeds",
                "20",
                "--out_json",
                eval_json,
            ],
            os.environ.copy(),
        )
    else:
        _log(f"Skipping relational_probe for s{target_step}; exists: {eval_json}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt_dir", required=True)
    ap.add_argument("--base_resume_ckpt", required=True)
    ap.add_argument("--base_resume_step", type=int, default=30)
    ap.add_argument("--targets", default="300,3000,8000")
    ap.add_argument("--val_steps", type=int, default=1)
    args = ap.parse_args()

    config_path = str(Path(args.config))
    ckpt_dir = str(Path(args.ckpt_dir))
    base_resume_ckpt = str(Path(args.base_resume_ckpt))
    targets = [int(x.strip()) for x in args.targets.split(",") if x.strip()]

    _log("Learning-curve resume runner started")
    _log(f"config={config_path}")
    _log(f"ckpt_dir={ckpt_dir}")
    _log(f"base_resume_ckpt={base_resume_ckpt}")
    _log(f"targets={targets}")

    for t in targets:
        _log(f"=== TARGET {t} ===")
        ckpt = _run_train_to_target(
            target_step=t,
            ckpt_dir=ckpt_dir,
            config_path=config_path,
            base_resume_ckpt=base_resume_ckpt,
            base_resume_step=int(args.base_resume_step),
            val_steps=int(args.val_steps),
        )
        _log(f"Reached target checkpoint: {ckpt}")
        _run_evals(target_step=t, ckpt_path=ckpt)
        _log(f"Finished evals for target {t}")

    _log("All targets complete")


if __name__ == "__main__":
    main()

