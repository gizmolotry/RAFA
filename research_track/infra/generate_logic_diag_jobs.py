from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
from typing import Any

import yaml


def _to_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        v = value.strip().lower()
        if v in {"1", "true", "yes", "y", "on"}:
            return True
        if v in {"0", "false", "no", "n", "off"}:
            return False
    raise ValueError(f"Cannot coerce value to bool: {value!r}")


def _job_name(prefix: str, checkpoint_name: str, anchor_enabled: bool, intermediate_consistency_enabled: bool, num_steps: int) -> str:
    return (
        f"{prefix}_{checkpoint_name}"
        f"_a{int(anchor_enabled)}"
        f"_ic{int(intermediate_consistency_enabled)}"
        f"_ns{int(num_steps)}"
    )


def generate_jobs(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    defaults = dict(manifest.get("defaults", {}))
    axes = dict(manifest.get("axes", {}))
    named_runs = list(manifest.get("named_runs", []))

    prefix = str(defaults.get("eval_name_prefix", "logic_diag"))
    checkpoints = list(defaults.get("checkpoints", []))
    tasks = list(defaults.get("tasks", []))
    iterations = int(defaults.get("iterations", 25))
    inject_strength = float(defaults.get("inject_strength", 0.5))
    seed = int(defaults.get("seed", 1337))
    anchor_lambda = float(defaults.get("anchor_lambda", 0.15))
    anchor_lambda_min = float(defaults.get("anchor_lambda_min", 0.05))
    anchor_decay = float(defaults.get("anchor_decay", 0.9))
    intermediate_consistency_detach_target = _to_bool(defaults.get("intermediate_consistency_detach_target", False))

    anchor_vals = [_to_bool(x) for x in axes.get("anchor_enabled", [False])]
    ic_vals = [_to_bool(x) for x in axes.get("intermediate_consistency_enabled", [False])]
    num_steps_vals = [int(x) for x in axes.get("num_steps", [2])]

    jobs: list[dict[str, Any]] = []
    for checkpoint, anchor_enabled, intermediate_consistency_enabled, num_steps in itertools.product(
        checkpoints,
        anchor_vals,
        ic_vals,
        num_steps_vals,
    ):
        checkpoint_name = str(checkpoint["name"])
        jobs.append(
            {
                "name": _job_name(prefix, checkpoint_name, anchor_enabled, intermediate_consistency_enabled, num_steps),
                "checkpoint_name": checkpoint_name,
                "checkpoint": str(checkpoint["ckpt"]),
                "notes": str(checkpoint.get("notes", "")),
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
                "num_steps": int(num_steps),
            }
        )

    for entry in named_runs:
        checkpoint = dict(entry.get("checkpoint", {}))
        checkpoint_name = str(checkpoint.get("name", entry.get("checkpoint_name", "unnamed_ckpt")))
        jobs.append(
            {
                "name": str(entry.get("name") or _job_name(
                    prefix,
                    checkpoint_name,
                    _to_bool(entry.get("anchor_enabled", defaults.get("anchor_enabled", False))),
                    _to_bool(entry.get("intermediate_consistency_enabled", defaults.get("intermediate_consistency_enabled", False))),
                    int(entry.get("num_steps", defaults.get("num_steps", 2))),
                )),
                "checkpoint_name": checkpoint_name,
                "checkpoint": str(checkpoint.get("ckpt", entry.get("checkpoint"))),
                "notes": str(entry.get("notes", checkpoint.get("notes", ""))),
                "tasks": list(entry.get("tasks", tasks)),
                "iterations": int(entry.get("iterations", iterations)),
                "inject_strength": float(entry.get("inject_strength", inject_strength)),
                "seed": int(entry.get("seed", seed)),
                "anchor_enabled": _to_bool(entry.get("anchor_enabled", False)),
                "anchor_lambda": float(entry.get("anchor_lambda", anchor_lambda)),
                "anchor_lambda_min": float(entry.get("anchor_lambda_min", anchor_lambda_min)),
                "anchor_decay": float(entry.get("anchor_decay", anchor_decay)),
                "intermediate_consistency_enabled": _to_bool(entry.get("intermediate_consistency_enabled", False)),
                "intermediate_consistency_detach_target": _to_bool(
                    entry.get("intermediate_consistency_detach_target", intermediate_consistency_detach_target)
                ),
                "num_steps": int(entry.get("num_steps", 2)),
            }
        )

    dedup: dict[str, dict[str, Any]] = {}
    for job in jobs:
        dedup[job["name"]] = job
    return list(dedup.values())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True, help="Path to logic diagnostic manifest YAML.")
    ap.add_argument("--out", required=True, help="Output JSON file path.")
    args = ap.parse_args()

    manifest_path = Path(args.manifest)
    out_path = Path(args.out)
    with manifest_path.open("r", encoding="utf-8") as f:
        manifest = yaml.safe_load(f)

    jobs = generate_jobs(manifest)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump({"jobs": jobs}, f, indent=2)
    print(f"Wrote {len(jobs)} jobs to {out_path}")


if __name__ == "__main__":
    main()
