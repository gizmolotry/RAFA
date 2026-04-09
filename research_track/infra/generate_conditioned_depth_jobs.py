from __future__ import annotations

import argparse
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


def generate_jobs(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    defaults = dict(manifest.get("defaults", {}))
    named_runs = list(manifest.get("named_runs", []))
    if not named_runs:
        return []

    prefix = str(defaults.get("eval_name_prefix", "conditioned_depth"))
    val_steps = int(defaults.get("val_steps", 1))
    model_type = str(defaults.get("model_type", "rafa"))
    default_logic_diag = dict(defaults.get("logic_diag", {}))

    jobs: list[dict[str, Any]] = []
    for entry in named_runs:
        name = str(entry.get("name", "")).strip()
        if not name:
            raise ValueError("Each conditioned depth run must define a non-empty name.")
        merged_logic_diag = dict(default_logic_diag)
        merged_logic_diag.update(dict(entry.get("logic_diag", {})))
        jobs.append(
            {
                "name": name if name.startswith(prefix) else f"{prefix}_{name}",
                "checkpoint_name": str(entry.get("checkpoint_name", name)),
                "base_config": str(entry["base_config"]),
                "resume_ckpt": str(entry["resume_ckpt"]),
                "resume_global_step": int(entry.get("resume_global_step", 0)),
                "resume_epoch": int(entry.get("resume_epoch", 1)),
                "checkpoint_dir": str(entry["checkpoint_dir"]),
                "max_steps": int(entry["max_steps"]),
                "val_steps": int(entry.get("val_steps", val_steps)),
                "model_type": str(entry.get("model_type", model_type)),
                "text_condition_mode": str(entry.get("text_condition_mode", defaults.get("text_condition_mode", "film"))),
                "model_text_conditioned": _to_bool(
                    entry.get(
                        "model_text_conditioned",
                        defaults.get("model_text_conditioned", str(entry.get("text_condition_mode", defaults.get("text_condition_mode", "film"))).strip().lower() != "ifs_control"),
                    )
                ),
                "text_ifs_control_strength": float(entry.get("text_ifs_control_strength", defaults.get("text_ifs_control_strength", 0.25))),
                "num_steps": int(entry["num_steps"]),
                "anchor_enabled": _to_bool(entry.get("anchor_enabled", False)),
                "anchor_lambda": float(entry.get("anchor_lambda", defaults.get("anchor_lambda", 0.15))),
                "anchor_lambda_min": float(entry.get("anchor_lambda_min", defaults.get("anchor_lambda_min", 0.05))),
                "anchor_decay": float(entry.get("anchor_decay", defaults.get("anchor_decay", 0.9))),
                "intermediate_consistency_enabled": _to_bool(entry.get("intermediate_consistency_enabled", False)),
                "intermediate_consistency_detach_target": _to_bool(
                    entry.get(
                        "intermediate_consistency_detach_target",
                        defaults.get("intermediate_consistency_detach_target", False),
                    )
                ),
                "notes": str(entry.get("notes", "")),
                "logic_diag": merged_logic_diag,
            }
        )
    return jobs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True, help="Path to conditioned depth sweep manifest YAML.")
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
