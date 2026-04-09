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


def _run_name(
    prefix: str,
    memory_enabled: bool,
    ramanujan_enabled: bool,
    slow_clock_enabled: bool,
    conductor_enabled: bool,
    metamer_enabled: bool,
    text_prompt_conditioned: bool,
    text_condition_mode: str,
    model_type: str,
) -> str:
    if model_type == "baseline":
        return f"{prefix}_baseline"
    mem = "mem1" if memory_enabled else "mem0"
    ram = "ram1" if ramanujan_enabled else "ram0"
    slow = "slow1" if slow_clock_enabled else "slow0"
    cond = "cond1" if conductor_enabled else "cond0"
    meta = "meta1" if metamer_enabled else "meta0"
    base = f"{prefix}_{mem}_{ram}_{slow}_{cond}_{meta}"
    if text_prompt_conditioned:
        suffix = "_txt1"
        if str(text_condition_mode).strip().lower() == "ifs_control":
            suffix = "_txt1_ifs1"
        elif str(text_condition_mode).strip().lower() == "hybrid":
            suffix = "_txt1_hyb1"
        return f"{base}{suffix}"
    return base


def generate_jobs(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    defaults = manifest.get("defaults", {})
    axes = manifest.get("axes", {})
    named_runs = manifest.get("named_runs", [])

    memory_vals = [_to_bool(x) for x in axes.get("memory_enabled", [True])]
    ramanujan_vals = [_to_bool(x) for x in axes.get("ramanujan_enabled", [True])]
    slow_vals = [_to_bool(x) for x in axes.get("slow_clock_enabled", [True])]
    conductor_vals = [_to_bool(x) for x in axes.get("hyena_conductor_enabled", [True])]
    metamer_vals = [_to_bool(x) for x in axes.get("metamer_enabled", [True])]
    text_prompt_vals = [_to_bool(x) for x in axes.get("text_prompt_conditioned", [False])]
    text_condition_mode_default = str(defaults.get("text_condition_mode", "film"))
    semantic_condition_mode_default = str(defaults.get("semantic_condition_mode", "none"))
    eval_prefix = str(defaults.get("eval_name_prefix", "pathb_hypercube"))
    max_steps = int(defaults.get("max_steps", 1000))

    jobs: list[dict[str, Any]] = []

    for memory_enabled, ramanujan_enabled, slow_clock_enabled, conductor_enabled, metamer_enabled, text_prompt_conditioned in itertools.product(
        memory_vals, ramanujan_vals, slow_vals, conductor_vals, metamer_vals, text_prompt_vals
    ):
        name = _run_name(
            eval_prefix,
            memory_enabled,
            ramanujan_enabled,
            slow_clock_enabled,
            conductor_enabled,
            metamer_enabled,
            text_prompt_conditioned,
            text_condition_mode_default,
            "rafa",
        )
        jobs.append(
            {
                "name": name,
                "model_type": "rafa",
                "max_steps": max_steps,
                "memory_enabled": memory_enabled,
                "ablate_ramanujan": not ramanujan_enabled,
                "ablate_slow_clock": not slow_clock_enabled,
                "hyena_conductor_enabled": conductor_enabled,
                "metamer_enabled": metamer_enabled,
                "text_prompt_conditioned": text_prompt_conditioned,
                "text_condition_mode": text_condition_mode_default,
                "semantic_condition_mode": semantic_condition_mode_default,
                "model_text_conditioned": bool(defaults.get("model_text_conditioned", text_condition_mode_default != "ifs_control")),
                "text_encoder": str(defaults.get("text_encoder", "laion/clap-htsat-unfused")),
                "guidance_scale": float(defaults.get("guidance_scale", 3.0)),
                "p_uncond": float(defaults.get("p_uncond", 0.15)),
                "cond_mag_scale": float(defaults.get("cond_mag_scale", 0.1)),
                "text_ifs_control_strength": float(defaults.get("text_ifs_control_strength", 0.25)),
                "semantic_tension_strength": float(defaults.get("semantic_tension_strength", 0.75)),
                "semantic_tension_alpha_strength": float(defaults.get("semantic_tension_alpha_strength", 0.2)),
                "semantic_tension_temp_strength": float(defaults.get("semantic_tension_temp_strength", 0.35)),
                "semantic_tension_delta_strength": float(defaults.get("semantic_tension_delta_strength", 0.25)),
                "phase_qset": list(defaults.get("phase_qset", [])),
                "val_steps": int(defaults.get("val_steps", 50)),
                "require_clap_cond": bool(defaults.get("require_clap_cond", False)) or bool(text_prompt_conditioned),
                "notes": (
                    f"hypercube memory={int(memory_enabled)} "
                    f"ramanujan={int(ramanujan_enabled)} slow_clock={int(slow_clock_enabled)} "
                    f"conductor={int(conductor_enabled)} metamer={int(metamer_enabled)} "
                    f"text_prompt={int(text_prompt_conditioned)} text_mode={text_condition_mode_default} "
                    f"semantic_mode={semantic_condition_mode_default}"
                ),
                "use_existing_checkpoint": bool(defaults.get("use_existing_checkpoint", True)),
                "s3": dict(defaults.get("s3", {})),
            }
        )

    for r in named_runs:
        model_type = str(r.get("model_type", "rafa"))
        memory_enabled = _to_bool(r.get("memory_enabled", True))
        ramanujan_enabled = _to_bool(r.get("ramanujan_enabled", True))
        slow_clock_enabled = _to_bool(r.get("slow_clock_enabled", True))
        conductor_enabled = _to_bool(r.get("hyena_conductor_enabled", True))
        metamer_enabled = _to_bool(r.get("metamer_enabled", True))
        text_prompt_conditioned = _to_bool(r.get("text_prompt_conditioned", False))
        text_condition_mode = str(r.get("text_condition_mode", defaults.get("text_condition_mode", "film")))
        semantic_condition_mode = str(r.get("semantic_condition_mode", defaults.get("semantic_condition_mode", "none")))
        name = str(
            r.get("name")
            or _run_name(
                eval_prefix,
                memory_enabled,
                ramanujan_enabled,
                slow_clock_enabled,
                conductor_enabled,
                metamer_enabled,
                text_prompt_conditioned,
                text_condition_mode,
                model_type,
            )
        )
        jobs.append(
            {
                "name": name,
                "model_type": model_type,
                "max_steps": int(r.get("max_steps", max_steps)),
                "memory_enabled": memory_enabled,
                "ablate_ramanujan": not ramanujan_enabled,
                "ablate_slow_clock": not slow_clock_enabled,
                "hyena_conductor_enabled": conductor_enabled,
                "metamer_enabled": metamer_enabled,
                "text_prompt_conditioned": text_prompt_conditioned,
                "text_condition_mode": text_condition_mode,
                "semantic_condition_mode": semantic_condition_mode,
                "model_text_conditioned": bool(r.get("model_text_conditioned", defaults.get("model_text_conditioned", text_condition_mode != "ifs_control"))),
                "text_encoder": str(r.get("text_encoder", defaults.get("text_encoder", "laion/clap-htsat-unfused"))),
                "guidance_scale": float(r.get("guidance_scale", defaults.get("guidance_scale", 3.0))),
                "p_uncond": float(r.get("p_uncond", defaults.get("p_uncond", 0.15))),
                "cond_mag_scale": float(r.get("cond_mag_scale", defaults.get("cond_mag_scale", 0.1))),
                "text_ifs_control_strength": float(r.get("text_ifs_control_strength", defaults.get("text_ifs_control_strength", 0.25))),
                "semantic_tension_strength": float(r.get("semantic_tension_strength", defaults.get("semantic_tension_strength", 0.75))),
                "semantic_tension_alpha_strength": float(r.get("semantic_tension_alpha_strength", defaults.get("semantic_tension_alpha_strength", 0.2))),
                "semantic_tension_temp_strength": float(r.get("semantic_tension_temp_strength", defaults.get("semantic_tension_temp_strength", 0.35))),
                "semantic_tension_delta_strength": float(r.get("semantic_tension_delta_strength", defaults.get("semantic_tension_delta_strength", 0.25))),
                "phase_qset": list(r.get("phase_qset", defaults.get("phase_qset", []))),
                "val_steps": int(r.get("val_steps", defaults.get("val_steps", 50))),
                "require_clap_cond": bool(r.get("require_clap_cond", defaults.get("require_clap_cond", False))) or bool(text_prompt_conditioned),
                "notes": str(r.get("notes", "")),
                "use_existing_checkpoint": bool(r.get("use_existing_checkpoint", defaults.get("use_existing_checkpoint", True))),
                "s3": dict(r.get("s3", defaults.get("s3", {}))),
            }
        )

    dedup: dict[str, dict[str, Any]] = {}
    for j in jobs:
        dedup[j["name"]] = j
    return list(dedup.values())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True, help="Path to hypercube manifest YAML.")
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
