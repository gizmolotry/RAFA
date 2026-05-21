from __future__ import annotations

import argparse
import json
import subprocess
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any, Iterable, Sequence


RUNTIME_DIR = Path(__file__).resolve().parent
DEFAULT_BASE_CONFIG = Path(
    r"D:\RAFA\checkpoints_circleworld_proto\training_run_2026-04-24_agreement_scout_v1"
    r"\circleworld_real_anchor_config_cem_v1.json"
)
DEFAULT_CASES_JSON = Path(
    r"C:\Users\Andrew\AppData\Local\Temp\rafa_tokenburst\audio_predeclared_lockbox_v1_2026_05_06"
    r"\audio_predeclared_lockbox_cases.json"
)
DEFAULT_OUT_DIR = Path(
    r"C:\Users\Andrew\AppData\Local\Temp\rafa_tokenburst"
    r"\internal_phase_law_objective_scout_auto_2026_05_07"
)

PROBE_SCRIPT = RUNTIME_DIR / "run_audio_delta_mechanism_probe.py"
ABS_COMPARE_SCRIPT = RUNTIME_DIR / "compare_audio_lockbox_results.py"
VARIANT_COMPARE_SCRIPT = RUNTIME_DIR / "compare_internal_phase_law_variants.py"
SCORE_SCRIPT = RUNTIME_DIR / "score_internal_phase_law_objective.py"
JOINT_DIAGNOSTIC_SCRIPT = RUNTIME_DIR / "diagnose_internal_phase_law_joint_rows.py"

PROBE_JSON = "audio_delta_mechanism_probe.json"
ABS_COMPARE_JSON = "audio_lockbox_result_compare.json"
VARIANT_COMPARE_JSON = "internal_phase_law_variant_compare.json"
SCORE_JSON = "internal_phase_law_objective_score.json"
JOINT_DIAGNOSTIC_JSON = "internal_phase_law_joint_row_diagnostics.json"
MANIFEST_JSON = "internal_phase_law_objective_scout_manifest.json"
TRACK_REPORT = "TRACK_REPORT.md"

PHASE_LAW_OFF = {
    "phase_law_precondition_gain": 0.0,
    "phase_law_velocity_mix": 0.0,
    "phase_law_stability_gain": 1.0,
    "phase_law_softclip": 0.0,
    "phase_law_low_rank": 0,
    "phase_law_consensus_mix": 0.0,
    "phase_law_consensus_damping": 0.0,
    "phase_law_median_guard": 0.0,
    "phase_law_local_velocity_mix": 0.0,
    "phase_law_local_coherence_damping": 0.0,
    "phase_law_curvature_guard": 0.0,
    "phase_law_reentry_mix": 0.0,
    "phase_law_reentry_accel_mix": 0.0,
    "phase_law_reentry_causal": False,
}


def _default_variant_templates() -> list[dict[str, Any]]:
    return [
        {
            "name": "phase_law_off",
            "label": "noop",
            "description": "Explicit no-op/off internal phase-law control.",
            "updates": dict(PHASE_LAW_OFF),
            "is_baseline": True,
        },
        {
            "name": "internal_gate_025_lr12",
            "label": "gate025",
            "description": "Low-gain stability-gated internal phase-law preconditioner.",
            "updates": {
                **PHASE_LAW_OFF,
                "phase_law_precondition_gain": 0.25,
                "phase_law_velocity_mix": 0.0,
                "phase_law_stability_gain": 2.0,
                "phase_law_softclip": 0.0,
                "phase_law_low_rank": 12,
            },
            "is_baseline": False,
        },
        {
            "name": "internal_gate_050_lr12",
            "label": "gate050",
            "description": "Medium-gain stability-gated internal phase-law preconditioner.",
            "updates": {
                **PHASE_LAW_OFF,
                "phase_law_precondition_gain": 0.5,
                "phase_law_velocity_mix": 0.0,
                "phase_law_stability_gain": 2.0,
                "phase_law_softclip": 0.0,
                "phase_law_low_rank": 12,
            },
            "is_baseline": False,
        },
        {
            "name": "internal_velocity_025_lr12",
            "label": "velocity025",
            "description": "Low-gain stability-gated phase law with a small velocity mix.",
            "updates": {
                **PHASE_LAW_OFF,
                "phase_law_precondition_gain": 0.25,
                "phase_law_velocity_mix": 0.05,
                "phase_law_stability_gain": 2.0,
                "phase_law_softclip": 0.0,
                "phase_law_low_rank": 12,
            },
            "is_baseline": False,
        },
        {
            "name": "internal_softclip_050_lr12",
            "label": "softclip050",
            "description": "Medium-gain stability-gated phase law with soft clipping.",
            "updates": {
                **PHASE_LAW_OFF,
                "phase_law_precondition_gain": 0.5,
                "phase_law_velocity_mix": 0.0,
                "phase_law_stability_gain": 2.0,
                "phase_law_softclip": 0.35,
                "phase_law_low_rank": 12,
            },
            "is_baseline": False,
        },
    ]


def _safe_name(value: Any, fallback: str) -> str:
    raw = str(value or "").strip() or fallback
    safe = "".join(ch if ch.isalnum() or ch in ("_", "-") else "_" for ch in raw)
    return safe.strip("_") or fallback


def _normalize_variant_templates(raw_templates: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    allowed_updates = set(PHASE_LAW_OFF)
    variants: list[dict[str, Any]] = []
    for idx, raw in enumerate(raw_templates):
        if not isinstance(raw, dict):
            raise RuntimeError(f"Variant spec row {idx} must be a JSON object")
        updates = raw.get("updates", {})
        if not isinstance(updates, dict):
            raise RuntimeError(f"Variant spec row {idx} updates must be a JSON object")
        unknown = sorted(set(updates) - allowed_updates)
        if unknown:
            raise RuntimeError(f"Variant spec row {idx} has unknown phase-law keys: {unknown}")
        name = _safe_name(raw.get("name"), f"variant_{idx:03d}")
        label = _safe_name(raw.get("label"), name)
        normalized_updates = dict(PHASE_LAW_OFF)
        normalized_updates.update(updates)
        variants.append(
            {
                "name": name,
                "label": label,
                "description": str(raw.get("description") or ""),
                "updates": normalized_updates,
                "is_baseline": bool(raw.get("is_baseline", False)),
            }
        )

    if not any(variant["is_baseline"] for variant in variants):
        variants.insert(
            0,
            {
                "name": "phase_law_off",
                "label": "noop",
                "description": "Explicit no-op/off internal phase-law control.",
                "updates": dict(PHASE_LAW_OFF),
                "is_baseline": True,
            },
        )
    if sum(1 for variant in variants if variant["is_baseline"]) != 1:
        raise RuntimeError("Exactly one internal phase-law variant must set is_baseline=true")
    labels = [variant["label"] for variant in variants]
    if len(labels) != len(set(labels)):
        raise RuntimeError(f"Variant labels must be unique: {labels}")
    names = [variant["name"] for variant in variants]
    if len(names) != len(set(names)):
        raise RuntimeError(f"Variant names must be unique: {names}")
    return variants


def _variant_templates(spec_path: Path | None = None) -> list[dict[str, Any]]:
    if spec_path is None:
        return _default_variant_templates()
    payload = _load_json(spec_path)
    raw_variants = payload.get("variants") if isinstance(payload, dict) else payload
    if not isinstance(raw_variants, list):
        raise RuntimeError(f"Variant spec must be a list or object with a variants list: {spec_path}")
    return _normalize_variant_templates([row for row in raw_variants if isinstance(row, dict)])


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _csv_values(raw: str, *, label: str) -> str:
    values = [item.strip() for item in raw.split(",") if item.strip()]
    if not values:
        raise ValueError(f"At least one {label} is required")
    return ",".join(values)


def _csv_items(raw: str) -> list[str]:
    return [item.strip() for item in raw.split(",") if item.strip()]


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _validate_target_row_in_probe_grid(
    *,
    target_row: tuple[str, str, str, float],
    magnitude_modes: str,
    masks: str,
    mechanisms: str,
    gains: str,
) -> None:
    target_mode, target_mask, target_mechanism, target_gain = target_row
    missing: list[str] = []
    if target_mode not in _csv_items(magnitude_modes):
        missing.append(f"mode={target_mode}")
    if target_mask not in _csv_items(masks):
        missing.append(f"mask={target_mask}")
    if target_mechanism not in _csv_items(mechanisms):
        missing.append(f"mechanism={target_mechanism}")
    gains_float = [_as_float(item, default=float("nan")) for item in _csv_items(gains)]
    if not any(abs(value - target_gain) <= 1e-9 for value in gains_float):
        missing.append(f"gain={target_gain}")
    if missing:
        raise ValueError(
            "Target row is not included in the probe grid: "
            + ", ".join(missing)
            + ". Add it to --magnitude-modes/--masks/--mechanisms/--gains or choose a different target."
        )


def _command_to_string(command: Sequence[str]) -> str:
    return subprocess.list2cmdline([str(item) for item in command])


def _manifest_path_for_cases(cases_json: Path) -> Path | None:
    candidate = cases_json.with_name(cases_json.name.replace("_cases.json", "_manifest.json"))
    if candidate != cases_json and candidate.exists():
        return candidate.resolve()
    manifests = sorted(cases_json.parent.glob("*manifest*.json"))
    return manifests[0].resolve() if len(manifests) == 1 else None


def _iter_manifest_roles(value: Any) -> Iterable[str]:
    if isinstance(value, dict):
        role = value.get("role")
        if role:
            yield str(role)
        for item in value.values():
            yield from _iter_manifest_roles(item)
    elif isinstance(value, list):
        for item in value:
            yield from _iter_manifest_roles(item)


def _effective_exclude_roles(manifest_path: Path | None, requested_roles: Sequence[str]) -> list[str]:
    requested = [role for role in (str(item).strip() for item in requested_roles) if role]
    if not requested or manifest_path is None or not manifest_path.exists():
        return sorted(set(requested))

    try:
        actual_roles = sorted(set(_iter_manifest_roles(_load_json(manifest_path))))
    except (OSError, json.JSONDecodeError):
        return sorted(set(requested))

    effective = set(requested)
    for requested_role in requested:
        for actual_role in actual_roles:
            if requested_role == actual_role or requested_role in actual_role:
                effective.add(actual_role)
    return sorted(effective)


def _config_body(payload: dict[str, Any]) -> dict[str, Any]:
    raw = payload.get("config") if "config" in payload else payload
    if not isinstance(raw, dict):
        raise RuntimeError("Base Circleworld config JSON must contain an object config.")
    return raw


def _write_variant_configs(base_config: Path, out_dir: Path, variant_spec_json: Path | None = None) -> list[dict[str, Any]]:
    base_payload = _load_json(base_config)
    if not isinstance(base_payload, dict):
        raise RuntimeError(f"Base Circleworld config must be a JSON object: {base_config}")

    config_dir = out_dir / "configs"
    variants: list[dict[str, Any]] = []
    for template in _variant_templates(variant_spec_json):
        payload = deepcopy(base_payload)
        config = _config_body(payload)
        config.update(template["updates"])
        payload["internal_phase_law_objective_scout"] = {
            "variant_name": template["name"],
            "variant_label": template["label"],
            "description": template["description"],
            "updates": template["updates"],
            "baseline_noop": bool(template["is_baseline"]),
        }

        config_path = config_dir / f"{template['name']}.json"
        _write_json(config_path, payload)
        variants.append(
            {
                **template,
                "config_path": str(config_path.resolve()),
                "run_dir": str((out_dir / "probe_runs" / template["name"]).resolve()),
                "result_path": str((out_dir / "probe_runs" / template["name"] / PROBE_JSON).resolve()),
            }
        )
    return variants


def _probe_command(
    *,
    variant: dict[str, Any],
    cases_json: Path,
    device: str,
    num_cases: int,
    mechanisms: str,
    gains: str,
    magnitude_modes: str,
    masks: str,
) -> list[str]:
    return [
        sys.executable,
        str(PROBE_SCRIPT),
        "--config",
        str(variant["config_path"]),
        "--out-dir",
        str(variant["run_dir"]),
        "--device",
        device,
        "--num-cases",
        str(num_cases),
        "--magnitude-modes",
        magnitude_modes,
        "--masks",
        masks,
        "--mechanisms",
        mechanisms,
        "--gains",
        gains,
        "--cases-json",
        str(cases_json),
    ]


def _absolute_compare_command(
    *,
    variants: Sequence[dict[str, Any]],
    out_dir: Path,
    manifest_path: Path | None,
    exclude_roles: Sequence[str],
) -> list[str]:
    command = [sys.executable, str(ABS_COMPARE_SCRIPT)]
    for variant in variants:
        command.extend(["--result", f"{variant['label']}={variant['result_path']}"])
    if manifest_path is not None:
        command.extend(["--manifest", str(manifest_path)])
    command.extend(["--out-dir", str(out_dir), "--all-rows"])
    for role in exclude_roles:
        command.extend(["--exclude-role", role])
    return command


def _variant_compare_command(
    *,
    variants: Sequence[dict[str, Any]],
    out_dir: Path,
    manifest_path: Path | None,
    exclude_roles: Sequence[str],
) -> list[str]:
    baseline = next(variant for variant in variants if variant["is_baseline"])
    command = [
        sys.executable,
        str(VARIANT_COMPARE_SCRIPT),
        "--baseline",
        f"{baseline['label']}={baseline['result_path']}",
        "--out-dir",
        str(out_dir),
    ]
    for variant in variants:
        if not variant["is_baseline"]:
            command.extend(["--result", f"{variant['label']}={variant['result_path']}"])
    if manifest_path is not None:
        command.extend(["--manifest", str(manifest_path)])
    for role in exclude_roles:
        command.extend(["--exclude-role", role])
    return command


def _score_command(
    *,
    absolute_compare: Path,
    variant_compare: Path,
    out_dir: Path,
    target_row: tuple[str, str, str, float],
) -> list[str]:
    return [
        sys.executable,
        str(SCORE_SCRIPT),
        "--absolute-compare",
        str(absolute_compare),
        "--variant-compare",
        str(variant_compare),
        "--out-dir",
        str(out_dir),
        "--target-mode",
        target_row[0],
        "--target-mask",
        target_row[1],
        "--target-mechanism",
        target_row[2],
        "--target-gain",
        str(target_row[3]),
    ]


def _joint_diagnostic_command(
    *,
    absolute_compare: Path,
    variant_compare: Path,
    out_dir: Path,
    target_row: tuple[str, str, str, float],
) -> list[str]:
    return [
        sys.executable,
        str(JOINT_DIAGNOSTIC_SCRIPT),
        "--absolute-compare",
        str(absolute_compare),
        "--variant-compare",
        str(variant_compare),
        "--out-dir",
        str(out_dir),
        "--target-mode",
        target_row[0],
        "--target-mask",
        target_row[1],
        "--target-mechanism",
        target_row[2],
        "--target-gain",
        str(target_row[3]),
    ]


def _run_command(command: Sequence[str]) -> None:
    subprocess.run([str(item) for item in command], check=True)


def _build_manifest(
    *,
    status: str,
    base_config: Path,
    cases_json: Path,
    out_dir: Path,
    manifest_path: Path | None,
    requested_exclude_roles: Sequence[str],
    effective_exclude_roles: Sequence[str],
    variants: Sequence[dict[str, Any]],
    commands: Sequence[dict[str, Any]],
    dry_run: bool,
    settings: dict[str, Any],
) -> dict[str, Any]:
    return {
        "runtime": "circleworld_proto",
        "schema": "circleworld_internal_phase_law_objective_scout_auto_v0",
        "status": status,
        "dry_run": bool(dry_run),
        "base_config": str(base_config),
        "cases_json": str(cases_json),
        "lockbox_manifest": str(manifest_path) if manifest_path is not None else None,
        "out_dir": str(out_dir),
        "settings": settings,
        "requested_exclude_roles": list(requested_exclude_roles),
        "effective_exclude_roles": list(effective_exclude_roles),
        "scripts": {
            "probe": str(PROBE_SCRIPT),
            "absolute_compare": str(ABS_COMPARE_SCRIPT),
            "variant_compare": str(VARIANT_COMPARE_SCRIPT),
            "score": str(SCORE_SCRIPT),
            "joint_diagnostics": str(JOINT_DIAGNOSTIC_SCRIPT),
        },
        "outputs": {
            "manifest": str((out_dir / MANIFEST_JSON).resolve()),
            "absolute_compare_json": str((out_dir / "absolute_compare" / ABS_COMPARE_JSON).resolve()),
            "variant_compare_json": str((out_dir / "variant_compare" / VARIANT_COMPARE_JSON).resolve()),
            "objective_score_json": str((out_dir / "objective_score" / SCORE_JSON).resolve()),
            "joint_diagnostics_json": str((out_dir / "joint_diagnostics" / JOINT_DIAGNOSTIC_JSON).resolve()),
            "track_report": str((out_dir / TRACK_REPORT).resolve()),
        },
        "variants": list(variants),
        "commands": list(commands),
    }


def _load_optional_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    payload = _load_json(path)
    return payload if isinstance(payload, dict) else None


def _fmt(value: Any) -> str:
    if value is None:
        return "NA"
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def _write_report(path: Path, manifest: dict[str, Any]) -> None:
    score = _load_optional_json(Path(manifest["outputs"]["objective_score_json"]))
    joint = _load_optional_json(Path(manifest["outputs"]["joint_diagnostics_json"]))
    lines = [
        "# Internal Phase-Law Objective Scout Track Report",
        "",
        f"- status: `{manifest['status']}`",
        f"- dry run: `{manifest['dry_run']}`",
        f"- base config: `{manifest['base_config']}`",
        f"- cases: `{manifest['cases_json']}`",
        f"- lockbox manifest: `{manifest.get('lockbox_manifest')}`",
        f"- target row: `{(manifest.get('settings') or {}).get('target_row')}`",
        f"- requested excluded roles: `{', '.join(manifest.get('requested_exclude_roles') or []) or 'none'}`",
        f"- effective excluded roles: `{', '.join(manifest.get('effective_exclude_roles') or []) or 'none'}`",
        "",
        "## Variants",
        "",
        "| label | name | baseline | precondition | velocity | stability | softclip | low rank | consensus | damping | median guard | local velocity | local damping | curvature guard | reentry | reentry accel | reentry causal | result |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for variant in manifest.get("variants", []):
        updates = variant.get("updates", {})
        lines.append(
            "| {label} | {name} | {baseline} | {pre} | {vel} | {stab} | {clip} | {rank} | {cons} | {damp} | {guard} | {local_vel} | {local_damp} | {curv} | {reentry} | {reentry_accel} | {reentry_causal} | `{result}` |".format(
                label=variant.get("label"),
                name=variant.get("name"),
                baseline=variant.get("is_baseline"),
                pre=_fmt(updates.get("phase_law_precondition_gain")),
                vel=_fmt(updates.get("phase_law_velocity_mix")),
                stab=_fmt(updates.get("phase_law_stability_gain")),
                clip=_fmt(updates.get("phase_law_softclip")),
                rank=_fmt(updates.get("phase_law_low_rank")),
                cons=_fmt(updates.get("phase_law_consensus_mix")),
                damp=_fmt(updates.get("phase_law_consensus_damping")),
                guard=_fmt(updates.get("phase_law_median_guard")),
                local_vel=_fmt(updates.get("phase_law_local_velocity_mix")),
                local_damp=_fmt(updates.get("phase_law_local_coherence_damping")),
                curv=_fmt(updates.get("phase_law_curvature_guard")),
                reentry=_fmt(updates.get("phase_law_reentry_mix")),
                reentry_accel=_fmt(updates.get("phase_law_reentry_accel_mix")),
                reentry_causal=_fmt(updates.get("phase_law_reentry_causal")),
                result=variant.get("result_path"),
            )
        )

    lines.extend(
        [
            "",
            "## Pipeline Outputs",
            "",
            f"- manifest: `{manifest['outputs']['manifest']}`",
            f"- absolute compare: `{manifest['outputs']['absolute_compare_json']}`",
            f"- variant compare: `{manifest['outputs']['variant_compare_json']}`",
            f"- objective score: `{manifest['outputs']['objective_score_json']}`",
            f"- joint diagnostics: `{manifest['outputs']['joint_diagnostics_json']}`",
        ]
    )
    if score is not None:
        lines.extend(
            [
                "",
                "## Objective Score",
                "",
                f"- status: `{score.get('status')}`",
                f"- best run: `{score.get('best_run')}`",
                f"- best decision: `{score.get('best_decision')}`",
                f"- best score: `{_fmt(score.get('best_score'))}`",
                f"- target candidates: `{_fmt(score.get('target_row_candidate_count', score.get('target_low_energy_candidate_count')))}` / `{_fmt(score.get('target_row_count', score.get('target_low_energy_row_count')))}`",
                f"- target best run / decision: `{score.get('target_row_best_run', score.get('target_low_energy_best_run'))}` / `{score.get('target_row_best_decision', score.get('target_low_energy_best_decision'))}`",
                f"- target direct / absolute / nonbad corr delta: `{_fmt(score.get('target_row_best_direct_mean_corr_delta', score.get('target_low_energy_best_direct_mean_corr_delta')))}` / `{_fmt(score.get('target_row_best_absolute_mean_corr_delta', score.get('target_low_energy_best_absolute_mean_corr_delta')))}` / `{_fmt(score.get('target_row_best_nonbad_bin_mean_corr_delta', score.get('target_low_energy_best_nonbad_bin_mean_corr_delta')))}`",
            ]
        )
    elif manifest["dry_run"]:
        lines.extend(["", "## Objective Score", "", "- skipped because `--dry-run` was set."])

    if joint is not None:
        best = joint.get("best_row") if isinstance(joint.get("best_row"), dict) else {}
        lines.extend(
            [
                "",
                "## Joint Row Diagnostics",
                "",
                f"- status: `{joint.get('status')}`",
                f"- candidates: `{_fmt(joint.get('candidate_count'))}` / rows `{_fmt(joint.get('row_count'))}`",
                f"- mean nonbad-bin corr delta: `{_fmt(joint.get('mean_absolute_nonbad_bin_corr_delta'))}`",
                f"- best row: `{best.get('run_label')}` / `{best.get('magnitude_mode')}` / `{best.get('mask_mode')}` / gain `{_fmt(best.get('gain'))}`",
                f"- best decision: `{best.get('decision')}`",
                f"- target candidates: `{_fmt(joint.get('target_row_candidate_count', joint.get('target_low_energy_candidate_count')))}` / `{_fmt(joint.get('target_row_count', joint.get('target_low_energy_row_count')))}`",
                f"- target best run / decision: `{joint.get('target_row_best_run', joint.get('target_low_energy_best_run'))}` / `{joint.get('target_row_best_decision', joint.get('target_low_energy_best_decision'))}`",
                f"- target direct / absolute / nonbad corr delta: `{_fmt(joint.get('target_row_best_direct_mean_corr_delta', joint.get('target_low_energy_best_direct_mean_corr_delta')))}` / `{_fmt(joint.get('target_row_best_absolute_mean_corr_delta', joint.get('target_low_energy_best_absolute_mean_corr_delta')))}` / `{_fmt(joint.get('target_row_best_nonbad_bin_mean_corr_delta', joint.get('target_low_energy_best_nonbad_bin_mean_corr_delta')))}`",
            ]
        )
    elif manifest["dry_run"]:
        lines.extend(["", "## Joint Row Diagnostics", "", "- skipped because `--dry-run` was set."])

    lines.extend(["", "## Commands", ""])
    for command in manifest.get("commands", []):
        lines.append(f"- `{command.get('name')}`: `{command.get('status')}`")
        lines.append(f"  `{command.get('command')}`")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_scout(
    *,
    base_config: Path,
    cases_json: Path,
    lockbox_manifest: Path | None,
    variant_spec_json: Path | None,
    out_dir: Path,
    device: str,
    num_cases: int,
    mechanisms: str,
    gains: str,
    magnitude_modes: str,
    masks: str,
    target_row: tuple[str, str, str, float],
    exclude_roles: Sequence[str],
    dry_run: bool,
) -> dict[str, Any]:
    base_config = base_config.resolve()
    cases_json = cases_json.resolve()
    variant_spec_json = variant_spec_json.resolve() if variant_spec_json is not None else None
    out_dir = out_dir.resolve()
    mechanisms = _csv_values(mechanisms, label="mechanism")
    gains = _csv_values(gains, label="gain")
    magnitude_modes = _csv_values(magnitude_modes, label="magnitude mode")
    masks = _csv_values(masks, label="mask")
    _validate_target_row_in_probe_grid(
        target_row=target_row,
        magnitude_modes=magnitude_modes,
        masks=masks,
        mechanisms=mechanisms,
        gains=gains,
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    variants = _write_variant_configs(base_config, out_dir, variant_spec_json)
    lockbox_manifest = lockbox_manifest.resolve() if lockbox_manifest is not None else _manifest_path_for_cases(cases_json)
    effective_excludes = _effective_exclude_roles(lockbox_manifest, exclude_roles)

    abs_dir = (out_dir / "absolute_compare").resolve()
    variant_dir = (out_dir / "variant_compare").resolve()
    score_dir = (out_dir / "objective_score").resolve()
    joint_dir = (out_dir / "joint_diagnostics").resolve()

    planned_commands: list[dict[str, Any]] = []
    for variant in variants:
        command = _probe_command(
            variant=variant,
            cases_json=cases_json,
            device=device,
            num_cases=num_cases,
            mechanisms=mechanisms,
            gains=gains,
            magnitude_modes=magnitude_modes,
            masks=masks,
        )
        planned_commands.append(
            {"name": f"probe:{variant['label']}", "kind": "probe", "status": "planned", "command": _command_to_string(command)}
        )

    abs_command = _absolute_compare_command(
        variants=variants,
        out_dir=abs_dir,
        manifest_path=lockbox_manifest,
        exclude_roles=effective_excludes,
    )
    variant_command = _variant_compare_command(
        variants=variants,
        out_dir=variant_dir,
        manifest_path=lockbox_manifest,
        exclude_roles=effective_excludes,
    )
    score_command = _score_command(
        absolute_compare=abs_dir / ABS_COMPARE_JSON,
        variant_compare=variant_dir / VARIANT_COMPARE_JSON,
        out_dir=score_dir,
        target_row=target_row,
    )
    joint_command = _joint_diagnostic_command(
        absolute_compare=abs_dir / ABS_COMPARE_JSON,
        variant_compare=variant_dir / VARIANT_COMPARE_JSON,
        out_dir=joint_dir,
        target_row=target_row,
    )
    planned_commands.extend(
        [
            {"name": "compare:absolute", "kind": "absolute_compare", "status": "planned", "command": _command_to_string(abs_command)},
            {"name": "compare:variant_vs_noop", "kind": "variant_compare", "status": "planned", "command": _command_to_string(variant_command)},
            {"name": "score:locked_objective", "kind": "score", "status": "planned", "command": _command_to_string(score_command)},
            {"name": "diagnose:joint_rows", "kind": "joint_diagnostics", "status": "planned", "command": _command_to_string(joint_command)},
        ]
    )

    settings = {
        "device": device,
        "num_cases": int(num_cases),
        "mechanisms": mechanisms,
        "gains": gains,
        "magnitude_modes": magnitude_modes,
        "masks": masks,
        "target_row": {
            "magnitude_mode": target_row[0],
            "mask_mode": target_row[1],
            "mechanism": target_row[2],
            "gain": target_row[3],
        },
        "variant_spec_json": str(variant_spec_json) if variant_spec_json is not None else None,
    }
    manifest = _build_manifest(
        status="dry_run_planned" if dry_run else "planned",
        base_config=base_config,
        cases_json=cases_json,
        out_dir=out_dir,
        manifest_path=lockbox_manifest,
        requested_exclude_roles=exclude_roles,
        effective_exclude_roles=effective_excludes,
        variants=variants,
        commands=planned_commands,
        dry_run=dry_run,
        settings=settings,
    )
    manifest_path = out_dir / MANIFEST_JSON
    _write_json(manifest_path, manifest)
    _write_report(out_dir / TRACK_REPORT, manifest)
    if dry_run:
        return manifest

    command_lookup = {
        command_info["command"]: command_info
        for command_info in planned_commands
    }
    status = "completed"
    try:
        for variant in variants:
            command = _probe_command(
                variant=variant,
                cases_json=cases_json,
                device=device,
                num_cases=num_cases,
                mechanisms=mechanisms,
                gains=gains,
                magnitude_modes=magnitude_modes,
                masks=masks,
            )
            command_info = command_lookup[_command_to_string(command)]
            command_info["status"] = "running"
            _write_json(manifest_path, manifest)
            _run_command(command)
            command_info["status"] = "completed"

        for command in (abs_command, variant_command, score_command, joint_command):
            command_info = command_lookup[_command_to_string(command)]
            command_info["status"] = "running"
            _write_json(manifest_path, manifest)
            _run_command(command)
            command_info["status"] = "completed"
    except subprocess.CalledProcessError as exc:
        status = "failed"
        for command_info in planned_commands:
            if command_info.get("status") == "running":
                command_info["status"] = "failed"
                command_info["returncode"] = exc.returncode
                break
        manifest["status"] = status
        _write_json(manifest_path, manifest)
        _write_report(out_dir / TRACK_REPORT, manifest)
        raise

    manifest["status"] = status
    _write_json(manifest_path, manifest)
    _write_report(out_dir / TRACK_REPORT, manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate and run internal phase-law variants against the locked audio objective."
    )
    parser.add_argument("--base-config", default=str(DEFAULT_BASE_CONFIG))
    parser.add_argument("--cases-json", default=str(DEFAULT_CASES_JSON))
    parser.add_argument(
        "--variant-spec-json",
        default=None,
        help="Optional JSON list/object of phase-law variants. A no-op baseline is inserted if omitted.",
    )
    parser.add_argument(
        "--lockbox-manifest",
        default=None,
        help="Optional lockbox manifest override. Defaults to a sibling manifest inferred from --cases-json.",
    )
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num-cases", type=int, default=0, help="0 means all cases in this harness.")
    parser.add_argument("--mechanisms", default="raw")
    parser.add_argument("--gains", default="0.0,1.0,2.0")
    parser.add_argument("--magnitude-modes", default="prefix_hold,flat")
    parser.add_argument("--masks", default="all_bins,high_energy_bins,low_energy_bins")
    parser.add_argument("--target-mode", default="flat")
    parser.add_argument("--target-mask", default="low_energy_bins")
    parser.add_argument("--target-mechanism", default="raw")
    parser.add_argument("--target-gain", type=float, default=2.0)
    parser.add_argument(
        "--exclude-role",
        action="append",
        default=["single_source"],
        help="Exclude a manifest role from compares. May be repeated.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Write configs, manifest, and report without subprocesses.")
    args = parser.parse_args()

    summary = run_scout(
        base_config=Path(args.base_config),
        cases_json=Path(args.cases_json),
        lockbox_manifest=Path(args.lockbox_manifest) if args.lockbox_manifest else None,
        variant_spec_json=Path(args.variant_spec_json) if args.variant_spec_json else None,
        out_dir=Path(args.out_dir),
        device=str(args.device),
        num_cases=int(args.num_cases),
        mechanisms=str(args.mechanisms),
        gains=str(args.gains),
        magnitude_modes=str(args.magnitude_modes),
        masks=str(args.masks),
        target_row=(
            str(args.target_mode),
            str(args.target_mask),
            str(args.target_mechanism),
            float(args.target_gain),
        ),
        exclude_roles=args.exclude_role or [],
        dry_run=bool(args.dry_run),
    )
    print(
        json.dumps(
            {
                "status": summary["status"],
                "dry_run": summary["dry_run"],
                "manifest": str(Path(summary["out_dir"]) / MANIFEST_JSON),
                "track_report": str(Path(summary["out_dir"]) / TRACK_REPORT),
                "variant_count": len(summary["variants"]),
                "objective_score": summary["outputs"]["objective_score_json"],
                "joint_diagnostics": summary["outputs"]["joint_diagnostics_json"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
