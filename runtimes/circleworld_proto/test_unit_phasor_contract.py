from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[2]
CORE = ROOT / "core"
LINEAGE = ROOT / "lineages" / "04_positive_replacement"
for path in (ROOT, CORE, LINEAGE):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))



def _exit_if_help_requested_without_runtime() -> None:
    if any(arg in ("-h", "--help") for arg in sys.argv[1:]):
        print(f"usage: {Path(__file__).name} [runtime-coupled contract options]")
        print()
        print("This contract is coupled to the Circleworld runtime/signature lane.")
        print("Run it from a Circleworld integration worktree for full argument parsing and execution.")
        raise SystemExit(0)


_exit_if_help_requested_without_runtime()

from ablate_formalization import _generate_seed_phase, make_seed_rafa
from circleworld import clone_circleworld_state, perturb_circleworld_state, recurse_circleworld
from evaluate_circleworld import _load_circle_cfg, _safe_device
from lib_blackwell import NakedRAFA
from rafa_math_tools import phase_to_phasor, phasor_normalize


DEFAULT_CONFIG = (
    Path(r"C:\Users\Andrew\AppData\Local\Temp\rafa_tokenburst")
    / "child_writeback_ablation_v3_unit_phasor_fixed_cuda_2026_05_06"
    / "support_operator_floor_high"
    / "circleworld_config.json"
)
DEFAULT_DIRECT_EXPORT_WAVS = (
    ROOT / "wav_files" / "soundbible_airplane-takeoff_c752b8ba8b.wav",
    ROOT / "wav_files" / "soundbible_steam-engine-running_68e20d27d1.wav",
)

PHASOR_KEYS = {"phase_state", "mixed_phase_state", "phase_modes", "seed_state"}


def _as_float(value: torch.Tensor | float | int) -> float:
    if torch.is_tensor(value):
        return float(value.detach().cpu().item())
    return float(value)


def _phasor_norm_stats(path: str, tensor: torch.Tensor) -> dict[str, Any]:
    z = tensor.detach()
    norm = torch.linalg.vector_norm(z, dim=-1)
    err = (norm - 1.0).abs()
    finite = torch.isfinite(z).all(dim=-1) & torch.isfinite(norm)
    finite_fraction = 1.0 if bool(finite.all().item()) else _as_float(finite.double().mean())
    return {
        "path": path,
        "shape": list(z.shape),
        "dtype": str(z.dtype),
        "device": str(z.device),
        "max_abs_norm_error": _as_float(err.max()),
        "mean_abs_norm_error": _as_float(err.mean()),
        "min_norm": _as_float(norm.min()),
        "max_norm": _as_float(norm.max()),
        "finite_fraction": finite_fraction,
    }


def _is_phasor_path(path: str, value: Any) -> bool:
    if not torch.is_tensor(value):
        return False
    if value.dim() < 1 or value.size(-1) != 2:
        return False
    key = path.split(".")[-1]
    return key in PHASOR_KEYS


def _collect_phasor_stats(value: Any, path: str, out: list[dict[str, Any]]) -> None:
    if _is_phasor_path(path, value):
        out.append(_phasor_norm_stats(path, value))
        return
    if isinstance(value, dict):
        for key, child in value.items():
            _collect_phasor_stats(child, f"{path}.{key}", out)
        return
    if isinstance(value, (list, tuple)):
        for idx, child in enumerate(value):
            _collect_phasor_stats(child, f"{path}[{idx}]", out)


def _group_for_path(path: str) -> str:
    if path.startswith("nested."):
        if "identity_probe_state" in path:
            return "nested_identity_probe"
        if "pre_unroll" in path:
            return "nested_pre_unroll"
        if "branch_start" in path:
            return "nested_branch_start"
        return "nested"
    if path.startswith("export."):
        if "blend" in path:
            return "export_blend"
        return "export"
    if path.startswith("direct_export."):
        if "blend" in path or "final_phase" in path:
            return "direct_export_blend"
        return "direct_export"
    if path.startswith("audio_continuation."):
        if "final_phase" in path:
            return "audio_continuation_final"
        return "audio_continuation"
    if path.startswith("input."):
        return "input"
    if "child_worlds" in path:
        return "child_worlds"
    if "multimode_history" in path:
        return "multimode_history"
    if "multimode_state" in path:
        return "multimode_state"
    if "packets" in path:
        return "packets"
    if "mixed_phase_state" in path:
        return "mixed_phase_state"
    if path.endswith(".phase_state"):
        return "phase_state"
    return "other"


def _aggregate_stats(rows: list[dict[str, Any]], tolerance: float) -> dict[str, Any]:
    if not rows:
        return {
            "status": "fail_no_phasor_tensors",
            "tensor_count": 0,
            "max_abs_norm_error": None,
            "max_phasor_norm_error": None,
            "mean_abs_norm_error": None,
            "min_finite_fraction": None,
            "num_norm_violations": 0,
            "violating_paths": [],
            "phase_only_contract_passed": False,
            "hidden_magnitude_channel_detected": False,
            "passing": False,
        }
    max_err = max(float(row["max_abs_norm_error"]) for row in rows)
    mean_err = sum(float(row["mean_abs_norm_error"]) for row in rows) / len(rows)
    min_finite = min(float(row["finite_fraction"]) for row in rows)
    passing = bool(max_err <= tolerance and min_finite >= 1.0)
    violations = [row for row in rows if float(row["max_abs_norm_error"]) > tolerance or float(row["finite_fraction"]) < 1.0]
    return {
        "status": "pass" if passing else "fail",
        "tensor_count": len(rows),
        "max_abs_norm_error": max_err,
        "max_phasor_norm_error": max_err,
        "mean_abs_norm_error": mean_err,
        "min_finite_fraction": min_finite,
        "num_norm_violations": len(violations),
        "violating_paths": [str(row["path"]) for row in violations[:25]],
        "phase_only_contract_passed": passing,
        "hidden_magnitude_channel_detected": bool(violations),
        "tolerance": float(tolerance),
        "passing": passing,
    }


def _aggregate_by_group(rows: list[dict[str, Any]]) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(_group_for_path(str(row["path"])), []).append(row)
    out: dict[str, Any] = {}
    for group, group_rows in sorted(groups.items()):
        out[group] = {
            "tensor_count": len(group_rows),
            "max_abs_norm_error": max(float(row["max_abs_norm_error"]) for row in group_rows),
            "mean_abs_norm_error": sum(float(row["mean_abs_norm_error"]) for row in group_rows) / len(group_rows),
            "min_finite_fraction": min(float(row["finite_fraction"]) for row in group_rows),
            "worst_path": max(group_rows, key=lambda row: float(row["max_abs_norm_error"]))["path"],
        }
    return out


def _mixture_norm_stats(path: str, state: dict[str, Any], cfg: Any) -> dict[str, Any] | None:
    phase_modes = state.get("phase_modes")
    logits = state.get("mode_logits")
    support = state.get("mode_support")
    if not (torch.is_tensor(phase_modes) and torch.is_tensor(logits) and torch.is_tensor(support)):
        return None
    if phase_modes.dim() < 5 or phase_modes.size(-1) != 2:
        return None
    weights = torch.softmax(logits / max(1e-4, float(cfg.readout_temperature)), dim=-1) * support.clamp_min(1e-4)
    weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-8)
    raw = (weights.unsqueeze(-1) * phase_modes).sum(dim=3)
    norm = torch.linalg.vector_norm(raw.detach(), dim=-1)
    finite = torch.isfinite(raw).all(dim=-1) & torch.isfinite(norm)
    finite_fraction = 1.0 if bool(finite.all().item()) else _as_float(finite.double().mean())
    return {
        "path": path,
        "shape": list(raw.shape),
        "min_raw_mixture_norm": _as_float(norm.min()),
        "mean_raw_mixture_norm": _as_float(norm.mean()),
        "max_raw_mixture_norm": _as_float(norm.max()),
        "frac_norm_lt_1e_6": _as_float((norm < 1.0e-6).float().mean()),
        "frac_norm_lt_1e_4": _as_float((norm < 1.0e-4).float().mean()),
        "frac_norm_lt_1e_3": _as_float((norm < 1.0e-3).float().mean()),
        "finite_fraction": finite_fraction,
    }


def _collect_mixture_stats(value: Any, path: str, cfg: Any, out: list[dict[str, Any]]) -> None:
    if isinstance(value, dict):
        row = _mixture_norm_stats(path, value, cfg)
        if row is not None:
            out.append(row)
        for key, child in value.items():
            _collect_mixture_stats(child, f"{path}.{key}", cfg, out)
        return
    if isinstance(value, (list, tuple)):
        for idx, child in enumerate(value):
            _collect_mixture_stats(child, f"{path}[{idx}]", cfg, out)


def _aggregate_mixture_stats(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {
            "status": "not_applicable",
            "state_count": 0,
            "min_raw_mixture_norm": None,
            "frac_norm_lt_1e_4": None,
            "min_finite_fraction": None,
        }
    min_norm = min(float(row["min_raw_mixture_norm"]) for row in rows)
    max_low = max(float(row["frac_norm_lt_1e_4"]) for row in rows)
    min_finite = min(float(row["finite_fraction"]) for row in rows)
    status = "pass" if min_norm >= 1.0e-4 and max_low == 0.0 and min_finite >= 1.0 else "needs_review"
    return {
        "status": status,
        "state_count": len(rows),
        "min_raw_mixture_norm": min_norm,
        "mean_raw_mixture_norm": sum(float(row["mean_raw_mixture_norm"]) for row in rows) / len(rows),
        "max_raw_mixture_norm": max(float(row["max_raw_mixture_norm"]) for row in rows),
        "max_frac_norm_lt_1e_6": max(float(row["frac_norm_lt_1e_6"]) for row in rows),
        "max_frac_norm_lt_1e_4": max_low,
        "max_frac_norm_lt_1e_3": max(float(row["frac_norm_lt_1e_3"]) for row in rows),
        "min_finite_fraction": min_finite,
        "worst_path": min(rows, key=lambda row: float(row["min_raw_mixture_norm"]))["path"],
    }


def _parse_seeds(raw: str, source: str) -> list[int]:
    text = str(raw).strip()
    if not text:
        raise ValueError("--seeds must be a positive count or comma-separated seed list")
    if "," in text:
        return [int(part.strip()) for part in text.split(",") if part.strip()]
    count = int(text)
    if count <= 0:
        raise ValueError("--seeds count must be positive")
    base = 6100 if source == "naked_rafa" else 5100
    return [base + idx for idx in range(count)]


def _seed_plan(seed_source: str, seeds: str) -> list[tuple[str, int]]:
    sources = ["synthetic", "naked_rafa"] if seed_source == "both" else [seed_source]
    plan: list[tuple[str, int]] = []
    for source in sources:
        plan.extend((source, seed) for seed in _parse_seeds(seeds, source))
    return plan


def _direct_export_wav_paths(raw: str) -> list[Path]:
    text = str(raw).strip()
    if not text:
        return [path for path in DEFAULT_DIRECT_EXPORT_WAVS if path.exists()]
    paths = [Path(part.strip()) for part in text.replace(";", ",").split(",") if part.strip()]
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Direct export WAV path(s) not found: {missing}")
    return paths


def _mode_for_cfg(cfg: Any) -> str:
    return cfg.branching_mode if cfg.branching_mode in {"native_multimode", "native_multimode_childworld"} else "active_packets"


def _run_case(
    *,
    cfg: Any,
    rafa_core: NakedRAFA | None,
    source: str,
    seed: int,
    time_steps: int,
    device: torch.device,
    tolerance: float,
) -> dict[str, Any]:
    phase_state = _generate_seed_phase(
        rafa_core=rafa_core,
        batch_size=1,
        time_steps=time_steps,
        device=device,
        seed_source=source,
        seed=int(seed),
    ).detach()
    run_mode = _mode_for_cfg(cfg)
    run = recurse_circleworld(phase_state, cfg=cfg, depth=cfg.recursion_depth, mode=run_mode)

    rows: list[dict[str, Any]] = []
    _collect_phasor_stats({"phase_state": phase_state}, "input", rows)
    _collect_phasor_stats(run, "run", rows)
    mixture_rows: list[dict[str, Any]] = []
    _collect_mixture_stats(run, "run", cfg, mixture_rows)
    aggregate = _aggregate_stats(rows, tolerance=tolerance)
    return {
        "surface": "heldout",
        "source": source,
        "seed": int(seed),
        "mode": run_mode,
        "tensor_count": len(rows),
        "aggregate": aggregate,
        "groups": _aggregate_by_group(rows),
        "mixture_cancellation": _aggregate_mixture_stats(mixture_rows),
        "mixture_rows": mixture_rows,
        "rows": rows,
        "child_world_count": len(run.get("child_worlds", []) or []),
        "child_event_count": len(run.get("child_event_history", []) or []),
    }


def _nested_branch_specs(cfg: Any) -> list[dict[str, Any]]:
    from test_nested_commitment import _lifecycle_cfg

    return [
        {"name": "base", "kind": None, "strength": 0.0, "seed": 0, "cfg": cfg},
        {"name": "logit_tilt", "kind": "logit_tilt", "strength": 1.1, "seed": 101, "cfg": cfg},
        {"name": "support_expand", "kind": "support_expand", "strength": 0.35, "seed": 202, "cfg": cfg},
        {"name": "support_suppress", "kind": "support_suppress", "strength": 0.60, "seed": 303, "cfg": cfg},
        {"name": "qtrace_shift", "kind": "qtrace_shift", "strength": 0.60, "seed": 404, "cfg": cfg},
        {"name": "child_phase_shift", "kind": "child_phase_shift", "strength": 0.90, "seed": 505, "cfg": cfg},
        {"name": "child_support_boost", "kind": "child_support_boost", "strength": 0.35, "seed": 606, "cfg": cfg},
        {"name": "child_qtrace_shift", "kind": "child_qtrace_shift", "strength": 0.60, "seed": 707, "cfg": cfg},
        {"name": "child_parent_mix_shift", "kind": "child_parent_mix_shift", "strength": 0.80, "seed": 1001, "cfg": cfg},
        {
            "name": "child_writeback_gate_shift",
            "kind": "child_writeback_ready_shift",
            "strength": 1.00,
            "seed": 1444,
            "cfg": _lifecycle_cfg(cfg, "writeback"),
            "pre_unroll": "child_only",
            "extra_depth": 1,
        },
        {
            "name": "child_mode1_replace_50_shift",
            "kind": "child_phase_shift",
            "strength": 1.10,
            "seed": 2222,
            "cfg": _lifecycle_cfg(cfg, "retention"),
            "pre_unroll": "child_delayed_remix",
            "readout_mode": "child_mode1_replace_50",
            "extra_depth": 1,
        },
        {
            "name": "child_mode1_replace_85_shift",
            "kind": "child_writeback_ready_shift",
            "strength": 1.10,
            "seed": 2333,
            "cfg": _lifecycle_cfg(cfg, "writeback"),
            "pre_unroll": "child_delayed_remix",
            "readout_mode": "child_mode1_replace_85",
            "extra_depth": 1,
        },
    ]


def _select_nested_fork_state(states: list[Any]) -> tuple[int, Any]:
    fallback_idx = max(0, len(states) - 1)
    for idx, state in enumerate(states):
        if not isinstance(state, dict):
            continue
        children = state.get("child_worlds") or []
        if any(isinstance(child, dict) and child.get("active", False) for child in children):
            return idx, state
    for idx, state in enumerate(states):
        if isinstance(state, dict) and state.get("child_worlds"):
            return idx, state
    return fallback_idx, states[fallback_idx]


def _apply_nested_pre_unroll(state: dict[str, Any], cfg: Any, pre_unroll: str | None) -> dict[str, Any]:
    from test_nested_commitment import _child_only_pre_unroll

    if pre_unroll == "child_only":
        return _child_only_pre_unroll(state, cfg, steps=2, parent_write_scale=0.0, mode0_protect=0.97)
    if pre_unroll == "child_delayed_remix":
        return _child_only_pre_unroll(state, cfg, steps=2, parent_write_scale=0.10, mode0_protect=0.97)
    return state


def _run_nested_case(
    *,
    cfg: Any,
    rafa_core: NakedRAFA | None,
    source: str,
    seed: int,
    time_steps: int,
    device: torch.device,
    tolerance: float,
    nested_depth: int,
) -> dict[str, Any]:
    from test_nested_commitment import _apply_readout_override, _run_depth_trace

    phase_state = _generate_seed_phase(
        rafa_core=rafa_core,
        batch_size=1,
        time_steps=time_steps,
        device=device,
        seed_source=source,
        seed=int(seed),
    ).detach()
    run_mode = _mode_for_cfg(cfg)
    trace = _run_depth_trace(phase_state, cfg=cfg, depth=max(1, int(nested_depth)), mode=run_mode)
    fork_idx, fork_state_raw = _select_nested_fork_state(trace.get("states", []))
    fork_state = clone_circleworld_state(fork_state_raw)

    rows: list[dict[str, Any]] = []
    mixture_rows: list[dict[str, Any]] = []
    _collect_phasor_stats({"phase_state": phase_state}, f"nested.{source}_{seed}.input", rows)
    _collect_phasor_stats(trace, f"nested.{source}_{seed}.trace", rows)
    _collect_mixture_stats(trace, f"nested.{source}_{seed}.trace", cfg, mixture_rows)

    branch_count = 0
    for spec in _nested_branch_specs(cfg):
        branch_count += 1
        branch_name = str(spec["name"])
        branch_cfg = spec["cfg"]
        if spec["kind"] is None:
            branch_state = clone_circleworld_state(fork_state)
        else:
            branch_state = perturb_circleworld_state(
                fork_state,
                cfg,
                str(spec["kind"]),
                float(spec["strength"]),
                int(spec["seed"]),
            )
        _collect_phasor_stats(branch_state, f"nested.{source}_{seed}.{branch_name}.branch_start", rows)
        _collect_mixture_stats(branch_state, f"nested.{source}_{seed}.{branch_name}.branch_start", branch_cfg, mixture_rows)
        if isinstance(branch_state, dict) and spec.get("pre_unroll"):
            branch_state = _apply_nested_pre_unroll(branch_state, branch_cfg, str(spec.get("pre_unroll")))
            _collect_phasor_stats(branch_state, f"nested.{source}_{seed}.{branch_name}.pre_unroll", rows)
            _collect_mixture_stats(branch_state, f"nested.{source}_{seed}.{branch_name}.pre_unroll", branch_cfg, mixture_rows)
        branch_depth = max(1, int(nested_depth) + int(spec.get("extra_depth", 0)) - int(fork_idx))
        branch_run = _run_depth_trace(branch_state, cfg=branch_cfg, depth=branch_depth, mode=run_mode)
        branch_run = _apply_readout_override(branch_run, branch_cfg, spec.get("readout_mode"))
        _collect_phasor_stats(branch_run, f"nested.{source}_{seed}.{branch_name}.continuation", rows)
        _collect_mixture_stats(branch_run, f"nested.{source}_{seed}.{branch_name}.continuation", branch_cfg, mixture_rows)

    return {
        "surface": "nested",
        "source": source,
        "seed": int(seed),
        "mode": run_mode,
        "fork_depth": int(fork_idx),
        "branch_count": int(branch_count),
        "tensor_count": len(rows),
        "aggregate": _aggregate_stats(rows, tolerance=tolerance),
        "groups": _aggregate_by_group(rows),
        "mixture_cancellation": _aggregate_mixture_stats(mixture_rows),
        "mixture_rows": mixture_rows,
        "rows": rows,
        "child_world_count": sum(
            len(state.get("child_worlds", []) or []) for state in trace.get("states", []) if isinstance(state, dict)
        ),
        "child_event_count": len(trace.get("state_bundle", {}).get("child_event_history", []) or [])
        if isinstance(trace.get("state_bundle"), dict)
        else 0,
    }


def _run_export_case(
    *,
    cfg: Any,
    rafa_core: NakedRAFA | None,
    source: str,
    seed: int,
    time_steps: int,
    device: torch.device,
    tolerance: float,
    phase_blend: float,
) -> dict[str, Any]:
    phase_state = _generate_seed_phase(
        rafa_core=rafa_core,
        batch_size=1,
        time_steps=time_steps,
        device=device,
        seed_source=source,
        seed=int(seed),
    ).detach()
    original_phase = torch.atan2(phase_state[..., 1], phase_state[..., 0])
    original_z = phase_to_phasor(original_phase)
    run_mode = _mode_for_cfg(cfg)
    run = recurse_circleworld(phase_state, cfg=cfg, depth=cfg.recursion_depth, mode=run_mode)
    blend = float(min(1.0, max(0.0, phase_blend)))
    blended_z = phasor_normalize((1.0 - blend) * original_z + blend * run["phase_state"])

    rows: list[dict[str, Any]] = []
    mixture_rows: list[dict[str, Any]] = []
    _collect_phasor_stats({"phase_state": phase_state}, f"export.{source}_{seed}.input", rows)
    _collect_phasor_stats({"phase_state": original_z}, f"export.{source}_{seed}.original_phase_to_phasor", rows)
    _collect_phasor_stats(run, f"export.{source}_{seed}.run", rows)
    _collect_phasor_stats({"phase_state": blended_z}, f"export.{source}_{seed}.blend", rows)
    _collect_mixture_stats(run, f"export.{source}_{seed}.run", cfg, mixture_rows)
    return {
        "surface": "export",
        "source": source,
        "seed": int(seed),
        "mode": run_mode,
        "phase_blend": blend,
        "tensor_count": len(rows),
        "aggregate": _aggregate_stats(rows, tolerance=tolerance),
        "groups": _aggregate_by_group(rows),
        "mixture_cancellation": _aggregate_mixture_stats(mixture_rows),
        "mixture_rows": mixture_rows,
        "rows": rows,
        "child_world_count": len(run.get("child_worlds", []) or []),
        "child_event_count": len(run.get("child_event_history", []) or []),
    }


def _run_direct_export_case(
    *,
    cfg: Any,
    wav_path: Path,
    device_name: str,
    phase_blend: float,
    clip_seconds: int | None,
    tolerance: float,
) -> dict[str, Any]:
    from config import load_config
    from export_circleworld_audio import _blend_phase, prepare_reference_audio
    from stft_utils import compute_stft, inverse_stft

    runtime_cfg = load_config()
    stft_cfg = runtime_cfg["data"]["stft"]
    device = _safe_device(device_name)
    wav, _sr = prepare_reference_audio(
        wav_path=wav_path,
        device_name=device_name,
        clip_seconds_override=clip_seconds,
    )
    wav = wav.to(device)
    mag, phase = compute_stft(wav, stft_cfg)
    phase_state = phase_to_phasor(phase)
    run_mode = _mode_for_cfg(cfg)
    run = recurse_circleworld(phase_state, cfg=cfg, depth=cfg.recursion_depth, mode=run_mode)
    blend = float(min(1.0, max(0.0, phase_blend)))
    original_z = phase_to_phasor(phase)
    raw_blend = (1.0 - blend) * original_z + blend * run["phase_state"]
    raw_blend_norm = torch.linalg.vector_norm(raw_blend.detach(), dim=-1)
    final_phase = _blend_phase(phase, run["phase_state"], blend=blend)
    final_z = phase_to_phasor(final_phase)
    out_wav = inverse_stft(mag, final_phase, stft_cfg).squeeze(0)

    rows: list[dict[str, Any]] = []
    mixture_rows: list[dict[str, Any]] = []
    stem = wav_path.stem.replace(".", "_").replace(" ", "_")
    _collect_phasor_stats({"phase_state": phase_state}, f"direct_export.{stem}.stft_phase_state", rows)
    _collect_phasor_stats(run, f"direct_export.{stem}.run", rows)
    _collect_phasor_stats({"phase_state": final_z}, f"direct_export.{stem}.final_phase", rows)
    _collect_mixture_stats(run, f"direct_export.{stem}.run", cfg, mixture_rows)
    aggregate = _aggregate_stats(rows, tolerance=tolerance)
    return {
        "surface": "direct_export",
        "source": "wav",
        "seed": 0,
        "wav_path": str(wav_path),
        "mode": run_mode,
        "phase_blend": blend,
        "clip_seconds": clip_seconds,
        "tensor_count": len(rows),
        "aggregate": aggregate,
        "groups": _aggregate_by_group(rows),
        "mixture_cancellation": _aggregate_mixture_stats(mixture_rows),
        "mixture_rows": mixture_rows,
        "rows": rows,
        "direct_blend_raw_min_norm": _as_float(raw_blend_norm.min()),
        "direct_blend_raw_mean_norm": _as_float(raw_blend_norm.mean()),
        "direct_blend_raw_max_norm": _as_float(raw_blend_norm.max()),
        "output_finite_fraction": _as_float(torch.isfinite(out_wav).float().mean()),
        "child_world_count": len(run.get("child_worlds", []) or []),
        "child_event_count": len(run.get("child_event_history", []) or []),
    }


def _run_audio_continuation_case(
    *,
    cfg: Any,
    wav_path: Path,
    device_name: str,
    prefix_seconds: float,
    future_seconds: float,
    magnitude_mode: str,
    tolerance: float,
) -> dict[str, Any]:
    import math

    from benchmark_audio_continuation import (
        _build_magnitude_canvas,
        _extend_phase_from_prefix,
        _phase_to_angle,
        _prepare_anchor,
        _render_future_from_stft,
    )
    from config import load_config
    from stft_utils import compute_stft

    runtime_cfg = load_config()
    sr = int(runtime_cfg["data"]["sample_rate"])
    stft_cfg = runtime_cfg["data"]["stft"]
    device = _safe_device(device_name)
    prefix_samples = max(1, int(round(float(prefix_seconds) * sr)))
    future_samples = max(1, int(round(float(future_seconds) * sr)))
    total_samples = prefix_samples + future_samples
    wav = _prepare_anchor(wav_path, target_sr=sr, total_samples=total_samples)
    prefix = wav[..., :prefix_samples].to(device)

    prefix_mag, prefix_phase = compute_stft(prefix, stft_cfg)
    hop = int(stft_cfg["hop"])
    future_frames = max(1, int(math.ceil(float(future_samples) / float(hop))))
    initial_phase = _extend_phase_from_prefix(prefix_phase, future_frames=future_frames)
    initial_z = phase_to_phasor(initial_phase)

    run_mode = _mode_for_cfg(cfg)
    run = recurse_circleworld(initial_z, cfg=cfg, depth=cfg.recursion_depth, mode=run_mode)
    run_phase = _phase_to_angle(run["phase_state"])
    final_phase = torch.cat([prefix_phase, run_phase[..., prefix_phase.size(-1) :]], dim=-1)
    final_z = phase_to_phasor(final_phase)
    mag_canvas, mag_flags = _build_magnitude_canvas(prefix_mag, future_frames=future_frames, mode=magnitude_mode)
    future = _render_future_from_stft(
        mag_canvas=mag_canvas,
        phase_canvas=final_phase,
        stft_cfg=stft_cfg,
        prefix_samples=prefix_samples,
        future_samples=future_samples,
    )

    rows: list[dict[str, Any]] = []
    mixture_rows: list[dict[str, Any]] = []
    stem = wav_path.stem.replace(".", "_").replace(" ", "_")
    _collect_phasor_stats({"phase_state": initial_z}, f"audio_continuation.{stem}.initial_z", rows)
    _collect_phasor_stats(run, f"audio_continuation.{stem}.run", rows)
    _collect_phasor_stats({"phase_state": final_z}, f"audio_continuation.{stem}.final_phase", rows)
    _collect_mixture_stats(run, f"audio_continuation.{stem}.run", cfg, mixture_rows)
    return {
        "surface": "audio_continuation",
        "source": "wav",
        "seed": 0,
        "wav_path": str(wav_path),
        "mode": run_mode,
        "prefix_seconds": float(prefix_seconds),
        "future_seconds": float(future_seconds),
        "magnitude_mode": str(magnitude_mode),
        "magnitude_flags": mag_flags,
        "tensor_count": len(rows),
        "aggregate": _aggregate_stats(rows, tolerance=tolerance),
        "groups": _aggregate_by_group(rows),
        "mixture_cancellation": _aggregate_mixture_stats(mixture_rows),
        "mixture_rows": mixture_rows,
        "rows": rows,
        "output_finite_fraction": _as_float(torch.isfinite(future).double().mean()),
        "child_world_count": len(run.get("child_worlds", []) or []),
        "child_event_count": len(run.get("child_event_history", []) or []),
    }


def _run_negative_control(
    *,
    rafa_core: NakedRAFA | None,
    source: str,
    seed: int,
    time_steps: int,
    device: torch.device,
    tolerance: float,
    scale: float,
) -> dict[str, Any]:
    phase_state = _generate_seed_phase(
        rafa_core=rafa_core,
        batch_size=1,
        time_steps=time_steps,
        device=device,
        seed_source=source,
        seed=int(seed),
    ).detach()
    corrupted = phase_state.clone() * float(scale)
    rows: list[dict[str, Any]] = []
    _collect_phasor_stats(
        {"phase_state": corrupted},
        f"negative_control.{source}_{seed}.scaled_phase_state",
        rows,
    )
    aggregate = _aggregate_stats(rows, tolerance=tolerance)
    detected = bool(
        aggregate.get("hidden_magnitude_channel_detected")
        and not aggregate.get("phase_only_contract_passed", False)
    )
    return {
        "surface": "negative_control",
        "source": source,
        "seed": int(seed),
        "scale": float(scale),
        "expected_status": "fail",
        "detected": detected,
        "tensor_count": len(rows),
        "aggregate": aggregate,
        "rows": rows,
    }


def _source_summaries(cases: list[dict[str, Any]]) -> dict[str, Any]:
    by_source: dict[str, list[dict[str, Any]]] = {}
    for case in cases:
        by_source.setdefault(str(case["source"]), []).append(case)
    out: dict[str, Any] = {}
    for source, rows in sorted(by_source.items()):
        out[source] = {
            "num_cases": len(rows),
            "max_abs_norm_error": max(float(row["aggregate"]["max_abs_norm_error"]) for row in rows),
            "mean_abs_norm_error": sum(float(row["aggregate"]["mean_abs_norm_error"]) for row in rows) / len(rows),
            "min_finite_fraction": min(float(row["aggregate"]["min_finite_fraction"]) for row in rows),
            "passing_cases": sum(1 for row in rows if row["aggregate"]["passing"]),
            "child_world_count": sum(int(row.get("child_world_count", 0)) for row in rows),
            "child_event_count": sum(int(row.get("child_event_count", 0)) for row in rows),
            "min_raw_mixture_norm": min(
                float(row.get("mixture_cancellation", {}).get("min_raw_mixture_norm", 1.0)) for row in rows
            ),
        }
    return out


def _surface_summaries(cases: list[dict[str, Any]]) -> dict[str, Any]:
    by_surface: dict[str, list[dict[str, Any]]] = {}
    for case in cases:
        by_surface.setdefault(str(case.get("surface", "heldout")), []).append(case)
    out: dict[str, Any] = {}
    for surface, rows in sorted(by_surface.items()):
        out[surface] = {
            "num_cases": len(rows),
            "max_abs_norm_error": max(float(row["aggregate"]["max_abs_norm_error"]) for row in rows),
            "mean_abs_norm_error": sum(float(row["aggregate"]["mean_abs_norm_error"]) for row in rows) / len(rows),
            "min_finite_fraction": min(float(row["aggregate"]["min_finite_fraction"]) for row in rows),
            "passing_cases": sum(1 for row in rows if row["aggregate"]["passing"]),
            "min_raw_mixture_norm": min(
                float(row.get("mixture_cancellation", {}).get("min_raw_mixture_norm", 1.0)) for row in rows
            ),
        }
    return out


def _write_markdown(path: Path, payload: dict[str, Any]) -> None:
    overall = payload["overall"]
    lines = [
        "# Circleworld Unit-Phasor Contract Audit",
        "",
        "This audit treats RAFA's phase-only claim as an executable runtime contract: every tensor that is meant to carry phase as `[re, im]` should stay unit modulus and finite.",
        "",
        "## Summary",
        "",
        f"- status: `{overall['status']}`",
        f"- tolerance: `{overall['tolerance']}`",
        f"- max absolute norm error: `{overall['max_abs_norm_error']}`",
        f"- mean absolute norm error: `{overall['mean_abs_norm_error']}`",
        f"- minimum finite fraction: `{overall['min_finite_fraction']}`",
        f"- norm violations: `{overall['num_norm_violations']}`",
        f"- hidden magnitude channel detected: `{overall['hidden_magnitude_channel_detected']}`",
        f"- raw mixture cancellation status: `{payload['mixture_cancellation']['status']}`",
        f"- minimum raw mixture norm: `{payload['mixture_cancellation']['min_raw_mixture_norm']}`",
        f"- tensor count: `{overall['tensor_count']}`",
        f"- seed plan: `{payload['seed_plan']}`",
        f"- config: `{payload['config_path']}`",
        "",
        "## Source Summary",
        "",
        "| source | cases | max error | mean error | min finite | min raw mix | passing | children | events |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for source, row in payload["source_summaries"].items():
        lines.append(
            "| {source} | {cases} | {maxerr} | {meanerr} | {finite} | {mix} | {passing} | {children} | {events} |".format(
                source=source,
                cases=row["num_cases"],
                maxerr=row["max_abs_norm_error"],
                meanerr=row["mean_abs_norm_error"],
                finite=row["min_finite_fraction"],
                mix=row["min_raw_mixture_norm"],
                passing=row["passing_cases"],
                children=row["child_world_count"],
                events=row["child_event_count"],
            )
        )
    lines.extend(
        [
            "",
            "## Surface Summary",
            "",
            "| surface | cases | max error | mean error | min finite | min raw mix | passing |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for surface, row in payload["surface_summaries"].items():
        lines.append(
            "| {surface} | {cases} | {maxerr} | {meanerr} | {finite} | {mix} | {passing} |".format(
                surface=surface,
                cases=row["num_cases"],
                maxerr=row["max_abs_norm_error"],
                meanerr=row["mean_abs_norm_error"],
                finite=row["min_finite_fraction"],
                mix=row["min_raw_mixture_norm"],
                passing=row["passing_cases"],
            )
        )
    lines.extend(
        [
            "",
            "## Group Summary",
            "",
            "| group | tensors | max error | mean error | min finite | worst path |",
            "|---|---:|---:|---:|---:|---|",
        ]
    )
    for group, row in payload["group_summaries"].items():
        lines.append(
            "| {group} | {count} | {maxerr} | {meanerr} | {finite} | `{worst}` |".format(
                group=group,
                count=row["tensor_count"],
                maxerr=row["max_abs_norm_error"],
                meanerr=row["mean_abs_norm_error"],
                finite=row["min_finite_fraction"],
                worst=row["worst_path"],
            )
        )
    negative = payload.get("negative_control")
    if isinstance(negative, dict):
        neg_agg = negative.get("aggregate", {})
        lines.extend(
            [
                "",
                "## Negative Control",
                "",
                f"- enabled: `True`",
                f"- expected status: `{negative.get('expected_status')}`",
                f"- detected: `{negative.get('detected')}`",
                f"- source/seed: `{negative.get('source')}_{negative.get('seed')}`",
                f"- deliberate scale: `{negative.get('scale')}`",
                f"- max absolute norm error: `{neg_agg.get('max_abs_norm_error')}`",
                f"- norm violations: `{neg_agg.get('num_norm_violations')}`",
                f"- hidden magnitude channel detected: `{neg_agg.get('hidden_magnitude_channel_detected')}`",
            ]
        )
    lines.extend(
        [
            "",
            "## Raw Mixture Cancellation",
            "",
            "| path | min norm | mean norm | max norm | frac < 1e-4 | finite |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in payload.get("mixture_rows", []):
        lines.append(
            "| {path} | {minn} | {mean} | {maxn} | {low} | {finite} |".format(
                path=f"`{row['path']}`",
                minn=row["min_raw_mixture_norm"],
                mean=row["mean_raw_mixture_norm"],
                maxn=row["max_raw_mixture_norm"],
                low=row["frac_norm_lt_1e_4"],
                finite=row["finite_fraction"],
            )
        )
    lines.extend(
        [
            "",
            "## Case Summary",
            "",
            "| surface | source | seed | tensors | max error | min finite | children |",
            "|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for case in payload["cases"]:
        lines.append(
            "| {surface} | {source} | {seed} | {count} | {maxerr} | {finite} | {children} |".format(
                surface=case.get("surface", "heldout"),
                source=case["source"],
                seed=case["seed"],
                count=case["tensor_count"],
                maxerr=case["aggregate"]["max_abs_norm_error"],
                finite=case["aggregate"]["min_finite_fraction"],
                children=case.get("child_world_count", 0),
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_audit(
    *,
    config_path: Path,
    out_dir: Path,
    time_steps: int,
    device_name: str,
    seed_source: str,
    seeds: str,
    surface: str,
    nested_seed_source: str,
    nested_seeds: str,
    nested_depth: int,
    phase_blend: float,
    tolerance: float,
    include_negative_control: bool = False,
    negative_control_scale: float = 0.83,
    direct_export_wavs: str = "",
    direct_export_clip_seconds: int | None = 2,
    audio_continuation_wavs: str = "",
    audio_continuation_prefix_seconds: float = 1.0,
    audio_continuation_future_seconds: float = 1.0,
    audio_continuation_magnitude_mode: str = "prefix_hold",
) -> dict[str, Any]:
    cfg = _load_circle_cfg(config_path)
    device = _safe_device(device_name)
    plan = _seed_plan(seed_source, seeds)
    nested_plan = _seed_plan(nested_seed_source, nested_seeds)
    direct_wav_paths = _direct_export_wav_paths(direct_export_wavs)
    continuation_wav_paths = _direct_export_wav_paths(audio_continuation_wavs)
    active_surfaces = (
        ["heldout", "nested", "export", "direct_export", "audio_continuation"] if surface == "all" else [surface]
    )
    needs_rafa = any(src == "naked_rafa" for src, _ in (plan + nested_plan))
    rafa_core = make_seed_rafa(dev=device.type) if needs_rafa else None

    cases: list[dict[str, Any]] = []
    if "heldout" in active_surfaces:
        cases.extend(
            _run_case(
                cfg=cfg,
                rafa_core=rafa_core,
                source=source,
                seed=seed,
                time_steps=time_steps,
                device=device,
                tolerance=tolerance,
            )
            for source, seed in plan
        )
    if "nested" in active_surfaces:
        cases.extend(
            _run_nested_case(
                cfg=cfg,
                rafa_core=rafa_core,
                source=source,
                seed=seed,
                time_steps=time_steps,
                device=device,
                tolerance=tolerance,
                nested_depth=nested_depth,
            )
            for source, seed in nested_plan
        )
    if "export" in active_surfaces:
        cases.extend(
            _run_export_case(
                cfg=cfg,
                rafa_core=rafa_core,
                source=source,
                seed=seed,
                time_steps=time_steps,
                device=device,
                tolerance=tolerance,
                phase_blend=phase_blend,
            )
            for source, seed in plan
        )
    if "direct_export" in active_surfaces:
        if not direct_wav_paths:
            raise FileNotFoundError("No direct export WAV paths are available.")
        cases.extend(
            _run_direct_export_case(
                cfg=cfg,
                wav_path=wav_path,
                device_name=str(device_name),
                phase_blend=phase_blend,
                clip_seconds=direct_export_clip_seconds,
                tolerance=tolerance,
            )
            for wav_path in direct_wav_paths
        )
    if "audio_continuation" in active_surfaces:
        if not continuation_wav_paths:
            raise FileNotFoundError("No audio continuation WAV paths are available.")
        cases.extend(
            _run_audio_continuation_case(
                cfg=cfg,
                wav_path=wav_path,
                device_name=str(device_name),
                prefix_seconds=audio_continuation_prefix_seconds,
                future_seconds=audio_continuation_future_seconds,
                magnitude_mode=audio_continuation_magnitude_mode,
                tolerance=tolerance,
            )
            for wav_path in continuation_wav_paths
        )
    all_rows = [row for case in cases for row in case["rows"]]
    mixture_rows = [row for case in cases for row in case.get("mixture_rows", [])]
    negative_control = None
    if include_negative_control:
        neg_source, neg_seed = plan[0]
        negative_control = _run_negative_control(
            rafa_core=rafa_core,
            source=neg_source,
            seed=neg_seed,
            time_steps=time_steps,
            device=device,
            tolerance=tolerance,
            scale=negative_control_scale,
        )
    payload = {
        "config_path": str(config_path),
        "device_requested": str(device_name),
        "device": str(device),
        "time_steps": int(time_steps),
        "surface": str(surface),
        "active_surfaces": active_surfaces,
        "seed_plan": [{"source": source, "seed": int(seed)} for source, seed in plan],
        "nested_seed_plan": [{"source": source, "seed": int(seed)} for source, seed in nested_plan],
        "direct_export_wavs": [str(path) for path in direct_wav_paths],
        "direct_export_clip_seconds": direct_export_clip_seconds,
        "audio_continuation_wavs": [str(path) for path in continuation_wav_paths],
        "audio_continuation_prefix_seconds": float(audio_continuation_prefix_seconds),
        "audio_continuation_future_seconds": float(audio_continuation_future_seconds),
        "audio_continuation_magnitude_mode": str(audio_continuation_magnitude_mode),
        "nested_depth": int(nested_depth),
        "phase_blend": float(phase_blend),
        "tolerance": float(tolerance),
        "negative_control_enabled": bool(include_negative_control),
        "negative_control": negative_control,
        "overall": _aggregate_stats(all_rows, tolerance=tolerance),
        "mixture_cancellation": _aggregate_mixture_stats(mixture_rows),
        "source_summaries": _source_summaries(cases),
        "surface_summaries": _surface_summaries(cases),
        "group_summaries": _aggregate_by_group(all_rows),
        "mixture_rows": mixture_rows,
        "cases": cases,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "unit_phasor_contract_report.json"
    md_path = out_dir / "UNIT_PHASOR_CONTRACT.md"
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    _write_markdown(md_path, payload)
    return {"json": str(json_path), "markdown": str(md_path), "overall": payload["overall"]}


def main() -> None:
    ap = argparse.ArgumentParser(description="Audit Circleworld runtime phasor tensors for unit-modulus and finiteness.")
    ap.add_argument("--config", default=str(DEFAULT_CONFIG))
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--time-steps", type=int, default=96)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed-source", choices=["synthetic", "naked_rafa", "both"], default="both")
    ap.add_argument("--seeds", default="3", help="Positive count per source or comma-separated explicit seed list.")
    ap.add_argument(
        "--surface",
        choices=["heldout", "nested", "export", "direct_export", "audio_continuation", "all"],
        default="heldout",
    )
    ap.add_argument("--nested-seed-source", choices=["synthetic", "naked_rafa", "both"], default="naked_rafa")
    ap.add_argument("--nested-seeds", default="9100,9101,9102")
    ap.add_argument("--nested-depth", type=int, default=3)
    ap.add_argument("--phase-blend", type=float, default=1.0)
    ap.add_argument("--tolerance", type=float, default=1.0e-5)
    ap.add_argument("--include-negative-control", action="store_true")
    ap.add_argument("--negative-control-scale", type=float, default=0.83)
    ap.add_argument(
        "--direct-export-wavs",
        default="",
        help="Comma- or semicolon-separated WAV paths for literal export-path tensor audit. Empty uses default benchmark anchors.",
    )
    ap.add_argument("--direct-export-clip-seconds", type=int, default=2)
    ap.add_argument(
        "--audio-continuation-wavs",
        default="",
        help="Comma- or semicolon-separated WAV paths for audio-continuation tensor audit. Empty uses default benchmark anchors.",
    )
    ap.add_argument("--audio-continuation-prefix-seconds", type=float, default=1.0)
    ap.add_argument("--audio-continuation-future-seconds", type=float, default=1.0)
    ap.add_argument("--audio-continuation-magnitude-mode", choices=["prefix_hold", "flat"], default="prefix_hold")
    args = ap.parse_args()
    result = run_audit(
        config_path=Path(args.config),
        out_dir=Path(args.out_dir),
        time_steps=int(args.time_steps),
        device_name=str(args.device),
        seed_source=str(args.seed_source),
        seeds=str(args.seeds),
        surface=str(args.surface),
        nested_seed_source=str(args.nested_seed_source),
        nested_seeds=str(args.nested_seeds),
        nested_depth=int(args.nested_depth),
        phase_blend=float(args.phase_blend),
        tolerance=float(args.tolerance),
        include_negative_control=bool(args.include_negative_control),
        negative_control_scale=float(args.negative_control_scale),
        direct_export_wavs=str(args.direct_export_wavs),
        direct_export_clip_seconds=args.direct_export_clip_seconds,
        audio_continuation_wavs=str(args.audio_continuation_wavs),
        audio_continuation_prefix_seconds=float(args.audio_continuation_prefix_seconds),
        audio_continuation_future_seconds=float(args.audio_continuation_future_seconds),
        audio_continuation_magnitude_mode=str(args.audio_continuation_magnitude_mode),
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
