from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

import torch

ROOT = Path(__file__).resolve().parents[2]
CORE = ROOT / "core"
LINEAGE = ROOT / "lineages" / "04_positive_replacement"
RUNTIME = ROOT / "runtimes" / "circleworld_proto"
for path in (ROOT, CORE, LINEAGE, RUNTIME):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from common_io import as_float as _as_float  # noqa: E402
from common_io import mean as _mean  # noqa: E402
from common_io import parse_gain_csv as _parse_csv_floats  # noqa: E402
from benchmark_audio_continuation import (  # noqa: E402
    MAGNITUDE_MODES,
    _build_magnitude_canvas,
    _build_phase_seed_canvas,
    _compare_waveforms,
    _fit_phase_frames,
    _jsonify,
    _load_cases,
    _load_circle_cfg,
    _loop_reentry_metrics,
    _phase_to_angle,
    _prepare_anchor,
    _render_future_from_stft,
    _safe_device,
    _wrap_angle,
)
from circleworld import recurse_circleworld, summarize_circleworld_run  # noqa: E402
from config import load_config  # noqa: E402
from rafa_math_tools import phase_to_phasor  # noqa: E402
from stft_utils import compute_stft  # noqa: E402


PHASE_SEED_POLICY = "copy_last_waveform_phase"
DEFAULT_GAINS = (-1.0, 0.0, 0.25, 0.5, 1.0, 2.0)
CORE_MASK_MODES = ("all_bins", "high_energy_bins", "low_energy_bins")
PHASE_MASK_MODES = (
    "phase_stable_bins",
    "phase_dynamic_bins",
    "phase_low_motion_bins",
    "phase_coherent_motion_bins",
    "phase_curvature_bins",
    "phase_router_bins",
)
MASK_MODES = CORE_MASK_MODES + PHASE_MASK_MODES
OUTPUT_JSON = "audio_circle_delta_probe.json"
OUTPUT_MD = "AUDIO_CIRCLE_DELTA_PROBE.md"


def _parse_csv(raw: str, allowed: Sequence[str], label: str) -> list[str]:
    values = [item.strip() for item in raw.split(",") if item.strip()]
    if not values:
        raise ValueError(f"At least one {label} is required")
    unknown = sorted(set(values) - set(allowed))
    if unknown:
        raise ValueError(f"Unknown {label}: {unknown}; expected one of {tuple(allowed)}")
    return values
def _default_magnitude_modes() -> list[str]:
    requested = ("prefix_hold", "flat")
    return [mode for mode in requested if mode in MAGNITUDE_MODES]


def _quantile_1d(values: torch.Tensor, q: float) -> torch.Tensor:
    flat = values.detach().reshape(-1)
    if flat.numel() == 0:
        return torch.tensor(0.0, device=values.device, dtype=values.dtype)
    ordered = torch.sort(flat).values
    index = int(round(float(q) * float(max(0, ordered.numel() - 1))))
    return ordered[max(0, min(index, int(ordered.numel() - 1)))]


def _minmax01(values: torch.Tensor) -> torch.Tensor:
    low = values.min()
    high = values.max()
    return (values - low) / (high - low).clamp_min(1.0e-8)


def _future_mask_from_prefix_energy(
    prefix_mag: torch.Tensor,
    future_frames: int,
    mask_mode: str,
    prefix_phase: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    if mask_mode not in MASK_MODES:
        raise ValueError(f"Unknown mask mode: {mask_mode}; expected one of {MASK_MODES}")

    bins = int(prefix_mag.size(1))
    if mask_mode == "all_bins":
        mask = torch.ones((1, bins, future_frames), device=prefix_mag.device, dtype=prefix_mag.dtype)
        return mask, {
            "mask_mode": mask_mode,
            "mask_source": "all_stft_bins",
            "prefix_energy_threshold_source": "none",
            "mask_bin_fraction": 1.0,
            "future_target_magnitude_reused": False,
            "target_future_stft_magnitude_accessed": False,
        }

    prefix_energy = (prefix_mag.detach() * prefix_mag.detach()).mean(dim=-1, keepdim=True)
    if mask_mode == "high_energy_bins":
        threshold = _quantile_1d(prefix_energy, 2.0 / 3.0)
        bin_mask = prefix_energy >= threshold
        if int(bin_mask.sum().detach().cpu().item()) == 0:
            bin_mask = prefix_energy == prefix_energy.max()
        comparator = "prefix_bin_energy_gte_q66"
    elif mask_mode == "low_energy_bins":
        threshold = _quantile_1d(prefix_energy, 1.0 / 3.0)
        bin_mask = prefix_energy <= threshold
        if int(bin_mask.sum().detach().cpu().item()) == 0:
            bin_mask = prefix_energy == prefix_energy.min()
        comparator = "prefix_bin_energy_lte_q33"
    else:
        if prefix_phase is None:
            raise ValueError(f"Mask mode {mask_mode} requires prefix_phase")
        if prefix_phase.size(-1) > 1:
            velocity = _wrap_angle(prefix_phase[..., 1:] - prefix_phase[..., :-1])
            mean_abs_velocity = velocity.abs().mean(dim=-1, keepdim=True)
            stability = torch.cos(velocity).mean(dim=-1, keepdim=True)
            if velocity.size(-1) > 1:
                curvature = _wrap_angle(velocity[..., 1:] - velocity[..., :-1]).abs().mean(dim=-1, keepdim=True)
            else:
                curvature = torch.zeros_like(mean_abs_velocity)
        else:
            mean_abs_velocity = torch.zeros_like(prefix_mag[..., :1])
            stability = torch.ones_like(prefix_mag[..., :1])
            curvature = torch.zeros_like(prefix_mag[..., :1])
        if mask_mode == "phase_stable_bins":
            score = stability
            threshold = _quantile_1d(score, 2.0 / 3.0)
            bin_mask = score >= threshold
            comparator = "prefix_phase_stability_gte_q66"
        elif mask_mode == "phase_dynamic_bins":
            score = mean_abs_velocity
            threshold = _quantile_1d(score, 2.0 / 3.0)
            bin_mask = score >= threshold
            comparator = "prefix_phase_abs_velocity_gte_q66"
        elif mask_mode == "phase_low_motion_bins":
            score = mean_abs_velocity
            threshold = _quantile_1d(score, 1.0 / 3.0)
            bin_mask = score <= threshold
            comparator = "prefix_phase_abs_velocity_lte_q33"
        elif mask_mode == "phase_coherent_motion_bins":
            # Stable phase drift is a local law hint: coherent motion without relying on energy or future bins.
            score = _minmax01(stability) * (0.5 + 0.5 * _minmax01(mean_abs_velocity))
            threshold = _quantile_1d(score, 2.0 / 3.0)
            bin_mask = score >= threshold
            comparator = "prefix_phase_coherent_motion_gte_q66"
        elif mask_mode == "phase_curvature_bins":
            score = curvature
            threshold = _quantile_1d(score, 2.0 / 3.0)
            bin_mask = score >= threshold
            comparator = "prefix_phase_curvature_gte_q66"
        elif mask_mode == "phase_router_bins":
            stable = _minmax01(stability)
            motion = _minmax01(mean_abs_velocity)
            smooth = 1.0 - _minmax01(curvature)
            score = 0.55 * stable + 0.25 * motion + 0.20 * smooth
            threshold = _quantile_1d(score, 2.0 / 3.0)
            bin_mask = score >= threshold
            comparator = "prefix_phase_router_score_gte_q66"
        else:
            raise ValueError(f"Unknown phase mask mode: {mask_mode}")
        if int(bin_mask.sum().detach().cpu().item()) == 0:
            bin_mask = score == score.max() if "gte" in comparator else score == score.min()

    mask = bin_mask.to(dtype=prefix_mag.dtype).expand(-1, -1, future_frames).clone()
    return mask, {
        "mask_mode": mask_mode,
        "mask_source": "prefix_only_mean_stft_bin_energy"
        if mask_mode in CORE_MASK_MODES
        else "prefix_only_stft_phase_velocity",
        "prefix_energy_threshold_source": comparator,
        "prefix_energy_threshold": float(threshold.detach().cpu().item()),
        "mask_bin_fraction": float(bin_mask.float().mean().detach().cpu().item()),
        "uses_prefix_phase": bool(mask_mode in PHASE_MASK_MODES),
        "future_target_magnitude_reused": False,
        "target_future_stft_magnitude_accessed": False,
        "future_target_phase_reused": False,
        "target_future_stft_phase_accessed": False,
    }


def _phase_delta_stats(delta_future: torch.Tensor, mask: torch.Tensor) -> dict[str, float]:
    masked = delta_future * mask
    active = mask > 0.0
    active_delta = delta_future[active] if bool(active.any().detach().cpu().item()) else delta_future.reshape(-1)
    return {
        "future_mean_abs_phase_delta": float(delta_future.abs().mean().detach().cpu().item()),
        "future_rms_phase_delta": float(torch.sqrt(torch.mean(delta_future * delta_future)).detach().cpu().item()),
        "future_mean_cos_phase_delta": float(torch.cos(delta_future).mean().detach().cpu().item()),
        "masked_mean_abs_phase_delta": float(masked.abs().mean().detach().cpu().item()),
        "active_bin_mean_abs_phase_delta": float(active_delta.abs().mean().detach().cpu().item()),
    }


def _apply_future_delta(
    seed_phase: torch.Tensor,
    delta_future: torch.Tensor,
    mask: torch.Tensor,
    gain: float,
    prefix_frames: int,
) -> torch.Tensor:
    future = _wrap_angle(seed_phase[..., prefix_frames:] + float(gain) * delta_future * mask)
    return torch.cat([seed_phase[..., :prefix_frames], future], dim=-1)


def _aggregate(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, float], list[dict[str, Any]]] = defaultdict(list)
    seed_rows_by_case: dict[tuple[str, str, str], dict[str, Any]] = {}
    for row in rows:
        mode = str(row["magnitude_mode"])
        mask = str(row["mask_mode"])
        gain = float(row["gain"])
        grouped[(mode, mask, gain)].append(row)
        if abs(gain) < 1.0e-12:
            seed_rows_by_case[(str(row["name"]), mode, mask)] = row

    out: list[dict[str, Any]] = []
    for (mode, mask, gain), group in sorted(grouped.items(), key=lambda item: (item[0][0], item[0][1], item[0][2])):
        corr_deltas: list[float] = []
        mse_deltas: list[float] = []
        corr_wins = 0
        mse_wins = 0
        for row in group:
            seed = seed_rows_by_case.get((str(row["name"]), mode, mask))
            if seed is None:
                continue
            corr_delta = _as_float(row["target_corr"]) - _as_float(seed["target_corr"])
            mse_delta = _as_float(row["target_mse"]) - _as_float(seed["target_mse"])
            corr_deltas.append(corr_delta)
            mse_deltas.append(mse_delta)
            corr_wins += int(corr_delta > 1.0e-6)
            mse_wins += int(mse_delta < -1.0e-9)

        mean_corr_delta = _mean(corr_deltas)
        mean_mse_delta = _mean(mse_deltas)
        corr_win_fraction = float(corr_wins / max(1, len(corr_deltas)))
        mse_win_fraction = float(mse_wins / max(1, len(mse_deltas)))
        mean_vs_seed_corr = _mean([_as_float(row["vs_gain0_corr"]) for row in group])
        if abs(gain) < 1.0e-12:
            verdict = "seed_baseline"
        elif mean_corr_delta > 0.02 and corr_win_fraction >= 0.5 and mean_mse_delta <= 1.0e-4:
            verdict = "delta_helpful"
        elif mean_mse_delta < -1.0e-4 and mse_win_fraction >= 0.5 and mean_corr_delta >= -0.005:
            verdict = "delta_mse_helpful_candidate"
        elif mean_mse_delta < -1.0e-4 and mse_win_fraction >= 0.5:
            verdict = "delta_mse_corr_tradeoff"
        elif (
            mean_corr_delta < -0.005
            or mean_mse_delta > 1.0e-4
            or (mean_vs_seed_corr >= 0.95 and abs(mean_corr_delta) <= 0.005 and mean_mse_delta >= -1.0e-5)
        ):
            verdict = "delta_decorative_or_harmful"
        else:
            verdict = "needs_review"

        out.append(
            {
                "magnitude_mode": mode,
                "mask_mode": mask,
                "gain": gain,
                "case_count": len(group),
                "mean_target_corr": _mean([_as_float(row["target_corr"]) for row in group]),
                "mean_target_mae": _mean([_as_float(row["target_mae"]) for row in group]),
                "mean_target_mse": _mean([_as_float(row["target_mse"]) for row in group]),
                "mean_vs_gain0_corr": mean_vs_seed_corr,
                "mean_vs_gain0_mse": _mean([_as_float(row["vs_gain0_mse"]) for row in group]),
                "mean_target_corr_delta_vs_gain0": mean_corr_delta,
                "mean_target_mse_delta_vs_gain0": mean_mse_delta,
                "target_corr_win_fraction_vs_gain0": corr_win_fraction,
                "target_mse_win_fraction_vs_gain0": mse_win_fraction,
                "mean_loop_autocorr_peak": _mean([_as_float(row["loop_autocorr_peak"]) for row in group]),
                "mean_first_chunk_reentry": _mean([_as_float(row["first_chunk_reentry"]) for row in group]),
                "mean_mask_bin_fraction": _mean([_as_float(row["mask_bin_fraction"]) for row in group]),
                "mean_active_bin_abs_phase_delta": _mean(
                    [_as_float(row["phase_delta_stats"]["active_bin_mean_abs_phase_delta"]) for row in group]
                ),
                "verdict": verdict,
            }
        )
    return out


def _overall_status(aggregate: list[dict[str, Any]]) -> str:
    nonzero = [row for row in aggregate if abs(float(row["gain"])) > 1.0e-12]
    if any(row.get("verdict") == "delta_helpful" for row in nonzero):
        return "delta_helpful"
    if any(row.get("verdict") in {"delta_mse_helpful_candidate", "delta_mse_corr_tradeoff"} for row in nonzero):
        return "needs_review"
    if nonzero and all(row.get("verdict") == "delta_decorative_or_harmful" for row in nonzero):
        return "delta_decorative_or_harmful"
    return "needs_review"


def _merge_access_flags(
    *,
    phase_flags: dict[str, Any],
    magnitude_flags: Sequence[dict[str, Any]],
) -> dict[str, bool]:
    return {
        "future_target_magnitude_reused": bool(
            any(bool(flags.get("future_target_magnitude_reused")) for flags in magnitude_flags)
        ),
        "target_future_stft_magnitude_accessed": bool(
            any(bool(flags.get("target_future_stft_magnitude_accessed")) for flags in magnitude_flags)
        ),
        "future_target_phase_reused": bool(phase_flags.get("future_target_phase_reused")),
        "target_future_stft_phase_accessed": bool(phase_flags.get("target_future_stft_phase_accessed")),
    }


def _future_access_flags(rows: Sequence[dict[str, Any]]) -> dict[str, bool]:
    return {
        "future_target_magnitude_reused": bool(
            any(bool(row.get("future_target_magnitude_reused")) for row in rows)
        ),
        "target_future_stft_magnitude_accessed": bool(
            any(bool(row.get("target_future_stft_magnitude_accessed")) for row in rows)
        ),
        "future_target_phase_reused": bool(any(bool(row.get("future_target_phase_reused")) for row in rows)),
        "target_future_stft_phase_accessed": bool(
            any(bool(row.get("target_future_stft_phase_accessed")) for row in rows)
        ),
    }


def _run_case(
    *,
    name: str,
    wav_path: Path,
    circle_cfg: Any,
    stft_cfg: dict[str, Any],
    sr: int,
    device: torch.device,
    prefix_seconds: float,
    future_seconds: float,
    magnitude_modes: Sequence[str],
    mask_modes: Sequence[str],
    gains: Sequence[float],
) -> dict[str, Any]:
    prefix_samples = max(1, int(round(prefix_seconds * sr)))
    future_samples = max(1, int(round(future_seconds * sr)))
    total_samples = prefix_samples + future_samples

    wav = _prepare_anchor(wav_path, target_sr=sr, total_samples=total_samples)
    prefix = wav[..., :prefix_samples].to(device)
    target_future = wav[..., prefix_samples:total_samples].reshape(-1).to(device)

    prefix_mag, prefix_phase = compute_stft(prefix, stft_cfg)
    prefix_frames = int(prefix_phase.size(-1))
    hop = int(stft_cfg["hop"])
    future_frames = max(1, int(math.ceil(float(future_samples) / float(hop))))

    seed_phase, seed_flags = _build_phase_seed_canvas(
        prefix=prefix,
        prefix_phase=prefix_phase,
        future_frames=future_frames,
        future_samples=future_samples,
        sr=sr,
        stft_cfg=stft_cfg,
        policy=PHASE_SEED_POLICY,
    )
    seed_phase = _fit_phase_frames(seed_phase, prefix_frames + future_frames)
    seed_z = phase_to_phasor(seed_phase)

    run_mode = (
        circle_cfg.branching_mode
        if circle_cfg.branching_mode in {"native_multimode", "native_multimode_childworld"}
        else "active_packets"
    )
    run = recurse_circleworld(seed_z, cfg=circle_cfg, depth=circle_cfg.recursion_depth, mode=run_mode)
    final_phase_raw = _fit_phase_frames(_phase_to_angle(run["phase_state"]), prefix_frames + future_frames)
    final_phase = torch.cat([prefix_phase, final_phase_raw[..., prefix_frames:]], dim=-1)
    delta_future = _wrap_angle(final_phase[..., prefix_frames:] - seed_phase[..., prefix_frames:])

    rows: list[dict[str, Any]] = []
    seed_future_by_mode_mask: dict[tuple[str, str], torch.Tensor] = {}
    for mode in magnitude_modes:
        mag_canvas, mag_flags = _build_magnitude_canvas(prefix_mag, future_frames=future_frames, mode=mode)
        for mask_mode in mask_modes:
            mask, mask_flags = _future_mask_from_prefix_energy(
                prefix_mag=prefix_mag,
                prefix_phase=prefix_phase,
                future_frames=future_frames,
                mask_mode=mask_mode,
            )
            stats = _phase_delta_stats(delta_future, mask)
            seed_phase_with_mask = _apply_future_delta(
                seed_phase=seed_phase,
                delta_future=delta_future,
                mask=mask,
                gain=0.0,
                prefix_frames=prefix_frames,
            )
            seed_future = _render_future_from_stft(
                mag_canvas=mag_canvas,
                phase_canvas=seed_phase_with_mask,
                stft_cfg=stft_cfg,
                prefix_samples=prefix_samples,
                future_samples=future_samples,
            )
            seed_future_by_mode_mask[(mode, mask_mode)] = seed_future
            for gain in gains:
                phase = _apply_future_delta(
                    seed_phase=seed_phase,
                    delta_future=delta_future,
                    mask=mask,
                    gain=float(gain),
                    prefix_frames=prefix_frames,
                )
                pred = _render_future_from_stft(
                    mag_canvas=mag_canvas,
                    phase_canvas=phase,
                    stft_cfg=stft_cfg,
                    prefix_samples=prefix_samples,
                    future_samples=future_samples,
                )
                target_metrics = _compare_waveforms(target_future, pred)
                seed_metrics = _compare_waveforms(seed_future_by_mode_mask[(mode, mask_mode)], pred)
                loop = _loop_reentry_metrics(pred, sr)
                access_flags = _merge_access_flags(
                    phase_flags=seed_flags,
                    magnitude_flags=(mag_flags, mask_flags),
                )
                rows.append(
                    {
                        "name": name,
                        "source_wav": str(wav_path),
                        "magnitude_mode": mode,
                        "mask_mode": mask_mode,
                        "gain": float(gain),
                        "target_corr": target_metrics["corr"],
                        "target_mae": target_metrics["mae"],
                        "target_mse": target_metrics["mse"],
                        "target_rms": target_metrics["target_rms"],
                        "output_rms": target_metrics["output_rms"],
                        "vs_gain0_corr": seed_metrics["corr"],
                        "vs_gain0_mae": seed_metrics["mae"],
                        "vs_gain0_mse": seed_metrics["mse"],
                        "loop_autocorr_peak": loop["loop_autocorr_peak"],
                        "first_chunk_reentry": loop["first_chunk_reentry"],
                        "mask_bin_fraction": mask_flags["mask_bin_fraction"],
                        "phase_delta_stats": stats,
                        "phase_seed_flags": seed_flags,
                        "magnitude_flags": mag_flags,
                        "mask_flags": mask_flags,
                        "future_target_audio_used_for_metrics_only": True,
                        **access_flags,
                    }
                )

    access_flags = _future_access_flags(rows)
    return {
        "name": name,
        "source_wav": str(wav_path),
        "sample_rate": sr,
        "prefix_samples": prefix_samples,
        "future_samples": future_samples,
        "prefix_stft_frames": prefix_frames,
        "future_stft_frames": int(future_frames),
        "phase_seed_policy": PHASE_SEED_POLICY,
        "phase_seed_flags": seed_flags,
        "future_target_audio_used_for_metrics_only": True,
        **access_flags,
        "circleworld_meta": _jsonify(summarize_circleworld_run(run)),
        "rows": rows,
    }


def run_probe(
    *,
    config_path: Path,
    out_dir: Path,
    device_name: str,
    num_cases: int,
    prefix_seconds: float,
    future_seconds: float,
    magnitude_modes: Sequence[str],
    mask_modes: Sequence[str],
    gains: Sequence[float],
    cases_json: Path | None = None,
) -> dict[str, Any]:
    if not magnitude_modes:
        raise ValueError("At least one magnitude mode is required")
    if not mask_modes:
        raise ValueError("At least one mask mode is required")
    if prefix_seconds <= 0.0 or future_seconds <= 0.0:
        raise ValueError("prefix_seconds and future_seconds must be positive")

    cfg = load_config(str(ROOT / "config.yaml")) if (ROOT / "config.yaml").exists() else load_config()
    sr = int(cfg["data"]["sample_rate"])
    stft_cfg = dict(cfg["data"]["stft"])
    circle_cfg = _load_circle_cfg(config_path)
    device = _safe_device(device_name)
    out_dir.mkdir(parents=True, exist_ok=True)

    cases = list(_load_cases(cases_json).items())
    if num_cases > 0:
        cases = cases[:num_cases]
    if not cases:
        raise RuntimeError("No cases selected for audio Circleworld delta probe")

    case_rows = [
        _run_case(
            name=name,
            wav_path=wav_path,
            circle_cfg=circle_cfg,
            stft_cfg=stft_cfg,
            sr=sr,
            device=device,
            prefix_seconds=prefix_seconds,
            future_seconds=future_seconds,
            magnitude_modes=magnitude_modes,
            mask_modes=mask_modes,
            gains=gains,
        )
        for name, wav_path in cases
    ]
    rows = [row for case in case_rows for row in case["rows"]]
    aggregate = _aggregate(rows)
    access_flags = _future_access_flags(rows)
    status = _overall_status(aggregate)
    if any(access_flags.values()):
        status = "invalid_future_leakage"
    summary = _jsonify(
        {
            "runtime": "circleworld_proto",
            "schema": "circleworld_audio_delta_probe_v0",
            "status": status,
            "config_path": str(config_path),
            "out_dir": str(out_dir),
            "device": str(device),
            "requested_device": str(device_name),
            "sample_rate": sr,
            "prefix_seconds": float(prefix_seconds),
            "future_seconds": float(future_seconds),
            "magnitude_modes": list(magnitude_modes),
            "mask_modes": list(mask_modes),
            "gains": [float(gain) for gain in gains],
            "phase_seed_policy": PHASE_SEED_POLICY,
            "case_count": len(case_rows),
            "future_target_audio_used_for_metrics_only": True,
            **access_flags,
            "aggregate": aggregate,
            "cases": case_rows,
        }
    )
    (out_dir / OUTPUT_JSON).write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (out_dir / OUTPUT_MD).write_text(_markdown_summary(summary), encoding="utf-8")
    return summary


def _markdown_summary(summary: dict[str, Any]) -> str:
    lines = [
        "# Audio Circle Delta Probe",
        "",
        "This tests whether Circleworld future-frame phase deltas improve a strong legal copyphase seed.",
        "The seed uses `copy_last_waveform_phase`; target future audio is used only for metrics.",
        "",
        f"- status: `{summary['status']}`",
        f"- config: `{summary['config_path']}`",
        f"- cases: {summary['case_count']}",
        f"- phase seed policy: `{summary['phase_seed_policy']}`",
        f"- magnitude modes: `{', '.join(summary['magnitude_modes'])}`",
        f"- masks: `{', '.join(summary['mask_modes'])}`",
        f"- gains: `{', '.join(str(gain) for gain in summary['gains'])}`",
        f"- future target magnitude reused: `{summary['future_target_magnitude_reused']}`",
        f"- target future STFT magnitude accessed: `{summary['target_future_stft_magnitude_accessed']}`",
        f"- future target phase reused: `{summary['future_target_phase_reused']}`",
        f"- target future STFT phase accessed: `{summary['target_future_stft_phase_accessed']}`",
        "",
        "## Aggregate Rows",
        "",
        "| mode | mask | gain | cases | corr | MAE | MSE | vs gain0 corr | corr delta | MSE delta | corr wins | MSE wins | mask bins | active abs delta | verdict |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in summary["aggregate"]:
        lines.append(
            "| {mode} | {mask} | {gain:.6g} | {cases} | {corr:.6g} | {mae:.6g} | {mse:.6g} | {vseed:.6g} | {dcorr:.6g} | {dmse:.6g} | {wcorr:.6g} | {wmse:.6g} | {bins:.6g} | {adelta:.6g} | {verdict} |".format(
                mode=row["magnitude_mode"],
                mask=row["mask_mode"],
                gain=float(row["gain"]),
                cases=int(row["case_count"]),
                corr=float(row["mean_target_corr"]),
                mae=float(row["mean_target_mae"]),
                mse=float(row["mean_target_mse"]),
                vseed=float(row["mean_vs_gain0_corr"]),
                dcorr=float(row["mean_target_corr_delta_vs_gain0"]),
                dmse=float(row["mean_target_mse_delta_vs_gain0"]),
                wcorr=float(row["target_corr_win_fraction_vs_gain0"]),
                wmse=float(row["target_mse_win_fraction_vs_gain0"]),
                bins=float(row["mean_mask_bin_fraction"]),
                adelta=float(row["mean_active_bin_abs_phase_delta"]),
                verdict=row["verdict"],
            )
        )

    lines.extend(
        [
            "",
            "## Case Rows",
            "",
            "| case | mode | mask | gain | corr | MAE | MSE | vs gain0 corr | vs gain0 MSE | loop peak | reentry |",
            "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for case in summary["cases"]:
        for row in case["rows"]:
            lines.append(
                "| {case} | {mode} | {mask} | {gain:.6g} | {corr:.6g} | {mae:.6g} | {mse:.6g} | {vseed:.6g} | {vmse:.6g} | {loop:.6g} | {reentry:.6g} |".format(
                    case=row["name"],
                    mode=row["magnitude_mode"],
                    mask=row["mask_mode"],
                    gain=float(row["gain"]),
                    corr=float(row["target_corr"]),
                    mae=float(row["target_mae"]),
                    mse=float(row["target_mse"]),
                    vseed=float(row["vs_gain0_corr"]),
                    vmse=float(row["vs_gain0_mse"]),
                    loop=float(row["loop_autocorr_peak"]),
                    reentry=float(row["first_chunk_reentry"]),
                )
            )

    lines.extend(
        [
            "",
            "## Interpretation Guard",
            "",
            "- Gain `0.0` is the legal copyphase seed rendered with the same prefix-derived magnitude and mask context.",
            "- A positive result requires nonzero masked Circleworld deltas to beat gain `0.0` on target metrics.",
            "- Masks are built from prefix-only STFT bin energy; no future target STFT magnitude or phase is accessed.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    default_modes = _default_magnitude_modes()
    ap = argparse.ArgumentParser(
        description="Probe additive Circleworld future phase deltas against a copy-last-waveform phase seed."
    )
    ap.add_argument("--config", required=True, help="Circleworld JSON config path.")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--num-cases", type=int, default=3)
    ap.add_argument("--prefix-seconds", type=float, default=2.0)
    ap.add_argument("--future-seconds", type=float, default=2.0)
    ap.add_argument("--magnitude-modes", default=",".join(default_modes))
    ap.add_argument("--masks", default=",".join(CORE_MASK_MODES))
    ap.add_argument("--gains", default=",".join(str(value) for value in DEFAULT_GAINS))
    ap.add_argument(
        "--cases-json",
        default=None,
        help="Optional JSON mapping case names to real anchor WAV paths. Defaults to the expanded Circleworld anchors if present.",
    )
    args = ap.parse_args()
    summary = run_probe(
        config_path=Path(args.config),
        out_dir=Path(args.out_dir),
        device_name=str(args.device),
        num_cases=int(args.num_cases),
        prefix_seconds=float(args.prefix_seconds),
        future_seconds=float(args.future_seconds),
        magnitude_modes=_parse_csv(str(args.magnitude_modes), MAGNITUDE_MODES, "magnitude mode"),
        mask_modes=_parse_csv(str(args.masks), MASK_MODES, "mask mode"),
        gains=_parse_csv_floats(str(args.gains)),
        cases_json=Path(args.cases_json) if args.cases_json else None,
    )
    print(
        json.dumps(
            {
                "json": str(Path(args.out_dir) / OUTPUT_JSON),
                "markdown": str(Path(args.out_dir) / OUTPUT_MD),
                "status": summary["status"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
