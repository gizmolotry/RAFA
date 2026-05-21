from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
CORE = ROOT / "core"
LINEAGE = ROOT / "lineages" / "04_positive_replacement"
RUNTIME = ROOT / "runtimes" / "circleworld_proto"
for path in (ROOT, CORE, LINEAGE, RUNTIME):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from common_io import as_float as _as_float  # noqa: E402
from common_io import mean as _mean  # noqa: E402
from common_io import median as _median  # noqa: E402
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
from run_audio_circle_delta_probe import (  # noqa: E402
    CORE_MASK_MODES,
    MASK_MODES,
    _apply_future_delta,
    _future_mask_from_prefix_energy,
    _merge_access_flags,
)
from phase_native_audio_operators import (  # noqa: E402
    CORE_MECHANISMS as OPERATOR_CORE_MECHANISMS,
    EXPERIMENTAL_MECHANISMS as OPERATOR_EXPERIMENTAL_MECHANISMS,
    MECHANISMS as OPERATOR_MECHANISMS,
    mechanism_delta as _operator_mechanism_delta,
    phase_delta_stats as _operator_phase_delta_stats,
)
from stft_utils import compute_stft  # noqa: E402


PHASE_SEED_POLICY = "copy_last_waveform_phase"
CORE_MECHANISMS = (
    "raw",
    "time_smooth_3",
    "time_smooth_7",
    "freq_smooth_5",
    "energy_weighted",
    "phase_velocity_coherent",
    "phase_velocity_incoherent",
    "energy_velocity_coherent",
    "energy_velocity_time_smooth_3",
    "future_decay_taper",
)
EXPERIMENTAL_MECHANISMS = (
    "stable_time_smooth_3",
    "stable_energy_time_smooth_3",
    "stable_velocity_time_smooth_3",
    "stable_energy_velocity_time_smooth_3",
    "stable_softclip_time_smooth_3",
    "stable_energy_velocity_softclip_time_smooth_3",
    "anti_reentry_phase_shear",
    "anti_reentry_curvature_shear",
    "anti_reentry_delta_shear_mix",
    "anti_reentry_late_decorrelator",
    "anti_reentry_staggered_decorrelator",
    "anti_reentry_delta_decorrelator_mix",
)
MECHANISMS = CORE_MECHANISMS + EXPERIMENTAL_MECHANISMS
DEFAULT_GAINS = (-1.0, 0.0, 0.5, 1.0, 2.0)
OUTPUT_JSON = "audio_delta_mechanism_probe.json"
OUTPUT_MD = "AUDIO_DELTA_MECHANISM_PROBE.md"
MECHANISM_CANDIDATE_MIN_CORR_DELTA = 0.02
MECHANISM_CANDIDATE_MIN_CORR_WIN_FRACTION = 0.67


def _parse_csv(raw: str, allowed: Sequence[str], label: str) -> list[str]:
    values = [item.strip() for item in raw.split(",") if item.strip()]
    if not values:
        raise ValueError(f"At least one {label} is required")
    unknown = sorted(set(values) - set(allowed))
    if unknown:
        raise ValueError(f"Unknown {label}: {unknown}; expected one of {tuple(allowed)}")
    return values
def _circular_smooth_time(delta: torch.Tensor, kernel_size: int) -> torch.Tensor:
    kernel_size = max(1, int(kernel_size))
    if kernel_size <= 1 or delta.size(-1) <= 1:
        return delta
    pad_left = kernel_size // 2
    pad_right = kernel_size - 1 - pad_left
    b, f, t = delta.shape
    cos = torch.cos(delta).reshape(b * f, 1, t)
    sin = torch.sin(delta).reshape(b * f, 1, t)
    cos_s = F.avg_pool1d(F.pad(cos, (pad_left, pad_right), mode="replicate"), kernel_size=kernel_size, stride=1)
    sin_s = F.avg_pool1d(F.pad(sin, (pad_left, pad_right), mode="replicate"), kernel_size=kernel_size, stride=1)
    return torch.atan2(sin_s.reshape(b, f, t), cos_s.reshape(b, f, t))


def _circular_smooth_freq(delta: torch.Tensor, kernel_size: int) -> torch.Tensor:
    kernel_size = max(1, int(kernel_size))
    if kernel_size <= 1 or delta.size(1) <= 1:
        return delta
    pad_left = kernel_size // 2
    pad_right = kernel_size - 1 - pad_left
    b, f, t = delta.shape
    cos = torch.cos(delta).permute(0, 2, 1).reshape(b * t, 1, f)
    sin = torch.sin(delta).permute(0, 2, 1).reshape(b * t, 1, f)
    cos_s = F.avg_pool1d(F.pad(cos, (pad_left, pad_right), mode="replicate"), kernel_size=kernel_size, stride=1)
    sin_s = F.avg_pool1d(F.pad(sin, (pad_left, pad_right), mode="replicate"), kernel_size=kernel_size, stride=1)
    return torch.atan2(sin_s.reshape(b, t, f), cos_s.reshape(b, t, f)).permute(0, 2, 1)


def _prefix_energy_weight(prefix_mag: torch.Tensor) -> torch.Tensor:
    energy = (prefix_mag.detach() * prefix_mag.detach()).mean(dim=-1, keepdim=True)
    denom = energy.amax(dim=1, keepdim=True).clamp_min(1.0e-8)
    return (energy / denom).clamp(0.0, 1.0)


def _phase_velocity_coherence(prefix_phase: torch.Tensor, history_frames: int = 8) -> torch.Tensor:
    if prefix_phase.size(-1) < 2:
        return torch.ones(prefix_phase.shape[:2] + (1,), device=prefix_phase.device, dtype=prefix_phase.dtype)
    diff = _wrap_angle(prefix_phase[..., 1:] - prefix_phase[..., :-1])
    diff = diff[..., -max(1, int(history_frames)) :]
    mean_cos = torch.cos(diff).mean(dim=-1, keepdim=True)
    mean_sin = torch.sin(diff).mean(dim=-1, keepdim=True)
    return torch.sqrt(mean_cos * mean_cos + mean_sin * mean_sin).clamp(0.0, 1.0)


def _prefix_circular_velocity(prefix_phase: torch.Tensor, history_frames: int = 8) -> torch.Tensor:
    if prefix_phase.size(-1) < 2:
        return torch.zeros(prefix_phase.shape[:2] + (1,), device=prefix_phase.device, dtype=prefix_phase.dtype)
    diff = _wrap_angle(prefix_phase[..., 1:] - prefix_phase[..., :-1])
    diff = diff[..., -max(1, int(history_frames)) :]
    mean_cos = torch.cos(diff).mean(dim=-1, keepdim=True)
    mean_sin = torch.sin(diff).mean(dim=-1, keepdim=True)
    return torch.atan2(mean_sin, mean_cos)


def _prefix_circular_acceleration(prefix_phase: torch.Tensor, history_frames: int = 8) -> torch.Tensor:
    if prefix_phase.size(-1) < 3:
        return torch.zeros(prefix_phase.shape[:2] + (1,), device=prefix_phase.device, dtype=prefix_phase.dtype)
    diff = _wrap_angle(prefix_phase[..., 1:] - prefix_phase[..., :-1])
    accel = _wrap_angle(diff[..., 1:] - diff[..., :-1])
    accel = accel[..., -max(1, int(history_frames)) :]
    mean_cos = torch.cos(accel).mean(dim=-1, keepdim=True)
    mean_sin = torch.sin(accel).mean(dim=-1, keepdim=True)
    return torch.atan2(mean_sin, mean_cos)


def _center_frequency_law(x: torch.Tensor) -> torch.Tensor:
    return x - x.mean(dim=1, keepdim=True)


def _anti_reentry_phase_shear(prefix_phase: torch.Tensor, future_frames: int, *, power: float = 1.35) -> torch.Tensor:
    velocity = _center_frequency_law(_prefix_circular_velocity(prefix_phase, history_frames=8))
    ramp = torch.linspace(0.0, 1.0, max(1, int(future_frames)), device=prefix_phase.device, dtype=prefix_phase.dtype)
    ramp = ramp.clamp_min(0.0).pow(float(power)).view(1, 1, -1)
    return _soft_clip_phase_delta(velocity * ramp, limit=0.55)


def _anti_reentry_curvature_shear(prefix_phase: torch.Tensor, future_frames: int) -> torch.Tensor:
    accel = _center_frequency_law(_prefix_circular_acceleration(prefix_phase, history_frames=8))
    ramp = torch.linspace(0.0, 1.0, max(1, int(future_frames)), device=prefix_phase.device, dtype=prefix_phase.dtype)
    ramp = (ramp * ramp).view(1, 1, -1)
    return _soft_clip_phase_delta(accel * ramp, limit=0.45)


def _frequency_hash(prefix_phase: torch.Tensor) -> torch.Tensor:
    freq = torch.arange(prefix_phase.size(1), device=prefix_phase.device, dtype=prefix_phase.dtype).view(1, -1, 1)
    return torch.sin(freq * 1.61803398875 + 0.37)


def _anti_reentry_late_decorrelator(prefix_phase: torch.Tensor, future_frames: int) -> torch.Tensor:
    # A deterministic, prefix-independent phase-only decorrelator that is quiet at the boundary
    # and grows later, so it targets replay across chunks rather than the initial carrier.
    frames = max(1, int(future_frames))
    t = torch.linspace(0.0, 1.0, frames, device=prefix_phase.device, dtype=prefix_phase.dtype).view(1, 1, -1)
    late = torch.sigmoid((t - 0.38) * 10.0)
    freq_hash = _frequency_hash(prefix_phase)
    return _soft_clip_phase_delta(freq_hash * late, limit=0.85)


def _anti_reentry_staggered_decorrelator(prefix_phase: torch.Tensor, future_frames: int) -> torch.Tensor:
    frames = max(1, int(future_frames))
    t = torch.linspace(0.0, 1.0, frames, device=prefix_phase.device, dtype=prefix_phase.dtype).view(1, 1, -1)
    freq = torch.arange(prefix_phase.size(1), device=prefix_phase.device, dtype=prefix_phase.dtype).view(1, -1, 1)
    stagger = torch.sin((freq + 1.0) * (2.0 * math.pi * t) + freq * 0.73)
    late = torch.sigmoid((t - 0.25) * 8.0)
    return _soft_clip_phase_delta(stagger * late, limit=0.75)


def _prefix_magnitude_stability(prefix_mag: torch.Tensor, history_frames: int = 8) -> torch.Tensor:
    if prefix_mag.size(-1) < 2:
        return torch.ones(prefix_mag.shape[:2] + (1,), device=prefix_mag.device, dtype=prefix_mag.dtype)
    mag = torch.log1p(prefix_mag.detach().clamp_min(0.0))
    flux = (mag[..., 1:] - mag[..., :-1]).abs()
    flux = flux[..., -max(1, int(history_frames)) :]
    flux = flux.mean(dim=-1, keepdim=True)
    denom = flux.amax(dim=1, keepdim=True).clamp_min(1.0e-8)
    return (1.0 - flux / denom).clamp(0.0, 1.0)


def _soft_clip_phase_delta(delta: torch.Tensor, limit: float = 0.35) -> torch.Tensor:
    limit = max(1.0e-6, float(limit))
    return limit * torch.tanh(delta / limit)


def _future_taper(delta: torch.Tensor, start: float = 1.0, end: float = 0.25) -> torch.Tensor:
    if delta.size(-1) <= 1:
        return delta * float(start)
    ramp = torch.linspace(float(start), float(end), delta.size(-1), device=delta.device, dtype=delta.dtype)
    return delta * ramp.view(1, 1, -1)


def _mechanism_delta(
    *,
    mechanism: str,
    raw_delta: torch.Tensor,
    prefix_mag: torch.Tensor,
    prefix_phase: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, Any]]:
    energy = _prefix_energy_weight(prefix_mag)
    coherence = _phase_velocity_coherence(prefix_phase, history_frames=8)
    stability = _prefix_magnitude_stability(prefix_mag, history_frames=8)
    if mechanism == "raw":
        shaped = raw_delta
        source = "circleworld_raw_delta"
    elif mechanism == "time_smooth_3":
        shaped = _circular_smooth_time(raw_delta, kernel_size=3)
        source = "circleworld_delta_circular_time_smooth_3"
    elif mechanism == "time_smooth_7":
        shaped = _circular_smooth_time(raw_delta, kernel_size=7)
        source = "circleworld_delta_circular_time_smooth_7"
    elif mechanism == "freq_smooth_5":
        shaped = _circular_smooth_freq(raw_delta, kernel_size=5)
        source = "circleworld_delta_circular_frequency_smooth_5"
    elif mechanism == "energy_weighted":
        shaped = raw_delta * energy
        source = "circleworld_delta_prefix_energy_weighted"
    elif mechanism == "phase_velocity_coherent":
        shaped = raw_delta * coherence
        source = "circleworld_delta_prefix_phase_velocity_coherence_weighted"
    elif mechanism == "phase_velocity_incoherent":
        shaped = raw_delta * (1.0 - coherence)
        source = "circleworld_delta_prefix_phase_velocity_incoherence_weighted"
    elif mechanism == "energy_velocity_coherent":
        shaped = raw_delta * energy * coherence
        source = "circleworld_delta_prefix_energy_and_phase_velocity_coherence_weighted"
    elif mechanism == "energy_velocity_time_smooth_3":
        shaped = _circular_smooth_time(raw_delta, kernel_size=3) * energy * coherence
        source = "circleworld_delta_time_smooth_then_prefix_energy_velocity_coherence_weighted"
    elif mechanism == "future_decay_taper":
        shaped = _future_taper(raw_delta)
        source = "circleworld_delta_future_decay_taper"
    elif mechanism == "stable_time_smooth_3":
        shaped = _circular_smooth_time(raw_delta, kernel_size=3) * stability
        source = "circleworld_delta_time_smooth_3_prefix_magnitude_stability_weighted"
    elif mechanism == "stable_energy_time_smooth_3":
        shaped = _circular_smooth_time(raw_delta, kernel_size=3) * energy * stability
        source = "circleworld_delta_time_smooth_3_prefix_energy_and_stability_weighted"
    elif mechanism == "stable_velocity_time_smooth_3":
        shaped = _circular_smooth_time(raw_delta, kernel_size=3) * coherence * stability
        source = "circleworld_delta_time_smooth_3_prefix_velocity_coherence_and_stability_weighted"
    elif mechanism == "stable_energy_velocity_time_smooth_3":
        shaped = _circular_smooth_time(raw_delta, kernel_size=3) * energy * coherence * stability
        source = "circleworld_delta_time_smooth_3_prefix_energy_velocity_and_stability_weighted"
    elif mechanism == "stable_softclip_time_smooth_3":
        shaped = _soft_clip_phase_delta(_circular_smooth_time(raw_delta, kernel_size=3)) * stability
        source = "circleworld_delta_time_smooth_3_softclipped_prefix_stability_weighted"
    elif mechanism == "stable_energy_velocity_softclip_time_smooth_3":
        shaped = _soft_clip_phase_delta(_circular_smooth_time(raw_delta, kernel_size=3)) * energy * coherence * stability
        source = "circleworld_delta_time_smooth_3_softclipped_prefix_energy_velocity_stability_weighted"
    elif mechanism == "anti_reentry_phase_shear":
        shaped = _anti_reentry_phase_shear(prefix_phase, raw_delta.size(-1))
        source = "prefix_phase_velocity_centered_future_shear"
    elif mechanism == "anti_reentry_curvature_shear":
        shaped = _anti_reentry_curvature_shear(prefix_phase, raw_delta.size(-1))
        source = "prefix_phase_acceleration_centered_future_shear"
    elif mechanism == "anti_reentry_delta_shear_mix":
        shear = _anti_reentry_phase_shear(prefix_phase, raw_delta.size(-1))
        shaped = _soft_clip_phase_delta(_circular_smooth_time(raw_delta, kernel_size=3) + shear, limit=0.75) * (
            0.5 + 0.5 * stability
        )
        source = "circleworld_time_smooth_delta_plus_prefix_phase_velocity_reentry_shear"
    elif mechanism == "anti_reentry_late_decorrelator":
        shaped = _anti_reentry_late_decorrelator(prefix_phase, raw_delta.size(-1))
        source = "deterministic_late_frequency_phase_decorrelator"
    elif mechanism == "anti_reentry_staggered_decorrelator":
        shaped = _anti_reentry_staggered_decorrelator(prefix_phase, raw_delta.size(-1))
        source = "deterministic_staggered_time_frequency_phase_decorrelator"
    elif mechanism == "anti_reentry_delta_decorrelator_mix":
        decor = _anti_reentry_late_decorrelator(prefix_phase, raw_delta.size(-1))
        shaped = _soft_clip_phase_delta(_circular_smooth_time(raw_delta, kernel_size=3) + decor, limit=0.85) * (
            0.5 + 0.5 * stability
        )
        source = "circleworld_time_smooth_delta_plus_late_phase_decorrelator"
    else:
        raise ValueError(f"Unknown mechanism: {mechanism}; expected one of {MECHANISMS}")
    return _wrap_angle(shaped), {
        "mechanism": mechanism,
        "mechanism_source": source,
        "mean_abs_raw_delta": float(raw_delta.abs().mean().detach().cpu().item()),
        "mean_abs_shaped_delta": float(shaped.abs().mean().detach().cpu().item()),
        "mean_energy_weight": float(energy.mean().detach().cpu().item()),
        "mean_phase_velocity_coherence": float(coherence.mean().detach().cpu().item()),
        "mean_prefix_magnitude_stability": float(stability.mean().detach().cpu().item()),
        "future_target_magnitude_reused": False,
        "target_future_stft_magnitude_accessed": False,
        "future_target_phase_reused": False,
        "target_future_stft_phase_accessed": False,
    }


def _phase_delta_stats(delta: torch.Tensor, mask: torch.Tensor) -> dict[str, float]:
    masked = delta * mask
    active = mask > 0.0
    active_delta = delta[active] if bool(active.any().detach().cpu().item()) else delta.reshape(-1)
    return {
        "future_mean_abs_phase_delta": float(delta.abs().mean().detach().cpu().item()),
        "future_rms_phase_delta": float(torch.sqrt(torch.mean(delta * delta)).detach().cpu().item()),
        "future_mean_cos_phase_delta": float(torch.cos(delta).mean().detach().cpu().item()),
        "masked_mean_abs_phase_delta": float(masked.abs().mean().detach().cpu().item()),
        "active_bin_mean_abs_phase_delta": float(active_delta.abs().mean().detach().cpu().item()),
    }


def _aggregate(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str, float], list[dict[str, Any]]] = defaultdict(list)
    gain0: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    for row in rows:
        mode = str(row["magnitude_mode"])
        mask = str(row["mask_mode"])
        mechanism = str(row["mechanism"])
        gain = float(row["gain"])
        grouped[(mode, mask, mechanism, gain)].append(row)
        if abs(gain) < 1.0e-12:
            gain0[(str(row["name"]), mode, mask, mechanism)] = row

    out: list[dict[str, Any]] = []
    for (mode, mask, mechanism, gain), group in sorted(grouped.items(), key=lambda item: item[0]):
        corr_deltas: list[float] = []
        mse_deltas: list[float] = []
        corr_wins = 0
        mse_wins = 0
        for row in group:
            base = gain0.get((str(row["name"]), mode, mask, mechanism))
            if not base:
                continue
            corr_delta = _as_float(row["target_corr"]) - _as_float(base["target_corr"])
            mse_delta = _as_float(row["target_mse"]) - _as_float(base["target_mse"])
            corr_deltas.append(corr_delta)
            mse_deltas.append(mse_delta)
            corr_wins += int(corr_delta > 1.0e-6)
            mse_wins += int(mse_delta < -1.0e-9)
        mean_corr_delta = _mean(corr_deltas)
        mean_mse_delta = _mean(mse_deltas)
        corr_win_fraction = float(corr_wins / max(1, len(corr_deltas)))
        mse_win_fraction = float(mse_wins / max(1, len(mse_deltas)))
        mean_vs_gain0_corr = _mean([_as_float(row["vs_gain0_corr"]) for row in group])
        if abs(gain) < 1.0e-12:
            verdict = "seed_baseline"
        elif (
            mean_corr_delta > MECHANISM_CANDIDATE_MIN_CORR_DELTA
            and corr_win_fraction >= MECHANISM_CANDIDATE_MIN_CORR_WIN_FRACTION
            and mean_mse_delta <= 0.0
        ):
            verdict = "mechanism_helpful_candidate"
        elif mean_corr_delta > 0.0 and mean_mse_delta <= 0.0:
            verdict = "mechanism_tradeoff_signal"
        elif mean_mse_delta < 0.0:
            verdict = "mse_only_or_corr_tradeoff"
        elif mean_vs_gain0_corr >= 0.95 and abs(mean_corr_delta) <= 0.005:
            verdict = "carrier_locked_decorative"
        else:
            verdict = "not_improved"
        out.append(
            {
                "magnitude_mode": mode,
                "mask_mode": mask,
                "mechanism": mechanism,
                "gain": gain,
                "case_count": len(group),
                "mean_target_corr": _mean([_as_float(row["target_corr"]) for row in group]),
                "mean_target_mae": _mean([_as_float(row["target_mae"]) for row in group]),
                "mean_target_mse": _mean([_as_float(row["target_mse"]) for row in group]),
                "mean_vs_gain0_corr": mean_vs_gain0_corr,
                "mean_vs_gain0_mse": _mean([_as_float(row["vs_gain0_mse"]) for row in group]),
                "mean_target_corr_delta_vs_gain0": mean_corr_delta,
                "median_target_corr_delta_vs_gain0": _median(corr_deltas),
                "mean_target_mse_delta_vs_gain0": mean_mse_delta,
                "target_corr_win_fraction_vs_gain0": corr_win_fraction,
                "target_mse_win_fraction_vs_gain0": mse_win_fraction,
                "mean_loop_autocorr_peak": _mean([_as_float(row["loop_autocorr_peak"]) for row in group]),
                "mean_first_chunk_reentry": _mean([_as_float(row["first_chunk_reentry"]) for row in group]),
                "mean_mask_bin_fraction": _mean([_as_float(row["mask_bin_fraction"]) for row in group]),
                "mean_active_bin_abs_phase_delta": _mean(
                    [_as_float(row["phase_delta_stats"]["active_bin_mean_abs_phase_delta"]) for row in group]
                ),
                "mean_abs_shaped_delta": _mean(
                    [_as_float(row["mechanism_flags"]["mean_abs_shaped_delta"]) for row in group]
                ),
                "verdict": verdict,
            }
        )
    return out


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


def _overall_status(aggregate: Sequence[dict[str, Any]], flags: dict[str, bool]) -> str:
    if any(flags.values()):
        return "invalid_future_leakage"
    nonzero = [row for row in aggregate if abs(_as_float(row.get("gain"))) > 1.0e-12]
    if any(row.get("verdict") == "mechanism_helpful_candidate" for row in nonzero):
        return "mechanism_candidate_found"
    if any(row.get("verdict") == "mechanism_tradeoff_signal" for row in nonzero):
        return "mechanism_tradeoff_signal_only"
    if any(row.get("verdict") == "mse_only_or_corr_tradeoff" for row in nonzero):
        return "mse_tradeoff_only"
    return "no_mechanism_candidate"


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
    mechanisms: Sequence[str],
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
    raw_delta = _wrap_angle(final_phase[..., prefix_frames:] - seed_phase[..., prefix_frames:])

    rows: list[dict[str, Any]] = []
    seed_future_by_key: dict[tuple[str, str, str], torch.Tensor] = {}
    for mechanism in mechanisms:
        shaped_delta, mechanism_flags = _mechanism_delta(
            mechanism=mechanism,
            raw_delta=raw_delta,
            prefix_mag=prefix_mag,
            prefix_phase=prefix_phase,
        )
        for mode in magnitude_modes:
            mag_canvas, mag_flags = _build_magnitude_canvas(prefix_mag, future_frames=future_frames, mode=mode)
            for mask_mode in mask_modes:
                mask, mask_flags = _future_mask_from_prefix_energy(
                    prefix_mag,
                    future_frames=future_frames,
                    mask_mode=mask_mode,
                    prefix_phase=prefix_phase,
                )
                seed_phase_canvas = _apply_future_delta(
                    seed_phase=seed_phase,
                    delta_future=shaped_delta,
                    mask=mask,
                    gain=0.0,
                    prefix_frames=prefix_frames,
                )
                seed_future = _render_future_from_stft(
                    mag_canvas=mag_canvas,
                    phase_canvas=seed_phase_canvas,
                    stft_cfg=stft_cfg,
                    prefix_samples=prefix_samples,
                    future_samples=future_samples,
                )
                seed_future_by_key[(mode, mask_mode, mechanism)] = seed_future
                stats = _phase_delta_stats(shaped_delta, mask)
                for gain in gains:
                    phase = _apply_future_delta(
                        seed_phase=seed_phase,
                        delta_future=shaped_delta,
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
                    seed_metrics = _compare_waveforms(seed_future_by_key[(mode, mask_mode, mechanism)], pred)
                    loop = _loop_reentry_metrics(pred, sr)
                    access_flags = _merge_access_flags(
                        phase_flags=seed_flags,
                        magnitude_flags=(mag_flags, mask_flags, mechanism_flags),
                    )
                    rows.append(
                        {
                            "name": name,
                            "source_wav": str(wav_path),
                            "magnitude_mode": mode,
                            "mask_mode": mask_mode,
                            "mechanism": mechanism,
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
                            "mechanism_flags": mechanism_flags,
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
    mechanisms: Sequence[str],
    gains: Sequence[float],
    cases_json: Path | None = None,
) -> dict[str, Any]:
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
        raise RuntimeError("No cases selected for audio delta mechanism probe")

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
            mechanisms=mechanisms,
            gains=gains,
        )
        for name, wav_path in cases
    ]
    rows = [row for case in case_rows for row in case["rows"]]
    aggregate = _aggregate(rows)
    flags = _future_access_flags(rows)
    summary = _jsonify(
        {
            "runtime": "circleworld_proto",
            "schema": "circleworld_audio_delta_mechanism_probe_v0",
            "status": _overall_status(aggregate, flags),
            "config_path": str(config_path),
            "out_dir": str(out_dir),
            "device": str(device),
            "requested_device": str(device_name),
            "sample_rate": sr,
            "prefix_seconds": float(prefix_seconds),
            "future_seconds": float(future_seconds),
            "magnitude_modes": list(magnitude_modes),
            "mask_modes": list(mask_modes),
            "mechanisms": list(mechanisms),
            "gains": [float(gain) for gain in gains],
            "phase_seed_policy": PHASE_SEED_POLICY,
            "case_count": len(case_rows),
            "future_target_audio_used_for_metrics_only": True,
            **flags,
            "aggregate": aggregate,
            "cases": case_rows,
        }
    )
    (out_dir / OUTPUT_JSON).write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (out_dir / OUTPUT_MD).write_text(_markdown_summary(summary), encoding="utf-8")
    return summary


def _markdown_summary(summary: dict[str, Any]) -> str:
    def fmt(value: Any) -> str:
        if value is None:
            return "NA"
        if isinstance(value, (int, float)):
            return f"{float(value):.6g}"
        return str(value)

    lines = [
        "# Audio Delta Mechanism Probe",
        "",
        "This tests fixed prefix-only shaping mechanisms over Circleworld future-frame deltas.",
        "Gain 0 is the legal copyphase carrier; mechanisms are not selected per case.",
        "",
        f"- status: `{summary['status']}`",
        f"- config: `{summary['config_path']}`",
        f"- cases: `{summary['case_count']}`",
        f"- phase seed policy: `{summary['phase_seed_policy']}`",
        f"- mechanisms: `{', '.join(summary['mechanisms'])}`",
        f"- future target magnitude reused: `{summary['future_target_magnitude_reused']}`",
        f"- future target phase reused: `{summary['future_target_phase_reused']}`",
        "",
        "## Aggregate Rows",
        "",
        "| mode | mask | mechanism | gain | corr | MSE | vs gain0 corr | corr delta | median corr delta | MSE delta | corr wins | MSE wins | shaped abs delta | verdict |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in summary["aggregate"]:
        lines.append(
            "| {mode} | {mask} | {mechanism} | {gain} | {corr} | {mse} | {vcorr} | {dcorr} | {mcorr} | {dmse} | {wcorr} | {wmse} | {sdelta} | {verdict} |".format(
                mode=row.get("magnitude_mode"),
                mask=row.get("mask_mode"),
                mechanism=row.get("mechanism"),
                gain=fmt(row.get("gain")),
                corr=fmt(row.get("mean_target_corr")),
                mse=fmt(row.get("mean_target_mse")),
                vcorr=fmt(row.get("mean_vs_gain0_corr")),
                dcorr=fmt(row.get("mean_target_corr_delta_vs_gain0")),
                mcorr=fmt(row.get("median_target_corr_delta_vs_gain0")),
                dmse=fmt(row.get("mean_target_mse_delta_vs_gain0")),
                wcorr=fmt(row.get("target_corr_win_fraction_vs_gain0")),
                wmse=fmt(row.get("target_mse_win_fraction_vs_gain0")),
                sdelta=fmt(row.get("mean_abs_shaped_delta")),
                verdict=row.get("verdict"),
            )
        )
    lines.extend(
        [
            "",
            "## Interpretation Guard",
            "",
            "- Mechanisms use prefix-only energy or phase-velocity statistics; target future STFT is used only for metrics.",
            "- A mechanism candidate requires the fixed row to improve correlation materially, not just MSE.",
            "- Best mechanism rows are diagnostic unless the mechanism and objective row are frozen before a lockbox run.",
        ]
    )
    return "\n".join(lines) + "\n"


# Keep this legacy probe runner wired to the canonical operator-bank module.
# The local definitions above remain for historical diff readability, but the
# exported globals used by _run_case and main are rebound at import time.
CORE_MECHANISMS = OPERATOR_CORE_MECHANISMS
EXPERIMENTAL_MECHANISMS = OPERATOR_EXPERIMENTAL_MECHANISMS
MECHANISMS = OPERATOR_MECHANISMS
_mechanism_delta = _operator_mechanism_delta
_phase_delta_stats = _operator_phase_delta_stats


def main() -> None:
    ap = argparse.ArgumentParser(description="Probe fixed prefix-only mechanisms for shaping Circleworld audio deltas.")
    ap.add_argument("--config", required=True, help="Circleworld JSON config path.")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--num-cases", type=int, default=3)
    ap.add_argument("--prefix-seconds", type=float, default=1.0)
    ap.add_argument("--future-seconds", type=float, default=1.0)
    ap.add_argument("--magnitude-modes", default="prefix_hold,flat")
    ap.add_argument("--masks", default=",".join(CORE_MASK_MODES))
    ap.add_argument("--mechanisms", default=",".join(CORE_MECHANISMS))
    ap.add_argument("--gains", default=",".join(str(value) for value in DEFAULT_GAINS))
    ap.add_argument("--cases-json", default=None)
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
        mechanisms=_parse_csv(str(args.mechanisms), MECHANISMS, "mechanism"),
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
