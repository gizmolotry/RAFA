from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn.functional as F


CORE_MECHANISMS: tuple[str, ...] = (
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

EXPERIMENTAL_MECHANISMS: tuple[str, ...] = (
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

MECHANISMS: tuple[str, ...] = CORE_MECHANISMS + EXPERIMENTAL_MECHANISMS


def wrap_angle(angle: torch.Tensor) -> torch.Tensor:
    return torch.atan2(torch.sin(angle), torch.cos(angle))


def circular_smooth_time(delta: torch.Tensor, kernel_size: int) -> torch.Tensor:
    kernel_size = max(1, int(kernel_size))
    if kernel_size <= 1 or delta.size(-1) <= 1:
        return delta
    pad_left = kernel_size // 2
    pad_right = kernel_size - 1 - pad_left
    bsz, f_bins, t_len = delta.shape
    cos = torch.cos(delta).reshape(bsz * f_bins, 1, t_len)
    sin = torch.sin(delta).reshape(bsz * f_bins, 1, t_len)
    cos_s = F.avg_pool1d(F.pad(cos, (pad_left, pad_right), mode="replicate"), kernel_size=kernel_size, stride=1)
    sin_s = F.avg_pool1d(F.pad(sin, (pad_left, pad_right), mode="replicate"), kernel_size=kernel_size, stride=1)
    return torch.atan2(sin_s.reshape(bsz, f_bins, t_len), cos_s.reshape(bsz, f_bins, t_len))


def circular_smooth_freq(delta: torch.Tensor, kernel_size: int) -> torch.Tensor:
    kernel_size = max(1, int(kernel_size))
    if kernel_size <= 1 or delta.size(1) <= 1:
        return delta
    pad_left = kernel_size // 2
    pad_right = kernel_size - 1 - pad_left
    bsz, f_bins, t_len = delta.shape
    cos = torch.cos(delta).permute(0, 2, 1).reshape(bsz * t_len, 1, f_bins)
    sin = torch.sin(delta).permute(0, 2, 1).reshape(bsz * t_len, 1, f_bins)
    cos_s = F.avg_pool1d(F.pad(cos, (pad_left, pad_right), mode="replicate"), kernel_size=kernel_size, stride=1)
    sin_s = F.avg_pool1d(F.pad(sin, (pad_left, pad_right), mode="replicate"), kernel_size=kernel_size, stride=1)
    return torch.atan2(sin_s.reshape(bsz, t_len, f_bins), cos_s.reshape(bsz, t_len, f_bins)).permute(0, 2, 1)


def prefix_energy_weight(prefix_mag: torch.Tensor) -> torch.Tensor:
    energy = (prefix_mag.detach() * prefix_mag.detach()).mean(dim=-1, keepdim=True)
    denom = energy.amax(dim=1, keepdim=True).clamp_min(1.0e-8)
    return (energy / denom).clamp(0.0, 1.0)


def phase_velocity_coherence(prefix_phase: torch.Tensor, history_frames: int = 8) -> torch.Tensor:
    if prefix_phase.size(-1) < 2:
        return torch.ones(prefix_phase.shape[:2] + (1,), device=prefix_phase.device, dtype=prefix_phase.dtype)
    diff = wrap_angle(prefix_phase[..., 1:] - prefix_phase[..., :-1])
    diff = diff[..., -max(1, int(history_frames)) :]
    mean_cos = torch.cos(diff).mean(dim=-1, keepdim=True)
    mean_sin = torch.sin(diff).mean(dim=-1, keepdim=True)
    return torch.sqrt(mean_cos * mean_cos + mean_sin * mean_sin).clamp(0.0, 1.0)


def prefix_circular_velocity(prefix_phase: torch.Tensor, history_frames: int = 8) -> torch.Tensor:
    if prefix_phase.size(-1) < 2:
        return torch.zeros(prefix_phase.shape[:2] + (1,), device=prefix_phase.device, dtype=prefix_phase.dtype)
    diff = wrap_angle(prefix_phase[..., 1:] - prefix_phase[..., :-1])
    diff = diff[..., -max(1, int(history_frames)) :]
    mean_cos = torch.cos(diff).mean(dim=-1, keepdim=True)
    mean_sin = torch.sin(diff).mean(dim=-1, keepdim=True)
    return torch.atan2(mean_sin, mean_cos)


def prefix_circular_acceleration(prefix_phase: torch.Tensor, history_frames: int = 8) -> torch.Tensor:
    if prefix_phase.size(-1) < 3:
        return torch.zeros(prefix_phase.shape[:2] + (1,), device=prefix_phase.device, dtype=prefix_phase.dtype)
    diff = wrap_angle(prefix_phase[..., 1:] - prefix_phase[..., :-1])
    accel = wrap_angle(diff[..., 1:] - diff[..., :-1])
    accel = accel[..., -max(1, int(history_frames)) :]
    mean_cos = torch.cos(accel).mean(dim=-1, keepdim=True)
    mean_sin = torch.sin(accel).mean(dim=-1, keepdim=True)
    return torch.atan2(mean_sin, mean_cos)


def center_frequency_law(x: torch.Tensor) -> torch.Tensor:
    return x - x.mean(dim=1, keepdim=True)


def soft_clip_phase_delta(delta: torch.Tensor, limit: float = 0.35) -> torch.Tensor:
    limit = max(1.0e-6, float(limit))
    return limit * torch.tanh(delta / limit)


def anti_reentry_phase_shear(prefix_phase: torch.Tensor, future_frames: int, *, power: float = 1.35) -> torch.Tensor:
    velocity = center_frequency_law(prefix_circular_velocity(prefix_phase, history_frames=8))
    ramp = torch.linspace(0.0, 1.0, max(1, int(future_frames)), device=prefix_phase.device, dtype=prefix_phase.dtype)
    ramp = ramp.clamp_min(0.0).pow(float(power)).view(1, 1, -1)
    return soft_clip_phase_delta(velocity * ramp, limit=0.55)


def anti_reentry_curvature_shear(prefix_phase: torch.Tensor, future_frames: int) -> torch.Tensor:
    accel = center_frequency_law(prefix_circular_acceleration(prefix_phase, history_frames=8))
    ramp = torch.linspace(0.0, 1.0, max(1, int(future_frames)), device=prefix_phase.device, dtype=prefix_phase.dtype)
    ramp = (ramp * ramp).view(1, 1, -1)
    return soft_clip_phase_delta(accel * ramp, limit=0.45)


def frequency_hash(prefix_phase: torch.Tensor) -> torch.Tensor:
    freq = torch.arange(prefix_phase.size(1), device=prefix_phase.device, dtype=prefix_phase.dtype).view(1, -1, 1)
    return torch.sin(freq * 1.61803398875 + 0.37)


def anti_reentry_late_decorrelator(prefix_phase: torch.Tensor, future_frames: int) -> torch.Tensor:
    frames = max(1, int(future_frames))
    t = torch.linspace(0.0, 1.0, frames, device=prefix_phase.device, dtype=prefix_phase.dtype).view(1, 1, -1)
    late = torch.sigmoid((t - 0.38) * 10.0)
    return soft_clip_phase_delta(frequency_hash(prefix_phase) * late, limit=0.85)


def anti_reentry_staggered_decorrelator(prefix_phase: torch.Tensor, future_frames: int) -> torch.Tensor:
    frames = max(1, int(future_frames))
    t = torch.linspace(0.0, 1.0, frames, device=prefix_phase.device, dtype=prefix_phase.dtype).view(1, 1, -1)
    freq = torch.arange(prefix_phase.size(1), device=prefix_phase.device, dtype=prefix_phase.dtype).view(1, -1, 1)
    stagger = torch.sin((freq + 1.0) * (2.0 * math.pi * t) + freq * 0.73)
    late = torch.sigmoid((t - 0.25) * 8.0)
    return soft_clip_phase_delta(stagger * late, limit=0.75)


def prefix_magnitude_stability(prefix_mag: torch.Tensor, history_frames: int = 8) -> torch.Tensor:
    if prefix_mag.size(-1) < 2:
        return torch.ones(prefix_mag.shape[:2] + (1,), device=prefix_mag.device, dtype=prefix_mag.dtype)
    mag = torch.log1p(prefix_mag.detach().clamp_min(0.0))
    flux = (mag[..., 1:] - mag[..., :-1]).abs()
    flux = flux[..., -max(1, int(history_frames)) :]
    flux = flux.mean(dim=-1, keepdim=True)
    denom = flux.amax(dim=1, keepdim=True).clamp_min(1.0e-8)
    return (1.0 - flux / denom).clamp(0.0, 1.0)


def future_taper(delta: torch.Tensor, start: float = 1.0, end: float = 0.25) -> torch.Tensor:
    if delta.size(-1) <= 1:
        return delta * float(start)
    ramp = torch.linspace(float(start), float(end), delta.size(-1), device=delta.device, dtype=delta.dtype)
    return delta * ramp.view(1, 1, -1)


def mechanism_delta(
    *,
    mechanism: str,
    raw_delta: torch.Tensor,
    prefix_mag: torch.Tensor,
    prefix_phase: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, Any]]:
    energy = prefix_energy_weight(prefix_mag)
    coherence = phase_velocity_coherence(prefix_phase, history_frames=8)
    stability = prefix_magnitude_stability(prefix_mag, history_frames=8)
    if mechanism == "raw":
        shaped = raw_delta
        source = "circleworld_raw_delta"
    elif mechanism == "time_smooth_3":
        shaped = circular_smooth_time(raw_delta, kernel_size=3)
        source = "circleworld_delta_circular_time_smooth_3"
    elif mechanism == "time_smooth_7":
        shaped = circular_smooth_time(raw_delta, kernel_size=7)
        source = "circleworld_delta_circular_time_smooth_7"
    elif mechanism == "freq_smooth_5":
        shaped = circular_smooth_freq(raw_delta, kernel_size=5)
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
        shaped = circular_smooth_time(raw_delta, kernel_size=3) * energy * coherence
        source = "circleworld_delta_time_smooth_then_prefix_energy_velocity_coherence_weighted"
    elif mechanism == "future_decay_taper":
        shaped = future_taper(raw_delta)
        source = "circleworld_delta_future_decay_taper"
    elif mechanism == "stable_time_smooth_3":
        shaped = circular_smooth_time(raw_delta, kernel_size=3) * stability
        source = "circleworld_delta_time_smooth_3_prefix_magnitude_stability_weighted"
    elif mechanism == "stable_energy_time_smooth_3":
        shaped = circular_smooth_time(raw_delta, kernel_size=3) * energy * stability
        source = "circleworld_delta_time_smooth_3_prefix_energy_and_stability_weighted"
    elif mechanism == "stable_velocity_time_smooth_3":
        shaped = circular_smooth_time(raw_delta, kernel_size=3) * coherence * stability
        source = "circleworld_delta_time_smooth_3_prefix_velocity_coherence_and_stability_weighted"
    elif mechanism == "stable_energy_velocity_time_smooth_3":
        shaped = circular_smooth_time(raw_delta, kernel_size=3) * energy * coherence * stability
        source = "circleworld_delta_time_smooth_3_prefix_energy_velocity_and_stability_weighted"
    elif mechanism == "stable_softclip_time_smooth_3":
        shaped = soft_clip_phase_delta(circular_smooth_time(raw_delta, kernel_size=3)) * stability
        source = "circleworld_delta_time_smooth_3_softclipped_prefix_stability_weighted"
    elif mechanism == "stable_energy_velocity_softclip_time_smooth_3":
        shaped = soft_clip_phase_delta(circular_smooth_time(raw_delta, kernel_size=3)) * energy * coherence * stability
        source = "circleworld_delta_time_smooth_3_softclipped_prefix_energy_velocity_stability_weighted"
    elif mechanism == "anti_reentry_phase_shear":
        shaped = anti_reentry_phase_shear(prefix_phase, raw_delta.size(-1))
        source = "prefix_phase_velocity_centered_future_shear"
    elif mechanism == "anti_reentry_curvature_shear":
        shaped = anti_reentry_curvature_shear(prefix_phase, raw_delta.size(-1))
        source = "prefix_phase_acceleration_centered_future_shear"
    elif mechanism == "anti_reentry_delta_shear_mix":
        shear = anti_reentry_phase_shear(prefix_phase, raw_delta.size(-1))
        shaped = soft_clip_phase_delta(circular_smooth_time(raw_delta, kernel_size=3) + shear, limit=0.75) * (
            0.5 + 0.5 * stability
        )
        source = "circleworld_time_smooth_delta_plus_prefix_phase_velocity_reentry_shear"
    elif mechanism == "anti_reentry_late_decorrelator":
        shaped = anti_reentry_late_decorrelator(prefix_phase, raw_delta.size(-1))
        source = "deterministic_late_frequency_phase_decorrelator"
    elif mechanism == "anti_reentry_staggered_decorrelator":
        shaped = anti_reentry_staggered_decorrelator(prefix_phase, raw_delta.size(-1))
        source = "deterministic_staggered_time_frequency_phase_decorrelator"
    elif mechanism == "anti_reentry_delta_decorrelator_mix":
        decor = anti_reentry_late_decorrelator(prefix_phase, raw_delta.size(-1))
        shaped = soft_clip_phase_delta(circular_smooth_time(raw_delta, kernel_size=3) + decor, limit=0.85) * (
            0.5 + 0.5 * stability
        )
        source = "circleworld_time_smooth_delta_plus_late_phase_decorrelator"
    else:
        raise ValueError(f"Unknown mechanism: {mechanism}; expected one of {MECHANISMS}")
    return wrap_angle(shaped), {
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


def phase_delta_stats(delta: torch.Tensor, mask: torch.Tensor) -> dict[str, float]:
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


def phasor_unit_norm_error_from_phase(phase: torch.Tensor) -> float:
    phasor = torch.stack([torch.cos(phase), torch.sin(phase)], dim=-1)
    return float((phasor.norm(dim=-1) - 1.0).abs().amax().detach().cpu().item())


def operator_bank_manifest() -> dict[str, Any]:
    return {
        "schema": "phase_native_audio_operator_bank_v1",
        "core_mechanisms": list(CORE_MECHANISMS),
        "experimental_mechanisms": list(EXPERIMENTAL_MECHANISMS),
        "mechanism_count": len(MECHANISMS),
        "future_access_contract": {
            "future_target_magnitude_reused": False,
            "target_future_stft_magnitude_accessed": False,
            "future_target_phase_reused": False,
            "target_future_stft_phase_accessed": False,
        },
    }


# Compatibility aliases for older probe-style code.
_mechanism_delta = mechanism_delta
_phase_delta_stats = phase_delta_stats
_wrap_angle = wrap_angle
