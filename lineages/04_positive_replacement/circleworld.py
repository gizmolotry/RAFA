from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from rafa_relational_signature import (
    build_relational_signature_library,
    extract_relational_signatures,
    summarize_relational_signature_bank,
)
from rafa_math_tools import (
    phasor_apply_delta,
    phasor_normalize,
    ramanujan_relative_score_phasor,
)


@dataclass
class CircleworldConfig:
    qset: tuple[int, ...] = (2, 3, 4, 5, 6, 8, 12)
    q_weights: tuple[float, ...] = (1.0, 1.0, 0.9, 0.8, 0.7, 0.6, 0.5)
    promotion_threshold: float = 0.58
    max_promotions: int = 4
    recursion_depth: int = 2
    residue_scale: float = 0.75
    child_law_gain: float = 0.25
    attack_window: int = 8
    persistence_momentum: float = 0.6
    soft_matryoshka_enabled: bool = False
    matryoshka_rank: int = 24
    phase_law_precondition_gain: float = 0.0
    phase_law_velocity_mix: float = 0.0
    phase_law_stability_gain: float = 1.0
    phase_law_softclip: float = 0.0
    phase_law_low_rank: int = 0
    phase_law_consensus_mix: float = 0.0
    phase_law_consensus_damping: float = 0.0
    phase_law_median_guard: float = 0.0
    phase_law_local_velocity_mix: float = 0.0
    phase_law_local_coherence_damping: float = 0.0
    phase_law_curvature_guard: float = 0.0
    phase_law_reentry_mix: float = 0.0
    phase_law_reentry_accel_mix: float = 0.0
    phase_law_reentry_causal: bool = False
    prefix_fracs: tuple[float, ...] = (0.125, 0.25, 0.5)
    slow_persistence: float = 0.96
    fast_persistence: float = 0.22
    persistence_curve: float = 1.8
    slow_write_scale: float = 0.08
    fast_write_scale: float = 1.0
    write_curve: float = 1.4
    prefix_coarse_weight: float = 0.40
    prefix_mid_weight: float = 0.25
    branching_mode: str = "single_path"
    num_modes: int = 2
    readout_mode: str = "weighted_mixture"
    branch_law_version: str = "parametric_v1"
    q_trace_rank: int = 4
    dormant_logit: float = -6.0
    dormant_support: float = 0.02
    dormant_q_scale: float = 0.02
    split_pressure: float = 0.24
    split_seed_scale: float = 0.10
    split_support_gain: float = 0.45
    merge_pressure: float = 0.16
    merge_phase_tol: float = 0.82
    merge_support_overlap_weight: float = 0.50
    survival_coherence_weight: float = 0.40
    survival_arc_weight: float = 0.55
    survival_qtrace_weight: float = 0.25
    survival_residue_penalty: float = 0.30
    collapse_sharpness: float = 1.25
    support_decay: float = 0.10
    support_spread: int = 5
    support_overlap_penalty: float = 0.15
    anti_fixation_weight: float = 0.20
    readout_temperature: float = 0.85
    slot2_support_threshold: float = 0.10
    real_branch_threshold: float = 0.12
    child_branch_parent_threshold: float = 0.12
    child_branch_writeback_threshold: float = 0.0
    child_branch_meso_threshold: float = 0.0
    child_branch_live_threshold: float = 0.0
    mode_perturb_window_frac: float = 0.18
    qtrace_momentum: float = 0.85
    mask_neighborhood: int = 3
    topology_mask_gain: float = 0.55
    complexity_mask_gain: float = 0.60
    context_mask_gain: float = 0.45
    contrastive_mask_gain: float = 0.70
    aux_mask_suppression: float = 0.35
    instability_mask_gain: float = 0.75
    defect_phase_gain: float = 0.40
    defect_q_gain: float = 0.30
    defect_residue_gain: float = 0.15
    defect_sharpness_gain: float = 0.15
    defect_world_grad_gain: float = 0.20
    instability_seed_scale: float = 0.18
    instability_support_gain: float = 0.28
    instability_logit_gain: float = 0.22
    branch_kernel_version: str = "ramanujan"
    relation_attention_gain: float = 0.65
    relation_attention_sharpness: float = 1.10
    relation_value_gain: float = 0.30
    relation_support_gain: float = 0.22
    relation_logit_gain: float = 0.24
    relation_qtrace_gain: float = 0.14
    relation_residual_mix: float = 0.60
    child_spawn_threshold: float = 0.34
    child_max_worlds: int = 4
    child_min_age_for_writeback: int = 2
    child_support_window: int = 12
    child_support_decay: float = 0.12
    child_survival_coherence_weight: float = 0.42
    child_survival_qtrace_weight: float = 0.24
    child_survival_residue_penalty: float = 0.22
    child_writeback_gain: float = 0.32
    child_writeback_rank: int = 12
    child_writeback_temperature: float = 0.85
    child_writeback_budget: float = 1.25
    child_parent_mix: float = 0.15
    child_parent_mix_early: float = 0.08
    child_operator_seed_gain: float = 2.40
    child_operator_promotability_gain: float = 0.60
    child_writeback_phase_delta_gain: float = 1.00
    child_writeback_operator_mix: float = 0.25
    child_writeback_phase_floor_target: float = 0.30
    child_writeback_phase_floor_threshold_mult: float = 2.50
    child_writeback_phase_floor_gain: float = 2.10
    child_writeback_parent_mix_cap: float = 0.04
    child_support_writeback_gain: float = 1.00
    child_support_writeback_floor_gain: float = 0.35
    child_assay_writeback_scale_enabled: bool = False
    child_assay_writeback_scale_default: float = 1.0
    child_kill_threshold: float = 0.08
    child_local_ifs_enabled: bool = False
    child_local_steps: int = 1
    child_local_support_only: bool = True
    child_local_coherence_retention_enabled: bool = False
    child_local_coherence_retention_mix: float = 0.0
    child_local_coherence_floor: float = 0.0
    child_local_coherence_floor_support: float = 0.18
    child_local_coherence_causal_gate_enabled: bool = False
    child_local_coherence_causal_min_delta: float = 0.02
    child_local_coherence_causal_full_delta: float = 0.18
    child_local_coherence_causal_gate_floor: float = 0.0
    law_packet_merge_threshold: float = 0.92
    law_packet_min_score: float = 0.25
    law_packet_topk_families: int = 8


BRANCH_LAW_FEATURE_NAMES: tuple[str, ...] = (
    "phase_agreement",
    "q_trace_similarity",
    "major_arc_compatibility",
    "minor_residue_disagreement",
    "support_overlap",
    "branch_defect",
    "mode0_coherence",
    "mode1_coherence",
    "inherited_rational_history",
    "mode0_support",
    "mode1_support",
    "q_trace_disagreement",
)

LEARNED_BRANCH_LAW_OUTPUT_NAMES: tuple[str, ...] = (
    "split_drive",
    "coexistence_drive",
    "merge_drive",
    "survival_delta0",
    "survival_delta1",
    "collapse_pressure",
    "support_delta0",
    "support_delta1",
    "coherence_target",
    "qtrace_inheritance_mix",
)


class RafaLearnedBranchLawV0(nn.Module):
    """Tiny shadow branch law used by assays, not production runtime selection."""

    def __init__(
        self,
        input_dim: int = len(BRANCH_LAW_FEATURE_NAMES),
        hidden_dim: int = 32,
        output_dim: int = len(LEARNED_BRANCH_LAW_OUTPUT_NAMES),
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, features: torch.Tensor) -> dict[str, torch.Tensor]:
        raw = self.net(features)
        out: dict[str, torch.Tensor] = {}
        sigmoid_names = {
            "split_drive",
            "coexistence_drive",
            "merge_drive",
            "collapse_pressure",
            "coherence_target",
            "qtrace_inheritance_mix",
        }
        for idx, name in enumerate(LEARNED_BRANCH_LAW_OUTPUT_NAMES):
            value = raw[..., idx]
            out[name] = torch.sigmoid(value) if name in sigmoid_names else torch.tanh(value)
        return out


def branch_law_feature_names() -> tuple[str, ...]:
    return BRANCH_LAW_FEATURE_NAMES


def learned_branch_law_output_names() -> tuple[str, ...]:
    return LEARNED_BRANCH_LAW_OUTPUT_NAMES


def _dct_basis(size: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    n = torch.arange(size, device=device, dtype=dtype)
    k = n.view(-1, 1)
    basis = torch.cos(torch.pi / float(size) * (n + 0.5) * k)
    basis[0] = basis[0] / math.sqrt(2.0)
    basis = basis * math.sqrt(2.0 / float(size))
    return basis


def _basis_project(z: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    return torch.einsum("kf,bftc->bktc", basis, z)


def _basis_reconstruct(coeff: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    return torch.einsum("fk,bktc->bftc", basis.transpose(0, 1), coeff)


def _prefix_count(size: int, frac: float) -> int:
    return max(1, min(size, int(round(size * float(frac)))))


def _basis_profiles(size: int, cfg: CircleworldConfig, device: torch.device, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
    idx = torch.linspace(0.0, 1.0, steps=size, device=device, dtype=dtype)
    persistence = cfg.slow_persistence + (cfg.fast_persistence - cfg.slow_persistence) * idx.pow(cfg.persistence_curve)
    write_scale = cfg.slow_write_scale + (cfg.fast_write_scale - cfg.slow_write_scale) * idx.pow(cfg.write_curve)
    return persistence.clamp(0.01, 0.999), write_scale.clamp(0.01, 4.0)


def _low_rank_delta(delta: torch.Tensor, rank: int) -> torch.Tensor:
    if delta.dim() != 3:
        raise ValueError("delta must have shape [B, F, T]")
    f_bins = delta.size(1)
    rank = max(1, min(int(rank), f_bins))
    basis = _dct_basis(f_bins, device=delta.device, dtype=delta.dtype)
    coeff = torch.einsum("kf,bft->bkt", basis[:rank], delta)
    return torch.einsum("fk,bkt->bft", basis[:rank].transpose(0, 1), coeff)


def _wrapped_delta(delta: torch.Tensor) -> torch.Tensor:
    return torch.atan2(torch.sin(delta), torch.cos(delta))


def _phase_velocity_from_state(z: torch.Tensor) -> torch.Tensor:
    if z.size(2) < 2:
        return z.new_zeros(z.shape[:-1])
    vel = _phasor_phase_delta(z[:, :, :-1, :], z[:, :, 1:, :])
    out = z.new_zeros(z.shape[:-1])
    out[:, :, 1:] = vel
    out[:, :, 0] = vel[:, :, 0]
    return out


def _phase_causal_velocity_from_state(z: torch.Tensor) -> torch.Tensor:
    if z.size(2) < 2:
        return z.new_zeros(z.shape[:-1])
    vel = _phasor_phase_delta(z[:, :, :-1, :], z[:, :, 1:, :])
    out = z.new_zeros(z.shape[:-1])
    out[:, :, 1:] = vel
    return out


def _phase_temporal_stability_gate(z: torch.Tensor, stability_gain: float) -> torch.Tensor:
    if z.size(2) < 3 or stability_gain <= 0.0:
        return z.new_ones(z.shape[:-1])
    vel = _phasor_phase_delta(z[:, :, :-1, :], z[:, :, 1:, :])
    accel = _wrapped_delta(vel[:, :, 1:] - vel[:, :, :-1]).abs()
    accel_gate = torch.exp(-float(stability_gain) * accel).clamp(0.0, 1.0)
    gate = z.new_ones(z.shape[:-1])
    gate[:, :, 2:] = accel_gate
    gate[:, :, 1] = accel_gate[:, :, 0]
    gate[:, :, 0] = gate[:, :, 1]
    return gate


def _circular_time_smooth(angle: torch.Tensor, kernel_size: int = 3) -> torch.Tensor:
    kernel_size = max(1, int(kernel_size))
    if kernel_size <= 1 or angle.size(-1) <= 1:
        return angle
    pad_left = kernel_size // 2
    pad_right = kernel_size - 1 - pad_left
    b, f, t = angle.shape
    cos = torch.cos(angle).reshape(b * f, 1, t)
    sin = torch.sin(angle).reshape(b * f, 1, t)
    cos_s = F.avg_pool1d(F.pad(cos, (pad_left, pad_right), mode="replicate"), kernel_size=kernel_size, stride=1)
    sin_s = F.avg_pool1d(F.pad(sin, (pad_left, pad_right), mode="replicate"), kernel_size=kernel_size, stride=1)
    return torch.atan2(sin_s.reshape(b, f, t), cos_s.reshape(b, f, t))


def _phase_velocity_consensus(z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    vel = _phase_velocity_from_state(z)
    if vel.numel() == 0:
        return vel, torch.ones_like(vel)
    vel_t = _circular_time_smooth(vel, kernel_size=3)
    freq_cos = torch.cos(vel_t).mean(dim=1, keepdim=True)
    freq_sin = torch.sin(vel_t).mean(dim=1, keepdim=True)
    freq_consensus = torch.atan2(freq_sin, freq_cos).expand_as(vel_t)
    freq_coherence = torch.sqrt(freq_cos * freq_cos + freq_sin * freq_sin).clamp(0.0, 1.0).expand_as(vel_t)
    local_agreement = (0.5 + 0.5 * torch.cos(_wrapped_delta(vel - freq_consensus))).clamp(0.0, 1.0)
    consensus_gate = (freq_coherence * local_agreement).clamp(0.0, 1.0)
    return freq_consensus, consensus_gate


def _phase_local_velocity_carrier(z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    vel = _phase_velocity_from_state(z)
    if vel.numel() == 0:
        return vel, torch.ones_like(vel)

    vel_t = _circular_time_smooth(vel, kernel_size=3)
    if vel_t.size(1) <= 1:
        local_carrier = vel_t
        local_coherence = torch.ones_like(vel_t)
    else:
        b, f, t = vel_t.shape
        cos = torch.cos(vel_t).permute(0, 2, 1).reshape(b * t, 1, f)
        sin = torch.sin(vel_t).permute(0, 2, 1).reshape(b * t, 1, f)
        cos_s = F.avg_pool1d(F.pad(cos, (1, 1), mode="replicate"), kernel_size=3, stride=1)
        sin_s = F.avg_pool1d(F.pad(sin, (1, 1), mode="replicate"), kernel_size=3, stride=1)
        cos_s = cos_s.reshape(b, t, f).permute(0, 2, 1)
        sin_s = sin_s.reshape(b, t, f).permute(0, 2, 1)
        local_carrier = torch.atan2(sin_s, cos_s)
        local_coherence = torch.sqrt(cos_s * cos_s + sin_s * sin_s).clamp(0.0, 1.0)

    local_agreement = (0.5 + 0.5 * torch.cos(_wrapped_delta(vel - local_carrier))).clamp(0.0, 1.0)
    local_gate = (local_coherence * local_agreement).clamp(0.0, 1.0)
    return local_carrier, local_gate


def _phase_temporal_reentry_carrier(
    z: torch.Tensor,
    accel_mix: float,
    stability_gain: float,
    causal: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    vel = _phase_causal_velocity_from_state(z) if causal else _phase_velocity_from_state(z)
    if vel.numel() == 0:
        return vel, torch.ones_like(vel)

    accel = torch.zeros_like(vel)
    if vel.size(2) > 1:
        accel[:, :, 1:] = _wrapped_delta(vel[:, :, 1:] - vel[:, :, :-1])
        if not causal:
            accel[:, :, 0] = accel[:, :, 1]

    carrier = vel
    if vel.size(2) > 1:
        prev = vel.clone()
        prev[:, :, 1:] = vel[:, :, :-1]
        carrier = _wrapped_delta(0.65 * vel + 0.35 * prev)
    if accel_mix != 0.0:
        carrier = _wrapped_delta(carrier + float(accel_mix) * accel)

    strength = max(0.0, float(stability_gain))
    gate = torch.exp(-strength * accel.abs()).clamp(0.0, 1.0) if strength > 0.0 else torch.ones_like(vel)
    return carrier, gate


def _phase_curvature_gate(z: torch.Tensor, strength: float) -> torch.Tensor:
    strength = max(0.0, float(strength))
    if strength <= 0.0 or z.size(2) < 3:
        return z.new_ones(z.shape[:-1])
    vel = _phase_velocity_from_state(z)
    accel = z.new_zeros(vel.shape)
    accel[:, :, 1:] = _wrapped_delta(vel[:, :, 1:] - vel[:, :, :-1]).abs()
    accel[:, :, 0] = accel[:, :, 1]
    return torch.exp(-strength * accel).clamp(0.0, 1.0)


def _median_guard_delta(delta: torch.Tensor, strength: float) -> torch.Tensor:
    strength = max(0.0, min(1.0, float(strength)))
    if strength <= 0.0 or delta.numel() == 0:
        return delta
    flat = delta.abs().flatten(1)
    median = flat.median(dim=1).values.view(delta.size(0), 1, 1).clamp_min(1.0e-6)
    guard = (median / (median + delta.abs())).clamp(0.0, 1.0)
    return delta * ((1.0 - strength) + strength * guard)


def _precondition_phase_delta(
    z: torch.Tensor,
    delta: torch.Tensor,
    cfg: CircleworldConfig,
) -> torch.Tensor:
    gain = max(0.0, min(1.0, float(cfg.phase_law_precondition_gain)))
    velocity_mix = float(cfg.phase_law_velocity_mix)
    softclip = max(0.0, float(cfg.phase_law_softclip))
    low_rank = int(cfg.phase_law_low_rank)
    consensus_mix = float(cfg.phase_law_consensus_mix)
    consensus_damping = max(0.0, min(1.0, float(cfg.phase_law_consensus_damping)))
    median_guard = max(0.0, min(1.0, float(cfg.phase_law_median_guard)))
    local_velocity_mix = float(cfg.phase_law_local_velocity_mix)
    local_coherence_damping = max(0.0, min(1.0, float(cfg.phase_law_local_coherence_damping)))
    curvature_guard = max(0.0, min(1.0, float(cfg.phase_law_curvature_guard)))
    reentry_mix = max(0.0, min(0.25, float(cfg.phase_law_reentry_mix)))
    reentry_accel_mix = max(0.0, min(1.25, float(cfg.phase_law_reentry_accel_mix)))
    if (
        gain <= 0.0
        and velocity_mix == 0.0
        and softclip <= 0.0
        and low_rank <= 0
        and consensus_mix == 0.0
        and consensus_damping <= 0.0
        and median_guard <= 0.0
        and local_velocity_mix == 0.0
        and local_coherence_damping <= 0.0
        and curvature_guard <= 0.0
        and reentry_mix == 0.0
    ):
        return delta

    gate = _phase_temporal_stability_gate(z, float(cfg.phase_law_stability_gain))
    out = delta
    if gain > 0.0:
        out = out * ((1.0 - gain) + gain * gate)
    if velocity_mix != 0.0:
        out = out + velocity_mix * gate * _phase_velocity_from_state(z)
    if consensus_mix != 0.0 or consensus_damping > 0.0:
        consensus, consensus_gate = _phase_velocity_consensus(z)
        if consensus_mix != 0.0:
            out = out + consensus_mix * gate * consensus_gate * consensus
        if consensus_damping > 0.0:
            agreement = (0.5 + 0.5 * torch.cos(_wrapped_delta(out - consensus))).clamp(0.0, 1.0)
            damping_gate = (consensus_gate * agreement).clamp(0.0, 1.0)
            out = out * ((1.0 - consensus_damping) + consensus_damping * damping_gate)
    if local_velocity_mix != 0.0 or local_coherence_damping > 0.0:
        local_carrier, local_gate = _phase_local_velocity_carrier(z)
        if local_velocity_mix != 0.0:
            out = out + local_velocity_mix * gate * local_gate * local_carrier
        if local_coherence_damping > 0.0:
            out = out * ((1.0 - local_coherence_damping) + local_coherence_damping * local_gate)
    if reentry_mix != 0.0:
        reentry, reentry_gate = _phase_temporal_reentry_carrier(
            z,
            accel_mix=reentry_accel_mix,
            stability_gain=float(cfg.phase_law_stability_gain),
            causal=bool(cfg.phase_law_reentry_causal),
        )
        out = out + reentry_mix * gate * reentry_gate * reentry
    if curvature_guard > 0.0:
        curvature_gate = _phase_curvature_gate(z, float(cfg.phase_law_stability_gain)).clamp(0.0, 1.0)
        out = out * ((1.0 - curvature_guard) + curvature_guard * curvature_gate)
    if low_rank > 0:
        out = _low_rank_delta(out, rank=min(low_rank, max(1, z.size(1))))
    out = _median_guard_delta(out, median_guard)
    if softclip > 0.0:
        out = softclip * torch.tanh(out / max(softclip, 1e-6))
    return out


def _soft_matryoshka_update(
    z: torch.Tensor,
    delta: torch.Tensor,
    cfg: CircleworldConfig,
) -> torch.Tensor:
    target = phasor_apply_delta(z, delta)
    f_bins = z.size(1)
    basis = _dct_basis(f_bins, device=z.device, dtype=z.dtype)
    coeff_cur = _basis_project(z, basis)
    coeff_tgt = _basis_project(target, basis)
    persistence, write_scale = _basis_profiles(f_bins, cfg, device=z.device, dtype=z.dtype)
    refine = ((1.0 - persistence) * write_scale).view(1, -1, 1, 1)
    coeff_next = coeff_cur + refine * (coeff_tgt - coeff_cur)
    z_next = _basis_reconstruct(coeff_next, basis)
    return phasor_normalize(z_next)


def _prefix_metrics(
    z: torch.Tensor,
    cfg: CircleworldConfig,
) -> list[dict[str, float]]:
    f_bins = z.size(1)
    basis = _dct_basis(f_bins, device=z.device, dtype=z.dtype)
    coeff = _basis_project(z, basis)
    metrics: list[dict[str, float]] = []
    for frac in cfg.prefix_fracs:
        k = _prefix_count(f_bins, frac)
        coeff_prefix = coeff.clone()
        coeff_prefix[:, k:] = 0.0
        z_prefix = phasor_normalize(_basis_reconstruct(coeff_prefix, basis))
        arc = hardy_littlewood_arc_field(z_prefix, cfg.qset, cfg.q_weights)
        promo = promotability_field(arc, persistence_momentum=cfg.persistence_momentum)
        metrics.append(
            {
                "frac": float(frac),
                "major_mass": float(arc["major_mass"].mean().item()),
                "minor_residue": float(arc["minor_residue"].mean().item()),
                "promotability": float(promo["promotability"].mean().item()),
            }
        )
    return metrics


def _clone_state_value(value: Any) -> Any:
    if torch.is_tensor(value):
        return value.detach().clone()
    if isinstance(value, dict):
        return {k: _clone_state_value(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_clone_state_value(v) for v in value]
    return value


def clone_circleworld_state(state: Any) -> Any:
    return _clone_state_value(state)


def _phase_alignment(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return (a[..., 0] * b[..., 0] + a[..., 1] * b[..., 1]).clamp(-1.0, 1.0)


def _weighted_field_mean(field: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    denom = weight.sum()
    if not bool((denom > 0).item()):
        return field.new_tensor(0.0)
    return (field * weight).sum() / denom.clamp_min(1e-8)


def _phase_only_branch_fields(
    mode0: torch.Tensor,
    mode1: torch.Tensor,
    support: torch.Tensor,
    cfg: CircleworldConfig,
) -> dict[str, torch.Tensor]:
    phase_align = ((_phase_alignment(mode0, mode1) + 1.0) * 0.5).clamp(0.0, 1.0)
    phase_wall = (1.0 - phase_align).clamp(0.0, 1.0)
    slot2_live = (support[..., 1] > cfg.slot2_support_threshold).to(mode0.dtype)
    support_sum = (support[..., 0] + support[..., 1]).clamp_min(1e-8)
    support_balance = (1.0 - (support[..., 0] - support[..., 1]).abs() / support_sum).clamp(0.0, 1.0)
    phase_only_surface = phase_wall * slot2_live
    phase_only_balanced_surface = phase_only_surface * support_balance
    return {
        "phase_align": phase_align,
        "phase_wall": phase_wall,
        "slot2_live": slot2_live,
        "support_balance": support_balance,
        "surface": phase_only_surface,
        "balanced_surface": phase_only_balanced_surface,
    }


def _safe_cosine_similarity(a: torch.Tensor, b: torch.Tensor, dim: int = -1) -> torch.Tensor:
    a_norm = a.norm(dim=dim)
    b_norm = b.norm(dim=dim)
    denom = (a_norm * b_norm).clamp_min(1e-8)
    return (a * b).sum(dim=dim) / denom


def _expand_time_to_freq(time_field: torch.Tensor, f_bins: int) -> torch.Tensor:
    return time_field.unsqueeze(1).expand(-1, f_bins, -1)


def _spread_support_time(support: torch.Tensor, kernel: int) -> torch.Tensor:
    kernel = max(1, int(kernel))
    if kernel % 2 == 0:
        kernel += 1
    if kernel <= 1:
        return support.clamp(0.0, 1.0)
    pad = kernel // 2
    bsz, f_bins, t_len, k_modes = support.shape
    x = support.permute(0, 1, 3, 2).reshape(bsz * f_bins * k_modes, 1, t_len)
    x = F.avg_pool1d(F.pad(x, (pad, pad), mode="replicate"), kernel_size=kernel, stride=1)
    if x.size(-1) != t_len:
        x = F.interpolate(x, size=t_len, mode="linear", align_corners=True)
    return x.reshape(bsz, f_bins, k_modes, t_len).permute(0, 1, 3, 2).clamp(0.0, 1.0)


def _time_gradient_field(time_field: torch.Tensor) -> torch.Tensor:
    if time_field.size(-1) <= 1:
        return torch.zeros_like(time_field)
    prev = torch.roll(time_field, shifts=1, dims=-1)
    prev[..., 0] = time_field[..., 0]
    nxt = torch.roll(time_field, shifts=-1, dims=-1)
    nxt[..., -1] = time_field[..., -1]
    return (0.5 * (time_field - prev).abs() + 0.5 * (nxt - time_field).abs()).clamp(0.0, 1.0)


def _window_bounds(t_idx: int, t_len: int, window: int) -> tuple[int, int]:
    width = max(2, int(window))
    t0 = max(0, int(t_idx) - width // 2)
    t1 = min(t_len, t0 + width)
    t0 = max(0, t1 - width)
    return t0, t1


def _window_mask_like(reference: torch.Tensor, t0: int, t1: int) -> torch.Tensor:
    mask = torch.zeros(reference.shape[:-1], device=reference.device, dtype=reference.dtype)
    mask[..., t0:t1] = 1.0
    return mask


def _masked_mean(field: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask_f = mask
    while mask_f.dim() < field.dim():
        mask_f = mask_f.unsqueeze(-1)
    weighted = field * mask_f
    denom = mask_f.sum().clamp_min(1e-8)
    return weighted.sum() / denom


def _packet_law_signature(
    packet: dict[str, torch.Tensor],
    arc: dict[str, torch.Tensor],
    promo: dict[str, torch.Tensor],
    cfg: CircleworldConfig,
) -> torch.Tensor:
    t_len = arc["major_mass"].size(-1)
    t_idx = int(packet["time_index"].item())
    t0, t1 = _window_bounds(t_idx, t_len, cfg.child_support_window)
    q_mass = packet["q_mass"].squeeze(0)
    q_mass = q_mass / q_mass.sum().clamp_min(1e-8)
    b = int(packet["batch_index"].item())
    window = slice(t0, t1)
    major = arc["major_mass"][b, window].mean()
    residue = arc["minor_residue"][b, window].mean()
    harmonic = arc["harmonic_ratio"][b, window].mean()
    persistence = promo["persistence"][b, window].mean()
    promotability = promo["promotability"][b, window].mean()
    concentration = arc["concentration"][b, window].mean()
    sharpness = arc["sharpness"][b, window].mean()
    pair_strength = arc["pair_scores"][b].abs().mean()
    attack = packet["attack_law"].mean()
    decay = packet["decay_law"].mean()
    reentry = packet["reentry_bias"].mean()
    signature = torch.cat(
        [
            q_mass,
            torch.stack(
                [
                    major,
                    residue,
                    harmonic,
                    persistence,
                    promotability,
                    concentration,
                    sharpness,
                    pair_strength,
                    attack,
                    decay,
                    reentry,
                ]
            ),
        ],
        dim=0,
    )
    return F.normalize(signature, dim=0)


def _child_support_mask(phase_state: torch.Tensor, support_window: tuple[int, int]) -> torch.Tensor:
    t0, t1 = support_window
    return _window_mask_like(phase_state, t0, t1)


def _child_operator_phase_bias(
    law_signature: torch.Tensor,
    reference: torch.Tensor,
    support_mask: torch.Tensor,
    cfg: CircleworldConfig,
    child_id: int,
) -> torch.Tensor:
    f_bins = reference.size(1)
    t_len = reference.size(2)
    q_count = len(cfg.qset)
    q_values_flat = torch.tensor(cfg.qset, device=reference.device, dtype=reference.dtype)
    q_values = q_values_flat.view(-1, 1, 1)
    q_weights_flat = law_signature[:q_count].to(device=reference.device, dtype=reference.dtype).abs()
    q_weights_flat = q_weights_flat / q_weights_flat.sum().clamp_min(1e-8)
    dominant_q = (q_weights_flat * q_values_flat).sum().clamp_min(1.0)
    freq = torch.linspace(-1.0, 1.0, steps=f_bins, device=reference.device, dtype=reference.dtype).view(1, f_bins, 1)
    time = torch.linspace(0.0, 1.0, steps=t_len, device=reference.device, dtype=reference.dtype).view(1, 1, t_len)
    tail = law_signature[q_count:].to(device=reference.device, dtype=reference.dtype)
    lifecycle_gain = tail.abs().mean().clamp(0.25, 1.25) if tail.numel() else reference.new_tensor(0.5)
    sharpness = tail[6].abs().clamp(0.0, 1.0) if tail.numel() > 6 else reference.new_tensor(0.5)
    promotability = tail[4].abs().clamp(0.0, 1.0) if tail.numel() > 4 else reference.new_tensor(0.5)
    rational_drive = torch.sin(math.pi * dominant_q * time + math.pi * (1.0 + sharpness) * freq)
    support_carrier = torch.sin(math.pi * time) * torch.cos(math.pi * (1.0 + dominant_q / 12.0) * freq)
    operator_drive = (0.70 * rational_drive + 0.30 * support_carrier).squeeze(0)
    branch_sign = -1.0 if int(child_id) % 2 else 1.0
    seed_gain = cfg.child_operator_seed_gain + cfg.child_operator_promotability_gain * promotability
    raw = branch_sign * cfg.split_seed_scale * seed_gain * lifecycle_gain * support_mask * operator_drive.unsqueeze(0)
    return _low_rank_delta(raw, rank=min(cfg.child_writeback_rank, max(4, f_bins // 4))).clamp(-3.14159, 3.14159)


def _summarize_child_event_history(child_events: list[dict[str, Any]]) -> dict[str, float]:
    if not child_events:
        return {
            "spawn_count": 0.0,
            "kill_count": 0.0,
            "writeback_count": 0.0,
            "local_ifs_count": 0.0,
            "max_age": 0.0,
        }
    spawns = sum(1.0 for ev in child_events if ev.get("event") == "spawn")
    kills = sum(1.0 for ev in child_events if ev.get("event") == "collapse")
    writes = sum(1.0 for ev in child_events if ev.get("event") == "writeback")
    local_ifs = sum(1.0 for ev in child_events if ev.get("event") == "child_local_ifs")
    max_age = max(float(ev.get("survival_age", 0.0)) for ev in child_events)
    return {
        "spawn_count": spawns,
        "kill_count": kills,
        "writeback_count": writes,
        "local_ifs_count": local_ifs,
        "max_age": max_age,
    }


def _smooth_field_2d(field: torch.Tensor, kernel: int) -> torch.Tensor:
    kernel = max(1, int(kernel))
    if kernel % 2 == 0:
        kernel += 1
    if kernel <= 1:
        return field.clamp(0.0, 1.0)
    pad = kernel // 2
    x = field.unsqueeze(1)
    x = F.avg_pool2d(F.pad(x, (pad, pad, pad, pad), mode="replicate"), kernel_size=kernel, stride=1)
    if x.size(-2) != field.size(-2) or x.size(-1) != field.size(-1):
        x = F.interpolate(x, size=field.shape[-2:], mode="bilinear", align_corners=True)
    return x.squeeze(1).clamp(0.0, 1.0)


def _relation_kernel_field(
    mode0: torch.Tensor,
    mode1: torch.Tensor,
    q_profile0: torch.Tensor,
    q_profile1: torch.Tensor,
    q_sim: torch.Tensor,
    phase_align: torch.Tensor,
    cfg: CircleworldConfig,
) -> torch.Tensor:
    perq_align = ((_safe_cosine_similarity(q_profile0, q_profile1, dim=-1) + 1.0) * 0.5).clamp(0.0, 1.0)
    perq_align = _expand_time_to_freq(perq_align, mode0.size(1))
    if cfg.branch_kernel_version == "ramanujan":
        return (0.55 * perq_align + 0.25 * q_sim + 0.20 * phase_align).clamp(0.0, 1.0)
    if cfg.branch_kernel_version == "phase_only":
        return phase_align.clamp(0.0, 1.0)
    if cfg.branch_kernel_version == "qtrace_only":
        return q_sim.clamp(0.0, 1.0)
    if cfg.branch_kernel_version == "uniform":
        return torch.ones_like(phase_align)
    return (0.35 * perq_align + 0.35 * q_sim + 0.30 * phase_align).clamp(0.0, 1.0)


def _relational_branch_attention(
    evidence0: torch.Tensor,
    evidence1: torch.Tensor,
    coherence0: torch.Tensor,
    coherence1: torch.Tensor,
    q0_strength: torch.Tensor,
    q1_strength: torch.Tensor,
    residue0: torch.Tensor,
    residue1: torch.Tensor,
    sharpness0: torch.Tensor,
    sharpness1: torch.Tensor,
    support: torch.Tensor,
    occupancy_prev: torch.Tensor,
    world_context: torch.Tensor,
    branch_pos: torch.Tensor,
    relation_kernel: torch.Tensor,
    cfg: CircleworldConfig,
) -> dict[str, torch.Tensor]:
    feat0 = torch.stack(
        [
            evidence0,
            coherence0,
            q0_strength,
            (1.0 - residue0).clamp(0.0, 1.0),
            (1.0 - sharpness0).clamp(0.0, 1.0),
            support[..., 0],
            occupancy_prev[..., 0],
            world_context,
        ],
        dim=-1,
    )
    feat1 = torch.stack(
        [
            evidence1,
            coherence1,
            q1_strength,
            (1.0 - residue1).clamp(0.0, 1.0),
            (1.0 - sharpness1).clamp(0.0, 1.0),
            support[..., 1],
            occupancy_prev[..., 1],
            world_context,
        ],
        dim=-1,
    )
    feat = torch.stack([feat0, feat1], dim=3)
    query = torch.stack(
        [
            feat[..., 0] + 0.45 * feat[..., 1],
            feat[..., 2] + 0.25 * feat[..., 6],
            feat[..., 7] + 0.20 * feat[..., 3] - 0.15 * (1.0 - feat[..., 4]),
            feat[..., 5],
        ],
        dim=-1,
    )
    key = torch.stack(
        [
            feat[..., 1] + 0.35 * feat[..., 0],
            feat[..., 2] + 0.20 * feat[..., 5],
            feat[..., 7] + 0.20 * feat[..., 3],
            feat[..., 6],
        ],
        dim=-1,
    )
    value = torch.stack(
        [
            feat[..., 0],
            feat[..., 1],
            feat[..., 2],
            feat[..., 3],
            feat[..., 5],
            feat[..., 6],
            feat[..., 7],
        ],
        dim=-1,
    )
    query = F.normalize(query, dim=-1)
    key = F.normalize(key, dim=-1)
    attn_logits = torch.einsum("bftmd,bftnd->bftmn", query, key) / math.sqrt(float(query.size(-1)))
    kernel_gate = (cfg.relation_attention_gain * relation_kernel * branch_pos).clamp(0.0, 1.0)
    forward_bias = cfg.relation_attention_sharpness * kernel_gate * (1.0 - support[..., 1])
    back_bias = 0.45 * cfg.relation_attention_sharpness * kernel_gate * support[..., 1]
    self0 = 0.20 * cfg.relation_attention_sharpness * (1.0 - kernel_gate)
    self1 = 0.15 * cfg.relation_attention_sharpness * (1.0 - kernel_gate)
    bias = attn_logits.new_zeros(attn_logits.shape)
    bias[..., 0, 0] = self0
    bias[..., 1, 1] = self1
    bias[..., 1, 0] = forward_bias
    bias[..., 0, 1] = back_bias
    attn = torch.softmax(attn_logits + bias, dim=-1)
    message = torch.einsum("bftmn,bftnd->bftmd", attn, value)
    handoff_drive = (
        cfg.relation_value_gain
        * kernel_gate
        * attn[..., 1, 0]
        * (0.40 + 0.35 * message[..., 1, 1] + 0.25 * message[..., 1, 6])
    ).clamp(0.0, 1.5)
    return {
        "attn": attn,
        "message": message,
        "handoff_drive": handoff_drive,
        "kernel_gate": kernel_gate,
    }


def _compress_q_trace(per_q: torch.Tensor, rank: int) -> torch.Tensor:
    rank = max(1, min(int(rank), per_q.size(1)))
    if rank == per_q.size(1):
        return per_q
    return F.interpolate(per_q.permute(0, 2, 1), size=rank, mode="linear", align_corners=True).permute(0, 2, 1)


def _mode_to_freq_profile(per_q: torch.Tensor, f_bins: int) -> torch.Tensor:
    q_mass = per_q.abs().mean(dim=2)
    q_mass = q_mass / q_mass.sum(dim=1, keepdim=True).clamp_min(1e-8)
    return F.interpolate(q_mass.unsqueeze(1), size=f_bins, mode="linear", align_corners=True).squeeze(1)


def _harmonic_inharmonic_energy_timewise(
    z: torch.Tensor,
    rat_ratios: tuple[tuple[int, int], ...] = ((1, 1), (2, 1), (3, 2), (4, 3)),
    harmonic_qs: tuple[int, ...] = (2, 3, 4, 6, 8, 12),
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Circleworld-local variant that assumes z is [B, F, T, 2] and always returns [B, T].
    This avoids the orientation guessing in the shared helper, which breaks on real STFTs
    where T > F.
    """
    if z.dim() != 4 or z.size(-1) != 2:
        raise ValueError("z must have shape [B, F, T, 2]")

    zt = z.transpose(1, 2)  # [B, T, F, 2]
    bsz, t_len, _, _ = zt.shape
    zi = zt.unsqueeze(3)  # [B, T, F, 1, 2]
    zj = zt.unsqueeze(2)  # [B, T, 1, F, 2]
    zj_conj = torch.stack([zj[..., 0], -zj[..., 1]], dim=-1)

    r_re = zi[..., 0] * zj_conj[..., 0] - zi[..., 1] * zj_conj[..., 1]
    r_im = zi[..., 0] * zj_conj[..., 1] + zi[..., 1] * zj_conj[..., 0]
    theta = torch.atan2(r_im, r_re)  # [B, T, F, F]

    all_qs = {int(r[0]) for r in rat_ratios} | {int(r[1]) for r in rat_ratios}
    h_energy = torch.zeros((bsz, t_len), device=z.device, dtype=z.dtype)
    i_energy = torch.zeros((bsz, t_len), device=z.device, dtype=z.dtype)
    for q in all_qs:
        coh = torch.cos(float(q) * theta).mean(dim=(2, 3))  # [B, T]
        if q in harmonic_qs:
            h_energy = h_energy + coh
        else:
            i_energy = i_energy + coh
    return h_energy, i_energy

def hardy_littlewood_arc_field(
    z: torch.Tensor,
    qset: tuple[int, ...],
    q_weights: tuple[float, ...],
) -> dict[str, torch.Tensor]:
    """
    Returns a graded circle-method style field over time.

    z shape: [B, Q, T, 2]
    """
    if z.dim() != 4 or z.size(-1) != 2:
        raise ValueError("z must have shape [B, Q, T, 2]")

    device = z.device
    dtype = z.dtype
    q_weights_t = torch.tensor(q_weights, dtype=dtype, device=device)

    per_q = []
    pair_scores = []
    for b in range(z.size(0)):
        zb = z[b]  # [Q, T, 2]
        score_matrix = ramanujan_relative_score_phasor(
            zb, qs=qset, q_weights=q_weights_t
        )
        pair_scores.append(score_matrix)
        rel_theta = torch.atan2(
            zb[..., 1].unsqueeze(1) * zb[..., 0].unsqueeze(0)
            - zb[..., 0].unsqueeze(1) * zb[..., 1].unsqueeze(0),
            zb[..., 0].unsqueeze(1) * zb[..., 0].unsqueeze(0)
            + zb[..., 1].unsqueeze(1) * zb[..., 1].unsqueeze(0)
            + 1e-12,
        )
        q_scores = []
        for q, w in zip(qset, q_weights):
            q_scores.append(float(w) * torch.cos(float(q) * rel_theta).mean(dim=(0, 1)))
        per_q.append(torch.stack(q_scores, dim=0))
    per_q = torch.stack(per_q, dim=0)  # [B, K, T]
    pair_scores = torch.stack(pair_scores, dim=0)

    weights = q_weights_t.view(1, -1, 1)
    weighted = per_q * weights
    major_mass = weighted.clamp_min(0.0).sum(dim=1) / weights.sum().clamp_min(1e-8)
    major_mass = major_mass.clamp(0.0, 1.0)
    concentration = per_q.abs().amax(dim=1)
    sharpness = per_q.std(dim=1, unbiased=False)
    harmonic_energy, inharmonic_energy = _harmonic_inharmonic_energy_timewise(
        z, [(1, 1), (2, 1), (3, 2), (4, 3)], harmonic_qs=(2, 3, 4, 6, 8, 12)
    )
    total_energy = (harmonic_energy + inharmonic_energy).clamp_min(1e-8)
    minor_residue = (inharmonic_energy / total_energy).clamp(0.0, 1.0)
    harmonic_ratio = (harmonic_energy / total_energy).clamp(0.0, 1.0)

    return {
        "per_q": per_q,
        "pair_scores": pair_scores,
        "major_mass": major_mass,
        "minor_residue": minor_residue,
        "harmonic_ratio": harmonic_ratio,
        "harmonic_energy": harmonic_energy,
        "inharmonic_energy": inharmonic_energy,
        "concentration": concentration,
        "sharpness": sharpness,
    }


def promotability_field(
    arc: dict[str, torch.Tensor],
    persistence_momentum: float = 0.6,
) -> dict[str, torch.Tensor]:
    major = arc["major_mass"]
    residue = arc["minor_residue"]
    sharpness = arc["sharpness"]
    concentration = arc["concentration"]

    prev = torch.roll(major, shifts=1, dims=-1)
    prev[..., 0] = major[..., 0]
    persistence = 1.0 - (major - prev).abs()
    persistence = persistence.clamp(0.0, 1.0)
    persistence = (
        persistence_momentum * persistence
        + (1.0 - persistence_momentum) * major
    ).clamp(0.0, 1.0)

    promotability = (
        0.45 * major
        + 0.25 * concentration.clamp(0.0, 1.0)
        + 0.20 * persistence
        + 0.10 * sharpness.clamp(0.0, 1.0)
        - 0.30 * residue
    )
    promotability = promotability.clamp(0.0, 1.0)
    return {
        "persistence": persistence,
        "promotability": promotability,
    }


def branch_law_feature_fields_from_state(state: dict[str, torch.Tensor], cfg: CircleworldConfig) -> dict[str, torch.Tensor]:
    phase_modes = state["phase_modes"]
    support = state["mode_support"].clamp(0.0, 1.0)
    q_trace = state["mode_q_trace"]
    mode_coherence = state.get("mode_coherence", support).clamp(0.0, 1.0)
    mode0 = phase_modes[..., 0, :]
    mode1 = phase_modes[..., 1, :]
    arc0 = hardy_littlewood_arc_field(mode0, cfg.qset, cfg.q_weights)
    arc1 = hardy_littlewood_arc_field(mode1, cfg.qset, cfg.q_weights)
    phase_fields = _phase_only_branch_fields(mode0, mode1, support, cfg)
    q0 = q_trace[..., 0, :]
    q1 = q_trace[..., 1, :]
    q_sim = ((_safe_cosine_similarity(q0, q1, dim=-1) + 1.0) * 0.5).clamp(0.0, 1.0)
    q_disagreement = (q0 - q1).abs().mean(dim=-1).clamp(0.0, 1.0)
    f_bins = mode0.size(1)
    major_arc_compatibility = (
        1.0 - _expand_time_to_freq((arc0["major_mass"] - arc1["major_mass"]).abs(), f_bins)
    ).clamp(0.0, 1.0)
    minor_residue_disagreement = _expand_time_to_freq(
        (arc0["minor_residue"] - arc1["minor_residue"]).abs(),
        f_bins,
    ).clamp(0.0, 1.0)
    residue_mix = 0.5 * (
        _expand_time_to_freq(arc0["minor_residue"], f_bins)
        + _expand_time_to_freq(arc1["minor_residue"], f_bins)
    )
    sharpness_mix = 0.5 * (
        _expand_time_to_freq(arc0["sharpness"], f_bins)
        + _expand_time_to_freq(arc1["sharpness"], f_bins)
    )
    branch_defect = (
        cfg.defect_phase_gain * phase_fields["phase_wall"]
        + cfg.defect_q_gain * q_disagreement
        + cfg.defect_residue_gain * residue_mix
        + cfg.defect_sharpness_gain * sharpness_mix
    ).clamp(0.0, 1.0)
    inherited_rational_history = (0.5 * (q0.abs().mean(dim=-1) + q1.abs().mean(dim=-1))).clamp(0.0, 1.0)
    return {
        "phase_agreement": phase_fields["phase_align"],
        "q_trace_similarity": q_sim,
        "major_arc_compatibility": major_arc_compatibility,
        "minor_residue_disagreement": minor_residue_disagreement,
        "support_overlap": (support[..., 0] * support[..., 1]).clamp(0.0, 1.0),
        "branch_defect": branch_defect,
        "mode0_coherence": mode_coherence[..., 0],
        "mode1_coherence": mode_coherence[..., 1],
        "inherited_rational_history": inherited_rational_history,
        "mode0_support": support[..., 0],
        "mode1_support": support[..., 1],
        "q_trace_disagreement": q_disagreement,
    }


def branch_law_feature_tensor(feature_fields: dict[str, torch.Tensor]) -> torch.Tensor:
    return torch.stack([feature_fields[name] for name in BRANCH_LAW_FEATURE_NAMES], dim=-1)


def learned_branch_law_targets_from_features(
    features: torch.Tensor | dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    fields = (
        features
        if isinstance(features, dict)
        else {name: features[..., idx] for idx, name in enumerate(BRANCH_LAW_FEATURE_NAMES)}
    )
    phase = fields["phase_agreement"].clamp(0.0, 1.0)
    q_sim = fields["q_trace_similarity"].clamp(0.0, 1.0)
    major = fields["major_arc_compatibility"].clamp(0.0, 1.0)
    residue_gap = fields["minor_residue_disagreement"].clamp(0.0, 1.0)
    support0 = fields["mode0_support"].clamp(0.0, 1.0)
    support1 = fields["mode1_support"].clamp(0.0, 1.0)
    coh0 = fields["mode0_coherence"].clamp(0.0, 1.0)
    coh1 = fields["mode1_coherence"].clamp(0.0, 1.0)
    defect = fields["branch_defect"].clamp(0.0, 1.0)
    rational_history = fields["inherited_rational_history"].clamp(0.0, 1.0)
    q_disagreement = fields["q_trace_disagreement"].clamp(0.0, 1.0)

    support_pair = (support0 * support1).clamp_min(0.0).sqrt()
    coherence_pair = 0.5 * (coh0 + coh1)
    incompatibility = (0.50 * (1.0 - phase) + 0.30 * q_disagreement + 0.20 * residue_gap).clamp(0.0, 1.0)
    lawfulness = (coherence_pair * (0.5 + 0.5 * rational_history)).clamp(0.0, 1.0)
    merge_drive = (phase * q_sim * major * (1.0 - residue_gap) * support_pair).clamp(0.0, 1.0)
    collapse_pressure = (
        0.35 * (1.0 - coh1)
        + 0.25 * residue_gap
        + 0.25 * defect
        + 0.15 * (1.0 - support1)
        - 0.20 * rational_history
    ).clamp(0.0, 1.0)
    coexistence_drive = (lawfulness * support_pair * incompatibility * (1.0 - 0.5 * merge_drive)).clamp(0.0, 1.0)
    split_drive = (lawfulness * support0 * incompatibility * (0.5 + 0.5 * defect) * (1.0 - merge_drive)).clamp(0.0, 1.0)
    support_delta0 = (coh0 * major - 0.5 * defect - support0).clamp(-1.0, 1.0)
    support_delta1 = (coh1 * (0.5 + 0.5 * incompatibility) - collapse_pressure - support1).clamp(-1.0, 1.0)
    survival_delta0 = (2.0 * coh0 * support0 - 1.0).clamp(-1.0, 1.0)
    survival_delta1 = (2.0 * coh1 * support1 * (1.0 - collapse_pressure) - 1.0).clamp(-1.0, 1.0)
    coherence_target = (lawfulness * (0.65 + 0.35 * major) * (1.0 - 0.35 * residue_gap)).clamp(0.0, 1.0)
    qtrace_inheritance_mix = (q_sim * major * (0.5 + 0.5 * rational_history)).clamp(0.0, 1.0)
    return {
        "split_drive": split_drive,
        "coexistence_drive": coexistence_drive,
        "merge_drive": merge_drive,
        "survival_delta0": survival_delta0,
        "survival_delta1": survival_delta1,
        "collapse_pressure": collapse_pressure,
        "support_delta0": support_delta0,
        "support_delta1": support_delta1,
        "coherence_target": coherence_target,
        "qtrace_inheritance_mix": qtrace_inheritance_mix,
    }


def learned_branch_law_output_tensor(outputs: dict[str, torch.Tensor]) -> torch.Tensor:
    return torch.stack([outputs[name] for name in LEARNED_BRANCH_LAW_OUTPUT_NAMES], dim=-1)


def build_promoted_packets(
    z: torch.Tensor,
    arc: dict[str, torch.Tensor],
    promo: dict[str, torch.Tensor],
    cfg: CircleworldConfig,
) -> list[dict[str, torch.Tensor]]:
    major = arc["major_mass"]
    promo_field = promo["promotability"]
    per_q = arc["per_q"]
    bsz, q_bins, t_len, _ = z.shape
    packets: list[dict[str, torch.Tensor]] = []

    attack_window = min(cfg.attack_window, t_len)
    for b in range(bsz):
        top_vals, top_idx = torch.topk(
            promo_field[b], k=min(cfg.max_promotions, t_len), largest=True
        )
        for score, t_idx in zip(top_vals.tolist(), top_idx.tolist()):
            if score < cfg.promotion_threshold:
                continue
            t0 = max(0, t_idx - attack_window // 2)
            t1 = min(t_len, t0 + attack_window)
            seed_mean = z[b : b + 1, :, t0:t1].mean(dim=2, keepdim=True)
            seed_center = phasor_normalize(z[b : b + 1, :, t_idx : t_idx + 1])
            seed_norm = torch.linalg.vector_norm(seed_mean, dim=-1, keepdim=True)
            seed = torch.where(seed_norm > 1.0e-4, phasor_normalize(seed_mean), seed_center)
            q_mass = per_q[b : b + 1, :, t0:t1].mean(dim=2)
            q_mass = torch.softmax(q_mass, dim=-1)
            attack = (major[b, min(t_len - 1, t1 - 1)] - major[b, t0]).view(1, 1)
            decay = (major[b, -1] - major[b, t_idx]).view(1, 1)
            packets.append(
                {
                    "batch_index": torch.tensor([b], device=z.device),
                    "time_index": torch.tensor([t_idx], device=z.device),
                    "score": torch.tensor([score], dtype=z.dtype, device=z.device),
                    "seed_state": seed,
                    "q_mass": q_mass,
                    "attack_law": attack,
                    "decay_law": decay,
                    "reentry_bias": major[b, t_idx].view(1, 1),
                }
            )
    return packets


def promote_law_packets(
    packets: list[dict[str, torch.Tensor]],
    arc: dict[str, torch.Tensor],
    promo: dict[str, torch.Tensor],
    cfg: CircleworldConfig,
    depth_index: int,
) -> list[dict[str, torch.Tensor]]:
    if not packets:
        return []
    law_packets: list[dict[str, torch.Tensor]] = []
    t_len = arc["major_mass"].size(-1)
    for packet in packets:
        score = float(packet["score"].item())
        if score < cfg.law_packet_min_score:
            continue
        b = int(packet["batch_index"].item())
        t_idx = int(packet["time_index"].item())
        t0 = max(0, t_idx - cfg.attack_window // 2)
        t1 = min(t_len, t0 + max(2, cfg.attack_window))
        window = slice(t0, t1)
        q_mass = packet["q_mass"].squeeze(0)
        q_mass = q_mass / q_mass.sum().clamp_min(1e-8)
        major = arc["major_mass"][b, window].mean()
        residue = arc["minor_residue"][b, window].mean()
        harmonic = arc["harmonic_ratio"][b, window].mean()
        persistence = promo["persistence"][b, window].mean()
        promotability = promo["promotability"][b, window].mean()
        concentration = arc["concentration"][b, window].mean()
        sharpness = arc["sharpness"][b, window].mean()
        pair_strength = arc["pair_scores"][b].abs().mean()
        attack = packet["attack_law"].mean()
        decay = packet["decay_law"].mean()
        reentry = packet["reentry_bias"].mean()
        signature = torch.cat(
            [
                q_mass,
                torch.stack(
                    [
                        major,
                        residue,
                        harmonic,
                        persistence,
                        promotability,
                        concentration,
                        sharpness,
                        pair_strength,
                        attack,
                        decay,
                        reentry,
                    ]
                ),
            ],
            dim=0,
        )
        signature = F.normalize(signature, dim=0)
        law_packets.append(
            {
                **packet,
                "depth_index": torch.tensor([depth_index], device=signature.device),
                "major_mass": major.view(1),
                "minor_residue": residue.view(1),
                "harmonic_ratio": harmonic.view(1),
                "persistence": persistence.view(1),
                "promotability": promotability.view(1),
                "concentration": concentration.view(1),
                "sharpness": sharpness.view(1),
                "pair_strength": pair_strength.view(1),
                "law_signature": signature,
                "law_window": torch.tensor([t0, t1], device=signature.device),
            }
        )
    return law_packets


def build_law_token_library(
    law_packets: list[dict[str, torch.Tensor]],
    cfg: CircleworldConfig,
) -> dict[str, Any]:
    if not law_packets:
        return {"families": [], "num_law_packets": 0, "num_families": 0, "mean_family_size": 0.0}
    families: list[dict[str, Any]] = []
    for packet in law_packets:
        sig = packet["law_signature"]
        best_idx = -1
        best_sim = -1.0
        for idx, family in enumerate(families):
            proto = family["prototype_signature"]
            sim = float(F.cosine_similarity(sig, proto, dim=0).item())
            if sim > best_sim:
                best_sim = sim
                best_idx = idx
        if best_idx >= 0 and best_sim >= cfg.law_packet_merge_threshold:
            family = families[best_idx]
            count = family["count"] + 1
            family["count"] = count
            weight_prev = float(count - 1) / float(count)
            weight_new = 1.0 / float(count)
            for key in (
                "prototype_signature",
                "prototype_q_mass",
                "mean_score",
                "mean_major_mass",
                "mean_minor_residue",
                "mean_harmonic_ratio",
                "mean_persistence",
                "mean_promotability",
                "mean_pair_strength",
            ):
                if key in ("prototype_signature", "prototype_q_mass"):
                    family[key] = F.normalize(weight_prev * family[key] + weight_new * packet["law_signature" if key == "prototype_signature" else "q_mass"].squeeze(0), dim=0)
                else:
                    src = {
                        "mean_score": packet["score"].mean(),
                        "mean_major_mass": packet["major_mass"].mean(),
                        "mean_minor_residue": packet["minor_residue"].mean(),
                        "mean_harmonic_ratio": packet["harmonic_ratio"].mean(),
                        "mean_persistence": packet["persistence"].mean(),
                        "mean_promotability": packet["promotability"].mean(),
                        "mean_pair_strength": packet["pair_strength"].mean(),
                    }[key]
                    family[key] = weight_prev * family[key] + weight_new * src
            family["members"].append(
                {
                    "depth_index": int(packet["depth_index"].item()),
                    "time_index": int(packet["time_index"].item()),
                    "score": float(packet["score"].item()),
                }
            )
            family["best_similarity"] = max(float(family.get("best_similarity", 0.0)), best_sim)
        else:
            families.append(
                {
                    "family_index": len(families),
                    "count": 1,
                    "prototype_signature": packet["law_signature"].clone(),
                    "prototype_q_mass": packet["q_mass"].squeeze(0).clone(),
                    "mean_score": packet["score"].mean().clone(),
                    "mean_major_mass": packet["major_mass"].mean().clone(),
                    "mean_minor_residue": packet["minor_residue"].mean().clone(),
                    "mean_harmonic_ratio": packet["harmonic_ratio"].mean().clone(),
                    "mean_persistence": packet["persistence"].mean().clone(),
                    "mean_promotability": packet["promotability"].mean().clone(),
                    "mean_pair_strength": packet["pair_strength"].mean().clone(),
                    "members": [
                        {
                            "depth_index": int(packet["depth_index"].item()),
                            "time_index": int(packet["time_index"].item()),
                            "score": float(packet["score"].item()),
                        }
                    ],
                    "best_similarity": 1.0,
                }
            )
    families.sort(key=lambda fam: (fam["count"], float(fam["mean_score"].item() if torch.is_tensor(fam["mean_score"]) else fam["mean_score"])), reverse=True)
    for idx, family in enumerate(families):
        family["family_index"] = idx
    trimmed = families[: max(1, cfg.law_packet_topk_families)]
    mean_family_size = float(sum(f["count"] for f in trimmed) / max(1, len(trimmed)))
    return {
        "families": trimmed,
        "num_law_packets": len(law_packets),
        "num_families": len(trimmed),
        "mean_family_size": mean_family_size,
    }


def serialize_law_token_library(library: dict[str, Any]) -> dict[str, Any]:
    out = {
        "num_law_packets": int(library.get("num_law_packets", 0)),
        "num_families": int(library.get("num_families", 0)),
        "mean_family_size": float(library.get("mean_family_size", 0.0)),
        "families": [],
    }
    for family in library.get("families", []):
        out["families"].append(
            {
                "family_index": int(family["family_index"]),
                "count": int(family["count"]),
                "prototype_signature": family["prototype_signature"].detach().cpu().tolist(),
                "prototype_q_mass": family["prototype_q_mass"].detach().cpu().tolist(),
                "mean_score": float(family["mean_score"].item() if torch.is_tensor(family["mean_score"]) else family["mean_score"]),
                "mean_major_mass": float(family["mean_major_mass"].item() if torch.is_tensor(family["mean_major_mass"]) else family["mean_major_mass"]),
                "mean_minor_residue": float(family["mean_minor_residue"].item() if torch.is_tensor(family["mean_minor_residue"]) else family["mean_minor_residue"]),
                "mean_harmonic_ratio": float(family["mean_harmonic_ratio"].item() if torch.is_tensor(family["mean_harmonic_ratio"]) else family["mean_harmonic_ratio"]),
                "mean_persistence": float(family["mean_persistence"].item() if torch.is_tensor(family["mean_persistence"]) else family["mean_persistence"]),
                "mean_promotability": float(family["mean_promotability"].item() if torch.is_tensor(family["mean_promotability"]) else family["mean_promotability"]),
                "mean_pair_strength": float(family["mean_pair_strength"].item() if torch.is_tensor(family["mean_pair_strength"]) else family["mean_pair_strength"]),
                "best_similarity": float(family.get("best_similarity", 0.0)),
                "members": list(family["members"]),
            }
        )
    return out


def apply_passive_packets(
    z: torch.Tensor,
    packets: list[dict[str, torch.Tensor]],
    gain: float,
) -> torch.Tensor:
    if not packets:
        return z.clone()
    out = z.clone()
    for packet in packets:
        b = int(packet["batch_index"].item())
        seed = packet["seed_state"]
        score = packet["score"].view(1, 1, 1, 1)
        seed_expanded = seed.expand(1, seed.size(1), out.size(2), seed.size(3))
        out[b : b + 1] = phasor_normalize(
            (1.0 - gain * score) * out[b : b + 1] + (gain * score) * seed_expanded
        )
    return out


def apply_active_packets(
    z: torch.Tensor,
    packets: list[dict[str, torch.Tensor]],
    cfg: CircleworldConfig,
) -> torch.Tensor:
    if not packets:
        return z.clone()
    out = z.clone()
    qset = torch.tensor(cfg.qset, dtype=z.dtype, device=z.device).view(1, -1)
    for packet in packets:
        b = int(packet["batch_index"].item())
        q_mass = packet["q_mass"]
        score = packet["score"].view(1, 1)
        attack = packet["attack_law"]
        decay = packet["decay_law"]
        reentry = packet["reentry_bias"]

        q_profile = q_mass * (qset / qset.max().clamp_min(1.0))
        freq_profile = F.interpolate(
            q_profile.unsqueeze(1),
            size=out.size(1),
            mode="linear",
            align_corners=True,
        ).squeeze(1)
        phase_bias = cfg.child_law_gain * score * freq_profile
        phase_bias = phase_bias + 0.5 * attack - 0.25 * decay + 0.1 * reentry
        delta = phase_bias.unsqueeze(-1).expand(-1, -1, out.size(2))
        delta = _precondition_phase_delta(out[b : b + 1], delta, cfg)
        if cfg.soft_matryoshka_enabled:
            delta = _low_rank_delta(delta, rank=cfg.matryoshka_rank)
            out[b : b + 1] = _soft_matryoshka_update(out[b : b + 1], delta, cfg)
        else:
            out[b : b + 1] = phasor_apply_delta(out[b : b + 1], delta)
    return out


def child_circleworld_step(
    child: dict[str, Any],
    cfg: CircleworldConfig,
    parent_mode0: torch.Tensor,
    parent_mode1: torch.Tensor,
    *,
    parent_coupled: bool = True,
) -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
    phase = phasor_normalize(child["phase_state"])
    previous_phase = phase
    support_window = tuple(child.get("support_window", (0, phase.size(2))))
    window_mask = _child_support_mask(phase, support_window).clamp(0.0, 1.0)
    support_mask = torch.maximum(child.get("mode_support", window_mask).clamp(0.0, 1.0), 0.50 * window_mask)
    previous_coherence = child.get("mode_coherence", support_mask)
    if not torch.is_tensor(previous_coherence) or previous_coherence.shape != support_mask.shape:
        previous_coherence = torch.zeros_like(support_mask)
    previous_coherence = previous_coherence.clamp(0.0, 1.0)
    packet_count = 0.0
    phase_delta_terms: list[torch.Tensor] = []
    local_steps = max(1, int(cfg.child_local_steps))

    for _ in range(local_steps):
        before = phase
        arc = hardy_littlewood_arc_field(phase, cfg.qset, cfg.q_weights)
        promo = promotability_field(arc, persistence_momentum=cfg.persistence_momentum)
        packets = build_promoted_packets(phase, arc, promo, cfg)
        transported = apply_active_packets(phase, packets, cfg)
        packet_count += float(len(packets))
        if cfg.child_local_support_only:
            outside_anchor = parent_mode1 if parent_coupled else phase
            phase = phasor_normalize(
                support_mask.unsqueeze(-1) * transported
                + (1.0 - support_mask).unsqueeze(-1) * outside_anchor
            )
        else:
            phase = phasor_normalize(transported)
        phase_delta_terms.append(_masked_mean(_phasor_phase_delta(before, phase).abs(), support_mask))

    child_arc = hardy_littlewood_arc_field(phase, cfg.qset, cfg.q_weights)
    child_promo = promotability_field(child_arc, persistence_momentum=cfg.persistence_momentum)
    child_q = _compress_q_trace(child_arc["per_q"], cfg.q_trace_rank)
    child_q_expanded = child_q.permute(0, 2, 1).unsqueeze(1).expand(-1, phase.size(1), -1, -1)
    q_strength = _expand_time_to_freq(child_q.abs().mean(dim=1), phase.size(1)).clamp(0.0, 1.0)
    support_evidence = (
        0.45 * _expand_time_to_freq(child_promo["promotability"], phase.size(1))
        + 0.35 * _expand_time_to_freq(child_arc["major_mass"], phase.size(1))
        + 0.20 * q_strength
        - 0.25 * _expand_time_to_freq(child_arc["minor_residue"], phase.size(1))
    ).clamp(0.0, 1.0)
    support_target = torch.maximum(0.50 * support_mask, window_mask * support_evidence)
    support_next = (
        (1.0 - cfg.child_support_decay) * support_mask
        + cfg.child_support_decay * support_target
    ).clamp(0.0, 1.0)
    coherence_raw = (
        cfg.child_survival_coherence_weight * _expand_time_to_freq(child_arc["major_mass"], phase.size(1))
        + cfg.child_survival_qtrace_weight * q_strength
        - cfg.child_survival_residue_penalty * _expand_time_to_freq(child_arc["minor_residue"], phase.size(1))
    ).clamp(0.0, 1.0)
    parent_phase_delta = _phasor_phase_delta(phase, parent_mode1).abs()
    causal_gate = torch.ones_like(coherence_raw)
    if cfg.child_local_coherence_causal_gate_enabled:
        min_delta = max(0.0, float(cfg.child_local_coherence_causal_min_delta))
        full_delta = max(min_delta + 1.0e-6, float(cfg.child_local_coherence_causal_full_delta))
        causal_gate = ((parent_phase_delta - min_delta) / (full_delta - min_delta)).clamp(0.0, 1.0)
        gate_floor = max(0.0, min(1.0, float(cfg.child_local_coherence_causal_gate_floor)))
        if gate_floor > 0.0:
            causal_gate = torch.maximum(causal_gate, gate_floor * (support_next > 0.0).to(causal_gate.dtype))
        causal_gate = (causal_gate * torch.maximum(window_mask, support_next)).clamp(0.0, 1.0)
    coherence_next = coherence_raw
    retention_delta = torch.zeros_like(coherence_raw)
    floor_delta = torch.zeros_like(coherence_raw)
    retention_gate_loss = torch.zeros_like(coherence_raw)
    floor_gate_loss = torch.zeros_like(coherence_raw)
    if cfg.child_local_coherence_retention_enabled:
        retention_mix = float(max(0.0, min(1.0, cfg.child_local_coherence_retention_mix)))
        if retention_mix > 0.0:
            retention_target = previous_coherence * (0.35 + 0.65 * support_next)
            retained_ungated = ((1.0 - retention_mix) * coherence_raw + retention_mix * retention_target).clamp(
                0.0, 1.0
            )
            retention_mix_field = (retention_mix * causal_gate).clamp(0.0, 1.0)
            retained = (
                (1.0 - retention_mix_field) * coherence_raw
                + retention_mix_field * retention_target
            ).clamp(0.0, 1.0)
            coherence_next = torch.maximum(coherence_next, retained)
            retention_delta = (coherence_next - coherence_raw).clamp_min(0.0)
            retention_gate_loss = (torch.maximum(coherence_raw, retained_ungated) - coherence_next).clamp_min(0.0)
        floor_value = float(max(0.0, min(1.0, cfg.child_local_coherence_floor)))
        if floor_value > 0.0:
            floor_gate = (support_next >= float(cfg.child_local_coherence_floor_support)).to(support_next.dtype)
            floor_target = floor_value * floor_gate * causal_gate
            floor_target_ungated = floor_value * floor_gate
            floored = torch.maximum(coherence_next, floor_target)
            floor_delta = (floored - coherence_next).clamp_min(0.0)
            floor_gate_loss = (torch.maximum(coherence_next, floor_target_ungated) - floored).clamp_min(0.0)
            coherence_next = floored
        coherence_next = (coherence_next * torch.maximum(window_mask, support_next)).clamp(0.0, 1.0)
    q_prev = child.get("q_trace")
    if not torch.is_tensor(q_prev) or q_prev.shape != child_q_expanded.shape:
        q_prev = torch.zeros_like(child_q_expanded)
    child["phase_state"] = phasor_normalize(phase)
    child["mode_support"] = support_next
    child["mode_coherence"] = coherence_next
    child["q_trace"] = cfg.qtrace_momentum * q_prev + (1.0 - cfg.qtrace_momentum) * (
        child_q_expanded * support_next.unsqueeze(-1)
    )
    child["survival_age"] = int(child.get("survival_age", 0)) + local_steps
    child["child_local_ifs_steps"] = int(child.get("child_local_ifs_steps", 0)) + local_steps
    phase_delta = _masked_mean(_phasor_phase_delta(previous_phase, child["phase_state"]).abs(), support_next)
    metrics = {
        "child_local_ifs_step_count": phase.new_tensor(float(local_steps)),
        "child_local_ifs_packet_count": phase.new_tensor(packet_count),
        "child_local_ifs_phase_delta": phase_delta,
        "child_local_ifs_support_mean": support_next.mean(),
        "child_local_ifs_coherence_mean": _masked_mean(coherence_next, support_next),
        "child_local_ifs_coherence_raw_mean": _masked_mean(coherence_raw, support_next),
        "child_local_ifs_coherence_retention_delta": _masked_mean(retention_delta, support_next),
        "child_local_ifs_coherence_floor_delta": _masked_mean(floor_delta, support_next),
        "child_local_ifs_parent_phase_delta": _masked_mean(parent_phase_delta, support_next),
        "child_local_ifs_causal_gate_mean": _masked_mean(causal_gate, support_next),
        "child_local_ifs_causal_retention_loss": _masked_mean(retention_gate_loss, support_next),
        "child_local_ifs_causal_floor_loss": _masked_mean(floor_gate_loss, support_next),
        "child_local_ifs_parent_coupled": phase.new_tensor(1.0 if parent_coupled else 0.0),
        "child_local_ifs_step_phase_delta": (
            torch.stack(phase_delta_terms).mean() if phase_delta_terms else phase.new_tensor(0.0)
        ),
    }
    return child, metrics


def _init_multimode_state(z: torch.Tensor, cfg: CircleworldConfig) -> dict[str, torch.Tensor]:
    if cfg.num_modes != 2:
        raise ValueError("native_multimode v1 requires num_modes=2")
    base = phasor_normalize(z)
    f_bins = base.size(1)
    t_len = base.size(2)
    freq_basis = torch.linspace(-1.0, 1.0, steps=f_bins, device=base.device, dtype=base.dtype).view(1, f_bins, 1)
    time_basis = torch.sin(torch.linspace(0.0, torch.pi, steps=t_len, device=base.device, dtype=base.dtype)).view(1, 1, t_len)
    seed_delta = cfg.split_seed_scale * freq_basis * time_basis
    seed_delta = _low_rank_delta(seed_delta, rank=min(cfg.matryoshka_rank, max(4, f_bins // 4)))
    slot1 = phasor_apply_delta(base, seed_delta)

    phase_modes = torch.stack([base, slot1], dim=3)
    mode_logits = torch.full((base.size(0), f_bins, t_len, 2), float(cfg.dormant_logit), device=base.device, dtype=base.dtype)
    mode_logits[..., 0] = 2.5
    mode_support = torch.full_like(mode_logits, float(cfg.dormant_support))
    mode_support[..., 0] = 1.0
    mode_coherence = torch.full_like(mode_logits, float(cfg.dormant_support))
    mode_coherence[..., 0] = 1.0
    mode_q_trace = torch.zeros((base.size(0), f_bins, t_len, 2, cfg.q_trace_rank), device=base.device, dtype=base.dtype)
    mode_q_trace[..., 1, :] = float(cfg.dormant_q_scale)
    state = {
        "phase_modes": phasor_normalize(phase_modes),
        "mode_logits": mode_logits,
        "mode_support": mode_support,
        "mode_coherence": mode_coherence,
        "mode_q_trace": mode_q_trace,
        "child_worlds": [],
        "child_event_history": [],
        "next_child_id": 0,
    }
    state["mixed_phase_state"] = _mix_multimode_state(state, cfg)
    return state


def _empty_child_metrics(reference: torch.Tensor) -> dict[str, torch.Tensor]:
    zero = reference.new_tensor(0.0)
    return {
        "child_world_count": zero,
        "live_child_fraction": zero,
        "child_age_mean": zero,
        "child_writeback_mass": zero,
        "child_writeback_gate_mass": zero,
        "child_phase_writeback_delta_mass": zero,
        "child_parent_phase_writeback_delta_mass": zero,
        "child_support_writeback_mass": zero,
        "child_assay_writeback_scale_mean": zero,
        "child_assay_writeback_scale_min": zero,
        "child_logit_writeback_mass": zero,
        "child_qtrace_writeback_mass": zero,
        "child_parent_divergence": zero,
        "child_sibling_divergence": zero,
        "child_local_ifs_step_count": zero,
        "child_local_ifs_packet_count": zero,
        "child_local_ifs_phase_delta": zero,
        "child_local_ifs_support_mean": zero,
        "child_local_ifs_coherence_mean": zero,
        "child_local_ifs_coherence_raw_mean": zero,
        "child_local_ifs_coherence_retention_delta": zero,
        "child_local_ifs_coherence_floor_delta": zero,
        "child_local_ifs_parent_phase_delta": zero,
        "child_local_ifs_causal_gate_mean": zero,
        "child_local_ifs_causal_retention_loss": zero,
        "child_local_ifs_causal_floor_loss": zero,
        "defect_without_branch_penalty": zero,
        "branch_resolution_delay": zero,
    }


def _spawn_child_worlds(
    state: dict[str, Any],
    cfg: CircleworldConfig,
    mode1_packets: list[dict[str, torch.Tensor]],
    mode1: torch.Tensor,
    mode0: torch.Tensor,
    q1: torch.Tensor,
    branch_pos: torch.Tensor,
    defect_mask: torch.Tensor,
    q_disagreement: torch.Tensor,
    phase_wall: torch.Tensor,
    branch_neg: torch.Tensor,
    mode1_mask: torch.Tensor,
    mode1_arc: dict[str, torch.Tensor],
    mode1_promo: dict[str, torch.Tensor],
) -> list[dict[str, Any]]:
    children = list(state.get("child_worlds", []))
    active_count = sum(1 for child in children if child.get("active", False))
    if active_count >= cfg.child_max_worlds:
        return children
    sorted_packets = sorted(mode1_packets, key=lambda pkt: float(pkt["score"].item()), reverse=True)
    candidate_rows: list[dict[str, Any]] = []
    spawned_any = False
    for packet in sorted_packets:
        if active_count >= cfg.child_max_worlds:
            break
        t_idx = int(packet["time_index"].item())
        t0, t1 = _window_bounds(t_idx, mode1.size(2), cfg.child_support_window)
        local_branch = float(branch_pos[..., t0:t1].mean().item())
        local_defect = float(defect_mask[..., t0:t1].mean().item())
        local_q = float(q_disagreement[..., t0:t1].mean().item())
        local_phase = float(phase_wall[..., t0:t1].mean().item())
        seed_state = packet["seed_state"].expand_as(mode1)
        support_mask = _child_support_mask(mode1, (t0, t1))
        seed_parent_div = float(
            _masked_mean(
                1.0 - ((_phase_alignment(seed_state, mode0) + 1.0) * 0.5),
                support_mask,
            ).item()
        )
        local_incompat = 0.40 * local_q + 0.30 * local_phase + 0.30 * seed_parent_div
        local_neg = float(branch_neg[..., t0:t1].mean().item())
        local_support = float(mode1_mask[..., t0:t1].mean().item())
        packet_score = float(packet["score"].item())
        q_mass = packet["q_mass"].squeeze(0)
        q_peak = float((q_mass / q_mass.sum().clamp_min(1e-8)).amax().item())
        spawn_score = (
            0.24 * local_branch
            + 0.22 * local_defect
            + 0.24 * local_incompat
            + 0.16 * packet_score
            + 0.10 * local_support
            + 0.08 * q_peak
            - 0.08 * local_neg
        )
        dynamic_threshold = max(
            0.12,
            float(cfg.child_spawn_threshold)
            - 0.16 * packet_score
            - 0.12 * local_defect
            - 0.08 * seed_parent_div,
        )
        candidate_rows.append(
            {
                "packet": packet,
                "t_idx": t_idx,
                "t0": t0,
                "t1": t1,
                "support_mask": support_mask,
                "spawn_score": float(spawn_score),
                "dynamic_threshold": float(dynamic_threshold),
                "local_branch": float(local_branch),
                "local_defect": float(local_defect),
                "local_q": float(local_q),
                "local_phase": float(local_phase),
                "local_incompatibility": float(local_incompat),
                "local_support": float(local_support),
                "local_neg": float(local_neg),
                "packet_score": float(packet_score),
                "seed_parent_divergence": float(seed_parent_div),
                "q_peak": float(q_peak),
            }
        )
        if spawn_score < dynamic_threshold:
            continue
        if local_incompat < 0.025 or local_support < 0.015:
            continue
        child_id = int(state.get("next_child_id", 0))
        law_signature = _packet_law_signature(packet, mode1_arc, mode1_promo, cfg)
        child_phase = phasor_normalize(
            support_mask.unsqueeze(-1) * seed_state
            + (1.0 - support_mask).unsqueeze(-1) * mode1
        )
        child_phase = phasor_apply_delta(
            child_phase,
            _child_operator_phase_bias(law_signature, mode1, support_mask, cfg, child_id),
        )
        child = {
            "child_id": child_id,
            "parent_depth": int(len(state.get("child_event_history", []))),
            "spawn_time_index": t_idx,
            "origin_mode_index": 1,
            "support_window": (t0, t1),
            "survival_age": 0,
            "phase_state": child_phase,
            "mode_support": support_mask.clone(),
            "mode_coherence": (support_mask * mode1_mask).clamp(0.0, 1.0),
            "q_trace": (q1 * support_mask.unsqueeze(-1)).clone(),
            "law_signature": law_signature.clone(),
            "writeback_budget": float(cfg.child_writeback_budget),
            "active": True,
            "collapsed": False,
        }
        children.append(child)
        state["next_child_id"] = int(state.get("next_child_id", 0)) + 1
        state.setdefault("child_event_history", []).append(
            {
                "event": "spawn",
                "child_id": int(child["child_id"]),
                "spawn_time_index": int(t_idx),
                "support_window": [int(t0), int(t1)],
                "spawn_score": float(spawn_score),
                "local_branch": float(local_branch),
                "local_defect": float(local_defect),
                "local_incompatibility": float(local_incompat),
                "seed_parent_divergence": float(seed_parent_div),
                "packet_score": float(packet_score),
                "forced_spawn": False,
                "survival_age": 0.0,
            }
        )
        active_count += 1
        spawned_any = True
    unresolved_mass = float(
        (
            0.45 * branch_pos.mean()
            + 0.35 * defect_mask.mean()
            + 0.20 * q_disagreement.mean()
        ).item()
    )
    if (not spawned_any) and active_count < cfg.child_max_worlds and candidate_rows:
        candidate_rows.sort(
            key=lambda row: (
                row["spawn_score"] + 0.25 * row["seed_parent_divergence"] + 0.20 * row["packet_score"]
            ),
            reverse=True,
        )
        fallback_limit = min(cfg.child_max_worlds - active_count, 2)
        for row in candidate_rows[:fallback_limit]:
            if unresolved_mass < 0.06:
                break
            if row["local_incompatibility"] < 0.02:
                continue
            packet = row["packet"]
            support_mask = row["support_mask"]
            seed_state = packet["seed_state"].expand_as(mode1)
            child_id = int(state.get("next_child_id", 0))
            law_signature = _packet_law_signature(packet, mode1_arc, mode1_promo, cfg)
            child_phase = phasor_normalize(
                support_mask.unsqueeze(-1) * seed_state
                + (1.0 - support_mask).unsqueeze(-1) * mode1
            )
            child_phase = phasor_apply_delta(
                child_phase,
                _child_operator_phase_bias(law_signature, mode1, support_mask, cfg, child_id),
            )
            child = {
                "child_id": child_id,
                "parent_depth": int(len(state.get("child_event_history", []))),
                "spawn_time_index": int(row["t_idx"]),
                "origin_mode_index": 1,
                "support_window": (int(row["t0"]), int(row["t1"])),
                "survival_age": 0,
                "phase_state": child_phase,
                "mode_support": support_mask.clone(),
                "mode_coherence": (support_mask * mode1_mask).clamp(0.0, 1.0),
                "q_trace": (q1 * support_mask.unsqueeze(-1)).clone(),
                "law_signature": law_signature.clone(),
                "writeback_budget": float(cfg.child_writeback_budget),
                "active": True,
                "collapsed": False,
            }
            children.append(child)
            state["next_child_id"] = int(state.get("next_child_id", 0)) + 1
            state.setdefault("child_event_history", []).append(
                {
                    "event": "spawn",
                    "child_id": int(child["child_id"]),
                    "spawn_time_index": int(row["t_idx"]),
                    "support_window": [int(row["t0"]), int(row["t1"])],
                    "spawn_score": float(row["spawn_score"]),
                    "local_branch": float(row["local_branch"]),
                    "local_defect": float(row["local_defect"]),
                    "local_incompatibility": float(row["local_incompatibility"]),
                    "seed_parent_divergence": float(row["seed_parent_divergence"]),
                    "packet_score": float(row["packet_score"]),
                    "forced_spawn": True,
                    "survival_age": 0.0,
                }
            )
            active_count += 1
            spawned_any = True
            if active_count >= cfg.child_max_worlds:
                break
    return children


def _evolve_child_worlds(
    state: dict[str, Any],
    cfg: CircleworldConfig,
    parent_mode0: torch.Tensor,
    parent_mode1: torch.Tensor,
    parent_branch_defect: torch.Tensor,
) -> tuple[list[dict[str, Any]], dict[str, torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor]:
    children = list(state.get("child_worlds", []))
    if not children:
        empty = _empty_child_metrics(parent_mode0[..., 0])
        zero_delta = torch.zeros(parent_mode0.shape[:-1], device=parent_mode0.device, dtype=parent_mode0.dtype)
        zero_support = torch.zeros(parent_mode0.shape[:-1], device=parent_mode0.device, dtype=parent_mode0.dtype)
        return [], empty, zero_delta, zero_delta, zero_support

    writeback_delta0 = torch.zeros(parent_mode0.shape[:-1], device=parent_mode0.device, dtype=parent_mode0.dtype)
    writeback_delta1 = torch.zeros(parent_mode1.shape[:-1], device=parent_mode1.device, dtype=parent_mode1.dtype)
    writeback_support1 = torch.zeros(parent_mode1.shape[:-1], device=parent_mode1.device, dtype=parent_mode1.dtype)
    live_children: list[dict[str, Any]] = []
    parent_div_terms: list[torch.Tensor] = []
    sibling_div_terms: list[torch.Tensor] = []
    age_terms: list[torch.Tensor] = []
    writeback_terms: list[torch.Tensor] = []
    phase_writeback_terms: list[torch.Tensor] = []
    parent_phase_writeback_terms: list[torch.Tensor] = []
    support_writeback_terms: list[torch.Tensor] = []
    assay_writeback_scale_terms: list[torch.Tensor] = []
    local_ifs_step_terms: list[torch.Tensor] = []
    local_ifs_packet_terms: list[torch.Tensor] = []
    local_ifs_delta_terms: list[torch.Tensor] = []
    local_ifs_support_terms: list[torch.Tensor] = []
    local_ifs_coherence_terms: list[torch.Tensor] = []
    local_ifs_coherence_raw_terms: list[torch.Tensor] = []
    local_ifs_coherence_retention_terms: list[torch.Tensor] = []
    local_ifs_coherence_floor_terms: list[torch.Tensor] = []
    local_ifs_parent_phase_delta_terms: list[torch.Tensor] = []
    local_ifs_causal_gate_terms: list[torch.Tensor] = []
    local_ifs_causal_retention_loss_terms: list[torch.Tensor] = []
    local_ifs_causal_floor_loss_terms: list[torch.Tensor] = []

    for child in children:
        if not child.get("active", False):
            continue
        t0, t1 = child["support_window"]
        support_mask = child["mode_support"].clamp(0.0, 1.0)
        if cfg.child_local_ifs_enabled:
            child, local_metrics = child_circleworld_step(child, cfg, parent_mode0, parent_mode1)
            state.setdefault("child_event_history", []).append(
                {
                    "event": "child_local_ifs",
                    "child_id": int(child["child_id"]),
                    "support_window": [int(t0), int(t1)],
                    "survival_age": float(child["survival_age"]),
                    "step_count": float(local_metrics["child_local_ifs_step_count"].item()),
                    "packet_count": float(local_metrics["child_local_ifs_packet_count"].item()),
                    "phase_delta": float(local_metrics["child_local_ifs_phase_delta"].item()),
                    "support_mean": float(local_metrics["child_local_ifs_support_mean"].item()),
                    "coherence_mean": float(local_metrics["child_local_ifs_coherence_mean"].item()),
                    "coherence_raw_mean": float(local_metrics["child_local_ifs_coherence_raw_mean"].item()),
                    "coherence_retention_delta": float(local_metrics["child_local_ifs_coherence_retention_delta"].item()),
                    "coherence_floor_delta": float(local_metrics["child_local_ifs_coherence_floor_delta"].item()),
                    "parent_phase_delta": float(local_metrics["child_local_ifs_parent_phase_delta"].item()),
                    "causal_gate_mean": float(local_metrics["child_local_ifs_causal_gate_mean"].item()),
                    "causal_retention_loss": float(local_metrics["child_local_ifs_causal_retention_loss"].item()),
                    "causal_floor_loss": float(local_metrics["child_local_ifs_causal_floor_loss"].item()),
                }
            )
            local_ifs_step_terms.append(local_metrics["child_local_ifs_step_count"])
            local_ifs_packet_terms.append(local_metrics["child_local_ifs_packet_count"])
            local_ifs_delta_terms.append(local_metrics["child_local_ifs_phase_delta"])
            local_ifs_support_terms.append(local_metrics["child_local_ifs_support_mean"])
            local_ifs_coherence_terms.append(local_metrics["child_local_ifs_coherence_mean"])
            local_ifs_coherence_raw_terms.append(local_metrics["child_local_ifs_coherence_raw_mean"])
            local_ifs_coherence_retention_terms.append(local_metrics["child_local_ifs_coherence_retention_delta"])
            local_ifs_coherence_floor_terms.append(local_metrics["child_local_ifs_coherence_floor_delta"])
            local_ifs_parent_phase_delta_terms.append(local_metrics["child_local_ifs_parent_phase_delta"])
            local_ifs_causal_gate_terms.append(local_metrics["child_local_ifs_causal_gate_mean"])
            local_ifs_causal_retention_loss_terms.append(local_metrics["child_local_ifs_causal_retention_loss"])
            local_ifs_causal_floor_loss_terms.append(local_metrics["child_local_ifs_causal_floor_loss"])
            support_mask = child["mode_support"].clamp(0.0, 1.0)
        child_arc = hardy_littlewood_arc_field(child["phase_state"], cfg.qset, cfg.q_weights)
        child_promo = promotability_field(child_arc, persistence_momentum=cfg.persistence_momentum)
        child_q = _compress_q_trace(child_arc["per_q"], cfg.q_trace_rank)
        child_q_expanded = child_q.permute(0, 2, 1).unsqueeze(1).expand(-1, child["phase_state"].size(1), -1, -1)
        child_coherence = (
            cfg.child_survival_coherence_weight * _expand_time_to_freq(child_arc["major_mass"], child["phase_state"].size(1))
            + cfg.child_survival_qtrace_weight * _expand_time_to_freq(child_q.abs().mean(dim=1), child["phase_state"].size(1))
            - cfg.child_survival_residue_penalty * _expand_time_to_freq(child_arc["minor_residue"], child["phase_state"].size(1))
        ).clamp(0.0, 1.0)
        if not cfg.child_local_ifs_enabled:
            child["survival_age"] = int(child.get("survival_age", 0)) + 1
            child["mode_support"] = (
                (1.0 - cfg.child_support_decay) * support_mask
                + cfg.child_support_decay * _child_support_mask(child["phase_state"], child["support_window"])
            ).clamp(0.0, 1.0)
            child["mode_coherence"] = child_coherence.clamp(0.0, 1.0)
            child["q_trace"] = cfg.qtrace_momentum * child["q_trace"] + (1.0 - cfg.qtrace_momentum) * (
                child_q_expanded * child["mode_support"].unsqueeze(-1)
            )
        else:
            child["mode_coherence"] = torch.maximum(child["mode_coherence"], child_coherence).clamp(0.0, 1.0)

        support_mass = child["mode_support"].mean()
        coherence_mass = _masked_mean(child["mode_coherence"], child["mode_support"])
        residue_mass = _masked_mean(_expand_time_to_freq(child_arc["minor_residue"], child["phase_state"].size(1)), child["mode_support"])
        parent_div = _masked_mean(
            1.0 - ((_phase_alignment(child["phase_state"], parent_mode0) + 1.0) * 0.5),
            child["mode_support"],
        ).clamp(0.0, 1.0)
        sibling_div = _masked_mean(
            1.0 - ((_phase_alignment(child["phase_state"], parent_mode1) + 1.0) * 0.5),
            child["mode_support"],
        ).clamp(0.0, 1.0)

        if float((0.55 * support_mass + 0.45 * coherence_mass - 0.35 * residue_mass).item()) < cfg.child_kill_threshold:
            child["active"] = False
            child["collapsed"] = True
            state.setdefault("child_event_history", []).append(
                {
                    "event": "collapse",
                    "child_id": int(child["child_id"]),
                    "support_window": [int(t0), int(t1)],
                    "survival_age": float(child["survival_age"]),
                    "support_mass": float(support_mass.item()),
                    "coherence_mass": float(coherence_mass.item()),
                    "residue_mass": float(residue_mass.item()),
                }
            )
            continue

        phase_delta = _phasor_phase_delta(parent_mode1, child["phase_state"])
        local_defect = _masked_mean(parent_branch_defect, child["mode_support"]).clamp(0.0, 1.0)
        budget = float(child.get("writeback_budget", 0.0))
        base_ready = (
            child["survival_age"] >= cfg.child_min_age_for_writeback
            and budget > 0.0
            and float(parent_div.item()) > 0.04
        )
        early_ready = (
            child["survival_age"] >= 1
            and budget > 0.0
            and float(parent_div.item()) > 0.10
            and float(sibling_div.item()) > 0.08
            and float(coherence_mass.item()) > 0.12
            and float(local_defect.item()) > 0.10
        )
        writeback_ready = base_ready or early_ready
        assay_writeback_scale = float(cfg.child_assay_writeback_scale_default)
        if cfg.child_assay_writeback_scale_enabled:
            assay_writeback_scale = float(child.get("assay_writeback_scale", assay_writeback_scale))
        assay_writeback_scale = max(0.0, min(1.0, assay_writeback_scale))
        assay_writeback_scale_t = parent_mode1.new_tensor(assay_writeback_scale)
        writeback_effective_ready = bool(writeback_ready and assay_writeback_scale > 1.0e-6)
        if writeback_ready:
            writeback_gate = torch.sigmoid(
                (
                    cfg.child_writeback_temperature
                    * (
                        child["mode_coherence"]
                        + local_defect
                        + 0.50 * parent_div
                        + 0.25 * sibling_div
                    )
                    - 0.45
                )
            ) * child["mode_support"]
            writeback_gain = cfg.child_writeback_gain * (1.25 if early_ready and not base_ready else 1.0)
            operator_delta = _child_operator_phase_bias(
                child["law_signature"],
                parent_mode1,
                child["mode_support"],
                cfg,
                int(child["child_id"]),
            )
            operator_sign = torch.where(operator_delta >= 0.0, torch.ones_like(operator_delta), -torch.ones_like(operator_delta))
            target_phase = parent_mode1.new_tensor(
                max(
                    float(cfg.child_writeback_phase_floor_target),
                    float(cfg.child_writeback_phase_floor_threshold_mult) * float(cfg.real_branch_threshold),
                )
            )
            phase_floor = (
                operator_sign
                * F.relu(target_phase - phase_delta.abs())
                * child["mode_support"]
                    * (0.30 + 0.70 * writeback_gate)
            )
            delta = writeback_gain * assay_writeback_scale_t * _low_rank_delta(
                writeback_gate
                * (
                    cfg.child_writeback_phase_delta_gain * phase_delta
                    + cfg.child_writeback_operator_mix * operator_delta
                )
                + cfg.child_writeback_phase_floor_gain * phase_floor,
                rank=min(cfg.child_writeback_rank, max(4, parent_mode1.size(1) // 4)),
            )
            writeback_delta1 = writeback_delta1 + delta
            parent_mix = cfg.child_parent_mix_early if early_ready and not base_ready else cfg.child_parent_mix
            parent_mix = min(float(parent_mix), float(cfg.child_writeback_parent_mix_cap))
            writeback_delta0 = writeback_delta0 + parent_mix * delta
            phase_write_mass = delta.abs().mean()
            parent_phase_write_mass = (parent_mix * delta).abs().mean()
            identity_divergence = (0.5 * parent_div + 0.5 * sibling_div).clamp(0.0, 1.0)
            support_write = assay_writeback_scale_t * cfg.child_support_writeback_gain * (
                torch.maximum(
                    writeback_gate * (0.25 + 0.75 * identity_divergence),
                    cfg.child_support_writeback_floor_gain
                    * child["mode_support"]
                    * child["mode_coherence"]
                    * identity_divergence,
                )
                * (0.55 + 0.45 * local_defect)
            ).clamp(0.0, 1.0)
            writeback_support1 = torch.maximum(writeback_support1, support_write)
            write_mass = (assay_writeback_scale_t * writeback_gate).mean()
            support_write_mass = support_write.mean()
            child["writeback_budget"] = max(0.0, float(child["writeback_budget"]) - float(write_mass.item()))
            state.setdefault("child_event_history", []).append(
                {
                    "event": "writeback",
                    "child_id": int(child["child_id"]),
                    "support_window": [int(t0), int(t1)],
                    "survival_age": float(child["survival_age"]),
                    "writeback_mass": float(write_mass.item()),
                    "writeback_gate_mass": float(write_mass.item()),
                    "phase_writeback_delta_mass": float(phase_write_mass.item()),
                    "parent_phase_writeback_delta_mass": float(parent_phase_write_mass.item()),
                    "support_writeback_mass": float(support_write_mass.item()),
                    "parent_divergence": float(parent_div.item()),
                    "sibling_divergence": float(sibling_div.item()),
                    "early_writeback": bool(early_ready and not base_ready),
                    "assay_writeback_scale": float(assay_writeback_scale),
                }
            )
            writeback_terms.append(write_mass)
            phase_writeback_terms.append(phase_write_mass)
            parent_phase_writeback_terms.append(parent_phase_write_mass)
            support_writeback_terms.append(support_write_mass)
            assay_writeback_scale_terms.append(assay_writeback_scale_t)

        child_retain = child["mode_support"]
        if writeback_effective_ready:
            child_retain = torch.maximum(child_retain, 0.75 * child["mode_support"] + 0.20 * child["mode_coherence"])
        child_retain = child_retain.clamp(0.0, 1.0)
        child["phase_state"] = phasor_normalize(
            child_retain.unsqueeze(-1) * child["phase_state"]
            + (1.0 - child_retain).unsqueeze(-1) * parent_mode1
        )
        live_children.append(child)
        parent_div_terms.append(parent_div)
        sibling_div_terms.append(sibling_div)
        age_terms.append(parent_div.new_tensor(float(child["survival_age"])))

    child_count = parent_mode0.new_tensor(float(len(live_children)))
    child_metrics = {
        "child_world_count": child_count,
        "live_child_fraction": child_count / max(1.0, float(cfg.child_max_worlds)),
        "child_age_mean": torch.stack(age_terms).mean() if age_terms else parent_mode0.new_tensor(0.0),
        "child_writeback_mass": torch.stack(writeback_terms).mean() if writeback_terms else parent_mode0.new_tensor(0.0),
        "child_writeback_gate_mass": torch.stack(writeback_terms).mean() if writeback_terms else parent_mode0.new_tensor(0.0),
        "child_phase_writeback_delta_mass": (
            torch.stack(phase_writeback_terms).mean() if phase_writeback_terms else parent_mode0.new_tensor(0.0)
        ),
        "child_parent_phase_writeback_delta_mass": (
            torch.stack(parent_phase_writeback_terms).mean() if parent_phase_writeback_terms else parent_mode0.new_tensor(0.0)
        ),
        "child_support_writeback_mass": (
            torch.stack(support_writeback_terms).mean() if support_writeback_terms else parent_mode0.new_tensor(0.0)
        ),
        "child_assay_writeback_scale_mean": (
            torch.stack(assay_writeback_scale_terms).mean()
            if assay_writeback_scale_terms
            else parent_mode0.new_tensor(0.0)
        ),
        "child_assay_writeback_scale_min": (
            torch.stack(assay_writeback_scale_terms).min()
            if assay_writeback_scale_terms
            else parent_mode0.new_tensor(0.0)
        ),
        "child_logit_writeback_mass": parent_mode0.new_tensor(0.0),
        "child_qtrace_writeback_mass": parent_mode0.new_tensor(0.0),
        "child_parent_divergence": torch.stack(parent_div_terms).mean() if parent_div_terms else parent_mode0.new_tensor(0.0),
        "child_sibling_divergence": torch.stack(sibling_div_terms).mean() if sibling_div_terms else parent_mode0.new_tensor(0.0),
        "child_local_ifs_step_count": (
            torch.stack(local_ifs_step_terms).mean() if local_ifs_step_terms else parent_mode0.new_tensor(0.0)
        ),
        "child_local_ifs_packet_count": (
            torch.stack(local_ifs_packet_terms).mean() if local_ifs_packet_terms else parent_mode0.new_tensor(0.0)
        ),
        "child_local_ifs_phase_delta": (
            torch.stack(local_ifs_delta_terms).mean() if local_ifs_delta_terms else parent_mode0.new_tensor(0.0)
        ),
        "child_local_ifs_support_mean": (
            torch.stack(local_ifs_support_terms).mean() if local_ifs_support_terms else parent_mode0.new_tensor(0.0)
        ),
        "child_local_ifs_coherence_mean": (
            torch.stack(local_ifs_coherence_terms).mean() if local_ifs_coherence_terms else parent_mode0.new_tensor(0.0)
        ),
        "child_local_ifs_coherence_raw_mean": (
            torch.stack(local_ifs_coherence_raw_terms).mean() if local_ifs_coherence_raw_terms else parent_mode0.new_tensor(0.0)
        ),
        "child_local_ifs_coherence_retention_delta": (
            torch.stack(local_ifs_coherence_retention_terms).mean() if local_ifs_coherence_retention_terms else parent_mode0.new_tensor(0.0)
        ),
        "child_local_ifs_coherence_floor_delta": (
            torch.stack(local_ifs_coherence_floor_terms).mean() if local_ifs_coherence_floor_terms else parent_mode0.new_tensor(0.0)
        ),
        "child_local_ifs_parent_phase_delta": (
            torch.stack(local_ifs_parent_phase_delta_terms).mean() if local_ifs_parent_phase_delta_terms else parent_mode0.new_tensor(0.0)
        ),
        "child_local_ifs_causal_gate_mean": (
            torch.stack(local_ifs_causal_gate_terms).mean() if local_ifs_causal_gate_terms else parent_mode0.new_tensor(0.0)
        ),
        "child_local_ifs_causal_retention_loss": (
            torch.stack(local_ifs_causal_retention_loss_terms).mean()
            if local_ifs_causal_retention_loss_terms
            else parent_mode0.new_tensor(0.0)
        ),
        "child_local_ifs_causal_floor_loss": (
            torch.stack(local_ifs_causal_floor_loss_terms).mean()
            if local_ifs_causal_floor_loss_terms
            else parent_mode0.new_tensor(0.0)
        ),
        "defect_without_branch_penalty": (parent_branch_defect.mean() * (child_count <= 0).float()),
        "branch_resolution_delay": torch.stack(age_terms).mean() if age_terms else parent_mode0.new_tensor(0.0),
    }
    return live_children, child_metrics, writeback_delta0, writeback_delta1, writeback_support1


def _mode_occupancy(state: dict[str, torch.Tensor], cfg: CircleworldConfig) -> torch.Tensor:
    logits = state["mode_logits"]
    support = state["mode_support"].clamp_min(0.0)
    weights = torch.softmax(logits / max(1e-4, float(cfg.readout_temperature)), dim=-1) * support.clamp_min(1e-4)
    return weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-8)


def _mix_multimode_state(state: dict[str, torch.Tensor], cfg: CircleworldConfig) -> torch.Tensor:
    weights = _mode_occupancy(state, cfg)
    mixed = (weights.unsqueeze(-1) * state["phase_modes"]).sum(dim=3)
    return phasor_normalize(mixed)


def extract_phase_state(state_or_run: Any, cfg: CircleworldConfig | None = None) -> torch.Tensor:
    if isinstance(state_or_run, dict):
        if "phase_state" in state_or_run and torch.is_tensor(state_or_run["phase_state"]):
            return state_or_run["phase_state"]
        if "phase_modes" in state_or_run:
            if cfg is None:
                raise ValueError("cfg is required to extract phase state from multimode state")
            return _mix_multimode_state(state_or_run, cfg)
    if torch.is_tensor(state_or_run):
        return state_or_run
    raise ValueError("Unsupported state/run type")


def _multimode_branch_seed(state: dict[str, torch.Tensor], cfg: CircleworldConfig) -> torch.Tensor:
    mode0 = state["phase_modes"][..., 0, :]
    f_bins = mode0.size(1)
    t_len = mode0.size(2)
    freq_basis = torch.linspace(-1.0, 1.0, steps=f_bins, device=mode0.device, dtype=mode0.dtype).view(1, f_bins, 1)
    time_basis = torch.sin(torch.linspace(0.0, math.pi, steps=t_len, device=mode0.device, dtype=mode0.dtype)).view(1, 1, t_len)
    seed_delta = cfg.split_seed_scale * freq_basis * time_basis
    return _low_rank_delta(seed_delta, rank=min(cfg.matryoshka_rank, max(4, f_bins // 4)))


def _phasor_phase_delta(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    cross = a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]
    dot = a[..., 0] * b[..., 0] + a[..., 1] * b[..., 1]
    return torch.atan2(cross, dot)


def _multimode_step(
    current: torch.Tensor | dict[str, torch.Tensor],
    cfg: CircleworldConfig,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor], list[dict[str, torch.Tensor]]]:
    state = _init_multimode_state(current, cfg) if torch.is_tensor(current) else clone_circleworld_state(current)
    state.setdefault("child_worlds", [])
    state.setdefault("child_event_history", [])
    state.setdefault("next_child_id", 0)
    child_enabled = bool(cfg.branching_mode == "native_multimode_childworld")
    phase_modes = state["phase_modes"]
    logits = state["mode_logits"]
    support = state["mode_support"]
    q_trace = state["mode_q_trace"]

    mode_blocks: list[dict[str, torch.Tensor]] = []
    mode_packets: list[list[dict[str, torch.Tensor]]] = []
    transported_modes = []
    q_targets = []
    q_profiles = []
    local_evidences = []
    mode_coherences = []
    for mode_idx in range(cfg.num_modes):
        z_mode = phase_modes[..., mode_idx, :]
        arc = hardy_littlewood_arc_field(z_mode, cfg.qset, cfg.q_weights)
        promo = promotability_field(arc, persistence_momentum=cfg.persistence_momentum)
        packets = build_promoted_packets(z_mode, arc, promo, cfg)
        transported = apply_active_packets(z_mode, packets, cfg)
        transported_modes.append(transported)
        q_targets.append(_compress_q_trace(arc["per_q"], cfg.q_trace_rank))
        time_evidence = (
            0.55 * promo["promotability"]
            + 0.30 * arc["major_mass"]
            + 0.15 * promo["persistence"]
            - 0.30 * arc["minor_residue"]
        ).clamp(0.0, 1.0)
        freq_profile = _mode_to_freq_profile(arc["per_q"], transported.size(1))
        q_profiles.append(arc["per_q"].permute(0, 2, 1))
        local_evidence = (freq_profile.unsqueeze(-1) * time_evidence.unsqueeze(1)).clamp(0.0, 1.0)
        q_strength = _expand_time_to_freq(q_targets[-1].abs().mean(dim=1), transported.size(1)).clamp(0.0, 1.0)
        coherence = (
            0.45 * local_evidence
            + 0.25 * _expand_time_to_freq(arc["major_mass"], transported.size(1))
            + 0.15 * (1.0 - _expand_time_to_freq(arc["minor_residue"], transported.size(1)))
            + 0.15 * q_strength
        ).clamp(0.0, 1.0)
        mode_blocks.append({**arc, **promo})
        mode_packets.append(packets)
        local_evidences.append(local_evidence)
        mode_coherences.append(coherence)

    world_phase = _mix_multimode_state(state, cfg)
    world_arc = hardy_littlewood_arc_field(world_phase, cfg.qset, cfg.q_weights)
    world_promo = promotability_field(world_arc, persistence_momentum=cfg.persistence_momentum)

    mode0 = transported_modes[0]
    mode1 = transported_modes[1]
    evidence0 = local_evidences[0]
    evidence1 = local_evidences[1]
    coherence0 = mode_coherences[0]
    coherence1 = mode_coherences[1]
    residue0 = _expand_time_to_freq(mode_blocks[0]["minor_residue"], mode0.size(1))
    residue1 = _expand_time_to_freq(mode_blocks[1]["minor_residue"], mode1.size(1))
    sharpness0 = _expand_time_to_freq(mode_blocks[0]["sharpness"], mode0.size(1))
    sharpness1 = _expand_time_to_freq(mode_blocks[1]["sharpness"], mode1.size(1))

    phase_only_fields = _phase_only_branch_fields(mode0, mode1, support, cfg)
    phase_align = phase_only_fields["phase_align"]
    q0 = q_targets[0].permute(0, 2, 1).unsqueeze(1).expand(-1, mode0.size(1), -1, -1)
    q1 = q_targets[1].permute(0, 2, 1).unsqueeze(1).expand(-1, mode0.size(1), -1, -1)
    q_sim = ((_safe_cosine_similarity(q0, q1, dim=-1) + 1.0) * 0.5).clamp(0.0, 1.0)
    major_gap = _expand_time_to_freq((mode_blocks[0]["major_mass"] - mode_blocks[1]["major_mass"]).abs(), mode0.size(1)).clamp(0.0, 1.0)
    residue_gap = _expand_time_to_freq((mode_blocks[0]["minor_residue"] - mode_blocks[1]["minor_residue"]).abs(), mode0.size(1)).clamp(0.0, 1.0)
    support_overlap = (support[..., 0] * support[..., 1]).clamp(0.0, 1.0)
    compat = (0.45 * phase_align + 0.25 * q_sim + 0.20 * (1.0 - major_gap) - 0.10 * residue_gap).clamp(0.0, 1.0)
    ambiguity = ((1.0 - compat) * evidence0 * (1.0 - support[..., 1])).clamp(0.0, 1.0)

    topology_raw = _smooth_field_2d(
        0.45 * phase_align + 0.35 * q_sim + 0.20 * ((support[..., 0] + support[..., 1]) * 0.5),
        cfg.mask_neighborhood,
    )
    topology_mask = ((1.0 - cfg.topology_mask_gain) + cfg.topology_mask_gain * topology_raw).clamp(0.0, 1.0)
    density = _smooth_field_2d((0.5 * (evidence0 + evidence1)).clamp(0.0, 1.0), cfg.mask_neighborhood)
    complexity_raw = (
        0.35 * (0.5 * (sharpness0 + sharpness1))
        + 0.25 * residue_gap
        + 0.20 * major_gap
        + 0.20 * (1.0 - density)
    ).clamp(0.0, 1.0)
    complexity_mask = ((1.0 - cfg.complexity_mask_gain) + cfg.complexity_mask_gain * complexity_raw).clamp(0.0, 1.0)
    world_context_raw = (
        0.45 * _expand_time_to_freq(world_promo["promotability"], mode0.size(1))
        + 0.35 * _expand_time_to_freq(world_arc["major_mass"], mode0.size(1))
        + 0.20 * (1.0 - _expand_time_to_freq(world_arc["minor_residue"], mode0.size(1)))
    ).clamp(0.0, 1.0)
    context_mask = ((1.0 - cfg.context_mask_gain) + cfg.context_mask_gain * _smooth_field_2d(world_context_raw, cfg.mask_neighborhood)).clamp(0.0, 1.0)
    world_major_grad = _expand_time_to_freq(_time_gradient_field(world_arc["major_mass"]), mode0.size(1))
    world_promo_grad = _expand_time_to_freq(_time_gradient_field(world_promo["promotability"]), mode0.size(1))
    q_disagreement = (q0 - q1).abs().mean(dim=-1).clamp(0.0, 1.0)
    phase_wall = phase_only_fields["phase_wall"]
    residue_mix = (0.5 * (residue0 + residue1)).clamp(0.0, 1.0)
    sharpness_mix = (0.5 * (sharpness0 + sharpness1)).clamp(0.0, 1.0)
    defect_raw = (
        cfg.defect_phase_gain * phase_wall
        + cfg.defect_q_gain * q_disagreement
        + cfg.defect_residue_gain * residue_mix
        + cfg.defect_sharpness_gain * sharpness_mix
        + cfg.defect_world_grad_gain * (0.5 * world_major_grad + 0.5 * world_promo_grad)
    ).clamp(0.0, 1.0)
    defect_mask = (
        (1.0 - cfg.instability_mask_gain) * defect_raw
        + cfg.instability_mask_gain * _smooth_field_2d(defect_raw, cfg.mask_neighborhood)
    ).clamp(0.0, 1.0)
    branch_pos_raw = (ambiguity * complexity_mask * context_mask * topology_mask + 0.85 * defect_mask * context_mask).clamp(0.0, 1.0)
    branch_neg_raw = (compat * (1.0 - ambiguity) * (1.0 - complexity_raw) * topology_mask * (1.0 - 0.5 * defect_mask)).clamp(0.0, 1.0)
    branch_pos = ((1.0 - cfg.contrastive_mask_gain) * ambiguity + cfg.contrastive_mask_gain * branch_pos_raw).clamp(0.0, 1.0)
    branch_neg = ((1.0 - cfg.contrastive_mask_gain) * (compat * (1.0 - ambiguity)) + cfg.contrastive_mask_gain * branch_neg_raw).clamp(0.0, 1.0)
    mode0_mask = (topology_mask * context_mask).clamp(0.0, 1.0)
    mode1_mask = ((complexity_mask + defect_mask).clamp(0.0, 1.0) * context_mask).clamp(0.0, 1.0)

    phase_delta = _phasor_phase_delta(mode0, mode1)
    defect_seed = cfg.instability_seed_scale * _low_rank_delta(
        defect_mask * phase_delta,
        rank=min(cfg.matryoshka_rank, max(4, mode0.size(1) // 4)),
    )
    split_seed = (_multimode_branch_seed(state, cfg) * branch_pos + defect_seed).clamp(-3.14159, 3.14159)
    slot1_seeded = phasor_apply_delta(mode0, split_seed)

    merge_strength = (
        cfg.merge_pressure
        * F.relu(compat - cfg.merge_phase_tol)
        * (support_overlap ** max(0.0, cfg.merge_support_overlap_weight))
    ).clamp(0.0, 1.0)
    mode0_next = phasor_normalize((1.0 - 0.5 * merge_strength).unsqueeze(-1) * mode0 + (0.5 * merge_strength).unsqueeze(-1) * mode1)
    slot1_mixed = phasor_normalize((1.0 - cfg.split_pressure * ambiguity).unsqueeze(-1) * mode1 + (cfg.split_pressure * ambiguity).unsqueeze(-1) * slot1_seeded)
    mode1_next = phasor_normalize((1.0 - 0.5 * merge_strength).unsqueeze(-1) * slot1_mixed + (0.5 * merge_strength).unsqueeze(-1) * mode0)

    support0_target = ((1.0 - cfg.support_decay) * support[..., 0] + evidence0 * mode0_mask).clamp(0.0, 1.0)
    support1_target = (
        (1.0 - cfg.support_decay) * support[..., 1]
        + evidence1 * mode1_mask
        + cfg.split_support_gain * branch_pos
        + cfg.instability_support_gain * defect_mask
        - cfg.aux_mask_suppression * branch_neg
    ).clamp(0.0, 1.0)
    support_next = torch.stack([support0_target, support1_target], dim=-1)
    support_next = _spread_support_time(support_next, cfg.support_spread)

    q0_strength = q0.abs().mean(dim=-1) * mode0_mask
    q1_strength = q1.abs().mean(dim=-1) * mode1_mask
    occupancy_prev = _mode_occupancy(state, cfg)
    relation_kernel = _relation_kernel_field(
        mode0=mode0,
        mode1=mode1,
        q_profile0=q_profiles[0],
        q_profile1=q_profiles[1],
        q_sim=q_sim,
        phase_align=phase_align,
        cfg=cfg,
    )
    relation_metrics = {
        "attn": None,
        "handoff_drive": torch.zeros_like(branch_pos),
        "kernel_gate": torch.zeros_like(branch_pos),
        "mean_attn_10": torch.zeros((), device=mode0.device, dtype=mode0.dtype),
        "mean_attn_01": torch.zeros((), device=mode0.device, dtype=mode0.dtype),
    }
    if cfg.branch_law_version == "relational_qkv_v2":
        relation = _relational_branch_attention(
            evidence0=evidence0,
            evidence1=evidence1,
            coherence0=coherence0,
            coherence1=coherence1,
            q0_strength=q0_strength,
            q1_strength=q1_strength,
            residue0=residue0,
            residue1=residue1,
            sharpness0=sharpness0,
            sharpness1=sharpness1,
            support=support,
            occupancy_prev=occupancy_prev,
            world_context=world_context_raw,
            branch_pos=branch_pos,
            relation_kernel=relation_kernel,
            cfg=cfg,
        )
        relation_support = cfg.relation_residual_mix * cfg.relation_support_gain * relation["handoff_drive"]
        relation_logit = cfg.relation_residual_mix * cfg.relation_logit_gain * relation["handoff_drive"]
        support_next[..., 1] = (support_next[..., 1] + relation_support).clamp(0.0, 1.0)
        support_next[..., 0] = (support_next[..., 0] - 0.35 * relation_support).clamp(0.0, 1.0)
        relation_metrics = {
            "attn": relation["attn"],
            "handoff_drive": relation["handoff_drive"],
            "kernel_gate": relation["kernel_gate"],
            "mean_attn_10": relation["attn"][..., 1, 0].mean(),
            "mean_attn_01": relation["attn"][..., 0, 1].mean(),
        }
    else:
        relation_logit = torch.zeros_like(branch_pos)
    anti_fix = cfg.anti_fixation_weight * (branch_pos + 0.5 * defect_mask) * occupancy_prev[..., 0]
    update0 = (
        cfg.survival_arc_weight * evidence0 * mode0_mask
        + cfg.survival_coherence_weight * coherence0 * mode0_mask
        + cfg.survival_qtrace_weight * q0_strength
        - cfg.survival_residue_penalty * residue0
        - cfg.support_overlap_penalty * support_overlap * compat
    )
    update1 = (
        cfg.survival_arc_weight * evidence1 * mode1_mask
        + cfg.survival_coherence_weight * coherence1 * mode1_mask
        + cfg.survival_qtrace_weight * q1_strength
        - cfg.survival_residue_penalty * residue1
        - cfg.support_overlap_penalty * support_overlap * compat
        + cfg.split_pressure * branch_pos
        + cfg.instability_logit_gain * defect_mask
        + anti_fix
        + relation_logit
        - cfg.aux_mask_suppression * branch_neg
    )
    logits_next = logits.clone()
    logits_next[..., 0] = (1.0 - cfg.support_decay) * logits[..., 0] + update0
    logits_next[..., 1] = (1.0 - cfg.support_decay) * logits[..., 1] + update1

    occupancy_mid = torch.softmax(logits_next / max(1e-4, float(cfg.readout_temperature)), dim=-1)
    collapse_mass = (
        cfg.collapse_sharpness
        * compat
        * F.relu(occupancy_mid[..., 0] - 0.85)
        * (support_overlap + branch_neg).clamp(0.0, 1.0)
    ).clamp(0.0, 4.0)
    logits_next[..., 1] = logits_next[..., 1] - collapse_mass

    q_trace_next = q_trace.clone()
    q_trace_next[..., 0, :] = cfg.qtrace_momentum * q_trace[..., 0, :] + (1.0 - cfg.qtrace_momentum) * (q0 * mode0_mask.unsqueeze(-1))
    q_trace_next[..., 1, :] = cfg.qtrace_momentum * q_trace[..., 1, :] + (1.0 - cfg.qtrace_momentum) * (q1 * branch_pos.unsqueeze(-1))
    if cfg.branch_law_version == "relational_qkv_v2":
        q_trace_next[..., 1, :] = q_trace_next[..., 1, :] + (
            cfg.relation_residual_mix
            * cfg.relation_qtrace_gain
            * relation_metrics["handoff_drive"].unsqueeze(-1)
            * (
                0.70 * q0 * relation_kernel.unsqueeze(-1)
                + 0.30 * q1
            )
        )

    child_metrics = _empty_child_metrics(mode0[..., 0])
    child_worlds_next = list(state.get("child_worlds", []))
    if child_enabled:
        child_worlds_next, child_metrics, child_delta0, child_delta1, child_support1 = _evolve_child_worlds(
            state=state,
            cfg=cfg,
            parent_mode0=mode0_next,
            parent_mode1=mode1_next,
            parent_branch_defect=defect_mask,
        )
        if child_worlds_next:
            child_delta0 = _precondition_phase_delta(mode0_next, child_delta0.clamp(-3.14159, 3.14159), cfg)
            child_delta1 = _precondition_phase_delta(mode1_next, child_delta1.clamp(-3.14159, 3.14159), cfg)
            mode0_next = phasor_apply_delta(mode0_next, child_delta0)
            mode1_next = phasor_apply_delta(mode1_next, child_delta1)
            child_identity = child_support1.clamp(0.0, 1.0)
            support_next[..., 1] = torch.maximum(support_next[..., 1], child_identity).clamp(0.0, 1.0)
            child_logit_write = cfg.child_writeback_gain * (
                0.50 + 0.50 * child_metrics["child_parent_divergence"].clamp(0.0, 1.0)
            ) * child_identity
            logits_next[..., 1] = logits_next[..., 1] + child_logit_write
            child_qtrace_write = (1.0 - cfg.qtrace_momentum) * child_identity.unsqueeze(-1) * q1
            q_trace_next[..., 1, :] = q_trace_next[..., 1, :] + child_qtrace_write
            child_metrics["child_logit_writeback_mass"] = child_logit_write.abs().mean()
            child_metrics["child_qtrace_writeback_mass"] = child_qtrace_write.abs().mean()
        state["child_worlds"] = child_worlds_next
        child_worlds_next = _spawn_child_worlds(
            state=state,
            cfg=cfg,
            mode1_packets=mode_packets[1],
            mode1=mode1_next,
            mode0=mode0_next,
            q1=q_trace_next[..., 1, :],
            branch_pos=branch_pos,
            defect_mask=defect_mask,
            q_disagreement=q_disagreement,
            phase_wall=phase_wall,
            branch_neg=branch_neg,
            mode1_mask=mode1_mask,
            mode1_arc=mode_blocks[1],
            mode1_promo={"persistence": mode_blocks[1]["persistence"], "promotability": mode_blocks[1]["promotability"]},
        )

    phase_only_fields_next = _phase_only_branch_fields(mode0_next, mode1_next, support_next, cfg)
    coherence_next = torch.stack([coherence0, coherence1], dim=-1)
    state_next = {
        "phase_modes": torch.stack([mode0_next, mode1_next], dim=3),
        "mode_logits": logits_next,
        "mode_support": support_next,
        "mode_coherence": coherence_next,
        "mode_q_trace": q_trace_next,
        "phase_only_branch_surface": phase_only_fields_next["surface"],
        "phase_only_branch_surface_balanced": phase_only_fields_next["balanced_surface"],
        "phase_only_branch_balance": phase_only_fields_next["support_balance"],
        "phase_only_branch_live_mask": phase_only_fields_next["slot2_live"],
        "child_worlds": child_worlds_next,
        "child_event_history": list(state.get("child_event_history", [])),
        "next_child_id": int(state.get("next_child_id", 0)),
    }
    state_next["mixed_phase_state"] = _mix_multimode_state(state_next, cfg)

    mixed_arc = hardy_littlewood_arc_field(state_next["mixed_phase_state"], cfg.qset, cfg.q_weights)
    mixed_promo = promotability_field(mixed_arc, persistence_momentum=cfg.persistence_momentum)
    step_metrics = {
        "split_mass": ambiguity.mean(),
        "merge_mass": merge_strength.mean(),
        "collapse_mass": collapse_mass.mean(),
        "mode_divergence": ((1.0 - phase_align) * (support_next[..., 0] * support_next[..., 1])).mean(),
        "mode_qtrace_divergence": (q0 - q1).abs().mean(),
        "slot2_live_fraction": (support_next[..., 1] > cfg.slot2_support_threshold).float().mean(),
        "branch_support_overlap": support_overlap.mean(),
        "dominant_mode_occupancy": _mode_occupancy(state_next, cfg).amax(dim=-1).mean(),
        "branch_fixation_penalty": (compat * occupancy_mid[..., 0] * (1.0 - support_next[..., 1])).mean(),
        "topology_mask_mean": topology_mask.mean(),
        "complexity_mask_mean": complexity_mask.mean(),
        "context_mask_mean": context_mask.mean(),
        "branch_defect_mean": defect_mask.mean(),
        "branch_world_grad_mean": (0.5 * world_major_grad + 0.5 * world_promo_grad).mean(),
        "branch_q_disagreement_mean": q_disagreement.mean(),
        "branch_phase_wall_mean": phase_wall.mean(),
        "branch_seed_energy": split_seed.abs().mean(),
        "branch_positive_mask": branch_pos.mean(),
        "branch_negative_mask": branch_neg.mean(),
        "phase_only_branch_surface_mean": phase_only_fields_next["surface"].mean(),
        "phase_only_branch_surface_peak": phase_only_fields_next["surface"].amax(),
        "phase_only_branch_phase_wall_mean": phase_only_fields_next["phase_wall"].mean(),
        "phase_only_branch_support_balance_mean": _weighted_field_mean(
            phase_only_fields_next["support_balance"],
            phase_only_fields_next["slot2_live"],
        ),
        "phase_only_branch_distinctness": _weighted_field_mean(
            phase_only_fields_next["phase_wall"],
            phase_only_fields_next["slot2_live"],
        ),
        "phase_only_branch_distinctness_balanced": _weighted_field_mean(
            phase_only_fields_next["phase_wall"],
            phase_only_fields_next["slot2_live"] * phase_only_fields_next["support_balance"],
        ),
        "phase_only_real_branch_fraction": (
            ((phase_only_fields_next["slot2_live"] > 0.0) & (phase_only_fields_next["phase_wall"] > cfg.real_branch_threshold))
            .to(mode0.dtype)
            .mean()
        ),
        "decorative_slot2_fraction": ((support_next[..., 1] > cfg.slot2_support_threshold).float() * (branch_pos < 0.10).float()).mean(),
        "relation_kernel_mean": relation_kernel.mean(),
        "relation_handoff_drive": relation_metrics["handoff_drive"].mean(),
        "relation_attn_10": relation_metrics["mean_attn_10"],
        "relation_attn_01": relation_metrics["mean_attn_01"],
        "child_world_count": child_metrics["child_world_count"],
        "live_child_fraction": child_metrics["live_child_fraction"],
        "child_age_mean": child_metrics["child_age_mean"],
        "child_writeback_mass": child_metrics["child_writeback_mass"],
        "child_writeback_gate_mass": child_metrics["child_writeback_gate_mass"],
        "child_phase_writeback_delta_mass": child_metrics["child_phase_writeback_delta_mass"],
        "child_parent_phase_writeback_delta_mass": child_metrics["child_parent_phase_writeback_delta_mass"],
        "child_support_writeback_mass": child_metrics["child_support_writeback_mass"],
        "child_assay_writeback_scale_mean": child_metrics["child_assay_writeback_scale_mean"],
        "child_assay_writeback_scale_min": child_metrics["child_assay_writeback_scale_min"],
        "child_logit_writeback_mass": child_metrics["child_logit_writeback_mass"],
        "child_qtrace_writeback_mass": child_metrics["child_qtrace_writeback_mass"],
        "child_parent_divergence": child_metrics["child_parent_divergence"],
        "child_sibling_divergence": child_metrics["child_sibling_divergence"],
        "child_local_ifs_step_count": child_metrics["child_local_ifs_step_count"],
        "child_local_ifs_packet_count": child_metrics["child_local_ifs_packet_count"],
        "child_local_ifs_phase_delta": child_metrics["child_local_ifs_phase_delta"],
        "child_local_ifs_support_mean": child_metrics["child_local_ifs_support_mean"],
        "child_local_ifs_coherence_mean": child_metrics["child_local_ifs_coherence_mean"],
        "child_local_ifs_coherence_raw_mean": child_metrics["child_local_ifs_coherence_raw_mean"],
        "child_local_ifs_coherence_retention_delta": child_metrics["child_local_ifs_coherence_retention_delta"],
        "child_local_ifs_coherence_floor_delta": child_metrics["child_local_ifs_coherence_floor_delta"],
        "child_local_ifs_parent_phase_delta": child_metrics["child_local_ifs_parent_phase_delta"],
        "child_local_ifs_causal_gate_mean": child_metrics["child_local_ifs_causal_gate_mean"],
        "child_local_ifs_causal_retention_loss": child_metrics["child_local_ifs_causal_retention_loss"],
        "child_local_ifs_causal_floor_loss": child_metrics["child_local_ifs_causal_floor_loss"],
        "defect_without_branch_penalty": child_metrics["defect_without_branch_penalty"],
        "branch_resolution_delay": child_metrics["branch_resolution_delay"],
    }
    block = {
        **mixed_arc,
        **mixed_promo,
        **step_metrics,
        "phase_only_branch_surface": phase_only_fields_next["surface"],
        "phase_only_branch_surface_balanced": phase_only_fields_next["balanced_surface"],
        "phase_only_branch_balance": phase_only_fields_next["support_balance"],
        "phase_only_branch_live_mask": phase_only_fields_next["slot2_live"],
    }
    packets: list[dict[str, torch.Tensor]] = []
    for mode_idx, packets_mode in enumerate(mode_packets):
        for packet in packets_mode:
            item = dict(packet)
            item["mode_index"] = torch.tensor([mode_idx], device=packet["batch_index"].device)
            packets.append(item)
    return state_next, block, packets


def circleworld_step(
    current: torch.Tensor | dict[str, torch.Tensor],
    cfg: CircleworldConfig,
    mode: str = "active_packets",
) -> tuple[torch.Tensor | dict[str, torch.Tensor], dict[str, torch.Tensor], list[dict[str, torch.Tensor]]]:
    if mode in {"native_multimode", "native_multimode_childworld"}:
        return _multimode_step(current, cfg)
    if not torch.is_tensor(current):
        current = extract_phase_state(current, cfg)
    arc = hardy_littlewood_arc_field(current, cfg.qset, cfg.q_weights)
    promo = promotability_field(arc, persistence_momentum=cfg.persistence_momentum)
    packets = build_promoted_packets(current, arc, promo, cfg)
    block = {**arc, **promo}

    if mode == "no_promotion":
        return current, block, packets
    if mode == "passive_packets":
        return apply_passive_packets(current, packets, gain=cfg.child_law_gain), block, packets
    if mode == "active_packets":
        return apply_active_packets(current, packets, cfg), block, packets
    raise ValueError(f"Unknown circleworld mode: {mode}")


def _mode_support_window_length(support: torch.Tensor, threshold: float) -> float:
    mask = support > threshold
    if not bool(mask.any().item()):
        return 0.0
    per_trace = mask.float().sum(dim=2)
    active = per_trace > 0
    return float(per_trace[active].mean().item()) if bool(active.any().item()) else 0.0


def _support_handoff_count(occupancy: torch.Tensor) -> float:
    dominant = occupancy.argmax(dim=-1)
    if dominant.size(2) <= 1:
        return 0.0
    handoff = (dominant[:, :, 1:] != dominant[:, :, :-1]).float().sum(dim=2)
    return float(handoff.mean().item())


def perturb_circleworld_state(
    state: torch.Tensor | dict[str, torch.Tensor],
    cfg: CircleworldConfig,
    kind: str,
    strength: float,
    seed: int,
) -> torch.Tensor | dict[str, torch.Tensor]:
    if torch.is_tensor(state):
        gen = torch.Generator(device=state.device)
        gen.manual_seed(int(seed))
        delta = float(strength) * torch.randn(
            (state.size(0), state.size(1), state.size(2)),
            device=state.device,
            dtype=state.dtype,
            generator=gen,
        )
        return phasor_apply_delta(state, delta)

    out = clone_circleworld_state(state)
    logits = out["mode_logits"]
    support = out["mode_support"]
    q_trace = out["mode_q_trace"]
    child_worlds = out.get("child_worlds", [])
    t_len = logits.size(2)
    span = max(1, int(round(t_len * cfg.mode_perturb_window_frac)))
    t0 = max(0, (t_len // 2) - (span // 2))
    t1 = min(t_len, t0 + span)
    if kind == "logit_tilt":
        logits[:, :, t0:t1, 1] = logits[:, :, t0:t1, 1] + float(strength)
    elif kind == "support_expand":
        support[:, :, t0:t1, 1] = (support[:, :, t0:t1, 1] + float(strength)).clamp(0.0, 1.0)
    elif kind == "support_suppress":
        support[:, :, t0:t1, 1] = (support[:, :, t0:t1, 1] * max(0.0, 1.0 - float(strength))).clamp(0.0, 1.0)
    elif kind == "qtrace_shift":
        gen = torch.Generator(device=q_trace.device)
        gen.manual_seed(int(seed))
        noise = 0.25 * float(strength) * torch.randn(
            q_trace[:, :, t0:t1, 1, :].shape,
            device=q_trace.device,
            dtype=q_trace.dtype,
            generator=gen,
        )
        q_trace[:, :, t0:t1, 1, :] = (q_trace[:, :, t0:t1, 1, :] + noise).clamp(-2.0, 2.0)
    elif kind in {
        "child_phase_shift",
        "child_support_boost",
        "child_qtrace_shift",
        "child_support_window_shift",
        "child_writeback_budget_shift",
        "child_parent_mix_shift",
        "child_survival_age_shift",
        "child_writeback_ready_shift",
    }:
        if not child_worlds:
            out["mixed_phase_state"] = _mix_multimode_state(out, cfg)
            return out
        def _child_score(child: dict[str, Any]) -> float:
            support_mass = float(child.get("mode_support", support[..., 1]).mean().item()) if torch.is_tensor(child.get("mode_support")) else 0.0
            coherence_mass = float(child.get("mode_coherence", support[..., 1]).mean().item()) if torch.is_tensor(child.get("mode_coherence")) else 0.0
            age = float(child.get("survival_age", 0.0))
            return age + support_mass + coherence_mass
        active_children = [child for child in child_worlds if child.get("active", False)]
        target_child = max(active_children or child_worlds, key=_child_score)
        child_t0, child_t1 = target_child.get("support_window", (t0, t1))
        child_t0 = max(0, min(int(child_t0), t_len - 1))
        child_t1 = max(child_t0 + 1, min(int(child_t1), t_len))
        if kind == "child_phase_shift":
            gen = torch.Generator(device=target_child["phase_state"].device)
            gen.manual_seed(int(seed))
            delta = 0.20 * float(strength) * torch.randn(
                target_child["phase_state"][..., child_t0:child_t1, 0].shape,
                device=target_child["phase_state"].device,
                dtype=target_child["phase_state"].dtype,
                generator=gen,
            )
            target_child["phase_state"][..., child_t0:child_t1, :] = phasor_apply_delta(
                target_child["phase_state"][..., child_t0:child_t1, :],
                delta,
            )
        elif kind == "child_support_boost":
            target_child["mode_support"][..., child_t0:child_t1] = (
                target_child["mode_support"][..., child_t0:child_t1] + float(strength)
            ).clamp(0.0, 1.0)
            target_child["mode_coherence"][..., child_t0:child_t1] = (
                target_child["mode_coherence"][..., child_t0:child_t1] + 0.5 * float(strength)
            ).clamp(0.0, 1.0)
        elif kind == "child_qtrace_shift":
            gen = torch.Generator(device=target_child["q_trace"].device)
            gen.manual_seed(int(seed))
            noise = 0.25 * float(strength) * torch.randn(
                target_child["q_trace"][..., child_t0:child_t1, :].shape,
                device=target_child["q_trace"].device,
                dtype=target_child["q_trace"].dtype,
                generator=gen,
            )
            target_child["q_trace"][..., child_t0:child_t1, :] = (
                target_child["q_trace"][..., child_t0:child_t1, :] + noise
            ).clamp(-2.0, 2.0)
        elif kind == "child_support_window_shift":
            current_len = max(1, child_t1 - child_t0)
            expand = max(1, int(round(current_len * max(0.5, float(strength)))))
            new_t0 = max(0, child_t0 - expand)
            new_t1 = min(t_len, child_t1 + expand)
            target_child["support_window"] = (int(new_t0), int(new_t1))
            widened = _child_support_mask(target_child["phase_state"], (new_t0, new_t1))
            target_child["mode_support"] = torch.maximum(
                target_child["mode_support"],
                (0.65 + 0.35 * float(strength)) * widened,
            ).clamp(0.0, 1.0)
            target_child["mode_coherence"] = torch.maximum(
                target_child["mode_coherence"],
                (0.55 + 0.45 * float(strength)) * widened,
            ).clamp(0.0, 1.0)
            target_child["survival_age"] = max(
                int(target_child.get("survival_age", 0)),
                int(cfg.child_min_age_for_writeback),
            )
        elif kind == "child_writeback_budget_shift":
            widened = _child_support_mask(target_child["phase_state"], (child_t0, child_t1))
            target_child["writeback_budget"] = float(target_child.get("writeback_budget", cfg.child_writeback_budget)) * (
                1.0 + 1.5 * float(strength)
            )
            target_child["survival_age"] = max(
                int(target_child.get("survival_age", 0)),
                int(cfg.child_min_age_for_writeback),
            )
            target_child["mode_support"] = torch.maximum(
                target_child["mode_support"],
                (0.50 + 0.50 * float(strength)) * widened,
            ).clamp(0.0, 1.0)
            target_child["mode_coherence"] = torch.maximum(
                target_child["mode_coherence"],
                (0.45 + 0.55 * float(strength)) * widened,
            ).clamp(0.0, 1.0)
            phase_parent = out["phase_modes"][..., 1, :]
            local_delta = _masked_mean(
                _phasor_phase_delta(target_child["phase_state"], phase_parent).abs(),
                target_child["mode_support"],
            )
            child_delta = local_delta.unsqueeze(-1) * target_child["mode_support"] * (0.35 + 0.65 * float(strength))
            out["phase_modes"][..., 1, :] = phasor_apply_delta(out["phase_modes"][..., 1, :], child_delta)
            out["phase_modes"][..., 0, :] = phasor_apply_delta(out["phase_modes"][..., 0, :], 0.35 * child_delta)
        elif kind == "child_parent_mix_shift":
            widened = _child_support_mask(target_child["phase_state"], (child_t0, child_t1))
            mix = (widened * (0.30 + 0.45 * float(strength))).clamp(0.0, 0.95)
            target_child["survival_age"] = max(
                int(target_child.get("survival_age", 0)),
                int(cfg.child_min_age_for_writeback),
            )
            out["phase_modes"][..., 1, :] = phasor_normalize(
                (1.0 - mix).unsqueeze(-1) * out["phase_modes"][..., 1, :]
                + mix.unsqueeze(-1) * target_child["phase_state"]
            )
            out["phase_modes"][..., 0, :] = phasor_normalize(
                (1.0 - 0.35 * mix).unsqueeze(-1) * out["phase_modes"][..., 0, :]
                + (0.35 * mix).unsqueeze(-1) * target_child["phase_state"]
            )
            out["mode_support"][..., 1] = torch.maximum(out["mode_support"][..., 1], mix).clamp(0.0, 1.0)
            out["mode_logits"][..., 1] = out["mode_logits"][..., 1] + 0.75 * float(strength) * mix
        elif kind == "child_survival_age_shift":
            widened = _child_support_mask(target_child["phase_state"], (child_t0, child_t1))
            bonus_age = max(1, int(round(2.0 * float(strength))))
            target_child["survival_age"] = max(
                int(target_child.get("survival_age", 0)),
                int(cfg.child_min_age_for_writeback) + bonus_age,
            )
            target_child["mode_support"] = torch.maximum(
                target_child["mode_support"],
                (0.60 + 0.35 * float(strength)) * widened,
            ).clamp(0.0, 1.0)
            target_child["mode_coherence"] = torch.maximum(
                target_child["mode_coherence"],
                (0.55 + 0.40 * float(strength)) * widened,
            ).clamp(0.0, 1.0)
            target_child["active"] = True
            target_child["collapsed"] = False
        elif kind == "child_writeback_ready_shift":
            widened = _child_support_mask(target_child["phase_state"], (child_t0, child_t1))
            target_child["writeback_budget"] = max(
                float(target_child.get("writeback_budget", cfg.child_writeback_budget)),
                float(cfg.child_writeback_budget) * (1.25 + 1.5 * float(strength)),
            )
            target_child["survival_age"] = max(
                int(target_child.get("survival_age", 0)),
                int(cfg.child_min_age_for_writeback) + 1,
            )
            target_child["mode_support"] = torch.maximum(
                target_child["mode_support"],
                (0.65 + 0.30 * float(strength)) * widened,
            ).clamp(0.0, 1.0)
            target_child["mode_coherence"] = torch.maximum(
                target_child["mode_coherence"],
                (0.60 + 0.35 * float(strength)) * widened,
            ).clamp(0.0, 1.0)
            phase_parent = out["phase_modes"][..., 1, :]
            local_delta = _masked_mean(
                _phasor_phase_delta(target_child["phase_state"], phase_parent).abs(),
                target_child["mode_support"],
            )
            child_delta = local_delta.unsqueeze(-1) * target_child["mode_support"] * (0.25 + 0.55 * float(strength))
            out["phase_modes"][..., 1, :] = phasor_apply_delta(out["phase_modes"][..., 1, :], child_delta)
            out["mode_support"][..., 1] = torch.maximum(
                out["mode_support"][..., 1],
                (0.35 + 0.40 * float(strength)) * widened,
            ).clamp(0.0, 1.0)
            out["mode_logits"][..., 1] = out["mode_logits"][..., 1] + 0.35 * float(strength) * widened
    else:
        raise ValueError(f"Unknown perturbation kind: {kind}")
    out["mixed_phase_state"] = _mix_multimode_state(out, cfg)
    return out


def recurse_circleworld(
    z: torch.Tensor,
    cfg: CircleworldConfig,
    depth: int | None = None,
    mode: str = "active_packets",
) -> dict[str, Any]:
    depth = cfg.recursion_depth if depth is None else depth
    if mode in {"native_multimode", "native_multimode_childworld"} or cfg.branching_mode in {"native_multimode", "native_multimode_childworld"}:
        mode = "native_multimode_childworld" if cfg.branching_mode == "native_multimode_childworld" or mode == "native_multimode_childworld" else "native_multimode"
        current: torch.Tensor | dict[str, torch.Tensor] = _init_multimode_state(phasor_normalize(z), cfg)
    else:
        current = phasor_normalize(z)
    history: list[dict[str, torch.Tensor]] = []
    all_packets: list[list[dict[str, torch.Tensor]]] = []
    multimode_history: list[dict[str, torch.Tensor]] = []

    for _ in range(max(1, depth)):
        current, block, packets = circleworld_step(current, cfg=cfg, mode=mode)
        history.append(block)
        all_packets.append(packets)
        if mode in {"native_multimode", "native_multimode_childworld"}:
            multimode_history.append(clone_circleworld_state(current))

    phase_state = extract_phase_state(current, cfg)
    final_arc = hardy_littlewood_arc_field(phase_state, cfg.qset, cfg.q_weights)
    final_promo = promotability_field(final_arc, persistence_momentum=cfg.persistence_momentum)
    law_packets: list[dict[str, torch.Tensor]] = []
    for depth_idx, (block, depth_packets) in enumerate(zip(history, all_packets)):
        law_packets.extend(promote_law_packets(depth_packets, block, {"persistence": block["persistence"], "promotability": block["promotability"]}, cfg, depth_idx))
    law_token_library = build_law_token_library(law_packets, cfg)
    out = {
        "phase_state": phase_state,
        "history": history,
        "packets": all_packets,
        "final_arc": final_arc,
        "final_promo": final_promo,
        "law_packets": law_packets,
        "law_token_library": law_token_library,
        "config": cfg,
        "mode": mode,
    }
    if mode in {"native_multimode", "native_multimode_childworld"}:
        out["multimode_state"] = current
        out["mixed_phase_state"] = phase_state
        out["multimode_history"] = multimode_history
        out["child_worlds"] = clone_circleworld_state(current.get("child_worlds", []))
        out["child_event_history"] = list(current.get("child_event_history", []))
    relational_signatures = extract_relational_signatures(out)
    out["relational_signatures"] = relational_signatures
    out["relational_signature_library"] = build_relational_signature_library(relational_signatures, cfg)
    return out


def summarize_circleworld_run(run: dict[str, Any]) -> dict[str, float]:
    initial = run["history"][0]
    final_arc = run["final_arc"]
    final_promo = run["final_promo"]
    packets = run["packets"]
    cfg = run.get("config")

    num_packets = float(sum(len(p) for p in packets))
    major_gain = (final_arc["major_mass"].mean() - initial["major_mass"].mean()).item()
    residue_drop = (initial["minor_residue"].mean() - final_arc["minor_residue"].mean()).item()
    promo_gain = (
        final_promo["promotability"].mean() - initial["promotability"].mean()
    ).item()
    per_q_abs = final_arc["per_q"].abs()
    q_mass = per_q_abs.mean(dim=(0, 2))
    q_mass = q_mass / q_mass.sum().clamp_min(1e-8)
    dominant_q_share = float(q_mass.max().item())
    q_entropy = float(
        (-(q_mass * (q_mass.clamp_min(1e-8).log())).sum() / torch.log(q_mass.new_tensor(float(q_mass.numel())))).item()
    )
    major_saturation = float(final_arc["major_mass"].clamp_min(0.92).sub(0.92).mean().item())
    prefix_alignment = 0.0
    prefix_delta = 0.0
    coarse_prefix_major = float(final_arc["major_mass"].mean().item())
    coarse_prefix_prom = float(final_promo["promotability"].mean().item())
    if cfg is not None and cfg.soft_matryoshka_enabled:
        prefixes = _prefix_metrics(run["phase_state"], cfg)
        if prefixes:
            weights = []
            for item in prefixes:
                frac = float(item["frac"])
                if frac <= 0.13:
                    weights.append(cfg.prefix_coarse_weight)
                elif frac <= 0.26:
                    weights.append(cfg.prefix_mid_weight)
                else:
                    weights.append(1.0 - cfg.prefix_coarse_weight - cfg.prefix_mid_weight)
            weight_sum = max(1e-8, sum(weights))
            full_major = float(final_arc["major_mass"].mean().item())
            full_residue = float(final_arc["minor_residue"].mean().item())
            full_promo = float(final_promo["promotability"].mean().item())
            align_terms = []
            delta_terms = []
            for item, weight in zip(prefixes, weights):
                align_terms.append(
                    weight
                    * (
                        abs(item["major_mass"] - full_major)
                        + 0.5 * abs(item["minor_residue"] - full_residue)
                        + 0.5 * abs(item["promotability"] - full_promo)
                    )
                )
                delta_terms.append(weight * abs(item["major_mass"] - full_major))
            prefix_alignment = float(max(0.0, 1.0 - sum(align_terms) / weight_sum))
            prefix_delta = float(sum(delta_terms) / weight_sum)
            coarse_prefix_major = float(prefixes[0]["major_mass"])
            coarse_prefix_prom = float(prefixes[0]["promotability"])

    out = {
        "initial_major_mass": float(initial["major_mass"].mean().item()),
        "final_major_mass": float(final_arc["major_mass"].mean().item()),
        "initial_minor_residue": float(initial["minor_residue"].mean().item()),
        "final_minor_residue": float(final_arc["minor_residue"].mean().item()),
        "initial_promotability": float(initial["promotability"].mean().item()),
        "final_promotability": float(final_promo["promotability"].mean().item()),
        "major_gain": float(major_gain),
        "residue_drop": float(residue_drop),
        "promotability_gain": float(promo_gain),
        "num_promotions": float(num_packets),
        "dominant_q_share": dominant_q_share,
        "q_entropy": q_entropy,
        "major_saturation": major_saturation,
        "prefix_alignment": prefix_alignment,
        "prefix_delta": prefix_delta,
        "coarse_prefix_major_mass": coarse_prefix_major,
        "coarse_prefix_promotability": coarse_prefix_prom,
    }
    law_lib = run.get("law_token_library")
    if isinstance(law_lib, dict):
        families = list(law_lib.get("families", []))
        dominant_law_family_share = 0.0
        law_family_entropy = 0.0
        dominant_law_q_share = 0.0
        law_top_q_entropy = 0.0
        num_law_top_q_unique = 0.0
        if families:
            counts = final_arc["major_mass"].new_tensor([float(f["count"]) for f in families])
            probs = counts / counts.sum().clamp_min(1e-8)
            dominant_law_family_share = float(probs.max().item())
            if probs.numel() > 1:
                law_family_entropy = float(
                    (-(probs * probs.clamp_min(1e-8).log()).sum() / torch.log(probs.new_tensor(float(probs.numel())))).item()
                )
            weighted_q = final_arc["major_mass"].new_zeros(len(cfg.qset))
            top_q_counts = final_arc["major_mass"].new_zeros(len(cfg.qset))
            for family, count in zip(families, counts):
                q_mass = family["prototype_q_mass"]
                if not torch.is_tensor(q_mass):
                    q_mass = final_arc["major_mass"].new_tensor(q_mass)
                q_mass = q_mass / q_mass.sum().clamp_min(1e-8)
                weighted_q = weighted_q + count * q_mass
                top_q_counts[int(q_mass.argmax().item())] += count
            weighted_q = weighted_q / weighted_q.sum().clamp_min(1e-8)
            dominant_law_q_share = float(weighted_q.max().item())
            top_q_probs = top_q_counts / top_q_counts.sum().clamp_min(1e-8)
            num_law_top_q_unique = float((top_q_counts > 0).float().sum().item())
            if top_q_probs.numel() > 1:
                law_top_q_entropy = float(
                    (-(top_q_probs * top_q_probs.clamp_min(1e-8).log()).sum() / torch.log(top_q_probs.new_tensor(float(top_q_probs.numel())))).item()
                )
        out.update(
            {
                "num_law_packets": float(law_lib.get("num_law_packets", 0)),
                "num_law_families": float(law_lib.get("num_families", 0)),
                "mean_law_family_size": float(law_lib.get("mean_family_size", 0.0)),
                "dominant_law_family_share": dominant_law_family_share,
                "law_family_entropy": law_family_entropy,
                "dominant_law_q_share": dominant_law_q_share,
                "law_top_q_entropy": law_top_q_entropy,
                "num_law_top_q_unique": num_law_top_q_unique,
            }
        )
    rel_bank = run.get("relational_signatures")
    rel_library = run.get("relational_signature_library")
    if rel_bank is not None:
        out.update(summarize_relational_signature_bank(rel_bank, rel_library if isinstance(rel_library, dict) else None))
    if run.get("mode") in {"native_multimode", "native_multimode_childworld"} and "multimode_state" in run:
        state = run["multimode_state"]
        occupancy = _mode_occupancy(state, cfg)
        support = state["mode_support"]
        phase_modes = state["phase_modes"]
        q_trace = state["mode_q_trace"]
        phase_only_fields = _phase_only_branch_fields(phase_modes[..., 0, :], phase_modes[..., 1, :], support, cfg)
        support_overlap = (support[..., 0] * support[..., 1]).clamp(0.0, 1.0)
        phase_div_field = (1.0 - ((_phase_alignment(phase_modes[..., 0, :], phase_modes[..., 1, :]) + 1.0) * 0.5)) * support_overlap
        phase_div = phase_div_field.mean()
        q_div = ((q_trace[..., 0, :] - q_trace[..., 1, :]).abs().mean(dim=-1) * support_overlap).mean()
        slot2_live = (support[..., 1] > cfg.slot2_support_threshold).float()
        phase_only_branch_mask = (phase_only_fields["slot2_live"] > 0.0) & (
            phase_only_fields["phase_wall"] > cfg.real_branch_threshold
        )
        phase_only_real_branch = phase_only_branch_mask.to(support.dtype).mean()
        phase_only_excess_branch = (
            (phase_only_branch_mask & (phase_div_field.detach() <= cfg.real_branch_threshold)).to(support.dtype).mean()
        )
        decorative_slot2_low_phase = (
            ((slot2_live > 0.5) & (phase_only_fields["phase_wall"] <= cfg.real_branch_threshold)).to(support.dtype).mean()
            if slot2_live.numel() > 0
            else support.new_tensor(0.0)
        )
        real_branch = (
            ((slot2_live > 0.5) & (phase_div_field.detach() > cfg.real_branch_threshold)).float().mean()
            if slot2_live.numel() > 0
            else support.new_tensor(0.0)
        )
        step_metrics = run["history"]
        child_real_branch = 0.0
        child_meso_effect = 0.0
        if "child_world_count" in step_metrics[0]:
            child_live_fraction = float(torch.stack([blk["live_child_fraction"] for blk in step_metrics]).mean().item())
            child_parent_div = float(torch.stack([blk["child_parent_divergence"] for blk in step_metrics]).mean().item())
            child_writeback = float(torch.stack([blk["child_writeback_mass"] for blk in step_metrics]).mean().item())
            child_meso_effect = child_parent_div * max(0.05, child_writeback)
            child_branch_ready = (
                child_live_fraction >= cfg.child_branch_live_threshold
                and child_parent_div >= cfg.child_branch_parent_threshold
                and child_writeback >= cfg.child_branch_writeback_threshold
                and child_meso_effect >= cfg.child_branch_meso_threshold
            )
            child_real_branch = child_live_fraction if child_branch_ready else 0.0
        out.update(
            {
                "mean_active_mode_count": float(((support > cfg.slot2_support_threshold).float().sum(dim=-1)).mean().item()),
                "mean_slot2_live_fraction": float(slot2_live.mean().item()),
                "mean_slot2_support_mass": float(support[..., 1].mean().item()),
                "mean_slot2_persistence": float(slot2_live.mean(dim=2).mean().item()),
                "mean_branch_support_overlap": float(support_overlap.mean().item()),
                "mean_split_events": float(torch.stack([blk["split_mass"] for blk in step_metrics]).mean().item()),
                "mean_merge_events": float(torch.stack([blk["merge_mass"] for blk in step_metrics]).mean().item()),
                "mean_collapse_events": float(torch.stack([blk["collapse_mass"] for blk in step_metrics]).mean().item()),
                "mean_mode_divergence": float(phase_div.item()),
                "mean_mode_qtrace_divergence": float(q_div.item()),
                "mean_dominant_mode_occupancy": float(occupancy.amax(dim=-1).mean().item()),
                "mean_support_window_length": float(_mode_support_window_length(support[..., 1], cfg.slot2_support_threshold)),
                "mean_support_handoff_count": float(_support_handoff_count(occupancy)),
                "mean_branch_fixation_penalty": float(torch.stack([blk["branch_fixation_penalty"] for blk in step_metrics]).mean().item()),
                "mean_topology_mask": float(torch.stack([blk["topology_mask_mean"] for blk in step_metrics]).mean().item()),
                "mean_complexity_mask": float(torch.stack([blk["complexity_mask_mean"] for blk in step_metrics]).mean().item()),
                "mean_context_mask": float(torch.stack([blk["context_mask_mean"] for blk in step_metrics]).mean().item()),
                "mean_branch_defect": float(torch.stack([blk["branch_defect_mean"] for blk in step_metrics]).mean().item()),
                "mean_branch_world_grad": float(torch.stack([blk["branch_world_grad_mean"] for blk in step_metrics]).mean().item()),
                "mean_branch_q_disagreement": float(torch.stack([blk["branch_q_disagreement_mean"] for blk in step_metrics]).mean().item()),
                "mean_branch_phase_wall": float(torch.stack([blk["branch_phase_wall_mean"] for blk in step_metrics]).mean().item()),
                "mean_branch_seed_energy": float(torch.stack([blk["branch_seed_energy"] for blk in step_metrics]).mean().item()),
                "mean_branch_positive_mask": float(torch.stack([blk["branch_positive_mask"] for blk in step_metrics]).mean().item()),
                "mean_branch_negative_mask": float(torch.stack([blk["branch_negative_mask"] for blk in step_metrics]).mean().item()),
                "mean_phase_only_branch_surface": float(torch.stack([blk["phase_only_branch_surface_mean"] for blk in step_metrics]).mean().item()),
                "mean_phase_only_branch_surface_peak": float(torch.stack([blk["phase_only_branch_surface_peak"] for blk in step_metrics]).mean().item()),
                "mean_phase_only_branch_phase_wall": float(torch.stack([blk["phase_only_branch_phase_wall_mean"] for blk in step_metrics]).mean().item()),
                "mean_phase_only_branch_support_balance": float(torch.stack([blk["phase_only_branch_support_balance_mean"] for blk in step_metrics]).mean().item()),
                "mean_phase_only_branch_distinctness": float(torch.stack([blk["phase_only_branch_distinctness"] for blk in step_metrics]).mean().item()),
                "mean_phase_only_branch_distinctness_balanced": float(
                    torch.stack([blk["phase_only_branch_distinctness_balanced"] for blk in step_metrics]).mean().item()
                ),
                "mean_decorative_slot2_fraction": float(torch.stack([blk["decorative_slot2_fraction"] for blk in step_metrics]).mean().item()),
                "mean_relation_kernel": float(torch.stack([blk["relation_kernel_mean"] for blk in step_metrics]).mean().item()),
                "mean_relation_handoff_drive": float(torch.stack([blk["relation_handoff_drive"] for blk in step_metrics]).mean().item()),
                "mean_relation_attn_10": float(torch.stack([blk["relation_attn_10"] for blk in step_metrics]).mean().item()),
                "mean_relation_attn_01": float(torch.stack([blk["relation_attn_01"] for blk in step_metrics]).mean().item()),
                "mean_child_world_count": float(torch.stack([blk["child_world_count"] for blk in step_metrics]).mean().item()),
                "mean_live_child_fraction": float(torch.stack([blk["live_child_fraction"] for blk in step_metrics]).mean().item()),
                "mean_child_age": float(torch.stack([blk["child_age_mean"] for blk in step_metrics]).mean().item()),
                "mean_child_writeback_mass": float(torch.stack([blk["child_writeback_mass"] for blk in step_metrics]).mean().item()),
                "mean_child_writeback_gate_mass": float(
                    torch.stack([blk["child_writeback_gate_mass"] for blk in step_metrics]).mean().item()
                ),
                "mean_child_phase_writeback_delta_mass": float(
                    torch.stack([blk["child_phase_writeback_delta_mass"] for blk in step_metrics]).mean().item()
                ),
                "mean_child_parent_phase_writeback_delta_mass": float(
                    torch.stack([blk["child_parent_phase_writeback_delta_mass"] for blk in step_metrics]).mean().item()
                ),
                "mean_child_support_writeback_mass": float(
                    torch.stack([blk["child_support_writeback_mass"] for blk in step_metrics]).mean().item()
                ),
                "mean_child_assay_writeback_scale": float(
                    torch.stack([blk["child_assay_writeback_scale_mean"] for blk in step_metrics]).mean().item()
                ),
                "min_child_assay_writeback_scale": float(
                    torch.stack([blk["child_assay_writeback_scale_min"] for blk in step_metrics]).min().item()
                ),
                "mean_child_logit_writeback_mass": float(
                    torch.stack([blk["child_logit_writeback_mass"] for blk in step_metrics]).mean().item()
                ),
                "mean_child_qtrace_writeback_mass": float(
                    torch.stack([blk["child_qtrace_writeback_mass"] for blk in step_metrics]).mean().item()
                ),
                "mean_child_parent_divergence": float(torch.stack([blk["child_parent_divergence"] for blk in step_metrics]).mean().item()),
                "mean_child_sibling_divergence": float(torch.stack([blk["child_sibling_divergence"] for blk in step_metrics]).mean().item()),
                "mean_child_local_ifs_step_count": float(
                    torch.stack([blk["child_local_ifs_step_count"] for blk in step_metrics]).mean().item()
                ),
                "mean_child_local_ifs_packet_count": float(
                    torch.stack([blk["child_local_ifs_packet_count"] for blk in step_metrics]).mean().item()
                ),
                "mean_child_local_ifs_phase_delta": float(
                    torch.stack([blk["child_local_ifs_phase_delta"] for blk in step_metrics]).mean().item()
                ),
                "mean_child_local_ifs_support": float(
                    torch.stack([blk["child_local_ifs_support_mean"] for blk in step_metrics]).mean().item()
                ),
                "mean_child_local_ifs_coherence": float(
                    torch.stack([blk["child_local_ifs_coherence_mean"] for blk in step_metrics]).mean().item()
                ),
                "mean_child_local_ifs_coherence_raw": float(
                    torch.stack([blk["child_local_ifs_coherence_raw_mean"] for blk in step_metrics]).mean().item()
                ),
                "mean_child_local_ifs_coherence_retention_delta": float(
                    torch.stack([blk["child_local_ifs_coherence_retention_delta"] for blk in step_metrics]).mean().item()
                ),
                "mean_child_local_ifs_coherence_floor_delta": float(
                    torch.stack([blk["child_local_ifs_coherence_floor_delta"] for blk in step_metrics]).mean().item()
                ),
                "mean_child_local_ifs_parent_phase_delta": float(
                    torch.stack([blk["child_local_ifs_parent_phase_delta"] for blk in step_metrics]).mean().item()
                ),
                "mean_child_local_ifs_causal_gate": float(
                    torch.stack([blk["child_local_ifs_causal_gate_mean"] for blk in step_metrics]).mean().item()
                ),
                "mean_child_local_ifs_causal_retention_loss": float(
                    torch.stack([blk["child_local_ifs_causal_retention_loss"] for blk in step_metrics]).mean().item()
                ),
                "mean_child_local_ifs_causal_floor_loss": float(
                    torch.stack([blk["child_local_ifs_causal_floor_loss"] for blk in step_metrics]).mean().item()
                ),
                "mean_defect_without_branch_penalty": float(torch.stack([blk["defect_without_branch_penalty"] for blk in step_metrics]).mean().item()),
                "mean_branch_resolution_delay": float(torch.stack([blk["branch_resolution_delay"] for blk in step_metrics]).mean().item()),
                "parent_real_branch_fraction": float(real_branch.item()) if torch.is_tensor(real_branch) else float(real_branch),
                "child_real_branch_fraction": float(child_real_branch),
                "real_branch_fraction": max(float(real_branch.item()) if torch.is_tensor(real_branch) else float(real_branch), child_real_branch),
                "phase_only_real_branch_fraction": float(phase_only_real_branch.item()),
                "phase_only_excess_branch_fraction": float(phase_only_excess_branch.item()),
                "decorative_slot2_low_phase_fraction": float(decorative_slot2_low_phase.item()),
                "final_phase_only_branch_surface": float(phase_only_fields["surface"].mean().item()),
                "final_phase_only_branch_surface_peak": float(phase_only_fields["surface"].amax().item()),
                "final_phase_only_branch_phase_wall": float(phase_only_fields["phase_wall"].mean().item()),
                "final_phase_only_branch_live_fraction": float(phase_only_fields["slot2_live"].mean().item()),
                "final_phase_only_branch_support_balance": float(
                    _weighted_field_mean(phase_only_fields["support_balance"], phase_only_fields["slot2_live"]).item()
                ),
                "final_phase_only_branch_distinctness": float(
                    _weighted_field_mean(phase_only_fields["phase_wall"], phase_only_fields["slot2_live"]).item()
                ),
                "final_phase_only_branch_distinctness_balanced": float(
                    _weighted_field_mean(
                        phase_only_fields["phase_wall"],
                        phase_only_fields["slot2_live"] * phase_only_fields["support_balance"],
                    ).item()
                ),
                "meso_branch_effect": max(float((phase_div * support[..., 1].mean()).item()), child_meso_effect),
                "silent_singlepath_fraction": float(1.0 - slot2_live.mean().item()),
            }
        )
        child_history = _summarize_child_event_history(run.get("child_event_history", []))
        out.update(
            {
                "child_spawn_count": float(child_history["spawn_count"]),
                "child_kill_count": float(child_history["kill_count"]),
                "child_writeback_count": float(child_history["writeback_count"]),
                "child_local_ifs_count": float(child_history["local_ifs_count"]),
                "child_max_age": float(child_history["max_age"]),
            }
        )
    return out


def circleworld_loss(
    run: dict[str, Any],
    target_major_mass: float = 0.85,
    target_promotability: float = 0.65,
    target_attack: float = 0.05,
) -> dict[str, torch.Tensor]:
    final_arc = run["final_arc"]
    final_promo = run["final_promo"]
    initial = run["history"][0]
    cfg = run.get("config")

    major = final_arc["major_mass"].mean()
    residue = final_arc["minor_residue"].mean()
    promo = final_promo["promotability"].mean()
    initial_major = initial["major_mass"]
    final_major = final_arc["major_mass"]
    attack = (final_major[..., 1:] - final_major[..., :-1]).clamp_min(0.0).mean()
    per_q_abs = final_arc["per_q"].abs()
    q_mass = per_q_abs.mean(dim=(0, 2))
    q_mass = q_mass / q_mass.sum().clamp_min(1e-8)
    dominant_q_share = q_mass.max()
    q_entropy = -(q_mass * q_mass.clamp_min(1e-8).log()).sum() / torch.log(
        q_mass.new_tensor(float(q_mass.numel()))
    )
    major_saturation = final_major.mean()

    l_major = F.mse_loss(major, major.new_tensor(target_major_mass))
    l_residue = residue
    l_promo = F.mse_loss(promo, promo.new_tensor(target_promotability))
    l_attack = F.mse_loss(attack, attack.new_tensor(target_attack))
    l_gain = F.relu(initial_major.mean() - major)
    l_q_dom = F.relu(dominant_q_share - dominant_q_share.new_tensor(0.55))
    l_q_entropy = F.relu(q_entropy.new_tensor(0.60) - q_entropy)
    l_major_sat = F.relu(major_saturation - major_saturation.new_tensor(0.92))
    l_prefix = major.new_tensor(0.0)
    if cfg is not None and cfg.soft_matryoshka_enabled:
        prefixes = _prefix_metrics(run["phase_state"], cfg)
        if prefixes:
            full_major = major.detach()
            full_promo = promo.detach()
            full_residue = residue.detach()
            parts = []
            for idx, item in enumerate(prefixes):
                weight = cfg.prefix_coarse_weight if idx == 0 else (
                    cfg.prefix_mid_weight if idx == 1 else max(0.05, 1.0 - cfg.prefix_coarse_weight - cfg.prefix_mid_weight)
                )
                item_major = major.new_tensor(item["major_mass"])
                item_promo = major.new_tensor(item["promotability"])
                item_residue = major.new_tensor(item["minor_residue"])
                parts.append(
                    weight
                    * (
                        F.mse_loss(item_major, full_major)
                        + 0.5 * F.mse_loss(item_promo, full_promo)
                        + 0.5 * F.mse_loss(item_residue, full_residue)
                    )
                )
            l_prefix = sum(parts) / max(1, len(parts))
    l_branch = major.new_tensor(0.0)
    if run.get("mode") in {"native_multimode", "native_multimode_childworld"} and "multimode_state" in run and cfg is not None:
        state = run["multimode_state"]
        occupancy = _mode_occupancy(state, cfg)
        support = state["mode_support"]
        slot2_live = (support[..., 1] > cfg.slot2_support_threshold).float().mean()
        phase_align = ((_phase_alignment(state["phase_modes"][..., 0, :], state["phase_modes"][..., 1, :]) + 1.0) * 0.5)
        phase_div = ((1.0 - phase_align) * support[..., 0] * support[..., 1]).mean()
        branch_pos = torch.stack([blk["branch_positive_mask"] for blk in run["history"]]).mean()
        branch_neg = torch.stack([blk["branch_negative_mask"] for blk in run["history"]]).mean()
        branch_defect = torch.stack([blk["branch_defect_mean"] for blk in run["history"]]).mean()
        branch_seed_energy = torch.stack([blk["branch_seed_energy"] for blk in run["history"]]).mean()
        decorative_slot2 = torch.stack([blk["decorative_slot2_fraction"] for blk in run["history"]]).mean()
        relation_handoff = torch.stack([blk["relation_handoff_drive"] for blk in run["history"]]).mean()
        child_world_count = torch.stack([blk["child_world_count"] for blk in run["history"]]).mean()
        child_live_fraction = torch.stack([blk["live_child_fraction"] for blk in run["history"]]).mean()
        child_age = torch.stack([blk["child_age_mean"] for blk in run["history"]]).mean()
        child_writeback = torch.stack([blk["child_writeback_mass"] for blk in run["history"]]).mean()
        child_parent_div = torch.stack([blk["child_parent_divergence"] for blk in run["history"]]).mean()
        child_sibling_div = torch.stack([blk["child_sibling_divergence"] for blk in run["history"]]).mean()
        defect_without_branch = torch.stack([blk["defect_without_branch_penalty"] for blk in run["history"]]).mean()
        branch_delay = torch.stack([blk["branch_resolution_delay"] for blk in run["history"]]).mean()
        l_branch = (
            0.35 * F.relu(slot2_live.new_tensor(0.03) - slot2_live)
            + 0.35 * F.relu(phase_div.new_tensor(0.05) - phase_div)
            + 0.30 * F.relu(occupancy.amax(dim=-1).mean() - occupancy.new_tensor(0.985))
            + 0.20 * decorative_slot2
            + 0.20 * F.relu(branch_pos.new_tensor(0.10) - branch_pos)
            + 0.15 * branch_neg
            + 0.15 * F.relu(relation_handoff.new_tensor(0.02) - relation_handoff)
            + 0.28 * F.relu(child_live_fraction.new_tensor(0.08) - child_live_fraction)
            + 0.25 * F.relu(child_age.new_tensor(float(cfg.child_min_age_for_writeback)) - child_age)
            + 0.22 * F.relu(child_writeback.new_tensor(0.015) - child_writeback)
            + 0.20 * F.relu(child_parent_div.new_tensor(0.04) - child_parent_div)
            + 0.18 * F.relu(child_sibling_div.new_tensor(0.04) - child_sibling_div)
            + 0.12 * F.relu(child_world_count.new_tensor(1.0) - child_world_count)
            + 0.25 * defect_without_branch
            - 0.08 * branch_defect
            - 0.05 * branch_seed_energy
            - 0.06 * branch_delay
        )

    total = (
        l_major
        + 0.75 * l_residue
        + 0.5 * l_promo
        + 0.25 * l_attack
        + 0.5 * l_gain
        + 0.75 * l_q_dom
        + 0.35 * l_q_entropy
        + 0.35 * l_major_sat
        + 0.45 * l_prefix
        + 0.40 * l_branch
    )

    return {
        "loss": total,
        "l_major": l_major,
        "l_residue": l_residue,
        "l_promo": l_promo,
        "l_attack": l_attack,
        "l_gain": l_gain,
        "l_q_dom": l_q_dom,
        "l_q_entropy": l_q_entropy,
        "l_major_sat": l_major_sat,
        "l_prefix": l_prefix,
        "l_branch": l_branch,
    }
