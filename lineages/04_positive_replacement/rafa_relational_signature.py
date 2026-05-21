from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F


@dataclass
class RafaRelationalSignatureConfig:
    signature_dim: int = 768
    prefix_dims: tuple[int, ...] = (128, 256, 384, 512, 640, 768)
    operator_seed_dim: int = 128
    branch_profile_dim: int = 4
    library_merge_threshold: float = 0.92
    library_topk_families: int = 8
    residual_mix: float = 0.22


@dataclass
class RafaRelationalSignatureV0:
    h: torch.Tensor
    q_profile: torch.Tensor
    arc_profile: torch.Tensor
    temporal_profile: torch.Tensor
    support_profile: torch.Tensor
    branch_profile: torch.Tensor
    operator_seed: torch.Tensor
    confidence: torch.Tensor
    packet_index: torch.Tensor
    depth_index: torch.Tensor
    time_index: torch.Tensor
    prefix_dims: tuple[int, ...]
    coarse_profile: torch.Tensor | None = None

    @property
    def num_signatures(self) -> int:
        return int(self.h.size(0))

    def prefix_view(self, dim: int) -> torch.Tensor:
        dim = max(1, min(int(dim), self.h.size(-1)))
        return self.h[:, :dim]


@dataclass
class RelationalFactorPack:
    coarse_profile: torch.Tensor
    q_profile: torch.Tensor
    arc_profile: torch.Tensor
    temporal_profile: torch.Tensor
    support_profile: torch.Tensor
    branch_profile: torch.Tensor
    law_signature: torch.Tensor
    confidence: torch.Tensor
    operator_seed: torch.Tensor

    def validate(self, sig_cfg: RafaRelationalSignatureConfig | None = None) -> RelationalFactorPack:
        tensors = {
            "coarse_profile": self.coarse_profile,
            "q_profile": self.q_profile,
            "arc_profile": self.arc_profile,
            "temporal_profile": self.temporal_profile,
            "support_profile": self.support_profile,
            "law_signature": self.law_signature,
            "confidence": self.confidence,
            "operator_seed": self.operator_seed,
        }
        for name, value in tensors.items():
            if not torch.is_tensor(value):
                raise TypeError(f"relational factor '{name}' must be a torch.Tensor")
            if value.dim() != 2:
                raise ValueError(f"relational factor '{name}' must have shape [N, D], got {tuple(value.shape)}")
        if not torch.is_tensor(self.branch_profile):
            raise TypeError("relational factor 'branch_profile' must be a torch.Tensor")
        if self.branch_profile.dim() != 3:
            raise ValueError(f"relational factor 'branch_profile' must have shape [N, modes, D], got {tuple(self.branch_profile.shape)}")

        batch = int(self.q_profile.size(0))
        for name, value in {**tensors, "branch_profile": self.branch_profile}.items():
            if int(value.size(0)) != batch:
                raise ValueError(f"relational factor '{name}' batch mismatch: expected {batch}, got {int(value.size(0))}")
        if self.confidence.size(-1) != 1:
            raise ValueError(f"relational factor 'confidence' must have shape [N, 1], got {tuple(self.confidence.shape)}")
        if sig_cfg is not None:
            expected_seed_dim = int(sig_cfg.operator_seed_dim)
            if self.operator_seed.size(-1) != expected_seed_dim:
                raise ValueError(
                    f"relational factor 'operator_seed' dim mismatch: expected {expected_seed_dim}, got {int(self.operator_seed.size(-1))}"
                )
            expected_branch_dim = int(sig_cfg.branch_profile_dim)
            if self.branch_profile.size(-1) != expected_branch_dim:
                raise ValueError(
                    f"relational factor 'branch_profile' dim mismatch: expected {expected_branch_dim}, got {int(self.branch_profile.size(-1))}"
                )
        return self

    def named_factors(self) -> dict[str, torch.Tensor]:
        return {
            "coarse": self.coarse_profile,
            "coarse_profile": self.coarse_profile,
            "q": self.q_profile,
            "q_profile": self.q_profile,
            "arc": self.arc_profile,
            "arc_profile": self.arc_profile,
            "temporal": self.temporal_profile,
            "temporal_profile": self.temporal_profile,
            "support": self.support_profile,
            "support_profile": self.support_profile,
            "branch": self.branch_profile,
            "branch_profile": self.branch_profile,
            "law_signature": self.law_signature,
            "confidence": self.confidence,
            "operator": self.operator_seed,
            "operator_seed": self.operator_seed,
        }


def signature_prefix_contract(sig_cfg: RafaRelationalSignatureConfig | None = None) -> dict[str, Any]:
    sig_cfg = sig_cfg or RafaRelationalSignatureConfig()
    return {
        "signature_dim": int(sig_cfg.signature_dim),
        "prefix_dims": list(sig_cfg.prefix_dims),
        "operator_seed_dim": int(sig_cfg.operator_seed_dim),
        "branch_profile_dim": int(sig_cfg.branch_profile_dim),
        "library_merge_threshold": float(sig_cfg.library_merge_threshold),
        "library_topk_families": int(sig_cfg.library_topk_families),
        "residual_mix": float(sig_cfg.residual_mix),
    }


def _empty_bank(
    q_bins: int,
    num_modes: int,
    sig_cfg: RafaRelationalSignatureConfig,
) -> RafaRelationalSignatureV0:
    branch_dim = sig_cfg.branch_profile_dim
    return RafaRelationalSignatureV0(
        h=torch.zeros(0, sig_cfg.signature_dim),
        q_profile=torch.zeros(0, q_bins),
        arc_profile=torch.zeros(0, 5),
        temporal_profile=torch.zeros(0, 8),
        support_profile=torch.zeros(0, 6),
        branch_profile=torch.zeros(0, num_modes, branch_dim),
        operator_seed=torch.zeros(0, sig_cfg.operator_seed_dim),
        confidence=torch.zeros(0, 1),
        packet_index=torch.zeros(0, dtype=torch.long),
        depth_index=torch.zeros(0, dtype=torch.long),
        time_index=torch.zeros(0, dtype=torch.long),
        prefix_dims=tuple(sig_cfg.prefix_dims),
        coarse_profile=None,
    )


def _normalize_rows(x: torch.Tensor) -> torch.Tensor:
    if x.numel() == 0:
        return x
    return F.normalize(x, dim=-1)


def _stretch_features(features: torch.Tensor, out_dim: int) -> torch.Tensor:
    if features.size(-1) == out_dim:
        return features
    return F.interpolate(
        features.unsqueeze(1),
        size=int(out_dim),
        mode="linear",
        align_corners=True,
    ).squeeze(1)


def _phase_alignment(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return (a[..., 0] * b[..., 0] + a[..., 1] * b[..., 1]).clamp(-1.0, 1.0)


def _cfg_qset(cfg: Any, q_bins: int) -> torch.Tensor:
    qset = tuple(getattr(cfg, "qset", tuple(range(1, q_bins + 1))))
    if len(qset) != q_bins:
        qset = tuple(range(1, q_bins + 1))
    return torch.tensor(qset, dtype=torch.float32)


def _safe_entropy(prob: torch.Tensor) -> torch.Tensor:
    if prob.numel() <= 1:
        return prob.new_tensor(0.0)
    denom = torch.log(prob.new_tensor(float(prob.numel()))).clamp_min(1e-8)
    return (-(prob * prob.clamp_min(1e-8).log()).sum() / denom).clamp(0.0, 1.0)


def _packet_branch_profile(
    run: dict[str, Any],
    packet: dict[str, torch.Tensor],
    num_modes: int,
    branch_dim: int,
) -> torch.Tensor:
    base = packet["q_mass"]
    profile = torch.zeros(num_modes, branch_dim, device=base.device, dtype=base.dtype)
    profile[0, 0] = 1.0
    profile[0, 1] = 1.0
    profile[0, 2] = 1.0
    profile[0, 3] = 1.0
    multimode_history = run.get("multimode_history")
    if not multimode_history:
        return profile

    depth_idx = int(packet.get("depth_index", torch.tensor([0])).item())
    if depth_idx >= len(multimode_history):
        return profile
    state = multimode_history[depth_idx]
    phase_modes = state.get("phase_modes")
    mode_logits = state.get("mode_logits")
    mode_support = state.get("mode_support")
    mode_q_trace = state.get("mode_q_trace")
    if phase_modes is None or mode_logits is None or mode_support is None or mode_q_trace is None:
        return profile

    b = int(packet["batch_index"].item())
    window = packet.get("law_window")
    if window is None:
        t_idx = int(packet["time_index"].item())
        t0, t1 = max(0, t_idx - 4), t_idx + 4
    else:
        t0 = int(window[0].item())
        t1 = int(window[1].item())

    logits = torch.softmax(mode_logits[b, :, t0:t1, :], dim=-1).mean(dim=(0, 1))
    support = mode_support[b, :, t0:t1, :].mean(dim=(0, 1))
    q_trace = mode_q_trace[b, :, t0:t1, :, :].mean(dim=(0, 1))
    phase_window = phase_modes[b, :, t0:t1, :, :]
    ref = phase_window[..., 0, :]
    coherence = []
    observed_modes = phase_window.size(-2)
    for k in range(observed_modes):
        aligned = (_phase_alignment(phase_window[..., k, :], ref) + 1.0) * 0.5
        coherence.append(aligned.mean())
    coherence_t = torch.stack(coherence, dim=0)
    q_presence = q_trace.mean(dim=-1)
    q_presence = q_presence / q_presence.amax().clamp_min(1e-8)

    use_modes = min(num_modes, observed_modes)
    profile[:use_modes, 0] = logits[:use_modes]
    profile[:use_modes, 1] = support[:use_modes]
    profile[:use_modes, 2] = coherence_t[:use_modes]
    profile[:use_modes, 3] = q_presence[:use_modes]
    return profile.clamp(0.0, 1.0)


def _packet_support_profile(
    packet: dict[str, torch.Tensor],
    q_profile: torch.Tensor,
    branch_profile: torch.Tensor,
    cfg: Any,
) -> torch.Tensor:
    q_bins = q_profile.numel()
    qset = _cfg_qset(cfg, q_bins).to(q_profile.device, q_profile.dtype)
    t0 = int(packet["law_window"][0].item()) if "law_window" in packet else int(packet["time_index"].item())
    t1 = int(packet["law_window"][1].item()) if "law_window" in packet else t0 + 1
    t_len = max(t1, int(getattr(cfg, "child_support_window", 1)))
    window_width = max(1, t1 - t0)
    window_center = t0 + 0.5 * window_width
    q_center = (q_profile * qset).sum() / qset.amax().clamp_min(1.0)
    q_spread = torch.sqrt((q_profile * (qset - (q_profile * qset).sum()).pow(2)).sum().clamp_min(1e-8))
    q_spread = q_spread / qset.amax().clamp_min(1.0)
    support_mass = branch_profile[..., 1].mean()
    locality = 1.0 - min(1.0, float(window_width) / float(max(t_len, 1)))
    return torch.tensor(
        [
            float(window_center) / float(max(t_len, 1)),
            float(window_width) / float(max(t_len, 1)),
            float(q_center.item()),
            float(q_spread.item()),
            float(locality),
            float(support_mass.item()),
        ],
        device=q_profile.device,
        dtype=q_profile.dtype,
    ).clamp(0.0, 1.0)


def _packet_temporal_profile(
    run: dict[str, Any],
    packet: dict[str, torch.Tensor],
    q_profile: torch.Tensor,
) -> torch.Tensor:
    depth_idx = int(packet.get("depth_index", torch.tensor([0])).item())
    history = run.get("history", [])
    block = history[depth_idx] if depth_idx < len(history) else None

    persistence = float(packet.get("persistence", torch.tensor([0.0])).mean().item())
    promotability = float(packet.get("promotability", torch.tensor([0.0])).mean().item())
    attack = float(packet.get("attack_law", torch.tensor([0.0])).mean().item())
    decay = float(packet.get("decay_law", torch.tensor([0.0])).mean().item())
    reentry = float(packet.get("reentry_bias", torch.tensor([0.0])).mean().item())
    loop_tendency = max(0.0, min(1.0, reentry * max(0.0, persistence)))

    phase_drift = 0.0
    winding_rate = 0.0
    if block is not None:
        b = int(packet["batch_index"].item())
        t0 = int(packet["law_window"][0].item()) if "law_window" in packet else int(packet["time_index"].item())
        t1 = int(packet["law_window"][1].item()) if "law_window" in packet else min(t0 + 2, block["major_mass"].size(-1))
        if "pair_scores" in block:
            pair_window = block["pair_scores"][b, t0:t1].detach()
            if pair_window.numel() > 1:
                phase_drift = float(pair_window.diff().abs().mean().clamp(0.0, 1.0).item())
        if "per_q" in block:
            q_window = block["per_q"][b, :, t0:t1].abs().detach()
            if q_window.numel() > 0:
                q_window = q_window / q_window.sum(dim=0, keepdim=True).clamp_min(1e-8)
                if q_window.size(-1) > 1:
                    winding_rate = float((q_window[:, 1:] - q_window[:, :-1]).abs().mean().clamp(0.0, 1.0).item())

    q_entropy = float(_safe_entropy(q_profile).item())
    return torch.tensor(
        [
            persistence,
            promotability,
            attack,
            decay,
            reentry,
            loop_tendency,
            phase_drift,
            0.5 * (winding_rate + q_entropy),
        ],
        dtype=q_profile.dtype,
        device=q_profile.device,
    ).clamp(0.0, 1.0)


def _compose_coarse_profile(
    arc_profile: torch.Tensor,
    law_signature: torch.Tensor,
    confidence: torch.Tensor,
) -> torch.Tensor:
    return _normalize_rows(torch.cat([arc_profile, law_signature, confidence], dim=-1))


def construct_operator_seed(
    *,
    coarse_profile: torch.Tensor,
    q_profile: torch.Tensor,
    temporal_profile: torch.Tensor,
    support_profile: torch.Tensor,
    branch_profile: torch.Tensor,
    sig_cfg: RafaRelationalSignatureConfig,
) -> torch.Tensor:
    branch_flat = branch_profile.reshape(branch_profile.size(0), -1)
    seed_source = _normalize_rows(
        torch.cat(
            [
                coarse_profile,
                q_profile,
                temporal_profile,
                support_profile,
                branch_flat,
            ],
            dim=-1,
        )
    )
    return _normalize_rows(_stretch_features(seed_source, sig_cfg.operator_seed_dim))


def _build_operator_seed(
    *,
    coarse_profile: torch.Tensor,
    q_profile: torch.Tensor,
    temporal_profile: torch.Tensor,
    support_profile: torch.Tensor,
    branch_profile: torch.Tensor,
    sig_cfg: RafaRelationalSignatureConfig,
) -> torch.Tensor:
    return construct_operator_seed(
        coarse_profile=coarse_profile,
        q_profile=q_profile,
        temporal_profile=temporal_profile,
        support_profile=support_profile,
        branch_profile=branch_profile,
        sig_cfg=sig_cfg,
    )


def build_relational_factor_pack(
    *,
    q_profile: torch.Tensor,
    arc_profile: torch.Tensor,
    temporal_profile: torch.Tensor,
    support_profile: torch.Tensor,
    branch_profile: torch.Tensor,
    law_signature: torch.Tensor,
    confidence: torch.Tensor,
    sig_cfg: RafaRelationalSignatureConfig,
    operator_seed: torch.Tensor | None = None,
) -> RelationalFactorPack:
    coarse_profile = _compose_coarse_profile(
        arc_profile=arc_profile,
        law_signature=law_signature,
        confidence=confidence,
    )
    if operator_seed is None:
        operator_seed = construct_operator_seed(
            coarse_profile=coarse_profile,
            q_profile=q_profile,
            temporal_profile=temporal_profile,
            support_profile=support_profile,
            branch_profile=branch_profile,
            sig_cfg=sig_cfg,
        )
    else:
        operator_seed = _normalize_rows(_stretch_features(operator_seed, sig_cfg.operator_seed_dim))
    return RelationalFactorPack(
        coarse_profile=coarse_profile,
        q_profile=q_profile,
        arc_profile=arc_profile,
        temporal_profile=temporal_profile,
        support_profile=support_profile,
        branch_profile=branch_profile,
        law_signature=law_signature,
        confidence=confidence,
        operator_seed=operator_seed,
    ).validate(sig_cfg)


def _assemble_dense_body(
    q_profile: torch.Tensor,
    arc_profile: torch.Tensor,
    temporal_profile: torch.Tensor,
    support_profile: torch.Tensor,
    branch_profile: torch.Tensor,
    law_signature: torch.Tensor,
    confidence: torch.Tensor,
    sig_cfg: RafaRelationalSignatureConfig,
    operator_seed: torch.Tensor | None = None,
) -> torch.Tensor:
    factor_pack = build_relational_factor_pack(
        q_profile=q_profile,
        arc_profile=arc_profile,
        temporal_profile=temporal_profile,
        support_profile=support_profile,
        branch_profile=branch_profile,
        law_signature=law_signature,
        confidence=confidence,
        operator_seed=operator_seed,
        sig_cfg=sig_cfg,
    )
    return _assemble_dense_body_from_factor_pack(factor_pack=factor_pack, sig_cfg=sig_cfg)


def _assemble_dense_body_from_factor_pack(
    factor_pack: RelationalFactorPack,
    sig_cfg: RafaRelationalSignatureConfig,
) -> torch.Tensor:
    factor_pack = factor_pack.validate(sig_cfg)
    branch_flat = factor_pack.branch_profile.reshape(factor_pack.branch_profile.size(0), -1)
    residual = _normalize_rows(
        torch.cat(
            [
                factor_pack.coarse_profile,
                factor_pack.q_profile,
                factor_pack.temporal_profile,
                factor_pack.support_profile,
                branch_flat,
                factor_pack.operator_seed,
            ],
            dim=-1,
        )
    )
    q_values = torch.linspace(
        0.0,
        1.0,
        steps=factor_pack.q_profile.size(-1),
        device=factor_pack.q_profile.device,
        dtype=factor_pack.q_profile.dtype,
    ).view(1, -1)
    q_center = (factor_pack.q_profile * q_values).sum(dim=-1, keepdim=True)
    q_spread = torch.sqrt(
        (factor_pack.q_profile * (q_values - q_center).pow(2)).sum(dim=-1, keepdim=True).clamp_min(1e-8)
    )
    q_stats = torch.cat([q_center, q_spread], dim=-1)

    segments = [
        torch.cat(
            [
                factor_pack.arc_profile[:, [0, 2, 3, 4]],
                factor_pack.temporal_profile[:, :5],
                factor_pack.support_profile[:, [0, 1, 4, 5]],
                factor_pack.confidence,
            ],
            dim=-1,
        ),
        torch.cat(
            [
                factor_pack.q_profile,
                q_stats,
                factor_pack.arc_profile[:, :2],
                factor_pack.temporal_profile[:, [0, 1, 6, 7]],
            ],
            dim=-1,
        ),
        torch.cat(
            [
                factor_pack.temporal_profile,
                factor_pack.arc_profile[:, [0, 1]],
                factor_pack.support_profile[:, :2],
                factor_pack.confidence,
            ],
            dim=-1,
        ),
        torch.cat(
            [
                factor_pack.support_profile,
                factor_pack.temporal_profile[:, [4, 5, 6]],
                factor_pack.arc_profile[:, [3, 4]],
                q_stats,
            ],
            dim=-1,
        ),
        torch.cat(
            [
                branch_flat,
                factor_pack.q_profile,
                factor_pack.support_profile[:, [4, 5]],
                factor_pack.operator_seed,
            ],
            dim=-1,
        ),
        torch.cat(
            [
                factor_pack.coarse_profile,
                factor_pack.operator_seed,
                branch_flat,
                factor_pack.temporal_profile[:, 5:],
                factor_pack.support_profile,
                factor_pack.confidence,
            ],
            dim=-1,
        ),
    ]
    prefix_dims = tuple(sig_cfg.prefix_dims)
    segment_dims = []
    prev = 0
    for dim in prefix_dims:
        segment_dims.append(int(dim) - prev)
        prev = int(dim)

    mixed_segments: list[torch.Tensor] = []
    for source, seg_dim in zip(segments, segment_dims):
        local = _stretch_features(_normalize_rows(source), seg_dim)
        res = _stretch_features(residual, seg_dim)
        mixed = _normalize_rows((1.0 - sig_cfg.residual_mix) * local + sig_cfg.residual_mix * res)
        mixed_segments.append(mixed)
    return _normalize_rows(torch.cat(mixed_segments, dim=-1))


def extract_relational_factor_pack(
    run: dict[str, Any],
    sig_cfg: RafaRelationalSignatureConfig | None = None,
) -> tuple[RelationalFactorPack | None, list[int], list[int], list[int], int, int]:
    sig_cfg = sig_cfg or RafaRelationalSignatureConfig()
    law_packets = list(run.get("law_packets", []))
    cfg = run.get("config")
    num_modes = int(getattr(cfg, "num_modes", 2))
    q_bins = len(getattr(cfg, "qset", (2, 3, 4, 5, 6, 8, 12)))
    if not law_packets:
        return None, [], [], [], q_bins, num_modes

    q_rows: list[torch.Tensor] = []
    arc_rows: list[torch.Tensor] = []
    temporal_rows: list[torch.Tensor] = []
    support_rows: list[torch.Tensor] = []
    branch_rows: list[torch.Tensor] = []
    confidence_rows: list[torch.Tensor] = []
    law_sig_rows: list[torch.Tensor] = []
    packet_ids: list[int] = []
    depth_ids: list[int] = []
    time_ids: list[int] = []

    for packet_idx, packet in enumerate(law_packets):
        q_profile = packet["q_mass"].detach().view(-1)
        q_profile = q_profile / q_profile.sum().clamp_min(1e-8)
        arc_profile = torch.stack(
            [
                packet["major_mass"].mean(),
                packet["minor_residue"].mean(),
                packet["harmonic_ratio"].mean(),
                packet["concentration"].mean(),
                packet["sharpness"].mean(),
            ],
            dim=0,
        ).detach()
        branch_profile = _packet_branch_profile(run, packet, num_modes=num_modes, branch_dim=sig_cfg.branch_profile_dim).detach()
        temporal_profile = _packet_temporal_profile(run, packet, q_profile=q_profile).detach()
        support_profile = _packet_support_profile(packet, q_profile=q_profile, branch_profile=branch_profile, cfg=cfg).detach()
        confidence = torch.tensor(
            [
                float(
                    (
                        packet["score"].mean()
                        + packet["major_mass"].mean()
                        + packet["promotability"].mean()
                    ).div(3.0).clamp(0.0, 1.0).item()
                )
            ],
            device=q_profile.device,
            dtype=q_profile.dtype,
        )
        q_rows.append(q_profile)
        arc_rows.append(arc_profile)
        temporal_rows.append(temporal_profile)
        support_rows.append(support_profile)
        branch_rows.append(branch_profile)
        confidence_rows.append(confidence)
        law_sig_rows.append(packet["law_signature"].detach().view(-1))
        packet_ids.append(packet_idx)
        depth_ids.append(int(packet["depth_index"].item()))
        time_ids.append(int(packet["time_index"].item()))

    q_tensor = torch.stack(q_rows, dim=0)
    arc_tensor = torch.stack(arc_rows, dim=0)
    temporal_tensor = torch.stack(temporal_rows, dim=0)
    support_tensor = torch.stack(support_rows, dim=0)
    branch_tensor = torch.stack(branch_rows, dim=0)
    confidence_tensor = torch.stack(confidence_rows, dim=0)
    law_sig_tensor = torch.stack(law_sig_rows, dim=0)
    factor_pack = build_relational_factor_pack(
        q_profile=q_tensor,
        arc_profile=arc_tensor,
        temporal_profile=temporal_tensor,
        support_profile=support_tensor,
        branch_profile=branch_tensor,
        law_signature=law_sig_tensor,
        confidence=confidence_tensor,
        sig_cfg=sig_cfg,
    )
    return factor_pack, packet_ids, depth_ids, time_ids, q_bins, num_modes


def extract_relational_signatures(
    run: dict[str, Any],
    sig_cfg: RafaRelationalSignatureConfig | None = None,
) -> RafaRelationalSignatureV0:
    sig_cfg = sig_cfg or RafaRelationalSignatureConfig()
    factor_pack, packet_ids, depth_ids, time_ids, q_bins, num_modes = extract_relational_factor_pack(run, sig_cfg)
    if factor_pack is None:
        return _empty_bank(q_bins=q_bins, num_modes=num_modes, sig_cfg=sig_cfg)
    h = _assemble_dense_body_from_factor_pack(factor_pack=factor_pack, sig_cfg=sig_cfg)
    return RafaRelationalSignatureV0(
        h=h,
        q_profile=factor_pack.q_profile,
        arc_profile=factor_pack.arc_profile,
        temporal_profile=factor_pack.temporal_profile,
        support_profile=factor_pack.support_profile,
        branch_profile=factor_pack.branch_profile,
        operator_seed=factor_pack.operator_seed,
        confidence=factor_pack.confidence,
        packet_index=torch.tensor(packet_ids, dtype=torch.long, device=h.device),
        depth_index=torch.tensor(depth_ids, dtype=torch.long, device=h.device),
        time_index=torch.tensor(time_ids, dtype=torch.long, device=h.device),
        prefix_dims=tuple(sig_cfg.prefix_dims),
        coarse_profile=factor_pack.coarse_profile,
    )


def concat_relational_signature_banks(
    banks: list[RafaRelationalSignatureV0],
) -> RafaRelationalSignatureV0:
    if not banks:
        return _empty_bank(q_bins=7, num_modes=2, sig_cfg=RafaRelationalSignatureConfig())
    non_empty = [bank for bank in banks if bank.h.numel() > 0]
    if not non_empty:
        return _empty_bank(
            q_bins=banks[0].q_profile.size(-1),
            num_modes=banks[0].branch_profile.size(1),
            sig_cfg=RafaRelationalSignatureConfig(
                signature_dim=banks[0].h.size(-1),
                prefix_dims=banks[0].prefix_dims,
                operator_seed_dim=banks[0].operator_seed.size(-1),
                branch_profile_dim=banks[0].branch_profile.size(-1),
            ),
        )
    ref = non_empty[0]
    return RafaRelationalSignatureV0(
        h=torch.cat([bank.h for bank in non_empty], dim=0),
        q_profile=torch.cat([bank.q_profile for bank in non_empty], dim=0),
        arc_profile=torch.cat([bank.arc_profile for bank in non_empty], dim=0),
        temporal_profile=torch.cat([bank.temporal_profile for bank in non_empty], dim=0),
        support_profile=torch.cat([bank.support_profile for bank in non_empty], dim=0),
        branch_profile=torch.cat([bank.branch_profile for bank in non_empty], dim=0),
        operator_seed=torch.cat([bank.operator_seed for bank in non_empty], dim=0),
        confidence=torch.cat([bank.confidence for bank in non_empty], dim=0),
        packet_index=torch.cat([bank.packet_index for bank in non_empty], dim=0),
        depth_index=torch.cat([bank.depth_index for bank in non_empty], dim=0),
        time_index=torch.cat([bank.time_index for bank in non_empty], dim=0),
        prefix_dims=ref.prefix_dims,
        coarse_profile=(
            torch.cat([bank.coarse_profile for bank in non_empty if bank.coarse_profile is not None], dim=0)
            if all(bank.coarse_profile is not None for bank in non_empty)
            else None
        ),
    )


def build_relational_signature_library(
    bank: RafaRelationalSignatureV0,
    cfg: Any | None = None,
    sig_cfg: RafaRelationalSignatureConfig | None = None,
) -> dict[str, Any]:
    sig_cfg = sig_cfg or RafaRelationalSignatureConfig()
    if bank.num_signatures == 0:
        return {"families": [], "num_signatures": 0, "num_families": 0, "mean_family_size": 0.0}

    merge_threshold = float(getattr(cfg, "law_packet_merge_threshold", sig_cfg.library_merge_threshold))
    topk = int(getattr(cfg, "law_packet_topk_families", sig_cfg.library_topk_families))
    families: list[dict[str, Any]] = []
    for idx in range(bank.num_signatures):
        sig = bank.h[idx]
        best_idx = -1
        best_sim = -1.0
        for family_idx, family in enumerate(families):
            sim = float(F.cosine_similarity(sig, family["prototype_h"], dim=0).item())
            if sim > best_sim:
                best_sim = sim
                best_idx = family_idx
        if best_idx >= 0 and best_sim >= merge_threshold:
            family = families[best_idx]
            count = family["count"] + 1
            prev_w = float(count - 1) / float(count)
            new_w = 1.0 / float(count)
            family["count"] = count
            for key in ("prototype_h", "prototype_q_profile", "prototype_arc_profile", "prototype_temporal_profile", "prototype_support_profile", "prototype_branch_profile", "prototype_operator_seed"):
                family[key] = prev_w * family[key] + new_w * {
                    "prototype_h": bank.h[idx],
                    "prototype_q_profile": bank.q_profile[idx],
                    "prototype_arc_profile": bank.arc_profile[idx],
                    "prototype_temporal_profile": bank.temporal_profile[idx],
                    "prototype_support_profile": bank.support_profile[idx],
                    "prototype_branch_profile": bank.branch_profile[idx],
                    "prototype_operator_seed": bank.operator_seed[idx],
                }[key]
                if key in {"prototype_h", "prototype_q_profile", "prototype_operator_seed"}:
                    family[key] = _normalize_rows(family[key].unsqueeze(0)).squeeze(0)
            family["mean_confidence"] = prev_w * family["mean_confidence"] + new_w * bank.confidence[idx].mean()
            family["mean_branch_mass"] = prev_w * family["mean_branch_mass"] + new_w * bank.branch_profile[idx, :, 1].mean()
            family["mean_q_entropy"] = prev_w * family["mean_q_entropy"] + new_w * _safe_entropy(bank.q_profile[idx])
            family["members"].append(
                {
                    "packet_index": int(bank.packet_index[idx].item()),
                    "depth_index": int(bank.depth_index[idx].item()),
                    "time_index": int(bank.time_index[idx].item()),
                    "confidence": float(bank.confidence[idx].mean().item()),
                }
            )
            family["best_similarity"] = max(float(family["best_similarity"]), best_sim)
        else:
            families.append(
                {
                    "family_index": len(families),
                    "count": 1,
                    "prototype_h": bank.h[idx].clone(),
                    "prototype_q_profile": bank.q_profile[idx].clone(),
                    "prototype_arc_profile": bank.arc_profile[idx].clone(),
                    "prototype_temporal_profile": bank.temporal_profile[idx].clone(),
                    "prototype_support_profile": bank.support_profile[idx].clone(),
                    "prototype_branch_profile": bank.branch_profile[idx].clone(),
                    "prototype_operator_seed": bank.operator_seed[idx].clone(),
                    "mean_confidence": bank.confidence[idx].mean().clone(),
                    "mean_branch_mass": bank.branch_profile[idx, :, 1].mean().clone(),
                    "mean_q_entropy": _safe_entropy(bank.q_profile[idx]).clone(),
                    "members": [
                        {
                            "packet_index": int(bank.packet_index[idx].item()),
                            "depth_index": int(bank.depth_index[idx].item()),
                            "time_index": int(bank.time_index[idx].item()),
                            "confidence": float(bank.confidence[idx].mean().item()),
                        }
                    ],
                    "best_similarity": 1.0,
                }
            )
    ranked = sorted(
        families,
        key=lambda item: (
            float(item["count"]),
            float(item["mean_confidence"].item() if torch.is_tensor(item["mean_confidence"]) else item["mean_confidence"]),
        ),
        reverse=True,
    )
    trimmed = ranked[: max(1, min(topk, len(ranked)))]
    mean_family_size = float(sum(int(f["count"]) for f in trimmed) / max(1, len(trimmed)))
    return {
        "families": trimmed,
        "num_signatures": bank.num_signatures,
        "num_families": len(trimmed),
        "mean_family_size": mean_family_size,
        "prefix_dims": tuple(bank.prefix_dims),
    }


def serialize_relational_signature_bank(bank: RafaRelationalSignatureV0) -> dict[str, Any]:
    return {
        "signature_dim": int(bank.h.size(-1)),
        "prefix_dims": [int(dim) for dim in bank.prefix_dims],
        "num_signatures": int(bank.num_signatures),
        "signatures": [
            {
                "packet_index": int(bank.packet_index[idx].item()),
                "depth_index": int(bank.depth_index[idx].item()),
                "time_index": int(bank.time_index[idx].item()),
                "confidence": float(bank.confidence[idx].mean().item()),
                "h": bank.h[idx].detach().cpu().tolist(),
                "q_profile": bank.q_profile[idx].detach().cpu().tolist(),
                "arc_profile": bank.arc_profile[idx].detach().cpu().tolist(),
                "temporal_profile": bank.temporal_profile[idx].detach().cpu().tolist(),
                "support_profile": bank.support_profile[idx].detach().cpu().tolist(),
                "branch_profile": bank.branch_profile[idx].detach().cpu().tolist(),
                "operator_seed": bank.operator_seed[idx].detach().cpu().tolist(),
                **(
                    {"coarse_profile": bank.coarse_profile[idx].detach().cpu().tolist()}
                    if bank.coarse_profile is not None
                    else {}
                ),
            }
            for idx in range(bank.num_signatures)
        ],
    }


def serialize_relational_signature_library(library: dict[str, Any]) -> dict[str, Any]:
    out = {
        "num_signatures": int(library.get("num_signatures", 0)),
        "num_families": int(library.get("num_families", 0)),
        "mean_family_size": float(library.get("mean_family_size", 0.0)),
        "prefix_dims": [int(dim) for dim in library.get("prefix_dims", ())],
        "families": [],
    }
    for family in library.get("families", []):
        out["families"].append(
            {
                "family_index": int(family["family_index"]),
                "count": int(family["count"]),
                "prototype_h": family["prototype_h"].detach().cpu().tolist(),
                "prototype_q_profile": family["prototype_q_profile"].detach().cpu().tolist(),
                "prototype_arc_profile": family["prototype_arc_profile"].detach().cpu().tolist(),
                "prototype_temporal_profile": family["prototype_temporal_profile"].detach().cpu().tolist(),
                "prototype_support_profile": family["prototype_support_profile"].detach().cpu().tolist(),
                "prototype_branch_profile": family["prototype_branch_profile"].detach().cpu().tolist(),
                "prototype_operator_seed": family["prototype_operator_seed"].detach().cpu().tolist(),
                "mean_confidence": float(family["mean_confidence"].item() if torch.is_tensor(family["mean_confidence"]) else family["mean_confidence"]),
                "mean_branch_mass": float(family["mean_branch_mass"].item() if torch.is_tensor(family["mean_branch_mass"]) else family["mean_branch_mass"]),
                "mean_q_entropy": float(family["mean_q_entropy"].item() if torch.is_tensor(family["mean_q_entropy"]) else family["mean_q_entropy"]),
                "best_similarity": float(family["best_similarity"]),
                "members": list(family["members"]),
            }
        )
    return out


def summarize_relational_signature_bank(
    bank: RafaRelationalSignatureV0,
    library: dict[str, Any] | None = None,
) -> dict[str, float]:
    if bank.num_signatures == 0:
        return {
            "num_relational_signatures": 0.0,
            "num_relational_signature_families": 0.0,
            "mean_relational_signature_confidence": 0.0,
            "mean_relational_signature_q_entropy": 0.0,
            "mean_relational_branch_mass": 0.0,
            "dominant_relational_family_share": 0.0,
        }
    q_entropy = torch.stack([_safe_entropy(row) for row in bank.q_profile], dim=0)
    branch_mass = bank.branch_profile[..., 1].mean(dim=-1)
    out = {
        "num_relational_signatures": float(bank.num_signatures),
        "num_relational_signature_families": 0.0,
        "mean_relational_signature_confidence": float(bank.confidence.mean().item()),
        "mean_relational_signature_q_entropy": float(q_entropy.mean().item()),
        "mean_relational_branch_mass": float(branch_mass.mean().item()),
        "dominant_relational_family_share": 0.0,
    }
    if library:
        counts = [float(f["count"]) for f in library.get("families", [])]
        if counts:
            total = max(1e-8, sum(counts))
            out["num_relational_signature_families"] = float(library.get("num_families", len(counts)))
            out["dominant_relational_family_share"] = float(max(counts) / total)
    return out
