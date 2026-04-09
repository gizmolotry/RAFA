from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F

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
            seed = z[b : b + 1, :, t0:t1].mean(dim=2, keepdim=True)
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


def apply_passive_packets(
    z: torch.Tensor,
    packets: list[dict[str, torch.Tensor]],
    gain: float,
) -> torch.Tensor:
    if not packets:
        return z
    out = z
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
        return z
    out = z
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
        out[b : b + 1] = phasor_apply_delta(out[b : b + 1], delta)
    return out


def recurse_circleworld(
    z: torch.Tensor,
    cfg: CircleworldConfig,
    depth: int | None = None,
    mode: str = "active_packets",
) -> dict[str, Any]:
    depth = cfg.recursion_depth if depth is None else depth
    current = phasor_normalize(z)
    history: list[dict[str, torch.Tensor]] = []
    all_packets: list[list[dict[str, torch.Tensor]]] = []

    for d in range(max(1, depth)):
        arc = hardy_littlewood_arc_field(current, cfg.qset, cfg.q_weights)
        promo = promotability_field(arc, persistence_momentum=cfg.persistence_momentum)
        packets = build_promoted_packets(current, arc, promo, cfg)
        history.append({**arc, **promo})
        all_packets.append(packets)

        if mode == "no_promotion":
            continue
        if mode == "passive_packets":
            current = apply_passive_packets(current, packets, gain=cfg.child_law_gain)
        elif mode == "active_packets":
            current = apply_active_packets(current, packets, cfg)
        else:
            raise ValueError(f"Unknown circleworld mode: {mode}")

    final_arc = hardy_littlewood_arc_field(current, cfg.qset, cfg.q_weights)
    final_promo = promotability_field(final_arc, persistence_momentum=cfg.persistence_momentum)
    return {
        "phase_state": current,
        "history": history,
        "packets": all_packets,
        "final_arc": final_arc,
        "final_promo": final_promo,
    }


def summarize_circleworld_run(run: dict[str, Any]) -> dict[str, float]:
    initial = run["history"][0]
    final_arc = run["final_arc"]
    final_promo = run["final_promo"]
    packets = run["packets"]

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

    return {
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
    }


def circleworld_loss(
    run: dict[str, Any],
    target_major_mass: float = 0.85,
    target_promotability: float = 0.65,
    target_attack: float = 0.05,
) -> dict[str, torch.Tensor]:
    final_arc = run["final_arc"]
    final_promo = run["final_promo"]
    initial = run["history"][0]

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
    total = (
        l_major
        + 0.75 * l_residue
        + 0.5 * l_promo
        + 0.25 * l_attack
        + 0.5 * l_gain
        + 0.75 * l_q_dom
        + 0.35 * l_q_entropy
        + 0.35 * l_major_sat
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
    }
