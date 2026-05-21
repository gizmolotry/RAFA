from __future__ import annotations

import argparse
import json
import math
import random
import sys
from copy import replace
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
CORE = ROOT / "core"
LINEAGE = ROOT / "lineages" / "04_positive_replacement"
for path in (ROOT, CORE, LINEAGE):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from benchmark_audio_continuity import (
    _chunk_vectors,
    _cosine,
    _frame_rms,
    _normalized_autocorr,
    analyze_wav,
    compare_continuity_summaries,
    continuity_delta_snapshot,
    continuity_snapshot,
)
from ablate_formalization import _generate_seed_phase, make_seed_rafa
from circleworld import (
    CircleworldConfig,
    _evolve_child_worlds,
    _phase_alignment,
    apply_passive_packets,
    circleworld_step,
    clone_circleworld_state,
    extract_phase_state,
    hardy_littlewood_arc_field,
    promotability_field,
    perturb_circleworld_state,
    summarize_circleworld_run,
)
from export_circleworld_audio import _load_circle_cfg, _phasor_to_phase, _save_wav, prepare_reference_audio
from rafa_math_tools import phasor_apply_delta, phasor_normalize
from resonant_law_objects import composition_summary, law_objects_from_state, run_retrieval_assay
from stft_utils import compute_stft, inverse_stft
from config import load_config

IDENTITY_CARRY_SUPPORT_MIN = 0.18
IDENTITY_CARRY_COHERENCE_MIN = 0.10
IDENTITY_CARRY_BUDGET_MIN = 0.60
IDENTITY_CARRY_BUDGET_RETAINED_MIN = 0.55
IDENTITY_CARRY_FRACTION_MIN = 0.50
IDENTITY_CARRY_STEPS_MIN = 2.0
BRANCH_FAMILY_ORDER: tuple[str, ...] = ("perturb", "readout", "continuation", "mode_replace", "promotion", "other")
NESTED_ASSAY_FAMILIES: tuple[str, ...] = ("readout", "continuation", "mode_replace")
NESTED_ASSAY_LABELS: tuple[str, ...] = (
    "nested_sibling",
    "overwrite_or_world_jump",
    "over_rigid_or_frozen",
    "ambiguous_middle",
)
DEFAULT_ASSAY_KILLSWITCH: dict[str, Any] = {
    "disable_continuation_mode1_write": False,
    "continuation_parent_only": False,
    "continuation_write_only_readout": False,
    "scramble_continuation_child_phase": False,
    "disable_branch_pre_unroll": False,
    "disable_branch_childworld_runtime": False,
    "disable_continuation_pre_unroll": False,
    "disable_mode_replace_pre_unroll": False,
    "disable_direct_child_continuation_mix": False,
    "strip_child_worlds_before_branch": False,
    "continuation_mode0_parent_only": False,
    "continuation_clamp_child_identity": False,
    "continuation_child_phase_control": "",
    "mode_replace_clamp_child_identity": False,
    "mode_replace_child_phase_control": "",
    "mode_replace_gate_floor": 0.0,
    "mode_replace_ratio_scale": 1.0,
    "child_volume_control": "",
    "child_volume_top_k": 0,
    "child_volume_drop_strongest": False,
    "child_volume_min_support": 0.0,
    "fragment_child_records": False,
    "fragment_child_parts": 2,
    "fragment_child_axis": "support_window",
    "fragment_child_phase_mode": "preserve",
    "fragment_child_budget_scale": 1.0,
    "fragment_child_replace_source": True,
    "fragment_child_scope": "strongest",
    "enable_grandchild_probe": False,
    "grandchild_probe_depth": 1,
    "grandchild_probe_parts": 2,
    "grandchild_probe_budget_scale": 0.5,
    "grandchild_probe_replace_parent": False,
    "grandchild_probe_flatten_to_child_worlds": True,
    "enable_child_predictive_probe": False,
    "enable_child_predictive_assimilation": False,
    "child_predictive_bind_gain": 0.85,
    "child_predictive_min_residual_reduction": 0.0,
    "child_predictive_min_boundary_match": 0.0,
    "child_predictive_gate_floor": 0.0,
    "child_predictive_support_power": 1.0,
    "child_predictive_logit_gain": 0.35,
    "child_predictive_target_horizon": 1,
    "child_predictive_candidate": "raw_child_delta",
    "child_predictive_oracle_mix": 1.0,
    "child_predictive_export_dataset": False,
    "child_predictive_dataset_dir": "",
    "child_predictive_dataset_tag": "parent_phase_projector",
    "learned_parent_bridge_boundary_center": 0.952,
    "learned_parent_bridge_boundary_scale": 120.0,
    "learned_parent_bridge_support_target": 0.158,
    "learned_parent_bridge_volume_effective_target": 4.80,
    "learned_parent_bridge_top_share_target": 0.26,
    "learned_parent_bridge_acceptance_target": 0.13,
    "learned_parent_bridge_acceptance_floor": 0.0,
    "learned_parent_bridge_score_threshold": 0.50,
    "learned_parent_bridge_boundary_weight": 0.50,
    "learned_parent_bridge_volume_weight": 0.25,
    "learned_parent_bridge_support_weight": 0.15,
    "learned_parent_bridge_coherence_weight": 0.10,
    "learned_parent_bridge_prewrite_policy": "scale",
    "learned_parent_bridge_delta_scale": 0.0,
    "learned_parent_bridge_delta_source": "raw_child_delta",
    "learned_parent_bridge_projector_coeffs": [],
    "learned_parent_bridge_projector_feature_names": [],
    "learned_parent_bridge_carrier_coeffs": [],
    "learned_parent_bridge_carrier_feature_names": [],
    "learned_parent_bridge_phasor_carrier_cos_coeffs": [],
    "learned_parent_bridge_phasor_carrier_sin_coeffs": [],
    "learned_parent_bridge_phasor_carrier_feature_names": [],
    "learned_parent_bridge_authority_source": "",
    "learned_parent_bridge_authority_coeffs": [],
    "learned_parent_bridge_authority_feature_names": [],
    "enable_parent_ontology_field": False,
    "disable_parent_ontology_charge": False,
    "disable_parent_ontology_support_gate": False,
    "disable_parent_ontology_mode_logits": False,
    "disable_parent_ontology_mode_support": False,
    "disable_mode_replace_child_phase_carrier": False,
    "parent_ontology_charge_gain": 1.0,
    "parent_ontology_use_in_mode_replace": False,
    "parent_ontology_mode_replace_source": "ontology_delta",
    "parent_ontology_mode_replace_delta_gain": 1.0,
    "parent_ontology_mode1_logit_boost": 0.0,
    "parent_ontology_mode0_logit_suppress": 0.0,
    "parent_ontology_support_boost": 1.0,
    "parent_ontology_gate_floor": 0.0,
}
ASSAY_KILLSWITCH: dict[str, Any] = dict(DEFAULT_ASSAY_KILLSWITCH)


def _assay_flag(name: str) -> bool:
    return bool(ASSAY_KILLSWITCH.get(name, DEFAULT_ASSAY_KILLSWITCH.get(name, False)))


def _assay_value(name: str, default: Any = None) -> Any:
    return ASSAY_KILLSWITCH.get(name, DEFAULT_ASSAY_KILLSWITCH.get(name, default))


def _safe_device(requested: str) -> torch.device:
    if requested == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _circleworld_step(
    z: torch.Tensor | dict[str, torch.Tensor],
    cfg: CircleworldConfig,
    mode: str = "active_packets",
) -> tuple[torch.Tensor | dict[str, torch.Tensor], dict[str, torch.Tensor], list[dict[str, torch.Tensor]]]:
    return circleworld_step(z, cfg=cfg, mode=mode)


def _run_depth_trace(
    z0: Any,
    cfg: CircleworldConfig,
    depth: int,
    mode: str = "active_packets",
) -> dict[str, Any]:
    if mode in {"native_multimode", "native_multimode_childworld"} or cfg.branching_mode in {"native_multimode", "native_multimode_childworld"}:
        current: torch.Tensor | dict[str, torch.Tensor] = clone_circleworld_state(z0)
        mode = "native_multimode_childworld" if cfg.branching_mode == "native_multimode_childworld" or mode == "native_multimode_childworld" else "native_multimode"
    else:
        current = phasor_normalize(z0)
    history: list[dict[str, torch.Tensor]] = []
    packets_by_depth: list[list[dict[str, torch.Tensor]]] = []
    states: list[Any] = [clone_circleworld_state(current)]
    for _ in range(depth):
        current, block, packets = _circleworld_step(current, cfg=cfg, mode=mode)
        history.append(block)
        packets_by_depth.append(packets)
        states.append(clone_circleworld_state(current))
    final_phase = extract_phase_state(current, cfg)
    final_arc = hardy_littlewood_arc_field(final_phase, cfg.qset, cfg.q_weights)
    final_promo = promotability_field(final_arc, persistence_momentum=cfg.persistence_momentum)
    return {
        "phase_state": final_phase,
        "state_bundle": current,
        "history": history,
        "packets": packets_by_depth,
        "states": states,
        "final_arc": final_arc,
        "final_promo": final_promo,
    }


def _seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _seeded_rand(shape: tuple[int, ...], device: torch.device, seed: int, offset: int = 0) -> torch.Tensor:
    gen = torch.Generator(device="cpu")
    gen.manual_seed(int(seed) + int(offset))
    return torch.rand(shape, generator=gen).to(device)


def _generate_seed_magnitude(
    batch_size: int,
    freq_bins: int,
    time_steps: int,
    device: torch.device,
    seed: int,
) -> torch.Tensor:
    t = torch.linspace(0.0, 1.0, steps=time_steps, device=device).view(1, 1, -1)
    f = torch.linspace(0.0, 1.0, steps=freq_bins, device=device).view(1, -1, 1)
    base = 0.04 + 0.12 * _seeded_rand((batch_size, freq_bins, time_steps), device=device, seed=seed, offset=17)
    low_rank = (
        0.18 * torch.sin(2.0 * torch.pi * (1.0 + 2.0 * f) * t)
        + 0.10 * torch.cos(2.0 * torch.pi * (0.5 + 1.5 * t) * (0.25 + f))
    ).abs()
    envelope = (0.75 + 0.25 * torch.cos(2.0 * torch.pi * (t - 0.2)).abs()).clamp(0.5, 1.0)
    return (base + low_rank) * envelope


def _default_seeded_nested_cases() -> list[dict[str, Any]]:
    return [
        {
            "name": "naked_seed_9100",
            "seed_source": "naked_rafa",
            "seed": 9100,
            "time_steps": 128,
            "warmup_depth": 5,
            "continuation_depth": 3,
            "require_live_child_start": True,
        },
        {
            "name": "naked_seed_9101",
            "seed_source": "naked_rafa",
            "seed": 9101,
            "time_steps": 128,
            "warmup_depth": 5,
            "continuation_depth": 3,
            "require_live_child_start": True,
        },
        {
            "name": "naked_seed_9102",
            "seed_source": "naked_rafa",
            "seed": 9102,
            "time_steps": 128,
            "warmup_depth": 5,
            "continuation_depth": 3,
            "require_live_child_start": True,
        },
    ]


def _phase_noise_perturb(z: torch.Tensor, scale: float, seed: int) -> torch.Tensor:
    _seed_all(seed)
    delta = scale * torch.randn((z.size(0), z.size(1), z.size(2)), device=z.device, dtype=z.dtype)
    return phasor_apply_delta(z, delta)


def _packet_seed_perturb(
    z: torch.Tensor,
    packets: list[dict[str, torch.Tensor]],
    strength: float,
    rank: int = 0,
) -> torch.Tensor:
    if not packets:
        return z
    ordered = sorted(packets, key=lambda p: float(p["score"].item()), reverse=True)
    packet = ordered[min(rank, len(ordered) - 1)]
    return apply_passive_packets(z, [packet], gain=strength)


def _promotion_cfg(cfg: CircleworldConfig, direction: str) -> CircleworldConfig:
    if direction == "lo":
        return replace(
            cfg,
            promotion_threshold=max(0.45, cfg.promotion_threshold - 0.05),
            max_promotions=min(cfg.max_promotions + 1, 8),
            child_law_gain=min(cfg.child_law_gain + 0.05, 1.0),
        )
    if direction == "hi":
        return replace(
            cfg,
            promotion_threshold=min(0.90, cfg.promotion_threshold + 0.05),
            max_promotions=max(cfg.max_promotions - 1, 1),
            child_law_gain=max(cfg.child_law_gain - 0.05, 0.05),
        )
    raise ValueError(direction)


def _lifecycle_cfg(cfg: CircleworldConfig, direction: str) -> CircleworldConfig:
    if direction == "retention":
        return replace(
            cfg,
            child_support_decay=max(0.01, 0.55 * cfg.child_support_decay),
            child_kill_threshold=max(0.0, cfg.child_kill_threshold - 0.035),
            child_writeback_gain=min(1.0, cfg.child_writeback_gain + 0.12),
            child_parent_mix_early=min(0.35, cfg.child_parent_mix_early + 0.06),
        )
    if direction == "writeback":
        return replace(
            cfg,
            child_min_age_for_writeback=max(1, cfg.child_min_age_for_writeback - 1),
            child_kill_threshold=max(0.0, cfg.child_kill_threshold - 0.025),
            child_writeback_gain=min(1.0, cfg.child_writeback_gain + 0.18),
            child_parent_mix=min(0.45, cfg.child_parent_mix + 0.08),
            child_parent_mix_early=min(0.40, cfg.child_parent_mix_early + 0.10),
        )
    raise ValueError(direction)


def _child_only_pre_unroll(
    state: dict[str, Any],
    cfg: CircleworldConfig,
    *,
    steps: int = 1,
    parent_write_scale: float = 1.0,
    mode0_protect: float = 0.0,
) -> dict[str, Any]:
    out = clone_circleworld_state(state)
    if not out.get("child_worlds"):
        out["mixed_phase_state"] = extract_phase_state(out, cfg)
        return out
    base_mode0 = out["phase_modes"][..., 0, :].clone()
    for _ in range(max(1, int(steps))):
        parent_mode0 = out["phase_modes"][..., 0, :]
        parent_mode1 = out["phase_modes"][..., 1, :]
        support_overlap = (out["mode_support"][..., 0] * out["mode_support"][..., 1]).clamp(0.0, 1.0)
        parent_branch_defect = ((1.0 - ((_phase_alignment(parent_mode0, parent_mode1) + 1.0) * 0.5)) * support_overlap).clamp(0.0, 1.0)
        child_worlds, child_metrics, delta0, delta1, child_identity_support = _evolve_child_worlds(
            state=out,
            cfg=cfg,
            parent_mode0=parent_mode0,
            parent_mode1=parent_mode1,
            parent_branch_defect=parent_branch_defect,
        )
        out["child_worlds"] = child_worlds
        if not child_worlds:
            break
        out["phase_modes"][..., 0, :] = phasor_apply_delta(
            out["phase_modes"][..., 0, :],
            (float(parent_write_scale) * delta0).clamp(-3.14159, 3.14159),
        )
        out["phase_modes"][..., 1, :] = phasor_apply_delta(
            out["phase_modes"][..., 1, :],
            delta1.clamp(-3.14159, 3.14159),
        )
        if mode0_protect > 0.0:
            protect = float(max(0.0, min(0.98, mode0_protect)))
            out["phase_modes"][..., 0, :] = phasor_normalize(
                (1.0 - protect) * out["phase_modes"][..., 0, :] + protect * base_mode0
            )
        child_support = torch.stack([child["mode_support"] for child in child_worlds], dim=0).amax(dim=0)
        support_boost = torch.maximum(
            child_identity_support.clamp(0.0, 1.0),
            (0.30 * child_support + 0.40 * child_metrics["child_writeback_mass"]).clamp(0.0, 1.0),
        )
        out["mode_support"][..., 1] = torch.maximum(out["mode_support"][..., 1], support_boost).clamp(0.0, 1.0)
        out["mode_logits"][..., 1] = out["mode_logits"][..., 1] + 0.35 * support_boost
    out["mixed_phase_state"] = extract_phase_state(out, cfg)
    return out


def _child_readout_score(child: dict[str, Any]) -> float:
    support_mass = float(child.get("mode_support", 0.0).mean().item()) if torch.is_tensor(child.get("mode_support")) else 0.0
    coherence_mass = float(child.get("mode_coherence", 0.0).mean().item()) if torch.is_tensor(child.get("mode_coherence")) else 0.0
    writeback_budget = float(child.get("writeback_budget", 0.0))
    survival_age = float(child.get("survival_age", 0.0))
    return support_mass + coherence_mass + 0.20 * writeback_budget + 0.12 * survival_age


def _select_strongest_child(state: dict[str, Any]) -> dict[str, Any] | None:
    children = [child for child in state.get("child_worlds", []) if child.get("active", False)]
    if not children:
        children = list(state.get("child_worlds", []))
    if not children:
        return None
    return max(children, key=_child_readout_score)


def _child_support_mean(child: dict[str, Any]) -> float:
    support = child.get("mode_support")
    if torch.is_tensor(support):
        return float(support.detach().float().mean().item())
    return 0.0


def _child_generation(child: dict[str, Any]) -> int:
    try:
        return int(child.get("generation", 1))
    except Exception:
        return 1


def _child_volume_summary(state: dict[str, Any]) -> dict[str, float]:
    children = list(state.get("child_worlds", [])) if isinstance(state, dict) else []
    active = [child for child in children if child.get("active", False)]
    scored = active if active else children
    scores = np.asarray([max(0.0, _child_readout_score(child)) for child in scored], dtype=np.float64)
    support = np.asarray([max(0.0, _child_support_mean(child)) for child in scored], dtype=np.float64)
    score_sum = float(scores.sum()) if scores.size else 0.0
    if score_sum > 0.0:
        probs = scores / score_sum
        entropy = float(-(probs * np.log(np.maximum(probs, 1.0e-12))).sum())
        effective = float(np.exp(entropy))
        top_share = float(probs.max())
    else:
        entropy = 0.0
        effective = 0.0
        top_share = 0.0
    return {
        "total_count": float(len(children)),
        "active_count": float(len(active)),
        "scored_count": float(len(scored)),
        "support_mass": float(support.sum()) if support.size else 0.0,
        "mean_support": float(support.mean()) if support.size else 0.0,
        "top_score_share": top_share,
        "score_entropy": entropy,
        "effective_count": effective,
        "max_generation": float(max([_child_generation(child) for child in children], default=0)),
        "grandchild_count": float(sum(1 for child in children if _child_generation(child) > 1)),
    }


def _mask_span_like(support: torch.Tensor, start: int, end: int, *, axis: str) -> torch.Tensor:
    mask = torch.zeros_like(support)
    if support.numel() == 0:
        return mask
    axis = str(axis or "support_window").strip().lower()
    if axis == "freq" and support.dim() >= 2:
        f_len = support.size(-2)
        s0 = max(0, min(int(start), int(f_len)))
        s1 = max(s0 + 1, min(int(end), int(f_len)))
        mask[..., s0:s1, :] = 1.0
    else:
        t_len = support.size(-1)
        s0 = max(0, min(int(start), int(t_len)))
        s1 = max(s0 + 1, min(int(end), int(t_len)))
        mask[..., s0:s1] = 1.0
    return mask


def _partition_spans(start: int, end: int, parts: int, *, min_width: int = 1) -> list[tuple[int, int]]:
    start_i = int(start)
    end_i = max(start_i + 1, int(end))
    width = max(1, end_i - start_i)
    part_count = max(1, min(int(parts), width))
    spans: list[tuple[int, int]] = []
    for idx in range(part_count):
        left = start_i + int(round(idx * width / part_count))
        right = start_i + int(round((idx + 1) * width / part_count))
        right = max(right, left + max(1, int(min_width)))
        right = min(end_i, right)
        if right > left:
            spans.append((left, right))
    return spans or [(start_i, end_i)]


def _support_extent(child: dict[str, Any], *, axis: str) -> tuple[int, int]:
    support = child.get("mode_support")
    if not torch.is_tensor(support) or support.numel() == 0:
        return (0, 1)
    axis = str(axis or "support_window").strip().lower()
    if axis == "freq" and support.dim() >= 2:
        return (0, int(support.size(-2)))
    window = child.get("support_window")
    if isinstance(window, (list, tuple)) and len(window) == 2:
        return (max(0, int(window[0])), max(int(window[0]) + 1, int(window[1])))
    return (0, int(support.size(-1)))


def _deterministic_random_phasor_like(reference: torch.Tensor, seed: int) -> torch.Tensor:
    basis = torch.arange(
        int(reference[..., 0].numel()),
        device=reference.device,
        dtype=reference.dtype,
    ).reshape(reference.shape[:-1])
    angle = torch.frac(torch.sin(basis * 12.9898 + float(seed) * 78.233) * 43758.5453)
    angle = (angle * (2.0 * math.pi)) - math.pi
    return torch.stack([torch.cos(angle), torch.sin(angle)], dim=-1)


def _child_phase_variant(
    phase: torch.Tensor,
    support_mask: torch.Tensor,
    *,
    mode: str,
    child_id: int,
    fragment_index: int,
) -> torch.Tensor:
    phase_mode = str(mode or "preserve").strip().lower()
    if phase_mode == "roll":
        candidate = torch.roll(phase, shifts=int(fragment_index) + 1, dims=-2)
    elif phase_mode == "random":
        candidate = _deterministic_random_phasor_like(phase, seed=int(child_id) * 1009 + int(fragment_index))
    elif phase_mode in {"zero", "canonical"}:
        candidate = torch.zeros_like(phase)
        candidate[..., 0] = 1.0
    else:
        candidate = phase
    mask = support_mask.unsqueeze(-1).clamp(0.0, 1.0)
    return phasor_normalize((1.0 - mask) * phase + mask * candidate)


def _fragment_child_record(
    child: dict[str, Any],
    *,
    state: dict[str, Any],
    parts: int,
    axis: str,
    phase_mode: str,
    budget_scale: float,
    generation: int | None = None,
    parent_child_id: int | None = None,
    preserve_first_child_id: bool = True,
) -> tuple[list[dict[str, Any]], int]:
    support = child.get("mode_support")
    phase = child.get("phase_state")
    if not torch.is_tensor(support) or not torch.is_tensor(phase):
        return [], int(state.get("next_child_id", 0))
    start, end = _support_extent(child, axis=axis)
    spans = _partition_spans(start, end, max(1, int(parts)))
    existing_ids = [
        int(item.get("child_id", -1))
        for item in state.get("child_worlds", [])
        if isinstance(item, dict) and item.get("child_id") is not None
    ]
    next_child_id = max(int(state.get("next_child_id", 0)), max(existing_ids, default=-1) + 1)
    try:
        original_child_id = int(child.get("child_id", -1))
    except Exception:
        original_child_id = -1
    try:
        root_origin_child_id = int(child.get("origin_child_id", original_child_id))
    except Exception:
        root_origin_child_id = original_child_id
    if root_origin_child_id < 0:
        root_origin_child_id = original_child_id
    fragments: list[dict[str, Any]] = []
    for idx, (s0, s1) in enumerate(spans):
        new_child = clone_circleworld_state(child)
        if idx == 0 and preserve_first_child_id and original_child_id >= 0:
            child_id = original_child_id
        else:
            child_id = next_child_id
            next_child_id += 1
        mask = _mask_span_like(support, s0, s1, axis=axis)
        new_child["child_id"] = int(child_id)
        new_child["mode_support"] = (support * mask).clamp(0.0, 1.0)
        coherence = child.get("mode_coherence")
        if torch.is_tensor(coherence):
            new_child["mode_coherence"] = (coherence * mask).clamp(0.0, 1.0)
        new_child["phase_state"] = _child_phase_variant(
            phase,
            mask,
            mode=phase_mode,
            child_id=child_id,
            fragment_index=idx,
        )
        if str(axis or "support_window").strip().lower() == "freq":
            new_child["support_window"] = tuple(child.get("support_window", (0, int(support.size(-1)))))
            new_child["support_axis"] = "freq"
            new_child["freq_support_window"] = (int(s0), int(s1))
        else:
            new_child["support_window"] = (int(s0), int(s1))
            new_child["support_axis"] = "time"
        budget = float(child.get("writeback_budget", 0.0))
        new_child["writeback_budget"] = float(budget * max(0.0, float(budget_scale)) / max(1, len(spans)))
        new_child["active"] = bool(float(new_child["mode_support"].mean().item()) > 0.0 and not child.get("collapsed", False))
        new_child["collapsed"] = bool(child.get("collapsed", False))
        new_child["origin_child_id"] = int(root_origin_child_id)
        new_child["parent_child_id"] = int(parent_child_id if parent_child_id is not None else original_child_id)
        new_child["generation"] = int(generation if generation is not None else _child_generation(child))
        new_child["fragment_index"] = int(idx)
        new_child["fragment_count"] = int(len(spans))
        fragments.append(new_child)
    state["next_child_id"] = int(next_child_id)
    return fragments, next_child_id


def _apply_child_volume_control(state: dict[str, Any]) -> tuple[dict[str, Any], dict[str, float]]:
    control = str(_assay_value("child_volume_control", "") or "").strip().lower()
    drop_strongest = bool(_assay_flag("child_volume_drop_strongest") or control == "drop_strongest")
    top_k = int(_assay_value("child_volume_top_k", 0) or 0)
    if control == "top1":
        top_k = 1
    min_support = float(_assay_value("child_volume_min_support", 0.0) or 0.0)
    children = list(state.get("child_worlds", []))
    if not children:
        return state, {"volume_removed_count": 0.0}
    ranked = sorted(children, key=_child_readout_score, reverse=True)
    keep_ids: set[int] | None = None
    if drop_strongest and ranked:
        strongest_id = int(ranked[0].get("child_id", -1))
        keep_ids = {int(child.get("child_id", -2)) for child in children if int(child.get("child_id", -2)) != strongest_id}
    elif top_k > 0:
        keep_ids = {int(child.get("child_id", -1)) for child in ranked[: max(1, top_k)]}
    if keep_ids is not None:
        children = [child for child in children if int(child.get("child_id", -2)) in keep_ids]
    if min_support > 0.0 or control == "thin_low_support":
        children = [child for child in children if _child_support_mean(child) >= min_support]
    removed = float(len(state.get("child_worlds", [])) - len(children))
    state["child_worlds"] = children
    return state, {"volume_removed_count": removed}


def _apply_assay_child_partitioning(state: Any, cfg: CircleworldConfig) -> tuple[Any, dict[str, Any]]:
    if not isinstance(state, dict) or "phase_modes" not in state:
        return state, {}
    out = clone_circleworld_state(state)
    before = _child_volume_summary(out)
    volume_metrics: dict[str, float] = {"volume_removed_count": 0.0}
    if (
        str(_assay_value("child_volume_control", "") or "").strip()
        or _assay_flag("child_volume_drop_strongest")
        or float(_assay_value("child_volume_min_support", 0.0) or 0.0) > 0.0
        or int(_assay_value("child_volume_top_k", 0) or 0) > 0
    ):
        out, volume_metrics = _apply_child_volume_control(out)

    fragment_count = 0
    if _assay_flag("fragment_child_records"):
        axis = str(_assay_value("fragment_child_axis", "support_window") or "support_window")
        phase_mode = str(_assay_value("fragment_child_phase_mode", "preserve") or "preserve")
        parts = max(1, int(_assay_value("fragment_child_parts", 2) or 2))
        budget_scale = float(_assay_value("fragment_child_budget_scale", 1.0) or 1.0)
        scope = str(_assay_value("fragment_child_scope", "strongest") or "strongest").strip().lower()
        replace_source = _assay_flag("fragment_child_replace_source")
        children = list(out.get("child_worlds", []))
        if children:
            if scope == "all":
                sources = [child for child in children if child.get("active", False)] or children
            else:
                strongest = _select_strongest_child(out)
                sources = [strongest] if strongest is not None else []
            source_ids = {int(child.get("child_id", -1)) for child in sources}
            fragments: list[dict[str, Any]] = []
            for source in sources:
                made, _ = _fragment_child_record(
                    source,
                    state=out,
                    parts=parts,
                    axis=axis,
                    phase_mode=phase_mode,
                    budget_scale=budget_scale,
                    generation=_child_generation(source),
                    parent_child_id=int(source.get("child_id", -1)),
                    preserve_first_child_id=True,
                )
                fragments.extend(made)
            if replace_source:
                children = [child for child in children if int(child.get("child_id", -2)) not in source_ids]
            out["child_worlds"] = children + fragments
            fragment_count = len(fragments)
            out.setdefault("child_event_history", []).append(
                {
                    "event": "assay_fragment_child_records",
                    "fragment_count": float(fragment_count),
                    "parts": float(parts),
                    "axis": axis,
                    "phase_mode": phase_mode,
                    "replace_source": 1.0 if replace_source else 0.0,
                }
            )

    grandchild_count = 0
    if _assay_flag("enable_grandchild_probe"):
        depth = max(1, int(_assay_value("grandchild_probe_depth", 1) or 1))
        parts = max(1, int(_assay_value("grandchild_probe_parts", 2) or 2))
        budget_scale = float(_assay_value("grandchild_probe_budget_scale", 0.5) or 0.5)
        replace_parent = _assay_flag("grandchild_probe_replace_parent")
        flatten = _assay_flag("grandchild_probe_flatten_to_child_worlds")
        axis = str(_assay_value("fragment_child_axis", "support_window") or "support_window")
        phase_mode = str(_assay_value("fragment_child_phase_mode", "preserve") or "preserve")
        sources = [_select_strongest_child(out)] if _select_strongest_child(out) is not None else []
        all_new: list[dict[str, Any]] = []
        replaced_ids: set[int] = set()
        for _depth_idx in range(depth):
            next_sources: list[dict[str, Any]] = []
            for source in sources:
                if source is None:
                    continue
                parent_id = int(source.get("child_id", -1))
                made, _ = _fragment_child_record(
                    source,
                    state=out,
                    parts=parts,
                    axis=axis,
                    phase_mode=phase_mode,
                    budget_scale=budget_scale,
                    generation=_child_generation(source) + 1,
                    parent_child_id=parent_id,
                    preserve_first_child_id=False,
                )
                for idx, child in enumerate(made):
                    child["grandchild_index"] = int(idx)
                    child["grandchild_probe"] = True
                next_sources = made
                all_new.extend(made)
                if replace_parent:
                    replaced_ids.add(parent_id)
            sources = next_sources
        if flatten and all_new:
            children = [child for child in out.get("child_worlds", []) if int(child.get("child_id", -2)) not in replaced_ids]
            out["child_worlds"] = children + all_new
        grandchild_count = len(all_new)
        out.setdefault("child_event_history", []).append(
            {
                "event": "assay_grandchild_probe",
                "grandchild_count": float(grandchild_count),
                "depth": float(depth),
                "parts": float(parts),
                "replace_parent": 1.0 if replace_parent else 0.0,
                "flattened": 1.0 if flatten else 0.0,
            }
        )

    after = _child_volume_summary(out)
    metrics = {
        "applied": 1.0
        if (
            fragment_count > 0
            or grandchild_count > 0
            or float(volume_metrics.get("volume_removed_count", 0.0)) > 0.0
        )
        else 0.0,
        "before_total_count": before["total_count"],
        "before_active_count": before["active_count"],
        "before_effective_count": before["effective_count"],
        "before_top_score_share": before["top_score_share"],
        "before_grandchild_count": before["grandchild_count"],
        "after_total_count": after["total_count"],
        "after_active_count": after["active_count"],
        "after_effective_count": after["effective_count"],
        "after_top_score_share": after["top_score_share"],
        "after_support_mass": after["support_mass"],
        "after_mean_support": after["mean_support"],
        "after_grandchild_count": after["grandchild_count"],
        "after_max_generation": after["max_generation"],
        "volume_removed_count": float(volume_metrics.get("volume_removed_count", 0.0)),
        "fragment_count": float(fragment_count),
        "grandchild_count": float(grandchild_count),
        "child_max_worlds_cfg": float(getattr(cfg, "child_max_worlds", 0)),
    }
    return out, metrics


def _phase_delta_from_to(source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    cross = source[..., 0] * target[..., 1] - source[..., 1] * target[..., 0]
    dot = source[..., 0] * target[..., 0] + source[..., 1] * target[..., 1]
    return torch.atan2(cross, dot)


def _support_weighted_mean(
    value: torch.Tensor,
    weight: torch.Tensor,
    *,
    dims: tuple[int, ...],
) -> torch.Tensor:
    numer = (value * weight).sum(dim=dims, keepdim=True)
    denom = weight.sum(dim=dims, keepdim=True).clamp_min(1.0e-8)
    return numer / denom


def _low_rank_child_phase_delta(delta: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
    if delta.ndim < 3:
        return delta
    freq_profile = _support_weighted_mean(delta, gate, dims=(-1,))
    time_profile = _support_weighted_mean(delta, gate, dims=(-2,))
    global_profile = _support_weighted_mean(delta, gate, dims=(-2, -1))
    return (freq_profile + time_profile - global_profile).clamp(-math.pi, math.pi)


def _weighted_abs_phase_error(a: torch.Tensor, b: torch.Tensor, weight: torch.Tensor) -> float:
    delta = _phase_delta_from_to(a, b).abs()
    w = weight.clamp(0.0, 1.0)
    denom = float(w.sum().clamp_min(1.0e-8).item())
    return float((delta * w).sum().item() / denom)


def _support_boundary_mask(support: torch.Tensor) -> torch.Tensor:
    if support.numel() == 0:
        return torch.zeros_like(support)
    time_edge = (support - torch.roll(support, shifts=1, dims=-1)).abs()
    time_edge = torch.maximum(time_edge, (support - torch.roll(support, shifts=-1, dims=-1)).abs())
    if support.dim() >= 3:
        freq_edge = (support - torch.roll(support, shifts=1, dims=-2)).abs()
        freq_edge = torch.maximum(freq_edge, (support - torch.roll(support, shifts=-1, dims=-2)).abs())
        edge = torch.maximum(time_edge, freq_edge)
    else:
        edge = time_edge
    return torch.maximum(edge, 0.10 * support).clamp(0.0, 1.0)


def _weighted_mean_scalar(value: torch.Tensor, weight: torch.Tensor) -> float:
    w = weight.clamp(0.0, 1.0)
    denom = float(w.sum().clamp_min(1.0e-8).item())
    return float((value * w).sum().item() / denom)


def _mode1_occupancy_mean(state: dict[str, Any], cfg: CircleworldConfig, weight: torch.Tensor | None = None) -> float:
    if not isinstance(state, dict) or "mode_logits" not in state or "mode_support" not in state:
        return 0.0
    logits = state["mode_logits"]
    support = state["mode_support"].clamp_min(0.0)
    weights = torch.softmax(logits / max(1e-4, float(cfg.readout_temperature)), dim=-1) * support.clamp_min(1e-4)
    occ = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1.0e-8)
    mode1 = occ[..., 1]
    if weight is not None:
        return _weighted_mean_scalar(mode1, weight)
    return float(mode1.mean().item())


def _empty_child_predictive_metrics() -> dict[str, float]:
    return {
        "enabled": 0.0,
        "applied": 0.0,
        "child_id": -1.0,
        "baseline_error": 0.0,
        "candidate_error": 0.0,
        "residual_reduction": 0.0,
        "positive_residual_reduction": 0.0,
        "boundary_match": 0.0,
        "support_gate_mean": 0.0,
        "support_gate_max": 0.0,
        "explained_parent_defect": 0.0,
        "target_horizon": 1.0,
        "target_residual_abs_mean": 0.0,
        "raw_child_delta_abs_mean": 0.0,
        "candidate_delta_abs_mean": 0.0,
        "oracle_candidate_error": 0.0,
        "oracle_residual_reduction": 0.0,
        "learned_parent_bridge_score": 0.0,
        "learned_parent_bridge_acceptance": 0.0,
        "learned_parent_bridge_authority_acceptance": 0.0,
        "learned_parent_bridge_boundary_score": 0.0,
        "learned_parent_bridge_volume_score": 0.0,
        "learned_parent_bridge_support_score": 0.0,
        "learned_parent_bridge_coherence_score": 0.0,
        "learned_parent_bridge_delta_scale": 0.0,
        "learned_parent_bridge_prewrite_gate": 0.0,
        "learned_parent_bridge_effective_delta_scale": 0.0,
        "parent_phase_projector_dataset_exported": 0.0,
        "ontology_binding_mass": 0.0,
        "parent_ontology_charge_mean": 0.0,
        "parent_ontology_charge_max": 0.0,
        "parent_ontology_mode1_occupancy_before": 0.0,
        "parent_ontology_mode1_occupancy_after": 0.0,
        "applied_delta_abs_mean": 0.0,
        "mode1_support_boost": 0.0,
        "mode1_logit_boost": 0.0,
    }


def _child_gate_for_predictive_binding(child: dict[str, Any]) -> torch.Tensor | None:
    support = child.get("mode_support")
    phase = child.get("phase_state")
    if not torch.is_tensor(support) or not torch.is_tensor(phase):
        return None
    if torch.is_tensor(child.get("mode_coherence")):
        coherence = child["mode_coherence"].clamp(0.0, 1.0)
    else:
        coherence = torch.ones_like(support)
    support_power = max(0.05, float(_assay_value("child_predictive_support_power", 1.0) or 1.0))
    gate = support.clamp(0.0, 1.0).pow(support_power) * (0.20 + 0.80 * coherence)
    gate_floor = float(_assay_value("child_predictive_gate_floor", 0.0) or 0.0)
    if gate_floor > 0.0:
        gate = torch.maximum(gate, torch.full_like(gate, max(0.0, min(1.0, gate_floor))))
    return gate.clamp(0.0, 1.0)


def _sigmoid_unit(value: float) -> float:
    x = max(-60.0, min(60.0, float(value)))
    return float(1.0 / (1.0 + math.exp(-x)))


PARENT_AUTHORITY_FEATURES: tuple[str, ...] = (
    "bias",
    "boundary_match",
    "boundary_score",
    "support_gate_mean",
    "support_score",
    "support_gate_max",
    "raw_low_rank_abs_mean",
    "raw_low_rank_sq_mean",
    "parent_tangent_abs_mean",
    "boundary_weighted_mean",
    "raw_abs_x_support",
    "raw_abs_x_boundary",
    "boundary_x_support",
    "support_x_boundary_score",
    "raw_minus_tangent_abs_mean",
)


def _weighted_scalar_feature_mean(value: torch.Tensor | None, gate: torch.Tensor) -> float:
    if not torch.is_tensor(value) or value.shape != gate.shape:
        return 0.0
    weight = gate.clamp(0.0, 1.0)
    denom = float(weight.sum().clamp_min(1.0e-8).item())
    return float((value * weight).sum().item() / denom)


def _parent_authority_scalar_features(
    *,
    boundary_match: float,
    boundary_score: float,
    support_score: float,
    gate: torch.Tensor,
    raw_low_rank_delta: torch.Tensor | None,
    parent_tangent_low_rank_delta: torch.Tensor | None,
    boundary_mask: torch.Tensor | None,
) -> list[float]:
    support_mean = float(gate.mean().item())
    support_max = float(gate.max().item())
    raw_abs = _weighted_scalar_feature_mean(raw_low_rank_delta.abs() if torch.is_tensor(raw_low_rank_delta) else None, gate)
    raw_sq = _weighted_scalar_feature_mean((raw_low_rank_delta * raw_low_rank_delta) if torch.is_tensor(raw_low_rank_delta) else None, gate)
    tangent_abs = _weighted_scalar_feature_mean(parent_tangent_low_rank_delta.abs() if torch.is_tensor(parent_tangent_low_rank_delta) else None, gate)
    boundary_mean = _weighted_scalar_feature_mean(boundary_mask.clamp(0.0, 1.0) if torch.is_tensor(boundary_mask) else None, gate)
    return [
        1.0,
        float(boundary_match),
        float(boundary_score),
        support_mean,
        float(support_score),
        support_max,
        raw_abs,
        raw_sq,
        tangent_abs,
        boundary_mean,
        raw_abs * support_mean,
        raw_abs * boundary_mean,
        boundary_mean * support_mean,
        support_mean * float(boundary_score),
        abs(raw_abs - tangent_abs),
    ]


def _linear_parent_authority_acceptance(features: list[float]) -> float | None:
    coeffs_raw = _assay_value("learned_parent_bridge_authority_coeffs", [])
    names_raw = _assay_value("learned_parent_bridge_authority_feature_names", [])
    if not isinstance(coeffs_raw, (list, tuple)) or not coeffs_raw:
        return None
    if isinstance(names_raw, (list, tuple)) and names_raw and list(names_raw) != list(PARENT_AUTHORITY_FEATURES):
        return None
    if len(coeffs_raw) != len(features):
        return None
    value = sum(float(weight) * float(feature) for weight, feature in zip(coeffs_raw, features))
    return float(max(0.0, min(1.0, value)))


def _score_learned_parent_bridge(
    state: dict[str, Any],
    child: dict[str, Any],
    gate: torch.Tensor,
    *,
    boundary_match: float,
    raw_low_rank_delta: torch.Tensor | None = None,
    parent_tangent_low_rank_delta: torch.Tensor | None = None,
    boundary_mask: torch.Tensor | None = None,
) -> dict[str, float]:
    volume = _child_volume_summary(state)
    boundary_center = float(_assay_value("learned_parent_bridge_boundary_center", 0.952) or 0.952)
    boundary_scale = float(_assay_value("learned_parent_bridge_boundary_scale", 120.0) or 120.0)
    boundary_score = _sigmoid_unit((float(boundary_match) - boundary_center) * boundary_scale)

    effective_target = max(1.0e-6, float(_assay_value("learned_parent_bridge_volume_effective_target", 4.80) or 4.80))
    top_share_target = max(1.0e-6, float(_assay_value("learned_parent_bridge_top_share_target", 0.26) or 0.26))
    effective_score = max(0.0, min(1.0, float(volume.get("effective_count", 0.0)) / effective_target))
    top_share_score = max(0.0, min(1.0, float(volume.get("top_score_share", 0.0)) / top_share_target))
    volume_score = 0.50 * effective_score + 0.50 * top_share_score

    support_target = max(1.0e-6, float(_assay_value("learned_parent_bridge_support_target", 0.158) or 0.158))
    support_score = max(0.0, min(1.0, float(gate.mean().item()) / support_target))

    coherence_tensor = child.get("mode_coherence")
    if torch.is_tensor(coherence_tensor):
        coherence_score = _weighted_mean_scalar(coherence_tensor.clamp(0.0, 1.0), gate)
    else:
        coherence_score = 1.0

    weights = {
        "boundary": max(0.0, float(_assay_value("learned_parent_bridge_boundary_weight", 0.50) or 0.50)),
        "volume": max(0.0, float(_assay_value("learned_parent_bridge_volume_weight", 0.25) or 0.25)),
        "support": max(0.0, float(_assay_value("learned_parent_bridge_support_weight", 0.15) or 0.15)),
        "coherence": max(0.0, float(_assay_value("learned_parent_bridge_coherence_weight", 0.10) or 0.10)),
    }
    denom = max(1.0e-8, sum(weights.values()))
    score = (
        weights["boundary"] * boundary_score
        + weights["volume"] * volume_score
        + weights["support"] * support_score
        + weights["coherence"] * coherence_score
    ) / denom

    acceptance_target = max(0.0, float(_assay_value("learned_parent_bridge_acceptance_target", 0.13) or 0.13))
    acceptance_floor = max(0.0, float(_assay_value("learned_parent_bridge_acceptance_floor", 0.0) or 0.0))
    threshold = max(0.0, min(1.0, float(_assay_value("learned_parent_bridge_score_threshold", 0.50) or 0.50)))
    authority_features = _parent_authority_scalar_features(
        boundary_match=boundary_match,
        boundary_score=boundary_score,
        support_score=support_score,
        gate=gate,
        raw_low_rank_delta=raw_low_rank_delta,
        parent_tangent_low_rank_delta=parent_tangent_low_rank_delta,
        boundary_mask=boundary_mask,
    )
    trained_authority = _linear_parent_authority_acceptance(authority_features)
    authority_source = str(_assay_value("learned_parent_bridge_authority_source", "") or "").strip().lower()
    acceptance = 0.0
    if score >= threshold:
        if authority_source in {"trained_linear_authority", "learned_authority", "authority_head"} and trained_authority is not None:
            acceptance = max(acceptance_floor, trained_authority)
        else:
            acceptance = max(acceptance_floor, acceptance_target * score)
    return {
        "score": float(max(0.0, min(1.0, score))),
        "acceptance": float(max(0.0, min(1.0, acceptance))),
        "authority_acceptance": float(trained_authority) if trained_authority is not None else 0.0,
        "boundary_score": float(max(0.0, min(1.0, boundary_score))),
        "volume_score": float(max(0.0, min(1.0, volume_score))),
        "support_score": float(max(0.0, min(1.0, support_score))),
        "coherence_score": float(max(0.0, min(1.0, coherence_score))),
        "delta_scale": float(_assay_value("learned_parent_bridge_delta_scale", 0.0) or 0.0),
        "threshold": float(threshold),
    }


PARENT_PHASE_PROJECTOR_FEATURES: tuple[str, ...] = (
    "bias",
    "raw_child_delta",
    "parent_tangent_delta",
    "signed_parent_tangent",
    "gate",
    "boundary_mask",
    "child_parent_alignment_centered",
    "raw_times_gate",
    "tangent_times_gate",
    "raw_times_boundary",
    "tangent_times_boundary",
    "raw_times_alignment",
    "tangent_times_alignment",
)


PARENT_PHASOR_CARRIER_FEATURES: tuple[str, ...] = (
    "bias",
    "parent_cos",
    "parent_sin",
    "raw_delta",
    "sin_raw_delta",
    "cos_raw_delta",
    "raw_low_rank_delta",
    "parent_tangent_delta",
    "parent_tangent_low_rank_delta",
    "gate",
    "boundary_mask",
    "child_parent_alignment_centered",
    "parent_cos_x_cos_raw",
    "parent_sin_x_sin_raw",
    "parent_sin_x_cos_raw",
    "parent_cos_x_sin_raw",
    "gate_x_sin_raw",
    "gate_x_cos_raw",
    "boundary_x_sin_raw",
    "boundary_x_cos_raw",
)


def _parent_phase_projector_features(
    *,
    raw_low_rank_delta: torch.Tensor,
    parent_tangent_low_rank_delta: torch.Tensor,
    gate: torch.Tensor,
    boundary_mask: torch.Tensor,
    boundary_align: torch.Tensor,
) -> torch.Tensor:
    gate = gate.clamp(0.0, 1.0)
    boundary = boundary_mask.clamp(0.0, 1.0)
    alignment = (boundary_align.clamp(0.0, 1.0) - 0.5) * 2.0
    signed_parent_tangent = torch.sign(raw_low_rank_delta) * parent_tangent_low_rank_delta.abs()
    return torch.stack(
        [
            torch.ones_like(gate),
            raw_low_rank_delta,
            parent_tangent_low_rank_delta,
            signed_parent_tangent,
            gate,
            boundary,
            alignment,
            raw_low_rank_delta * gate,
            parent_tangent_low_rank_delta * gate,
            raw_low_rank_delta * boundary,
            parent_tangent_low_rank_delta * boundary,
            raw_low_rank_delta * alignment,
            parent_tangent_low_rank_delta * alignment,
        ],
        dim=0,
    )


def _parent_phasor_carrier_features(
    *,
    parent_mode1: torch.Tensor,
    raw_delta: torch.Tensor,
    raw_low_rank_delta: torch.Tensor,
    parent_tangent_delta: torch.Tensor,
    parent_tangent_low_rank_delta: torch.Tensor,
    gate: torch.Tensor,
    boundary_mask: torch.Tensor,
    boundary_align: torch.Tensor,
) -> torch.Tensor:
    parent_cos = parent_mode1[..., 0]
    parent_sin = parent_mode1[..., 1]
    sin_raw = torch.sin(raw_delta)
    cos_raw = torch.cos(raw_delta)
    gate = gate.clamp(0.0, 1.0)
    boundary = boundary_mask.clamp(0.0, 1.0)
    alignment = (boundary_align.clamp(0.0, 1.0) - 0.5) * 2.0
    return torch.stack(
        [
            torch.ones_like(gate),
            parent_cos,
            parent_sin,
            raw_delta,
            sin_raw,
            cos_raw,
            raw_low_rank_delta,
            parent_tangent_delta,
            parent_tangent_low_rank_delta,
            gate,
            boundary,
            alignment,
            parent_cos * cos_raw,
            parent_sin * sin_raw,
            parent_sin * cos_raw,
            parent_cos * sin_raw,
            gate * sin_raw,
            gate * cos_raw,
            boundary * sin_raw,
            boundary * cos_raw,
        ],
        dim=0,
    )


def _linear_parent_phase_projector_delta(features: torch.Tensor) -> torch.Tensor | None:
    coeffs_raw = _assay_value("learned_parent_bridge_projector_coeffs", [])
    if not isinstance(coeffs_raw, (list, tuple)) or len(coeffs_raw) == 0:
        return None
    coeffs = torch.tensor([float(item) for item in coeffs_raw], dtype=features.dtype, device=features.device)
    if int(coeffs.numel()) != int(features.shape[0]):
        return None
    view_shape = (int(coeffs.numel()),) + (1,) * (features.ndim - 1)
    return (features * coeffs.view(view_shape)).sum(dim=0).clamp(-math.pi, math.pi)


def _linear_parent_phasor_carrier_phase(features: torch.Tensor) -> torch.Tensor | None:
    cos_raw = _assay_value("learned_parent_bridge_phasor_carrier_cos_coeffs", [])
    sin_raw = _assay_value("learned_parent_bridge_phasor_carrier_sin_coeffs", [])
    if not isinstance(cos_raw, (list, tuple)) or not isinstance(sin_raw, (list, tuple)):
        return None
    if len(cos_raw) == 0 or len(sin_raw) == 0 or len(cos_raw) != len(sin_raw):
        return None
    cos_coeffs = torch.tensor([float(item) for item in cos_raw], dtype=features.dtype, device=features.device)
    sin_coeffs = torch.tensor([float(item) for item in sin_raw], dtype=features.dtype, device=features.device)
    if int(cos_coeffs.numel()) != int(features.shape[0]):
        return None
    view_shape = (int(cos_coeffs.numel()),) + (1,) * (features.ndim - 1)
    cos_part = (features * cos_coeffs.view(view_shape)).sum(dim=0)
    sin_part = (features * sin_coeffs.view(view_shape)).sum(dim=0)
    return phasor_normalize(torch.stack([cos_part, sin_part], dim=-1))


def _linear_parent_phase_carrier_delta(features: torch.Tensor) -> torch.Tensor | None:
    coeffs_raw = _assay_value("learned_parent_bridge_carrier_coeffs", [])
    if not isinstance(coeffs_raw, (list, tuple)) or len(coeffs_raw) == 0:
        return None
    coeffs = torch.tensor([float(item) for item in coeffs_raw], dtype=features.dtype, device=features.device)
    if int(coeffs.numel()) != int(features.shape[0]):
        return None
    view_shape = (int(coeffs.numel()),) + (1,) * (features.ndim - 1)
    return (features * coeffs.view(view_shape)).sum(dim=0).clamp(-math.pi, math.pi)


def _export_parent_phase_projector_example(
    *,
    context: dict[str, Any] | None,
    features: torch.Tensor,
    child_phase: torch.Tensor,
    parent_mode0: torch.Tensor,
    parent_mode1: torch.Tensor,
    baseline_mode1: torch.Tensor,
    target_mode1: torch.Tensor,
    raw_delta: torch.Tensor,
    parent_tangent_delta: torch.Tensor,
    target_residual_delta: torch.Tensor,
    target_delta: torch.Tensor,
    candidate_delta: torch.Tensor,
    gate: torch.Tensor,
    boundary_mask: torch.Tensor,
    raw_low_rank_delta: torch.Tensor,
    parent_tangent_low_rank_delta: torch.Tensor,
    child: dict[str, Any],
    metrics: dict[str, float],
) -> str | None:
    if not _assay_flag("child_predictive_export_dataset"):
        return None
    dataset_dir = str(_assay_value("child_predictive_dataset_dir", "") or "").strip()
    if not dataset_dir and context:
        dataset_dir = str(context.get("dataset_dir") or "")
    if not dataset_dir:
        return None
    out_dir = Path(dataset_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    case_name = str((context or {}).get("case", "unknown_case")).replace("\\", "_").replace("/", "_")
    branch_name = str((context or {}).get("branch", "unknown_branch")).replace("\\", "_").replace("/", "_")
    tag = str(_assay_value("child_predictive_dataset_tag", "parent_phase_projector") or "parent_phase_projector")
    file_name = f"{tag}__{case_name}__{branch_name}.npz"
    out_path = out_dir / file_name
    np.savez_compressed(
        out_path,
        features=features.detach().cpu().numpy().astype(np.float32),
        child_phase=child_phase.detach().cpu().numpy().astype(np.float32),
        parent_mode0=parent_mode0.detach().cpu().numpy().astype(np.float32),
        parent_mode1=parent_mode1.detach().cpu().numpy().astype(np.float32),
        baseline_mode1=baseline_mode1.detach().cpu().numpy().astype(np.float32),
        target_mode1=target_mode1.detach().cpu().numpy().astype(np.float32),
        raw_delta=raw_delta.detach().cpu().numpy().astype(np.float32),
        parent_tangent_delta=parent_tangent_delta.detach().cpu().numpy().astype(np.float32),
        target_residual_delta=target_residual_delta.detach().cpu().numpy().astype(np.float32),
        target_delta=target_delta.detach().cpu().numpy().astype(np.float32),
        candidate_delta=candidate_delta.detach().cpu().numpy().astype(np.float32),
        gate=gate.detach().cpu().numpy().astype(np.float32),
        boundary_mask=boundary_mask.detach().cpu().numpy().astype(np.float32),
        raw_child_delta=raw_low_rank_delta.detach().cpu().numpy().astype(np.float32),
        parent_tangent_low_rank_delta=parent_tangent_low_rank_delta.detach().cpu().numpy().astype(np.float32),
        feature_names=np.array(PARENT_PHASE_PROJECTOR_FEATURES, dtype=object),
        case=np.array(case_name, dtype=object),
        branch=np.array(branch_name, dtype=object),
        child_id=np.array(float(child.get("child_id", -1)), dtype=np.float32),
        metrics=np.array(json.dumps(metrics, sort_keys=True), dtype=object),
    )
    return str(out_path)


def _apply_child_predictive_assimilation(
    state: Any,
    cfg: CircleworldConfig,
    *,
    context: dict[str, Any] | None = None,
) -> tuple[Any, dict[str, float]]:
    if not isinstance(state, dict) or "phase_modes" not in state:
        return state, _empty_child_predictive_metrics()
    if not (_assay_flag("enable_child_predictive_probe") or _assay_flag("enable_child_predictive_assimilation")):
        return state, _empty_child_predictive_metrics()
    child = _select_strongest_child(state)
    if child is None:
        return state, _empty_child_predictive_metrics()
    child_phase = child.get("phase_state")
    gate = _child_gate_for_predictive_binding(child)
    if not torch.is_tensor(child_phase) or gate is None:
        return state, _empty_child_predictive_metrics()

    # The child earns ontology by predicting the child-enabled next parent state better than parent-only dynamics.
    no_child_cfg = _without_childworld_runtime_cfg(cfg)
    parent_only_state = _strip_child_worlds_for_killswitch(state, cfg)
    target_horizon = max(1, int(_assay_value("child_predictive_target_horizon", 1) or 1))
    target_run = _run_depth_trace(state, cfg=cfg, depth=target_horizon, mode=cfg.branching_mode)
    baseline_run = _run_depth_trace(parent_only_state, cfg=no_child_cfg, depth=target_horizon, mode=no_child_cfg.branching_mode)
    target_state = target_run.get("state_bundle")
    baseline_state = baseline_run.get("state_bundle")
    if not isinstance(target_state, dict) or not isinstance(baseline_state, dict):
        return state, _empty_child_predictive_metrics()

    parent_mode0 = state["phase_modes"][..., 0, :]
    parent_mode1 = state["phase_modes"][..., 1, :]
    raw_delta = _phase_delta_from_to(parent_mode1, child_phase)
    target_mode1 = target_state["phase_modes"][..., 1, :]
    baseline_mode1 = baseline_state["phase_modes"][..., 1, :]
    parent_tangent_delta = _phase_delta_from_to(parent_mode1, baseline_mode1)
    target_residual_delta = _phase_delta_from_to(baseline_mode1, target_mode1)
    raw_low_rank_delta = _low_rank_child_phase_delta(raw_delta, gate)
    parent_tangent_low_rank_delta = _low_rank_child_phase_delta(parent_tangent_delta, gate)
    oracle_low_rank_delta = _low_rank_child_phase_delta(target_residual_delta, gate)
    boundary_mask = _support_boundary_mask(gate)
    boundary_align = ((_phase_alignment(child_phase, parent_mode1) + 1.0) * 0.5).clamp(0.0, 1.0)
    projector_features = _parent_phase_projector_features(
        raw_low_rank_delta=raw_low_rank_delta,
        parent_tangent_low_rank_delta=parent_tangent_low_rank_delta,
        gate=gate,
        boundary_mask=boundary_mask,
        boundary_align=boundary_align,
    )
    phasor_carrier_features = _parent_phasor_carrier_features(
        parent_mode1=parent_mode1,
        raw_delta=raw_delta,
        raw_low_rank_delta=raw_low_rank_delta,
        parent_tangent_delta=parent_tangent_delta,
        parent_tangent_low_rank_delta=parent_tangent_low_rank_delta,
        gate=gate,
        boundary_mask=boundary_mask,
        boundary_align=boundary_align,
    )
    candidate_mode = str(_assay_value("child_predictive_candidate", "raw_child_delta") or "raw_child_delta").strip().lower()
    oracle_mix = max(0.0, min(1.0, float(_assay_value("child_predictive_oracle_mix", 1.0) or 1.0)))
    learned_bridge_mode = candidate_mode in {"learned_parent_bridge", "scout_parent_bridge", "learned_bridge"}
    trained_projector_delta = _linear_parent_phase_projector_delta(projector_features)
    trained_carrier_delta = _linear_parent_phase_carrier_delta(projector_features)
    trained_phasor_carrier_phase = _linear_parent_phasor_carrier_phase(phasor_carrier_features)
    requested_delta_scale = 1.0
    prewrite_gate = 1.0
    effective_delta_scale = 1.0
    if candidate_mode in {"oracle_parent_residual", "oracle_residual", "parent_residual"}:
        low_rank_delta = oracle_low_rank_delta
    elif candidate_mode in {"blend_raw_oracle", "hybrid"}:
        low_rank_delta = (1.0 - oracle_mix) * raw_low_rank_delta + oracle_mix * oracle_low_rank_delta
    elif learned_bridge_mode:
        delta_scale = float(_assay_value("learned_parent_bridge_delta_scale", 0.0) or 0.0)
        requested_delta_scale = max(0.0, delta_scale)
        delta_source = str(_assay_value("learned_parent_bridge_delta_source", "raw_child_delta") or "raw_child_delta").strip().lower()
        if delta_source in {"trained_linear_projector", "linear_parent_projector", "learned_parent_projector"} and trained_projector_delta is not None:
            base_delta = trained_projector_delta
        elif delta_source in {"parent_tangent", "parent_only_tangent", "parent_law_tangent"}:
            base_delta = parent_tangent_low_rank_delta
        elif delta_source in {"anti_parent_tangent", "negative_parent_tangent"}:
            base_delta = -parent_tangent_low_rank_delta
        elif delta_source in {"child_signed_parent_tangent", "signed_parent_tangent"}:
            base_delta = torch.sign(raw_low_rank_delta) * parent_tangent_low_rank_delta.abs()
        elif delta_source in {"anti_child_signed_parent_tangent", "negative_child_signed_parent_tangent"}:
            base_delta = -torch.sign(raw_low_rank_delta) * parent_tangent_low_rank_delta.abs()
        else:
            base_delta = raw_low_rank_delta
        prewrite_policy = str(_assay_value("learned_parent_bridge_prewrite_policy", "scale") or "scale").strip().lower()
        if prewrite_policy in {"disabled", "off", "none", "carrier_only", "authority_only", "no_delta", "no_prewrite"}:
            prewrite_gate = 0.0
        else:
            prewrite_gate = 1.0
        effective_delta_scale = requested_delta_scale * prewrite_gate
        low_rank_delta = effective_delta_scale * base_delta
    else:
        low_rank_delta = raw_low_rank_delta
    probe_gain = float(_assay_value("child_predictive_bind_gain", 0.85) or 0.85)
    probe_state = clone_circleworld_state(parent_only_state)
    probe_state["phase_modes"][..., 1, :] = phasor_apply_delta(parent_mode1, (probe_gain * gate * low_rank_delta).clamp(-math.pi, math.pi))
    probe_state["mixed_phase_state"] = extract_phase_state(probe_state, cfg)
    candidate_run = _run_depth_trace(probe_state, cfg=no_child_cfg, depth=target_horizon, mode=no_child_cfg.branching_mode)
    candidate_state = candidate_run.get("state_bundle")
    if not isinstance(candidate_state, dict):
        return state, _empty_child_predictive_metrics()

    candidate_mode1 = candidate_state["phase_modes"][..., 1, :]
    baseline_error = _weighted_abs_phase_error(baseline_mode1, target_mode1, gate)
    candidate_error = _weighted_abs_phase_error(candidate_mode1, target_mode1, gate)
    residual_reduction = float((baseline_error - candidate_error) / max(1.0e-8, baseline_error))
    positive_reduction = max(0.0, residual_reduction)
    oracle_probe_state = clone_circleworld_state(parent_only_state)
    oracle_probe_state["phase_modes"][..., 1, :] = phasor_apply_delta(parent_mode1, (probe_gain * gate * oracle_low_rank_delta).clamp(-math.pi, math.pi))
    oracle_probe_state["mixed_phase_state"] = extract_phase_state(oracle_probe_state, cfg)
    oracle_run = _run_depth_trace(oracle_probe_state, cfg=no_child_cfg, depth=target_horizon, mode=no_child_cfg.branching_mode)
    oracle_state = oracle_run.get("state_bundle")
    if isinstance(oracle_state, dict):
        oracle_candidate_error = _weighted_abs_phase_error(oracle_state["phase_modes"][..., 1, :], target_mode1, gate)
        oracle_residual_reduction = float((baseline_error - oracle_candidate_error) / max(1.0e-8, baseline_error))
    else:
        oracle_candidate_error = 0.0
        oracle_residual_reduction = 0.0

    boundary_match = _weighted_mean_scalar(boundary_align, boundary_mask)
    explained_defect = float(max(0.0, baseline_error - candidate_error))
    learned_bridge_scores = (
        _score_learned_parent_bridge(
            state,
            child,
            gate,
            boundary_match=boundary_match,
            raw_low_rank_delta=raw_low_rank_delta,
            parent_tangent_low_rank_delta=parent_tangent_low_rank_delta,
            boundary_mask=boundary_mask,
        )
        if learned_bridge_mode
        else {}
    )
    acceptance = (
        float(learned_bridge_scores.get("acceptance", 0.0))
        if learned_bridge_mode
        else positive_reduction * max(0.0, min(1.0, boundary_match))
    )
    min_reduction = float(_assay_value("child_predictive_min_residual_reduction", 0.0) or 0.0)
    min_boundary = float(_assay_value("child_predictive_min_boundary_match", 0.0) or 0.0)
    if learned_bridge_mode:
        should_apply = bool(
            _assay_flag("enable_child_predictive_assimilation")
            and boundary_match >= min_boundary
            and float(learned_bridge_scores.get("score", 0.0)) >= float(learned_bridge_scores.get("threshold", 0.0))
            and acceptance > 0.0
        )
    else:
        should_apply = bool(
            _assay_flag("enable_child_predictive_assimilation")
            and residual_reduction >= min_reduction
            and boundary_match >= min_boundary
            and acceptance > 0.0
        )

    out = clone_circleworld_state(state)
    applied_delta = out["phase_modes"][..., 1, :].new_zeros(gate.shape)
    support_boost = out["phase_modes"][..., 1, :].new_zeros(gate.shape)
    logit_boost = out["phase_modes"][..., 1, :].new_zeros(gate.shape)
    ontology_charge = out["phase_modes"][..., 1, :].new_zeros(gate.shape)
    mode1_occupancy_before = _mode1_occupancy_mean(out, cfg, gate)
    mode1_occupancy_after = mode1_occupancy_before
    if should_apply:
        applied_delta = (probe_gain * acceptance * gate * low_rank_delta).clamp(-math.pi, math.pi)
        out["phase_modes"][..., 1, :] = phasor_apply_delta(out["phase_modes"][..., 1, :], applied_delta)
        binding_gate = (acceptance * gate).clamp(0.0, 1.0)
        if _assay_flag("enable_parent_ontology_field"):
            charge_gain = float(_assay_value("parent_ontology_charge_gain", 1.0) or 1.0)
            ontology_charge = (charge_gain * binding_gate).clamp(0.0, 1.0)
            if _assay_flag("disable_parent_ontology_charge"):
                ontology_charge = torch.zeros_like(ontology_charge)
            existing_charge = out.get("ontology_charge")
            if torch.is_tensor(existing_charge) and existing_charge.shape == ontology_charge.shape:
                ontology_charge = torch.maximum(existing_charge, ontology_charge).clamp(0.0, 1.0)
            out["ontology_charge"] = ontology_charge
            out["mode_child_binding"] = ontology_charge
            out["ontology_phase_delta"] = low_rank_delta.detach().clone()
            if trained_carrier_delta is not None:
                out["ontology_carrier_delta"] = trained_carrier_delta.detach().clone()
            if trained_phasor_carrier_phase is not None:
                out["ontology_carrier_phase"] = trained_phasor_carrier_phase.detach().clone()
            out["ontology_child_id"] = int(child.get("child_id", -1))
        if "mode_support" in out and not _assay_flag("disable_parent_ontology_mode_support"):
            support_boost = binding_gate
            out["mode_support"][..., 1] = torch.maximum(out["mode_support"][..., 1], binding_gate).clamp(0.0, 1.0)
        if "mode_logits" in out and not _assay_flag("disable_parent_ontology_mode_logits"):
            logit_gain = float(_assay_value("child_predictive_logit_gain", 0.35) or 0.35)
            logit_boost = logit_gain * binding_gate
            out["mode_logits"][..., 1] = out["mode_logits"][..., 1] + logit_boost
        out["mixed_phase_state"] = extract_phase_state(out, cfg)
        mode1_occupancy_after = _mode1_occupancy_mean(out, cfg, gate)
        out.setdefault("child_event_history", []).append(
            {
                "event": "assay_child_predictive_assimilation",
                "child_id": float(child.get("child_id", -1)),
                "residual_reduction": float(residual_reduction),
                "boundary_match": float(boundary_match),
                "ontology_binding_mass": float((acceptance * gate).mean().item()),
                "parent_ontology_charge_mean": float(ontology_charge.mean().item()),
            }
        )

    metrics = _empty_child_predictive_metrics()
    metrics.update(
        {
            "enabled": 1.0,
            "applied": 1.0 if should_apply else 0.0,
            "child_id": float(child.get("child_id", -1)),
            "baseline_error": float(baseline_error),
            "candidate_error": float(candidate_error),
            "residual_reduction": float(residual_reduction),
            "positive_residual_reduction": float(positive_reduction),
            "boundary_match": float(boundary_match),
            "support_gate_mean": float(gate.mean().item()),
            "support_gate_max": float(gate.max().item()),
            "explained_parent_defect": float(explained_defect),
            "target_horizon": float(target_horizon),
            "target_residual_abs_mean": float((target_residual_delta.abs() * gate).sum().item() / gate.sum().clamp_min(1.0e-8).item()),
            "raw_child_delta_abs_mean": float((raw_low_rank_delta.abs() * gate).sum().item() / gate.sum().clamp_min(1.0e-8).item()),
            "candidate_delta_abs_mean": float((low_rank_delta.abs() * gate).sum().item() / gate.sum().clamp_min(1.0e-8).item()),
            "oracle_candidate_error": float(oracle_candidate_error),
            "oracle_residual_reduction": float(oracle_residual_reduction),
            "learned_parent_bridge_score": float(learned_bridge_scores.get("score", 0.0)),
            "learned_parent_bridge_acceptance": float(learned_bridge_scores.get("acceptance", 0.0)),
            "learned_parent_bridge_authority_acceptance": float(learned_bridge_scores.get("authority_acceptance", 0.0)),
            "learned_parent_bridge_boundary_score": float(learned_bridge_scores.get("boundary_score", 0.0)),
            "learned_parent_bridge_volume_score": float(learned_bridge_scores.get("volume_score", 0.0)),
            "learned_parent_bridge_support_score": float(learned_bridge_scores.get("support_score", 0.0)),
            "learned_parent_bridge_coherence_score": float(learned_bridge_scores.get("coherence_score", 0.0)),
            "learned_parent_bridge_delta_scale": float(learned_bridge_scores.get("delta_scale", 0.0)),
            "learned_parent_bridge_prewrite_gate": float(prewrite_gate) if learned_bridge_mode else 0.0,
            "learned_parent_bridge_effective_delta_scale": float(effective_delta_scale) if learned_bridge_mode else 0.0,
            "ontology_binding_mass": float((acceptance * gate).mean().item()) if should_apply else 0.0,
            "parent_ontology_charge_mean": float(ontology_charge.mean().item()) if should_apply else 0.0,
            "parent_ontology_charge_max": float(ontology_charge.max().item()) if should_apply else 0.0,
            "parent_ontology_mode1_occupancy_before": float(mode1_occupancy_before),
            "parent_ontology_mode1_occupancy_after": float(mode1_occupancy_after),
            "applied_delta_abs_mean": float((applied_delta.abs() * gate).mean().item()) if should_apply else 0.0,
            "mode1_support_boost": float(support_boost.mean().item()) if should_apply else 0.0,
            "mode1_logit_boost": float(logit_boost.mean().item()) if should_apply else 0.0,
        }
    )
    exported_path = _export_parent_phase_projector_example(
        context=context,
        features=projector_features,
        child_phase=child_phase,
        parent_mode0=parent_mode0,
        parent_mode1=parent_mode1,
        baseline_mode1=baseline_mode1,
        target_mode1=target_mode1,
        raw_delta=raw_delta,
        parent_tangent_delta=parent_tangent_delta,
        target_residual_delta=target_residual_delta,
        target_delta=oracle_low_rank_delta,
        candidate_delta=low_rank_delta,
        gate=gate,
        boundary_mask=boundary_mask,
        raw_low_rank_delta=raw_low_rank_delta,
        parent_tangent_low_rank_delta=parent_tangent_low_rank_delta,
        child=child,
        metrics=metrics,
    )
    if exported_path:
        metrics["parent_phase_projector_dataset_exported"] = 1.0
    return out, metrics


def _carry_parent_ontology_fields(source: Any, target: Any) -> Any:
    if not isinstance(source, dict) or not isinstance(target, dict):
        return target
    for key in (
        "ontology_charge",
        "mode_child_binding",
        "ontology_phase_delta",
        "ontology_carrier_delta",
        "ontology_carrier_phase",
        "ontology_child_id",
    ):
        if key in source:
            target[key] = clone_circleworld_state(source[key])
    return target


def _child_mode1_continuation_write(
    cont_state: dict[str, Any],
    child: dict[str, Any],
    *,
    readout_mode: str,
) -> dict[str, float]:
    child_phase = child.get("phase_state")
    child_support = child.get("mode_support")
    child_coherence = child.get("mode_coherence")
    if not torch.is_tensor(child_phase) or not torch.is_tensor(child_support):
        return {
            "enabled": 0.0,
            "gate_mean": 0.0,
            "gate_max": 0.0,
            "raw_delta_abs_mean": 0.0,
            "low_rank_delta_abs_mean": 0.0,
            "applied_delta_abs_mean": 0.0,
        }

    support = child_support.clamp(0.0, 1.0)
    if torch.is_tensor(child_coherence):
        coherence = child_coherence.clamp(0.0, 1.0)
    else:
        coherence = torch.ones_like(support)
    gate = (support * (0.25 + 0.75 * coherence)).clamp(0.0, 1.0)
    parent_mode1 = cont_state["phase_modes"][..., 1, :]
    raw_delta = _phase_delta_from_to(parent_mode1, child_phase)
    low_rank_delta = _low_rank_child_phase_delta(raw_delta, gate)
    write_gain = 0.85 if readout_mode == "child_continuation_dominant" else 0.60
    applied_delta = (float(write_gain) * gate * low_rank_delta).clamp(-math.pi, math.pi)
    cont_state["phase_modes"][..., 1, :] = phasor_apply_delta(parent_mode1, applied_delta)
    if "mode_support" in cont_state:
        cont_state["mode_support"][..., 1] = torch.maximum(cont_state["mode_support"][..., 1], gate).clamp(0.0, 1.0)
    if "mode_logits" in cont_state:
        cont_state["mode_logits"][..., 1] = cont_state["mode_logits"][..., 1] + 0.20 * gate
    return {
        "enabled": 1.0,
        "gate_mean": float(gate.mean().item()),
        "gate_max": float(gate.max().item()),
        "raw_delta_abs_mean": float((raw_delta.abs() * gate).sum().item() / gate.sum().clamp_min(1.0e-8).item()),
        "low_rank_delta_abs_mean": float((low_rank_delta.abs() * gate).sum().item() / gate.sum().clamp_min(1.0e-8).item()),
        "applied_delta_abs_mean": float((applied_delta.abs() * gate).sum().item() / gate.sum().clamp_min(1.0e-8).item()),
        "write_gain": float(write_gain),
    }


def _disabled_continuation_write_metrics(reason: str) -> dict[str, float | str]:
    return {
        "enabled": 0.0,
        "gate_mean": 0.0,
        "gate_max": 0.0,
        "raw_delta_abs_mean": 0.0,
        "low_rank_delta_abs_mean": 0.0,
        "applied_delta_abs_mean": 0.0,
        "write_gain": 0.0,
        "disabled_reason": str(reason),
    }


def _scramble_child_phase_for_killswitch(child_phase: torch.Tensor, *, mode: str = "roll") -> torch.Tensor:
    mode = str(mode or "roll")
    if mode == "zero":
        out = torch.zeros_like(child_phase)
        out[..., 0] = 1.0
        return out
    if mode == "random":
        gen = torch.Generator(device=child_phase.device)
        gen.manual_seed(730193)
        theta = (2.0 * math.pi) * torch.rand(
            child_phase.shape[:-1],
            device=child_phase.device,
            dtype=child_phase.dtype,
            generator=gen,
        )
        return torch.stack([torch.cos(theta), torch.sin(theta)], dim=-1)
    if child_phase.ndim < 4:
        return phasor_normalize(-child_phase)
    freq_shift = max(1, int(child_phase.shape[-3] // 3))
    time_shift = max(1, int(child_phase.shape[-2] // 5))
    return phasor_normalize(torch.roll(child_phase, shifts=(freq_shift, time_shift), dims=(-3, -2)))


def _controlled_child_phase_for_killswitch(
    child: dict[str, Any],
    children: list[dict[str, Any]],
    *,
    mode: str,
) -> torch.Tensor | None:
    child_phase = child.get("phase_state")
    if not torch.is_tensor(child_phase):
        return None
    mode = str(mode or "").strip().lower()
    if mode == "cross_child":
        child_id = int(child.get("child_id", -1))
        for other in children:
            if int(other.get("child_id", -1)) == child_id:
                continue
            other_phase = other.get("phase_state")
            if torch.is_tensor(other_phase) and other_phase.shape == child_phase.shape:
                return phasor_normalize(other_phase)
        return _scramble_child_phase_for_killswitch(child_phase, mode="roll")
    if mode in {"roll", "random", "zero"}:
        return _scramble_child_phase_for_killswitch(child_phase, mode=mode)
    return child_phase


def _clamp_child_identity_for_killswitch(child: dict[str, Any]) -> dict[str, Any]:
    out = dict(child)
    support = out.get("mode_support")
    coherence = out.get("mode_coherence")
    if torch.is_tensor(support):
        out["mode_support"] = torch.maximum(support, torch.full_like(support, 0.36)).clamp(0.0, 1.0)
    if torch.is_tensor(coherence):
        out["mode_coherence"] = torch.maximum(coherence, torch.full_like(coherence, 0.28)).clamp(0.0, 1.0)
    out["writeback_budget"] = max(float(out.get("writeback_budget", 0.0)), 1.0)
    out["active"] = True
    out["collapsed"] = False
    return out


def _strip_child_worlds_for_killswitch(state: dict[str, Any], cfg: CircleworldConfig) -> dict[str, Any]:
    out = clone_circleworld_state(state)
    out["child_worlds"] = []
    out["child_event_history"] = list(out.get("child_event_history", []))
    if "phase_modes" in out:
        out["mixed_phase_state"] = extract_phase_state(out, cfg)
    return out


def _without_childworld_runtime_cfg(cfg: CircleworldConfig) -> CircleworldConfig:
    if cfg.branching_mode != "native_multimode_childworld":
        return cfg
    return replace(cfg, branching_mode="native_multimode")


def _child_by_id(state: dict[str, Any], child_id: int | None) -> dict[str, Any] | None:
    if child_id is None:
        return None
    for child in state.get("child_worlds", []):
        if int(child.get("child_id", -1)) == int(child_id):
            return child
    return None


def _identity_carry_thresholds() -> dict[str, float]:
    return {
        "support_min": IDENTITY_CARRY_SUPPORT_MIN,
        "coherence_min": IDENTITY_CARRY_COHERENCE_MIN,
        "budget_min": IDENTITY_CARRY_BUDGET_MIN,
        "budget_retained_min": IDENTITY_CARRY_BUDGET_RETAINED_MIN,
        "qualified_fraction_min": IDENTITY_CARRY_FRACTION_MIN,
        "qualified_steps_min": IDENTITY_CARRY_STEPS_MIN,
    }


def _child_state_identity_metrics(
    state: dict[str, Any],
    child_id: int | None,
) -> dict[str, float]:
    child = _child_by_id(state, child_id)
    thresholds = _identity_carry_thresholds()
    if child is None or not child.get("active", False):
        return {
            "active": 0.0,
            "qualified_active": 0.0,
            "support_mean": 0.0,
            "coherence_mean": 0.0,
            "budget": 0.0,
            "parent_divergence": 0.0,
            "sibling_divergence": 0.0,
        }
    child_phase = child.get("phase_state")
    child_support = child.get("mode_support")
    child_coherence = child.get("mode_coherence")
    if not torch.is_tensor(child_phase) or not torch.is_tensor(child_support):
        return {
            "active": 0.0,
            "qualified_active": 0.0,
            "support_mean": 0.0,
            "coherence_mean": 0.0,
            "budget": 0.0,
            "parent_divergence": 0.0,
            "sibling_divergence": 0.0,
        }
    parent_mode0 = state["phase_modes"][..., 0, :]
    parent_mode1 = state["phase_modes"][..., 1, :]
    support_mean = float(child_support.mean().item())
    if torch.is_tensor(child_coherence):
        coherence_mean = float(
            (child_coherence.clamp(0.0, 1.0) * child_support.clamp(0.0, 1.0)).sum().item()
            / child_support.clamp(0.0, 1.0).sum().clamp_min(1e-8).item()
        )
    else:
        coherence_mean = 0.0
    budget = float(child.get("writeback_budget", 0.0))
    parent_div = float(
        ((1.0 - ((_phase_alignment(child_phase, parent_mode0) + 1.0) * 0.5)) * child_support).sum().item()
        / child_support.sum().clamp_min(1e-8).item()
    )
    sibling_div = float(
        ((1.0 - ((_phase_alignment(child_phase, parent_mode1) + 1.0) * 0.5)) * child_support).sum().item()
        / child_support.sum().clamp_min(1e-8).item()
    )
    qualified_active = 1.0 if (
        support_mean >= thresholds["support_min"]
        and coherence_mean >= thresholds["coherence_min"]
        and budget >= thresholds["budget_min"]
    ) else 0.0
    return {
        "active": 1.0,
        "qualified_active": qualified_active,
        "support_mean": support_mean,
        "coherence_mean": coherence_mean,
        "budget": budget,
        "parent_divergence": parent_div,
        "sibling_divergence": sibling_div,
    }


def _branch_identity_summary(
    anchor_child_id: int | None,
    branch_states: list[dict[str, Any]],
) -> dict[str, float]:
    thresholds = _identity_carry_thresholds()
    if anchor_child_id is None or not branch_states:
        return {
            "anchor_child_id": -1.0,
            "same_child_carry_fraction": 0.0,
            "same_child_carry_steps": 0.0,
            "same_child_qualified_carry_fraction": 0.0,
            "same_child_qualified_carry_steps": 0.0,
            "same_child_support_mean": 0.0,
            "same_child_coherence_mean": 0.0,
            "same_child_budget_retained": 0.0,
            "same_child_qualified_budget_retained": 0.0,
            "same_child_min_support": 0.0,
            "same_child_min_coherence": 0.0,
            "same_child_min_budget": 0.0,
            "same_child_parent_divergence": 0.0,
            "same_child_sibling_divergence": 0.0,
            "same_child_support_threshold": thresholds["support_min"],
            "same_child_coherence_threshold": thresholds["coherence_min"],
            "same_child_budget_threshold": thresholds["budget_min"],
            "same_child_budget_retained_threshold": thresholds["budget_retained_min"],
            "same_child_qualified_fraction_threshold": thresholds["qualified_fraction_min"],
            "same_child_qualified_steps_threshold": thresholds["qualified_steps_min"],
        }
    per_state = [_child_state_identity_metrics(state, anchor_child_id) for state in branch_states]
    active_rows = [row for row in per_state if row["active"] > 0.0]
    qualified_rows = [row for row in per_state if row["qualified_active"] > 0.0]
    initial_budget = active_rows[0]["budget"] if active_rows else 0.0
    final_budget = active_rows[-1]["budget"] if active_rows else 0.0
    budget_retained = final_budget / max(1e-8, initial_budget) if initial_budget > 0.0 else final_budget
    initial_qualified_budget = qualified_rows[0]["budget"] if qualified_rows else 0.0
    final_qualified_budget = qualified_rows[-1]["budget"] if qualified_rows else 0.0
    qualified_budget_retained = (
        final_qualified_budget / max(1e-8, initial_qualified_budget)
        if initial_qualified_budget > 0.0
        else final_qualified_budget
    )
    carry_fraction = float(sum(row["active"] for row in per_state) / max(1, len(per_state)))
    carry_steps = float(max(0, sum(1 for row in per_state if row["active"] > 0.0) - 1))
    qualified_carry_fraction = float(sum(row["qualified_active"] for row in per_state) / max(1, len(per_state)))
    qualified_carry_steps = float(max(0, sum(1 for row in per_state if row["qualified_active"] > 0.0) - 1))
    return {
        "anchor_child_id": float(anchor_child_id),
        "same_child_carry_fraction": carry_fraction,
        "same_child_carry_steps": carry_steps,
        "same_child_qualified_carry_fraction": qualified_carry_fraction,
        "same_child_qualified_carry_steps": qualified_carry_steps,
        "same_child_support_mean": float(np.mean([row["support_mean"] for row in active_rows])) if active_rows else 0.0,
        "same_child_coherence_mean": float(np.mean([row["coherence_mean"] for row in active_rows])) if active_rows else 0.0,
        "same_child_budget_retained": float(budget_retained),
        "same_child_qualified_budget_retained": float(qualified_budget_retained),
        "same_child_min_support": float(np.min([row["support_mean"] for row in active_rows])) if active_rows else 0.0,
        "same_child_min_coherence": float(np.min([row["coherence_mean"] for row in active_rows])) if active_rows else 0.0,
        "same_child_min_budget": float(np.min([row["budget"] for row in active_rows])) if active_rows else 0.0,
        "same_child_parent_divergence": float(np.mean([row["parent_divergence"] for row in active_rows])) if active_rows else 0.0,
        "same_child_sibling_divergence": float(np.mean([row["sibling_divergence"] for row in active_rows])) if active_rows else 0.0,
        "same_child_support_threshold": thresholds["support_min"],
        "same_child_coherence_threshold": thresholds["coherence_min"],
        "same_child_budget_threshold": thresholds["budget_min"],
        "same_child_budget_retained_threshold": thresholds["budget_retained_min"],
        "same_child_qualified_fraction_threshold": thresholds["qualified_fraction_min"],
        "same_child_qualified_steps_threshold": thresholds["qualified_steps_min"],
    }


def _child_local_readout_phase(
    state: dict[str, Any],
    cfg: CircleworldConfig,
    *,
    readout_mode: str,
) -> torch.Tensor:
    parent_phase = extract_phase_state(state, cfg)
    children = [child for child in state.get("child_worlds", []) if child.get("active", False)]
    if not children:
        return parent_phase

    child = max(children, key=_child_readout_score)
    child_phase = child.get("phase_state")
    child_support = child.get("mode_support")
    child_coherence = child.get("mode_coherence")
    if not torch.is_tensor(child_phase) or not torch.is_tensor(child_support):
        return parent_phase

    parent_mode1 = state["phase_modes"][..., 1, :]
    if torch.is_tensor(child_coherence):
        support_gate = torch.maximum(child_support, 0.45 * child_coherence).clamp(0.0, 1.0)
    else:
        support_gate = child_support.clamp(0.0, 1.0)

    if readout_mode == "child_local":
        dominance = 0.84
        child_branch = phasor_normalize((1.0 - dominance) * parent_mode1 + dominance * child_phase)
        return phasor_normalize(
            (1.0 - support_gate).unsqueeze(-1) * parent_phase
            + support_gate.unsqueeze(-1) * child_branch
        )

    if readout_mode == "child_dominant":
        dominance = 0.96
        gate = (0.20 + 0.95 * support_gate).clamp(0.0, 1.0)
        child_branch = phasor_normalize((1.0 - dominance) * parent_mode1 + dominance * child_phase)
        return phasor_normalize(
            (1.0 - gate).unsqueeze(-1) * parent_mode1
            + gate.unsqueeze(-1) * child_branch
        )

    return parent_phase


def _child_continuation_phase(
    state: dict[str, Any],
    cfg: CircleworldConfig,
    *,
    steps: int,
    readout_mode: str,
) -> tuple[torch.Tensor, dict[str, Any]]:
    if _assay_flag("continuation_parent_only"):
        probe_state = clone_circleworld_state(state)
        parent_phase = (
            phasor_normalize(probe_state["phase_modes"][..., 0, :])
            if _assay_flag("continuation_mode0_parent_only") and "phase_modes" in probe_state
            else extract_phase_state(probe_state, cfg)
        )
        probe_state["continuation_mode1_write"] = _disabled_continuation_write_metrics("continuation_parent_only")
        probe_state["mixed_phase_state"] = parent_phase
        return parent_phase, probe_state
    if _assay_flag("disable_continuation_pre_unroll"):
        cont_state = clone_circleworld_state(state)
        cont_state["mixed_phase_state"] = extract_phase_state(cont_state, cfg)
    else:
        cont_state = _child_only_pre_unroll(
            state,
            cfg,
            steps=max(1, int(steps)),
            parent_write_scale=0.0,
            mode0_protect=0.97,
        )
    parent_phase = extract_phase_state(cont_state, cfg)
    children = [child for child in cont_state.get("child_worlds", []) if child.get("active", False)]
    if not children:
        return parent_phase, cont_state

    child = max(children, key=_child_readout_score)
    if _assay_flag("continuation_clamp_child_identity"):
        child = _clamp_child_identity_for_killswitch(child)
    child_phase = child.get("phase_state")
    child_support = child.get("mode_support")
    child_coherence = child.get("mode_coherence")
    if not torch.is_tensor(child_phase) or not torch.is_tensor(child_support):
        return parent_phase, cont_state
    phase_control = str(_assay_value("continuation_child_phase_control", "") or "").strip().lower()
    if _assay_flag("scramble_continuation_child_phase") and not phase_control:
        phase_control = "roll"
    if phase_control:
        child = dict(child)
        controlled_phase = _controlled_child_phase_for_killswitch(child, children, mode=phase_control)
        if torch.is_tensor(controlled_phase):
            child_phase = controlled_phase
            child["phase_state"] = child_phase
    if _assay_flag("continuation_clamp_child_identity") or phase_control:
        child_id = int(child.get("child_id", -1))
        cont_state["child_worlds"] = [
            child if int(other.get("child_id", -2)) == child_id else other
            for other in cont_state.get("child_worlds", [])
        ]
        child_support = child.get("mode_support")
        child_coherence = child.get("mode_coherence")

    if _assay_flag("disable_continuation_mode1_write"):
        write_metrics = _disabled_continuation_write_metrics("disable_continuation_mode1_write")
    else:
        write_metrics = _child_mode1_continuation_write(cont_state, child, readout_mode=readout_mode)
    cont_state["continuation_mode1_write"] = write_metrics
    parent_phase = (
        phasor_normalize(cont_state["phase_modes"][..., 0, :])
        if _assay_flag("continuation_mode0_parent_only")
        else extract_phase_state(cont_state, cfg)
    )
    if _assay_flag("continuation_write_only_readout") or _assay_flag("disable_direct_child_continuation_mix"):
        cont_state["mixed_phase_state"] = parent_phase
        return parent_phase, cont_state
    if torch.is_tensor(child_coherence):
        support_gate = torch.maximum(child_support, 0.55 * child_coherence).clamp(0.0, 1.0)
    else:
        support_gate = child_support.clamp(0.0, 1.0)
    parent_mode1 = cont_state["phase_modes"][..., 1, :]

    if readout_mode == "child_continuation_local":
        child_branch = phasor_normalize(0.08 * parent_mode1 + 0.92 * child_phase)
        phase_state = phasor_normalize(
            (1.0 - support_gate).unsqueeze(-1) * parent_phase
            + support_gate.unsqueeze(-1) * child_branch
        )
        cont_state["mixed_phase_state"] = phase_state
        return phase_state, cont_state

    if readout_mode == "child_continuation_dominant":
        gate = (0.35 + 0.90 * support_gate).clamp(0.0, 1.0)
        child_branch = phasor_normalize(0.03 * parent_mode1 + 0.97 * child_phase)
        phase_state = phasor_normalize(
            (1.0 - gate).unsqueeze(-1) * parent_mode1
            + gate.unsqueeze(-1) * child_branch
        )
        cont_state["mixed_phase_state"] = phase_state
        return phase_state, cont_state

    return parent_phase, cont_state


def _child_mode1_replace_phase(
    state: dict[str, Any],
    cfg: CircleworldConfig,
    *,
    replace_ratio: float,
    steps: int = 2,
) -> tuple[torch.Tensor, dict[str, Any]]:
    if _assay_flag("disable_mode_replace_pre_unroll"):
        cont_state = clone_circleworld_state(state)
        cont_state["mixed_phase_state"] = extract_phase_state(cont_state, cfg)
    else:
        cont_state = _child_only_pre_unroll(
            state,
            cfg,
            steps=max(1, int(steps)),
            parent_write_scale=0.0,
            mode0_protect=0.97,
        )
    child = _select_strongest_child(cont_state)
    if child is None:
        phase_state = extract_phase_state(cont_state, cfg)
        cont_state["mixed_phase_state"] = phase_state
        cont_state["mode1_replace_metrics"] = {
            "enabled": 0.0,
            "replace_ratio": float(replace_ratio),
            "support_gate_mean": 0.0,
            "support_gate_max": 0.0,
            "static_record": 1.0 if _assay_flag("disable_mode_replace_pre_unroll") else 0.0,
        }
        return phase_state, cont_state

    children = [item for item in cont_state.get("child_worlds", []) if item.get("active", False)]
    if _assay_flag("mode_replace_clamp_child_identity"):
        child = _clamp_child_identity_for_killswitch(child)
    child_phase = child.get("phase_state")
    child_support = child.get("mode_support")
    child_coherence = child.get("mode_coherence")
    if not torch.is_tensor(child_phase) or not torch.is_tensor(child_support):
        phase_state = extract_phase_state(cont_state, cfg)
        cont_state["mixed_phase_state"] = phase_state
        cont_state["mode1_replace_metrics"] = {
            "enabled": 0.0,
            "replace_ratio": float(replace_ratio),
            "support_gate_mean": 0.0,
            "support_gate_max": 0.0,
            "static_record": 1.0 if _assay_flag("disable_mode_replace_pre_unroll") else 0.0,
        }
        return phase_state, cont_state
    phase_control = str(_assay_value("mode_replace_child_phase_control", "") or "").strip().lower()
    if phase_control:
        child = dict(child)
        controlled_phase = _controlled_child_phase_for_killswitch(child, children, mode=phase_control)
        if torch.is_tensor(controlled_phase):
            child_phase = controlled_phase
            child["phase_state"] = child_phase
    if _assay_flag("mode_replace_clamp_child_identity") or phase_control:
        child_id = int(child.get("child_id", -1))
        cont_state["child_worlds"] = [
            child if int(other.get("child_id", -2)) == child_id else other
            for other in cont_state.get("child_worlds", [])
        ]
        child_support = child.get("mode_support")
        child_coherence = child.get("mode_coherence")

    if torch.is_tensor(child_coherence):
        support_gate = torch.maximum(child_support, 0.55 * child_coherence).clamp(0.0, 1.0)
    else:
        support_gate = child_support.clamp(0.0, 1.0)
    gate_floor = float(_assay_value("mode_replace_gate_floor", 0.0) or 0.0)
    if gate_floor > 0.0:
        support_gate = torch.maximum(support_gate, torch.full_like(support_gate, max(0.0, min(1.0, gate_floor)))).clamp(0.0, 1.0)
    parent_mode1 = cont_state["phase_modes"][..., 1, :]
    ontology_used = 0.0
    ontology_charge_mean = 0.0
    ontology_charge_max = 0.0
    ontology_carrier_used = 0.0
    ontology_mode1_occ_before = _mode1_occupancy_mean(cont_state, cfg, support_gate)
    ontology_mode1_occ_after = ontology_mode1_occ_before
    if _assay_flag("disable_mode_replace_child_phase_carrier"):
        child_phase = parent_mode1
    if _assay_flag("parent_ontology_use_in_mode_replace"):
        ontology_charge = cont_state.get("ontology_charge")
        ontology_delta = cont_state.get("ontology_phase_delta")
        ontology_carrier_delta = cont_state.get("ontology_carrier_delta")
        ontology_carrier_phase = cont_state.get("ontology_carrier_phase")
        if torch.is_tensor(ontology_charge) and ontology_charge.shape == support_gate.shape:
            if not _assay_flag("disable_parent_ontology_support_gate"):
                gate_boost = float(_assay_value("parent_ontology_support_boost", 1.0) or 1.0)
                support_gate = torch.maximum(support_gate, (gate_boost * ontology_charge).clamp(0.0, 1.0)).clamp(0.0, 1.0)
                ontology_floor = float(_assay_value("parent_ontology_gate_floor", 0.0) or 0.0)
                if ontology_floor > 0.0:
                    support_gate = torch.maximum(
                        support_gate,
                        torch.full_like(support_gate, max(0.0, min(1.0, ontology_floor))),
                    ).clamp(0.0, 1.0)
            ontology_charge_mean = float(ontology_charge.mean().item())
            ontology_charge_max = float(ontology_charge.max().item())
            ontology_used = 1.0
            if "mode_logits" in cont_state and not _assay_flag("disable_parent_ontology_mode_logits"):
                boost = float(_assay_value("parent_ontology_mode1_logit_boost", 0.0) or 0.0)
                suppress = float(_assay_value("parent_ontology_mode0_logit_suppress", 0.0) or 0.0)
                if boost != 0.0:
                    cont_state["mode_logits"][..., 1] = cont_state["mode_logits"][..., 1] + boost * ontology_charge
                if suppress != 0.0:
                    cont_state["mode_logits"][..., 0] = cont_state["mode_logits"][..., 0] - suppress * ontology_charge
            if "mode_support" in cont_state and not _assay_flag("disable_parent_ontology_mode_support"):
                cont_state["mode_support"][..., 1] = torch.maximum(cont_state["mode_support"][..., 1], ontology_charge).clamp(0.0, 1.0)
            source = str(_assay_value("parent_ontology_mode_replace_source", "ontology_delta") or "ontology_delta").strip().lower()
            if (
                torch.is_tensor(ontology_carrier_phase)
                and ontology_carrier_phase.shape == parent_mode1.shape
                and source in {"trained_phasor_carrier_map", "learned_phasor_carrier_map", "trained_phasor_carrier", "blend_child_trained_phasor_carrier"}
            ):
                if source == "blend_child_trained_phasor_carrier":
                    child_phase = phasor_normalize(0.50 * child_phase + 0.50 * ontology_carrier_phase)
                else:
                    child_phase = phasor_normalize(ontology_carrier_phase)
                ontology_carrier_used = 1.0
            elif (
                torch.is_tensor(ontology_carrier_delta)
                and ontology_carrier_delta.shape == support_gate.shape
                and source in {"trained_carrier_map", "learned_carrier_map", "trained_carrier", "blend_child_trained_carrier"}
            ):
                carrier_phase = phasor_apply_delta(parent_mode1, ontology_carrier_delta.clamp(-math.pi, math.pi))
                if source == "blend_child_trained_carrier":
                    child_phase = phasor_normalize(0.50 * child_phase + 0.50 * carrier_phase)
                else:
                    child_phase = carrier_phase
                ontology_carrier_used = 1.0
            elif torch.is_tensor(ontology_delta) and ontology_delta.shape == support_gate.shape and source in {"ontology_delta", "blend_child_ontology"}:
                delta_gain = float(_assay_value("parent_ontology_mode_replace_delta_gain", 1.0) or 1.0)
                ontology_phase = phasor_apply_delta(parent_mode1, (delta_gain * ontology_delta).clamp(-math.pi, math.pi))
                if source == "blend_child_ontology":
                    child_phase = phasor_normalize(0.50 * child_phase + 0.50 * ontology_phase)
                else:
                    child_phase = ontology_phase
            ontology_mode1_occ_after = _mode1_occupancy_mean(cont_state, cfg, support_gate)
    ratio_scale = float(_assay_value("mode_replace_ratio_scale", 1.0) or 1.0)
    ratio = float(max(0.0, min(1.0, float(replace_ratio) * ratio_scale)))
    replaced_mode1 = phasor_normalize(
        (1.0 - (ratio * support_gate)).unsqueeze(-1) * parent_mode1
        + (ratio * support_gate).unsqueeze(-1) * child_phase
    )
    cont_state["phase_modes"][..., 1, :] = replaced_mode1
    phase_state = extract_phase_state(cont_state, cfg)
    cont_state["mixed_phase_state"] = phase_state
    cont_state["mode1_replace_metrics"] = {
        "enabled": 1.0,
        "replace_ratio": ratio,
        "support_gate_mean": float(support_gate.mean().item()),
        "support_gate_max": float(support_gate.max().item()),
        "gate_floor": float(max(0.0, min(1.0, gate_floor))),
        "ratio_scale": float(ratio_scale),
        "static_record": 1.0 if _assay_flag("disable_mode_replace_pre_unroll") else 0.0,
        "parent_ontology_used": float(ontology_used),
        "parent_ontology_charge_mean": float(ontology_charge_mean),
        "parent_ontology_charge_max": float(ontology_charge_max),
        "parent_ontology_carrier_used": float(ontology_carrier_used),
        "parent_ontology_mode1_occupancy_before": float(ontology_mode1_occ_before),
        "parent_ontology_mode1_occupancy_after": float(ontology_mode1_occ_after),
    }
    return phase_state, cont_state


def _apply_readout_override(
    run: dict[str, Any],
    cfg: CircleworldConfig,
    readout_mode: str | None,
) -> dict[str, Any]:
    if not readout_mode:
        return run
    state_bundle = run.get("state_bundle")
    if not isinstance(state_bundle, dict) or "phase_modes" not in state_bundle:
        return run

    probe_state = state_bundle
    if readout_mode in {"child_local", "child_dominant"}:
        phase_state = _child_local_readout_phase(state_bundle, cfg, readout_mode=readout_mode)
    elif readout_mode in {"child_continuation_local", "child_continuation_dominant"}:
        phase_state, probe_state = _child_continuation_phase(
            state_bundle,
            cfg,
            steps=2 if readout_mode == "child_continuation_dominant" else 1,
            readout_mode=readout_mode,
        )
    elif readout_mode == "child_mode1_replace_50":
        phase_state, probe_state = _child_mode1_replace_phase(state_bundle, cfg, replace_ratio=0.50, steps=2)
    elif readout_mode == "child_mode1_replace_85":
        phase_state, probe_state = _child_mode1_replace_phase(state_bundle, cfg, replace_ratio=0.85, steps=2)
    else:
        return run
    run["phase_state"] = phase_state
    state_bundle["mixed_phase_state"] = phase_state
    if isinstance(probe_state, dict) and isinstance(probe_state.get("continuation_mode1_write"), dict):
        run["continuation_mode1_write"] = dict(probe_state["continuation_mode1_write"])
    if isinstance(probe_state, dict) and isinstance(probe_state.get("mode1_replace_metrics"), dict):
        run["mode1_replace_metrics"] = dict(probe_state["mode1_replace_metrics"])
    run["identity_probe_state"] = clone_circleworld_state(probe_state)
    run["final_arc"] = hardy_littlewood_arc_field(phase_state, cfg.qset, cfg.q_weights)
    run["final_promo"] = promotability_field(run["final_arc"], persistence_momentum=cfg.persistence_momentum)
    run["readout_override_mode"] = readout_mode
    return run


def _render_phase_state(mag: torch.Tensor, phase_state: torch.Tensor, stft_cfg: dict[str, Any], out_path: Path, sr: int) -> torch.Tensor:
    phase = _phasor_to_phase(phase_state)
    wav = inverse_stft(mag, phase, stft_cfg).squeeze(0)
    _save_wav(wav, out_path, sr)
    return wav.detach().cpu()


def _spectral_envelope(wav: np.ndarray, sr: int, n_fft: int = 2048, hop: int = 512) -> np.ndarray:
    if len(wav) < n_fft:
        wav = np.pad(wav, (0, n_fft - len(wav)))
    win = np.hanning(n_fft).astype(np.float32)
    vecs = []
    for start in range(0, len(wav) - n_fft + 1, hop):
        seg = wav[start : start + n_fft]
        spec = np.fft.rfft(seg * win)
        vecs.append(np.log1p(np.abs(spec)).astype(np.float32))
    if not vecs:
        vecs = [np.log1p(np.abs(np.fft.rfft(wav[:n_fft] * win))).astype(np.float32)]
    return np.mean(np.stack(vecs, axis=0), axis=0)


def _energy_contour(wav: np.ndarray, sr: int, window_seconds: float = 0.5, hop_seconds: float = 0.25) -> np.ndarray:
    frame = max(128, int(round(sr * window_seconds)))
    hop = max(64, int(round(sr * hop_seconds)))
    return _frame_rms(wav, frame=frame, hop=hop)


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) == 0 or len(b) == 0:
        return 0.0
    n = min(len(a), len(b))
    a = a[:n].astype(np.float64)
    b = b[:n].astype(np.float64)
    a = a - a.mean()
    b = b - b.mean()
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom <= 1e-12:
        return 0.0
    return float(np.dot(a, b) / denom)


def _mean_chunk_similarity(wav_a: np.ndarray, wav_b: np.ndarray, sr: int, chunk_seconds: float = 2.0) -> float:
    chunks_a = _chunk_vectors(wav_a, sr, chunk_seconds)
    chunks_b = _chunk_vectors(wav_b, sr, chunk_seconds)
    n = min(len(chunks_a), len(chunks_b))
    if n == 0:
        return 0.0
    return float(np.mean([_cosine(chunks_a[i], chunks_b[i]) for i in range(n)]))


def _short_texture_distance(wav_a: np.ndarray, wav_b: np.ndarray, sr: int, chunk_ms: float = 120.0) -> float:
    chunk_len = max(256, int(round(sr * chunk_ms / 1000.0)))
    n = min(len(wav_a), len(wav_b))
    wav_a = wav_a[:n]
    wav_b = wav_b[:n]
    vals = []
    for start in range(0, max(1, n - chunk_len + 1), chunk_len // 2):
        a = wav_a[start : start + chunk_len]
        b = wav_b[start : start + chunk_len]
        if len(a) < chunk_len or len(b) < chunk_len:
            break
        spec_a = np.log1p(np.abs(np.fft.rfft(a * np.hanning(chunk_len))))
        spec_b = np.log1p(np.abs(np.fft.rfft(b * np.hanning(chunk_len))))
        vals.append(float(np.mean(np.abs(spec_a - spec_b))))
    return float(np.mean(vals)) if vals else 0.0


def _q_profile(block: dict[str, torch.Tensor]) -> np.ndarray:
    per_q_abs = block["per_q"].abs().mean(dim=(0, 2))
    q_mass = per_q_abs / per_q_abs.sum().clamp_min(1e-8)
    return q_mass.detach().cpu().numpy().astype(np.float32)


def _packet_succession_vector(run: dict[str, Any]) -> np.ndarray:
    counts = np.array([len(level) for level in run["packets"]], dtype=np.float32)
    if counts.size == 0:
        return np.zeros(1, dtype=np.float32)
    score_means = []
    for level in run["packets"]:
        if level:
            score_means.append(float(np.mean([float(p["score"].item()) for p in level])))
        else:
            score_means.append(0.0)
    return np.concatenate([counts, np.array(score_means, dtype=np.float32)])


def _scalar_metric(value: Any) -> float:
    if torch.is_tensor(value):
        return float(value.item())
    return float(value)


def _child_signal_metrics(block: dict[str, Any] | None, live_child_threshold: float) -> dict[str, float]:
    if not isinstance(block, dict):
        return {
            "live_child_fraction": 0.0,
            "child_world_count": 0.0,
            "child_writeback_mass": 0.0,
            "child_parent_divergence": 0.0,
            "child_score": 0.0,
            "live_child_gate_passed": 0.0,
        }
    live_child = _scalar_metric(block.get("live_child_fraction", 0.0))
    child_count = _scalar_metric(block.get("child_world_count", 0.0))
    child_writeback = _scalar_metric(block.get("child_writeback_mass", 0.0))
    child_parent_div = _scalar_metric(block.get("child_parent_divergence", 0.0))
    child_score = live_child + 0.25 * child_count + 0.5 * child_writeback + 0.5 * child_parent_div
    return {
        "live_child_fraction": float(live_child),
        "child_world_count": float(child_count),
        "child_writeback_mass": float(child_writeback),
        "child_parent_divergence": float(child_parent_div),
        "child_score": float(child_score),
        "live_child_gate_passed": 1.0 if (live_child >= live_child_threshold or child_count > 0.0) else 0.0,
    }


def _branch_family_tag(branch_name: str, *, readout_mode: str | None) -> str:
    mode = str(readout_mode or "")
    if mode in {"child_local", "child_dominant"}:
        return "readout"
    if mode.startswith("child_continuation_"):
        return "continuation"
    if mode.startswith("child_mode1_replace_") or mode.startswith("static_child_mode1_replace_"):
        return "mode_replace"
    if str(branch_name).startswith("promotion_"):
        return "promotion"
    if branch_name:
        return "perturb"
    return "other"


def _branch_family_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts = {family: 0 for family in BRANCH_FAMILY_ORDER}
    for row in rows:
        family = str(row.get("family", "other"))
        counts.setdefault(family, 0)
        counts[family] += 1
    return counts


def _family_specific_label(family: str, label: str) -> str:
    if family in NESTED_ASSAY_FAMILIES and label in NESTED_ASSAY_LABELS:
        return f"{family}_{label}"
    return label


def _empty_assay_family_label_counts() -> dict[str, dict[str, int]]:
    return {
        family: {f"{family}_{label}": 0 for label in NESTED_ASSAY_LABELS}
        for family in NESTED_ASSAY_FAMILIES
    }


def _branch_identity_float(row: dict[str, Any], key: str, default: float = 0.0) -> float:
    identity = row.get("branch_identity", {})
    if isinstance(identity, dict):
        return float(identity.get(key, default))
    return float(default)


def _row_float(row: dict[str, Any], key: str, default: float = 0.0) -> float:
    return float(row.get(key, default))


def _assay_family_rollup(rows: list[dict[str, Any]]) -> dict[str, Any]:
    label_counts = _empty_assay_family_label_counts()
    nested_counts = {family: 0 for family in NESTED_ASSAY_FAMILIES}
    branch_counts = {family: 0 for family in NESTED_ASSAY_FAMILIES}
    response: dict[str, Any] = {
        "assay_family_label_counts": label_counts,
        "assay_family_nested_sibling_counts": nested_counts,
        "assay_family_branch_counts": branch_counts,
    }
    for family in NESTED_ASSAY_FAMILIES:
        family_rows = [row for row in rows if str(row.get("family", "other")) == family]
        branch_counts[family] = int(len(family_rows))
        for row in family_rows:
            family_label = str(row.get("family_label") or _family_specific_label(family, str(row.get("label", ""))))
            label_counts[family].setdefault(family_label, 0)
            label_counts[family][family_label] += 1
            if row.get("label") == "nested_sibling":
                nested_counts[family] += 1
        fraction = float(nested_counts[family] / max(1, branch_counts[family]))
        mean_score = float(np.mean([_branch_response_score(row) for row in family_rows])) if family_rows else 0.0
        mean_identity = float(
            np.mean([_branch_identity_float(row, "same_child_qualified_carry_fraction") for row in family_rows])
        ) if family_rows else 0.0
        mean_budget = float(np.mean([_branch_identity_float(row, "same_child_budget_retained") for row in family_rows])) if family_rows else 0.0
        mean_support = float(np.mean([_branch_identity_float(row, "same_child_support_mean") for row in family_rows])) if family_rows else 0.0
        mean_coherence = float(np.mean([_branch_identity_float(row, "same_child_coherence_mean") for row in family_rows])) if family_rows else 0.0
        mean_response = float(np.mean([_row_float(row, "readout_sibling_response") for row in family_rows])) if family_rows else 0.0
        mean_coarse = float(np.mean([_row_float(row, "coarse_env_corr") for row in family_rows])) if family_rows else 0.0
        mean_q_corr = float(np.mean([_row_float(row, "fine_q_profile_corr") for row in family_rows])) if family_rows else 0.0
        readiness = [_nested_sibling_readiness(row) for row in family_rows]
        mean_readiness = float(np.mean(readiness)) if readiness else 0.0
        max_readiness = float(np.max(readiness)) if readiness else 0.0
        write_rows = [dict(row.get("continuation_mode1_write", {})) for row in family_rows if isinstance(row.get("continuation_mode1_write", {}), dict)]
        mean_write_enabled = float(np.mean([float(row.get("enabled", 0.0)) for row in write_rows])) if write_rows else 0.0
        mean_write_delta = float(np.mean([float(row.get("applied_delta_abs_mean", 0.0)) for row in write_rows])) if write_rows else 0.0
        replace_rows = [dict(row.get("mode1_replace_metrics", {})) for row in family_rows if isinstance(row.get("mode1_replace_metrics", {}), dict)]
        mean_replace_enabled = float(np.mean([float(row.get("enabled", 0.0)) for row in replace_rows])) if replace_rows else 0.0
        mean_replace_ratio = float(np.mean([float(row.get("replace_ratio", 0.0)) for row in replace_rows])) if replace_rows else 0.0
        mean_replace_gate = float(np.mean([float(row.get("support_gate_mean", 0.0)) for row in replace_rows])) if replace_rows else 0.0
        mean_replace_static = float(np.mean([float(row.get("static_record", 0.0)) for row in replace_rows])) if replace_rows else 0.0
        mean_replace_gate_floor = float(np.mean([float(row.get("gate_floor", 0.0)) for row in replace_rows])) if replace_rows else 0.0
        mean_replace_ratio_scale = float(np.mean([float(row.get("ratio_scale", 1.0)) for row in replace_rows])) if replace_rows else 0.0
        mean_replace_ontology_used = float(np.mean([float(row.get("parent_ontology_used", 0.0)) for row in replace_rows])) if replace_rows else 0.0
        mean_replace_ontology_charge = float(np.mean([float(row.get("parent_ontology_charge_mean", 0.0)) for row in replace_rows])) if replace_rows else 0.0
        mean_replace_ontology_carrier_used = float(np.mean([float(row.get("parent_ontology_carrier_used", 0.0)) for row in replace_rows])) if replace_rows else 0.0
        mean_replace_ontology_occ_after = float(np.mean([float(row.get("parent_ontology_mode1_occupancy_after", 0.0)) for row in replace_rows])) if replace_rows else 0.0
        partition_rows = [
            dict(row.get("assay_child_partition", {}))
            for row in family_rows
            if isinstance(row.get("assay_child_partition", {}), dict)
        ]
        mean_volume_active = float(np.mean([float(row.get("after_active_count", 0.0)) for row in partition_rows])) if partition_rows else 0.0
        mean_volume_effective = float(np.mean([float(row.get("after_effective_count", 0.0)) for row in partition_rows])) if partition_rows else 0.0
        mean_volume_top_share = float(np.mean([float(row.get("after_top_score_share", 0.0)) for row in partition_rows])) if partition_rows else 0.0
        mean_fragment_count = float(np.mean([float(row.get("fragment_count", 0.0)) for row in partition_rows])) if partition_rows else 0.0
        mean_grandchild_count = float(np.mean([float(row.get("grandchild_count", 0.0)) for row in partition_rows])) if partition_rows else 0.0
        max_generation = float(np.max([float(row.get("after_max_generation", 0.0)) for row in partition_rows])) if partition_rows else 0.0
        predictive_rows = [
            dict(row.get("child_predictive_assimilation", {}))
            for row in family_rows
            if isinstance(row.get("child_predictive_assimilation", {}), dict)
        ]
        mean_predictive_reduction = float(np.mean([float(row.get("residual_reduction", 0.0)) for row in predictive_rows])) if predictive_rows else 0.0
        mean_predictive_positive = float(np.mean([float(row.get("positive_residual_reduction", 0.0)) for row in predictive_rows])) if predictive_rows else 0.0
        mean_predictive_boundary = float(np.mean([float(row.get("boundary_match", 0.0)) for row in predictive_rows])) if predictive_rows else 0.0
        mean_predictive_binding = float(np.mean([float(row.get("ontology_binding_mass", 0.0)) for row in predictive_rows])) if predictive_rows else 0.0
        mean_predictive_applied = float(np.mean([float(row.get("applied", 0.0)) for row in predictive_rows])) if predictive_rows else 0.0
        response[f"assay_{family}_branch_count"] = float(branch_counts[family])
        response[f"assay_{family}_nested_sibling_count"] = float(nested_counts[family])
        response[f"assay_{family}_nested_sibling_fraction"] = fraction
        response[f"assay_{family}_mean_child_response_score"] = mean_score
        response[f"assay_{family}_mean_branch_identity_qualified_carry"] = mean_identity
        response[f"assay_{family}_mean_branch_identity_budget_retained"] = mean_budget
        response[f"assay_{family}_mean_branch_identity_support"] = mean_support
        response[f"assay_{family}_mean_branch_identity_coherence"] = mean_coherence
        response[f"assay_{family}_mean_readout_sibling_response"] = mean_response
        response[f"assay_{family}_mean_coarse_env_corr"] = mean_coarse
        response[f"assay_{family}_mean_fine_q_profile_corr"] = mean_q_corr
        response[f"assay_{family}_mean_nested_sibling_readiness"] = mean_readiness
        response[f"assay_{family}_max_nested_sibling_readiness"] = max_readiness
        response[f"assay_{family}_mean_continuation_write_enabled"] = mean_write_enabled
        response[f"assay_{family}_mean_continuation_write_delta"] = mean_write_delta
        response[f"assay_{family}_mean_mode1_replace_enabled"] = mean_replace_enabled
        response[f"assay_{family}_mean_mode1_replace_ratio"] = mean_replace_ratio
        response[f"assay_{family}_mean_mode1_replace_gate"] = mean_replace_gate
        response[f"assay_{family}_mean_mode1_replace_static_record"] = mean_replace_static
        response[f"assay_{family}_mean_mode1_replace_gate_floor"] = mean_replace_gate_floor
        response[f"assay_{family}_mean_mode1_replace_ratio_scale"] = mean_replace_ratio_scale
        response[f"assay_{family}_mean_mode1_replace_parent_ontology_used"] = mean_replace_ontology_used
        response[f"assay_{family}_mean_mode1_replace_parent_ontology_charge"] = mean_replace_ontology_charge
        response[f"assay_{family}_mean_mode1_replace_parent_ontology_carrier_used"] = mean_replace_ontology_carrier_used
        response[f"assay_{family}_mean_mode1_replace_parent_ontology_mode1_occupancy_after"] = mean_replace_ontology_occ_after
        response[f"assay_{family}_mean_child_volume_active_count"] = mean_volume_active
        response[f"assay_{family}_mean_child_volume_effective_count"] = mean_volume_effective
        response[f"assay_{family}_mean_child_volume_top_score_share"] = mean_volume_top_share
        response[f"assay_{family}_mean_child_fragment_count"] = mean_fragment_count
        response[f"assay_{family}_mean_child_grandchild_count"] = mean_grandchild_count
        response[f"assay_{family}_max_child_generation"] = max_generation
        response[f"assay_{family}_mean_child_predictive_residual_reduction"] = mean_predictive_reduction
        response[f"assay_{family}_mean_child_predictive_positive_residual_reduction"] = mean_predictive_positive
        response[f"assay_{family}_mean_child_predictive_boundary_match"] = mean_predictive_boundary
        response[f"assay_{family}_mean_child_predictive_ontology_binding_mass"] = mean_predictive_binding
        response[f"assay_{family}_mean_child_predictive_applied_fraction"] = mean_predictive_applied
    return response


def _child_record_survival_score(best_identity: dict[str, Any], thresholds: dict[str, float]) -> float:
    support_denom = max(1.0e-6, float(thresholds.get("support_min", 0.0)))
    coherence_denom = max(1.0e-6, float(thresholds.get("coherence_min", 0.0)))
    budget_denom = max(1.0e-6, float(thresholds.get("budget_retained_min", 0.0)))
    criteria = [
        float(best_identity.get("same_child_qualified_carry_fraction", 0.0)),
        float(best_identity.get("same_child_budget_retained", 0.0)) / budget_denom,
        float(best_identity.get("same_child_support_mean", 0.0)) / support_denom,
        float(best_identity.get("same_child_coherence_mean", 0.0)) / coherence_denom,
    ]
    return float(max(0.0, min(1.0, min(criteria))))


def _branch_label(
    coarse_env_corr: float,
    meso_chunk_similarity: float,
    fine_texture_distance: float,
    q_profile_corr: float,
    residue_delta: float,
    same_child_carry_fraction: float = 0.0,
    same_child_carry_steps: float = 0.0,
    same_child_qualified_carry_fraction: float = 0.0,
    same_child_qualified_carry_steps: float = 0.0,
    same_child_budget_retained: float = 0.0,
    readout_sibling_response: float = 0.0,
    readout_only_baseline: float = 0.0,
) -> str:
    thresholds = _identity_carry_thresholds()
    if coarse_env_corr < 0.75 or q_profile_corr < 0.60:
        return "overwrite_or_world_jump"
    if (
        same_child_carry_fraction > 0.0
        and same_child_carry_steps >= 1.0
        and same_child_qualified_carry_fraction > thresholds["qualified_fraction_min"]
        and same_child_qualified_carry_steps >= thresholds["qualified_steps_min"]
        and same_child_budget_retained >= thresholds["budget_retained_min"]
        and coarse_env_corr >= 0.99
        and q_profile_corr >= 0.995
        and readout_sibling_response > readout_only_baseline
    ):
        return "nested_sibling"
    if coarse_env_corr > 0.92 and meso_chunk_similarity > 0.92 and fine_texture_distance < 0.08 and residue_delta < 0.01:
        return "over_rigid_or_frozen"
    return "ambiguous_middle"


def _nested_sibling_readiness(row: dict[str, Any]) -> float:
    thresholds = _identity_carry_thresholds()
    carry = dict(row.get("branch_identity", {}))
    response_target = max(1.0e-6, float(row.get("readout_only_baseline", 0.0)))
    criteria = [
        float(carry.get("same_child_carry_fraction", 0.0)) / 0.5,
        float(carry.get("same_child_carry_steps", 0.0)) / 1.0,
        float(carry.get("same_child_qualified_carry_fraction", 0.0)) / max(1.0e-6, thresholds["qualified_fraction_min"]),
        float(carry.get("same_child_qualified_carry_steps", 0.0)) / max(1.0e-6, thresholds["qualified_steps_min"]),
        float(carry.get("same_child_budget_retained", 0.0)) / max(1.0e-6, thresholds["budget_retained_min"]),
        float(row.get("coarse_env_corr", 0.0)) / 0.99,
        float(row.get("fine_q_profile_corr", 0.0)) / 0.995,
        float(row.get("readout_sibling_response", 0.0)) / response_target,
    ]
    return float(max(0.0, min(1.0, min(criteria))))


def _child_history_summary(run: dict[str, Any]) -> dict[str, float]:
    events = list(run.get("child_event_history", []))
    if not events:
        return {
            "spawn_count": 0.0,
            "kill_count": 0.0,
            "writeback_count": 0.0,
            "max_age": 0.0,
        }
    return {
        "spawn_count": float(sum(1 for ev in events if ev.get("event") == "spawn")),
        "kill_count": float(sum(1 for ev in events if ev.get("event") == "collapse")),
        "writeback_count": float(sum(1 for ev in events if ev.get("event") == "writeback")),
        "max_age": float(max(float(ev.get("survival_age", 0.0)) for ev in events)),
    }


def _branch_response_score(metrics: dict[str, Any]) -> float:
    probe = dict(metrics.get("child_branch_probe", {}))
    carry = dict(metrics.get("branch_identity", {}))
    thresholds = _identity_carry_thresholds()
    coarse = float(metrics.get("coarse_env_corr", 0.0))
    meso = float(metrics.get("meso_chunk_similarity", 0.0))
    texture = float(metrics.get("fine_texture_distance", 0.0))
    world_jump = 1.0 if metrics.get("label") == "overwrite_or_world_jump" else 0.0
    rigid = 1.0 if metrics.get("label") == "over_rigid_or_frozen" else 0.0
    nested = 1.0 if metrics.get("label") == "nested_sibling" else 0.0
    readout_sibling = float(metrics.get("readout_sibling_response", 0.0))
    score = 0.0
    score += 3.0 * nested
    score += 8.0 * float(probe.get("real_branch_fraction", 0.0))
    score += 180.0 * float(probe.get("meso_branch_effect", 0.0))
    score += 10.0 * float(probe.get("child_writeback_mass", 0.0))
    score += 12.0 * float(probe.get("child_parent_divergence", 0.0))
    score += 14.0 * float(probe.get("child_sibling_divergence", 0.0))
    score += 1.5 * float(carry.get("same_child_carry_fraction", 0.0))
    score += 0.5 * float(carry.get("same_child_carry_steps", 0.0))
    score += 4.0 * float(carry.get("same_child_qualified_carry_fraction", 0.0))
    score += 1.5 * float(carry.get("same_child_qualified_carry_steps", 0.0))
    score += 2.5 * max(
        0.0,
        float(carry.get("same_child_budget_retained", 0.0)) - thresholds["budget_retained_min"],
    )
    score += 20.0 * readout_sibling
    score += 1.2 * max(0.0, coarse - 0.90)
    score += 0.8 * max(0.0, 0.95 - meso)
    score -= 10.0 * max(0.0, 0.82 - coarse)
    score -= 2.0 * rigid
    score -= 3.0 * world_jump
    score -= 2.5 * max(0.0, 0.001 - texture)
    return float(score)


def _empty_case_nested_response(
    thresholds: dict[str, float],
    *,
    live_child_start_found: bool,
    selected_fork_has_live_child: bool,
    selected_live_child_gate_passed: bool,
    selected_child_signal_found: bool,
) -> dict[str, Any]:
    selected_phase_only_eligibility = bool(selected_child_signal_found and not selected_fork_has_live_child)
    selected_live_child_eligibility = bool(selected_live_child_gate_passed or selected_child_signal_found)
    selected_nested_score_eligibility = bool(selected_fork_has_live_child or selected_child_signal_found)
    return {
        "mean_child_response_score": 0.0,
        "max_child_response_score": 0.0,
        "mean_child_active_fraction": 0.0,
        "mean_child_meso_response": 0.0,
        "mean_coarse_preservation": 0.0,
        "mean_world_jump_penalty": 0.0,
        "nested_sibling_fraction": 0.0,
        "over_rigid_fraction": 0.0,
        "mean_child_survival_signal": 0.0,
        "mean_readout_sibling_response": 0.0,
        "direct_readout_dependency": 0.0,
        "final_direct_mix_shortcut_score": 0.0,
        "child_record_survival_score": 0.0,
        "parent_mode_conversion_sibling_fraction": 0.0,
        "mode_replace_conversion_score": 0.0,
        "mean_child_volume_active_count": 0.0,
        "mean_child_volume_effective_count": 0.0,
        "mean_child_volume_top_score_share": 0.0,
        "mean_assay_child_fragment_count": 0.0,
        "max_assay_child_fragment_count": 0.0,
        "mean_assay_grandchild_count": 0.0,
        "max_assay_grandchild_count": 0.0,
        "max_assay_child_generation": 0.0,
        "mean_child_predictive_residual_reduction": 0.0,
        "max_child_predictive_residual_reduction": 0.0,
        "mean_child_predictive_positive_residual_reduction": 0.0,
        "mean_child_predictive_boundary_match": 0.0,
        "mean_child_predictive_explained_parent_defect": 0.0,
        "mean_child_predictive_ontology_binding_mass": 0.0,
        "mean_child_predictive_applied_fraction": 0.0,
        "mean_child_predictive_target_residual_abs": 0.0,
        "mean_child_predictive_raw_delta_abs": 0.0,
        "mean_child_predictive_candidate_delta_abs": 0.0,
        "mean_child_predictive_oracle_residual_reduction": 0.0,
        "max_child_predictive_oracle_residual_reduction": 0.0,
        "mean_child_predictive_learned_bridge_score": 0.0,
        "mean_child_predictive_learned_bridge_acceptance": 0.0,
        "mean_child_predictive_learned_bridge_authority_acceptance": 0.0,
        "mean_child_predictive_learned_bridge_boundary_score": 0.0,
        "mean_child_predictive_learned_bridge_volume_score": 0.0,
        "mean_child_predictive_learned_bridge_support_score": 0.0,
        "mean_child_predictive_learned_bridge_coherence_score": 0.0,
        "mean_child_predictive_learned_bridge_delta_scale": 0.0,
        "mean_child_predictive_learned_bridge_prewrite_gate": 0.0,
        "mean_child_predictive_learned_bridge_effective_delta_scale": 0.0,
        "mean_parent_ontology_charge": 0.0,
        "max_parent_ontology_charge": 0.0,
        "mean_parent_ontology_mode1_occupancy_after": 0.0,
        "mean_branch_identity_carry": 0.0,
        "mean_branch_identity_qualified_carry": 0.0,
        "max_branch_identity_carry": 0.0,
        "max_branch_identity_qualified_carry": 0.0,
        "max_branch_identity_qualified_steps": 0.0,
        "max_branch_identity_budget_retained": 0.0,
        "max_branch_identity_parent_div": 0.0,
        "max_branch_identity_sibling_div": 0.0,
        "best_identity_branch": "",
        "best_identity_support_mean": 0.0,
        "best_identity_coherence_mean": 0.0,
        "best_identity_min_support": 0.0,
        "best_identity_min_coherence": 0.0,
        "mean_branch_identity_budget_retained": 0.0,
        "mean_branch_identity_parent_div": 0.0,
        "mean_branch_identity_sibling_div": 0.0,
        "readout_branch_count": 0.0,
        "mean_continuity_recurrence_score": 0.0,
        "max_continuity_recurrence_score": 0.0,
        "mean_continuity_adjacent_chunk_similarity": 0.0,
        "mean_continuity_nonlocal_chunk_repeat": 0.0,
        "mean_continuity_first_chunk_reentry": 0.0,
        "mean_continuity_loop_autocorr_peak": 0.0,
        "mean_continuity_delta_magnitude": 0.0,
        "max_continuity_delta_magnitude": 0.0,
        "live_child_start_gate_passed": 1.0 if live_child_start_found else 0.0,
        "selected_live_child_gate_passed": 1.0 if selected_live_child_gate_passed else 0.0,
        "selected_child_signal_found": 1.0 if selected_child_signal_found else 0.0,
        "selected_fork_has_live_child": 1.0 if selected_fork_has_live_child else 0.0,
        "selected_fork_has_child_signal": 1.0 if selected_child_signal_found else 0.0,
        "selected_phase_only_eligibility": 1.0 if selected_phase_only_eligibility else 0.0,
        "selected_fork_phase_only_eligible": 1.0 if selected_phase_only_eligibility else 0.0,
        "selected_live_child_eligibility": 1.0 if selected_live_child_eligibility else 0.0,
        "selected_fork_eligible_for_nested_score": 1.0 if selected_nested_score_eligibility else 0.0,
        "branch_family_counts": {family: 0 for family in BRANCH_FAMILY_ORDER},
        "family_breakdown": {},
        "assay_family_label_counts": _empty_assay_family_label_counts(),
        "assay_family_nested_sibling_counts": {family: 0 for family in NESTED_ASSAY_FAMILIES},
        "assay_family_branch_counts": {family: 0 for family in NESTED_ASSAY_FAMILIES},
        "assay_readout_branch_count": 0.0,
        "assay_readout_nested_sibling_count": 0.0,
        "assay_readout_nested_sibling_fraction": 0.0,
        "assay_readout_mean_child_response_score": 0.0,
        "assay_readout_mean_branch_identity_qualified_carry": 0.0,
        "assay_continuation_branch_count": 0.0,
        "assay_continuation_nested_sibling_count": 0.0,
        "assay_continuation_nested_sibling_fraction": 0.0,
        "assay_continuation_mean_child_response_score": 0.0,
        "assay_continuation_mean_branch_identity_qualified_carry": 0.0,
        "assay_mode_replace_branch_count": 0.0,
        "assay_mode_replace_nested_sibling_count": 0.0,
        "assay_mode_replace_nested_sibling_fraction": 0.0,
        "assay_mode_replace_mean_child_response_score": 0.0,
        "assay_mode_replace_mean_branch_identity_qualified_carry": 0.0,
        "branch_identity_support_threshold": thresholds["support_min"],
        "branch_identity_coherence_threshold": thresholds["coherence_min"],
        "branch_identity_budget_threshold": thresholds["budget_min"],
        "branch_identity_budget_retained_threshold": thresholds["budget_retained_min"],
        "branch_identity_qualified_fraction_threshold": thresholds["qualified_fraction_min"],
        "branch_identity_qualified_steps_threshold": thresholds["qualified_steps_min"],
        "eligible_for_nested_score": 1.0 if selected_nested_score_eligibility else 0.0,
    }


def _case_nested_response(
    branch_metrics: dict[str, Any],
    *,
    live_child_start_found: bool,
    selected_fork_has_live_child: bool,
    selected_live_child_gate_passed: bool,
    selected_child_signal_found: bool,
) -> dict[str, Any]:
    thresholds = _identity_carry_thresholds()
    response = _empty_case_nested_response(
        thresholds,
        live_child_start_found=live_child_start_found,
        selected_fork_has_live_child=selected_fork_has_live_child,
        selected_live_child_gate_passed=selected_live_child_gate_passed,
        selected_child_signal_found=selected_child_signal_found,
    )
    if not branch_metrics:
        return response

    branch_items = [(str(name), metrics) for name, metrics in branch_metrics.items()]
    rows = [metrics for _, metrics in branch_items]
    branch_family_counts = _branch_family_counts(rows)
    response_scores = [_branch_response_score(row) for row in rows]
    readout_rows = [row for row in rows if str(row.get("family", "other")) == "readout"]
    survival_signal = [
        float(row.get("child_branch_probe", {}).get("real_branch_fraction", 0.0))
        + float(row.get("child_branch_probe", {}).get("child_writeback_mass", 0.0))
        + float(row.get("child_branch_probe", {}).get("child_parent_divergence", 0.0))
        for row in rows
    ]
    readout_signal = [float(row.get("readout_sibling_response", 0.0)) for row in readout_rows]
    carry_rows = [dict(row.get("branch_identity", {})) for row in rows]
    named_carry_rows = [
        (name, dict(row.get("branch_identity", {})))
        for name, row in branch_items
        if isinstance(row.get("branch_identity", {}), dict)
    ]
    best_identity_name = ""
    best_identity = {}
    if named_carry_rows:
        best_identity_name, best_identity = max(
            named_carry_rows,
            key=lambda item: (
                float(item[1].get("same_child_qualified_carry_fraction", 0.0)),
                float(item[1].get("same_child_carry_fraction", 0.0)),
                float(item[1].get("same_child_qualified_carry_steps", 0.0)),
                float(item[1].get("same_child_sibling_divergence", 0.0)),
            ),
        )
    child_active = [
        1.0
        if (
            float(row.get("child_branch_probe", {}).get("real_branch_fraction", 0.0)) > 0.0
            or float(row.get("child_branch_probe", {}).get("child_writeback_mass", 0.0)) > 0.0
            or float(row.get("child_summary", {}).get("spawn_count", 0.0)) > 0.0
            or float(row.get("child_summary", {}).get("writeback_count", 0.0)) > 0.0
        )
        else 0.0
        for row in rows
    ]
    continuity_rows = [dict(row.get("continuity_summary", {})) for row in rows if row.get("continuity_summary")]
    continuity_delta_rows = [dict(row.get("continuity_delta", {})) for row in rows if row.get("continuity_delta")]
    partition_rows = [
        dict(row.get("assay_child_partition", {}))
        for row in rows
        if isinstance(row.get("assay_child_partition", {}), dict)
    ]
    predictive_rows = [
        dict(row.get("child_predictive_assimilation", {}))
        for row in rows
        if isinstance(row.get("child_predictive_assimilation", {}), dict)
    ]
    family_breakdown: dict[str, Any] = {}
    label_names = ("nested_sibling", "overwrite_or_world_jump", "over_rigid_or_frozen", "ambiguous_middle")
    assay_family_rollup = _assay_family_rollup(rows)
    readout_fraction = float(assay_family_rollup.get("assay_readout_nested_sibling_fraction", 0.0))
    continuation_fraction = float(assay_family_rollup.get("assay_continuation_nested_sibling_fraction", 0.0))
    mode_replace_fraction = float(assay_family_rollup.get("assay_mode_replace_nested_sibling_fraction", 0.0))
    causal_fraction = max(continuation_fraction, mode_replace_fraction)
    direct_readout_dependency = max(0.0, readout_fraction - causal_fraction)
    final_direct_mix_shortcut_score = max(0.0, continuation_fraction - mode_replace_fraction)
    child_record_survival = _child_record_survival_score(best_identity, thresholds) if best_identity else 0.0
    mode_replace_conversion_score = (
        mode_replace_fraction
        * child_record_survival
        * max(0.0, min(1.0, float(assay_family_rollup.get("assay_mode_replace_mean_fine_q_profile_corr", 0.0))))
        * max(0.0, min(1.0, 1.0 - float(np.mean([1.0 if row.get("label") == "overwrite_or_world_jump" else 0.0 for row in rows]))))
    )
    for family in sorted({str(row.get("family", "other")) for row in rows}):
        family_items = [(name, row) for name, row in branch_items if str(row.get("family", "other")) == family]
        family_rows = [row for _, row in family_items]
        family_readout_signal = [float(row.get("readout_sibling_response", 0.0)) for row in family_rows]
        family_carry_rows = [dict(row.get("branch_identity", {})) for row in family_rows]
        family_continuity = [dict(row.get("continuity_summary", {})) for row in family_rows if row.get("continuity_summary")]
        family_continuity_delta = [dict(row.get("continuity_delta", {})) for row in family_rows if row.get("continuity_delta")]
        family_partition_rows = [
            dict(row.get("assay_child_partition", {}))
            for row in family_rows
            if isinstance(row.get("assay_child_partition", {}), dict)
        ]
        family_predictive_rows = [
            dict(row.get("child_predictive_assimilation", {}))
            for row in family_rows
            if isinstance(row.get("child_predictive_assimilation", {}), dict)
        ]
        family_label_counts = {}
        if family in NESTED_ASSAY_FAMILIES:
            for label in label_names:
                label_key = _family_specific_label(family, label)
                family_label_counts[label_key] = int(
                    sum(
                        1
                        for row in family_rows
                        if str(row.get("family_label") or _family_specific_label(family, str(row.get("label", "")))) == label_key
                    )
                )
        family_breakdown[family] = {
            "branch_count": int(len(family_rows)),
            "label_counts": {
                label: int(sum(1 for row in family_rows if row.get("label") == label))
                for label in label_names
            },
            "family_label_counts": family_label_counts,
            "nested_sibling_fraction": float(np.mean([1.0 if row.get("label") == "nested_sibling" else 0.0 for row in family_rows])),
            "mean_child_response_score": float(np.mean([_branch_response_score(row) for row in family_rows])),
            "mean_readout_sibling_response": float(np.mean(family_readout_signal)) if family_readout_signal else 0.0,
            "mean_branch_identity_carry": float(np.mean([float(row.get("same_child_carry_fraction", 0.0)) for row in family_carry_rows])) if family_carry_rows else 0.0,
            "mean_branch_identity_qualified_carry": float(np.mean([float(row.get("same_child_qualified_carry_fraction", 0.0)) for row in family_carry_rows])) if family_carry_rows else 0.0,
            "mean_branch_identity_budget_retained": float(np.mean([float(row.get("same_child_budget_retained", 0.0)) for row in family_carry_rows])) if family_carry_rows else 0.0,
            "mean_branch_identity_support": float(np.mean([float(row.get("same_child_support_mean", 0.0)) for row in family_carry_rows])) if family_carry_rows else 0.0,
            "mean_branch_identity_coherence": float(np.mean([float(row.get("same_child_coherence_mean", 0.0)) for row in family_carry_rows])) if family_carry_rows else 0.0,
            "mean_coarse_env_corr": float(np.mean([float(row.get("coarse_env_corr", 0.0)) for row in family_rows])) if family_rows else 0.0,
            "mean_fine_q_profile_corr": float(np.mean([float(row.get("fine_q_profile_corr", 0.0)) for row in family_rows])) if family_rows else 0.0,
            "mean_nested_sibling_readiness": float(np.mean([_nested_sibling_readiness(row) for row in family_rows])) if family_rows else 0.0,
            "max_nested_sibling_readiness": float(np.max([_nested_sibling_readiness(row) for row in family_rows])) if family_rows else 0.0,
            "mean_child_volume_active_count": float(np.mean([float(row.get("after_active_count", 0.0)) for row in family_partition_rows])) if family_partition_rows else 0.0,
            "mean_child_volume_effective_count": float(np.mean([float(row.get("after_effective_count", 0.0)) for row in family_partition_rows])) if family_partition_rows else 0.0,
            "mean_child_volume_top_score_share": float(np.mean([float(row.get("after_top_score_share", 0.0)) for row in family_partition_rows])) if family_partition_rows else 0.0,
            "mean_child_fragment_count": float(np.mean([float(row.get("fragment_count", 0.0)) for row in family_partition_rows])) if family_partition_rows else 0.0,
            "mean_child_grandchild_count": float(np.mean([float(row.get("grandchild_count", 0.0)) for row in family_partition_rows])) if family_partition_rows else 0.0,
            "max_child_generation": float(np.max([float(row.get("after_max_generation", 0.0)) for row in family_partition_rows])) if family_partition_rows else 0.0,
            "mean_child_predictive_residual_reduction": float(np.mean([float(row.get("residual_reduction", 0.0)) for row in family_predictive_rows])) if family_predictive_rows else 0.0,
            "mean_child_predictive_positive_residual_reduction": float(np.mean([float(row.get("positive_residual_reduction", 0.0)) for row in family_predictive_rows])) if family_predictive_rows else 0.0,
            "mean_child_predictive_boundary_match": float(np.mean([float(row.get("boundary_match", 0.0)) for row in family_predictive_rows])) if family_predictive_rows else 0.0,
            "mean_child_predictive_ontology_binding_mass": float(np.mean([float(row.get("ontology_binding_mass", 0.0)) for row in family_predictive_rows])) if family_predictive_rows else 0.0,
            "mean_child_predictive_applied_fraction": float(np.mean([float(row.get("applied", 0.0)) for row in family_predictive_rows])) if family_predictive_rows else 0.0,
            "mean_child_predictive_learned_bridge_score": float(np.mean([float(row.get("learned_parent_bridge_score", 0.0)) for row in family_predictive_rows])) if family_predictive_rows else 0.0,
            "mean_child_predictive_learned_bridge_acceptance": float(np.mean([float(row.get("learned_parent_bridge_acceptance", 0.0)) for row in family_predictive_rows])) if family_predictive_rows else 0.0,
            "mean_child_predictive_learned_bridge_authority_acceptance": float(np.mean([float(row.get("learned_parent_bridge_authority_acceptance", 0.0)) for row in family_predictive_rows])) if family_predictive_rows else 0.0,
            "mean_child_predictive_learned_bridge_delta_scale": float(np.mean([float(row.get("learned_parent_bridge_delta_scale", 0.0)) for row in family_predictive_rows])) if family_predictive_rows else 0.0,
            "mean_child_predictive_learned_bridge_prewrite_gate": float(np.mean([float(row.get("learned_parent_bridge_prewrite_gate", 0.0)) for row in family_predictive_rows])) if family_predictive_rows else 0.0,
            "mean_child_predictive_learned_bridge_effective_delta_scale": float(np.mean([float(row.get("learned_parent_bridge_effective_delta_scale", 0.0)) for row in family_predictive_rows])) if family_predictive_rows else 0.0,
            "mean_continuity_recurrence_score": float(np.mean([float(row.get("recurrence_score", 0.0)) for row in family_continuity])) if family_continuity else 0.0,
            "mean_continuity_delta_magnitude": float(np.mean([float(row.get("max_abs_mean_delta", 0.0)) for row in family_continuity_delta])) if family_continuity_delta else 0.0,
            "branch_continuity": {
                name: {
                    "continuity_summary": row.get("continuity_summary", {}),
                    "continuity_delta": row.get("continuity_delta", {}),
                }
                for name, row in family_items
                if row.get("continuity_summary") or row.get("continuity_delta")
            },
        }

    response.update(
        {
            "mean_child_response_score": float(np.mean(response_scores)),
            "max_child_response_score": float(np.max(response_scores)),
            "mean_child_active_fraction": float(np.mean(child_active)),
            "mean_child_meso_response": float(np.mean([float(row.get("child_branch_probe", {}).get("meso_branch_effect", 0.0)) for row in rows])),
            "mean_coarse_preservation": float(np.mean([float(row.get("coarse_env_corr", 0.0)) for row in rows])),
            "mean_world_jump_penalty": float(np.mean([1.0 if row.get("label") == "overwrite_or_world_jump" else 0.0 for row in rows])),
            "nested_sibling_fraction": float(np.mean([1.0 if row.get("label") == "nested_sibling" else 0.0 for row in rows])),
            "over_rigid_fraction": float(np.mean([1.0 if row.get("label") == "over_rigid_or_frozen" else 0.0 for row in rows])),
            "mean_child_survival_signal": float(np.mean(survival_signal)),
            "mean_readout_sibling_response": float(np.mean(readout_signal)) if readout_signal else 0.0,
            "direct_readout_dependency": float(direct_readout_dependency),
            "final_direct_mix_shortcut_score": float(final_direct_mix_shortcut_score),
            "child_record_survival_score": float(child_record_survival),
            "parent_mode_conversion_sibling_fraction": float(mode_replace_fraction),
            "mode_replace_conversion_score": float(mode_replace_conversion_score),
            "mean_child_volume_active_count": float(np.mean([float(row.get("after_active_count", 0.0)) for row in partition_rows])) if partition_rows else 0.0,
            "mean_child_volume_effective_count": float(np.mean([float(row.get("after_effective_count", 0.0)) for row in partition_rows])) if partition_rows else 0.0,
            "mean_child_volume_top_score_share": float(np.mean([float(row.get("after_top_score_share", 0.0)) for row in partition_rows])) if partition_rows else 0.0,
            "mean_assay_child_fragment_count": float(np.mean([float(row.get("fragment_count", 0.0)) for row in partition_rows])) if partition_rows else 0.0,
            "max_assay_child_fragment_count": float(np.max([float(row.get("fragment_count", 0.0)) for row in partition_rows])) if partition_rows else 0.0,
            "mean_assay_grandchild_count": float(np.mean([float(row.get("grandchild_count", 0.0)) for row in partition_rows])) if partition_rows else 0.0,
            "max_assay_grandchild_count": float(np.max([float(row.get("grandchild_count", 0.0)) for row in partition_rows])) if partition_rows else 0.0,
            "max_assay_child_generation": float(np.max([float(row.get("after_max_generation", 0.0)) for row in partition_rows])) if partition_rows else 0.0,
            "mean_child_predictive_residual_reduction": float(np.mean([float(row.get("residual_reduction", 0.0)) for row in predictive_rows])) if predictive_rows else 0.0,
            "max_child_predictive_residual_reduction": float(np.max([float(row.get("residual_reduction", 0.0)) for row in predictive_rows])) if predictive_rows else 0.0,
            "mean_child_predictive_positive_residual_reduction": float(np.mean([float(row.get("positive_residual_reduction", 0.0)) for row in predictive_rows])) if predictive_rows else 0.0,
            "mean_child_predictive_boundary_match": float(np.mean([float(row.get("boundary_match", 0.0)) for row in predictive_rows])) if predictive_rows else 0.0,
            "mean_child_predictive_explained_parent_defect": float(np.mean([float(row.get("explained_parent_defect", 0.0)) for row in predictive_rows])) if predictive_rows else 0.0,
            "mean_child_predictive_ontology_binding_mass": float(np.mean([float(row.get("ontology_binding_mass", 0.0)) for row in predictive_rows])) if predictive_rows else 0.0,
            "mean_child_predictive_applied_fraction": float(np.mean([float(row.get("applied", 0.0)) for row in predictive_rows])) if predictive_rows else 0.0,
            "mean_child_predictive_target_residual_abs": float(np.mean([float(row.get("target_residual_abs_mean", 0.0)) for row in predictive_rows])) if predictive_rows else 0.0,
            "mean_child_predictive_raw_delta_abs": float(np.mean([float(row.get("raw_child_delta_abs_mean", 0.0)) for row in predictive_rows])) if predictive_rows else 0.0,
            "mean_child_predictive_candidate_delta_abs": float(np.mean([float(row.get("candidate_delta_abs_mean", 0.0)) for row in predictive_rows])) if predictive_rows else 0.0,
            "mean_child_predictive_oracle_residual_reduction": float(np.mean([float(row.get("oracle_residual_reduction", 0.0)) for row in predictive_rows])) if predictive_rows else 0.0,
            "max_child_predictive_oracle_residual_reduction": float(np.max([float(row.get("oracle_residual_reduction", 0.0)) for row in predictive_rows])) if predictive_rows else 0.0,
            "mean_child_predictive_learned_bridge_score": float(np.mean([float(row.get("learned_parent_bridge_score", 0.0)) for row in predictive_rows])) if predictive_rows else 0.0,
            "mean_child_predictive_learned_bridge_acceptance": float(np.mean([float(row.get("learned_parent_bridge_acceptance", 0.0)) for row in predictive_rows])) if predictive_rows else 0.0,
            "mean_child_predictive_learned_bridge_authority_acceptance": float(np.mean([float(row.get("learned_parent_bridge_authority_acceptance", 0.0)) for row in predictive_rows])) if predictive_rows else 0.0,
            "mean_child_predictive_learned_bridge_boundary_score": float(np.mean([float(row.get("learned_parent_bridge_boundary_score", 0.0)) for row in predictive_rows])) if predictive_rows else 0.0,
            "mean_child_predictive_learned_bridge_volume_score": float(np.mean([float(row.get("learned_parent_bridge_volume_score", 0.0)) for row in predictive_rows])) if predictive_rows else 0.0,
            "mean_child_predictive_learned_bridge_support_score": float(np.mean([float(row.get("learned_parent_bridge_support_score", 0.0)) for row in predictive_rows])) if predictive_rows else 0.0,
            "mean_child_predictive_learned_bridge_coherence_score": float(np.mean([float(row.get("learned_parent_bridge_coherence_score", 0.0)) for row in predictive_rows])) if predictive_rows else 0.0,
            "mean_child_predictive_learned_bridge_delta_scale": float(np.mean([float(row.get("learned_parent_bridge_delta_scale", 0.0)) for row in predictive_rows])) if predictive_rows else 0.0,
            "mean_child_predictive_learned_bridge_prewrite_gate": float(np.mean([float(row.get("learned_parent_bridge_prewrite_gate", 0.0)) for row in predictive_rows])) if predictive_rows else 0.0,
            "mean_child_predictive_learned_bridge_effective_delta_scale": float(np.mean([float(row.get("learned_parent_bridge_effective_delta_scale", 0.0)) for row in predictive_rows])) if predictive_rows else 0.0,
            "mean_parent_ontology_charge": float(np.mean([float(row.get("parent_ontology_charge_mean", 0.0)) for row in predictive_rows])) if predictive_rows else 0.0,
            "max_parent_ontology_charge": float(np.max([float(row.get("parent_ontology_charge_max", 0.0)) for row in predictive_rows])) if predictive_rows else 0.0,
            "mean_parent_ontology_mode1_occupancy_after": float(np.mean([float(row.get("parent_ontology_mode1_occupancy_after", 0.0)) for row in predictive_rows])) if predictive_rows else 0.0,
            "mean_branch_identity_carry": float(np.mean([float(row.get("same_child_carry_fraction", 0.0)) for row in carry_rows])) if carry_rows else 0.0,
            "mean_branch_identity_qualified_carry": float(np.mean([float(row.get("same_child_qualified_carry_fraction", 0.0)) for row in carry_rows])) if carry_rows else 0.0,
            "max_branch_identity_carry": float(np.max([float(row.get("same_child_carry_fraction", 0.0)) for row in carry_rows])) if carry_rows else 0.0,
            "max_branch_identity_qualified_carry": float(np.max([float(row.get("same_child_qualified_carry_fraction", 0.0)) for row in carry_rows])) if carry_rows else 0.0,
            "max_branch_identity_qualified_steps": float(np.max([float(row.get("same_child_qualified_carry_steps", 0.0)) for row in carry_rows])) if carry_rows else 0.0,
            "max_branch_identity_budget_retained": float(np.max([float(row.get("same_child_budget_retained", 0.0)) for row in carry_rows])) if carry_rows else 0.0,
            "max_branch_identity_parent_div": float(np.max([float(row.get("same_child_parent_divergence", 0.0)) for row in carry_rows])) if carry_rows else 0.0,
            "max_branch_identity_sibling_div": float(np.max([float(row.get("same_child_sibling_divergence", 0.0)) for row in carry_rows])) if carry_rows else 0.0,
            "best_identity_branch": best_identity_name,
            "best_identity_support_mean": float(best_identity.get("same_child_support_mean", 0.0)),
            "best_identity_coherence_mean": float(best_identity.get("same_child_coherence_mean", 0.0)),
            "best_identity_min_support": float(best_identity.get("same_child_min_support", 0.0)),
            "best_identity_min_coherence": float(best_identity.get("same_child_min_coherence", 0.0)),
            "best_identity_branch_metrics": best_identity,
            "mean_branch_identity_budget_retained": float(np.mean([float(row.get("same_child_budget_retained", 0.0)) for row in carry_rows])) if carry_rows else 0.0,
            "mean_branch_identity_parent_div": float(np.mean([float(row.get("same_child_parent_divergence", 0.0)) for row in carry_rows])) if carry_rows else 0.0,
            "mean_branch_identity_sibling_div": float(np.mean([float(row.get("same_child_sibling_divergence", 0.0)) for row in carry_rows])) if carry_rows else 0.0,
            "readout_branch_count": float(len(readout_rows)),
            "mean_continuity_recurrence_score": float(np.mean([float(row.get("recurrence_score", 0.0)) for row in continuity_rows])) if continuity_rows else 0.0,
            "max_continuity_recurrence_score": float(np.max([float(row.get("recurrence_score", 0.0)) for row in continuity_rows])) if continuity_rows else 0.0,
            "mean_continuity_adjacent_chunk_similarity": float(np.mean([float(row.get("adjacent_chunk_similarity", 0.0)) for row in continuity_rows])) if continuity_rows else 0.0,
            "mean_continuity_nonlocal_chunk_repeat": float(np.mean([float(row.get("nonlocal_chunk_repeat", 0.0)) for row in continuity_rows])) if continuity_rows else 0.0,
            "mean_continuity_first_chunk_reentry": float(np.mean([float(row.get("first_chunk_reentry", 0.0)) for row in continuity_rows])) if continuity_rows else 0.0,
            "mean_continuity_loop_autocorr_peak": float(np.mean([float(row.get("loop_autocorr_peak", 0.0)) for row in continuity_rows])) if continuity_rows else 0.0,
            "mean_continuity_delta_magnitude": float(np.mean([float(row.get("max_abs_mean_delta", 0.0)) for row in continuity_delta_rows])) if continuity_delta_rows else 0.0,
            "max_continuity_delta_magnitude": float(np.max([float(row.get("max_abs_mean_delta", 0.0)) for row in continuity_delta_rows])) if continuity_delta_rows else 0.0,
            "branch_family_counts": branch_family_counts,
            "family_breakdown": family_breakdown,
            **assay_family_rollup,
            "branch_continuity": {
                name: {
                    "family": str(row.get("family", "other")),
                    "continuity_summary": row.get("continuity_summary", {}),
                    "continuity_delta": row.get("continuity_delta", {}),
                }
                for name, row in branch_items
                if row.get("continuity_summary") or row.get("continuity_delta")
            },
        }
    )
    return response


def _compare_branch(
    base_wav: np.ndarray,
    branch_wav: np.ndarray,
    sr: int,
    base_run: dict[str, Any],
    base_continuity: dict[str, Any],
    branch_run: dict[str, Any],
    branch_path: Path,
    branch_identity: dict[str, float] | None = None,
) -> dict[str, Any]:
    env_base = _spectral_envelope(base_wav, sr)
    env_branch = _spectral_envelope(branch_wav, sr)
    energy_base = _energy_contour(base_wav, sr)
    energy_branch = _energy_contour(branch_wav, sr)
    q_base = _q_profile(base_run["final_arc"])
    q_branch = _q_profile(branch_run["final_arc"])
    continuity = analyze_wav(branch_path, min_loop_seconds=0.5, max_loop_seconds=4.0, chunk_seconds=2.0)
    continuity_compare = compare_continuity_summaries(
        continuity,
        base_continuity,
        lhs_label=branch_path.stem,
        rhs_label=Path(base_run["render_wav"]).stem,
    )
    coarse_env_corr = _corr(env_base, env_branch)
    energy_corr = _corr(energy_base, energy_branch)
    meso_chunk_similarity = _mean_chunk_similarity(base_wav, branch_wav, sr)
    packet_succession_similarity = _corr(_packet_succession_vector(base_run), _packet_succession_vector(branch_run))
    q_profile_corr = _corr(q_base, q_branch)
    fine_texture_distance = _short_texture_distance(base_wav, branch_wav, sr)
    residue_delta = abs(
        float(base_run["final_arc"]["minor_residue"].mean().item())
        - float(branch_run["final_arc"]["minor_residue"].mean().item())
    )
    loop_period_delta = abs(
        float(continuity["loop_period_seconds"])
        - float(base_continuity["loop_period_seconds"])
    )
    summary = summarize_circleworld_run(branch_run)
    branch_identity = dict(branch_identity or {})
    readout_sibling_response = (
        max(0.0, coarse_env_corr - 0.95)
        + 2.0 * max(0.0, q_profile_corr - 0.95)
        + 15.0 * fine_texture_distance
        - 2.0 * (1.0 if coarse_env_corr < 0.75 or q_profile_corr < 0.60 else 0.0)
    )
    return {
        "coarse_env_corr": coarse_env_corr,
        "coarse_energy_corr": energy_corr,
        "meso_chunk_similarity": meso_chunk_similarity,
        "meso_packet_succession_similarity": packet_succession_similarity,
        "meso_loop_period_delta": loop_period_delta,
        "fine_texture_distance": fine_texture_distance,
        "fine_q_profile_corr": q_profile_corr,
        "fine_minor_residue_delta": residue_delta,
        "continuity": continuity,
        "continuity_summary": continuity_snapshot(continuity),
        "continuity_delta": continuity_delta_snapshot(continuity_compare),
        "readout_sibling_response": float(readout_sibling_response),
        "continuation_mode1_write": dict(branch_run.get("continuation_mode1_write", {})),
        "mode1_replace_metrics": dict(branch_run.get("mode1_replace_metrics", {})),
        "assay_child_partition": dict(branch_run.get("assay_child_partition", {})),
        "child_predictive_assimilation": dict(branch_run.get("child_predictive_assimilation", {})),
        "branch_identity": branch_identity,
        "child_summary": _child_history_summary(branch_run),
        "child_branch_probe": {
            "real_branch_fraction": float(summary.get("real_branch_fraction", 0.0)),
            "meso_branch_effect": float(summary.get("meso_branch_effect", 0.0)),
            "child_writeback_mass": float(summary.get("mean_child_writeback_mass", 0.0)),
            "child_parent_divergence": float(summary.get("mean_child_parent_divergence", 0.0)),
            "child_sibling_divergence": float(summary.get("mean_child_sibling_divergence", 0.0)),
        },
    }


def _make_case(
    name: str,
    reference_wav: Path | None,
    config_path: Path,
    out_dir: Path,
    device: torch.device,
    total_depth: int,
    fork_depth: int,
    fork_selector: str,
    live_child_threshold: float,
    mode: str,
    case: dict[str, Any] | None = None,
) -> dict[str, Any]:
    cfg = _load_circle_cfg(config_path)
    stft_cfg = load_config()["data"]["stft"]
    sample_rate = int(load_config()["data"].get("sample_rate", 16000))
    case = dict(case or {})
    if reference_wav is not None:
        wav, sr = prepare_reference_audio(reference_wav, device_name=device.type, clip_seconds_override=10)
        mag, phase = compute_stft(wav, stft_cfg)
        z0 = phasor_normalize(torch.stack([torch.cos(phase), torch.sin(phase)], dim=-1))
        case_spec = {"reference_wav": str(reference_wav)}
        warmup_depth = int(total_depth)
        continuation_depth = max(1, total_depth - max(1, fork_depth))
        require_live_child_start = False
    else:
        seed_source = str(case.get("seed_source", "naked_rafa"))
        seed = int(case.get("seed", 0))
        time_steps = int(case.get("time_steps", 128))
        rafa_core = make_seed_rafa(dev=device.type) if seed_source == "naked_rafa" else None
        z0 = _generate_seed_phase(
            rafa_core=rafa_core,
            batch_size=1,
            time_steps=time_steps,
            device=device,
            seed_source=seed_source,
            seed=seed,
        ).detach()
        mag = _generate_seed_magnitude(
            batch_size=1,
            freq_bins=int(z0.size(1)),
            time_steps=int(z0.size(2)),
            device=device,
            seed=seed,
        )
        sr = sample_rate
        case_spec = {
            "seed_source": seed_source,
            "seed": seed,
            "time_steps": time_steps,
        }
        warmup_depth = int(case.get("warmup_depth", max(total_depth, 5)))
        continuation_depth = int(case.get("continuation_depth", total_depth))
        require_live_child_start = bool(case.get("require_live_child_start", True))

    run_mode = cfg.branching_mode if cfg.branching_mode in {"native_multimode", "native_multimode_childworld"} else mode
    trace = _run_depth_trace(z0, cfg=cfg, depth=max(1, warmup_depth), mode=run_mode)
    states = trace["states"]
    selected_fork_depth = int(fork_depth)
    live_child_start_found = False
    selected_child_score = 0.0
    if fork_selector != "fixed" and trace["history"]:
        best_idx = None
        best_score = -1.0
        fallback_idx = None
        fallback_score = -1.0
        for idx, block in enumerate(trace["history"], start=1):
            child_signal = _child_signal_metrics(block, live_child_threshold)
            live_child = float(child_signal["live_child_fraction"])
            child_count = float(child_signal["child_world_count"])
            child_score = float(child_signal["child_score"])
            if child_score > fallback_score:
                fallback_idx = idx
                fallback_score = child_score
            if fork_selector == "first_live_child":
                if live_child >= live_child_threshold or child_count > 0.0:
                    best_idx = idx
                    best_score = child_score
                    break
            elif fork_selector == "max_live_child":
                if live_child >= live_child_threshold or child_count > 0.0:
                    if child_score > best_score:
                        best_idx = idx
                        best_score = child_score
        if best_idx is not None:
            selected_fork_depth = int(best_idx)
            live_child_start_found = True
            selected_child_score = float(best_score)
        elif require_live_child_start and fallback_idx is not None:
            selected_fork_depth = int(fallback_idx)
            selected_child_score = float(fallback_score)
    selected_fork_depth = max(1, min(int(selected_fork_depth), max(1, warmup_depth)))
    selected_block = trace["history"][selected_fork_depth - 1] if trace["history"] else None
    selected_child_signal = _child_signal_metrics(selected_block, live_child_threshold)
    if selected_child_score <= 0.0:
        selected_child_score = float(selected_child_signal["child_score"])
    selected_live_child_gate_passed = bool(selected_child_signal["live_child_gate_passed"] > 0.0)
    selected_fork_has_live_child = bool(selected_live_child_gate_passed)
    selected_child_signal_found = bool(selected_child_signal["child_score"] > 0.0)
    selected_phase_only_eligible = bool(selected_child_signal_found and not selected_fork_has_live_child)
    selected_live_child_eligibility = bool(selected_live_child_gate_passed or selected_child_signal_found)

    early_state = clone_circleworld_state(states[selected_fork_depth])
    anchor_child = _select_strongest_child(early_state) if isinstance(early_state, dict) else None
    anchor_child_id = int(anchor_child.get("child_id")) if isinstance(anchor_child, dict) and anchor_child.get("child_id") is not None else None
    mid_state = clone_circleworld_state(states[min(max(1, warmup_depth) - 1, max(selected_fork_depth + 1, 1))])
    late_state = clone_circleworld_state(trace["state_bundle"])

    case_dir = out_dir / name
    case_dir.mkdir(parents=True, exist_ok=True)

    early_wav_path = case_dir / "state_early.wav"
    mid_wav_path = case_dir / "state_mid.wav"
    late_wav_path = case_dir / "state_late.wav"
    _render_phase_state(mag, extract_phase_state(early_state, cfg), stft_cfg, early_wav_path, sr)
    _render_phase_state(mag, extract_phase_state(mid_state, cfg), stft_cfg, mid_wav_path, sr)
    _render_phase_state(mag, extract_phase_state(late_state, cfg), stft_cfg, late_wav_path, sr)

    remaining_depth = max(1, continuation_depth if reference_wav is None else total_depth - selected_fork_depth)
    if run_mode in {"native_multimode", "native_multimode_childworld"}:
        branches: list[dict[str, Any]] = [
            {"name": "base", "state": early_state, "cfg": cfg},
            {"name": "logit_tilt", "state": perturb_circleworld_state(early_state, cfg, "logit_tilt", 1.1, 101), "cfg": cfg},
            {"name": "support_expand", "state": perturb_circleworld_state(early_state, cfg, "support_expand", 0.35, 202), "cfg": cfg},
            {"name": "support_suppress", "state": perturb_circleworld_state(early_state, cfg, "support_suppress", 0.60, 303), "cfg": cfg},
            {"name": "qtrace_shift", "state": perturb_circleworld_state(early_state, cfg, "qtrace_shift", 0.60, 404), "cfg": cfg},
            {"name": "child_phase_shift", "state": perturb_circleworld_state(early_state, cfg, "child_phase_shift", 0.90, 505), "cfg": cfg},
            {"name": "child_support_boost", "state": perturb_circleworld_state(early_state, cfg, "child_support_boost", 0.35, 606), "cfg": cfg},
            {"name": "child_qtrace_shift", "state": perturb_circleworld_state(early_state, cfg, "child_qtrace_shift", 0.60, 707), "cfg": cfg},
            {"name": "child_support_window_shift", "state": perturb_circleworld_state(early_state, cfg, "child_support_window_shift", 0.85, 808), "cfg": cfg},
            {"name": "child_writeback_budget_shift", "state": perturb_circleworld_state(early_state, cfg, "child_writeback_budget_shift", 0.90, 909), "cfg": cfg, "extra_depth": 1},
            {"name": "child_parent_mix_shift", "state": perturb_circleworld_state(early_state, cfg, "child_parent_mix_shift", 0.80, 1001), "cfg": cfg},
            {"name": "child_survival_age_shift", "state": perturb_circleworld_state(early_state, cfg, "child_survival_age_shift", 0.90, 1111), "cfg": cfg, "extra_depth": 1},
            {"name": "child_writeback_ready_shift", "state": perturb_circleworld_state(early_state, cfg, "child_writeback_ready_shift", 0.95, 1222), "cfg": cfg, "extra_depth": 1},
            {
                "name": "child_retention_gate_shift",
                "state": perturb_circleworld_state(early_state, cfg, "child_survival_age_shift", 0.95, 1333),
                "cfg": _lifecycle_cfg(cfg, "retention"),
                "extra_depth": 1,
                "pre_unroll": "child_only",
            },
            {
                "name": "child_writeback_gate_shift",
                "state": perturb_circleworld_state(early_state, cfg, "child_writeback_ready_shift", 1.00, 1444),
                "cfg": _lifecycle_cfg(cfg, "writeback"),
                "extra_depth": 1,
                "pre_unroll": "child_only",
            },
            {
                "name": "child_parent_protect_shift",
                "state": perturb_circleworld_state(early_state, cfg, "child_writeback_ready_shift", 1.00, 1555),
                "cfg": _lifecycle_cfg(cfg, "writeback"),
                "extra_depth": 1,
                "pre_unroll": "child_parent_protect",
            },
            {
                "name": "child_delayed_remix_shift",
                "state": perturb_circleworld_state(early_state, cfg, "child_survival_age_shift", 1.00, 1666),
                "cfg": _lifecycle_cfg(cfg, "retention"),
                "extra_depth": 1,
                "pre_unroll": "child_delayed_remix",
            },
            {
                "name": "child_local_readout_shift",
                "state": perturb_circleworld_state(early_state, cfg, "child_phase_shift", 1.05, 1777),
                "cfg": _lifecycle_cfg(cfg, "retention"),
                "extra_depth": 1,
                "pre_unroll": "child_parent_protect",
                "readout_mode": "child_local",
            },
            {
                "name": "child_dominant_readout_shift",
                "state": perturb_circleworld_state(early_state, cfg, "child_writeback_ready_shift", 1.05, 1888),
                "cfg": _lifecycle_cfg(cfg, "writeback"),
                "extra_depth": 1,
                "pre_unroll": "child_delayed_remix",
                "readout_mode": "child_dominant",
            },
            {
                "name": "child_local_continuation_shift",
                "state": perturb_circleworld_state(early_state, cfg, "child_phase_shift", 1.10, 1999),
                "cfg": _lifecycle_cfg(cfg, "retention"),
                "extra_depth": 1,
                "pre_unroll": "child_parent_protect",
                "readout_mode": "child_continuation_local",
            },
            {
                "name": "child_dominant_continuation_shift",
                "state": perturb_circleworld_state(early_state, cfg, "child_writeback_ready_shift", 1.10, 2111),
                "cfg": _lifecycle_cfg(cfg, "writeback"),
                "extra_depth": 1,
                "pre_unroll": "child_delayed_remix",
                "readout_mode": "child_continuation_dominant",
            },
            {
                "name": "child_mode1_replace_50_shift",
                "state": perturb_circleworld_state(early_state, cfg, "child_phase_shift", 1.10, 2222),
                "cfg": _lifecycle_cfg(cfg, "retention"),
                "extra_depth": 1,
                "pre_unroll": "child_delayed_remix",
                "readout_mode": "child_mode1_replace_50",
            },
            {
                "name": "child_mode1_replace_85_shift",
                "state": perturb_circleworld_state(early_state, cfg, "child_writeback_ready_shift", 1.10, 2333),
                "cfg": _lifecycle_cfg(cfg, "writeback"),
                "extra_depth": 1,
                "pre_unroll": "child_delayed_remix",
                "readout_mode": "child_mode1_replace_85",
            },
            {"name": "promotion_lo", "state": clone_circleworld_state(early_state), "cfg": _promotion_cfg(cfg, "lo")},
            {"name": "promotion_hi", "state": clone_circleworld_state(early_state), "cfg": _promotion_cfg(cfg, "hi")},
        ]
    else:
        early_packets = trace["packets"][max(0, selected_fork_depth - 1)] if selected_fork_depth > 0 else []
        branches = [
            {"name": "base", "state": early_state, "cfg": cfg},
            {"name": "fine_noise_small", "state": _phase_noise_perturb(early_state, 0.015, seed=101), "cfg": cfg},
            {"name": "fine_noise_large", "state": _phase_noise_perturb(early_state, 0.05, seed=202), "cfg": cfg},
            {"name": "packet_primary", "state": _packet_seed_perturb(early_state, early_packets, strength=0.18, rank=0), "cfg": cfg},
            {"name": "packet_secondary", "state": _packet_seed_perturb(early_state, early_packets, strength=0.12, rank=1), "cfg": cfg},
            {"name": "promotion_lo", "state": early_state, "cfg": _promotion_cfg(cfg, "lo")},
            {"name": "promotion_hi", "state": early_state, "cfg": _promotion_cfg(cfg, "hi")},
        ]

    rendered: dict[str, dict[str, Any]] = {}
    base_wave_np: np.ndarray | None = None
    base_run: dict[str, Any] | None = None
    case_law_objects: list[dict[str, Any]] = []

    for branch in branches:
        branch_name = branch["name"]
        branch_cfg = branch["cfg"]
        branch_family = _branch_family_tag(branch_name, readout_mode=branch.get("readout_mode"))
        runtime_cfg = _without_childworld_runtime_cfg(branch_cfg) if _assay_flag("disable_branch_childworld_runtime") else branch_cfg
        branch_mode = runtime_cfg.branching_mode if runtime_cfg.branching_mode in {"native_multimode", "native_multimode_childworld"} else mode
        branch_state = branch["state"]
        if _assay_flag("strip_child_worlds_before_branch") and isinstance(branch_state, dict) and "phase_modes" in branch_state:
            branch_state = _strip_child_worlds_for_killswitch(branch_state, branch_cfg)
        branch_state, child_partition_metrics = _apply_assay_child_partitioning(branch_state, branch_cfg)
        branch_state, child_predictive_metrics = _apply_child_predictive_assimilation(
            branch_state,
            branch_cfg,
            context={"case": name, "branch": branch_name, "case_dir": str(case_dir)},
        )
        if not _assay_flag("disable_branch_pre_unroll"):
            if branch.get("pre_unroll") == "child_only":
                branch_state = _child_only_pre_unroll(branch_state, branch_cfg)
            elif branch.get("pre_unroll") == "child_parent_protect":
                branch_state = _child_only_pre_unroll(
                    branch_state,
                    branch_cfg,
                    steps=1,
                    parent_write_scale=0.20,
                    mode0_protect=0.85,
                )
            elif branch.get("pre_unroll") == "child_delayed_remix":
                branch_state = _child_only_pre_unroll(
                    branch_state,
                    branch_cfg,
                    steps=2,
                    parent_write_scale=0.0,
                    mode0_protect=0.92,
                )
        branch_law_objects = law_objects_from_state(
            branch_state,
            context={"case": name, "branch": branch_name, "family": branch_family, "source": "branch_start"},
        )
        branch_depth = max(1, remaining_depth + int(branch.get("extra_depth", 0)))
        run = _run_depth_trace(branch_state, cfg=runtime_cfg, depth=branch_depth, mode=branch_mode)
        if isinstance(run.get("state_bundle"), dict):
            run["state_bundle"] = _carry_parent_ontology_fields(branch_state, run["state_bundle"])
            branch_law_objects.extend(
                law_objects_from_state(
                    run["state_bundle"],
                    context={"case": name, "branch": branch_name, "family": branch_family, "source": "branch_final"},
                )
            )
        run = _apply_readout_override(run, branch_cfg, branch.get("readout_mode"))
        run["assay_child_partition"] = dict(child_partition_metrics)
        run["child_predictive_assimilation"] = dict(child_predictive_metrics)
        branch_identity_states = [clone_circleworld_state(branch_state)]
        branch_identity_states.extend(
            clone_circleworld_state(state)
            for state in run.get("states", [])[1:]
            if isinstance(state, dict) and "phase_modes" in state
        )
        probe_state = run.get("identity_probe_state")
        if isinstance(probe_state, dict) and "phase_modes" in probe_state:
            branch_identity_states.append(clone_circleworld_state(probe_state))
        branch_identity = _branch_identity_summary(anchor_child_id, branch_identity_states)
        out_path = case_dir / f"{branch_name}.wav"
        wav_out = _render_phase_state(mag, run["phase_state"], stft_cfg, out_path, sr).numpy()
        run["render_wav"] = str(out_path)
        continuation_write = dict(run.get("continuation_mode1_write", {}))
        continuation_write_executed = float(continuation_write.get("enabled", 0.0)) > 0.0
        case_law_objects.extend(branch_law_objects)
        rendered[branch_name] = {
            "summary": summarize_circleworld_run(run),
            "wav_path": str(out_path),
            "cfg": {
                "promotion_threshold": branch_cfg.promotion_threshold,
                "max_promotions": branch_cfg.max_promotions,
                "child_law_gain": branch_cfg.child_law_gain,
                "readout_mode": branch.get("readout_mode"),
                "family": branch_family,
                "assay_disable_branch_pre_unroll": 1.0 if _assay_flag("disable_branch_pre_unroll") else 0.0,
                "assay_disable_branch_childworld_runtime": 1.0 if _assay_flag("disable_branch_childworld_runtime") else 0.0,
                "continuation_mode1_write_path": "child_support_coherence_low_rank_delta"
                if continuation_write_executed
                else None,
                "continuation_mode1_write_executed": 1.0 if continuation_write_executed else 0.0,
            },
            "run": run,
            "wave_np": wav_out,
            "child_history": list(run.get("child_event_history", [])),
            "branch_identity": branch_identity,
            "family": branch_family,
            "assay_child_partition": dict(child_partition_metrics),
            "child_predictive_assimilation": dict(child_predictive_metrics),
            "resonant_law_object_count": len(branch_law_objects),
            "resonant_law_object_ids": [str(obj.get("object_id", "")) for obj in branch_law_objects[:24]],
        }
        if branch_name == "base":
            base_wave_np = wav_out
            base_run = run

    assert base_wave_np is not None and base_run is not None
    base_continuity = analyze_wav(Path(base_run["render_wav"]), min_loop_seconds=0.5, max_loop_seconds=4.0, chunk_seconds=2.0)

    branch_metrics: dict[str, Any] = {}
    for branch_name, payload in rendered.items():
        if branch_name == "base":
            continue
        branch_metrics[branch_name] = _compare_branch(
            base_wav=base_wave_np,
            branch_wav=payload["wave_np"],
            sr=sr,
            base_run=base_run,
            base_continuity=base_continuity,
            branch_run=payload["run"],
            branch_path=Path(payload["wav_path"]),
            branch_identity=payload.get("branch_identity"),
        )
        branch_metrics[branch_name]["summary"] = payload["summary"]
        branch_metrics[branch_name]["cfg"] = payload["cfg"]
        branch_metrics[branch_name]["wav_path"] = payload["wav_path"]
        branch_metrics[branch_name]["child_history"] = payload["child_history"]
        branch_metrics[branch_name]["family"] = payload["family"]
        branch_metrics[branch_name]["assay_child_partition"] = dict(payload.get("assay_child_partition", {}))
        branch_metrics[branch_name]["child_predictive_assimilation"] = dict(payload.get("child_predictive_assimilation", {}))
        branch_metrics[branch_name]["resonant_law_object_count"] = int(payload.get("resonant_law_object_count", 0))
        branch_metrics[branch_name]["resonant_law_object_ids"] = list(payload.get("resonant_law_object_ids", []))

    readout_only_rows = [
        metrics
        for name, metrics in branch_metrics.items()
        if str(metrics.get("cfg", {}).get("readout_mode") or "") in {"child_local", "child_dominant"}
    ]
    readout_only_baseline = float(np.mean([float(row.get("readout_sibling_response", 0.0)) for row in readout_only_rows])) if readout_only_rows else 0.0
    for metrics in branch_metrics.values():
        metrics["readout_only_baseline"] = readout_only_baseline
        carry = dict(metrics.get("branch_identity", {}))
        metrics["label"] = _branch_label(
            coarse_env_corr=float(metrics.get("coarse_env_corr", 0.0)),
            meso_chunk_similarity=float(metrics.get("meso_chunk_similarity", 0.0)),
            fine_texture_distance=float(metrics.get("fine_texture_distance", 0.0)),
            q_profile_corr=float(metrics.get("fine_q_profile_corr", 0.0)),
            residue_delta=float(metrics.get("fine_minor_residue_delta", 0.0)),
            same_child_carry_fraction=float(carry.get("same_child_carry_fraction", 0.0)),
            same_child_carry_steps=float(carry.get("same_child_carry_steps", 0.0)),
            same_child_qualified_carry_fraction=float(carry.get("same_child_qualified_carry_fraction", 0.0)),
            same_child_qualified_carry_steps=float(carry.get("same_child_qualified_carry_steps", 0.0)),
            same_child_budget_retained=float(carry.get("same_child_budget_retained", 0.0)),
            readout_sibling_response=float(metrics.get("readout_sibling_response", 0.0)),
            readout_only_baseline=readout_only_baseline,
        )
        metrics["family_label"] = _family_specific_label(str(metrics.get("family", "other")), str(metrics["label"]))

    counts = {
        "nested_sibling": sum(1 for m in branch_metrics.values() if m["label"] == "nested_sibling"),
        "overwrite_or_world_jump": sum(1 for m in branch_metrics.values() if m["label"] == "overwrite_or_world_jump"),
        "over_rigid_or_frozen": sum(1 for m in branch_metrics.values() if m["label"] == "over_rigid_or_frozen"),
        "ambiguous_middle": sum(1 for m in branch_metrics.values() if m["label"] == "ambiguous_middle"),
    }
    branch_family_counts = _branch_family_counts(list(branch_metrics.values()))
    if counts["nested_sibling"] >= 3 and counts["overwrite_or_world_jump"] == 0:
        verdict = "evidence_of_nested_commitment"
    elif counts["over_rigid_or_frozen"] >= 3:
        verdict = "over_rigid_attractor"
    elif counts["overwrite_or_world_jump"] >= 2:
        verdict = "overwrite_dominant"
    else:
        verdict = "mixed_or_inconclusive"

    case_nested = _case_nested_response(
        branch_metrics,
        live_child_start_found=live_child_start_found,
        selected_fork_has_live_child=selected_fork_has_live_child,
        selected_live_child_gate_passed=selected_live_child_gate_passed,
        selected_child_signal_found=selected_child_signal_found,
    )
    resonant_retrieval = run_retrieval_assay(case_law_objects)
    resonant_composition = composition_summary(case_law_objects)

    result = {
        "case": name,
        "reference_wav": str(reference_wav) if reference_wav else None,
        "case_spec": case_spec,
        "config_path": str(config_path),
        "nested_identity_thresholds": _identity_carry_thresholds(),
        "total_depth": total_depth,
        "warmup_depth": int(max(1, warmup_depth)),
        "fork_depth": selected_fork_depth,
        "requested_fork_depth": fork_depth,
        "fork_selector": fork_selector,
        "live_child_threshold": live_child_threshold,
        "live_child_start_found": bool(live_child_start_found),
        "selected_fork_has_live_child": bool(selected_fork_has_live_child),
        "selected_live_child_gate_passed": bool(selected_live_child_gate_passed),
        "selected_child_signal_found": bool(selected_child_signal_found),
        "selected_phase_only_eligible": bool(selected_phase_only_eligible),
        "selected_live_child_eligible": bool(selected_live_child_eligibility),
        "selected_child_score": float(selected_child_score),
        "selected_child_signal": selected_child_signal,
        "remaining_depth": remaining_depth,
        "state_wavs": {
            "early": str(early_wav_path),
            "mid": str(mid_wav_path),
            "late": str(late_wav_path),
        },
        "base_continuity_summary": continuity_snapshot(base_continuity),
        "branch_metrics": branch_metrics,
        "case_nested_response": case_nested,
        "resonant_law_objects": case_law_objects,
        "resonant_retrieval": resonant_retrieval["summary"],
        "resonant_retrieval_rows": resonant_retrieval["rows"],
        "resonant_composition": resonant_composition,
        "base_child_history": list(base_run.get("child_event_history", [])),
        "branch_family_counts": branch_family_counts,
        "counts": counts,
        "verdict": verdict,
    }
    (case_dir / "nested_commitment_summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def evaluate_nested_commitment_report(
    *,
    config_path: Path,
    out_dir: Path,
    device_name: str,
    depth: int,
    fork_depth: int,
    fork_selector: str,
    live_child_threshold: float,
    mode: str,
    cases: list[dict[str, Any]],
    assay_killswitch: dict[str, Any] | None = None,
) -> dict[str, Any]:
    device = _safe_device(device_name)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    previous_killswitch = dict(ASSAY_KILLSWITCH)
    ASSAY_KILLSWITCH.clear()
    ASSAY_KILLSWITCH.update(DEFAULT_ASSAY_KILLSWITCH)
    ASSAY_KILLSWITCH.update({str(k): v for k, v in dict(assay_killswitch or {}).items()})
    try:
        for case in cases:
            rows.append(
                _make_case(
                    name=str(case["name"]),
                    reference_wav=Path(case["reference_wav"]) if case.get("reference_wav") else None,
                    config_path=config_path,
                    out_dir=out_dir,
                    device=device,
                    total_depth=int(depth),
                    fork_depth=int(fork_depth),
                    fork_selector=str(fork_selector),
                    live_child_threshold=float(live_child_threshold),
                    mode=mode,
                    case=case,
                )
            )
    finally:
        ASSAY_KILLSWITCH.clear()
        ASSAY_KILLSWITCH.update(previous_killswitch)

    aggregate = {
        "config_path": str(config_path),
        "device": str(device),
        "assay_killswitch": {str(k): v for k, v in dict(assay_killswitch or {}).items()},
        "nested_identity_thresholds": _identity_carry_thresholds(),
        "depth": int(depth),
        "fork_depth": int(fork_depth),
        "fork_selector": str(fork_selector),
        "live_child_threshold": float(live_child_threshold),
        "mode": mode,
        "cases": rows,
    }
    verdict_counts: dict[str, int] = {}
    for row in rows:
        verdict_counts[row["verdict"]] = verdict_counts.get(row["verdict"], 0) + 1
    aggregate["verdict_counts"] = verdict_counts
    aggregate["overall_read"] = (
        "nested_commitment_signal_present"
        if verdict_counts.get("evidence_of_nested_commitment", 0) >= max(1, math.ceil(len(rows) / 2))
        else "nested_commitment_not_yet_established"
    )
    branch_family_counts = {family: 0 for family in BRANCH_FAMILY_ORDER}
    for row in rows:
        for family, count in dict(row.get("branch_family_counts", {})).items():
            branch_family_counts.setdefault(str(family), 0)
            branch_family_counts[str(family)] += int(count)
    aggregate["branch_family_counts"] = branch_family_counts
    eligible_rows = [row for row in rows if float(row.get("case_nested_response", {}).get("eligible_for_nested_score", 0.0)) > 0.0]
    aggregate["num_cases"] = int(len(rows))
    aggregate["num_live_child_start_cases"] = int(sum(1 for row in rows if bool(row.get("live_child_start_found", False))))
    aggregate["num_selected_fork_has_live_child_cases"] = int(sum(1 for row in rows if bool(row.get("selected_fork_has_live_child", False))))
    aggregate["num_selected_live_child_eligible_cases"] = int(sum(1 for row in rows if bool(row.get("selected_live_child_eligible", False))))
    aggregate["num_selected_phase_only_eligible_cases"] = int(sum(1 for row in rows if bool(row.get("selected_phase_only_eligible", False))))
    aggregate["mean_selected_fork_has_live_child"] = float(
        np.mean([1.0 if bool(row.get("selected_fork_has_live_child", False)) else 0.0 for row in rows])
    ) if rows else 0.0
    aggregate["mean_selected_child_signal_found"] = float(
        np.mean([1.0 if bool(row.get("selected_child_signal_found", False)) else 0.0 for row in rows])
    ) if rows else 0.0
    aggregate["mean_selected_live_child_eligible"] = float(
        np.mean([1.0 if bool(row.get("selected_live_child_eligible", False)) else 0.0 for row in rows])
    ) if rows else 0.0
    aggregate["mean_selected_phase_only_eligible"] = float(
        np.mean([1.0 if bool(row.get("selected_phase_only_eligible", False)) else 0.0 for row in rows])
    ) if rows else 0.0
    aggregate["num_eligible_nested_cases"] = int(len(eligible_rows))
    metric_rows = eligible_rows if eligible_rows else []
    aggregate["assay_family_label_counts"] = _empty_assay_family_label_counts()
    aggregate["assay_family_nested_sibling_counts"] = {family: 0 for family in NESTED_ASSAY_FAMILIES}
    aggregate["assay_family_branch_counts"] = {family: 0 for family in NESTED_ASSAY_FAMILIES}
    for row in rows:
        nested_response = dict(row.get("case_nested_response", {}))
        row_label_counts = dict(nested_response.get("assay_family_label_counts", {}))
        row_nested_counts = dict(nested_response.get("assay_family_nested_sibling_counts", {}))
        row_branch_counts = dict(nested_response.get("assay_family_branch_counts", {}))
        for family in NESTED_ASSAY_FAMILIES:
            for label, count in dict(row_label_counts.get(family, {})).items():
                aggregate["assay_family_label_counts"][family].setdefault(str(label), 0)
                aggregate["assay_family_label_counts"][family][str(label)] += int(count)
            aggregate["assay_family_nested_sibling_counts"][family] += int(row_nested_counts.get(family, 0))
            aggregate["assay_family_branch_counts"][family] += int(row_branch_counts.get(family, 0))
    for family in NESTED_ASSAY_FAMILIES:
        aggregate[f"mean_assay_{family}_branch_count"] = float(
            np.mean([float(row.get("case_nested_response", {}).get(f"assay_{family}_branch_count", 0.0)) for row in metric_rows])
        ) if metric_rows else 0.0
        aggregate[f"mean_assay_{family}_nested_sibling_fraction"] = float(
            np.mean([float(row.get("case_nested_response", {}).get(f"assay_{family}_nested_sibling_fraction", 0.0)) for row in metric_rows])
        ) if metric_rows else 0.0
        aggregate[f"mean_assay_{family}_child_response_score"] = float(
            np.mean([float(row.get("case_nested_response", {}).get(f"assay_{family}_mean_child_response_score", 0.0)) for row in metric_rows])
        ) if metric_rows else 0.0
        aggregate[f"mean_assay_{family}_branch_identity_qualified_carry"] = float(
            np.mean([float(row.get("case_nested_response", {}).get(f"assay_{family}_mean_branch_identity_qualified_carry", 0.0)) for row in metric_rows])
        ) if metric_rows else 0.0
        for metric in (
            "branch_identity_budget_retained",
            "branch_identity_support",
            "branch_identity_coherence",
            "readout_sibling_response",
            "coarse_env_corr",
            "fine_q_profile_corr",
            "nested_sibling_readiness",
            "max_nested_sibling_readiness",
            "continuation_write_enabled",
            "continuation_write_delta",
            "mode1_replace_enabled",
            "mode1_replace_ratio",
            "mode1_replace_gate",
            "mode1_replace_static_record",
            "mode1_replace_gate_floor",
            "mode1_replace_ratio_scale",
            "child_volume_active_count",
            "child_volume_effective_count",
            "child_volume_top_score_share",
            "child_fragment_count",
            "child_grandchild_count",
            "max_child_generation",
            "child_predictive_residual_reduction",
            "child_predictive_positive_residual_reduction",
            "child_predictive_boundary_match",
            "child_predictive_ontology_binding_mass",
            "child_predictive_applied_fraction",
            "child_predictive_target_residual_abs",
            "child_predictive_raw_delta_abs",
            "child_predictive_candidate_delta_abs",
            "child_predictive_oracle_residual_reduction",
            "max_child_predictive_oracle_residual_reduction",
            "child_predictive_learned_bridge_score",
            "child_predictive_learned_bridge_acceptance",
            "child_predictive_learned_bridge_boundary_score",
            "child_predictive_learned_bridge_volume_score",
            "child_predictive_learned_bridge_support_score",
            "child_predictive_learned_bridge_coherence_score",
            "child_predictive_learned_bridge_delta_scale",
            "mode1_replace_parent_ontology_used",
            "mode1_replace_parent_ontology_charge",
            "mode1_replace_parent_ontology_carrier_used",
            "mode1_replace_parent_ontology_mode1_occupancy_after",
        ):
            case_key = f"assay_{family}_mean_{metric}"
            if metric.startswith("max_"):
                case_key = f"assay_{family}_{metric}"
            aggregate[f"mean_assay_{family}_{metric}"] = float(
                np.mean(
                    [
                        float(row.get("case_nested_response", {}).get(case_key, 0.0))
                        for row in metric_rows
                    ]
                )
            ) if metric_rows else 0.0
    if metric_rows:
        aggregate["mean_child_response_score"] = float(np.mean([float(row.get("case_nested_response", {}).get("mean_child_response_score", 0.0)) for row in metric_rows]))
        aggregate["max_child_response_score"] = float(np.max([float(row.get("case_nested_response", {}).get("max_child_response_score", 0.0)) for row in metric_rows]))
        aggregate["mean_child_active_fraction"] = float(np.mean([float(row.get("case_nested_response", {}).get("mean_child_active_fraction", 0.0)) for row in metric_rows]))
        aggregate["mean_child_meso_response"] = float(np.mean([float(row.get("case_nested_response", {}).get("mean_child_meso_response", 0.0)) for row in metric_rows]))
        aggregate["mean_coarse_preservation"] = float(np.mean([float(row.get("case_nested_response", {}).get("mean_coarse_preservation", 0.0)) for row in metric_rows]))
        aggregate["mean_world_jump_penalty"] = float(np.mean([float(row.get("case_nested_response", {}).get("mean_world_jump_penalty", 0.0)) for row in metric_rows]))
        aggregate["mean_nested_sibling_fraction"] = float(np.mean([float(row.get("case_nested_response", {}).get("nested_sibling_fraction", 0.0)) for row in metric_rows]))
        aggregate["mean_over_rigid_fraction"] = float(np.mean([float(row.get("case_nested_response", {}).get("over_rigid_fraction", 0.0)) for row in metric_rows]))
        aggregate["mean_child_survival_signal"] = float(np.mean([float(row.get("case_nested_response", {}).get("mean_child_survival_signal", 0.0)) for row in metric_rows]))
        aggregate["mean_readout_sibling_response"] = float(np.mean([float(row.get("case_nested_response", {}).get("mean_readout_sibling_response", 0.0)) for row in metric_rows]))
        aggregate["mean_direct_readout_dependency"] = float(np.mean([float(row.get("case_nested_response", {}).get("direct_readout_dependency", 0.0)) for row in metric_rows]))
        aggregate["mean_final_direct_mix_shortcut_score"] = float(np.mean([float(row.get("case_nested_response", {}).get("final_direct_mix_shortcut_score", 0.0)) for row in metric_rows]))
        aggregate["mean_child_record_survival_score"] = float(np.mean([float(row.get("case_nested_response", {}).get("child_record_survival_score", 0.0)) for row in metric_rows]))
        aggregate["mean_parent_mode_conversion_sibling_fraction"] = float(np.mean([float(row.get("case_nested_response", {}).get("parent_mode_conversion_sibling_fraction", 0.0)) for row in metric_rows]))
        aggregate["mean_mode_replace_conversion_score"] = float(np.mean([float(row.get("case_nested_response", {}).get("mode_replace_conversion_score", 0.0)) for row in metric_rows]))
        aggregate["mean_child_volume_active_count"] = float(np.mean([float(row.get("case_nested_response", {}).get("mean_child_volume_active_count", 0.0)) for row in metric_rows]))
        aggregate["mean_child_volume_effective_count"] = float(np.mean([float(row.get("case_nested_response", {}).get("mean_child_volume_effective_count", 0.0)) for row in metric_rows]))
        aggregate["mean_child_volume_top_score_share"] = float(np.mean([float(row.get("case_nested_response", {}).get("mean_child_volume_top_score_share", 0.0)) for row in metric_rows]))
        aggregate["mean_assay_child_fragment_count"] = float(np.mean([float(row.get("case_nested_response", {}).get("mean_assay_child_fragment_count", 0.0)) for row in metric_rows]))
        aggregate["max_assay_child_fragment_count"] = float(np.max([float(row.get("case_nested_response", {}).get("max_assay_child_fragment_count", 0.0)) for row in metric_rows]))
        aggregate["mean_assay_grandchild_count"] = float(np.mean([float(row.get("case_nested_response", {}).get("mean_assay_grandchild_count", 0.0)) for row in metric_rows]))
        aggregate["max_assay_grandchild_count"] = float(np.max([float(row.get("case_nested_response", {}).get("max_assay_grandchild_count", 0.0)) for row in metric_rows]))
        aggregate["max_assay_child_generation"] = float(np.max([float(row.get("case_nested_response", {}).get("max_assay_child_generation", 0.0)) for row in metric_rows]))
        aggregate["mean_child_predictive_residual_reduction"] = float(np.mean([float(row.get("case_nested_response", {}).get("mean_child_predictive_residual_reduction", 0.0)) for row in metric_rows]))
        aggregate["max_child_predictive_residual_reduction"] = float(np.max([float(row.get("case_nested_response", {}).get("max_child_predictive_residual_reduction", 0.0)) for row in metric_rows]))
        aggregate["mean_child_predictive_positive_residual_reduction"] = float(np.mean([float(row.get("case_nested_response", {}).get("mean_child_predictive_positive_residual_reduction", 0.0)) for row in metric_rows]))
        aggregate["mean_child_predictive_boundary_match"] = float(np.mean([float(row.get("case_nested_response", {}).get("mean_child_predictive_boundary_match", 0.0)) for row in metric_rows]))
        aggregate["mean_child_predictive_explained_parent_defect"] = float(np.mean([float(row.get("case_nested_response", {}).get("mean_child_predictive_explained_parent_defect", 0.0)) for row in metric_rows]))
        aggregate["mean_child_predictive_ontology_binding_mass"] = float(np.mean([float(row.get("case_nested_response", {}).get("mean_child_predictive_ontology_binding_mass", 0.0)) for row in metric_rows]))
        aggregate["mean_child_predictive_applied_fraction"] = float(np.mean([float(row.get("case_nested_response", {}).get("mean_child_predictive_applied_fraction", 0.0)) for row in metric_rows]))
        aggregate["mean_child_predictive_target_residual_abs"] = float(np.mean([float(row.get("case_nested_response", {}).get("mean_child_predictive_target_residual_abs", 0.0)) for row in metric_rows]))
        aggregate["mean_child_predictive_raw_delta_abs"] = float(np.mean([float(row.get("case_nested_response", {}).get("mean_child_predictive_raw_delta_abs", 0.0)) for row in metric_rows]))
        aggregate["mean_child_predictive_candidate_delta_abs"] = float(np.mean([float(row.get("case_nested_response", {}).get("mean_child_predictive_candidate_delta_abs", 0.0)) for row in metric_rows]))
        aggregate["mean_child_predictive_oracle_residual_reduction"] = float(np.mean([float(row.get("case_nested_response", {}).get("mean_child_predictive_oracle_residual_reduction", 0.0)) for row in metric_rows]))
        aggregate["max_child_predictive_oracle_residual_reduction"] = float(np.max([float(row.get("case_nested_response", {}).get("max_child_predictive_oracle_residual_reduction", 0.0)) for row in metric_rows]))
        aggregate["mean_child_predictive_learned_bridge_score"] = float(np.mean([float(row.get("case_nested_response", {}).get("mean_child_predictive_learned_bridge_score", 0.0)) for row in metric_rows]))
        aggregate["mean_child_predictive_learned_bridge_acceptance"] = float(np.mean([float(row.get("case_nested_response", {}).get("mean_child_predictive_learned_bridge_acceptance", 0.0)) for row in metric_rows]))
        aggregate["mean_child_predictive_learned_bridge_authority_acceptance"] = float(np.mean([float(row.get("case_nested_response", {}).get("mean_child_predictive_learned_bridge_authority_acceptance", 0.0)) for row in metric_rows]))
        aggregate["mean_child_predictive_learned_bridge_boundary_score"] = float(np.mean([float(row.get("case_nested_response", {}).get("mean_child_predictive_learned_bridge_boundary_score", 0.0)) for row in metric_rows]))
        aggregate["mean_child_predictive_learned_bridge_volume_score"] = float(np.mean([float(row.get("case_nested_response", {}).get("mean_child_predictive_learned_bridge_volume_score", 0.0)) for row in metric_rows]))
        aggregate["mean_child_predictive_learned_bridge_support_score"] = float(np.mean([float(row.get("case_nested_response", {}).get("mean_child_predictive_learned_bridge_support_score", 0.0)) for row in metric_rows]))
        aggregate["mean_child_predictive_learned_bridge_coherence_score"] = float(np.mean([float(row.get("case_nested_response", {}).get("mean_child_predictive_learned_bridge_coherence_score", 0.0)) for row in metric_rows]))
        aggregate["mean_child_predictive_learned_bridge_delta_scale"] = float(np.mean([float(row.get("case_nested_response", {}).get("mean_child_predictive_learned_bridge_delta_scale", 0.0)) for row in metric_rows]))
        aggregate["mean_child_predictive_learned_bridge_prewrite_gate"] = float(np.mean([float(row.get("case_nested_response", {}).get("mean_child_predictive_learned_bridge_prewrite_gate", 0.0)) for row in metric_rows]))
        aggregate["mean_child_predictive_learned_bridge_effective_delta_scale"] = float(np.mean([float(row.get("case_nested_response", {}).get("mean_child_predictive_learned_bridge_effective_delta_scale", 0.0)) for row in metric_rows]))
        aggregate["mean_parent_ontology_charge"] = float(np.mean([float(row.get("case_nested_response", {}).get("mean_parent_ontology_charge", 0.0)) for row in metric_rows]))
        aggregate["max_parent_ontology_charge"] = float(np.max([float(row.get("case_nested_response", {}).get("max_parent_ontology_charge", 0.0)) for row in metric_rows]))
        aggregate["mean_parent_ontology_mode1_occupancy_after"] = float(np.mean([float(row.get("case_nested_response", {}).get("mean_parent_ontology_mode1_occupancy_after", 0.0)) for row in metric_rows]))
        aggregate["mean_branch_identity_carry"] = float(np.mean([float(row.get("case_nested_response", {}).get("mean_branch_identity_carry", 0.0)) for row in metric_rows]))
        aggregate["mean_branch_identity_qualified_carry"] = float(np.mean([float(row.get("case_nested_response", {}).get("mean_branch_identity_qualified_carry", 0.0)) for row in metric_rows]))
        aggregate["mean_max_branch_identity_carry"] = float(np.mean([float(row.get("case_nested_response", {}).get("max_branch_identity_carry", 0.0)) for row in metric_rows]))
        aggregate["mean_max_branch_identity_qualified_carry"] = float(np.mean([float(row.get("case_nested_response", {}).get("max_branch_identity_qualified_carry", 0.0)) for row in metric_rows]))
        aggregate["mean_max_branch_identity_qualified_steps"] = float(np.mean([float(row.get("case_nested_response", {}).get("max_branch_identity_qualified_steps", 0.0)) for row in metric_rows]))
        aggregate["mean_max_branch_identity_budget_retained"] = float(np.mean([float(row.get("case_nested_response", {}).get("max_branch_identity_budget_retained", 0.0)) for row in metric_rows]))
        aggregate["mean_max_branch_identity_parent_div"] = float(np.mean([float(row.get("case_nested_response", {}).get("max_branch_identity_parent_div", 0.0)) for row in metric_rows]))
        aggregate["mean_max_branch_identity_sibling_div"] = float(np.mean([float(row.get("case_nested_response", {}).get("max_branch_identity_sibling_div", 0.0)) for row in metric_rows]))
        aggregate["max_branch_identity_qualified_carry"] = float(np.max([float(row.get("case_nested_response", {}).get("max_branch_identity_qualified_carry", 0.0)) for row in metric_rows]))
        aggregate["max_branch_identity_qualified_steps"] = float(np.max([float(row.get("case_nested_response", {}).get("max_branch_identity_qualified_steps", 0.0)) for row in metric_rows]))
        aggregate["mean_best_identity_support_mean"] = float(np.mean([float(row.get("case_nested_response", {}).get("best_identity_support_mean", 0.0)) for row in metric_rows]))
        aggregate["mean_best_identity_coherence_mean"] = float(np.mean([float(row.get("case_nested_response", {}).get("best_identity_coherence_mean", 0.0)) for row in metric_rows]))
        aggregate["mean_best_identity_min_support"] = float(np.mean([float(row.get("case_nested_response", {}).get("best_identity_min_support", 0.0)) for row in metric_rows]))
        aggregate["mean_best_identity_min_coherence"] = float(np.mean([float(row.get("case_nested_response", {}).get("best_identity_min_coherence", 0.0)) for row in metric_rows]))
        aggregate["mean_branch_identity_budget_retained"] = float(np.mean([float(row.get("case_nested_response", {}).get("mean_branch_identity_budget_retained", 0.0)) for row in metric_rows]))
        aggregate["mean_branch_identity_parent_div"] = float(np.mean([float(row.get("case_nested_response", {}).get("mean_branch_identity_parent_div", 0.0)) for row in metric_rows]))
        aggregate["mean_branch_identity_sibling_div"] = float(np.mean([float(row.get("case_nested_response", {}).get("mean_branch_identity_sibling_div", 0.0)) for row in metric_rows]))
        aggregate["mean_readout_branch_count"] = float(np.mean([float(row.get("case_nested_response", {}).get("readout_branch_count", 0.0)) for row in metric_rows]))
        aggregate["mean_continuity_recurrence_score"] = float(np.mean([float(row.get("case_nested_response", {}).get("mean_continuity_recurrence_score", 0.0)) for row in metric_rows]))
        aggregate["max_continuity_recurrence_score"] = float(np.max([float(row.get("case_nested_response", {}).get("max_continuity_recurrence_score", 0.0)) for row in metric_rows]))
        aggregate["mean_continuity_adjacent_chunk_similarity"] = float(np.mean([float(row.get("case_nested_response", {}).get("mean_continuity_adjacent_chunk_similarity", 0.0)) for row in metric_rows]))
        aggregate["mean_continuity_nonlocal_chunk_repeat"] = float(np.mean([float(row.get("case_nested_response", {}).get("mean_continuity_nonlocal_chunk_repeat", 0.0)) for row in metric_rows]))
        aggregate["mean_continuity_first_chunk_reentry"] = float(np.mean([float(row.get("case_nested_response", {}).get("mean_continuity_first_chunk_reentry", 0.0)) for row in metric_rows]))
        aggregate["mean_continuity_loop_autocorr_peak"] = float(np.mean([float(row.get("case_nested_response", {}).get("mean_continuity_loop_autocorr_peak", 0.0)) for row in metric_rows]))
        aggregate["mean_continuity_delta_magnitude"] = float(np.mean([float(row.get("case_nested_response", {}).get("mean_continuity_delta_magnitude", 0.0)) for row in metric_rows]))
        aggregate["max_continuity_delta_magnitude"] = float(np.max([float(row.get("case_nested_response", {}).get("max_continuity_delta_magnitude", 0.0)) for row in metric_rows]))
    else:
        aggregate["mean_child_response_score"] = 0.0
        aggregate["max_child_response_score"] = 0.0
        aggregate["mean_child_active_fraction"] = 0.0
        aggregate["mean_child_meso_response"] = 0.0
        aggregate["mean_coarse_preservation"] = 0.0
        aggregate["mean_world_jump_penalty"] = 0.0
        aggregate["mean_nested_sibling_fraction"] = 0.0
        aggregate["mean_over_rigid_fraction"] = 0.0
        aggregate["mean_child_survival_signal"] = 0.0
        aggregate["mean_readout_sibling_response"] = 0.0
        aggregate["mean_direct_readout_dependency"] = 0.0
        aggregate["mean_final_direct_mix_shortcut_score"] = 0.0
        aggregate["mean_child_record_survival_score"] = 0.0
        aggregate["mean_parent_mode_conversion_sibling_fraction"] = 0.0
        aggregate["mean_mode_replace_conversion_score"] = 0.0
        aggregate["mean_child_volume_active_count"] = 0.0
        aggregate["mean_child_volume_effective_count"] = 0.0
        aggregate["mean_child_volume_top_score_share"] = 0.0
        aggregate["mean_assay_child_fragment_count"] = 0.0
        aggregate["max_assay_child_fragment_count"] = 0.0
        aggregate["mean_assay_grandchild_count"] = 0.0
        aggregate["max_assay_grandchild_count"] = 0.0
        aggregate["max_assay_child_generation"] = 0.0
        aggregate["mean_child_predictive_residual_reduction"] = 0.0
        aggregate["max_child_predictive_residual_reduction"] = 0.0
        aggregate["mean_child_predictive_positive_residual_reduction"] = 0.0
        aggregate["mean_child_predictive_boundary_match"] = 0.0
        aggregate["mean_child_predictive_explained_parent_defect"] = 0.0
        aggregate["mean_child_predictive_ontology_binding_mass"] = 0.0
        aggregate["mean_child_predictive_applied_fraction"] = 0.0
        aggregate["mean_child_predictive_target_residual_abs"] = 0.0
        aggregate["mean_child_predictive_raw_delta_abs"] = 0.0
        aggregate["mean_child_predictive_candidate_delta_abs"] = 0.0
        aggregate["mean_child_predictive_oracle_residual_reduction"] = 0.0
        aggregate["max_child_predictive_oracle_residual_reduction"] = 0.0
        aggregate["mean_child_predictive_learned_bridge_score"] = 0.0
        aggregate["mean_child_predictive_learned_bridge_acceptance"] = 0.0
        aggregate["mean_child_predictive_learned_bridge_boundary_score"] = 0.0
        aggregate["mean_child_predictive_learned_bridge_volume_score"] = 0.0
        aggregate["mean_child_predictive_learned_bridge_support_score"] = 0.0
        aggregate["mean_child_predictive_learned_bridge_coherence_score"] = 0.0
        aggregate["mean_child_predictive_learned_bridge_delta_scale"] = 0.0
        aggregate["mean_parent_ontology_charge"] = 0.0
        aggregate["max_parent_ontology_charge"] = 0.0
        aggregate["mean_parent_ontology_mode1_occupancy_after"] = 0.0
        aggregate["mean_branch_identity_carry"] = 0.0
        aggregate["mean_branch_identity_qualified_carry"] = 0.0
        aggregate["mean_max_branch_identity_carry"] = 0.0
        aggregate["mean_max_branch_identity_qualified_carry"] = 0.0
        aggregate["mean_max_branch_identity_qualified_steps"] = 0.0
        aggregate["mean_max_branch_identity_budget_retained"] = 0.0
        aggregate["mean_max_branch_identity_parent_div"] = 0.0
        aggregate["mean_max_branch_identity_sibling_div"] = 0.0
        aggregate["max_branch_identity_qualified_carry"] = 0.0
        aggregate["max_branch_identity_qualified_steps"] = 0.0
        aggregate["mean_best_identity_support_mean"] = 0.0
        aggregate["mean_best_identity_coherence_mean"] = 0.0
        aggregate["mean_best_identity_min_support"] = 0.0
        aggregate["mean_best_identity_min_coherence"] = 0.0
        aggregate["mean_branch_identity_budget_retained"] = 0.0
        aggregate["mean_branch_identity_parent_div"] = 0.0
        aggregate["mean_branch_identity_sibling_div"] = 0.0
        aggregate["mean_readout_branch_count"] = 0.0
        aggregate["mean_continuity_recurrence_score"] = 0.0
        aggregate["max_continuity_recurrence_score"] = 0.0
        aggregate["mean_continuity_adjacent_chunk_similarity"] = 0.0
        aggregate["mean_continuity_nonlocal_chunk_repeat"] = 0.0
        aggregate["mean_continuity_first_chunk_reentry"] = 0.0
        aggregate["mean_continuity_loop_autocorr_peak"] = 0.0
        aggregate["mean_continuity_delta_magnitude"] = 0.0
        aggregate["max_continuity_delta_magnitude"] = 0.0

    all_law_objects = [
        obj
        for row in rows
        for obj in list(row.get("resonant_law_objects", []))
        if isinstance(obj, dict)
    ]
    aggregate_retrieval = run_retrieval_assay(all_law_objects)
    aggregate_composition = composition_summary(all_law_objects)
    aggregate["resonant_law_object_count"] = int(len(all_law_objects))
    aggregate["resonant_retrieval"] = aggregate_retrieval["summary"]
    aggregate["resonant_retrieval_rows"] = aggregate_retrieval["rows"]
    aggregate["resonant_composition"] = aggregate_composition

    (out_dir / "nested_commitment_report.json").write_text(json.dumps(aggregate, indent=2), encoding="utf-8")
    return aggregate


def main() -> None:
    ap = argparse.ArgumentParser(description="Run a Circleworld nested-commitment / fork-resume test.")
    ap.add_argument("--config", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--depth", type=int, default=3)
    ap.add_argument("--fork-depth", type=int, default=1)
    ap.add_argument("--fork-selector", choices=["fixed", "first_live_child", "max_live_child"], default="fixed")
    ap.add_argument("--live-child-threshold", type=float, default=0.05)
    ap.add_argument("--mode", default="active_packets", choices=["no_promotion", "passive_packets", "active_packets", "native_multimode", "native_multimode_childworld"])
    ap.add_argument("--case-json", default=None, help="Optional JSON list of reference-wav or seeded nested-case specs")
    ap.add_argument("--assay-killswitch-json", default=None, help="Optional JSON object/path for assay-only causality kill-switch flags.")
    args = ap.parse_args()

    device = _safe_device(args.device)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.case_json:
        raw_cases = json.loads(Path(args.case_json).read_text(encoding="utf-8-sig"))
        if isinstance(raw_cases, dict):
            cases = [{"name": str(name), "reference_wav": str(path)} for name, path in raw_cases.items()]
        else:
            cases = raw_cases
    else:
        cases = _default_seeded_nested_cases()
    assay_killswitch: dict[str, Any] = {}
    if args.assay_killswitch_json:
        raw = str(args.assay_killswitch_json)
        maybe_path = Path(raw)
        if maybe_path.exists():
            loaded = json.loads(maybe_path.read_text(encoding="utf-8-sig"))
        else:
            loaded = json.loads(raw)
        if not isinstance(loaded, dict):
            raise ValueError("--assay-killswitch-json must resolve to a JSON object")
        assay_killswitch = {str(k): v for k, v in loaded.items()}

    aggregate = evaluate_nested_commitment_report(
        config_path=Path(args.config),
        out_dir=out_dir,
        device_name=args.device,
        depth=int(args.depth),
        fork_depth=int(args.fork_depth),
        fork_selector=str(args.fork_selector),
        live_child_threshold=float(args.live_child_threshold),
        mode=args.mode,
        cases=cases,
        assay_killswitch=assay_killswitch,
    )
    print(json.dumps(aggregate, indent=2))


if __name__ == "__main__":
    main()
