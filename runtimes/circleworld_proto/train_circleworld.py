from __future__ import annotations

import argparse
import json
import math
import random
import sys
import wave
from copy import deepcopy
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

from ablate_formalization import _generate_seed_phase, _load_cfg, make_seed_rafa
from circleworld import CircleworldConfig, circleworld_loss, recurse_circleworld, summarize_circleworld_run
from lib_blackwell import NakedRAFA
from config import load_config
from stft_utils import compute_stft
from rafa_math_tools import phase_to_phasor


def _set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _safe_device(requested: str) -> torch.device:
    if requested == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


_CHILD_WRITEBACK_CONTROL_DEFAULTS: dict[str, float] = {
    "child_operator_seed_gain": 2.40,
    "child_operator_promotability_gain": 0.60,
    "child_writeback_phase_delta_gain": 1.00,
    "child_writeback_operator_mix": 0.25,
    "child_writeback_phase_floor_target": 0.30,
    "child_writeback_phase_floor_threshold_mult": 2.50,
    "child_writeback_phase_floor_gain": 2.10,
    "child_writeback_parent_mix_cap": 0.04,
    "child_support_writeback_gain": 1.00,
    "child_support_writeback_floor_gain": 0.35,
}

_CHILD_WRITEBACK_CONTROL_STDS: dict[str, float] = {
    "child_operator_seed_gain": 0.30,
    "child_operator_promotability_gain": 0.12,
    "child_writeback_phase_delta_gain": 0.12,
    "child_writeback_operator_mix": 0.05,
    "child_writeback_phase_floor_target": 0.04,
    "child_writeback_phase_floor_threshold_mult": 0.30,
    "child_writeback_phase_floor_gain": 0.25,
    "child_writeback_parent_mix_cap": 0.01,
    "child_support_writeback_gain": 0.12,
    "child_support_writeback_floor_gain": 0.05,
}

_CHILD_WRITEBACK_CONTROL_BOUNDS: dict[str, tuple[float, float]] = {
    "child_operator_seed_gain": (0.0, 6.0),
    "child_operator_promotability_gain": (0.0, 3.0),
    "child_writeback_phase_delta_gain": (0.0, 3.0),
    "child_writeback_operator_mix": (0.0, 2.0),
    "child_writeback_phase_floor_target": (0.0, 0.75),
    "child_writeback_phase_floor_threshold_mult": (0.5, 6.0),
    "child_writeback_phase_floor_gain": (0.0, 6.0),
    "child_writeback_parent_mix_cap": (0.0, 0.25),
    "child_support_writeback_gain": (0.0, 3.0),
    "child_support_writeback_floor_gain": (0.0, 1.0),
}


_CHILD_LOCAL_COHERENCE_CONTROL_DEFAULTS: dict[str, float] = {
    "child_local_coherence_retention_mix": 0.0,
    "child_local_coherence_floor": 0.0,
    "child_local_coherence_floor_support": 0.18,
    "child_local_coherence_causal_min_delta": 0.02,
    "child_local_coherence_causal_full_delta": 0.18,
    "child_local_coherence_causal_gate_floor": 0.0,
}

_CHILD_LOCAL_COHERENCE_CONTROL_STDS: dict[str, float] = {
    "child_local_coherence_retention_mix": 0.08,
    "child_local_coherence_floor": 0.04,
    "child_local_coherence_floor_support": 0.02,
    "child_local_coherence_causal_min_delta": 0.01,
    "child_local_coherence_causal_full_delta": 0.04,
    "child_local_coherence_causal_gate_floor": 0.05,
}

_CHILD_LOCAL_COHERENCE_CONTROL_BOUNDS: dict[str, tuple[float, float]] = {
    "child_local_coherence_retention_mix": (0.0, 1.0),
    "child_local_coherence_floor": (0.0, 1.0),
    "child_local_coherence_floor_support": (0.0, 1.0),
    "child_local_coherence_causal_min_delta": (0.0, 1.0),
    "child_local_coherence_causal_full_delta": (0.001, 2.0),
    "child_local_coherence_causal_gate_floor": (0.0, 1.0),
}


_PHASE_LAW_CONTROL_DEFAULTS: dict[str, float] = {
    "phase_law_precondition_gain": 0.0,
    "phase_law_velocity_mix": 0.0,
    "phase_law_stability_gain": 1.0,
    "phase_law_softclip": 0.0,
    "phase_law_low_rank": 0.0,
    "phase_law_consensus_mix": 0.0,
    "phase_law_consensus_damping": 0.0,
    "phase_law_median_guard": 0.0,
    "phase_law_local_velocity_mix": 0.0,
    "phase_law_local_coherence_damping": 0.0,
    "phase_law_curvature_guard": 0.0,
    "phase_law_reentry_mix": 0.0,
    "phase_law_reentry_accel_mix": 0.0,
}

_PHASE_LAW_CONTROL_BOUNDS: dict[str, tuple[float, float]] = {
    "phase_law_precondition_gain": (0.0, 0.75),
    "phase_law_velocity_mix": (0.0, 0.20),
    "phase_law_stability_gain": (0.50, 3.00),
    "phase_law_softclip": (0.0, 1.00),
    "phase_law_low_rank": (0.0, 32.0),
    "phase_law_consensus_mix": (0.0, 0.12),
    "phase_law_consensus_damping": (0.0, 0.80),
    "phase_law_median_guard": (0.0, 0.80),
    "phase_law_local_velocity_mix": (0.0, 0.16),
    "phase_law_local_coherence_damping": (0.0, 0.80),
    "phase_law_curvature_guard": (0.0, 0.80),
    "phase_law_reentry_mix": (0.0, 0.16),
    "phase_law_reentry_accel_mix": (0.0, 1.25),
}

_PHASE_LAW_CONTROL_STDS: dict[str, float] = {
    "phase_law_precondition_gain": 0.08,
    "phase_law_velocity_mix": 0.03,
    "phase_law_stability_gain": 0.25,
    "phase_law_softclip": 0.08,
    "phase_law_low_rank": 3.0,
    "phase_law_consensus_mix": 0.02,
    "phase_law_consensus_damping": 0.12,
    "phase_law_median_guard": 0.12,
    "phase_law_local_velocity_mix": 0.03,
    "phase_law_local_coherence_damping": 0.12,
    "phase_law_curvature_guard": 0.12,
    "phase_law_reentry_mix": 0.03,
    "phase_law_reentry_accel_mix": 0.20,
}


def _phase_law_control_kwargs(src: dict[str, Any], fallback: CircleworldConfig | None = None) -> dict[str, float | int | bool]:
    return {
        "phase_law_precondition_gain": float(
            src.get("phase_law_precondition_gain", getattr(fallback, "phase_law_precondition_gain", 0.0))
        ),
        "phase_law_velocity_mix": float(src.get("phase_law_velocity_mix", getattr(fallback, "phase_law_velocity_mix", 0.0))),
        "phase_law_stability_gain": float(
            src.get("phase_law_stability_gain", getattr(fallback, "phase_law_stability_gain", 1.0))
        ),
        "phase_law_softclip": float(src.get("phase_law_softclip", getattr(fallback, "phase_law_softclip", 0.0))),
        "phase_law_low_rank": int(round(float(src.get("phase_law_low_rank", getattr(fallback, "phase_law_low_rank", 0))))),
        "phase_law_consensus_mix": float(
            src.get("phase_law_consensus_mix", getattr(fallback, "phase_law_consensus_mix", 0.0))
        ),
        "phase_law_consensus_damping": float(
            src.get("phase_law_consensus_damping", getattr(fallback, "phase_law_consensus_damping", 0.0))
        ),
        "phase_law_median_guard": float(
            src.get("phase_law_median_guard", getattr(fallback, "phase_law_median_guard", 0.0))
        ),
        "phase_law_local_velocity_mix": float(
            src.get("phase_law_local_velocity_mix", getattr(fallback, "phase_law_local_velocity_mix", 0.0))
        ),
        "phase_law_local_coherence_damping": float(
            src.get(
                "phase_law_local_coherence_damping",
                getattr(fallback, "phase_law_local_coherence_damping", 0.0),
            )
        ),
        "phase_law_curvature_guard": float(
            src.get("phase_law_curvature_guard", getattr(fallback, "phase_law_curvature_guard", 0.0))
        ),
        "phase_law_reentry_mix": float(
            src.get("phase_law_reentry_mix", getattr(fallback, "phase_law_reentry_mix", 0.0))
        ),
        "phase_law_reentry_accel_mix": float(
            src.get("phase_law_reentry_accel_mix", getattr(fallback, "phase_law_reentry_accel_mix", 0.0))
        ),
        "phase_law_reentry_causal": bool(
            src.get("phase_law_reentry_causal", getattr(fallback, "phase_law_reentry_causal", False))
        ),
    }


def _phase_law_control_values(cfg: CircleworldConfig) -> dict[str, float | int | bool]:
    return {
        "phase_law_precondition_gain": float(cfg.phase_law_precondition_gain),
        "phase_law_velocity_mix": float(cfg.phase_law_velocity_mix),
        "phase_law_stability_gain": float(cfg.phase_law_stability_gain),
        "phase_law_softclip": float(cfg.phase_law_softclip),
        "phase_law_low_rank": int(cfg.phase_law_low_rank),
        "phase_law_consensus_mix": float(cfg.phase_law_consensus_mix),
        "phase_law_consensus_damping": float(cfg.phase_law_consensus_damping),
        "phase_law_median_guard": float(cfg.phase_law_median_guard),
        "phase_law_local_velocity_mix": float(cfg.phase_law_local_velocity_mix),
        "phase_law_local_coherence_damping": float(cfg.phase_law_local_coherence_damping),
        "phase_law_curvature_guard": float(cfg.phase_law_curvature_guard),
        "phase_law_reentry_mix": float(cfg.phase_law_reentry_mix),
        "phase_law_reentry_accel_mix": float(cfg.phase_law_reentry_accel_mix),
        "phase_law_reentry_causal": bool(cfg.phase_law_reentry_causal),
    }


def _materialize_phase_law_control_kwargs(vec: dict[str, Any], base: CircleworldConfig) -> dict[str, float | int]:
    out: dict[str, float | int] = {}
    for key, default in _PHASE_LAW_CONTROL_DEFAULTS.items():
        lo, hi = _PHASE_LAW_CONTROL_BOUNDS[key]
        raw = float(vec.get(key, getattr(base, key, default)))
        value = min(hi, max(lo, raw))
        out[key] = int(round(value)) if key == "phase_law_low_rank" else float(value)
    return out


def _child_writeback_control_kwargs(src: dict[str, Any], fallback: CircleworldConfig | None = None) -> dict[str, float]:
    return {
        key: float(src.get(key, getattr(fallback, key, default) if fallback is not None else default))
        for key, default in _CHILD_WRITEBACK_CONTROL_DEFAULTS.items()
    }


def _materialize_child_writeback_control_kwargs(vec: dict[str, Any], base: CircleworldConfig) -> dict[str, float]:
    out: dict[str, float] = {}
    for key, default in _CHILD_WRITEBACK_CONTROL_DEFAULTS.items():
        lo, hi = _CHILD_WRITEBACK_CONTROL_BOUNDS[key]
        raw = float(vec.get(key, getattr(base, key, default)))
        out[key] = float(min(hi, max(lo, raw)))
    return out


def _child_writeback_control_values(cfg: CircleworldConfig) -> dict[str, float]:
    return {key: float(getattr(cfg, key, default)) for key, default in _CHILD_WRITEBACK_CONTROL_DEFAULTS.items()}


def _child_local_coherence_kwargs(src: dict[str, Any], fallback: CircleworldConfig | None = None) -> dict[str, float | bool]:
    out: dict[str, float | bool] = {
        "child_local_coherence_retention_enabled": bool(
            src.get(
                "child_local_coherence_retention_enabled",
                getattr(fallback, "child_local_coherence_retention_enabled", False),
            )
        ),
        "child_local_coherence_causal_gate_enabled": bool(
            src.get(
                "child_local_coherence_causal_gate_enabled",
                getattr(fallback, "child_local_coherence_causal_gate_enabled", False),
            )
        ),
    }
    for key, default in _CHILD_LOCAL_COHERENCE_CONTROL_DEFAULTS.items():
        out[key] = float(src.get(key, getattr(fallback, key, default) if fallback is not None else default))
    return out


def _materialize_child_local_coherence_kwargs(vec: dict[str, Any], base: CircleworldConfig) -> dict[str, float | bool]:
    out: dict[str, float | bool] = {
        "child_local_coherence_retention_enabled": bool(
            vec.get("child_local_coherence_retention_enabled", base.child_local_coherence_retention_enabled)
        ),
        "child_local_coherence_causal_gate_enabled": bool(
            vec.get("child_local_coherence_causal_gate_enabled", base.child_local_coherence_causal_gate_enabled)
        ),
    }
    for key, default in _CHILD_LOCAL_COHERENCE_CONTROL_DEFAULTS.items():
        lo, hi = _CHILD_LOCAL_COHERENCE_CONTROL_BOUNDS[key]
        raw = float(vec.get(key, getattr(base, key, default)))
        out[key] = float(min(hi, max(lo, raw)))
    return out


def _child_local_coherence_values(cfg: CircleworldConfig) -> dict[str, float | bool]:
    out: dict[str, float | bool] = {
        "child_local_coherence_retention_enabled": bool(getattr(cfg, "child_local_coherence_retention_enabled", False)),
        "child_local_coherence_causal_gate_enabled": bool(
            getattr(cfg, "child_local_coherence_causal_gate_enabled", False)
        ),
    }
    for key, default in _CHILD_LOCAL_COHERENCE_CONTROL_DEFAULTS.items():
        out[key] = float(getattr(cfg, key, default))
    return out


def _default_circle_cfg() -> CircleworldConfig:
    raw = _load_cfg().get("circleworld", {})
    return CircleworldConfig(
        qset=tuple(raw.get("qset", (2, 3, 4, 5, 6, 8, 12))),
        q_weights=tuple(raw.get("q_weights", (1.0, 1.0, 0.9, 0.8, 0.7, 0.6, 0.5))),
        promotion_threshold=float(raw.get("promotion_threshold", 0.58)),
        max_promotions=int(raw.get("max_promotions", 4)),
        recursion_depth=int(raw.get("recursion_depth", 2)),
        residue_scale=float(raw.get("residue_scale", 0.75)),
        child_law_gain=float(raw.get("child_law_gain", 0.25)),
        attack_window=int(raw.get("attack_window", 8)),
        persistence_momentum=float(raw.get("persistence_momentum", 0.6)),
        soft_matryoshka_enabled=bool(raw.get("soft_matryoshka_enabled", False)),
        matryoshka_rank=int(raw.get("matryoshka_rank", 24)),
        **_phase_law_control_kwargs(raw),
        prefix_fracs=tuple(raw.get("prefix_fracs", (0.125, 0.25, 0.5))),
        slow_persistence=float(raw.get("slow_persistence", 0.96)),
        fast_persistence=float(raw.get("fast_persistence", 0.22)),
        persistence_curve=float(raw.get("persistence_curve", 1.8)),
        slow_write_scale=float(raw.get("slow_write_scale", 0.08)),
        fast_write_scale=float(raw.get("fast_write_scale", 1.0)),
        write_curve=float(raw.get("write_curve", 1.4)),
        prefix_coarse_weight=float(raw.get("prefix_coarse_weight", 0.40)),
        prefix_mid_weight=float(raw.get("prefix_mid_weight", 0.25)),
        branching_mode=str(raw.get("branching_mode", "single_path")),
        num_modes=int(raw.get("num_modes", 2)),
        readout_mode=str(raw.get("readout_mode", "weighted_mixture")),
        branch_law_version=str(raw.get("branch_law_version", "parametric_v1")),
        q_trace_rank=int(raw.get("q_trace_rank", 4)),
        dormant_logit=float(raw.get("dormant_logit", -6.0)),
        dormant_support=float(raw.get("dormant_support", 0.02)),
        dormant_q_scale=float(raw.get("dormant_q_scale", 0.02)),
        split_pressure=float(raw.get("split_pressure", 0.24)),
        split_seed_scale=float(raw.get("split_seed_scale", 0.10)),
        split_support_gain=float(raw.get("split_support_gain", 0.45)),
        merge_pressure=float(raw.get("merge_pressure", 0.16)),
        merge_phase_tol=float(raw.get("merge_phase_tol", 0.82)),
        merge_support_overlap_weight=float(raw.get("merge_support_overlap_weight", 0.50)),
        survival_coherence_weight=float(raw.get("survival_coherence_weight", 0.40)),
        survival_arc_weight=float(raw.get("survival_arc_weight", 0.55)),
        survival_qtrace_weight=float(raw.get("survival_qtrace_weight", 0.25)),
        survival_residue_penalty=float(raw.get("survival_residue_penalty", 0.30)),
        collapse_sharpness=float(raw.get("collapse_sharpness", 1.25)),
        support_decay=float(raw.get("support_decay", 0.10)),
        support_spread=int(raw.get("support_spread", 5)),
        support_overlap_penalty=float(raw.get("support_overlap_penalty", 0.15)),
        anti_fixation_weight=float(raw.get("anti_fixation_weight", 0.20)),
        readout_temperature=float(raw.get("readout_temperature", 0.85)),
        slot2_support_threshold=float(raw.get("slot2_support_threshold", 0.10)),
        real_branch_threshold=float(raw.get("real_branch_threshold", 0.12)),
        child_branch_parent_threshold=float(raw.get("child_branch_parent_threshold", raw.get("real_branch_threshold", 0.12))),
        child_branch_writeback_threshold=float(raw.get("child_branch_writeback_threshold", 0.0)),
        child_branch_meso_threshold=float(raw.get("child_branch_meso_threshold", 0.0)),
        child_branch_live_threshold=float(raw.get("child_branch_live_threshold", 0.0)),
        mode_perturb_window_frac=float(raw.get("mode_perturb_window_frac", 0.18)),
        qtrace_momentum=float(raw.get("qtrace_momentum", 0.85)),
        mask_neighborhood=int(raw.get("mask_neighborhood", 3)),
        topology_mask_gain=float(raw.get("topology_mask_gain", 0.55)),
        complexity_mask_gain=float(raw.get("complexity_mask_gain", 0.60)),
        context_mask_gain=float(raw.get("context_mask_gain", 0.45)),
        contrastive_mask_gain=float(raw.get("contrastive_mask_gain", 0.70)),
        aux_mask_suppression=float(raw.get("aux_mask_suppression", 0.35)),
        instability_mask_gain=float(raw.get("instability_mask_gain", 0.75)),
        defect_phase_gain=float(raw.get("defect_phase_gain", 0.40)),
        defect_q_gain=float(raw.get("defect_q_gain", 0.30)),
        defect_residue_gain=float(raw.get("defect_residue_gain", 0.15)),
        defect_sharpness_gain=float(raw.get("defect_sharpness_gain", 0.15)),
        defect_world_grad_gain=float(raw.get("defect_world_grad_gain", 0.20)),
        instability_seed_scale=float(raw.get("instability_seed_scale", 0.18)),
        instability_support_gain=float(raw.get("instability_support_gain", 0.28)),
        instability_logit_gain=float(raw.get("instability_logit_gain", 0.22)),
        branch_kernel_version=str(raw.get("branch_kernel_version", "ramanujan")),
        relation_attention_gain=float(raw.get("relation_attention_gain", 0.65)),
        relation_attention_sharpness=float(raw.get("relation_attention_sharpness", 1.10)),
        relation_value_gain=float(raw.get("relation_value_gain", 0.30)),
        relation_support_gain=float(raw.get("relation_support_gain", 0.22)),
        relation_logit_gain=float(raw.get("relation_logit_gain", 0.24)),
        relation_qtrace_gain=float(raw.get("relation_qtrace_gain", 0.14)),
        relation_residual_mix=float(raw.get("relation_residual_mix", 0.60)),
        child_spawn_threshold=float(raw.get("child_spawn_threshold", 0.34)),
        child_max_worlds=int(raw.get("child_max_worlds", 4)),
        child_min_age_for_writeback=int(raw.get("child_min_age_for_writeback", 2)),
        child_support_window=int(raw.get("child_support_window", 12)),
        child_support_decay=float(raw.get("child_support_decay", 0.12)),
        child_survival_coherence_weight=float(raw.get("child_survival_coherence_weight", 0.42)),
        child_survival_qtrace_weight=float(raw.get("child_survival_qtrace_weight", 0.24)),
        child_survival_residue_penalty=float(raw.get("child_survival_residue_penalty", 0.22)),
        child_writeback_gain=float(raw.get("child_writeback_gain", 0.32)),
        child_writeback_rank=int(raw.get("child_writeback_rank", 12)),
        child_writeback_temperature=float(raw.get("child_writeback_temperature", 0.85)),
        child_writeback_budget=float(raw.get("child_writeback_budget", 1.25)),
        child_parent_mix=float(raw.get("child_parent_mix", 0.15)),
        child_parent_mix_early=float(raw.get("child_parent_mix_early", 0.08)),
        **_child_writeback_control_kwargs(raw),
        child_kill_threshold=float(raw.get("child_kill_threshold", 0.08)),
        child_local_ifs_enabled=bool(raw.get("child_local_ifs_enabled", False)),
        child_local_steps=int(raw.get("child_local_steps", 1)),
        child_local_support_only=bool(raw.get("child_local_support_only", True)),
        **_child_local_coherence_kwargs(raw),
        law_packet_merge_threshold=float(raw.get("law_packet_merge_threshold", 0.92)),
        law_packet_min_score=float(raw.get("law_packet_min_score", 0.25)),
        law_packet_topk_families=int(raw.get("law_packet_topk_families", 8)),
    )


def _load_circle_cfg(path: Path) -> CircleworldConfig:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    cfg = payload["config"] if "config" in payload else payload
    return CircleworldConfig(
        qset=tuple(cfg.get("qset", (2, 3, 4, 5, 6, 8, 12))),
        q_weights=tuple(cfg["q_weights"]),
        promotion_threshold=float(cfg["promotion_threshold"]),
        max_promotions=int(cfg["max_promotions"]),
        recursion_depth=int(cfg["recursion_depth"]),
        residue_scale=float(cfg.get("residue_scale", 0.75)),
        child_law_gain=float(cfg["child_law_gain"]),
        attack_window=int(cfg["attack_window"]),
        persistence_momentum=float(cfg["persistence_momentum"]),
        soft_matryoshka_enabled=bool(cfg.get("soft_matryoshka_enabled", False)),
        matryoshka_rank=int(cfg.get("matryoshka_rank", 24)),
        **_phase_law_control_kwargs(cfg),
        prefix_fracs=tuple(cfg.get("prefix_fracs", (0.125, 0.25, 0.5))),
        slow_persistence=float(cfg.get("slow_persistence", 0.96)),
        fast_persistence=float(cfg.get("fast_persistence", 0.22)),
        persistence_curve=float(cfg.get("persistence_curve", 1.8)),
        slow_write_scale=float(cfg.get("slow_write_scale", 0.08)),
        fast_write_scale=float(cfg.get("fast_write_scale", 1.0)),
        write_curve=float(cfg.get("write_curve", 1.4)),
        prefix_coarse_weight=float(cfg.get("prefix_coarse_weight", 0.40)),
        prefix_mid_weight=float(cfg.get("prefix_mid_weight", 0.25)),
        branching_mode=str(cfg.get("branching_mode", "single_path")),
        num_modes=int(cfg.get("num_modes", 2)),
        readout_mode=str(cfg.get("readout_mode", "weighted_mixture")),
        branch_law_version=str(cfg.get("branch_law_version", "parametric_v1")),
        q_trace_rank=int(cfg.get("q_trace_rank", 4)),
        dormant_logit=float(cfg.get("dormant_logit", -6.0)),
        dormant_support=float(cfg.get("dormant_support", 0.02)),
        dormant_q_scale=float(cfg.get("dormant_q_scale", 0.02)),
        split_pressure=float(cfg.get("split_pressure", 0.24)),
        split_seed_scale=float(cfg.get("split_seed_scale", 0.10)),
        split_support_gain=float(cfg.get("split_support_gain", 0.45)),
        merge_pressure=float(cfg.get("merge_pressure", 0.16)),
        merge_phase_tol=float(cfg.get("merge_phase_tol", 0.82)),
        merge_support_overlap_weight=float(cfg.get("merge_support_overlap_weight", 0.50)),
        survival_coherence_weight=float(cfg.get("survival_coherence_weight", 0.40)),
        survival_arc_weight=float(cfg.get("survival_arc_weight", 0.55)),
        survival_qtrace_weight=float(cfg.get("survival_qtrace_weight", 0.25)),
        survival_residue_penalty=float(cfg.get("survival_residue_penalty", 0.30)),
        collapse_sharpness=float(cfg.get("collapse_sharpness", 1.25)),
        support_decay=float(cfg.get("support_decay", 0.10)),
        support_spread=int(cfg.get("support_spread", 5)),
        support_overlap_penalty=float(cfg.get("support_overlap_penalty", 0.15)),
        anti_fixation_weight=float(cfg.get("anti_fixation_weight", 0.20)),
        readout_temperature=float(cfg.get("readout_temperature", 0.85)),
        slot2_support_threshold=float(cfg.get("slot2_support_threshold", 0.10)),
        real_branch_threshold=float(cfg.get("real_branch_threshold", 0.12)),
        child_branch_parent_threshold=float(cfg.get("child_branch_parent_threshold", cfg.get("real_branch_threshold", 0.12))),
        child_branch_writeback_threshold=float(cfg.get("child_branch_writeback_threshold", 0.0)),
        child_branch_meso_threshold=float(cfg.get("child_branch_meso_threshold", 0.0)),
        child_branch_live_threshold=float(cfg.get("child_branch_live_threshold", 0.0)),
        mode_perturb_window_frac=float(cfg.get("mode_perturb_window_frac", 0.18)),
        qtrace_momentum=float(cfg.get("qtrace_momentum", 0.85)),
        mask_neighborhood=int(cfg.get("mask_neighborhood", 3)),
        topology_mask_gain=float(cfg.get("topology_mask_gain", 0.55)),
        complexity_mask_gain=float(cfg.get("complexity_mask_gain", 0.60)),
        context_mask_gain=float(cfg.get("context_mask_gain", 0.45)),
        contrastive_mask_gain=float(cfg.get("contrastive_mask_gain", 0.70)),
        aux_mask_suppression=float(cfg.get("aux_mask_suppression", 0.35)),
        instability_mask_gain=float(cfg.get("instability_mask_gain", 0.75)),
        defect_phase_gain=float(cfg.get("defect_phase_gain", 0.40)),
        defect_q_gain=float(cfg.get("defect_q_gain", 0.30)),
        defect_residue_gain=float(cfg.get("defect_residue_gain", 0.15)),
        defect_sharpness_gain=float(cfg.get("defect_sharpness_gain", 0.15)),
        defect_world_grad_gain=float(cfg.get("defect_world_grad_gain", 0.20)),
        instability_seed_scale=float(cfg.get("instability_seed_scale", 0.18)),
        instability_support_gain=float(cfg.get("instability_support_gain", 0.28)),
        instability_logit_gain=float(cfg.get("instability_logit_gain", 0.22)),
        branch_kernel_version=str(cfg.get("branch_kernel_version", "ramanujan")),
        relation_attention_gain=float(cfg.get("relation_attention_gain", 0.65)),
        relation_attention_sharpness=float(cfg.get("relation_attention_sharpness", 1.10)),
        relation_value_gain=float(cfg.get("relation_value_gain", 0.30)),
        relation_support_gain=float(cfg.get("relation_support_gain", 0.22)),
        relation_logit_gain=float(cfg.get("relation_logit_gain", 0.24)),
        relation_qtrace_gain=float(cfg.get("relation_qtrace_gain", 0.14)),
        relation_residual_mix=float(cfg.get("relation_residual_mix", 0.60)),
        child_spawn_threshold=float(cfg.get("child_spawn_threshold", 0.34)),
        child_max_worlds=int(cfg.get("child_max_worlds", 4)),
        child_min_age_for_writeback=int(cfg.get("child_min_age_for_writeback", 2)),
        child_support_window=int(cfg.get("child_support_window", 12)),
        child_support_decay=float(cfg.get("child_support_decay", 0.12)),
        child_survival_coherence_weight=float(cfg.get("child_survival_coherence_weight", 0.42)),
        child_survival_qtrace_weight=float(cfg.get("child_survival_qtrace_weight", 0.24)),
        child_survival_residue_penalty=float(cfg.get("child_survival_residue_penalty", 0.22)),
        child_writeback_gain=float(cfg.get("child_writeback_gain", 0.32)),
        child_writeback_rank=int(cfg.get("child_writeback_rank", 12)),
        child_writeback_temperature=float(cfg.get("child_writeback_temperature", 0.85)),
        child_writeback_budget=float(cfg.get("child_writeback_budget", 1.25)),
        child_parent_mix=float(cfg.get("child_parent_mix", 0.15)),
        child_parent_mix_early=float(cfg.get("child_parent_mix_early", 0.08)),
        **_child_writeback_control_kwargs(cfg),
        child_kill_threshold=float(cfg.get("child_kill_threshold", 0.08)),
        child_local_ifs_enabled=bool(cfg.get("child_local_ifs_enabled", False)),
        child_local_steps=int(cfg.get("child_local_steps", 1)),
        child_local_support_only=bool(cfg.get("child_local_support_only", True)),
        **_child_local_coherence_kwargs(cfg),
        law_packet_merge_threshold=float(cfg.get("law_packet_merge_threshold", 0.92)),
        law_packet_min_score=float(cfg.get("law_packet_min_score", 0.25)),
        law_packet_topk_families=int(cfg.get("law_packet_topk_families", 8)),
    )


def _materialize_cfg(base: CircleworldConfig, vec: dict[str, Any]) -> CircleworldConfig:
    support_spread = int(min(15, max(1, round(vec.get("support_spread", base.support_spread)))))
    if support_spread % 2 == 0:
        support_spread = min(15, support_spread + 1)
    mask_neighborhood = int(min(7, max(1, round(vec.get("mask_neighborhood", base.mask_neighborhood)))))
    if mask_neighborhood % 2 == 0:
        mask_neighborhood = min(7, mask_neighborhood + 1)
    return CircleworldConfig(
        qset=base.qset,
        q_weights=tuple(float(max(0.1, x)) for x in vec["q_weights"]),
        promotion_threshold=float(min(0.9, max(0.25, vec["promotion_threshold"]))),
        max_promotions=int(min(8, max(1, round(vec["max_promotions"])))),
        recursion_depth=base.recursion_depth,
        residue_scale=base.residue_scale,
        child_law_gain=float(min(1.0, max(0.01, vec["child_law_gain"]))),
        attack_window=int(min(24, max(2, round(vec["attack_window"])))),
        persistence_momentum=float(min(0.98, max(0.05, vec["persistence_momentum"]))),
        soft_matryoshka_enabled=bool(vec.get("soft_matryoshka_enabled", base.soft_matryoshka_enabled)),
        matryoshka_rank=int(min(64, max(4, round(vec.get("matryoshka_rank", base.matryoshka_rank))))),
        **_materialize_phase_law_control_kwargs(vec, base),
        prefix_fracs=tuple(base.prefix_fracs),
        slow_persistence=float(min(0.995, max(0.70, vec.get("slow_persistence", base.slow_persistence)))),
        fast_persistence=float(min(0.80, max(0.02, vec.get("fast_persistence", base.fast_persistence)))),
        persistence_curve=float(min(3.5, max(0.5, vec.get("persistence_curve", base.persistence_curve)))),
        slow_write_scale=float(min(0.60, max(0.01, vec.get("slow_write_scale", base.slow_write_scale)))),
        fast_write_scale=float(min(2.5, max(0.20, vec.get("fast_write_scale", base.fast_write_scale)))),
        write_curve=float(min(3.5, max(0.5, vec.get("write_curve", base.write_curve)))),
        prefix_coarse_weight=float(min(0.8, max(0.05, vec.get("prefix_coarse_weight", base.prefix_coarse_weight)))),
        prefix_mid_weight=float(min(0.8, max(0.05, vec.get("prefix_mid_weight", base.prefix_mid_weight)))),
        branching_mode=str(vec.get("branching_mode", base.branching_mode)),
        num_modes=int(base.num_modes),
        readout_mode=str(base.readout_mode),
        branch_law_version=str(vec.get("branch_law_version", base.branch_law_version)),
        q_trace_rank=int(base.q_trace_rank),
        dormant_logit=float(base.dormant_logit),
        dormant_support=float(base.dormant_support),
        dormant_q_scale=float(base.dormant_q_scale),
        split_pressure=float(min(1.2, max(0.01, vec.get("split_pressure", base.split_pressure)))),
        split_seed_scale=float(min(0.75, max(0.01, vec.get("split_seed_scale", base.split_seed_scale)))),
        split_support_gain=float(min(1.5, max(0.01, vec.get("split_support_gain", base.split_support_gain)))),
        merge_pressure=float(min(1.0, max(0.01, vec.get("merge_pressure", base.merge_pressure)))),
        merge_phase_tol=float(min(0.98, max(0.40, vec.get("merge_phase_tol", base.merge_phase_tol)))),
        merge_support_overlap_weight=float(min(2.0, max(0.05, vec.get("merge_support_overlap_weight", base.merge_support_overlap_weight)))),
        survival_coherence_weight=float(min(1.5, max(0.05, vec.get("survival_coherence_weight", base.survival_coherence_weight)))),
        survival_arc_weight=float(min(1.5, max(0.05, vec.get("survival_arc_weight", base.survival_arc_weight)))),
        survival_qtrace_weight=float(min(1.5, max(0.01, vec.get("survival_qtrace_weight", base.survival_qtrace_weight)))),
        survival_residue_penalty=float(min(1.5, max(0.01, vec.get("survival_residue_penalty", base.survival_residue_penalty)))),
        collapse_sharpness=float(min(4.0, max(0.05, vec.get("collapse_sharpness", base.collapse_sharpness)))),
        support_decay=float(min(0.60, max(0.01, vec.get("support_decay", base.support_decay)))),
        support_spread=support_spread,
        support_overlap_penalty=float(min(1.5, max(0.0, vec.get("support_overlap_penalty", base.support_overlap_penalty)))),
        anti_fixation_weight=float(min(1.5, max(0.0, vec.get("anti_fixation_weight", base.anti_fixation_weight)))),
        readout_temperature=float(min(2.0, max(0.10, vec.get("readout_temperature", base.readout_temperature)))),
        slot2_support_threshold=float(base.slot2_support_threshold),
        real_branch_threshold=float(base.real_branch_threshold),
        child_branch_parent_threshold=float(vec.get("child_branch_parent_threshold", base.child_branch_parent_threshold)),
        child_branch_writeback_threshold=float(vec.get("child_branch_writeback_threshold", base.child_branch_writeback_threshold)),
        child_branch_meso_threshold=float(vec.get("child_branch_meso_threshold", base.child_branch_meso_threshold)),
        child_branch_live_threshold=float(vec.get("child_branch_live_threshold", base.child_branch_live_threshold)),
        mode_perturb_window_frac=float(base.mode_perturb_window_frac),
        qtrace_momentum=float(min(0.99, max(0.40, vec.get("qtrace_momentum", base.qtrace_momentum)))),
        mask_neighborhood=mask_neighborhood,
        topology_mask_gain=float(min(1.0, max(0.0, vec.get("topology_mask_gain", base.topology_mask_gain)))),
        complexity_mask_gain=float(min(1.0, max(0.0, vec.get("complexity_mask_gain", base.complexity_mask_gain)))),
        context_mask_gain=float(min(1.0, max(0.0, vec.get("context_mask_gain", base.context_mask_gain)))),
        contrastive_mask_gain=float(min(1.0, max(0.0, vec.get("contrastive_mask_gain", base.contrastive_mask_gain)))),
        aux_mask_suppression=float(min(1.5, max(0.0, vec.get("aux_mask_suppression", base.aux_mask_suppression)))),
        instability_mask_gain=float(min(1.0, max(0.0, vec.get("instability_mask_gain", base.instability_mask_gain)))),
        defect_phase_gain=float(min(1.5, max(0.0, vec.get("defect_phase_gain", base.defect_phase_gain)))),
        defect_q_gain=float(min(1.5, max(0.0, vec.get("defect_q_gain", base.defect_q_gain)))),
        defect_residue_gain=float(min(1.5, max(0.0, vec.get("defect_residue_gain", base.defect_residue_gain)))),
        defect_sharpness_gain=float(min(1.5, max(0.0, vec.get("defect_sharpness_gain", base.defect_sharpness_gain)))),
        defect_world_grad_gain=float(min(1.5, max(0.0, vec.get("defect_world_grad_gain", base.defect_world_grad_gain)))),
        instability_seed_scale=float(min(0.75, max(0.0, vec.get("instability_seed_scale", base.instability_seed_scale)))),
        instability_support_gain=float(min(1.5, max(0.0, vec.get("instability_support_gain", base.instability_support_gain)))),
        instability_logit_gain=float(min(1.5, max(0.0, vec.get("instability_logit_gain", base.instability_logit_gain)))),
        branch_kernel_version=str(vec.get("branch_kernel_version", base.branch_kernel_version)),
        relation_attention_gain=float(min(1.5, max(0.0, vec.get("relation_attention_gain", base.relation_attention_gain)))),
        relation_attention_sharpness=float(min(3.0, max(0.05, vec.get("relation_attention_sharpness", base.relation_attention_sharpness)))),
        relation_value_gain=float(min(1.5, max(0.0, vec.get("relation_value_gain", base.relation_value_gain)))),
        relation_support_gain=float(min(1.5, max(0.0, vec.get("relation_support_gain", base.relation_support_gain)))),
        relation_logit_gain=float(min(1.5, max(0.0, vec.get("relation_logit_gain", base.relation_logit_gain)))),
        relation_qtrace_gain=float(min(1.5, max(0.0, vec.get("relation_qtrace_gain", base.relation_qtrace_gain)))),
        relation_residual_mix=float(min(1.0, max(0.0, vec.get("relation_residual_mix", base.relation_residual_mix)))),
        child_spawn_threshold=float(min(0.95, max(0.05, vec.get("child_spawn_threshold", base.child_spawn_threshold)))),
        child_max_worlds=int(min(8, max(1, round(vec.get("child_max_worlds", base.child_max_worlds))))),
        child_min_age_for_writeback=int(min(6, max(1, round(vec.get("child_min_age_for_writeback", base.child_min_age_for_writeback))))),
        child_support_window=int(min(32, max(4, round(vec.get("child_support_window", base.child_support_window))))),
        child_support_decay=float(min(0.5, max(0.01, vec.get("child_support_decay", base.child_support_decay)))),
        child_survival_coherence_weight=float(min(1.2, max(0.05, vec.get("child_survival_coherence_weight", base.child_survival_coherence_weight)))),
        child_survival_qtrace_weight=float(min(1.0, max(0.05, vec.get("child_survival_qtrace_weight", base.child_survival_qtrace_weight)))),
        child_survival_residue_penalty=float(min(1.0, max(0.01, vec.get("child_survival_residue_penalty", base.child_survival_residue_penalty)))),
        child_writeback_gain=float(min(1.2, max(0.01, vec.get("child_writeback_gain", base.child_writeback_gain)))),
        child_writeback_rank=int(min(32, max(4, round(vec.get("child_writeback_rank", base.child_writeback_rank))))),
        child_writeback_temperature=float(min(2.5, max(0.1, vec.get("child_writeback_temperature", base.child_writeback_temperature)))),
        child_writeback_budget=float(min(4.0, max(0.1, vec.get("child_writeback_budget", base.child_writeback_budget)))),
        child_parent_mix=float(min(0.5, max(0.02, vec.get("child_parent_mix", base.child_parent_mix)))),
        child_parent_mix_early=float(min(0.4, max(0.01, vec.get("child_parent_mix_early", base.child_parent_mix_early)))),
        **_materialize_child_writeback_control_kwargs(vec, base),
        child_kill_threshold=float(min(0.5, max(0.01, vec.get("child_kill_threshold", base.child_kill_threshold)))),
        child_local_ifs_enabled=bool(vec.get("child_local_ifs_enabled", base.child_local_ifs_enabled)),
        child_local_steps=int(min(8, max(1, round(vec.get("child_local_steps", base.child_local_steps))))),
        child_local_support_only=bool(vec.get("child_local_support_only", base.child_local_support_only)),
        **_materialize_child_local_coherence_kwargs(vec, base),
        law_packet_merge_threshold=float(min(0.995, max(0.70, vec.get("law_packet_merge_threshold", base.law_packet_merge_threshold)))),
        law_packet_min_score=float(min(0.95, max(0.05, vec.get("law_packet_min_score", base.law_packet_min_score)))),
        law_packet_topk_families=int(min(24, max(1, round(vec.get("law_packet_topk_families", base.law_packet_topk_families))))),
    )


def _candidate_from_mean_std(mean: dict[str, Any], std: dict[str, Any], rng: random.Random) -> dict[str, Any]:
    out = {
        "q_weights": [],
        "promotion_threshold": rng.gauss(mean["promotion_threshold"], std["promotion_threshold"]),
        "max_promotions": rng.gauss(mean["max_promotions"], std["max_promotions"]),
        "child_law_gain": rng.gauss(mean["child_law_gain"], std["child_law_gain"]),
        "attack_window": rng.gauss(mean["attack_window"], std["attack_window"]),
        "persistence_momentum": rng.gauss(mean["persistence_momentum"], std["persistence_momentum"]),
        "matryoshka_rank": rng.gauss(mean["matryoshka_rank"], std["matryoshka_rank"]),
        "slow_persistence": rng.gauss(mean["slow_persistence"], std["slow_persistence"]),
        "fast_persistence": rng.gauss(mean["fast_persistence"], std["fast_persistence"]),
        "persistence_curve": rng.gauss(mean["persistence_curve"], std["persistence_curve"]),
        "slow_write_scale": rng.gauss(mean["slow_write_scale"], std["slow_write_scale"]),
        "fast_write_scale": rng.gauss(mean["fast_write_scale"], std["fast_write_scale"]),
        "write_curve": rng.gauss(mean["write_curve"], std["write_curve"]),
        "prefix_coarse_weight": rng.gauss(mean["prefix_coarse_weight"], std["prefix_coarse_weight"]),
        "prefix_mid_weight": rng.gauss(mean["prefix_mid_weight"], std["prefix_mid_weight"]),
        "split_pressure": rng.gauss(mean["split_pressure"], std["split_pressure"]),
        "split_seed_scale": rng.gauss(mean["split_seed_scale"], std["split_seed_scale"]),
        "split_support_gain": rng.gauss(mean["split_support_gain"], std["split_support_gain"]),
        "merge_pressure": rng.gauss(mean["merge_pressure"], std["merge_pressure"]),
        "merge_phase_tol": rng.gauss(mean["merge_phase_tol"], std["merge_phase_tol"]),
        "merge_support_overlap_weight": rng.gauss(mean["merge_support_overlap_weight"], std["merge_support_overlap_weight"]),
        "survival_coherence_weight": rng.gauss(mean["survival_coherence_weight"], std["survival_coherence_weight"]),
        "survival_arc_weight": rng.gauss(mean["survival_arc_weight"], std["survival_arc_weight"]),
        "survival_qtrace_weight": rng.gauss(mean["survival_qtrace_weight"], std["survival_qtrace_weight"]),
        "survival_residue_penalty": rng.gauss(mean["survival_residue_penalty"], std["survival_residue_penalty"]),
        "collapse_sharpness": rng.gauss(mean["collapse_sharpness"], std["collapse_sharpness"]),
        "support_decay": rng.gauss(mean["support_decay"], std["support_decay"]),
        "support_spread": rng.gauss(mean["support_spread"], std["support_spread"]),
        "support_overlap_penalty": rng.gauss(mean["support_overlap_penalty"], std["support_overlap_penalty"]),
        "anti_fixation_weight": rng.gauss(mean["anti_fixation_weight"], std["anti_fixation_weight"]),
        "readout_temperature": rng.gauss(mean["readout_temperature"], std["readout_temperature"]),
        "qtrace_momentum": rng.gauss(mean["qtrace_momentum"], std["qtrace_momentum"]),
        "mask_neighborhood": rng.gauss(mean["mask_neighborhood"], std["mask_neighborhood"]),
        "topology_mask_gain": rng.gauss(mean["topology_mask_gain"], std["topology_mask_gain"]),
        "complexity_mask_gain": rng.gauss(mean["complexity_mask_gain"], std["complexity_mask_gain"]),
        "context_mask_gain": rng.gauss(mean["context_mask_gain"], std["context_mask_gain"]),
        "contrastive_mask_gain": rng.gauss(mean["contrastive_mask_gain"], std["contrastive_mask_gain"]),
        "aux_mask_suppression": rng.gauss(mean["aux_mask_suppression"], std["aux_mask_suppression"]),
        "instability_mask_gain": rng.gauss(mean["instability_mask_gain"], std["instability_mask_gain"]),
        "defect_phase_gain": rng.gauss(mean["defect_phase_gain"], std["defect_phase_gain"]),
        "defect_q_gain": rng.gauss(mean["defect_q_gain"], std["defect_q_gain"]),
        "defect_residue_gain": rng.gauss(mean["defect_residue_gain"], std["defect_residue_gain"]),
        "defect_sharpness_gain": rng.gauss(mean["defect_sharpness_gain"], std["defect_sharpness_gain"]),
        "defect_world_grad_gain": rng.gauss(mean["defect_world_grad_gain"], std["defect_world_grad_gain"]),
        "instability_seed_scale": rng.gauss(mean["instability_seed_scale"], std["instability_seed_scale"]),
        "instability_support_gain": rng.gauss(mean["instability_support_gain"], std["instability_support_gain"]),
        "instability_logit_gain": rng.gauss(mean["instability_logit_gain"], std["instability_logit_gain"]),
        "relation_attention_gain": rng.gauss(mean["relation_attention_gain"], std["relation_attention_gain"]),
        "relation_attention_sharpness": rng.gauss(mean["relation_attention_sharpness"], std["relation_attention_sharpness"]),
        "relation_value_gain": rng.gauss(mean["relation_value_gain"], std["relation_value_gain"]),
        "relation_support_gain": rng.gauss(mean["relation_support_gain"], std["relation_support_gain"]),
        "relation_logit_gain": rng.gauss(mean["relation_logit_gain"], std["relation_logit_gain"]),
        "relation_qtrace_gain": rng.gauss(mean["relation_qtrace_gain"], std["relation_qtrace_gain"]),
        "relation_residual_mix": rng.gauss(mean["relation_residual_mix"], std["relation_residual_mix"]),
        "child_spawn_threshold": rng.gauss(mean.get("child_spawn_threshold", 0.34), std.get("child_spawn_threshold", 0.03)),
        "child_max_worlds": rng.gauss(mean.get("child_max_worlds", 4.0), std.get("child_max_worlds", 0.5)),
        "child_min_age_for_writeback": rng.gauss(mean.get("child_min_age_for_writeback", 2.0), std.get("child_min_age_for_writeback", 0.4)),
        "child_support_window": rng.gauss(mean.get("child_support_window", 12.0), std.get("child_support_window", 1.5)),
        "child_support_decay": rng.gauss(mean.get("child_support_decay", 0.12), std.get("child_support_decay", 0.02)),
        "child_survival_coherence_weight": rng.gauss(mean.get("child_survival_coherence_weight", 0.42), std.get("child_survival_coherence_weight", 0.03)),
        "child_survival_qtrace_weight": rng.gauss(mean.get("child_survival_qtrace_weight", 0.24), std.get("child_survival_qtrace_weight", 0.03)),
        "child_survival_residue_penalty": rng.gauss(mean.get("child_survival_residue_penalty", 0.22), std.get("child_survival_residue_penalty", 0.03)),
        "child_writeback_gain": rng.gauss(mean.get("child_writeback_gain", 0.32), std.get("child_writeback_gain", 0.03)),
        "child_writeback_rank": rng.gauss(mean.get("child_writeback_rank", 12.0), std.get("child_writeback_rank", 1.5)),
        "child_writeback_temperature": rng.gauss(mean.get("child_writeback_temperature", 0.85), std.get("child_writeback_temperature", 0.03)),
        "child_writeback_budget": rng.gauss(mean.get("child_writeback_budget", 1.25), std.get("child_writeback_budget", 0.08)),
        "child_parent_mix": rng.gauss(mean.get("child_parent_mix", 0.15), std.get("child_parent_mix", 0.02)),
        "child_parent_mix_early": rng.gauss(mean.get("child_parent_mix_early", 0.08), std.get("child_parent_mix_early", 0.015)),
        "child_kill_threshold": rng.gauss(mean.get("child_kill_threshold", 0.08), std.get("child_kill_threshold", 0.02)),
        "child_local_ifs_enabled": bool(mean.get("child_local_ifs_enabled", False)),
        "child_local_steps": mean.get("child_local_steps", 1),
        "child_local_support_only": bool(mean.get("child_local_support_only", True)),
        "child_local_coherence_retention_enabled": bool(mean.get("child_local_coherence_retention_enabled", False)),
        "law_packet_merge_threshold": rng.gauss(mean["law_packet_merge_threshold"], std["law_packet_merge_threshold"]),
        "law_packet_min_score": rng.gauss(mean["law_packet_min_score"], std["law_packet_min_score"]),
        "law_packet_topk_families": rng.gauss(mean["law_packet_topk_families"], std["law_packet_topk_families"]),
        "soft_matryoshka_enabled": mean.get("soft_matryoshka_enabled", False),
        "branching_mode": mean.get("branching_mode", "single_path"),
        "branch_law_version": mean.get("branch_law_version", "parametric_v1"),
        "branch_kernel_version": mean.get("branch_kernel_version", "ramanujan"),
    }
    for key, default in _PHASE_LAW_CONTROL_DEFAULTS.items():
        out[key] = rng.gauss(mean.get(key, default), std.get(key, _PHASE_LAW_CONTROL_STDS[key]))
    for key, default in _CHILD_WRITEBACK_CONTROL_DEFAULTS.items():
        out[key] = rng.gauss(mean.get(key, default), std.get(key, _CHILD_WRITEBACK_CONTROL_STDS[key]))
    for key, default in _CHILD_LOCAL_COHERENCE_CONTROL_DEFAULTS.items():
        out[key] = rng.gauss(mean.get(key, default), std.get(key, _CHILD_LOCAL_COHERENCE_CONTROL_STDS[key]))
    for idx, val in enumerate(mean["q_weights"]):
        out["q_weights"].append(rng.gauss(val, std["q_weights"][idx]))
    return out


def _seed_plan(
    device: torch.device,
    train_count: int,
    val_count: int,
    include_naked_rafa: bool = True,
) -> tuple[list[tuple[str, int]], list[tuple[str, int]]]:
    train: list[tuple[str, int]] = [("synthetic", 1100 + i) for i in range(train_count)]
    val: list[tuple[str, int]] = [("synthetic", 2100 + i) for i in range(val_count)]
    if device.type == "cuda" and include_naked_rafa:
        train.extend([("naked_rafa", 3100 + i) for i in range(max(2, train_count // 2))])
        val.extend([("naked_rafa", 4100 + i) for i in range(max(2, val_count // 2))])
    return train, val


def _graduation_anchor_wavs() -> tuple[list[str], list[str]]:
    train = [
        "D:\\RAFA\\wav_files\\soundbible_muscle-car_9fce13702c.wav",
        "D:\\RAFA\\wav_files\\soundbible_steam-engine-running_68e20d27d1.wav",
        "D:\\RAFA\\wav_files\\soundbible_mystic-chanting-4_a38f2bbe68.wav",
        "D:\\RAFA\\wav_files\\soundbible_metal-clang_d06ac0429e.wav",
        "D:\\RAFA\\wav_files\\soundbible_spooky-drone_2efbfd965b.wav",
        "D:\\RAFA\\wav_files\\soundbible_airplane-takeoff_c752b8ba8b.wav",
    ]
    val = [
        "D:\\RAFA\\wav_files\\soundbible_muscle-car_2f21ca885f.wav",
        "D:\\RAFA\\wav_files\\soundbible_mystic-chanting-4_7962fb220f.wav",
        "D:\\RAFA\\wav_files\\soundbible_anvil-impact_c8021375ce.wav",
        "D:\\RAFA\\wav_files\\soundbible_spooky-drone_7cc45e3229.wav",
        "D:\\RAFA\\wav_files\\soundbible_airplane-landing-airport_caf74bfdeb.wav",
    ]
    return train, val


def _load_local_pcm_wav(path: Path) -> tuple[torch.Tensor, int]:
    with wave.open(str(path), "rb") as wf:
        sr = int(wf.getframerate())
        ch = int(wf.getnchannels())
        sampwidth = int(wf.getsampwidth())
        nframes = int(wf.getnframes())
        raw = wf.readframes(nframes)

    if sampwidth == 1:
        arr = np.frombuffer(raw, dtype=np.uint8).astype(np.float32)
        arr = (arr - 128.0) / 128.0
    elif sampwidth == 2:
        arr = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    elif sampwidth == 3:
        b = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 3)
        v = (
            b[:, 0].astype(np.int32)
            | (b[:, 1].astype(np.int32) << 8)
            | (b[:, 2].astype(np.int32) << 16)
        )
        sign = 1 << 23
        v = (v ^ sign) - sign
        arr = v.astype(np.float32) / 8388608.0
    elif sampwidth == 4:
        arr = np.frombuffer(raw, dtype=np.int32).astype(np.float32) / 2147483648.0
    else:
        raise RuntimeError(f"Unsupported WAV sample width: {sampwidth} bytes")

    arr = arr.reshape(-1, ch).T
    wav = torch.from_numpy(arr)
    return wav, sr


def _native_resample(wav: torch.Tensor, orig_sr: int, target_sr: int) -> torch.Tensor:
    if orig_sr == target_sr:
        return wav
    ratio = float(target_sr) / float(orig_sr)
    target_len = int(round(wav.size(1) * ratio))
    return F.interpolate(
        wav.unsqueeze(0),
        size=target_len,
        mode="linear",
        align_corners=False,
    ).squeeze(0)


def _crop_or_tile(wav: torch.Tensor, target_len: int) -> torch.Tensor:
    if wav.size(1) < target_len:
        reps = int(np.ceil(float(target_len) / float(max(1, wav.size(1)))))
        wav = wav.repeat(1, reps)
    return wav[:, :target_len]


def _phase_state_from_wav(
    wav_path: Path,
    target_time_steps: int,
    device: torch.device,
) -> torch.Tensor:
    cfg = load_config()
    sr = int(cfg["data"]["sample_rate"])
    stft_cfg = cfg["data"]["stft"]
    hop = int(stft_cfg["hop"])
    winl = int(stft_cfg["win_length"])
    target_len = hop * max(1, target_time_steps - 1) + winl

    wav, wav_sr = _load_local_pcm_wav(wav_path)
    wav = wav.float()
    if wav.size(0) > 1:
        wav = wav.mean(dim=0, keepdim=True)
    wav = _native_resample(wav, wav_sr, sr)
    wav = _crop_or_tile(wav, target_len).to(device)
    mag, phase = compute_stft(wav, stft_cfg)
    if phase.size(-1) > target_time_steps:
        phase = phase[..., :target_time_steps]
    elif phase.size(-1) < target_time_steps:
        pad = target_time_steps - phase.size(-1)
        phase = F.pad(phase, (0, pad), mode="replicate")
    return phase_to_phasor(phase).detach()


def _build_dataset(
    plan: list[tuple[str, int]],
    time_steps: int,
    device: torch.device,
    anchor_wavs: list[str] | None = None,
) -> list[dict[str, Any]]:
    dataset: list[dict[str, Any]] = []
    rafa_core: NakedRAFA | None = make_seed_rafa(dev=device.type) if any(src == "naked_rafa" for src, _ in plan) else None
    for source, seed in plan:
        _set_seed(seed)
        phase_state = _generate_seed_phase(
            rafa_core=rafa_core,
            batch_size=1,
            time_steps=time_steps,
            device=device,
            seed_source=source,
            seed=seed,
        ).detach()
        dataset.append(
            {
                "source": source,
                "seed": seed,
                "phase_state": phase_state,
            }
        )
    for wav_path in anchor_wavs or []:
        phase_state = _phase_state_from_wav(Path(wav_path), target_time_steps=time_steps, device=device)
        dataset.append(
            {
                "source": "graduation_anchor",
                "seed": int(abs(hash(wav_path)) % 10_000_000),
                "wav_path": str(wav_path),
                "phase_state": phase_state,
            }
        )
    return dataset


def _score_run(
    run: dict[str, Any],
    score_cfg: dict[str, float] | None = None,
) -> tuple[float, dict[str, float]]:
    summary = summarize_circleworld_run(run)
    losses = circleworld_loss(run)
    score_cfg = score_cfg or {}
    target_dom = float(score_cfg.get("target_dom", 0.70))
    target_entropy = float(score_cfg.get("target_entropy", 0.45))
    min_promotions = float(score_cfg.get("min_promotions", 3.0))
    w_major = float(score_cfg.get("w_major", 1.0))
    w_residue = float(score_cfg.get("w_residue", 1.0))
    w_promo = float(score_cfg.get("w_promo", 0.5))
    w_final_major = float(score_cfg.get("w_final_major", 0.25))
    w_final_minor = float(score_cfg.get("w_final_minor", 0.25))
    w_dom = float(score_cfg.get("w_dom", 0.40))
    w_entropy = float(score_cfg.get("w_entropy", 0.20))
    w_major_sat = float(score_cfg.get("w_major_sat", 0.50))
    w_loss = float(score_cfg.get("w_loss", 0.05))
    w_dom_target = float(score_cfg.get("w_dom_target", 0.0))
    w_entropy_target = float(score_cfg.get("w_entropy_target", 0.0))
    w_promote_floor = float(score_cfg.get("w_promote_floor", 0.0))
    w_rel_signature_count = float(score_cfg.get("w_rel_signature_count", 0.0))
    w_rel_signature_families = float(score_cfg.get("w_rel_signature_families", 0.0))
    w_rel_signature_confidence = float(score_cfg.get("w_rel_signature_confidence", 0.0))
    w_rel_signature_q_entropy = float(score_cfg.get("w_rel_signature_q_entropy", 0.0))
    w_rel_branch_mass = float(score_cfg.get("w_rel_branch_mass", 0.0))
    w_rel_dominant_family = float(score_cfg.get("w_rel_dominant_family", 0.0))
    target_rel_dominant_family_share = float(score_cfg.get("target_rel_dominant_family_share", 1.0))
    dom_target_err = abs(summary["dominant_q_share"] - target_dom)
    entropy_target_err = abs(summary["q_entropy"] - target_entropy)
    promote_floor = max(0.0, min_promotions - summary["num_promotions"])
    score = (
        w_major * summary["major_gain"]
        + w_residue * summary["residue_drop"]
        + w_promo * summary["promotability_gain"]
        + w_final_major * summary["final_major_mass"]
        - w_final_minor * summary["final_minor_residue"]
        - w_dom * summary["dominant_q_share"]
        + w_entropy * summary["q_entropy"]
        - w_major_sat * summary["major_saturation"]
        - w_loss * float(losses["loss"].item())
        - w_dom_target * dom_target_err
        - w_entropy_target * entropy_target_err
        - w_promote_floor * promote_floor
        + w_rel_signature_count * float(summary.get("num_relational_signatures", 0.0))
        + w_rel_signature_families * float(summary.get("num_relational_signature_families", 0.0))
        + w_rel_signature_confidence * float(summary.get("mean_relational_signature_confidence", 0.0))
        + w_rel_signature_q_entropy * float(summary.get("mean_relational_signature_q_entropy", 0.0))
        + w_rel_branch_mass * float(summary.get("mean_relational_branch_mass", 0.0))
        - w_rel_dominant_family
        * max(0.0, float(summary.get("dominant_relational_family_share", 0.0)) - target_rel_dominant_family_share)
    )
    metrics = {
        **summary,
        "loss": float(losses["loss"].item()),
        "l_q_dom": float(losses["l_q_dom"].item()),
        "l_q_entropy": float(losses["l_q_entropy"].item()),
        "l_major_sat": float(losses["l_major_sat"].item()),
        "dom_target_err": float(dom_target_err),
        "entropy_target_err": float(entropy_target_err),
        "promote_floor_penalty": float(promote_floor),
        "score": float(score),
    }
    return float(score), metrics


def _evaluate_cfg(
    cfg: CircleworldConfig,
    dataset: list[dict[str, Any]],
    mode: str,
    score_cfg: dict[str, float] | None = None,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    run_mode = cfg.branching_mode if cfg.branching_mode in {"native_multimode", "native_multimode_childworld"} else mode
    for item in dataset:
        run = recurse_circleworld(item["phase_state"], cfg=cfg, depth=cfg.recursion_depth, mode=run_mode)
        score, metrics = _score_run(run, score_cfg=score_cfg)
        rows.append(
            {
                "source": item["source"],
                "seed": int(item["seed"]),
                "score": score,
                **metrics,
            }
        )

    def _mean(key: str) -> float:
        return float(sum(row[key] for row in rows) / max(1, len(rows)))

    return {
        "num_samples": len(rows),
        "mean_score": _mean("score"),
        "mean_loss": _mean("loss"),
        "mean_major_gain": _mean("major_gain"),
        "mean_residue_drop": _mean("residue_drop"),
        "mean_promotability_gain": _mean("promotability_gain"),
        "mean_final_major_mass": _mean("final_major_mass"),
        "mean_final_minor_residue": _mean("final_minor_residue"),
        "mean_dominant_q_share": _mean("dominant_q_share"),
        "mean_q_entropy": _mean("q_entropy"),
        "mean_major_saturation": _mean("major_saturation"),
        "mean_l_q_dom": _mean("l_q_dom"),
        "mean_l_q_entropy": _mean("l_q_entropy"),
        "mean_l_major_sat": _mean("l_major_sat"),
        "mean_dom_target_err": _mean("dom_target_err"),
        "mean_entropy_target_err": _mean("entropy_target_err"),
        "mean_promote_floor_penalty": _mean("promote_floor_penalty"),
        "mean_num_promotions": _mean("num_promotions"),
        "mean_slot2_live_fraction": _mean("mean_slot2_live_fraction") if rows and "mean_slot2_live_fraction" in rows[0] else 0.0,
        "mean_real_branch_fraction": _mean("real_branch_fraction") if rows and "real_branch_fraction" in rows[0] else 0.0,
        "mean_meso_branch_effect": _mean("meso_branch_effect") if rows and "meso_branch_effect" in rows[0] else 0.0,
        "mean_silent_singlepath_fraction": _mean("silent_singlepath_fraction") if rows and "silent_singlepath_fraction" in rows[0] else 0.0,
        "mean_branch_positive_mask": _mean("mean_branch_positive_mask") if rows and "mean_branch_positive_mask" in rows[0] else 0.0,
        "mean_branch_negative_mask": _mean("mean_branch_negative_mask") if rows and "mean_branch_negative_mask" in rows[0] else 0.0,
        "mean_decorative_slot2_fraction": _mean("mean_decorative_slot2_fraction") if rows and "mean_decorative_slot2_fraction" in rows[0] else 0.0,
        "mean_num_relational_signatures": _mean("num_relational_signatures") if rows and "num_relational_signatures" in rows[0] else 0.0,
        "mean_num_relational_signature_families": _mean("num_relational_signature_families") if rows and "num_relational_signature_families" in rows[0] else 0.0,
        "mean_relational_signature_confidence": _mean("mean_relational_signature_confidence") if rows and "mean_relational_signature_confidence" in rows[0] else 0.0,
        "mean_relational_signature_q_entropy": _mean("mean_relational_signature_q_entropy") if rows and "mean_relational_signature_q_entropy" in rows[0] else 0.0,
        "mean_relational_branch_mass": _mean("mean_relational_branch_mass") if rows and "mean_relational_branch_mass" in rows[0] else 0.0,
        "mean_dominant_relational_family_share": _mean("dominant_relational_family_share") if rows and "dominant_relational_family_share" in rows[0] else 0.0,
        "rows": rows,
    }


def _baseline_report(
    base_cfg: CircleworldConfig,
    train_set: list[dict[str, Any]],
    val_set: list[dict[str, Any]],
    score_cfg: dict[str, float] | None = None,
) -> dict[str, Any]:
    report: dict[str, Any] = {}
    for mode in ("no_promotion", "passive_packets", "active_packets"):
        report[mode] = {
            "train": _evaluate_cfg(base_cfg, train_set, mode=mode, score_cfg=score_cfg),
            "val": _evaluate_cfg(base_cfg, val_set, mode=mode, score_cfg=score_cfg),
        }
    return report


def train_circleworld(
    out_dir: Path,
    checkpoint_dir: Path,
    iterations: int,
    population: int,
    elite_count: int,
    time_steps: int,
    recursion_depth: int,
    device_name: str,
    train_count: int,
    val_count: int,
    seed: int,
    init_config_path: Path | None = None,
    score_cfg: dict[str, float] | None = None,
    include_graduation_anchors: bool = False,
    include_naked_rafa: bool = True,
) -> dict[str, Any]:
    device = _safe_device(device_name)
    _set_seed(seed)
    base_cfg = _load_circle_cfg(init_config_path) if init_config_path else _default_circle_cfg()
    base_cfg.recursion_depth = recursion_depth

    train_plan, val_plan = _seed_plan(
        device,
        train_count=train_count,
        val_count=val_count,
        include_naked_rafa=include_naked_rafa,
    )
    grad_train_wavs, grad_val_wavs = _graduation_anchor_wavs() if include_graduation_anchors else ([], [])
    train_set = _build_dataset(train_plan, time_steps=time_steps, device=device, anchor_wavs=grad_train_wavs)
    val_set = _build_dataset(val_plan, time_steps=time_steps, device=device, anchor_wavs=grad_val_wavs)

    baseline = _baseline_report(base_cfg, train_set, val_set, score_cfg=score_cfg)

    mean = {
        "q_weights": list(base_cfg.q_weights),
        "promotion_threshold": float(base_cfg.promotion_threshold),
        "max_promotions": float(base_cfg.max_promotions),
        "child_law_gain": float(base_cfg.child_law_gain),
        "attack_window": float(base_cfg.attack_window),
        "persistence_momentum": float(base_cfg.persistence_momentum),
        "matryoshka_rank": float(base_cfg.matryoshka_rank),
        "slow_persistence": float(base_cfg.slow_persistence),
        "fast_persistence": float(base_cfg.fast_persistence),
        "persistence_curve": float(base_cfg.persistence_curve),
        "slow_write_scale": float(base_cfg.slow_write_scale),
        "fast_write_scale": float(base_cfg.fast_write_scale),
        "write_curve": float(base_cfg.write_curve),
        "prefix_coarse_weight": float(base_cfg.prefix_coarse_weight),
        "prefix_mid_weight": float(base_cfg.prefix_mid_weight),
        "split_pressure": float(base_cfg.split_pressure),
        "split_seed_scale": float(base_cfg.split_seed_scale),
        "split_support_gain": float(base_cfg.split_support_gain),
        "merge_pressure": float(base_cfg.merge_pressure),
        "merge_phase_tol": float(base_cfg.merge_phase_tol),
        "merge_support_overlap_weight": float(base_cfg.merge_support_overlap_weight),
        "survival_coherence_weight": float(base_cfg.survival_coherence_weight),
        "survival_arc_weight": float(base_cfg.survival_arc_weight),
        "survival_qtrace_weight": float(base_cfg.survival_qtrace_weight),
        "survival_residue_penalty": float(base_cfg.survival_residue_penalty),
        "collapse_sharpness": float(base_cfg.collapse_sharpness),
        "support_decay": float(base_cfg.support_decay),
        "support_spread": float(base_cfg.support_spread),
        "support_overlap_penalty": float(base_cfg.support_overlap_penalty),
        "anti_fixation_weight": float(base_cfg.anti_fixation_weight),
        "readout_temperature": float(base_cfg.readout_temperature),
        "qtrace_momentum": float(base_cfg.qtrace_momentum),
        "mask_neighborhood": float(base_cfg.mask_neighborhood),
        "topology_mask_gain": float(base_cfg.topology_mask_gain),
        "complexity_mask_gain": float(base_cfg.complexity_mask_gain),
        "context_mask_gain": float(base_cfg.context_mask_gain),
        "contrastive_mask_gain": float(base_cfg.contrastive_mask_gain),
        "aux_mask_suppression": float(base_cfg.aux_mask_suppression),
        "instability_mask_gain": float(base_cfg.instability_mask_gain),
        "defect_phase_gain": float(base_cfg.defect_phase_gain),
        "defect_q_gain": float(base_cfg.defect_q_gain),
        "defect_residue_gain": float(base_cfg.defect_residue_gain),
        "defect_sharpness_gain": float(base_cfg.defect_sharpness_gain),
        "defect_world_grad_gain": float(base_cfg.defect_world_grad_gain),
        "instability_seed_scale": float(base_cfg.instability_seed_scale),
        "instability_support_gain": float(base_cfg.instability_support_gain),
        "instability_logit_gain": float(base_cfg.instability_logit_gain),
        "relation_attention_gain": float(base_cfg.relation_attention_gain),
        "relation_attention_sharpness": float(base_cfg.relation_attention_sharpness),
        "relation_value_gain": float(base_cfg.relation_value_gain),
        "relation_support_gain": float(base_cfg.relation_support_gain),
        "relation_logit_gain": float(base_cfg.relation_logit_gain),
        "relation_qtrace_gain": float(base_cfg.relation_qtrace_gain),
        "relation_residual_mix": float(base_cfg.relation_residual_mix),
        "child_kill_threshold": float(base_cfg.child_kill_threshold),
        "child_local_ifs_enabled": bool(base_cfg.child_local_ifs_enabled),
        "child_local_steps": float(base_cfg.child_local_steps),
        "child_local_support_only": bool(base_cfg.child_local_support_only),
        **_child_local_coherence_values(base_cfg),
        "law_packet_merge_threshold": float(base_cfg.law_packet_merge_threshold),
        "law_packet_min_score": float(base_cfg.law_packet_min_score),
        "law_packet_topk_families": float(base_cfg.law_packet_topk_families),
        **_phase_law_control_values(base_cfg),
        "soft_matryoshka_enabled": bool(base_cfg.soft_matryoshka_enabled),
        "branching_mode": str(base_cfg.branching_mode),
        "branch_law_version": str(base_cfg.branch_law_version),
        "branch_kernel_version": str(base_cfg.branch_kernel_version),
    }
    std = {
        "q_weights": [0.20 for _ in base_cfg.q_weights],
        "promotion_threshold": 0.10,
        "max_promotions": 1.25,
        "child_law_gain": 0.12,
        "attack_window": 2.0,
        "persistence_momentum": 0.10,
        "matryoshka_rank": 4.0,
        "slow_persistence": 0.05,
        "fast_persistence": 0.08,
        "persistence_curve": 0.30,
        "slow_write_scale": 0.04,
        "fast_write_scale": 0.15,
        "write_curve": 0.30,
        "prefix_coarse_weight": 0.08,
        "prefix_mid_weight": 0.08,
        "split_pressure": 0.08,
        "split_seed_scale": 0.04,
        "split_support_gain": 0.12,
        "merge_pressure": 0.06,
        "merge_phase_tol": 0.05,
        "merge_support_overlap_weight": 0.10,
        "survival_coherence_weight": 0.08,
        "survival_arc_weight": 0.08,
        "survival_qtrace_weight": 0.06,
        "survival_residue_penalty": 0.06,
        "collapse_sharpness": 0.15,
        "support_decay": 0.04,
        "support_spread": 1.0,
        "support_overlap_penalty": 0.05,
        "anti_fixation_weight": 0.06,
        "readout_temperature": 0.05,
        "qtrace_momentum": 0.04,
        "mask_neighborhood": 0.75,
        "topology_mask_gain": 0.05,
        "complexity_mask_gain": 0.05,
        "context_mask_gain": 0.05,
        "contrastive_mask_gain": 0.05,
        "aux_mask_suppression": 0.05,
        "instability_mask_gain": 0.05,
        "defect_phase_gain": 0.06,
        "defect_q_gain": 0.06,
        "defect_residue_gain": 0.04,
        "defect_sharpness_gain": 0.04,
        "defect_world_grad_gain": 0.05,
        "instability_seed_scale": 0.04,
        "instability_support_gain": 0.06,
        "instability_logit_gain": 0.05,
        "relation_attention_gain": 0.05,
        "relation_attention_sharpness": 0.08,
        "relation_value_gain": 0.04,
        "relation_support_gain": 0.04,
        "relation_logit_gain": 0.04,
        "relation_qtrace_gain": 0.04,
        "relation_residual_mix": 0.05,
        "child_kill_threshold": 0.02,
        **_CHILD_LOCAL_COHERENCE_CONTROL_STDS,
        "law_packet_merge_threshold": 0.03,
        "law_packet_min_score": 0.05,
        "law_packet_topk_families": 1.5,
    }
    mean.update({key: float(value) for key, value in _phase_law_control_values(base_cfg).items()})
    mean.update(_child_writeback_control_values(base_cfg))
    std.update(_PHASE_LAW_CONTROL_STDS)
    std.update(_CHILD_WRITEBACK_CONTROL_STDS)

    history: list[dict[str, Any]] = []
    best_state: dict[str, Any] | None = None
    rng = random.Random(seed)

    for step in range(iterations):
        candidates: list[dict[str, Any]] = []
        for _ in range(population):
            cand = _candidate_from_mean_std(mean, std, rng)
            cfg = _materialize_cfg(base_cfg, cand)
            train_eval = _evaluate_cfg(cfg, train_set, mode="active_packets", score_cfg=score_cfg)
            val_eval = _evaluate_cfg(cfg, val_set, mode="active_packets", score_cfg=score_cfg)
            row = {
                "iteration": step,
                "candidate": cand,
                "materialized_cfg": {
                    "q_weights": list(cfg.q_weights),
                    "promotion_threshold": cfg.promotion_threshold,
                    "max_promotions": cfg.max_promotions,
                    "child_law_gain": cfg.child_law_gain,
                    "attack_window": cfg.attack_window,
                    "persistence_momentum": cfg.persistence_momentum,
                    "recursion_depth": cfg.recursion_depth,
                    "soft_matryoshka_enabled": cfg.soft_matryoshka_enabled,
                    "matryoshka_rank": cfg.matryoshka_rank,
                    **_phase_law_control_values(cfg),
                    "slow_persistence": cfg.slow_persistence,
                    "fast_persistence": cfg.fast_persistence,
                    "persistence_curve": cfg.persistence_curve,
                    "slow_write_scale": cfg.slow_write_scale,
                    "fast_write_scale": cfg.fast_write_scale,
                    "write_curve": cfg.write_curve,
                    "prefix_coarse_weight": cfg.prefix_coarse_weight,
                    "prefix_mid_weight": cfg.prefix_mid_weight,
                    "split_pressure": cfg.split_pressure,
                    "split_seed_scale": cfg.split_seed_scale,
                    "split_support_gain": cfg.split_support_gain,
                    "merge_pressure": cfg.merge_pressure,
                    "merge_phase_tol": cfg.merge_phase_tol,
                    "merge_support_overlap_weight": cfg.merge_support_overlap_weight,
                    "survival_coherence_weight": cfg.survival_coherence_weight,
                    "survival_arc_weight": cfg.survival_arc_weight,
                    "survival_qtrace_weight": cfg.survival_qtrace_weight,
                    "survival_residue_penalty": cfg.survival_residue_penalty,
                    "collapse_sharpness": cfg.collapse_sharpness,
                    "support_decay": cfg.support_decay,
                    "support_spread": cfg.support_spread,
                      "support_overlap_penalty": cfg.support_overlap_penalty,
                      "anti_fixation_weight": cfg.anti_fixation_weight,
                      "readout_temperature": cfg.readout_temperature,
                      "qtrace_momentum": cfg.qtrace_momentum,
                      "mask_neighborhood": cfg.mask_neighborhood,
                      "topology_mask_gain": cfg.topology_mask_gain,
                      "complexity_mask_gain": cfg.complexity_mask_gain,
                    "context_mask_gain": cfg.context_mask_gain,
                    "contrastive_mask_gain": cfg.contrastive_mask_gain,
                    "aux_mask_suppression": cfg.aux_mask_suppression,
                    "instability_mask_gain": cfg.instability_mask_gain,
                    "defect_phase_gain": cfg.defect_phase_gain,
                    "defect_q_gain": cfg.defect_q_gain,
                    "defect_residue_gain": cfg.defect_residue_gain,
                    "defect_sharpness_gain": cfg.defect_sharpness_gain,
                    "defect_world_grad_gain": cfg.defect_world_grad_gain,
                    "instability_seed_scale": cfg.instability_seed_scale,
                    "instability_support_gain": cfg.instability_support_gain,
                    "instability_logit_gain": cfg.instability_logit_gain,
                    "relation_attention_gain": cfg.relation_attention_gain,
                    "relation_attention_sharpness": cfg.relation_attention_sharpness,
                    "relation_value_gain": cfg.relation_value_gain,
                    "relation_support_gain": cfg.relation_support_gain,
                    "relation_logit_gain": cfg.relation_logit_gain,
                    "relation_qtrace_gain": cfg.relation_qtrace_gain,
                    "relation_residual_mix": cfg.relation_residual_mix,
                    **_child_writeback_control_values(cfg),
                    "child_kill_threshold": cfg.child_kill_threshold,
                    "child_local_ifs_enabled": cfg.child_local_ifs_enabled,
                    "child_local_steps": cfg.child_local_steps,
                    "child_local_support_only": cfg.child_local_support_only,
                    **_child_local_coherence_values(cfg),
                    "law_packet_merge_threshold": cfg.law_packet_merge_threshold,
                    "law_packet_min_score": cfg.law_packet_min_score,
                    "law_packet_topk_families": cfg.law_packet_topk_families,
                    "branching_mode": cfg.branching_mode,
                    "branch_law_version": cfg.branch_law_version,
                    "branch_kernel_version": cfg.branch_kernel_version,
                },
                "train": train_eval,
                "val": val_eval,
            }
            candidates.append(row)

        candidates.sort(key=lambda item: (item["val"]["mean_score"], item["train"]["mean_score"]), reverse=True)
        elites = candidates[:elite_count]
        history.extend(candidates)

        mean = {
            "q_weights": [
                sum(row["materialized_cfg"]["q_weights"][i] for row in elites) / len(elites)
                for i in range(len(base_cfg.q_weights))
            ],
            "promotion_threshold": sum(row["materialized_cfg"]["promotion_threshold"] for row in elites) / len(elites),
            "max_promotions": sum(row["materialized_cfg"]["max_promotions"] for row in elites) / len(elites),
            "child_law_gain": sum(row["materialized_cfg"]["child_law_gain"] for row in elites) / len(elites),
            "attack_window": sum(row["materialized_cfg"]["attack_window"] for row in elites) / len(elites),
            "persistence_momentum": sum(row["materialized_cfg"]["persistence_momentum"] for row in elites) / len(elites),
            "matryoshka_rank": sum(row["materialized_cfg"]["matryoshka_rank"] for row in elites) / len(elites),
            "slow_persistence": sum(row["materialized_cfg"]["slow_persistence"] for row in elites) / len(elites),
            "fast_persistence": sum(row["materialized_cfg"]["fast_persistence"] for row in elites) / len(elites),
            "persistence_curve": sum(row["materialized_cfg"]["persistence_curve"] for row in elites) / len(elites),
            "slow_write_scale": sum(row["materialized_cfg"]["slow_write_scale"] for row in elites) / len(elites),
            "fast_write_scale": sum(row["materialized_cfg"]["fast_write_scale"] for row in elites) / len(elites),
            "write_curve": sum(row["materialized_cfg"]["write_curve"] for row in elites) / len(elites),
            "prefix_coarse_weight": sum(row["materialized_cfg"]["prefix_coarse_weight"] for row in elites) / len(elites),
            "prefix_mid_weight": sum(row["materialized_cfg"]["prefix_mid_weight"] for row in elites) / len(elites),
            "split_pressure": sum(row["materialized_cfg"]["split_pressure"] for row in elites) / len(elites),
            "split_seed_scale": sum(row["materialized_cfg"]["split_seed_scale"] for row in elites) / len(elites),
            "split_support_gain": sum(row["materialized_cfg"]["split_support_gain"] for row in elites) / len(elites),
            "merge_pressure": sum(row["materialized_cfg"]["merge_pressure"] for row in elites) / len(elites),
            "merge_phase_tol": sum(row["materialized_cfg"]["merge_phase_tol"] for row in elites) / len(elites),
            "merge_support_overlap_weight": sum(row["materialized_cfg"]["merge_support_overlap_weight"] for row in elites) / len(elites),
            "survival_coherence_weight": sum(row["materialized_cfg"]["survival_coherence_weight"] for row in elites) / len(elites),
            "survival_arc_weight": sum(row["materialized_cfg"]["survival_arc_weight"] for row in elites) / len(elites),
            "survival_qtrace_weight": sum(row["materialized_cfg"]["survival_qtrace_weight"] for row in elites) / len(elites),
            "survival_residue_penalty": sum(row["materialized_cfg"]["survival_residue_penalty"] for row in elites) / len(elites),
            "collapse_sharpness": sum(row["materialized_cfg"]["collapse_sharpness"] for row in elites) / len(elites),
            "support_decay": sum(row["materialized_cfg"]["support_decay"] for row in elites) / len(elites),
            "support_spread": sum(row["materialized_cfg"]["support_spread"] for row in elites) / len(elites),
            "support_overlap_penalty": sum(row["materialized_cfg"]["support_overlap_penalty"] for row in elites) / len(elites),
            "anti_fixation_weight": sum(row["materialized_cfg"]["anti_fixation_weight"] for row in elites) / len(elites),
            "readout_temperature": sum(row["materialized_cfg"]["readout_temperature"] for row in elites) / len(elites),
            "qtrace_momentum": sum(row["materialized_cfg"]["qtrace_momentum"] for row in elites) / len(elites),
            "mask_neighborhood": sum(row["materialized_cfg"]["mask_neighborhood"] for row in elites) / len(elites),
            "topology_mask_gain": sum(row["materialized_cfg"]["topology_mask_gain"] for row in elites) / len(elites),
            "complexity_mask_gain": sum(row["materialized_cfg"]["complexity_mask_gain"] for row in elites) / len(elites),
            "context_mask_gain": sum(row["materialized_cfg"]["context_mask_gain"] for row in elites) / len(elites),
            "contrastive_mask_gain": sum(row["materialized_cfg"]["contrastive_mask_gain"] for row in elites) / len(elites),
            "aux_mask_suppression": sum(row["materialized_cfg"]["aux_mask_suppression"] for row in elites) / len(elites),
            "instability_mask_gain": sum(row["materialized_cfg"]["instability_mask_gain"] for row in elites) / len(elites),
            "defect_phase_gain": sum(row["materialized_cfg"]["defect_phase_gain"] for row in elites) / len(elites),
            "defect_q_gain": sum(row["materialized_cfg"]["defect_q_gain"] for row in elites) / len(elites),
            "defect_residue_gain": sum(row["materialized_cfg"]["defect_residue_gain"] for row in elites) / len(elites),
            "defect_sharpness_gain": sum(row["materialized_cfg"]["defect_sharpness_gain"] for row in elites) / len(elites),
            "defect_world_grad_gain": sum(row["materialized_cfg"]["defect_world_grad_gain"] for row in elites) / len(elites),
            "instability_seed_scale": sum(row["materialized_cfg"]["instability_seed_scale"] for row in elites) / len(elites),
            "instability_support_gain": sum(row["materialized_cfg"]["instability_support_gain"] for row in elites) / len(elites),
            "instability_logit_gain": sum(row["materialized_cfg"]["instability_logit_gain"] for row in elites) / len(elites),
            "relation_attention_gain": sum(row["materialized_cfg"]["relation_attention_gain"] for row in elites) / len(elites),
            "relation_attention_sharpness": sum(row["materialized_cfg"]["relation_attention_sharpness"] for row in elites) / len(elites),
            "relation_value_gain": sum(row["materialized_cfg"]["relation_value_gain"] for row in elites) / len(elites),
            "relation_support_gain": sum(row["materialized_cfg"]["relation_support_gain"] for row in elites) / len(elites),
            "relation_logit_gain": sum(row["materialized_cfg"]["relation_logit_gain"] for row in elites) / len(elites),
            "relation_qtrace_gain": sum(row["materialized_cfg"]["relation_qtrace_gain"] for row in elites) / len(elites),
            "relation_residual_mix": sum(row["materialized_cfg"]["relation_residual_mix"] for row in elites) / len(elites),
            "child_kill_threshold": sum(row["materialized_cfg"]["child_kill_threshold"] for row in elites) / len(elites),
            "child_local_ifs_enabled": mean.get("child_local_ifs_enabled", False),
            "child_local_steps": mean.get("child_local_steps", 1),
            "child_local_support_only": mean.get("child_local_support_only", True),
            "child_local_coherence_retention_enabled": mean.get("child_local_coherence_retention_enabled", False),
            "law_packet_merge_threshold": sum(row["materialized_cfg"]["law_packet_merge_threshold"] for row in elites) / len(elites),
            "law_packet_min_score": sum(row["materialized_cfg"]["law_packet_min_score"] for row in elites) / len(elites),
            "law_packet_topk_families": sum(row["materialized_cfg"]["law_packet_topk_families"] for row in elites) / len(elites),
            "soft_matryoshka_enabled": mean["soft_matryoshka_enabled"],
            "branching_mode": mean["branching_mode"],
            "branch_law_version": mean["branch_law_version"],
            "branch_kernel_version": mean["branch_kernel_version"],
        }
        for key in _PHASE_LAW_CONTROL_DEFAULTS:
            mean[key] = sum(float(row["materialized_cfg"][key]) for row in elites) / len(elites)
        for key in _CHILD_WRITEBACK_CONTROL_DEFAULTS:
            mean[key] = sum(row["materialized_cfg"][key] for row in elites) / len(elites)
        for key in _CHILD_LOCAL_COHERENCE_CONTROL_DEFAULTS:
            mean[key] = sum(row["materialized_cfg"][key] for row in elites) / len(elites)

        def _elite_std(values: list[float], floor: float) -> float:
            if len(values) <= 1:
                return floor
            mean_v = sum(values) / len(values)
            var = sum((v - mean_v) ** 2 for v in values) / len(values)
            return max(floor, math.sqrt(var) * 0.85)

        std = {
            "q_weights": [
                _elite_std([row["materialized_cfg"]["q_weights"][i] for row in elites], 0.04)
                for i in range(len(base_cfg.q_weights))
            ],
            "promotion_threshold": _elite_std([row["materialized_cfg"]["promotion_threshold"] for row in elites], 0.03),
            "max_promotions": _elite_std([float(row["materialized_cfg"]["max_promotions"]) for row in elites], 0.50),
            "child_law_gain": _elite_std([row["materialized_cfg"]["child_law_gain"] for row in elites], 0.03),
            "attack_window": _elite_std([float(row["materialized_cfg"]["attack_window"]) for row in elites], 0.75),
            "persistence_momentum": _elite_std([row["materialized_cfg"]["persistence_momentum"] for row in elites], 0.03),
            "matryoshka_rank": _elite_std([float(row["materialized_cfg"]["matryoshka_rank"]) for row in elites], 2.0),
            "slow_persistence": _elite_std([row["materialized_cfg"]["slow_persistence"] for row in elites], 0.02),
            "fast_persistence": _elite_std([row["materialized_cfg"]["fast_persistence"] for row in elites], 0.03),
            "persistence_curve": _elite_std([row["materialized_cfg"]["persistence_curve"] for row in elites], 0.12),
            "slow_write_scale": _elite_std([row["materialized_cfg"]["slow_write_scale"] for row in elites], 0.02),
            "fast_write_scale": _elite_std([row["materialized_cfg"]["fast_write_scale"] for row in elites], 0.05),
            "write_curve": _elite_std([row["materialized_cfg"]["write_curve"] for row in elites], 0.12),
            "prefix_coarse_weight": _elite_std([row["materialized_cfg"]["prefix_coarse_weight"] for row in elites], 0.03),
            "prefix_mid_weight": _elite_std([row["materialized_cfg"]["prefix_mid_weight"] for row in elites], 0.03),
            "split_pressure": _elite_std([row["materialized_cfg"]["split_pressure"] for row in elites], 0.03),
            "split_seed_scale": _elite_std([row["materialized_cfg"]["split_seed_scale"] for row in elites], 0.02),
            "split_support_gain": _elite_std([row["materialized_cfg"]["split_support_gain"] for row in elites], 0.05),
            "merge_pressure": _elite_std([row["materialized_cfg"]["merge_pressure"] for row in elites], 0.02),
            "merge_phase_tol": _elite_std([row["materialized_cfg"]["merge_phase_tol"] for row in elites], 0.02),
            "merge_support_overlap_weight": _elite_std([row["materialized_cfg"]["merge_support_overlap_weight"] for row in elites], 0.04),
            "survival_coherence_weight": _elite_std([row["materialized_cfg"]["survival_coherence_weight"] for row in elites], 0.03),
            "survival_arc_weight": _elite_std([row["materialized_cfg"]["survival_arc_weight"] for row in elites], 0.03),
            "survival_qtrace_weight": _elite_std([row["materialized_cfg"]["survival_qtrace_weight"] for row in elites], 0.02),
            "survival_residue_penalty": _elite_std([row["materialized_cfg"]["survival_residue_penalty"] for row in elites], 0.02),
            "collapse_sharpness": _elite_std([row["materialized_cfg"]["collapse_sharpness"] for row in elites], 0.05),
            "support_decay": _elite_std([row["materialized_cfg"]["support_decay"] for row in elites], 0.02),
            "support_spread": _elite_std([float(row["materialized_cfg"]["support_spread"]) for row in elites], 0.5),
            "support_overlap_penalty": _elite_std([row["materialized_cfg"]["support_overlap_penalty"] for row in elites], 0.02),
            "anti_fixation_weight": _elite_std([row["materialized_cfg"]["anti_fixation_weight"] for row in elites], 0.02),
            "readout_temperature": _elite_std([row["materialized_cfg"]["readout_temperature"] for row in elites], 0.02),
            "qtrace_momentum": _elite_std([row["materialized_cfg"]["qtrace_momentum"] for row in elites], 0.02),
            "mask_neighborhood": _elite_std([float(row["materialized_cfg"]["mask_neighborhood"]) for row in elites], 0.5),
            "topology_mask_gain": _elite_std([row["materialized_cfg"]["topology_mask_gain"] for row in elites], 0.02),
            "complexity_mask_gain": _elite_std([row["materialized_cfg"]["complexity_mask_gain"] for row in elites], 0.02),
            "context_mask_gain": _elite_std([row["materialized_cfg"]["context_mask_gain"] for row in elites], 0.02),
            "contrastive_mask_gain": _elite_std([row["materialized_cfg"]["contrastive_mask_gain"] for row in elites], 0.02),
            "aux_mask_suppression": _elite_std([row["materialized_cfg"]["aux_mask_suppression"] for row in elites], 0.02),
            "instability_mask_gain": _elite_std([row["materialized_cfg"]["instability_mask_gain"] for row in elites], 0.02),
            "defect_phase_gain": _elite_std([row["materialized_cfg"]["defect_phase_gain"] for row in elites], 0.02),
            "defect_q_gain": _elite_std([row["materialized_cfg"]["defect_q_gain"] for row in elites], 0.02),
            "defect_residue_gain": _elite_std([row["materialized_cfg"]["defect_residue_gain"] for row in elites], 0.02),
            "defect_sharpness_gain": _elite_std([row["materialized_cfg"]["defect_sharpness_gain"] for row in elites], 0.02),
            "defect_world_grad_gain": _elite_std([row["materialized_cfg"]["defect_world_grad_gain"] for row in elites], 0.02),
            "instability_seed_scale": _elite_std([row["materialized_cfg"]["instability_seed_scale"] for row in elites], 0.02),
            "instability_support_gain": _elite_std([row["materialized_cfg"]["instability_support_gain"] for row in elites], 0.02),
            "instability_logit_gain": _elite_std([row["materialized_cfg"]["instability_logit_gain"] for row in elites], 0.02),
            "relation_attention_gain": _elite_std([row["materialized_cfg"]["relation_attention_gain"] for row in elites], 0.02),
            "relation_attention_sharpness": _elite_std([row["materialized_cfg"]["relation_attention_sharpness"] for row in elites], 0.03),
            "relation_value_gain": _elite_std([row["materialized_cfg"]["relation_value_gain"] for row in elites], 0.02),
            "relation_support_gain": _elite_std([row["materialized_cfg"]["relation_support_gain"] for row in elites], 0.02),
            "relation_logit_gain": _elite_std([row["materialized_cfg"]["relation_logit_gain"] for row in elites], 0.02),
            "relation_qtrace_gain": _elite_std([row["materialized_cfg"]["relation_qtrace_gain"] for row in elites], 0.02),
            "relation_residual_mix": _elite_std([row["materialized_cfg"]["relation_residual_mix"] for row in elites], 0.02),
            "child_kill_threshold": _elite_std([row["materialized_cfg"]["child_kill_threshold"] for row in elites], 0.01),
            "law_packet_merge_threshold": _elite_std([row["materialized_cfg"]["law_packet_merge_threshold"] for row in elites], 0.01),
            "law_packet_min_score": _elite_std([row["materialized_cfg"]["law_packet_min_score"] for row in elites], 0.02),
            "law_packet_topk_families": _elite_std([float(row["materialized_cfg"]["law_packet_topk_families"]) for row in elites], 0.5),
        }
        for key, floor in _PHASE_LAW_CONTROL_STDS.items():
            std[key] = _elite_std([float(row["materialized_cfg"][key]) for row in elites], floor * 0.5)
        for key, floor in _CHILD_WRITEBACK_CONTROL_STDS.items():
            std[key] = _elite_std([row["materialized_cfg"][key] for row in elites], floor * 0.5)
        for key, floor in _CHILD_LOCAL_COHERENCE_CONTROL_STDS.items():
            std[key] = _elite_std([row["materialized_cfg"][key] for row in elites], floor * 0.5)

        candidate_best = deepcopy(candidates[0])
        if best_state is None or candidate_best["val"]["mean_score"] > best_state["val"]["mean_score"]:
            best_state = candidate_best

    assert best_state is not None
    best_cfg = _materialize_cfg(base_cfg, best_state["materialized_cfg"])
    best_train = _evaluate_cfg(best_cfg, train_set, mode="active_packets", score_cfg=score_cfg)
    best_val = _evaluate_cfg(best_cfg, val_set, mode="active_packets", score_cfg=score_cfg)

    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_cfg_path = checkpoint_dir / "circleworld_config_cem_v1.json"
    history_path = out_dir / "search_history.json"
    summary_path = out_dir / "train_summary.json"
    baseline_path = out_dir / "baseline_report.json"

    best_config_payload = {
        "runtime": "circleworld_proto",
        "schema": "circleworld_v0",
        "trainer": "cem_parameter_search",
        "mode": "active_packets",
        "seed": seed,
        "device": str(device),
        "time_steps": time_steps,
        "init_config_path": str(init_config_path) if init_config_path else None,
        "include_graduation_anchors": include_graduation_anchors,
        "include_naked_rafa": include_naked_rafa,
        "score_cfg": score_cfg or {},
        "train_plan": [{"source": src, "seed": s} for src, s in train_plan],
        "val_plan": [{"source": src, "seed": s} for src, s in val_plan],
        "graduation_anchor_train_wavs": grad_train_wavs,
        "graduation_anchor_val_wavs": grad_val_wavs,
        "config": {
            "q_weights": list(best_cfg.q_weights),
            "promotion_threshold": best_cfg.promotion_threshold,
            "max_promotions": best_cfg.max_promotions,
            "child_law_gain": best_cfg.child_law_gain,
            "attack_window": best_cfg.attack_window,
            "persistence_momentum": best_cfg.persistence_momentum,
            "recursion_depth": best_cfg.recursion_depth,
            "soft_matryoshka_enabled": best_cfg.soft_matryoshka_enabled,
            "matryoshka_rank": best_cfg.matryoshka_rank,
            **_phase_law_control_values(best_cfg),
            "prefix_fracs": list(best_cfg.prefix_fracs),
            "slow_persistence": best_cfg.slow_persistence,
            "fast_persistence": best_cfg.fast_persistence,
            "persistence_curve": best_cfg.persistence_curve,
            "slow_write_scale": best_cfg.slow_write_scale,
            "fast_write_scale": best_cfg.fast_write_scale,
            "write_curve": best_cfg.write_curve,
            "prefix_coarse_weight": best_cfg.prefix_coarse_weight,
            "prefix_mid_weight": best_cfg.prefix_mid_weight,
            "branching_mode": best_cfg.branching_mode,
            "num_modes": best_cfg.num_modes,
            "readout_mode": best_cfg.readout_mode,
            "branch_law_version": best_cfg.branch_law_version,
            "q_trace_rank": best_cfg.q_trace_rank,
            "dormant_logit": best_cfg.dormant_logit,
            "dormant_support": best_cfg.dormant_support,
            "dormant_q_scale": best_cfg.dormant_q_scale,
            "split_pressure": best_cfg.split_pressure,
            "split_seed_scale": best_cfg.split_seed_scale,
            "split_support_gain": best_cfg.split_support_gain,
            "merge_pressure": best_cfg.merge_pressure,
            "merge_phase_tol": best_cfg.merge_phase_tol,
            "merge_support_overlap_weight": best_cfg.merge_support_overlap_weight,
            "survival_coherence_weight": best_cfg.survival_coherence_weight,
            "survival_arc_weight": best_cfg.survival_arc_weight,
            "survival_qtrace_weight": best_cfg.survival_qtrace_weight,
            "survival_residue_penalty": best_cfg.survival_residue_penalty,
            "collapse_sharpness": best_cfg.collapse_sharpness,
            "support_decay": best_cfg.support_decay,
            "support_spread": best_cfg.support_spread,
              "support_overlap_penalty": best_cfg.support_overlap_penalty,
              "anti_fixation_weight": best_cfg.anti_fixation_weight,
              "readout_temperature": best_cfg.readout_temperature,
              "slot2_support_threshold": best_cfg.slot2_support_threshold,
              "real_branch_threshold": best_cfg.real_branch_threshold,
              "mode_perturb_window_frac": best_cfg.mode_perturb_window_frac,
              "qtrace_momentum": best_cfg.qtrace_momentum,
              "mask_neighborhood": best_cfg.mask_neighborhood,
              "topology_mask_gain": best_cfg.topology_mask_gain,
              "complexity_mask_gain": best_cfg.complexity_mask_gain,
              "context_mask_gain": best_cfg.context_mask_gain,
              "contrastive_mask_gain": best_cfg.contrastive_mask_gain,
              "aux_mask_suppression": best_cfg.aux_mask_suppression,
              "instability_mask_gain": best_cfg.instability_mask_gain,
              "defect_phase_gain": best_cfg.defect_phase_gain,
              "defect_q_gain": best_cfg.defect_q_gain,
              "defect_residue_gain": best_cfg.defect_residue_gain,
              "defect_sharpness_gain": best_cfg.defect_sharpness_gain,
              "defect_world_grad_gain": best_cfg.defect_world_grad_gain,
              "instability_seed_scale": best_cfg.instability_seed_scale,
              "instability_support_gain": best_cfg.instability_support_gain,
              "instability_logit_gain": best_cfg.instability_logit_gain,
              "branch_kernel_version": best_cfg.branch_kernel_version,
              "relation_attention_gain": best_cfg.relation_attention_gain,
              "relation_attention_sharpness": best_cfg.relation_attention_sharpness,
              "relation_value_gain": best_cfg.relation_value_gain,
              "relation_support_gain": best_cfg.relation_support_gain,
              "relation_logit_gain": best_cfg.relation_logit_gain,
              "relation_qtrace_gain": best_cfg.relation_qtrace_gain,
              "relation_residual_mix": best_cfg.relation_residual_mix,
              **_child_writeback_control_values(best_cfg),
              "child_kill_threshold": best_cfg.child_kill_threshold,
              "child_local_ifs_enabled": best_cfg.child_local_ifs_enabled,
              "child_local_steps": best_cfg.child_local_steps,
              "child_local_support_only": best_cfg.child_local_support_only,
              **_child_local_coherence_values(best_cfg),
              "law_packet_merge_threshold": best_cfg.law_packet_merge_threshold,
              "law_packet_min_score": best_cfg.law_packet_min_score,
              "law_packet_topk_families": best_cfg.law_packet_topk_families,
          },
    }
    best_cfg_path.write_text(json.dumps(best_config_payload, indent=2), encoding="utf-8")
    history_path.write_text(json.dumps(history, indent=2), encoding="utf-8")
    baseline_path.write_text(json.dumps(baseline, indent=2), encoding="utf-8")

    summary = {
        "runtime": "circleworld_proto",
        "schema": "circleworld_v0",
        "status": "completed",
        "trainer": "cem_parameter_search",
        "seed": seed,
        "device": str(device),
        "iterations": iterations,
        "population": population,
        "elite_count": elite_count,
        "time_steps": time_steps,
        "checkpoint": str(best_cfg_path),
        "baseline_active_val": baseline["active_packets"]["val"],
        "best_active_train": best_train,
        "best_active_val": best_val,
        "best_candidate": best_state,
        "best_config": best_config_payload["config"],
        "artifact_paths": {
            "summary": str(summary_path),
            "baseline_report": str(baseline_path),
            "search_history": str(history_path),
            "checkpoint": str(best_cfg_path),
        },
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description="Train Circleworld parameters in an isolated prototype lane.")
    ap.add_argument("--out-dir", default=str(ROOT / "outputs" / "circleworld_proto" / "proper_attempt_2026-04-06"))
    ap.add_argument("--checkpoint-dir", default=str(ROOT / "checkpoints_circleworld_proto" / "proper_attempt_2026-04-06"))
    ap.add_argument("--iterations", type=int, default=8)
    ap.add_argument("--population", type=int, default=10)
    ap.add_argument("--elite-count", type=int, default=3)
    ap.add_argument("--time-steps", type=int, default=96)
    ap.add_argument("--recursion-depth", type=int, default=2)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--train-count", type=int, default=6)
    ap.add_argument("--val-count", type=int, default=3)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--init-config", default=None)
    ap.add_argument("--target-dom", type=float, default=0.70)
    ap.add_argument("--target-entropy", type=float, default=0.45)
    ap.add_argument("--min-promotions", type=float, default=3.0)
    ap.add_argument("--w-dom-target", type=float, default=0.0)
    ap.add_argument("--w-entropy-target", type=float, default=0.0)
    ap.add_argument("--w-promote-floor", type=float, default=0.0)
    ap.add_argument("--include-graduation-anchors", action="store_true")
    ap.add_argument("--no-naked-rafa", action="store_true")
    args = ap.parse_args()

    score_cfg = {
        "target_dom": float(args.target_dom),
        "target_entropy": float(args.target_entropy),
        "min_promotions": float(args.min_promotions),
        "w_dom_target": float(args.w_dom_target),
        "w_entropy_target": float(args.w_entropy_target),
        "w_promote_floor": float(args.w_promote_floor),
    }

    summary = train_circleworld(
        out_dir=Path(args.out_dir),
        checkpoint_dir=Path(args.checkpoint_dir),
        iterations=args.iterations,
        population=args.population,
        elite_count=args.elite_count,
        time_steps=args.time_steps,
        recursion_depth=args.recursion_depth,
        device_name=args.device,
        train_count=args.train_count,
        val_count=args.val_count,
        seed=args.seed,
        init_config_path=Path(args.init_config) if args.init_config else None,
        score_cfg=score_cfg,
        include_graduation_anchors=bool(args.include_graduation_anchors),
        include_naked_rafa=not bool(args.no_naked_rafa),
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
