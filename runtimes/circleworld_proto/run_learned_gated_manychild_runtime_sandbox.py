from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
LINEAGE = ROOT / "lineages" / "04_positive_replacement"
RUNTIME = Path(__file__).resolve().parent
for path in (ROOT, LINEAGE, RUNTIME):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from ablate_formalization import make_seed_rafa  # noqa: E402
from build_shadow_branch_law_table import shadow_branch_law_row  # noqa: E402
from causal_operator_selector import causal_pair_feature_vector  # noqa: E402
from circleworld import circleworld_step, clone_circleworld_state  # noqa: E402
from evaluate_circleworld import _load_circle_cfg, _safe_device  # noqa: E402
from evaluate_shadow_branch_law_calibration import _load_model  # noqa: E402
from resonant_law_objects import run_retrieval_assay, score_query_to_law_object  # noqa: E402
from run_contrastive_structural_embedding_probe import operator_safety_proxy, recurrence_compatibility  # noqa: E402
from run_learned_gated_multistep_operator_sandbox import (  # noqa: E402
    _as_float,
    _masked_mean,
    _mean,
    _parse_seeds,
    _predict_gate_fields,
    _prepare_case,
)
from run_learned_gated_writeback_sandbox import _soft_writeback_scale  # noqa: E402
from run_resonant_operator_causality_assay import (  # noqa: E402
    _apply_child_operator,
    _load_causal_selector,
    _object_child_mapping,
    _phase_delta_from_to,
    _select_operator_triplet,
)
from test_nested_commitment import ASSAY_KILLSWITCH, DEFAULT_ASSAY_KILLSWITCH  # noqa: E402


def _child_union_gate(state: dict[str, Any]) -> torch.Tensor:
    parent_mode1 = state["phase_modes"][..., 1, :]
    gate = torch.zeros(parent_mode1.shape[:-1], device=parent_mode1.device, dtype=parent_mode1.dtype)
    for child in state.get("child_worlds", []) or []:
        if not isinstance(child, dict) or not child.get("active", False):
            continue
        support = child.get("mode_support")
        if not torch.is_tensor(support) or support.shape != gate.shape:
            continue
        coherence = child.get("mode_coherence")
        if not torch.is_tensor(coherence) or coherence.shape != gate.shape:
            coherence = torch.ones_like(support)
        gate = torch.maximum(gate, support.clamp(0.0, 1.0) * (0.20 + 0.80 * coherence.clamp(0.0, 1.0)))
    if float(gate.max().detach().cpu().item()) <= 0.0:
        gate = torch.ones_like(gate)
    return gate.clamp(0.0, 1.0)


def _force_children_writeback_ready(state: dict[str, Any], cfg: Any) -> None:
    for child in state.get("child_worlds", []) or []:
        if not isinstance(child, dict) or not child.get("active", False):
            continue
        child["survival_age"] = max(float(child.get("survival_age", 0.0)), float(cfg.child_min_age_for_writeback))
        child["writeback_budget"] = max(float(child.get("writeback_budget", 0.0)), float(cfg.child_writeback_budget))


def _predict_causal_pair(
    causal_selector: dict[str, Any] | None,
    query: dict[str, Any],
    candidate: dict[str, Any],
    row: dict[str, Any],
    device: torch.device,
) -> dict[str, float]:
    if causal_selector is None:
        return {
            "route_score": _as_float(row.get("final_score", row.get("geometric_score", 0.0))),
            "movement": _as_float(row.get("candidate_safety_proxy", 0.0)),
            "jump_safety": _as_float(row.get("candidate_safety_proxy", 0.0)),
            "identity": 1.0 if bool(row.get("same_family", False)) else 0.0,
        }
    values = causal_pair_feature_vector(query, candidate, row)
    feature_dim = int(causal_selector["feature_dim"])
    if len(values) < feature_dim:
        values.extend([0.0] * (feature_dim - len(values)))
    elif len(values) > feature_dim:
        values = values[:feature_dim]
    tensor = torch.tensor([values], dtype=torch.float32, device=device)
    model = causal_selector["model"]
    with torch.no_grad():
        output = model(tensor)
    if isinstance(output, dict):
        return {
            "route_score": float(output["route_score"].squeeze(0).detach().cpu().item()),
            "movement": float(output["movement"].squeeze(0).detach().cpu().item()),
            "jump_safety": float(output["jump_safety"].squeeze(0).detach().cpu().item()),
            "identity": float(output["identity"].squeeze(0).detach().cpu().item()),
        }
    score = float(output.squeeze(0).detach().cpu().item())
    return {"route_score": score, "movement": score, "jump_safety": score, "identity": score}


def _best_decoy_metrics(
    objects: list[dict[str, Any]],
    by_id: dict[str, dict[str, Any]],
    operator_metrics: dict[str, dict[str, float]],
    candidate: dict[str, Any],
) -> dict[str, float]:
    candidate_family = str(candidate.get("local_family_key", ""))
    decoys: list[tuple[float, dict[str, float]]] = []
    for obj in objects:
        obj_id = str(obj.get("object_id", ""))
        if obj_id == str(candidate.get("object_id", "")):
            continue
        if str(obj.get("local_family_key", "")) == candidate_family:
            continue
        metrics = operator_metrics.get(obj_id)
        if metrics is None and obj_id in by_id:
            metrics = {}
        decoys.append((_as_float((metrics or {}).get("world_jump_proxy", 0.0)), metrics or {}))
    if not decoys:
        return {}
    decoys.sort(key=lambda item: item[0], reverse=True)
    return decoys[0][1]


def _assign_learned_scales(
    state: dict[str, Any],
    *,
    learned_model: torch.nn.Module,
    causal_selector: dict[str, Any] | None,
    device: torch.device,
    variant: str,
    writeback_threshold: float,
    soft_floor: float,
    soft_full: float,
    case_label: str,
    operator_gain: float,
) -> dict[str, Any]:
    objects, by_id = _object_child_mapping(state, case=case_label, branch=f"manychild_{variant}")
    if not objects:
        return {"object_count": 0, "mean_score": 0.0, "mean_scale": 0.0, "nonzero_scale_fraction": 0.0, "assignments": []}
    retrieval = run_retrieval_assay(objects, top_k=3)
    query_id, _, _, _, _ = _select_operator_triplet(objects, retrieval)
    by_object = {str(obj.get("object_id", "")): obj for obj in objects}
    query = by_object.get(query_id) or objects[0]
    operator_metrics: dict[str, dict[str, float]] = {}
    for obj in objects:
        obj_id = str(obj.get("object_id", ""))
        child = by_id.get(obj_id)
        if child is not None:
            operator_metrics[obj_id] = _apply_child_operator(state, child, gain=float(operator_gain))

    assignments: list[dict[str, Any]] = []
    for obj in objects:
        obj_id = str(obj.get("object_id", ""))
        child = by_id.get(obj_id)
        if child is None:
            continue
        if obj_id == str(query.get("object_id", "")):
            pair_row = {
                "geometric_score": 1.0,
                "final_score": 1.0,
                "same_local_family": 1.0,
                "same_structural_family": 1.0,
                "same_family": True,
                "same_object": True,
                "identity_membrane_pass": 1.0,
                "strong_recurrence_pass": 1.0,
                "recurrence_compatibility": 1.0,
                "query_safety_proxy": float(operator_safety_proxy(query)),
                "candidate_safety_proxy": float(operator_safety_proxy(obj)),
            }
        else:
            pair_row = score_query_to_law_object(query, obj, family_key_field="local_family_key")
            pair_row["recurrence_compatibility"] = float(recurrence_compatibility(query, obj))
            pair_row["query_safety_proxy"] = float(operator_safety_proxy(query))
            pair_row["candidate_safety_proxy"] = float(operator_safety_proxy(obj))
            pair_row["identity_membrane_pass"] = 1.0 if bool(pair_row.get("same_family", False)) else 0.0
            pair_row["same_local_family"] = 1.0 if bool(pair_row.get("same_family", False)) else 0.0
            pair_row["same_structural_family"] = 1.0 if bool(pair_row.get("same_family", False)) else 0.0
            pair_row["strong_recurrence_pass"] = (
                1.0 if _as_float(pair_row.get("recurrence_compatibility", 0.0)) >= 0.90 else 0.0
            )
        selector_pred = _predict_causal_pair(causal_selector, query, obj, pair_row, device)
        metrics = operator_metrics.get(obj_id, {})
        decoy_metrics = _best_decoy_metrics(objects, by_id, operator_metrics, obj)
        case = {
            "source": "manychild_runtime",
            "seed": 0,
            "causal_selector_model_score": selector_pred["route_score"],
            "causal_selector_movement_pred": selector_pred["movement"],
            "causal_selector_jump_safety_pred": selector_pred["jump_safety"],
            "causal_selector_identity_pred": selector_pred["identity"],
            "causal_selector_identity_membrane_pass": _as_float(pair_row.get("identity_membrane_pass", 0.0)),
            "causal_selector_same_local_family": _as_float(pair_row.get("same_local_family", 0.0)),
            "causal_selector_same_structural_family": _as_float(pair_row.get("same_structural_family", 0.0)),
            "causal_selector_recurrence_compatibility": _as_float(pair_row.get("recurrence_compatibility", 0.0)),
            "causal_parent_phase_divergence": _as_float(metrics.get("parent_phase_divergence", 0.0)),
            "causal_support_shift": _as_float(metrics.get("support_shift", 0.0)),
            "causal_qtrace_shift_proxy": _as_float(metrics.get("qtrace_shift_proxy", 0.0)),
            "causal_world_jump_proxy": _as_float(metrics.get("world_jump_proxy", 0.0)),
            "decoy_parent_phase_divergence": _as_float(decoy_metrics.get("parent_phase_divergence", 0.0)),
            "decoy_support_shift": _as_float(decoy_metrics.get("support_shift", 0.0)),
            "decoy_qtrace_shift_proxy": _as_float(decoy_metrics.get("qtrace_shift_proxy", 0.0)),
            "decoy_world_jump_proxy": _as_float(decoy_metrics.get("world_jump_proxy", 0.0)),
        }
        measured = shadow_branch_law_row(case, source_path="manychild_runtime")
        predicted = _predict_gate_fields(learned_model, measured, device)
        score = _as_float(predicted.get("predicted_writeback_score", 0.0))
        if variant in {"learned_hard", "learned_hard_alloc"}:
            scale = 1.0 if score >= float(writeback_threshold) else 0.0
        elif variant in {
            "learned_soft",
            "learned_soft_alloc",
            "learned_soft_alloc_loose",
            "learned_soft_develop_alloc",
            "learned_soft_directional_coalition",
            "learned_soft_directional_coalition_boost",
            "learned_soft_target_coalition",
            "learned_soft_contrastive_sibling_target",
            "learned_soft_target_authority_scout",
        }:
            scale = _soft_writeback_scale(score, soft_floor=float(soft_floor), soft_full=float(soft_full))
        else:
            raise ValueError(f"Unsupported learned variant: {variant}")
        child["assay_writeback_scale"] = float(scale)
        assignments.append(
            {
                "child_id": int(child.get("child_id", -1)),
                "object_id": obj_id,
                "local_family_key": str(obj.get("local_family_key", "")),
                "is_query": obj_id == str(query.get("object_id", "")),
                "writeback_score": score,
                "scale": float(scale),
                "raw_scale": float(scale),
                "route_score": selector_pred["route_score"],
                "identity_pred": selector_pred["identity"],
                "jump_safety_pred": selector_pred["jump_safety"],
                "recurrence_compatibility": _as_float(pair_row.get("recurrence_compatibility", 0.0)),
                "causal_parent_phase_divergence": _as_float(metrics.get("parent_phase_divergence", 0.0)),
                "causal_world_jump_proxy": _as_float(metrics.get("world_jump_proxy", 0.0)),
                "decoy_world_jump_proxy": _as_float(decoy_metrics.get("world_jump_proxy", 0.0)),
            }
        )
    allocator_info: dict[str, Any] = {}
    if variant in {
        "learned_hard_alloc",
        "learned_soft_alloc",
        "learned_soft_alloc_loose",
        "learned_soft_develop_alloc",
        "learned_soft_directional_coalition",
        "learned_soft_directional_coalition_boost",
        "learned_soft_target_coalition",
        "learned_soft_contrastive_sibling_target",
        "learned_soft_target_authority_scout",
    }:
        if variant == "learned_soft_target_coalition":
            allocator_info = _apply_target_conditioned_coalition_allocation(state, assignments, by_id)
        elif variant == "learned_soft_contrastive_sibling_target":
            allocator_info = _apply_contrastive_sibling_target_allocation(state, assignments, by_id)
        elif variant == "learned_soft_target_authority_scout":
            allocator_info = _apply_target_authority_scout_allocation(state, assignments, by_id)
        elif variant in {"learned_soft_directional_coalition", "learned_soft_directional_coalition_boost"}:
            allocator_info = _apply_directional_coalition_allocation(state, assignments, by_id, profile=variant)
        else:
            allocator_info = _apply_ecology_allocation(state, assignments, by_id, profile=variant)
    return {
        "object_count": len(objects),
        "query_object_id": str(query.get("object_id", "")),
        "mean_score": _mean([_as_float(row.get("writeback_score", 0.0)) for row in assignments]),
        "mean_scale": _mean([_as_float(row.get("scale", 0.0)) for row in assignments]),
        "mean_raw_scale": _mean([_as_float(row.get("raw_scale", row.get("scale", 0.0))) for row in assignments]),
        "nonzero_scale_fraction": _mean([1.0 if _as_float(row.get("scale", 0.0)) > 1.0e-6 else 0.0 for row in assignments]),
        **allocator_info,
        "assignments": assignments,
    }


def _return_direction_vector(
    state: dict[str, Any],
    child: dict[str, Any],
) -> torch.Tensor:
    parent_mode1 = state["phase_modes"][..., 1, :]
    child_phase = child.get("phase_state")
    if not torch.is_tensor(child_phase) or child_phase.shape != parent_mode1.shape:
        return torch.zeros(parent_mode1.shape[:-1], device=parent_mode1.device, dtype=parent_mode1.dtype).flatten()
    support = child.get("mode_support")
    if not torch.is_tensor(support) or support.shape != parent_mode1.shape[:-1]:
        support = torch.ones(parent_mode1.shape[:-1], device=parent_mode1.device, dtype=parent_mode1.dtype)
    coherence = child.get("mode_coherence")
    if not torch.is_tensor(coherence) or coherence.shape != support.shape:
        coherence = torch.ones_like(support)
    gate = support.clamp(0.0, 1.0) * (0.20 + 0.80 * coherence.clamp(0.0, 1.0))
    return (_phase_delta_from_to(parent_mode1, child_phase).clamp(-3.14159265, 3.14159265) * gate).detach().float().flatten()


def _apply_directional_coalition_allocation(
    state: dict[str, Any],
    assignments: list[dict[str, Any]],
    by_object_id: dict[str, dict[str, Any]],
    *,
    profile: str = "learned_soft_directional_coalition",
) -> dict[str, Any]:
    if not assignments:
        return {
            "allocation_enabled": True,
            "allocation_profile": "learned_soft_directional_coalition",
            "mean_developmental_drive": 0.0,
            "directional_coalition_size": 0.0,
            "directional_alignment_mean": 0.0,
            "directional_opposition_mean": 0.0,
        }

    directions: list[torch.Tensor] = []
    weights: list[float] = []
    developmental_drives: list[float] = []
    boosted = str(profile).endswith("_boost")
    for row in assignments:
        child = by_object_id.get(str(row.get("object_id", "")))
        if isinstance(child, dict):
            directions.append(_return_direction_vector(state, child))
        else:
            parent_mode1 = state["phase_modes"][..., 1, :]
            directions.append(torch.zeros(parent_mode1.shape[:-1], device=parent_mode1.device, dtype=parent_mode1.dtype).flatten())
        safety = max(0.0, min(1.0, _as_float(row.get("jump_safety_pred", 0.0))))
        score = max(0.0, min(1.0, _as_float(row.get("writeback_score", 0.0))))
        recurrence = max(0.0, min(1.0, _as_float(row.get("recurrence_compatibility", 0.0))))
        raw_scale = max(0.0, min(1.0, _as_float(row.get("scale", 0.0))))
        movement = max(0.0, min(1.0, _as_float(row.get("causal_parent_phase_divergence", 0.0)) / 0.35))
        jump = max(0.0, min(1.0, _as_float(row.get("causal_world_jump_proxy", 0.0)) / 0.08))
        developmental_drive = max(
            0.0,
            min(1.0, movement * (0.42 + 0.58 * safety) * (0.40 + 0.60 * recurrence) - 0.38 * jump),
        )
        developmental_drives.append(float(developmental_drive))
        weights.append(float(raw_scale * (0.18 + 0.82 * score) * (0.35 + 0.65 * safety) * (0.45 + 0.55 * developmental_drive)))

    direction_stack = torch.stack(directions, dim=0)
    norms = torch.linalg.vector_norm(direction_stack, dim=1).clamp_min(1.0e-8)
    cos = (direction_stack @ direction_stack.T) / (norms[:, None] * norms[None, :])
    cos = torch.nan_to_num(cos, nan=0.0, posinf=0.0, neginf=0.0).clamp(-1.0, 1.0)
    weight_t = torch.tensor(weights, dtype=torch.float32, device=cos.device)
    positive = cos.clamp_min(0.0)
    opposition = (-cos).clamp_min(0.0)
    seed_scores = weight_t + (positive * weight_t[None, :]).sum(dim=1) - 0.55 * (opposition * weight_t[None, :]).sum(dim=1)
    seed_idx = int(torch.argmax(seed_scores).detach().cpu().item())
    alignment = positive[seed_idx]
    opposition_to_seed = opposition[seed_idx]
    member_threshold = 0.14 if boosted else 0.22
    member_mask = (alignment >= member_threshold) | (torch.arange(len(assignments), device=alignment.device) == seed_idx)

    raw_allocations: list[float] = []
    family_counts: dict[str, int] = {}
    for row in assignments:
        key = str(row.get("local_family_key", ""))
        family_counts[key] = family_counts.get(key, 0) + 1
    for idx, row in enumerate(assignments):
        raw_scale = max(0.0, min(1.0, _as_float(row.get("scale", 0.0))))
        safety = max(0.0, min(1.0, _as_float(row.get("jump_safety_pred", 0.0))))
        score = max(0.0, min(1.0, _as_float(row.get("writeback_score", 0.0))))
        dev = developmental_drives[idx]
        align = float(alignment[idx].detach().cpu().item())
        oppose = float(opposition_to_seed[idx].detach().cpu().item())
        family_count = max(1, family_counts.get(str(row.get("local_family_key", "")), 1))
        redundancy_penalty = 1.0 + 0.06 * max(0, family_count - 1)
        if bool(member_mask[idx].detach().cpu().item()):
            if boosted:
                coalition_quality = (
                    (0.35 + 0.65 * align)
                    * (0.42 + 0.58 * dev)
                    * (0.50 + 0.50 * safety)
                    * (0.45 + 0.55 * score)
                )
                allocation = 1.35 * raw_scale * coalition_quality / redundancy_penalty
            else:
                coalition_quality = (0.20 + 0.80 * align) * (0.25 + 0.75 * dev) * (0.40 + 0.60 * safety) * (0.35 + 0.65 * score)
                allocation = raw_scale * coalition_quality / redundancy_penalty
        else:
            allocation = raw_scale * (0.035 if boosted else 0.015) * max(0.0, 1.0 - oppose)
        row["directional_alignment_to_seed"] = align
        row["directional_opposition_to_seed"] = oppose
        row["directional_coalition_member"] = 1.0 if bool(member_mask[idx].detach().cpu().item()) else 0.0
        raw_allocations.append(float(max(0.0, allocation)))

    budget_mean = 0.86 if boosted else 0.52
    budget = float(budget_mean * max(1, len(assignments)))
    total_raw = float(np.sum(raw_allocations))
    shrink = min(1.0, budget / max(total_raw, 1.0e-8))
    for row, allocation in zip(assignments, raw_allocations):
        allocated_scale = max(0.0, min(1.0, float(allocation * shrink)))
        row["allocated_scale"] = allocated_scale
        row["scale"] = allocated_scale
        child = by_object_id.get(str(row.get("object_id", "")))
        if isinstance(child, dict):
            child["assay_writeback_scale"] = allocated_scale

    member_values = [float(alignment[idx].detach().cpu().item()) for idx in range(len(assignments)) if bool(member_mask[idx].detach().cpu().item())]
    return {
        "allocation_enabled": True,
        "allocation_profile": str(profile),
        "allocation_budget_mean_scale": float(budget_mean),
        "allocation_total_raw_authority": total_raw,
        "allocation_total_allocated_authority": float(np.sum([_as_float(row.get("allocated_scale", 0.0)) for row in assignments])),
        "allocation_shrink_factor": float(shrink),
        "mean_overlap_load": 0.0,
        "mean_redundancy_penalty": 1.0,
        "mean_developmental_drive": _mean(developmental_drives),
        "directional_seed_score": float(seed_scores[seed_idx].detach().cpu().item()),
        "directional_coalition_size": float(sum(1 for idx in range(len(assignments)) if bool(member_mask[idx].detach().cpu().item()))),
        "directional_alignment_mean": _mean(member_values),
        "directional_opposition_mean": float(opposition_to_seed.mean().detach().cpu().item()),
    }


def _collapse_direction_vector(state: dict[str, Any]) -> torch.Tensor:
    phase_modes = state["phase_modes"]
    parent_mode0 = phase_modes[..., 0, :]
    parent_mode1 = phase_modes[..., 1, :]
    return _phase_delta_from_to(parent_mode1, parent_mode0).clamp(-3.14159265, 3.14159265).detach().float().flatten()


def _controlled_sibling_target_vector(state: dict[str, Any], target_radians: float = 0.085) -> torch.Tensor:
    phase_modes = state["phase_modes"]
    parent_mode1 = phase_modes[..., 1, :]
    support = _child_union_gate(state).detach().float().clamp(0.0, 1.0)
    batch, freq_count, time_count = support.shape
    freq_axis = torch.linspace(-1.0, 1.0, freq_count, device=support.device, dtype=support.dtype).view(1, freq_count, 1)
    time_axis = torch.linspace(-1.0, 1.0, time_count, device=support.device, dtype=support.dtype).view(1, 1, time_count)
    freq_axis = freq_axis.expand(batch, freq_count, time_count)
    time_axis = time_axis.expand(batch, freq_count, time_count)
    carrier = (
        torch.sin(2.0 * torch.pi * (0.31 * time_axis + 0.17 * freq_axis + 0.11 * freq_axis * time_axis))
        + 0.55 * torch.cos(2.0 * torch.pi * (0.13 * time_axis - 0.23 * freq_axis))
    )
    support_mass = support.sum().clamp_min(1.0e-8)
    carrier_mean = (carrier * support).sum() / support_mass
    raw = (carrier - carrier_mean) * support
    raw_vec = raw.detach().float().flatten()

    collapse = _collapse_direction_vector(state)
    collapse_norm = torch.linalg.vector_norm(collapse).clamp_min(1.0e-8)
    collapse_unit = collapse / collapse_norm
    tangent = raw_vec - torch.dot(raw_vec, collapse_unit) * collapse_unit
    tangent_norm = torch.linalg.vector_norm(tangent)
    if float(tangent_norm.detach().cpu().item()) <= 1.0e-8:
        return torch.zeros(parent_mode1.shape[:-1], device=parent_mode1.device, dtype=parent_mode1.dtype).flatten()
    active_scale = torch.sqrt(support_mass.detach().float()).clamp_min(1.0)
    return (tangent / tangent_norm.clamp_min(1.0e-8)) * float(target_radians) * active_scale


def _apply_target_conditioned_coalition_allocation(
    state: dict[str, Any],
    assignments: list[dict[str, Any]],
    by_object_id: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    if not assignments:
        return {
            "allocation_enabled": True,
            "allocation_profile": "learned_soft_target_coalition",
            "mean_developmental_drive": 0.0,
            "directional_coalition_size": 0.0,
            "directional_alignment_mean": 0.0,
            "directional_opposition_mean": 0.0,
            "target_projection_mean": 0.0,
            "target_norm": 0.0,
        }

    collapse = _collapse_direction_vector(state)
    collapse_norm = torch.linalg.vector_norm(collapse).clamp_min(1.0e-8)
    collapse_unit = collapse / collapse_norm

    directions: list[torch.Tensor] = []
    tangent_directions: list[torch.Tensor] = []
    target_weights: list[float] = []
    developmental_drives: list[float] = []
    collapse_components: list[float] = []
    for row in assignments:
        child = by_object_id.get(str(row.get("object_id", "")))
        direction = _return_direction_vector(state, child) if isinstance(child, dict) else torch.zeros_like(collapse)
        collapse_component_t = torch.dot(direction, collapse_unit)
        tangent = direction - collapse_component_t * collapse_unit
        directions.append(direction)
        tangent_directions.append(tangent)

        safety = max(0.0, min(1.0, _as_float(row.get("jump_safety_pred", 0.0))))
        score = max(0.0, min(1.0, _as_float(row.get("writeback_score", 0.0))))
        recurrence = max(0.0, min(1.0, _as_float(row.get("recurrence_compatibility", 0.0))))
        raw_scale = max(0.0, min(1.0, _as_float(row.get("scale", 0.0))))
        movement = max(0.0, min(1.0, _as_float(row.get("causal_parent_phase_divergence", 0.0)) / 0.35))
        jump = max(0.0, min(1.0, _as_float(row.get("causal_world_jump_proxy", 0.0)) / 0.08))
        collapse_abs = min(1.0, abs(float(collapse_component_t.detach().cpu().item())) / float(collapse_norm.detach().cpu().item()))
        developmental_drive = max(
            0.0,
            min(
                1.0,
                movement * (0.45 + 0.55 * safety) * (0.45 + 0.55 * recurrence)
                - 0.32 * jump
                - 0.22 * collapse_abs,
            ),
        )
        target_weight = raw_scale * (0.20 + 0.80 * score) * (0.35 + 0.65 * safety) * (0.35 + 0.65 * developmental_drive)
        target_weights.append(float(max(0.0, target_weight)))
        developmental_drives.append(float(developmental_drive))
        collapse_components.append(float(collapse_component_t.detach().cpu().item()))

    if float(np.sum(target_weights)) <= 1.0e-8:
        target_weights = [1.0 for _ in assignments]
    target = torch.zeros_like(collapse)
    for tangent, weight in zip(tangent_directions, target_weights):
        target = target + float(weight) * tangent
    target_norm = torch.linalg.vector_norm(target)
    if float(target_norm.detach().cpu().item()) <= 1.0e-8:
        target = torch.stack(tangent_directions, dim=0).mean(dim=0)
        target_norm = torch.linalg.vector_norm(target)
    target_unit = target / target_norm.clamp_min(1.0e-8)

    raw_allocations: list[float] = []
    target_projections: list[float] = []
    family_counts: dict[str, int] = {}
    for row in assignments:
        key = str(row.get("local_family_key", ""))
        family_counts[key] = family_counts.get(key, 0) + 1

    for row, direction, tangent, dev, collapse_component in zip(
        assignments, directions, tangent_directions, developmental_drives, collapse_components
    ):
        raw_scale = max(0.0, min(1.0, _as_float(row.get("scale", 0.0))))
        safety = max(0.0, min(1.0, _as_float(row.get("jump_safety_pred", 0.0))))
        score = max(0.0, min(1.0, _as_float(row.get("writeback_score", 0.0))))
        tangent_norm = torch.linalg.vector_norm(tangent).clamp_min(1.0e-8)
        target_projection = float((torch.dot(tangent, target_unit) / tangent_norm).detach().cpu().item())
        target_projection = max(-1.0, min(1.0, target_projection))
        positive_projection = max(0.0, target_projection)
        collapse_penalty = min(1.0, abs(collapse_component) / float(collapse_norm.detach().cpu().item()))
        family_count = max(1, family_counts.get(str(row.get("local_family_key", "")), 1))
        redundancy_penalty = 1.0 + 0.05 * max(0, family_count - 1)
        if positive_projection >= 0.12:
            allocation = (
                raw_scale
                * (0.18 + 0.82 * positive_projection)
                * (0.35 + 0.65 * dev)
                * (0.45 + 0.55 * safety)
                * (0.40 + 0.60 * score)
                * (1.0 - 0.35 * collapse_penalty)
                / redundancy_penalty
            )
        else:
            allocation = raw_scale * 0.01 * max(0.0, 1.0 + target_projection)
        row["target_projection"] = target_projection
        row["target_coalition_member"] = 1.0 if positive_projection >= 0.12 else 0.0
        row["collapse_component"] = collapse_component
        raw_allocations.append(float(max(0.0, allocation)))
        target_projections.append(target_projection)

    budget = float(0.72 * max(1, len(assignments)))
    total_raw = float(np.sum(raw_allocations))
    shrink = min(1.0, budget / max(total_raw, 1.0e-8))
    for row, allocation in zip(assignments, raw_allocations):
        allocated_scale = max(0.0, min(1.0, float(allocation * shrink)))
        row["allocated_scale"] = allocated_scale
        row["scale"] = allocated_scale
        child = by_object_id.get(str(row.get("object_id", "")))
        if isinstance(child, dict):
            child["assay_writeback_scale"] = allocated_scale

    coalition_members = [row for row in assignments if _as_float(row.get("target_coalition_member", 0.0)) > 0.5]
    member_projection = [_as_float(row.get("target_projection", 0.0)) for row in coalition_members]
    opposition = [max(0.0, -_as_float(row.get("target_projection", 0.0))) for row in assignments]
    return {
        "allocation_enabled": True,
        "allocation_profile": "learned_soft_target_coalition",
        "allocation_budget_mean_scale": 0.72,
        "allocation_total_raw_authority": total_raw,
        "allocation_total_allocated_authority": float(np.sum([_as_float(row.get("allocated_scale", 0.0)) for row in assignments])),
        "allocation_shrink_factor": float(shrink),
        "mean_overlap_load": 0.0,
        "mean_redundancy_penalty": 1.0,
        "mean_developmental_drive": _mean(developmental_drives),
        "directional_coalition_size": float(len(coalition_members)),
        "directional_alignment_mean": _mean(member_projection),
        "directional_opposition_mean": _mean(opposition),
        "target_projection_mean": _mean(target_projections),
        "target_norm": float(target_norm.detach().cpu().item()),
        "collapse_component_abs_mean": _mean([abs(value) for value in collapse_components]),
    }


def _apply_contrastive_sibling_target_allocation(
    state: dict[str, Any],
    assignments: list[dict[str, Any]],
    by_object_id: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    if not assignments:
        return {
            "allocation_enabled": True,
            "allocation_profile": "learned_soft_contrastive_sibling_target",
            "target_source": "controlled_tangent_sibling",
            "mean_developmental_drive": 0.0,
            "directional_coalition_size": 0.0,
            "directional_alignment_mean": 0.0,
            "directional_opposition_mean": 0.0,
            "target_projection_mean": 0.0,
            "target_norm": 0.0,
            "contrastive_target_projection_mean": 0.0,
            "contrastive_target_norm": 0.0,
        }

    collapse = _collapse_direction_vector(state)
    collapse_norm = torch.linalg.vector_norm(collapse).clamp_min(1.0e-8)
    collapse_unit = collapse / collapse_norm
    target = _controlled_sibling_target_vector(state)
    target_norm = torch.linalg.vector_norm(target)
    target_unit = target / target_norm.clamp_min(1.0e-8)

    directions: list[torch.Tensor] = []
    tangent_directions: list[torch.Tensor] = []
    developmental_drives: list[float] = []
    collapse_components: list[float] = []
    target_projections: list[float] = []
    for row in assignments:
        child = by_object_id.get(str(row.get("object_id", "")))
        direction = _return_direction_vector(state, child) if isinstance(child, dict) else torch.zeros_like(collapse)
        collapse_component_t = torch.dot(direction, collapse_unit)
        tangent = direction - collapse_component_t * collapse_unit
        tangent_norm = torch.linalg.vector_norm(tangent).clamp_min(1.0e-8)
        target_projection = float((torch.dot(tangent, target_unit) / tangent_norm).detach().cpu().item())
        target_projection = max(-1.0, min(1.0, target_projection))

        safety = max(0.0, min(1.0, _as_float(row.get("jump_safety_pred", 0.0))))
        recurrence = max(0.0, min(1.0, _as_float(row.get("recurrence_compatibility", 0.0))))
        movement = max(0.0, min(1.0, _as_float(row.get("causal_parent_phase_divergence", 0.0)) / 0.35))
        jump = max(0.0, min(1.0, _as_float(row.get("causal_world_jump_proxy", 0.0)) / 0.08))
        collapse_abs = min(1.0, abs(float(collapse_component_t.detach().cpu().item())) / float(collapse_norm.detach().cpu().item()))
        positive_projection = max(0.0, target_projection)
        developmental_drive = max(
            0.0,
            min(
                1.0,
                movement * (0.42 + 0.58 * safety) * (0.35 + 0.65 * recurrence) * (0.25 + 0.75 * positive_projection)
                - 0.30 * jump
                - 0.20 * collapse_abs,
            ),
        )

        directions.append(direction)
        tangent_directions.append(tangent)
        developmental_drives.append(float(developmental_drive))
        collapse_components.append(float(collapse_component_t.detach().cpu().item()))
        target_projections.append(float(target_projection))

    family_counts: dict[str, int] = {}
    for row in assignments:
        key = str(row.get("local_family_key", ""))
        family_counts[key] = family_counts.get(key, 0) + 1

    raw_allocations: list[float] = []
    for row, target_projection, dev, collapse_component in zip(
        assignments, target_projections, developmental_drives, collapse_components
    ):
        raw_scale = max(0.0, min(1.0, _as_float(row.get("scale", 0.0))))
        safety = max(0.0, min(1.0, _as_float(row.get("jump_safety_pred", 0.0))))
        score = max(0.0, min(1.0, _as_float(row.get("writeback_score", 0.0))))
        recurrence = max(0.0, min(1.0, _as_float(row.get("recurrence_compatibility", 0.0))))
        positive_projection = max(0.0, float(target_projection))
        collapse_penalty = min(1.0, abs(float(collapse_component)) / float(collapse_norm.detach().cpu().item()))
        family_count = max(1, family_counts.get(str(row.get("local_family_key", "")), 1))
        redundancy_penalty = 1.0 + 0.05 * max(0, family_count - 1)
        if positive_projection >= 0.02:
            allocation = (
                raw_scale
                * (0.10 + 0.90 * positive_projection)
                * (0.32 + 0.68 * dev)
                * (0.44 + 0.56 * safety)
                * (0.36 + 0.64 * score)
                * (0.35 + 0.65 * recurrence)
                * (1.0 - 0.38 * collapse_penalty)
                / redundancy_penalty
            )
        else:
            allocation = raw_scale * 0.004 * max(0.0, 1.0 + float(target_projection))
        row["target_projection"] = float(target_projection)
        row["contrastive_target_projection"] = float(target_projection)
        row["target_coalition_member"] = 1.0 if positive_projection >= 0.02 else 0.0
        row["collapse_component"] = float(collapse_component)
        raw_allocations.append(float(max(0.0, allocation)))

    budget_mean = 0.78
    budget = float(budget_mean * max(1, len(assignments)))
    total_raw = float(np.sum(raw_allocations))
    shrink = min(1.0, budget / max(total_raw, 1.0e-8))
    for row, allocation in zip(assignments, raw_allocations):
        allocated_scale = max(0.0, min(1.0, float(allocation * shrink)))
        row["allocated_scale"] = allocated_scale
        row["scale"] = allocated_scale
        child = by_object_id.get(str(row.get("object_id", "")))
        if isinstance(child, dict):
            child["assay_writeback_scale"] = allocated_scale

    coalition_members = [row for row in assignments if _as_float(row.get("target_coalition_member", 0.0)) > 0.5]
    member_projection = [_as_float(row.get("target_projection", 0.0)) for row in coalition_members]
    opposition = [max(0.0, -_as_float(row.get("target_projection", 0.0))) for row in assignments]
    return {
        "allocation_enabled": True,
        "allocation_profile": "learned_soft_contrastive_sibling_target",
        "allocation_budget_mean_scale": budget_mean,
        "allocation_total_raw_authority": total_raw,
        "allocation_total_allocated_authority": float(np.sum([_as_float(row.get("allocated_scale", 0.0)) for row in assignments])),
        "allocation_shrink_factor": float(shrink),
        "target_source": "controlled_tangent_sibling",
        "mean_overlap_load": 0.0,
        "mean_redundancy_penalty": 1.0,
        "mean_developmental_drive": _mean(developmental_drives),
        "directional_coalition_size": float(len(coalition_members)),
        "directional_alignment_mean": _mean(member_projection),
        "directional_opposition_mean": _mean(opposition),
        "target_projection_mean": _mean(target_projections),
        "target_norm": float(target_norm.detach().cpu().item()),
        "contrastive_target_projection_mean": _mean(target_projections),
        "contrastive_target_norm": float(target_norm.detach().cpu().item()),
        "collapse_component_abs_mean": _mean([abs(value) for value in collapse_components]),
    }


def _apply_target_authority_scout_allocation(
    state: dict[str, Any],
    assignments: list[dict[str, Any]],
    by_object_id: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    if not assignments:
        return {
            "allocation_enabled": True,
            "allocation_profile": "learned_soft_target_authority_scout",
            "target_source": "controlled_tangent_sibling",
            "mean_developmental_drive": 0.0,
            "directional_coalition_size": 0.0,
            "directional_alignment_mean": 0.0,
            "directional_opposition_mean": 0.0,
            "target_projection_mean": 0.0,
            "target_norm": 0.0,
            "contrastive_target_projection_mean": 0.0,
            "contrastive_target_norm": 0.0,
        }

    collapse = _collapse_direction_vector(state)
    collapse_norm = torch.linalg.vector_norm(collapse).clamp_min(1.0e-8)
    collapse_unit = collapse / collapse_norm
    target = _controlled_sibling_target_vector(state)
    target_norm = torch.linalg.vector_norm(target)
    target_unit = target / target_norm.clamp_min(1.0e-8)

    target_projections: list[float] = []
    collapse_components: list[float] = []
    developmental_drives: list[float] = []
    raw_allocations: list[float] = []
    for row in assignments:
        child = by_object_id.get(str(row.get("object_id", "")))
        direction = _return_direction_vector(state, child) if isinstance(child, dict) else torch.zeros_like(collapse)
        collapse_component_t = torch.dot(direction, collapse_unit)
        tangent = direction - collapse_component_t * collapse_unit
        tangent_norm = torch.linalg.vector_norm(tangent).clamp_min(1.0e-8)
        projection = float((torch.dot(tangent, target_unit) / tangent_norm).detach().cpu().item())
        projection = max(-1.0, min(1.0, projection))
        positive_projection = max(0.0, projection)
        collapse_penalty = min(1.0, abs(float(collapse_component_t.detach().cpu().item())) / float(collapse_norm.detach().cpu().item()))
        safety = max(0.0, min(1.0, _as_float(row.get("jump_safety_pred", 0.0))))
        score = max(0.0, min(1.0, _as_float(row.get("writeback_score", 0.0))))
        recurrence = max(0.0, min(1.0, _as_float(row.get("recurrence_compatibility", 0.0))))
        raw_scale = max(0.0, min(1.0, _as_float(row.get("scale", 0.0))))
        movement = max(0.0, min(1.0, _as_float(row.get("causal_parent_phase_divergence", 0.0)) / 0.35))
        jump = max(0.0, min(1.0, _as_float(row.get("causal_world_jump_proxy", 0.0)) / 0.08))
        developmental_drive = max(
            0.0,
            min(
                1.0,
                (0.18 + 0.82 * positive_projection)
                * (0.30 + 0.70 * movement)
                * (0.45 + 0.55 * recurrence)
                * (0.55 + 0.45 * safety)
                - 0.35 * jump
                - 0.18 * collapse_penalty,
            ),
        )
        if positive_projection >= 0.05:
            authority_floor = 0.34 + 0.54 * positive_projection
            allocation = (
                max(raw_scale, authority_floor)
                * (0.55 + 0.45 * safety)
                * (0.45 + 0.55 * score)
                * (0.45 + 0.55 * recurrence)
                * (1.0 - 0.30 * collapse_penalty)
            )
        elif positive_projection >= 0.02:
            allocation = raw_scale * (0.08 + 0.42 * positive_projection) * (0.50 + 0.50 * safety)
        else:
            allocation = raw_scale * 0.002 * max(0.0, 1.0 + projection)
        row["target_projection"] = projection
        row["contrastive_target_projection"] = projection
        row["target_coalition_member"] = 1.0 if positive_projection >= 0.05 else 0.0
        row["collapse_component"] = float(collapse_component_t.detach().cpu().item())
        row["target_authority_scout"] = 1.0
        target_projections.append(float(projection))
        collapse_components.append(float(collapse_component_t.detach().cpu().item()))
        developmental_drives.append(float(developmental_drive))
        raw_allocations.append(float(max(0.0, allocation)))

    budget_mean = 1.20
    budget = float(budget_mean * max(1, len(assignments)))
    total_raw = float(np.sum(raw_allocations))
    shrink = min(1.0, budget / max(total_raw, 1.0e-8))
    for row, allocation in zip(assignments, raw_allocations):
        allocated_scale = max(0.0, min(1.0, float(allocation * shrink)))
        row["allocated_scale"] = allocated_scale
        row["scale"] = allocated_scale
        child = by_object_id.get(str(row.get("object_id", "")))
        if isinstance(child, dict):
            child["assay_writeback_scale"] = allocated_scale

    coalition_members = [row for row in assignments if _as_float(row.get("target_coalition_member", 0.0)) > 0.5]
    member_projection = [_as_float(row.get("target_projection", 0.0)) for row in coalition_members]
    opposition = [max(0.0, -_as_float(row.get("target_projection", 0.0))) for row in assignments]
    return {
        "allocation_enabled": True,
        "allocation_profile": "learned_soft_target_authority_scout",
        "allocation_budget_mean_scale": budget_mean,
        "allocation_total_raw_authority": total_raw,
        "allocation_total_allocated_authority": float(np.sum([_as_float(row.get("allocated_scale", 0.0)) for row in assignments])),
        "allocation_shrink_factor": float(shrink),
        "target_source": "controlled_tangent_sibling",
        "mean_overlap_load": 0.0,
        "mean_redundancy_penalty": 1.0,
        "mean_developmental_drive": _mean(developmental_drives),
        "directional_coalition_size": float(len(coalition_members)),
        "directional_alignment_mean": _mean(member_projection),
        "directional_opposition_mean": _mean(opposition),
        "target_projection_mean": _mean(target_projections),
        "target_norm": float(target_norm.detach().cpu().item()),
        "contrastive_target_projection_mean": _mean(target_projections),
        "contrastive_target_norm": float(target_norm.detach().cpu().item()),
        "collapse_component_abs_mean": _mean([abs(value) for value in collapse_components]),
    }


def _apply_ecology_allocation(
    state: dict[str, Any],
    assignments: list[dict[str, Any]],
    by_object_id: dict[str, dict[str, Any]],
    *,
    profile: str = "learned_soft_alloc",
) -> dict[str, Any]:
    if not assignments:
        return {
            "allocation_enabled": True,
            "allocation_budget_mean_scale": 0.0,
            "allocation_shrink_factor": 0.0,
            "mean_overlap_load": 0.0,
            "mean_redundancy_penalty": 0.0,
        }
    parent_mode1 = state["phase_modes"][..., 1, :]
    supports: list[torch.Tensor] = []
    for row in assignments:
        child = by_object_id.get(str(row.get("object_id", "")))
        support = child.get("mode_support") if isinstance(child, dict) else None
        if not torch.is_tensor(support) or support.shape != parent_mode1.shape[:-1]:
            support = torch.zeros(parent_mode1.shape[:-1], device=parent_mode1.device, dtype=parent_mode1.dtype)
        supports.append(support.detach().float().clamp(0.0, 1.0))
    support_stack = torch.stack(supports, dim=0) if supports else torch.zeros((0, *parent_mode1.shape[:-1]), device=parent_mode1.device)
    support_load = support_stack.sum(dim=0) if support_stack.numel() else torch.zeros(parent_mode1.shape[:-1], device=parent_mode1.device)

    family_counts: dict[str, int] = {}
    for row in assignments:
        key = str(row.get("local_family_key", ""))
        family_counts[key] = family_counts.get(key, 0) + 1

    loose = str(profile).endswith("_loose")
    developmental = "develop" in str(profile)
    if developmental:
        budget_mean = 0.74
        overlap_coeff = 0.055
        redundancy_coeff = 0.08
        score_power = 1.0
        safety_floor = 0.48
    else:
        budget_mean = 0.58 if loose else 0.34
        overlap_coeff = 0.12 if loose else 0.55
        redundancy_coeff = 0.12 if loose else 0.32
        score_power = 1.25 if loose else 2.0
        safety_floor = 0.38 if loose else 0.25

    raw_authorities: list[float] = []
    overlap_loads: list[float] = []
    redundancy_penalties: list[float] = []
    developmental_drives: list[float] = []
    for idx, row in enumerate(assignments):
        support = support_stack[idx]
        denom = float(support.sum().clamp_min(1.0e-8).item())
        overlap_load = float(((support_load - support) * support).sum().item() / denom)
        family_count = max(1, family_counts.get(str(row.get("local_family_key", "")), 1))
        redundancy_penalty = 1.0 + redundancy_coeff * max(0, family_count - 1)
        safety = max(0.0, min(1.0, _as_float(row.get("jump_safety_pred", 0.0))))
        score = max(0.0, min(1.0, _as_float(row.get("writeback_score", 0.0))))
        recurrence = max(0.0, min(1.0, _as_float(row.get("recurrence_compatibility", 0.0))))
        raw_scale = max(0.0, min(1.0, _as_float(row.get("scale", 0.0))))
        overlap_penalty = 1.0 + overlap_coeff * max(0.0, overlap_load)
        movement = max(0.0, min(1.0, _as_float(row.get("causal_parent_phase_divergence", 0.0)) / 0.35))
        jump = max(0.0, min(1.0, _as_float(row.get("causal_world_jump_proxy", 0.0)) / 0.08))
        developmental_drive = max(0.0, min(1.0, movement * (0.35 + 0.65 * safety) * (0.45 + 0.55 * recurrence) - 0.45 * jump))
        if developmental:
            quality = max(0.0, min(1.0, 0.35 * score + 0.45 * developmental_drive + 0.20 * recurrence))
            authority = raw_scale * (0.08 + 0.92 * quality) * (safety_floor + (1.0 - safety_floor) * safety)
        else:
            authority = raw_scale * (0.15 + 0.85 * (score**score_power)) * (safety_floor + (1.0 - safety_floor) * safety)
        authority = authority / (overlap_penalty * redundancy_penalty)
        raw_authorities.append(float(max(0.0, authority)))
        overlap_loads.append(overlap_load)
        redundancy_penalties.append(redundancy_penalty)
        developmental_drives.append(float(developmental_drive))

    child_count = max(1, len(assignments))
    total_authority = float(np.sum(raw_authorities))
    budget = float(budget_mean * child_count)
    shrink = min(1.0, budget / max(total_authority, 1.0e-8))
    for row, authority in zip(assignments, raw_authorities):
        allocated_scale = max(0.0, min(1.0, float(authority * shrink)))
        row["allocated_scale"] = allocated_scale
        row["scale"] = allocated_scale
        child = by_object_id.get(str(row.get("object_id", "")))
        if isinstance(child, dict):
            child["assay_writeback_scale"] = allocated_scale
    return {
        "allocation_enabled": True,
        "allocation_profile": str(profile),
        "allocation_budget_mean_scale": float(budget_mean),
        "allocation_total_raw_authority": total_authority,
        "allocation_total_allocated_authority": float(np.sum([_as_float(row.get("allocated_scale", 0.0)) for row in assignments])),
        "allocation_shrink_factor": float(shrink),
        "mean_overlap_load": _mean(overlap_loads),
        "mean_redundancy_penalty": _mean(redundancy_penalties),
        "mean_developmental_drive": _mean(developmental_drives),
    }


def _assign_fixed_scales(state: dict[str, Any], scale: float) -> dict[str, Any]:
    assignments: list[dict[str, Any]] = []
    for child in state.get("child_worlds", []) or []:
        if not isinstance(child, dict) or not child.get("active", False):
            continue
        child["assay_writeback_scale"] = float(scale)
        assignments.append({"child_id": int(child.get("child_id", -1)), "scale": float(scale)})
    return {
        "object_count": len(assignments),
        "mean_score": float(scale),
        "mean_scale": float(scale),
        "mean_raw_scale": float(scale),
        "nonzero_scale_fraction": 1.0 if float(scale) > 1.0e-6 and assignments else 0.0,
        "allocation_enabled": False,
        "assignments": assignments,
    }


def run_manychild_recurrence(
    state: dict[str, Any],
    cfg: Any,
    *,
    learned_model: torch.nn.Module,
    causal_selector: dict[str, Any] | None,
    device: torch.device,
    variant: str,
    runtime_steps: int,
    writeback_threshold: float,
    soft_floor: float,
    soft_full: float,
    case_label: str,
    operator_gain: float,
) -> dict[str, Any]:
    sandbox = clone_circleworld_state(state)
    _force_children_writeback_ready(sandbox, cfg)
    initial_child_count = len(sandbox.get("child_worlds", []) or [])
    runtime_cfg = replace(
        cfg,
        branching_mode="native_multimode_childworld",
        child_max_worlds=max(int(cfg.child_max_worlds), int(initial_child_count)),
        child_assay_writeback_scale_enabled=True,
        child_assay_writeback_scale_default=1.0,
    )
    gate = _child_union_gate(sandbox)
    initial_mode0 = sandbox["phase_modes"][..., 0, :].clone()
    initial_mode1 = sandbox["phase_modes"][..., 1, :].clone()
    initial_alignment = _masked_mean((initial_mode0 * initial_mode1).sum(dim=-1), gate)
    step_rows: list[dict[str, Any]] = []
    for step_idx in range(max(0, int(runtime_steps))):
        _force_children_writeback_ready(sandbox, runtime_cfg)
        if variant == "blocked_zero":
            scale_info = _assign_fixed_scales(sandbox, 0.0)
        elif variant == "ungated":
            scale_info = _assign_fixed_scales(sandbox, 1.0)
        else:
            scale_info = _assign_learned_scales(
                sandbox,
                learned_model=learned_model,
                causal_selector=causal_selector,
                device=device,
                variant=variant,
                writeback_threshold=float(writeback_threshold),
                soft_floor=float(soft_floor),
                soft_full=float(soft_full),
                case_label=f"{case_label}:step{step_idx}",
                operator_gain=float(operator_gain),
            )
        sandbox, block, _ = circleworld_step(sandbox, runtime_cfg, mode="native_multimode_childworld")
        step_rows.append(
            {
                "step": int(step_idx),
                "scale_info": scale_info,
                "child_world_count": _as_float(block.get("child_world_count", 0.0)),
                "child_writeback_mass": _as_float(block.get("child_writeback_mass", 0.0)),
                "child_phase_writeback_delta_mass": _as_float(block.get("child_phase_writeback_delta_mass", 0.0)),
                "child_support_writeback_mass": _as_float(block.get("child_support_writeback_mass", 0.0)),
                "child_assay_writeback_scale_mean": _as_float(block.get("child_assay_writeback_scale_mean", 0.0)),
                "child_parent_divergence": _as_float(block.get("child_parent_divergence", 0.0)),
                "child_sibling_divergence": _as_float(block.get("child_sibling_divergence", 0.0)),
            }
        )
    final_mode1 = sandbox["phase_modes"][..., 1, :]
    final_alignment = _masked_mean((initial_mode0 * final_mode1).sum(dim=-1), gate)
    events = list(sandbox.get("child_event_history", []))
    return {
        "variant": variant,
        "runtime_steps": int(runtime_steps),
        "initial_child_count": float(initial_child_count),
        "final_child_count": float(len(sandbox.get("child_worlds", []) or [])),
        "final_parent_phase_divergence": _masked_mean(_phase_delta_from_to(initial_mode1, final_mode1).abs(), gate),
        "final_world_jump_proxy": float(max(0.0, initial_alignment - final_alignment)),
        "mean_assignment_scale": _mean([_as_float(row["scale_info"].get("mean_scale", 0.0)) for row in step_rows]),
        "mean_assignment_raw_scale": _mean([_as_float(row["scale_info"].get("mean_raw_scale", 0.0)) for row in step_rows]),
        "mean_assignment_nonzero_fraction": _mean(
            [_as_float(row["scale_info"].get("nonzero_scale_fraction", 0.0)) for row in step_rows]
        ),
        "mean_assignment_score": _mean([_as_float(row["scale_info"].get("mean_score", 0.0)) for row in step_rows]),
        "mean_allocation_shrink_factor": _mean(
            [_as_float(row["scale_info"].get("allocation_shrink_factor", 1.0)) for row in step_rows]
        ),
        "mean_overlap_load": _mean([_as_float(row["scale_info"].get("mean_overlap_load", 0.0)) for row in step_rows]),
        "mean_developmental_drive": _mean(
            [_as_float(row["scale_info"].get("mean_developmental_drive", 0.0)) for row in step_rows]
        ),
        "mean_directional_coalition_size": _mean(
            [_as_float(row["scale_info"].get("directional_coalition_size", 0.0)) for row in step_rows]
        ),
        "mean_directional_alignment": _mean(
            [_as_float(row["scale_info"].get("directional_alignment_mean", 0.0)) for row in step_rows]
        ),
        "mean_directional_opposition": _mean(
            [_as_float(row["scale_info"].get("directional_opposition_mean", 0.0)) for row in step_rows]
        ),
        "mean_target_projection": _mean(
            [_as_float(row["scale_info"].get("target_projection_mean", 0.0)) for row in step_rows]
        ),
        "mean_target_norm": _mean([_as_float(row["scale_info"].get("target_norm", 0.0)) for row in step_rows]),
        "mean_contrastive_target_projection": _mean(
            [_as_float(row["scale_info"].get("contrastive_target_projection_mean", 0.0)) for row in step_rows]
        ),
        "mean_contrastive_target_norm": _mean(
            [_as_float(row["scale_info"].get("contrastive_target_norm", 0.0)) for row in step_rows]
        ),
        "mean_collapse_component_abs": _mean(
            [_as_float(row["scale_info"].get("collapse_component_abs_mean", 0.0)) for row in step_rows]
        ),
        "mean_child_writeback_mass": _mean([_as_float(row.get("child_writeback_mass", 0.0)) for row in step_rows]),
        "sum_child_writeback_mass": float(np.sum([_as_float(row.get("child_writeback_mass", 0.0)) for row in step_rows])),
        "mean_child_phase_writeback_delta_mass": _mean(
            [_as_float(row.get("child_phase_writeback_delta_mass", 0.0)) for row in step_rows]
        ),
        "mean_child_support_writeback_mass": _mean(
            [_as_float(row.get("child_support_writeback_mass", 0.0)) for row in step_rows]
        ),
        "writeback_event_count": float(len([event for event in events if event.get("event") == "writeback"])),
        "collapse_event_count": float(len([event for event in events if event.get("event") == "collapse"])),
        "initial_parent0_alignment": float(initial_alignment),
        "final_parent0_alignment": float(final_alignment),
        "step_rows": step_rows,
    }


def run_manychild_runtime_sandbox(
    *,
    config_path: Path,
    out_dir: Path,
    device_name: str,
    seed_source: str,
    seeds: list[int],
    time_steps: int,
    warmup_depth: int,
    operator_gain: float,
    grandchild_depth: int,
    grandchild_parts: int,
    runtime_steps: int,
    learned_branch_checkpoint: Path,
    causal_selector_checkpoint: Path,
    writeback_threshold: float,
    soft_floor: float,
    soft_full: float,
    causal_residual_weight: float,
    causal_residual_cap: float,
    causal_membrane_mode: str,
) -> dict[str, str]:
    cfg = _load_circle_cfg(config_path)
    device = _safe_device(device_name)
    rafa_core = make_seed_rafa(dev=device.type) if seed_source == "naked_rafa" else None
    learned_model = _load_model(learned_branch_checkpoint, device)
    causal_selector = _load_causal_selector(causal_selector_checkpoint, device)
    rows: list[dict[str, Any]] = []
    previous = dict(ASSAY_KILLSWITCH)
    ASSAY_KILLSWITCH.clear()
    ASSAY_KILLSWITCH.update(DEFAULT_ASSAY_KILLSWITCH)
    ASSAY_KILLSWITCH.update(
        {
            "enable_grandchild_probe": True,
            "grandchild_probe_depth": int(grandchild_depth),
            "grandchild_probe_parts": int(grandchild_parts),
            "grandchild_probe_flatten_to_child_worlds": True,
            "disable_branch_childworld_runtime": True,
            "disable_branch_pre_unroll": True,
        }
    )
    try:
        for seed in seeds:
            prepared = _prepare_case(
                cfg=cfg,
                rafa_core=rafa_core,
                source=seed_source,
                seed=int(seed),
                time_steps=int(time_steps),
                warmup_depth=int(warmup_depth),
                device=device,
                operator_gain=float(operator_gain),
                grandchild_depth=int(grandchild_depth),
                grandchild_parts=int(grandchild_parts),
                causal_selector=causal_selector,
                causal_residual_weight=float(causal_residual_weight),
                causal_residual_cap=float(causal_residual_cap),
                causal_membrane_mode=str(causal_membrane_mode),
            )
            base = {
                "source": seed_source,
                "seed": int(seed),
                "config_path": str(config_path),
                "fork_depth": int(prepared["fork_depth"]),
                "grandchild_depth": int(grandchild_depth),
                "grandchild_parts": int(grandchild_parts),
                "initial_law_object_count": int(prepared["case_row"].get("law_object_count", 0)),
            }
            for variant in (
                "blocked_zero",
                "ungated",
                "learned_hard",
                "learned_soft",
                "learned_hard_alloc",
                "learned_soft_alloc",
                "learned_soft_alloc_loose",
                "learned_soft_develop_alloc",
                "learned_soft_directional_coalition",
                "learned_soft_directional_coalition_boost",
                "learned_soft_target_coalition",
                "learned_soft_contrastive_sibling_target",
                "learned_soft_target_authority_scout",
            ):
                metrics = run_manychild_recurrence(
                    prepared["state"],
                    cfg,
                    learned_model=learned_model,
                    causal_selector=causal_selector,
                    device=device,
                    variant=variant,
                    runtime_steps=int(runtime_steps),
                    writeback_threshold=float(writeback_threshold),
                    soft_floor=float(soft_floor),
                    soft_full=float(soft_full),
                    case_label=f"{seed_source}_{seed}",
                    operator_gain=float(operator_gain),
                )
                row = dict(base)
                row.update(metrics)
                rows.append(row)
    finally:
        ASSAY_KILLSWITCH.clear()
        ASSAY_KILLSWITCH.update(previous)

    aggregate = _aggregate_rows(rows)
    payload = {
        "schema": "learned_gated_manychild_runtime_sandbox_v0",
        "config_path": str(config_path),
        "learned_branch_checkpoint": str(learned_branch_checkpoint),
        "causal_selector_checkpoint": str(causal_selector_checkpoint),
        "seed_source": seed_source,
        "seeds": [int(seed) for seed in seeds],
        "time_steps": int(time_steps),
        "warmup_depth": int(warmup_depth),
        "operator_gain": float(operator_gain),
        "grandchild_depth": int(grandchild_depth),
        "grandchild_parts": int(grandchild_parts),
        "runtime_steps": int(runtime_steps),
        "writeback_threshold": float(writeback_threshold),
        "soft_floor": float(soft_floor),
        "soft_full": float(soft_full),
        "aggregate": aggregate,
        "rows": rows,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "learned_gated_manychild_runtime_sandbox.json"
    md_path = out_dir / "LEARNED_GATED_MANYCHILD_RUNTIME_SANDBOX.md"
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    _write_markdown(md_path, payload)
    return {"json": str(json_path), "markdown": str(md_path), "status": aggregate["status"]}


def _aggregate_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    variants = sorted({str(row.get("variant", "")) for row in rows})
    by_variant: dict[str, dict[str, float]] = {}
    for variant in variants:
        subset = [row for row in rows if str(row.get("variant", "")) == variant]
        by_variant[variant] = {
            "row_count": float(len(subset)),
            "mean_assignment_scale": _mean([_as_float(row.get("mean_assignment_scale", 0.0)) for row in subset]),
            "mean_assignment_raw_scale": _mean([_as_float(row.get("mean_assignment_raw_scale", 0.0)) for row in subset]),
            "mean_assignment_nonzero_fraction": _mean(
                [_as_float(row.get("mean_assignment_nonzero_fraction", 0.0)) for row in subset]
            ),
            "mean_allocation_shrink_factor": _mean(
                [_as_float(row.get("mean_allocation_shrink_factor", 1.0)) for row in subset]
            ),
            "mean_overlap_load": _mean([_as_float(row.get("mean_overlap_load", 0.0)) for row in subset]),
            "mean_developmental_drive": _mean([_as_float(row.get("mean_developmental_drive", 0.0)) for row in subset]),
            "mean_directional_coalition_size": _mean(
                [_as_float(row.get("mean_directional_coalition_size", 0.0)) for row in subset]
            ),
            "mean_directional_alignment": _mean([_as_float(row.get("mean_directional_alignment", 0.0)) for row in subset]),
            "mean_directional_opposition": _mean([_as_float(row.get("mean_directional_opposition", 0.0)) for row in subset]),
            "mean_target_projection": _mean([_as_float(row.get("mean_target_projection", 0.0)) for row in subset]),
            "mean_target_norm": _mean([_as_float(row.get("mean_target_norm", 0.0)) for row in subset]),
            "mean_contrastive_target_projection": _mean(
                [_as_float(row.get("mean_contrastive_target_projection", 0.0)) for row in subset]
            ),
            "mean_contrastive_target_norm": _mean(
                [_as_float(row.get("mean_contrastive_target_norm", 0.0)) for row in subset]
            ),
            "mean_collapse_component_abs": _mean(
                [_as_float(row.get("mean_collapse_component_abs", 0.0)) for row in subset]
            ),
            "mean_final_parent_phase_divergence": _mean(
                [_as_float(row.get("final_parent_phase_divergence", 0.0)) for row in subset]
            ),
            "mean_final_world_jump_proxy": _mean([_as_float(row.get("final_world_jump_proxy", 0.0)) for row in subset]),
            "mean_child_writeback_mass": _mean([_as_float(row.get("mean_child_writeback_mass", 0.0)) for row in subset]),
            "mean_sum_child_writeback_mass": _mean([_as_float(row.get("sum_child_writeback_mass", 0.0)) for row in subset]),
            "mean_child_phase_writeback_delta_mass": _mean(
                [_as_float(row.get("mean_child_phase_writeback_delta_mass", 0.0)) for row in subset]
            ),
            "mean_writeback_event_count": _mean([_as_float(row.get("writeback_event_count", 0.0)) for row in subset]),
            "mean_collapse_event_count": _mean([_as_float(row.get("collapse_event_count", 0.0)) for row in subset]),
        }
    blocked = by_variant.get("blocked_zero", {})
    ungated = by_variant.get("ungated", {})
    for variant, summary in by_variant.items():
        parent = summary.get("mean_final_parent_phase_divergence", 0.0)
        jump = summary.get("mean_final_world_jump_proxy", 0.0)
        b_parent = blocked.get("mean_final_parent_phase_divergence", 0.0)
        b_jump = blocked.get("mean_final_world_jump_proxy", 0.0)
        u_parent = ungated.get("mean_final_parent_phase_divergence", 0.0)
        u_jump = ungated.get("mean_final_world_jump_proxy", 0.0)
        summary["parent_divergence_lift_vs_blocked"] = parent - b_parent
        summary["world_jump_lift_vs_blocked"] = jump - b_jump
        summary["parent_retention_vs_ungated"] = parent / u_parent if u_parent > 1.0e-8 else 0.0
        summary["world_jump_retention_vs_ungated"] = jump / u_jump if u_jump > 1.0e-8 else 0.0
    return {
        "status": "pass_learned_gated_manychild_runtime_sandbox" if rows else "fail_no_rows",
        "row_count": len(rows),
        "variant_count": len(variants),
        "variants": by_variant,
    }


def _write_markdown(path: Path, payload: dict[str, Any]) -> None:
    agg = payload["aggregate"]
    lines = [
        "# Learned-Gated Many-Child Runtime Sandbox",
        "",
        "This assay assigns writeback scales to all live child worlds before each actual `circleworld_step()` call.",
        "",
        "## Summary",
        "",
        f"- status: `{agg['status']}`",
        f"- rows: `{agg['row_count']}`",
        f"- runtime steps: `{payload['runtime_steps']}`",
        f"- grandchild depth: `{payload['grandchild_depth']}`",
        "",
        "## Variants",
        "",
        "| variant | rows | scale | raw scale | nonzero frac | shrink | overlap | dev drive | coalition | align | target | ctarget | parent div | world jump | writeback | parent lift | jump lift |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for variant, summary in agg.get("variants", {}).items():
        lines.append(
            "| {variant} | {rows} | {scale} | {raw} | {nz} | {shrink} | {overlap} | {dev} | {coalition} | {align} | {target} | {ctarget} | {parent} | {jump} | {writeback} | {plift} | {jlift} |".format(
                variant=variant,
                rows=summary.get("row_count", 0.0),
                scale=summary.get("mean_assignment_scale", 0.0),
                raw=summary.get("mean_assignment_raw_scale", summary.get("mean_assignment_scale", 0.0)),
                nz=summary.get("mean_assignment_nonzero_fraction", 0.0),
                shrink=summary.get("mean_allocation_shrink_factor", 1.0),
                overlap=summary.get("mean_overlap_load", 0.0),
                dev=summary.get("mean_developmental_drive", 0.0),
                coalition=summary.get("mean_directional_coalition_size", 0.0),
                align=summary.get("mean_directional_alignment", 0.0),
                target=summary.get("mean_target_projection", 0.0),
                ctarget=summary.get("mean_contrastive_target_projection", 0.0),
                parent=summary.get("mean_final_parent_phase_divergence", 0.0),
                jump=summary.get("mean_final_world_jump_proxy", 0.0),
                writeback=summary.get("mean_child_writeback_mass", 0.0),
                plift=summary.get("parent_divergence_lift_vs_blocked", 0.0),
                jlift=summary.get("world_jump_lift_vs_blocked", 0.0),
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description="Run learned gates over all live child worlds in actual Circleworld recurrence.")
    ap.add_argument("--config", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--learned-branch-checkpoint", required=True)
    ap.add_argument("--causal-selector-checkpoint", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed-source", choices=["synthetic", "naked_rafa"], default="naked_rafa")
    ap.add_argument("--seeds", default="9114,9115,9116")
    ap.add_argument("--time-steps", type=int, default=64)
    ap.add_argument("--warmup-depth", type=int, default=3)
    ap.add_argument("--operator-gain", type=float, default=0.85)
    ap.add_argument("--grandchild-depth", type=int, default=2)
    ap.add_argument("--grandchild-parts", type=int, default=2)
    ap.add_argument("--runtime-steps", type=int, default=4)
    ap.add_argument("--writeback-threshold", type=float, default=0.78)
    ap.add_argument("--soft-floor", type=float, default=0.70)
    ap.add_argument("--soft-full", type=float, default=0.88)
    ap.add_argument("--causal-residual-weight", type=float, default=0.20)
    ap.add_argument("--causal-residual-cap", type=float, default=0.05)
    ap.add_argument("--causal-membrane-mode", choices=["identity", "off"], default="off")
    args = ap.parse_args()
    result = run_manychild_runtime_sandbox(
        config_path=Path(args.config),
        out_dir=Path(args.out_dir),
        device_name=str(args.device),
        seed_source=str(args.seed_source),
        seeds=_parse_seeds(args.seeds),
        time_steps=int(args.time_steps),
        warmup_depth=int(args.warmup_depth),
        operator_gain=float(args.operator_gain),
        grandchild_depth=int(args.grandchild_depth),
        grandchild_parts=int(args.grandchild_parts),
        runtime_steps=int(args.runtime_steps),
        learned_branch_checkpoint=Path(args.learned_branch_checkpoint),
        causal_selector_checkpoint=Path(args.causal_selector_checkpoint),
        writeback_threshold=float(args.writeback_threshold),
        soft_floor=float(args.soft_floor),
        soft_full=float(args.soft_full),
        causal_residual_weight=float(args.causal_residual_weight),
        causal_residual_cap=float(args.causal_residual_cap),
        causal_membrane_mode=str(args.causal_membrane_mode),
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
