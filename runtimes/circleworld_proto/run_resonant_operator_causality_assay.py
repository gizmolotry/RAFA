from __future__ import annotations

import argparse
import json
import math
import sys
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

from ablate_formalization import _generate_seed_phase, make_seed_rafa  # noqa: E402
from causal_operator_selector import (  # noqa: E402
    CausalOperatorSelector,
    MultiHeadCausalOperatorSelector,
    causal_pair_feature_vector,
    selector_route_scores,
)
from circleworld import clone_circleworld_state  # noqa: E402
from evaluate_circleworld import _load_circle_cfg, _safe_device  # noqa: E402
from resonant_law_objects import (  # noqa: E402
    LAW_OBJECT_FEATURE_NAMES,
    law_object_feature_vector,
    law_objects_from_state,
    run_retrieval_assay,
    score_query_to_law_object,
)
from run_contrastive_structural_embedding_probe import (  # noqa: E402
    ContrastiveLawEmbedding,
    operator_safety_proxy,
    recurrence_compatibility,
)
from rafa_math_tools import phasor_apply_delta, phasor_normalize  # noqa: E402
from test_nested_commitment import (  # noqa: E402
    ASSAY_KILLSWITCH,
    DEFAULT_ASSAY_KILLSWITCH,
    _apply_assay_child_partitioning,
    _generate_seed_magnitude,
    _run_depth_trace,
)


def _as_float(value: Any, default: float = 0.0) -> float:
    if torch.is_tensor(value):
        if value.numel() == 0:
            return float(default)
        return float(value.detach().float().mean().cpu().item())
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _phase_delta_from_to(src: torch.Tensor, dst: torch.Tensor) -> torch.Tensor:
    dot = (src * dst).sum(dim=-1).clamp(-1.0, 1.0)
    cross = src[..., 0] * dst[..., 1] - src[..., 1] * dst[..., 0]
    return torch.atan2(cross, dot)


def _masked_mean(value: torch.Tensor, mask: torch.Tensor | None = None) -> float:
    if mask is None or mask.shape != value.shape:
        return float(value.detach().float().mean().cpu().item())
    weight = mask.detach().float().clamp(0.0, 1.0)
    denom = float(weight.sum().clamp_min(1.0e-8).item())
    return float((value.detach().float() * weight).sum().item() / denom)


def _select_fork_state(states: list[Any]) -> tuple[int, dict[str, Any]]:
    best_idx = -1
    best_count = -1
    best_state: dict[str, Any] | None = None
    for idx, state in enumerate(states):
        if not isinstance(state, dict) or "phase_modes" not in state:
            continue
        children = [child for child in state.get("child_worlds", []) or [] if isinstance(child, dict)]
        active_count = sum(1 for child in children if child.get("active", False))
        count = active_count if active_count else len(children)
        if count > best_count:
            best_idx = idx
            best_count = count
            best_state = state
    if best_state is None:
        raise RuntimeError("No multimode state with child worlds was found for operator causality assay.")
    return best_idx, clone_circleworld_state(best_state)


def _object_child_mapping(state: dict[str, Any], *, case: str, branch: str) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    objects = law_objects_from_state(state, context={"case": case, "branch": branch, "family": "operator", "source": "operator_state"})
    by_id: dict[str, dict[str, Any]] = {}
    for child, obj in zip(state.get("child_worlds", []) or [], objects):
        if isinstance(child, dict):
            by_id[str(obj.get("object_id", ""))] = child
    return objects, by_id


def _select_operator_triplet(
    objects: list[dict[str, Any]],
    retrieval: dict[str, Any],
) -> tuple[str, str, str, dict[str, Any] | None, dict[str, Any] | None]:
    rows = list(retrieval.get("rows", []) or [])
    rows.sort(
        key=lambda row: (
            float(row.get("top1_family_hit_excluding_self", 0.0)),
            float(row.get("resonance_margin", 0.0)),
            float(row.get("top_score", 0.0)),
        ),
        reverse=True,
    )
    if rows:
        row = rows[0]
        query_id = str(row.get("query_object_id", ""))
        selected_id = str(row.get("top_object_id", ""))
        by_id = {str(obj.get("object_id", "")): obj for obj in objects}
        query = by_id.get(query_id)
        selected_row: dict[str, Any] | None = None
        decoy_row: dict[str, Any] | None = None
        if query is not None:
            scored = [
                score_query_to_law_object(query, candidate, family_key_field="local_family_key")
                for candidate in objects
            ]
            scored.sort(key=lambda candidate_row: float(candidate_row.get("final_score", 0.0)), reverse=True)
            nonself = [candidate_row for candidate_row in scored if not bool(candidate_row.get("same_object", False))]
            nonself_positive = [candidate_row for candidate_row in nonself if bool(candidate_row.get("same_family", False))]
            selected_row = nonself_positive[0] if nonself_positive else (nonself[0] if nonself else None)
            if selected_row is not None:
                selected_id = str(selected_row.get("candidate_object_id", ""))
            decoys = [
                candidate_row
                for candidate_row in scored
                if not bool(candidate_row.get("same_object", False))
                and not bool(candidate_row.get("same_family", False))
            ]
            decoy_row = decoys[0] if decoys else None
        decoy_id = str(decoy_row.get("candidate_object_id", "")) if decoy_row else ""
        selected_report = dict(row)
        if selected_row is not None:
            selected_report.update(
                {
                    "top_object_id": str(selected_row.get("candidate_object_id", "")),
                    "top_score": float(selected_row.get("final_score", 0.0)),
                    "top_geometric_score": float(selected_row.get("geometric_score", 0.0)),
                    "top_learned_residual": float(selected_row.get("learned_residual", 0.0)),
                    "top1_self_hit": 0.0,
                    "top1_family_hit_including_self": 1.0 if bool(selected_row.get("same_family", False)) else 0.0,
                    "top1_family_hit_excluding_self": 1.0 if bool(selected_row.get("same_family", False)) else 0.0,
                }
            )
        return query_id, selected_id, decoy_id, selected_report, decoy_row
    if len(objects) >= 2:
        return str(objects[0].get("object_id", "")), str(objects[1].get("object_id", "")), "", None, None
    if objects:
        return str(objects[0].get("object_id", "")), str(objects[0].get("object_id", "")), "", None, None
    return "", "", "", None, None


def _load_learned_selector(checkpoint_path: Path | None, device: torch.device) -> dict[str, Any] | None:
    if checkpoint_path is None or not str(checkpoint_path):
        return None
    path = Path(checkpoint_path)
    if not path.exists():
        raise FileNotFoundError(f"Learned selector checkpoint not found: {path}")
    checkpoint = torch.load(path, map_location=device)
    config = dict(checkpoint.get("config", {}) or {})
    feature_names = list(checkpoint.get("feature_names", LAW_OBJECT_FEATURE_NAMES))
    feature_dim = len(feature_names) if feature_names else len(LAW_OBJECT_FEATURE_NAMES)
    model = ContrastiveLawEmbedding(
        feature_dim=feature_dim,
        embedding_dim=int(config.get("embedding_dim", 32)),
        hidden_dim=int(config.get("hidden_dim", 96)),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return {
        "checkpoint_path": str(path),
        "model": model,
        "feature_dim": int(feature_dim),
        "feature_names": feature_names,
        "config": config,
        "summary": dict(checkpoint.get("summary", {}) or {}),
    }


def _load_causal_selector(checkpoint_path: Path | None, device: torch.device) -> dict[str, Any] | None:
    if checkpoint_path is None or not str(checkpoint_path):
        return None
    path = Path(checkpoint_path)
    if not path.exists():
        raise FileNotFoundError(f"Causal selector checkpoint not found: {path}")
    checkpoint = torch.load(path, map_location=device)
    feature_dim = int(checkpoint.get("feature_dim", 0))
    hidden_dim = int(checkpoint.get("hidden_dim", 96))
    model_kind = str(checkpoint.get("model_kind", checkpoint.get("config", {}).get("model_kind", "single")))
    if feature_dim <= 0:
        raise ValueError(f"Causal selector checkpoint has invalid feature_dim: {path}")
    if model_kind == "multihead":
        model = MultiHeadCausalOperatorSelector(feature_dim=feature_dim, hidden_dim=hidden_dim).to(device)
    else:
        model = CausalOperatorSelector(feature_dim=feature_dim, hidden_dim=hidden_dim).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return {
        "checkpoint_path": str(path),
        "model": model,
        "feature_dim": feature_dim,
        "hidden_dim": hidden_dim,
        "model_kind": model_kind,
        "config": dict(checkpoint.get("config", {}) or {}),
        "summary": dict(checkpoint.get("summary", {}) or {}),
    }


def _feature_tensor_for_objects(objects: list[dict[str, Any]], feature_dim: int, device: torch.device) -> torch.Tensor:
    rows: list[list[float]] = []
    for obj in objects:
        values = list(law_object_feature_vector(obj))
        if len(values) < feature_dim:
            values.extend([0.0] * (feature_dim - len(values)))
        elif len(values) > feature_dim:
            values = values[:feature_dim]
        rows.append(values)
    if not rows:
        return torch.zeros((0, int(feature_dim)), dtype=torch.float32, device=device)
    return torch.tensor(rows, dtype=torch.float32, device=device)


def _select_recurrence_candidate(
    objects: list[dict[str, Any]],
    query_id: str,
) -> dict[str, Any] | None:
    by_id = {str(obj.get("object_id", "")): obj for obj in objects}
    query = by_id.get(query_id)
    if query is None:
        return None
    rows: list[dict[str, Any]] = []
    for candidate in objects:
        if str(candidate.get("object_id", "")) == query_id:
            continue
        base = score_query_to_law_object(query, candidate, family_key_field="local_family_key")
        base["recurrence_compatibility"] = float(recurrence_compatibility(query, candidate))
        base["candidate_safety_target"] = float(operator_safety_proxy(candidate))
        rows.append(base)
    rows.sort(
        key=lambda row: (
            float(row.get("recurrence_compatibility", 0.0)),
            float(row.get("geometric_score", 0.0)),
            float(row.get("candidate_safety_target", 0.0)),
        ),
        reverse=True,
    )
    return rows[0] if rows else None


def _select_learned_candidate(
    objects: list[dict[str, Any]],
    query_id: str,
    learned_selector: dict[str, Any] | None,
    *,
    residual_weight: float,
    safety_weight: float,
    residual_cap: float,
    membrane_mode: str,
    strong_recurrence_threshold: float,
    strong_geometric_threshold: float,
    device: torch.device,
) -> dict[str, Any] | None:
    if learned_selector is None:
        return None
    by_id = {str(obj.get("object_id", "")): idx for idx, obj in enumerate(objects)}
    query_idx = by_id.get(query_id)
    if query_idx is None:
        return None
    feature_dim = int(learned_selector["feature_dim"])
    features = _feature_tensor_for_objects(objects, feature_dim, device)
    if features.numel() == 0:
        return None
    model = learned_selector["model"]
    with torch.no_grad():
        embeddings, safety_pred = model(features)
    query = objects[query_idx]
    query_embedding = embeddings[query_idx]
    rows: list[dict[str, Any]] = []
    for idx, candidate in enumerate(objects):
        if idx == query_idx:
            continue
        base = score_query_to_law_object(query, candidate, family_key_field="local_family_key")
        same_structural = (
            str(base.get("query_structural_family_key", ""))
            == str(base.get("candidate_structural_family_key", ""))
        )
        recurrence = float(recurrence_compatibility(query, candidate))
        strong_recurrence_pass = (
            recurrence >= float(strong_recurrence_threshold)
            and float(base.get("geometric_score", 0.0)) >= float(strong_geometric_threshold)
        )
        membrane_pass = bool(base.get("same_family", False)) or same_structural or strong_recurrence_pass
        embedding_cosine = float((query_embedding * embeddings[idx]).sum().clamp(-1.0, 1.0).item())
        embedding_score = 0.5 + 0.5 * embedding_cosine
        candidate_safety = float(safety_pred[idx].clamp(0.0, 1.0).item())
        raw_learned_residual = float(residual_weight) * (embedding_score - 0.5)
        raw_learned_residual += float(safety_weight) * (candidate_safety - 0.5)
        cap = max(0.0, float(residual_cap))
        learned_residual = max(0.0, min(cap, raw_learned_residual)) if membrane_pass else 0.0
        hybrid_score = max(0.0, min(1.0, float(base.get("geometric_score", 0.0)) + learned_residual))
        base.update(
            {
                "learned_embedding_cosine": embedding_cosine,
                "learned_embedding_score": float(embedding_score),
                "learned_safety_pred": candidate_safety,
                "learned_safety_target": float(operator_safety_proxy(candidate)),
                "raw_learned_residual": float(raw_learned_residual),
                "learned_residual_cap": float(cap),
                "learned_residual": float(learned_residual),
                "learned_hybrid_score": float(hybrid_score),
                "recurrence_compatibility": recurrence,
                "learned_identity_membrane_pass": 1.0 if membrane_pass else 0.0,
                "learned_strong_recurrence_pass": 1.0 if strong_recurrence_pass else 0.0,
            }
        )
        rows.append(base)
    mode = str(membrane_mode or "identity").strip().lower()
    rows.sort(
        key=lambda row: (
            float(row.get("learned_identity_membrane_pass", 0.0)) if mode in {"identity", "prefer_identity"} else 0.0,
            float(row.get("learned_hybrid_score", 0.0)),
            float(row.get("learned_embedding_score", 0.0)),
            float(row.get("learned_safety_pred", 0.0)),
        ),
        reverse=True,
    )
    return rows[0] if rows else None


def _select_causal_candidate(
    objects: list[dict[str, Any]],
    query_id: str,
    causal_selector: dict[str, Any] | None,
    *,
    residual_weight: float,
    residual_cap: float,
    membrane_mode: str,
    strong_recurrence_threshold: float,
    strong_geometric_threshold: float,
    device: torch.device,
) -> dict[str, Any] | None:
    if causal_selector is None:
        return None
    by_id = {str(obj.get("object_id", "")): obj for obj in objects}
    query = by_id.get(query_id)
    if query is None:
        return None
    model = causal_selector["model"]
    feature_dim = int(causal_selector["feature_dim"])
    rows: list[dict[str, Any]] = []
    features: list[list[float]] = []
    for candidate in objects:
        candidate_id = str(candidate.get("object_id", ""))
        if candidate_id == query_id:
            continue
        base = score_query_to_law_object(query, candidate, family_key_field="local_family_key")
        same_structural = (
            str(base.get("query_structural_family_key", ""))
            == str(base.get("candidate_structural_family_key", ""))
        )
        recurrence = float(recurrence_compatibility(query, candidate))
        strong_recurrence_pass = (
            recurrence >= float(strong_recurrence_threshold)
            and float(base.get("geometric_score", 0.0)) >= float(strong_geometric_threshold)
        )
        membrane_pass = bool(base.get("same_family", False)) or same_structural or strong_recurrence_pass
        base.update(
            {
                "recurrence_compatibility": recurrence,
                "same_local_family": 1.0 if bool(base.get("same_family", False)) else 0.0,
                "same_structural_family": 1.0 if same_structural else 0.0,
                "identity_membrane_pass": 1.0 if membrane_pass else 0.0,
                "strong_recurrence_pass": 1.0 if strong_recurrence_pass else 0.0,
                "query_safety_proxy": float(operator_safety_proxy(query)),
                "candidate_safety_proxy": float(operator_safety_proxy(candidate)),
            }
        )
        values = causal_pair_feature_vector(query, candidate, base)
        if len(values) < feature_dim:
            values.extend([0.0] * (feature_dim - len(values)))
        elif len(values) > feature_dim:
            values = values[:feature_dim]
        rows.append(base)
        features.append(values)
    if not rows:
        return None
    feature_tensor = torch.tensor(features, dtype=torch.float32, device=device)
    with torch.no_grad():
        output = model(feature_tensor)
        model_scores = selector_route_scores(output).detach().float().cpu().tolist()
        movement_scores = output["movement"].detach().float().cpu().tolist() if isinstance(output, dict) else [0.0] * len(rows)
        jump_scores = output["jump_safety"].detach().float().cpu().tolist() if isinstance(output, dict) else [0.0] * len(rows)
        identity_scores = output["identity"].detach().float().cpu().tolist() if isinstance(output, dict) else [0.0] * len(rows)
    for row, model_score, movement_score, jump_score, identity_score in zip(
        rows, model_scores, movement_scores, jump_scores, identity_scores
    ):
        raw_residual = float(residual_weight) * (float(model_score) - 0.5)
        cap = max(0.0, float(residual_cap))
        membrane_pass = float(row.get("identity_membrane_pass", 0.0)) > 0.0
        residual = max(0.0, min(cap, raw_residual)) if membrane_pass else 0.0
        hybrid_score = max(0.0, min(1.0, float(row.get("geometric_score", 0.0)) + residual))
        row.update(
            {
                "causal_selector_model_score": float(model_score),
                "causal_selector_movement_pred": float(movement_score),
                "causal_selector_jump_safety_pred": float(jump_score),
                "causal_selector_identity_pred": float(identity_score),
                "causal_selector_raw_residual": float(raw_residual),
                "causal_selector_residual": float(residual),
                "causal_selector_residual_cap": float(cap),
                "causal_selector_hybrid_score": float(hybrid_score),
            }
        )
    mode = str(membrane_mode or "identity").strip().lower()
    rows.sort(
        key=lambda row: (
            float(row.get("identity_membrane_pass", 0.0)) if mode in {"identity", "prefer_identity"} else 0.0,
            float(row.get("causal_selector_hybrid_score", 0.0)),
            float(row.get("causal_selector_model_score", 0.0)),
        ),
        reverse=True,
    )
    return rows[0]


def _prefixed_metrics(prefix: str, metrics: dict[str, float]) -> dict[str, float]:
    return {f"{prefix}_{key}": _as_float(value) for key, value in metrics.items()}


def _apply_child_operator(state: dict[str, Any], child: dict[str, Any], gain: float) -> dict[str, float]:
    phase_modes = state["phase_modes"]
    parent_mode0 = phase_modes[..., 0, :]
    parent_mode1 = phase_modes[..., 1, :]
    child_phase = phasor_normalize(child["phase_state"])
    support = child.get("mode_support")
    if not torch.is_tensor(support) or support.shape != parent_mode1.shape[:-1]:
        support = torch.ones(parent_mode1.shape[:-1], device=parent_mode1.device, dtype=parent_mode1.dtype)
    support = support.clamp(0.0, 1.0)
    coherence = child.get("mode_coherence")
    if not torch.is_tensor(coherence) or coherence.shape != support.shape:
        coherence = torch.ones_like(support)
    gate = (support * (0.20 + 0.80 * coherence.clamp(0.0, 1.0))).clamp(0.0, 1.0)
    raw_delta = _phase_delta_from_to(parent_mode1, child_phase).clamp(-math.pi, math.pi)
    applied_delta = (float(gain) * gate * raw_delta).clamp(-math.pi, math.pi)
    candidate_mode1 = phasor_apply_delta(parent_mode1, applied_delta)
    inert_mode1 = parent_mode1
    candidate_phase_div = _masked_mean(_phase_delta_from_to(inert_mode1, candidate_mode1).abs(), gate)
    parent0_alignment_before = _masked_mean((parent_mode0 * inert_mode1).sum(dim=-1), gate)
    parent0_alignment_after = _masked_mean((parent_mode0 * candidate_mode1).sum(dim=-1), gate)
    qtrace_shift = 0.0
    mode_q_trace = state.get("mode_q_trace")
    child_q = child.get("q_trace")
    if torch.is_tensor(mode_q_trace) and torch.is_tensor(child_q) and mode_q_trace.dim() >= 5:
        parent_q = mode_q_trace[..., 1, :]
        if parent_q.shape == child_q.shape:
            qtrace_shift = _masked_mean((parent_q - child_q).abs().mean(dim=-1), gate)
    return {
        "operator_applied": 1.0,
        "operator_gain": float(gain),
        "support_gate_mean": float(gate.mean().item()),
        "support_gate_max": float(gate.max().item()),
        "raw_delta_abs_mean": _masked_mean(raw_delta.abs(), gate),
        "applied_delta_abs_mean": _masked_mean(applied_delta.abs(), gate),
        "parent_phase_divergence": candidate_phase_div,
        "support_shift": float(gate.mean().item()),
        "qtrace_shift_proxy": float(qtrace_shift),
        "parent0_alignment_before": float(parent0_alignment_before),
        "parent0_alignment_after": float(parent0_alignment_after),
        "world_jump_proxy": float(max(0.0, parent0_alignment_before - parent0_alignment_after)),
        "inert_parent_phase_divergence": 0.0,
    }


def _run_one_case(
    *,
    cfg: Any,
    rafa_core: Any,
    source: str,
    seed: int,
    time_steps: int,
    warmup_depth: int,
    device: torch.device,
    operator_gain: float,
    learned_selector: dict[str, Any] | None,
    learned_residual_weight: float,
    learned_safety_weight: float,
    learned_residual_cap: float,
    learned_membrane_mode: str,
    learned_strong_recurrence_threshold: float,
    learned_strong_geometric_threshold: float,
    causal_selector: dict[str, Any] | None,
    causal_residual_weight: float,
    causal_residual_cap: float,
    causal_membrane_mode: str,
) -> dict[str, Any]:
    phase_state = _generate_seed_phase(
        rafa_core=rafa_core if source == "naked_rafa" else None,
        batch_size=1,
        time_steps=time_steps,
        device=device,
        seed_source=source,
        seed=seed,
    ).detach()
    _ = _generate_seed_magnitude(1, int(phase_state.size(1)), int(phase_state.size(2)), device, seed)
    mode = cfg.branching_mode if cfg.branching_mode in {"native_multimode", "native_multimode_childworld"} else "native_multimode_childworld"
    trace = _run_depth_trace(phase_state, cfg=cfg, depth=warmup_depth, mode=mode)
    fork_idx, fork_state = _select_fork_state(list(trace.get("states", [])))
    branch_state, partition = _apply_assay_child_partitioning(fork_state, cfg)
    objects, by_id = _object_child_mapping(branch_state, case=f"{source}_{seed}", branch="operator")
    retrieval = run_retrieval_assay(objects, top_k=3)
    query_id, candidate_id, decoy_id, row, decoy_row = _select_operator_triplet(objects, retrieval)
    recurrence_row = _select_recurrence_candidate(objects, query_id)
    recurrence_id = str(recurrence_row.get("candidate_object_id", "")) if recurrence_row else ""
    learned_row = _select_learned_candidate(
        objects,
        query_id,
        learned_selector,
        residual_weight=float(learned_residual_weight),
        safety_weight=float(learned_safety_weight),
        residual_cap=float(learned_residual_cap),
        membrane_mode=str(learned_membrane_mode),
        strong_recurrence_threshold=float(learned_strong_recurrence_threshold),
        strong_geometric_threshold=float(learned_strong_geometric_threshold),
        device=device,
    )
    learned_id = str(learned_row.get("candidate_object_id", "")) if learned_row else ""
    causal_row = _select_causal_candidate(
        objects,
        query_id,
        causal_selector,
        residual_weight=float(causal_residual_weight),
        residual_cap=float(causal_residual_cap),
        membrane_mode=str(causal_membrane_mode),
        strong_recurrence_threshold=float(learned_strong_recurrence_threshold),
        strong_geometric_threshold=float(learned_strong_geometric_threshold),
        device=device,
    )
    causal_id = str(causal_row.get("candidate_object_id", "")) if causal_row else ""
    child = by_id.get(candidate_id) or by_id.get(query_id)
    if child is None:
        return {
            "source": source,
            "seed": int(seed),
            "fork_depth": int(fork_idx),
            "law_object_count": len(objects),
            "status": "fail_no_candidate_child",
            "retrieval": retrieval["summary"],
            "partition": partition,
        }
    metrics = _apply_child_operator(branch_state, child, gain=operator_gain)
    decoy_metrics: dict[str, float] = {}
    decoy_child = by_id.get(decoy_id)
    if decoy_child is not None:
        decoy_metrics = _apply_child_operator(branch_state, decoy_child, gain=operator_gain)
    recurrence_metrics: dict[str, float] = {}
    recurrence_child = by_id.get(recurrence_id)
    if recurrence_child is not None:
        recurrence_metrics = _apply_child_operator(branch_state, recurrence_child, gain=operator_gain)
    learned_metrics: dict[str, float] = {}
    learned_child = by_id.get(learned_id)
    if learned_child is not None:
        learned_metrics = _apply_child_operator(branch_state, learned_child, gain=operator_gain)
    causal_metrics: dict[str, float] = {}
    causal_child = by_id.get(causal_id)
    if causal_child is not None:
        causal_metrics = _apply_child_operator(branch_state, causal_child, gain=operator_gain)
    metrics.update(
        {
            "source": source,
            "seed": int(seed),
            "fork_depth": int(fork_idx),
            "law_object_count": len(objects),
            "query_object_id": query_id,
            "candidate_object_id": candidate_id,
            "recurrence_candidate_object_id": recurrence_id,
            "learned_candidate_object_id": learned_id,
            "causal_candidate_object_id": causal_id,
            "decoy_candidate_object_id": decoy_id,
            "selected_resonance_margin": _as_float(row.get("resonance_margin", 0.0)) if row else 0.0,
            "selected_top_score": _as_float(row.get("top_score", 0.0)) if row else 0.0,
            "selected_top_family_hit_excluding_self": _as_float(row.get("top1_family_hit_excluding_self", 0.0)) if row else 0.0,
            "recurrence_selector_score": _as_float(recurrence_row.get("recurrence_compatibility", 0.0)) if recurrence_row else 0.0,
            "recurrence_selector_geometric_score": _as_float(recurrence_row.get("geometric_score", 0.0)) if recurrence_row else 0.0,
            "recurrence_selector_same_local_family": 1.0
            if recurrence_row is not None and bool(recurrence_row.get("same_family", False))
            else 0.0,
            "recurrence_selector_same_structural_family": 1.0
            if recurrence_row is not None
            and str(recurrence_row.get("query_structural_family_key", ""))
            == str(recurrence_row.get("candidate_structural_family_key", ""))
            else 0.0,
            "learned_selector_available": 1.0 if learned_selector is not None else 0.0,
            "learned_selector_hybrid_score": _as_float(learned_row.get("learned_hybrid_score", 0.0)) if learned_row else 0.0,
            "learned_selector_embedding_score": _as_float(learned_row.get("learned_embedding_score", 0.0)) if learned_row else 0.0,
            "learned_selector_geometric_score": _as_float(learned_row.get("geometric_score", 0.0)) if learned_row else 0.0,
            "learned_selector_residual": _as_float(learned_row.get("learned_residual", 0.0)) if learned_row else 0.0,
            "learned_selector_raw_residual": _as_float(learned_row.get("raw_learned_residual", 0.0)) if learned_row else 0.0,
            "learned_selector_residual_cap": _as_float(learned_row.get("learned_residual_cap", 0.0)) if learned_row else 0.0,
            "learned_selector_safety_pred": _as_float(learned_row.get("learned_safety_pred", 0.0)) if learned_row else 0.0,
            "learned_selector_safety_target": _as_float(learned_row.get("learned_safety_target", 0.0)) if learned_row else 0.0,
            "learned_selector_recurrence_compatibility": _as_float(learned_row.get("recurrence_compatibility", 0.0)) if learned_row else 0.0,
            "learned_selector_identity_membrane_pass": _as_float(learned_row.get("learned_identity_membrane_pass", 0.0)) if learned_row else 0.0,
            "learned_selector_strong_recurrence_pass": _as_float(learned_row.get("learned_strong_recurrence_pass", 0.0)) if learned_row else 0.0,
            "learned_selector_same_local_family": 1.0
            if learned_row is not None and bool(learned_row.get("same_family", False))
            else 0.0,
            "learned_selector_same_structural_family": 1.0
            if learned_row is not None
            and str(learned_row.get("query_structural_family_key", ""))
            == str(learned_row.get("candidate_structural_family_key", ""))
            else 0.0,
            "learned_selector_agrees_with_selected": 1.0 if learned_id and learned_id == candidate_id else 0.0,
            "learned_selector_agrees_with_recurrence": 1.0 if learned_id and learned_id == recurrence_id else 0.0,
            "causal_selector_available": 1.0 if causal_selector is not None else 0.0,
            "causal_selector_model_score": _as_float(causal_row.get("causal_selector_model_score", 0.0)) if causal_row else 0.0,
            "causal_selector_movement_pred": _as_float(causal_row.get("causal_selector_movement_pred", 0.0)) if causal_row else 0.0,
            "causal_selector_jump_safety_pred": _as_float(causal_row.get("causal_selector_jump_safety_pred", 0.0)) if causal_row else 0.0,
            "causal_selector_identity_pred": _as_float(causal_row.get("causal_selector_identity_pred", 0.0)) if causal_row else 0.0,
            "causal_selector_hybrid_score": _as_float(causal_row.get("causal_selector_hybrid_score", 0.0)) if causal_row else 0.0,
            "causal_selector_geometric_score": _as_float(causal_row.get("geometric_score", 0.0)) if causal_row else 0.0,
            "causal_selector_residual": _as_float(causal_row.get("causal_selector_residual", 0.0)) if causal_row else 0.0,
            "causal_selector_raw_residual": _as_float(causal_row.get("causal_selector_raw_residual", 0.0)) if causal_row else 0.0,
            "causal_selector_recurrence_compatibility": _as_float(causal_row.get("recurrence_compatibility", 0.0)) if causal_row else 0.0,
            "causal_selector_identity_membrane_pass": _as_float(causal_row.get("identity_membrane_pass", 0.0)) if causal_row else 0.0,
            "causal_selector_strong_recurrence_pass": _as_float(causal_row.get("strong_recurrence_pass", 0.0)) if causal_row else 0.0,
            "causal_selector_same_local_family": _as_float(causal_row.get("same_local_family", 0.0)) if causal_row else 0.0,
            "causal_selector_same_structural_family": _as_float(causal_row.get("same_structural_family", 0.0)) if causal_row else 0.0,
            "causal_selector_agrees_with_selected": 1.0 if causal_id and causal_id == candidate_id else 0.0,
            "causal_selector_agrees_with_recurrence": 1.0 if causal_id and causal_id == recurrence_id else 0.0,
            "causal_selector_agrees_with_learned": 1.0 if causal_id and causal_id == learned_id else 0.0,
            "decoy_score": _as_float(decoy_row.get("final_score", 0.0)) if decoy_row else 0.0,
            "decoy_operator_applied": _as_float(decoy_metrics.get("operator_applied", 0.0)),
            "decoy_parent_phase_divergence": _as_float(decoy_metrics.get("parent_phase_divergence", 0.0)),
            "decoy_support_shift": _as_float(decoy_metrics.get("support_shift", 0.0)),
            "decoy_qtrace_shift_proxy": _as_float(decoy_metrics.get("qtrace_shift_proxy", 0.0)),
            "decoy_world_jump_proxy": _as_float(decoy_metrics.get("world_jump_proxy", 0.0)),
            "selected_decoy_phase_divergence_gap": float(
                metrics["parent_phase_divergence"] - _as_float(decoy_metrics.get("parent_phase_divergence", 0.0))
            ),
            "selected_decoy_world_jump_gap": float(
                metrics["world_jump_proxy"] - _as_float(decoy_metrics.get("world_jump_proxy", 0.0))
            ),
            **_prefixed_metrics("recurrence", recurrence_metrics),
            **_prefixed_metrics("learned", learned_metrics),
            **_prefixed_metrics("causal", causal_metrics),
            "recurrence_selected_phase_divergence_gap": float(
                _as_float(recurrence_metrics.get("parent_phase_divergence", 0.0)) - metrics["parent_phase_divergence"]
            ),
            "recurrence_selected_world_jump_gap": float(
                _as_float(recurrence_metrics.get("world_jump_proxy", 0.0)) - metrics["world_jump_proxy"]
            ),
            "learned_selected_phase_divergence_gap": float(
                _as_float(learned_metrics.get("parent_phase_divergence", 0.0)) - metrics["parent_phase_divergence"]
            ),
            "learned_selected_world_jump_gap": float(
                _as_float(learned_metrics.get("world_jump_proxy", 0.0)) - metrics["world_jump_proxy"]
            ),
            "learned_decoy_phase_divergence_gap": float(
                _as_float(learned_metrics.get("parent_phase_divergence", 0.0))
                - _as_float(decoy_metrics.get("parent_phase_divergence", 0.0))
            ),
            "learned_decoy_world_jump_gap": float(
                _as_float(learned_metrics.get("world_jump_proxy", 0.0))
                - _as_float(decoy_metrics.get("world_jump_proxy", 0.0))
            ),
            "causal_selected_phase_divergence_gap": float(
                _as_float(causal_metrics.get("parent_phase_divergence", 0.0)) - metrics["parent_phase_divergence"]
            ),
            "causal_selected_world_jump_gap": float(
                _as_float(causal_metrics.get("world_jump_proxy", 0.0)) - metrics["world_jump_proxy"]
            ),
            "causal_decoy_phase_divergence_gap": float(
                _as_float(causal_metrics.get("parent_phase_divergence", 0.0))
                - _as_float(decoy_metrics.get("parent_phase_divergence", 0.0))
            ),
            "causal_decoy_world_jump_gap": float(
                _as_float(causal_metrics.get("world_jump_proxy", 0.0))
                - _as_float(decoy_metrics.get("world_jump_proxy", 0.0))
            ),
            "decoy_higher_movement_than_causal": 1.0
            if _as_float(decoy_metrics.get("parent_phase_divergence", 0.0))
            > _as_float(causal_metrics.get("parent_phase_divergence", 0.0))
            else 0.0,
            "decoy_higher_jump_than_causal": 1.0
            if _as_float(decoy_metrics.get("world_jump_proxy", 0.0))
            > _as_float(causal_metrics.get("world_jump_proxy", 0.0))
            else 0.0,
            "hard_decoy_against_causal": 1.0
            if (
                _as_float(decoy_metrics.get("parent_phase_divergence", 0.0))
                > _as_float(causal_metrics.get("parent_phase_divergence", 0.0))
                and _as_float(decoy_metrics.get("world_jump_proxy", 0.0))
                > _as_float(causal_metrics.get("world_jump_proxy", 0.0))
            )
            else 0.0,
            "causal_safety_win_over_decoy": 1.0
            if _as_float(causal_metrics.get("world_jump_proxy", 0.0))
            < _as_float(decoy_metrics.get("world_jump_proxy", 0.0))
            else 0.0,
            "causal_authority_win_over_decoy": 1.0
            if (
                _as_float(causal_row.get("identity_membrane_pass", 0.0) if causal_row else 0.0) > 0.0
                and _as_float(causal_metrics.get("world_jump_proxy", 0.0))
                < _as_float(decoy_metrics.get("world_jump_proxy", 0.0))
            )
            else 0.0,
            "decoy_causal_phase_divergence_gap": float(
                _as_float(decoy_metrics.get("parent_phase_divergence", 0.0))
                - _as_float(causal_metrics.get("parent_phase_divergence", 0.0))
            ),
            "decoy_causal_world_jump_gap": float(
                _as_float(decoy_metrics.get("world_jump_proxy", 0.0))
                - _as_float(causal_metrics.get("world_jump_proxy", 0.0))
            ),
            "retrieval": retrieval["summary"],
            "partition": partition,
            "status": "pass_operator_causal_delta" if metrics["parent_phase_divergence"] > 0.0 else "fail_zero_operator_delta",
        }
    )
    return metrics


def _write_markdown(path: Path, payload: dict[str, Any]) -> None:
    agg = payload["aggregate"]
    lines = [
        "# Resonant Operator Causality Assay",
        "",
        "This assay selects a retrieved child law object and applies its operator-valued phase payload against an inert no-op control.",
        "",
        "## Summary",
        "",
        f"- status: `{agg['status']}`",
        f"- cases: `{agg['case_count']}`",
        f"- mean law objects: `{agg['mean_law_object_count']}`",
        f"- mean parent phase divergence: `{agg['mean_parent_phase_divergence']}`",
        f"- mean inert parent phase divergence: `{agg['mean_inert_parent_phase_divergence']}`",
        f"- mean support shift: `{agg['mean_support_shift']}`",
        f"- mean qtrace shift proxy: `{agg['mean_qtrace_shift_proxy']}`",
        f"- mean world jump proxy: `{agg['mean_world_jump_proxy']}`",
        f"- mean selected resonance margin: `{agg['mean_selected_resonance_margin']}`",
        f"- mean decoy parent phase divergence: `{agg['mean_decoy_parent_phase_divergence']}`",
        f"- mean decoy world jump proxy: `{agg['mean_decoy_world_jump_proxy']}`",
        f"- mean selected-decoy phase divergence gap: `{agg['mean_selected_decoy_phase_divergence_gap']}`",
        f"- mean selected-decoy world jump gap: `{agg['mean_selected_decoy_world_jump_gap']}`",
        f"- mean recurrence parent phase divergence: `{agg.get('mean_recurrence_parent_phase_divergence', 0.0)}`",
        f"- mean recurrence world jump proxy: `{agg.get('mean_recurrence_world_jump_proxy', 0.0)}`",
        f"- mean learned parent phase divergence: `{agg.get('mean_learned_parent_phase_divergence', 0.0)}`",
        f"- mean learned world jump proxy: `{agg.get('mean_learned_world_jump_proxy', 0.0)}`",
        f"- mean learned hybrid score: `{agg.get('mean_learned_selector_hybrid_score', 0.0)}`",
        f"- mean learned selector recurrence compatibility: `{agg.get('mean_learned_selector_recurrence_compatibility', 0.0)}`",
        f"- learned selector identity membrane pass: `{agg.get('mean_learned_selector_identity_membrane_pass', 0.0)}`",
        f"- learned selector strong recurrence pass: `{agg.get('mean_learned_selector_strong_recurrence_pass', 0.0)}`",
        f"- learned selector agreement with geometric selected: `{agg.get('mean_learned_selector_agrees_with_selected', 0.0)}`",
        f"- learned selector agreement with recurrence selected: `{agg.get('mean_learned_selector_agrees_with_recurrence', 0.0)}`",
        f"- mean causal parent phase divergence: `{agg.get('mean_causal_parent_phase_divergence', 0.0)}`",
        f"- mean causal world jump proxy: `{agg.get('mean_causal_world_jump_proxy', 0.0)}`",
        f"- mean causal selector model score: `{agg.get('mean_causal_selector_model_score', 0.0)}`",
        f"- causal selector identity membrane pass: `{agg.get('mean_causal_selector_identity_membrane_pass', 0.0)}`",
        f"- causal selector agreement with recurrence selected: `{agg.get('mean_causal_selector_agrees_with_recurrence', 0.0)}`",
        f"- hard decoy against causal fraction: `{agg.get('mean_hard_decoy_against_causal', 0.0)}`",
        f"- causal safety win over decoy fraction: `{agg.get('mean_causal_safety_win_over_decoy', 0.0)}`",
        f"- causal authority win over decoy fraction: `{agg.get('mean_causal_authority_win_over_decoy', 0.0)}`",
        f"- mean decoy-causal phase divergence gap: `{agg.get('mean_decoy_causal_phase_divergence_gap', 0.0)}`",
        f"- mean decoy-causal world jump gap: `{agg.get('mean_decoy_causal_world_jump_gap', 0.0)}`",
        "",
        "## Cases",
        "",
        "| source | seed | status | law objects | geo div | recur div | learned div | causal div | decoy div | geo jump | causal jump | decoy jump | learned score | causal score |",
        "|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in payload["cases"]:
        lines.append(
            "| {source} | {seed} | {status} | {objects} | {div} | {recur_div} | {learned_div} | {causal_div} | {decoy_div} | {jump} | {causal_jump} | {decoy_jump} | {learned_score} | {causal_score} |".format(
                source=row.get("source"),
                seed=row.get("seed"),
                status=row.get("status"),
                objects=row.get("law_object_count"),
                div=row.get("parent_phase_divergence", 0.0),
                recur_div=row.get("recurrence_parent_phase_divergence", 0.0),
                learned_div=row.get("learned_parent_phase_divergence", 0.0),
                causal_div=row.get("causal_parent_phase_divergence", 0.0),
                decoy_div=row.get("decoy_parent_phase_divergence", 0.0),
                jump=row.get("world_jump_proxy", 0.0),
                causal_jump=row.get("causal_world_jump_proxy", 0.0),
                decoy_jump=row.get("decoy_world_jump_proxy", 0.0),
                learned_score=row.get("learned_selector_hybrid_score", 0.0),
                causal_score=row.get("causal_selector_hybrid_score", 0.0),
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_assay(
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
    learned_selector_checkpoint: Path | None = None,
    learned_residual_weight: float = 0.25,
    learned_safety_weight: float = 0.10,
    learned_residual_cap: float = 0.05,
    learned_membrane_mode: str = "identity",
    learned_strong_recurrence_threshold: float = 0.96,
    learned_strong_geometric_threshold: float = 0.94,
    causal_selector_checkpoint: Path | None = None,
    causal_residual_weight: float = 0.20,
    causal_residual_cap: float = 0.05,
    causal_membrane_mode: str = "identity",
) -> dict[str, str]:
    cfg = _load_circle_cfg(config_path)
    device = _safe_device(device_name)
    learned_selector = _load_learned_selector(learned_selector_checkpoint, device)
    causal_selector = _load_causal_selector(causal_selector_checkpoint, device)
    rafa_core = make_seed_rafa(dev=device.type) if seed_source == "naked_rafa" else None
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
        rows = [
            _run_one_case(
                cfg=cfg,
                rafa_core=rafa_core,
                source=seed_source,
                seed=int(seed),
                time_steps=int(time_steps),
                warmup_depth=int(warmup_depth),
                device=device,
                operator_gain=float(operator_gain),
                learned_selector=learned_selector,
                learned_residual_weight=float(learned_residual_weight),
                learned_safety_weight=float(learned_safety_weight),
                learned_residual_cap=float(learned_residual_cap),
                learned_membrane_mode=str(learned_membrane_mode),
                learned_strong_recurrence_threshold=float(learned_strong_recurrence_threshold),
                learned_strong_geometric_threshold=float(learned_strong_geometric_threshold),
                causal_selector=causal_selector,
                causal_residual_weight=float(causal_residual_weight),
                causal_residual_cap=float(causal_residual_cap),
                causal_membrane_mode=str(causal_membrane_mode),
            )
            for seed in seeds
        ]
    finally:
        ASSAY_KILLSWITCH.clear()
        ASSAY_KILLSWITCH.update(previous)
    aggregate = {
        "status": "pass_operator_causality_present"
        if any(float(row.get("parent_phase_divergence", 0.0)) > 0.0 for row in rows)
        else "fail_no_operator_causality",
        "case_count": len(rows),
        "mean_law_object_count": float(np.mean([float(row.get("law_object_count", 0.0)) for row in rows])) if rows else 0.0,
        "mean_parent_phase_divergence": float(np.mean([float(row.get("parent_phase_divergence", 0.0)) for row in rows])) if rows else 0.0,
        "mean_inert_parent_phase_divergence": float(np.mean([float(row.get("inert_parent_phase_divergence", 0.0)) for row in rows])) if rows else 0.0,
        "mean_support_shift": float(np.mean([float(row.get("support_shift", 0.0)) for row in rows])) if rows else 0.0,
        "mean_qtrace_shift_proxy": float(np.mean([float(row.get("qtrace_shift_proxy", 0.0)) for row in rows])) if rows else 0.0,
        "mean_world_jump_proxy": float(np.mean([float(row.get("world_jump_proxy", 0.0)) for row in rows])) if rows else 0.0,
        "mean_selected_resonance_margin": float(np.mean([float(row.get("selected_resonance_margin", 0.0)) for row in rows])) if rows else 0.0,
        "mean_decoy_parent_phase_divergence": float(np.mean([float(row.get("decoy_parent_phase_divergence", 0.0)) for row in rows])) if rows else 0.0,
        "mean_decoy_world_jump_proxy": float(np.mean([float(row.get("decoy_world_jump_proxy", 0.0)) for row in rows])) if rows else 0.0,
        "mean_selected_decoy_phase_divergence_gap": float(np.mean([float(row.get("selected_decoy_phase_divergence_gap", 0.0)) for row in rows])) if rows else 0.0,
        "mean_selected_decoy_world_jump_gap": float(np.mean([float(row.get("selected_decoy_world_jump_gap", 0.0)) for row in rows])) if rows else 0.0,
        "mean_recurrence_parent_phase_divergence": float(np.mean([float(row.get("recurrence_parent_phase_divergence", 0.0)) for row in rows])) if rows else 0.0,
        "mean_recurrence_world_jump_proxy": float(np.mean([float(row.get("recurrence_world_jump_proxy", 0.0)) for row in rows])) if rows else 0.0,
        "mean_recurrence_selector_score": float(np.mean([float(row.get("recurrence_selector_score", 0.0)) for row in rows])) if rows else 0.0,
        "mean_learned_parent_phase_divergence": float(np.mean([float(row.get("learned_parent_phase_divergence", 0.0)) for row in rows])) if rows else 0.0,
        "mean_learned_world_jump_proxy": float(np.mean([float(row.get("learned_world_jump_proxy", 0.0)) for row in rows])) if rows else 0.0,
        "mean_learned_selector_hybrid_score": float(np.mean([float(row.get("learned_selector_hybrid_score", 0.0)) for row in rows])) if rows else 0.0,
        "mean_learned_selector_embedding_score": float(np.mean([float(row.get("learned_selector_embedding_score", 0.0)) for row in rows])) if rows else 0.0,
        "mean_learned_selector_raw_residual": float(np.mean([float(row.get("learned_selector_raw_residual", 0.0)) for row in rows])) if rows else 0.0,
        "mean_learned_selector_residual": float(np.mean([float(row.get("learned_selector_residual", 0.0)) for row in rows])) if rows else 0.0,
        "mean_learned_selector_safety_pred": float(np.mean([float(row.get("learned_selector_safety_pred", 0.0)) for row in rows])) if rows else 0.0,
        "mean_learned_selector_safety_target": float(np.mean([float(row.get("learned_selector_safety_target", 0.0)) for row in rows])) if rows else 0.0,
        "mean_learned_selector_recurrence_compatibility": float(np.mean([float(row.get("learned_selector_recurrence_compatibility", 0.0)) for row in rows])) if rows else 0.0,
        "mean_learned_selector_identity_membrane_pass": float(np.mean([float(row.get("learned_selector_identity_membrane_pass", 0.0)) for row in rows])) if rows else 0.0,
        "mean_learned_selector_strong_recurrence_pass": float(np.mean([float(row.get("learned_selector_strong_recurrence_pass", 0.0)) for row in rows])) if rows else 0.0,
        "mean_learned_selector_same_local_family": float(np.mean([float(row.get("learned_selector_same_local_family", 0.0)) for row in rows])) if rows else 0.0,
        "mean_learned_selector_same_structural_family": float(np.mean([float(row.get("learned_selector_same_structural_family", 0.0)) for row in rows])) if rows else 0.0,
        "mean_learned_selector_agrees_with_selected": float(np.mean([float(row.get("learned_selector_agrees_with_selected", 0.0)) for row in rows])) if rows else 0.0,
        "mean_learned_selector_agrees_with_recurrence": float(np.mean([float(row.get("learned_selector_agrees_with_recurrence", 0.0)) for row in rows])) if rows else 0.0,
        "mean_learned_selected_phase_divergence_gap": float(np.mean([float(row.get("learned_selected_phase_divergence_gap", 0.0)) for row in rows])) if rows else 0.0,
        "mean_learned_selected_world_jump_gap": float(np.mean([float(row.get("learned_selected_world_jump_gap", 0.0)) for row in rows])) if rows else 0.0,
        "mean_learned_decoy_phase_divergence_gap": float(np.mean([float(row.get("learned_decoy_phase_divergence_gap", 0.0)) for row in rows])) if rows else 0.0,
        "mean_learned_decoy_world_jump_gap": float(np.mean([float(row.get("learned_decoy_world_jump_gap", 0.0)) for row in rows])) if rows else 0.0,
        "mean_causal_parent_phase_divergence": float(np.mean([float(row.get("causal_parent_phase_divergence", 0.0)) for row in rows])) if rows else 0.0,
        "mean_causal_world_jump_proxy": float(np.mean([float(row.get("causal_world_jump_proxy", 0.0)) for row in rows])) if rows else 0.0,
        "mean_causal_selector_model_score": float(np.mean([float(row.get("causal_selector_model_score", 0.0)) for row in rows])) if rows else 0.0,
        "mean_causal_selector_movement_pred": float(np.mean([float(row.get("causal_selector_movement_pred", 0.0)) for row in rows])) if rows else 0.0,
        "mean_causal_selector_jump_safety_pred": float(np.mean([float(row.get("causal_selector_jump_safety_pred", 0.0)) for row in rows])) if rows else 0.0,
        "mean_causal_selector_identity_pred": float(np.mean([float(row.get("causal_selector_identity_pred", 0.0)) for row in rows])) if rows else 0.0,
        "mean_causal_selector_hybrid_score": float(np.mean([float(row.get("causal_selector_hybrid_score", 0.0)) for row in rows])) if rows else 0.0,
        "mean_causal_selector_residual": float(np.mean([float(row.get("causal_selector_residual", 0.0)) for row in rows])) if rows else 0.0,
        "mean_causal_selector_recurrence_compatibility": float(np.mean([float(row.get("causal_selector_recurrence_compatibility", 0.0)) for row in rows])) if rows else 0.0,
        "mean_causal_selector_identity_membrane_pass": float(np.mean([float(row.get("causal_selector_identity_membrane_pass", 0.0)) for row in rows])) if rows else 0.0,
        "mean_causal_selector_strong_recurrence_pass": float(np.mean([float(row.get("causal_selector_strong_recurrence_pass", 0.0)) for row in rows])) if rows else 0.0,
        "mean_causal_selector_same_local_family": float(np.mean([float(row.get("causal_selector_same_local_family", 0.0)) for row in rows])) if rows else 0.0,
        "mean_causal_selector_same_structural_family": float(np.mean([float(row.get("causal_selector_same_structural_family", 0.0)) for row in rows])) if rows else 0.0,
        "mean_causal_selector_agrees_with_selected": float(np.mean([float(row.get("causal_selector_agrees_with_selected", 0.0)) for row in rows])) if rows else 0.0,
        "mean_causal_selector_agrees_with_recurrence": float(np.mean([float(row.get("causal_selector_agrees_with_recurrence", 0.0)) for row in rows])) if rows else 0.0,
        "mean_causal_selector_agrees_with_learned": float(np.mean([float(row.get("causal_selector_agrees_with_learned", 0.0)) for row in rows])) if rows else 0.0,
        "mean_causal_selected_phase_divergence_gap": float(np.mean([float(row.get("causal_selected_phase_divergence_gap", 0.0)) for row in rows])) if rows else 0.0,
        "mean_causal_selected_world_jump_gap": float(np.mean([float(row.get("causal_selected_world_jump_gap", 0.0)) for row in rows])) if rows else 0.0,
        "mean_causal_decoy_phase_divergence_gap": float(np.mean([float(row.get("causal_decoy_phase_divergence_gap", 0.0)) for row in rows])) if rows else 0.0,
        "mean_causal_decoy_world_jump_gap": float(np.mean([float(row.get("causal_decoy_world_jump_gap", 0.0)) for row in rows])) if rows else 0.0,
        "mean_decoy_higher_movement_than_causal": float(np.mean([float(row.get("decoy_higher_movement_than_causal", 0.0)) for row in rows])) if rows else 0.0,
        "mean_decoy_higher_jump_than_causal": float(np.mean([float(row.get("decoy_higher_jump_than_causal", 0.0)) for row in rows])) if rows else 0.0,
        "mean_hard_decoy_against_causal": float(np.mean([float(row.get("hard_decoy_against_causal", 0.0)) for row in rows])) if rows else 0.0,
        "mean_causal_safety_win_over_decoy": float(np.mean([float(row.get("causal_safety_win_over_decoy", 0.0)) for row in rows])) if rows else 0.0,
        "mean_causal_authority_win_over_decoy": float(np.mean([float(row.get("causal_authority_win_over_decoy", 0.0)) for row in rows])) if rows else 0.0,
        "mean_decoy_causal_phase_divergence_gap": float(np.mean([float(row.get("decoy_causal_phase_divergence_gap", 0.0)) for row in rows])) if rows else 0.0,
        "mean_decoy_causal_world_jump_gap": float(np.mean([float(row.get("decoy_causal_world_jump_gap", 0.0)) for row in rows])) if rows else 0.0,
    }
    payload = {
        "config_path": str(config_path),
        "device_requested": device_name,
        "device": str(device),
        "seed_source": seed_source,
        "seeds": [int(seed) for seed in seeds],
        "time_steps": int(time_steps),
        "warmup_depth": int(warmup_depth),
        "operator_gain": float(operator_gain),
        "grandchild_depth": int(grandchild_depth),
        "grandchild_parts": int(grandchild_parts),
        "learned_selector_checkpoint": str(learned_selector_checkpoint) if learned_selector_checkpoint else "",
        "learned_residual_weight": float(learned_residual_weight),
        "learned_safety_weight": float(learned_safety_weight),
        "learned_residual_cap": float(learned_residual_cap),
        "learned_membrane_mode": str(learned_membrane_mode),
        "learned_strong_recurrence_threshold": float(learned_strong_recurrence_threshold),
        "learned_strong_geometric_threshold": float(learned_strong_geometric_threshold),
        "causal_selector_checkpoint": str(causal_selector_checkpoint) if causal_selector_checkpoint else "",
        "causal_residual_weight": float(causal_residual_weight),
        "causal_residual_cap": float(causal_residual_cap),
        "causal_membrane_mode": str(causal_membrane_mode),
        "learned_selector_summary": {
            "available": learned_selector is not None,
            "checkpoint_path": str(learned_selector.get("checkpoint_path", "")) if learned_selector else "",
            "feature_dim": int(learned_selector.get("feature_dim", 0)) if learned_selector else 0,
            "source_summary": dict(learned_selector.get("summary", {}) or {}) if learned_selector else {},
        },
        "causal_selector_summary": {
            "available": causal_selector is not None,
            "checkpoint_path": str(causal_selector.get("checkpoint_path", "")) if causal_selector else "",
            "feature_dim": int(causal_selector.get("feature_dim", 0)) if causal_selector else 0,
            "source_summary": dict(causal_selector.get("summary", {}) or {}) if causal_selector else {},
        },
        "aggregate": aggregate,
        "cases": rows,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "resonant_operator_causality_assay.json"
    md_path = out_dir / "RESONANT_OPERATOR_CAUSALITY_ASSAY.md"
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    _write_markdown(md_path, payload)
    return {"json": str(json_path), "markdown": str(md_path), "status": aggregate["status"]}


def _parse_seeds(raw: str) -> list[int]:
    return [int(part.strip()) for part in str(raw).replace(";", ",").split(",") if part.strip()]


def main() -> None:
    ap = argparse.ArgumentParser(description="Run a resonant operator causality assay over live child law objects.")
    ap.add_argument("--config", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed-source", choices=["synthetic", "naked_rafa"], default="synthetic")
    ap.add_argument("--seeds", default="9100", help="Comma-separated seeds, for example 9100,9101,9102.")
    ap.add_argument("--time-steps", type=int, default=64)
    ap.add_argument("--warmup-depth", type=int, default=3)
    ap.add_argument("--operator-gain", type=float, default=0.85)
    ap.add_argument("--grandchild-depth", type=int, default=1)
    ap.add_argument("--grandchild-parts", type=int, default=2)
    ap.add_argument("--learned-selector-checkpoint", default="")
    ap.add_argument("--learned-residual-weight", type=float, default=0.25)
    ap.add_argument("--learned-safety-weight", type=float, default=0.10)
    ap.add_argument("--learned-residual-cap", type=float, default=0.05)
    ap.add_argument("--learned-membrane-mode", choices=["off", "identity", "prefer_identity"], default="identity")
    ap.add_argument("--learned-strong-recurrence-threshold", type=float, default=0.96)
    ap.add_argument("--learned-strong-geometric-threshold", type=float, default=0.94)
    ap.add_argument("--causal-selector-checkpoint", default="")
    ap.add_argument("--causal-residual-weight", type=float, default=0.20)
    ap.add_argument("--causal-residual-cap", type=float, default=0.05)
    ap.add_argument("--causal-membrane-mode", choices=["off", "identity", "prefer_identity"], default="identity")
    args = ap.parse_args()
    print(
        json.dumps(
            run_assay(
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
                learned_selector_checkpoint=Path(args.learned_selector_checkpoint)
                if str(args.learned_selector_checkpoint).strip()
                else None,
                learned_residual_weight=float(args.learned_residual_weight),
                learned_safety_weight=float(args.learned_safety_weight),
                learned_residual_cap=float(args.learned_residual_cap),
                learned_membrane_mode=str(args.learned_membrane_mode),
                learned_strong_recurrence_threshold=float(args.learned_strong_recurrence_threshold),
                learned_strong_geometric_threshold=float(args.learned_strong_geometric_threshold),
                causal_selector_checkpoint=Path(args.causal_selector_checkpoint)
                if str(args.causal_selector_checkpoint).strip()
                else None,
                causal_residual_weight=float(args.causal_residual_weight),
                causal_residual_cap=float(args.causal_residual_cap),
                causal_membrane_mode=str(args.causal_membrane_mode),
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
