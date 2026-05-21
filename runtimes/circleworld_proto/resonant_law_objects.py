from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import torch

GEOMETRIC_COMPONENTS: tuple[str, ...] = (
    "q_compatibility",
    "major_arc_overlap",
    "minor_residue_fit",
    "support_overlap",
    "boundary_compatibility",
    "temporal_law_compatibility",
    "reentry_safety",
)

GEOMETRIC_WEIGHTS: dict[str, float] = {
    "q_compatibility": 0.28,
    "major_arc_overlap": 0.16,
    "minor_residue_fit": 0.14,
    "support_overlap": 0.14,
    "boundary_compatibility": 0.10,
    "temporal_law_compatibility": 0.10,
    "reentry_safety": 0.08,
}

LAW_OBJECT_Q_FEATURE_COUNT = 8
LAW_OBJECT_FEATURE_NAMES: tuple[str, ...] = tuple(
    [f"q_profile_{idx}" for idx in range(LAW_OBJECT_Q_FEATURE_COUNT)]
    + [
        "q_entropy",
        "dominant_q_share",
        "major_arc_proxy",
        "minor_residue_proxy",
        "support_mean",
        "coherence_mean",
        "boundary_score",
        "temporal_center",
        "temporal_width",
        "window_start",
        "window_end",
        "generation",
        "operator_norm",
        "writeback_budget",
        "support_mass",
        "coherence_mass",
        "causal_use_ready",
        "is_childworld",
        "is_branch_summary",
    ]
)


def _as_float(value: Any, default: float = 0.0) -> float:
    if torch.is_tensor(value):
        if value.numel() == 0:
            return float(default)
        return float(value.detach().float().mean().cpu().item())
    if value is None:
        return float(default)
    try:
        if isinstance(value, str) and not value.strip():
            return float(default)
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _as_int(value: Any, default: int = -1) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return int(default)


def _clamp01(value: float) -> float:
    return float(max(0.0, min(1.0, value)))


def _safe_list(values: Any, max_len: int | None = None) -> list[float]:
    if torch.is_tensor(values):
        raw = values.detach().float().flatten().cpu().tolist()
    elif isinstance(values, (list, tuple)):
        raw = []
        for item in values:
            if isinstance(item, (list, tuple)):
                raw.extend(_safe_list(item))
            else:
                raw.append(_as_float(item))
    elif values is None:
        raw = []
    else:
        raw = [_as_float(values)]
    if max_len is not None:
        raw = raw[: max(0, int(max_len))]
    return [float(x) for x in raw if math.isfinite(float(x))]


def _normalize_profile(values: Any, *, fallback_len: int = 4) -> list[float]:
    raw = [abs(x) for x in _safe_list(values)]
    if not raw:
        raw = [0.0] * int(fallback_len)
    total = sum(raw)
    if total <= 1.0e-12:
        return [1.0 / len(raw)] * len(raw)
    return [float(x / total) for x in raw]


def _mean_last_dim_profile(value: Any, *, fallback_len: int = 4) -> list[float]:
    if not torch.is_tensor(value) or value.numel() == 0:
        return _normalize_profile([], fallback_len=fallback_len)
    tensor = value.detach().float().abs()
    if tensor.dim() == 0:
        return _normalize_profile([float(tensor.item())], fallback_len=fallback_len)
    if tensor.dim() == 1:
        return _normalize_profile(tensor, fallback_len=fallback_len)
    dims = tuple(range(tensor.dim() - 1))
    return _normalize_profile(tensor.mean(dim=dims), fallback_len=fallback_len)


def _cosine01(left: list[float], right: list[float]) -> float:
    n = min(len(left), len(right))
    if n <= 0:
        return 0.0
    a = left[:n]
    b = right[:n]
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na <= 1.0e-12 or nb <= 1.0e-12:
        return 0.0
    return _clamp01(0.5 + 0.5 * (dot / (na * nb)))


def _safe_masked_mean(value: Any, mask: Any | None = None, default: float = 0.0) -> float:
    if not torch.is_tensor(value) or value.numel() == 0:
        return float(default)
    tensor = value.detach().float()
    if torch.is_tensor(mask) and mask.shape == tensor.shape:
        weight = mask.detach().float().clamp(0.0, 1.0)
        denom = float(weight.sum().clamp_min(1.0e-8).item())
        return float((tensor * weight).sum().item() / denom)
    return float(tensor.mean().item())


def _time_profile_from_support(support: Any, support_window: Any = None) -> dict[str, float]:
    if torch.is_tensor(support) and support.numel() > 0 and support.dim() >= 3:
        weight = support.detach().float().clamp(0.0, 1.0)
        time_weight = weight.mean(dim=tuple(range(weight.dim() - 1)))
        t_len = int(time_weight.numel())
        if t_len <= 1:
            return {"center": 0.0, "width": 1.0, "window_start": 0.0, "window_end": 1.0}
        total = float(time_weight.sum().item())
        positions = torch.linspace(0.0, 1.0, steps=t_len, device=time_weight.device, dtype=time_weight.dtype)
        if total <= 1.0e-8:
            center = 0.5
            width = 1.0
        else:
            center = float((positions * time_weight).sum().item() / total)
            variance = float((((positions - center) ** 2) * time_weight).sum().item() / total)
            width = _clamp01(2.0 * math.sqrt(max(0.0, variance)))
        active = torch.nonzero(time_weight > 1.0e-6, as_tuple=False).flatten()
        if active.numel() > 0:
            start = float(active.min().item()) / float(max(1, t_len - 1))
            end = float(active.max().item()) / float(max(1, t_len - 1))
        else:
            start = 0.0
            end = 1.0
        return {"center": center, "width": width, "window_start": start, "window_end": end}

    if isinstance(support_window, (list, tuple)) and len(support_window) >= 2:
        start = max(0.0, float(support_window[0]))
        end = max(start, float(support_window[1]))
        span = max(1.0, end)
        start_n = _clamp01(start / span)
        end_n = _clamp01(end / span)
        return {
            "center": _clamp01(0.5 * (start_n + end_n)),
            "width": _clamp01(end_n - start_n),
            "window_start": start_n,
            "window_end": end_n,
        }
    return {"center": 0.5, "width": 1.0, "window_start": 0.0, "window_end": 1.0}


def _window_overlap(left: dict[str, Any], right: dict[str, Any]) -> float:
    l0 = _as_float(left.get("window_start", 0.0))
    l1 = _as_float(left.get("window_end", 1.0))
    r0 = _as_float(right.get("window_start", 0.0))
    r1 = _as_float(right.get("window_end", 1.0))
    inter = max(0.0, min(l1, r1) - max(l0, r0))
    union = max(1.0e-8, max(l1, r1) - min(l0, r0))
    return _clamp01(inter / union)


def _phase_alignment_to_parent(child: dict[str, Any], state: dict[str, Any] | None) -> float:
    phase = child.get("phase_state")
    support = child.get("mode_support")
    if not (torch.is_tensor(phase) and isinstance(state, dict)):
        return 0.5
    phase_modes = state.get("phase_modes")
    if not torch.is_tensor(phase_modes) or phase_modes.dim() < 5 or phase_modes.size(3) < 2:
        return 0.5
    parent_mode1 = phase_modes[..., 1, :]
    if parent_mode1.shape != phase.shape:
        return 0.5
    align = (phase * parent_mode1).sum(dim=-1)
    return _clamp01(0.5 + 0.5 * _safe_masked_mean(align, support, default=0.0))


def _law_signature_profile(value: Any, q_len: int) -> tuple[list[float], float]:
    raw = _safe_list(value)
    if not raw:
        return _normalize_profile([], fallback_len=q_len), 0.0
    q_part = raw[: max(1, int(q_len))]
    norm = math.sqrt(sum(x * x for x in raw))
    return _normalize_profile(q_part, fallback_len=q_len), float(norm)


def _object_family(case: str, child_id: int, object_kind: str, branch_family: str = "") -> str:
    if child_id >= 0:
        return f"{case}:child:{child_id}"
    if branch_family:
        return f"{case}:family:{branch_family}"
    return f"{case}:{object_kind}"


def _bucket(value: Any, bins: int = 6) -> int:
    return int(round(_clamp01(_as_float(value, 0.0)) * max(1, int(bins) - 1)))


def _dominant_index(values: Any) -> int:
    raw = _safe_list(values)
    if not raw:
        return -1
    return int(max(range(len(raw)), key=lambda idx: raw[idx]))


def structural_family_key_from_key(object_kind: str, key: dict[str, Any]) -> str:
    """Case-independent, geometry-first family key for cross-seed law retrieval."""
    q_profile = _safe_list(key.get("q_profile"))
    q_idx = _dominant_index(q_profile)
    q_share = max(q_profile) if q_profile else 0.0
    return ":".join(
        [
            "struct",
            str(object_kind or "law_object"),
            f"q{q_idx}",
            f"qs{_bucket(q_share)}",
            f"arc{_bucket(key.get('major_arc_proxy', 0.0))}",
            f"res{_bucket(1.0 - _as_float(key.get('minor_residue_proxy', 1.0)))}",
            f"sup{_bucket(key.get('support_mean', 0.0))}",
            f"coh{_bucket(key.get('coherence_mean', 0.0))}",
            f"tc{_bucket(key.get('temporal_center', 0.5))}",
            f"tw{_bucket(key.get('temporal_width', 1.0))}",
        ]
    )


def object_family_value(obj: dict[str, Any], family_key_field: str = "family_key") -> str:
    field = str(family_key_field or "family_key")
    value = obj.get(field)
    if value is not None and str(value) != "":
        return str(value)
    if field == "local_family_key":
        return str(obj.get("family_key", ""))
    if field == "structural_family_key":
        key = dict(obj.get("key", obj.get("query", {})) or {})
        return structural_family_key_from_key(str(obj.get("object_kind", "law_object")), key)
    return str(obj.get("family_key", ""))


def law_object_feature_vector(obj: dict[str, Any], q_len: int = LAW_OBJECT_Q_FEATURE_COUNT) -> list[float]:
    """Compact learned-probe feature surface for RAFA law objects."""
    key = dict(obj.get("key", obj.get("query", {})) or {})
    value = dict(obj.get("value", {}) or {})
    q_profile = _normalize_profile(_safe_list(key.get("q_profile"), max_len=q_len), fallback_len=q_len)
    if len(q_profile) < q_len:
        q_profile = q_profile + [0.0] * (q_len - len(q_profile))
    q_profile = q_profile[:q_len]
    q_entropy = 0.0
    if q_profile:
        denom = math.log(max(2, len(q_profile)))
        q_entropy = float(-sum(p * math.log(max(1.0e-12, p)) for p in q_profile) / denom)
    object_kind = str(obj.get("object_kind", ""))
    return [
        *[float(x) for x in q_profile],
        _clamp01(q_entropy),
        _clamp01(_as_float(key.get("dominant_q_share", max(q_profile) if q_profile else 0.0))),
        _clamp01(_as_float(key.get("major_arc_proxy", 0.0))),
        _clamp01(_as_float(key.get("minor_residue_proxy", 0.0))),
        _clamp01(_as_float(key.get("support_mean", value.get("support_mass", 0.0)))),
        _clamp01(_as_float(key.get("coherence_mean", value.get("coherence_mass", 0.0)))),
        _clamp01(_as_float(key.get("boundary_score", 0.5))),
        _clamp01(_as_float(key.get("temporal_center", 0.5))),
        _clamp01(_as_float(key.get("temporal_width", 1.0))),
        _clamp01(_as_float(key.get("window_start", 0.0))),
        _clamp01(_as_float(key.get("window_end", 1.0))),
        _clamp01(_as_float(obj.get("generation", 1.0)) / 8.0),
        _clamp01(_as_float(value.get("operator_norm", 0.0))),
        _clamp01(_as_float(value.get("writeback_budget", 0.0)) / 8.0),
        _clamp01(_as_float(value.get("support_mass", key.get("support_mean", 0.0)))),
        _clamp01(_as_float(value.get("coherence_mass", key.get("coherence_mean", 0.0)))),
        _clamp01(_as_float(value.get("causal_use_ready", 0.0))),
        1.0 if object_kind == "childworld" else 0.0,
        1.0 if object_kind == "branch_summary" else 0.0,
    ]


def child_law_object(
    child: dict[str, Any],
    *,
    state: dict[str, Any] | None = None,
    context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    context = dict(context or {})
    case = str(context.get("case", "unknown_case"))
    branch = str(context.get("branch", "state"))
    source = str(context.get("source", "child_state"))
    child_id = _as_int(child.get("child_id", -1))
    origin_child_id = _as_int(child.get("origin_child_id", child_id))
    family_child_id = origin_child_id if origin_child_id >= 0 else child_id
    generation = _as_int(child.get("generation", 1), 1)
    parent_child_id = _as_int(child.get("parent_child_id", -1))
    fragment_index = _as_int(child.get("fragment_index", -1))
    support = child.get("mode_support")
    coherence = child.get("mode_coherence")
    q_profile = _mean_last_dim_profile(child.get("q_trace"), fallback_len=4)
    sig_profile, operator_norm = _law_signature_profile(child.get("law_signature"), len(q_profile))
    if q_profile and max(q_profile) <= 1.0e-12 and sig_profile:
        q_profile = sig_profile
    support_mean = _safe_masked_mean(support, default=0.0)
    coherence_mean = _safe_masked_mean(coherence, support, default=0.0)
    budget = _as_float(child.get("writeback_budget", 0.0))
    temporal = _time_profile_from_support(support, child.get("support_window"))
    boundary = _phase_alignment_to_parent(child, state)
    residue_proxy = _clamp01(1.0 - coherence_mean)
    dominant_q_share = max(q_profile) if q_profile else 0.0
    branch_family = str(context.get("family", "childworld"))
    object_id = (
        f"{case}:{branch}:{source}:child:{child_id}:origin:{family_child_id}:"
        f"parent:{parent_child_id}:gen:{generation}:frag:{fragment_index}"
    )
    key = {
        "q_profile": q_profile,
        "dominant_q_share": dominant_q_share,
        "major_arc_proxy": _clamp01(0.5 * dominant_q_share + 0.5 * coherence_mean),
        "minor_residue_proxy": residue_proxy,
        "support_mean": support_mean,
        "coherence_mean": coherence_mean,
        "boundary_score": boundary,
        "temporal_center": temporal["center"],
        "temporal_width": temporal["width"],
        "window_start": temporal["window_start"],
        "window_end": temporal["window_end"],
    }
    local_family_key = _object_family(case, family_child_id, "childworld", branch_family)
    structural_family_key = structural_family_key_from_key("childworld", key)
    return {
        "object_id": object_id,
        "object_kind": "childworld",
        "source": source,
        "case": case,
        "branch": branch,
        "branch_family": branch_family,
        "family_key": local_family_key,
        "local_family_key": local_family_key,
        "structural_family_key": structural_family_key,
        "child_instance_key": _object_family(case, child_id, "childworld", branch_family),
        "child_id": float(child_id),
        "generation": float(generation),
        "parent_child_id": float(parent_child_id),
        "origin_child_id": float(origin_child_id),
        "fragment_index": float(fragment_index),
        "query": dict(key),
        "key": key,
        "value": {
            "operator_kind": "child_phase_writeback",
            "operator_norm": operator_norm,
            "writeback_budget": budget,
            "support_mass": support_mean,
            "coherence_mass": coherence_mean,
            "causal_use_ready": 1.0 if support_mean > 0.0 and coherence_mean > 0.0 and budget > 0.0 else 0.0,
        },
    }


def law_objects_from_state(state: Any, *, context: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    if not isinstance(state, dict):
        return []
    out: list[dict[str, Any]] = []
    for child in state.get("child_worlds", []) or []:
        if isinstance(child, dict):
            out.append(child_law_object(child, state=state, context=context))
    return out


def branch_summary_law_object(case: str, branch: str, metrics: dict[str, Any]) -> dict[str, Any]:
    branch_identity = dict(metrics.get("branch_identity", {}) or {})
    predictive = dict(metrics.get("child_predictive_assimilation", {}) or {})
    partition = dict(metrics.get("assay_child_partition", {}) or {})
    replace = dict(metrics.get("mode1_replace_metrics", {}) or {})
    family = str(metrics.get("family", metrics.get("cfg", {}).get("family", "branch_summary")))
    child_id = _as_int(
        predictive.get("child_id", branch_identity.get("anchor_child_id", metrics.get("ontology_child_id", -1)))
    )
    support = _as_float(branch_identity.get("same_child_support_mean", partition.get("after_mean_support", 0.0)))
    coherence = _as_float(branch_identity.get("same_child_coherence_mean", 0.0))
    budget = _as_float(branch_identity.get("same_child_budget_retained", 0.0))
    q_corr = _clamp01(_as_float(metrics.get("fine_q_profile_corr", 0.0)))
    residue_delta = _clamp01(_as_float(metrics.get("fine_minor_residue_delta", 1.0)))
    coarse = _clamp01(_as_float(metrics.get("coarse_env_corr", 0.0)))
    response = _clamp01(_as_float(metrics.get("readout_sibling_response", 0.0)))
    boundary = _clamp01(_as_float(predictive.get("boundary_match", replace.get("parent_ontology_charge_mean", 0.5))))
    q_profile = _normalize_profile([q_corr, 1.0 - residue_delta, coarse, response], fallback_len=4)
    temporal_center = _clamp01(0.5 + 0.25 * _as_float(metrics.get("meso_loop_period_delta", 0.0)))
    temporal_width = _clamp01(0.35 + 0.30 * response + 0.20 * support)
    key = {
        "q_profile": q_profile,
        "dominant_q_share": max(q_profile) if q_profile else 0.0,
        "major_arc_proxy": _clamp01(0.5 * coarse + 0.5 * q_corr),
        "minor_residue_proxy": residue_delta,
        "support_mean": support,
        "coherence_mean": coherence,
        "boundary_score": boundary,
        "temporal_center": temporal_center,
        "temporal_width": temporal_width,
        "window_start": _clamp01(temporal_center - 0.5 * temporal_width),
        "window_end": _clamp01(temporal_center + 0.5 * temporal_width),
    }
    object_id = f"{case}:{branch}:branch_summary:child:{child_id}"
    local_family_key = _object_family(str(case), child_id, "branch_summary", family)
    structural_family_key = structural_family_key_from_key("branch_summary", key)
    return {
        "object_id": object_id,
        "object_kind": "branch_summary",
        "source": "nested_branch_metrics",
        "case": str(case),
        "branch": str(branch),
        "branch_family": family,
        "family_key": local_family_key,
        "local_family_key": local_family_key,
        "structural_family_key": structural_family_key,
        "child_id": float(child_id),
        "generation": _as_float(partition.get("after_max_generation", 1.0), 1.0),
        "parent_child_id": -1.0,
        "origin_child_id": float(child_id),
        "query": dict(key),
        "key": key,
        "value": {
            "operator_kind": "branch_summary_operator_proxy",
            "operator_norm": _clamp01(response + _as_float(predictive.get("raw_child_delta_abs_mean", 0.0))),
            "writeback_budget": budget,
            "support_mass": support,
            "coherence_mass": coherence,
            "causal_use_ready": _as_float(predictive.get("applied", 0.0)),
        },
    }


def law_objects_from_nested_case_summary(case_summary: dict[str, Any]) -> list[dict[str, Any]]:
    case = str(case_summary.get("case", "unknown_case"))
    out: list[dict[str, Any]] = []
    for branch, metrics in dict(case_summary.get("branch_metrics", {}) or {}).items():
        if isinstance(metrics, dict):
            out.append(branch_summary_law_object(case, str(branch), metrics))
    return out


def score_query_to_law_object(
    query_object: dict[str, Any],
    candidate: dict[str, Any],
    *,
    family_key_field: str = "family_key",
) -> dict[str, Any]:
    query = dict(query_object.get("query", {}) or {})
    key = dict(candidate.get("key", {}) or {})
    q_compat = _cosine01(_safe_list(query.get("q_profile")), _safe_list(key.get("q_profile")))
    major = 1.0 - abs(_as_float(query.get("major_arc_proxy", 0.0)) - _as_float(key.get("major_arc_proxy", 0.0)))
    residue = 1.0 - abs(
        _as_float(query.get("minor_residue_proxy", 0.0)) - _as_float(key.get("minor_residue_proxy", 0.0))
    )
    support_overlap = _window_overlap(query, key)
    boundary = 1.0 - abs(_as_float(query.get("boundary_score", 0.5)) - _as_float(key.get("boundary_score", 0.5)))
    temporal = 1.0 - 0.65 * abs(_as_float(query.get("temporal_center", 0.5)) - _as_float(key.get("temporal_center", 0.5)))
    temporal -= 0.35 * abs(_as_float(query.get("temporal_width", 1.0)) - _as_float(key.get("temporal_width", 1.0)))
    same_object = str(query_object.get("object_id")) == str(candidate.get("object_id"))
    query_family = object_family_value(query_object, family_key_field)
    candidate_family = object_family_value(candidate, family_key_field)
    same_family = query_family == candidate_family
    reentry_risk = 0.35 if same_object else (0.10 if same_family else 0.0)
    components = {
        "q_compatibility": _clamp01(q_compat),
        "major_arc_overlap": _clamp01(major),
        "minor_residue_fit": _clamp01(residue),
        "support_overlap": _clamp01(support_overlap),
        "boundary_compatibility": _clamp01(boundary),
        "temporal_law_compatibility": _clamp01(temporal),
        "reentry_safety": _clamp01(1.0 - reentry_risk),
    }
    geometric_score = sum(GEOMETRIC_WEIGHTS[name] * components[name] for name in GEOMETRIC_COMPONENTS)
    learned_residual = _as_float(candidate.get("learned_residual", 0.0))
    final_score = _clamp01(geometric_score + learned_residual)
    return {
        "query_object_id": str(query_object.get("object_id", "")),
        "candidate_object_id": str(candidate.get("object_id", "")),
        "family_key_field": str(family_key_field or "family_key"),
        "query_family_key": query_family,
        "candidate_family_key": candidate_family,
        "query_local_family_key": object_family_value(query_object, "local_family_key"),
        "candidate_local_family_key": object_family_value(candidate, "local_family_key"),
        "query_structural_family_key": object_family_value(query_object, "structural_family_key"),
        "candidate_structural_family_key": object_family_value(candidate, "structural_family_key"),
        "same_object": bool(same_object),
        "same_family": bool(same_family),
        "components": components,
        "geometric_score": float(geometric_score),
        "learned_residual": float(learned_residual),
        "final_score": float(final_score),
    }


def _score_entropy(scores: list[float]) -> float:
    if not scores:
        return 0.0
    scaled = [math.exp(max(-50.0, min(50.0, 8.0 * (score - max(scores))))) for score in scores]
    total = sum(scaled)
    if total <= 1.0e-12:
        return 0.0
    probs = [x / total for x in scaled]
    entropy = -sum(p * math.log(max(1.0e-12, p)) for p in probs)
    return float(entropy / max(1.0e-12, math.log(max(2, len(probs)))))


def _mean(rows: list[float]) -> float:
    return float(sum(rows) / len(rows)) if rows else 0.0


def run_retrieval_assay(
    objects: list[dict[str, Any]],
    *,
    top_k: int = 3,
    family_key_field: str = "family_key",
) -> dict[str, Any]:
    clean_objects = [obj for obj in objects if isinstance(obj, dict) and obj.get("object_id")]
    rows: list[dict[str, Any]] = []
    for query in clean_objects:
        scored = [
            score_query_to_law_object(query, candidate, family_key_field=family_key_field)
            for candidate in clean_objects
        ]
        scored.sort(key=lambda row: float(row["final_score"]), reverse=True)
        if not scored:
            continue
        decoys = [row for row in scored if not row["same_family"]]
        positives = [row for row in scored if row["same_family"]]
        nonself = [row for row in scored if not row["same_object"]]
        nonself_positive = [row for row in nonself if row["same_family"]]
        top = scored[0]
        top_nonself = nonself[0] if nonself else None
        best_decoy_score = float(decoys[0]["final_score"]) if decoys else 0.0
        best_positive_score = float(positives[0]["final_score"]) if positives else 0.0
        topk = scored[: max(1, int(top_k))]
        rows.append(
            {
                "query_object_id": str(query.get("object_id")),
                "family_key_field": str(family_key_field or "family_key"),
                "query_family_key": object_family_value(query, family_key_field),
                "query_local_family_key": object_family_value(query, "local_family_key"),
                "query_structural_family_key": object_family_value(query, "structural_family_key"),
                "candidate_count": len(scored),
                "positive_count": len(positives),
                "nonself_positive_count": len(nonself_positive),
                "top_object_id": str(top.get("candidate_object_id")),
                "top_family_key": str(top.get("candidate_family_key")),
                "top_score": float(top["final_score"]),
                "top_geometric_score": float(top["geometric_score"]),
                "top_learned_residual": float(top["learned_residual"]),
                "top1_family_hit_including_self": 1.0 if bool(top["same_family"]) else 0.0,
                "top1_self_hit": 1.0 if bool(top["same_object"]) else 0.0,
                "top1_family_hit_excluding_self": (
                    1.0 if top_nonself is not None and bool(top_nonself["same_family"]) else 0.0
                ),
                "eligible_excluding_self": 1.0 if nonself_positive else 0.0,
                "topk_family_hit": 1.0 if any(bool(row["same_family"]) for row in topk) else 0.0,
                "best_positive_score": best_positive_score,
                "best_decoy_score": best_decoy_score,
                "resonance_margin": float(best_positive_score - best_decoy_score),
                "decoy_suppression_ratio": float(
                    best_positive_score / max(1.0e-8, best_decoy_score)
                    if best_decoy_score > 0.0
                    else (1.0 if best_positive_score > 0.0 else 0.0)
                ),
                "resonance_entropy": _score_entropy([float(row["final_score"]) for row in scored]),
                "top_components": dict(top["components"]),
            }
        )
    eligible = [row for row in rows if float(row["eligible_excluding_self"]) > 0.0]
    summary = {
        "object_count": len(clean_objects),
        "query_count": len(rows),
        "eligible_excluding_self_count": len(eligible),
        "family_key_field": str(family_key_field or "family_key"),
        "family_count": len({object_family_value(obj, family_key_field) for obj in clean_objects}),
        "local_family_count": len({object_family_value(obj, "local_family_key") for obj in clean_objects}),
        "structural_family_count": len({object_family_value(obj, "structural_family_key") for obj in clean_objects}),
        "top1_family_accuracy_including_self": _mean([float(row["top1_family_hit_including_self"]) for row in rows]),
        "top1_self_hit_fraction": _mean([float(row["top1_self_hit"]) for row in rows]),
        "top1_family_accuracy_excluding_self": _mean(
            [float(row["top1_family_hit_excluding_self"]) for row in eligible]
        ),
        "topk_family_accuracy": _mean([float(row["topk_family_hit"]) for row in rows]),
        "mean_resonance_margin": _mean([float(row["resonance_margin"]) for row in rows]),
        "mean_decoy_suppression_ratio": _mean([float(row["decoy_suppression_ratio"]) for row in rows]),
        "mean_resonance_entropy": _mean([float(row["resonance_entropy"]) for row in rows]),
        "mean_top_geometric_score": _mean([float(row["top_geometric_score"]) for row in rows]),
        "mean_top_learned_residual": _mean([float(row["top_learned_residual"]) for row in rows]),
        "geometry_components_reported": list(GEOMETRIC_COMPONENTS),
        "status": "pass_nonself_retrieval_signal"
        if eligible and _mean([float(row["top1_family_hit_excluding_self"]) for row in eligible]) > 0.0
        else ("pass_self_retrieval_smoke" if rows and _mean([float(row["top1_family_hit_including_self"]) for row in rows]) > 0.0 else "fail_no_retrieval_signal"),
    }
    return {"summary": summary, "rows": rows}


def composition_summary(objects: list[dict[str, Any]]) -> dict[str, Any]:
    children = [obj for obj in objects if str(obj.get("object_kind")) == "childworld"]
    pairs: list[dict[str, Any]] = []
    by_case_branch_id: dict[tuple[str, str, int], dict[str, Any]] = {}
    by_case_id: dict[tuple[str, int], dict[str, Any]] = {}
    by_id: dict[int, dict[str, Any]] = {}
    for obj in children:
        child_id = int(float(obj.get("child_id", -1)))
        case = str(obj.get("case", ""))
        branch = str(obj.get("branch", ""))
        by_case_branch_id.setdefault((case, branch, child_id), obj)
        by_case_id.setdefault((case, child_id), obj)
        by_id.setdefault(child_id, obj)
    for obj in children:
        parent_id = int(float(obj.get("parent_child_id", -1)))
        child_id = int(float(obj.get("child_id", -1)))
        if parent_id < 0 or parent_id == child_id:
            continue
        case = str(obj.get("case", ""))
        branch = str(obj.get("branch", ""))
        parent = by_case_branch_id.get((case, branch, parent_id))
        if parent is None:
            parent = by_case_id.get((case, parent_id))
        if parent is None:
            parent = by_id.get(parent_id)
        if parent is None:
            continue
        if str(parent.get("object_id")) == str(obj.get("object_id")):
            continue
        child_generation = _as_float(obj.get("generation", 0.0))
        parent_generation = _as_float(parent.get("generation", 0.0))
        if child_generation <= parent_generation:
            continue
        score = score_query_to_law_object(parent, obj)
        pairs.append(score)
    return {
        "childworld_object_count": len(children),
        "cross_depth_pair_count": len(pairs),
        "mean_cross_depth_geometric_score": _mean([float(row["geometric_score"]) for row in pairs]),
        "mean_cross_depth_q_compatibility": _mean(
            [float(row["components"]["q_compatibility"]) for row in pairs]
        ),
        "mean_cross_depth_support_overlap": _mean(
            [float(row["components"]["support_overlap"]) for row in pairs]
        ),
        "status": "cross_depth_pairs_present" if pairs else "no_cross_depth_pairs",
    }


def load_law_objects_from_json(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    objects: list[dict[str, Any]] = []
    if isinstance(payload, dict):
        direct = payload.get("resonant_law_objects")
        if isinstance(direct, list):
            objects.extend(obj for obj in direct if isinstance(obj, dict))
        if payload.get("branch_metrics"):
            objects.extend(law_objects_from_nested_case_summary(payload))
        for case in payload.get("cases", []) or []:
            if isinstance(case, dict):
                nested = case.get("resonant_law_objects")
                if isinstance(nested, list):
                    objects.extend(obj for obj in nested if isinstance(obj, dict))
                if case.get("branch_metrics"):
                    objects.extend(law_objects_from_nested_case_summary(case))
    return objects


def synthetic_law_objects() -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for family_idx, family in enumerate(("reed", "bell", "engine")):
        for variant in range(2):
            q_profile = _normalize_profile(
                [
                    1.0 if family_idx == 0 else 0.1,
                    1.0 if family_idx == 1 else 0.1,
                    1.0 if family_idx == 2 else 0.1,
                    0.25 + 0.05 * variant,
                ]
            )
            key = {
                "q_profile": q_profile,
                "dominant_q_share": max(q_profile),
                "major_arc_proxy": 0.75 - 0.05 * family_idx,
                "minor_residue_proxy": 0.10 + 0.15 * family_idx,
                "support_mean": 0.35 + 0.05 * variant,
                "coherence_mean": 0.80 - 0.05 * family_idx,
                "boundary_score": 0.72 + 0.03 * variant,
                "temporal_center": 0.25 + 0.20 * family_idx,
                "temporal_width": 0.25 + 0.05 * variant,
                "window_start": 0.15 + 0.20 * family_idx,
                "window_end": 0.40 + 0.20 * family_idx + 0.05 * variant,
            }
            out.append(
                {
                    "object_id": f"synthetic:{family}:{variant}",
                    "object_kind": "childworld",
                    "source": "synthetic_smoke",
                    "case": "synthetic",
                    "branch": f"{family}_{variant}",
                    "branch_family": family,
                    "family_key": f"synthetic:{family}",
                    "child_id": float(family_idx * 10 + variant),
                    "generation": 1.0 + float(variant),
                    "parent_child_id": float(family_idx * 10) if variant else -1.0,
                    "origin_child_id": float(family_idx * 10),
                    "query": dict(key),
                    "key": key,
                    "value": {
                        "operator_kind": "synthetic_phase_operator",
                        "operator_norm": 0.5 + 0.1 * variant,
                        "writeback_budget": 1.0,
                        "support_mass": key["support_mean"],
                        "coherence_mass": key["coherence_mean"],
                        "causal_use_ready": 1.0,
                    },
                }
            )
    return out


def write_assay_report(path: Path, payload: dict[str, Any]) -> None:
    summary = dict(payload.get("retrieval", {}).get("summary", {}) or {})
    composition = dict(payload.get("composition", {}) or {})
    case_retrieval = dict(payload.get("case_retrieval", {}) or {})
    retrieval_scopes = dict(payload.get("retrieval_scopes", {}) or {})
    lines = [
        "# Resonant Child Retrieval Assay",
        "",
        "This assay treats packet/child/branch artifacts as RAFA law objects and tests whether partial relational queries retrieve compatible families above decoys.",
        "",
        "## Summary",
        "",
        f"- status: `{summary.get('status', 'unknown')}`",
        f"- object count: `{summary.get('object_count', 0)}`",
        f"- family count: `{summary.get('family_count', 0)}`",
        f"- query count: `{summary.get('query_count', 0)}`",
        f"- eligible nonself queries: `{summary.get('eligible_excluding_self_count', 0)}`",
        f"- top1 family accuracy including self: `{summary.get('top1_family_accuracy_including_self', 0.0)}`",
        f"- top1 family accuracy excluding self: `{summary.get('top1_family_accuracy_excluding_self', 0.0)}`",
        f"- top-k family accuracy: `{summary.get('topk_family_accuracy', 0.0)}`",
        f"- mean resonance margin: `{summary.get('mean_resonance_margin', 0.0)}`",
        f"- mean decoy suppression ratio: `{summary.get('mean_decoy_suppression_ratio', 0.0)}`",
        f"- mean resonance entropy: `{summary.get('mean_resonance_entropy', 0.0)}`",
        f"- mean top geometric score: `{summary.get('mean_top_geometric_score', 0.0)}`",
        f"- mean top learned residual: `{summary.get('mean_top_learned_residual', 0.0)}`",
        "",
        "## Composition",
        "",
        f"- status: `{composition.get('status', 'unknown')}`",
        f"- childworld objects: `{composition.get('childworld_object_count', 0)}`",
        f"- cross-depth pairs: `{composition.get('cross_depth_pair_count', 0)}`",
        f"- mean cross-depth geometric score: `{composition.get('mean_cross_depth_geometric_score', 0.0)}`",
        f"- mean cross-depth q compatibility: `{composition.get('mean_cross_depth_q_compatibility', 0.0)}`",
        f"- mean cross-depth support overlap: `{composition.get('mean_cross_depth_support_overlap', 0.0)}`",
        "",
        "## Case-Local Retrieval",
        "",
        f"- cases: `{case_retrieval.get('case_count', 0)}`",
        f"- passing cases: `{case_retrieval.get('passing_case_count', 0)}`",
        f"- mean case-local top1 excluding self: `{case_retrieval.get('mean_case_top1_family_accuracy_excluding_self', 0.0)}`",
        f"- mean case-local resonance margin: `{case_retrieval.get('mean_case_resonance_margin', 0.0)}`",
        "",
        "| case | objects | status | top1 excl self | margin | cross-depth pairs |",
        "|---|---:|---|---:|---:|---:|",
    ]
    for row in case_retrieval.get("rows", []) or []:
        if not isinstance(row, dict):
            continue
        lines.append(
            "| "
            f"{row.get('case', 'unknown')} | "
            f"{row.get('object_count', 0)} | "
            f"{row.get('status', 'unknown')} | "
            f"{row.get('top1_family_accuracy_excluding_self', 0.0)} | "
            f"{row.get('mean_resonance_margin', 0.0)} | "
            f"{row.get('cross_depth_pair_count', 0)} |"
        )
    if retrieval_scopes:
        lines.extend(
            [
                "",
                "## Retrieval Scopes",
                "",
                "| scope | family field | families | status | top1 excl self | margin | case-local top1 | top cross-case | top same-structural |",
                "|---|---|---:|---|---:|---:|---:|---:|---:|",
            ]
        )
        for scope_name, scope_payload in retrieval_scopes.items():
            if not isinstance(scope_payload, dict):
                continue
            scope_summary = dict(scope_payload.get("summary", {}) or {})
            scope_case = dict(scope_payload.get("case_retrieval", {}) or {})
            scope_top = dict(scope_payload.get("top_candidate_case_summary", {}) or {})
            lines.append(
                "| "
                f"{scope_name} | "
                f"{scope_payload.get('family_key_field', scope_summary.get('family_key_field', 'family_key'))} | "
                f"{scope_summary.get('family_count', 0)} | "
                f"{scope_summary.get('status', 'unknown')} | "
                f"{scope_summary.get('top1_family_accuracy_excluding_self', 0.0)} | "
                f"{scope_summary.get('mean_resonance_margin', 0.0)} | "
                f"{scope_case.get('mean_case_top1_family_accuracy_excluding_self', 0.0)} | "
                f"{scope_top.get('top_cross_case_fraction', 0.0)} | "
                f"{scope_top.get('top_same_structural_family_fraction', 0.0)} |"
            )
    lines.extend(
        [
            "",
            "## Geometry Components",
            "",
        ]
    )
    for component in summary.get("geometry_components_reported", GEOMETRIC_COMPONENTS):
        lines.append(f"- `{component}`")
    lines.extend(["", "## Sources", ""])
    for source in payload.get("source_files", []):
        lines.append(f"- `{source}`")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
