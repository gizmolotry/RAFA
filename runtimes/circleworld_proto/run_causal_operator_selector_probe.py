from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
LINEAGE = ROOT / "lineages" / "04_positive_replacement"
RUNTIME = Path(__file__).resolve().parent
for path in (ROOT, LINEAGE, RUNTIME):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from ablate_formalization import _generate_seed_phase, make_seed_rafa  # noqa: E402
from causal_operator_selector import (  # noqa: E402
    PAIR_SCALAR_NAMES,
    CausalOperatorSelector,
    MultiHeadCausalOperatorSelector,
    causal_pair_feature_dim,
    causal_pair_feature_vector,
    selector_route_scores,
)
from evaluate_circleworld import _load_circle_cfg, _safe_device  # noqa: E402
from resonant_law_objects import (  # noqa: E402
    object_family_value,
    run_retrieval_assay,
    score_query_to_law_object,
)
from run_contrastive_structural_embedding_probe import operator_safety_proxy, recurrence_compatibility  # noqa: E402
from run_resonant_operator_causality_assay import (  # noqa: E402
    _apply_child_operator,
    _object_child_mapping,
    _select_fork_state,
    _select_operator_triplet,
)
from test_nested_commitment import (  # noqa: E402
    ASSAY_KILLSWITCH,
    DEFAULT_ASSAY_KILLSWITCH,
    _apply_assay_child_partitioning,
    _generate_seed_magnitude,
    _run_depth_trace,
)


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _mean(values: list[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def _strong_recurrence_pass(row: dict[str, Any], recurrence_threshold: float, geometric_threshold: float) -> bool:
    return (
        _as_float(row.get("recurrence_compatibility", 0.0)) >= float(recurrence_threshold)
        and _as_float(row.get("geometric_score", 0.0)) >= float(geometric_threshold)
    )


def _target_rows_for_query(
    rows: list[dict[str, Any]],
    *,
    decoy_cap: float,
) -> list[dict[str, Any]]:
    max_div = max([_as_float(row.get("parent_phase_divergence", 0.0)) for row in rows] + [1.0e-8])
    max_jump = max([_as_float(row.get("world_jump_proxy", 0.0)) for row in rows] + [1.0e-8])
    out: list[dict[str, Any]] = []
    for row in rows:
        movement = _as_float(row.get("parent_phase_divergence", 0.0)) / max_div
        jump = _as_float(row.get("world_jump_proxy", 0.0)) / max_jump
        recurrence = _as_float(row.get("recurrence_compatibility", 0.0))
        identity = max(_as_float(row.get("same_local_family", 0.0)), _as_float(row.get("same_structural_family", 0.0)))
        membrane = _as_float(row.get("identity_membrane_pass", 0.0))
        target = 0.42 * movement + 0.20 * (1.0 - jump) + 0.20 * recurrence + 0.18 * identity
        if membrane <= 0.0:
            target = min(float(decoy_cap), target)
        clone = dict(row)
        clone.update(
            {
                "movement_norm": float(movement),
                "world_jump_norm": float(jump),
                "causal_operator_target": float(max(0.0, min(1.0, target))),
            }
        )
        out.append(clone)
    return out


def _hard_decoy_summary(rows: list[dict[str, Any]]) -> dict[str, float]:
    """Summarize wrong-family candidates that move the parent more while jumping harder."""
    by_query: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_query.setdefault(str(row.get("query_object_id", "")), []).append(row)

    comparable = 0
    hard_queries = 0
    safety_wins = 0
    movement_tradeoffs = 0
    high_movement_wrong_family_count = 0
    high_jump_wrong_family_count = 0
    hard_wrong_family_count = 0
    movement_gaps: list[float] = []
    jump_gaps: list[float] = []
    membrane_divs: list[float] = []
    decoy_divs: list[float] = []
    membrane_jumps: list[float] = []
    decoy_jumps: list[float] = []

    for query_rows in by_query.values():
        membrane_rows = [row for row in query_rows if _as_float(row.get("identity_membrane_pass", 0.0)) > 0.0]
        decoy_rows = [row for row in query_rows if _as_float(row.get("identity_membrane_pass", 0.0)) <= 0.0]
        if not membrane_rows or not decoy_rows:
            continue
        comparable += 1
        best_membrane = max(
            membrane_rows,
            key=lambda row: (
                _as_float(row.get("causal_operator_target", 0.0)),
                _as_float(row.get("recurrence_compatibility", 0.0)),
                _as_float(row.get("parent_phase_divergence", 0.0)),
            ),
        )
        best_decoy = max(
            decoy_rows,
            key=lambda row: (
                _as_float(row.get("parent_phase_divergence", 0.0)),
                _as_float(row.get("world_jump_proxy", 0.0)),
                _as_float(row.get("geometric_score", 0.0)),
            ),
        )
        membrane_div = _as_float(best_membrane.get("parent_phase_divergence", 0.0))
        decoy_div = _as_float(best_decoy.get("parent_phase_divergence", 0.0))
        membrane_jump = _as_float(best_membrane.get("world_jump_proxy", 0.0))
        decoy_jump = _as_float(best_decoy.get("world_jump_proxy", 0.0))
        movement_gap = decoy_div - membrane_div
        jump_gap = decoy_jump - membrane_jump

        membrane_divs.append(membrane_div)
        decoy_divs.append(decoy_div)
        membrane_jumps.append(membrane_jump)
        decoy_jumps.append(decoy_jump)
        movement_gaps.append(movement_gap)
        jump_gaps.append(jump_gap)
        if movement_gap > 0.0:
            movement_tradeoffs += 1
        if jump_gap > 0.0:
            safety_wins += 1
        if movement_gap > 0.0 and jump_gap > 0.0:
            hard_queries += 1

        for decoy in decoy_rows:
            decoy_row_div = _as_float(decoy.get("parent_phase_divergence", 0.0))
            decoy_row_jump = _as_float(decoy.get("world_jump_proxy", 0.0))
            if decoy_row_div > membrane_div:
                high_movement_wrong_family_count += 1
            if decoy_row_jump > membrane_jump:
                high_jump_wrong_family_count += 1
            if decoy_row_div > membrane_div and decoy_row_jump > membrane_jump:
                hard_wrong_family_count += 1

    return {
        "query_count": float(len(by_query)),
        "comparable_query_count": float(comparable),
        "hard_decoy_query_count": float(hard_queries),
        "hard_decoy_fraction": float(hard_queries / comparable) if comparable else 0.0,
        "movement_tradeoff_fraction": float(movement_tradeoffs / comparable) if comparable else 0.0,
        "membrane_safety_win_fraction": float(safety_wins / comparable) if comparable else 0.0,
        "high_movement_wrong_family_count": float(high_movement_wrong_family_count),
        "high_jump_wrong_family_count": float(high_jump_wrong_family_count),
        "hard_wrong_family_count": float(hard_wrong_family_count),
        "mean_best_membrane_parent_phase_divergence": _mean(membrane_divs),
        "mean_best_decoy_parent_phase_divergence": _mean(decoy_divs),
        "mean_best_membrane_world_jump_proxy": _mean(membrane_jumps),
        "mean_best_decoy_world_jump_proxy": _mean(decoy_jumps),
        "mean_best_decoy_movement_gap": _mean(movement_gaps),
        "mean_best_decoy_jump_gap": _mean(jump_gaps),
    }


def _collect_case_rows(
    *,
    cfg: Any,
    rafa_core: Any,
    source: str,
    seed: int,
    time_steps: int,
    warmup_depth: int,
    device: torch.device,
    operator_gain: float,
    recurrence_threshold: float,
    geometric_threshold: float,
    decoy_cap: float,
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
    case_name = f"{source}_{seed}"
    objects, by_id = _object_child_mapping(branch_state, case=case_name, branch="causal_operator")
    by_object_id = {str(obj.get("object_id", "")): obj for obj in objects}
    retrieval = run_retrieval_assay(objects, top_k=3)
    query_id, selected_id, decoy_id, row, decoy_row = _select_operator_triplet(objects, retrieval)
    rows: list[dict[str, Any]] = []
    for query in objects:
        query_id_full = str(query.get("object_id", ""))
        query_rows: list[dict[str, Any]] = []
        for candidate in objects:
            candidate_id = str(candidate.get("object_id", ""))
            if candidate_id == query_id_full:
                continue
            child = by_id.get(candidate_id)
            if child is None:
                continue
            score = score_query_to_law_object(query, candidate, family_key_field="local_family_key")
            recurrence = float(recurrence_compatibility(query, candidate))
            same_structural = (
                object_family_value(query, "structural_family_key")
                == object_family_value(candidate, "structural_family_key")
            )
            strong = _strong_recurrence_pass(
                {
                    "recurrence_compatibility": recurrence,
                    "geometric_score": score.get("geometric_score", 0.0),
                },
                recurrence_threshold,
                geometric_threshold,
            )
            membrane = bool(score.get("same_family", False)) or same_structural or strong
            metrics = _apply_child_operator(branch_state, child, gain=operator_gain)
            query_rows.append(
                {
                    "case": case_name,
                    "seed": int(seed),
                    "fork_depth": int(fork_idx),
                    "query_object_id": query_id_full,
                    "candidate_object_id": candidate_id,
                    "geometric_score": _as_float(score.get("geometric_score", 0.0)),
                    "recurrence_compatibility": recurrence,
                    "same_local_family": 1.0 if bool(score.get("same_family", False)) else 0.0,
                    "same_structural_family": 1.0 if same_structural else 0.0,
                    "identity_membrane_pass": 1.0 if membrane else 0.0,
                    "strong_recurrence_pass": 1.0 if strong else 0.0,
                    "query_safety_proxy": float(operator_safety_proxy(query)),
                    "candidate_safety_proxy": float(operator_safety_proxy(candidate)),
                    "parent_phase_divergence": _as_float(metrics.get("parent_phase_divergence", 0.0)),
                    "world_jump_proxy": _as_float(metrics.get("world_jump_proxy", 0.0)),
                    "support_shift": _as_float(metrics.get("support_shift", 0.0)),
                    "qtrace_shift_proxy": _as_float(metrics.get("qtrace_shift_proxy", 0.0)),
                }
            )
        rows.extend(_target_rows_for_query(query_rows, decoy_cap=decoy_cap))
    return {
        "case": case_name,
        "seed": int(seed),
        "fork_depth": int(fork_idx),
        "law_object_count": len(objects),
        "candidate_row_count": len(rows),
        "assay_query_object_id": query_id,
        "assay_selected_object_id": selected_id,
        "assay_decoy_object_id": decoy_id,
        "assay_selected_top_score": _as_float(row.get("top_score", 0.0)) if row else 0.0,
        "assay_decoy_score": _as_float(decoy_row.get("final_score", 0.0)) if decoy_row else 0.0,
        "partition": partition,
        "hard_decoy_summary": _hard_decoy_summary(rows),
        "rows": rows,
        "objects": by_object_id,
    }


def _features_and_targets(rows: list[dict[str, Any]], objects: dict[str, dict[str, Any]], device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    features: list[list[float]] = []
    targets: list[float] = []
    for row in rows:
        query = objects[str(row.get("query_object_id", ""))]
        candidate = objects[str(row.get("candidate_object_id", ""))]
        features.append(causal_pair_feature_vector(query, candidate, row))
        targets.append(_as_float(row.get("causal_operator_target", 0.0)))
    return torch.tensor(features, dtype=torch.float32, device=device), torch.tensor(targets, dtype=torch.float32, device=device)


def _target_heads(rows: list[dict[str, Any]], route_targets: torch.Tensor, device: torch.device) -> dict[str, torch.Tensor]:
    return {
        "route_score": route_targets,
        "movement": torch.tensor([_as_float(row.get("movement_norm", 0.0)) for row in rows], dtype=torch.float32, device=device),
        "jump_safety": torch.tensor(
            [1.0 - _as_float(row.get("world_jump_norm", 1.0)) for row in rows],
            dtype=torch.float32,
            device=device,
        ),
        "identity": torch.tensor(
            [
                max(_as_float(row.get("same_local_family", 0.0)), _as_float(row.get("same_structural_family", 0.0)))
                for row in rows
            ],
            dtype=torch.float32,
            device=device,
        ),
    }


def _make_model(model_kind: str, feature_dim: int, hidden_dim: int) -> torch.nn.Module:
    if str(model_kind).lower() == "multihead":
        return MultiHeadCausalOperatorSelector(feature_dim, hidden_dim=hidden_dim)
    return CausalOperatorSelector(feature_dim, hidden_dim=hidden_dim)


def _train_model(
    model: torch.nn.Module,
    features: torch.Tensor,
    targets: torch.Tensor,
    rows: list[dict[str, Any]],
    *,
    epochs: int,
    batch_size: int,
    lr: float,
    model_kind: str,
) -> dict[str, float]:
    if features.numel() == 0:
        return {"final_loss": 0.0}
    target_heads = _target_heads(rows, targets, features.device)
    opt = torch.optim.AdamW(model.parameters(), lr=float(lr), weight_decay=1.0e-3)
    last_loss = 0.0
    last_route_loss = 0.0
    last_movement_loss = 0.0
    last_jump_loss = 0.0
    last_identity_loss = 0.0
    for _ in range(max(1, int(epochs))):
        perm = torch.randperm(features.size(0), device=features.device)
        losses: list[float] = []
        route_losses: list[float] = []
        movement_losses: list[float] = []
        jump_losses: list[float] = []
        identity_losses: list[float] = []
        for start in range(0, int(features.size(0)), max(1, int(batch_size))):
            idx = perm[start : start + max(1, int(batch_size))]
            output = model(features[idx])
            pred = selector_route_scores(output)
            route_loss = F.mse_loss(pred, targets[idx])
            if isinstance(output, dict) and str(model_kind).lower() == "multihead":
                movement_loss = F.mse_loss(output["movement"], target_heads["movement"][idx])
                jump_loss = F.mse_loss(output["jump_safety"], target_heads["jump_safety"][idx])
                identity_loss = F.binary_cross_entropy(output["identity"].clamp(1.0e-6, 1.0 - 1.0e-6), target_heads["identity"][idx])
                loss = route_loss + 0.20 * movement_loss + 0.20 * jump_loss + 0.20 * identity_loss
            else:
                movement_loss = torch.zeros((), device=features.device)
                jump_loss = torch.zeros((), device=features.device)
                identity_loss = torch.zeros((), device=features.device)
                loss = route_loss
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            losses.append(float(loss.detach().cpu().item()))
            route_losses.append(float(route_loss.detach().cpu().item()))
            movement_losses.append(float(movement_loss.detach().cpu().item()))
            jump_losses.append(float(jump_loss.detach().cpu().item()))
            identity_losses.append(float(identity_loss.detach().cpu().item()))
        last_loss = _mean(losses)
        last_route_loss = _mean(route_losses)
        last_movement_loss = _mean(movement_losses)
        last_jump_loss = _mean(jump_losses)
        last_identity_loss = _mean(identity_losses)
    return {
        "final_loss": float(last_loss),
        "final_route_loss": float(last_route_loss),
        "final_movement_loss": float(last_movement_loss),
        "final_jump_safety_loss": float(last_jump_loss),
        "final_identity_loss": float(last_identity_loss),
    }


def _evaluate_rows(
    model: torch.nn.Module,
    rows: list[dict[str, Any]],
    objects: dict[str, dict[str, Any]],
    device: torch.device,
) -> dict[str, Any]:
    by_query: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_query.setdefault(str(row.get("query_object_id", "")), []).append(row)
    out_rows: list[dict[str, Any]] = []
    model.eval()
    with torch.no_grad():
        for query_id, query_rows in sorted(by_query.items()):
            feats, _ = _features_and_targets(query_rows, objects, device)
            output = model(feats)
            pred = selector_route_scores(output).detach().cpu().tolist()
            movement_pred = output["movement"].detach().cpu().tolist() if isinstance(output, dict) else [0.0] * len(query_rows)
            jump_pred = output["jump_safety"].detach().cpu().tolist() if isinstance(output, dict) else [0.0] * len(query_rows)
            identity_pred = output["identity"].detach().cpu().tolist() if isinstance(output, dict) else [0.0] * len(query_rows)
            scored: list[dict[str, Any]] = []
            for row, score, move_score, jump_score, ident_score in zip(query_rows, pred, movement_pred, jump_pred, identity_pred):
                clone = dict(row)
                clone["model_score"] = float(score)
                clone["movement_pred"] = float(move_score)
                clone["jump_safety_pred"] = float(jump_score)
                clone["identity_pred"] = float(ident_score)
                scored.append(clone)
            raw_top = max(scored, key=lambda row: float(row.get("model_score", 0.0)))
            membrane_top = max(
                scored,
                key=lambda row: (
                    float(row.get("identity_membrane_pass", 0.0)),
                    float(row.get("model_score", 0.0)),
                    float(row.get("causal_operator_target", 0.0)),
                ),
            )
            recurrence_top = max(
                scored,
                key=lambda row: (
                    float(row.get("recurrence_compatibility", 0.0)),
                    float(row.get("geometric_score", 0.0)),
                ),
            )
            target_top = max(scored, key=lambda row: float(row.get("causal_operator_target", 0.0)))
            out_rows.append(
                {
                    "query_object_id": query_id,
                    "raw_candidate_object_id": str(raw_top.get("candidate_object_id", "")),
                    "membrane_candidate_object_id": str(membrane_top.get("candidate_object_id", "")),
                    "recurrence_candidate_object_id": str(recurrence_top.get("candidate_object_id", "")),
                    "target_candidate_object_id": str(target_top.get("candidate_object_id", "")),
                    "raw_model_score": float(raw_top.get("model_score", 0.0)),
                    "membrane_model_score": float(membrane_top.get("model_score", 0.0)),
                    "membrane_movement_pred": float(membrane_top.get("movement_pred", 0.0)),
                    "membrane_jump_safety_pred": float(membrane_top.get("jump_safety_pred", 0.0)),
                    "membrane_identity_pred": float(membrane_top.get("identity_pred", 0.0)),
                    "target_top_score": float(target_top.get("causal_operator_target", 0.0)),
                    "raw_target_score": float(raw_top.get("causal_operator_target", 0.0)),
                    "membrane_target_score": float(membrane_top.get("causal_operator_target", 0.0)),
                    "membrane_parent_phase_divergence": float(membrane_top.get("parent_phase_divergence", 0.0)),
                    "membrane_world_jump_proxy": float(membrane_top.get("world_jump_proxy", 0.0)),
                    "membrane_same_local_family": float(membrane_top.get("same_local_family", 0.0)),
                    "membrane_same_structural_family": float(membrane_top.get("same_structural_family", 0.0)),
                    "membrane_identity_pass": float(membrane_top.get("identity_membrane_pass", 0.0)),
                    "membrane_agrees_with_recurrence": 1.0
                    if str(membrane_top.get("candidate_object_id", ""))
                    == str(recurrence_top.get("candidate_object_id", ""))
                    else 0.0,
                    "membrane_agrees_with_target": 1.0
                    if str(membrane_top.get("candidate_object_id", ""))
                    == str(target_top.get("candidate_object_id", ""))
                    else 0.0,
                    "raw_identity_pass": float(raw_top.get("identity_membrane_pass", 0.0)),
                }
            )
    return {
        "query_count": len(out_rows),
        "mean_membrane_target_score": _mean([float(row["membrane_target_score"]) for row in out_rows]),
        "mean_target_top_score": _mean([float(row["target_top_score"]) for row in out_rows]),
        "mean_membrane_parent_phase_divergence": _mean([float(row["membrane_parent_phase_divergence"]) for row in out_rows]),
        "mean_membrane_world_jump_proxy": _mean([float(row["membrane_world_jump_proxy"]) for row in out_rows]),
        "mean_membrane_identity_pass": _mean([float(row["membrane_identity_pass"]) for row in out_rows]),
        "mean_membrane_same_local_family": _mean([float(row["membrane_same_local_family"]) for row in out_rows]),
        "mean_membrane_same_structural_family": _mean([float(row["membrane_same_structural_family"]) for row in out_rows]),
        "mean_membrane_agrees_with_recurrence": _mean([float(row["membrane_agrees_with_recurrence"]) for row in out_rows]),
        "mean_membrane_agrees_with_target": _mean([float(row["membrane_agrees_with_target"]) for row in out_rows]),
        "mean_raw_identity_pass": _mean([float(row["raw_identity_pass"]) for row in out_rows]),
        "hard_decoy_summary": _hard_decoy_summary(rows),
        "rows": out_rows,
    }


def _write_markdown(path: Path, payload: dict[str, Any]) -> None:
    summary = payload["summary"]
    lines = [
        "# Causal Operator Selector Probe",
        "",
        "This assay trains an operator router from actual candidate writeback outcomes. It is assay-only and does not change Circleworld runtime.",
        "",
        "## Summary",
        "",
        f"- status: `{summary['status']}`",
        f"- cases: `{summary['case_count']}`",
        f"- total candidate rows: `{summary['candidate_row_count']}`",
        f"- model kind: `{summary.get('model_kind', 'single')}`",
        f"- final training loss: `{summary['final_loss']}`",
        f"- final route loss: `{summary.get('final_route_loss', 0.0)}`",
        f"- final movement loss: `{summary.get('final_movement_loss', 0.0)}`",
        f"- final jump-safety loss: `{summary.get('final_jump_safety_loss', 0.0)}`",
        f"- final identity loss: `{summary.get('final_identity_loss', 0.0)}`",
        f"- mean holdout membrane target score: `{summary['mean_holdout_membrane_target_score']}`",
        f"- mean holdout target-top score: `{summary['mean_holdout_target_top_score']}`",
        f"- mean holdout membrane identity pass: `{summary['mean_holdout_membrane_identity_pass']}`",
        f"- mean holdout membrane agreement with recurrence: `{summary['mean_holdout_membrane_agrees_with_recurrence']}`",
        f"- hard-decoy fraction: `{summary.get('hard_decoy_fraction', 0.0)}`",
        f"- hard wrong-family candidate count: `{summary.get('hard_wrong_family_count', 0.0)}`",
        f"- mean best decoy movement gap: `{summary.get('mean_best_decoy_movement_gap', 0.0)}`",
        f"- mean best decoy jump gap: `{summary.get('mean_best_decoy_jump_gap', 0.0)}`",
        f"- membrane safety win fraction: `{summary.get('membrane_safety_win_fraction', 0.0)}`",
        f"- checkpoint: `{payload.get('checkpoint_path', '')}`",
        "",
        "## Holdout Folds",
        "",
        "| holdout | queries | target score | target top | identity pass | same local | agrees recurrence | parent div | world jump |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for fold in payload["folds"]:
        eval_summary = fold["evaluation"]
        lines.append(
            "| {holdout} | {queries} | {target} | {target_top} | {identity} | {local} | {agree_rec} | {div} | {jump} |".format(
                holdout=fold["holdout_case"],
                queries=eval_summary["query_count"],
                target=eval_summary["mean_membrane_target_score"],
                target_top=eval_summary["mean_target_top_score"],
                identity=eval_summary["mean_membrane_identity_pass"],
                local=eval_summary["mean_membrane_same_local_family"],
                agree_rec=eval_summary["mean_membrane_agrees_with_recurrence"],
                div=eval_summary["mean_membrane_parent_phase_divergence"],
                jump=eval_summary["mean_membrane_world_jump_proxy"],
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_probe(
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
    recurrence_threshold: float,
    geometric_threshold: float,
    decoy_cap: float,
    epochs: int,
    batch_size: int,
    lr: float,
    hidden_dim: int,
    model_kind: str,
    seed: int,
) -> dict[str, str]:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    cfg = _load_circle_cfg(config_path)
    device = _safe_device(device_name)
    rafa_core = make_seed_rafa(dev=device.type) if seed_source == "naked_rafa" else None
    previous = dict(ASSAY_KILLSWITCH)
    case_payloads: list[dict[str, Any]] = []
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
        for run_seed in seeds:
            case_payloads.append(
                _collect_case_rows(
                    cfg=cfg,
                    rafa_core=rafa_core,
                    source=seed_source,
                    seed=int(run_seed),
                    time_steps=int(time_steps),
                    warmup_depth=int(warmup_depth),
                    device=device,
                    operator_gain=float(operator_gain),
                    recurrence_threshold=float(recurrence_threshold),
                    geometric_threshold=float(geometric_threshold),
                    decoy_cap=float(decoy_cap),
                )
            )
    finally:
        ASSAY_KILLSWITCH.clear()
        ASSAY_KILLSWITCH.update(previous)

    all_objects: dict[str, dict[str, Any]] = {}
    all_rows: list[dict[str, Any]] = []
    for case in case_payloads:
        all_objects.update(case["objects"])
        all_rows.extend(case["rows"])
    feature_dim = causal_pair_feature_dim()
    folds: list[dict[str, Any]] = []
    for holdout in sorted({str(row["case"]) for row in all_rows}):
        train_rows = [row for row in all_rows if str(row["case"]) != holdout]
        holdout_rows = [row for row in all_rows if str(row["case"]) == holdout]
        train_features, train_targets = _features_and_targets(train_rows, all_objects, device)
        model = _make_model(model_kind, feature_dim, hidden_dim=hidden_dim).to(device)
        train_summary = _train_model(
            model,
            train_features,
            train_targets,
            train_rows,
            epochs=epochs,
            batch_size=batch_size,
            lr=lr,
            model_kind=model_kind,
        )
        evaluation = _evaluate_rows(model, holdout_rows, all_objects, device)
        folds.append(
            {
                "holdout_case": holdout,
                "train_row_count": len(train_rows),
                "holdout_row_count": len(holdout_rows),
                "training": train_summary,
                "evaluation": evaluation,
            }
        )

    train_features, train_targets = _features_and_targets(all_rows, all_objects, device)
    final_model = _make_model(model_kind, feature_dim, hidden_dim=hidden_dim).to(device)
    final_train = _train_model(
        final_model,
        train_features,
        train_targets,
        all_rows,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        model_kind=model_kind,
    )
    hard_decoy_summary = _hard_decoy_summary(all_rows)
    summary = {
        "status": "pass_causal_operator_selector_probe"
        if _mean([fold["evaluation"]["mean_membrane_identity_pass"] for fold in folds]) > 0.0
        else "fail_causal_operator_selector_probe",
        "case_count": len(case_payloads),
        "candidate_row_count": len(all_rows),
        "feature_dim": int(feature_dim),
        "model_kind": str(model_kind),
        **final_train,
        "mean_holdout_membrane_target_score": _mean(
            [fold["evaluation"]["mean_membrane_target_score"] for fold in folds]
        ),
        "mean_holdout_target_top_score": _mean([fold["evaluation"]["mean_target_top_score"] for fold in folds]),
        "mean_holdout_membrane_identity_pass": _mean(
            [fold["evaluation"]["mean_membrane_identity_pass"] for fold in folds]
        ),
        "mean_holdout_membrane_same_local_family": _mean(
            [fold["evaluation"]["mean_membrane_same_local_family"] for fold in folds]
        ),
        "mean_holdout_membrane_agrees_with_recurrence": _mean(
            [fold["evaluation"]["mean_membrane_agrees_with_recurrence"] for fold in folds]
        ),
        "hard_decoy_fraction": float(hard_decoy_summary.get("hard_decoy_fraction", 0.0)),
        "hard_decoy_query_count": float(hard_decoy_summary.get("hard_decoy_query_count", 0.0)),
        "hard_wrong_family_count": float(hard_decoy_summary.get("hard_wrong_family_count", 0.0)),
        "mean_best_decoy_movement_gap": float(hard_decoy_summary.get("mean_best_decoy_movement_gap", 0.0)),
        "mean_best_decoy_jump_gap": float(hard_decoy_summary.get("mean_best_decoy_jump_gap", 0.0)),
        "membrane_safety_win_fraction": float(hard_decoy_summary.get("membrane_safety_win_fraction", 0.0)),
    }
    payload = {
        "config_path": str(config_path),
        "device_requested": device_name,
        "device": str(device),
        "seed_source": seed_source,
        "seeds": [int(run_seed) for run_seed in seeds],
        "time_steps": int(time_steps),
        "warmup_depth": int(warmup_depth),
        "operator_gain": float(operator_gain),
        "grandchild_depth": int(grandchild_depth),
        "grandchild_parts": int(grandchild_parts),
        "recurrence_threshold": float(recurrence_threshold),
        "geometric_threshold": float(geometric_threshold),
        "decoy_cap": float(decoy_cap),
        "pair_scalar_names": list(PAIR_SCALAR_NAMES),
        "summary": summary,
        "hard_decoy_summary": hard_decoy_summary,
        "case_summaries": [
            {key: value for key, value in case.items() if key not in {"rows", "objects"}}
            for case in case_payloads
        ],
        "folds": folds,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "causal_operator_selector_probe.json"
    md_path = out_dir / "CAUSAL_OPERATOR_SELECTOR_PROBE.md"
    checkpoint_path = out_dir / "causal_operator_selector_probe.pt"
    torch.save(
        {
            "model_state_dict": final_model.state_dict(),
            "feature_dim": int(feature_dim),
            "hidden_dim": int(hidden_dim),
            "model_kind": str(model_kind),
            "pair_scalar_names": list(PAIR_SCALAR_NAMES),
            "config": {
                "recurrence_threshold": float(recurrence_threshold),
                "geometric_threshold": float(geometric_threshold),
                "decoy_cap": float(decoy_cap),
                "epochs": int(epochs),
                "batch_size": int(batch_size),
                "lr": float(lr),
                "model_kind": str(model_kind),
                "seed": int(seed),
            },
            "summary": summary,
        },
        checkpoint_path,
    )
    payload["checkpoint_path"] = str(checkpoint_path)
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    _write_markdown(md_path, payload)
    return {"json": str(json_path), "markdown": str(md_path), "checkpoint": str(checkpoint_path), "status": summary["status"]}


def _parse_seeds(raw: str) -> list[int]:
    return [int(part.strip()) for part in str(raw).replace(";", ",").split(",") if part.strip()]


def main() -> None:
    ap = argparse.ArgumentParser(description="Train/evaluate a causal operator selector over candidate child writebacks.")
    ap.add_argument("--config", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed-source", choices=["synthetic", "naked_rafa"], default="naked_rafa")
    ap.add_argument("--seeds", default="9100,9101,9102")
    ap.add_argument("--time-steps", type=int, default=64)
    ap.add_argument("--warmup-depth", type=int, default=3)
    ap.add_argument("--operator-gain", type=float, default=0.85)
    ap.add_argument("--grandchild-depth", type=int, default=1)
    ap.add_argument("--grandchild-parts", type=int, default=2)
    ap.add_argument("--recurrence-threshold", type=float, default=0.96)
    ap.add_argument("--geometric-threshold", type=float, default=0.94)
    ap.add_argument("--decoy-cap", type=float, default=0.35)
    ap.add_argument("--epochs", type=int, default=240)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=0.002)
    ap.add_argument("--hidden-dim", type=int, default=96)
    ap.add_argument("--model-kind", choices=["single", "multihead"], default="single")
    ap.add_argument("--seed", type=int, default=2026)
    args = ap.parse_args()
    print(
        json.dumps(
            run_probe(
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
                recurrence_threshold=float(args.recurrence_threshold),
                geometric_threshold=float(args.geometric_threshold),
                decoy_cap=float(args.decoy_cap),
                epochs=int(args.epochs),
                batch_size=int(args.batch_size),
                lr=float(args.lr),
                hidden_dim=int(args.hidden_dim),
                model_kind=str(args.model_kind),
                seed=int(args.seed),
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
