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
from build_shadow_branch_law_table import shadow_branch_law_row  # noqa: E402
from circleworld import clone_circleworld_state, learned_branch_law_output_names, learned_branch_law_output_tensor  # noqa: E402
from evaluate_circleworld import _load_circle_cfg, _safe_device  # noqa: E402
from evaluate_shadow_branch_law_calibration import _load_model, shadow_prediction_to_branch_fields  # noqa: E402
from rafa_math_tools import phasor_apply_delta, phasor_normalize  # noqa: E402
from resonant_law_objects import run_retrieval_assay  # noqa: E402
from run_learned_gated_writeback_sandbox import _soft_writeback_scale  # noqa: E402
from run_resonant_operator_causality_assay import (  # noqa: E402
    _apply_child_operator,
    _load_causal_selector,
    _object_child_mapping,
    _phase_delta_from_to,
    _select_causal_candidate,
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
from train_shadow_branch_law_from_table import shadow_branch_feature_names, shadow_row_to_feature_target  # noqa: E402


def _as_float(value: Any, default: float = 0.0) -> float:
    if torch.is_tensor(value):
        if value.numel() == 0:
            return float(default)
        return float(value.detach().float().mean().cpu().item())
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _mean(values: list[float]) -> float:
    return float(np.mean(values)) if values else 0.0


def _masked_mean(value: torch.Tensor, mask: torch.Tensor | None = None) -> float:
    if mask is None or mask.shape != value.shape:
        return float(value.detach().float().mean().cpu().item())
    weight = mask.detach().float().clamp(0.0, 1.0)
    denom = float(weight.sum().clamp_min(1.0e-8).item())
    return float((value.detach().float() * weight).sum().item() / denom)


def _child_gate(state: dict[str, Any], child: dict[str, Any]) -> torch.Tensor:
    parent_mode1 = state["phase_modes"][..., 1, :]
    support = child.get("mode_support")
    if not torch.is_tensor(support) or support.shape != parent_mode1.shape[:-1]:
        support = torch.ones(parent_mode1.shape[:-1], device=parent_mode1.device, dtype=parent_mode1.dtype)
    coherence = child.get("mode_coherence")
    if not torch.is_tensor(coherence) or coherence.shape != support.shape:
        coherence = torch.ones_like(support)
    return (support.clamp(0.0, 1.0) * (0.20 + 0.80 * coherence.clamp(0.0, 1.0))).clamp(0.0, 1.0)


def _apply_scaled_child_operator_step(
    state: dict[str, Any],
    child: dict[str, Any],
    *,
    operator_gain: float,
    writeback_scale: float,
) -> dict[str, float]:
    phase_modes = state["phase_modes"]
    parent_mode0 = phase_modes[..., 0, :]
    parent_mode1 = phase_modes[..., 1, :]
    child_phase = phasor_normalize(child["phase_state"])
    gate = _child_gate(state, child)
    raw_delta = _phase_delta_from_to(parent_mode1, child_phase).clamp(-math.pi, math.pi)
    applied_delta = (float(operator_gain) * float(writeback_scale) * gate * raw_delta).clamp(-math.pi, math.pi)
    candidate_mode1 = phasor_apply_delta(parent_mode1, applied_delta)
    parent0_alignment_before = _masked_mean((parent_mode0 * parent_mode1).sum(dim=-1), gate)
    parent0_alignment_after = _masked_mean((parent_mode0 * candidate_mode1).sum(dim=-1), gate)
    step_parent_div = _masked_mean(_phase_delta_from_to(parent_mode1, candidate_mode1).abs(), gate)
    state["phase_modes"] = phase_modes.clone()
    state["phase_modes"][..., 1, :] = phasor_normalize(candidate_mode1)
    state["mixed_phase_state"] = state["phase_modes"][..., 1, :]
    return {
        "step_parent_phase_divergence": step_parent_div,
        "step_world_jump_proxy": float(max(0.0, parent0_alignment_before - parent0_alignment_after)),
        "step_gate_mean": float(gate.mean().detach().cpu().item()),
        "step_raw_delta_abs_mean": _masked_mean(raw_delta.abs(), gate),
        "step_applied_delta_abs_mean": _masked_mean(applied_delta.abs(), gate),
        "parent0_alignment_before": float(parent0_alignment_before),
        "parent0_alignment_after": float(parent0_alignment_after),
    }


def run_scaled_operator_recurrence(
    state: dict[str, Any],
    child: dict[str, Any],
    *,
    operator_gain: float,
    writeback_scale: float,
    steps: int,
) -> dict[str, float]:
    sandbox = clone_circleworld_state(state)
    child_clone = clone_circleworld_state(child)
    gate = _child_gate(sandbox, child_clone)
    initial_mode1 = sandbox["phase_modes"][..., 1, :].clone()
    parent_mode0 = sandbox["phase_modes"][..., 0, :].clone()
    initial_alignment = _masked_mean((parent_mode0 * initial_mode1).sum(dim=-1), gate)
    step_rows: list[dict[str, float]] = []
    for _ in range(max(0, int(steps))):
        step_rows.append(
            _apply_scaled_child_operator_step(
                sandbox,
                child_clone,
                operator_gain=float(operator_gain),
                writeback_scale=float(writeback_scale),
            )
        )
    final_mode1 = sandbox["phase_modes"][..., 1, :]
    final_alignment = _masked_mean((parent_mode0 * final_mode1).sum(dim=-1), gate)
    return {
        "steps": int(steps),
        "writeback_scale": float(writeback_scale),
        "final_parent_phase_divergence": _masked_mean(_phase_delta_from_to(initial_mode1, final_mode1).abs(), gate),
        "cumulative_step_parent_phase_divergence": _mean(
            [_as_float(row.get("step_parent_phase_divergence", 0.0)) for row in step_rows]
        )
        * len(step_rows),
        "final_world_jump_proxy": float(max(0.0, initial_alignment - final_alignment)),
        "cumulative_world_jump_proxy": sum(_as_float(row.get("step_world_jump_proxy", 0.0)) for row in step_rows),
        "mean_step_gate": _mean([_as_float(row.get("step_gate_mean", 0.0)) for row in step_rows]),
        "mean_step_raw_delta_abs": _mean([_as_float(row.get("step_raw_delta_abs_mean", 0.0)) for row in step_rows]),
        "mean_step_applied_delta_abs": _mean([_as_float(row.get("step_applied_delta_abs_mean", 0.0)) for row in step_rows]),
        "initial_parent0_alignment": float(initial_alignment),
        "final_parent0_alignment": float(final_alignment),
    }


def _parse_seeds(raw: str) -> list[int]:
    return [int(part.strip()) for part in str(raw).replace(";", ",").split(",") if part.strip()]


def _prepare_case(
    *,
    cfg: Any,
    rafa_core: Any,
    source: str,
    seed: int,
    time_steps: int,
    warmup_depth: int,
    device: torch.device,
    operator_gain: float,
    grandchild_depth: int,
    grandchild_parts: int,
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
    objects, by_id = _object_child_mapping(branch_state, case=f"{source}_{seed}", branch="multistep_operator")
    retrieval = run_retrieval_assay(objects, top_k=3)
    query_id, selected_id, decoy_id, _, _ = _select_operator_triplet(objects, retrieval)
    causal_row = _select_causal_candidate(
        objects,
        query_id,
        causal_selector,
        residual_weight=float(causal_residual_weight),
        residual_cap=float(causal_residual_cap),
        membrane_mode=str(causal_membrane_mode),
        strong_recurrence_threshold=0.96,
        strong_geometric_threshold=0.94,
        device=device,
    )
    causal_id = str(causal_row.get("candidate_object_id", "")) if causal_row else selected_id
    causal_child = by_id.get(causal_id) or by_id.get(selected_id) or by_id.get(query_id)
    decoy_child = by_id.get(decoy_id)
    if causal_child is None:
        raise RuntimeError(f"No causal child selected for seed {seed}")
    causal_metrics = _apply_child_operator(branch_state, causal_child, gain=operator_gain)
    decoy_metrics = _apply_child_operator(branch_state, decoy_child, gain=operator_gain) if decoy_child is not None else {}
    case_row = {
        "source": source,
        "seed": int(seed),
        "fork_depth": int(fork_idx),
        "law_object_count": len(objects),
        "query_object_id": query_id,
        "causal_candidate_object_id": causal_id,
        "decoy_candidate_object_id": decoy_id,
        "causal_selector_model_score": _as_float(causal_row.get("causal_selector_model_score", 0.0)) if causal_row else 0.0,
        "causal_selector_movement_pred": _as_float(causal_row.get("causal_selector_movement_pred", 0.0)) if causal_row else 0.0,
        "causal_selector_jump_safety_pred": _as_float(causal_row.get("causal_selector_jump_safety_pred", 0.0)) if causal_row else 0.0,
        "causal_selector_identity_pred": _as_float(causal_row.get("causal_selector_identity_pred", 0.0)) if causal_row else 0.0,
        "causal_selector_recurrence_compatibility": _as_float(causal_row.get("recurrence_compatibility", 0.0)) if causal_row else 0.0,
        "causal_selector_identity_membrane_pass": _as_float(causal_row.get("identity_membrane_pass", 0.0)) if causal_row else 0.0,
        "causal_selector_same_local_family": _as_float(causal_row.get("same_local_family", 0.0)) if causal_row else 0.0,
        "causal_selector_same_structural_family": _as_float(causal_row.get("same_structural_family", 0.0)) if causal_row else 0.0,
        "causal_parent_phase_divergence": _as_float(causal_metrics.get("parent_phase_divergence", 0.0)),
        "causal_support_shift": _as_float(causal_metrics.get("support_shift", 0.0)),
        "causal_qtrace_shift_proxy": _as_float(causal_metrics.get("qtrace_shift_proxy", 0.0)),
        "causal_world_jump_proxy": _as_float(causal_metrics.get("world_jump_proxy", 0.0)),
        "decoy_parent_phase_divergence": _as_float(decoy_metrics.get("parent_phase_divergence", 0.0)),
        "decoy_support_shift": _as_float(decoy_metrics.get("support_shift", 0.0)),
        "decoy_qtrace_shift_proxy": _as_float(decoy_metrics.get("qtrace_shift_proxy", 0.0)),
        "decoy_world_jump_proxy": _as_float(decoy_metrics.get("world_jump_proxy", 0.0)),
        "partition": partition,
    }
    return {
        "state": branch_state,
        "child": causal_child,
        "case_row": case_row,
        "retrieval": retrieval.get("summary", {}),
        "fork_depth": int(fork_idx),
        "grandchild_depth": int(grandchild_depth),
        "grandchild_parts": int(grandchild_parts),
    }


def _predict_gate_fields(model: torch.nn.Module, measured: dict[str, Any], device: torch.device) -> dict[str, float]:
    features, _ = shadow_row_to_feature_target(measured)
    feature_tensor = torch.tensor([features], dtype=torch.float32, device=device)
    with torch.no_grad():
        pred = learned_branch_law_output_tensor(model(feature_tensor)).squeeze(0).detach().cpu()
    prediction_by_name = {
        name: float(pred[idx].item())
        for idx, name in enumerate(learned_branch_law_output_names())
    }
    return shadow_prediction_to_branch_fields(prediction_by_name)


def run_multistep_operator_sandbox(
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
    recurrence_steps: int,
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
    previous = dict(ASSAY_KILLSWITCH)
    rows: list[dict[str, Any]] = []
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
            measured = shadow_branch_law_row(prepared["case_row"], source_path=str(config_path))
            predicted = _predict_gate_fields(learned_model, measured, device)
            score = _as_float(predicted.get("predicted_writeback_score", 0.0))
            hard_scale = 1.0 if score >= float(writeback_threshold) else 0.0
            soft_scale = _soft_writeback_scale(score, soft_floor=float(soft_floor), soft_full=float(soft_full))
            variant_scales = {
                "ungated": 1.0,
                "hard_gate": hard_scale,
                "soft_gate": soft_scale,
            }
            variant_rows = {
                name: run_scaled_operator_recurrence(
                    prepared["state"],
                    prepared["child"],
                    operator_gain=float(operator_gain),
                    writeback_scale=float(scale),
                    steps=int(recurrence_steps),
                )
                for name, scale in variant_scales.items()
            }
            base = {
                "source": seed_source,
                "seed": int(seed),
                "config_path": str(config_path),
                "fork_depth": int(prepared["fork_depth"]),
                "grandchild_depth": int(grandchild_depth),
                "law_object_count": int(prepared["case_row"].get("law_object_count", 0)),
                "target_writeback_permission": _as_float(measured.get("writeback_permission", 0.0)),
                "writeback_score": score,
                "hard_writeback_scale": hard_scale,
                "soft_writeback_scale": soft_scale,
                "predicted_survival_delta": _as_float(predicted.get("predicted_survival_delta", 0.0)),
                "predicted_collapse_pressure": _as_float(predicted.get("predicted_collapse_pressure", 0.0)),
                "measured_survival_delta": _as_float(measured.get("survival_delta", 0.0)),
                "measured_collapse_pressure": _as_float(measured.get("collapse_pressure", 0.0)),
                "causal_parent_phase_divergence": _as_float(prepared["case_row"].get("causal_parent_phase_divergence", 0.0)),
                "causal_world_jump_proxy": _as_float(prepared["case_row"].get("causal_world_jump_proxy", 0.0)),
            }
            for variant, metrics in variant_rows.items():
                row = dict(base)
                row.update({"variant": variant, **metrics})
                rows.append(row)
    finally:
        ASSAY_KILLSWITCH.clear()
        ASSAY_KILLSWITCH.update(previous)

    aggregate = _aggregate_rows(rows)
    payload = {
        "schema": "learned_gated_multistep_operator_sandbox_v0",
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
        "recurrence_steps": int(recurrence_steps),
        "writeback_threshold": float(writeback_threshold),
        "soft_floor": float(soft_floor),
        "soft_full": float(soft_full),
        "aggregate": aggregate,
        "rows": rows,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "learned_gated_multistep_operator_sandbox.json"
    md_path = out_dir / "LEARNED_GATED_MULTISTEP_OPERATOR_SANDBOX.md"
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
            "mean_writeback_scale": _mean([_as_float(row.get("writeback_scale", 0.0)) for row in subset]),
            "mean_final_parent_phase_divergence": _mean(
                [_as_float(row.get("final_parent_phase_divergence", 0.0)) for row in subset]
            ),
            "mean_cumulative_step_parent_phase_divergence": _mean(
                [_as_float(row.get("cumulative_step_parent_phase_divergence", 0.0)) for row in subset]
            ),
            "mean_final_world_jump_proxy": _mean([_as_float(row.get("final_world_jump_proxy", 0.0)) for row in subset]),
            "mean_cumulative_world_jump_proxy": _mean(
                [_as_float(row.get("cumulative_world_jump_proxy", 0.0)) for row in subset]
            ),
            "mean_step_applied_delta_abs": _mean([_as_float(row.get("mean_step_applied_delta_abs", 0.0)) for row in subset]),
        }
    ungated = by_variant.get("ungated", {})
    for variant, summary in by_variant.items():
        if variant == "ungated":
            summary["parent_retention_vs_ungated"] = 1.0
            summary["world_jump_retention_vs_ungated"] = 1.0
            continue
        parent = summary.get("mean_final_parent_phase_divergence", 0.0)
        jump = summary.get("mean_final_world_jump_proxy", 0.0)
        u_parent = ungated.get("mean_final_parent_phase_divergence", 0.0)
        u_jump = ungated.get("mean_final_world_jump_proxy", 0.0)
        summary["parent_retention_vs_ungated"] = parent / u_parent if u_parent > 1.0e-8 else 0.0
        summary["world_jump_retention_vs_ungated"] = jump / u_jump if u_jump > 1.0e-8 else 0.0
        summary["world_jump_reduction_vs_ungated"] = u_jump - jump
        summary["parent_divergence_reduction_vs_ungated"] = u_parent - parent
        summary["influence_per_jump_retained"] = summary["parent_retention_vs_ungated"] / max(
            summary["world_jump_retention_vs_ungated"], 1.0e-8
        )
    return {
        "status": "pass_learned_gated_multistep_operator_sandbox" if rows else "fail_no_rows",
        "row_count": len(rows),
        "variant_count": len(variants),
        "variants": by_variant,
    }


def _write_markdown(path: Path, payload: dict[str, Any]) -> None:
    agg = payload["aggregate"]
    lines = [
        "# Learned-Gated Multi-Step Operator Sandbox",
        "",
        "This assay repeatedly applies the selected child operator under ungated, hard-gated, and soft-gated writeback scales.",
        "",
        "## Summary",
        "",
        f"- status: `{agg['status']}`",
        f"- rows: `{agg['row_count']}`",
        f"- recurrence steps: `{payload['recurrence_steps']}`",
        f"- grandchild depth: `{payload['grandchild_depth']}`",
        f"- soft floor/full: `{payload['soft_floor']}` / `{payload['soft_full']}`",
        "",
        "## Variants",
        "",
        "| variant | rows | scale | parent div | world jump | parent retained | jump retained | influence/jump |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for variant, summary in agg.get("variants", {}).items():
        lines.append(
            "| {variant} | {rows} | {scale} | {parent} | {jump} | {pretain} | {jretain} | {ratio} |".format(
                variant=variant,
                rows=summary.get("row_count", 0.0),
                scale=summary.get("mean_writeback_scale", 0.0),
                parent=summary.get("mean_final_parent_phase_divergence", 0.0),
                jump=summary.get("mean_final_world_jump_proxy", 0.0),
                pretain=summary.get("parent_retention_vs_ungated", 0.0),
                jretain=summary.get("world_jump_retention_vs_ungated", 0.0),
                ratio=summary.get("influence_per_jump_retained", 1.0 if variant == "ungated" else 0.0),
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description="Run learned hard/soft gates in a multi-step operator recurrence sandbox.")
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
    ap.add_argument("--recurrence-steps", type=int, default=4)
    ap.add_argument("--writeback-threshold", type=float, default=0.78)
    ap.add_argument("--soft-floor", type=float, default=0.70)
    ap.add_argument("--soft-full", type=float, default=0.88)
    ap.add_argument("--causal-residual-weight", type=float, default=0.20)
    ap.add_argument("--causal-residual-cap", type=float, default=0.05)
    ap.add_argument("--causal-membrane-mode", choices=["identity", "off"], default="off")
    args = ap.parse_args()
    result = run_multistep_operator_sandbox(
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
        recurrence_steps=int(args.recurrence_steps),
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
