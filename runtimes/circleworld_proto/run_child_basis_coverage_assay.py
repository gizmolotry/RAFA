from __future__ import annotations

import argparse
import json
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

from ablate_formalization import make_seed_rafa  # noqa: E402
from circleworld import clone_circleworld_state  # noqa: E402
from evaluate_circleworld import _load_circle_cfg, _safe_device  # noqa: E402
from evaluate_shadow_branch_law_calibration import _load_model  # noqa: E402
from rafa_math_tools import phasor_apply_delta, phasor_normalize  # noqa: E402
from run_learned_gated_manychild_runtime_sandbox import (  # noqa: E402
    _child_union_gate,
    _collapse_direction_vector,
    _controlled_sibling_target_vector,
    _return_direction_vector,
    run_manychild_recurrence,
)
from run_learned_gated_multistep_operator_sandbox import (  # noqa: E402
    _as_float,
    _mean,
    _parse_seeds,
    _prepare_case,
)
from run_resonant_operator_causality_assay import _load_causal_selector  # noqa: E402
from test_nested_commitment import ASSAY_KILLSWITCH, DEFAULT_ASSAY_KILLSWITCH  # noqa: E402


def _carrier_target_vector(state: dict[str, Any], *, target_index: int, target_radians: float) -> torch.Tensor:
    phase_modes = state["phase_modes"]
    parent_mode1 = phase_modes[..., 1, :]
    support = _child_union_gate(state).detach().float().clamp(0.0, 1.0)
    batch, freq_count, time_count = support.shape
    freq_axis = torch.linspace(-1.0, 1.0, freq_count, device=support.device, dtype=support.dtype).view(1, freq_count, 1)
    time_axis = torch.linspace(-1.0, 1.0, time_count, device=support.device, dtype=support.dtype).view(1, 1, time_count)
    freq_axis = freq_axis.expand(batch, freq_count, time_count)
    time_axis = time_axis.expand(batch, freq_count, time_count)
    idx = float(target_index + 1)
    carrier = (
        torch.sin(2.0 * torch.pi * ((0.17 + 0.037 * idx) * time_axis + (0.11 + 0.019 * idx) * freq_axis))
        + 0.45 * torch.cos(2.0 * torch.pi * ((0.07 + 0.013 * idx) * time_axis - (0.19 + 0.017 * idx) * freq_axis))
        + 0.25 * torch.sin(2.0 * torch.pi * ((0.05 + 0.011 * idx) * freq_axis * time_axis))
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


def _target_vectors(state: dict[str, Any], *, target_count: int, target_radians: float) -> list[torch.Tensor]:
    count = max(1, int(target_count))
    targets = [_controlled_sibling_target_vector(state, target_radians=float(target_radians))]
    targets.extend(
        _carrier_target_vector(state, target_index=idx, target_radians=float(target_radians))
        for idx in range(max(0, count - 1))
    )
    return targets


def _basis_rows(state: dict[str, Any], targets: list[torch.Tensor]) -> tuple[list[dict[str, float]], dict[str, float]]:
    children = [child for child in (state.get("child_worlds", []) or []) if isinstance(child, dict) and child.get("active", False)]
    collapse = _collapse_direction_vector(state)
    collapse_norm = torch.linalg.vector_norm(collapse).clamp_min(1.0e-8)
    collapse_unit = collapse / collapse_norm
    directions: list[torch.Tensor] = []
    tangent_directions: list[torch.Tensor] = []
    collapse_abs: list[float] = []
    child_norms: list[float] = []
    for child in children:
        direction = _return_direction_vector(state, child)
        component = torch.dot(direction, collapse_unit)
        tangent = direction - component * collapse_unit
        directions.append(direction)
        tangent_directions.append(tangent)
        collapse_abs.append(abs(float(component.detach().cpu().item())) / float(collapse_norm.detach().cpu().item()))
        child_norms.append(float(torch.linalg.vector_norm(direction).detach().cpu().item()))

    target_rows: list[dict[str, float]] = []
    all_projections: list[float] = []
    positive_02: list[float] = []
    positive_05: list[float] = []
    positive_10: list[float] = []
    max_projections: list[float] = []
    top3_projections: list[float] = []
    for target_index, target in enumerate(targets):
        target_norm = torch.linalg.vector_norm(target).clamp_min(1.0e-8)
        target_unit = target / target_norm
        projections: list[float] = []
        for tangent in tangent_directions:
            tangent_norm = torch.linalg.vector_norm(tangent).clamp_min(1.0e-8)
            projection = float((torch.dot(tangent, target_unit) / tangent_norm).detach().cpu().item())
            projections.append(max(-1.0, min(1.0, projection)))
        sorted_pos = sorted([max(0.0, value) for value in projections], reverse=True)
        max_projection = sorted_pos[0] if sorted_pos else 0.0
        top3_projection = float(np.mean(sorted_pos[:3])) if sorted_pos else 0.0
        row = {
            "target_index": float(target_index),
            "target_norm": float(target_norm.detach().cpu().item()),
            "mean_projection": _mean(projections),
            "max_positive_projection": max_projection,
            "top3_positive_projection": top3_projection,
            "positive_fraction_002": _mean([1.0 if value >= 0.02 else 0.0 for value in projections]),
            "positive_fraction_005": _mean([1.0 if value >= 0.05 else 0.0 for value in projections]),
            "positive_fraction_010": _mean([1.0 if value >= 0.10 else 0.0 for value in projections]),
        }
        target_rows.append(row)
        all_projections.extend(projections)
        max_projections.append(max_projection)
        top3_projections.append(top3_projection)
        positive_02.append(row["positive_fraction_002"])
        positive_05.append(row["positive_fraction_005"])
        positive_10.append(row["positive_fraction_010"])

    rank = 0.0
    effective_rank = 0.0
    pairwise_alignment = 0.0
    if tangent_directions:
        matrix = torch.stack(tangent_directions, dim=0).float()
        norms = torch.linalg.vector_norm(matrix, dim=1).clamp_min(1.0e-8)
        normalized = matrix / norms[:, None]
        if normalized.shape[0] >= 2:
            cos = (normalized @ normalized.T).clamp(-1.0, 1.0)
            mask = ~torch.eye(normalized.shape[0], dtype=torch.bool, device=normalized.device)
            pairwise_alignment = float(cos[mask].mean().detach().cpu().item())
        singular = torch.linalg.svdvals(normalized)
        singular = singular[singular > 1.0e-6]
        rank = float(singular.numel())
        if singular.numel() > 0:
            effective_rank = float(((singular.sum() ** 2) / (singular.square().sum().clamp_min(1.0e-8))).detach().cpu().item())

    summary = {
        "child_count": float(len(children)),
        "target_count": float(len(targets)),
        "mean_child_direction_norm": _mean(child_norms),
        "mean_collapse_component_abs": _mean(collapse_abs),
        "basis_rank": rank,
        "basis_effective_rank": effective_rank,
        "child_pairwise_alignment_mean": pairwise_alignment,
        "mean_projection": _mean(all_projections),
        "mean_max_positive_projection": _mean(max_projections),
        "mean_top3_positive_projection": _mean(top3_projections),
        "mean_positive_fraction_002": _mean(positive_02),
        "mean_positive_fraction_005": _mean(positive_05),
        "mean_positive_fraction_010": _mean(positive_10),
        "covered_target_fraction_005": _mean([1.0 if value >= 0.05 else 0.0 for value in max_projections]),
        "covered_target_fraction_010": _mean([1.0 if value >= 0.10 else 0.0 for value in max_projections]),
    }
    return target_rows, summary


def _augment_target_children(
    state: dict[str, Any],
    targets: list[torch.Tensor],
    *,
    max_augments: int,
) -> dict[str, Any]:
    out = clone_circleworld_state(state)
    children = [child for child in (out.get("child_worlds", []) or []) if isinstance(child, dict)]
    active = [child for child in children if child.get("active", False)]
    template = active[0] if active else (children[0] if children else {})
    parent_mode1 = out["phase_modes"][..., 1, :]
    support = _child_union_gate(out).detach().float().clamp(0.0, 1.0)
    coherence = torch.where(support > 0.0, torch.ones_like(support), torch.zeros_like(support))
    max_id = max([int(child.get("child_id", -1)) for child in children] or [-1])
    for idx, target in enumerate(targets[: max(0, int(max_augments))]):
        delta = target.reshape(parent_mode1.shape[:-1]).to(device=parent_mode1.device, dtype=parent_mode1.dtype)
        phase = phasor_normalize(phasor_apply_delta(parent_mode1, delta))
        child = clone_circleworld_state(template) if isinstance(template, dict) else {}
        child_id = max_id + idx + 1
        child["child_id"] = int(child_id)
        child["origin_child_id"] = int(child_id)
        child["parent_child_id"] = int(template.get("child_id", -1)) if isinstance(template, dict) else -1
        child["generation"] = int(template.get("generation", 1)) + 1 if isinstance(template, dict) else 1
        child["fragment_index"] = 9000 + idx
        child["active"] = True
        child["collapsed"] = False
        child["phase_state"] = phase
        child["mode_support"] = support.clone()
        child["mode_coherence"] = coherence.clone()
        child["survival_age"] = max(2.0, float(child.get("survival_age", 0.0)))
        child["writeback_budget"] = max(1.0, float(child.get("writeback_budget", 0.0)))
        child["law_signature"] = target.detach().float()[: min(16, target.numel())].clone()
        child["assay_writeback_scale"] = 1.0
        out.setdefault("child_worlds", []).append(child)
    return out


def _aggregate(rows: list[dict[str, Any]], recurrence_rows: list[dict[str, Any]]) -> dict[str, Any]:
    phases = sorted({str(row.get("phase", "")) for row in rows})
    by_phase: dict[str, dict[str, float]] = {}
    for phase in phases:
        subset = [row for row in rows if str(row.get("phase", "")) == phase]
        by_phase[phase] = {
            "row_count": float(len(subset)),
            "mean_child_count": _mean([_as_float(row.get("child_count", 0.0)) for row in subset]),
            "mean_basis_rank": _mean([_as_float(row.get("basis_rank", 0.0)) for row in subset]),
            "mean_basis_effective_rank": _mean([_as_float(row.get("basis_effective_rank", 0.0)) for row in subset]),
            "mean_pairwise_alignment": _mean([_as_float(row.get("child_pairwise_alignment_mean", 0.0)) for row in subset]),
            "mean_max_positive_projection": _mean([_as_float(row.get("mean_max_positive_projection", 0.0)) for row in subset]),
            "mean_top3_positive_projection": _mean([_as_float(row.get("mean_top3_positive_projection", 0.0)) for row in subset]),
            "covered_target_fraction_005": _mean([_as_float(row.get("covered_target_fraction_005", 0.0)) for row in subset]),
            "covered_target_fraction_010": _mean([_as_float(row.get("covered_target_fraction_010", 0.0)) for row in subset]),
            "mean_collapse_component_abs": _mean([_as_float(row.get("mean_collapse_component_abs", 0.0)) for row in subset]),
        }
    recur_by_phase: dict[str, dict[str, float]] = {}
    for phase in sorted({str(row.get("phase", "")) for row in recurrence_rows}):
        subset = [row for row in recurrence_rows if str(row.get("phase", "")) == phase]
        recur_by_phase[phase] = {
            "row_count": float(len(subset)),
            "mean_assignment_scale": _mean([_as_float(row.get("mean_assignment_scale", 0.0)) for row in subset]),
            "mean_contrastive_target_projection": _mean(
                [_as_float(row.get("mean_contrastive_target_projection", 0.0)) for row in subset]
            ),
            "mean_child_writeback_mass": _mean([_as_float(row.get("mean_child_writeback_mass", 0.0)) for row in subset]),
            "mean_final_parent_phase_divergence": _mean(
                [_as_float(row.get("final_parent_phase_divergence", 0.0)) for row in subset]
            ),
            "mean_final_world_jump_proxy": _mean([_as_float(row.get("final_world_jump_proxy", 0.0)) for row in subset]),
            "mean_parent_lift_vs_blocked": _mean([_as_float(row.get("parent_divergence_lift_vs_blocked", 0.0)) for row in subset]),
            "mean_jump_lift_vs_blocked": _mean([_as_float(row.get("world_jump_lift_vs_blocked", 0.0)) for row in subset]),
        }
    recur_by_phase_variant: dict[str, dict[str, float]] = {}
    phase_variants = sorted({(str(row.get("phase", "")), str(row.get("variant", ""))) for row in recurrence_rows})
    for phase, variant in phase_variants:
        key = f"{phase}:{variant}"
        subset = [
            row
            for row in recurrence_rows
            if str(row.get("phase", "")) == phase and str(row.get("variant", "")) == variant
        ]
        recur_by_phase_variant[key] = {
            "phase": phase,
            "variant": variant,
            "row_count": float(len(subset)),
            "mean_assignment_scale": _mean([_as_float(row.get("mean_assignment_scale", 0.0)) for row in subset]),
            "mean_contrastive_target_projection": _mean(
                [_as_float(row.get("mean_contrastive_target_projection", 0.0)) for row in subset]
            ),
            "mean_child_writeback_mass": _mean([_as_float(row.get("mean_child_writeback_mass", 0.0)) for row in subset]),
            "mean_final_parent_phase_divergence": _mean(
                [_as_float(row.get("final_parent_phase_divergence", 0.0)) for row in subset]
            ),
            "mean_final_world_jump_proxy": _mean([_as_float(row.get("final_world_jump_proxy", 0.0)) for row in subset]),
            "mean_parent_lift_vs_blocked": _mean([_as_float(row.get("parent_divergence_lift_vs_blocked", 0.0)) for row in subset]),
            "mean_jump_lift_vs_blocked": _mean([_as_float(row.get("world_jump_lift_vs_blocked", 0.0)) for row in subset]),
        }
    for key, summary in recur_by_phase_variant.items():
        phase = str(summary.get("phase", ""))
        blocked = recur_by_phase_variant.get(f"{phase}:blocked_zero")
        if blocked:
            summary["mean_parent_lift_vs_blocked"] = (
                _as_float(summary.get("mean_final_parent_phase_divergence", 0.0))
                - _as_float(blocked.get("mean_final_parent_phase_divergence", 0.0))
            )
            summary["mean_jump_lift_vs_blocked"] = (
                _as_float(summary.get("mean_final_world_jump_proxy", 0.0))
                - _as_float(blocked.get("mean_final_world_jump_proxy", 0.0))
            )
    return {
        "status": "pass_child_basis_coverage_assay" if rows else "fail_no_rows",
        "row_count": len(rows),
        "recurrence_row_count": len(recurrence_rows),
        "coverage_by_phase": by_phase,
        "recurrence_by_phase": recur_by_phase,
        "recurrence_by_phase_variant": recur_by_phase_variant,
    }


def _write_markdown(path: Path, payload: dict[str, Any]) -> None:
    agg = payload["aggregate"]
    lines = [
        "# Child Basis Coverage Assay",
        "",
        "This assay measures whether live child worlds span controlled tangent sibling target directions.",
        "",
        "## Summary",
        "",
        f"- status: `{agg['status']}`",
        f"- rows: `{agg['row_count']}`",
        f"- recurrence rows: `{agg['recurrence_row_count']}`",
        f"- target count: `{payload['target_count']}`",
        f"- augmentation count: `{payload['augmentation_count']}`",
        "",
        "## Coverage",
        "",
        "| phase | rows | children | rank | eff rank | pair align | max proj | top3 proj | cov@.05 | cov@.10 | collapse |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for phase, summary in agg.get("coverage_by_phase", {}).items():
        lines.append(
            "| {phase} | {rows} | {children} | {rank} | {erank} | {align} | {maxp} | {top3} | {cov05} | {cov10} | {collapse} |".format(
                phase=phase,
                rows=summary.get("row_count", 0.0),
                children=summary.get("mean_child_count", 0.0),
                rank=summary.get("mean_basis_rank", 0.0),
                erank=summary.get("mean_basis_effective_rank", 0.0),
                align=summary.get("mean_pairwise_alignment", 0.0),
                maxp=summary.get("mean_max_positive_projection", 0.0),
                top3=summary.get("mean_top3_positive_projection", 0.0),
                cov05=summary.get("covered_target_fraction_005", 0.0),
                cov10=summary.get("covered_target_fraction_010", 0.0),
                collapse=summary.get("mean_collapse_component_abs", 0.0),
            )
        )
    lines.extend(
        [
            "",
            "## Contrastive Runtime Check",
            "",
            "| phase | rows | scale | ctarget | writeback | parent div | world jump | parent lift | jump lift |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    recurrence_table = agg.get("recurrence_by_phase_variant") or agg.get("recurrence_by_phase", {})
    for phase, summary in recurrence_table.items():
        lines.append(
            "| {phase} | {rows} | {scale} | {ctarget} | {writeback} | {parent} | {jump} | {plift} | {jlift} |".format(
                phase=phase,
                rows=summary.get("row_count", 0.0),
                scale=summary.get("mean_assignment_scale", 0.0),
                ctarget=summary.get("mean_contrastive_target_projection", 0.0),
                writeback=summary.get("mean_child_writeback_mass", 0.0),
                parent=summary.get("mean_final_parent_phase_divergence", 0.0),
                jump=summary.get("mean_final_world_jump_proxy", 0.0),
                plift=summary.get("mean_parent_lift_vs_blocked", 0.0),
                jlift=summary.get("mean_jump_lift_vs_blocked", 0.0),
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_child_basis_coverage_assay(
    *,
    config_path: Path,
    out_dir: Path,
    learned_branch_checkpoint: Path,
    causal_selector_checkpoint: Path,
    device_name: str,
    seed_source: str,
    seeds: list[int],
    time_steps: int,
    warmup_depth: int,
    operator_gain: float,
    grandchild_depth: int,
    grandchild_parts: int,
    target_count: int,
    target_radians: float,
    augmentation_count: int,
    runtime_steps: int,
    writeback_threshold: float,
    soft_floor: float,
    soft_full: float,
) -> dict[str, str]:
    cfg = _load_circle_cfg(config_path)
    device = _safe_device(device_name)
    rafa_core = make_seed_rafa(dev=device.type) if seed_source == "naked_rafa" else None
    learned_model = _load_model(learned_branch_checkpoint, device)
    causal_selector = _load_causal_selector(causal_selector_checkpoint, device)
    rows: list[dict[str, Any]] = []
    target_rows: list[dict[str, Any]] = []
    recurrence_rows: list[dict[str, Any]] = []
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
                causal_residual_weight=0.0,
                causal_residual_cap=0.0,
                causal_membrane_mode="off",
            )
            base_state = prepared["state"]
            targets = _target_vectors(base_state, target_count=int(target_count), target_radians=float(target_radians))
            augmented_state = _augment_target_children(base_state, targets, max_augments=int(augmentation_count))
            for phase, state in (("natural", base_state), ("target_augmented", augmented_state)):
                phase_targets = targets if phase == "natural" else _target_vectors(
                    state, target_count=int(target_count), target_radians=float(target_radians)
                )
                per_target, summary = _basis_rows(state, phase_targets)
                summary.update(
                    {
                        "source": seed_source,
                        "seed": int(seed),
                        "phase": phase,
                        "grandchild_depth": int(grandchild_depth),
                        "grandchild_parts": int(grandchild_parts),
                        "fork_depth": int(prepared["fork_depth"]),
                    }
                )
                rows.append(summary)
                for row in per_target:
                    out = dict(row)
                    out.update({"source": seed_source, "seed": int(seed), "phase": phase})
                    target_rows.append(out)
                for variant in (
                    "blocked_zero",
                    "learned_soft_contrastive_sibling_target",
                    "learned_soft_target_authority_scout",
                ):
                    metrics = run_manychild_recurrence(
                        state,
                        cfg,
                        learned_model=learned_model,
                        causal_selector=causal_selector,
                        device=device,
                        variant=variant,
                        runtime_steps=int(runtime_steps),
                        writeback_threshold=float(writeback_threshold),
                        soft_floor=float(soft_floor),
                        soft_full=float(soft_full),
                        case_label=f"{seed_source}_{seed}_{phase}_{variant}",
                        operator_gain=float(operator_gain),
                    )
                    metrics.update({"source": seed_source, "seed": int(seed), "phase": phase})
                    recurrence_rows.append(metrics)
    finally:
        ASSAY_KILLSWITCH.clear()
        ASSAY_KILLSWITCH.update(previous)

    payload = {
        "schema": "child_basis_coverage_assay_v0",
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
        "target_count": int(target_count),
        "target_radians": float(target_radians),
        "augmentation_count": int(augmentation_count),
        "runtime_steps": int(runtime_steps),
        "writeback_threshold": float(writeback_threshold),
        "soft_floor": float(soft_floor),
        "soft_full": float(soft_full),
        "aggregate": _aggregate(rows, recurrence_rows),
        "rows": rows,
        "target_rows": target_rows,
        "recurrence_rows": recurrence_rows,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "child_basis_coverage_assay.json"
    md_path = out_dir / "CHILD_BASIS_COVERAGE_ASSAY.md"
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    _write_markdown(md_path, payload)
    return {"json": str(json_path), "markdown": str(md_path), "status": payload["aggregate"]["status"]}


def main() -> None:
    ap = argparse.ArgumentParser(description="Measure child-world basis coverage against controlled sibling targets.")
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
    ap.add_argument("--grandchild-depth", type=int, default=3)
    ap.add_argument("--grandchild-parts", type=int, default=2)
    ap.add_argument("--target-count", type=int, default=8)
    ap.add_argument("--target-radians", type=float, default=0.085)
    ap.add_argument("--augmentation-count", type=int, default=8)
    ap.add_argument("--runtime-steps", type=int, default=4)
    ap.add_argument("--writeback-threshold", type=float, default=0.78)
    ap.add_argument("--soft-floor", type=float, default=0.70)
    ap.add_argument("--soft-full", type=float, default=0.88)
    args = ap.parse_args()
    result = run_child_basis_coverage_assay(
        config_path=Path(args.config),
        out_dir=Path(args.out_dir),
        learned_branch_checkpoint=Path(args.learned_branch_checkpoint),
        causal_selector_checkpoint=Path(args.causal_selector_checkpoint),
        device_name=str(args.device),
        seed_source=str(args.seed_source),
        seeds=_parse_seeds(args.seeds),
        time_steps=int(args.time_steps),
        warmup_depth=int(args.warmup_depth),
        operator_gain=float(args.operator_gain),
        grandchild_depth=int(args.grandchild_depth),
        grandchild_parts=int(args.grandchild_parts),
        target_count=int(args.target_count),
        target_radians=float(args.target_radians),
        augmentation_count=int(args.augmentation_count),
        runtime_steps=int(args.runtime_steps),
        writeback_threshold=float(args.writeback_threshold),
        soft_floor=float(args.soft_floor),
        soft_full=float(args.soft_full),
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
