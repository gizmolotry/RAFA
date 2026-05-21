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
from circleworld import circleworld_step, clone_circleworld_state  # noqa: E402
from evaluate_circleworld import _load_circle_cfg, _safe_device  # noqa: E402
from evaluate_shadow_branch_law_calibration import _load_model  # noqa: E402
from run_learned_gated_multistep_operator_sandbox import (  # noqa: E402
    _as_float,
    _masked_mean,
    _mean,
    _parse_seeds,
    _predict_gate_fields,
    _prepare_case,
)
from run_learned_gated_writeback_sandbox import _soft_writeback_scale  # noqa: E402
from run_resonant_operator_causality_assay import _load_causal_selector, _phase_delta_from_to  # noqa: E402
from test_nested_commitment import ASSAY_KILLSWITCH, DEFAULT_ASSAY_KILLSWITCH  # noqa: E402


def _child_gate(state: dict[str, Any], child: dict[str, Any]) -> torch.Tensor:
    parent_mode1 = state["phase_modes"][..., 1, :]
    support = child.get("mode_support")
    if not torch.is_tensor(support) or support.shape != parent_mode1.shape[:-1]:
        support = torch.ones(parent_mode1.shape[:-1], device=parent_mode1.device, dtype=parent_mode1.dtype)
    coherence = child.get("mode_coherence")
    if not torch.is_tensor(coherence) or coherence.shape != support.shape:
        coherence = torch.ones_like(support)
    return (support.clamp(0.0, 1.0) * (0.20 + 0.80 * coherence.clamp(0.0, 1.0))).clamp(0.0, 1.0)


def _block_float(block: dict[str, Any], key: str) -> float:
    return _as_float(block.get(key, 0.0))


def _isolate_selected_child(
    state: dict[str, Any],
    child: dict[str, Any],
    cfg: Any,
    *,
    scale: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    sandbox = clone_circleworld_state(state)
    selected_id = int(child.get("child_id", -1))
    selected_child: dict[str, Any] | None = None
    for candidate in sandbox.get("child_worlds", []):
        if int(candidate.get("child_id", -2)) == selected_id:
            selected_child = candidate
            break
    if selected_child is None:
        selected_child = clone_circleworld_state(child)
    selected_child["active"] = True
    selected_child["collapsed"] = False
    selected_child["survival_age"] = max(float(selected_child.get("survival_age", 0.0)), float(cfg.child_min_age_for_writeback))
    selected_child["writeback_budget"] = max(float(selected_child.get("writeback_budget", 0.0)), float(cfg.child_writeback_budget))
    selected_child["assay_writeback_scale"] = float(scale)
    sandbox["child_worlds"] = [selected_child]
    sandbox["child_event_history"] = []
    sandbox["next_child_id"] = max(int(sandbox.get("next_child_id", selected_id + 1)), selected_id + 1)
    return sandbox, selected_child


def run_runtime_recurrence(
    state: dict[str, Any],
    child: dict[str, Any],
    cfg: Any,
    *,
    writeback_scale: float,
    runtime_steps: int,
) -> dict[str, Any]:
    runtime_cfg = replace(
        cfg,
        branching_mode="native_multimode_childworld",
        child_max_worlds=1,
        child_assay_writeback_scale_enabled=True,
        child_assay_writeback_scale_default=float(writeback_scale),
    )
    sandbox, selected_child = _isolate_selected_child(state, child, cfg, scale=float(writeback_scale))
    gate = _child_gate(sandbox, selected_child)
    initial_mode0 = sandbox["phase_modes"][..., 0, :].clone()
    initial_mode1 = sandbox["phase_modes"][..., 1, :].clone()
    initial_alignment = _masked_mean((initial_mode0 * initial_mode1).sum(dim=-1), gate)
    step_rows: list[dict[str, float]] = []
    for step_idx in range(max(0, int(runtime_steps))):
        for live_child in sandbox.get("child_worlds", []):
            live_child["assay_writeback_scale"] = float(writeback_scale)
        sandbox, block, _ = circleworld_step(sandbox, runtime_cfg, mode="native_multimode_childworld")
        step_rows.append(
            {
                "step": float(step_idx),
                "child_world_count": _block_float(block, "child_world_count"),
                "live_child_fraction": _block_float(block, "live_child_fraction"),
                "child_writeback_mass": _block_float(block, "child_writeback_mass"),
                "child_phase_writeback_delta_mass": _block_float(block, "child_phase_writeback_delta_mass"),
                "child_support_writeback_mass": _block_float(block, "child_support_writeback_mass"),
                "child_assay_writeback_scale_mean": _block_float(block, "child_assay_writeback_scale_mean"),
                "child_parent_divergence": _block_float(block, "child_parent_divergence"),
                "child_sibling_divergence": _block_float(block, "child_sibling_divergence"),
                "child_local_ifs_step_count": _block_float(block, "child_local_ifs_step_count"),
            }
        )
    final_mode1 = sandbox["phase_modes"][..., 1, :]
    final_alignment = _masked_mean((initial_mode0 * final_mode1).sum(dim=-1), gate)
    event_history = list(sandbox.get("child_event_history", []))
    writeback_events = [event for event in event_history if event.get("event") == "writeback"]
    return {
        "runtime_steps": int(runtime_steps),
        "writeback_scale": float(writeback_scale),
        "final_parent_phase_divergence": _masked_mean(_phase_delta_from_to(initial_mode1, final_mode1).abs(), gate),
        "final_world_jump_proxy": float(max(0.0, initial_alignment - final_alignment)),
        "mean_child_writeback_mass": _mean([row["child_writeback_mass"] for row in step_rows]),
        "sum_child_writeback_mass": float(np.sum([row["child_writeback_mass"] for row in step_rows])) if step_rows else 0.0,
        "mean_child_phase_writeback_delta_mass": _mean([row["child_phase_writeback_delta_mass"] for row in step_rows]),
        "mean_child_support_writeback_mass": _mean([row["child_support_writeback_mass"] for row in step_rows]),
        "mean_child_assay_writeback_scale": _mean([row["child_assay_writeback_scale_mean"] for row in step_rows]),
        "mean_child_parent_divergence_metric": _mean([row["child_parent_divergence"] for row in step_rows]),
        "mean_child_sibling_divergence_metric": _mean([row["child_sibling_divergence"] for row in step_rows]),
        "mean_live_child_fraction": _mean([row["live_child_fraction"] for row in step_rows]),
        "final_child_world_count": float(len(sandbox.get("child_worlds", []))),
        "writeback_event_count": float(len(writeback_events)),
        "collapse_event_count": float(len([event for event in event_history if event.get("event") == "collapse"])),
        "initial_parent0_alignment": float(initial_alignment),
        "final_parent0_alignment": float(final_alignment),
        "step_rows": step_rows,
    }


def run_learned_gated_runtime_sandbox(
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
            measured = shadow_branch_law_row(prepared["case_row"], source_path=str(config_path))
            predicted = _predict_gate_fields(learned_model, measured, device)
            score = _as_float(predicted.get("predicted_writeback_score", 0.0))
            hard_scale = 1.0 if score >= float(writeback_threshold) else 0.0
            soft_scale = _soft_writeback_scale(score, soft_floor=float(soft_floor), soft_full=float(soft_full))
            variants = {
                "blocked_zero": 0.0,
                "ungated": 1.0,
                "hard_gate": hard_scale,
                "soft_gate": soft_scale,
            }
            base = {
                "source": seed_source,
                "seed": int(seed),
                "config_path": str(config_path),
                "fork_depth": int(prepared["fork_depth"]),
                "grandchild_depth": int(grandchild_depth),
                "grandchild_parts": int(grandchild_parts),
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
            for variant, scale in variants.items():
                metrics = run_runtime_recurrence(
                    prepared["state"],
                    prepared["child"],
                    cfg,
                    writeback_scale=float(scale),
                    runtime_steps=int(runtime_steps),
                )
                row = dict(base)
                row.update({"variant": variant, **metrics})
                rows.append(row)
    finally:
        ASSAY_KILLSWITCH.clear()
        ASSAY_KILLSWITCH.update(previous)

    aggregate = _aggregate_rows(rows)
    payload = {
        "schema": "learned_gated_runtime_sandbox_v0",
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
    json_path = out_dir / "learned_gated_runtime_sandbox.json"
    md_path = out_dir / "LEARNED_GATED_RUNTIME_SANDBOX.md"
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
            "mean_final_world_jump_proxy": _mean([_as_float(row.get("final_world_jump_proxy", 0.0)) for row in subset]),
            "mean_child_writeback_mass": _mean([_as_float(row.get("mean_child_writeback_mass", 0.0)) for row in subset]),
            "mean_sum_child_writeback_mass": _mean([_as_float(row.get("sum_child_writeback_mass", 0.0)) for row in subset]),
            "mean_child_phase_writeback_delta_mass": _mean(
                [_as_float(row.get("mean_child_phase_writeback_delta_mass", 0.0)) for row in subset]
            ),
            "mean_child_support_writeback_mass": _mean(
                [_as_float(row.get("mean_child_support_writeback_mass", 0.0)) for row in subset]
            ),
            "mean_live_child_fraction": _mean([_as_float(row.get("mean_live_child_fraction", 0.0)) for row in subset]),
            "mean_writeback_event_count": _mean([_as_float(row.get("writeback_event_count", 0.0)) for row in subset]),
            "mean_collapse_event_count": _mean([_as_float(row.get("collapse_event_count", 0.0)) for row in subset]),
        }
    ungated = by_variant.get("ungated", {})
    blocked = by_variant.get("blocked_zero", {})
    for variant, summary in by_variant.items():
        if variant == "ungated":
            summary["parent_retention_vs_ungated"] = 1.0
            summary["world_jump_retention_vs_ungated"] = 1.0
        else:
            parent = summary.get("mean_final_parent_phase_divergence", 0.0)
            jump = summary.get("mean_final_world_jump_proxy", 0.0)
            u_parent = ungated.get("mean_final_parent_phase_divergence", 0.0)
            u_jump = ungated.get("mean_final_world_jump_proxy", 0.0)
            summary["parent_retention_vs_ungated"] = parent / u_parent if u_parent > 1.0e-8 else 0.0
            summary["world_jump_retention_vs_ungated"] = jump / u_jump if u_jump > 1.0e-8 else 0.0
            summary["world_jump_reduction_vs_ungated"] = u_jump - jump
            summary["parent_divergence_reduction_vs_ungated"] = u_parent - parent
        parent = summary.get("mean_final_parent_phase_divergence", 0.0)
        jump = summary.get("mean_final_world_jump_proxy", 0.0)
        b_parent = blocked.get("mean_final_parent_phase_divergence", 0.0)
        b_jump = blocked.get("mean_final_world_jump_proxy", 0.0)
        summary["parent_divergence_lift_vs_blocked"] = parent - b_parent
        summary["world_jump_lift_vs_blocked"] = jump - b_jump
    return {
        "status": "pass_learned_gated_runtime_sandbox" if rows else "fail_no_rows",
        "row_count": len(rows),
        "variant_count": len(variants),
        "variants": by_variant,
    }


def _write_markdown(path: Path, payload: dict[str, Any]) -> None:
    agg = payload["aggregate"]
    lines = [
        "# Learned-Gated Runtime Sandbox",
        "",
        "This assay routes the learned shadow branch-law gate through actual `circleworld_step()` child writeback, not a detached operator recurrence.",
        "",
        "## Summary",
        "",
        f"- status: `{agg['status']}`",
        f"- rows: `{agg['row_count']}`",
        f"- runtime steps: `{payload['runtime_steps']}`",
        f"- grandchild depth: `{payload['grandchild_depth']}`",
        f"- soft floor/full: `{payload['soft_floor']}` / `{payload['soft_full']}`",
        "",
        "## Variants",
        "",
        "| variant | rows | scale | parent div | world jump | child writeback | events | parent retained | jump retained | parent lift vs blocked | jump lift vs blocked |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for variant, summary in agg.get("variants", {}).items():
        lines.append(
            "| {variant} | {rows} | {scale} | {parent} | {jump} | {writeback} | {events} | {pretain} | {jretain} | {plift} | {jlift} |".format(
                variant=variant,
                rows=summary.get("row_count", 0.0),
                scale=summary.get("mean_writeback_scale", 0.0),
                parent=summary.get("mean_final_parent_phase_divergence", 0.0),
                jump=summary.get("mean_final_world_jump_proxy", 0.0),
                writeback=summary.get("mean_child_writeback_mass", 0.0),
                events=summary.get("mean_writeback_event_count", 0.0),
                pretain=summary.get("parent_retention_vs_ungated", 0.0),
                jretain=summary.get("world_jump_retention_vs_ungated", 0.0),
                plift=summary.get("parent_divergence_lift_vs_blocked", 0.0),
                jlift=summary.get("world_jump_lift_vs_blocked", 0.0),
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description="Run learned hard/soft gates through actual Circleworld child writeback.")
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
    result = run_learned_gated_runtime_sandbox(
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
