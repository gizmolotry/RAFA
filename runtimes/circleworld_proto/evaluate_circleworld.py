from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[2]
CORE = ROOT / "core"
LINEAGE = ROOT / "lineages" / "04_positive_replacement"
for path in (ROOT, CORE, LINEAGE):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from ablate_formalization import _generate_seed_phase, make_seed_rafa
from circleworld import CircleworldConfig, circleworld_loss, recurse_circleworld, summarize_circleworld_run
from lib_blackwell import NakedRAFA


def _safe_device(requested: str) -> torch.device:
    if requested == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


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
        phase_law_precondition_gain=float(cfg.get("phase_law_precondition_gain", 0.0)),
        phase_law_velocity_mix=float(cfg.get("phase_law_velocity_mix", 0.0)),
        phase_law_stability_gain=float(cfg.get("phase_law_stability_gain", 1.0)),
        phase_law_softclip=float(cfg.get("phase_law_softclip", 0.0)),
        phase_law_low_rank=int(cfg.get("phase_law_low_rank", 0)),
        phase_law_consensus_mix=float(cfg.get("phase_law_consensus_mix", 0.0)),
        phase_law_consensus_damping=float(cfg.get("phase_law_consensus_damping", 0.0)),
        phase_law_median_guard=float(cfg.get("phase_law_median_guard", 0.0)),
        phase_law_local_velocity_mix=float(cfg.get("phase_law_local_velocity_mix", 0.0)),
        phase_law_local_coherence_damping=float(cfg.get("phase_law_local_coherence_damping", 0.0)),
        phase_law_curvature_guard=float(cfg.get("phase_law_curvature_guard", 0.0)),
        phase_law_reentry_mix=float(cfg.get("phase_law_reentry_mix", 0.0)),
        phase_law_reentry_accel_mix=float(cfg.get("phase_law_reentry_accel_mix", 0.0)),
        phase_law_reentry_causal=bool(cfg.get("phase_law_reentry_causal", False)),
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
        child_operator_seed_gain=float(cfg.get("child_operator_seed_gain", 2.40)),
        child_operator_promotability_gain=float(cfg.get("child_operator_promotability_gain", 0.60)),
        child_writeback_phase_delta_gain=float(cfg.get("child_writeback_phase_delta_gain", 1.00)),
        child_writeback_operator_mix=float(cfg.get("child_writeback_operator_mix", 0.25)),
        child_writeback_phase_floor_target=float(cfg.get("child_writeback_phase_floor_target", 0.30)),
        child_writeback_phase_floor_threshold_mult=float(cfg.get("child_writeback_phase_floor_threshold_mult", 2.50)),
        child_writeback_phase_floor_gain=float(cfg.get("child_writeback_phase_floor_gain", 2.10)),
        child_writeback_parent_mix_cap=float(cfg.get("child_writeback_parent_mix_cap", 0.04)),
        child_support_writeback_gain=float(cfg.get("child_support_writeback_gain", 1.00)),
        child_support_writeback_floor_gain=float(cfg.get("child_support_writeback_floor_gain", 0.35)),
        child_kill_threshold=float(cfg.get("child_kill_threshold", 0.08)),
        child_local_ifs_enabled=bool(cfg.get("child_local_ifs_enabled", False)),
        child_local_steps=int(cfg.get("child_local_steps", 1)),
        child_local_support_only=bool(cfg.get("child_local_support_only", True)),
        child_local_coherence_retention_enabled=bool(cfg.get("child_local_coherence_retention_enabled", False)),
        child_local_coherence_retention_mix=float(cfg.get("child_local_coherence_retention_mix", 0.0)),
        child_local_coherence_floor=float(cfg.get("child_local_coherence_floor", 0.0)),
        child_local_coherence_floor_support=float(cfg.get("child_local_coherence_floor_support", 0.18)),
        child_local_coherence_causal_gate_enabled=bool(
            cfg.get("child_local_coherence_causal_gate_enabled", False)
        ),
        child_local_coherence_causal_min_delta=float(cfg.get("child_local_coherence_causal_min_delta", 0.02)),
        child_local_coherence_causal_full_delta=float(cfg.get("child_local_coherence_causal_full_delta", 0.18)),
        child_local_coherence_causal_gate_floor=float(cfg.get("child_local_coherence_causal_gate_floor", 0.0)),
        law_packet_merge_threshold=float(cfg.get("law_packet_merge_threshold", 0.92)),
        law_packet_min_score=float(cfg.get("law_packet_min_score", 0.25)),
        law_packet_topk_families=int(cfg.get("law_packet_topk_families", 8)),
    )


def _heldout_plan(device: torch.device) -> list[tuple[str, int]]:
    plan: list[tuple[str, int]] = [("synthetic", 5100 + i) for i in range(6)]
    if device.type == "cuda":
        plan.extend([("naked_rafa", 6100 + i) for i in range(3)])
    return plan


def _trajectory_rows(run: dict[str, Any], source: str, seed: int) -> list[dict[str, Any]]:
    hist = run["history"]
    rows: list[dict[str, Any]] = []
    for depth_idx, block in enumerate(hist):
        major = block["major_mass"][0].detach().cpu()
        residue = block["minor_residue"][0].detach().cpu()
        promo = block["promotability"][0].detach().cpu()
        per_q = block["per_q"][0].abs().detach().cpu()  # [K, T]
        q_mass = per_q / per_q.sum(dim=0, keepdim=True).clamp_min(1e-8)
        dom = q_mass.max(dim=0).values
        entropy = -(q_mass * q_mass.clamp_min(1e-8).log()).sum(dim=0) / torch.log(torch.tensor(float(q_mass.size(0))))
        for t in range(major.numel()):
            rows.append(
                {
                    "source": source,
                    "seed": seed,
                    "depth": depth_idx,
                    "t": t,
                    "major_mass": float(major[t].item()),
                    "minor_residue": float(residue[t].item()),
                    "promotability": float(promo[t].item()),
                    "dominant_q_share": float(dom[t].item()),
                    "q_entropy": float(entropy[t].item()),
                }
            )
    return rows


def evaluate_circle_cfg(
    cfg: CircleworldConfig,
    *,
    time_steps: int,
    device_name: str,
    plan: list[tuple[str, int]] | None = None,
    config_path: Path | None = None,
    out_dir: Path | None = None,
    include_trajectory: bool = True,
) -> dict[str, Any]:
    device = _safe_device(device_name)
    resolved_plan = list(plan) if plan is not None else _heldout_plan(device)
    rafa_core: NakedRAFA | None = make_seed_rafa(dev=device.type) if any(src == "naked_rafa" for src, _ in resolved_plan) else None

    rows: list[dict[str, Any]] = []
    traj_rows: list[dict[str, Any]] = []
    for source, seed in resolved_plan:
        phase_state = _generate_seed_phase(
            rafa_core=rafa_core,
            batch_size=1,
            time_steps=time_steps,
            device=device,
            seed_source=source,
            seed=seed,
        ).detach()
        run_mode = cfg.branching_mode if cfg.branching_mode in {"native_multimode", "native_multimode_childworld"} else "active_packets"
        run = recurse_circleworld(phase_state, cfg=cfg, depth=cfg.recursion_depth, mode=run_mode)
        summary = summarize_circleworld_run(run)
        losses = circleworld_loss(run)
        rows.append(
            {
                "source": str(source),
                "seed": int(seed),
                **summary,
                "loss": float(losses["loss"].item()),
                "l_q_dom": float(losses["l_q_dom"].item()),
                "l_q_entropy": float(losses["l_q_entropy"].item()),
                "l_major_sat": float(losses["l_major_sat"].item()),
            }
        )
        if include_trajectory:
            traj_rows.extend(_trajectory_rows(run, source=str(source), seed=int(seed)))

    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)

    csv_path: str | None = None
    if include_trajectory and out_dir is not None and traj_rows:
        csv_file = out_dir / "heldout_trajectory.csv"
        with csv_file.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(traj_rows[0].keys()))
            writer.writeheader()
            writer.writerows(traj_rows)
        csv_path = str(csv_file)

    def mean_key(items: list[dict[str, Any]], key: str) -> float:
        return float(sum(float(row.get(key, 0.0)) for row in items) / len(items))

    summary = {
        "config_path": str(config_path) if config_path is not None else None,
        "device": str(device),
        "time_steps": time_steps,
        "num_samples": len(rows),
        "mean_major_gain": float(sum(r["major_gain"] for r in rows) / len(rows)),
        "mean_residue_drop": float(sum(r["residue_drop"] for r in rows) / len(rows)),
        "mean_promotability_gain": float(sum(r["promotability_gain"] for r in rows) / len(rows)),
        "mean_dominant_q_share": float(sum(r["dominant_q_share"] for r in rows) / len(rows)),
        "mean_q_entropy": float(sum(r["q_entropy"] for r in rows) / len(rows)),
        "mean_major_saturation": float(sum(r["major_saturation"] for r in rows) / len(rows)),
        "mean_loss": float(sum(r["loss"] for r in rows) / len(rows)),
        "mean_l_q_dom": float(sum(r["l_q_dom"] for r in rows) / len(rows)),
        "mean_l_q_entropy": float(sum(r["l_q_entropy"] for r in rows) / len(rows)),
        "mean_l_major_sat": float(sum(r["l_major_sat"] for r in rows) / len(rows)),
        "mean_slot2_live_fraction": float(sum(r.get("mean_slot2_live_fraction", 0.0) for r in rows) / len(rows)),
        "mean_real_branch_fraction": float(sum(r.get("real_branch_fraction", 0.0) for r in rows) / len(rows)),
        "mean_parent_real_branch_fraction": mean_key(rows, "parent_real_branch_fraction"),
        "mean_child_real_branch_fraction": mean_key(rows, "child_real_branch_fraction"),
        "mean_meso_branch_effect": float(sum(r.get("meso_branch_effect", 0.0) for r in rows) / len(rows)),
        "mean_silent_singlepath_fraction": float(sum(r.get("silent_singlepath_fraction", 0.0) for r in rows) / len(rows)),
        "mean_branch_positive_mask": float(sum(r.get("mean_branch_positive_mask", 0.0) for r in rows) / len(rows)),
        "mean_branch_negative_mask": float(sum(r.get("mean_branch_negative_mask", 0.0) for r in rows) / len(rows)),
        "mean_decorative_slot2_fraction": float(sum(r.get("mean_decorative_slot2_fraction", 0.0) for r in rows) / len(rows)),
        "mean_relation_kernel": float(sum(r.get("mean_relation_kernel", 0.0) for r in rows) / len(rows)),
        "mean_relation_handoff_drive": float(sum(r.get("mean_relation_handoff_drive", 0.0) for r in rows) / len(rows)),
        "mean_relation_attn_10": float(sum(r.get("mean_relation_attn_10", 0.0) for r in rows) / len(rows)),
        "mean_relation_attn_01": float(sum(r.get("mean_relation_attn_01", 0.0) for r in rows) / len(rows)),
        "mean_branch_defect": float(sum(r.get("mean_branch_defect", 0.0) for r in rows) / len(rows)),
        "mean_branch_world_grad": float(sum(r.get("mean_branch_world_grad", 0.0) for r in rows) / len(rows)),
        "mean_branch_q_disagreement": float(sum(r.get("mean_branch_q_disagreement", 0.0) for r in rows) / len(rows)),
        "mean_branch_phase_wall": float(sum(r.get("mean_branch_phase_wall", 0.0) for r in rows) / len(rows)),
        "mean_branch_seed_energy": float(sum(r.get("mean_branch_seed_energy", 0.0) for r in rows) / len(rows)),
        "mean_phase_only_branch_surface": mean_key(rows, "mean_phase_only_branch_surface"),
        "mean_phase_only_branch_surface_peak": mean_key(rows, "mean_phase_only_branch_surface_peak"),
        "mean_phase_only_branch_phase_wall": mean_key(rows, "mean_phase_only_branch_phase_wall"),
        "mean_phase_only_branch_support_balance": mean_key(rows, "mean_phase_only_branch_support_balance"),
        "mean_phase_only_branch_distinctness": mean_key(rows, "mean_phase_only_branch_distinctness"),
        "mean_phase_only_branch_distinctness_balanced": mean_key(rows, "mean_phase_only_branch_distinctness_balanced"),
        "phase_only_real_branch_fraction": mean_key(rows, "phase_only_real_branch_fraction"),
        "phase_only_excess_branch_fraction": mean_key(rows, "phase_only_excess_branch_fraction"),
        "decorative_slot2_low_phase_fraction": mean_key(rows, "decorative_slot2_low_phase_fraction"),
        "final_phase_only_branch_surface": mean_key(rows, "final_phase_only_branch_surface"),
        "final_phase_only_branch_surface_peak": mean_key(rows, "final_phase_only_branch_surface_peak"),
        "final_phase_only_branch_phase_wall": mean_key(rows, "final_phase_only_branch_phase_wall"),
        "final_phase_only_branch_live_fraction": mean_key(rows, "final_phase_only_branch_live_fraction"),
        "final_phase_only_branch_support_balance": mean_key(rows, "final_phase_only_branch_support_balance"),
        "final_phase_only_branch_distinctness": mean_key(rows, "final_phase_only_branch_distinctness"),
        "final_phase_only_branch_distinctness_balanced": mean_key(rows, "final_phase_only_branch_distinctness_balanced"),
        "mean_child_world_count": float(sum(r.get("mean_child_world_count", 0.0) for r in rows) / len(rows)),
        "mean_live_child_fraction": float(sum(r.get("mean_live_child_fraction", 0.0) for r in rows) / len(rows)),
        "mean_child_age": float(sum(r.get("mean_child_age", 0.0) for r in rows) / len(rows)),
        "mean_child_writeback_mass": float(sum(r.get("mean_child_writeback_mass", 0.0) for r in rows) / len(rows)),
        "mean_child_writeback_gate_mass": mean_key(rows, "mean_child_writeback_gate_mass"),
        "mean_child_phase_writeback_delta_mass": mean_key(rows, "mean_child_phase_writeback_delta_mass"),
        "mean_child_parent_phase_writeback_delta_mass": mean_key(rows, "mean_child_parent_phase_writeback_delta_mass"),
        "mean_child_support_writeback_mass": float(sum(r.get("mean_child_support_writeback_mass", 0.0) for r in rows) / len(rows)),
        "mean_child_logit_writeback_mass": mean_key(rows, "mean_child_logit_writeback_mass"),
        "mean_child_qtrace_writeback_mass": mean_key(rows, "mean_child_qtrace_writeback_mass"),
        "mean_child_local_ifs_step_count": mean_key(rows, "mean_child_local_ifs_step_count"),
        "mean_child_local_ifs_packet_count": mean_key(rows, "mean_child_local_ifs_packet_count"),
        "mean_child_local_ifs_phase_delta": mean_key(rows, "mean_child_local_ifs_phase_delta"),
        "mean_child_local_ifs_support": mean_key(rows, "mean_child_local_ifs_support"),
        "mean_child_local_ifs_coherence": mean_key(rows, "mean_child_local_ifs_coherence"),
        "mean_child_local_ifs_coherence_raw": mean_key(rows, "mean_child_local_ifs_coherence_raw"),
        "mean_child_local_ifs_coherence_retention_delta": mean_key(rows, "mean_child_local_ifs_coherence_retention_delta"),
        "mean_child_local_ifs_coherence_floor_delta": mean_key(rows, "mean_child_local_ifs_coherence_floor_delta"),
        "mean_child_local_ifs_parent_phase_delta": mean_key(rows, "mean_child_local_ifs_parent_phase_delta"),
        "mean_child_local_ifs_causal_gate": mean_key(rows, "mean_child_local_ifs_causal_gate"),
        "mean_child_local_ifs_causal_retention_loss": mean_key(rows, "mean_child_local_ifs_causal_retention_loss"),
        "mean_child_local_ifs_causal_floor_loss": mean_key(rows, "mean_child_local_ifs_causal_floor_loss"),
        "mean_child_parent_divergence": float(sum(r.get("mean_child_parent_divergence", 0.0) for r in rows) / len(rows)),
        "mean_child_sibling_divergence": float(sum(r.get("mean_child_sibling_divergence", 0.0) for r in rows) / len(rows)),
        "mean_defect_without_branch_penalty": float(sum(r.get("mean_defect_without_branch_penalty", 0.0) for r in rows) / len(rows)),
        "mean_branch_resolution_delay": float(sum(r.get("mean_branch_resolution_delay", 0.0) for r in rows) / len(rows)),
        "mean_num_law_packets": float(sum(r.get("num_law_packets", 0.0) for r in rows) / len(rows)),
        "mean_num_law_families": float(sum(r.get("num_law_families", 0.0) for r in rows) / len(rows)),
        "mean_law_family_size": float(sum(r.get("mean_law_family_size", 0.0) for r in rows) / len(rows)),
        "mean_dominant_law_family_share": float(sum(r.get("dominant_law_family_share", 0.0) for r in rows) / len(rows)),
        "mean_law_family_entropy": float(sum(r.get("law_family_entropy", 0.0) for r in rows) / len(rows)),
        "mean_dominant_law_q_share": float(sum(r.get("dominant_law_q_share", 0.0) for r in rows) / len(rows)),
        "mean_law_top_q_entropy": float(sum(r.get("law_top_q_entropy", 0.0) for r in rows) / len(rows)),
        "mean_num_law_top_q_unique": float(sum(r.get("num_law_top_q_unique", 0.0) for r in rows) / len(rows)),
        "mean_num_relational_signatures": float(sum(r.get("num_relational_signatures", 0.0) for r in rows) / len(rows)),
        "mean_num_relational_signature_families": float(sum(r.get("num_relational_signature_families", 0.0) for r in rows) / len(rows)),
        "mean_relational_signature_confidence": float(sum(r.get("mean_relational_signature_confidence", 0.0) for r in rows) / len(rows)),
        "mean_relational_signature_q_entropy": float(sum(r.get("mean_relational_signature_q_entropy", 0.0) for r in rows) / len(rows)),
        "mean_relational_branch_mass": float(sum(r.get("mean_relational_branch_mass", 0.0) for r in rows) / len(rows)),
        "mean_dominant_relational_family_share": float(sum(r.get("dominant_relational_family_share", 0.0) for r in rows) / len(rows)),
        "by_source": {},
        "rows": rows,
        "trajectory_csv": csv_path,
    }
    for source in sorted({str(r["source"]) for r in rows}):
        subset = [r for r in rows if r["source"] == source]
        summary["by_source"][source] = {
            "count": len(subset),
            "mean_real_branch_fraction": float(sum(r.get("real_branch_fraction", 0.0) for r in subset) / len(subset)),
            "mean_parent_real_branch_fraction": mean_key(subset, "parent_real_branch_fraction"),
            "mean_child_real_branch_fraction": mean_key(subset, "child_real_branch_fraction"),
            "mean_meso_branch_effect": float(sum(r.get("meso_branch_effect", 0.0) for r in subset) / len(subset)),
            "mean_child_world_count": float(sum(r.get("mean_child_world_count", 0.0) for r in subset) / len(subset)),
            "mean_live_child_fraction": float(sum(r.get("mean_live_child_fraction", 0.0) for r in subset) / len(subset)),
            "mean_child_writeback_mass": float(sum(r.get("mean_child_writeback_mass", 0.0) for r in subset) / len(subset)),
            "mean_child_writeback_gate_mass": mean_key(subset, "mean_child_writeback_gate_mass"),
            "mean_child_phase_writeback_delta_mass": mean_key(subset, "mean_child_phase_writeback_delta_mass"),
            "mean_child_parent_phase_writeback_delta_mass": mean_key(subset, "mean_child_parent_phase_writeback_delta_mass"),
            "mean_child_support_writeback_mass": float(
                sum(r.get("mean_child_support_writeback_mass", 0.0) for r in subset) / len(subset)
            ),
            "mean_child_logit_writeback_mass": mean_key(subset, "mean_child_logit_writeback_mass"),
            "mean_child_qtrace_writeback_mass": mean_key(subset, "mean_child_qtrace_writeback_mass"),
            "mean_child_local_ifs_step_count": mean_key(subset, "mean_child_local_ifs_step_count"),
            "mean_child_local_ifs_packet_count": mean_key(subset, "mean_child_local_ifs_packet_count"),
            "mean_child_local_ifs_phase_delta": mean_key(subset, "mean_child_local_ifs_phase_delta"),
            "mean_child_local_ifs_support": mean_key(subset, "mean_child_local_ifs_support"),
            "mean_child_local_ifs_coherence": mean_key(subset, "mean_child_local_ifs_coherence"),
            "mean_child_local_ifs_coherence_raw": mean_key(subset, "mean_child_local_ifs_coherence_raw"),
            "mean_child_local_ifs_coherence_retention_delta": mean_key(subset, "mean_child_local_ifs_coherence_retention_delta"),
            "mean_child_local_ifs_coherence_floor_delta": mean_key(subset, "mean_child_local_ifs_coherence_floor_delta"),
            "mean_child_local_ifs_parent_phase_delta": mean_key(subset, "mean_child_local_ifs_parent_phase_delta"),
            "mean_child_local_ifs_causal_gate": mean_key(subset, "mean_child_local_ifs_causal_gate"),
            "mean_child_local_ifs_causal_retention_loss": mean_key(
                subset, "mean_child_local_ifs_causal_retention_loss"
            ),
            "mean_child_local_ifs_causal_floor_loss": mean_key(subset, "mean_child_local_ifs_causal_floor_loss"),
            "mean_child_parent_divergence": float(sum(r.get("mean_child_parent_divergence", 0.0) for r in subset) / len(subset)),
            "mean_child_sibling_divergence": float(sum(r.get("mean_child_sibling_divergence", 0.0) for r in subset) / len(subset)),
            "mean_phase_only_branch_surface": mean_key(subset, "mean_phase_only_branch_surface"),
            "mean_phase_only_branch_surface_peak": mean_key(subset, "mean_phase_only_branch_surface_peak"),
            "mean_phase_only_branch_phase_wall": mean_key(subset, "mean_phase_only_branch_phase_wall"),
            "mean_phase_only_branch_support_balance": mean_key(subset, "mean_phase_only_branch_support_balance"),
            "mean_phase_only_branch_distinctness": mean_key(subset, "mean_phase_only_branch_distinctness"),
            "mean_phase_only_branch_distinctness_balanced": mean_key(subset, "mean_phase_only_branch_distinctness_balanced"),
            "phase_only_real_branch_fraction": mean_key(subset, "phase_only_real_branch_fraction"),
            "phase_only_excess_branch_fraction": mean_key(subset, "phase_only_excess_branch_fraction"),
            "decorative_slot2_low_phase_fraction": mean_key(subset, "decorative_slot2_low_phase_fraction"),
            "final_phase_only_branch_surface": mean_key(subset, "final_phase_only_branch_surface"),
            "final_phase_only_branch_surface_peak": mean_key(subset, "final_phase_only_branch_surface_peak"),
            "final_phase_only_branch_phase_wall": mean_key(subset, "final_phase_only_branch_phase_wall"),
            "final_phase_only_branch_live_fraction": mean_key(subset, "final_phase_only_branch_live_fraction"),
            "final_phase_only_branch_support_balance": mean_key(subset, "final_phase_only_branch_support_balance"),
            "final_phase_only_branch_distinctness": mean_key(subset, "final_phase_only_branch_distinctness"),
            "final_phase_only_branch_distinctness_balanced": mean_key(subset, "final_phase_only_branch_distinctness_balanced"),
            "mean_num_law_families": float(sum(r.get("num_law_families", 0.0) for r in subset) / len(subset)),
            "mean_law_family_entropy": float(sum(r.get("law_family_entropy", 0.0) for r in subset) / len(subset)),
            "mean_num_relational_signatures": float(sum(r.get("num_relational_signatures", 0.0) for r in subset) / len(subset)),
            "mean_num_relational_signature_families": float(sum(r.get("num_relational_signature_families", 0.0) for r in subset) / len(subset)),
            "mean_relational_signature_confidence": float(sum(r.get("mean_relational_signature_confidence", 0.0) for r in subset) / len(subset)),
            "mean_relational_signature_q_entropy": float(sum(r.get("mean_relational_signature_q_entropy", 0.0) for r in subset) / len(subset)),
            "mean_relational_branch_mass": float(sum(r.get("mean_relational_branch_mass", 0.0) for r in subset) / len(subset)),
        }

    if out_dir is not None:
        (out_dir / "heldout_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def evaluate_config(config_path: Path, out_dir: Path, time_steps: int, device_name: str) -> dict[str, Any]:
    cfg = _load_circle_cfg(config_path)
    return evaluate_circle_cfg(
        cfg,
        time_steps=time_steps,
        device_name=device_name,
        config_path=config_path,
        out_dir=out_dir,
        include_trajectory=True,
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate a Circleworld config on held-out seeds and export trajectories.")
    ap.add_argument("--config", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--time-steps", type=int, default=128)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    summary = evaluate_config(
        config_path=Path(args.config),
        out_dir=Path(args.out_dir),
        time_steps=args.time_steps,
        device_name=args.device,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
