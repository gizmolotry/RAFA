from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
CORE = ROOT / "core"
LINEAGE = ROOT / "lineages" / "04_positive_replacement"
for path in (ROOT, CORE, LINEAGE):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from circleworld import (
    CircleworldConfig,
    RafaLearnedBranchLawV0,
    branch_law_feature_fields_from_state,
    branch_law_feature_names,
    branch_law_feature_tensor,
    child_circleworld_step,
    clone_circleworld_state,
    circleworld_step,
    learned_branch_law_output_names,
    learned_branch_law_output_tensor,
    learned_branch_law_targets_from_features,
    recurse_circleworld,
    summarize_circleworld_run,
)
from evaluate_circleworld import _load_circle_cfg, _safe_device
from rafa_math_tools import phase_to_phasor, phasor_normalize


DEFAULT_OUT_DIR = (
    Path(r"C:\Users\Andrew\AppData\Local\Temp\rafa_tokenburst")
    / "learned_branch_law_child_ifs_assay_2026_05_09"
)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _coherent_seed(seed: int, f_bins: int, t_len: int, device: torch.device) -> torch.Tensor:
    generator = torch.Generator(device=device)
    generator.manual_seed(int(seed))
    freq = torch.linspace(-1.0, 1.0, steps=f_bins, device=device).view(1, f_bins, 1)
    time = torch.linspace(0.0, 1.0, steps=t_len, device=device).view(1, 1, t_len)
    q = float((2, 3, 4, 5, 6, 8)[seed % 6])
    branch = 0.25 * torch.sin(math.pi * (1.0 + q / 12.0) * freq) * torch.sin(math.pi * time)
    drift = 2.0 * math.pi * (0.35 * q * time + 0.18 * freq * time)
    noise = 0.18 * torch.randn((1, f_bins, t_len), generator=generator, device=device)
    return phase_to_phasor(drift + branch + noise)


def _phase_delta(a: torch.Tensor, b: torch.Tensor, mask: torch.Tensor | None = None) -> float:
    cross = a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]
    dot = a[..., 0] * b[..., 0] + a[..., 1] * b[..., 1]
    delta = torch.atan2(cross, dot).abs()
    if mask is not None:
        denom = mask.sum().clamp_min(1e-8)
        return float((delta * mask).sum().div(denom).detach().cpu().item())
    return float(delta.mean().detach().cpu().item())


def _tensor_stats(x: torch.Tensor) -> dict[str, float]:
    y = x.detach().float().cpu()
    return {
        "mean": float(y.mean().item()),
        "std": float(y.std(unbiased=False).item()),
        "min": float(y.min().item()),
        "max": float(y.max().item()),
    }


def _collect_branch_examples(
    cfg: CircleworldConfig,
    seeds: list[int],
    *,
    depth: int,
    f_bins: int,
    t_len: int,
    device: torch.device,
    max_points_per_state: int,
) -> tuple[torch.Tensor, torch.Tensor, list[dict[str, Any]], list[dict[str, Any]]]:
    features: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []
    run_summaries: list[dict[str, Any]] = []
    state_rows: list[dict[str, Any]] = []
    for seed in seeds:
        z = _coherent_seed(seed, f_bins, t_len, device)
        run = recurse_circleworld(z, cfg=cfg, depth=depth, mode="native_multimode_childworld")
        run_summaries.append(summarize_circleworld_run(run))
        for state_idx, state in enumerate(run.get("multimode_history", [])):
            fields = branch_law_feature_fields_from_state(state, cfg)
            feature = branch_law_feature_tensor(fields).reshape(-1, len(branch_law_feature_names())).detach()
            target = learned_branch_law_output_tensor(learned_branch_law_targets_from_features(feature)).detach()
            if max_points_per_state > 0 and feature.size(0) > max_points_per_state:
                idx = torch.linspace(0, feature.size(0) - 1, steps=max_points_per_state, device=feature.device).long()
                feature = feature.index_select(0, idx)
                target = target.index_select(0, idx)
            features.append(feature)
            targets.append(target)
            state_rows.append(
                {
                    "seed": int(seed),
                    "state_index": int(state_idx),
                    "feature_count": int(feature.size(0)),
                    "child_world_count": int(len(state.get("child_worlds", []) or [])),
                }
            )
    if not features:
        raise RuntimeError("No branch-law feature states were collected")
    return torch.cat(features, dim=0), torch.cat(targets, dim=0), run_summaries, state_rows


def _train_shadow_law(
    features: torch.Tensor,
    targets: torch.Tensor,
    *,
    epochs: int,
    lr: float,
    seed: int,
) -> tuple[RafaLearnedBranchLawV0, dict[str, Any]]:
    torch.manual_seed(int(seed))
    model = RafaLearnedBranchLawV0(input_dim=features.size(-1), output_dim=targets.size(-1)).to(features.device)
    opt = torch.optim.AdamW(model.parameters(), lr=float(lr), weight_decay=1.0e-4)
    loss_history: list[float] = []
    for _ in range(max(1, int(epochs))):
        opt.zero_grad(set_to_none=True)
        pred = learned_branch_law_output_tensor(model(features))
        loss = F.mse_loss(pred, targets)
        loss.backward()
        opt.step()
        loss_history.append(float(loss.detach().cpu().item()))

    probe = features[: min(2048, features.size(0))].detach().clone().requires_grad_(True)
    pred_probe = learned_branch_law_output_tensor(model(probe))
    saliency_loss = F.mse_loss(pred_probe, targets[: probe.size(0)])
    saliency_loss.backward()
    saliency = probe.grad.detach().abs().mean(dim=0)
    with torch.no_grad():
        pred_all = learned_branch_law_output_tensor(model(features))
        output_rows = {}
        for idx, name in enumerate(learned_branch_law_output_names()):
            output_rows[name] = {
                "prediction": _tensor_stats(pred_all[:, idx]),
                "target": _tensor_stats(targets[:, idx]),
                "mae": float((pred_all[:, idx] - targets[:, idx]).abs().mean().detach().cpu().item()),
            }
    report = {
        "train_loss_initial": loss_history[0],
        "train_loss_final": loss_history[-1],
        "loss_history_tail": loss_history[-10:],
        "feature_saliency": {
            name: float(saliency[idx].detach().cpu().item())
            for idx, name in enumerate(branch_law_feature_names())
        },
        "outputs": output_rows,
    }
    return model, report


def _first_active_child(state: dict[str, Any]) -> dict[str, Any] | None:
    for child in state.get("child_worlds", []) or []:
        if child.get("active", False):
            return child
    return None


def _child_probe_suite(
    cfg: CircleworldConfig,
    seeds: list[int],
    *,
    depth: int,
    f_bins: int,
    t_len: int,
    device: torch.device,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    sibling_rows: list[dict[str, Any]] = []
    for seed in seeds:
        z = _coherent_seed(seed, f_bins, t_len, device)
        enabled_cfg = replace(cfg, child_local_ifs_enabled=True)
        no_write_cfg = replace(
            enabled_cfg,
            child_writeback_gain=0.0,
            child_writeback_budget=0.0,
            child_parent_mix=0.0,
            child_parent_mix_early=0.0,
            child_support_writeback_gain=0.0,
            child_support_writeback_floor_gain=0.0,
        )
        run_enabled = recurse_circleworld(z, cfg=enabled_cfg, depth=depth, mode="native_multimode_childworld")
        state = run_enabled["multimode_state"]
        child = _first_active_child(state)
        if child is None:
            rows.append({"seed": int(seed), "status": "no_live_child"})
            continue
        parent_mode0 = state["phase_modes"][..., 0, :]
        parent_mode1 = state["phase_modes"][..., 1, :]
        isolated_child = clone_circleworld_state(child)
        coupled_child = clone_circleworld_state(child)
        isolated_child, isolated_metrics = child_circleworld_step(
            isolated_child,
            enabled_cfg,
            parent_mode0,
            parent_mode1,
            parent_coupled=False,
        )
        coupled_child, coupled_metrics = child_circleworld_step(
            coupled_child,
            enabled_cfg,
            parent_mode0,
            parent_mode1,
            parent_coupled=True,
        )
        support = coupled_child["mode_support"].clamp(0.0, 1.0)
        coupled_vs_isolated = _phase_delta(coupled_child["phase_state"], isolated_child["phase_state"], support)
        run_no_write = recurse_circleworld(z, cfg=no_write_cfg, depth=depth, mode="native_multimode_childworld")
        parent_writeback_divergence = _phase_delta(run_enabled["phase_state"], run_no_write["phase_state"])
        rows.append(
            {
                "seed": int(seed),
                "status": "live_child_evaluated",
                "child_id": int(child["child_id"]),
                "isolated_support": float(isolated_metrics["child_local_ifs_support_mean"].detach().cpu().item()),
                "isolated_coherence": float(isolated_metrics["child_local_ifs_coherence_mean"].detach().cpu().item()),
                "isolated_phase_delta": float(isolated_metrics["child_local_ifs_phase_delta"].detach().cpu().item()),
                "coupled_support": float(coupled_metrics["child_local_ifs_support_mean"].detach().cpu().item()),
                "coupled_coherence": float(coupled_metrics["child_local_ifs_coherence_mean"].detach().cpu().item()),
                "coupled_phase_delta": float(coupled_metrics["child_local_ifs_phase_delta"].detach().cpu().item()),
                "coupled_vs_isolated_phase_delta": float(coupled_vs_isolated),
                "parent_writeback_divergence": float(parent_writeback_divergence),
                "child_local_packet_count": float(coupled_metrics["child_local_ifs_packet_count"].detach().cpu().item()),
            }
        )
        active_children = [c for c in state.get("child_worlds", []) or [] if c.get("active", False)]
        if len(active_children) >= 2:
            a, b = active_children[0], active_children[1]
            mask = torch.maximum(a["mode_support"], b["mode_support"]).clamp(0.0, 1.0)
            sibling_rows.append(
                {
                    "seed": int(seed),
                    "child_a": int(a["child_id"]),
                    "child_b": int(b["child_id"]),
                    "sibling_phase_delta": _phase_delta(a["phase_state"], b["phase_state"], mask),
                    "support_union": float(mask.mean().detach().cpu().item()),
                }
            )
    evaluated = [row for row in rows if row.get("status") == "live_child_evaluated"]
    return {
        "rows": rows,
        "sibling_rows": sibling_rows,
        "summary": {
            "seed_count": len(seeds),
            "live_child_case_count": len(evaluated),
            "sibling_case_count": len(sibling_rows),
            "mean_isolated_coherence": (
                sum(float(row["isolated_coherence"]) for row in evaluated) / len(evaluated) if evaluated else 0.0
            ),
            "mean_coupled_vs_isolated_phase_delta": (
                sum(float(row["coupled_vs_isolated_phase_delta"]) for row in evaluated) / len(evaluated) if evaluated else 0.0
            ),
            "mean_parent_writeback_divergence": (
                sum(float(row["parent_writeback_divergence"]) for row in evaluated) / len(evaluated) if evaluated else 0.0
            ),
            "mean_sibling_phase_delta": (
                sum(float(row["sibling_phase_delta"]) for row in sibling_rows) / len(sibling_rows) if sibling_rows else 0.0
            ),
        },
    }


def _learned_output_fields(
    model: RafaLearnedBranchLawV0,
    state: dict[str, Any],
    cfg: CircleworldConfig,
) -> dict[str, torch.Tensor]:
    fields = branch_law_feature_fields_from_state(state, cfg)
    feature = branch_law_feature_tensor(fields)
    shape = feature.shape[:-1]
    with torch.no_grad():
        outputs = model(feature.reshape(-1, feature.size(-1)))
    return {name: value.reshape(shape) for name, value in outputs.items()}


def _apply_learned_branch_law_sandbox(
    state: dict[str, Any],
    cfg: CircleworldConfig,
    model: RafaLearnedBranchLawV0,
    *,
    gain: float,
) -> tuple[dict[str, Any], dict[str, float]]:
    sandbox = clone_circleworld_state(state)
    outputs = _learned_output_fields(model, sandbox, cfg)
    support = sandbox["mode_support"].clone().clamp(0.0, 1.0)
    logits = sandbox["mode_logits"].clone()
    coherence = sandbox.get("mode_coherence", support).clone().clamp(0.0, 1.0)
    q_trace = sandbox["mode_q_trace"].clone()

    g = float(gain)
    split = outputs["split_drive"].clamp(0.0, 1.0)
    coexist = outputs["coexistence_drive"].clamp(0.0, 1.0)
    merge = outputs["merge_drive"].clamp(0.0, 1.0)
    collapse = outputs["collapse_pressure"].clamp(0.0, 1.0)
    survival0 = outputs["survival_delta0"].clamp(-1.0, 1.0)
    survival1 = outputs["survival_delta1"].clamp(-1.0, 1.0)
    support0_delta = outputs["support_delta0"].clamp(-1.0, 1.0)
    support1_delta = outputs["support_delta1"].clamp(-1.0, 1.0)
    coherence_target = outputs["coherence_target"].clamp(0.0, 1.0)
    qmix = outputs["qtrace_inheritance_mix"].clamp(0.0, 1.0).unsqueeze(-1)

    # Assay-only control: nudge branch fields with the learned law, then let the
    # normal runtime continue from the cloned state.
    logits[..., 0] = logits[..., 0] + g * (0.35 * survival0 + 0.20 * merge - 0.10 * split)
    logits[..., 1] = logits[..., 1] + g * (0.35 * survival1 + 0.30 * split + 0.25 * coexist - 0.30 * collapse)
    support[..., 0] = (support[..., 0] + g * (0.12 * support0_delta + 0.05 * merge - 0.03 * split)).clamp(0.0, 1.0)
    support[..., 1] = (
        support[..., 1]
        + g * (0.12 * support1_delta + 0.10 * split + 0.10 * coexist - 0.08 * collapse)
    ).clamp(0.0, 1.0)
    coherence[..., 0] = ((1.0 - 0.15 * g) * coherence[..., 0] + (0.15 * g) * coherence_target).clamp(0.0, 1.0)
    coherence[..., 1] = ((1.0 - 0.25 * g) * coherence[..., 1] + (0.25 * g) * coherence_target).clamp(0.0, 1.0)
    q_trace[..., 1, :] = ((1.0 - 0.20 * g * qmix) * q_trace[..., 1, :] + (0.20 * g * qmix) * q_trace[..., 0, :]).clamp(
        -1.0,
        1.0,
    )

    sandbox["phase_modes"] = phasor_normalize(sandbox["phase_modes"])
    sandbox["mode_logits"] = logits
    sandbox["mode_support"] = support
    sandbox["mode_coherence"] = coherence
    sandbox["mode_q_trace"] = q_trace
    metrics = {
        "mean_split_drive": float(split.mean().detach().cpu().item()),
        "mean_coexistence_drive": float(coexist.mean().detach().cpu().item()),
        "mean_merge_drive": float(merge.mean().detach().cpu().item()),
        "mean_collapse_pressure": float(collapse.mean().detach().cpu().item()),
        "mean_support1_delta": float((support[..., 1] - state["mode_support"][..., 1]).mean().detach().cpu().item()),
        "mean_logit1_delta": float((logits[..., 1] - state["mode_logits"][..., 1]).mean().detach().cpu().item()),
        "mean_coherence1_delta": float(
            (coherence[..., 1] - state.get("mode_coherence", state["mode_support"])[..., 1]).mean().detach().cpu().item()
        ),
    }
    return sandbox, metrics


def _sandbox_control_suite(
    cfg: CircleworldConfig,
    model: RafaLearnedBranchLawV0,
    seeds: list[int],
    *,
    depth: int,
    f_bins: int,
    t_len: int,
    device: torch.device,
    gain: float,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    enabled_cfg = replace(cfg, child_local_ifs_enabled=True)
    for seed in seeds:
        z = _coherent_seed(seed, f_bins, t_len, device)
        run = recurse_circleworld(z, cfg=enabled_cfg, depth=depth, mode="native_multimode_childworld")
        state = run["multimode_state"]
        param_next, param_block, _ = circleworld_step(clone_circleworld_state(state), enabled_cfg, mode="native_multimode_childworld")
        sandbox_state, law_metrics = _apply_learned_branch_law_sandbox(state, enabled_cfg, model, gain=gain)
        learned_next, learned_block, _ = circleworld_step(sandbox_state, enabled_cfg, mode="native_multimode_childworld")
        support_mask = torch.maximum(
            param_next["mode_support"][..., 1].clamp(0.0, 1.0),
            learned_next["mode_support"][..., 1].clamp(0.0, 1.0),
        )
        rows.append(
            {
                "seed": int(seed),
                "gain": float(gain),
                "sandbox_vs_parametric_phase_delta": _phase_delta(
                    learned_next["mixed_phase_state"],
                    param_next["mixed_phase_state"],
                    support_mask,
                ),
                "parametric_live_child_fraction": float(param_block["live_child_fraction"].detach().cpu().item()),
                "learned_live_child_fraction": float(learned_block["live_child_fraction"].detach().cpu().item()),
                "parametric_writeback_mass": float(param_block["child_writeback_mass"].detach().cpu().item()),
                "learned_writeback_mass": float(learned_block["child_writeback_mass"].detach().cpu().item()),
                "parametric_parent_divergence": float(param_block["child_parent_divergence"].detach().cpu().item()),
                "learned_parent_divergence": float(learned_block["child_parent_divergence"].detach().cpu().item()),
                "parametric_sibling_divergence": float(param_block["child_sibling_divergence"].detach().cpu().item()),
                "learned_sibling_divergence": float(learned_block["child_sibling_divergence"].detach().cpu().item()),
                "parametric_slot2_live_fraction": float(param_block["slot2_live_fraction"].detach().cpu().item()),
                "learned_slot2_live_fraction": float(learned_block["slot2_live_fraction"].detach().cpu().item()),
                **law_metrics,
            }
        )
    return {
        "enabled": True,
        "gain": float(gain),
        "rows": rows,
        "summary": {
            "seed_count": len(seeds),
            "mean_phase_delta": sum(float(row["sandbox_vs_parametric_phase_delta"]) for row in rows) / len(rows) if rows else 0.0,
            "mean_live_child_delta": (
                sum(float(row["learned_live_child_fraction"] - row["parametric_live_child_fraction"]) for row in rows) / len(rows)
                if rows
                else 0.0
            ),
            "mean_writeback_delta": (
                sum(float(row["learned_writeback_mass"] - row["parametric_writeback_mass"]) for row in rows) / len(rows)
                if rows
                else 0.0
            ),
            "mean_parent_divergence_delta": (
                sum(float(row["learned_parent_divergence"] - row["parametric_parent_divergence"]) for row in rows) / len(rows)
                if rows
                else 0.0
            ),
            "mean_sibling_divergence_delta": (
                sum(float(row["learned_sibling_divergence"] - row["parametric_sibling_divergence"]) for row in rows) / len(rows)
                if rows
                else 0.0
            ),
        },
    }


def _markdown_report(payload: dict[str, Any]) -> str:
    shadow = payload["shadow_law"]
    child = payload["child_probe"]["summary"]
    sandbox = payload.get("sandbox_control", {"enabled": False, "summary": {}})
    lines = [
        "# Learned Branch Law / Child-Local IFS Assay",
        "",
        f"- schema: `{payload['schema']}`",
        f"- feature rows: `{payload['feature_rows']}`",
        f"- shadow train loss final: `{shadow['train_loss_final']:.6f}`",
        f"- live child cases: `{child['live_child_case_count']}` / `{child['seed_count']}`",
        f"- sibling cases: `{child['sibling_case_count']}`",
        f"- mean isolated coherence: `{child['mean_isolated_coherence']:.6f}`",
        f"- mean parent writeback divergence: `{child['mean_parent_writeback_divergence']:.6f}`",
    ]
    if sandbox.get("enabled", False):
        sandbox_summary = sandbox.get("summary", {})
        lines.extend(
            [
                f"- sandbox mean phase delta: `{float(sandbox_summary.get('mean_phase_delta', 0.0)):.6f}`",
                f"- sandbox mean live-child delta: `{float(sandbox_summary.get('mean_live_child_delta', 0.0)):.6f}`",
                f"- sandbox mean writeback delta: `{float(sandbox_summary.get('mean_writeback_delta', 0.0)):.6f}`",
            ]
        )
    lines.extend(["", "## Top Feature Saliency"])
    saliency = sorted(shadow["feature_saliency"].items(), key=lambda item: item[1], reverse=True)
    for name, value in saliency[:8]:
        lines.append(f"- `{name}`: `{value:.6f}`")
    lines.extend(["", "## Output MAE"])
    for name, row in shadow["outputs"].items():
        lines.append(f"- `{name}`: `{row['mae']:.6f}`")
    return "\n".join(lines) + "\n"


def run_assay(
    out_dir: Path,
    config_path: Path | None,
    device_name: str,
    seeds: list[int],
    depth: int,
    f_bins: int,
    t_len: int,
    epochs: int,
    lr: float,
    child_local_steps: int,
    max_points_per_state: int,
    sandbox_control: bool,
    sandbox_gain: float,
) -> dict[str, Any]:
    device = _safe_device(device_name)
    base_cfg = _load_circle_cfg(config_path) if config_path else CircleworldConfig()
    probe_cfg = replace(
        base_cfg,
        branching_mode="native_multimode_childworld",
        branch_law_version="parametric_v1",
        child_local_ifs_enabled=False,
        child_local_steps=max(1, int(child_local_steps)),
        child_spawn_threshold=min(float(base_cfg.child_spawn_threshold), 0.12),
        child_kill_threshold=min(float(base_cfg.child_kill_threshold), 0.02),
    )
    features, targets, run_summaries, state_rows = _collect_branch_examples(
        probe_cfg,
        seeds,
        depth=depth,
        f_bins=f_bins,
        t_len=t_len,
        device=device,
        max_points_per_state=max_points_per_state,
    )
    model, shadow_report = _train_shadow_law(features, targets, epochs=epochs, lr=lr, seed=1701)
    child_report = _child_probe_suite(
        replace(probe_cfg, child_local_ifs_enabled=True),
        seeds,
        depth=depth,
        f_bins=f_bins,
        t_len=t_len,
        device=device,
    )
    sandbox_report = (
        _sandbox_control_suite(
            replace(probe_cfg, child_local_ifs_enabled=True),
            model,
            seeds,
            depth=depth,
            f_bins=f_bins,
            t_len=t_len,
            device=device,
            gain=sandbox_gain,
        )
        if sandbox_control
        else {"enabled": False, "rows": [], "summary": {}}
    )
    parametric_means = {
        key: sum(float(row.get(key, 0.0)) for row in run_summaries) / len(run_summaries)
        for key in (
            "mean_split_events",
            "mean_merge_events",
            "mean_collapse_events",
            "mean_live_child_fraction",
            "mean_child_writeback_mass",
            "real_branch_fraction",
        )
    }
    payload = {
        "schema": "circleworld_learned_branch_law_child_ifs_assay_v0",
        "config_path": str(config_path) if config_path else None,
        "device": str(device),
        "seeds": seeds,
        "feature_names": list(branch_law_feature_names()),
        "output_names": list(learned_branch_law_output_names()),
        "feature_rows": int(features.size(0)),
        "feature_stats": {
            name: _tensor_stats(features[:, idx])
            for idx, name in enumerate(branch_law_feature_names())
        },
        "target_stats": {
            name: _tensor_stats(targets[:, idx])
            for idx, name in enumerate(learned_branch_law_output_names())
        },
        "parametric_reference": parametric_means,
        "state_rows": state_rows,
        "shadow_law": shadow_report,
        "child_probe": child_report,
        "sandbox_control": sandbox_report,
        "status": (
            "child_local_ifs_signal_present"
            if child_report["summary"]["live_child_case_count"] > 0
            and child_report["summary"]["mean_parent_writeback_divergence"] > 0.0
            else "needs_more_branch_active_child_cases"
        ),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_json(out_dir / "learned_branch_law_assay.json", payload)
    (out_dir / "LEARNED_BRANCH_LAW_ASSAY.md").write_text(_markdown_report(payload), encoding="utf-8")
    torch.save(model.state_dict(), out_dir / "rafa_learned_branch_law_v0.pt")
    return payload


def _parse_seeds(raw: str) -> list[int]:
    if "," in raw:
        return [int(part.strip()) for part in raw.split(",") if part.strip()]
    count = int(raw)
    return [9100 + idx for idx in range(max(1, count))]


def main() -> None:
    ap = argparse.ArgumentParser(description="Train and evaluate a shadow learned branch law plus child-local IFS probes.")
    ap.add_argument("--config", type=Path, default=None)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seeds", default="9100,9101,9102")
    ap.add_argument("--depth", type=int, default=4)
    ap.add_argument("--f-bins", type=int, default=16)
    ap.add_argument("--time-steps", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--lr", type=float, default=3.0e-3)
    ap.add_argument("--child-local-steps", type=int, default=2)
    ap.add_argument("--max-points-per-state", type=int, default=384)
    ap.add_argument("--sandbox-control", action="store_true", help="Let the learned law nudge cloned branch fields for one sandbox step.")
    ap.add_argument("--sandbox-gain", type=float, default=0.75)
    args = ap.parse_args()
    payload = run_assay(
        out_dir=args.out_dir,
        config_path=args.config,
        device_name=args.device,
        seeds=_parse_seeds(args.seeds),
        depth=args.depth,
        f_bins=args.f_bins,
        t_len=args.time_steps,
        epochs=args.epochs,
        lr=args.lr,
        child_local_steps=args.child_local_steps,
        max_points_per_state=args.max_points_per_state,
        sandbox_control=bool(args.sandbox_control),
        sandbox_gain=float(args.sandbox_gain),
    )
    print(json.dumps({"status": payload["status"], "out_dir": str(args.out_dir)}, indent=2))


if __name__ == "__main__":
    main()
