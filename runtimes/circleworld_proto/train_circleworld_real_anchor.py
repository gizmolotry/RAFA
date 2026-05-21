from __future__ import annotations

import argparse
import json
import math
import random
import sys
from copy import deepcopy
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
from config import load_config
from circleworld import build_law_token_library, circleworld_loss, recurse_circleworld, summarize_circleworld_run
from export_circleworld_audio import _blend_phase, _load_circle_cfg, prepare_reference_audio
from evaluate_circleworld import evaluate_circle_cfg
from rafa_relational_signature import build_relational_signature_library, concat_relational_signature_banks
from stft_utils import compute_stft, inverse_stft
from test_nested_commitment import evaluate_nested_commitment_report
from train_circleworld import (
    _CHILD_WRITEBACK_CONTROL_DEFAULTS,
    _CHILD_WRITEBACK_CONTROL_STDS,
    _CHILD_LOCAL_COHERENCE_CONTROL_DEFAULTS,
    _CHILD_LOCAL_COHERENCE_CONTROL_STDS,
    _PHASE_LAW_CONTROL_DEFAULTS,
    _PHASE_LAW_CONTROL_STDS,
    _candidate_from_mean_std,
    _child_local_coherence_values,
    _child_writeback_control_values,
    _graduation_anchor_wavs,
    _materialize_cfg,
    _phase_law_control_values,
    _safe_device,
    _set_seed,
)


def _load_json_arg(value: str | None, default: Any) -> Any:
    if not value:
        return default
    path = Path(value)
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8-sig"))
    return json.loads(value)


def _load_named_json_arg(value: str | None, default: Any, nested_key: str | None = None) -> Any:
    payload = _load_json_arg(value, default)
    if nested_key and isinstance(payload, dict) and nested_key in payload:
        return payload[nested_key]
    return payload


def _corrcoef_1d(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.float().view(-1)
    b = b.float().view(-1)
    a = a - a.mean()
    b = b - b.mean()
    denom = a.std(unbiased=False) * b.std(unbiased=False)
    if float(denom.item()) < 1e-12:
        return 0.0
    return float(((a * b).mean() / denom).item())


def _window_penalty(value: float, low: float | None, high: float | None) -> float:
    if low is not None and value < low:
        return float(low - value)
    if high is not None and value > high:
        return float(value - high)
    return 0.0


def _case_weight_for(item: dict[str, Any], case_weights: dict[str, float] | None) -> float:
    if not case_weights:
        return 1.0
    haystacks = [
        str(item.get("name", "")),
        str(item.get("wav_path", "")),
    ]
    for key, value in case_weights.items():
        if any(key in hay for hay in haystacks):
            return float(value)
    return 1.0


def _normalized_entropy(probs: torch.Tensor) -> float:
    if probs.numel() <= 1:
        return 0.0
    probs = probs / probs.sum().clamp_min(1e-8)
    return float((-(probs * probs.clamp_min(1e-8).log()).sum() / torch.log(probs.new_tensor(float(probs.numel())))).item())


def _law_library_stats(library: dict[str, Any], qset: tuple[int, ...]) -> dict[str, float]:
    families = list(library.get("families", []))
    if not families:
        return {
            "aggregate_num_law_packets": 0.0,
            "aggregate_num_law_families": 0.0,
            "aggregate_mean_law_family_size": 0.0,
            "aggregate_dominant_law_family_share": 1.0,
            "aggregate_law_family_entropy": 0.0,
            "aggregate_dominant_law_q_share": 1.0,
            "aggregate_law_top_q_entropy": 0.0,
            "aggregate_num_law_top_q_unique": 0.0,
        }
    counts = torch.tensor([float(f["count"]) for f in families], dtype=torch.float32)
    family_probs = counts / counts.sum().clamp_min(1e-8)
    weighted_q = torch.zeros(len(qset), dtype=torch.float32)
    top_q_counts = torch.zeros(len(qset), dtype=torch.float32)
    for family, count in zip(families, counts):
        q_mass = family["prototype_q_mass"]
        if not torch.is_tensor(q_mass):
            q_mass = torch.tensor(q_mass, dtype=torch.float32)
        q_mass = q_mass.detach().cpu().float()
        q_mass = q_mass / q_mass.sum().clamp_min(1e-8)
        weighted_q = weighted_q + count * q_mass
        top_q_counts[int(q_mass.argmax().item())] += count
    weighted_q = weighted_q / weighted_q.sum().clamp_min(1e-8)
    top_q_probs = top_q_counts / top_q_counts.sum().clamp_min(1e-8)
    return {
        "aggregate_num_law_packets": float(library.get("num_law_packets", 0)),
        "aggregate_num_law_families": float(library.get("num_families", 0)),
        "aggregate_mean_law_family_size": float(library.get("mean_family_size", 0.0)),
        "aggregate_dominant_law_family_share": float(family_probs.max().item()),
        "aggregate_law_family_entropy": _normalized_entropy(family_probs),
        "aggregate_dominant_law_q_share": float(weighted_q.max().item()),
        "aggregate_law_top_q_entropy": _normalized_entropy(top_q_probs),
        "aggregate_num_law_top_q_unique": float((top_q_counts > 0).float().sum().item()),
    }


def _signature_library_stats(library: dict[str, Any]) -> dict[str, float]:
    families = list(library.get("families", []))
    if not families:
        return {
            "aggregate_num_relational_signatures": 0.0,
            "aggregate_num_relational_signature_families": 0.0,
            "aggregate_mean_relational_family_size": 0.0,
            "aggregate_dominant_relational_family_share": 1.0,
            "aggregate_relational_signature_confidence": 0.0,
            "aggregate_relational_branch_mass": 0.0,
            "aggregate_relational_q_entropy": 0.0,
        }
    counts = torch.tensor([float(f["count"]) for f in families], dtype=torch.float32)
    family_probs = counts / counts.sum().clamp_min(1e-8)
    confs = torch.tensor([float(f.get("mean_confidence", 0.0)) for f in families], dtype=torch.float32)
    branch_masses = torch.tensor([float(f.get("mean_branch_mass", 0.0)) for f in families], dtype=torch.float32)
    q_entropies = torch.tensor([float(f.get("mean_q_entropy", 0.0)) for f in families], dtype=torch.float32)
    weight = counts / counts.sum().clamp_min(1e-8)
    return {
        "aggregate_num_relational_signatures": float(library.get("num_signatures", 0)),
        "aggregate_num_relational_signature_families": float(library.get("num_families", 0)),
        "aggregate_mean_relational_family_size": float(library.get("mean_family_size", 0.0)),
        "aggregate_dominant_relational_family_share": float(family_probs.max().item()),
        "aggregate_relational_signature_confidence": float((weight * confs).sum().item()),
        "aggregate_relational_branch_mass": float((weight * branch_masses).sum().item()),
        "aggregate_relational_q_entropy": float((weight * q_entropies).sum().item()),
    }


def _transfer_probe_mean(rows: list[dict[str, Any]], key: str) -> float:
    if not rows:
        return 0.0
    return float(sum(float(r.get(key, 0.0)) for r in rows) / len(rows))


def _aggregate_transfer_probe_runs(runs: list[dict[str, Any]]) -> dict[str, Any]:
    if not runs:
        return {"enabled": False, "score": 0.0, "rows": [], "by_source": {}, "num_repeats": 0}

    scalar_keys = [
        "score",
        "min_branch",
        "min_phase_only_branch",
        "min_phase_only_distinctness",
        "min_meso",
        "min_writeback",
        "min_parent_div",
        "min_live_child",
        "branch_gap",
        "phase_only_branch_gap",
        "phase_only_distinctness_gap",
        "meso_gap",
        "writeback_gap",
        "parent_div_gap",
        "law_family_gap",
        "law_entropy_gap",
        "naked_branch",
        "naked_phase_only_branch",
        "naked_phase_only_distinctness",
        "naked_meso",
        "naked_writeback",
        "naked_parent_div",
        "naked_live_child",
        "naked_child_world_count",
        "naked_branch_defect",
        "naked_defect_without_branch",
        "synth_branch_defect",
        "synth_defect_without_branch",
        "synth_phase_only_branch",
        "synth_phase_only_distinctness",
        "naked_law_families",
        "naked_law_entropy",
        "synth_law_families",
        "synth_law_entropy",
        "pressure_floor_ratio",
        "pressure_weighted_ratio",
        "pressure_weighted_shortfall_ratio",
        "pressure_branch_first_score",
        "pressure_branch_live_coupling",
        "pressure_writeback_parent_coupling",
        "naked_pressure_floor_ratio",
        "naked_pressure_weighted_ratio",
        "naked_pressure_weighted_shortfall_ratio",
        "naked_pressure_branch_first_score",
        "naked_pressure_branch_live_coupling",
        "naked_pressure_writeback_parent_coupling",
    ]
    aggregated: dict[str, Any] = {
        "enabled": bool(runs[0].get("enabled", False)),
        "rows": list(runs[-1].get("rows", [])),
        "num_repeats": len(runs),
        "repeat_scores": [float(run.get("score", 0.0)) for run in runs],
        "by_source": {},
        "repeats": [
            {
                "score": float(run.get("score", 0.0)),
                "naked_branch": float(run.get("naked_branch", 0.0)),
                "naked_phase_only_branch": float(run.get("naked_phase_only_branch", 0.0)),
                "naked_phase_only_distinctness": float(run.get("naked_phase_only_distinctness", 0.0)),
                "naked_writeback": float(run.get("naked_writeback", 0.0)),
                "naked_parent_div": float(run.get("naked_parent_div", 0.0)),
                "naked_live_child": float(run.get("naked_live_child", 0.0)),
                "naked_law_families": float(run.get("naked_law_families", 0.0)),
                "naked_law_entropy": float(run.get("naked_law_entropy", 0.0)),
            }
            for run in runs
        ],
    }
    for key in scalar_keys:
        values = [float(run.get(key, 0.0)) for run in runs]
        aggregated[key] = float(sum(values) / len(values))
        aggregated[f"{key}_min"] = float(min(values))
        aggregated[f"{key}_max"] = float(max(values))
        aggregated[f"{key}_spread"] = float(max(values) - min(values))

    source_names = sorted({str(src) for run in runs for src in run.get("by_source", {}).keys()})
    for source in source_names:
        entries = [run.get("by_source", {}).get(source, {}) for run in runs]
        metric_names = sorted({str(k) for entry in entries for k in entry.keys()})
        stats: dict[str, float] = {}
        for metric in metric_names:
            values = [float(entry.get(metric, 0.0)) for entry in entries]
            stats[metric] = float(sum(values) / len(values))
            stats[f"{metric}_min"] = float(min(values))
            stats[f"{metric}_max"] = float(max(values))
            stats[f"{metric}_spread"] = float(max(values) - min(values))
        aggregated["by_source"][source] = stats
    return aggregated


def _score_cfg_weight(
    score_cfg: dict[str, float],
    key: str,
    *fallback_keys: str,
    scale: float = 1.0,
    default: float = 0.0,
) -> float:
    if key in score_cfg:
        return float(score_cfg.get(key, 0.0))
    values = [abs(float(score_cfg.get(name, 0.0))) for name in fallback_keys if name in score_cfg]
    if values:
        return float(scale * sum(values) / len(values))
    return float(default)


def _pressure_profile(
    *,
    branch: float,
    live_child: float,
    writeback: float,
    parent_div: float,
    branch_target: float,
    live_child_target: float,
    writeback_target: float,
    parent_div_target: float,
) -> dict[str, Any]:
    metric_rows = {
        "branch": (float(branch), float(branch_target), 1.75),
        "live_child": (float(live_child), float(live_child_target), 1.0),
        "writeback": (float(writeback), float(writeback_target), 1.0),
        "parent_div": (float(parent_div), float(parent_div_target), 1.0),
    }
    components: dict[str, dict[str, float]] = {}
    floor_ratio = float("inf")
    floor_margin = float("inf")
    weighted_ratio_sum = 0.0
    weighted_shortfall_sum = 0.0
    weighted_deficit_sum = 0.0
    weight_total = 0.0
    met_target = 0.0
    clipped_ratios: list[float] = []
    raw_ratios: dict[str, float] = {}

    for name, (value, target, weight) in metric_rows.items():
        gap = float(value - target)
        if target > 1e-8:
            ratio = float(value / target)
            shortfall_ratio = float(max(0.0, 1.0 - ratio))
        else:
            ratio = 1.0 if value > 0.0 else 0.0
            shortfall_ratio = 0.0
        clipped_ratio = float(max(0.0, min(1.5, ratio)))
        deficit = float(max(0.0, -gap))
        components[name] = {
            "value": float(value),
            "target": float(target),
            "gap": gap,
            "ratio": ratio,
            "clipped_ratio": clipped_ratio,
            "shortfall_ratio": shortfall_ratio,
            "deficit": deficit,
            "weight": float(weight),
        }
        raw_ratios[name] = ratio
        clipped_ratios.append(clipped_ratio)
        floor_ratio = min(floor_ratio, ratio)
        floor_margin = min(floor_margin, gap)
        weighted_ratio_sum += float(weight) * clipped_ratio
        weighted_shortfall_sum += float(weight) * shortfall_ratio
        weighted_deficit_sum += float(weight) * deficit
        weight_total += float(weight)
        met_target += 1.0 if gap >= 0.0 else 0.0

    if floor_ratio == float("inf"):
        floor_ratio = 0.0
    if floor_margin == float("inf"):
        floor_margin = 0.0
    denom = max(weight_total, 1e-8)
    weighted_ratio = float(weighted_ratio_sum / denom)
    weighted_shortfall_ratio = float(weighted_shortfall_sum / denom)
    weighted_deficit = float(weighted_deficit_sum / denom)
    mean_ratio = float(sum(clipped_ratios) / len(clipped_ratios)) if clipped_ratios else 0.0
    met_target_fraction = float(met_target / len(metric_rows)) if metric_rows else 0.0
    branch_live_coupling = float(min(components["branch"]["clipped_ratio"], components["live_child"]["clipped_ratio"]))
    branch_writeback_coupling = float(min(components["branch"]["clipped_ratio"], components["writeback"]["clipped_ratio"]))
    survival_writeback_coupling = float(min(components["live_child"]["clipped_ratio"], components["writeback"]["clipped_ratio"]))
    writeback_parent_coupling = float(min(components["writeback"]["clipped_ratio"], components["parent_div"]["clipped_ratio"]))
    branch_first_score = float(
        0.40 * components["branch"]["clipped_ratio"]
        + 0.20 * components["live_child"]["clipped_ratio"]
        + 0.20 * components["writeback"]["clipped_ratio"]
        + 0.20 * components["parent_div"]["clipped_ratio"]
        - weighted_shortfall_ratio
    )
    return {
        "components": components,
        "raw_ratios": raw_ratios,
        "floor_ratio": float(floor_ratio),
        "floor_margin": float(floor_margin),
        "mean_ratio": mean_ratio,
        "weighted_ratio": weighted_ratio,
        "weighted_shortfall_ratio": weighted_shortfall_ratio,
        "weighted_deficit": weighted_deficit,
        "met_target_fraction": met_target_fraction,
        "branch_live_coupling": branch_live_coupling,
        "branch_writeback_coupling": branch_writeback_coupling,
        "survival_writeback_coupling": survival_writeback_coupling,
        "writeback_parent_coupling": writeback_parent_coupling,
        "branch_first_score": branch_first_score,
    }


def _metric_floor(rows: list[dict[str, Any]], key: str, source: str | None = None) -> float:
    subset = rows if source is None else [row for row in rows if str(row.get("source", "")) == source]
    if not subset:
        return 0.0
    return float(min(float(row.get(key, 0.0)) for row in subset))


def _heldout_pressure_metrics(
    heldout_eval: dict[str, Any],
    probe_cfg: dict[str, Any] | None,
) -> dict[str, Any]:
    probe_cfg = dict(probe_cfg or {})
    rows = list(heldout_eval.get("rows", []))
    metrics = {
        "real_branch_floor": _metric_floor(rows, "real_branch_fraction"),
        "live_child_floor": _metric_floor(rows, "mean_live_child_fraction"),
        "child_writeback_floor": _metric_floor(rows, "mean_child_writeback_mass"),
        "child_parent_divergence_floor": _metric_floor(rows, "mean_child_parent_divergence"),
        "naked_real_branch_floor": _metric_floor(rows, "real_branch_fraction", source="naked_rafa"),
        "naked_live_child_floor": _metric_floor(rows, "mean_live_child_fraction", source="naked_rafa"),
        "naked_child_writeback_floor": _metric_floor(rows, "mean_child_writeback_mass", source="naked_rafa"),
        "naked_child_parent_divergence_floor": _metric_floor(rows, "mean_child_parent_divergence", source="naked_rafa"),
        "synthetic_real_branch_floor": _metric_floor(rows, "real_branch_fraction", source="synthetic"),
        "synthetic_live_child_floor": _metric_floor(rows, "mean_live_child_fraction", source="synthetic"),
        "synthetic_child_writeback_floor": _metric_floor(rows, "mean_child_writeback_mass", source="synthetic"),
        "synthetic_child_parent_divergence_floor": _metric_floor(rows, "mean_child_parent_divergence", source="synthetic"),
    }
    metrics["real_branch_floor_gap"] = float(max(0.0, float(heldout_eval.get("mean_real_branch_fraction", 0.0)) - metrics["real_branch_floor"]))
    metrics["live_child_floor_gap"] = float(max(0.0, float(heldout_eval.get("mean_live_child_fraction", 0.0)) - metrics["live_child_floor"]))
    metrics["child_writeback_floor_gap"] = float(max(0.0, float(heldout_eval.get("mean_child_writeback_mass", 0.0)) - metrics["child_writeback_floor"]))
    metrics["child_parent_divergence_floor_gap"] = float(
        max(0.0, float(heldout_eval.get("mean_child_parent_divergence", 0.0)) - metrics["child_parent_divergence_floor"])
    )
    metrics["naked_real_branch_floor_gap"] = float(
        max(0.0, float(dict(heldout_eval.get("by_source", {}).get("naked_rafa", {})).get("mean_real_branch_fraction", 0.0)) - metrics["naked_real_branch_floor"])
    )
    metrics["naked_live_child_floor_gap"] = float(
        max(0.0, float(dict(heldout_eval.get("by_source", {}).get("naked_rafa", {})).get("mean_live_child_fraction", 0.0)) - metrics["naked_live_child_floor"])
    )
    metrics["naked_child_writeback_floor_gap"] = float(
        max(0.0, float(dict(heldout_eval.get("by_source", {}).get("naked_rafa", {})).get("mean_child_writeback_mass", 0.0)) - metrics["naked_child_writeback_floor"])
    )
    metrics["naked_child_parent_divergence_floor_gap"] = float(
        max(
            0.0,
            float(dict(heldout_eval.get("by_source", {}).get("naked_rafa", {})).get("mean_child_parent_divergence", 0.0))
            - metrics["naked_child_parent_divergence_floor"],
        )
    )
    pressure = _pressure_profile(
        branch=metrics["real_branch_floor"],
        live_child=metrics["live_child_floor"],
        writeback=metrics["child_writeback_floor"],
        parent_div=metrics["child_parent_divergence_floor"],
        branch_target=float(probe_cfg.get("min_branch_target", 0.0)),
        live_child_target=float(probe_cfg.get("min_live_child_target", 0.0)),
        writeback_target=float(probe_cfg.get("min_writeback_target", 0.0)),
        parent_div_target=float(probe_cfg.get("min_parent_div_target", 0.0)),
    )
    naked_pressure = _pressure_profile(
        branch=metrics["naked_real_branch_floor"],
        live_child=metrics["naked_live_child_floor"],
        writeback=metrics["naked_child_writeback_floor"],
        parent_div=metrics["naked_child_parent_divergence_floor"],
        branch_target=float(probe_cfg.get("min_naked_branch_target", probe_cfg.get("min_branch_target", 0.0))),
        live_child_target=float(probe_cfg.get("min_naked_live_child_target", probe_cfg.get("min_live_child_target", 0.0))),
        writeback_target=float(probe_cfg.get("min_naked_writeback_target", probe_cfg.get("min_writeback_target", 0.0))),
        parent_div_target=float(probe_cfg.get("min_naked_parent_div_target", probe_cfg.get("min_parent_div_target", 0.0))),
    )
    metrics.update(
        {
            "pressure_metrics": pressure,
            "naked_pressure_metrics": naked_pressure,
            "pressure_floor_ratio": float(pressure["floor_ratio"]),
            "pressure_weighted_ratio": float(pressure["weighted_ratio"]),
            "pressure_weighted_shortfall_ratio": float(pressure["weighted_shortfall_ratio"]),
            "pressure_branch_first_score": float(pressure["branch_first_score"]),
            "pressure_branch_live_coupling": float(pressure["branch_live_coupling"]),
            "pressure_writeback_parent_coupling": float(pressure["writeback_parent_coupling"]),
            "naked_pressure_floor_ratio": float(naked_pressure["floor_ratio"]),
            "naked_pressure_weighted_ratio": float(naked_pressure["weighted_ratio"]),
            "naked_pressure_weighted_shortfall_ratio": float(naked_pressure["weighted_shortfall_ratio"]),
            "naked_pressure_branch_first_score": float(naked_pressure["branch_first_score"]),
            "naked_pressure_branch_live_coupling": float(naked_pressure["branch_live_coupling"]),
            "naked_pressure_writeback_parent_coupling": float(naked_pressure["writeback_parent_coupling"]),
        }
    )
    return metrics


def _selection_gate_status(
    transfer_probe: dict[str, Any],
    probe_cfg: dict[str, Any] | None,
    score_cfg: dict[str, float],
) -> dict[str, Any]:
    probe_cfg = dict(probe_cfg or {})
    enabled = bool(score_cfg.get("use_hard_selection_gate", 0.0))
    if not enabled:
        return {
            "enabled": False,
            "passed": True,
            "margin": 0.0,
            "failed_checks": [],
            "checks": {},
        }

    checks = {
        "min_branch": (
            float(transfer_probe.get("min_branch_min", transfer_probe.get("min_branch", 0.0))),
            float(probe_cfg.get("min_branch_target", 0.0)),
        ),
        "min_phase_only_branch": (
            float(transfer_probe.get("min_phase_only_branch_min", transfer_probe.get("min_phase_only_branch", 0.0))),
            float(probe_cfg.get("min_phase_only_branch_target", 0.0)),
        ),
        "min_phase_only_distinctness": (
            float(transfer_probe.get("min_phase_only_distinctness_min", transfer_probe.get("min_phase_only_distinctness", 0.0))),
            float(probe_cfg.get("min_phase_only_distinctness_target", 0.0)),
        ),
        "min_meso": (
            float(transfer_probe.get("min_meso_min", transfer_probe.get("min_meso", 0.0))),
            float(probe_cfg.get("min_meso_target", 0.0)),
        ),
        "min_writeback": (
            float(transfer_probe.get("min_writeback_min", transfer_probe.get("min_writeback", 0.0))),
            float(probe_cfg.get("min_writeback_target", 0.0)),
        ),
        "min_parent_div": (
            float(transfer_probe.get("min_parent_div_min", transfer_probe.get("min_parent_div", 0.0))),
            float(probe_cfg.get("min_parent_div_target", 0.0)),
        ),
        "min_live_child": (
            float(transfer_probe.get("min_live_child_min", transfer_probe.get("min_live_child", 0.0))),
            float(probe_cfg.get("min_live_child_target", 0.0)),
        ),
        "naked_branch": (
            float(transfer_probe.get("naked_branch_min", transfer_probe.get("naked_branch", 0.0))),
            float(probe_cfg.get("min_naked_branch_target", probe_cfg.get("min_branch_target", 0.0))),
        ),
        "naked_phase_only_branch": (
            float(transfer_probe.get("naked_phase_only_branch_min", transfer_probe.get("naked_phase_only_branch", 0.0))),
            float(probe_cfg.get("min_naked_phase_only_branch_target", probe_cfg.get("min_phase_only_branch_target", 0.0))),
        ),
        "naked_phase_only_distinctness": (
            float(transfer_probe.get("naked_phase_only_distinctness_min", transfer_probe.get("naked_phase_only_distinctness", 0.0))),
            float(probe_cfg.get("min_naked_phase_only_distinctness_target", probe_cfg.get("min_phase_only_distinctness_target", 0.0))),
        ),
        "naked_meso": (
            float(transfer_probe.get("naked_meso_min", transfer_probe.get("naked_meso", 0.0))),
            float(probe_cfg.get("min_naked_meso_target", probe_cfg.get("min_meso_target", 0.0))),
        ),
        "naked_writeback": (
            float(transfer_probe.get("naked_writeback_min", transfer_probe.get("naked_writeback", 0.0))),
            float(probe_cfg.get("min_naked_writeback_target", probe_cfg.get("min_writeback_target", 0.0))),
        ),
        "naked_parent_div": (
            float(transfer_probe.get("naked_parent_div_min", transfer_probe.get("naked_parent_div", 0.0))),
            float(probe_cfg.get("min_naked_parent_div_target", probe_cfg.get("min_parent_div_target", 0.0))),
        ),
        "naked_live_child": (
            float(transfer_probe.get("naked_live_child_min", transfer_probe.get("naked_live_child", 0.0))),
            float(probe_cfg.get("min_naked_live_child_target", probe_cfg.get("min_live_child_target", 0.0))),
        ),
        "naked_law_families": (
            float(transfer_probe.get("naked_law_families_min", transfer_probe.get("naked_law_families", 0.0))),
            float(probe_cfg.get("min_naked_law_family_target", probe_cfg.get("min_law_family_target", 0.0))),
        ),
        "naked_law_entropy": (
            float(transfer_probe.get("naked_law_entropy_min", transfer_probe.get("naked_law_entropy", 0.0))),
            float(probe_cfg.get("min_naked_law_entropy_target", probe_cfg.get("min_law_entropy_target", 0.0))),
        ),
    }
    optional_checks = {
        "pressure_floor_ratio": (
            float(transfer_probe.get("pressure_floor_ratio", 0.0)),
            probe_cfg.get("min_pressure_floor_ratio_target"),
        ),
        "naked_pressure_floor_ratio": (
            float(transfer_probe.get("naked_pressure_floor_ratio", 0.0)),
            probe_cfg.get("min_naked_pressure_floor_ratio_target"),
        ),
        "pressure_branch_live_coupling": (
            float(transfer_probe.get("pressure_branch_live_coupling", 0.0)),
            probe_cfg.get("min_pressure_branch_live_coupling_target"),
        ),
        "naked_pressure_branch_live_coupling": (
            float(transfer_probe.get("naked_pressure_branch_live_coupling", 0.0)),
            probe_cfg.get("min_naked_pressure_branch_live_coupling_target"),
        ),
        "pressure_writeback_parent_coupling": (
            float(transfer_probe.get("pressure_writeback_parent_coupling", 0.0)),
            probe_cfg.get("min_pressure_writeback_parent_coupling_target"),
        ),
        "naked_pressure_writeback_parent_coupling": (
            float(transfer_probe.get("naked_pressure_writeback_parent_coupling", 0.0)),
            probe_cfg.get("min_naked_pressure_writeback_parent_coupling_target"),
        ),
    }
    for name, (value, target) in optional_checks.items():
        if target is not None:
            checks[name] = (value, float(target))

    failed_checks: list[str] = []
    margin = float("inf")
    check_rows: dict[str, dict[str, float]] = {}
    for name, (value, target) in checks.items():
        delta = float(value - target)
        check_rows[name] = {"value": value, "target": target, "delta": delta}
        if target > 0.0:
            margin = min(margin, delta)
            if delta < 0.0:
                failed_checks.append(name)
    if margin == float("inf"):
        margin = 0.0

    return {
        "enabled": True,
        "passed": len(failed_checks) == 0,
        "margin": margin,
        "failed_checks": failed_checks,
        "checks": check_rows,
    }


def _selection_sort_key(row: dict[str, Any]) -> tuple[float, ...]:
    gate = row.get("selection_gate", {})
    transfer_probe = row.get("transfer_probe", {})
    nested_adjusted = float(row["val"].get("mean_score_with_nested_probe", row["val"].get("mean_score_with_transfer", row["val"]["mean_score"])))
    nested_score = float(row.get("nested_probe_score", 0.0))
    if bool(row.get("selection_prefer_nested_probe", False)):
        nested_enabled = 1.0 if row.get("nested_probe", {}).get("enabled", False) else 0.0
        return (
            nested_enabled,
            nested_score,
            nested_adjusted,
            1.0 if gate.get("passed", True) else 0.0,
            float(transfer_probe.get("score_min", transfer_probe.get("score", 0.0))),
            float(row["val"]["mean_score"]),
            float(row["train"]["mean_score"]),
        )
    return (
        1.0 if gate.get("passed", True) else 0.0,
        float(transfer_probe.get("naked_pressure_branch_first_score", 0.0)),
        float(transfer_probe.get("naked_pressure_floor_ratio", 0.0)),
        float(transfer_probe.get("pressure_branch_first_score", 0.0)),
        float(transfer_probe.get("pressure_floor_ratio", 0.0)),
        float(transfer_probe.get("naked_pressure_branch_live_coupling", 0.0)),
        float(transfer_probe.get("naked_pressure_writeback_parent_coupling", 0.0)),
        float(transfer_probe.get("naked_branch", 0.0)),
        float(transfer_probe.get("naked_live_child", 0.0)),
        float(transfer_probe.get("naked_writeback", 0.0)),
        float(transfer_probe.get("naked_parent_div", 0.0)),
        float(gate.get("margin", 0.0)),
        float(transfer_probe.get("score_min", transfer_probe.get("score", 0.0))),
        float(row.get("nested_probe_score", 0.0)),
        nested_adjusted,
        float(row["train"]["mean_score"]),
    )


def _selection_candidate_key(row: dict[str, Any]) -> str:
    return json.dumps(row.get("materialized_cfg", {}), sort_keys=True)


def _jsonify_cfg_value(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple):
        return [_jsonify_cfg_value(v) for v in value]
    if isinstance(value, list):
        return [_jsonify_cfg_value(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _jsonify_cfg_value(v) for k, v in value.items()}
    return value


def _cfg_payload(circle_cfg: Any) -> dict[str, Any]:
    return {"config": {str(k): _jsonify_cfg_value(v) for k, v in vars(circle_cfg).items()}}


def _default_nested_cases() -> list[dict[str, str]]:
    return [
        {
            "name": "airplane_takeoff",
            "reference_wav": str(
                ROOT / "outputs" / "circleworld_proto" / "inference_2026-04-09_realanchor_constrained_ab" / "airplane_takeoff_anchor_reference_10s.wav"
            ),
        },
        {
            "name": "sax_like_clarinet",
            "reference_wav": str(
                ROOT / "outputs" / "circleworld_proto" / "inference_2026-04-09_realanchor_constrained_ab" / "sax_like_clarinet_anchor_reference_10s.wav"
            ),
        },
    ]


def _seeded_nested_case_suite(suite_name: str = "default") -> list[dict[str, Any]]:
    normalized = str(suite_name or "default").strip().lower()
    if normalized in {"default", "seeded_branch_active"}:
        return [
            {
                "name": "naked_seed_9100",
                "seed_source": "naked_rafa",
                "seed": 9100,
                "time_steps": 128,
                "warmup_depth": 5,
                "continuation_depth": 3,
                "require_live_child_start": True,
            },
            {
                "name": "naked_seed_9101",
                "seed_source": "naked_rafa",
                "seed": 9101,
                "time_steps": 128,
                "warmup_depth": 5,
                "continuation_depth": 3,
                "require_live_child_start": True,
            },
            {
                "name": "naked_seed_9102",
                "seed_source": "naked_rafa",
                "seed": 9102,
                "time_steps": 128,
                "warmup_depth": 5,
                "continuation_depth": 3,
                "require_live_child_start": True,
            },
        ]
    if normalized in {"branch_recovery_soft_guard_v1", "branch_recovery_soft_guard"}:
        return [
            {
                "name": "naked_seed_9200",
                "seed_source": "naked_rafa",
                "seed": 9200,
                "time_steps": 128,
                "warmup_depth": 5,
                "continuation_depth": 3,
                "require_live_child_start": True,
            },
            {
                "name": "naked_seed_9201",
                "seed_source": "naked_rafa",
                "seed": 9201,
                "time_steps": 128,
                "warmup_depth": 5,
                "continuation_depth": 3,
                "require_live_child_start": True,
            },
            {
                "name": "naked_seed_9202",
                "seed_source": "naked_rafa",
                "seed": 9202,
                "time_steps": 128,
                "warmup_depth": 5,
                "continuation_depth": 3,
                "require_live_child_start": True,
            },
        ]
    raise ValueError(f"Unknown selection_nested_case_suite={suite_name!r}")


def _default_seeded_nested_cases() -> list[dict[str, Any]]:
    return _seeded_nested_case_suite("default")


def _load_nested_cases(probe_cfg: dict[str, Any] | None) -> list[dict[str, Any]]:
    probe_cfg = dict(probe_cfg or {})
    inline_plan = probe_cfg.get("selection_nested_case_plan")
    if inline_plan:
        if isinstance(inline_plan, str):
            return list(json.loads(inline_plan))
        return list(inline_plan)
    case_json = probe_cfg.get("selection_nested_cases_json")
    if case_json:
        raw_cases = json.loads(Path(case_json).read_text(encoding="utf-8-sig"))
        if isinstance(raw_cases, dict):
            return [{"name": str(name), "reference_wav": str(path)} for name, path in raw_cases.items()]
        return list(raw_cases)
    case_suite = probe_cfg.get("selection_nested_case_suite")
    if case_suite:
        return _seeded_nested_case_suite(str(case_suite))
    case_source = str(probe_cfg.get("selection_nested_case_source", "reference_audio"))
    if case_source == "seeded_branch_active":
        return _default_seeded_nested_cases()
    if case_source == "reference_audio":
        return _default_nested_cases()
    raise ValueError(f"Unknown selection_nested_case_source={case_source!r}")


def _first_float(source: dict[str, Any], keys: tuple[str, ...]) -> float | None:
    for key in keys:
        if key in source:
            return float(source.get(key, 0.0))
    return None


def _nested_family_sibling_fraction(nested_eval: dict[str, Any], family: str) -> float:
    family_key = family.replace("-", "_")
    direct = _first_float(
        nested_eval,
        (
            f"mean_assay_{family_key}_nested_sibling_fraction",
            f"assay_{family_key}_nested_sibling_fraction",
            f"mean_{family_key}_nested_sibling_fraction",
            f"{family_key}_nested_sibling_fraction",
            f"mean_nested_{family_key}_sibling_fraction",
            f"nested_{family_key}_sibling_fraction",
        ),
    )
    if direct is not None:
        return direct

    family_breakdown = nested_eval.get("family_breakdown", {})
    if isinstance(family_breakdown, dict):
        row = family_breakdown.get(family_key) or family_breakdown.get(family_key.replace("_", "-"))
        if isinstance(row, dict):
            value = _first_float(row, ("nested_sibling_fraction", "mean_nested_sibling_fraction"))
            if value is not None:
                return value

    case_values: list[float] = []
    for case in nested_eval.get("cases", []) or []:
        if not isinstance(case, dict):
            continue
        case_response = case.get("case_nested_response", {})
        if not isinstance(case_response, dict):
            continue
        family_breakdown = case_response.get("family_breakdown", {})
        if not isinstance(family_breakdown, dict):
            continue
        row = family_breakdown.get(family_key) or family_breakdown.get(family_key.replace("_", "-"))
        if not isinstance(row, dict):
            continue
        value = _first_float(row, ("nested_sibling_fraction", "mean_nested_sibling_fraction"))
        if value is not None:
            case_values.append(value)
    if case_values:
        return float(sum(case_values) / len(case_values))
    return 0.0


def _nested_family_metric(nested_eval: dict[str, Any], family: str, metric: str) -> float:
    family_key = family.replace("-", "_")
    metric_key = metric.replace("-", "_")
    direct = _first_float(
        nested_eval,
        (
            f"mean_assay_{family_key}_{metric_key}",
            f"assay_{family_key}_{metric_key}",
            f"mean_{family_key}_{metric_key}",
            f"{family_key}_{metric_key}",
        ),
    )
    if direct is not None:
        return direct

    family_breakdown = nested_eval.get("family_breakdown", {})
    if isinstance(family_breakdown, dict):
        row = family_breakdown.get(family_key) or family_breakdown.get(family_key.replace("_", "-"))
        if isinstance(row, dict):
            value = _first_float(row, (metric_key, f"mean_{metric_key}"))
            if value is not None:
                return value

    case_values: list[float] = []
    for case in nested_eval.get("cases", []) or []:
        if not isinstance(case, dict):
            continue
        case_response = case.get("case_nested_response", {})
        if not isinstance(case_response, dict):
            continue
        value = _first_float(
            case_response,
            (
                f"assay_{family_key}_mean_{metric_key}",
                f"assay_{family_key}_{metric_key}",
                f"mean_assay_{family_key}_{metric_key}",
            ),
        )
        if value is not None:
            case_values.append(value)
            continue
        family_breakdown = case_response.get("family_breakdown", {})
        if not isinstance(family_breakdown, dict):
            continue
        row = family_breakdown.get(family_key) or family_breakdown.get(family_key.replace("_", "-"))
        if not isinstance(row, dict):
            continue
        value = _first_float(row, (metric_key, f"mean_{metric_key}"))
        if value is not None:
            case_values.append(value)
    if case_values:
        return float(sum(case_values) / len(case_values))
    return 0.0


def _nested_readout_without_continuation_penalty(
    nested_eval: dict[str, Any],
    readout_sibling: float,
    continuation_sibling: float,
    mode_replace_sibling: float = 0.0,
) -> float:
    direct = _first_float(
        nested_eval,
        (
            "mean_readout_without_continuation_nested_sibling_fraction",
            "readout_without_continuation_nested_sibling_fraction",
            "mean_readout_only_without_continuation_fraction",
            "readout_only_without_continuation_fraction",
            "mean_readout_only_without_continuation_success_fraction",
            "readout_only_without_continuation_success_fraction",
        ),
    )
    if direct is not None:
        return direct

    case_values: list[float] = []
    for case in nested_eval.get("cases", []) or []:
        if not isinstance(case, dict):
            continue
        case_response = case.get("case_nested_response", {})
        if not isinstance(case_response, dict):
            continue
        readout = _nested_family_sibling_fraction(case_response, "readout")
        continuation = _nested_family_sibling_fraction(case_response, "continuation")
        mode_replace = _nested_family_sibling_fraction(case_response, "mode_replace")
        causal_sibling = max(continuation, mode_replace)
        case_values.append(max(0.0, readout - causal_sibling))
    if case_values:
        return float(sum(case_values) / len(case_values))
    causal_sibling = max(continuation_sibling, mode_replace_sibling)
    return float(max(0.0, readout_sibling - causal_sibling))


def _nested_probe_score(
    nested_eval: dict[str, Any],
    score_cfg: dict[str, float],
    probe_cfg: dict[str, Any] | None = None,
) -> float:
    probe_cfg = dict(probe_cfg or {})
    score = 0.0
    response = float(nested_eval.get("mean_child_response_score", 0.0))
    active = float(nested_eval.get("mean_child_active_fraction", 0.0))
    meso = float(nested_eval.get("mean_child_meso_response", 0.0))
    sibling = float(nested_eval.get("mean_nested_sibling_fraction", 0.0))
    readout_sibling = _nested_family_sibling_fraction(nested_eval, "readout")
    continuation_sibling = _nested_family_sibling_fraction(nested_eval, "continuation")
    mode_replace_sibling = _nested_family_sibling_fraction(nested_eval, "mode_replace")
    causal_sibling = max(continuation_sibling, mode_replace_sibling)
    continuation_budget = _nested_family_metric(nested_eval, "continuation", "branch_identity_budget_retained")
    continuation_response = _nested_family_metric(nested_eval, "continuation", "readout_sibling_response")
    continuation_q_corr = _nested_family_metric(nested_eval, "continuation", "fine_q_profile_corr")
    continuation_write_delta = _nested_family_metric(nested_eval, "continuation", "continuation_write_delta")
    continuation_readiness = _nested_family_metric(nested_eval, "continuation", "nested_sibling_readiness")
    continuation_max_readiness = _nested_family_metric(nested_eval, "continuation", "max_nested_sibling_readiness")
    mode_replace_budget = _nested_family_metric(nested_eval, "mode_replace", "branch_identity_budget_retained")
    mode_replace_response = _nested_family_metric(nested_eval, "mode_replace", "readout_sibling_response")
    mode_replace_q_corr = _nested_family_metric(nested_eval, "mode_replace", "fine_q_profile_corr")
    mode_replace_readiness = _nested_family_metric(nested_eval, "mode_replace", "nested_sibling_readiness")
    mode_replace_max_readiness = _nested_family_metric(nested_eval, "mode_replace", "max_nested_sibling_readiness")
    readout_without_continuation = _nested_readout_without_continuation_penalty(
        nested_eval=nested_eval,
        readout_sibling=readout_sibling,
        continuation_sibling=continuation_sibling,
        mode_replace_sibling=mode_replace_sibling,
    )
    direct_readout_dependency = _first_float(
        nested_eval,
        (
            "mean_direct_readout_dependency",
            "direct_readout_dependency",
        ),
    )
    if direct_readout_dependency is None:
        direct_readout_dependency = readout_without_continuation
    final_direct_mix_shortcut = _first_float(
        nested_eval,
        (
            "mean_final_direct_mix_shortcut_score",
            "final_direct_mix_shortcut_score",
        ),
    )
    if final_direct_mix_shortcut is None:
        final_direct_mix_shortcut = max(0.0, continuation_sibling - mode_replace_sibling)
    child_record_survival_score = _first_float(
        nested_eval,
        (
            "mean_child_record_survival_score",
            "child_record_survival_score",
        ),
    )
    if child_record_survival_score is None:
        child_record_survival_score = 0.0
    parent_mode_conversion = _first_float(
        nested_eval,
        (
            "mean_parent_mode_conversion_sibling_fraction",
            "parent_mode_conversion_sibling_fraction",
        ),
    )
    if parent_mode_conversion is None:
        parent_mode_conversion = mode_replace_sibling
    mode_replace_conversion_score = _first_float(
        nested_eval,
        (
            "mean_mode_replace_conversion_score",
            "mode_replace_conversion_score",
        ),
    )
    if mode_replace_conversion_score is None:
        mode_replace_conversion_score = mode_replace_sibling * max(0.0, min(1.0, child_record_survival_score))
    coarse = float(nested_eval.get("mean_coarse_preservation", 0.0))
    identity_carry = float(nested_eval.get("mean_branch_identity_carry", 0.0))
    qualified_carry = float(nested_eval.get("mean_branch_identity_qualified_carry", 0.0))
    max_qualified_carry = float(nested_eval.get("mean_max_branch_identity_qualified_carry", nested_eval.get("max_branch_identity_qualified_carry", 0.0)))
    max_qualified_steps = float(nested_eval.get("mean_max_branch_identity_qualified_steps", nested_eval.get("max_branch_identity_qualified_steps", 0.0)))
    best_identity_support = float(nested_eval.get("mean_best_identity_support_mean", 0.0))
    best_identity_coherence = float(nested_eval.get("mean_best_identity_coherence_mean", 0.0))
    best_identity_min_support = float(nested_eval.get("mean_best_identity_min_support", 0.0))
    best_identity_min_coherence = float(nested_eval.get("mean_best_identity_min_coherence", 0.0))
    budget_retained = float(nested_eval.get("mean_branch_identity_budget_retained", 0.0))
    identity_parent_div = float(nested_eval.get("mean_branch_identity_parent_div", 0.0))
    identity_sibling_div = float(nested_eval.get("mean_branch_identity_sibling_div", 0.0))
    case_count = max(1.0, float(nested_eval.get("num_cases", 0.0) or 0.0))
    live_start_fraction = float(nested_eval.get("num_live_child_start_cases", 0.0)) / case_count
    selected_live_fraction = float(nested_eval.get("mean_selected_fork_has_live_child", 0.0))
    over_rigid = float(nested_eval.get("mean_over_rigid_fraction", 0.0))
    world_jump = float(nested_eval.get("mean_world_jump_penalty", 0.0))
    score += float(score_cfg.get("w_nested_probe_response", 0.0)) * response
    score += float(score_cfg.get("w_nested_probe_active_fraction", 0.0)) * active
    score += float(score_cfg.get("w_nested_probe_live_start_fraction", 0.0)) * live_start_fraction
    score += float(score_cfg.get("w_nested_probe_selected_live_fraction", 0.0)) * selected_live_fraction
    score += float(score_cfg.get("w_nested_probe_meso_response", 0.0)) * meso
    score += float(score_cfg.get("w_nested_probe_sibling_fraction", 0.0)) * sibling
    score += float(score_cfg.get("w_nested_probe_readout_sibling_fraction", 0.0)) * readout_sibling
    score += float(score_cfg.get("w_nested_probe_continuation_sibling_fraction", 0.0)) * continuation_sibling
    score += float(score_cfg.get("w_nested_probe_mode_replace_sibling_fraction", 0.0)) * mode_replace_sibling
    score += float(score_cfg.get("w_nested_probe_causal_sibling_fraction", 0.0)) * causal_sibling
    score += float(score_cfg.get("w_nested_probe_continuation_budget_retained", 0.0)) * continuation_budget
    score += float(score_cfg.get("w_nested_probe_continuation_readout_response", 0.0)) * continuation_response
    score += float(score_cfg.get("w_nested_probe_continuation_q_corr", 0.0)) * continuation_q_corr
    score += float(score_cfg.get("w_nested_probe_continuation_write_delta", 0.0)) * continuation_write_delta
    score += float(score_cfg.get("w_nested_probe_continuation_readiness", 0.0)) * continuation_readiness
    score += float(score_cfg.get("w_nested_probe_continuation_max_readiness", 0.0)) * continuation_max_readiness
    score += float(score_cfg.get("w_nested_probe_mode_replace_budget_retained", 0.0)) * mode_replace_budget
    score += float(score_cfg.get("w_nested_probe_mode_replace_readout_response", 0.0)) * mode_replace_response
    score += float(score_cfg.get("w_nested_probe_mode_replace_q_corr", 0.0)) * mode_replace_q_corr
    score += float(score_cfg.get("w_nested_probe_mode_replace_readiness", 0.0)) * mode_replace_readiness
    score += float(score_cfg.get("w_nested_probe_mode_replace_max_readiness", 0.0)) * mode_replace_max_readiness
    score += float(score_cfg.get("w_nested_probe_parent_mode_conversion_sibling_fraction", 0.0)) * parent_mode_conversion
    score += float(score_cfg.get("w_nested_probe_child_record_survival_score", 0.0)) * child_record_survival_score
    score += float(score_cfg.get("w_nested_probe_mode_replace_conversion_score", 0.0)) * mode_replace_conversion_score
    score += float(score_cfg.get("w_nested_probe_identity_carry", 0.0)) * identity_carry
    score += float(score_cfg.get("w_nested_probe_qualified_carry", 0.0)) * qualified_carry
    score += float(score_cfg.get("w_nested_probe_max_qualified_carry", 0.0)) * max_qualified_carry
    score += float(score_cfg.get("w_nested_probe_max_qualified_steps", 0.0)) * max_qualified_steps
    score += float(score_cfg.get("w_nested_probe_best_identity_support", 0.0)) * best_identity_support
    score += float(score_cfg.get("w_nested_probe_best_identity_coherence", 0.0)) * best_identity_coherence
    score += float(score_cfg.get("w_nested_probe_best_identity_min_support", 0.0)) * best_identity_min_support
    score += float(score_cfg.get("w_nested_probe_best_identity_min_coherence", 0.0)) * best_identity_min_coherence
    score += float(score_cfg.get("w_nested_probe_budget_retained", 0.0)) * budget_retained
    score += float(score_cfg.get("w_nested_probe_identity_parent_div", 0.0)) * identity_parent_div
    score += float(score_cfg.get("w_nested_probe_identity_sibling_div", 0.0)) * identity_sibling_div
    score += float(score_cfg.get("w_nested_probe_coarse_preservation", 0.0)) * max(
        0.0, coarse - float(score_cfg.get("target_nested_probe_coarse_preservation", 0.90))
    )
    score -= float(score_cfg.get("w_nested_probe_over_rigid", 0.0)) * over_rigid
    score -= float(score_cfg.get("w_nested_probe_world_jump", 0.0)) * world_jump
    score -= float(score_cfg.get("w_nested_probe_readout_without_continuation_penalty", 0.0)) * readout_without_continuation
    score -= float(score_cfg.get("w_nested_probe_direct_readout_dependency_penalty", 0.0)) * float(direct_readout_dependency)
    score -= float(score_cfg.get("w_nested_probe_final_direct_mix_shortcut_penalty", 0.0)) * float(final_direct_mix_shortcut)
    score -= float(score_cfg.get("w_nested_probe_continuation_budget_shortfall", 0.0)) * max(
        0.0,
        float(score_cfg.get("target_nested_probe_continuation_budget_retained", 0.70)) - continuation_budget,
    )
    score -= float(score_cfg.get("w_nested_probe_continuation_response_shortfall", 0.0)) * max(
        0.0,
        float(score_cfg.get("target_nested_probe_continuation_readout_response", 0.30)) - continuation_response,
    )
    score -= float(score_cfg.get("w_nested_probe_continuation_q_corr_shortfall", 0.0)) * max(
        0.0,
        float(score_cfg.get("target_nested_probe_continuation_q_corr", 0.995)) - continuation_q_corr,
    )
    score -= float(score_cfg.get("w_nested_probe_mode_replace_response_shortfall", 0.0)) * max(
        0.0,
        float(score_cfg.get("target_nested_probe_mode_replace_readout_response", 0.30)) - mode_replace_response,
    )
    score -= float(score_cfg.get("w_nested_probe_continuation_readiness_shortfall", 0.0)) * max(
        0.0,
        float(score_cfg.get("target_nested_probe_continuation_readiness", 1.0)) - continuation_max_readiness,
    )
    score -= float(score_cfg.get("w_nested_probe_mode_replace_readiness_shortfall", 0.0)) * max(
        0.0,
        float(score_cfg.get("target_nested_probe_mode_replace_readiness", 1.0)) - mode_replace_max_readiness,
    )
    score -= float(score_cfg.get("w_nested_probe_child_record_survival_shortfall", 0.0)) * max(
        0.0,
        float(score_cfg.get("target_nested_probe_child_record_survival_score", 1.0)) - float(child_record_survival_score),
    )
    score -= float(score_cfg.get("w_nested_probe_mode_replace_conversion_shortfall", 0.0)) * max(
        0.0,
        float(score_cfg.get("target_nested_probe_mode_replace_conversion_score", 0.10)) - float(mode_replace_conversion_score),
    )
    score -= float(score_cfg.get("w_nested_probe_response_shortfall", 0.0)) * max(
        0.0, float(probe_cfg.get("selection_nested_min_response", float("-inf"))) - response
    )
    score -= float(score_cfg.get("w_nested_probe_active_shortfall", 0.0)) * max(
        0.0, float(probe_cfg.get("selection_nested_min_active_fraction", 0.0)) - active
    )
    score -= float(score_cfg.get("w_nested_probe_live_start_shortfall", 0.0)) * max(
        0.0, float(probe_cfg.get("selection_nested_min_live_child_start_fraction", 0.0)) - live_start_fraction
    )
    score -= float(score_cfg.get("w_nested_probe_selected_live_shortfall", 0.0)) * max(
        0.0, float(probe_cfg.get("selection_nested_min_selected_live_child_fraction", 0.0)) - selected_live_fraction
    )
    score -= float(score_cfg.get("w_nested_probe_meso_shortfall", 0.0)) * max(
        0.0, float(probe_cfg.get("selection_nested_min_meso_response", 0.0)) - meso
    )
    score -= float(score_cfg.get("w_nested_probe_identity_carry_shortfall", 0.0)) * max(
        0.0, float(probe_cfg.get("selection_nested_min_identity_carry", 0.0)) - identity_carry
    )
    score -= float(score_cfg.get("w_nested_probe_qualified_carry_shortfall", 0.0)) * max(
        0.0, float(probe_cfg.get("selection_nested_min_qualified_carry", 0.0)) - qualified_carry
    )
    score -= float(score_cfg.get("w_nested_probe_max_qualified_carry_shortfall", 0.0)) * max(
        0.0, float(probe_cfg.get("selection_nested_min_max_qualified_carry", 0.0)) - max_qualified_carry
    )
    score -= float(score_cfg.get("w_nested_probe_max_qualified_steps_shortfall", 0.0)) * max(
        0.0, float(probe_cfg.get("selection_nested_min_max_qualified_steps", 0.0)) - max_qualified_steps
    )
    score -= float(score_cfg.get("w_nested_probe_best_identity_support_shortfall", 0.0)) * max(
        0.0, float(probe_cfg.get("selection_nested_min_best_identity_support", 0.0)) - best_identity_support
    )
    score -= float(score_cfg.get("w_nested_probe_best_identity_coherence_shortfall", 0.0)) * max(
        0.0, float(probe_cfg.get("selection_nested_min_best_identity_coherence", 0.0)) - best_identity_coherence
    )
    score -= float(score_cfg.get("w_nested_probe_best_identity_min_support_shortfall", 0.0)) * max(
        0.0, float(probe_cfg.get("selection_nested_min_best_identity_min_support", 0.0)) - best_identity_min_support
    )
    score -= float(score_cfg.get("w_nested_probe_best_identity_min_coherence_shortfall", 0.0)) * max(
        0.0, float(probe_cfg.get("selection_nested_min_best_identity_min_coherence", 0.0)) - best_identity_min_coherence
    )
    score -= float(score_cfg.get("w_nested_probe_budget_retained_shortfall", 0.0)) * max(
        0.0, float(probe_cfg.get("selection_nested_min_budget_retained", 0.0)) - budget_retained
    )
    score -= float(score_cfg.get("w_nested_probe_identity_parent_div_shortfall", 0.0)) * max(
        0.0, float(probe_cfg.get("selection_nested_min_identity_parent_div", 0.0)) - identity_parent_div
    )
    score -= float(score_cfg.get("w_nested_probe_identity_sibling_div_shortfall", 0.0)) * max(
        0.0, float(probe_cfg.get("selection_nested_min_identity_sibling_div", 0.0)) - identity_sibling_div
    )
    return float(score)


def _evaluate_transfer_probe(
    circle_cfg,
    device_name: str,
    score_cfg: dict[str, float],
    probe_cfg: dict[str, Any] | None = None,
) -> dict[str, Any]:
    probe_cfg = dict(probe_cfg or {})
    def shortfall(target: Any, observed: float) -> float:
        return max(0.0, float(target) - float(observed))

    if not probe_cfg.get("enabled", False):
        return {"enabled": False, "score": 0.0, "rows": [], "by_source": {}}

    device = _safe_device(device_name)
    time_steps = int(probe_cfg.get("time_steps", 128))
    probe_plan = list(
        probe_cfg.get(
            "plan",
            [
                ("synthetic", 7100),
                ("synthetic", 7101),
                ("naked_rafa", 8100),
                ("naked_rafa", 8101),
            ],
        )
    )
    rafa_core = make_seed_rafa(dev=device.type) if any(src == "naked_rafa" for src, _ in probe_plan) else None

    rows: list[dict[str, Any]] = []
    run_mode = circle_cfg.branching_mode if circle_cfg.branching_mode in {"native_multimode", "native_multimode_childworld"} else "active_packets"
    for source, seed in probe_plan:
        phase_state = _generate_seed_phase(
            rafa_core=rafa_core,
            batch_size=1,
            time_steps=time_steps,
            device=device,
            seed_source=source,
            seed=seed,
        ).detach()
        run = recurse_circleworld(
            phase_state,
            cfg=circle_cfg,
            depth=circle_cfg.recursion_depth,
            mode=run_mode,
        )
        summary = summarize_circleworld_run(run)
        rows.append({"source": source, "seed": seed, **summary})

    by_source: dict[str, dict[str, float]] = {}
    for source in sorted({str(r["source"]) for r in rows}):
        subset = [r for r in rows if r["source"] == source]
        by_source[source] = {
            "count": float(len(subset)),
            "mean_real_branch_fraction": _transfer_probe_mean(subset, "real_branch_fraction"),
            "mean_parent_real_branch_fraction": _transfer_probe_mean(subset, "parent_real_branch_fraction"),
            "mean_child_real_branch_fraction": _transfer_probe_mean(subset, "child_real_branch_fraction"),
            "mean_meso_branch_effect": _transfer_probe_mean(subset, "meso_branch_effect"),
            "phase_only_real_branch_fraction": _transfer_probe_mean(subset, "phase_only_real_branch_fraction"),
            "phase_only_excess_branch_fraction": _transfer_probe_mean(subset, "phase_only_excess_branch_fraction"),
            "decorative_slot2_low_phase_fraction": _transfer_probe_mean(subset, "decorative_slot2_low_phase_fraction"),
            "mean_phase_only_branch_distinctness": _transfer_probe_mean(subset, "mean_phase_only_branch_distinctness"),
            "mean_phase_only_branch_distinctness_balanced": _transfer_probe_mean(subset, "mean_phase_only_branch_distinctness_balanced"),
            "mean_phase_only_branch_phase_wall": _transfer_probe_mean(subset, "mean_phase_only_branch_phase_wall"),
            "mean_child_writeback_mass": _transfer_probe_mean(subset, "mean_child_writeback_mass"),
            "mean_child_local_ifs_step_count": _transfer_probe_mean(subset, "mean_child_local_ifs_step_count"),
            "mean_child_local_ifs_packet_count": _transfer_probe_mean(subset, "mean_child_local_ifs_packet_count"),
            "mean_child_local_ifs_phase_delta": _transfer_probe_mean(subset, "mean_child_local_ifs_phase_delta"),
            "mean_child_local_ifs_support": _transfer_probe_mean(subset, "mean_child_local_ifs_support"),
            "mean_child_local_ifs_coherence": _transfer_probe_mean(subset, "mean_child_local_ifs_coherence"),
            "mean_child_local_ifs_coherence_raw": _transfer_probe_mean(subset, "mean_child_local_ifs_coherence_raw"),
            "mean_child_local_ifs_coherence_retention_delta": _transfer_probe_mean(
                subset, "mean_child_local_ifs_coherence_retention_delta"
            ),
            "mean_child_local_ifs_coherence_floor_delta": _transfer_probe_mean(
                subset, "mean_child_local_ifs_coherence_floor_delta"
            ),
            "mean_child_local_ifs_parent_phase_delta": _transfer_probe_mean(
                subset, "mean_child_local_ifs_parent_phase_delta"
            ),
            "mean_child_local_ifs_causal_gate": _transfer_probe_mean(subset, "mean_child_local_ifs_causal_gate"),
            "mean_child_local_ifs_causal_retention_loss": _transfer_probe_mean(
                subset, "mean_child_local_ifs_causal_retention_loss"
            ),
            "mean_child_local_ifs_causal_floor_loss": _transfer_probe_mean(
                subset, "mean_child_local_ifs_causal_floor_loss"
            ),
            "mean_child_parent_divergence": _transfer_probe_mean(subset, "mean_child_parent_divergence"),
            "mean_child_sibling_divergence": _transfer_probe_mean(subset, "mean_child_sibling_divergence"),
            "mean_child_world_count": _transfer_probe_mean(subset, "mean_child_world_count"),
            "mean_live_child_fraction": _transfer_probe_mean(subset, "mean_live_child_fraction"),
            "mean_branch_defect": _transfer_probe_mean(subset, "mean_branch_defect"),
            "mean_defect_without_branch_penalty": _transfer_probe_mean(subset, "mean_defect_without_branch_penalty"),
            "mean_branch_resolution_delay": _transfer_probe_mean(subset, "mean_branch_resolution_delay"),
            "mean_num_law_families": _transfer_probe_mean(subset, "num_law_families"),
            "mean_law_family_entropy": _transfer_probe_mean(subset, "law_family_entropy"),
            "mean_num_law_top_q_unique": _transfer_probe_mean(subset, "num_law_top_q_unique"),
        }

    synth = by_source.get("synthetic", {})
    naked = by_source.get("naked_rafa", {})
    synth_branch = float(synth.get("mean_real_branch_fraction", 0.0))
    naked_branch = float(naked.get("mean_real_branch_fraction", 0.0))
    synth_phase_branch = float(synth.get("phase_only_real_branch_fraction", 0.0))
    naked_phase_branch = float(naked.get("phase_only_real_branch_fraction", 0.0))
    synth_phase_distinctness = float(synth.get("mean_phase_only_branch_distinctness", 0.0))
    naked_phase_distinctness = float(naked.get("mean_phase_only_branch_distinctness", 0.0))
    synth_meso = float(synth.get("mean_meso_branch_effect", 0.0))
    naked_meso = float(naked.get("mean_meso_branch_effect", 0.0))
    synth_writeback = float(synth.get("mean_child_writeback_mass", 0.0))
    naked_writeback = float(naked.get("mean_child_writeback_mass", 0.0))
    synth_div = float(synth.get("mean_child_parent_divergence", 0.0))
    naked_div = float(naked.get("mean_child_parent_divergence", 0.0))
    synth_live = float(synth.get("mean_live_child_fraction", 0.0))
    naked_live = float(naked.get("mean_live_child_fraction", 0.0))
    synth_child_worlds = float(synth.get("mean_child_world_count", 0.0))
    naked_child_worlds = float(naked.get("mean_child_world_count", 0.0))
    synth_defect = float(synth.get("mean_branch_defect", 0.0))
    naked_defect = float(naked.get("mean_branch_defect", 0.0))
    synth_defect_wo_branch = float(synth.get("mean_defect_without_branch_penalty", 0.0))
    naked_defect_wo_branch = float(naked.get("mean_defect_without_branch_penalty", 0.0))
    synth_delay = float(synth.get("mean_branch_resolution_delay", 0.0))
    naked_delay = float(naked.get("mean_branch_resolution_delay", 0.0))
    synth_law_families = float(synth.get("mean_num_law_families", 0.0))
    naked_law_families = float(naked.get("mean_num_law_families", 0.0))
    synth_law_entropy = float(synth.get("mean_law_family_entropy", 0.0))
    naked_law_entropy = float(naked.get("mean_law_family_entropy", 0.0))

    min_branch = min(synth_branch, naked_branch) if naked else synth_branch
    min_phase_branch = min(synth_phase_branch, naked_phase_branch) if naked else synth_phase_branch
    min_phase_distinctness = min(synth_phase_distinctness, naked_phase_distinctness) if naked else synth_phase_distinctness
    min_meso = min(synth_meso, naked_meso) if naked else synth_meso
    min_writeback = min(synth_writeback, naked_writeback) if naked else synth_writeback
    min_div = min(synth_div, naked_div) if naked else synth_div
    min_live = min(synth_live, naked_live) if naked else synth_live
    min_law_families = min(synth_law_families, naked_law_families) if naked else synth_law_families
    min_law_entropy = min(synth_law_entropy, naked_law_entropy) if naked else synth_law_entropy

    branch_gap = abs(synth_branch - naked_branch) if naked else 0.0
    phase_branch_gap = abs(synth_phase_branch - naked_phase_branch) if naked else 0.0
    phase_distinctness_gap = abs(synth_phase_distinctness - naked_phase_distinctness) if naked else 0.0
    meso_gap = abs(synth_meso - naked_meso) if naked else 0.0
    writeback_gap = abs(synth_writeback - naked_writeback) if naked else 0.0
    div_gap = abs(synth_div - naked_div) if naked else 0.0
    law_family_gap = abs(synth_law_families - naked_law_families) if naked else 0.0
    law_entropy_gap = abs(synth_law_entropy - naked_law_entropy) if naked else 0.0
    pressure_metrics = _pressure_profile(
        branch=min_branch,
        live_child=min_live,
        writeback=min_writeback,
        parent_div=min_div,
        branch_target=float(probe_cfg.get("min_branch_target", 0.15)),
        live_child_target=float(probe_cfg.get("min_live_child_target", 0.10)),
        writeback_target=float(probe_cfg.get("min_writeback_target", 0.02)),
        parent_div_target=float(probe_cfg.get("min_parent_div_target", 0.05)),
    )
    naked_pressure_metrics = _pressure_profile(
        branch=naked_branch,
        live_child=naked_live,
        writeback=naked_writeback,
        parent_div=naked_div,
        branch_target=float(probe_cfg.get("min_naked_branch_target", probe_cfg.get("min_branch_target", 0.15))),
        live_child_target=float(probe_cfg.get("min_naked_live_child_target", probe_cfg.get("min_live_child_target", 0.10))),
        writeback_target=float(probe_cfg.get("min_naked_writeback_target", probe_cfg.get("min_writeback_target", 0.02))),
        parent_div_target=float(probe_cfg.get("min_naked_parent_div_target", probe_cfg.get("min_parent_div_target", 0.05))),
    )

    score = 0.0
    score += float(score_cfg.get("w_transfer_min_branch", 0.0)) * min_branch
    score += float(score_cfg.get("w_transfer_min_phase_only_branch", 0.0)) * min_phase_branch
    score += float(score_cfg.get("w_transfer_min_phase_only_distinctness", 0.0)) * min_phase_distinctness
    score += float(score_cfg.get("w_transfer_min_meso", 0.0)) * min_meso
    score += float(score_cfg.get("w_transfer_min_writeback", 0.0)) * min_writeback
    score += float(score_cfg.get("w_transfer_min_parent_div", 0.0)) * min_div
    score += float(score_cfg.get("w_transfer_min_live_child", 0.0)) * min_live
    score += float(score_cfg.get("w_transfer_min_law_families", 0.0)) * min_law_families
    score += float(score_cfg.get("w_transfer_min_law_entropy", 0.0)) * min_law_entropy
    score -= float(score_cfg.get("w_transfer_branch_gap", 0.0)) * branch_gap
    score -= float(score_cfg.get("w_transfer_phase_only_branch_gap", 0.0)) * phase_branch_gap
    score -= float(score_cfg.get("w_transfer_phase_only_distinctness_gap", 0.0)) * phase_distinctness_gap
    score -= float(score_cfg.get("w_transfer_meso_gap", 0.0)) * meso_gap
    score -= float(score_cfg.get("w_transfer_writeback_gap", 0.0)) * writeback_gap
    score -= float(score_cfg.get("w_transfer_parent_div_gap", 0.0)) * div_gap
    score -= float(score_cfg.get("w_transfer_law_family_gap", 0.0)) * law_family_gap
    score -= float(score_cfg.get("w_transfer_law_entropy_gap", 0.0)) * law_entropy_gap
    score -= float(score_cfg.get("w_transfer_branch_shortfall", 0.0)) * max(0.0, float(probe_cfg.get("min_branch_target", 0.15)) - min_branch)
    score -= float(score_cfg.get("w_transfer_phase_only_branch_shortfall", 0.0)) * max(0.0, float(probe_cfg.get("min_phase_only_branch_target", 0.0)) - min_phase_branch)
    score -= float(score_cfg.get("w_transfer_phase_only_distinctness_shortfall", 0.0)) * max(0.0, float(probe_cfg.get("min_phase_only_distinctness_target", 0.0)) - min_phase_distinctness)
    score -= float(score_cfg.get("w_transfer_writeback_shortfall", 0.0)) * max(0.0, float(probe_cfg.get("min_writeback_target", 0.02)) - min_writeback)
    score -= float(score_cfg.get("w_transfer_parent_div_shortfall", 0.0)) * max(0.0, float(probe_cfg.get("min_parent_div_target", 0.05)) - min_div)
    score -= float(score_cfg.get("w_transfer_meso_shortfall", 0.0)) * max(0.0, float(probe_cfg.get("min_meso_target", 0.004)) - min_meso)
    score -= float(score_cfg.get("w_transfer_live_child_shortfall", 0.0)) * max(0.0, float(probe_cfg.get("min_live_child_target", 0.10)) - min_live)
    score -= float(score_cfg.get("w_transfer_law_family_shortfall", 0.0)) * max(0.0, float(probe_cfg.get("min_law_family_target", 1.0)) - min_law_families)
    score -= float(score_cfg.get("w_transfer_law_entropy_shortfall", 0.0)) * max(0.0, float(probe_cfg.get("min_law_entropy_target", 0.0)) - min_law_entropy)
    score += _score_cfg_weight(
        score_cfg,
        "w_transfer_pressure_floor",
        "w_transfer_min_branch",
        "w_transfer_min_live_child",
        "w_transfer_min_writeback",
        "w_transfer_min_parent_div",
        scale=0.12,
    ) * float(pressure_metrics["floor_ratio"])
    score += _score_cfg_weight(
        score_cfg,
        "w_transfer_branch_first_score",
        "w_transfer_min_branch",
        "w_transfer_min_live_child",
        "w_transfer_min_writeback",
        "w_transfer_min_parent_div",
        scale=0.15,
    ) * float(pressure_metrics["branch_first_score"])
    score += _score_cfg_weight(
        score_cfg,
        "w_transfer_branch_live_coupling",
        "w_transfer_min_branch",
        "w_transfer_min_live_child",
        scale=0.12,
    ) * float(pressure_metrics["branch_live_coupling"])
    score += _score_cfg_weight(
        score_cfg,
        "w_transfer_writeback_parent_coupling",
        "w_transfer_min_writeback",
        "w_transfer_min_parent_div",
        scale=0.12,
    ) * float(pressure_metrics["writeback_parent_coupling"])
    score -= _score_cfg_weight(
        score_cfg,
        "w_transfer_pressure_shortfall",
        "w_transfer_branch_shortfall",
        "w_transfer_live_child_shortfall",
        "w_transfer_writeback_shortfall",
        "w_transfer_parent_div_shortfall",
        scale=0.10,
    ) * float(pressure_metrics["weighted_shortfall_ratio"])
    if naked:
        score += float(score_cfg.get("w_naked_min_branch", 0.0)) * naked_branch
        score += float(score_cfg.get("w_naked_min_phase_only_branch", 0.0)) * naked_phase_branch
        score += float(score_cfg.get("w_naked_min_phase_only_distinctness", 0.0)) * naked_phase_distinctness
        score += float(score_cfg.get("w_naked_min_meso", 0.0)) * naked_meso
        score += float(score_cfg.get("w_naked_min_writeback", 0.0)) * naked_writeback
        score += float(score_cfg.get("w_naked_min_parent_div", 0.0)) * naked_div
        score += float(score_cfg.get("w_naked_min_live_child", 0.0)) * naked_live
        score += float(score_cfg.get("w_naked_min_law_families", 0.0)) * naked_law_families
        score += float(score_cfg.get("w_naked_min_law_entropy", 0.0)) * naked_law_entropy
        score -= float(score_cfg.get("w_naked_defect_without_branch", 0.0)) * naked_defect_wo_branch
        score -= float(score_cfg.get("w_naked_branch_delay", 0.0)) * max(0.0, naked_delay)
        score -= float(score_cfg.get("w_naked_branch_shortfall", 0.0)) * shortfall(
            probe_cfg.get("min_naked_branch_target", probe_cfg.get("min_branch_target", 0.15)), naked_branch
        )
        score -= float(score_cfg.get("w_naked_phase_only_branch_shortfall", 0.0)) * shortfall(
            probe_cfg.get("min_naked_phase_only_branch_target", probe_cfg.get("min_phase_only_branch_target", 0.0)),
            naked_phase_branch,
        )
        score -= float(score_cfg.get("w_naked_phase_only_distinctness_shortfall", 0.0)) * shortfall(
            probe_cfg.get(
                "min_naked_phase_only_distinctness_target",
                probe_cfg.get("min_phase_only_distinctness_target", 0.0),
            ),
            naked_phase_distinctness,
        )
        score -= float(score_cfg.get("w_naked_writeback_shortfall", 0.0)) * shortfall(
            probe_cfg.get("min_naked_writeback_target", probe_cfg.get("min_writeback_target", 0.02)), naked_writeback
        )
        score -= float(score_cfg.get("w_naked_parent_div_shortfall", 0.0)) * shortfall(
            probe_cfg.get("min_naked_parent_div_target", probe_cfg.get("min_parent_div_target", 0.05)), naked_div
        )
        score -= float(score_cfg.get("w_naked_meso_shortfall", 0.0)) * shortfall(
            probe_cfg.get("min_naked_meso_target", probe_cfg.get("min_meso_target", 0.004)), naked_meso
        )
        score -= float(score_cfg.get("w_naked_live_child_shortfall", 0.0)) * shortfall(
            probe_cfg.get("min_naked_live_child_target", probe_cfg.get("min_live_child_target", 0.10)), naked_live
        )
        score -= float(score_cfg.get("w_naked_law_family_shortfall", 0.0)) * shortfall(
            probe_cfg.get("min_naked_law_family_target", probe_cfg.get("min_law_family_target", 1.0)),
            naked_law_families,
        )
        score -= float(score_cfg.get("w_naked_law_entropy_shortfall", 0.0)) * shortfall(
            probe_cfg.get("min_naked_law_entropy_target", probe_cfg.get("min_law_entropy_target", 0.0)),
            naked_law_entropy,
        )
        score -= float(score_cfg.get("w_transfer_naked_zero_branch", 0.0)) * (1.0 if naked_branch <= 1e-8 else 0.0)
        score -= float(score_cfg.get("w_transfer_naked_zero_writeback", 0.0)) * (1.0 if naked_writeback <= 1e-8 else 0.0)
        score -= float(score_cfg.get("w_naked_zero_child_worlds", 0.0)) * (1.0 if naked_child_worlds <= 1e-8 else 0.0)
        defect_gate = max(0.0, naked_defect - float(probe_cfg.get("naked_defect_gate", 0.10)))
        score -= float(score_cfg.get("w_naked_high_defect_zero_branch", 0.0)) * defect_gate * (1.0 if naked_branch <= 1e-8 else 0.0)
        score -= float(score_cfg.get("w_naked_high_defect_zero_writeback", 0.0)) * defect_gate * (1.0 if naked_writeback <= 1e-8 else 0.0)
        score += _score_cfg_weight(
            score_cfg,
            "w_naked_pressure_floor",
            "w_naked_min_branch",
            "w_naked_min_live_child",
            "w_naked_min_writeback",
            "w_naked_min_parent_div",
            scale=0.14,
        ) * float(naked_pressure_metrics["floor_ratio"])
        score += _score_cfg_weight(
            score_cfg,
            "w_naked_branch_first_score",
            "w_naked_min_branch",
            "w_naked_min_live_child",
            "w_naked_min_writeback",
            "w_naked_min_parent_div",
            scale=0.18,
        ) * float(naked_pressure_metrics["branch_first_score"])
        score += _score_cfg_weight(
            score_cfg,
            "w_naked_branch_live_coupling",
            "w_naked_min_branch",
            "w_naked_min_live_child",
            scale=0.14,
        ) * float(naked_pressure_metrics["branch_live_coupling"])
        score += _score_cfg_weight(
            score_cfg,
            "w_naked_writeback_parent_coupling",
            "w_naked_min_writeback",
            "w_naked_min_parent_div",
            scale=0.14,
        ) * float(naked_pressure_metrics["writeback_parent_coupling"])
        score -= _score_cfg_weight(
            score_cfg,
            "w_naked_pressure_shortfall",
            "w_naked_branch_shortfall",
            "w_naked_live_child_shortfall",
            "w_naked_writeback_shortfall",
            "w_naked_parent_div_shortfall",
            scale=0.12,
        ) * float(naked_pressure_metrics["weighted_shortfall_ratio"])

    return {
        "enabled": True,
        "score": float(score),
        "rows": rows,
        "by_source": by_source,
        "min_branch": float(min_branch),
        "min_phase_only_branch": float(min_phase_branch),
        "min_phase_only_distinctness": float(min_phase_distinctness),
        "min_meso": float(min_meso),
        "min_writeback": float(min_writeback),
        "min_parent_div": float(min_div),
        "min_live_child": float(min_live),
        "branch_gap": float(branch_gap),
        "phase_only_branch_gap": float(phase_branch_gap),
        "phase_only_distinctness_gap": float(phase_distinctness_gap),
        "meso_gap": float(meso_gap),
        "writeback_gap": float(writeback_gap),
        "parent_div_gap": float(div_gap),
        "law_family_gap": float(law_family_gap),
        "law_entropy_gap": float(law_entropy_gap),
        "naked_branch": float(naked_branch),
        "naked_phase_only_branch": float(naked_phase_branch),
        "naked_phase_only_distinctness": float(naked_phase_distinctness),
        "naked_meso": float(naked_meso),
        "synth_phase_only_branch": float(synth_phase_branch),
        "synth_phase_only_distinctness": float(synth_phase_distinctness),
        "naked_writeback": float(naked_writeback),
        "naked_parent_div": float(naked_div),
        "naked_live_child": float(naked_live),
        "naked_child_world_count": float(naked_child_worlds),
        "naked_branch_defect": float(naked_defect),
        "naked_defect_without_branch": float(naked_defect_wo_branch),
        "synth_branch_defect": float(synth_defect),
        "synth_defect_without_branch": float(synth_defect_wo_branch),
        "naked_law_families": float(naked_law_families),
        "naked_law_entropy": float(naked_law_entropy),
        "synth_law_families": float(synth_law_families),
        "synth_law_entropy": float(synth_law_entropy),
        "pressure_metrics": pressure_metrics,
        "naked_pressure_metrics": naked_pressure_metrics,
        "pressure_floor_ratio": float(pressure_metrics["floor_ratio"]),
        "pressure_weighted_ratio": float(pressure_metrics["weighted_ratio"]),
        "pressure_weighted_shortfall_ratio": float(pressure_metrics["weighted_shortfall_ratio"]),
        "pressure_branch_first_score": float(pressure_metrics["branch_first_score"]),
        "pressure_branch_live_coupling": float(pressure_metrics["branch_live_coupling"]),
        "pressure_writeback_parent_coupling": float(pressure_metrics["writeback_parent_coupling"]),
        "naked_pressure_floor_ratio": float(naked_pressure_metrics["floor_ratio"]),
        "naked_pressure_weighted_ratio": float(naked_pressure_metrics["weighted_ratio"]),
        "naked_pressure_weighted_shortfall_ratio": float(naked_pressure_metrics["weighted_shortfall_ratio"]),
        "naked_pressure_branch_first_score": float(naked_pressure_metrics["branch_first_score"]),
        "naked_pressure_branch_live_coupling": float(naked_pressure_metrics["branch_live_coupling"]),
        "naked_pressure_writeback_parent_coupling": float(naked_pressure_metrics["writeback_parent_coupling"]),
    }


def _evaluate_transfer_probe_repeated(
    circle_cfg,
    device_name: str,
    score_cfg: dict[str, float],
    probe_cfg: dict[str, Any] | None = None,
) -> dict[str, Any]:
    probe_cfg = dict(probe_cfg or {})
    repeats = max(1, int(probe_cfg.get("selection_probe_repeats", 1)))
    runs = [
        _evaluate_transfer_probe(
            circle_cfg,
            device_name=device_name,
            score_cfg=score_cfg,
            probe_cfg=probe_cfg,
        )
        for _ in range(repeats)
    ]
    return _aggregate_transfer_probe_runs(runs)


def _evaluate_selection_heldout(
    circle_cfg,
    device_name: str,
    probe_cfg: dict[str, Any] | None = None,
) -> dict[str, Any]:
    probe_cfg = dict(probe_cfg or {})
    if not probe_cfg.get("enabled", False):
        return {"enabled": False, "rows": [], "by_source": {}}

    time_steps = int(probe_cfg.get("time_steps", 128))
    heldout_plan = list(
        probe_cfg.get(
            "plan",
            [
                ("synthetic", 5100),
                ("synthetic", 5101),
                ("synthetic", 5102),
                ("synthetic", 5103),
                ("synthetic", 5104),
                ("synthetic", 5105),
                ("naked_rafa", 6100),
                ("naked_rafa", 6101),
                ("naked_rafa", 6102),
            ],
        )
    )
    summary = evaluate_circle_cfg(
        circle_cfg,
        time_steps=time_steps,
        device_name=device_name,
        plan=heldout_plan,
        include_trajectory=False,
    )
    summary["enabled"] = True
    summary.update(_heldout_pressure_metrics(summary, probe_cfg))
    return summary


def _selection_heldout_gate_status(
    heldout_eval: dict[str, Any],
    probe_cfg: dict[str, Any] | None,
    score_cfg: dict[str, float],
) -> dict[str, Any]:
    probe_cfg = dict(probe_cfg or {})
    enabled = bool(score_cfg.get("use_hard_selection_gate", 0.0)) and bool(heldout_eval.get("enabled", False))
    if not enabled:
        return {
            "enabled": False,
            "passed": True,
            "margin": 0.0,
            "failed_checks": [],
            "checks": {},
        }

    naked = dict(heldout_eval.get("by_source", {}).get("naked_rafa", {}))
    checks = {
        "heldout_branch": (
            float(heldout_eval.get("mean_real_branch_fraction", 0.0)),
            float(probe_cfg.get("min_branch_target", 0.0)),
        ),
        "heldout_phase_only_branch": (
            float(heldout_eval.get("phase_only_real_branch_fraction", 0.0)),
            float(probe_cfg.get("min_phase_only_branch_target", 0.0)),
        ),
        "heldout_phase_only_distinctness": (
            float(heldout_eval.get("mean_phase_only_branch_distinctness", 0.0)),
            float(probe_cfg.get("min_phase_only_distinctness_target", 0.0)),
        ),
        "heldout_meso": (
            float(heldout_eval.get("mean_meso_branch_effect", 0.0)),
            float(probe_cfg.get("min_meso_target", 0.0)),
        ),
        "heldout_writeback": (
            float(heldout_eval.get("mean_child_writeback_mass", 0.0)),
            float(probe_cfg.get("min_writeback_target", 0.0)),
        ),
        "heldout_parent_div": (
            float(heldout_eval.get("mean_child_parent_divergence", 0.0)),
            float(probe_cfg.get("min_parent_div_target", 0.0)),
        ),
        "heldout_live_child": (
            float(heldout_eval.get("mean_live_child_fraction", 0.0)),
            float(probe_cfg.get("min_live_child_target", 0.0)),
        ),
        "heldout_naked_branch": (
            float(naked.get("mean_real_branch_fraction", 0.0)),
            float(probe_cfg.get("min_naked_branch_target", probe_cfg.get("min_branch_target", 0.0))),
        ),
        "heldout_naked_phase_only_branch": (
            float(naked.get("phase_only_real_branch_fraction", 0.0)),
            float(probe_cfg.get("min_naked_phase_only_branch_target", probe_cfg.get("min_phase_only_branch_target", 0.0))),
        ),
        "heldout_naked_phase_only_distinctness": (
            float(naked.get("mean_phase_only_branch_distinctness", 0.0)),
            float(probe_cfg.get("min_naked_phase_only_distinctness_target", probe_cfg.get("min_phase_only_distinctness_target", 0.0))),
        ),
        "heldout_naked_meso": (
            float(naked.get("mean_meso_branch_effect", 0.0)),
            float(probe_cfg.get("min_naked_meso_target", probe_cfg.get("min_meso_target", 0.0))),
        ),
        "heldout_naked_writeback": (
            float(naked.get("mean_child_writeback_mass", 0.0)),
            float(probe_cfg.get("min_naked_writeback_target", probe_cfg.get("min_writeback_target", 0.0))),
        ),
        "heldout_naked_parent_div": (
            float(naked.get("mean_child_parent_divergence", 0.0)),
            float(probe_cfg.get("min_naked_parent_div_target", probe_cfg.get("min_parent_div_target", 0.0))),
        ),
        "heldout_naked_live_child": (
            float(naked.get("mean_live_child_fraction", 0.0)),
            float(probe_cfg.get("min_naked_live_child_target", probe_cfg.get("min_live_child_target", 0.0))),
        ),
        "heldout_naked_law_families": (
            float(naked.get("mean_num_law_families", 0.0)),
            float(probe_cfg.get("min_naked_law_family_target", probe_cfg.get("min_law_family_target", 0.0))),
        ),
        "heldout_naked_law_entropy": (
            float(naked.get("mean_law_family_entropy", 0.0)),
            float(probe_cfg.get("min_naked_law_entropy_target", probe_cfg.get("min_law_entropy_target", 0.0))),
        ),
    }
    heldout_floor_scale = float(probe_cfg.get("selection_heldout_floor_scale", 0.80))
    heldout_naked_floor_scale = float(probe_cfg.get("selection_heldout_naked_floor_scale", 0.90))
    checks.update(
        {
            "heldout_branch_floor": (
                float(heldout_eval.get("real_branch_floor", 0.0)),
                float(probe_cfg.get("selection_heldout_branch_floor_target", heldout_floor_scale * probe_cfg.get("min_branch_target", 0.0))),
            ),
            "heldout_live_child_floor": (
                float(heldout_eval.get("live_child_floor", 0.0)),
                float(probe_cfg.get("selection_heldout_live_child_floor_target", heldout_floor_scale * probe_cfg.get("min_live_child_target", 0.0))),
            ),
            "heldout_writeback_floor": (
                float(heldout_eval.get("child_writeback_floor", 0.0)),
                float(probe_cfg.get("selection_heldout_writeback_floor_target", heldout_floor_scale * probe_cfg.get("min_writeback_target", 0.0))),
            ),
            "heldout_parent_div_floor": (
                float(heldout_eval.get("child_parent_divergence_floor", 0.0)),
                float(probe_cfg.get("selection_heldout_parent_div_floor_target", heldout_floor_scale * probe_cfg.get("min_parent_div_target", 0.0))),
            ),
            "heldout_naked_branch_floor": (
                float(heldout_eval.get("naked_real_branch_floor", 0.0)),
                float(
                    probe_cfg.get(
                        "selection_heldout_naked_branch_floor_target",
                        heldout_naked_floor_scale * probe_cfg.get("min_naked_branch_target", probe_cfg.get("min_branch_target", 0.0)),
                    )
                ),
            ),
            "heldout_naked_live_child_floor": (
                float(heldout_eval.get("naked_live_child_floor", 0.0)),
                float(
                    probe_cfg.get(
                        "selection_heldout_naked_live_child_floor_target",
                        heldout_naked_floor_scale * probe_cfg.get("min_naked_live_child_target", probe_cfg.get("min_live_child_target", 0.0)),
                    )
                ),
            ),
            "heldout_naked_writeback_floor": (
                float(heldout_eval.get("naked_child_writeback_floor", 0.0)),
                float(
                    probe_cfg.get(
                        "selection_heldout_naked_writeback_floor_target",
                        heldout_naked_floor_scale * probe_cfg.get("min_naked_writeback_target", probe_cfg.get("min_writeback_target", 0.0)),
                    )
                ),
            ),
            "heldout_naked_parent_div_floor": (
                float(heldout_eval.get("naked_child_parent_divergence_floor", 0.0)),
                float(
                    probe_cfg.get(
                        "selection_heldout_naked_parent_div_floor_target",
                        heldout_naked_floor_scale * probe_cfg.get("min_naked_parent_div_target", probe_cfg.get("min_parent_div_target", 0.0)),
                    )
                ),
            ),
        }
    )
    optional_checks = {
        "heldout_pressure_floor_ratio": (
            float(heldout_eval.get("pressure_floor_ratio", 0.0)),
            probe_cfg.get("selection_heldout_pressure_floor_ratio_target"),
        ),
        "heldout_naked_pressure_floor_ratio": (
            float(heldout_eval.get("naked_pressure_floor_ratio", 0.0)),
            probe_cfg.get("selection_heldout_naked_pressure_floor_ratio_target"),
        ),
        "heldout_pressure_branch_live_coupling": (
            float(heldout_eval.get("pressure_branch_live_coupling", 0.0)),
            probe_cfg.get("selection_heldout_pressure_branch_live_coupling_target"),
        ),
        "heldout_naked_pressure_branch_live_coupling": (
            float(heldout_eval.get("naked_pressure_branch_live_coupling", 0.0)),
            probe_cfg.get("selection_heldout_naked_pressure_branch_live_coupling_target"),
        ),
        "heldout_pressure_writeback_parent_coupling": (
            float(heldout_eval.get("pressure_writeback_parent_coupling", 0.0)),
            probe_cfg.get("selection_heldout_pressure_writeback_parent_coupling_target"),
        ),
        "heldout_naked_pressure_writeback_parent_coupling": (
            float(heldout_eval.get("naked_pressure_writeback_parent_coupling", 0.0)),
            probe_cfg.get("selection_heldout_naked_pressure_writeback_parent_coupling_target"),
        ),
    }
    for name, (value, target) in optional_checks.items():
        if target is not None:
            checks[name] = (value, float(target))

    failed_checks: list[str] = []
    margin = float("inf")
    check_rows: dict[str, dict[str, float]] = {}
    for name, (value, target) in checks.items():
        delta = float(value - target)
        check_rows[name] = {"value": value, "target": target, "delta": delta}
        if target > 0.0:
            margin = min(margin, delta)
            if delta < 0.0:
                failed_checks.append(name)
    if margin == float("inf"):
        margin = 0.0

    return {
        "enabled": True,
        "passed": len(failed_checks) == 0,
        "margin": margin,
        "failed_checks": failed_checks,
        "checks": check_rows,
    }


def _selection_probe_heldout_agreement(
    transfer_probe: dict[str, Any],
    heldout_eval: dict[str, Any],
) -> dict[str, Any]:
    naked = dict(heldout_eval.get("by_source", {}).get("naked_rafa", {}))
    deltas = {
        "branch": abs(float(transfer_probe.get("min_branch_min", transfer_probe.get("min_branch", 0.0))) - float(heldout_eval.get("mean_real_branch_fraction", 0.0))),
        "meso": abs(float(transfer_probe.get("min_meso_min", transfer_probe.get("min_meso", 0.0))) - float(heldout_eval.get("mean_meso_branch_effect", 0.0))),
        "writeback": abs(float(transfer_probe.get("min_writeback_min", transfer_probe.get("min_writeback", 0.0))) - float(heldout_eval.get("mean_child_writeback_mass", 0.0))),
        "parent_div": abs(float(transfer_probe.get("min_parent_div_min", transfer_probe.get("min_parent_div", 0.0))) - float(heldout_eval.get("mean_child_parent_divergence", 0.0))),
        "naked_branch": abs(float(transfer_probe.get("naked_branch_min", transfer_probe.get("naked_branch", 0.0))) - float(naked.get("mean_real_branch_fraction", 0.0))),
        "naked_meso": abs(float(transfer_probe.get("naked_meso_min", transfer_probe.get("naked_meso", 0.0))) - float(naked.get("mean_meso_branch_effect", 0.0))),
        "naked_writeback": abs(float(transfer_probe.get("naked_writeback_min", transfer_probe.get("naked_writeback", 0.0))) - float(naked.get("mean_child_writeback_mass", 0.0))),
        "naked_parent_div": abs(float(transfer_probe.get("naked_parent_div_min", transfer_probe.get("naked_parent_div", 0.0))) - float(naked.get("mean_child_parent_divergence", 0.0))),
    }
    floor_deltas = {
        "branch_floor": abs(float(transfer_probe.get("min_branch_min", transfer_probe.get("min_branch", 0.0))) - float(heldout_eval.get("real_branch_floor", 0.0))),
        "live_child_floor": abs(float(transfer_probe.get("min_live_child_min", transfer_probe.get("min_live_child", 0.0))) - float(heldout_eval.get("live_child_floor", 0.0))),
        "writeback_floor": abs(float(transfer_probe.get("min_writeback_min", transfer_probe.get("min_writeback", 0.0))) - float(heldout_eval.get("child_writeback_floor", 0.0))),
        "parent_div_floor": abs(float(transfer_probe.get("min_parent_div_min", transfer_probe.get("min_parent_div", 0.0))) - float(heldout_eval.get("child_parent_divergence_floor", 0.0))),
        "naked_branch_floor": abs(float(transfer_probe.get("naked_branch_min", transfer_probe.get("naked_branch", 0.0))) - float(heldout_eval.get("naked_real_branch_floor", 0.0))),
        "naked_live_child_floor": abs(float(transfer_probe.get("naked_live_child_min", transfer_probe.get("naked_live_child", 0.0))) - float(heldout_eval.get("naked_live_child_floor", 0.0))),
        "naked_writeback_floor": abs(float(transfer_probe.get("naked_writeback_min", transfer_probe.get("naked_writeback", 0.0))) - float(heldout_eval.get("naked_child_writeback_floor", 0.0))),
        "naked_parent_div_floor": abs(float(transfer_probe.get("naked_parent_div_min", transfer_probe.get("naked_parent_div", 0.0))) - float(heldout_eval.get("naked_child_parent_divergence_floor", 0.0))),
    }
    pressure_deltas = {
        "pressure_floor_ratio": abs(float(transfer_probe.get("pressure_floor_ratio", 0.0)) - float(heldout_eval.get("pressure_floor_ratio", 0.0))),
        "pressure_branch_first_score": abs(float(transfer_probe.get("pressure_branch_first_score", 0.0)) - float(heldout_eval.get("pressure_branch_first_score", 0.0))),
        "naked_pressure_floor_ratio": abs(float(transfer_probe.get("naked_pressure_floor_ratio", 0.0)) - float(heldout_eval.get("naked_pressure_floor_ratio", 0.0))),
        "naked_pressure_branch_first_score": abs(
            float(transfer_probe.get("naked_pressure_branch_first_score", 0.0)) - float(heldout_eval.get("naked_pressure_branch_first_score", 0.0))
        ),
    }
    weighted_sum = (
        0.8 * deltas["branch"]
        + 10.0 * deltas["meso"]
        + 2.0 * deltas["writeback"]
        + 2.0 * deltas["parent_div"]
        + 2.0 * deltas["naked_branch"]
        + 40.0 * deltas["naked_meso"]
        + 4.0 * deltas["naked_writeback"]
        + 4.0 * deltas["naked_parent_div"]
        + 2.4 * floor_deltas["branch_floor"]
        + 1.6 * floor_deltas["live_child_floor"]
        + 2.0 * floor_deltas["writeback_floor"]
        + 2.0 * floor_deltas["parent_div_floor"]
        + 3.4 * floor_deltas["naked_branch_floor"]
        + 2.0 * floor_deltas["naked_live_child_floor"]
        + 2.6 * floor_deltas["naked_writeback_floor"]
        + 2.6 * floor_deltas["naked_parent_div_floor"]
        + 2.4 * pressure_deltas["pressure_floor_ratio"]
        + 2.0 * pressure_deltas["pressure_branch_first_score"]
        + 3.2 * pressure_deltas["naked_pressure_floor_ratio"]
        + 2.6 * pressure_deltas["naked_pressure_branch_first_score"]
    )
    return {
        "deltas": deltas,
        "floor_deltas": floor_deltas,
        "pressure_deltas": pressure_deltas,
        "max_abs_delta": float(max([*deltas.values(), *floor_deltas.values(), *pressure_deltas.values()]) if (deltas or floor_deltas or pressure_deltas) else 0.0),
        "weighted_abs_delta": float(weighted_sum),
        "score": float(-weighted_sum),
    }


def _selection_nested_gate_status(
    nested_eval: dict[str, Any],
    probe_cfg: dict[str, Any] | None,
) -> dict[str, Any]:
    probe_cfg = dict(probe_cfg or {})
    enabled = bool(nested_eval.get("enabled", False))
    if not enabled:
        return {
            "enabled": False,
            "passed": True,
            "margin": 0.0,
            "failed_checks": [],
            "checks": {},
        }

    case_count = max(1.0, float(nested_eval.get("num_cases", 0.0) or 0.0))
    live_start_fraction = float(nested_eval.get("num_live_child_start_cases", 0.0)) / case_count
    selected_live_fraction = float(nested_eval.get("mean_selected_fork_has_live_child", 0.0))
    checks = {
        "nested_response": (
            float(nested_eval.get("mean_child_response_score", 0.0)),
            float(probe_cfg.get("selection_nested_min_response", float("-inf"))),
        ),
        "nested_active_fraction": (
            float(nested_eval.get("mean_child_active_fraction", 0.0)),
            float(probe_cfg.get("selection_nested_min_active_fraction", 0.0)),
        ),
        "nested_live_child_start_fraction": (
            live_start_fraction,
            float(probe_cfg.get("selection_nested_min_live_child_start_fraction", 0.0)),
        ),
        "nested_selected_live_child_fraction": (
            selected_live_fraction,
            float(probe_cfg.get("selection_nested_min_selected_live_child_fraction", 0.0)),
        ),
        "nested_meso_response": (
            float(nested_eval.get("mean_child_meso_response", 0.0)),
            float(probe_cfg.get("selection_nested_min_meso_response", 0.0)),
        ),
        "nested_sibling_fraction": (
            float(nested_eval.get("mean_nested_sibling_fraction", 0.0)),
            float(probe_cfg.get("selection_nested_min_sibling_fraction", 0.0)),
        ),
        "nested_identity_carry": (
            float(nested_eval.get("mean_branch_identity_carry", 0.0)),
            float(probe_cfg.get("selection_nested_min_identity_carry", 0.0)),
        ),
        "nested_qualified_carry": (
            float(nested_eval.get("mean_branch_identity_qualified_carry", 0.0)),
            float(probe_cfg.get("selection_nested_min_qualified_carry", 0.0)),
        ),
        "nested_max_qualified_carry": (
            float(nested_eval.get("mean_max_branch_identity_qualified_carry", nested_eval.get("max_branch_identity_qualified_carry", 0.0))),
            float(probe_cfg.get("selection_nested_min_max_qualified_carry", 0.0)),
        ),
        "nested_max_qualified_steps": (
            float(nested_eval.get("mean_max_branch_identity_qualified_steps", nested_eval.get("max_branch_identity_qualified_steps", 0.0))),
            float(probe_cfg.get("selection_nested_min_max_qualified_steps", 0.0)),
        ),
        "nested_best_identity_support": (
            float(nested_eval.get("mean_best_identity_support_mean", 0.0)),
            float(probe_cfg.get("selection_nested_min_best_identity_support", 0.0)),
        ),
        "nested_best_identity_coherence": (
            float(nested_eval.get("mean_best_identity_coherence_mean", 0.0)),
            float(probe_cfg.get("selection_nested_min_best_identity_coherence", 0.0)),
        ),
        "nested_best_identity_min_support": (
            float(nested_eval.get("mean_best_identity_min_support", 0.0)),
            float(probe_cfg.get("selection_nested_min_best_identity_min_support", 0.0)),
        ),
        "nested_best_identity_min_coherence": (
            float(nested_eval.get("mean_best_identity_min_coherence", 0.0)),
            float(probe_cfg.get("selection_nested_min_best_identity_min_coherence", 0.0)),
        ),
        "nested_budget_retained": (
            float(nested_eval.get("mean_branch_identity_budget_retained", 0.0)),
            float(probe_cfg.get("selection_nested_min_budget_retained", 0.0)),
        ),
        "nested_identity_parent_div": (
            float(nested_eval.get("mean_branch_identity_parent_div", 0.0)),
            float(probe_cfg.get("selection_nested_min_identity_parent_div", 0.0)),
        ),
        "nested_identity_sibling_div": (
            float(nested_eval.get("mean_branch_identity_sibling_div", 0.0)),
            float(probe_cfg.get("selection_nested_min_identity_sibling_div", 0.0)),
        ),
        "nested_mode_replace_readiness": (
            _nested_family_metric(nested_eval, "mode_replace", "max_nested_sibling_readiness"),
            float(probe_cfg.get("selection_nested_min_mode_replace_readiness", 0.0)),
        ),
        "nested_child_record_survival_score": (
            float(nested_eval.get("mean_child_record_survival_score", 0.0)),
            float(probe_cfg.get("selection_nested_min_child_record_survival_score", 0.0)),
        ),
        "nested_mode_replace_conversion_score": (
            float(nested_eval.get("mean_mode_replace_conversion_score", 0.0)),
            float(probe_cfg.get("selection_nested_min_mode_replace_conversion_score", 0.0)),
        ),
    }
    max_over_rigid = probe_cfg.get("selection_nested_max_over_rigid")
    max_world_jump = probe_cfg.get("selection_nested_max_world_jump")
    max_direct_readout_dependency = probe_cfg.get("selection_nested_max_direct_readout_dependency")
    max_final_direct_mix_shortcut = probe_cfg.get("selection_nested_max_final_direct_mix_shortcut")

    failed_checks: list[str] = []
    margin = float("inf")
    check_rows: dict[str, dict[str, float]] = {}
    for name, (value, target) in checks.items():
        delta = float(value - target)
        check_rows[name] = {"value": value, "target": target, "delta": delta}
        if math.isfinite(target):
            margin = min(margin, delta)
            if delta < 0.0:
                failed_checks.append(name)
    if max_over_rigid is not None:
        over_rigid_value = float(nested_eval.get("mean_over_rigid_fraction", 0.0))
        over_rigid_target = float(max_over_rigid)
        delta = float(over_rigid_target - over_rigid_value)
        check_rows["nested_over_rigid"] = {"value": over_rigid_value, "target": over_rigid_target, "delta": delta}
        margin = min(margin, delta)
        if delta < 0.0:
            failed_checks.append("nested_over_rigid")
    if max_world_jump is not None:
        world_jump_value = float(nested_eval.get("mean_world_jump_penalty", 0.0))
        world_jump_target = float(max_world_jump)
        delta = float(world_jump_target - world_jump_value)
        check_rows["nested_world_jump"] = {"value": world_jump_value, "target": world_jump_target, "delta": delta}
        margin = min(margin, delta)
        if delta < 0.0:
            failed_checks.append("nested_world_jump")
    if max_direct_readout_dependency is not None:
        direct_dependency_value = float(nested_eval.get("mean_direct_readout_dependency", 0.0))
        direct_dependency_target = float(max_direct_readout_dependency)
        delta = float(direct_dependency_target - direct_dependency_value)
        check_rows["nested_direct_readout_dependency"] = {
            "value": direct_dependency_value,
            "target": direct_dependency_target,
            "delta": delta,
        }
        margin = min(margin, delta)
        if delta < 0.0:
            failed_checks.append("nested_direct_readout_dependency")
    if max_final_direct_mix_shortcut is not None:
        shortcut_value = float(nested_eval.get("mean_final_direct_mix_shortcut_score", 0.0))
        shortcut_target = float(max_final_direct_mix_shortcut)
        delta = float(shortcut_target - shortcut_value)
        check_rows["nested_final_direct_mix_shortcut"] = {
            "value": shortcut_value,
            "target": shortcut_target,
            "delta": delta,
        }
        margin = min(margin, delta)
        if delta < 0.0:
            failed_checks.append("nested_final_direct_mix_shortcut")
    if margin == float("inf"):
        margin = 0.0

    return {
        "enabled": True,
        "passed": len(failed_checks) == 0,
        "margin": margin,
        "failed_checks": failed_checks,
        "checks": check_rows,
    }


def _selection_heldout_sort_key(row: dict[str, Any]) -> tuple[float, ...]:
    probe_gate = row.get("selection_gate", {})
    heldout_gate = row.get("selection_heldout_gate", {})
    agreement = row.get("selection_heldout_agreement", {})
    heldout = row.get("selection_heldout", {})
    nested = row.get("selection_nested", {})
    nested_probe = row.get("nested_probe", {})
    nested_gate = row.get("selection_nested_gate", {})
    nested_signal = nested if nested.get("enabled", False) else nested_probe
    nested_case_count = max(1.0, float(nested_signal.get("num_cases", 0.0) or 0.0))
    naked = dict(heldout.get("by_source", {}).get("naked_rafa", {}))
    return (
        1.0 if heldout_gate.get("passed", False) else 0.0,
        1.0 if nested_gate.get("passed", True) else 0.0,
        1.0 if probe_gate.get("passed", True) else 0.0,
        float(row.get("nested_probe_score", 0.0)),
        1.0 if nested_signal.get("enabled", False) else 0.0,
        float(nested_signal.get("num_live_child_start_cases", 0.0)) / nested_case_count,
        float(nested_signal.get("mean_selected_fork_has_live_child", 0.0)),
        float(nested_signal.get("mean_child_response_score", float("-inf"))),
        float(nested_signal.get("mean_child_active_fraction", 0.0)),
        float(nested_signal.get("mean_child_meso_response", 0.0)),
        float(nested_signal.get("mean_nested_sibling_fraction", 0.0)),
        float(nested_signal.get("mean_branch_identity_qualified_carry", 0.0)),
        float(nested_signal.get("mean_max_branch_identity_qualified_carry", nested_signal.get("max_branch_identity_qualified_carry", 0.0))),
        float(nested_signal.get("mean_max_branch_identity_qualified_steps", nested_signal.get("max_branch_identity_qualified_steps", 0.0))),
        float(nested_signal.get("mean_best_identity_coherence_mean", 0.0)),
        float(nested_signal.get("mean_best_identity_min_coherence", 0.0)),
        float(nested_signal.get("mean_best_identity_support_mean", 0.0)),
        float(nested_signal.get("mean_best_identity_min_support", 0.0)),
        float(nested_signal.get("mean_branch_identity_carry", 0.0)),
        float(nested_signal.get("mean_branch_identity_budget_retained", 0.0)),
        float(nested_signal.get("mean_branch_identity_sibling_div", 0.0)),
        float(heldout.get("naked_pressure_branch_first_score", 0.0)),
        float(heldout.get("naked_pressure_floor_ratio", 0.0)),
        float(heldout.get("pressure_branch_first_score", 0.0)),
        float(heldout.get("pressure_floor_ratio", 0.0)),
        float(heldout.get("naked_real_branch_floor", 0.0)),
        float(heldout.get("real_branch_floor", 0.0)),
        float(heldout.get("naked_live_child_floor", 0.0)),
        float(heldout.get("naked_child_writeback_floor", 0.0)),
        float(heldout.get("naked_child_parent_divergence_floor", 0.0)),
        -float(heldout.get("naked_real_branch_floor_gap", 0.0)),
        -float(heldout.get("naked_child_writeback_floor_gap", 0.0)),
        -float(heldout.get("naked_child_parent_divergence_floor_gap", 0.0)),
        float(heldout_gate.get("margin", -1.0)),
        float(nested_gate.get("margin", 0.0)),
        float(agreement.get("score", float("-inf"))),
        float(naked.get("mean_child_parent_divergence", 0.0)),
        float(naked.get("mean_real_branch_fraction", 0.0)),
        float(heldout.get("mean_child_parent_divergence", 0.0)),
        float(heldout.get("mean_meso_branch_effect", 0.0)),
        *_selection_sort_key(row),
    )


def _selection_nested_is_explicit_failure(row: dict[str, Any]) -> bool:
    nested_gate = row.get("selection_nested_gate", {})
    return bool(nested_gate.get("enabled", False)) and not bool(nested_gate.get("passed", False))


def _gate_failure_counts(rows: list[dict[str, Any]], gate_key: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        gate = row.get(gate_key, {})
        for name in gate.get("failed_checks", []) or []:
            key = str(name)
            counts[key] = counts.get(key, 0) + 1
    return counts


def _compact_candidate_failure(row: dict[str, Any], idx: int) -> dict[str, Any]:
    heldout = row.get("selection_heldout", {})
    nested = row.get("selection_nested", {})
    nested_gate = row.get("selection_nested_gate", {})
    nested_case_count = max(1.0, float(nested.get("num_cases", 0.0) or 0.0))
    return {
        "rank": int(idx),
        "iteration": int(row.get("iteration", -1)),
        "selection_gate_passed": bool(row.get("selection_gate", {}).get("passed", True)),
        "selection_gate_failed_checks": list(row.get("selection_gate", {}).get("failed_checks", []) or []),
        "heldout_gate_passed": bool(row.get("selection_heldout_gate", {}).get("passed", False)),
        "heldout_failed_checks": list(row.get("selection_heldout_gate", {}).get("failed_checks", []) or []),
        "nested_gate_evaluated": bool(nested_gate.get("enabled", False)),
        "nested_gate_passed": bool(nested_gate.get("passed", False)) if nested_gate.get("enabled", False) else None,
        "nested_failed_checks": list(nested_gate.get("failed_checks", []) or []),
        "nested_probe_score": float(row.get("nested_probe_score", 0.0)),
        "heldout_naked_branch_floor": float(heldout.get("naked_real_branch_floor", 0.0)),
        "heldout_naked_writeback_floor": float(heldout.get("naked_child_writeback_floor", 0.0)),
        "nested_response": float(nested.get("mean_child_response_score", 0.0)),
        "nested_active_fraction": float(nested.get("mean_child_active_fraction", 0.0)),
        "nested_live_child_start_fraction": float(nested.get("num_live_child_start_cases", 0.0)) / nested_case_count,
        "nested_selected_live_child_fraction": float(nested.get("mean_selected_fork_has_live_child", 0.0)),
        "nested_sibling_fraction": float(nested.get("mean_nested_sibling_fraction", 0.0)),
        "nested_identity_carry": float(nested.get("mean_branch_identity_carry", 0.0)),
        "nested_qualified_carry": float(nested.get("mean_branch_identity_qualified_carry", 0.0)),
        "nested_max_qualified_carry": float(nested.get("mean_max_branch_identity_qualified_carry", nested.get("max_branch_identity_qualified_carry", 0.0))),
        "nested_max_qualified_steps": float(nested.get("mean_max_branch_identity_qualified_steps", nested.get("max_branch_identity_qualified_steps", 0.0))),
        "nested_best_identity_support": float(nested.get("mean_best_identity_support_mean", 0.0)),
        "nested_best_identity_coherence": float(nested.get("mean_best_identity_coherence_mean", 0.0)),
        "nested_best_identity_min_support": float(nested.get("mean_best_identity_min_support", 0.0)),
        "nested_best_identity_min_coherence": float(nested.get("mean_best_identity_min_coherence", 0.0)),
        "nested_budget_retained": float(nested.get("mean_branch_identity_budget_retained", 0.0)),
        "nested_identity_parent_div": float(nested.get("mean_branch_identity_parent_div", 0.0)),
        "nested_identity_sibling_div": float(nested.get("mean_branch_identity_sibling_div", 0.0)),
    }


def _selection_fallback_diagnostics(
    *,
    selection_source: str,
    selected_state: dict[str, Any] | None,
    best_state: dict[str, Any] | None,
    best_passing_state: dict[str, Any] | None,
    best_agreement_state: dict[str, Any] | None,
    best_nested_pending_state: dict[str, Any] | None,
    agreement_candidates: list[dict[str, Any]],
    hard_gate_enabled: bool,
    require_heldout_agreement: bool,
    require_nested_response: bool,
    allow_missing_nested_fallback: bool,
    agreement_top_k: int,
    nested_top_k: int,
    nested_eval_top_k: int,
    limit: int = 8,
) -> dict[str, Any]:
    reasons: list[str] = []
    heldout_pass_count = sum(1 for row in agreement_candidates if row.get("selection_heldout_gate", {}).get("passed", False))
    nested_evaluated_count = sum(1 for row in agreement_candidates if row.get("selection_nested_gate", {}).get("enabled", False))
    nested_pass_count = sum(1 for row in agreement_candidates if row.get("selection_nested_gate", {}).get("passed", False))

    if selected_state is None:
        if hard_gate_enabled and best_passing_state is None:
            reasons.append("no_transfer_gate_passing_candidate")
        if require_heldout_agreement and agreement_top_k > 0:
            if not agreement_candidates:
                reasons.append("no_heldout_agreement_candidates_evaluated")
            elif heldout_pass_count == 0:
                reasons.append("no_candidate_passed_heldout_gate")
        if require_nested_response and nested_top_k > 0:
            if nested_evaluated_count == 0:
                reasons.append("no_candidate_nested_gate_evaluated")
            elif nested_pass_count == 0:
                reasons.append("no_candidate_passed_nested_gate")
            if allow_missing_nested_fallback and best_nested_pending_state is None:
                reasons.append("no_soft_nested_pending_candidate")
            elif not allow_missing_nested_fallback:
                reasons.append("missing_nested_response_fallback_disabled")
        if not reasons:
            reasons.append("hard_gate_candidate_unavailable")

    top_rows = agreement_candidates[:limit]
    if not top_rows and best_state is not None:
        top_rows = [best_state]

    return {
        "enabled": bool(hard_gate_enabled),
        "selection_source": str(selection_source),
        "used_init_config_fallback": selected_state is None,
        "candidate_unavailable_reasons": reasons,
        "requirements": {
            "require_heldout_agreement": bool(require_heldout_agreement),
            "require_nested_response": bool(require_nested_response),
            "allow_missing_nested_fallback": bool(allow_missing_nested_fallback),
            "agreement_top_k": int(agreement_top_k),
            "nested_top_k": int(nested_top_k),
            "nested_eval_top_k": int(nested_eval_top_k),
        },
        "availability": {
            "has_best_candidate": best_state is not None,
            "has_best_passing_candidate": best_passing_state is not None,
            "has_best_agreement_candidate": best_agreement_state is not None,
            "has_best_nested_pending_candidate": best_nested_pending_state is not None,
            "num_agreement_candidates": len(agreement_candidates),
            "num_heldout_gate_pass": int(heldout_pass_count),
            "num_nested_gate_evaluated": int(nested_evaluated_count),
            "num_nested_gate_pass": int(nested_pass_count),
        },
        "failure_counts": {
            "selection_gate": _gate_failure_counts(agreement_candidates, "selection_gate"),
            "heldout_gate": _gate_failure_counts(agreement_candidates, "selection_heldout_gate"),
            "nested_gate": _gate_failure_counts(agreement_candidates, "selection_nested_gate"),
        },
        "top_candidate_failures": [_compact_candidate_failure(row, idx) for idx, row in enumerate(top_rows)],
    }


def _anchor_dataset(
    device_name: str,
    clip_seconds: int,
    extra_train_wavs: list[str] | None = None,
    extra_val_wavs: list[str] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    cfg = load_config()
    stft_cfg = cfg["data"]["stft"]
    train_wavs, val_wavs = _graduation_anchor_wavs()
    train_wavs = list(train_wavs) + list(extra_train_wavs or [])
    val_wavs = list(val_wavs) + list(extra_val_wavs or [])
    device = _safe_device(device_name)

    def _build(paths: list[str]) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for wav_str in paths:
            wav_path = Path(wav_str)
            wav, _sr = prepare_reference_audio(
                wav_path=wav_path,
                device_name=device_name,
                clip_seconds_override=clip_seconds,
            )
            mag, phase = compute_stft(wav, stft_cfg)
            rows.append(
                {
                    "name": wav_path.stem,
                    "wav_path": str(wav_path),
                    "ref_wav": wav.squeeze(0).detach().to(device),
                    "mag": mag.detach().to(device),
                    "phase": phase.detach().to(device),
                    "phase_state": torch.stack([torch.cos(phase), torch.sin(phase)], dim=-1).detach().to(device),
                }
            )
        return rows

    return _build(train_wavs), _build(val_wavs)


def _evaluate_cfg_on_dataset(
    circle_cfg,
    dataset: list[dict[str, Any]],
    phase_blend: float,
    score_cfg: dict[str, float],
    case_weights: dict[str, float] | None = None,
    baseline_rows: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    all_law_packets: list[dict[str, Any]] = []
    all_signature_banks = []
    run_mode = circle_cfg.branching_mode if circle_cfg.branching_mode in {"native_multimode", "native_multimode_childworld"} else "active_packets"
    for item in dataset:
        run = recurse_circleworld(
            item["phase_state"],
            cfg=circle_cfg,
            depth=circle_cfg.recursion_depth,
            mode=run_mode,
        )
        summary = summarize_circleworld_run(run)
        all_law_packets.extend(run.get("law_packets", []))
        if "relational_signatures" in run:
            all_signature_banks.append(run["relational_signatures"])
        final_phase = _blend_phase(item["phase"], run["phase_state"], blend=phase_blend)
        out_wav = inverse_stft(item["mag"], final_phase, load_config()["data"]["stft"]).squeeze(0)
        ref_wav = item["ref_wav"][: out_wav.numel()]
        out_wav = out_wav[: ref_wav.numel()]

        mse = float(torch.mean((ref_wav - out_wav) ** 2).item())
        mae = float(torch.mean(torch.abs(ref_wav - out_wav)).item())
        corr = _corrcoef_1d(ref_wav, out_wav)

        dom = float(summary["dominant_q_share"])
        ent = float(summary["q_entropy"])
        promos = float(summary["num_promotions"])
        corr_band = _window_penalty(
            corr,
            score_cfg.get("corr_low"),
            score_cfg.get("corr_high"),
        )
        mae_band = _window_penalty(
            mae,
            score_cfg.get("mae_low"),
            score_cfg.get("mae_high"),
        )
        row_weight = _case_weight_for(item, case_weights)
        baseline_row = baseline_rows.get(item["name"]) if baseline_rows else None
        corr_regret = 0.0
        mae_regret = 0.0
        if baseline_row is not None:
            corr_margin = float(score_cfg.get("baseline_corr_margin", 0.0))
            mae_margin = float(score_cfg.get("baseline_mae_margin", 0.0))
            corr_regret = max(0.0, float(baseline_row["corr"]) - corr - corr_margin)
            mae_regret = max(0.0, mae - float(baseline_row["mae"]) - mae_margin)
        score = (
            - score_cfg["w_corr"] * abs(corr - score_cfg["target_corr"])
            - score_cfg["w_mae"] * abs(mae - score_cfg["target_mae"])
            - score_cfg.get("w_corr_band", 0.0) * corr_band
            - score_cfg.get("w_mae_band", 0.0) * mae_band
            - score_cfg.get("w_baseline_corr", 0.0) * corr_regret
            - score_cfg.get("w_baseline_mae", 0.0) * mae_regret
            - score_cfg["w_dom"] * abs(dom - score_cfg["target_dom"])
            - score_cfg["w_entropy"] * abs(ent - score_cfg["target_entropy"])
            - score_cfg["w_promote"] * max(0.0, score_cfg["min_promotions"] - promos)
            + score_cfg["w_residue"] * float(summary["residue_drop"])
            + score_cfg["w_promotability"] * float(summary["promotability_gain"])
            + score_cfg["w_major"] * float(summary["major_gain"])
            + score_cfg.get("w_prefix_alignment", 0.0) * float(summary.get("prefix_alignment", 0.0))
            - score_cfg.get("w_prefix_delta", 0.0) * float(summary.get("prefix_delta", 0.0))
            + score_cfg.get("w_real_branch", 0.0) * float(summary.get("real_branch_fraction", 0.0))
            + score_cfg.get("w_phase_only_real_branch", 0.0) * float(summary.get("phase_only_real_branch_fraction", 0.0))
            + score_cfg.get("w_phase_only_branch_distinctness", 0.0) * float(summary.get("mean_phase_only_branch_distinctness", 0.0))
            + score_cfg.get("w_phase_only_branch_phase_wall", 0.0) * float(summary.get("mean_phase_only_branch_phase_wall", 0.0))
            + score_cfg.get("w_meso_branch", 0.0) * float(summary.get("meso_branch_effect", 0.0))
            + score_cfg.get("w_slot2_live", 0.0) * float(summary.get("mean_slot2_live_fraction", 0.0))
            - score_cfg.get("w_silent_singlepath", 0.0) * float(summary.get("silent_singlepath_fraction", 0.0))
            + score_cfg.get("w_branch_positive_mask", 0.0) * float(summary.get("mean_branch_positive_mask", 0.0))
            - score_cfg.get("w_branch_negative_mask", 0.0) * float(summary.get("mean_branch_negative_mask", 0.0))
            - score_cfg.get("w_decorative_slot2", 0.0) * float(summary.get("mean_decorative_slot2_fraction", 0.0))
            + score_cfg.get("w_relation_handoff", 0.0) * float(summary.get("mean_relation_handoff_drive", 0.0))
            + score_cfg.get("w_relation_attn_10", 0.0) * float(summary.get("mean_relation_attn_10", 0.0))
            + score_cfg.get("w_branch_defect", 0.0) * float(summary.get("mean_branch_defect", 0.0))
            + score_cfg.get("w_branch_world_grad", 0.0) * float(summary.get("mean_branch_world_grad", 0.0))
            + score_cfg.get("w_branch_q_disagreement", 0.0) * float(summary.get("mean_branch_q_disagreement", 0.0))
            + score_cfg.get("w_branch_phase_wall", 0.0) * float(summary.get("mean_branch_phase_wall", 0.0))
            + score_cfg.get("w_branch_seed_energy", 0.0) * float(summary.get("mean_branch_seed_energy", 0.0))
            + score_cfg.get("w_child_world_count", 0.0) * float(summary.get("mean_child_world_count", 0.0))
            + score_cfg.get("w_live_child", 0.0) * float(summary.get("mean_live_child_fraction", 0.0))
            + score_cfg.get("w_child_age", 0.0) * float(summary.get("mean_child_age", 0.0))
            + score_cfg.get("w_child_writeback", 0.0) * float(summary.get("mean_child_writeback_mass", 0.0))
            + score_cfg.get("w_child_local_ifs_step_count", 0.0) * float(summary.get("mean_child_local_ifs_step_count", 0.0))
            + score_cfg.get("w_child_local_ifs_packet_count", 0.0) * float(summary.get("mean_child_local_ifs_packet_count", 0.0))
            + score_cfg.get("w_child_local_ifs_phase_delta", 0.0) * float(summary.get("mean_child_local_ifs_phase_delta", 0.0))
            + score_cfg.get("w_child_local_ifs_support", 0.0) * float(summary.get("mean_child_local_ifs_support", 0.0))
            + score_cfg.get("w_child_local_ifs_coherence", 0.0) * float(summary.get("mean_child_local_ifs_coherence", 0.0))
            + score_cfg.get("w_child_local_ifs_coherence_retention_delta", 0.0)
            * float(summary.get("mean_child_local_ifs_coherence_retention_delta", 0.0))
            + score_cfg.get("w_child_local_ifs_coherence_floor_delta", 0.0)
            * float(summary.get("mean_child_local_ifs_coherence_floor_delta", 0.0))
            + score_cfg.get("w_child_local_ifs_parent_phase_delta", 0.0)
            * float(summary.get("mean_child_local_ifs_parent_phase_delta", 0.0))
            + score_cfg.get("w_child_local_ifs_causal_gate", 0.0)
            * float(summary.get("mean_child_local_ifs_causal_gate", 0.0))
            - score_cfg.get("w_child_local_ifs_causal_retention_loss", 0.0)
            * float(summary.get("mean_child_local_ifs_causal_retention_loss", 0.0))
            - score_cfg.get("w_child_local_ifs_causal_floor_loss", 0.0)
            * float(summary.get("mean_child_local_ifs_causal_floor_loss", 0.0))
            + score_cfg.get("w_child_parent_div", 0.0) * float(summary.get("mean_child_parent_divergence", 0.0))
            + score_cfg.get("w_child_sibling_div", 0.0) * float(summary.get("mean_child_sibling_divergence", 0.0))
            - score_cfg.get("w_defect_without_branch", 0.0) * float(summary.get("mean_defect_without_branch_penalty", 0.0))
            + score_cfg.get("w_branch_delay", 0.0) * float(summary.get("mean_branch_resolution_delay", 0.0))
            + score_cfg.get("w_rel_signature_count", 0.0) * float(summary.get("num_relational_signatures", 0.0))
            + score_cfg.get("w_rel_signature_families", 0.0) * float(summary.get("num_relational_signature_families", 0.0))
            + score_cfg.get("w_rel_signature_confidence", 0.0) * float(summary.get("mean_relational_signature_confidence", 0.0))
            + score_cfg.get("w_rel_signature_q_entropy", 0.0) * float(summary.get("mean_relational_signature_q_entropy", 0.0))
            + score_cfg.get("w_rel_branch_mass", 0.0) * float(summary.get("mean_relational_branch_mass", 0.0))
            - score_cfg.get("w_rel_dominant_family", 0.0) * max(0.0, float(summary.get("dominant_relational_family_share", 0.0)) - score_cfg.get("target_rel_dominant_family_share", 1.0))
        )

        rows.append(
            {
                "name": item["name"],
                "wav_path": item["wav_path"],
                "score": float(score),
                "mse": mse,
                "mae": mae,
                "corr": corr,
                "corr_band_penalty": corr_band,
                "mae_band_penalty": mae_band,
                "corr_regret": corr_regret,
                "mae_regret": mae_regret,
                "row_weight": row_weight,
                **summary,
            }
        )

    def _mean(key: str) -> float:
        if not rows:
            return 0.0
        denom = sum(float(r.get("row_weight", 1.0)) for r in rows)
        if denom <= 0.0:
            return 0.0
        return float(sum(float(r[key]) * float(r.get("row_weight", 1.0)) for r in rows) / denom)

    aggregate_law_library = build_law_token_library(all_law_packets, circle_cfg)
    aggregate_law_stats = _law_library_stats(aggregate_law_library, circle_cfg.qset)
    aggregate_signature_bank = concat_relational_signature_banks(all_signature_banks)
    aggregate_signature_library = build_relational_signature_library(aggregate_signature_bank, circle_cfg)
    aggregate_signature_stats = _signature_library_stats(aggregate_signature_library)
    law_score = (
        - score_cfg.get("w_case_law_family_shortfall", 0.0)
        * max(0.0, score_cfg.get("target_case_law_families", 0.0) - _mean("num_law_families"))
        - score_cfg.get("w_case_law_entropy_shortfall", 0.0)
        * max(0.0, score_cfg.get("target_case_law_entropy", 0.0) - _mean("law_family_entropy"))
        - score_cfg.get("w_case_law_top_q_entropy_shortfall", 0.0)
        * max(0.0, score_cfg.get("target_case_law_top_q_entropy", 0.0) - _mean("law_top_q_entropy"))
        - score_cfg.get("w_case_law_dominant_family", 0.0)
        * max(0.0, _mean("dominant_law_family_share") - score_cfg.get("target_case_law_dominant_family_share", 1.0))
        - score_cfg.get("w_case_law_dominant_q", 0.0)
        * max(0.0, _mean("dominant_law_q_share") - score_cfg.get("target_case_law_dominant_q_share", 1.0))
        - score_cfg.get("w_aggregate_law_family_shortfall", 0.0)
        * max(0.0, score_cfg.get("target_aggregate_law_families", 0.0) - aggregate_law_stats["aggregate_num_law_families"])
        - score_cfg.get("w_aggregate_law_entropy_shortfall", 0.0)
        * max(0.0, score_cfg.get("target_aggregate_law_entropy", 0.0) - aggregate_law_stats["aggregate_law_family_entropy"])
        - score_cfg.get("w_aggregate_law_top_q_entropy_shortfall", 0.0)
        * max(0.0, score_cfg.get("target_aggregate_law_top_q_entropy", 0.0) - aggregate_law_stats["aggregate_law_top_q_entropy"])
        - score_cfg.get("w_aggregate_law_top_q_unique_shortfall", 0.0)
        * max(0.0, score_cfg.get("target_aggregate_law_top_q_unique", 0.0) - aggregate_law_stats["aggregate_num_law_top_q_unique"])
        - score_cfg.get("w_aggregate_law_dominant_family", 0.0)
        * max(0.0, aggregate_law_stats["aggregate_dominant_law_family_share"] - score_cfg.get("target_aggregate_law_dominant_family_share", 1.0))
        - score_cfg.get("w_aggregate_law_dominant_q", 0.0)
        * max(0.0, aggregate_law_stats["aggregate_dominant_law_q_share"] - score_cfg.get("target_aggregate_law_dominant_q_share", 1.0))
    )
    rel_score = (
        - score_cfg.get("w_case_rel_signature_family_shortfall", 0.0)
        * max(0.0, score_cfg.get("target_case_rel_signature_families", 0.0) - _mean("num_relational_signature_families"))
        - score_cfg.get("w_case_rel_signature_entropy_shortfall", 0.0)
        * max(0.0, score_cfg.get("target_case_rel_signature_q_entropy", 0.0) - _mean("mean_relational_signature_q_entropy"))
        - score_cfg.get("w_case_rel_signature_dominant_family", 0.0)
        * max(0.0, _mean("dominant_relational_family_share") - score_cfg.get("target_case_rel_dominant_family_share", 1.0))
        - score_cfg.get("w_aggregate_rel_signature_family_shortfall", 0.0)
        * max(0.0, score_cfg.get("target_aggregate_rel_signature_families", 0.0) - aggregate_signature_stats["aggregate_num_relational_signature_families"])
        - score_cfg.get("w_aggregate_rel_signature_confidence_shortfall", 0.0)
        * max(0.0, score_cfg.get("target_aggregate_rel_signature_confidence", 0.0) - aggregate_signature_stats["aggregate_relational_signature_confidence"])
        - score_cfg.get("w_aggregate_rel_signature_q_entropy_shortfall", 0.0)
        * max(0.0, score_cfg.get("target_aggregate_rel_signature_q_entropy", 0.0) - aggregate_signature_stats["aggregate_relational_q_entropy"])
        - score_cfg.get("w_aggregate_rel_signature_branch_mass_shortfall", 0.0)
        * max(0.0, score_cfg.get("target_aggregate_rel_branch_mass", 0.0) - aggregate_signature_stats["aggregate_relational_branch_mass"])
        - score_cfg.get("w_aggregate_rel_dominant_family", 0.0)
        * max(0.0, aggregate_signature_stats["aggregate_dominant_relational_family_share"] - score_cfg.get("target_aggregate_rel_dominant_family_share", 1.0))
    )
    mean_score = _mean("score") + float(law_score) + float(rel_score)

    return {
        "num_samples": len(rows),
        "mean_score": mean_score,
        "mean_mse": _mean("mse"),
        "mean_mae": _mean("mae"),
        "mean_corr": _mean("corr"),
        "mean_corr_band_penalty": _mean("corr_band_penalty"),
        "mean_mae_band_penalty": _mean("mae_band_penalty"),
        "mean_corr_regret": _mean("corr_regret"),
        "mean_mae_regret": _mean("mae_regret"),
        "mean_major_gain": _mean("major_gain"),
        "mean_residue_drop": _mean("residue_drop"),
        "mean_promotability_gain": _mean("promotability_gain"),
        "mean_dominant_q_share": _mean("dominant_q_share"),
        "mean_q_entropy": _mean("q_entropy"),
        "mean_num_promotions": _mean("num_promotions"),
        "mean_slot2_live_fraction": _mean("mean_slot2_live_fraction") if rows and "mean_slot2_live_fraction" in rows[0] else 0.0,
        "mean_real_branch_fraction": _mean("real_branch_fraction") if rows and "real_branch_fraction" in rows[0] else 0.0,
        "mean_parent_real_branch_fraction": _mean("parent_real_branch_fraction") if rows and "parent_real_branch_fraction" in rows[0] else 0.0,
        "mean_child_real_branch_fraction": _mean("child_real_branch_fraction") if rows and "child_real_branch_fraction" in rows[0] else 0.0,
        "mean_phase_only_excess_branch_fraction": _mean("phase_only_excess_branch_fraction") if rows and "phase_only_excess_branch_fraction" in rows[0] else 0.0,
        "mean_meso_branch_effect": _mean("meso_branch_effect") if rows and "meso_branch_effect" in rows[0] else 0.0,
        "mean_silent_singlepath_fraction": _mean("silent_singlepath_fraction") if rows and "silent_singlepath_fraction" in rows[0] else 0.0,
        "mean_branch_positive_mask": _mean("mean_branch_positive_mask") if rows and "mean_branch_positive_mask" in rows[0] else 0.0,
        "mean_branch_negative_mask": _mean("mean_branch_negative_mask") if rows and "mean_branch_negative_mask" in rows[0] else 0.0,
        "mean_branch_defect": _mean("mean_branch_defect") if rows and "mean_branch_defect" in rows[0] else 0.0,
        "mean_branch_world_grad": _mean("mean_branch_world_grad") if rows and "mean_branch_world_grad" in rows[0] else 0.0,
        "mean_branch_q_disagreement": _mean("mean_branch_q_disagreement") if rows and "mean_branch_q_disagreement" in rows[0] else 0.0,
        "mean_branch_phase_wall": _mean("mean_branch_phase_wall") if rows and "mean_branch_phase_wall" in rows[0] else 0.0,
        "mean_branch_seed_energy": _mean("mean_branch_seed_energy") if rows and "mean_branch_seed_energy" in rows[0] else 0.0,
        "mean_child_world_count": _mean("mean_child_world_count") if rows and "mean_child_world_count" in rows[0] else 0.0,
        "mean_live_child_fraction": _mean("mean_live_child_fraction") if rows and "mean_live_child_fraction" in rows[0] else 0.0,
        "mean_child_age": _mean("mean_child_age") if rows and "mean_child_age" in rows[0] else 0.0,
        "mean_child_writeback_mass": _mean("mean_child_writeback_mass") if rows and "mean_child_writeback_mass" in rows[0] else 0.0,
        "mean_child_writeback_gate_mass": _mean("mean_child_writeback_gate_mass") if rows and "mean_child_writeback_gate_mass" in rows[0] else 0.0,
        "mean_child_phase_writeback_delta_mass": (
            _mean("mean_child_phase_writeback_delta_mass") if rows and "mean_child_phase_writeback_delta_mass" in rows[0] else 0.0
        ),
        "mean_child_parent_phase_writeback_delta_mass": (
            _mean("mean_child_parent_phase_writeback_delta_mass")
            if rows and "mean_child_parent_phase_writeback_delta_mass" in rows[0]
            else 0.0
        ),
        "mean_child_support_writeback_mass": (
            _mean("mean_child_support_writeback_mass") if rows and "mean_child_support_writeback_mass" in rows[0] else 0.0
        ),
        "mean_child_logit_writeback_mass": _mean("mean_child_logit_writeback_mass") if rows and "mean_child_logit_writeback_mass" in rows[0] else 0.0,
        "mean_child_qtrace_writeback_mass": _mean("mean_child_qtrace_writeback_mass") if rows and "mean_child_qtrace_writeback_mass" in rows[0] else 0.0,
        "mean_child_local_ifs_step_count": _mean("mean_child_local_ifs_step_count") if rows and "mean_child_local_ifs_step_count" in rows[0] else 0.0,
        "mean_child_local_ifs_packet_count": _mean("mean_child_local_ifs_packet_count") if rows and "mean_child_local_ifs_packet_count" in rows[0] else 0.0,
        "mean_child_local_ifs_phase_delta": _mean("mean_child_local_ifs_phase_delta") if rows and "mean_child_local_ifs_phase_delta" in rows[0] else 0.0,
        "mean_child_local_ifs_support": _mean("mean_child_local_ifs_support") if rows and "mean_child_local_ifs_support" in rows[0] else 0.0,
        "mean_child_local_ifs_coherence": _mean("mean_child_local_ifs_coherence") if rows and "mean_child_local_ifs_coherence" in rows[0] else 0.0,
        "mean_child_local_ifs_coherence_raw": (
            _mean("mean_child_local_ifs_coherence_raw") if rows and "mean_child_local_ifs_coherence_raw" in rows[0] else 0.0
        ),
        "mean_child_local_ifs_coherence_retention_delta": (
            _mean("mean_child_local_ifs_coherence_retention_delta")
            if rows and "mean_child_local_ifs_coherence_retention_delta" in rows[0]
            else 0.0
        ),
        "mean_child_local_ifs_coherence_floor_delta": (
            _mean("mean_child_local_ifs_coherence_floor_delta")
            if rows and "mean_child_local_ifs_coherence_floor_delta" in rows[0]
            else 0.0
        ),
        "mean_child_local_ifs_parent_phase_delta": (
            _mean("mean_child_local_ifs_parent_phase_delta")
            if rows and "mean_child_local_ifs_parent_phase_delta" in rows[0]
            else 0.0
        ),
        "mean_child_local_ifs_causal_gate": (
            _mean("mean_child_local_ifs_causal_gate") if rows and "mean_child_local_ifs_causal_gate" in rows[0] else 0.0
        ),
        "mean_child_local_ifs_causal_retention_loss": (
            _mean("mean_child_local_ifs_causal_retention_loss")
            if rows and "mean_child_local_ifs_causal_retention_loss" in rows[0]
            else 0.0
        ),
        "mean_child_local_ifs_causal_floor_loss": (
            _mean("mean_child_local_ifs_causal_floor_loss")
            if rows and "mean_child_local_ifs_causal_floor_loss" in rows[0]
            else 0.0
        ),
        "mean_child_parent_divergence": _mean("mean_child_parent_divergence") if rows and "mean_child_parent_divergence" in rows[0] else 0.0,
        "mean_child_sibling_divergence": _mean("mean_child_sibling_divergence") if rows and "mean_child_sibling_divergence" in rows[0] else 0.0,
        "mean_defect_without_branch_penalty": _mean("mean_defect_without_branch_penalty") if rows and "mean_defect_without_branch_penalty" in rows[0] else 0.0,
        "mean_branch_resolution_delay": _mean("mean_branch_resolution_delay") if rows and "mean_branch_resolution_delay" in rows[0] else 0.0,
        "mean_decorative_slot2_fraction": _mean("mean_decorative_slot2_fraction") if rows and "mean_decorative_slot2_fraction" in rows[0] else 0.0,
        "mean_decorative_slot2_low_phase_fraction": _mean("decorative_slot2_low_phase_fraction") if rows and "decorative_slot2_low_phase_fraction" in rows[0] else 0.0,
        "mean_relation_kernel": _mean("mean_relation_kernel") if rows and "mean_relation_kernel" in rows[0] else 0.0,
        "mean_relation_handoff_drive": _mean("mean_relation_handoff_drive") if rows and "mean_relation_handoff_drive" in rows[0] else 0.0,
        "mean_relation_attn_10": _mean("mean_relation_attn_10") if rows and "mean_relation_attn_10" in rows[0] else 0.0,
        "mean_relation_attn_01": _mean("mean_relation_attn_01") if rows and "mean_relation_attn_01" in rows[0] else 0.0,
        "mean_num_law_packets": _mean("num_law_packets") if rows and "num_law_packets" in rows[0] else 0.0,
        "mean_num_law_families": _mean("num_law_families") if rows and "num_law_families" in rows[0] else 0.0,
        "mean_law_family_size": _mean("mean_law_family_size") if rows and "mean_law_family_size" in rows[0] else 0.0,
        "mean_dominant_law_family_share": _mean("dominant_law_family_share") if rows and "dominant_law_family_share" in rows[0] else 0.0,
        "mean_law_family_entropy": _mean("law_family_entropy") if rows and "law_family_entropy" in rows[0] else 0.0,
        "mean_dominant_law_q_share": _mean("dominant_law_q_share") if rows and "dominant_law_q_share" in rows[0] else 0.0,
        "mean_law_top_q_entropy": _mean("law_top_q_entropy") if rows and "law_top_q_entropy" in rows[0] else 0.0,
        "mean_num_law_top_q_unique": _mean("num_law_top_q_unique") if rows and "num_law_top_q_unique" in rows[0] else 0.0,
        "mean_num_relational_signatures": _mean("num_relational_signatures") if rows and "num_relational_signatures" in rows[0] else 0.0,
        "mean_num_relational_signature_families": _mean("num_relational_signature_families") if rows and "num_relational_signature_families" in rows[0] else 0.0,
        "mean_relational_signature_confidence": _mean("mean_relational_signature_confidence") if rows and "mean_relational_signature_confidence" in rows[0] else 0.0,
        "mean_relational_signature_q_entropy": _mean("mean_relational_signature_q_entropy") if rows and "mean_relational_signature_q_entropy" in rows[0] else 0.0,
        "mean_relational_branch_mass": _mean("mean_relational_branch_mass") if rows and "mean_relational_branch_mass" in rows[0] else 0.0,
        "mean_dominant_relational_family_share": _mean("dominant_relational_family_share") if rows and "dominant_relational_family_share" in rows[0] else 0.0,
        "law_score": float(law_score),
        "relational_signature_score": float(rel_score),
        "aggregate_law_library": {
            "num_law_packets": int(aggregate_law_library.get("num_law_packets", 0)),
            "num_law_families": int(aggregate_law_library.get("num_families", 0)),
            "mean_family_size": float(aggregate_law_library.get("mean_family_size", 0.0)),
            "family_counts": [int(f["count"]) for f in aggregate_law_library.get("families", [])],
        },
        "aggregate_relational_signature_library": {
            "num_signatures": int(aggregate_signature_library.get("num_signatures", 0)),
            "num_families": int(aggregate_signature_library.get("num_families", 0)),
            "mean_family_size": float(aggregate_signature_library.get("mean_family_size", 0.0)),
            "family_counts": [int(f["count"]) for f in aggregate_signature_library.get("families", [])],
        },
        **aggregate_law_stats,
        **aggregate_signature_stats,
        "total_weight": float(sum(float(r.get("row_weight", 1.0)) for r in rows)),
        "rows": rows,
    }


def train_circleworld_real_anchor(
    out_dir: Path,
    checkpoint_dir: Path,
    iterations: int,
    population: int,
    elite_count: int,
    seed: int,
    device_name: str,
    clip_seconds: int,
    phase_blend: float,
    init_config_path: Path,
    score_cfg: dict[str, float],
    case_weights: dict[str, float] | None = None,
    extra_train_wavs: list[str] | None = None,
    extra_val_wavs: list[str] | None = None,
    heldout_probe_cfg: dict[str, Any] | None = None,
) -> dict[str, Any]:
    _set_seed(seed)
    device = _safe_device(device_name)
    base_cfg = _load_circle_cfg(init_config_path)
    train_set, val_set = _anchor_dataset(
        device_name=device_name,
        clip_seconds=clip_seconds,
        extra_train_wavs=extra_train_wavs,
        extra_val_wavs=extra_val_wavs,
    )

    baseline_train = _evaluate_cfg_on_dataset(base_cfg, train_set, phase_blend=phase_blend, score_cfg=score_cfg, case_weights=case_weights)
    baseline_val = _evaluate_cfg_on_dataset(base_cfg, val_set, phase_blend=phase_blend, score_cfg=score_cfg, case_weights=case_weights)
    baseline_train_rows = {row["name"]: row for row in baseline_train["rows"]}
    baseline_val_rows = {row["name"]: row for row in baseline_val["rows"]}

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
        "relation_attention_gain": float(base_cfg.relation_attention_gain),
        "relation_attention_sharpness": float(base_cfg.relation_attention_sharpness),
        "relation_value_gain": float(base_cfg.relation_value_gain),
        "relation_support_gain": float(base_cfg.relation_support_gain),
        "relation_logit_gain": float(base_cfg.relation_logit_gain),
        "relation_qtrace_gain": float(base_cfg.relation_qtrace_gain),
        "relation_residual_mix": float(base_cfg.relation_residual_mix),
        "child_spawn_threshold": float(base_cfg.child_spawn_threshold),
        "child_max_worlds": float(base_cfg.child_max_worlds),
        "child_min_age_for_writeback": float(base_cfg.child_min_age_for_writeback),
        "child_support_window": float(base_cfg.child_support_window),
        "child_support_decay": float(base_cfg.child_support_decay),
        "child_survival_coherence_weight": float(base_cfg.child_survival_coherence_weight),
        "child_survival_qtrace_weight": float(base_cfg.child_survival_qtrace_weight),
        "child_survival_residue_penalty": float(base_cfg.child_survival_residue_penalty),
        "child_writeback_gain": float(base_cfg.child_writeback_gain),
        "child_writeback_rank": float(base_cfg.child_writeback_rank),
        "child_writeback_temperature": float(base_cfg.child_writeback_temperature),
        "child_writeback_budget": float(base_cfg.child_writeback_budget),
        "child_parent_mix": float(base_cfg.child_parent_mix),
        "child_parent_mix_early": float(base_cfg.child_parent_mix_early),
        "child_kill_threshold": float(base_cfg.child_kill_threshold),
        "child_local_ifs_enabled": bool(base_cfg.child_local_ifs_enabled),
        "child_local_steps": float(base_cfg.child_local_steps),
        "child_local_support_only": bool(base_cfg.child_local_support_only),
        **_child_local_coherence_values(base_cfg),
        "instability_mask_gain": float(base_cfg.instability_mask_gain),
        "defect_phase_gain": float(base_cfg.defect_phase_gain),
        "defect_q_gain": float(base_cfg.defect_q_gain),
        "defect_residue_gain": float(base_cfg.defect_residue_gain),
        "defect_sharpness_gain": float(base_cfg.defect_sharpness_gain),
        "defect_world_grad_gain": float(base_cfg.defect_world_grad_gain),
        "instability_seed_scale": float(base_cfg.instability_seed_scale),
        "instability_support_gain": float(base_cfg.instability_support_gain),
        "instability_logit_gain": float(base_cfg.instability_logit_gain),
        "law_packet_merge_threshold": float(base_cfg.law_packet_merge_threshold),
        "law_packet_min_score": float(base_cfg.law_packet_min_score),
        "law_packet_topk_families": float(base_cfg.law_packet_topk_families),
        **_phase_law_control_values(base_cfg),
        "soft_matryoshka_enabled": bool(base_cfg.soft_matryoshka_enabled),
        "branching_mode": str(base_cfg.branching_mode),
        "branch_law_version": str(base_cfg.branch_law_version),
        "branch_kernel_version": str(base_cfg.branch_kernel_version),
    }
    mean.update(_child_writeback_control_values(base_cfg))
    initial_mean = deepcopy(mean)
    std = {
        "q_weights": [0.15 for _ in base_cfg.q_weights],
        "promotion_threshold": 0.08,
        "max_promotions": 1.0,
        "child_law_gain": 0.10,
        "attack_window": 1.5,
        "persistence_momentum": 0.05,
        "matryoshka_rank": 4.0,
        "slow_persistence": 0.03,
        "fast_persistence": 0.05,
        "persistence_curve": 0.20,
        "slow_write_scale": 0.03,
        "fast_write_scale": 0.12,
        "write_curve": 0.20,
        "prefix_coarse_weight": 0.05,
        "prefix_mid_weight": 0.05,
        "split_pressure": 0.05,
        "split_seed_scale": 0.03,
        "split_support_gain": 0.06,
        "merge_pressure": 0.04,
        "merge_phase_tol": 0.03,
        "merge_support_overlap_weight": 0.05,
        "survival_coherence_weight": 0.04,
        "survival_arc_weight": 0.04,
        "survival_qtrace_weight": 0.04,
        "survival_residue_penalty": 0.04,
        "collapse_sharpness": 0.08,
        "support_decay": 0.03,
        "support_spread": 0.8,
        "support_overlap_penalty": 0.03,
        "anti_fixation_weight": 0.03,
        "readout_temperature": 0.04,
        "qtrace_momentum": 0.03,
        "mask_neighborhood": 0.5,
        "topology_mask_gain": 0.03,
        "complexity_mask_gain": 0.03,
        "context_mask_gain": 0.03,
        "contrastive_mask_gain": 0.03,
        "aux_mask_suppression": 0.03,
        "relation_attention_gain": 0.03,
        "relation_attention_sharpness": 0.05,
        "relation_value_gain": 0.03,
        "relation_support_gain": 0.03,
        "relation_logit_gain": 0.03,
        "relation_qtrace_gain": 0.03,
        "relation_residual_mix": 0.03,
        "child_spawn_threshold": 0.03,
        "child_max_worlds": 0.5,
        "child_min_age_for_writeback": 0.4,
        "child_support_window": 1.5,
        "child_support_decay": 0.02,
        "child_survival_coherence_weight": 0.03,
        "child_survival_qtrace_weight": 0.03,
        "child_survival_residue_penalty": 0.03,
        "child_writeback_gain": 0.03,
        "child_writeback_rank": 1.5,
        "child_writeback_temperature": 0.03,
        "child_writeback_budget": 0.08,
        "child_parent_mix": 0.02,
        "child_parent_mix_early": 0.015,
        "child_kill_threshold": 0.02,
        **_CHILD_LOCAL_COHERENCE_CONTROL_STDS,
        "instability_mask_gain": 0.03,
        "defect_phase_gain": 0.03,
        "defect_q_gain": 0.03,
        "defect_residue_gain": 0.02,
        "defect_sharpness_gain": 0.02,
        "defect_world_grad_gain": 0.02,
        "instability_seed_scale": 0.02,
        "instability_support_gain": 0.03,
        "instability_logit_gain": 0.03,
        "law_packet_merge_threshold": 0.02,
        "law_packet_min_score": 0.03,
        "law_packet_topk_families": 1.0,
    }
    std.update(_PHASE_LAW_CONTROL_STDS)
    std.update(_CHILD_WRITEBACK_CONTROL_STDS)

    history: list[dict[str, Any]] = []
    best_state: dict[str, Any] | None = None
    best_passing_state: dict[str, Any] | None = None
    best_agreement_state: dict[str, Any] | None = None
    best_nested_pending_state: dict[str, Any] | None = None
    agreement_candidates: list[dict[str, Any]] = []
    rng = random.Random(seed)
    selection_nested_top_k = max(0, int((heldout_probe_cfg or {}).get("selection_nested_top_k", 0)))
    nested_probe_top_k = max(0, int((heldout_probe_cfg or {}).get("nested_probe_top_k", 0)))
    nested_cases = _load_nested_cases(heldout_probe_cfg) if max(nested_probe_top_k, selection_nested_top_k) > 0 else []

    for step in range(iterations):
        candidates: list[dict[str, Any]] = []
        candidate_specs: list[tuple[str, dict[str, Any]]] = []
        if bool((heldout_probe_cfg or {}).get("selection_include_init_candidate", False)):
            candidate_specs.append(("init_checkpoint", deepcopy(initial_mean)))
        if bool((heldout_probe_cfg or {}).get("selection_include_mean_candidate", False)) and (
            step > 0 or not candidate_specs
        ):
            candidate_specs.append(("cem_mean", deepcopy(mean)))
        for _ in range(population):
            candidate_specs.append(("cem_sample", _candidate_from_mean_std(mean, std, rng)))

        for candidate_kind, cand in candidate_specs:
            cfg = _materialize_cfg(base_cfg, cand)
            train_eval = _evaluate_cfg_on_dataset(
                cfg,
                train_set,
                phase_blend=phase_blend,
                score_cfg=score_cfg,
                case_weights=case_weights,
                baseline_rows=baseline_train_rows,
            )
            val_eval = _evaluate_cfg_on_dataset(
                cfg,
                val_set,
                phase_blend=phase_blend,
                score_cfg=score_cfg,
                case_weights=case_weights,
                baseline_rows=baseline_val_rows,
            )
            transfer_probe = _evaluate_transfer_probe_repeated(
                cfg,
                device_name=device_name,
                score_cfg=score_cfg,
                probe_cfg=heldout_probe_cfg,
            )
            row = {
                "iteration": step,
                "candidate_kind": candidate_kind,
                "selection_prefer_nested_probe": bool((heldout_probe_cfg or {}).get("selection_prefer_nested_probe", False)),
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
                    "relation_attention_gain": cfg.relation_attention_gain,
                    "relation_attention_sharpness": cfg.relation_attention_sharpness,
                    "relation_value_gain": cfg.relation_value_gain,
                    "relation_support_gain": cfg.relation_support_gain,
                    "relation_logit_gain": cfg.relation_logit_gain,
                    "relation_qtrace_gain": cfg.relation_qtrace_gain,
                    "relation_residual_mix": cfg.relation_residual_mix,
                    "child_spawn_threshold": cfg.child_spawn_threshold,
                    "child_max_worlds": cfg.child_max_worlds,
                    "child_min_age_for_writeback": cfg.child_min_age_for_writeback,
                    "child_support_window": cfg.child_support_window,
                    "child_support_decay": cfg.child_support_decay,
                    "child_survival_coherence_weight": cfg.child_survival_coherence_weight,
                    "child_survival_qtrace_weight": cfg.child_survival_qtrace_weight,
                    "child_survival_residue_penalty": cfg.child_survival_residue_penalty,
                    "child_writeback_gain": cfg.child_writeback_gain,
                    "child_writeback_rank": cfg.child_writeback_rank,
                    "child_writeback_temperature": cfg.child_writeback_temperature,
                    "child_writeback_budget": cfg.child_writeback_budget,
                    "child_parent_mix": cfg.child_parent_mix,
                    "child_parent_mix_early": cfg.child_parent_mix_early,
                    **_child_writeback_control_values(cfg),
                    "child_kill_threshold": cfg.child_kill_threshold,
                    "child_local_ifs_enabled": cfg.child_local_ifs_enabled,
                    "child_local_steps": cfg.child_local_steps,
                    "child_local_support_only": cfg.child_local_support_only,
                    **_child_local_coherence_values(cfg),
                    "instability_mask_gain": cfg.instability_mask_gain,
                    "defect_phase_gain": cfg.defect_phase_gain,
                    "defect_q_gain": cfg.defect_q_gain,
                    "defect_residue_gain": cfg.defect_residue_gain,
                    "defect_sharpness_gain": cfg.defect_sharpness_gain,
                    "defect_world_grad_gain": cfg.defect_world_grad_gain,
                    "instability_seed_scale": cfg.instability_seed_scale,
                    "instability_support_gain": cfg.instability_support_gain,
                    "instability_logit_gain": cfg.instability_logit_gain,
                    "law_packet_merge_threshold": cfg.law_packet_merge_threshold,
                    "law_packet_min_score": cfg.law_packet_min_score,
                    "law_packet_topk_families": cfg.law_packet_topk_families,
                    "branching_mode": cfg.branching_mode,
                    "branch_law_version": cfg.branch_law_version,
                    "branch_kernel_version": cfg.branch_kernel_version,
                },
                "train": train_eval,
                "val": val_eval,
                "transfer_probe": transfer_probe,
            }
            row["val"]["mean_score_with_transfer"] = float(row["val"]["mean_score"] + transfer_probe["score"])
            row["selection_gate"] = _selection_gate_status(
                transfer_probe=transfer_probe,
                probe_cfg=heldout_probe_cfg,
                score_cfg=score_cfg,
            )
            row["nested_probe_score"] = 0.0
            candidates.append(row)

        candidates.sort(key=_selection_sort_key, reverse=True)
        if nested_probe_top_k > 0:
            nested_pool = sorted(candidates, key=_selection_sort_key, reverse=True)
            deduped_nested: list[dict[str, Any]] = []
            seen_nested_cfgs: set[str] = set()
            if bool((heldout_probe_cfg or {}).get("selection_force_guard_candidates_nested", False)):
                for row in candidates:
                    if str(row.get("candidate_kind", "")) not in {"init_checkpoint", "cem_mean"}:
                        continue
                    cfg_key = _selection_candidate_key(row)
                    if cfg_key in seen_nested_cfgs:
                        continue
                    seen_nested_cfgs.add(cfg_key)
                    deduped_nested.append(row)
                    if len(deduped_nested) >= nested_probe_top_k:
                        break
            for row in nested_pool:
                if len(deduped_nested) >= nested_probe_top_k:
                    break
                cfg_key = _selection_candidate_key(row)
                if cfg_key in seen_nested_cfgs:
                    continue
                seen_nested_cfgs.add(cfg_key)
                deduped_nested.append(row)

            for idx, row in enumerate(deduped_nested):
                nested_cfg = _materialize_cfg(base_cfg, row["materialized_cfg"])
                probe_dir = out_dir / "_nested_probe" / f"iter_{step:02d}" / f"candidate_{idx:02d}"
                probe_dir.mkdir(parents=True, exist_ok=True)
                probe_cfg_path = probe_dir / "candidate_config.json"
                probe_cfg_path.write_text(json.dumps(_cfg_payload(nested_cfg), indent=2), encoding="utf-8")
                nested_eval = evaluate_nested_commitment_report(
                    config_path=probe_cfg_path,
                    out_dir=probe_dir,
                    device_name=device_name,
                    depth=int((heldout_probe_cfg or {}).get("selection_nested_depth", 3)),
                    fork_depth=int((heldout_probe_cfg or {}).get("selection_nested_fork_depth", 1)),
                    fork_selector=str((heldout_probe_cfg or {}).get("selection_nested_fork_selector", "first_live_child")),
                    live_child_threshold=float((heldout_probe_cfg or {}).get("selection_nested_live_child_threshold", 0.05)),
                    mode=str((heldout_probe_cfg or {}).get("selection_nested_mode", "active_packets")),
                    cases=nested_cases,
                )
                nested_eval["enabled"] = True
                row["nested_probe"] = nested_eval
                row["nested_probe_score"] = _nested_probe_score(
                    nested_eval=nested_eval,
                    score_cfg=score_cfg,
                    probe_cfg=heldout_probe_cfg,
                )
                row["val"]["mean_score_with_nested_probe"] = float(
                    row["val"].get("mean_score_with_transfer", row["val"]["mean_score"]) + row["nested_probe_score"]
                )

            candidates.sort(key=_selection_sort_key, reverse=True)
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
            "relation_attention_gain": sum(row["materialized_cfg"]["relation_attention_gain"] for row in elites) / len(elites),
            "relation_attention_sharpness": sum(row["materialized_cfg"]["relation_attention_sharpness"] for row in elites) / len(elites),
            "relation_value_gain": sum(row["materialized_cfg"]["relation_value_gain"] for row in elites) / len(elites),
            "relation_support_gain": sum(row["materialized_cfg"]["relation_support_gain"] for row in elites) / len(elites),
            "relation_logit_gain": sum(row["materialized_cfg"]["relation_logit_gain"] for row in elites) / len(elites),
            "relation_qtrace_gain": sum(row["materialized_cfg"]["relation_qtrace_gain"] for row in elites) / len(elites),
            "relation_residual_mix": sum(row["materialized_cfg"]["relation_residual_mix"] for row in elites) / len(elites),
            "child_spawn_threshold": sum(row["materialized_cfg"]["child_spawn_threshold"] for row in elites) / len(elites),
            "child_max_worlds": sum(row["materialized_cfg"]["child_max_worlds"] for row in elites) / len(elites),
            "child_min_age_for_writeback": sum(row["materialized_cfg"]["child_min_age_for_writeback"] for row in elites) / len(elites),
            "child_support_window": sum(row["materialized_cfg"]["child_support_window"] for row in elites) / len(elites),
            "child_support_decay": sum(row["materialized_cfg"]["child_support_decay"] for row in elites) / len(elites),
            "child_survival_coherence_weight": sum(row["materialized_cfg"]["child_survival_coherence_weight"] for row in elites) / len(elites),
            "child_survival_qtrace_weight": sum(row["materialized_cfg"]["child_survival_qtrace_weight"] for row in elites) / len(elites),
            "child_survival_residue_penalty": sum(row["materialized_cfg"]["child_survival_residue_penalty"] for row in elites) / len(elites),
            "child_writeback_gain": sum(row["materialized_cfg"]["child_writeback_gain"] for row in elites) / len(elites),
            "child_writeback_rank": sum(row["materialized_cfg"]["child_writeback_rank"] for row in elites) / len(elites),
            "child_writeback_temperature": sum(row["materialized_cfg"]["child_writeback_temperature"] for row in elites) / len(elites),
            "child_writeback_budget": sum(row["materialized_cfg"]["child_writeback_budget"] for row in elites) / len(elites),
            "child_parent_mix": sum(row["materialized_cfg"]["child_parent_mix"] for row in elites) / len(elites),
            "child_parent_mix_early": sum(row["materialized_cfg"]["child_parent_mix_early"] for row in elites) / len(elites),
            "child_kill_threshold": sum(row["materialized_cfg"]["child_kill_threshold"] for row in elites) / len(elites),
            "child_local_ifs_enabled": mean.get("child_local_ifs_enabled", False),
            "child_local_steps": mean.get("child_local_steps", 1),
            "child_local_support_only": mean.get("child_local_support_only", True),
            "child_local_coherence_retention_enabled": mean.get("child_local_coherence_retention_enabled", False),
            "instability_mask_gain": sum(row["materialized_cfg"]["instability_mask_gain"] for row in elites) / len(elites),
            "defect_phase_gain": sum(row["materialized_cfg"]["defect_phase_gain"] for row in elites) / len(elites),
            "defect_q_gain": sum(row["materialized_cfg"]["defect_q_gain"] for row in elites) / len(elites),
            "defect_residue_gain": sum(row["materialized_cfg"]["defect_residue_gain"] for row in elites) / len(elites),
            "defect_sharpness_gain": sum(row["materialized_cfg"]["defect_sharpness_gain"] for row in elites) / len(elites),
            "defect_world_grad_gain": sum(row["materialized_cfg"]["defect_world_grad_gain"] for row in elites) / len(elites),
            "instability_seed_scale": sum(row["materialized_cfg"]["instability_seed_scale"] for row in elites) / len(elites),
            "instability_support_gain": sum(row["materialized_cfg"]["instability_support_gain"] for row in elites) / len(elites),
            "instability_logit_gain": sum(row["materialized_cfg"]["instability_logit_gain"] for row in elites) / len(elites),
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
                _elite_std([row["materialized_cfg"]["q_weights"][i] for row in elites], 0.03)
                for i in range(len(base_cfg.q_weights))
            ],
            "promotion_threshold": _elite_std([row["materialized_cfg"]["promotion_threshold"] for row in elites], 0.02),
            "max_promotions": _elite_std([float(row["materialized_cfg"]["max_promotions"]) for row in elites], 0.5),
            "child_law_gain": _elite_std([row["materialized_cfg"]["child_law_gain"] for row in elites], 0.02),
            "attack_window": _elite_std([float(row["materialized_cfg"]["attack_window"]) for row in elites], 0.5),
            "persistence_momentum": _elite_std([row["materialized_cfg"]["persistence_momentum"] for row in elites], 0.02),
            "matryoshka_rank": _elite_std([float(row["materialized_cfg"]["matryoshka_rank"]) for row in elites], 2.0),
            "slow_persistence": _elite_std([row["materialized_cfg"]["slow_persistence"] for row in elites], 0.015),
            "fast_persistence": _elite_std([row["materialized_cfg"]["fast_persistence"] for row in elites], 0.02),
            "persistence_curve": _elite_std([row["materialized_cfg"]["persistence_curve"] for row in elites], 0.08),
            "slow_write_scale": _elite_std([row["materialized_cfg"]["slow_write_scale"] for row in elites], 0.015),
            "fast_write_scale": _elite_std([row["materialized_cfg"]["fast_write_scale"] for row in elites], 0.04),
            "write_curve": _elite_std([row["materialized_cfg"]["write_curve"] for row in elites], 0.08),
            "prefix_coarse_weight": _elite_std([row["materialized_cfg"]["prefix_coarse_weight"] for row in elites], 0.02),
            "prefix_mid_weight": _elite_std([row["materialized_cfg"]["prefix_mid_weight"] for row in elites], 0.02),
            "split_pressure": _elite_std([row["materialized_cfg"]["split_pressure"] for row in elites], 0.02),
            "split_seed_scale": _elite_std([row["materialized_cfg"]["split_seed_scale"] for row in elites], 0.015),
            "split_support_gain": _elite_std([row["materialized_cfg"]["split_support_gain"] for row in elites], 0.03),
            "merge_pressure": _elite_std([row["materialized_cfg"]["merge_pressure"] for row in elites], 0.015),
            "merge_phase_tol": _elite_std([row["materialized_cfg"]["merge_phase_tol"] for row in elites], 0.015),
            "merge_support_overlap_weight": _elite_std([row["materialized_cfg"]["merge_support_overlap_weight"] for row in elites], 0.02),
            "survival_coherence_weight": _elite_std([row["materialized_cfg"]["survival_coherence_weight"] for row in elites], 0.02),
            "survival_arc_weight": _elite_std([row["materialized_cfg"]["survival_arc_weight"] for row in elites], 0.02),
            "survival_qtrace_weight": _elite_std([row["materialized_cfg"]["survival_qtrace_weight"] for row in elites], 0.015),
            "survival_residue_penalty": _elite_std([row["materialized_cfg"]["survival_residue_penalty"] for row in elites], 0.015),
            "collapse_sharpness": _elite_std([row["materialized_cfg"]["collapse_sharpness"] for row in elites], 0.04),
            "support_decay": _elite_std([row["materialized_cfg"]["support_decay"] for row in elites], 0.015),
            "support_spread": _elite_std([float(row["materialized_cfg"]["support_spread"]) for row in elites], 0.5),
            "support_overlap_penalty": _elite_std([row["materialized_cfg"]["support_overlap_penalty"] for row in elites], 0.015),
            "anti_fixation_weight": _elite_std([row["materialized_cfg"]["anti_fixation_weight"] for row in elites], 0.015),
            "readout_temperature": _elite_std([row["materialized_cfg"]["readout_temperature"] for row in elites], 0.015),
            "qtrace_momentum": _elite_std([row["materialized_cfg"]["qtrace_momentum"] for row in elites], 0.015),
            "mask_neighborhood": _elite_std([float(row["materialized_cfg"]["mask_neighborhood"]) for row in elites], 0.5),
            "topology_mask_gain": _elite_std([row["materialized_cfg"]["topology_mask_gain"] for row in elites], 0.015),
            "complexity_mask_gain": _elite_std([row["materialized_cfg"]["complexity_mask_gain"] for row in elites], 0.015),
            "context_mask_gain": _elite_std([row["materialized_cfg"]["context_mask_gain"] for row in elites], 0.015),
            "contrastive_mask_gain": _elite_std([row["materialized_cfg"]["contrastive_mask_gain"] for row in elites], 0.015),
            "aux_mask_suppression": _elite_std([row["materialized_cfg"]["aux_mask_suppression"] for row in elites], 0.015),
            "relation_attention_gain": _elite_std([row["materialized_cfg"]["relation_attention_gain"] for row in elites], 0.015),
            "relation_attention_sharpness": _elite_std([row["materialized_cfg"]["relation_attention_sharpness"] for row in elites], 0.02),
            "relation_value_gain": _elite_std([row["materialized_cfg"]["relation_value_gain"] for row in elites], 0.015),
            "relation_support_gain": _elite_std([row["materialized_cfg"]["relation_support_gain"] for row in elites], 0.015),
            "relation_logit_gain": _elite_std([row["materialized_cfg"]["relation_logit_gain"] for row in elites], 0.015),
            "relation_qtrace_gain": _elite_std([row["materialized_cfg"]["relation_qtrace_gain"] for row in elites], 0.015),
            "relation_residual_mix": _elite_std([row["materialized_cfg"]["relation_residual_mix"] for row in elites], 0.015),
            "child_spawn_threshold": _elite_std([row["materialized_cfg"]["child_spawn_threshold"] for row in elites], 0.015),
            "child_max_worlds": _elite_std([float(row["materialized_cfg"]["child_max_worlds"]) for row in elites], 0.5),
            "child_min_age_for_writeback": _elite_std([float(row["materialized_cfg"]["child_min_age_for_writeback"]) for row in elites], 0.4),
            "child_support_window": _elite_std([float(row["materialized_cfg"]["child_support_window"]) for row in elites], 0.75),
            "child_support_decay": _elite_std([row["materialized_cfg"]["child_support_decay"] for row in elites], 0.01),
            "child_survival_coherence_weight": _elite_std([row["materialized_cfg"]["child_survival_coherence_weight"] for row in elites], 0.015),
            "child_survival_qtrace_weight": _elite_std([row["materialized_cfg"]["child_survival_qtrace_weight"] for row in elites], 0.015),
            "child_survival_residue_penalty": _elite_std([row["materialized_cfg"]["child_survival_residue_penalty"] for row in elites], 0.015),
            "child_writeback_gain": _elite_std([row["materialized_cfg"]["child_writeback_gain"] for row in elites], 0.015),
            "child_writeback_rank": _elite_std([float(row["materialized_cfg"]["child_writeback_rank"]) for row in elites], 1.0),
            "child_writeback_temperature": _elite_std([row["materialized_cfg"]["child_writeback_temperature"] for row in elites], 0.015),
            "child_writeback_budget": _elite_std([row["materialized_cfg"]["child_writeback_budget"] for row in elites], 0.04),
            "child_parent_mix": _elite_std([row["materialized_cfg"]["child_parent_mix"] for row in elites], 0.01),
            "child_parent_mix_early": _elite_std([row["materialized_cfg"]["child_parent_mix_early"] for row in elites], 0.008),
            "child_kill_threshold": _elite_std([row["materialized_cfg"]["child_kill_threshold"] for row in elites], 0.01),
            **_CHILD_LOCAL_COHERENCE_CONTROL_STDS,
            "instability_mask_gain": _elite_std([row["materialized_cfg"]["instability_mask_gain"] for row in elites], 0.015),
            "defect_phase_gain": _elite_std([row["materialized_cfg"]["defect_phase_gain"] for row in elites], 0.015),
            "defect_q_gain": _elite_std([row["materialized_cfg"]["defect_q_gain"] for row in elites], 0.015),
            "defect_residue_gain": _elite_std([row["materialized_cfg"]["defect_residue_gain"] for row in elites], 0.01),
            "defect_sharpness_gain": _elite_std([row["materialized_cfg"]["defect_sharpness_gain"] for row in elites], 0.01),
            "defect_world_grad_gain": _elite_std([row["materialized_cfg"]["defect_world_grad_gain"] for row in elites], 0.01),
            "instability_seed_scale": _elite_std([row["materialized_cfg"]["instability_seed_scale"] for row in elites], 0.01),
            "instability_support_gain": _elite_std([row["materialized_cfg"]["instability_support_gain"] for row in elites], 0.015),
            "instability_logit_gain": _elite_std([row["materialized_cfg"]["instability_logit_gain"] for row in elites], 0.015),
            "law_packet_merge_threshold": _elite_std([row["materialized_cfg"]["law_packet_merge_threshold"] for row in elites], 0.01),
            "law_packet_min_score": _elite_std([row["materialized_cfg"]["law_packet_min_score"] for row in elites], 0.015),
            "law_packet_topk_families": _elite_std([float(row["materialized_cfg"]["law_packet_topk_families"]) for row in elites], 1.0),
        }
        for key, floor in _PHASE_LAW_CONTROL_STDS.items():
            std[key] = _elite_std([float(row["materialized_cfg"][key]) for row in elites], floor * 0.5)
        for key, floor in _CHILD_WRITEBACK_CONTROL_STDS.items():
            std[key] = _elite_std([row["materialized_cfg"][key] for row in elites], floor * 0.5)
        for key, floor in _CHILD_LOCAL_COHERENCE_CONTROL_STDS.items():
            std[key] = _elite_std([row["materialized_cfg"][key] for row in elites], floor * 0.5)

        candidate_best = deepcopy(candidates[0])
        if best_state is None or _selection_sort_key(candidate_best) > _selection_sort_key(best_state):
            best_state = candidate_best
        passing_candidates = [row for row in candidates if row.get("selection_gate", {}).get("passed", True)]
        if passing_candidates:
            candidate_best_passing = deepcopy(passing_candidates[0])
            if best_passing_state is None or _selection_sort_key(candidate_best_passing) > _selection_sort_key(best_passing_state):
                best_passing_state = candidate_best_passing

    assert best_state is not None
    hard_gate_enabled = bool(score_cfg.get("use_hard_selection_gate", 0.0))
    agreement_top_k = max(0, int((heldout_probe_cfg or {}).get("selection_heldout_top_k", 0)))
    nested_top_k = max(0, int((heldout_probe_cfg or {}).get("selection_nested_top_k", 0)))
    nested_eval_top_k = max(
        nested_top_k,
        max(0, int((heldout_probe_cfg or {}).get("selection_nested_eval_top_k", 0))),
    )
    require_heldout_agreement = bool((heldout_probe_cfg or {}).get("selection_require_heldout_agreement", False))
    require_nested_response = bool((heldout_probe_cfg or {}).get("selection_require_nested_response", False))
    allow_missing_nested_fallback = bool((heldout_probe_cfg or {}).get("selection_allow_missing_nested_fallback", False))
    if agreement_top_k > 0 and heldout_probe_cfg and heldout_probe_cfg.get("enabled", False):
        candidate_pool = sorted(history, key=_selection_sort_key, reverse=True)
        deduped_pool: list[dict[str, Any]] = []
        seen_cfgs: set[str] = set()
        for row in candidate_pool:
            cfg_key = _selection_candidate_key(row)
            if cfg_key in seen_cfgs:
                continue
            seen_cfgs.add(cfg_key)
            deduped_pool.append(deepcopy(row))
            if len(deduped_pool) >= agreement_top_k:
                break

        for row in deduped_pool:
            agreement_cfg = _materialize_cfg(base_cfg, row["materialized_cfg"])
            heldout_eval = _evaluate_selection_heldout(
                agreement_cfg,
                device_name=device_name,
                probe_cfg=heldout_probe_cfg,
            )
            heldout_gate = _selection_heldout_gate_status(
                heldout_eval=heldout_eval,
                probe_cfg=heldout_probe_cfg,
                score_cfg=score_cfg,
            )
            heldout_agreement = _selection_probe_heldout_agreement(
                transfer_probe=row["transfer_probe"],
                heldout_eval=heldout_eval,
            )
            row["selection_heldout"] = heldout_eval
            row["selection_heldout_gate"] = heldout_gate
            row["selection_heldout_agreement"] = heldout_agreement
            agreement_candidates.append(row)

        agreement_candidates.sort(key=_selection_heldout_sort_key, reverse=True)
        for idx, row in enumerate(agreement_candidates[:nested_eval_top_k]):
            agreement_cfg = _materialize_cfg(base_cfg, row["materialized_cfg"])
            candidate_dir = out_dir / "_selection_nested" / f"candidate_{idx:02d}"
            candidate_dir.mkdir(parents=True, exist_ok=True)
            candidate_cfg_path = candidate_dir / "candidate_config.json"
            candidate_payload = _cfg_payload(agreement_cfg)
            candidate_cfg_path.write_text(json.dumps(candidate_payload, indent=2), encoding="utf-8")
            nested_eval = evaluate_nested_commitment_report(
                config_path=candidate_cfg_path,
                out_dir=candidate_dir,
                device_name=device_name,
                depth=int((heldout_probe_cfg or {}).get("selection_nested_depth", 3)),
                fork_depth=int((heldout_probe_cfg or {}).get("selection_nested_fork_depth", 1)),
                fork_selector=str((heldout_probe_cfg or {}).get("selection_nested_fork_selector", "first_live_child")),
                live_child_threshold=float((heldout_probe_cfg or {}).get("selection_nested_live_child_threshold", 0.05)),
                mode=str((heldout_probe_cfg or {}).get("selection_nested_mode", "active_packets")),
                cases=nested_cases,
            )
            nested_eval["enabled"] = True
            row["selection_nested"] = nested_eval
            row["selection_nested_gate"] = _selection_nested_gate_status(
                nested_eval=nested_eval,
                probe_cfg=heldout_probe_cfg,
            )

        agreement_candidates.sort(key=_selection_heldout_sort_key, reverse=True)
        heldout_pass = [
            row
            for row in agreement_candidates
            if row.get("selection_heldout_gate", {}).get("passed", False)
            and (
                not require_nested_response
                or row.get("selection_nested_gate", {}).get("passed", False)
            )
        ]
        heldout_missing_nested_pass = []
        if require_nested_response and allow_missing_nested_fallback:
            heldout_missing_nested_pass = [
                row
                for row in agreement_candidates
                if row.get("selection_heldout_gate", {}).get("passed", False)
                and not _selection_nested_is_explicit_failure(row)
            ]
        if heldout_pass:
            best_agreement_state = deepcopy(heldout_pass[0])
        elif heldout_missing_nested_pass:
            best_nested_pending_state = deepcopy(heldout_missing_nested_pass[0])

    selected_state = best_state
    selection_source = "best_candidate"
    if hard_gate_enabled:
        if best_agreement_state is not None:
            selected_state = best_agreement_state
            selection_source = "best_agreement_candidate"
        elif require_nested_response and allow_missing_nested_fallback and best_nested_pending_state is not None:
            selected_state = best_nested_pending_state
            selection_source = "best_agreement_candidate_nested_pending"
        elif (require_heldout_agreement and agreement_top_k > 0) or (require_nested_response and nested_top_k > 0):
            selected_state = None
            selection_source = "init_config_fallback_nested_or_agreement_required"
        elif best_passing_state is not None:
            selected_state = best_passing_state
            selection_source = "best_passing_candidate"
        else:
            selected_state = None
            selection_source = "init_config_fallback"
    selection_fallback_diagnostics = _selection_fallback_diagnostics(
        selection_source=selection_source,
        selected_state=selected_state,
        best_state=best_state,
        best_passing_state=best_passing_state,
        best_agreement_state=best_agreement_state,
        best_nested_pending_state=best_nested_pending_state,
        agreement_candidates=agreement_candidates,
        hard_gate_enabled=hard_gate_enabled,
        require_heldout_agreement=require_heldout_agreement,
        require_nested_response=require_nested_response,
        allow_missing_nested_fallback=allow_missing_nested_fallback,
        agreement_top_k=agreement_top_k,
        nested_top_k=nested_top_k,
        nested_eval_top_k=nested_eval_top_k,
    )
    if selected_state is not None:
        best_cfg = _materialize_cfg(base_cfg, selected_state["materialized_cfg"])
    else:
        best_cfg = deepcopy(base_cfg)
    best_train = _evaluate_cfg_on_dataset(
        best_cfg,
        train_set,
        phase_blend=phase_blend,
        score_cfg=score_cfg,
        case_weights=case_weights,
        baseline_rows=baseline_train_rows,
    )
    best_val = _evaluate_cfg_on_dataset(
        best_cfg,
        val_set,
        phase_blend=phase_blend,
        score_cfg=score_cfg,
        case_weights=case_weights,
        baseline_rows=baseline_val_rows,
    )
    best_transfer_probe = _evaluate_transfer_probe_repeated(
        best_cfg,
        device_name=device_name,
        score_cfg=score_cfg,
        probe_cfg=heldout_probe_cfg,
    )
    best_val["mean_score_with_transfer"] = float(best_val["mean_score"] + best_transfer_probe["score"])

    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = checkpoint_dir / "circleworld_real_anchor_config_cem_v1.json"
    history_path = out_dir / "search_history.json"
    summary_path = out_dir / "train_summary.json"

    payload = {
        "runtime": "circleworld_proto",
        "schema": "circleworld_v0_real_anchor",
        "trainer": "cem_real_anchor",
        "seed": seed,
        "device": str(device),
        "clip_seconds": clip_seconds,
        "phase_blend": phase_blend,
        "init_config_path": str(init_config_path),
        "score_cfg": score_cfg,
        "heldout_probe_cfg": heldout_probe_cfg or {},
        "selection_source": selection_source,
        "selection_fallback_diagnostics": selection_fallback_diagnostics,
        "case_weights": case_weights or {},
        "train_anchor_wavs": [x["wav_path"] for x in train_set],
        "val_anchor_wavs": [x["wav_path"] for x in val_set],
        "config": {
            "qset": list(best_cfg.qset),
            "q_weights": list(best_cfg.q_weights),
            "residue_scale": best_cfg.residue_scale,
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
            "child_branch_parent_threshold": best_cfg.child_branch_parent_threshold,
            "child_branch_writeback_threshold": best_cfg.child_branch_writeback_threshold,
            "child_branch_meso_threshold": best_cfg.child_branch_meso_threshold,
            "child_branch_live_threshold": best_cfg.child_branch_live_threshold,
            "mode_perturb_window_frac": best_cfg.mode_perturb_window_frac,
            "qtrace_momentum": best_cfg.qtrace_momentum,
            "mask_neighborhood": best_cfg.mask_neighborhood,
            "topology_mask_gain": best_cfg.topology_mask_gain,
            "complexity_mask_gain": best_cfg.complexity_mask_gain,
            "context_mask_gain": best_cfg.context_mask_gain,
            "contrastive_mask_gain": best_cfg.contrastive_mask_gain,
            "aux_mask_suppression": best_cfg.aux_mask_suppression,
            "branch_kernel_version": best_cfg.branch_kernel_version,
            "relation_attention_gain": best_cfg.relation_attention_gain,
            "relation_attention_sharpness": best_cfg.relation_attention_sharpness,
            "relation_value_gain": best_cfg.relation_value_gain,
            "relation_support_gain": best_cfg.relation_support_gain,
            "relation_logit_gain": best_cfg.relation_logit_gain,
            "relation_qtrace_gain": best_cfg.relation_qtrace_gain,
            "relation_residual_mix": best_cfg.relation_residual_mix,
            "child_spawn_threshold": best_cfg.child_spawn_threshold,
            "child_max_worlds": best_cfg.child_max_worlds,
            "child_min_age_for_writeback": best_cfg.child_min_age_for_writeback,
            "child_support_window": best_cfg.child_support_window,
            "child_support_decay": best_cfg.child_support_decay,
            "child_survival_coherence_weight": best_cfg.child_survival_coherence_weight,
            "child_survival_qtrace_weight": best_cfg.child_survival_qtrace_weight,
            "child_survival_residue_penalty": best_cfg.child_survival_residue_penalty,
              "child_writeback_gain": best_cfg.child_writeback_gain,
              "child_writeback_rank": best_cfg.child_writeback_rank,
              "child_writeback_temperature": best_cfg.child_writeback_temperature,
              "child_writeback_budget": best_cfg.child_writeback_budget,
              "child_parent_mix": best_cfg.child_parent_mix,
              "child_parent_mix_early": best_cfg.child_parent_mix_early,
              **_child_writeback_control_values(best_cfg),
              "child_kill_threshold": best_cfg.child_kill_threshold,
            "child_local_ifs_enabled": best_cfg.child_local_ifs_enabled,
            "child_local_steps": best_cfg.child_local_steps,
            "child_local_support_only": best_cfg.child_local_support_only,
            **_child_local_coherence_values(best_cfg),
            "instability_mask_gain": best_cfg.instability_mask_gain,
            "defect_phase_gain": best_cfg.defect_phase_gain,
            "defect_q_gain": best_cfg.defect_q_gain,
            "defect_residue_gain": best_cfg.defect_residue_gain,
            "defect_sharpness_gain": best_cfg.defect_sharpness_gain,
            "defect_world_grad_gain": best_cfg.defect_world_grad_gain,
            "instability_seed_scale": best_cfg.instability_seed_scale,
            "instability_support_gain": best_cfg.instability_support_gain,
            "instability_logit_gain": best_cfg.instability_logit_gain,
            "law_packet_merge_threshold": best_cfg.law_packet_merge_threshold,
            "law_packet_min_score": best_cfg.law_packet_min_score,
            "law_packet_topk_families": best_cfg.law_packet_topk_families,
        },
    }
    ckpt_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    history_path.write_text(json.dumps(history, indent=2), encoding="utf-8")

    summary = {
        "runtime": "circleworld_proto",
        "schema": "circleworld_v0_real_anchor",
        "status": "completed",
        "trainer": "cem_real_anchor",
        "selection_source": selection_source,
        "seed": seed,
        "device": str(device),
        "iterations": iterations,
        "population": population,
        "elite_count": elite_count,
        "clip_seconds": clip_seconds,
        "phase_blend": phase_blend,
        "checkpoint": str(ckpt_path),
        "baseline_train": baseline_train,
        "baseline_val": baseline_val,
        "best_train": best_train,
        "best_val": best_val,
        "best_transfer_probe": best_transfer_probe,
        "best_candidate": best_state,
        "best_passing_candidate": best_passing_state,
        "best_agreement_candidate": best_agreement_state,
        "best_nested_pending_candidate": best_nested_pending_state,
        "selection_fallback_diagnostics": selection_fallback_diagnostics,
        "selection_agreement_candidates": agreement_candidates,
        "selected_candidate": selected_state,
        "best_candidate_gate": best_state.get("selection_gate", {}),
        "selected_candidate_gate": selected_state.get("selection_gate", {}) if selected_state is not None else {
            "enabled": hard_gate_enabled,
            "passed": False,
            "margin": 0.0,
            "failed_checks": ["no_passing_candidate"],
            "checks": {},
        },
        "selected_candidate_heldout_gate": selected_state.get("selection_heldout_gate", {}) if selected_state is not None else {
            "enabled": agreement_top_k > 0,
            "passed": False,
            "margin": 0.0,
            "failed_checks": ["no_selected_candidate"],
            "checks": {},
        },
        "selected_candidate_nested_gate": selected_state.get("selection_nested_gate", {}) if selected_state is not None else {
            "enabled": nested_top_k > 0,
            "passed": False,
            "margin": 0.0,
            "failed_checks": ["no_selected_candidate"],
            "checks": {},
        },
        "best_candidate_transfer_probe": best_state.get("transfer_probe", {}) if best_state is not None else {},
        "best_candidate_heldout_probe": best_state.get("heldout_probe", {}) if best_state is not None else {},
        "best_candidate_heldout_gate": best_state.get("selection_heldout_gate", {}) if best_state is not None else {
            "enabled": agreement_top_k > 0,
            "passed": False,
            "margin": 0.0,
            "failed_checks": ["no_best_candidate"],
            "checks": {},
        },
        "best_candidate_nested_gate": best_state.get("selection_nested_gate", {}) if best_state is not None else {
            "enabled": nested_top_k > 0,
            "passed": False,
            "margin": 0.0,
            "failed_checks": ["no_best_candidate"],
            "checks": {},
        },
        "best_candidate_nested_probe": best_state.get("nested_probe", {}) if best_state is not None else {
            "enabled": nested_probe_top_k > 0,
        },
        "selected_candidate_transfer_probe": selected_state.get("transfer_probe", {}) if selected_state is not None else {},
        "selected_candidate_heldout_probe": selected_state.get("heldout_probe", {}) if selected_state is not None else {},
        "selected_candidate_nested_probe": selected_state.get("nested_probe", {}) if selected_state is not None else {
            "enabled": nested_probe_top_k > 0,
        },
        "best_config": payload["config"],
        "artifact_paths": {
            "summary": str(summary_path),
            "search_history": str(history_path),
            "checkpoint": str(ckpt_path),
        },
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description="Train Circleworld directly on real-anchor audio comparisons.")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--checkpoint-dir", required=True)
    ap.add_argument("--iterations", type=int, default=40)
    ap.add_argument("--population", type=int, default=8)
    ap.add_argument("--elite-count", type=int, default=3)
    ap.add_argument("--seed", type=int, default=20260409)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--clip-seconds", type=int, default=10)
    ap.add_argument("--phase-blend", type=float, default=1.0)
    ap.add_argument("--init-config", required=True)
    ap.add_argument("--target-corr", type=float, default=0.78)
    ap.add_argument("--target-mae", type=float, default=0.08)
    ap.add_argument("--target-dom", type=float, default=0.62)
    ap.add_argument("--target-entropy", type=float, default=0.58)
    ap.add_argument("--min-promotions", type=float, default=8.0)
    ap.add_argument("--w-corr", type=float, default=1.0)
    ap.add_argument("--w-mae", type=float, default=3.0)
    ap.add_argument("--w-dom", type=float, default=0.4)
    ap.add_argument("--w-entropy", type=float, default=0.4)
    ap.add_argument("--w-promote", type=float, default=0.05)
    ap.add_argument("--w-residue", type=float, default=0.2)
    ap.add_argument("--w-promotability", type=float, default=0.1)
    ap.add_argument("--w-major", type=float, default=0.1)
    ap.add_argument("--w-prefix-alignment", type=float, default=0.35)
    ap.add_argument("--w-prefix-delta", type=float, default=0.20)
    ap.add_argument("--corr-low", type=float, default=None)
    ap.add_argument("--corr-high", type=float, default=None)
    ap.add_argument("--mae-low", type=float, default=None)
    ap.add_argument("--mae-high", type=float, default=None)
    ap.add_argument("--w-corr-band", type=float, default=0.0)
    ap.add_argument("--w-mae-band", type=float, default=0.0)
    ap.add_argument("--w-baseline-corr", type=float, default=0.0)
    ap.add_argument("--w-baseline-mae", type=float, default=0.0)
    ap.add_argument("--baseline-corr-margin", type=float, default=0.0)
    ap.add_argument("--baseline-mae-margin", type=float, default=0.0)
    ap.add_argument("--w-real-branch", type=float, default=0.0)
    ap.add_argument("--w-meso-branch", type=float, default=0.0)
    ap.add_argument("--w-slot2-live", type=float, default=0.0)
    ap.add_argument("--w-silent-singlepath", type=float, default=0.0)
    ap.add_argument("--w-branch-positive-mask", type=float, default=0.0)
    ap.add_argument("--w-branch-negative-mask", type=float, default=0.0)
    ap.add_argument("--w-decorative-slot2", type=float, default=0.0)
    ap.add_argument("--w-relation-handoff", type=float, default=0.0)
    ap.add_argument("--w-relation-attn-10", type=float, default=0.0)
    ap.add_argument("--w-branch-defect", type=float, default=0.0)
    ap.add_argument("--w-branch-world-grad", type=float, default=0.0)
    ap.add_argument("--w-branch-q-disagreement", type=float, default=0.0)
    ap.add_argument("--w-branch-phase-wall", type=float, default=0.0)
    ap.add_argument("--w-branch-seed-energy", type=float, default=0.0)
    ap.add_argument("--w-child-world-count", type=float, default=0.0)
    ap.add_argument("--w-live-child", type=float, default=0.0)
    ap.add_argument("--w-child-age", type=float, default=0.0)
    ap.add_argument("--w-child-writeback", type=float, default=0.0)
    ap.add_argument("--w-child-parent-div", type=float, default=0.0)
    ap.add_argument("--w-child-sibling-div", type=float, default=0.0)
    ap.add_argument("--w-defect-without-branch", type=float, default=0.0)
    ap.add_argument("--w-branch-delay", type=float, default=0.0)
    ap.add_argument("--w-rel-signature-count", type=float, default=0.0)
    ap.add_argument("--w-rel-signature-families", type=float, default=0.0)
    ap.add_argument("--w-rel-signature-confidence", type=float, default=0.0)
    ap.add_argument("--w-rel-signature-q-entropy", type=float, default=0.0)
    ap.add_argument("--w-rel-branch-mass", type=float, default=0.0)
    ap.add_argument("--w-rel-dominant-family", type=float, default=0.0)
    ap.add_argument("--target-rel-dominant-family-share", type=float, default=1.0)
    ap.add_argument("--target-case-law-families", type=float, default=0.0)
    ap.add_argument("--target-case-law-entropy", type=float, default=0.0)
    ap.add_argument("--target-case-law-top-q-entropy", type=float, default=0.0)
    ap.add_argument("--target-case-law-dominant-family-share", type=float, default=1.0)
    ap.add_argument("--target-case-law-dominant-q-share", type=float, default=1.0)
    ap.add_argument("--target-aggregate-law-families", type=float, default=0.0)
    ap.add_argument("--target-aggregate-law-entropy", type=float, default=0.0)
    ap.add_argument("--target-aggregate-law-top-q-entropy", type=float, default=0.0)
    ap.add_argument("--target-aggregate-law-top-q-unique", type=float, default=0.0)
    ap.add_argument("--target-aggregate-law-dominant-family-share", type=float, default=1.0)
    ap.add_argument("--target-aggregate-law-dominant-q-share", type=float, default=1.0)
    ap.add_argument("--w-case-law-family-shortfall", type=float, default=0.0)
    ap.add_argument("--w-case-law-entropy-shortfall", type=float, default=0.0)
    ap.add_argument("--w-case-law-top-q-entropy-shortfall", type=float, default=0.0)
    ap.add_argument("--w-case-law-dominant-family", type=float, default=0.0)
    ap.add_argument("--w-case-law-dominant-q", type=float, default=0.0)
    ap.add_argument("--w-aggregate-law-family-shortfall", type=float, default=0.0)
    ap.add_argument("--w-aggregate-law-entropy-shortfall", type=float, default=0.0)
    ap.add_argument("--w-aggregate-law-top-q-entropy-shortfall", type=float, default=0.0)
    ap.add_argument("--w-aggregate-law-top-q-unique-shortfall", type=float, default=0.0)
    ap.add_argument("--w-aggregate-law-dominant-family", type=float, default=0.0)
    ap.add_argument("--w-aggregate-law-dominant-q", type=float, default=0.0)
    ap.add_argument("--target-case-rel-signature-families", type=float, default=0.0)
    ap.add_argument("--target-case-rel-signature-q-entropy", type=float, default=0.0)
    ap.add_argument("--target-case-rel-dominant-family-share", type=float, default=1.0)
    ap.add_argument("--target-aggregate-rel-signature-families", type=float, default=0.0)
    ap.add_argument("--target-aggregate-rel-signature-confidence", type=float, default=0.0)
    ap.add_argument("--target-aggregate-rel-signature-q-entropy", type=float, default=0.0)
    ap.add_argument("--target-aggregate-rel-branch-mass", type=float, default=0.0)
    ap.add_argument("--target-aggregate-rel-dominant-family-share", type=float, default=1.0)
    ap.add_argument("--w-case-rel-signature-family-shortfall", type=float, default=0.0)
    ap.add_argument("--w-case-rel-signature-entropy-shortfall", type=float, default=0.0)
    ap.add_argument("--w-case-rel-signature-dominant-family", type=float, default=0.0)
    ap.add_argument("--w-aggregate-rel-signature-family-shortfall", type=float, default=0.0)
    ap.add_argument("--w-aggregate-rel-signature-confidence-shortfall", type=float, default=0.0)
    ap.add_argument("--w-aggregate-rel-signature-q-entropy-shortfall", type=float, default=0.0)
    ap.add_argument("--w-aggregate-rel-signature-branch-mass-shortfall", type=float, default=0.0)
    ap.add_argument("--w-aggregate-rel-dominant-family", type=float, default=0.0)
    ap.add_argument("--score-cfg-json", default=None, help="JSON object or path overlay applied to score_cfg after CLI parsing.")
    ap.add_argument("--heldout-probe-cfg-json", default=None, help="JSON object or path for heldout probe and branch-first selection settings.")
    ap.add_argument("--case-weights-json", default=None, help="JSON object mapping substrings to weights.")
    ap.add_argument("--extra-train-wavs-json", default=None, help="JSON list of extra train WAV paths.")
    ap.add_argument("--extra-val-wavs-json", default=None, help="JSON list of extra val WAV paths.")
    args = ap.parse_args()

    score_cfg = {
        "target_corr": float(args.target_corr),
        "target_mae": float(args.target_mae),
        "target_dom": float(args.target_dom),
        "target_entropy": float(args.target_entropy),
        "min_promotions": float(args.min_promotions),
        "w_corr": float(args.w_corr),
        "w_mae": float(args.w_mae),
        "w_dom": float(args.w_dom),
        "w_entropy": float(args.w_entropy),
        "w_promote": float(args.w_promote),
        "w_residue": float(args.w_residue),
        "w_promotability": float(args.w_promotability),
        "w_major": float(args.w_major),
        "w_prefix_alignment": float(args.w_prefix_alignment),
        "w_prefix_delta": float(args.w_prefix_delta),
        "corr_low": None if args.corr_low is None else float(args.corr_low),
        "corr_high": None if args.corr_high is None else float(args.corr_high),
        "mae_low": None if args.mae_low is None else float(args.mae_low),
        "mae_high": None if args.mae_high is None else float(args.mae_high),
        "w_corr_band": float(args.w_corr_band),
        "w_mae_band": float(args.w_mae_band),
        "w_baseline_corr": float(args.w_baseline_corr),
        "w_baseline_mae": float(args.w_baseline_mae),
        "baseline_corr_margin": float(args.baseline_corr_margin),
        "baseline_mae_margin": float(args.baseline_mae_margin),
        "w_real_branch": float(args.w_real_branch),
        "w_meso_branch": float(args.w_meso_branch),
        "w_slot2_live": float(args.w_slot2_live),
        "w_silent_singlepath": float(args.w_silent_singlepath),
        "w_branch_positive_mask": float(args.w_branch_positive_mask),
        "w_branch_negative_mask": float(args.w_branch_negative_mask),
        "w_decorative_slot2": float(args.w_decorative_slot2),
        "w_relation_handoff": float(args.w_relation_handoff),
        "w_relation_attn_10": float(args.w_relation_attn_10),
        "w_branch_defect": float(args.w_branch_defect),
        "w_branch_world_grad": float(args.w_branch_world_grad),
        "w_branch_q_disagreement": float(args.w_branch_q_disagreement),
        "w_branch_phase_wall": float(args.w_branch_phase_wall),
        "w_branch_seed_energy": float(args.w_branch_seed_energy),
        "w_child_world_count": float(args.w_child_world_count),
        "w_live_child": float(args.w_live_child),
        "w_child_age": float(args.w_child_age),
        "w_child_writeback": float(args.w_child_writeback),
        "w_child_parent_div": float(args.w_child_parent_div),
        "w_child_sibling_div": float(args.w_child_sibling_div),
        "w_defect_without_branch": float(args.w_defect_without_branch),
        "w_branch_delay": float(args.w_branch_delay),
        "w_rel_signature_count": float(args.w_rel_signature_count),
        "w_rel_signature_families": float(args.w_rel_signature_families),
        "w_rel_signature_confidence": float(args.w_rel_signature_confidence),
        "w_rel_signature_q_entropy": float(args.w_rel_signature_q_entropy),
        "w_rel_branch_mass": float(args.w_rel_branch_mass),
        "w_rel_dominant_family": float(args.w_rel_dominant_family),
        "target_rel_dominant_family_share": float(args.target_rel_dominant_family_share),
        "target_case_law_families": float(args.target_case_law_families),
        "target_case_law_entropy": float(args.target_case_law_entropy),
        "target_case_law_top_q_entropy": float(args.target_case_law_top_q_entropy),
        "target_case_law_dominant_family_share": float(args.target_case_law_dominant_family_share),
        "target_case_law_dominant_q_share": float(args.target_case_law_dominant_q_share),
        "target_aggregate_law_families": float(args.target_aggregate_law_families),
        "target_aggregate_law_entropy": float(args.target_aggregate_law_entropy),
        "target_aggregate_law_top_q_entropy": float(args.target_aggregate_law_top_q_entropy),
        "target_aggregate_law_top_q_unique": float(args.target_aggregate_law_top_q_unique),
        "target_aggregate_law_dominant_family_share": float(args.target_aggregate_law_dominant_family_share),
        "target_aggregate_law_dominant_q_share": float(args.target_aggregate_law_dominant_q_share),
        "w_case_law_family_shortfall": float(args.w_case_law_family_shortfall),
        "w_case_law_entropy_shortfall": float(args.w_case_law_entropy_shortfall),
        "w_case_law_top_q_entropy_shortfall": float(args.w_case_law_top_q_entropy_shortfall),
        "w_case_law_dominant_family": float(args.w_case_law_dominant_family),
        "w_case_law_dominant_q": float(args.w_case_law_dominant_q),
        "w_aggregate_law_family_shortfall": float(args.w_aggregate_law_family_shortfall),
        "w_aggregate_law_entropy_shortfall": float(args.w_aggregate_law_entropy_shortfall),
        "w_aggregate_law_top_q_entropy_shortfall": float(args.w_aggregate_law_top_q_entropy_shortfall),
        "w_aggregate_law_top_q_unique_shortfall": float(args.w_aggregate_law_top_q_unique_shortfall),
        "w_aggregate_law_dominant_family": float(args.w_aggregate_law_dominant_family),
        "w_aggregate_law_dominant_q": float(args.w_aggregate_law_dominant_q),
        "target_case_rel_signature_families": float(args.target_case_rel_signature_families),
        "target_case_rel_signature_q_entropy": float(args.target_case_rel_signature_q_entropy),
        "target_case_rel_dominant_family_share": float(args.target_case_rel_dominant_family_share),
        "target_aggregate_rel_signature_families": float(args.target_aggregate_rel_signature_families),
        "target_aggregate_rel_signature_confidence": float(args.target_aggregate_rel_signature_confidence),
        "target_aggregate_rel_signature_q_entropy": float(args.target_aggregate_rel_signature_q_entropy),
        "target_aggregate_rel_branch_mass": float(args.target_aggregate_rel_branch_mass),
        "target_aggregate_rel_dominant_family_share": float(args.target_aggregate_rel_dominant_family_share),
        "w_case_rel_signature_family_shortfall": float(args.w_case_rel_signature_family_shortfall),
        "w_case_rel_signature_entropy_shortfall": float(args.w_case_rel_signature_entropy_shortfall),
        "w_case_rel_signature_dominant_family": float(args.w_case_rel_signature_dominant_family),
        "w_aggregate_rel_signature_family_shortfall": float(args.w_aggregate_rel_signature_family_shortfall),
        "w_aggregate_rel_signature_confidence_shortfall": float(args.w_aggregate_rel_signature_confidence_shortfall),
        "w_aggregate_rel_signature_q_entropy_shortfall": float(args.w_aggregate_rel_signature_q_entropy_shortfall),
        "w_aggregate_rel_signature_branch_mass_shortfall": float(args.w_aggregate_rel_signature_branch_mass_shortfall),
        "w_aggregate_rel_dominant_family": float(args.w_aggregate_rel_dominant_family),
        "w_nested_probe_readout_sibling_fraction": 0.0,
        "w_nested_probe_continuation_sibling_fraction": 0.0,
        "w_nested_probe_mode_replace_sibling_fraction": 0.0,
        "w_nested_probe_causal_sibling_fraction": 0.0,
        "w_nested_probe_continuation_budget_retained": 0.0,
        "w_nested_probe_continuation_readout_response": 0.0,
        "w_nested_probe_continuation_q_corr": 0.0,
        "w_nested_probe_continuation_write_delta": 0.0,
        "w_nested_probe_continuation_readiness": 0.0,
        "w_nested_probe_continuation_max_readiness": 0.0,
        "w_nested_probe_mode_replace_budget_retained": 0.0,
        "w_nested_probe_mode_replace_readout_response": 0.0,
        "w_nested_probe_mode_replace_q_corr": 0.0,
        "w_nested_probe_mode_replace_readiness": 0.0,
        "w_nested_probe_mode_replace_max_readiness": 0.0,
        "w_nested_probe_parent_mode_conversion_sibling_fraction": 0.0,
        "w_nested_probe_child_record_survival_score": 0.0,
        "w_nested_probe_mode_replace_conversion_score": 0.0,
        "w_nested_probe_readout_without_continuation_penalty": 0.0,
        "w_nested_probe_direct_readout_dependency_penalty": 0.0,
        "w_nested_probe_final_direct_mix_shortcut_penalty": 0.0,
        "w_nested_probe_continuation_budget_shortfall": 0.0,
        "w_nested_probe_continuation_response_shortfall": 0.0,
        "w_nested_probe_continuation_q_corr_shortfall": 0.0,
        "w_nested_probe_mode_replace_response_shortfall": 0.0,
        "w_nested_probe_continuation_readiness_shortfall": 0.0,
        "w_nested_probe_mode_replace_readiness_shortfall": 0.0,
        "w_nested_probe_child_record_survival_shortfall": 0.0,
        "w_nested_probe_mode_replace_conversion_shortfall": 0.0,
        "target_nested_probe_continuation_budget_retained": 0.70,
        "target_nested_probe_continuation_readout_response": 0.30,
        "target_nested_probe_continuation_q_corr": 0.995,
        "target_nested_probe_mode_replace_readout_response": 0.30,
        "target_nested_probe_continuation_readiness": 1.0,
        "target_nested_probe_mode_replace_readiness": 1.0,
        "target_nested_probe_child_record_survival_score": 1.0,
        "target_nested_probe_mode_replace_conversion_score": 0.10,
    }
    score_cfg_overlay = _load_named_json_arg(args.score_cfg_json, {}, nested_key="score_cfg")
    if isinstance(score_cfg_overlay, dict):
        score_cfg.update({str(k): v for k, v in score_cfg_overlay.items()})
    heldout_probe_cfg = _load_named_json_arg(args.heldout_probe_cfg_json, {}, nested_key="heldout_probe_cfg")
    if not isinstance(heldout_probe_cfg, dict):
        heldout_probe_cfg = {}
    case_weights = _load_json_arg(args.case_weights_json, {})
    extra_train_wavs = _load_json_arg(args.extra_train_wavs_json, [])
    extra_val_wavs = _load_json_arg(args.extra_val_wavs_json, [])

    summary = train_circleworld_real_anchor(
        out_dir=Path(args.out_dir),
        checkpoint_dir=Path(args.checkpoint_dir),
        iterations=args.iterations,
        population=args.population,
        elite_count=args.elite_count,
        seed=args.seed,
        device_name=args.device,
        clip_seconds=args.clip_seconds,
        phase_blend=args.phase_blend,
        init_config_path=Path(args.init_config),
        score_cfg=score_cfg,
        case_weights=case_weights,
        extra_train_wavs=extra_train_wavs,
        extra_val_wavs=extra_val_wavs,
        heldout_probe_cfg=heldout_probe_cfg,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
