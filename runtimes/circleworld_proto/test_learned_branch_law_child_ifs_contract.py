from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
LINEAGE = ROOT / "lineages" / "04_positive_replacement"
RUNTIME = Path(__file__).resolve().parent
for path in (ROOT, LINEAGE, RUNTIME):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from circleworld import (
    CircleworldConfig,
    RafaLearnedBranchLawV0,
    branch_law_feature_names,
    child_circleworld_step,
    circleworld_step,
    learned_branch_law_output_names,
    learned_branch_law_output_tensor,
    learned_branch_law_targets_from_features,
)
from rafa_math_tools import phase_to_phasor, phasor_apply_delta
from test_nested_commitment import _child_state_identity_metrics
from train_shadow_branch_law_from_table import (
    shadow_branch_feature_names,
    shadow_row_to_feature_target,
)
from evaluate_shadow_branch_law_calibration import shadow_prediction_to_branch_fields
from run_learned_gated_writeback_sandbox import learned_gate_row_from_case
from run_learned_gated_multistep_operator_sandbox import run_scaled_operator_recurrence


def _seed_phase(f_bins: int = 8, t_len: int = 16) -> torch.Tensor:
    freq = torch.linspace(-1.0, 1.0, steps=f_bins).view(1, f_bins, 1)
    time = torch.linspace(0.0, 1.0, steps=t_len).view(1, 1, t_len)
    return phase_to_phasor(2.0 * torch.pi * (0.35 * time + 0.15 * freq * time))


def test_learned_branch_law_shape_and_target_contract() -> None:
    feature_dim = len(branch_law_feature_names())
    output_dim = len(learned_branch_law_output_names())
    features = torch.rand(2, 3, feature_dim)
    model = RafaLearnedBranchLawV0(input_dim=feature_dim, hidden_dim=16, output_dim=output_dim)
    outputs = model(features)
    output_tensor = learned_branch_law_output_tensor(outputs)
    targets = learned_branch_law_output_tensor(learned_branch_law_targets_from_features(features))

    assert output_tensor.shape == (2, 3, output_dim)
    assert targets.shape == output_tensor.shape
    for name in ("split_drive", "coexistence_drive", "merge_drive", "collapse_pressure"):
        assert torch.all(outputs[name] >= 0.0)
        assert torch.all(outputs[name] <= 1.0)
    for name in ("survival_delta0", "survival_delta1", "support_delta0", "support_delta1"):
        assert torch.all(outputs[name] >= -1.0)
        assert torch.all(outputs[name] <= 1.0)


def test_shadow_branch_table_row_maps_to_learned_branch_law_contract() -> None:
    features, targets = shadow_row_to_feature_target(
        {
            "causal_selector_model_score": 0.78,
            "identity_signal": 1.0,
            "recurrence_compatibility": 0.90,
            "jump_safety": 0.53,
            "coexistence_drive": 0.91,
            "merge_drive": 0.93,
            "survival_delta": 0.87,
            "collapse_pressure": 0.13,
            "support_delta": -0.11,
            "coherence_target": 0.88,
            "q_trace_inheritance_mix": 0.94,
            "hard_decoy_against_causal": 1.0,
            "causal_safety_win_over_decoy": 1.0,
            "causal_parent_phase_divergence": 0.18,
            "decoy_parent_phase_divergence": 0.19,
            "causal_world_jump_proxy": 0.017,
            "decoy_world_jump_proxy": 0.022,
        }
    )

    assert len(features) == len(shadow_branch_feature_names())
    assert len(targets) == len(learned_branch_law_output_names())
    assert all(-1.0 <= value <= 1.0 for value in features)
    assert all(-1.0 <= value <= 1.0 for value in targets)


def test_shadow_branch_prediction_fields_are_bounded() -> None:
    fields = shadow_prediction_to_branch_fields(
        {
            "coexistence_drive": 0.90,
            "merge_drive": 0.91,
            "survival_delta0": 0.72,
            "survival_delta1": 0.74,
            "collapse_pressure": 0.12,
            "support_delta0": -0.10,
            "support_delta1": -0.12,
            "coherence_target": 0.88,
            "qtrace_inheritance_mix": 0.93,
        }
    )

    for key, value in fields.items():
        if key == "predicted_support_delta":
            assert -1.0 <= value <= 1.0
        else:
            assert 0.0 <= value <= 1.0
    assert fields["predicted_writeback_score"] > 0.78


def test_learned_gate_row_suppresses_denied_writeback() -> None:
    row = learned_gate_row_from_case(
        {
            "source": "naked_rafa",
            "seed": 1,
            "causal_parent_phase_divergence": 0.25,
            "causal_world_jump_proxy": 0.04,
            "causal_support_shift": 0.10,
            "causal_qtrace_shift_proxy": 0.02,
        },
        predicted_fields={
            "predicted_writeback_score": 0.70,
            "predicted_survival_delta": 0.60,
            "predicted_collapse_pressure": 0.35,
            "predicted_coexistence_drive": 0.80,
            "predicted_q_trace_inheritance_mix": 0.90,
        },
        measured_shadow={"writeback_permission": 0.0, "survival_delta": 0.60, "collapse_pressure": 0.35},
        writeback_threshold=0.78,
    )

    assert row["predicted_writeback_permission"] == 0.0
    assert row["writeback_correct"] == 1.0
    assert row["ungated_parent_phase_divergence"] > 0.0
    assert row["gated_parent_phase_divergence"] == 0.0
    assert row["gated_world_jump_proxy"] == 0.0


def test_soft_learned_gate_attenuates_denied_writeback() -> None:
    row = learned_gate_row_from_case(
        {
            "source": "naked_rafa",
            "seed": 2,
            "causal_parent_phase_divergence": 0.25,
            "causal_world_jump_proxy": 0.04,
            "causal_support_shift": 0.10,
            "causal_qtrace_shift_proxy": 0.02,
        },
        predicted_fields={
            "predicted_writeback_score": 0.70,
            "predicted_survival_delta": 0.60,
            "predicted_collapse_pressure": 0.35,
            "predicted_coexistence_drive": 0.80,
            "predicted_q_trace_inheritance_mix": 0.90,
        },
        measured_shadow={"writeback_permission": 0.0, "survival_delta": 0.60, "collapse_pressure": 0.35},
        writeback_threshold=0.78,
        gate_mode="soft",
        soft_floor=0.65,
        soft_full=0.85,
    )

    assert row["predicted_writeback_permission"] == 0.0
    assert 0.0 < row["writeback_scale"] < 1.0
    assert abs(row["writeback_scale"] - 0.25) < 1.0e-6
    assert abs(row["gated_parent_phase_divergence"] - 0.0625) < 1.0e-6
    assert abs(row["gated_world_jump_proxy"] - 0.01) < 1.0e-6


def test_scaled_operator_recurrence_saturates_and_preserves_zero_gate() -> None:
    mode0 = _seed_phase(f_bins=5, t_len=9)
    support = torch.zeros(mode0.shape[:-1])
    support[:, :, 2:7] = 0.8
    mode1 = phasor_apply_delta(mode0, 0.12 * support)
    child = {
        "child_id": 7,
        "phase_state": phasor_apply_delta(mode1, 0.20 * support),
        "mode_support": support,
        "mode_coherence": support,
    }
    state = {"phase_modes": torch.stack([mode0, mode1], dim=3)}

    zero = run_scaled_operator_recurrence(state, child, operator_gain=0.85, writeback_scale=0.0, steps=3)
    full = run_scaled_operator_recurrence(state, child, operator_gain=0.85, writeback_scale=1.0, steps=3)

    assert zero["final_parent_phase_divergence"] == 0.0
    assert zero["final_world_jump_proxy"] == 0.0
    assert full["final_parent_phase_divergence"] > 0.0
    assert full["cumulative_step_parent_phase_divergence"] >= full["final_parent_phase_divergence"]


def test_child_circleworld_step_updates_child_state_and_preserves_phasors() -> None:
    cfg = CircleworldConfig(
        branching_mode="native_multimode_childworld",
        child_local_ifs_enabled=True,
        child_local_steps=2,
        child_support_window=8,
        child_local_coherence_retention_enabled=True,
        child_local_coherence_retention_mix=0.75,
        child_local_coherence_floor=0.35,
        child_local_coherence_floor_support=0.10,
        child_local_coherence_causal_gate_enabled=True,
        child_local_coherence_causal_min_delta=0.01,
        child_local_coherence_causal_full_delta=0.30,
        promotion_threshold=0.0,
        max_promotions=2,
    )
    parent_mode0 = _seed_phase()
    support = torch.zeros(parent_mode0.shape[:-1])
    support[:, :, 4:12] = 0.8
    parent_mode1 = phasor_apply_delta(parent_mode0, 0.20 * support)
    child = {
        "child_id": 3,
        "parent_depth": 0,
        "spawn_time_index": 8,
        "origin_mode_index": 1,
        "support_window": (4, 12),
        "survival_age": 0,
        "phase_state": phasor_apply_delta(parent_mode1, 0.18 * support),
        "mode_support": support.clone(),
        "mode_coherence": (support > 0).to(support.dtype),
        "q_trace": torch.zeros((*support.shape, cfg.q_trace_rank)),
        "law_signature": torch.ones(len(cfg.qset) + 11),
        "writeback_budget": 0.0,
        "active": True,
        "collapsed": False,
    }

    child_next, metrics = child_circleworld_step(child, cfg, parent_mode0, parent_mode1)
    norm_err = (torch.linalg.vector_norm(child_next["phase_state"], dim=-1) - 1.0).abs().max()

    assert child_next["survival_age"] == 2
    assert child_next["q_trace"].shape == (*support.shape, cfg.q_trace_rank)
    assert float(metrics["child_local_ifs_step_count"].item()) == 2.0
    assert float(metrics["child_local_ifs_support_mean"].item()) > 0.0
    assert float(metrics["child_local_ifs_coherence_mean"].item()) >= float(
        metrics["child_local_ifs_coherence_raw_mean"].item()
    ) - 1.0e-6
    assert float(metrics["child_local_ifs_coherence_retention_delta"].item()) >= -1.0e-6
    assert float(metrics["child_local_ifs_coherence_floor_delta"].item()) >= -1.0e-6
    assert float(metrics["child_local_ifs_parent_phase_delta"].item()) >= 0.0
    assert 0.0 <= float(metrics["child_local_ifs_causal_gate_mean"].item()) <= 1.0
    assert float(metrics["child_local_ifs_causal_retention_loss"].item()) >= -1.0e-6
    assert float(metrics["child_local_ifs_causal_floor_loss"].item()) >= -1.0e-6
    assert float(norm_err.item()) < 1.0e-5


def test_native_multimode_childworld_runs_config_gated_child_local_ifs() -> None:
    cfg = CircleworldConfig(
        branching_mode="native_multimode_childworld",
        child_local_ifs_enabled=True,
        child_local_steps=1,
        child_kill_threshold=-1.0,
        child_min_age_for_writeback=999,
        child_writeback_budget=0.0,
        promotion_threshold=0.0,
        max_promotions=2,
    )
    mode0 = _seed_phase()
    support = torch.zeros(mode0.shape[:-1])
    support[:, :, 5:13] = 0.75
    mode1 = phasor_apply_delta(mode0, 0.22 * support)
    phase_modes = torch.stack([mode0, mode1], dim=3)
    mode_logits = torch.zeros((*support.shape, 2))
    mode_logits[..., 0] = 1.5
    mode_logits[..., 1] = 0.5
    mode_support = torch.ones_like(mode_logits) * 0.1
    mode_support[..., 0] = 1.0
    mode_support[..., 1] = support
    child = {
        "child_id": 11,
        "parent_depth": 0,
        "spawn_time_index": 8,
        "origin_mode_index": 1,
        "support_window": (5, 13),
        "survival_age": 0,
        "phase_state": phasor_apply_delta(mode1, 0.15 * support),
        "mode_support": support.clone(),
        "mode_coherence": support.clone() * 0.4,
        "q_trace": torch.zeros((*support.shape, cfg.q_trace_rank)),
        "law_signature": torch.ones(len(cfg.qset) + 11),
        "writeback_budget": 0.0,
        "active": True,
        "collapsed": False,
    }
    state = {
        "phase_modes": phase_modes,
        "mode_logits": mode_logits,
        "mode_support": mode_support,
        "mode_coherence": mode_support.clone(),
        "mode_q_trace": torch.zeros((*support.shape, 2, cfg.q_trace_rank)),
        "child_worlds": [child],
        "child_event_history": [],
        "next_child_id": 12,
    }

    state_next, block, _ = circleworld_step(state, cfg, mode="native_multimode_childworld")
    child_events = state_next.get("child_event_history", [])

    assert float(block["child_local_ifs_step_count"].item()) > 0.0
    assert any(event.get("event") == "child_local_ifs" for event in child_events)
    norm_err = (torch.linalg.vector_norm(state_next["phase_modes"], dim=-1) - 1.0).abs().max()
    assert float(norm_err.item()) < 1.0e-5


def test_child_assay_writeback_scale_suppresses_runtime_return() -> None:
    cfg = CircleworldConfig(
        branching_mode="native_multimode_childworld",
        child_max_worlds=1,
        child_kill_threshold=-1.0,
        child_min_age_for_writeback=0,
        child_writeback_budget=1.0,
        child_writeback_gain=0.60,
        child_assay_writeback_scale_enabled=True,
        child_local_ifs_enabled=False,
        promotion_threshold=1.0,
        max_promotions=0,
    )
    mode0 = _seed_phase()
    support = torch.zeros(mode0.shape[:-1])
    support[:, :, 5:13] = 1.0
    mode1 = phasor_apply_delta(mode0, 0.20 * support)
    phase_modes = torch.stack([mode0, mode1], dim=3)
    mode_logits = torch.zeros((*support.shape, 2))
    mode_logits[..., 0] = 1.25
    mode_logits[..., 1] = 0.75
    mode_support = torch.zeros_like(mode_logits)
    mode_support[..., 0] = 1.0
    mode_support[..., 1] = support

    def _state(scale: float) -> dict[str, object]:
        child = {
            "child_id": 17,
            "parent_depth": 0,
            "spawn_time_index": 8,
            "origin_mode_index": 1,
            "support_window": (5, 13),
            "survival_age": 2,
            "phase_state": phasor_apply_delta(mode0, 0.55 * support),
            "mode_support": support.clone(),
            "mode_coherence": support.clone() * 0.80,
            "q_trace": torch.zeros((*support.shape, cfg.q_trace_rank)),
            "law_signature": torch.ones(len(cfg.qset) + 11),
            "writeback_budget": 1.0,
            "assay_writeback_scale": float(scale),
            "active": True,
            "collapsed": False,
        }
        return {
            "phase_modes": phase_modes.clone(),
            "mode_logits": mode_logits.clone(),
            "mode_support": mode_support.clone(),
            "mode_coherence": mode_support.clone(),
            "mode_q_trace": torch.zeros((*support.shape, 2, cfg.q_trace_rank)),
            "child_worlds": [child],
            "child_event_history": [],
            "next_child_id": 18,
        }

    state_zero, block_zero, _ = circleworld_step(_state(0.0), cfg, mode="native_multimode_childworld")
    state_one, block_one, _ = circleworld_step(_state(1.0), cfg, mode="native_multimode_childworld")

    assert float(block_zero["child_assay_writeback_scale_mean"].item()) == 0.0
    assert float(block_one["child_assay_writeback_scale_mean"].item()) == 1.0
    assert float(block_zero["child_writeback_mass"].item()) <= 1.0e-8
    assert float(block_one["child_writeback_mass"].item()) > 1.0e-4
    assert float(block_zero["child_phase_writeback_delta_mass"].item()) <= 1.0e-8
    assert float(block_one["child_phase_writeback_delta_mass"].item()) > 1.0e-4
    assert any(event.get("assay_writeback_scale") == 0.0 for event in state_zero.get("child_event_history", []))
    assert any(event.get("assay_writeback_scale") == 1.0 for event in state_one.get("child_event_history", []))


def test_nested_identity_coherence_is_support_masked() -> None:
    parent_mode0 = _seed_phase(f_bins=4, t_len=8)
    support = torch.zeros(parent_mode0.shape[:-1])
    support[:, :, 2:6] = 1.0
    parent_mode1 = phasor_apply_delta(parent_mode0, 0.15 * support)
    phase_modes = torch.stack([parent_mode0, parent_mode1], dim=3)
    child = {
        "child_id": 7,
        "phase_state": parent_mode1,
        "mode_support": support,
        "mode_coherence": support.clone(),
        "writeback_budget": 1.0,
        "active": True,
    }
    state = {"phase_modes": phase_modes, "child_worlds": [child]}

    metrics = _child_state_identity_metrics(state, child_id=7)

    assert metrics["support_mean"] == 0.5
    assert metrics["coherence_mean"] == 1.0
    assert metrics["qualified_active"] == 1.0
