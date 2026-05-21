from __future__ import annotations

import math
import sys
import tempfile
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
RUNTIME = ROOT / "runtimes" / "circleworld_proto"
for path in (ROOT, RUNTIME):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from phase_native_audio_operators import (  # noqa: E402
    MECHANISMS,
    mechanism_delta,
    phase_delta_stats,
    phasor_unit_norm_error_from_phase,
)
from profile_registry import (  # noqa: E402
    PHASE_NATIVE_AUDIO_OBJECTIVE_ROUTE_MODELS,
    route_dict,
    route_from_id,
    route_id,
    validate_route_policy_id,
)
from run_circleworld_operator_block import select_route_from_policy_payload  # noqa: E402
import train_phase_native_audio_objective_route_policy as objective_trainer  # noqa: E402


def test_phase_native_operator_bank_preserves_shape_and_finite_phase() -> None:
    torch.manual_seed(7)
    raw_delta = 0.35 * torch.randn(1, 16, 9)
    prefix_mag = torch.rand(1, 16, 12).abs() + 0.01
    prefix_phase = math.pi * (2.0 * torch.rand(1, 16, 12) - 1.0)
    mask = torch.ones_like(raw_delta)

    for mechanism in MECHANISMS:
        shaped, flags = mechanism_delta(
            mechanism=mechanism,
            raw_delta=raw_delta,
            prefix_mag=prefix_mag,
            prefix_phase=prefix_phase,
        )
        assert shaped.shape == raw_delta.shape
        assert torch.isfinite(shaped).all()
        assert float(shaped.abs().amax()) <= math.pi + 1.0e-5
        assert flags["future_target_magnitude_reused"] is False
        assert flags["target_future_stft_magnitude_accessed"] is False
        assert flags["future_target_phase_reused"] is False
        assert flags["target_future_stft_phase_accessed"] is False
        stats = phase_delta_stats(shaped, mask)
        assert stats["future_rms_phase_delta"] >= 0.0
        assert math.isfinite(stats["future_mean_cos_phase_delta"])


def test_phase_native_operator_bank_unit_phasor_contract() -> None:
    phase = torch.linspace(-math.pi, math.pi, steps=40).reshape(1, 4, 10)
    assert phasor_unit_norm_error_from_phase(phase) <= 1.0e-6


def test_operator_block_case_table_route_selection_matches_policy() -> None:
    payload = {
        "case_routes": {
            "family__case001": {
                "magnitude_mode": "prefix_hold",
                "mask_mode": "phase_router_bins",
                "mechanism": "anti_reentry_delta_shear_mix",
                "gain": 16.0,
            }
        },
        "fallback_route": {
            "magnitude_mode": "flat",
            "mask_mode": "all_bins",
            "mechanism": "raw",
            "gain": 0.5,
        },
    }
    route = select_route_from_policy_payload(payload, "family__case001", route_policy="case_table")
    assert route == payload["case_routes"]["family__case001"]


def test_operator_block_case_table_uses_fallback_for_unknown_case() -> None:
    payload = {
        "case_routes": {
            "known__case001": {
                "magnitude_mode": "prefix_hold",
                "mask_mode": "phase_router_bins",
                "mechanism": "anti_reentry_delta_shear_mix",
                "gain": 16.0,
            }
        },
        "fallback_route": {
            "magnitude_mode": "flat",
            "mask_mode": "all_bins",
            "mechanism": "raw",
            "gain": 0.5,
        },
    }

    route = select_route_from_policy_payload(payload, "unknown__case999", route_policy="case_table")
    assert route == payload["fallback_route"]


def test_phase_native_route_id_round_trip_is_stable() -> None:
    key = ("prefix_hold", "phase_router_bins", "anti_reentry_delta_shear_mix", 16.0)
    encoded = route_id(key)

    assert encoded == "prefix_hold|||phase_router_bins|||anti_reentry_delta_shear_mix|||16"
    assert route_from_id(encoded) == key
    assert route_dict(route_from_id(encoded)) == {
        "magnitude_mode": "prefix_hold",
        "mask_mode": "phase_router_bins",
        "mechanism": "anti_reentry_delta_shear_mix",
        "gain": 16.0,
    }


def test_objective_mlp_v1_is_registered_and_case_table_compatible() -> None:
    payload = {
        "schema": "phase_native_audio_objective_route_policy_v1",
        "model": "objective_mlp_v1",
        "case_routes": {
            "family__case001": {
                "magnitude_mode": "prefix_hold",
                "mask_mode": "phase_router_bins",
                "mechanism": "anti_reentry_delta_shear_mix",
                "gain": 16.0,
            }
        },
        "predictions": [
            {
                "case_name": "family__case001",
                "group": "family",
                "route_id": "prefix_hold|||phase_router_bins|||anti_reentry_delta_shear_mix|||16",
                "route": {
                    "magnitude_mode": "prefix_hold",
                    "mask_mode": "phase_router_bins",
                    "mechanism": "anti_reentry_delta_shear_mix",
                    "gain": 16.0,
                },
                "distance": 0.25,
            }
        ],
        "fallback_route": {
            "magnitude_mode": "flat",
            "mask_mode": "all_bins",
            "mechanism": "raw",
            "gain": 0.5,
        },
        "predicted_route_counts": {
            "prefix_hold|||phase_router_bins|||anti_reentry_delta_shear_mix|||16": 1,
        },
    }

    assert "objective_mlp_v1" in objective_trainer.OBJECTIVE_ROUTE_MODELS
    assert "objective_mlp_v1" in PHASE_NATIVE_AUDIO_OBJECTIVE_ROUTE_MODELS
    assert validate_route_policy_id("objective_mlp_v1") == "objective_mlp_v1"
    route = select_route_from_policy_payload(payload, "family__case001", route_policy="case_table")
    assert route == payload["predictions"][0]["route"]


def test_objective_score_mlp_v1_is_registered() -> None:
    assert "objective_score_mlp_v1" in objective_trainer.OBJECTIVE_ROUTE_MODELS
    assert "objective_score_mlp_v1" in PHASE_NATIVE_AUDIO_OBJECTIVE_ROUTE_MODELS
    assert validate_route_policy_id("objective_score_mlp_v1") == "objective_score_mlp_v1"


def test_objective_resonant_memory_v1_is_registered() -> None:
    assert "objective_resonant_memory_v1" in objective_trainer.OBJECTIVE_ROUTE_MODELS
    assert "objective_resonant_memory_v1" in PHASE_NATIVE_AUDIO_OBJECTIVE_ROUTE_MODELS
    assert validate_route_policy_id("objective_resonant_memory_v1") == "objective_resonant_memory_v1"


def test_objective_mlp_v1_training_emits_case_table_policy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    route_a = ("prefix_hold", "phase_router_bins", "anti_reentry_delta_shear_mix", 16.0)
    route_b = ("flat", "all_bins", "raw", 0.5)

    def fake_case_features(paths: list[Path]) -> dict[str, dict[str, float]]:
        if paths == [Path("target_delta.json")]:
            return {
                "family__target001": {"x": 0.05, "y": 0.10},
                "family__target002": {"x": 0.95, "y": 0.90},
            }
        return {
            "family__source001": {"x": 0.00, "y": 0.00},
            "family__source002": {"x": 0.10, "y": 0.05},
            "family__source003": {"x": 1.00, "y": 1.00},
            "family__source004": {"x": 0.90, "y": 0.95},
        }

    def fake_training_rows(
        train_sets: list[tuple[Path, list[Path]]],
        keys: list[str],
        *,
        margin: float,
    ) -> list[dict[str, object]]:
        del train_sets, margin
        source = fake_case_features([Path("source_delta.json")])
        labels = {
            "family__source001": route_a,
            "family__source002": route_a,
            "family__source003": route_b,
            "family__source004": route_b,
        }
        return [
            {
                "case_name": case_name,
                "group": "family",
                "route_id": objective_trainer._route_id(labels[case_name]),
                "route": objective_trainer._route_dict(labels[case_name]),
                "row_objective": 1.0,
                "vector": objective_trainer._vector(features, keys),
            }
            for case_name, features in sorted(source.items())
        ]

    monkeypatch.setattr(objective_trainer, "_case_features", fake_case_features)
    monkeypatch.setattr(objective_trainer, "_training_rows", fake_training_rows)

    with tempfile.TemporaryDirectory(prefix="objective_mlp_v1_", dir=Path.cwd()) as out_dir:
        summary = objective_trainer.train_objective_route_policy(
            train_sets=[(Path("suite.json"), [Path("source_delta.json")])],
            target_delta_json=Path("target_delta.json"),
            out_dir=Path(out_dir),
            model="objective_mlp_v1",
            margin=0.01,
        )

    assert summary["schema"] == "phase_native_audio_objective_route_policy_v1"
    assert summary["model"] == "objective_mlp_v1"
    assert summary["feature_keys"] == ["x", "y"]
    assert summary["source_future_metrics_used_for_route_training_labels"] is True
    assert summary["target_future_audio_used_for_route_selection"] is False
    assert summary["target_future_metrics_used_for_route_selection"] is False
    assert not any(
        token in key
        for key in summary["feature_keys"]
        for token in ("target_", "copy_last", "row_objective", "selected_route")
    )
    assert set(summary["case_routes"]) == {"family__target001", "family__target002"}
    assert len(summary["predictions"]) == 2
    assert sum(summary["predicted_route_counts"].values()) == 2
    assert summary["fallback_route"] in [objective_trainer._route_dict(route_a), objective_trainer._route_dict(route_b)]
    assert summary["training_diagnostics"]["class_count"] == 2
    assert summary["training_diagnostics"]["epochs"] > 0
    assert summary["training_diagnostics"]["final_loss"] is not None
    route = select_route_from_policy_payload(summary, "family__target001", route_policy="case_table")
    assert route == summary["case_routes"]["family__target001"]


def test_objective_score_mlp_v1_training_uses_candidate_objectives(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    route_a = ("prefix_hold", "phase_router_bins", "anti_reentry_delta_shear_mix", 16.0)
    route_b = ("flat", "all_bins", "raw", 0.5)
    zero_gain = ("flat", "all_bins", "raw", 0.0)

    def fake_case_features(paths: list[Path]) -> dict[str, dict[str, float]]:
        if paths == [Path("target_delta.json")]:
            return {
                "family__target001": {"x": 0.05, "y": 0.10},
                "family__target002": {"x": 0.95, "y": 0.90},
            }
        return {
            "family__source001": {"x": 0.00, "y": 0.00},
            "family__source002": {"x": 0.10, "y": 0.05},
            "family__source003": {"x": 1.00, "y": 1.00},
            "family__source004": {"x": 0.90, "y": 0.95},
        }

    def row(case_name: str, route: tuple[str, str, str, float], objective: float) -> dict[str, object]:
        return {
            "case_name": case_name,
            "group": "family",
            "magnitude_mode": route[0],
            "mask_mode": route[1],
            "mechanism": route[2],
            "gain": route[3],
            "corr_delta_vs_copy_last": objective,
            "mse_delta_vs_copy_last": 0.0,
            "loop_delta_vs_copy_last": 0.0,
            "harmful_replay_excess_delta_vs_copy_last": 0.0,
            "corr_delta_vs_gain0": 0.0,
        }

    def fake_collect_rows(
        suite_json: Path,
        delta_jsons: list[Path],
        margin: float,
    ) -> tuple[dict[str, object], list[dict[str, object]]]:
        del suite_json, delta_jsons, margin
        return {"future_access_clean": True}, [
            row("family__source001", route_a, 1.00),
            row("family__source001", route_b, 0.05),
            row("family__source002", route_a, 0.20),
            row("family__source002", route_a, 0.90),
            row("family__source002", route_b, 0.10),
            row("family__source003", route_a, 0.00),
            row("family__source003", route_b, 1.20),
            row("family__source004", route_a, 0.10),
            row("family__source004", route_b, 1.10),
            row("family__source004", zero_gain, 9.00),
        ]

    monkeypatch.setattr(objective_trainer, "_case_features", fake_case_features)
    monkeypatch.setattr(objective_trainer, "_collect_rows", fake_collect_rows)

    with tempfile.TemporaryDirectory(prefix="objective_score_mlp_v1_", dir=Path.cwd()) as out_dir:
        summary = objective_trainer.train_objective_route_policy(
            train_sets=[(Path("suite.json"), [Path("source_delta.json")])],
            target_delta_json=Path("target_delta.json"),
            out_dir=Path(out_dir),
            model="objective_score_mlp_v1",
            margin=0.01,
        )

    assert summary["schema"] == "phase_native_audio_objective_route_policy_v1"
    assert summary["model"] == "objective_score_mlp_v1"
    assert summary["feature_keys"] == ["x", "y"]
    assert summary["source_future_metrics_used_for_route_training_labels"] is True
    assert summary["target_future_audio_used_for_route_selection"] is False
    assert summary["target_future_metrics_used_for_route_selection"] is False
    assert not any(
        token in key
        for key in summary["feature_keys"]
        for token in (
            "target_",
            "copy_last",
            "row_objective",
            "selected_route",
            "corr_delta",
            "mse_delta",
            "loop_delta",
        )
    )
    assert set(summary["case_routes"]) == {"family__target001", "family__target002"}
    assert len(summary["predictions"]) == 2
    assert sum(summary["predicted_route_counts"].values()) == 2
    assert summary["fallback_route"] in [objective_trainer._route_dict(route_a), objective_trainer._route_dict(route_b)]
    assert set(summary["training_route_counts"]) == {
        objective_trainer._route_id(route_a),
        objective_trainer._route_id(route_b),
    }
    diagnostics = summary["training_diagnostics"]
    assert diagnostics["model_family"] == "score_mlp"
    assert diagnostics["candidate_route_count"] == 2
    assert diagnostics["epochs"] > 0
    assert diagnostics["final_loss"] is not None
    candidate_diagnostics = diagnostics["candidate_diagnostics"]
    assert candidate_diagnostics["raw_candidate_row_count"] == 10
    assert candidate_diagnostics["zero_gain_candidate_row_count"] == 1
    assert candidate_diagnostics["duplicate_case_route_candidate_row_count"] == 1
    assert candidate_diagnostics["candidate_row_count"] == 8
    assert candidate_diagnostics["candidate_route_count"] == 2
    assert candidate_diagnostics["source_future_access_clean"] is True
    route = select_route_from_policy_payload(summary, "family__target001", route_policy="case_table")
    assert route == summary["case_routes"]["family__target001"]


def test_objective_resonant_memory_v1_training_emits_component_diagnostics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    route_a = ("prefix_hold", "phase_router_bins", "anti_reentry_delta_shear_mix", 16.0)
    route_b = ("flat", "all_bins", "raw", 0.5)
    safe_feature_names = [
        "phase_alignment",
        "ramanujan_q_score",
        "arc_residue_balance",
        "mask_support_locality",
        "child_branch_mode",
        "loop_reentry_temporal",
        "case_horizon",
        "residual_energy",
    ]

    def feature_row(base: float) -> dict[str, float]:
        return {key: base for key in safe_feature_names}

    def fake_case_features(paths: list[Path]) -> dict[str, dict[str, float]]:
        if paths == [Path("target_delta.json")]:
            return {
                "family__target001": feature_row(0.05),
                "family__target002": feature_row(0.95),
            }
        return {
            "family__source001": feature_row(0.00),
            "family__source002": feature_row(0.10),
            "family__source003": feature_row(1.00),
            "family__source004": feature_row(0.90),
        }

    def fake_training_rows(
        train_sets: list[tuple[Path, list[Path]]],
        keys: list[str],
        *,
        margin: float,
    ) -> list[dict[str, object]]:
        del train_sets, margin
        source = fake_case_features([Path("source_delta.json")])
        labels = {
            "family__source001": (route_a, 1.5),
            "family__source002": (route_a, 1.2),
            "family__source003": (route_b, 1.4),
            "family__source004": (route_b, 1.1),
        }
        return [
            {
                "case_name": case_name,
                "group": "family",
                "route_id": objective_trainer._route_id(labels[case_name][0]),
                "route": objective_trainer._route_dict(labels[case_name][0]),
                "row_objective": labels[case_name][1],
                "vector": objective_trainer._vector(features, keys),
            }
            for case_name, features in sorted(source.items())
        ]

    monkeypatch.setattr(objective_trainer, "_case_features", fake_case_features)
    monkeypatch.setattr(objective_trainer, "_training_rows", fake_training_rows)

    with tempfile.TemporaryDirectory(prefix="objective_resonant_memory_v1_", dir=Path.cwd()) as out_dir:
        summary = objective_trainer.train_objective_route_policy(
            train_sets=[(Path("suite.json"), [Path("source_delta.json")])],
            target_delta_json=Path("target_delta.json"),
            out_dir=Path(out_dir),
            model="objective_resonant_memory_v1",
            margin=0.01,
        )

    assert summary["schema"] == "phase_native_audio_objective_route_policy_v1"
    assert summary["model"] == "objective_resonant_memory_v1"
    assert summary["feature_keys"] == sorted(safe_feature_names)
    assert summary["source_future_metrics_used_for_route_training_labels"] is True
    assert summary["target_future_audio_used_for_route_selection"] is False
    assert summary["target_future_metrics_used_for_route_selection"] is False
    assert not any(
        token in key
        for key in summary["feature_keys"]
        for token in (
            "target_",
            "copy_last",
            "row_objective",
            "selected_route",
            "corr_delta",
            "mse_delta",
            "loop_delta",
        )
    )
    assert set(summary["case_routes"]) == {"family__target001", "family__target002"}
    assert len(summary["predictions"]) == 2
    assert sum(summary["predicted_route_counts"].values()) == 2
    route = select_route_from_policy_payload(summary, "family__target001", route_policy="case_table")
    assert route == summary["case_routes"]["family__target001"]

    diagnostics = summary["training_diagnostics"]
    assert diagnostics["model_family"] == "resonant_memory"
    assert diagnostics["memory_source"] == "best_source_rows"
    assert diagnostics["memory_count"] == 4
    assert diagnostics["memory_route_count"] == 2
    assert diagnostics["fallback_prediction_count"] == 0
    assert set(diagnostics["component_weights"]) == {
        "phase",
        "q",
        "arc_residue",
        "support",
        "branch",
        "reentry",
        "case",
        "other",
    }
    for group in diagnostics["component_weights"]:
        assert group in diagnostics["feature_groups"]
    assert diagnostics["feature_groups"]["phase"] == ["phase_alignment"]
    assert diagnostics["feature_groups"]["q"] == ["ramanujan_q_score"]
    assert diagnostics["feature_groups"]["arc_residue"] == ["arc_residue_balance"]
    assert diagnostics["feature_groups"]["support"] == ["mask_support_locality"]
    assert diagnostics["feature_groups"]["branch"] == ["child_branch_mode"]
    assert diagnostics["feature_groups"]["reentry"] == ["loop_reentry_temporal"]
    assert diagnostics["feature_groups"]["case"] == ["case_horizon"]
    assert diagnostics["feature_groups"]["other"] == ["residual_energy"]
    assert diagnostics["target_route_activation_summaries"]

    first_prediction = summary["predictions"][0]
    assert "activation" in first_prediction
    assert set(first_prediction["resonance_components"]) == set(diagnostics["component_weights"])
    assert first_prediction["route_activation_summaries"]
    assert first_prediction["top_memory_preview"]
