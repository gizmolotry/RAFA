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

from resonant_law_objects import (  # noqa: E402
    GEOMETRIC_COMPONENTS,
    LAW_OBJECT_FEATURE_NAMES,
    child_law_object,
    composition_summary,
    law_object_feature_vector,
    run_retrieval_assay,
    score_query_to_law_object,
    synthetic_law_objects,
)
from rafa_math_tools import phase_to_phasor, phasor_apply_delta, phasor_normalize  # noqa: E402
from causal_operator_selector import (  # noqa: E402
    MultiHeadCausalOperatorSelector,
    causal_pair_feature_dim,
    causal_pair_feature_vector,
    selector_route_scores,
)
from build_shadow_branch_law_table import shadow_branch_law_row  # noqa: E402
from run_contrastive_structural_embedding_probe import (  # noqa: E402
    operator_safety_proxy,
    recurrence_compatibility,
)
from run_causal_operator_selector_probe import _target_rows_for_query  # noqa: E402


def _state_with_child() -> tuple[dict, dict]:
    freq = torch.linspace(-1.0, 1.0, steps=5).view(1, 5, 1)
    time = torch.linspace(0.0, 1.0, steps=9).view(1, 1, 9)
    mode0 = phase_to_phasor(2.0 * torch.pi * (0.25 * time + 0.10 * freq * time))
    support = torch.zeros(mode0.shape[:-1])
    support[:, :, 2:7] = 0.8
    mode1 = phasor_apply_delta(mode0, 0.20 * support)
    phase_modes = phasor_normalize(torch.stack([mode0, mode1], dim=3))
    child = {
        "child_id": 4,
        "parent_depth": 1,
        "spawn_time_index": 3,
        "origin_mode_index": 1,
        "support_window": (2, 7),
        "survival_age": 2,
        "phase_state": phasor_apply_delta(mode1, 0.13 * support),
        "mode_support": support,
        "mode_coherence": 0.75 * support,
        "q_trace": torch.stack([support, 0.25 * support, 0.05 * support, 0.02 * support], dim=-1),
        "law_signature": torch.tensor([0.8, 0.2, 0.1, 0.05, 0.3]),
        "writeback_budget": 1.2,
        "active": True,
        "collapsed": False,
    }
    state = {
        "phase_modes": phase_modes,
        "child_worlds": [child],
    }
    return state, child


def test_child_law_object_is_json_safe_and_operator_valued() -> None:
    state, child = _state_with_child()
    obj = child_law_object(child, state=state, context={"case": "case_a", "branch": "base"})

    assert obj["object_kind"] == "childworld"
    assert obj["family_key"] == "case_a:child:4"
    assert obj["local_family_key"] == "case_a:child:4"
    assert str(obj["structural_family_key"]).startswith("struct:childworld:")
    assert obj["value"]["operator_kind"] == "child_phase_writeback"
    assert obj["value"]["operator_norm"] > 0.0
    assert obj["value"]["causal_use_ready"] == 1.0
    assert abs(sum(obj["key"]["q_profile"]) - 1.0) < 1.0e-6
    assert 0.0 <= obj["key"]["boundary_score"] <= 1.0


def test_geometric_score_reports_required_components_without_learned_residual() -> None:
    objects = synthetic_law_objects()
    row = score_query_to_law_object(objects[0], objects[1])

    assert set(GEOMETRIC_COMPONENTS) == set(row["components"].keys())
    assert 0.0 <= row["geometric_score"] <= 1.0
    assert row["learned_residual"] == 0.0
    assert row["final_score"] == row["geometric_score"]


def test_retrieval_assay_has_nonself_family_signal_on_synthetic_bank() -> None:
    payload = run_retrieval_assay(synthetic_law_objects(), top_k=2)
    summary = payload["summary"]

    assert summary["object_count"] == 6
    assert summary["family_count"] == 3
    assert summary["eligible_excluding_self_count"] == 6
    assert summary["top1_family_accuracy_excluding_self"] > 0.0
    assert summary["mean_top_learned_residual"] == 0.0
    assert summary["geometry_components_reported"] == list(GEOMETRIC_COMPONENTS)


def test_structural_scope_is_case_independent() -> None:
    state, child = _state_with_child()
    left = child_law_object(child, state=state, context={"case": "case_a", "branch": "base"})
    right = child_law_object(child, state=state, context={"case": "case_b", "branch": "base"})
    local = score_query_to_law_object(left, right, family_key_field="local_family_key")
    structural = score_query_to_law_object(left, right, family_key_field="structural_family_key")

    assert left["local_family_key"] != right["local_family_key"]
    assert left["structural_family_key"] == right["structural_family_key"]
    assert local["same_family"] is False
    assert structural["same_family"] is True


def test_law_object_feature_vector_matches_contract() -> None:
    state, child = _state_with_child()
    obj = child_law_object(child, state=state, context={"case": "case_a", "branch": "base"})
    features = law_object_feature_vector(obj)

    assert len(features) == len(LAW_OBJECT_FEATURE_NAMES)
    assert all(0.0 <= value <= 1.0 for value in features)
    assert features[LAW_OBJECT_FEATURE_NAMES.index("is_childworld")] == 1.0


def test_recurrence_and_safety_surfaces_are_bounded() -> None:
    state, child = _state_with_child()
    left = child_law_object(child, state=state, context={"case": "case_a", "branch": "base"})
    right = child_law_object(child, state=state, context={"case": "case_b", "branch": "base"})

    assert 0.0 <= recurrence_compatibility(left, right) <= 1.0
    assert 0.0 <= operator_safety_proxy(left) <= 1.0
    assert recurrence_compatibility(left, right) > 0.9


def test_composition_summary_detects_parent_child_pairs() -> None:
    summary = composition_summary(synthetic_law_objects())

    assert summary["childworld_object_count"] == 6
    assert summary["cross_depth_pair_count"] == 3
    assert 0.0 <= summary["mean_cross_depth_geometric_score"] <= 1.0


def test_origin_child_id_defines_root_family_across_generations() -> None:
    state, root = _state_with_child()
    gen2 = dict(root)
    gen2.update({"child_id": 5, "parent_child_id": 4, "origin_child_id": 4, "generation": 2})
    gen3 = dict(root)
    gen3.update({"child_id": 6, "parent_child_id": 5, "origin_child_id": 4, "generation": 3})

    root_obj = child_law_object(root, state=state, context={"case": "case_a", "branch": "base"})
    gen3_obj = child_law_object(gen3, state=state, context={"case": "case_a", "branch": "base"})
    score = score_query_to_law_object(root_obj, gen3_obj)

    assert root_obj["family_key"] == gen3_obj["family_key"]
    assert gen3_obj["child_instance_key"] == "case_a:child:6"
    assert score["same_family"] is True


def test_composition_summary_skips_self_parent_fragments() -> None:
    state, root = _state_with_child()
    self_parent = dict(root)
    self_parent.update({"child_id": 4, "parent_child_id": 4, "origin_child_id": 4, "generation": 2})
    child = dict(root)
    child.update({"child_id": 5, "parent_child_id": 4, "origin_child_id": 4, "generation": 2})
    objects = [
        child_law_object(root, state=state, context={"case": "case_a", "branch": "base"}),
        child_law_object(self_parent, state=state, context={"case": "case_a", "branch": "base"}),
        child_law_object(child, state=state, context={"case": "case_a", "branch": "base"}),
    ]
    summary = composition_summary(objects)

    assert summary["cross_depth_pair_count"] == 1


def test_causal_operator_target_caps_nonmembrane_decoys() -> None:
    rows = _target_rows_for_query(
        [
            {
                "parent_phase_divergence": 0.9,
                "world_jump_proxy": 0.2,
                "recurrence_compatibility": 0.95,
                "same_local_family": 1.0,
                "same_structural_family": 0.0,
                "identity_membrane_pass": 1.0,
            },
            {
                "parent_phase_divergence": 1.0,
                "world_jump_proxy": 0.1,
                "recurrence_compatibility": 0.90,
                "same_local_family": 0.0,
                "same_structural_family": 0.0,
                "identity_membrane_pass": 0.0,
            },
        ],
        decoy_cap=0.35,
    )

    identity_target = rows[0]["causal_operator_target"]
    decoy_target = rows[1]["causal_operator_target"]

    assert identity_target > decoy_target
    assert decoy_target <= 0.35


def test_causal_pair_feature_vector_matches_contract() -> None:
    state, child = _state_with_child()
    query = child_law_object(child, state=state, context={"case": "case_a", "branch": "base"})
    candidate = child_law_object(child, state=state, context={"case": "case_b", "branch": "base"})
    row = {
        "geometric_score": 0.9,
        "recurrence_compatibility": 0.95,
        "same_local_family": 0.0,
        "same_structural_family": 1.0,
        "identity_membrane_pass": 1.0,
        "strong_recurrence_pass": 0.0,
        "query_safety_proxy": operator_safety_proxy(query),
        "candidate_safety_proxy": operator_safety_proxy(candidate),
    }
    features = causal_pair_feature_vector(query, candidate, row)

    assert len(features) == causal_pair_feature_dim()
    assert all(0.0 <= value <= 1.0 for value in features)


def test_multihead_causal_selector_reports_expected_heads() -> None:
    model = MultiHeadCausalOperatorSelector(causal_pair_feature_dim(), hidden_dim=16)
    features = torch.zeros(3, causal_pair_feature_dim())
    output = model(features)

    assert set(output) == {"route_score", "movement", "jump_safety", "identity"}
    assert selector_route_scores(output).shape == (3,)
    for value in output.values():
        assert torch.all(value >= 0.0)
        assert torch.all(value <= 1.0)


def test_shadow_branch_law_row_reports_bounded_assay_fields() -> None:
    row = shadow_branch_law_row(
        {
            "source": "naked_rafa",
            "seed": 9100,
            "causal_selector_model_score": 0.75,
            "causal_selector_identity_pred": 0.9,
            "causal_selector_recurrence_compatibility": 0.82,
            "causal_selector_jump_safety_pred": 0.95,
            "causal_parent_phase_divergence": 0.18,
            "decoy_parent_phase_divergence": 0.20,
            "causal_world_jump_proxy": 0.01,
            "decoy_world_jump_proxy": 0.02,
            "causal_support_shift": 0.03,
            "decoy_support_shift": 0.04,
            "causal_qtrace_shift_proxy": 0.02,
            "decoy_qtrace_shift_proxy": 0.03,
        }
    )

    for key in (
        "coexistence_drive",
        "merge_drive",
        "survival_delta",
        "collapse_pressure",
        "coherence_target",
        "q_trace_inheritance_mix",
        "writeback_permission",
    ):
        assert 0.0 <= row[key] <= 1.0
    assert row["hard_decoy_against_causal"] == 1.0
    assert row["writeback_permission"] == 1.0
