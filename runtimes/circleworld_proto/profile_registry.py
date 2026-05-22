from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Literal


ProfileFamily = Literal[
    "phase_native_audio_route_policy",
    "internal_phase_law",
    "childworld_runner",
]


@dataclass(frozen=True)
class RunnerProfile:
    """Stable metadata for one native runner/profile identifier."""

    profile_id: str
    family: ProfileFamily
    runner_script: str
    native_arg: str
    description: str
    tags: tuple[str, ...] = ()


@dataclass(frozen=True)
class LockboxProfile:
    """Stable name for a predeclared or holdout lockbox used by runner families."""

    lockbox_id: str
    description: str
    default_cases_artifact: str | None = None
    default_manifest_artifact: str | None = None
    tags: tuple[str, ...] = ()


PHASE_NATIVE_AUDIO_SELECTED_ROUTE_POLICIES: tuple[str, ...] = (
    "global",
    "family_table",
    "case_table",
)

PHASE_NATIVE_AUDIO_LEARNED_ROUTE_MODELS: tuple[str, ...] = (
    "nearest_centroid_route_v1",
    "nearest_case_route_v1",
    "knn5_route_vote_v1",
)

PHASE_NATIVE_AUDIO_FAMILY_ROUTE_MODELS: tuple[str, ...] = (
    "group_centroid_v1",
    "group_knn1_v1",
    "group_knn5_v1",
)

PHASE_NATIVE_AUDIO_OBJECTIVE_ROUTE_MODELS: tuple[str, ...] = (
    "objective_centroid_v1",
    "objective_knn1_v1",
    "objective_knn5_v1",
    "objective_mlp_v1",
    "objective_score_mlp_v1",
    "objective_rank_mlp_v1",
    "objective_rank_embed_mlp_v1",
    "objective_resonant_memory_v1",
)

PHASE_NATIVE_AUDIO_ROUTE_SELECTOR_COMPONENTS: tuple[str, ...] = (
    "objective_knn5_v1",
)

PHASE_NATIVE_AUDIO_ROUTE_POLICY_IDS: tuple[str, ...] = (
    *PHASE_NATIVE_AUDIO_SELECTED_ROUTE_POLICIES,
    *PHASE_NATIVE_AUDIO_LEARNED_ROUTE_MODELS,
    *PHASE_NATIVE_AUDIO_FAMILY_ROUTE_MODELS,
    *PHASE_NATIVE_AUDIO_OBJECTIVE_ROUTE_MODELS,
)

INTERNAL_PHASE_LAW_DEFAULT_VARIANT_NAMES: tuple[str, ...] = (
    "phase_law_off",
    "internal_gate_025_lr12",
    "internal_gate_050_lr12",
    "internal_velocity_025_lr12",
    "internal_softclip_050_lr12",
)

INTERNAL_PHASE_LAW_SPEC_PROFILE_NAMES: tuple[str, ...] = (
    "focused_v1",
    "wide_v1",
    "v2_guard_v1",
    "v3_local_v1",
    "v4_joint_nonbad_v1",
    "v5_low_energy_row_v1",
    "v6_low_energy_win_v1",
    "v7_low_energy_gain_ladder_v1",
    "v8_family_support_mask_v1",
    "v11_phase_router_direct_v1",
    "v12_phase_reentry_direct_v1",
    "v13_causal_reentry_direct_v1",
)

CHILDWORLD_RUNNER_PROFILE_NAMES: tuple[str, ...] = (
    "childworld_v1",
    "writeback_v2",
    "writeback_v3_identity",
    "writeback_v4_real_anchor_transfer",
    "writeback_v5_thresholded_real_branch",
    "writeback_v6_naked_survival_curriculum",
    "writeback_v7_audio_recovery_with_branch_floor",
    "writeback_v8_recovery_with_law_floor",
    "writeback_v9_aligned_probe_recovery",
    "writeback_v11_hard_gate_seeded",
    "writeback_v12_verified_gate_seeded",
    "writeback_v13_targeted_replay_recovery",
    "writeback_v15_nested_divergence_parentmix",
    "writeback_v16_benchmark_guard_nested",
    "writeback_v17_nested_response_gate",
    "writeback_v18_nested_probe_objective",
    "writeback_v19_seeded_nested_substrate",
    "writeback_v20_child_record_perturb",
    "branch_identity_guard_v1",
    "branch_writeback_floor_v1",
    "branch_balance_guard_v1",
    "branch_recovery_soft_guard_v1",
    "naked_phase_nested_recovery_v1",
    "mode_replace_conversion_v1",
)

LOCKBOX_NAMES: tuple[str, ...] = (
    "audio_predeclared_lockbox_v1_2026_05_06",
    "audio_predeclared_lockbox_v1_2026-05-06",
    "audio_predeclared_lockbox_probe_v1_cuda_2026_05_06",
    "audio_predeclared_lockbox_compare_v1_2026_05_06",
    "audio_predeclared_lockbox_compare_v1_no_single_source_2026_05_06",
    "audio_phase_law_scout_compare_no_single_source",
    "audio_internal_phase_law_raw_compare_no_single_source",
    "audio_internal_phase_law_focused_raw_compare_no_single_source",
    "audio_internal_phase_law_wide_raw_compare_no_single_source",
    "audio_internal_phase_law_v2_guard_raw_compare_no_single_source",
    "audio_internal_phase_law_v3_local_raw_compare_no_single_source",
    "audio_internal_phase_law_v4_joint_nonbad_raw_compare_no_single_source",
    "audio_internal_phase_law_v5_low_energy_row_raw_compare_no_single_source",
    "audio_internal_phase_law_v6_low_energy_win_raw_compare_no_single_source",
    "audio_internal_phase_law_v7_low_energy_gain_ladder_raw_compare_no_single_source",
    "audio_internal_phase_law_v8_family_support_mask_raw_compare_no_single_source",
    "audio_internal_phase_law_v9_phase_masks_raw_compare_no_single_source",
    "audio_electric_motor_holdout_cuda_2026_05_06",
    "original_lockbox",
    "fresh_lockbox",
    "third_lockbox",
    "shared_lockbox_battlefield",
)


PROFILE_REGISTRY: tuple[RunnerProfile, ...] = (
    *(
        RunnerProfile(
            profile_id=policy,
            family="phase_native_audio_route_policy",
            runner_script="run_phase_native_audio_selected_route.py",
            native_arg="--route-policy",
            description="Selected-route live renderer policy.",
            tags=("phase_native_audio", "selected_route"),
        )
        for policy in PHASE_NATIVE_AUDIO_SELECTED_ROUTE_POLICIES
    ),
    *(
        RunnerProfile(
            profile_id=model,
            family="phase_native_audio_route_policy",
            runner_script="train_phase_native_audio_route_policy.py",
            native_arg="--model",
            description="No-future learned route policy model.",
            tags=("phase_native_audio", "learned_route"),
        )
        for model in PHASE_NATIVE_AUDIO_LEARNED_ROUTE_MODELS
    ),
    *(
        RunnerProfile(
            profile_id=model,
            family="phase_native_audio_route_policy",
            runner_script="train_phase_native_audio_family_route_policy.py",
            native_arg="--model",
            description="No-future family classifier mapped to a frozen route table.",
            tags=("phase_native_audio", "family_route"),
        )
        for model in PHASE_NATIVE_AUDIO_FAMILY_ROUTE_MODELS
    ),
    *(
        RunnerProfile(
            profile_id=model,
            family="phase_native_audio_route_policy",
            runner_script="train_phase_native_audio_objective_route_policy.py",
            native_arg="--model",
            description="Objective-aware no-future route policy model.",
            tags=("phase_native_audio", "objective_route"),
        )
        for model in PHASE_NATIVE_AUDIO_OBJECTIVE_ROUTE_MODELS
    ),
    *(
        RunnerProfile(
            profile_id=component,
            family="phase_native_audio_route_policy",
            runner_script="package_phase_native_audio_route_selector_profile.py",
            native_arg="--component-id",
            description="Frozen route-selector component identifier.",
            tags=("phase_native_audio", "route_selector_component"),
        )
        for component in PHASE_NATIVE_AUDIO_ROUTE_SELECTOR_COMPONENTS
    ),
    *(
        RunnerProfile(
            profile_id=profile,
            family="internal_phase_law",
            runner_script="build_internal_phase_law_variant_spec.py",
            native_arg="--profile",
            description="Internal phase-law variant-spec profile.",
            tags=("internal_phase_law", "variant_spec"),
        )
        for profile in INTERNAL_PHASE_LAW_SPEC_PROFILE_NAMES
    ),
    *(
        RunnerProfile(
            profile_id=variant,
            family="internal_phase_law",
            runner_script="run_internal_phase_law_objective_scout.py",
            native_arg="variant.name",
            description="Default internal phase-law objective-scout variant name.",
            tags=("internal_phase_law", "objective_scout_default"),
        )
        for variant in INTERNAL_PHASE_LAW_DEFAULT_VARIANT_NAMES
    ),
    *(
        RunnerProfile(
            profile_id=profile,
            family="childworld_runner",
            runner_script="run_childworld_experiment.py",
            native_arg="--profile",
            description="Childworld experiment runner profile.",
            tags=("childworld",),
        )
        for profile in CHILDWORLD_RUNNER_PROFILE_NAMES
    ),
)

LOCKBOX_REGISTRY: tuple[LockboxProfile, ...] = (
    LockboxProfile(
        lockbox_id="audio_predeclared_lockbox_v1_2026_05_06",
        description="Canonical source/target predeclared audio lockbox spelling used by current runners.",
        default_cases_artifact="audio_predeclared_lockbox_cases.json",
        default_manifest_artifact="audio_predeclared_lockbox_manifest.json",
        tags=("audio", "predeclared", "canonical"),
    ),
    LockboxProfile(
        lockbox_id="audio_predeclared_lockbox_v1_2026-05-06",
        description="Legacy hyphenated-date spelling accepted by reset-suite defaults.",
        default_cases_artifact="audio_predeclared_lockbox_cases.json",
        default_manifest_artifact="audio_predeclared_lockbox_manifest.json",
        tags=("audio", "predeclared", "legacy_alias"),
    ),
    *(
        LockboxProfile(
            lockbox_id=name,
            description="Named audio lockbox, comparison output, or holdout role used by Circleworld audio runners.",
            tags=("audio", "lockbox"),
        )
        for name in LOCKBOX_NAMES
        if name
        not in {
            "audio_predeclared_lockbox_v1_2026_05_06",
            "audio_predeclared_lockbox_v1_2026-05-06",
        }
    ),
)


def _profile_index() -> dict[str, tuple[RunnerProfile, ...]]:
    index: dict[str, list[RunnerProfile]] = {}
    for profile in PROFILE_REGISTRY:
        index.setdefault(profile.profile_id, []).append(profile)
    return {key: tuple(value) for key, value in index.items()}


def _lockbox_index() -> dict[str, LockboxProfile]:
    return {lockbox.lockbox_id: lockbox for lockbox in LOCKBOX_REGISTRY}


def list_profiles(family: ProfileFamily | None = None) -> tuple[RunnerProfile, ...]:
    if family is None:
        return PROFILE_REGISTRY
    return tuple(profile for profile in PROFILE_REGISTRY if profile.family == family)


def list_profile_ids(family: ProfileFamily | None = None) -> tuple[str, ...]:
    return tuple(profile.profile_id for profile in list_profiles(family))


def list_route_policy_ids() -> tuple[str, ...]:
    return PHASE_NATIVE_AUDIO_ROUTE_POLICY_IDS


def list_internal_phase_law_profiles() -> tuple[str, ...]:
    return INTERNAL_PHASE_LAW_SPEC_PROFILE_NAMES


def list_lockbox_ids() -> tuple[str, ...]:
    return tuple(lockbox.lockbox_id for lockbox in LOCKBOX_REGISTRY)


def get_profiles(profile_id: str, family: ProfileFamily | None = None) -> tuple[RunnerProfile, ...]:
    matches = _profile_index().get(profile_id, ())
    if family is None:
        return matches
    return tuple(profile for profile in matches if profile.family == family)


def validate_profile_id(profile_id: str, family: ProfileFamily | None = None) -> str:
    if get_profiles(profile_id, family):
        return profile_id
    scope = family or "any"
    raise ValueError(f"Unknown Circleworld profile_id {profile_id!r} for family {scope!r}")


def validate_route_policy_id(route_policy_id: str) -> str:
    if route_policy_id in PHASE_NATIVE_AUDIO_ROUTE_POLICY_IDS:
        return route_policy_id
    raise ValueError(f"Unknown phase-native audio route policy id {route_policy_id!r}")


def validate_lockbox_id(lockbox_id: str) -> str:
    if lockbox_id in _lockbox_index():
        return lockbox_id
    raise ValueError(f"Unknown Circleworld lockbox id {lockbox_id!r}")


def case_group(case_name: str) -> str:
    return str(case_name).split("__", 1)[0]


def route_key_from_dict(raw: dict[str, Any]) -> tuple[str, str, str, float]:
    return (
        str(raw["magnitude_mode"]),
        str(raw["mask_mode"]),
        str(raw["mechanism"]),
        float(raw["gain"]),
    )


def route_dict(key: tuple[str, str, str, float]) -> dict[str, Any]:
    return {
        "magnitude_mode": key[0],
        "mask_mode": key[1],
        "mechanism": key[2],
        "gain": key[3],
    }


def route_id(key: tuple[str, str, str, float]) -> str:
    return f"{key[0]}|||{key[1]}|||{key[2]}|||{key[3]:.12g}"


def route_from_id(raw_route_id: str) -> tuple[str, str, str, float]:
    mode, mask, mechanism, gain = str(raw_route_id).split("|||")
    return mode, mask, mechanism, float(gain)


def unknown_profile_ids(profile_ids: Iterable[str], family: ProfileFamily | None = None) -> tuple[str, ...]:
    return tuple(profile_id for profile_id in profile_ids if not get_profiles(profile_id, family))


def unknown_lockbox_ids(lockbox_ids: Iterable[str]) -> tuple[str, ...]:
    known = _lockbox_index()
    return tuple(lockbox_id for lockbox_id in lockbox_ids if lockbox_id not in known)


__all__ = (
    "CHILDWORLD_RUNNER_PROFILE_NAMES",
    "INTERNAL_PHASE_LAW_DEFAULT_VARIANT_NAMES",
    "INTERNAL_PHASE_LAW_SPEC_PROFILE_NAMES",
    "LOCKBOX_NAMES",
    "LOCKBOX_REGISTRY",
    "PHASE_NATIVE_AUDIO_FAMILY_ROUTE_MODELS",
    "PHASE_NATIVE_AUDIO_LEARNED_ROUTE_MODELS",
    "PHASE_NATIVE_AUDIO_OBJECTIVE_ROUTE_MODELS",
    "PHASE_NATIVE_AUDIO_ROUTE_POLICY_IDS",
    "PHASE_NATIVE_AUDIO_ROUTE_SELECTOR_COMPONENTS",
    "PHASE_NATIVE_AUDIO_SELECTED_ROUTE_POLICIES",
    "PROFILE_REGISTRY",
    "LockboxProfile",
    "ProfileFamily",
    "RunnerProfile",
    "case_group",
    "get_profiles",
    "list_internal_phase_law_profiles",
    "list_lockbox_ids",
    "list_profile_ids",
    "list_profiles",
    "list_route_policy_ids",
    "route_dict",
    "route_from_id",
    "route_id",
    "route_key_from_dict",
    "unknown_lockbox_ids",
    "unknown_profile_ids",
    "validate_lockbox_id",
    "validate_profile_id",
    "validate_route_policy_id",
)
