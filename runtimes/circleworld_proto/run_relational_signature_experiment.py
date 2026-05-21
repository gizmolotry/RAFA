from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
RUNTIME = ROOT / "runtimes" / "circleworld_proto"
for path in (ROOT, RUNTIME):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


ACTIVE_INIT = (
    ROOT
    / "checkpoints_circleworld_proto"
    / "training_run_2026-04-24_agreement_scout_v1"
    / "circleworld_real_anchor_config_cem_v1.json"
)
BASELINE_INIT = (
    ROOT
    / "checkpoints_circleworld_proto"
    / "manual_candidates"
    / "circleworld_parentmix_220_100_2026-04-24.json"
)
SIGNATURE_INIT = (
    ROOT
    / "checkpoints_circleworld_proto"
    / "relsig_scout_balance_2026-05-04"
    / "circleworld_real_anchor_config_cem_v1.json"
)
BRANCHMASS_INIT = (
    ROOT
    / "checkpoints_circleworld_proto"
    / "tokenburst_2026-05-04_branchmass_balance_v1_run_retry"
    / "circleworld_real_anchor_config_cem_v1.json"
)
CONTROLLER_MANIFEST = (
    ROOT
    / "outputs"
    / "circleworld_proto"
    / "tokenburst_2026-05-04_controller"
    / "baseline_manifest.json"
)
DEFAULT_CASES = ROOT / "outputs" / "circleworld_proto" / "expanded_anchor_cases_2026-04-09.json"
REFERENCE_PROFILE = "balance_guard_v1"
TRACK_PROFILE_NAMES = (
    "export_diversity_floor_v1",
    "balance_guard_v2",
    "branchmass_balance_v1",
    "branchmass_plurality_v1",
)
SUPPORTED_EXPORT_COLLAPSE_KEYS = (
    "w_rel_signature_families",
    "w_rel_signature_q_entropy",
    "w_rel_branch_mass",
    "w_rel_dominant_family",
    "target_rel_dominant_family_share",
    "target_case_rel_signature_families",
    "target_case_rel_signature_q_entropy",
    "target_case_rel_dominant_family_share",
    "target_aggregate_rel_signature_families",
    "target_aggregate_rel_signature_confidence",
    "target_aggregate_rel_signature_q_entropy",
    "target_aggregate_rel_branch_mass",
    "target_aggregate_rel_dominant_family_share",
    "w_case_rel_signature_family_shortfall",
    "w_case_rel_signature_entropy_shortfall",
    "w_case_rel_signature_dominant_family",
    "w_aggregate_rel_signature_family_shortfall",
    "w_aggregate_rel_signature_confidence_shortfall",
    "w_aggregate_rel_signature_q_entropy_shortfall",
    "w_aggregate_rel_signature_branch_mass_shortfall",
    "w_aggregate_rel_dominant_family",
)
STRICTER_HIGHER_KEYS = {
    "w_rel_signature_families",
    "w_rel_signature_q_entropy",
    "w_rel_branch_mass",
    "w_rel_dominant_family",
    "target_case_rel_signature_families",
    "target_case_rel_signature_q_entropy",
    "target_aggregate_rel_signature_families",
    "target_aggregate_rel_signature_confidence",
    "target_aggregate_rel_signature_q_entropy",
    "target_aggregate_rel_branch_mass",
    "w_case_rel_signature_family_shortfall",
    "w_case_rel_signature_entropy_shortfall",
    "w_case_rel_signature_dominant_family",
    "w_aggregate_rel_signature_family_shortfall",
    "w_aggregate_rel_signature_confidence_shortfall",
    "w_aggregate_rel_signature_q_entropy_shortfall",
    "w_aggregate_rel_signature_branch_mass_shortfall",
    "w_aggregate_rel_dominant_family",
}
STRICTER_LOWER_KEYS = {
    "target_rel_dominant_family_share",
    "target_case_rel_dominant_family_share",
    "target_aggregate_rel_dominant_family_share",
}


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _profile_catalog() -> dict[str, dict[str, Any]]:
    return {
        "diversity_guard_v1": {
            "default_init_label": "branch_first_active",
            "lane": "legacy",
            "objective": "Favor broad relational-signature diversity over confidence concentration.",
            "optimizes": [
                "higher held-out signature family count",
                "higher aggregate export family count",
                "lower dominant family concentration",
            ],
            "dependencies": [],
            "score_overrides": {
                "w_rel_signature_families": 0.35,
                "w_rel_signature_confidence": 0.12,
                "w_rel_signature_q_entropy": 0.22,
                "w_rel_dominant_family": 0.50,
                "target_rel_dominant_family_share": 0.76,
                "target_case_rel_signature_families": 2.0,
                "target_case_rel_signature_q_entropy": 0.60,
                "target_case_rel_dominant_family_share": 0.82,
                "target_aggregate_rel_signature_families": 11.0,
                "target_aggregate_rel_signature_confidence": 0.66,
                "target_aggregate_rel_signature_q_entropy": 0.56,
                "w_case_rel_signature_family_shortfall": 0.45,
                "w_case_rel_signature_entropy_shortfall": 0.30,
                "w_case_rel_signature_dominant_family": 0.18,
                "w_aggregate_rel_signature_family_shortfall": 0.55,
                "w_aggregate_rel_signature_confidence_shortfall": 0.12,
                "w_aggregate_rel_signature_q_entropy_shortfall": 0.32,
                "w_aggregate_rel_dominant_family": 0.45,
            },
        },
        "confidence_guard_v1": {
            "default_init_label": "diversity_reference",
            "lane": "legacy",
            "objective": "Keep prefix and benchmark quality high while tolerating less signature spread.",
            "optimizes": [
                "higher benchmark corr / lower mae",
                "higher relational-signature confidence",
                "tighter prefix agreement",
            ],
            "dependencies": [],
            "score_overrides": {
                "target_corr": 0.84,
                "target_mae": 0.065,
                "w_corr": 0.50,
                "w_mae": 1.55,
                "w_prefix_alignment": 0.40,
                "w_prefix_delta": 0.25,
                "w_rel_signature_families": 0.12,
                "w_rel_signature_confidence": 0.34,
                "w_rel_signature_q_entropy": 0.12,
                "w_rel_branch_mass": 0.12,
                "w_rel_dominant_family": 0.25,
                "target_rel_dominant_family_share": 0.86,
                "target_case_rel_signature_families": 1.8,
                "target_case_rel_signature_q_entropy": 0.50,
                "target_case_rel_dominant_family_share": 0.88,
                "target_aggregate_rel_signature_families": 10.0,
                "target_aggregate_rel_signature_confidence": 0.71,
                "target_aggregate_rel_signature_q_entropy": 0.50,
                "w_case_rel_signature_family_shortfall": 0.20,
                "w_case_rel_signature_entropy_shortfall": 0.16,
                "w_case_rel_signature_dominant_family": 0.12,
                "w_aggregate_rel_signature_family_shortfall": 0.24,
                "w_aggregate_rel_signature_confidence_shortfall": 0.42,
                "w_aggregate_rel_signature_q_entropy_shortfall": 0.18,
                "w_aggregate_rel_dominant_family": 0.28,
            },
        },
        "balance_guard_v1": {
            "default_init_label": "branch_first_active",
            "lane": "legacy",
            "objective": "Balance confidence, diversity, and branch mass across held-out and export metrics.",
            "optimizes": [
                "balanced held-out family count and confidence",
                "moderate aggregate export diversity",
                "moderate relational branch mass",
            ],
            "dependencies": [],
            "score_overrides": {
                "w_rel_signature_families": 0.24,
                "w_rel_signature_confidence": 0.24,
                "w_rel_signature_q_entropy": 0.18,
                "w_rel_branch_mass": 0.16,
                "w_rel_dominant_family": 0.36,
                "target_rel_dominant_family_share": 0.80,
                "target_case_rel_signature_families": 1.9,
                "target_case_rel_signature_q_entropy": 0.56,
                "target_case_rel_dominant_family_share": 0.86,
                "target_aggregate_rel_signature_families": 10.0,
                "target_aggregate_rel_signature_confidence": 0.69,
                "target_aggregate_rel_signature_q_entropy": 0.54,
                "w_case_rel_signature_family_shortfall": 0.34,
                "w_case_rel_signature_entropy_shortfall": 0.24,
                "w_case_rel_signature_dominant_family": 0.16,
                "w_aggregate_rel_signature_family_shortfall": 0.40,
                "w_aggregate_rel_signature_confidence_shortfall": 0.28,
                "w_aggregate_rel_signature_q_entropy_shortfall": 0.24,
                "w_aggregate_rel_dominant_family": 0.36,
            },
        },
        "export_diversity_floor_v1": {
            "default_init_label": "diversity_reference",
            "lane": "signature_lane",
            "objective": "Impose a harder export diversity floor and explicitly punish dominant-family collapse on exported signature families.",
            "optimizes": [
                "higher aggregate export signature family count",
                "higher aggregate export q-entropy",
                "lower aggregate dominant family share",
            ],
            "dependencies": [],
            "score_overrides": {
                "w_rel_signature_families": 0.38,
                "w_rel_signature_confidence": 0.14,
                "w_rel_signature_q_entropy": 0.26,
                "w_rel_branch_mass": 0.18,
                "w_rel_dominant_family": 0.72,
                "target_rel_dominant_family_share": 0.74,
                "target_case_rel_signature_families": 2.2,
                "target_case_rel_signature_q_entropy": 0.62,
                "target_case_rel_dominant_family_share": 0.80,
                "target_aggregate_rel_signature_families": 12.5,
                "target_aggregate_rel_signature_confidence": 0.66,
                "target_aggregate_rel_signature_q_entropy": 0.58,
                "target_aggregate_rel_branch_mass": 0.46,
                "target_aggregate_rel_dominant_family_share": 0.68,
                "w_case_rel_signature_family_shortfall": 0.58,
                "w_case_rel_signature_entropy_shortfall": 0.34,
                "w_case_rel_signature_dominant_family": 0.28,
                "w_aggregate_rel_signature_family_shortfall": 0.88,
                "w_aggregate_rel_signature_confidence_shortfall": 0.12,
                "w_aggregate_rel_signature_q_entropy_shortfall": 0.40,
                "w_aggregate_rel_signature_branch_mass_shortfall": 0.24,
                "w_aggregate_rel_dominant_family": 0.72,
            },
        },
        "balance_guard_v2": {
            "default_init_label": "signature_lane_init",
            "lane": "signature_lane",
            "objective": "Keep the balance-guard blend but tighten export collapse penalties relative to balance_guard_v1.",
            "optimizes": [
                "balanced confidence and family diversity",
                "stronger export dominant-share ceiling",
                "higher aggregate branch-mass floor than v1",
            ],
            "dependencies": [],
            "score_overrides": {
                "target_corr": 0.825,
                "target_mae": 0.069,
                "w_corr": 0.46,
                "w_mae": 1.43,
                "w_rel_signature_families": 0.29,
                "w_rel_signature_confidence": 0.27,
                "w_rel_signature_q_entropy": 0.21,
                "w_rel_branch_mass": 0.18,
                "w_rel_dominant_family": 0.46,
                "target_rel_dominant_family_share": 0.78,
                "target_case_rel_signature_families": 2.0,
                "target_case_rel_signature_q_entropy": 0.58,
                "target_case_rel_dominant_family_share": 0.84,
                "target_aggregate_rel_signature_families": 10.8,
                "target_aggregate_rel_signature_confidence": 0.70,
                "target_aggregate_rel_signature_q_entropy": 0.55,
                "target_aggregate_rel_branch_mass": 0.47,
                "target_aggregate_rel_dominant_family_share": 0.76,
                "w_case_rel_signature_family_shortfall": 0.42,
                "w_case_rel_signature_entropy_shortfall": 0.28,
                "w_case_rel_signature_dominant_family": 0.20,
                "w_aggregate_rel_signature_family_shortfall": 0.54,
                "w_aggregate_rel_signature_confidence_shortfall": 0.30,
                "w_aggregate_rel_signature_q_entropy_shortfall": 0.28,
                "w_aggregate_rel_signature_branch_mass_shortfall": 0.28,
                "w_aggregate_rel_dominant_family": 0.50,
            },
        },
        "branchmass_balance_v1": {
            "default_init_label": "signature_lane_init",
            "lane": "signature_lane",
            "objective": "Raise relational branch mass without giving back the export diversity floor.",
            "optimizes": [
                "higher per-case and aggregate relational branch mass",
                "balanced export family count",
                "lower dominant family collapse while branch mass rises",
            ],
            "dependencies": [],
            "score_overrides": {
                "w_real_branch": 1.8,
                "w_meso_branch": 32000.0,
                "w_rel_signature_families": 0.24,
                "w_rel_signature_confidence": 0.22,
                "w_rel_signature_q_entropy": 0.20,
                "w_rel_branch_mass": 0.30,
                "w_rel_dominant_family": 0.48,
                "target_rel_dominant_family_share": 0.79,
                "target_case_rel_signature_families": 1.95,
                "target_case_rel_signature_q_entropy": 0.57,
                "target_case_rel_dominant_family_share": 0.85,
                "target_aggregate_rel_signature_families": 10.5,
                "target_aggregate_rel_signature_confidence": 0.69,
                "target_aggregate_rel_signature_q_entropy": 0.55,
                "target_aggregate_rel_branch_mass": 0.52,
                "target_aggregate_rel_dominant_family_share": 0.77,
                "w_case_rel_signature_family_shortfall": 0.36,
                "w_case_rel_signature_entropy_shortfall": 0.26,
                "w_case_rel_signature_dominant_family": 0.18,
                "w_aggregate_rel_signature_family_shortfall": 0.46,
                "w_aggregate_rel_signature_confidence_shortfall": 0.24,
                "w_aggregate_rel_signature_q_entropy_shortfall": 0.26,
                "w_aggregate_rel_signature_branch_mass_shortfall": 0.50,
                "w_aggregate_rel_dominant_family": 0.50,
            },
        },
        "branchmass_plurality_v1": {
            "default_init_label": "branchmass_signature_init",
            "lane": "signature_lane",
            "objective": "Start from the branchmass winner, then protect held-out family plurality while keeping export diversity pressure high.",
            "optimizes": [
                "preserved held-out signature family count from the branchmass init",
                "higher aggregate export family spread and q-entropy",
                "lower dominant-family collapse without overpaying for extra branch pressure",
            ],
            "dependencies": [],
            "score_overrides": {
                "w_corr": 0.40,
                "w_mae": 1.32,
                "w_real_branch": 1.65,
                "w_meso_branch": 28000.0,
                "w_rel_signature_families": 0.32,
                "w_rel_signature_confidence": 0.18,
                "w_rel_signature_q_entropy": 0.24,
                "w_rel_branch_mass": 0.22,
                "w_rel_dominant_family": 0.52,
                "target_rel_dominant_family_share": 0.78,
                "target_case_rel_signature_families": 2.10,
                "target_case_rel_signature_q_entropy": 0.60,
                "target_case_rel_dominant_family_share": 0.83,
                "target_aggregate_rel_signature_families": 12.5,
                "target_aggregate_rel_signature_confidence": 0.68,
                "target_aggregate_rel_signature_q_entropy": 0.57,
                "target_aggregate_rel_branch_mass": 0.50,
                "target_aggregate_rel_dominant_family_share": 0.75,
                "w_case_rel_signature_family_shortfall": 0.50,
                "w_case_rel_signature_entropy_shortfall": 0.32,
                "w_case_rel_signature_dominant_family": 0.24,
                "w_aggregate_rel_signature_family_shortfall": 0.68,
                "w_aggregate_rel_signature_confidence_shortfall": 0.18,
                "w_aggregate_rel_signature_q_entropy_shortfall": 0.34,
                "w_aggregate_rel_signature_branch_mass_shortfall": 0.34,
                "w_aggregate_rel_dominant_family": 0.62,
            },
        },
    }


def _profile_names() -> list[str]:
    return list(_profile_catalog().keys())


def _profile_spec(profile: str) -> dict[str, Any]:
    catalog = _profile_catalog()
    if profile not in catalog:
        raise RuntimeError(f"Unknown profile: {profile}")
    return catalog[profile]


def _init_path_by_label(label: str) -> Path:
    mapping = {
        "branch_first_active": ACTIVE_INIT,
        "diversity_reference": BASELINE_INIT,
        "signature_lane_init": SIGNATURE_INIT,
        "branchmass_signature_init": BRANCHMASS_INIT,
    }
    if label not in mapping:
        raise RuntimeError(f"Unknown init label: {label}")
    return mapping[label]


def _default_init_for_profile(profile: str) -> Path:
    return _init_path_by_label(str(_profile_spec(profile)["default_init_label"]))


def _signature_score_cfg(profile: str) -> dict[str, float]:
    base = {
        "target_corr": 0.82,
        "target_mae": 0.070,
        "target_dom": 0.62,
        "target_entropy": 0.58,
        "min_promotions": 8.0,
        "w_corr": 0.45,
        "w_mae": 1.40,
        "w_dom": 0.18,
        "w_entropy": 0.18,
        "w_promote": 0.05,
        "w_residue": 0.12,
        "w_promotability": 0.10,
        "w_major": 0.10,
        "w_prefix_alignment": 0.35,
        "w_prefix_delta": 0.22,
        "corr_low": 0.70,
        "corr_high": 0.98,
        "mae_low": 0.020,
        "mae_high": 0.120,
        "w_corr_band": 0.60,
        "w_mae_band": 0.80,
        "w_baseline_corr": 1.20,
        "w_baseline_mae": 1.00,
        "baseline_corr_margin": 0.00,
        "baseline_mae_margin": 0.00,
        "w_real_branch": 1.5,
        "w_meso_branch": 25000.0,
        "w_rel_signature_count": 0.05,
        "w_rel_signature_families": 0.20,
        "w_rel_signature_confidence": 0.20,
        "w_rel_signature_q_entropy": 0.16,
        "w_rel_branch_mass": 0.14,
        "w_rel_dominant_family": 0.35,
        "target_rel_dominant_family_share": 0.82,
        "target_case_law_families": 1.8,
        "target_case_law_entropy": 0.16,
        "target_aggregate_law_families": 5.0,
        "target_aggregate_law_entropy": 0.32,
        "w_case_law_family_shortfall": 0.10,
        "w_case_law_entropy_shortfall": 0.10,
        "w_aggregate_law_family_shortfall": 0.14,
        "w_aggregate_law_entropy_shortfall": 0.12,
        "target_case_rel_signature_families": 1.6,
        "target_case_rel_signature_q_entropy": 0.55,
        "target_case_rel_dominant_family_share": 0.90,
        "target_aggregate_rel_signature_families": 8.0,
        "target_aggregate_rel_signature_confidence": 0.68,
        "target_aggregate_rel_signature_q_entropy": 0.52,
        "target_aggregate_rel_branch_mass": 0.45,
        "target_aggregate_rel_dominant_family_share": 0.80,
        "w_case_rel_signature_family_shortfall": 0.30,
        "w_case_rel_signature_entropy_shortfall": 0.25,
        "w_case_rel_signature_dominant_family": 0.15,
        "w_aggregate_rel_signature_family_shortfall": 0.35,
        "w_aggregate_rel_signature_confidence_shortfall": 0.25,
        "w_aggregate_rel_signature_q_entropy_shortfall": 0.25,
        "w_aggregate_rel_signature_branch_mass_shortfall": 0.20,
        "w_aggregate_rel_dominant_family": 0.30,
    }
    base.update(_profile_spec(profile)["score_overrides"])
    return base


def _controller_baselines() -> dict[str, str]:
    manifest_payload: dict[str, Any] = {}
    if CONTROLLER_MANIFEST.exists():
        manifest_payload = _read_json(CONTROLLER_MANIFEST)
    return {
        "branch_first_active": str(manifest_payload.get("branch_first_active", ACTIVE_INIT)),
        "diversity_reference": str(manifest_payload.get("diversity_reference", BASELINE_INIT)),
        "signature_lane_init": str(manifest_payload.get("signature_lane_init", SIGNATURE_INIT)),
        "controller_manifest_path": str(CONTROLLER_MANIFEST),
    }


def _collapse_focus(score_cfg: dict[str, float]) -> dict[str, float]:
    return {key: float(score_cfg.get(key, 0.0)) for key in SUPPORTED_EXPORT_COLLAPSE_KEYS}


def _delta_rows(candidate: dict[str, float], reference: dict[str, float]) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for key in sorted(set(candidate.keys()) | set(reference.keys())):
        cv = float(candidate.get(key, 0.0))
        rv = float(reference.get(key, 0.0))
        if abs(cv - rv) < 1e-12:
            continue
        out[key] = {
            "candidate": cv,
            "reference": rv,
            "delta_candidate_minus_reference": cv - rv,
        }
    return out


def _strictness_summary(candidate: dict[str, float], reference: dict[str, float]) -> dict[str, list[str]]:
    stricter: list[str] = []
    relaxed: list[str] = []
    for key in sorted(STRICTER_HIGHER_KEYS):
        cv = float(candidate.get(key, 0.0))
        rv = float(reference.get(key, 0.0))
        if cv > rv:
            stricter.append(key)
        elif cv < rv:
            relaxed.append(key)
    for key in sorted(STRICTER_LOWER_KEYS):
        cv = float(candidate.get(key, 0.0))
        rv = float(reference.get(key, 0.0))
        if cv < rv:
            stricter.append(key)
        elif cv > rv:
            relaxed.append(key)
    return {
        "stricter_export_controls": stricter,
        "relaxed_export_controls": relaxed,
    }


def _profile_metadata(profile: str) -> dict[str, Any]:
    spec = _profile_spec(profile)
    return {
        "name": profile,
        "lane": str(spec["lane"]),
        "default_init_label": str(spec["default_init_label"]),
        "default_init_path": str(_default_init_for_profile(profile)),
        "objective": str(spec["objective"]),
        "optimizes": list(spec["optimizes"]),
        "dependencies": list(spec["dependencies"]),
    }


def _profile_catalog_payload() -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for profile in _profile_names():
        score_cfg = _signature_score_cfg(profile)
        payload[profile] = {
            **_profile_metadata(profile),
            "collapse_focus": _collapse_focus(score_cfg),
            "score_cfg": score_cfg,
        }
    return payload


def _track_summary_payload() -> dict[str, Any]:
    baselines = _controller_baselines()
    reference_cfg = _signature_score_cfg(REFERENCE_PROFILE)
    profiles: list[dict[str, Any]] = []
    for profile in TRACK_PROFILE_NAMES:
        score_cfg = _signature_score_cfg(profile)
        profiles.append(
            {
                **_profile_metadata(profile),
                "score_cfg": score_cfg,
                "collapse_focus": _collapse_focus(score_cfg),
                "delta_vs_reference_profile": _delta_rows(score_cfg, reference_cfg),
                "strictness_vs_reference_profile": _strictness_summary(score_cfg, reference_cfg),
            }
        )
    return {
        "track": "w3_signature_profiles",
        "scope": "circleworld_only",
        "owned_file": str(Path(__file__).resolve()),
        "created_utc": _utc_timestamp(),
        "reference_profile": REFERENCE_PROFILE,
        "controller_baselines": baselines,
        "supported_export_collapse_terms": list(SUPPORTED_EXPORT_COLLAPSE_KEYS),
        "profiles": profiles,
        "hard_dependencies": [],
        "optional_followups": [
            "Trainer does not currently score aggregate_mean_relational_family_size directly; export collapse is approximated through family-count floors, q-entropy floors, dominant-family ceilings, and branch-mass floors.",
            "Comparator could add a dedicated export-collapse verdict using aggregate_mean_relational_family_size in addition to aggregate family count and dominant share.",
        ],
    }


def _track_compare_payload() -> dict[str, Any]:
    reference_cfg = _signature_score_cfg(REFERENCE_PROFILE)
    comparisons: list[dict[str, Any]] = []
    for profile in TRACK_PROFILE_NAMES:
        score_cfg = _signature_score_cfg(profile)
        comparisons.append(
            {
                "profile": profile,
                "default_init_path": str(_default_init_for_profile(profile)),
                "objective": _profile_metadata(profile)["objective"],
                "strictness_vs_reference_profile": _strictness_summary(score_cfg, reference_cfg),
                "delta_vs_reference_profile": _delta_rows(score_cfg, reference_cfg),
                "collapse_focus": _collapse_focus(score_cfg),
            }
        )
    return {
        "track": "w3_signature_profiles",
        "reference_profile": REFERENCE_PROFILE,
        "reference_profile_cfg": reference_cfg,
        "comparisons": comparisons,
        "hard_dependencies": [],
        "optional_followups": [
            "No other track is required to parse or use these profiles.",
            "A trainer-side mean-family-size penalty would make export-family-collapse punishment more direct if another track adds it later.",
        ],
    }


def _track_report_markdown(summary_payload: dict[str, Any], compare_payload: dict[str, Any]) -> str:
    baselines = summary_payload["controller_baselines"]
    md_lines = [
        "# Circleworld Tokenburst W3 Signature Profiles",
        "",
        f"- track: `{summary_payload['track']}`",
        f"- scope: `{summary_payload['scope']}`",
        f"- owned file: `{summary_payload['owned_file']}`",
        f"- created utc: `{summary_payload['created_utc']}`",
        f"- controller manifest: `{baselines['controller_manifest_path']}`",
        f"- branch-first active: `{baselines['branch_first_active']}`",
        f"- diversity reference: `{baselines['diversity_reference']}`",
        f"- signature lane init: `{baselines['signature_lane_init']}`",
        f"- reference profile: `{summary_payload['reference_profile']}`",
        "",
        "## Implemented Profiles",
        "",
    ]
    for row in summary_payload["profiles"]:
        strictness = row["strictness_vs_reference_profile"]
        md_lines.extend(
            [
                f"### {row['name']}",
                f"- lane: `{row['lane']}`",
                f"- default init: `{row['default_init_path']}`",
                f"- objective: {row['objective']}",
                f"- optimizes: {', '.join(str(item) for item in row['optimizes'])}",
                f"- stricter than `{REFERENCE_PROFILE}` on: {', '.join(strictness['stricter_export_controls']) if strictness['stricter_export_controls'] else 'none'}",
                f"- relaxed vs `{REFERENCE_PROFILE}` on: {', '.join(strictness['relaxed_export_controls']) if strictness['relaxed_export_controls'] else 'none'}",
                "",
            ]
        )
    md_lines.extend(
        [
            "## Supported Anti-Collapse Terms",
            "",
            f"- {', '.join(summary_payload['supported_export_collapse_terms'])}",
            "",
            "## Dependencies",
            "",
            "- No hard dependency on other tracks.",
            "- Optional trainer or comparator follow-up: add direct scoring or verdicts for `aggregate_mean_relational_family_size`.",
            "",
            "## Comparison Payload",
            "",
            f"- compare entries: `{len(compare_payload['comparisons'])}`",
            "",
        ]
    )
    return "\n".join(md_lines)


def write_track_artifacts(out_dir: Path) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_payload = _track_summary_payload()
    compare_payload = _track_compare_payload()
    _write_json(out_dir / "track_summary.json", summary_payload)
    _write_json(out_dir / "track_compare.json", compare_payload)
    (out_dir / "TRACK_REPORT.md").write_text(
        _track_report_markdown(summary_payload, compare_payload),
        encoding="utf-8",
    )
    return {
        "out_dir": str(out_dir),
        "files": [
            str(out_dir / "track_summary.json"),
            str(out_dir / "track_compare.json"),
            str(out_dir / "TRACK_REPORT.md"),
        ],
        "profiles": list(TRACK_PROFILE_NAMES),
    }


def _headline(summary: dict[str, Any]) -> dict[str, float]:
    return {
        "benchmark_corr": float(summary.get("benchmark", {}).get("mean_corr", 0.0)),
        "benchmark_mae": float(summary.get("benchmark", {}).get("mean_mae", 0.0)),
        "heldout_real_branch_fraction": float(summary.get("heldout", {}).get("mean_real_branch_fraction", 0.0)),
        "heldout_signature_families": float(summary.get("heldout", {}).get("mean_num_relational_signature_families", 0.0)),
        "heldout_signature_confidence": float(summary.get("heldout", {}).get("mean_relational_signature_confidence", 0.0)),
        "library_signature_families": float(
            summary.get("law_token_library", {}).get("aggregate_relational_signature_library", {}).get("num_families", 0.0)
        ),
        "metamer_prefix_cos": float(summary.get("metamer", {}).get("mean_prefix_cos", 0.0)),
        "metamer_family_delta": float(summary.get("metamer", {}).get("mean_family_count_delta", 0.0)),
    }


def run_signature_experiment(
    *,
    profile: str,
    out_dir: Path,
    checkpoint_dir: Path,
    init_config_path: Path,
    iterations: int,
    population: int,
    elite_count: int,
    seed: int,
    device_name: str,
    clip_seconds: int,
    phase_blend: float,
    cases_path: Path,
) -> dict[str, Any]:
    from benchmark_audio_continuity import benchmark_folder
    from benchmark_circleworld_real_anchor import run_benchmark
    from build_law_token_library import build_library
    from evaluate_circleworld import evaluate_config
    from evaluate_relational_metamers import evaluate_relational_metamers
    from train_circleworld_real_anchor import train_circleworld_real_anchor

    score_cfg = _signature_score_cfg(profile)
    profile_metadata = _profile_metadata(profile)
    train_summary = train_circleworld_real_anchor(
        out_dir=out_dir / "train",
        checkpoint_dir=checkpoint_dir,
        iterations=iterations,
        population=population,
        elite_count=elite_count,
        seed=seed,
        device_name=device_name,
        clip_seconds=clip_seconds,
        phase_blend=phase_blend,
        init_config_path=init_config_path,
        score_cfg=score_cfg,
    )
    config_path = Path(train_summary["checkpoint"])
    heldout = evaluate_config(
        config_path=config_path,
        out_dir=out_dir / "heldout_eval",
        time_steps=128,
        device_name=device_name,
    )
    benchmark = run_benchmark(
        config_path=config_path,
        out_dir=out_dir / "benchmark_expanded",
        device_name=device_name,
        clip_seconds=clip_seconds,
        phase_blend=phase_blend,
    )
    continuity_circleworld = benchmark_folder(
        folder=out_dir / "benchmark_expanded",
        pattern="*_circleworld.wav",
        out_path=out_dir / "benchmark_expanded" / "continuity_circleworld.json",
        min_loop_seconds=0.5,
        max_loop_seconds=4.0,
        chunk_seconds=1.0,
    )
    continuity_reference = benchmark_folder(
        folder=out_dir / "benchmark_expanded",
        pattern="*_reference.wav",
        out_path=out_dir / "benchmark_expanded" / "continuity_reference.json",
        min_loop_seconds=0.5,
        max_loop_seconds=4.0,
        chunk_seconds=1.0,
    )
    library = build_library(
        config_path=config_path,
        out_dir=out_dir / "law_token_library",
        cases_path=cases_path,
        device_name=device_name,
        clip_seconds=clip_seconds,
    )
    metamer = evaluate_relational_metamers(
        config_path=config_path,
        out_dir=out_dir / "metamer_eval",
        cases_path=cases_path,
        device_name=device_name,
        clip_seconds=clip_seconds,
    )
    summary = {
        "profile": profile,
        "profile_metadata": profile_metadata,
        "init_config_path": str(init_config_path),
        "trained_config_path": str(config_path),
        "score_cfg": score_cfg,
        "train_summary": train_summary,
        "heldout": heldout,
        "benchmark": benchmark,
        "continuity_circleworld": continuity_circleworld,
        "continuity_reference": continuity_reference,
        "law_token_library": library,
        "metamer": metamer,
    }
    summary["headline"] = _headline(summary)
    _write_json(out_dir / "relational_signature_experiment_summary.json", summary)

    md_lines = [
        "# Circleworld Relational Signature Experiment",
        "",
        f"- profile: `{profile}`",
        f"- lane: `{profile_metadata['lane']}`",
        f"- objective: {profile_metadata['objective']}",
        f"- init config: `{init_config_path}`",
        f"- trained config: `{config_path}`",
        f"- benchmark corr / mae: `{summary['headline']['benchmark_corr']:.4f}` / `{summary['headline']['benchmark_mae']:.4f}`",
        f"- heldout branch fraction: `{summary['headline']['heldout_real_branch_fraction']:.4f}`",
        f"- heldout signature families: `{summary['headline']['heldout_signature_families']:.4f}`",
        f"- heldout signature confidence: `{summary['headline']['heldout_signature_confidence']:.4f}`",
        f"- library signature families: `{summary['headline']['library_signature_families']:.0f}`",
        f"- metamer prefix cos: `{summary['headline']['metamer_prefix_cos']:.4f}`",
        f"- metamer family drift: `{summary['headline']['metamer_family_delta']:.4f}`",
        "",
    ]
    (out_dir / "RELATIONAL_SIGNATURE_EXPERIMENT.md").write_text("\n".join(md_lines), encoding="utf-8")
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description="Run a signature-focused Circleworld training/eval experiment.")
    ap.add_argument("--profile", choices=_profile_names(), default=None)
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--checkpoint-dir", default=None)
    ap.add_argument("--init-config", default=None)
    ap.add_argument("--iterations", type=int, default=4)
    ap.add_argument("--population", type=int, default=4)
    ap.add_argument("--elite-count", type=int, default=2)
    ap.add_argument("--seed", type=int, default=20260504)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--clip-seconds", type=int, default=4)
    ap.add_argument("--phase-blend", type=float, default=1.0)
    ap.add_argument("--cases-json", default=str(DEFAULT_CASES))
    ap.add_argument("--list-profiles", action="store_true")
    ap.add_argument("--write-track-artifacts", default=None)
    args = ap.parse_args()

    result_payloads: list[dict[str, Any]] = []
    training_only = bool(args.profile and not args.list_profiles and not args.write_track_artifacts)

    if args.list_profiles:
        result_payloads.append({"profiles": _profile_catalog_payload()})

    if args.write_track_artifacts:
        result_payloads.append({"track_artifacts": write_track_artifacts(Path(args.write_track_artifacts))})

    if args.profile:
        if not args.out_dir:
            ap.error("--out-dir is required when --profile is provided.")
        if not args.checkpoint_dir:
            ap.error("--checkpoint-dir is required when --profile is provided.")
        init_config = Path(args.init_config) if args.init_config else _default_init_for_profile(args.profile)
        summary = run_signature_experiment(
            profile=args.profile,
            out_dir=Path(args.out_dir),
            checkpoint_dir=Path(args.checkpoint_dir),
            init_config_path=init_config,
            iterations=int(args.iterations),
            population=int(args.population),
            elite_count=int(args.elite_count),
            seed=int(args.seed),
            device_name=args.device,
            clip_seconds=int(args.clip_seconds),
            phase_blend=float(args.phase_blend),
            cases_path=Path(args.cases_json),
        )
        if training_only:
            print(json.dumps(summary["headline"], indent=2))
            return
        result_payloads.append({"headline": summary["headline"]})

    if not result_payloads:
        ap.error("Specify --profile to run an experiment, or use --list-profiles / --write-track-artifacts.")

    if len(result_payloads) == 1:
        print(json.dumps(result_payloads[0], indent=2))
    else:
        print(json.dumps(result_payloads, indent=2))


if __name__ == "__main__":
    main()
