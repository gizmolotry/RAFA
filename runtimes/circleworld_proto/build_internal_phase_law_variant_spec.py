from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable


TOKENBURST_ROOT = Path(r"C:\Users\Andrew\AppData\Local\Temp\rafa_tokenburst")
DEFAULT_OUT = (
    TOKENBURST_ROOT
    / "internal_phase_law_variant_specs_2026_05_07"
    / "internal_phase_law_focused_v1.json"
)
DEFAULT_REPORT = DEFAULT_OUT.with_suffix(".md")

PHASE_LAW_KEYS = (
    "phase_law_precondition_gain",
    "phase_law_velocity_mix",
    "phase_law_stability_gain",
    "phase_law_softclip",
    "phase_law_low_rank",
    "phase_law_consensus_mix",
    "phase_law_consensus_damping",
    "phase_law_median_guard",
    "phase_law_local_velocity_mix",
    "phase_law_local_coherence_damping",
    "phase_law_curvature_guard",
    "phase_law_reentry_mix",
    "phase_law_reentry_accel_mix",
    "phase_law_reentry_causal",
)


def _candidate(
    label: str,
    *,
    precondition: float,
    velocity: float,
    stability: float,
    softclip: float = 0.0,
    low_rank: int = 12,
    consensus_mix: float = 0.0,
    consensus_damping: float = 0.0,
    median_guard: float = 0.0,
    local_velocity_mix: float = 0.0,
    local_coherence_damping: float = 0.0,
    curvature_guard: float = 0.0,
    reentry_mix: float = 0.0,
    reentry_accel_mix: float = 0.0,
    reentry_causal: bool = False,
    description: str = "",
) -> dict[str, Any]:
    name = f"internal_{label}"
    return {
        "name": name,
        "label": label,
        "description": description or (
            f"pre={precondition}, velocity={velocity}, stability={stability}, "
            f"softclip={softclip}, low_rank={low_rank}, consensus_mix={consensus_mix}, "
            f"consensus_damping={consensus_damping}, median_guard={median_guard}, "
            f"local_velocity_mix={local_velocity_mix}, local_coherence_damping={local_coherence_damping}, "
            f"curvature_guard={curvature_guard}, reentry_mix={reentry_mix}, "
            f"reentry_accel_mix={reentry_accel_mix}, reentry_causal={reentry_causal}"
        ),
        "updates": {
            "phase_law_precondition_gain": float(precondition),
            "phase_law_velocity_mix": float(velocity),
            "phase_law_stability_gain": float(stability),
            "phase_law_softclip": float(softclip),
            "phase_law_low_rank": int(low_rank),
            "phase_law_consensus_mix": float(consensus_mix),
            "phase_law_consensus_damping": float(consensus_damping),
            "phase_law_median_guard": float(median_guard),
            "phase_law_local_velocity_mix": float(local_velocity_mix),
            "phase_law_local_coherence_damping": float(local_coherence_damping),
            "phase_law_curvature_guard": float(curvature_guard),
            "phase_law_reentry_mix": float(reentry_mix),
            "phase_law_reentry_accel_mix": float(reentry_accel_mix),
            "phase_law_reentry_causal": bool(reentry_causal),
        },
    }


def _focused_v1() -> list[dict[str, Any]]:
    return [
        _candidate("v015_vel002_stab15_lr8", precondition=0.15, velocity=0.02, stability=1.5, low_rank=8),
        _candidate("v020_vel004_stab15_lr8", precondition=0.20, velocity=0.04, stability=1.5, low_rank=8),
        _candidate("v025_vel005_stab15_lr8", precondition=0.25, velocity=0.05, stability=1.5, low_rank=8),
        _candidate("v025_vel005_stab20_lr8", precondition=0.25, velocity=0.05, stability=2.0, low_rank=8),
        _candidate("v025_vel005_stab20_lr16", precondition=0.25, velocity=0.05, stability=2.0, low_rank=16),
        _candidate("v025_vel005_stab30_lr12", precondition=0.25, velocity=0.05, stability=3.0, low_rank=12),
        _candidate("v030_vel005_stab20_lr12", precondition=0.30, velocity=0.05, stability=2.0, low_rank=12),
        _candidate("v035_vel005_stab20_lr12", precondition=0.35, velocity=0.05, stability=2.0, low_rank=12),
        _candidate("v025_vel008_stab20_lr12", precondition=0.25, velocity=0.08, stability=2.0, low_rank=12),
        _candidate("v025_vel010_stab20_lr12", precondition=0.25, velocity=0.10, stability=2.0, low_rank=12),
        _candidate("v025_vel005_stab20_clip020_lr12", precondition=0.25, velocity=0.05, stability=2.0, softclip=0.20, low_rank=12),
        _candidate("v025_vel005_stab20_clip035_lr12", precondition=0.25, velocity=0.05, stability=2.0, softclip=0.35, low_rank=12),
        _candidate("v035_vel005_stab20_clip020_lr12", precondition=0.35, velocity=0.05, stability=2.0, softclip=0.20, low_rank=12),
        _candidate("gate015_stab15_lr8", precondition=0.15, velocity=0.00, stability=1.5, low_rank=8),
        _candidate("gate025_stab15_lr12", precondition=0.25, velocity=0.00, stability=1.5, low_rank=12),
        _candidate("gate035_stab25_lr12", precondition=0.35, velocity=0.00, stability=2.5, low_rank=12),
    ]


def _wide_v1() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for precondition in (0.10, 0.20, 0.30, 0.40):
        for velocity in (0.00, 0.03, 0.06, 0.10):
            for stability in (1.25, 2.00):
                label = f"wide_p{int(precondition * 100):03d}_v{int(velocity * 100):03d}_s{int(stability * 100):03d}"
                rows.append(
                    _candidate(
                        label,
                        precondition=precondition,
                        velocity=velocity,
                        stability=stability,
                        low_rank=12,
                    )
                )
    return rows


def _v2_guard_v1() -> list[dict[str, Any]]:
    """Guarded phase-only candidates aimed at robust absolute rows, not bad-baseline rescue."""
    return [
        _candidate(
            "v2_cons004_damp025_guard025_p040_v010_s125",
            precondition=0.40,
            velocity=0.10,
            stability=1.25,
            consensus_mix=0.04,
            consensus_damping=0.25,
            median_guard=0.25,
            low_rank=12,
        ),
        _candidate(
            "v2_cons006_damp035_guard035_p040_v010_s125",
            precondition=0.40,
            velocity=0.10,
            stability=1.25,
            consensus_mix=0.06,
            consensus_damping=0.35,
            median_guard=0.35,
            low_rank=12,
        ),
        _candidate(
            "v2_cons004_damp050_guard050_p040_v006_s125",
            precondition=0.40,
            velocity=0.06,
            stability=1.25,
            consensus_mix=0.04,
            consensus_damping=0.50,
            median_guard=0.50,
            low_rank=12,
        ),
        _candidate(
            "v2_cons004_damp035_guard035_p025_v008_s200",
            precondition=0.25,
            velocity=0.08,
            stability=2.00,
            consensus_mix=0.04,
            consensus_damping=0.35,
            median_guard=0.35,
            low_rank=12,
        ),
        _candidate(
            "v2_cons006_damp050_guard050_p025_v008_s200",
            precondition=0.25,
            velocity=0.08,
            stability=2.00,
            consensus_mix=0.06,
            consensus_damping=0.50,
            median_guard=0.50,
            low_rank=12,
        ),
        _candidate(
            "v2_cons002_damp035_guard050_p010_v010_s125",
            precondition=0.10,
            velocity=0.10,
            stability=1.25,
            consensus_mix=0.02,
            consensus_damping=0.35,
            median_guard=0.50,
            low_rank=12,
        ),
        _candidate(
            "v2_cons004_damp050_guard035_p030_v010_s125",
            precondition=0.30,
            velocity=0.10,
            stability=1.25,
            consensus_mix=0.04,
            consensus_damping=0.50,
            median_guard=0.35,
            low_rank=12,
        ),
        _candidate(
            "v2_cons006_damp050_guard020_p040_v010_s200",
            precondition=0.40,
            velocity=0.10,
            stability=2.00,
            consensus_mix=0.06,
            consensus_damping=0.50,
            median_guard=0.20,
            low_rank=12,
        ),
        _candidate(
            "v2_cons008_damp035_guard035_p025_v000_s150",
            precondition=0.25,
            velocity=0.00,
            stability=1.50,
            consensus_mix=0.08,
            consensus_damping=0.35,
            median_guard=0.35,
            low_rank=12,
        ),
        _candidate(
            "v2_cons006_damp050_guard050_p035_v000_s200",
            precondition=0.35,
            velocity=0.00,
            stability=2.00,
            consensus_mix=0.06,
            consensus_damping=0.50,
            median_guard=0.50,
            low_rank=12,
        ),
    ]


def _v3_local_v1() -> list[dict[str, Any]]:
    """Local phase-only carriers that avoid frequency-wide consensus pressure."""
    return [
        _candidate(
            "v3_local004_damp020_curv020_p025_v000_s150",
            precondition=0.25,
            velocity=0.00,
            stability=1.50,
            local_velocity_mix=0.04,
            local_coherence_damping=0.20,
            curvature_guard=0.20,
            median_guard=0.20,
            low_rank=12,
        ),
        _candidate(
            "v3_local006_damp035_curv035_p025_v000_s200",
            precondition=0.25,
            velocity=0.00,
            stability=2.00,
            local_velocity_mix=0.06,
            local_coherence_damping=0.35,
            curvature_guard=0.35,
            median_guard=0.25,
            low_rank=12,
        ),
        _candidate(
            "v3_local008_damp050_curv050_p025_v000_s200",
            precondition=0.25,
            velocity=0.00,
            stability=2.00,
            local_velocity_mix=0.08,
            local_coherence_damping=0.50,
            curvature_guard=0.50,
            median_guard=0.35,
            low_rank=12,
        ),
        _candidate(
            "v3_local006_damp050_curv035_p040_v000_s125",
            precondition=0.40,
            velocity=0.00,
            stability=1.25,
            local_velocity_mix=0.06,
            local_coherence_damping=0.50,
            curvature_guard=0.35,
            median_guard=0.25,
            low_rank=12,
        ),
        _candidate(
            "v3_local010_damp050_curv050_p040_v000_s125",
            precondition=0.40,
            velocity=0.00,
            stability=1.25,
            local_velocity_mix=0.10,
            local_coherence_damping=0.50,
            curvature_guard=0.50,
            median_guard=0.35,
            low_rank=12,
        ),
        _candidate(
            "v3_local004_raw002_damp035_curv050_p025_s200",
            precondition=0.25,
            velocity=0.02,
            stability=2.00,
            local_velocity_mix=0.04,
            local_coherence_damping=0.35,
            curvature_guard=0.50,
            median_guard=0.35,
            low_rank=12,
        ),
        _candidate(
            "v3_local006_raw003_damp050_curv050_p030_s150",
            precondition=0.30,
            velocity=0.03,
            stability=1.50,
            local_velocity_mix=0.06,
            local_coherence_damping=0.50,
            curvature_guard=0.50,
            median_guard=0.35,
            low_rank=12,
        ),
        _candidate(
            "v3_local008_raw004_damp050_curv035_p040_s200",
            precondition=0.40,
            velocity=0.04,
            stability=2.00,
            local_velocity_mix=0.08,
            local_coherence_damping=0.50,
            curvature_guard=0.35,
            median_guard=0.25,
            low_rank=12,
        ),
        _candidate(
            "v3_damp_only_curv050_p025_s200",
            precondition=0.25,
            velocity=0.00,
            stability=2.00,
            local_velocity_mix=0.00,
            local_coherence_damping=0.50,
            curvature_guard=0.50,
            median_guard=0.35,
            low_rank=12,
        ),
        _candidate(
            "v3_local004_nolr_damp035_curv035_p025_s200",
            precondition=0.25,
            velocity=0.00,
            stability=2.00,
            local_velocity_mix=0.04,
            local_coherence_damping=0.35,
            curvature_guard=0.35,
            median_guard=0.25,
            low_rank=0,
        ),
    ]


def _v4_joint_nonbad_v1() -> list[dict[str, Any]]:
    """Near-miss bridge candidates aimed at same-row direct and non-bad absolute gains."""
    return [
        _candidate(
            "v4_local002_damp010_curv010_p015_s125_lr0",
            precondition=0.15,
            velocity=0.00,
            stability=1.25,
            local_velocity_mix=0.02,
            local_coherence_damping=0.10,
            curvature_guard=0.10,
            median_guard=0.10,
            low_rank=0,
        ),
        _candidate(
            "v4_local003_damp010_curv015_p020_s125_lr0",
            precondition=0.20,
            velocity=0.00,
            stability=1.25,
            local_velocity_mix=0.03,
            local_coherence_damping=0.10,
            curvature_guard=0.15,
            median_guard=0.12,
            low_rank=0,
        ),
        _candidate(
            "v4_local003_damp015_curv015_p020_s150_lr4",
            precondition=0.20,
            velocity=0.00,
            stability=1.50,
            local_velocity_mix=0.03,
            local_coherence_damping=0.15,
            curvature_guard=0.15,
            median_guard=0.12,
            low_rank=4,
        ),
        _candidate(
            "v4_local004_damp015_curv015_p020_s125_lr0",
            precondition=0.20,
            velocity=0.00,
            stability=1.25,
            local_velocity_mix=0.04,
            local_coherence_damping=0.15,
            curvature_guard=0.15,
            median_guard=0.15,
            low_rank=0,
        ),
        _candidate(
            "v4_local004_damp015_curv020_p025_s125_lr4",
            precondition=0.25,
            velocity=0.00,
            stability=1.25,
            local_velocity_mix=0.04,
            local_coherence_damping=0.15,
            curvature_guard=0.20,
            median_guard=0.15,
            low_rank=4,
        ),
        _candidate(
            "v4_local004_damp020_curv015_p020_s150_lr4",
            precondition=0.20,
            velocity=0.00,
            stability=1.50,
            local_velocity_mix=0.04,
            local_coherence_damping=0.20,
            curvature_guard=0.15,
            median_guard=0.15,
            low_rank=4,
        ),
        _candidate(
            "v4_local004_damp020_curv020_p025_s125_lr0",
            precondition=0.25,
            velocity=0.00,
            stability=1.25,
            local_velocity_mix=0.04,
            local_coherence_damping=0.20,
            curvature_guard=0.20,
            median_guard=0.18,
            low_rank=0,
        ),
        _candidate(
            "v4_local005_damp020_curv020_p025_s125_lr0",
            precondition=0.25,
            velocity=0.00,
            stability=1.25,
            local_velocity_mix=0.05,
            local_coherence_damping=0.20,
            curvature_guard=0.20,
            median_guard=0.18,
            low_rank=0,
        ),
        _candidate(
            "v4_local005_damp020_curv025_p025_s150_lr4",
            precondition=0.25,
            velocity=0.00,
            stability=1.50,
            local_velocity_mix=0.05,
            local_coherence_damping=0.20,
            curvature_guard=0.25,
            median_guard=0.18,
            low_rank=4,
        ),
        _candidate(
            "v4_local005_damp025_curv020_p030_s125_lr0",
            precondition=0.30,
            velocity=0.00,
            stability=1.25,
            local_velocity_mix=0.05,
            local_coherence_damping=0.25,
            curvature_guard=0.20,
            median_guard=0.20,
            low_rank=0,
        ),
        _candidate(
            "v4_local006_damp020_curv020_p020_s125_lr0",
            precondition=0.20,
            velocity=0.00,
            stability=1.25,
            local_velocity_mix=0.06,
            local_coherence_damping=0.20,
            curvature_guard=0.20,
            median_guard=0.18,
            low_rank=0,
        ),
        _candidate(
            "v4_local004_damp010_curv010_p000_s125_lr0",
            precondition=0.00,
            velocity=0.00,
            stability=1.25,
            local_velocity_mix=0.04,
            local_coherence_damping=0.10,
            curvature_guard=0.10,
            median_guard=0.10,
            low_rank=0,
            description="No global stability precondition; tests local carrier alone near the non-bad ridge.",
        ),
    ]


def _v5_low_energy_row_v1() -> list[dict[str, Any]]:
    """Target the v4 flat/low-energy near miss without reintroducing broad pressure."""
    return [
        _candidate(
            "v5_lowrow_local004_damp008_curv008_p000_s100_lr0",
            precondition=0.00,
            velocity=0.00,
            stability=1.00,
            local_velocity_mix=0.04,
            local_coherence_damping=0.08,
            curvature_guard=0.08,
            median_guard=0.08,
            low_rank=0,
        ),
        _candidate(
            "v5_lowrow_local004_damp010_curv010_p000_s125_lr0",
            precondition=0.00,
            velocity=0.00,
            stability=1.25,
            local_velocity_mix=0.04,
            local_coherence_damping=0.10,
            curvature_guard=0.10,
            median_guard=0.10,
            low_rank=0,
        ),
        _candidate(
            "v5_lowrow_local004_raw001_damp010_curv010_p000_s125_lr0",
            precondition=0.00,
            velocity=0.01,
            stability=1.25,
            local_velocity_mix=0.04,
            local_coherence_damping=0.10,
            curvature_guard=0.10,
            median_guard=0.10,
            low_rank=0,
        ),
        _candidate(
            "v5_lowrow_local004_raw002_damp010_curv010_p000_s125_lr0",
            precondition=0.00,
            velocity=0.02,
            stability=1.25,
            local_velocity_mix=0.04,
            local_coherence_damping=0.10,
            curvature_guard=0.10,
            median_guard=0.10,
            low_rank=0,
        ),
        _candidate(
            "v5_lowrow_local0035_damp008_curv008_p000_s125_lr0",
            precondition=0.00,
            velocity=0.00,
            stability=1.25,
            local_velocity_mix=0.035,
            local_coherence_damping=0.08,
            curvature_guard=0.08,
            median_guard=0.08,
            low_rank=0,
        ),
        _candidate(
            "v5_lowrow_local0045_damp010_curv010_p000_s125_lr0",
            precondition=0.00,
            velocity=0.00,
            stability=1.25,
            local_velocity_mix=0.045,
            local_coherence_damping=0.10,
            curvature_guard=0.10,
            median_guard=0.10,
            low_rank=0,
        ),
        _candidate(
            "v5_lowrow_local004_damp005_curv005_p000_s125_lr0",
            precondition=0.00,
            velocity=0.00,
            stability=1.25,
            local_velocity_mix=0.04,
            local_coherence_damping=0.05,
            curvature_guard=0.05,
            median_guard=0.05,
            low_rank=0,
        ),
        _candidate(
            "v5_lowrow_local004_damp010_curv010_p005_s125_lr0",
            precondition=0.05,
            velocity=0.00,
            stability=1.25,
            local_velocity_mix=0.04,
            local_coherence_damping=0.10,
            curvature_guard=0.10,
            median_guard=0.10,
            low_rank=0,
        ),
        _candidate(
            "v5_lowrow_local004_damp010_curv010_p000_s150_lr0",
            precondition=0.00,
            velocity=0.00,
            stability=1.50,
            local_velocity_mix=0.04,
            local_coherence_damping=0.10,
            curvature_guard=0.10,
            median_guard=0.10,
            low_rank=0,
        ),
        _candidate(
            "v5_lowrow_local004_damp010_curv015_p000_s125_lr0",
            precondition=0.00,
            velocity=0.00,
            stability=1.25,
            local_velocity_mix=0.04,
            local_coherence_damping=0.10,
            curvature_guard=0.15,
            median_guard=0.10,
            low_rank=0,
        ),
        _candidate(
            "v5_lowrow_local005_damp010_curv010_p000_s125_lr0",
            precondition=0.00,
            velocity=0.00,
            stability=1.25,
            local_velocity_mix=0.05,
            local_coherence_damping=0.10,
            curvature_guard=0.10,
            median_guard=0.10,
            low_rank=0,
        ),
        _candidate(
            "v5_lowrow_local004_damp010_curv010_p000_s125_lr2",
            precondition=0.00,
            velocity=0.00,
            stability=1.25,
            local_velocity_mix=0.04,
            local_coherence_damping=0.10,
            curvature_guard=0.10,
            median_guard=0.10,
            low_rank=2,
            description="Minimal low-rank bridge from the v4 near-miss row.",
        ),
    ]


def _v6_low_energy_win_v1() -> list[dict[str, Any]]:
    """Search lighter guards and small raw velocity to move the low-energy row's median/win rate."""
    return [
        _candidate(
            "v6_lowwin_local004_damp000_curv000_p000_s125_lr0",
            precondition=0.00,
            velocity=0.00,
            stability=1.25,
            local_velocity_mix=0.04,
            local_coherence_damping=0.00,
            curvature_guard=0.00,
            median_guard=0.00,
            low_rank=0,
        ),
        _candidate(
            "v6_lowwin_local004_damp002_curv002_p000_s125_lr0",
            precondition=0.00,
            velocity=0.00,
            stability=1.25,
            local_velocity_mix=0.04,
            local_coherence_damping=0.02,
            curvature_guard=0.02,
            median_guard=0.02,
            low_rank=0,
        ),
        _candidate(
            "v6_lowwin_local004_damp003_curv003_p000_s125_lr0",
            precondition=0.00,
            velocity=0.00,
            stability=1.25,
            local_velocity_mix=0.04,
            local_coherence_damping=0.03,
            curvature_guard=0.03,
            median_guard=0.03,
            low_rank=0,
        ),
        _candidate(
            "v6_lowwin_local004_damp004_curv004_p000_s125_lr0",
            precondition=0.00,
            velocity=0.00,
            stability=1.25,
            local_velocity_mix=0.04,
            local_coherence_damping=0.04,
            curvature_guard=0.04,
            median_guard=0.04,
            low_rank=0,
        ),
        _candidate(
            "v6_lowwin_local004_raw001_damp003_curv003_p000_s125_lr0",
            precondition=0.00,
            velocity=0.01,
            stability=1.25,
            local_velocity_mix=0.04,
            local_coherence_damping=0.03,
            curvature_guard=0.03,
            median_guard=0.03,
            low_rank=0,
        ),
        _candidate(
            "v6_lowwin_local004_raw002_damp003_curv003_p000_s125_lr0",
            precondition=0.00,
            velocity=0.02,
            stability=1.25,
            local_velocity_mix=0.04,
            local_coherence_damping=0.03,
            curvature_guard=0.03,
            median_guard=0.03,
            low_rank=0,
        ),
        _candidate(
            "v6_lowwin_local004_raw003_damp003_curv003_p000_s125_lr0",
            precondition=0.00,
            velocity=0.03,
            stability=1.25,
            local_velocity_mix=0.04,
            local_coherence_damping=0.03,
            curvature_guard=0.03,
            median_guard=0.03,
            low_rank=0,
        ),
        _candidate(
            "v6_lowwin_local0035_damp003_curv003_p000_s125_lr0",
            precondition=0.00,
            velocity=0.00,
            stability=1.25,
            local_velocity_mix=0.035,
            local_coherence_damping=0.03,
            curvature_guard=0.03,
            median_guard=0.03,
            low_rank=0,
        ),
        _candidate(
            "v6_lowwin_local0045_damp003_curv003_p000_s125_lr0",
            precondition=0.00,
            velocity=0.00,
            stability=1.25,
            local_velocity_mix=0.045,
            local_coherence_damping=0.03,
            curvature_guard=0.03,
            median_guard=0.03,
            low_rank=0,
        ),
        _candidate(
            "v6_lowwin_local004_damp003_curv000_p000_s125_lr0",
            precondition=0.00,
            velocity=0.00,
            stability=1.25,
            local_velocity_mix=0.04,
            local_coherence_damping=0.03,
            curvature_guard=0.00,
            median_guard=0.03,
            low_rank=0,
        ),
        _candidate(
            "v6_lowwin_local004_damp000_curv003_p000_s125_lr0",
            precondition=0.00,
            velocity=0.00,
            stability=1.25,
            local_velocity_mix=0.04,
            local_coherence_damping=0.00,
            curvature_guard=0.03,
            median_guard=0.03,
            low_rank=0,
        ),
        _candidate(
            "v6_lowwin_local004_damp003_curv003_p000_s100_lr0",
            precondition=0.00,
            velocity=0.00,
            stability=1.00,
            local_velocity_mix=0.04,
            local_coherence_damping=0.03,
            curvature_guard=0.03,
            median_guard=0.03,
            low_rank=0,
        ),
        _candidate(
            "v6_lowwin_local004_damp003_curv003_p000_s150_lr0",
            precondition=0.00,
            velocity=0.00,
            stability=1.50,
            local_velocity_mix=0.04,
            local_coherence_damping=0.03,
            curvature_guard=0.03,
            median_guard=0.03,
            low_rank=0,
        ),
        _candidate(
            "v6_lowwin_local004_raw002_damp002_curv002_p000_s100_lr0",
            precondition=0.00,
            velocity=0.02,
            stability=1.00,
            local_velocity_mix=0.04,
            local_coherence_damping=0.02,
            curvature_guard=0.02,
            median_guard=0.02,
            low_rank=0,
        ),
    ]


def _v7_low_energy_gain_ladder_v1() -> list[dict[str, Any]]:
    """Narrow v6 ridge variants for a target-row gain ladder and sign-flip diagnostics."""
    return [
        _candidate(
            "v7_gain_local004_raw003_damp003_curv003_s125",
            precondition=0.00,
            velocity=0.03,
            stability=1.25,
            local_velocity_mix=0.04,
            local_coherence_damping=0.03,
            curvature_guard=0.03,
            median_guard=0.03,
            low_rank=0,
        ),
        _candidate(
            "v7_gain_local004_raw002_damp003_curv003_s125",
            precondition=0.00,
            velocity=0.02,
            stability=1.25,
            local_velocity_mix=0.04,
            local_coherence_damping=0.03,
            curvature_guard=0.03,
            median_guard=0.03,
            low_rank=0,
        ),
        _candidate(
            "v7_gain_local004_raw0025_damp003_curv003_s125",
            precondition=0.00,
            velocity=0.025,
            stability=1.25,
            local_velocity_mix=0.04,
            local_coherence_damping=0.03,
            curvature_guard=0.03,
            median_guard=0.03,
            low_rank=0,
        ),
        _candidate(
            "v7_gain_local004_raw0035_damp003_curv003_s125",
            precondition=0.00,
            velocity=0.035,
            stability=1.25,
            local_velocity_mix=0.04,
            local_coherence_damping=0.03,
            curvature_guard=0.03,
            median_guard=0.03,
            low_rank=0,
        ),
        _candidate(
            "v7_gain_local0035_raw003_damp003_curv003_s125",
            precondition=0.00,
            velocity=0.03,
            stability=1.25,
            local_velocity_mix=0.035,
            local_coherence_damping=0.03,
            curvature_guard=0.03,
            median_guard=0.03,
            low_rank=0,
        ),
        _candidate(
            "v7_gain_local0045_raw003_damp003_curv003_s125",
            precondition=0.00,
            velocity=0.03,
            stability=1.25,
            local_velocity_mix=0.045,
            local_coherence_damping=0.03,
            curvature_guard=0.03,
            median_guard=0.03,
            low_rank=0,
        ),
        _candidate(
            "v7_gain_local004_raw003_damp000_curv003_s125",
            precondition=0.00,
            velocity=0.03,
            stability=1.25,
            local_velocity_mix=0.04,
            local_coherence_damping=0.00,
            curvature_guard=0.03,
            median_guard=0.03,
            low_rank=0,
        ),
        _candidate(
            "v7_gain_local004_raw003_damp003_curv000_s125",
            precondition=0.00,
            velocity=0.03,
            stability=1.25,
            local_velocity_mix=0.04,
            local_coherence_damping=0.03,
            curvature_guard=0.00,
            median_guard=0.03,
            low_rank=0,
        ),
        _candidate(
            "v7_gain_local004_raw003_damp002_curv002_s100",
            precondition=0.00,
            velocity=0.03,
            stability=1.00,
            local_velocity_mix=0.04,
            local_coherence_damping=0.02,
            curvature_guard=0.02,
            median_guard=0.02,
            low_rank=0,
        ),
    ]


def _v8_family_support_mask_v1() -> list[dict[str, Any]]:
    """Probe whether v7's locked families need different support masks rather than more gain."""
    return [
        _candidate(
            "v8_mask_local0035_raw003_damp003_curv003_s125",
            precondition=0.00,
            velocity=0.03,
            stability=1.25,
            local_velocity_mix=0.035,
            local_coherence_damping=0.03,
            curvature_guard=0.03,
            median_guard=0.03,
            low_rank=0,
            description="v7 target-flip selected run across support masks",
        ),
        _candidate(
            "v8_mask_local0045_raw003_damp003_curv003_s125",
            precondition=0.00,
            velocity=0.03,
            stability=1.25,
            local_velocity_mix=0.045,
            local_coherence_damping=0.03,
            curvature_guard=0.03,
            median_guard=0.03,
            low_rank=0,
            description="v7 best target-row mean run across support masks",
        ),
        _candidate(
            "v8_mask_local004_raw003_damp002_curv002_s100",
            precondition=0.00,
            velocity=0.03,
            stability=1.00,
            local_velocity_mix=0.04,
            local_coherence_damping=0.02,
            curvature_guard=0.02,
            median_guard=0.02,
            low_rank=0,
            description="v7 best global absolute/nonbad run across support masks",
        ),
        _candidate(
            "v8_mask_local004_raw002_damp003_curv003_s125",
            precondition=0.00,
            velocity=0.02,
            stability=1.25,
            local_velocity_mix=0.04,
            local_coherence_damping=0.03,
            curvature_guard=0.03,
            median_guard=0.03,
            low_rank=0,
            description="lower raw-velocity neighbor that won lower gains in v7",
        ),
        _candidate(
            "v8_mask_local004_raw0035_damp003_curv003_s125",
            precondition=0.00,
            velocity=0.035,
            stability=1.25,
            local_velocity_mix=0.04,
            local_coherence_damping=0.03,
            curvature_guard=0.03,
            median_guard=0.03,
            low_rank=0,
            description="higher raw-velocity neighbor with strong v7 mean pressure",
        ),
    ]


def _v11_phase_router_direct_v1() -> list[dict[str, Any]]:
    """Narrow router-mask follow-up aimed at crossing the strict direct-size gate."""
    return [
        _candidate(
            "v11_router_local004_raw003_damp002_curv002_s100",
            precondition=0.00,
            velocity=0.03,
            stability=1.00,
            local_velocity_mix=0.04,
            local_coherence_damping=0.02,
            curvature_guard=0.02,
            median_guard=0.02,
            low_rank=0,
            description="v10 top phase-support class row, repeated as an internal anchor.",
        ),
        _candidate(
            "v11_router_local0045_raw003_damp002_curv002_s100",
            precondition=0.00,
            velocity=0.03,
            stability=1.00,
            local_velocity_mix=0.045,
            local_coherence_damping=0.02,
            curvature_guard=0.02,
            median_guard=0.02,
            low_rank=0,
            description="Increase local phase carrier strength while keeping v10 damping/curvature.",
        ),
        _candidate(
            "v11_router_local005_raw003_damp002_curv002_s100",
            precondition=0.00,
            velocity=0.03,
            stability=1.00,
            local_velocity_mix=0.05,
            local_coherence_damping=0.02,
            curvature_guard=0.02,
            median_guard=0.02,
            low_rank=0,
            description="Stronger local carrier pressure against the direct-size gate.",
        ),
        _candidate(
            "v11_router_local004_raw0035_damp002_curv002_s100",
            precondition=0.00,
            velocity=0.035,
            stability=1.00,
            local_velocity_mix=0.04,
            local_coherence_damping=0.02,
            curvature_guard=0.02,
            median_guard=0.02,
            low_rank=0,
            description="Raise raw phase velocity from the strongest v10 class row.",
        ),
        _candidate(
            "v11_router_local0045_raw0035_damp002_curv002_s100",
            precondition=0.00,
            velocity=0.035,
            stability=1.00,
            local_velocity_mix=0.045,
            local_coherence_damping=0.02,
            curvature_guard=0.02,
            median_guard=0.02,
            low_rank=0,
            description="Joint raw/local velocity bump with v10 damping/curvature.",
        ),
        _candidate(
            "v11_router_local005_raw0035_damp002_curv002_s100",
            precondition=0.00,
            velocity=0.035,
            stability=1.00,
            local_velocity_mix=0.05,
            local_coherence_damping=0.02,
            curvature_guard=0.02,
            median_guard=0.02,
            low_rank=0,
            description="Aggressive raw/local bump while holding support-router guards fixed.",
        ),
        _candidate(
            "v11_router_local004_raw004_damp002_curv002_s100",
            precondition=0.00,
            velocity=0.04,
            stability=1.00,
            local_velocity_mix=0.04,
            local_coherence_damping=0.02,
            curvature_guard=0.02,
            median_guard=0.02,
            low_rank=0,
            description="Raw-velocity-only pressure above the v10 neighborhood.",
        ),
        _candidate(
            "v11_router_local0045_raw004_damp0025_curv0025_s110",
            precondition=0.00,
            velocity=0.04,
            stability=1.10,
            local_velocity_mix=0.045,
            local_coherence_damping=0.025,
            curvature_guard=0.025,
            median_guard=0.025,
            low_rank=0,
            description="Midpoint between v10 stable row and v10 higher-velocity target row.",
        ),
        _candidate(
            "v11_router_local004_raw0035_damp003_curv003_s125",
            precondition=0.00,
            velocity=0.035,
            stability=1.25,
            local_velocity_mix=0.04,
            local_coherence_damping=0.03,
            curvature_guard=0.03,
            median_guard=0.03,
            low_rank=0,
            description="v10 best joint-score target row, repeated as an internal anchor.",
        ),
        _candidate(
            "v11_router_local0045_raw0035_damp003_curv003_s125",
            precondition=0.00,
            velocity=0.035,
            stability=1.25,
            local_velocity_mix=0.045,
            local_coherence_damping=0.03,
            curvature_guard=0.03,
            median_guard=0.03,
            low_rank=0,
            description="Increase local carrier in the v10 best joint-score neighborhood.",
        ),
        _candidate(
            "v11_router_local004_raw004_damp003_curv003_s125",
            precondition=0.00,
            velocity=0.04,
            stability=1.25,
            local_velocity_mix=0.04,
            local_coherence_damping=0.03,
            curvature_guard=0.03,
            median_guard=0.03,
            low_rank=0,
            description="Raise raw phase velocity in the v10 best joint-score neighborhood.",
        ),
        _candidate(
            "v11_router_local005_raw004_damp003_curv003_s125",
            precondition=0.00,
            velocity=0.04,
            stability=1.25,
            local_velocity_mix=0.05,
            local_coherence_damping=0.03,
            curvature_guard=0.03,
            median_guard=0.03,
            low_rank=0,
            description="Highest-pressure router row in the bounded v11 search.",
        ),
    ]


def _v12_phase_reentry_direct_v1() -> list[dict[str, Any]]:
    """Phase-only temporal reentry carriers after v11 ruled out coefficient pressure alone."""
    rows: list[dict[str, Any]] = []
    specs = [
        ("v12_reentry_r002_a025_local004_raw003_damp002_curv002_s100", 0.02, 0.25, 0.04, 0.03, 0.02, 0.02, 1.00),
        ("v12_reentry_r004_a025_local004_raw003_damp002_curv002_s100", 0.04, 0.25, 0.04, 0.03, 0.02, 0.02, 1.00),
        ("v12_reentry_r006_a025_local004_raw003_damp002_curv002_s100", 0.06, 0.25, 0.04, 0.03, 0.02, 0.02, 1.00),
        ("v12_reentry_r004_a050_local004_raw003_damp002_curv002_s100", 0.04, 0.50, 0.04, 0.03, 0.02, 0.02, 1.00),
        ("v12_reentry_r006_a050_local004_raw003_damp002_curv002_s100", 0.06, 0.50, 0.04, 0.03, 0.02, 0.02, 1.00),
        ("v12_reentry_r008_a050_local004_raw003_damp002_curv002_s100", 0.08, 0.50, 0.04, 0.03, 0.02, 0.02, 1.00),
        ("v12_reentry_r004_a075_local004_raw0035_damp002_curv002_s100", 0.04, 0.75, 0.04, 0.035, 0.02, 0.02, 1.00),
        ("v12_reentry_r006_a075_local004_raw0035_damp002_curv002_s100", 0.06, 0.75, 0.04, 0.035, 0.02, 0.02, 1.00),
        ("v12_reentry_r008_a075_local004_raw0035_damp002_curv002_s100", 0.08, 0.75, 0.04, 0.035, 0.02, 0.02, 1.00),
        ("v12_reentry_r006_a050_local005_raw0035_damp002_curv002_s100", 0.06, 0.50, 0.05, 0.035, 0.02, 0.02, 1.00),
        ("v12_reentry_r006_a050_local000_raw000_damp000_curv000_s100", 0.06, 0.50, 0.00, 0.00, 0.00, 0.00, 1.00),
        ("v12_reentry_r008_a050_local000_raw000_damp000_curv000_s125", 0.08, 0.50, 0.00, 0.00, 0.00, 0.00, 1.25),
    ]
    for label, reentry, accel, local, velocity, damping, curvature, stability in specs:
        rows.append(
            _candidate(
                label,
                precondition=0.00,
                velocity=velocity,
                stability=stability,
                local_velocity_mix=local,
                local_coherence_damping=damping,
                curvature_guard=curvature,
                reentry_mix=reentry,
                reentry_accel_mix=accel,
                median_guard=damping,
                low_rank=0,
                description=(
                    "Default-off phase-only temporal reentry carrier; uses relative phase velocity and "
                    "wrapped acceleration with no export post-shaper or magnitude path."
                ),
            )
        )
    return rows


def _v13_causal_reentry_direct_v1() -> list[dict[str, Any]]:
    """Causality-clean temporal reentry rows without boundary lookahead or centered smoothing."""
    rows: list[dict[str, Any]] = []
    specs = [
        ("v13_causal_reentry_r004_a025_only_s100", 0.04, 0.25, 0.00, 0.00, 0.00, 0.00, 1.00),
        ("v13_causal_reentry_r006_a025_only_s100", 0.06, 0.25, 0.00, 0.00, 0.00, 0.00, 1.00),
        ("v13_causal_reentry_r008_a025_only_s125", 0.08, 0.25, 0.00, 0.00, 0.00, 0.00, 1.25),
        ("v13_causal_reentry_r004_a050_only_s100", 0.04, 0.50, 0.00, 0.00, 0.00, 0.00, 1.00),
        ("v13_causal_reentry_r006_a050_only_s100", 0.06, 0.50, 0.00, 0.00, 0.00, 0.00, 1.00),
        ("v13_causal_reentry_r008_a050_only_s125", 0.08, 0.50, 0.00, 0.00, 0.00, 0.00, 1.25),
        ("v13_causal_reentry_r006_a050_local004_raw003_damp002_curv002_s100", 0.06, 0.50, 0.04, 0.03, 0.02, 0.02, 1.00),
        ("v13_causal_reentry_r008_a050_local004_raw003_damp002_curv002_s100", 0.08, 0.50, 0.04, 0.03, 0.02, 0.02, 1.00),
        ("v13_causal_reentry_r006_a075_local004_raw0035_damp002_curv002_s100", 0.06, 0.75, 0.04, 0.035, 0.02, 0.02, 1.00),
        ("v13_causal_reentry_r008_a075_local004_raw0035_damp002_curv002_s100", 0.08, 0.75, 0.04, 0.035, 0.02, 0.02, 1.00),
    ]
    for label, reentry, accel, local, velocity, damping, curvature, stability in specs:
        rows.append(
            _candidate(
                label,
                precondition=0.00,
                velocity=velocity,
                stability=stability,
                local_velocity_mix=local,
                local_coherence_damping=damping,
                curvature_guard=curvature,
                reentry_mix=reentry,
                reentry_accel_mix=accel,
                reentry_causal=True,
                median_guard=damping,
                low_rank=0,
                description=(
                    "Default-off causal phase-only temporal reentry carrier. The reentry term uses past/current "
                    "relative phase velocity with a zero boundary and no centered smoothing."
                ),
            )
        )
    return rows


def _profile_variants(profile: str) -> list[dict[str, Any]]:
    if profile == "focused_v1":
        return _focused_v1()
    if profile == "wide_v1":
        return _wide_v1()
    if profile == "v2_guard_v1":
        return _v2_guard_v1()
    if profile == "v3_local_v1":
        return _v3_local_v1()
    if profile == "v4_joint_nonbad_v1":
        return _v4_joint_nonbad_v1()
    if profile == "v5_low_energy_row_v1":
        return _v5_low_energy_row_v1()
    if profile == "v6_low_energy_win_v1":
        return _v6_low_energy_win_v1()
    if profile == "v7_low_energy_gain_ladder_v1":
        return _v7_low_energy_gain_ladder_v1()
    if profile == "v8_family_support_mask_v1":
        return _v8_family_support_mask_v1()
    if profile == "v11_phase_router_direct_v1":
        return _v11_phase_router_direct_v1()
    if profile == "v12_phase_reentry_direct_v1":
        return _v12_phase_reentry_direct_v1()
    if profile == "v13_causal_reentry_direct_v1":
        return _v13_causal_reentry_direct_v1()
    raise ValueError(f"Unknown profile: {profile}")


def _with_limit(rows: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    if limit <= 0:
        return rows
    return rows[:limit]


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _fmt(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def _write_report(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# Internal Phase-Law Variant Spec",
        "",
        f"- schema: `{payload['schema']}`",
        f"- profile: `{payload['profile']}`",
        f"- variant count: `{len(payload['variants'])}`",
        f"- intended runner: `{payload['intended_runner']}`",
        "",
        "## Variants",
        "",
        "| label | precondition | velocity | stability | softclip | low rank | consensus | damping | median guard | local velocity | local damping | curvature guard | reentry | reentry accel | reentry causal |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in payload["variants"]:
        updates = row["updates"]
        lines.append(
            "| {label} | {pre} | {vel} | {stab} | {clip} | {rank} | {cons} | {damp} | {guard} | {local_vel} | {local_damp} | {curv} | {reentry} | {reentry_accel} | {reentry_causal} |".format(
                label=row["label"],
                pre=_fmt(updates["phase_law_precondition_gain"]),
                vel=_fmt(updates["phase_law_velocity_mix"]),
                stab=_fmt(updates["phase_law_stability_gain"]),
                clip=_fmt(updates["phase_law_softclip"]),
                rank=_fmt(updates["phase_law_low_rank"]),
                cons=_fmt(updates["phase_law_consensus_mix"]),
                damp=_fmt(updates["phase_law_consensus_damping"]),
                guard=_fmt(updates["phase_law_median_guard"]),
                local_vel=_fmt(updates["phase_law_local_velocity_mix"]),
                local_damp=_fmt(updates["phase_law_local_coherence_damping"]),
                curv=_fmt(updates["phase_law_curvature_guard"]),
                reentry=_fmt(updates["phase_law_reentry_mix"]),
                reentry_accel=_fmt(updates["phase_law_reentry_accel_mix"]),
                reentry_causal=_fmt(updates["phase_law_reentry_causal"]),
            )
        )
    lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def _validate_variants(rows: Iterable[dict[str, Any]]) -> None:
    labels: set[str] = set()
    names: set[str] = set()
    for row in rows:
        label = str(row.get("label") or "")
        name = str(row.get("name") or "")
        if not label or not name:
            raise RuntimeError("Every variant requires a name and label")
        if label in labels:
            raise RuntimeError(f"Duplicate label: {label}")
        if name in names:
            raise RuntimeError(f"Duplicate name: {name}")
        labels.add(label)
        names.add(name)
        updates = row.get("updates")
        if not isinstance(updates, dict):
            raise RuntimeError(f"Variant {label} updates must be an object")
        missing = sorted(set(PHASE_LAW_KEYS) - set(updates))
        unknown = sorted(set(updates) - set(PHASE_LAW_KEYS))
        if missing or unknown:
            raise RuntimeError(f"Variant {label} has missing={missing} unknown={unknown}")


def build_spec(*, profile: str, out: Path, report: Path | None, max_variants: int) -> dict[str, Any]:
    variants = _with_limit(_profile_variants(profile), max_variants)
    _validate_variants(variants)
    payload = {
        "schema": "circleworld_internal_phase_law_variant_spec_v0",
        "profile": profile,
        "variant_count": len(variants),
        "phase_law_keys": list(PHASE_LAW_KEYS),
        "intended_runner": str(Path(__file__).with_name("run_internal_phase_law_objective_scout.py")),
        "notes": [
            "Candidate-only specs are allowed; the runner inserts a no-op baseline if none is declared.",
            "This profile searches internal recurrence coefficients only, not export post-shapers.",
        ],
        "variants": variants,
    }
    _write_json(out, payload)
    if report is not None:
        _write_report(report, payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Build reproducible internal phase-law variant specs.")
    parser.add_argument(
        "--profile",
        choices=(
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
        ),
        default="focused_v1",
    )
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--report", default=str(DEFAULT_REPORT))
    parser.add_argument("--max-variants", type=int, default=0, help="0 keeps the full profile.")
    args = parser.parse_args()

    payload = build_spec(
        profile=args.profile,
        out=Path(args.out),
        report=Path(args.report) if args.report else None,
        max_variants=int(args.max_variants),
    )
    print(
        json.dumps(
            {
                "json": str(Path(args.out)),
                "report": str(Path(args.report)) if args.report else None,
                "profile": payload["profile"],
                "variant_count": payload["variant_count"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
