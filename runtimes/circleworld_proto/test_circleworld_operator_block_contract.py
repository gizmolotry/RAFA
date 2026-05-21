from __future__ import annotations

import math
import sys
from pathlib import Path


def _exit_if_help_requested_without_runtime() -> None:
    if any(arg in ("-h", "--help") for arg in sys.argv[1:]):
        print(f"usage: {Path(__file__).name} [runtime-coupled contract options]")
        print()
        print("This contract is coupled to the Circleworld runtime/signature lane.")
        print("Run it from a Circleworld integration worktree for full argument parsing and execution.")
        raise SystemExit(0)


_exit_if_help_requested_without_runtime()

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
from run_circleworld_operator_block import select_route_from_policy_payload  # noqa: E402


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
