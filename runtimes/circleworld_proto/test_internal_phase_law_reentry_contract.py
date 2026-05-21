from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch


ROOT = Path(__file__).resolve().parents[2]
LINEAGE = ROOT / "lineages" / "04_positive_replacement"
for path in (ROOT, LINEAGE):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from circleworld import CircleworldConfig, _precondition_phase_delta  # noqa: E402
from rafa_math_tools import phasor_apply_delta  # noqa: E402


DEFAULT_OUT_DIR = (
    Path(r"C:\Users\Andrew\AppData\Local\Temp\rafa_tokenburst")
    / "internal_phase_law_reentry_contract_2026_05_07"
)


def _make_ramp_phasor(*, device: torch.device) -> torch.Tensor:
    f_bins = 8
    t_bins = 16
    freq = torch.arange(f_bins, device=device, dtype=torch.float32).view(1, f_bins, 1)
    time = torch.arange(t_bins, device=device, dtype=torch.float32).view(1, 1, t_bins)
    phase = 0.17 * time + 0.011 * freq * time
    return torch.stack((torch.cos(phase), torch.sin(phase)), dim=-1)


def run_contract(*, device: str) -> dict[str, Any]:
    torch_device = torch.device(device)
    z = _make_ramp_phasor(device=torch_device)
    delta = torch.zeros(z.shape[:-1], device=torch_device, dtype=z.dtype)

    default_cfg = CircleworldConfig()
    default_delta = _precondition_phase_delta(z, delta, default_cfg)

    reentry_cfg = CircleworldConfig(phase_law_reentry_mix=0.06, phase_law_reentry_accel_mix=0.50)
    reentry_delta = _precondition_phase_delta(z, delta, reentry_cfg)
    reentry_state = phasor_apply_delta(z, reentry_delta)

    causal_cfg = CircleworldConfig(
        phase_law_reentry_mix=0.06,
        phase_law_reentry_accel_mix=0.50,
        phase_law_reentry_causal=True,
    )
    causal_delta = _precondition_phase_delta(z, delta, causal_cfg)
    causal_state = phasor_apply_delta(z, causal_delta)

    noncausal_t0 = float(reentry_delta[:, :, 0].abs().max().detach().cpu())
    causal_t0 = float(causal_delta[:, :, 0].abs().max().detach().cpu())
    default_noop = float((default_delta - delta).abs().max().detach().cpu())
    reentry_delta_mean = float((reentry_delta - delta).abs().mean().detach().cpu())
    causal_delta_mean = float((causal_delta - delta).abs().mean().detach().cpu())
    reentry_norm_error = float((reentry_state.norm(dim=-1) - 1.0).abs().max().detach().cpu())
    causal_norm_error = float((causal_state.norm(dim=-1) - 1.0).abs().max().detach().cpu())

    checks = {
        "default_noop": default_noop == 0.0,
        "reentry_changes_delta": reentry_delta_mean > 0.0,
        "reentry_unit_phasor": reentry_norm_error <= 1.0e-6,
        "causal_unit_phasor": causal_norm_error <= 1.0e-6,
        "causal_zero_t0": causal_t0 == 0.0,
        "noncausal_boundary_nonzero": noncausal_t0 > 0.0,
    }
    status = "pass" if all(checks.values()) else "fail"
    return {
        "schema": "circleworld_internal_phase_law_reentry_contract_v0",
        "status": status,
        "device": str(torch_device),
        "checks": checks,
        "metrics": {
            "default_noop_max_delta": default_noop,
            "reentry_delta_mean_abs": reentry_delta_mean,
            "causal_delta_mean_abs": causal_delta_mean,
            "reentry_unit_norm_max_error": reentry_norm_error,
            "causal_unit_norm_max_error": causal_norm_error,
            "reentry_noncausal_t0_max_abs": noncausal_t0,
            "reentry_causal_t0_max_abs": causal_t0,
        },
    }


def _write_report(path: Path, payload: dict[str, Any]) -> None:
    metrics = payload["metrics"]
    checks = payload["checks"]
    lines = [
        "# Internal Phase-Law Reentry Contract",
        "",
        f"- status: `{payload['status']}`",
        f"- device: `{payload['device']}`",
        "",
        "## Checks",
        "",
    ]
    for key, value in checks.items():
        lines.append(f"- `{key}`: `{value}`")
    lines.extend(
        [
            "",
            "## Metrics",
            "",
        ]
    )
    for key, value in metrics.items():
        lines.append(f"- `{key}`: `{value}`")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit Circleworld internal phase-law reentry invariants.")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = run_contract(device=args.device)
    json_path = out_dir / "internal_phase_law_reentry_contract.json"
    md_path = out_dir / "INTERNAL_PHASE_LAW_REENTRY_CONTRACT.md"
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    _write_report(md_path, payload)
    print(json.dumps({"json": str(json_path), "markdown": str(md_path), "status": payload["status"]}, indent=2))
    if payload["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
