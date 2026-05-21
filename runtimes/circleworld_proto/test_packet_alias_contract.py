from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[2]
CORE = ROOT / "core"
LINEAGE = ROOT / "lineages" / "04_positive_replacement"
for path in (ROOT, CORE, LINEAGE):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))



def _exit_if_help_requested_without_runtime() -> None:
    if any(arg in ("-h", "--help") for arg in sys.argv[1:]):
        print(f"usage: {Path(__file__).name} [runtime-coupled contract options]")
        print()
        print("This contract is coupled to the Circleworld runtime/signature lane.")
        print("Run it from a Circleworld integration worktree for full argument parsing and execution.")
        raise SystemExit(0)


_exit_if_help_requested_without_runtime()

from circleworld import (
    CircleworldConfig,
    apply_active_packets,
    apply_passive_packets,
    clone_circleworld_state,
    perturb_circleworld_state,
)
from rafa_math_tools import phase_to_phasor, phasor_apply_delta, phasor_normalize
from test_nested_commitment import _child_only_pre_unroll


def _safe_device(requested: str) -> torch.device:
    if requested == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _as_float(value: torch.Tensor | float | int) -> float:
    if torch.is_tensor(value):
        return float(value.detach().cpu().item())
    return float(value)


def _norm_error(z: torch.Tensor) -> float:
    norm = torch.linalg.vector_norm(z.detach(), dim=-1)
    return _as_float((norm - 1.0).abs().max())


def _collect_tensors(value: Any, prefix: str = "root") -> dict[str, torch.Tensor]:
    if torch.is_tensor(value):
        return {prefix: value}
    if isinstance(value, dict):
        out: dict[str, torch.Tensor] = {}
        for key, item in value.items():
            out.update(_collect_tensors(item, f"{prefix}.{key}"))
        return out
    if isinstance(value, (list, tuple)):
        out = {}
        for idx, item in enumerate(value):
            out.update(_collect_tensors(item, f"{prefix}[{idx}]"))
        return out
    return {}


def _max_tensor_delta(reference: Any, candidate: Any) -> float:
    ref_tensors = _collect_tensors(reference)
    cand_tensors = _collect_tensors(candidate)
    deltas = []
    for path, ref in ref_tensors.items():
        cand = cand_tensors.get(path)
        if cand is None or cand.shape != ref.shape:
            continue
        deltas.append((cand.detach() - ref.detach()).abs().max())
    if not deltas:
        return 0.0
    return _as_float(torch.stack([d.reshape(()) for d in deltas]).max())


def _has_shared_tensor_storage(left: Any, right: Any) -> bool:
    left_ptrs = {int(t.data_ptr()) for t in _collect_tensors(left).values()}
    if not left_ptrs:
        return False
    return any(int(t.data_ptr()) in left_ptrs for t in _collect_tensors(right).values())


def _make_state(device: torch.device) -> torch.Tensor:
    freq = torch.linspace(-0.7, 0.9, steps=17, device=device).view(1, 17, 1)
    time = torch.linspace(0.0, 2.0, steps=23, device=device).view(1, 1, 23)
    phase = freq * time + 0.13 * torch.sin(3.0 * time)
    return phase_to_phasor(phase)


def _alias_cfg() -> CircleworldConfig:
    return replace(
        CircleworldConfig(),
        branching_mode="native_multimode_childworld",
        child_kill_threshold=-1.0,
        child_min_age_for_writeback=999,
        child_writeback_budget=0.0,
    )


def _make_branch_state(device: torch.device, cfg: CircleworldConfig) -> dict[str, Any]:
    base = _make_state(device)
    phase = torch.linspace(0.0, 1.0, steps=base.size(2), device=device, dtype=base.dtype).view(1, 1, -1)
    mode1 = phasor_apply_delta(base, 0.27 * torch.sin(2.0 * torch.pi * phase))
    phase_modes = phasor_normalize(torch.stack([base, mode1], dim=3))
    mode_logits = torch.zeros(base.shape[:-1] + (2,), device=device, dtype=base.dtype)
    mode_logits[..., 0] = 2.0
    mode_logits[..., 1] = 0.35
    mode_support = torch.zeros_like(mode_logits)
    mode_support[..., 0] = 1.0
    mode_support[..., 1] = 0.45
    mode_coherence = mode_support.clone()
    q_trace = torch.zeros(base.shape[:-1] + (2, int(cfg.q_trace_rank)), device=device, dtype=base.dtype)
    child_support = torch.zeros(base.shape[:-1], device=device, dtype=base.dtype)
    child_support[:, :, 5:18] = 0.75
    child_phase = phasor_apply_delta(mode1.clone(), 0.18 * child_support)
    child = {
        "child_id": 7,
        "parent_depth": 0,
        "spawn_time_index": 8,
        "origin_mode_index": 1,
        "support_window": (5, 18),
        "survival_age": 1,
        "phase_state": child_phase,
        "mode_support": child_support.clone(),
        "mode_coherence": (0.5 * child_support).clamp(0.0, 1.0),
        "q_trace": torch.zeros(base.shape[:-1] + (int(cfg.q_trace_rank),), device=device, dtype=base.dtype),
        "law_signature": torch.zeros(int(cfg.q_trace_rank), device=device, dtype=base.dtype),
        "writeback_budget": 0.0,
        "active": True,
        "collapsed": False,
    }
    state = {
        "phase_modes": phase_modes,
        "mode_logits": mode_logits,
        "mode_support": mode_support,
        "mode_coherence": mode_coherence,
        "mode_q_trace": q_trace,
        "mixed_phase_state": base.clone(),
        "child_worlds": [child],
        "child_event_history": [],
        "next_child_id": 8,
    }
    return state


def _passive_packet(z: torch.Tensor) -> dict[str, torch.Tensor]:
    seed = phasor_apply_delta(z[:, :, :1, :], torch.full(z[:, :, :1, 0].shape, 0.41, device=z.device))
    return {
        "batch_index": torch.tensor(0, device=z.device),
        "seed_state": seed,
        "score": torch.tensor(0.8, device=z.device),
    }


def _active_packet(z: torch.Tensor, cfg: CircleworldConfig) -> dict[str, torch.Tensor]:
    q = torch.arange(1, len(cfg.qset) + 1, device=z.device, dtype=z.dtype).view(1, -1)
    q_mass = q / q.sum(dim=-1, keepdim=True)
    f_bins = z.size(1)
    freq = torch.linspace(-0.3, 0.3, steps=f_bins, device=z.device, dtype=z.dtype).view(1, f_bins)
    return {
        "batch_index": torch.tensor(0, device=z.device),
        "q_mass": q_mass,
        "score": torch.tensor([0.65], device=z.device, dtype=z.dtype),
        "attack_law": freq,
        "decay_law": -freq,
        "reentry_bias": torch.zeros_like(freq),
    }


def _run_one(kind: str, z: torch.Tensor, cfg: CircleworldConfig) -> dict[str, Any]:
    before = z.detach().clone()
    if kind == "passive":
        out = apply_passive_packets(z, [_passive_packet(z)], gain=0.5)
    elif kind == "active":
        out = apply_active_packets(z, [_active_packet(z, cfg)], cfg=cfg)
    else:
        raise ValueError(f"Unknown alias test kind: {kind}")
    input_delta = (z - before).abs().max()
    output_delta = (out - before).abs().max()
    shares_storage = bool(out.data_ptr() == z.data_ptr())
    return {
        "kind": kind,
        "input_max_abs_delta": _as_float(input_delta),
        "output_max_abs_delta": _as_float(output_delta),
        "output_max_norm_error": _norm_error(out),
        "input_output_share_storage": shares_storage,
        "passed": bool(_as_float(input_delta) == 0.0 and _as_float(output_delta) > 0.0 and not shares_storage),
    }


def _run_empty_packet(kind: str, z: torch.Tensor, cfg: CircleworldConfig) -> dict[str, Any]:
    before = z.detach().clone()
    if kind == "passive_empty":
        out = apply_passive_packets(z, [], gain=0.5)
    elif kind == "active_empty":
        out = apply_active_packets(z, [], cfg=cfg)
    else:
        raise ValueError(f"Unknown empty packet alias test kind: {kind}")
    input_delta = (z - before).abs().max()
    output_delta = (out - before).abs().max()
    shares_storage = bool(out.data_ptr() == z.data_ptr())
    return {
        "kind": kind,
        "input_max_abs_delta": _as_float(input_delta),
        "output_max_abs_delta": _as_float(output_delta),
        "output_max_norm_error": _norm_error(out),
        "input_output_share_storage": shares_storage,
        "passed": bool(_as_float(input_delta) == 0.0 and _as_float(output_delta) == 0.0 and not shares_storage),
    }


def _run_branch_clone_alias(device: torch.device, cfg: CircleworldConfig) -> dict[str, Any]:
    base = _make_branch_state(device, cfg)
    before = clone_circleworld_state(base)
    branch_a = clone_circleworld_state(base)
    branch_b = clone_circleworld_state(base)
    branch_a["phase_modes"][..., 1, :] = phasor_apply_delta(
        branch_a["phase_modes"][..., 1, :],
        torch.full(branch_a["phase_modes"][..., 1, 0].shape, 0.31, device=device, dtype=branch_a["phase_modes"].dtype),
    )
    branch_a["child_worlds"][0]["mode_support"].mul_(0.25)
    branch_a["child_worlds"][0]["phase_state"] = phasor_apply_delta(
        branch_a["child_worlds"][0]["phase_state"],
        0.19 * branch_a["child_worlds"][0]["mode_support"],
    )
    base_delta = _max_tensor_delta(before, base)
    sibling_delta = _max_tensor_delta(before, branch_b)
    edited_delta = _max_tensor_delta(before, branch_a)
    shares_storage = (
        _has_shared_tensor_storage(base, branch_a)
        or _has_shared_tensor_storage(base, branch_b)
        or _has_shared_tensor_storage(branch_a, branch_b)
    )
    return {
        "kind": "nested_branch_clone",
        "input_max_abs_delta": base_delta,
        "sibling_max_abs_delta": sibling_delta,
        "output_max_abs_delta": edited_delta,
        "output_max_norm_error": _norm_error(branch_a["phase_modes"]),
        "input_output_share_storage": shares_storage,
        "passed": bool(base_delta == 0.0 and sibling_delta == 0.0 and edited_delta > 0.0 and not shares_storage),
    }


def _run_child_perturb_alias(device: torch.device, cfg: CircleworldConfig) -> dict[str, Any]:
    base = _make_branch_state(device, cfg)
    before = clone_circleworld_state(base)
    out = perturb_circleworld_state(base, cfg, kind="child_support_boost", strength=0.65, seed=123)
    input_delta = _max_tensor_delta(before, base)
    output_delta = _max_tensor_delta(before, out)
    shares_storage = _has_shared_tensor_storage(base, out)
    return {
        "kind": "child_perturbation",
        "input_max_abs_delta": input_delta,
        "output_max_abs_delta": output_delta,
        "output_max_norm_error": _norm_error(out["phase_modes"]),
        "input_output_share_storage": shares_storage,
        "passed": bool(input_delta == 0.0 and output_delta > 0.0 and not shares_storage),
    }


def _run_child_pre_unroll_alias(device: torch.device, cfg: CircleworldConfig) -> dict[str, Any]:
    base = _make_branch_state(device, cfg)
    before = clone_circleworld_state(base)
    out = _child_only_pre_unroll(base, cfg, steps=1, parent_write_scale=0.0, mode0_protect=0.80)
    input_delta = _max_tensor_delta(before, base)
    output_delta = _max_tensor_delta(before, out)
    shares_storage = _has_shared_tensor_storage(base, out)
    return {
        "kind": "child_pre_unroll",
        "input_max_abs_delta": input_delta,
        "output_max_abs_delta": output_delta,
        "output_max_norm_error": _norm_error(out["phase_modes"]),
        "input_output_share_storage": shares_storage,
        "passed": bool(input_delta == 0.0 and output_delta > 0.0 and not shares_storage),
    }


def run_alias_audit(out_dir: Path, device_name: str, tolerance: float = 1.0e-7) -> dict[str, Any]:
    device = _safe_device(device_name)
    cfg = _alias_cfg()
    z = _make_state(device)
    rows = [
        _run_one("passive", z.clone(), cfg),
        _run_one("active", z.clone(), cfg),
        _run_empty_packet("passive_empty", z.clone(), cfg),
        _run_empty_packet("active_empty", z.clone(), cfg),
        _run_branch_clone_alias(device, cfg),
        _run_child_perturb_alias(device, cfg),
        _run_child_pre_unroll_alias(device, cfg),
    ]
    for row in rows:
        row["passed"] = bool(
            row["passed"]
            and float(row["input_max_abs_delta"]) <= tolerance
            and float(row["output_max_norm_error"]) <= 1.0e-5
        )
    payload = {
        "device_requested": device_name,
        "device": str(device),
        "tolerance": float(tolerance),
        "status": "pass" if all(row["passed"] for row in rows) else "fail",
        "rows": rows,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "packet_alias_contract_report.json"
    md_path = out_dir / "PACKET_ALIAS_CONTRACT.md"
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    lines = [
        "# Circleworld Packet Alias Contract",
        "",
        f"- status: `{payload['status']}`",
        f"- device: `{payload['device']}`",
        "",
        "| kind | input max delta | sibling max delta | output max delta | output norm error | shares storage | status |",
        "|---|---:|---:|---:|---:|---|---|",
    ]
    for row in rows:
        lines.append(
            "| {kind} | {input_delta} | {sibling_delta} | {output_delta} | {norm_error} | {shares} | {status} |".format(
                kind=row["kind"],
                input_delta=row["input_max_abs_delta"],
                sibling_delta=row.get("sibling_max_abs_delta", ""),
                output_delta=row["output_max_abs_delta"],
                norm_error=row["output_max_norm_error"],
                shares=row["input_output_share_storage"],
                status="pass" if row["passed"] else "fail",
            )
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"json": str(json_path), "markdown": str(md_path), "status": payload["status"]}


def main() -> None:
    ap = argparse.ArgumentParser(description="Audit packet applicators for input aliasing/mutation.")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--tolerance", type=float, default=1.0e-7)
    args = ap.parse_args()
    print(json.dumps(run_alias_audit(Path(args.out_dir), args.device, args.tolerance), indent=2))


if __name__ == "__main__":
    main()
