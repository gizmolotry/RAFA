from __future__ import annotations

import argparse
import json
import math
import random
import sys
from copy import replace
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
CORE = ROOT / "core"
LINEAGE = ROOT / "lineages" / "04_positive_replacement"
for path in (ROOT, CORE, LINEAGE):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from benchmark_audio_continuity import (
    _chunk_vectors,
    _cosine,
    _frame_rms,
    _normalized_autocorr,
    analyze_wav,
)
from circleworld import (
    CircleworldConfig,
    apply_passive_packets,
    build_promoted_packets,
    hardy_littlewood_arc_field,
    phasor_apply_delta,
    promotability_field,
    summarize_circleworld_run,
)
from export_circleworld_audio import _load_circle_cfg, _phasor_to_phase, _save_wav, prepare_reference_audio
from rafa_math_tools import phasor_normalize
from stft_utils import compute_stft, inverse_stft
from config import load_config


def _safe_device(requested: str) -> torch.device:
    if requested == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _circleworld_step(
    z: torch.Tensor,
    cfg: CircleworldConfig,
    mode: str = "active_packets",
) -> tuple[torch.Tensor, dict[str, torch.Tensor], list[dict[str, torch.Tensor]]]:
    arc = hardy_littlewood_arc_field(z, cfg.qset, cfg.q_weights)
    promo = promotability_field(arc, persistence_momentum=cfg.persistence_momentum)
    packets = build_promoted_packets(z, arc, promo, cfg)
    block = {**arc, **promo}

    if mode == "no_promotion":
        return z, block, packets
    if mode == "passive_packets":
        return apply_passive_packets(z, packets, gain=cfg.child_law_gain), block, packets
    if mode == "active_packets":
        from circleworld import apply_active_packets

        return apply_active_packets(z, packets, cfg), block, packets
    raise ValueError(f"Unknown mode: {mode}")


def _run_depth_trace(
    z0: torch.Tensor,
    cfg: CircleworldConfig,
    depth: int,
    mode: str = "active_packets",
) -> dict[str, Any]:
    current = phasor_normalize(z0)
    history: list[dict[str, torch.Tensor]] = []
    packets_by_depth: list[list[dict[str, torch.Tensor]]] = []
    states: list[torch.Tensor] = [current.detach().clone()]
    for _ in range(depth):
        current, block, packets = _circleworld_step(current, cfg=cfg, mode=mode)
        history.append(block)
        packets_by_depth.append(packets)
        states.append(current.detach().clone())
    final_arc = hardy_littlewood_arc_field(current, cfg.qset, cfg.q_weights)
    final_promo = promotability_field(final_arc, persistence_momentum=cfg.persistence_momentum)
    return {
        "phase_state": current,
        "history": history,
        "packets": packets_by_depth,
        "states": states,
        "final_arc": final_arc,
        "final_promo": final_promo,
    }


def _seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _phase_noise_perturb(z: torch.Tensor, scale: float, seed: int) -> torch.Tensor:
    _seed_all(seed)
    delta = scale * torch.randn((z.size(0), z.size(1), z.size(2)), device=z.device, dtype=z.dtype)
    return phasor_apply_delta(z, delta)


def _packet_seed_perturb(
    z: torch.Tensor,
    packets: list[dict[str, torch.Tensor]],
    strength: float,
    rank: int = 0,
) -> torch.Tensor:
    if not packets:
        return z
    ordered = sorted(packets, key=lambda p: float(p["score"].item()), reverse=True)
    packet = ordered[min(rank, len(ordered) - 1)]
    return apply_passive_packets(z, [packet], gain=strength)


def _promotion_cfg(cfg: CircleworldConfig, direction: str) -> CircleworldConfig:
    if direction == "lo":
        return replace(
            cfg,
            promotion_threshold=max(0.45, cfg.promotion_threshold - 0.05),
            max_promotions=min(cfg.max_promotions + 1, 8),
            child_law_gain=min(cfg.child_law_gain + 0.05, 1.0),
        )
    if direction == "hi":
        return replace(
            cfg,
            promotion_threshold=min(0.90, cfg.promotion_threshold + 0.05),
            max_promotions=max(cfg.max_promotions - 1, 1),
            child_law_gain=max(cfg.child_law_gain - 0.05, 0.05),
        )
    raise ValueError(direction)


def _render_phase_state(mag: torch.Tensor, phase_state: torch.Tensor, stft_cfg: dict[str, Any], out_path: Path, sr: int) -> torch.Tensor:
    phase = _phasor_to_phase(phase_state)
    wav = inverse_stft(mag, phase, stft_cfg).squeeze(0)
    _save_wav(wav, out_path, sr)
    return wav.detach().cpu()


def _spectral_envelope(wav: np.ndarray, sr: int, n_fft: int = 2048, hop: int = 512) -> np.ndarray:
    if len(wav) < n_fft:
        wav = np.pad(wav, (0, n_fft - len(wav)))
    win = np.hanning(n_fft).astype(np.float32)
    vecs = []
    for start in range(0, len(wav) - n_fft + 1, hop):
        seg = wav[start : start + n_fft]
        spec = np.fft.rfft(seg * win)
        vecs.append(np.log1p(np.abs(spec)).astype(np.float32))
    if not vecs:
        vecs = [np.log1p(np.abs(np.fft.rfft(wav[:n_fft] * win))).astype(np.float32)]
    return np.mean(np.stack(vecs, axis=0), axis=0)


def _energy_contour(wav: np.ndarray, sr: int, window_seconds: float = 0.5, hop_seconds: float = 0.25) -> np.ndarray:
    frame = max(128, int(round(sr * window_seconds)))
    hop = max(64, int(round(sr * hop_seconds)))
    return _frame_rms(wav, frame=frame, hop=hop)


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) == 0 or len(b) == 0:
        return 0.0
    n = min(len(a), len(b))
    a = a[:n].astype(np.float64)
    b = b[:n].astype(np.float64)
    a = a - a.mean()
    b = b - b.mean()
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom <= 1e-12:
        return 0.0
    return float(np.dot(a, b) / denom)


def _mean_chunk_similarity(wav_a: np.ndarray, wav_b: np.ndarray, sr: int, chunk_seconds: float = 2.0) -> float:
    chunks_a = _chunk_vectors(wav_a, sr, chunk_seconds)
    chunks_b = _chunk_vectors(wav_b, sr, chunk_seconds)
    n = min(len(chunks_a), len(chunks_b))
    if n == 0:
        return 0.0
    return float(np.mean([_cosine(chunks_a[i], chunks_b[i]) for i in range(n)]))


def _short_texture_distance(wav_a: np.ndarray, wav_b: np.ndarray, sr: int, chunk_ms: float = 120.0) -> float:
    chunk_len = max(256, int(round(sr * chunk_ms / 1000.0)))
    n = min(len(wav_a), len(wav_b))
    wav_a = wav_a[:n]
    wav_b = wav_b[:n]
    vals = []
    for start in range(0, max(1, n - chunk_len + 1), chunk_len // 2):
        a = wav_a[start : start + chunk_len]
        b = wav_b[start : start + chunk_len]
        if len(a) < chunk_len or len(b) < chunk_len:
            break
        spec_a = np.log1p(np.abs(np.fft.rfft(a * np.hanning(chunk_len))))
        spec_b = np.log1p(np.abs(np.fft.rfft(b * np.hanning(chunk_len))))
        vals.append(float(np.mean(np.abs(spec_a - spec_b))))
    return float(np.mean(vals)) if vals else 0.0


def _q_profile(block: dict[str, torch.Tensor]) -> np.ndarray:
    per_q_abs = block["per_q"].abs().mean(dim=(0, 2))
    q_mass = per_q_abs / per_q_abs.sum().clamp_min(1e-8)
    return q_mass.detach().cpu().numpy().astype(np.float32)


def _packet_succession_vector(run: dict[str, Any]) -> np.ndarray:
    counts = np.array([len(level) for level in run["packets"]], dtype=np.float32)
    if counts.size == 0:
        return np.zeros(1, dtype=np.float32)
    score_means = []
    for level in run["packets"]:
        if level:
            score_means.append(float(np.mean([float(p["score"].item()) for p in level])))
        else:
            score_means.append(0.0)
    return np.concatenate([counts, np.array(score_means, dtype=np.float32)])


def _branch_label(
    coarse_env_corr: float,
    meso_chunk_similarity: float,
    fine_texture_distance: float,
    q_profile_corr: float,
    residue_delta: float,
) -> str:
    if coarse_env_corr < 0.75 or q_profile_corr < 0.60:
        return "overwrite_or_world_jump"
    if coarse_env_corr > 0.92 and meso_chunk_similarity > 0.92 and fine_texture_distance < 0.08 and residue_delta < 0.01:
        return "over_rigid_or_frozen"
    if coarse_env_corr > 0.82 and meso_chunk_similarity > 0.65 and fine_texture_distance >= 0.05:
        return "nested_sibling"
    return "ambiguous_middle"


def _compare_branch(
    base_wav: np.ndarray,
    branch_wav: np.ndarray,
    sr: int,
    base_run: dict[str, Any],
    branch_run: dict[str, Any],
    branch_path: Path,
) -> dict[str, Any]:
    env_base = _spectral_envelope(base_wav, sr)
    env_branch = _spectral_envelope(branch_wav, sr)
    energy_base = _energy_contour(base_wav, sr)
    energy_branch = _energy_contour(branch_wav, sr)
    q_base = _q_profile(base_run["final_arc"])
    q_branch = _q_profile(branch_run["final_arc"])
    continuity = analyze_wav(branch_path, min_loop_seconds=0.5, max_loop_seconds=4.0, chunk_seconds=2.0)
    coarse_env_corr = _corr(env_base, env_branch)
    energy_corr = _corr(energy_base, energy_branch)
    meso_chunk_similarity = _mean_chunk_similarity(base_wav, branch_wav, sr)
    packet_succession_similarity = _corr(_packet_succession_vector(base_run), _packet_succession_vector(branch_run))
    q_profile_corr = _corr(q_base, q_branch)
    fine_texture_distance = _short_texture_distance(base_wav, branch_wav, sr)
    residue_delta = abs(
        float(base_run["final_arc"]["minor_residue"].mean().item())
        - float(branch_run["final_arc"]["minor_residue"].mean().item())
    )
    loop_period_delta = abs(
        float(continuity["loop_period_seconds"])
        - float(analyze_wav(Path(base_run["render_wav"]), 0.5, 4.0, 2.0)["loop_period_seconds"])
    )
    label = _branch_label(
        coarse_env_corr=coarse_env_corr,
        meso_chunk_similarity=meso_chunk_similarity,
        fine_texture_distance=fine_texture_distance,
        q_profile_corr=q_profile_corr,
        residue_delta=residue_delta,
    )
    return {
        "coarse_env_corr": coarse_env_corr,
        "coarse_energy_corr": energy_corr,
        "meso_chunk_similarity": meso_chunk_similarity,
        "meso_packet_succession_similarity": packet_succession_similarity,
        "meso_loop_period_delta": loop_period_delta,
        "fine_texture_distance": fine_texture_distance,
        "fine_q_profile_corr": q_profile_corr,
        "fine_minor_residue_delta": residue_delta,
        "continuity": continuity,
        "label": label,
    }


def _make_case(
    name: str,
    reference_wav: Path,
    config_path: Path,
    out_dir: Path,
    device: torch.device,
    total_depth: int,
    fork_depth: int,
    mode: str,
) -> dict[str, Any]:
    cfg = _load_circle_cfg(config_path)
    wav, sr = prepare_reference_audio(reference_wav, device_name=device.type, clip_seconds_override=10)
    stft_cfg = load_config()["data"]["stft"]
    mag, phase = compute_stft(wav, stft_cfg)
    z0 = phasor_normalize(torch.stack([torch.cos(phase), torch.sin(phase)], dim=-1))

    trace = _run_depth_trace(z0, cfg=cfg, depth=total_depth, mode=mode)
    states = trace["states"]
    early_state = states[fork_depth].detach().clone()
    mid_state = states[min(total_depth - 1, max(fork_depth + 1, 1))].detach().clone()
    late_state = trace["phase_state"].detach().clone()

    case_dir = out_dir / name
    case_dir.mkdir(parents=True, exist_ok=True)

    early_wav_path = case_dir / "state_early.wav"
    mid_wav_path = case_dir / "state_mid.wav"
    late_wav_path = case_dir / "state_late.wav"
    _render_phase_state(mag, early_state, stft_cfg, early_wav_path, sr)
    _render_phase_state(mag, mid_state, stft_cfg, mid_wav_path, sr)
    _render_phase_state(mag, late_state, stft_cfg, late_wav_path, sr)

    early_packets = trace["packets"][max(0, fork_depth - 1)] if fork_depth > 0 else []
    remaining_depth = max(1, total_depth - fork_depth)

    branches: list[dict[str, Any]] = [
        {"name": "base", "state": early_state, "cfg": cfg},
        {"name": "fine_noise_small", "state": _phase_noise_perturb(early_state, 0.015, seed=101), "cfg": cfg},
        {"name": "fine_noise_large", "state": _phase_noise_perturb(early_state, 0.05, seed=202), "cfg": cfg},
        {"name": "packet_primary", "state": _packet_seed_perturb(early_state, early_packets, strength=0.18, rank=0), "cfg": cfg},
        {"name": "packet_secondary", "state": _packet_seed_perturb(early_state, early_packets, strength=0.12, rank=1), "cfg": cfg},
        {"name": "promotion_lo", "state": early_state, "cfg": _promotion_cfg(cfg, "lo")},
        {"name": "promotion_hi", "state": early_state, "cfg": _promotion_cfg(cfg, "hi")},
    ]

    rendered: dict[str, dict[str, Any]] = {}
    base_wave_np: np.ndarray | None = None
    base_run: dict[str, Any] | None = None

    for branch in branches:
        branch_name = branch["name"]
        run = _run_depth_trace(branch["state"], cfg=branch["cfg"], depth=remaining_depth, mode=mode)
        out_path = case_dir / f"{branch_name}.wav"
        wav_out = _render_phase_state(mag, run["phase_state"], stft_cfg, out_path, sr).numpy()
        run["render_wav"] = str(out_path)
        rendered[branch_name] = {
            "summary": summarize_circleworld_run(run),
            "wav_path": str(out_path),
            "cfg": {
                "promotion_threshold": branch["cfg"].promotion_threshold,
                "max_promotions": branch["cfg"].max_promotions,
                "child_law_gain": branch["cfg"].child_law_gain,
            },
            "run": run,
            "wave_np": wav_out,
        }
        if branch_name == "base":
            base_wave_np = wav_out
            base_run = run

    assert base_wave_np is not None and base_run is not None

    branch_metrics: dict[str, Any] = {}
    for branch_name, payload in rendered.items():
        if branch_name == "base":
            continue
        branch_metrics[branch_name] = _compare_branch(
            base_wav=base_wave_np,
            branch_wav=payload["wave_np"],
            sr=sr,
            base_run=base_run,
            branch_run=payload["run"],
            branch_path=Path(payload["wav_path"]),
        )
        branch_metrics[branch_name]["summary"] = payload["summary"]
        branch_metrics[branch_name]["cfg"] = payload["cfg"]
        branch_metrics[branch_name]["wav_path"] = payload["wav_path"]

    counts = {
        "nested_sibling": sum(1 for m in branch_metrics.values() if m["label"] == "nested_sibling"),
        "overwrite_or_world_jump": sum(1 for m in branch_metrics.values() if m["label"] == "overwrite_or_world_jump"),
        "over_rigid_or_frozen": sum(1 for m in branch_metrics.values() if m["label"] == "over_rigid_or_frozen"),
        "ambiguous_middle": sum(1 for m in branch_metrics.values() if m["label"] == "ambiguous_middle"),
    }
    if counts["nested_sibling"] >= 3 and counts["overwrite_or_world_jump"] == 0:
        verdict = "evidence_of_nested_commitment"
    elif counts["over_rigid_or_frozen"] >= 3:
        verdict = "over_rigid_attractor"
    elif counts["overwrite_or_world_jump"] >= 2:
        verdict = "overwrite_dominant"
    else:
        verdict = "mixed_or_inconclusive"

    result = {
        "case": name,
        "reference_wav": str(reference_wav),
        "config_path": str(config_path),
        "total_depth": total_depth,
        "fork_depth": fork_depth,
        "remaining_depth": remaining_depth,
        "state_wavs": {
            "early": str(early_wav_path),
            "mid": str(mid_wav_path),
            "late": str(late_wav_path),
        },
        "branch_metrics": branch_metrics,
        "counts": counts,
        "verdict": verdict,
    }
    (case_dir / "nested_commitment_summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def main() -> None:
    ap = argparse.ArgumentParser(description="Run a Circleworld nested-commitment / fork-resume test.")
    ap.add_argument("--config", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--depth", type=int, default=3)
    ap.add_argument("--fork-depth", type=int, default=1)
    ap.add_argument("--mode", default="active_packets", choices=["no_promotion", "passive_packets", "active_packets"])
    ap.add_argument("--case-json", default=None, help="Optional JSON list of {name, reference_wav}")
    args = ap.parse_args()

    device = _safe_device(args.device)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.case_json:
        cases = json.loads(Path(args.case_json).read_text(encoding="utf-8"))
    else:
        cases = [
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

    rows = []
    for case in cases:
        rows.append(
            _make_case(
                name=str(case["name"]),
                reference_wav=Path(case["reference_wav"]),
                config_path=Path(args.config),
                out_dir=out_dir,
                device=device,
                total_depth=int(args.depth),
                fork_depth=int(args.fork_depth),
                mode=args.mode,
            )
        )

    aggregate = {
        "config_path": str(Path(args.config)),
        "device": str(device),
        "depth": int(args.depth),
        "fork_depth": int(args.fork_depth),
        "mode": args.mode,
        "cases": rows,
    }
    verdict_counts: dict[str, int] = {}
    for row in rows:
        verdict_counts[row["verdict"]] = verdict_counts.get(row["verdict"], 0) + 1
    aggregate["verdict_counts"] = verdict_counts
    aggregate["overall_read"] = (
        "nested_commitment_signal_present"
        if verdict_counts.get("evidence_of_nested_commitment", 0) >= max(1, math.ceil(len(rows) / 2))
        else "nested_commitment_not_yet_established"
    )

    (out_dir / "nested_commitment_report.json").write_text(json.dumps(aggregate, indent=2), encoding="utf-8")
    print(json.dumps(aggregate, indent=2))


if __name__ == "__main__":
    main()
