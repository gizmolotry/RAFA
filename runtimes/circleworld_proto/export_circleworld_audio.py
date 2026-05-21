from __future__ import annotations

import argparse
import json
import sys
import wave
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

from config import load_config
from circleworld import CircleworldConfig, recurse_circleworld, summarize_circleworld_run
from stft_utils import compute_stft, inverse_stft
from rafa_math_tools import phase_to_phasor, phasor_normalize


def _safe_device(requested: str) -> torch.device:
    if requested == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _load_circle_cfg(path: Path) -> CircleworldConfig:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    cfg = payload["config"] if "config" in payload else payload
    return CircleworldConfig(
        qset=tuple(cfg.get("qset", (2, 3, 4, 5, 6, 8, 12))),
        q_weights=tuple(cfg["q_weights"]),
        promotion_threshold=float(cfg["promotion_threshold"]),
        max_promotions=int(cfg["max_promotions"]),
        recursion_depth=int(cfg["recursion_depth"]),
        residue_scale=float(cfg.get("residue_scale", 0.75)),
        child_law_gain=float(cfg["child_law_gain"]),
        attack_window=int(cfg["attack_window"]),
        persistence_momentum=float(cfg["persistence_momentum"]),
        soft_matryoshka_enabled=bool(cfg.get("soft_matryoshka_enabled", False)),
        matryoshka_rank=int(cfg.get("matryoshka_rank", 24)),
        phase_law_precondition_gain=float(cfg.get("phase_law_precondition_gain", 0.0)),
        phase_law_velocity_mix=float(cfg.get("phase_law_velocity_mix", 0.0)),
        phase_law_stability_gain=float(cfg.get("phase_law_stability_gain", 1.0)),
        phase_law_softclip=float(cfg.get("phase_law_softclip", 0.0)),
        phase_law_low_rank=int(cfg.get("phase_law_low_rank", 0)),
        phase_law_consensus_mix=float(cfg.get("phase_law_consensus_mix", 0.0)),
        phase_law_consensus_damping=float(cfg.get("phase_law_consensus_damping", 0.0)),
        phase_law_median_guard=float(cfg.get("phase_law_median_guard", 0.0)),
        phase_law_local_velocity_mix=float(cfg.get("phase_law_local_velocity_mix", 0.0)),
        phase_law_local_coherence_damping=float(cfg.get("phase_law_local_coherence_damping", 0.0)),
        phase_law_curvature_guard=float(cfg.get("phase_law_curvature_guard", 0.0)),
        phase_law_reentry_mix=float(cfg.get("phase_law_reentry_mix", 0.0)),
        phase_law_reentry_accel_mix=float(cfg.get("phase_law_reentry_accel_mix", 0.0)),
        phase_law_reentry_causal=bool(cfg.get("phase_law_reentry_causal", False)),
        prefix_fracs=tuple(cfg.get("prefix_fracs", (0.125, 0.25, 0.5))),
        slow_persistence=float(cfg.get("slow_persistence", 0.96)),
        fast_persistence=float(cfg.get("fast_persistence", 0.22)),
        persistence_curve=float(cfg.get("persistence_curve", 1.8)),
        slow_write_scale=float(cfg.get("slow_write_scale", 0.08)),
        fast_write_scale=float(cfg.get("fast_write_scale", 1.0)),
        write_curve=float(cfg.get("write_curve", 1.4)),
        prefix_coarse_weight=float(cfg.get("prefix_coarse_weight", 0.40)),
        prefix_mid_weight=float(cfg.get("prefix_mid_weight", 0.25)),
        branching_mode=str(cfg.get("branching_mode", "single_path")),
        num_modes=int(cfg.get("num_modes", 2)),
        readout_mode=str(cfg.get("readout_mode", "weighted_mixture")),
        branch_law_version=str(cfg.get("branch_law_version", "parametric_v1")),
        q_trace_rank=int(cfg.get("q_trace_rank", 4)),
        dormant_logit=float(cfg.get("dormant_logit", -6.0)),
        dormant_support=float(cfg.get("dormant_support", 0.02)),
        dormant_q_scale=float(cfg.get("dormant_q_scale", 0.02)),
        split_pressure=float(cfg.get("split_pressure", 0.24)),
        split_seed_scale=float(cfg.get("split_seed_scale", 0.10)),
        split_support_gain=float(cfg.get("split_support_gain", 0.45)),
        merge_pressure=float(cfg.get("merge_pressure", 0.16)),
        merge_phase_tol=float(cfg.get("merge_phase_tol", 0.82)),
        merge_support_overlap_weight=float(cfg.get("merge_support_overlap_weight", 0.50)),
        survival_coherence_weight=float(cfg.get("survival_coherence_weight", 0.40)),
        survival_arc_weight=float(cfg.get("survival_arc_weight", 0.55)),
        survival_qtrace_weight=float(cfg.get("survival_qtrace_weight", 0.25)),
        survival_residue_penalty=float(cfg.get("survival_residue_penalty", 0.30)),
        collapse_sharpness=float(cfg.get("collapse_sharpness", 1.25)),
        support_decay=float(cfg.get("support_decay", 0.10)),
        support_spread=int(cfg.get("support_spread", 5)),
        support_overlap_penalty=float(cfg.get("support_overlap_penalty", 0.15)),
        anti_fixation_weight=float(cfg.get("anti_fixation_weight", 0.20)),
        readout_temperature=float(cfg.get("readout_temperature", 0.85)),
        slot2_support_threshold=float(cfg.get("slot2_support_threshold", 0.10)),
        real_branch_threshold=float(cfg.get("real_branch_threshold", 0.12)),
        child_branch_parent_threshold=float(cfg.get("child_branch_parent_threshold", cfg.get("real_branch_threshold", 0.12))),
        child_branch_writeback_threshold=float(cfg.get("child_branch_writeback_threshold", 0.0)),
        child_branch_meso_threshold=float(cfg.get("child_branch_meso_threshold", 0.0)),
        child_branch_live_threshold=float(cfg.get("child_branch_live_threshold", 0.0)),
        mode_perturb_window_frac=float(cfg.get("mode_perturb_window_frac", 0.18)),
        qtrace_momentum=float(cfg.get("qtrace_momentum", 0.85)),
        mask_neighborhood=int(cfg.get("mask_neighborhood", 3)),
        topology_mask_gain=float(cfg.get("topology_mask_gain", 0.55)),
        complexity_mask_gain=float(cfg.get("complexity_mask_gain", 0.60)),
        context_mask_gain=float(cfg.get("context_mask_gain", 0.45)),
        contrastive_mask_gain=float(cfg.get("contrastive_mask_gain", 0.70)),
        aux_mask_suppression=float(cfg.get("aux_mask_suppression", 0.35)),
        instability_mask_gain=float(cfg.get("instability_mask_gain", 0.75)),
        defect_phase_gain=float(cfg.get("defect_phase_gain", 0.40)),
        defect_q_gain=float(cfg.get("defect_q_gain", 0.30)),
        defect_residue_gain=float(cfg.get("defect_residue_gain", 0.15)),
        defect_sharpness_gain=float(cfg.get("defect_sharpness_gain", 0.15)),
        defect_world_grad_gain=float(cfg.get("defect_world_grad_gain", 0.20)),
        instability_seed_scale=float(cfg.get("instability_seed_scale", 0.18)),
        instability_support_gain=float(cfg.get("instability_support_gain", 0.28)),
        instability_logit_gain=float(cfg.get("instability_logit_gain", 0.22)),
        branch_kernel_version=str(cfg.get("branch_kernel_version", "ramanujan")),
        relation_attention_gain=float(cfg.get("relation_attention_gain", 0.65)),
        relation_attention_sharpness=float(cfg.get("relation_attention_sharpness", 1.10)),
        relation_value_gain=float(cfg.get("relation_value_gain", 0.30)),
        relation_support_gain=float(cfg.get("relation_support_gain", 0.22)),
        relation_logit_gain=float(cfg.get("relation_logit_gain", 0.24)),
        relation_qtrace_gain=float(cfg.get("relation_qtrace_gain", 0.14)),
        relation_residual_mix=float(cfg.get("relation_residual_mix", 0.60)),
        child_spawn_threshold=float(cfg.get("child_spawn_threshold", 0.34)),
        child_max_worlds=int(cfg.get("child_max_worlds", 4)),
        child_min_age_for_writeback=int(cfg.get("child_min_age_for_writeback", 2)),
        child_support_window=int(cfg.get("child_support_window", 12)),
        child_support_decay=float(cfg.get("child_support_decay", 0.12)),
        child_survival_coherence_weight=float(cfg.get("child_survival_coherence_weight", 0.42)),
        child_survival_qtrace_weight=float(cfg.get("child_survival_qtrace_weight", 0.24)),
        child_survival_residue_penalty=float(cfg.get("child_survival_residue_penalty", 0.22)),
        child_writeback_gain=float(cfg.get("child_writeback_gain", 0.32)),
        child_writeback_rank=int(cfg.get("child_writeback_rank", 12)),
        child_writeback_temperature=float(cfg.get("child_writeback_temperature", 0.85)),
        child_writeback_budget=float(cfg.get("child_writeback_budget", 1.25)),
        child_parent_mix=float(cfg.get("child_parent_mix", 0.15)),
        child_parent_mix_early=float(cfg.get("child_parent_mix_early", 0.08)),
        child_operator_seed_gain=float(cfg.get("child_operator_seed_gain", 2.40)),
        child_operator_promotability_gain=float(cfg.get("child_operator_promotability_gain", 0.60)),
        child_writeback_phase_delta_gain=float(cfg.get("child_writeback_phase_delta_gain", 1.00)),
        child_writeback_operator_mix=float(cfg.get("child_writeback_operator_mix", 0.25)),
        child_writeback_phase_floor_target=float(cfg.get("child_writeback_phase_floor_target", 0.30)),
        child_writeback_phase_floor_threshold_mult=float(cfg.get("child_writeback_phase_floor_threshold_mult", 2.50)),
        child_writeback_phase_floor_gain=float(cfg.get("child_writeback_phase_floor_gain", 2.10)),
        child_writeback_parent_mix_cap=float(cfg.get("child_writeback_parent_mix_cap", 0.04)),
        child_support_writeback_gain=float(cfg.get("child_support_writeback_gain", 1.00)),
        child_support_writeback_floor_gain=float(cfg.get("child_support_writeback_floor_gain", 0.35)),
        child_kill_threshold=float(cfg.get("child_kill_threshold", 0.08)),
        child_local_ifs_enabled=bool(cfg.get("child_local_ifs_enabled", False)),
        child_local_steps=int(cfg.get("child_local_steps", 1)),
        child_local_support_only=bool(cfg.get("child_local_support_only", True)),
        child_local_coherence_retention_enabled=bool(cfg.get("child_local_coherence_retention_enabled", False)),
        child_local_coherence_retention_mix=float(cfg.get("child_local_coherence_retention_mix", 0.0)),
        child_local_coherence_floor=float(cfg.get("child_local_coherence_floor", 0.0)),
        child_local_coherence_floor_support=float(cfg.get("child_local_coherence_floor_support", 0.18)),
        child_local_coherence_causal_gate_enabled=bool(
            cfg.get("child_local_coherence_causal_gate_enabled", False)
        ),
        child_local_coherence_causal_min_delta=float(cfg.get("child_local_coherence_causal_min_delta", 0.02)),
        child_local_coherence_causal_full_delta=float(cfg.get("child_local_coherence_causal_full_delta", 0.18)),
        child_local_coherence_causal_gate_floor=float(cfg.get("child_local_coherence_causal_gate_floor", 0.0)),
        law_packet_merge_threshold=float(cfg.get("law_packet_merge_threshold", 0.92)),
        law_packet_min_score=float(cfg.get("law_packet_min_score", 0.25)),
        law_packet_topk_families=int(cfg.get("law_packet_topk_families", 8)),
    )


def _native_resample(wav: torch.Tensor, orig_sr: int, target_sr: int) -> torch.Tensor:
    if orig_sr == target_sr:
        return wav
    ratio = float(target_sr) / float(orig_sr)
    target_len = int(round(wav.size(1) * ratio))
    return F.interpolate(
        wav.unsqueeze(0),
        size=target_len,
        mode="linear",
        align_corners=False,
    ).squeeze(0)


def _load_local_pcm_wav(path: Path) -> tuple[torch.Tensor, int]:
    with wave.open(str(path), "rb") as wf:
        sr = int(wf.getframerate())
        ch = int(wf.getnchannels())
        sampwidth = int(wf.getsampwidth())
        nframes = int(wf.getnframes())
        raw = wf.readframes(nframes)

    if sampwidth == 1:
        arr = np.frombuffer(raw, dtype=np.uint8).astype(np.float32)
        arr = (arr - 128.0) / 128.0
    elif sampwidth == 2:
        arr = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    elif sampwidth == 3:
        b = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 3)
        v = (
            b[:, 0].astype(np.int32)
            | (b[:, 1].astype(np.int32) << 8)
            | (b[:, 2].astype(np.int32) << 16)
        )
        sign = 1 << 23
        v = (v ^ sign) - sign
        arr = v.astype(np.float32) / 8388608.0
    elif sampwidth == 4:
        arr = np.frombuffer(raw, dtype=np.int32).astype(np.float32) / 2147483648.0
    else:
        raise RuntimeError(f"Unsupported WAV sample width: {sampwidth} bytes")

    arr = arr.reshape(-1, ch).T
    wav = torch.from_numpy(arr)
    return wav, sr


def _save_wav(wav: torch.Tensor, path: Path, sr: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    mono = wav.detach().cpu().float().view(-1).numpy()
    mono = mono / (np.abs(mono).max() + 1e-8)
    pcm = (mono * 32767.0).clip(-32768, 32767).astype(np.int16)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(pcm.tobytes())


def _crop_or_pad(wav: torch.Tensor, target_len: int) -> torch.Tensor:
    if wav.size(1) < target_len:
        reps = int(np.ceil(float(target_len) / float(max(1, wav.size(1)))))
        wav = wav.repeat(1, reps)
        wav = wav[:, :target_len]
    elif wav.size(1) > target_len:
        wav = wav[:, :target_len]
    return wav


def prepare_reference_audio(
    wav_path: Path,
    device_name: str,
    clip_seconds_override: int | None = None,
) -> tuple[torch.Tensor, int]:
    cfg = load_config()
    sr = int(cfg["data"]["sample_rate"])
    clip_seconds = int(clip_seconds_override if clip_seconds_override is not None else cfg["data"].get("clip_seconds", 2))
    target_len = sr * clip_seconds
    device = _safe_device(device_name)

    wav, wav_sr = _load_local_pcm_wav(wav_path)
    wav = wav.float()
    if wav.size(0) > 1:
        wav = wav.mean(dim=0, keepdim=True)
    wav = _native_resample(wav, wav_sr, sr)
    wav = _crop_or_pad(wav, target_len)
    return wav.to(device), sr


def render_reference_audio(
    wav_path: Path,
    out_path: Path,
    device_name: str,
    clip_seconds_override: int | None = None,
) -> dict[str, Any]:
    wav, sr = prepare_reference_audio(
        wav_path=wav_path,
        device_name=device_name,
        clip_seconds_override=clip_seconds_override,
    )
    _save_wav(wav.squeeze(0), out_path, sr=sr)
    return {
        "source_wav": str(wav_path),
        "out_wav": str(out_path),
        "device": str(_safe_device(device_name)),
        "sample_rate": sr,
        "clip_seconds": int(clip_seconds_override if clip_seconds_override is not None else load_config()["data"].get("clip_seconds", 2)),
    }


def _phasor_to_phase(z: torch.Tensor) -> torch.Tensor:
    return torch.atan2(z[..., 1], z[..., 0])


def _blend_phase(original_phase: torch.Tensor, final_z: torch.Tensor, blend: float) -> torch.Tensor:
    blend = float(min(1.0, max(0.0, blend)))
    orig_z = phase_to_phasor(original_phase)
    mixed = phasor_normalize((1.0 - blend) * orig_z + blend * final_z)
    return _phasor_to_phase(mixed)


def render_circleworld_audio(
    wav_path: Path,
    config_path: Path,
    out_path: Path,
    device_name: str,
    phase_blend: float,
    clip_seconds_override: int | None = None,
) -> dict[str, Any]:
    cfg = load_config()
    sr = int(cfg["data"]["sample_rate"])
    clip_seconds = int(clip_seconds_override if clip_seconds_override is not None else cfg["data"].get("clip_seconds", 2))
    stft_cfg = cfg["data"]["stft"]
    circle_cfg = _load_circle_cfg(config_path)
    device = _safe_device(device_name)

    wav, sr = prepare_reference_audio(
        wav_path=wav_path,
        device_name=device_name,
        clip_seconds_override=clip_seconds_override,
    )
    wav = wav.to(device)

    mag, phase = compute_stft(wav, stft_cfg)
    phase_state = phase_to_phasor(phase)
    run_mode = circle_cfg.branching_mode if circle_cfg.branching_mode in {"native_multimode", "native_multimode_childworld"} else "active_packets"
    run = recurse_circleworld(phase_state, cfg=circle_cfg, depth=circle_cfg.recursion_depth, mode=run_mode)
    summary = summarize_circleworld_run(run)
    final_phase = _blend_phase(phase, run["phase_state"], blend=phase_blend)
    out_wav = inverse_stft(mag, final_phase, stft_cfg).squeeze(0)
    _save_wav(out_wav, out_path, sr=sr)

    meta = {
        "source_wav": str(wav_path),
        "out_wav": str(out_path),
        "circleworld_config": str(config_path),
        "device": str(device),
        "sample_rate": sr,
        "clip_seconds": clip_seconds,
        "phase_blend": float(phase_blend),
        **summary,
    }
    return meta


def main() -> None:
    ap = argparse.ArgumentParser(description="Render Circleworld audio from a real local WAV anchor.")
    ap.add_argument("--wav-path", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--phase-blend", type=float, default=0.75)
    ap.add_argument("--clip-seconds", type=int, default=None)
    args = ap.parse_args()

    meta = render_circleworld_audio(
        wav_path=Path(args.wav_path),
        config_path=Path(args.config),
        out_path=Path(args.out),
        device_name=args.device,
        phase_blend=args.phase_blend,
        clip_seconds_override=args.clip_seconds,
    )
    meta_path = Path(args.out).with_suffix(".json")
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
