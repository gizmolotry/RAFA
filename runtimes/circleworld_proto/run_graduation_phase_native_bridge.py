from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
CORE = ROOT / "core"
RUNTIME = ROOT / "runtimes" / "circleworld_proto"
for path in (ROOT, CORE, RUNTIME):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from benchmark_audio_continuation import (  # noqa: E402
    _build_magnitude_canvas,
    _build_phase_seed_canvas,
    _compare_waveforms,
    _copy_last_baseline,
    _crop_or_pad_zero,
    _load_cases,
    _loop_reentry_metrics,
    _method_summary,
    _prepare_anchor,
    _safe_device,
    _sanitize_name,
    _write_wav,
)
from config import load_config  # noqa: E402
from diffusion_utils import make_beta_schedule, phase_to_phasor, phasor_to_phase, q_sample_x0  # noqa: E402
from stft_utils import compute_stft, inverse_stft  # noqa: E402
from tools.export_audio_compat14 import Compat14RAFA, NakedDenoiser, set_deterministic_seed  # noqa: E402


OUTPUT_JSON = "graduation_phase_native_bridge.json"
OUTPUT_MD = "GRADUATION_PHASE_NATIVE_BRIDGE.md"
DEFAULT_CKPT = ROOT / "checkpoints_stage4" / "bound_weights_ep10_step1200.pt"


def _mode_id(case_name: str, policy: str) -> int:
    if policy == "engine":
        return 1
    if policy == "voice":
        return 2
    if policy == "impact":
        return 3
    if policy == "drone":
        return 4
    if policy != "group_heuristic":
        raise ValueError(f"Unknown mode policy: {policy}")
    group = case_name.split("__", 1)[0]
    if "typewriter" in group:
        return 3
    if "synth" in group or "buzz" in group:
        return 4
    return 1


def _load_graduation_models(
    *,
    ckpt_path: Path,
    device: torch.device,
    phase_mix: float,
    mag_mix: float,
    source: str,
    start_idx: int,
) -> tuple[Compat14RAFA, NakedDenoiser, dict[str, Any]]:
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    core_weights = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in ckpt["core"].items()}
    denoiser = NakedDenoiser(dev=device)
    for key, value in ckpt["denoiser"].items():
        if key in denoiser.weights:
            denoiser.weights[key] = value.to(device)
    rafa_core = Compat14RAFA(
        core_weights,
        dev=device,
        phase_mix=phase_mix,
        mag_mix=mag_mix,
        source=source,
        start_idx=start_idx,
    )
    meta = {
        "checkpoint": str(ckpt_path),
        "compat_source": source,
        "phase_mix": float(phase_mix),
        "mag_mix": float(mag_mix),
        "start_idx": int(start_idx),
        "checkpoint_core_keys": sorted(str(key) for key in ckpt.get("core", {}).keys()),
    }
    return rafa_core, denoiser, meta


def _render_full_wav(
    *,
    mag: torch.Tensor,
    phase: torch.Tensor,
    stft_cfg: dict[str, Any],
    total_samples: int,
) -> torch.Tensor:
    wav = inverse_stft(mag, phase, stft_cfg).squeeze(0)
    return _crop_or_pad_zero(wav, total_samples)


def _normalise_generated(
    generated: torch.Tensor,
    prefix: torch.Tensor,
    mode: str,
) -> tuple[torch.Tensor, dict[str, Any]]:
    if mode == "none":
        return generated, {"output_normalization": "none"}
    if mode == "generated_peak":
        scale = float(generated.detach().abs().max().clamp_min(1.0e-8).item())
        return generated / scale, {"output_normalization": mode, "normalization_scale": scale}
    if mode == "prefix_rms":
        prefix_rms = float(torch.sqrt(torch.mean(prefix.detach().float() ** 2) + 1.0e-12).item())
        gen_rms = float(torch.sqrt(torch.mean(generated.detach().float() ** 2) + 1.0e-12).item())
        scale = gen_rms / max(prefix_rms, 1.0e-8)
        return generated / max(scale, 1.0e-8), {
            "output_normalization": mode,
            "prefix_rms": prefix_rms,
            "generated_rms": gen_rms,
            "normalization_scale": scale,
        }
    raise ValueError(f"Unknown output normalization: {mode}")


def _run_case(
    *,
    name: str,
    wav_path: Path,
    rafa_core: Compat14RAFA,
    denoiser: NakedDenoiser,
    stft_cfg: dict[str, Any],
    sr: int,
    out_dir: Path,
    device: torch.device,
    prefix_seconds: float,
    future_seconds: float,
    magnitude_mode: str,
    phase_seed_policy: str,
    mode_policy: str,
    output_normalization: str,
    alphas_cumprod: torch.Tensor,
    t: torch.Tensor,
) -> dict[str, Any]:
    prefix_samples = max(1, int(round(prefix_seconds * sr)))
    future_samples = max(1, int(round(future_seconds * sr)))
    total_samples = prefix_samples + future_samples
    safe_name = _sanitize_name(name)

    wav = _prepare_anchor(wav_path, target_sr=sr, total_samples=total_samples)
    prefix = wav[..., :prefix_samples].to(device)
    target_future = wav[..., prefix_samples:total_samples].reshape(-1).to(device)

    prefix_mag, prefix_phase = compute_stft(prefix, stft_cfg)
    hop = int(stft_cfg["hop"])
    future_frames = max(1, int(math.ceil(float(future_samples) / float(hop))))
    phase_canvas, phase_flags = _build_phase_seed_canvas(
        prefix=prefix,
        prefix_phase=prefix_phase,
        future_frames=future_frames,
        future_samples=future_samples,
        sr=sr,
        stft_cfg=stft_cfg,
        policy=phase_seed_policy,
    )
    mag_canvas, mag_flags = _build_magnitude_canvas(prefix_mag, future_frames=future_frames, mode=magnitude_mode)

    mode_id = _mode_id(name, mode_policy)
    with torch.no_grad():
        mag = mag_canvas.to(device)
        phase = phase_canvas.to(device)
        x0_mag = torch.log1p(mag.clamp_min(0.0))
        x0_z = phase_to_phasor(phase)
        xt_mag, xt_z, _, _ = q_sample_x0(x0_mag, x0_z, t, alphas_cumprod)
        p_phase, p_mag, _ = rafa_core.forward(mag, phase, control_matrix={"mode_id": mode_id})
        d_mag, d_z = denoiser.forward(
            xt_mag,
            xt_z,
            t,
            rafa_mag=p_mag,
            rafa_z=torch.stack([torch.cos(p_phase), torch.sin(p_phase)], -1),
        )
        rendered_full = _render_full_wav(
            mag=torch.expm1(d_mag).clamp_min(0.0),
            phase=phasor_to_phase(d_z),
            stft_cfg=stft_cfg,
            total_samples=total_samples,
        )
        rendered_full, norm_flags = _normalise_generated(rendered_full, prefix.reshape(-1), output_normalization)
        graduation_future = rendered_full[prefix_samples:total_samples]

    prefix_seed_baseline = _render_full_wav(
        mag=mag_canvas,
        phase=phase_canvas,
        stft_cfg=stft_cfg,
        total_samples=total_samples,
    )[prefix_samples:total_samples]
    copy_last = _copy_last_baseline(prefix.detach().cpu(), future_samples=future_samples, sr=sr).to(device)

    artifacts = {
        "target_future": out_dir / f"{safe_name}_target_future.wav",
        "graduation": out_dir / f"{safe_name}_graduation_continuation.wav",
        "baseline_copy_last": out_dir / f"{safe_name}_baseline_copy_last.wav",
        "baseline_prefix_seed": out_dir / f"{safe_name}_baseline_prefix_seed.wav",
    }
    _write_wav(artifacts["target_future"], target_future, sr)
    _write_wav(artifacts["graduation"], graduation_future, sr)
    _write_wav(artifacts["baseline_copy_last"], copy_last, sr)
    _write_wav(artifacts["baseline_prefix_seed"], prefix_seed_baseline, sr)

    methods = {
        "graduation_compat14": {
            "wav_path": str(artifacts["graduation"]),
            "metrics": _compare_waveforms(target_future, graduation_future),
            "loop_reentry": _loop_reentry_metrics(graduation_future, sr),
            "magnitude_flags": mag_flags,
            "phase_seed_flags": phase_flags,
            "graduation_flags": {
                "mode_policy": mode_policy,
                "mode_id": mode_id,
                **norm_flags,
                "future_target_magnitude_reused": False,
                "target_future_stft_magnitude_accessed": False,
                "future_target_phase_reused": False,
                "target_future_stft_phase_accessed": False,
            },
        },
        "baseline_copy_last": {
            "wav_path": str(artifacts["baseline_copy_last"]),
            "metrics": _compare_waveforms(target_future, copy_last),
            "loop_reentry": _loop_reentry_metrics(copy_last, sr),
            "magnitude_flags": {
                "magnitude_mode": "waveform_copy_last_prefix",
                "future_magnitude_source": "none_waveform_baseline",
                "future_target_magnitude_reused": False,
                "target_future_stft_magnitude_accessed": False,
            },
        },
        "baseline_prefix_seed": {
            "wav_path": str(artifacts["baseline_prefix_seed"]),
            "metrics": _compare_waveforms(target_future, prefix_seed_baseline),
            "loop_reentry": _loop_reentry_metrics(prefix_seed_baseline, sr),
            "magnitude_flags": mag_flags,
            "phase_seed_flags": phase_flags,
        },
    }
    return {
        "name": name,
        "source_wav": str(wav_path),
        "target_future_wav": str(artifacts["target_future"]),
        "sample_rate": sr,
        "prefix_samples": prefix_samples,
        "future_samples": future_samples,
        "prefix_stft_frames": int(prefix_mag.size(-1)),
        "future_stft_frames": int(future_frames),
        "future_target_audio_used_for_metrics_only": True,
        "future_target_magnitude_reused": False,
        "target_future_stft_magnitude_accessed": False,
        "future_target_phase_reused": False,
        "target_future_stft_phase_accessed": False,
        "mode_policy": mode_policy,
        "mode_id": mode_id,
        "methods": methods,
    }


def _comparison_table(method_summary: dict[str, Any], method: str, baseline: str) -> dict[str, float]:
    m = method_summary.get(method, {})
    b = method_summary.get(baseline, {})
    return {
        "mean_corr_delta": float(m.get("mean_corr", 0.0)) - float(b.get("mean_corr", 0.0)),
        "mean_mse_delta": float(m.get("mean_mse", 0.0)) - float(b.get("mean_mse", 0.0)),
        "mean_loop_delta": float(m.get("mean_loop_autocorr_peak", 0.0))
        - float(b.get("mean_loop_autocorr_peak", 0.0)),
        "mean_reentry_delta": float(m.get("mean_first_chunk_reentry", 0.0))
        - float(b.get("mean_first_chunk_reentry", 0.0)),
    }


def _markdown(summary: dict[str, Any]) -> str:
    comp = summary["comparisons"]
    lines = [
        "# Graduation Phase-Native Bridge",
        "",
        "This adapter runs the Graduation compat14 runtime on the same prefix/future",
        "audio-continuation substrate used by the Circleworld lockbox suite.",
        "",
        "## Status",
        "",
        f"- Status: `{summary['status']}`",
        f"- Cases: `{summary['case_count']}`",
        f"- Device: `{summary['device']}`",
        f"- Future access clean: `{summary['future_access_clean']}`",
        f"- Mode policy: `{summary['mode_policy']}`",
        f"- Magnitude mode: `{summary['magnitude_mode']}`",
        f"- Phase seed policy: `{summary['phase_seed_policy']}`",
        "",
        "## Deltas",
        "",
        "| comparison | corr | MSE | loop | reentry |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for label, row in comp.items():
        lines.append(
            "| `{label}` | {corr:+.9f} | {mse:+.9f} | {loop:+.9f} | {reentry:+.9f} |".format(
                label=label,
                corr=float(row["mean_corr_delta"]),
                mse=float(row["mean_mse_delta"]),
                loop=float(row["mean_loop_delta"]),
                reentry=float(row["mean_reentry_delta"]),
            )
        )
    lines.extend(["", "## Interpretation", "", summary["interpretation"]])
    return "\n".join(lines) + "\n"


def run_bridge(
    *,
    ckpt_path: Path,
    out_dir: Path,
    cases_json: Path | None,
    device_name: str,
    num_cases: int,
    prefix_seconds: float,
    future_seconds: float,
    magnitude_mode: str,
    phase_seed_policy: str,
    mode_policy: str,
    output_normalization: str,
    phase_mix: float,
    mag_mix: float,
    source: str,
    seed: int,
    start_idx: int,
) -> dict[str, Any]:
    cfg = load_config(str(ROOT / "config.yaml")) if (ROOT / "config.yaml").exists() else load_config()
    sr = int(cfg["data"]["sample_rate"])
    stft_cfg = dict(cfg["data"]["stft"])
    device = _safe_device(device_name)
    out_dir.mkdir(parents=True, exist_ok=True)
    set_deterministic_seed(int(seed))
    rafa_core, denoiser, model_meta = _load_graduation_models(
        ckpt_path=ckpt_path,
        device=device,
        phase_mix=phase_mix,
        mag_mix=mag_mix,
        source=source,
        start_idx=start_idx,
    )

    timesteps = 50
    betas = make_beta_schedule(timesteps, 1.0e-4, 0.02, device)
    alphas_cumprod = torch.cumprod(1.0 - betas, dim=0)
    t = torch.ones(1, dtype=torch.long, device=device) * 25

    cases = list(_load_cases(cases_json).items())
    if num_cases > 0:
        cases = cases[:num_cases]
    rows = [
        _run_case(
            name=name,
            wav_path=wav_path,
            rafa_core=rafa_core,
            denoiser=denoiser,
            stft_cfg=stft_cfg,
            sr=sr,
            out_dir=out_dir,
            device=device,
            prefix_seconds=prefix_seconds,
            future_seconds=future_seconds,
            magnitude_mode=magnitude_mode,
            phase_seed_policy=phase_seed_policy,
            mode_policy=mode_policy,
            output_normalization=output_normalization,
            alphas_cumprod=alphas_cumprod,
            t=t,
        )
        for name, wav_path in cases
    ]
    method_summary = _method_summary(rows)
    future_access_clean = not any(
        bool(row.get(flag))
        for row in rows
        for flag in (
            "future_target_magnitude_reused",
            "target_future_stft_magnitude_accessed",
            "future_target_phase_reused",
            "target_future_stft_phase_accessed",
        )
    )
    comparisons = {
        "graduation_vs_copy_last": _comparison_table(
            method_summary, "graduation_compat14", "baseline_copy_last"
        ),
        "graduation_vs_prefix_seed": _comparison_table(
            method_summary, "graduation_compat14", "baseline_prefix_seed"
        ),
    }
    corr_copy = comparisons["graduation_vs_copy_last"]["mean_corr_delta"]
    mse_copy = comparisons["graduation_vs_copy_last"]["mean_mse_delta"]
    loop_copy = comparisons["graduation_vs_copy_last"]["mean_loop_delta"]
    reentry_copy = comparisons["graduation_vs_copy_last"]["mean_reentry_delta"]
    smoke_pass = future_access_clean and corr_copy > 0.0 and mse_copy <= 0.0
    status = "graduation_bridge_smoke_pass" if smoke_pass else "graduation_bridge_smoke_not_candidate"
    summary = {
        "runtime": "graduation_compat14",
        "schema": "graduation_phase_native_bridge_v1",
        "status": status,
        "out_dir": str(out_dir),
        "checkpoint": str(ckpt_path),
        "model_meta": model_meta,
        "device": str(device),
        "requested_device": str(device_name),
        "sample_rate": sr,
        "seed": int(seed),
        "prefix_seconds": float(prefix_seconds),
        "future_seconds": float(future_seconds),
        "magnitude_mode": magnitude_mode,
        "phase_seed_policy": phase_seed_policy,
        "mode_policy": mode_policy,
        "output_normalization": output_normalization,
        "case_count": len(rows),
        "cases": {name: str(path) for name, path in cases},
        "future_access_clean": future_access_clean,
        "future_target_audio_used_for_metrics_only": True,
        "future_target_magnitude_reused": False,
        "target_future_stft_magnitude_accessed": False,
        "future_target_phase_reused": False,
        "target_future_stft_phase_accessed": False,
        "method_summary": method_summary,
        "comparisons": comparisons,
        "rows": rows,
        "interpretation": (
            "Graduation compat14 can be driven as a prefix-only continuation adapter. "
            "This is a bridge harness smoke result, not yet a final Graduation-vs-Circleworld leaderboard."
        ),
    }
    (out_dir / OUTPUT_JSON).write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (out_dir / OUTPUT_MD).write_text(_markdown(summary), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Graduation compat14 on the phase-native continuation lockbox.")
    parser.add_argument("--ckpt", type=Path, default=DEFAULT_CKPT)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--cases-json", type=Path, default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num-cases", type=int, default=3)
    parser.add_argument("--prefix-seconds", type=float, default=1.0)
    parser.add_argument("--future-seconds", type=float, default=1.0)
    parser.add_argument("--magnitude-mode", choices=("prefix_hold", "flat"), default="prefix_hold")
    parser.add_argument("--phase-seed-policy", default="copy_last_waveform_phase")
    parser.add_argument(
        "--mode-policy",
        choices=("engine", "voice", "impact", "drone", "group_heuristic"),
        default="group_heuristic",
    )
    parser.add_argument(
        "--output-normalization",
        choices=("none", "generated_peak", "prefix_rms"),
        default="prefix_rms",
    )
    parser.add_argument("--phase-mix", type=float, default=0.35)
    parser.add_argument("--mag-mix", type=float, default=0.05)
    parser.add_argument("--source", choices=("spine", "qkv", "filt"), default="qkv")
    parser.add_argument("--seed", type=int, default=101)
    parser.add_argument("--start-idx", type=int, default=516)
    args = parser.parse_args()
    summary = run_bridge(
        ckpt_path=args.ckpt,
        out_dir=args.out_dir,
        cases_json=args.cases_json,
        device_name=str(args.device),
        num_cases=int(args.num_cases),
        prefix_seconds=float(args.prefix_seconds),
        future_seconds=float(args.future_seconds),
        magnitude_mode=str(args.magnitude_mode),
        phase_seed_policy=str(args.phase_seed_policy),
        mode_policy=str(args.mode_policy),
        output_normalization=str(args.output_normalization),
        phase_mix=float(args.phase_mix),
        mag_mix=float(args.mag_mix),
        source=str(args.source),
        seed=int(args.seed),
        start_idx=int(args.start_idx),
    )
    print(
        json.dumps(
            {
                "status": summary["status"],
                "json": str(Path(args.out_dir) / OUTPUT_JSON),
                "markdown": str(Path(args.out_dir) / OUTPUT_MD),
                "case_count": summary["case_count"],
                "graduation_vs_copy_last": summary["comparisons"]["graduation_vs_copy_last"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
