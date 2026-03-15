from __future__ import annotations

import argparse
import copy
import csv
import json
import os
from pathlib import Path
import sys

import torch
from transformers import AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from diffusion_models import BaselineDenoiser, RAFADenoiser
from diffusion_utils import make_beta_schedule, phasor_normalize, phasor_to_phase
from hf_local import resolve_hf_pretrained_path
from sample_diffusion import _save_wav


def _pearson_corr(x: torch.Tensor, y: torch.Tensor) -> float:
    x = x.float()
    y = y.float()
    x = x - x.mean()
    y = y - y.mean()
    denom = torch.sqrt((x.square().sum() * y.square().sum()).clamp_min(1e-12))
    return float((x * y).sum().div(denom).item())


def _triangle_envelope(t_frames: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    if t_frames <= 1:
        return torch.zeros((t_frames,), device=device, dtype=dtype)
    pos = torch.linspace(0.0, 1.0, t_frames, device=device, dtype=dtype)
    return 1.0 - (2.0 * pos - 1.0).abs()


def _run_sampling_rollout(
    *,
    model: torch.nn.Module,
    cfg: dict,
    xt_mag_init: torch.Tensor,
    xt_z_init: torch.Tensor,
    idxs: list[int],
    alphas_cumprod: torch.Tensor,
    cond_tokens: dict[str, torch.Tensor] | None,
    uncond_tokens: dict[str, torch.Tensor] | None,
    use_cfg: bool,
    guidance: float,
) -> dict[str, torch.Tensor]:
    dev = xt_mag_init.device
    xt_mag = xt_mag_init.clone()
    xt_z = xt_z_init.clone()
    semantic_tension = None
    for k, ti in enumerate(idxs):
        t = torch.full((1,), int(ti), device=dev, dtype=torch.long)
        if use_cfg:
            mag_u, z_u, _, _ = model(xt_mag, xt_z, t, tokens=uncond_tokens)
            mag_c, z_c, p_ext_c, _ = model(xt_mag, xt_z, t, tokens=cond_tokens)
            pred_x0_mag = mag_u + guidance * (mag_c - mag_u)
            pred_x0_z = phasor_normalize(z_u + guidance * (z_c - z_u))
            p_ext = p_ext_c
        else:
            pred_x0_mag, pred_x0_z, p_ext, _ = model(xt_mag, xt_z, t, tokens=cond_tokens)
        if semantic_tension is None and isinstance(p_ext, dict):
            sc = p_ext.get("semantic_controls")
            if isinstance(sc, dict) and torch.is_tensor(sc.get("semantic_tension")):
                semantic_tension = sc["semantic_tension"][0, :, 0].detach().clone()
        if k + 1 >= len(idxs):
            xt_mag, xt_z = pred_x0_mag, pred_x0_z
            continue
        prev = int(idxs[k + 1])
        ab_t = alphas_cumprod[int(ti)]
        ab_prev = alphas_cumprod[prev]
        sqrt_ab_t = ab_t.sqrt()
        sqrt_omab_t = (1.0 - ab_t).sqrt().clamp_min(1e-8)
        sqrt_ab_prev = ab_prev.sqrt()
        sqrt_omab_prev = (1.0 - ab_prev).sqrt()

        eps_mag = (xt_mag - sqrt_ab_t * pred_x0_mag) / sqrt_omab_t
        xt_mag = sqrt_ab_prev * pred_x0_mag + sqrt_omab_prev * eps_mag

        eps_z = (xt_z - sqrt_ab_t * pred_x0_z) / sqrt_omab_t
        xt_z = phasor_normalize(sqrt_ab_prev * pred_x0_z + sqrt_omab_prev * eps_z)

    mag = torch.expm1(xt_mag).clamp_min(0.0)
    phase = phasor_to_phase(xt_z)
    z = mag * torch.exp(1j * phase)
    wav = torch.istft(
        z.squeeze(0),
        n_fft=int(cfg["data"]["stft"]["n_fft"]),
        hop_length=int(cfg["data"]["stft"]["hop"]),
        win_length=int(cfg["data"]["stft"]["win_length"]),
        window=torch.hann_window(int(cfg["data"]["stft"]["win_length"]), device=dev),
        length=int(cfg["data"]["sample_rate"] * cfg["data"]["clip_seconds"]),
    )
    if semantic_tension is None:
        semantic_tension = _triangle_envelope(mag.size(-1), device=dev, dtype=mag.dtype)
    return {
        "wav": wav.detach().clone(),
        "mag": mag.squeeze(0).detach().clone(),
        "phase": phase.squeeze(0).detach().clone(),
        "semantic_tension": semantic_tension.detach().clone(),
    }


def _q_energy_trajectory(
    mag: torch.Tensor,
    phase: torch.Tensor,
    *,
    qset: list[int],
    harmonic_qs: set[int],
    inharmonic_qs: set[int],
) -> tuple[torch.Tensor, torch.Tensor, dict[int, torch.Tensor]]:
    # mag/phase: [F, T]
    mag_t = mag.transpose(0, 1)  # [T, F]
    phase_t = phase.transpose(0, 1)  # [T, F]
    z = torch.stack([torch.cos(phase_t), torch.sin(phase_t)], dim=-1)  # [T,F,2]
    zi = z.unsqueeze(2)
    zj = z.unsqueeze(1)
    r_re = zi[..., 0] * zj[..., 0] + zi[..., 1] * zj[..., 1]
    r_im = zi[..., 1] * zj[..., 0] - zi[..., 0] * zj[..., 1]
    theta = torch.atan2(r_im, r_re)
    w = (mag_t.unsqueeze(2) * mag_t.unsqueeze(1)).clamp_min(1e-8)
    per_q: dict[int, torch.Tensor] = {}
    for q in qset:
        compat = 0.5 * (1.0 + torch.cos(float(q) * theta))
        per_q[int(q)] = (compat * w).sum(dim=(1, 2)) / w.sum(dim=(1, 2)).clamp_min(1e-8)
    harm = torch.stack([v for q, v in per_q.items() if q in harmonic_qs], dim=0).sum(dim=0)
    inharm = torch.stack([v for q, v in per_q.items() if q in inharmonic_qs], dim=0).sum(dim=0)
    return harm, inharm, per_q


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--seed", type=int, default=101)
    ap.add_argument("--steps", type=int, default=32)
    args = ap.parse_args()

    ckpt = torch.load(args.ckpt, map_location="cpu")
    cfg = copy.deepcopy(ckpt["config"])
    dcfg = cfg.setdefault("diffusion", {})
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(int(args.seed))
    if dev.type == "cuda":
        torch.cuda.manual_seed_all(int(args.seed))

    freq_bins = cfg["data"]["stft"]["n_fft"] // 2 + 1
    t_frames = int((cfg["data"]["sample_rate"] * cfg["data"]["clip_seconds"]) / cfg["data"]["stft"]["hop"]) + 1
    model_type = ckpt["diffusion"].get("model_type", "rafa")
    if model_type == "baseline":
        model = BaselineDenoiser(freq_bins).to(dev)
    else:
        model = RAFADenoiser(cfg).to(dev)
    model.load_state_dict(ckpt["model"], strict=False)
    model.eval()

    use_clap_cond = bool(dcfg.get("use_clap_cond", dcfg.get("use_clip_cond", False)))
    use_cfg = bool(dcfg.get("use_cfg", False)) and use_clap_cond
    guidance = float(dcfg.get("guidance_scale", 3.0))
    tokenizer = None
    cond_tokens = None
    uncond_tokens = None
    if use_clap_cond:
        try:
            text_encoder_id = resolve_hf_pretrained_path(cfg["model"]["text_encoder"], need_tokenizer=True)
            tokenizer = AutoTokenizer.from_pretrained(text_encoder_id, local_files_only=os.path.isdir(text_encoder_id))
        except Exception:
            tokenizer = None
            use_cfg = False
    if tokenizer is not None:
        cond_tokens = tokenizer([""], return_tensors="pt", padding=True, truncation=True)
        uncond_tokens = tokenizer([""], return_tensors="pt", padding=True, truncation=True)
        cond_tokens = {k: v.to(dev) for k, v in cond_tokens.items()}
        uncond_tokens = {k: v.to(dev) for k, v in uncond_tokens.items()}

    train_timesteps = int(ckpt["diffusion"]["timesteps"])
    timesteps = max(1, min(int(args.steps), train_timesteps))
    alphas_cumprod = ckpt["diffusion"]["alphas_cumprod"].to(dev)
    idxs = torch.linspace(train_timesteps - 1, 0, timesteps, device=dev).round().long().tolist()

    xt_mag = torch.randn(1, freq_bins, t_frames, device=dev)
    theta = 2.0 * torch.pi * torch.rand(1, freq_bins, t_frames, device=dev) - torch.pi
    xt_z = phasor_normalize(torch.stack([torch.cos(theta), torch.sin(theta)], dim=-1))

    result = _run_sampling_rollout(
        model=model,
        cfg=cfg,
        xt_mag_init=xt_mag,
        xt_z_init=xt_z,
        idxs=idxs,
        alphas_cumprod=alphas_cumprod,
        cond_tokens=cond_tokens,
        uncond_tokens=uncond_tokens,
        use_cfg=use_cfg,
        guidance=guidance,
    )

    mag = result["mag"]
    phase = result["phase"]
    tension = result["semantic_tension"]
    qset = [int(q) for q in cfg.get("phase_native_ifs", {}).get("qset", [2, 3, 4, 5, 6, 8, 12])]
    harmonic_qs = {2, 3, 4, 6, 8, 12}
    inharmonic_qs = {q for q in qset if q not in harmonic_qs}
    if not inharmonic_qs:
        inharmonic_qs = {5, 7, 11, 13}.intersection(set(qset))
    harm, inharm, per_q = _q_energy_trajectory(
        mag,
        phase,
        qset=qset,
        harmonic_qs=harmonic_qs,
        inharmonic_qs=inharmonic_qs,
    )
    ratio = inharm / (harm + inharm).clamp_min(1e-8)
    tension = tension[: ratio.numel()]
    r_tension = _pearson_corr(tension, ratio)
    peak_idx = int(torch.argmax(ratio).item())
    tension_peak_idx = int(torch.argmax(tension).item())
    peak_alignment_error = float(abs(peak_idx - tension_peak_idx) / max(1, ratio.numel() - 1))
    recovery_gap = float(abs(ratio[0].item() - ratio[-1].item()))

    repo_root = Path(__file__).resolve().parents[1]
    eval_dir = repo_root / "eval" / args.name
    eval_dir.mkdir(parents=True, exist_ok=True)
    wav_path = eval_dir / f"semantic_tension_seed{int(args.seed)}_s{timesteps}.wav"
    _save_wav(str(wav_path), result["wav"].unsqueeze(0), int(cfg["data"]["sample_rate"]))

    csv_path = eval_dir / "semantic_tension_trajectory.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        q_cols = [f"q_{q}" for q in qset]
        writer.writerow(["frame", "tension", "harmonic_energy", "inharmonic_energy", "inharmonic_ratio", *q_cols])
        for i in range(ratio.numel()):
            writer.writerow(
                [
                    i,
                    float(tension[i].item()),
                    float(harm[i].item()),
                    float(inharm[i].item()),
                    float(ratio[i].item()),
                    *[float(per_q[q][i].item()) for q in qset],
                ]
            )

    summary = {
        "name": args.name,
        "checkpoint": args.ckpt,
        "seed": int(args.seed),
        "steps": timesteps,
        "qset": qset,
        "harmonic_qs": sorted(harmonic_qs.intersection(set(qset))),
        "inharmonic_qs": sorted(inharmonic_qs),
        "tension_to_inharmonic_corr": r_tension,
        "peak_inharmonic_ratio": float(ratio.max().item()),
        "mean_inharmonic_ratio": float(ratio.mean().item()),
        "peak_alignment_error": peak_alignment_error,
        "recovery_gap": recovery_gap,
        "trajectory_csv": str(csv_path),
        "wav_path": str(wav_path),
    }
    summary_path = eval_dir / "semantic_tension_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
