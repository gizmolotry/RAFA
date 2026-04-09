from __future__ import annotations

import argparse
import copy
import json
import os
import wave

import numpy as np
import torch
import yaml

from diffusion_models import BaselineDenoiser, RAFADenoiser
from diffusion_utils import make_beta_schedule, phasor_normalize, phasor_to_phase, spectral_centroid
from stft_utils import compute_stft
from hf_local import resolve_hf_pretrained_path


def _save_wav(path: str, wav: torch.Tensor, sr: int) -> None:
    x = wav.detach().cpu()
    if x.dim() == 1:
        x = x.unsqueeze(0)
    x = x.clamp(-1.0, 1.0).numpy()
    pcm = (x * 32767.0).astype(np.int16)
    with wave.open(path, "wb") as wf:
        wf.setnchannels(int(x.shape[0]))
        wf.setsampwidth(2)
        wf.setframerate(int(sr))
        wf.writeframes(pcm.T.reshape(-1).tobytes())


def _stats(wav: torch.Tensor, mag: torch.Tensor, cfg: dict) -> dict[str, float]:
    rms = float(torch.sqrt(torch.mean(wav**2) + 1e-12).item())
    peak = float(torch.max(torch.abs(wav)).item())
    cent_t = spectral_centroid(mag.unsqueeze(0), int(cfg["data"]["sample_rate"]), int(cfg["data"]["stft"]["n_fft"]))[0]
    return {"rms": rms, "peak": peak, "centroid_hz": float(cent_t.mean().item())}


def _gru_module(model: torch.nn.Module) -> object | None:
    inner = getattr(model, "module", model)
    rafa = getattr(inner, "rafa", None)
    return getattr(rafa, "phase_gru", None)


def _set_gru_enabled(model: torch.nn.Module, enabled: bool) -> bool | None:
    gru = _gru_module(model)
    if gru is None or not hasattr(gru, "cfg"):
        return None
    prev = bool(getattr(gru.cfg, "enabled", True))
    gru.cfg.enabled = bool(enabled)
    return prev


def _state_effect(on_states: list[torch.Tensor], off_states: list[torch.Tensor]) -> float:
    vals: list[float] = []
    for on_state, off_state in zip(on_states, off_states):
        if on_state.shape != off_state.shape:
            continue
        vals.append(float((1.0 - (on_state * off_state).sum(dim=-1).mean()).item()))
    return float(sum(vals) / max(1, len(vals))) if vals else 0.0


@torch.no_grad()
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
    gru_enabled: bool | None,
) -> dict[str, object]:
    dev = xt_mag_init.device
    prev_enabled = None
    if gru_enabled is not None:
        prev_enabled = _set_gru_enabled(model, gru_enabled)
    try:
        xt_mag = xt_mag_init.clone()
        xt_z = xt_z_init.clone()
        phase_states: list[torch.Tensor] = []
        for k, ti in enumerate(idxs):
            t = torch.full((1,), int(ti), device=dev, dtype=torch.long)
            if use_cfg:
                mag_u, z_u, _, _ = model(xt_mag, xt_z, t, tokens=uncond_tokens)
                mag_c, z_c, p_ext_c, _ = model(xt_mag, xt_z, t, tokens=cond_tokens)
                pred_x0_mag = mag_u + guidance * (mag_c - mag_u)
                pred_x0_z = phasor_normalize(z_u + guidance * (z_c - z_u))
                phase_state = p_ext_c.get("phase_state") if isinstance(p_ext_c, dict) else None
            else:
                pred_x0_mag, pred_x0_z, p_ext_c, _ = model(xt_mag, xt_z, t, tokens=cond_tokens)
                phase_state = p_ext_c.get("phase_state") if isinstance(p_ext_c, dict) else None
            if torch.is_tensor(phase_state):
                phase_states.append(phase_state.detach().clone())
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
        top_ratio = float((mag.max(dim=1).values.mean() / mag.mean().clamp_min(1e-8)).item())
        cent_t = spectral_centroid(mag, int(cfg["data"]["sample_rate"]), int(cfg["data"]["stft"]["n_fft"]))
        cent_var = float(cent_t.var(unbiased=False).item())
        dphi = phasor_to_phase(xt_z)
        dphi_std = float((dphi[:, :, 1:] - dphi[:, :, :-1]).std().item()) if dphi.size(-1) > 1 else 0.0
        return {
            "wav": wav.detach().clone(),
            "mag": mag.detach().clone(),
            "phase_states": phase_states,
            "top_ratio": top_ratio,
            "cent_var": cent_var,
            "dphi_std": dphi_std,
        }
    finally:
        if prev_enabled is not None:
            _set_gru_enabled(model, prev_enabled)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", default="sample_diffusion.wav")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--prompt", default="cinematic rhythmic texture")
    ap.add_argument("--guidance", type=float, default=None)
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--allow_collapse", action="store_true")
    args = ap.parse_args()

    ckpt = torch.load(args.ckpt, map_location="cpu")
    cfg = ckpt["config"]
    model_state = ckpt["model"]
    has_ckpt_clap = any(k.startswith("text_encoder.") or k.startswith("film.") for k in model_state.keys())
    if bool(cfg.get("diffusion", {}).get("use_clap_cond", cfg.get("diffusion", {}).get("use_clip_cond", False))) and not has_ckpt_clap:
        cfg = copy.deepcopy(cfg)
        cfg.setdefault("diffusion", {})
        cfg["diffusion"]["use_clap_cond"] = False
        cfg["diffusion"]["use_cfg"] = False
        print("WARNING: checkpoint config requested CLAP conditioning but checkpoint weights do not include CLAP modules; sampling unconditioned.")
    dcfg = cfg["diffusion"]
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed = int(args.seed if args.seed is not None else dcfg.get("seed", 1337))
    torch.manual_seed(seed)
    if dev.type == "cuda":
        torch.cuda.manual_seed_all(seed)

    # Calculate exact t_frames based on compute_stft implementation
    dummy_wav = torch.zeros(1, int(cfg["data"]["sample_rate"] * cfg["data"]["clip_seconds"]))
    dummy_mag, _ = compute_stft(dummy_wav, cfg["data"]["stft"])
    t_frames = dummy_mag.size(-1)
    freq_bins = dummy_mag.size(1)
    print(f"DEBUG: Calculated t_frames={t_frames}, freq_bins={freq_bins}")
    model_type = ckpt["diffusion"].get("model_type", "rafa")
    if model_type == "baseline":
        model = BaselineDenoiser(freq_bins).to(dev)
    else:
        model = RAFADenoiser(cfg).to(dev)
    model.load_state_dict(model_state, strict=False)
    model.eval()
    use_clap_cond = bool(dcfg.get("use_clap_cond", dcfg.get("use_clip_cond", False)))
    use_cfg = bool(dcfg.get("use_cfg", False)) and use_clap_cond
    guidance = float(args.guidance if args.guidance is not None else dcfg.get("guidance_scale", 3.0))
    tokenizer = None
    if use_clap_cond:
        try:
            text_encoder_id = resolve_hf_pretrained_path(
                cfg["model"]["text_encoder"],
                need_tokenizer=True,
            )
            tokenizer = AutoTokenizer.from_pretrained(
                text_encoder_id,
                local_files_only=os.path.isdir(text_encoder_id),
            )
        except Exception as e:
            tokenizer = None
            use_cfg = False
            print(f"WARNING: CLAP tokenizer load failed; sampling unconditioned. reason={e}")
    cond_tokens = None
    uncond_tokens = None
    if tokenizer is not None:
        cond_tokens = tokenizer([args.prompt], return_tensors="pt", padding=True, truncation=True)
        uncond_tokens = tokenizer([""], return_tensors="pt", padding=True, truncation=True)
        cond_tokens = {k: v.to(dev) for k, v in cond_tokens.items()}
        uncond_tokens = {k: v.to(dev) for k, v in uncond_tokens.items()}

    train_timesteps = int(ckpt["diffusion"]["timesteps"])
    timesteps = int(args.steps if args.steps is not None else train_timesteps)
    timesteps = max(1, min(timesteps, train_timesteps))
    betas = ckpt["diffusion"]["betas"].to(dev)
    alphas = 1.0 - betas
    alphas_cumprod = ckpt["diffusion"]["alphas_cumprod"].to(dev)

    xt_mag = torch.randn(1, freq_bins, t_frames, device=dev)
    theta = 2.0 * torch.pi * torch.rand(1, freq_bins, t_frames, device=dev) - torch.pi
    xt_z = phasor_normalize(torch.stack([torch.cos(theta), torch.sin(theta)], dim=-1))

    idxs = torch.linspace(train_timesteps - 1, 0, timesteps, device=dev).round().long().tolist()
    result_on = _run_sampling_rollout(
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
        gru_enabled=True,
    )
    result_off = _run_sampling_rollout(
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
        gru_enabled=False,
    ) if _gru_module(model) is not None else None

    wav = result_on["wav"]
    mag = result_on["mag"]
    assert isinstance(wav, torch.Tensor)
    assert isinstance(mag, torch.Tensor)
    out = args.out
    _save_wav(out, wav.unsqueeze(0), int(cfg["data"]["sample_rate"]))
    stats = _stats(wav, mag.squeeze(0), cfg)
    top_ratio = float(result_on["top_ratio"])
    cent_var = float(result_on["cent_var"])
    dphi_std = float(result_on["dphi_std"])
    audio_effect = 0.0
    state_effect = 0.0
    coupling_score = 0.0
    if result_off is not None:
        wav_off = result_off["wav"]
        assert isinstance(wav_off, torch.Tensor)
        audio_effect = float((torch.norm(wav - wav_off) / torch.norm(wav).clamp_min(1e-8)).item())
        state_effect = _state_effect(
            result_on["phase_states"],
            result_off["phase_states"],
        )
        coupling_score = float(audio_effect * state_effect)
    metrics = {
        "saved": out,
        "seed": seed,
        "prompt": args.prompt,
        "steps": timesteps,
        "rms": stats["rms"],
        "peak": stats["peak"],
        "centroid_hz": stats["centroid_hz"],
        "top_ratio": top_ratio,
        "cent_var": cent_var,
        "dphi_std": dphi_std,
        "audio_effect": audio_effect,
        "state_effect": state_effect,
        "coupling_score": coupling_score,
    }
    with open(out + ".json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(
        f"saved={out} seed={seed} rms={stats['rms']:.6f} peak={stats['peak']:.6f} "
        f"centroid_hz={stats['centroid_hz']:.2f} top_ratio={top_ratio:.3f} cent_var={cent_var:.3f} "
        f"dphi_std={dphi_std:.3f} audio_effect={audio_effect:.4f} state_effect={state_effect:.4f} "
        f"coupling_score={coupling_score:.4f}"
    )
    if top_ratio > 30.0:
        print("WARNING: spectral concentration indicates possible beep collapse.")
    if cent_var < 1.0:
        print("WARNING: spectral centroid variance is very low; output may be static.")
    if (not args.allow_collapse) and stats["rms"] < 5e-4:
        raise RuntimeError("Generated audio is near-silent (beep/silence collapse).")


if __name__ == "__main__":
    main()
