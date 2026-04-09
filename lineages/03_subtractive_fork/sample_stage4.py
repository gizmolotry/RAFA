import argparse
import json
import os
import wave
import math
import torch
import numpy as np
import yaml
from pathlib import Path

from lib_blackwell import NakedRAFA, NakedDenoiser
from diffusion_utils import make_beta_schedule, phasor_normalize, phasor_to_phase, spectral_centroid

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

@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--out", default="sample_stage4.wav")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--allow_collapse", action="store_true")
    args = parser.parse_args()

    cfg_path = "config.yaml"
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)

    dev = torch.device("cuda")
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    # 1. Initialize Naked Models
    rafa_core = NakedRAFA(d_model=768, dev=dev)
    denoiser = NakedDenoiser(dev=dev)

    # 2. Load Weights
    ckpt = torch.load(args.ckpt, map_location=dev)
    for k, v in ckpt['core'].items():
        if k in rafa_core.weights:
            rafa_core.weights[k].copy_(v)
    for k, v in ckpt['denoiser'].items():
        if k in denoiser.weights:
            denoiser.weights[k].copy_(v)
    
    # 3. Setup Diffusion
    timesteps = 1000 # Default for Stage 4
    betas = make_beta_schedule(timesteps, 1e-4, 0.02, dev)
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)

    freq_bins = cfg["data"]["stft"]["n_fft"] // 2 + 1
    t_frames = int((cfg["data"]["sample_rate"] * cfg["data"]["clip_seconds"]) / cfg["data"]["stft"]["hop"]) + 1
    
    # 4. Generate Initial Noise
    xt_mag = torch.randn(1, freq_bins, t_frames, device=dev)
    theta = 2.0 * torch.pi * torch.rand(1, freq_bins, t_frames, device=dev) - torch.pi
    xt_z = phasor_normalize(torch.stack([torch.cos(theta), torch.sin(theta)], dim=-1))

    # 5. Denoising Loop (Unconditioned Generation for Stats)
    idxs = torch.linspace(timesteps - 1, 0, args.steps, device=dev).round().long().tolist()
    
    for k, ti in enumerate(idxs):
        t = torch.full((1,), int(ti), device=dev, dtype=torch.long)
        
        # Unconditioned sampling (None guidance)
        pred_x0_mag, pred_x0_z = denoiser.forward(xt_mag, xt_z, t, rafa_mag=None, rafa_z=None)
        
        if k + 1 >= len(idxs):
            xt_mag, xt_z = pred_x0_mag, pred_x0_z
            continue
            
        prev = int(idxs[k+1])
        ab_t = alphas_cumprod[int(ti)]
        ab_prev = alphas_cumprod[prev]
        
        sqrt_ab_t = ab_t.sqrt()
        sqrt_omab_t = (1.0 - ab_t).sqrt().clamp_min(1e-8)
        sqrt_ab_prev = ab_prev.sqrt()
        sqrt_omab_prev = (1.0 - ab_prev).sqrt()
        
        eps_mag = (xt_mag - sqrt_ab_t * pred_x0_mag) / sqrt_omab_t
        xt_mag = sqrt_ab_prev * pred_x0_mag + sqrt_omab_prev * eps_mag
        
        eps_z = (xt_z - sqrt_ab_t.unsqueeze(-1) * pred_x0_z) / sqrt_omab_t.unsqueeze(-1)
        xt_z = phasor_normalize(sqrt_ab_prev.unsqueeze(-1) * pred_x0_z + sqrt_omab_prev.unsqueeze(-1) * eps_z)

    # 6. Final Reconstruct
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

    # 7. Calculate Stats
    rms = float(torch.sqrt(torch.mean(wav**2) + 1e-12).item())
    peak = float(torch.max(torch.abs(wav)).item())
    
    top_ratio = float((mag.max(dim=1).values.mean() / mag.mean().clamp_min(1e-8)).item())
    cent_t = spectral_centroid(mag, int(cfg["data"]["sample_rate"]), int(cfg["data"]["stft"]["n_fft"]))
    cent_var = float(cent_t.var(unbiased=False).item())
    dphi = phasor_to_phase(xt_z)
    dphi_std = float((dphi[:, :, 1:] - dphi[:, :, :-1]).std().item()) if dphi.size(-1) > 1 else 0.0
    centroid_hz = float(cent_t.mean().item())

    # 8. Output results for path_b_eval.py
    print(
        f"saved={args.out} seed={args.seed} rms={rms:.6f} peak={peak:.6f} "
        f"centroid_hz={centroid_hz:.2f} top_ratio={top_ratio:.3f} cent_var={cent_var:.3f} "
        f"dphi_std={dphi_std:.3f} audio_effect=0.0 state_effect=0.0 coupling_score=0.0"
    )

    _save_wav(args.out, wav.unsqueeze(0), int(cfg["data"]["sample_rate"]))

if __name__ == "__main__":
    main()
