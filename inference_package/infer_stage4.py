import argparse
import os
import wave
import math
import torch
import numpy as np
import yaml
from pathlib import Path

# RAFA-PC Stage 4 Inference Driver (Sm_120 Safe)
# No transformers or nn.Module dependencies. Pure functional tensors.

from lib_blackwell import NakedRAFA, NakedDenoiser
from diffusion_utils import make_beta_schedule, phasor_normalize, phasor_to_phase

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
def infer(ckpt_path, out_path, mode_id=0, spectrum_id=1, seed=1337, steps=50):
    cfg_path = "config.yaml"
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    if dev.type == "cuda":
        torch.cuda.manual_seed_all(seed)

    # 1. Initialize Naked Models
    rafa_core = NakedRAFA(d_model=768, dev=dev)
    denoiser = NakedDenoiser(dev=dev)

    # 2. Load Weights
    print(f"Loading Checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=dev)
    for k, v in ckpt['core'].items():
        if k in rafa_core.weights:
            rafa_core.weights[k].copy_(v)
    for k, v in ckpt['denoiser'].items():
        if k in denoiser.weights:
            denoiser.weights[k].copy_(v)
    
    # 3. Setup Diffusion
    timesteps = 1000 
    betas = make_beta_schedule(timesteps, 1e-4, 0.02, dev)
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)

    freq_bins = cfg["data"]["stft"]["n_fft"] // 2 + 1
    t_frames = int((cfg["data"]["sample_rate"] * cfg["data"]["clip_seconds"]) / cfg["data"]["stft"]["hop"]) + 1
    
    # 4. Generate Core Guidance (The "Thought")
    # In Stage 4, we generate the guidance from the NakedRAFA core based on mode/spectrum.
    # We use empty magnitude/phase to start the core's own internal iteration.
    dummy_mag = torch.zeros(1, freq_bins, t_frames, device=dev)
    dummy_phase = torch.zeros(1, freq_bins, t_frames, device=dev)
    
    ctrl = {
        "mode_id": torch.tensor([mode_id], device=dev),
        "spectrum_id": torch.tensor([spectrum_id], device=dev)
    }
    
    print(f"Generating Guidance (Mode: {mode_id}, Spectrum: {spectrum_id})...")
    p_p, p_m, _ = rafa_core.forward(dummy_mag, dummy_phase, control_matrix=ctrl)
    
    guide_m = p_m.detach()
    guide_p = p_p.detach()
    guide_z = torch.stack([torch.cos(guide_p), torch.sin(guide_p)], -1)

    # 5. Denoising Loop
    xt_mag = torch.randn(1, freq_bins, t_frames, device=dev)
    theta = 2.0 * torch.pi * torch.rand(1, freq_bins, t_frames, device=dev) - torch.pi
    xt_z = phasor_normalize(torch.stack([torch.cos(theta), torch.sin(theta)], dim=-1))

    idxs = torch.linspace(timesteps - 1, 0, steps, device=dev).round().long().tolist()
    
    print(f"Sampling {steps} Diffusion Steps...")
    for k, ti in enumerate(idxs):
        t = torch.full((1,), int(ti), device=dev, dtype=torch.long)
        
        # Denoiser guided by the core's "thought"
        pred_x0_mag, pred_x0_z = denoiser.forward(xt_mag, xt_z, t, rafa_mag=guide_m, rafa_z=guide_z)
        
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

    # 6. Reconstruct Waveform
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

    print(f"Saving Result: {out_path}")
    _save_wav(out_path, wav.unsqueeze(0), int(cfg["data"]["sample_rate"]))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--mode", type=int, default=0, help="Mode ID (0-9)")
    parser.add_argument("--spectrum", type=int, default=1, help="Spectrum ID (0-3)")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--steps", type=int, default=50)
    args = parser.parse_args()
    
    infer(args.ckpt, args.out, mode_id=args.mode, spectrum_id=args.spectrum, seed=args.seed, steps=args.steps)
