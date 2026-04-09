import torch
import torch.nn.functional as F
import numpy as np
import os
import json
from lib_blackwell import NakedRAFA, NakedDenoiser
from dataset import RafaGuerrillaDataset
from config import load_config
from diffusion_utils import make_beta_schedule, phase_to_phasor, phasor_to_phase
import stft_utils

def q_sample_x0(x0_mag, x0_z, t, alphas_cumprod):
    sqrt_alpha_bar = torch.sqrt(alphas_cumprod[t]).view(-1, 1, 1)
    if x0_z.ndim == 4: sqrt_alpha_bar = sqrt_alpha_bar.unsqueeze(-1)
    sqrt_one_minus_alpha_bar = torch.sqrt(1 - alphas_cumprod[t]).view(-1, 1, 1)
    if x0_z.ndim == 4: sqrt_one_minus_alpha_bar = sqrt_one_minus_alpha_bar.unsqueeze(-1)
    noise_m = torch.randn_like(x0_mag)
    noise_z = torch.randn_like(x0_z)
    xt_m = sqrt_alpha_bar * x0_mag + sqrt_one_minus_alpha_bar * noise_m
    xt_z = sqrt_alpha_bar * x0_z + sqrt_one_minus_alpha_bar * noise_z
    return xt_m, xt_z, noise_m, noise_z

def run_stress_test():
    cfg = load_config()
    dev = torch.device("cuda")
    stft_cfg = cfg["data"]["stft"]
    
    # 1. Load Best Graduation weights
    rafa_core = NakedRAFA(d_model=768, dev=dev)
    denoiser = NakedDenoiser(dev=dev)
    ckpt = torch.load("checkpoints_stage4/bound_weights_ep99_step200.pt", map_location=dev, weights_only=False)
    rafa_core.weights = ckpt['core']
    denoiser.weights = ckpt['denoiser']
    rafa_core.qset = rafa_core.qset.to(dev); rafa_core.qw = rafa_core.qw.to(dev)

    # 2. Setup Audit Sample
    ds = RafaGuerrillaDataset(cfg, split="training")
    mag, phase, _, prompts, _ = ds[42]
    mag, phase = mag.to(dev).unsqueeze(0), phase.to(dev).unsqueeze(0)
    x0_mag, x0_z = torch.log1p(mag * 10.0), phase_to_phasor(phase)
    
    timesteps = 50
    betas = make_beta_schedule(timesteps, 1e-4, 0.02, dev)
    alphas_cumprod = torch.cumprod(1.0 - betas, dim=0)
    t = torch.ones(1, dtype=torch.long, device=dev) * 25
    xt_mag, xt_z, _, _ = q_sample_x0(x0_mag, x0_z, t, alphas_cumprod)
    
    # Ensure rank consistency for denoiser.forward
    if xt_mag.ndim == 4: xt_mag = xt_mag.squeeze(1)
    if xt_z.ndim == 5: xt_z = xt_z.squeeze(1)

    results = {}

    with torch.no_grad():
        # --- TEST 1: KNOCKOUT MATRIX ---
        p_p, p_m, p_ext = rafa_core.forward(mag, phase, control_matrix={"mode_id": 1})
        p_z = torch.stack([torch.cos(p_p), torch.sin(p_p)], -1)
        
        # A: Matched (Baseline)
        d_m_match, _ = denoiser.forward(xt_mag, xt_z, t, rafa_mag=p_m, rafa_z=p_z)
        results["matched"] = F.mse_loss(d_m_match, x0_mag).item()
        
        # B: Phase Knockout (Structural Zeroing)
        d_m_pko, _ = denoiser.forward(xt_mag, xt_z, t, rafa_mag=p_m, rafa_z=torch.zeros_like(p_z))
        results["phase_knockout"] = F.mse_loss(d_m_pko, x0_mag).item()
        
        # C: Magnitude Knockout (Spectral Darkness)
        d_m_mko, _ = denoiser.forward(xt_mag, xt_z, t, rafa_mag=torch.zeros_like(p_m), rafa_z=p_z)
        results["mag_knockout"] = F.mse_loss(d_m_mko, x0_mag).item()
        
        # D: Random Phase Sabotage (Adversarial Lattice)
        rand_z = torch.randn_like(p_z)
        rand_z = rand_z / torch.sqrt((rand_z**2).sum(-1, keepdim=True) + 1e-8)
        d_m_randp, _ = denoiser.forward(xt_mag, xt_z, t, rafa_mag=p_m, rafa_z=rand_z)
        results["random_phase"] = F.mse_loss(d_m_randp, x0_mag).item()

        # --- TEST 2: INTERNAL STATE TRACING ---
        h_slow = p_ext["h_slow"] 
        results["mind_variance_dark"] = h_slow.var().item()

    print("\n" + "="*60)
    print("RAFA-M HARD ABLATION SCORECARD")
    print("="*60)
    print(f"Matched Baseline Loss:  {results['matched']:.6f}")
    print(f"Phase Knockout Loss:    {results['phase_knockout']:.6f} (Delta: {((results['phase_knockout']/results['matched'])-1)*100:+.2f}%)")
    print(f"Mag Knockout Loss:      {results['mag_knockout']:.6f} (Delta: {((results['mag_knockout']/results['matched'])-1)*100:+.2f}%)")
    print(f"Random Phase Sabotage:  {results['random_phase']:.6f} (Delta: {((results['random_phase']/results['matched'])-1)*100:+.2f}%)")
    print(f"Mind Trace (Darkness):  Variance {results['mind_variance_dark']:.6f}")
    print("="*60)

    if results['random_phase'] > 10 * results['matched']:
        print("VERDICT: PHYSICALLY BOUND. The Mouth cannot speak without the correct Phase Key.")
    else:
        print("VERDICT: WEAK BINDING. The Mouth is using Magnitude shortcuts.")

    if results['mind_variance_dark'] > 0.01:
        print("VERDICT: TRUE PHASE-NATIVE. The Mind dreams logic in the dark.")

if __name__ == "__main__":
    run_stress_test()
