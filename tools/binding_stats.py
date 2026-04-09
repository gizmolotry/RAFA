import torch
import torch.nn.functional as F
import numpy as np
import os
from torch.utils.data import Subset
from lib_blackwell import NakedRAFA, NakedDenoiser
from dataset import RafaGuerrillaDataset
from config import load_config
from diffusion_utils import phase_to_phasor, q_sample_x0, make_beta_schedule

def compute_gap(ckpt_path):
    cfg = load_config()
    dev = torch.device("cuda")
    rafa_core = NakedRAFA(d_model=768, dev=dev)
    denoiser = NakedDenoiser(dev=dev)
    ckpt = torch.load(ckpt_path, map_location=dev, weights_only=False)
    rafa_core.weights = ckpt['core']
    denoiser.weights = ckpt['denoiser']

    ds = RafaGuerrillaDataset(cfg, split="training")
    loader = torch.utils.data.DataLoader(Subset(ds, list(range(128, 160))), batch_size=4)
    
    matched_losses = []
    shuffled_losses = []
    
    timesteps = 50
    betas = make_beta_schedule(timesteps, 1e-4, 0.02, dev)
    alphas_cumprod = torch.cumprod(1.0 - betas, dim=0)
    t = torch.ones(4, dtype=torch.long, device=dev) * 25

    with torch.no_grad():
        for batch in loader:
            mag, phase, _, prompts, _ = batch
            mag, phase = mag.to(dev), phase.to(dev)
            x0_mag, x0_z = torch.log1p(mag), phase_to_phasor(phase)
            xt_mag, xt_z, _, _ = q_sample_x0(x0_mag, x0_z, t[:mag.size(0)], alphas_cumprod)
            
            # Matched
            mode_ids = torch.tensor([1 if "engine" in str(p).lower() else 0 for p in prompts], device=dev)
            p_p, p_m, _ = rafa_core.forward(mag, phase, control_matrix={"mode_id": mode_ids})
            d_mag, d_z = denoiser.forward(xt_mag, xt_z, t[:mag.size(0)], rafa_mag=p_m, rafa_z=torch.stack([torch.cos(p_p), torch.sin(p_p)], -1))
            matched_losses.append((F.mse_loss(d_mag, x0_mag) + F.mse_loss(d_z, x0_z)).item())
            
            # Shuffled
            idx = torch.randperm(mag.size(0), device=dev)
            d_mag_s, d_z_s = denoiser.forward(xt_mag, xt_z, t[:mag.size(0)], rafa_mag=p_m[idx], rafa_z=torch.stack([torch.cos(p_p[idx]), torch.sin(p_p[idx])], -1))
            shuffled_losses.append((F.mse_loss(d_mag_s, x0_mag) + F.mse_loss(d_z_s, x0_z)).item())

    m_l, s_l = np.mean(matched_losses), np.mean(shuffled_losses)
    return {"mean_matched": m_l, "mean_shuffled": s_l, "ratio": s_l / m_l}

if __name__ == "__main__":
    res = compute_gap("checkpoints_stage4/bound_weights_ep9_step900.pt")
    import json
    print(json.dumps(res, indent=4))
