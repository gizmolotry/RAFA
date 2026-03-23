import torch
import torch.nn.functional as F
import numpy as np
import sys
import os
from lib_blackwell import NakedRAFA, NakedDenoiser
from dataset import RafaGuerrillaDataset
from config import load_config
from diffusion_utils import phase_to_phasor, q_sample_x0, make_beta_schedule

def run_audit(ckpt_path):
    cfg = load_config()
    dev = torch.device("cuda")
    
    # 1. Initialize and Load Audit Target
    rafa_core = NakedRAFA(dev=dev)
    denoiser = NakedDenoiser(dev=dev)
    
    if os.path.exists(ckpt_path):
        print(f"DEBUG: Auditing Checkpoint: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=dev)
        rafa_core.weights = ckpt['core']
        denoiser.weights = ckpt['denoiser']
    else:
        print(f"ERROR: Checkpoint {ckpt_path} not found.")
        return

    # 2. Grab Audit Batch
    train_ds = RafaGuerrillaDataset(cfg, split="training")
    loader = torch.utils.data.DataLoader(train_ds, batch_size=4, shuffle=True)
    batch = next(iter(loader))
    mag, phase, _, _, _ = batch
    mag, phase = mag.to(dev), phase.to(dev)
    
    x0_mag = torch.log1p(mag)
    x0_z = phase_to_phasor(phase)
    
    timesteps = 50
    betas = make_beta_schedule(timesteps, 1e-4, 0.02, dev)
    alphas_cumprod = torch.cumprod(1.0 - betas, dim=0)
    t = torch.ones(mag.size(0), dtype=torch.long, device=dev) * 25
    xt_mag, xt_z, _, _ = q_sample_x0(x0_mag, x0_z, t, alphas_cumprod)

    # --- PERTURBATION MATRIX ---
    
    # A: Normal
    p_p, p_m, _ = rafa_core.forward(mag, phase)
    d_mag_a, d_z_a = denoiser.forward(xt_mag, xt_z, t, rafa_mag=p_m, rafa_z=torch.stack([torch.cos(p_p), torch.sin(p_p)], -1))
    loss_a = F.mse_loss(d_mag_a, x0_mag) + F.mse_loss(d_z_a, x0_z)
    
    # B: Zeroed
    d_mag_b, d_z_b = denoiser.forward(xt_mag, xt_z, t, rafa_mag=None, rafa_z=None)
    loss_b = F.mse_loss(d_mag_b, x0_mag) + F.mse_loss(d_z_b, x0_z)
    
    # C: Shuffled
    idx = torch.randperm(mag.size(0), device=dev)
    p_m_shuf = p_m[idx]
    p_p_shuf = p_p[idx]
    d_mag_c, d_z_c = denoiser.forward(xt_mag, xt_z, t, rafa_mag=p_m_shuf, rafa_z=torch.stack([torch.cos(p_p_shuf), torch.sin(p_p_shuf)], -1))
    loss_c = F.mse_loss(d_mag_c, x0_mag) + F.mse_loss(d_z_c, x0_z)

    print("\n" + "="*60)
    print(f"CAUSAL BINDING AUDIT: {os.path.basename(ckpt_path)}")
    print("="*60)
    print(f"Condition A (Matched):  Loss {loss_a.item():.6f}")
    print(f"Condition B (Zeroed):   Loss {loss_b.item():.6f} | Delta: {((loss_b/loss_a)-1)*100:+.2f}%")
    print(f"Condition C (Shuffled): Loss {loss_c.item():.6f} | Delta: {((loss_c/loss_a)-1)*100:+.2f}%")
    print("="*60)
    
    if loss_b > 1.2 * loss_a:
        print("VERDICT: BINDING EMERGENCE. The Voice is listening.")
    else:
        print("VERDICT: BLACK MARKET. The Voice is ignoring the Brain.")

if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else "checkpoints_stage4/bound_weights_ep0_step0.pt"
    run_audit(target)
