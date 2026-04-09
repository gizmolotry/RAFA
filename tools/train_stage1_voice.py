import os
import sys
import time
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from dataset import RafaGuerrillaDataset
from config import load_config
from lib_blackwell import NakedDenoiser
from diffusion_utils import make_beta_schedule, extract, phase_to_phasor, q_sample_x0

# STAGE 1: VOICE BOOT CAMP
# Goal: Make the mouth competent. Train NakedDenoiser reconstruction only.

def main():
    cfg = load_config()
    dev = torch.device("cuda")
    
    # 1. Initialize Naked Denoiser (sm_120 Safe)
    model = NakedDenoiser(dev=dev)
    
    # 2. Setup Data (5.8k Local Corpus)
    train_ds = RafaGuerrillaDataset(cfg, split="training")
    loader = DataLoader(train_ds, batch_size=8, shuffle=True, num_workers=0)

    
    # 3. Setup Optimizer
    opt = torch.optim.AdamW(list(model.weights.values()), lr=1e-4)
    
    # 4. Diffusion Setup
    timesteps = 1000
    betas = make_beta_schedule(timesteps, 1e-4, 0.02, dev)
    alphas_cumprod = torch.cumprod(1.0 - betas, dim=0)
    
    print("STAGE 1 START: Voice Boot Camp (sm_120 Stable)")
    for ep in range(10):
        for i, batch in enumerate(loader):
            mag, phase, _, _, _ = batch
            mag, phase = mag.to(dev), phase.to(dev)
            
            x0_mag = torch.log1p(mag)
            x0_z = phase_to_phasor(phase)
            
            t = torch.randint(0, timesteps, (mag.size(0),), device=dev)
            xt_mag, xt_z, eps_mag, eps_z = q_sample_x0(x0_mag, x0_z, t, alphas_cumprod)
            
            pred_mag, pred_z = model.forward(xt_mag, xt_z, t)
            
            loss = F.mse_loss(pred_mag, x0_mag) + F.mse_loss(pred_z, x0_z)
            
            loss.backward()
            opt.step()
            opt.zero_grad()
            
            if i % 20 == 0:
                print(f"EP {ep} STEP {i} | VOICE LOSS: {loss.item():.6f}")
            
            if i % 1000 == 0:
                os.makedirs("checkpoints_stage1", exist_ok=True)
                torch.save(model.weights, f"checkpoints_stage1/voice_weights_ep{ep}_step{i}.pt")

if __name__ == "__main__":
    main()
