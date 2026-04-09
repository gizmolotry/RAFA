import sys
import os
import time
import torch
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader

# TOTAL PROJECT ISOLATION: No third-party RNN/Transformer imports allowed
# We use our sterile Blackwell core logic
from blackwell_model import RAFA
from dataset import RafaGuerrillaDataset
from config import load_config
import rafa_math_tools as rmt
from diffusion_utils import make_beta_schedule, extract, phase_to_phasor, q_sample_x0

# BLACKWELL sm_120 WORKAROUND
torch.backends.cudnn.enabled = False

def main():
    cfg = load_config()
    dev = torch.device("cuda")
    
    # 1. Initialize the Model (Physically isolated from rnn.py)
    print("DEBUG: Initializing Sterile RAFA...")
    # RAFA in blackwell_model.py is already a sterile implementation
    model = RAFA(cfg).to(dev)
    
    # 2. Setup Data
    train_ds = RafaGuerrillaDataset(cfg, split="training")
    loader = DataLoader(train_ds, batch_size=16, shuffle=True, num_workers=0)
    
    # 3. Setup Optimizer
    opt = torch.optim.AdamW(model.parameters(), lr=1e-5)
    
    # 4. Setup Diffusion Schedule
    timesteps = 50
    betas = make_beta_schedule(timesteps, 1e-4, 0.02, dev)
    alphas_cumprod = torch.cumprod(1.0 - betas, dim=0)
    
    print("DEBUG: Blackwell Master Training Started (sm_120 Native)")
    for ep in range(10):
        for i, batch in enumerate(loader):
            mag, phase, _, _, gt_tension = batch
            mag, phase = mag.to(dev), phase.to(dev)
            gt_tension = gt_tension.to(dev)
            
            # Diffusion Invariants
            x0_mag = torch.log1p(mag)
            x0_z = phase_to_phasor(phase)
            t = torch.randint(0, timesteps, (mag.size(0),), device=dev)
            xt_mag, xt_z, eps_mag, eps_z = q_sample_x0(x0_mag, x0_z, t, alphas_cumprod)
            
            # Forward Logic (Functional Path)
            # We call model() directly which executes our sterile sm_120 logic
            pred_p, pred_m, phase_ext, _ = model(xt_mag, xt_z, semantic_controls={"semantic_tension": gt_tension})
            
            # Ph.D. Level Relational Loss: Alignment to Ground-Truth Tension
            zf = phase_ext["phase_state"]
            h_e, i_e = rmt.harmonic_inharmonic_energy(zf, [tuple(x) for x in cfg['training']['rational_ratios']])
            actual = i_e / (h_e + i_e).clamp_min(1e-8)
            
            # Alignment Loss
            loss = F.mse_loss(actual, gt_tension)
            
            loss.backward()
            opt.step()
            opt.zero_grad()
            
            if i % 10 == 0:
                print(f"EP {ep} STEP {i} | LOSS: {loss.item():.6f} | VRAM: {torch.cuda.memory_allocated()/1e9:.2f}GB")
            if i >= 1000: break

if __name__ == "__main__":
    main()
