import sys
import os
import time
import torch
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader

# TOTAL FUNCTIONAL ESTABLISHMENT
# No RAFA class. No Module. Just pure math.

from dataset import RafaGuerrillaDataset
from config import load_config
import rafa_math_tools as rmt
from diffusion_utils import make_beta_schedule, extract, phase_to_phasor, q_sample_x0

# BLACKWELL sm_120 WORKAROUND
torch.backends.cudnn.enabled = False

def init_weights(dim_in, dim_out):
    w = torch.empty(dim_out, dim_in).cuda()
    torch.nn.init.xavier_uniform_(w)
    return torch.nn.Parameter(w)

def main():
    cfg = load_config()
    dev = torch.device("cuda")
    
    print("DEBUG: Initializing Raw Tensors (No nn.Module)...")
    freq_bins = 129
    d_model = 256
    
    # 1. Raw Weights for Gear 1
    w_proj = init_weights(6 * freq_bins, d_model)
    b_proj = torch.nn.Parameter(torch.zeros(d_model).cuda())
    
    w_ph = init_weights(d_model, freq_bins)
    w_mag = init_weights(d_model, freq_bins)
    
    # 2. Setup Data
    train_ds = RafaGuerrillaDataset(cfg, split="training")
    loader = DataLoader(train_ds, batch_size=8, shuffle=True, num_workers=0)
    
    # 3. Setup Diffusion
    timesteps = 50
    betas = make_beta_schedule(timesteps, 1e-4, 0.02, dev)
    alphas_cumprod = torch.cumprod(1.0 - betas, dim=0)
    
    print("DEBUG: Raw Tensor Training Started (Absolute sm_120 Isolation)")
    for i, batch in enumerate(loader):
        mag, phase, _, _, gt_tension = batch
        mag, phase = mag.to(dev), phase.to(dev)
        gt_tension = gt_tension.to(dev)
        
        # Simple Logic Pass
        x = torch.cat([mag]*6, dim=1).transpose(1, 2)
        x = F.linear(x, w_proj, b_proj)
        
        # We stop here to test if basic matmul triggers the bug
        loss = x.sum() * 0.0 + gt_tension.sum() * 0.0
        
        loss.backward()
        
        # Manual step
        with torch.no_grad():
            w_proj -= 1e-5 * w_proj.grad
            w_proj.grad.zero_()
            
        if i % 1 == 0:
            print(f"STEP {i} | SUCCESSFUL MATMUL | VRAM: {torch.cuda.memory_allocated()/1e9:.2f}GB")
        if i >= 10: break

if __name__ == "__main__":
    main()
