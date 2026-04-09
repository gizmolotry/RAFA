import os
import sys
import time
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from dataset import RafaGuerrillaDataset
from config import load_config
from lib_blackwell import NakedRAFA
from diffusion_utils import phase_to_phasor

# STAGE 2: SPINE BOOT CAMP
# Goal: Don't drop the civilization. Train NakedRAFA Gears/Clutches for long-range form.

def main():
    cfg = load_config()
    dev = torch.device("cuda")
    
    # 1. Initialize Naked RAFA (sm_120 Safe)
    model = NakedRAFA(dev=dev)
    
    # 2. Freeze Brain (Phase Solver) Weights
    # In NakedRAFA, brain weights start with 'ps_'
    params = []
    for name, p in model.weights.items():
        if "ps_" not in name:
            params.append(p)
        else:
            p.requires_grad = False # Freeze Brain
            
    opt = torch.optim.AdamW(params, lr=1e-4)
    
    # 3. Setup Data (Masked Span Reconstruction)
    train_ds = RafaGuerrillaDataset(cfg, split="training")
    loader = DataLoader(train_ds, batch_size=4, shuffle=True, num_workers=0)
    
    print("STAGE 2 START: Spine Boot Camp (sm_120 Stable)")
    for ep in range(10):
        for i, batch in enumerate(loader):
            mag, phase, _, _, _ = batch
            mag, phase = mag.to(dev), phase.to(dev)
            
            # --- TASK: MASKED RECONSTRUCTION ---
            mask = torch.ones_like(mag)
            t_start = torch.randint(0, mag.size(-1) - 50, (1,)).item()
            mask[..., t_start:t_start+50] = 0.0
            
            masked_mag = mag * mask
            masked_phase = phase * mask
            
            # Forward
            pred_p, pred_m, _ = model.forward(masked_mag, masked_phase)
            
            # SPINE LOSS: Alignment to original unmasked latents
            loss_mag = F.mse_loss(pred_m, torch.log1p(mag))
            loss_phase = F.mse_loss(torch.stack([torch.cos(pred_p), torch.sin(pred_p)], -1), phase_to_phasor(phase))
            
            loss = loss_mag + loss_phase
            
            loss.backward()
            opt.step()
            opt.zero_grad()
            
            if i % 20 == 0:
                print(f"EP {ep} STEP {i} | SPINE LOSS: {loss.item():.6f}")
            
            if i % 1000 == 0:
                os.makedirs("checkpoints_stage2", exist_ok=True)
                torch.save(model.weights, f"checkpoints_stage2/spine_weights_ep{ep}_step{i}.pt")

if __name__ == "__main__":
    main()
