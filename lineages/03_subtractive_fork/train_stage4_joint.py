import os
import sys
import time
import torch
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader
from dataset import RafaGuerrillaDataset
from config import load_config
from lib_blackwell import NakedRAFA, NakedDenoiser
import rafa_math_tools as rmt
from diffusion_utils import make_beta_schedule, phase_to_phasor, q_sample_x0

# STAGE 4: CONSTITUTIONAL BINDING (Aligned Anvil - RESUME)
# Goal: Stabilize at 13GB VRAM to finish the graduation.

def pearson_correlation_loss(pred, target, eps=1e-8):
    pred_mean = pred - pred.mean(dim=-1, keepdim=True)
    target_mean = target - target.mean(dim=-1, keepdim=True)
    cov = (pred_mean * target_mean).sum(dim=-1)
    pred_var = (pred_mean ** 2).sum(dim=-1)
    target_var = (target_mean ** 2).sum(dim=-1)
    corr = cov / torch.sqrt((pred_var * target_var).clamp_min(eps))
    return (1.0 - corr).mean()

def compile_buckets(prompts):
    modes = {"engine": 1, "voice": 2, "impact": 3, "drone": 4, "bell": 5}
    spectra = {"dark": 0, "warm": 1, "bright": 2, "broadband": 3}
    batch_mode = []
    batch_spec = []
    for p in prompts:
        p_l = str(p).lower()
        batch_mode.append(next((v for k,v in modes.items() if k in p_l), 0))
        batch_spec.append(next((v for k,v in spectra.items() if k in p_l), 1))
    return {"mode_id": torch.tensor(batch_mode), "spectrum_id": torch.tensor(batch_spec)}

def main():
    cfg = load_config()
    dev = torch.device("cuda")
    
    rafa_core = NakedRAFA(d_model=768, dev='cpu')
    denoiser = NakedDenoiser(dev='cpu')
    
    # 2. LOAD RECENT EPOCH 49 CHECKPOINT
    ckpt_path = "checkpoints_stage4/bound_weights_ep49_step100.pt"
    if os.path.exists(ckpt_path):
        try:
            ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
            # Organic Loading: Only overwrite what exists in the checkpoint
            w_source = ckpt['core'] if 'core' in ckpt else ckpt
            for k, v in w_source.items():
                if k in rafa_core.weights:
                    rafa_core.weights[k] = v

            d_source = ckpt['denoiser'] if 'denoiser' in ckpt else ckpt
            for k, v in d_source.items():
                if k in denoiser.weights:
                    denoiser.weights[k] = v
            print(f"DEBUG: Organically Resumed from Checkpoint: {ckpt_path}")
        except Exception as e:
            print(f"DEBUG: Load failed, starting fresh: {e}")

    # 3. Migrate to GPU
    for k in rafa_core.weights: 
        rafa_core.weights[k] = rafa_core.weights[k].to(dev).detach()
        assert rafa_core.weights[k].is_cuda, f"FATAL: {k} failed to migrate to GPU"
    for k in denoiser.weights: 
        denoiser.weights[k] = denoiser.weights[k].to(dev).detach().requires_grad_(True)
        assert denoiser.weights[k].is_cuda, f"FATAL: {k} failed to migrate to GPU"
    rafa_core.qset = rafa_core.qset.to(dev); rafa_core.qw = rafa_core.qw.to(dev)
    print("DEBUG: THE ANVIL GUARD-RAIL ACTIVE. Sm_120 registers locked.")

    # 4. Optimizer
    opt = torch.optim.AdamW(list(denoiser.weights.values()), lr=1e-4)
    scaler = torch.amp.GradScaler("cuda")
    
    # STABILIZED BATCH SIZE (BS=24 provides ~2.5GB safety margin)
    train_ds = RafaGuerrillaDataset(cfg, split="training")
    loader = DataLoader(train_ds, batch_size=24, shuffle=True, num_workers=0)
    
    timesteps = 50
    betas = make_beta_schedule(timesteps, 1e-4, 0.02, dev)
    alphas_cumprod = torch.cumprod(1.0 - betas, dim=0)
    
    print("STAGE 4 START: Formalized Joint Binding (Aligned Anvil - RESUME)")
    for ep in range(50, 100): # Start from 50
        for i, batch in enumerate(loader):
            mag, phase, _, prompts, gt_tension = batch
            mag, phase, gt_tension = mag.to(dev), phase.to(dev), gt_tension.to(dev)
            x0_mag = torch.log1p(mag * 10.0) 
            x0_z = phase_to_phasor(phase)
            
            ctrl = compile_buckets(prompts)
            with torch.no_grad():
                p_p, p_m, _ = rafa_core.forward(mag.float(), phase.float(), control_matrix={"mode_id": ctrl["mode_id"].to(dev), "spectrum_id": ctrl["spectrum_id"].to(dev)})
            
            with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                is_shuffled = (torch.rand(1).item() < 0.5)
                if is_shuffled:
                    idx = torch.randperm(mag.size(0), device=dev)
                    guide_m, guide_p = p_m[idx].detach(), p_p[idx].detach()
                else:
                    guide_m, guide_p = p_m, p_p
                
                t = torch.randint(0, timesteps, (mag.size(0),), device=dev)
                xt_mag, xt_z, _, _ = q_sample_x0(x0_mag, x0_z, t, alphas_cumprod)
                d_mag, d_z = denoiser.forward(xt_mag, xt_z, t, rafa_mag=guide_m.half(), rafa_z=torch.stack([torch.cos(guide_p), torch.sin(guide_p)], -1).half())
                
                if not is_shuffled:
                    loss = F.mse_loss(d_mag, x0_mag) + F.mse_loss(d_z, x0_z)
                    status = "MATCHED"
                else:
                    recon_err = F.mse_loss(d_mag, x0_mag) + F.mse_loss(d_z, x0_z)
                    loss = F.relu(0.2 - recon_err) 
                    status = "SHUFFLED"
            
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            opt.zero_grad()
            
            if i % 20 == 0:
                print(f"EP {ep} STEP {i} | {status} | LOSS: {loss.item():.4f} | VOICE_MSE: {F.mse_loss(d_mag, x0_mag).item():.4f}")
            
            if i % 100 == 0:
                os.makedirs("checkpoints_stage4", exist_ok=True)
                torch.save({'core': rafa_core.weights, 'denoiser': denoiser.weights}, f"checkpoints_stage4/bound_weights_ep{ep}_step{i}.pt")

if __name__ == "__main__":
    main()
