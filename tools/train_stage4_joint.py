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

# STAGE 4: CONSTITUTIONAL BINDING (AMP + Accumulation Edition)
# Goal: 100% Stability on 16GB VRAM with RAFA-M logic.

def pearson_correlation_loss(pred, target, eps=1e-8):
    pred_mean = pred - pred.mean(dim=-1, keepdim=True)
    target_mean = target - target.mean(dim=-1, keepdim=True)
    cov = (pred_mean * target_mean).sum(dim=-1)
    pred_var = (pred_mean ** 2).sum(dim=-1)
    target_var = (target_mean ** 2).sum(dim=-1)
    corr = cov / torch.sqrt((pred_var * target_var).clamp_min(eps))
    return (1.0 - corr).mean()

def main():
    cfg = load_config()
    dev = torch.device("cuda")
    
    rafa_core = NakedRAFA(dev='cpu')
    denoiser = NakedDenoiser(dev='cpu')
    
    # Safe Load
    voice_ckpt = "checkpoints_stage1/voice_weights_ep9_step0.pt"
    if os.path.exists(voice_ckpt):
        try:
            saved_weights = torch.load(voice_ckpt, map_location='cpu')
            for k, v in saved_weights.items():
                if k in denoiser.weights: denoiser.weights[k] = v
            print("DEBUG: Organically Merged Voice Checkpoint.")
        except: pass

    # Migrate to GPU
    for k in rafa_core.weights: rafa_core.weights[k] = rafa_core.weights[k].to(dev).detach().requires_grad_(True)
    for k in denoiser.weights: denoiser.weights[k] = denoiser.weights[k].to(dev).detach().requires_grad_(True)
    rafa_core.qset = rafa_core.qset.to(dev); rafa_core.qw = rafa_core.qw.to(dev)
    print("DEBUG: Organs migrated to Sm_120 registers.")

    # Constitutional Optimizer
    params = [
        {'params': [p for n, p in rafa_core.weights.items() if "ps_" in n], 'lr': 1e-6},
        {'params': [p for n, p in rafa_core.weights.items() if "ps_" not in n], 'lr': 1e-5},
        {'params': list(denoiser.weights.values()), 'lr': 1e-6}
    ]
    opt = torch.optim.AdamW(params)
    
    # --- AMP SETUP ---
    scaler = torch.amp.GradScaler("cuda")
    
    # Small Batch for VRAM safety
    train_ds = RafaGuerrillaDataset(cfg, split="training")
    loader = DataLoader(train_ds, batch_size=2, shuffle=True, num_workers=0)
    accum_steps = 4 # Effective BS = 8
    
    timesteps = 50
    betas = make_beta_schedule(timesteps, 1e-4, 0.02, dev)
    alphas_cumprod = torch.cumprod(1.0 - betas, dim=0)
    
    print("STAGE 4 START: Constitutional Binding (AMP Stable)")
    for ep in range(100):
        opt.zero_grad()
        for i, batch in enumerate(loader):
            mag, phase, _, prompts, gt_tension = batch
            mag, phase, gt_tension = mag.to(dev), phase.to(dev), gt_tension.to(dev)
            x0_mag = torch.log1p(mag)
            x0_z = phase_to_phasor(phase)
            
            # --- HETEROGENEOUS PRECISION FORWARD PASS ---
            # Brain stays in FP32 for Triton stability
            p_p, p_m, p_ext = rafa_core.forward(mag.float(), phase.to(dev).float())
            torch.cuda.synchronize()

            with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                is_shuffled = (torch.rand(1).item() < 0.5)
                if is_shuffled:
                    idx = torch.randperm(mag.size(0), device=dev)
                    guide_m, guide_p = p_m[idx].detach(), p_p[idx].detach()
                else:
                    guide_m, guide_p = p_m, p_p

                # Voice stays in FP16 (AMP)
                t = torch.randint(0, timesteps, (mag.size(0),), device=dev)
                xt_mag, xt_z, _, _ = q_sample_x0(x0_mag, x0_z, t, alphas_cumprod)
                # Ensure guidance is cast to match AMP context
                d_mag, d_z = denoiser.forward(xt_mag, xt_z, t, rafa_mag=guide_m.half(), rafa_z=torch.stack([torch.cos(guide_p), torch.sin(guide_p)], -1).half())

                # LOSS
                if not is_shuffled:
                    loss_voice = F.mse_loss(d_mag, x0_mag) + F.mse_loss(d_z, x0_z)
                    status = "MATCHED"
                else:
                    recon_err = F.mse_loss(d_mag, x0_mag) + F.mse_loss(d_z, x0_z)
                    loss_voice = F.relu(0.2 - recon_err) 
                    status = "SHUFFLED"
                
                zf = p_ext["phase_state"]
                h_e, i_e = rmt.harmonic_inharmonic_energy(zf, [tuple(x) for x in cfg['training']['rational_ratios']])
                actual = i_e / (h_e + i_e).clamp_min(1e-8)
                if actual.shape != gt_tension.shape:
                    actual = F.interpolate(actual.unsqueeze(1), size=gt_tension.size(1), mode="linear", align_corners=True).squeeze(1)
                loss_brain = pearson_correlation_loss(actual, gt_tension)
                
                loss = (0.8 * loss_voice + 0.2 * loss_brain) / accum_steps
            
            # --- AMP BACKWARD PASS ---
            scaler.scale(loss).backward()
            
            if (i + 1) % accum_steps == 0:
                scaler.unscale_(opt)
                # Flatten all parameters for clipping
                all_naked_params = list(rafa_core.weights.values()) + list(denoiser.weights.values())
                torch.nn.utils.clip_grad_norm_(all_naked_params, 0.1)
                scaler.step(opt)
                scaler.update()
                opt.zero_grad()
                torch.cuda.synchronize()
            
            if i % 20 == 0:
                print(f"STEP {i} | {status} | LOSS: {loss.item()*accum_steps:.4f} | VOICE_MSE: {F.mse_loss(d_mag, x0_mag).item():.4f} | CORR: {1.0-loss_brain.item():.4f}")
            
            if i % 100 == 0:
                os.makedirs("checkpoints_stage4", exist_ok=True)
                torch.save({'core': rafa_core.weights, 'denoiser': denoiser.weights}, f"checkpoints_stage4/bound_weights_ep{ep}_step{i}.pt")

if __name__ == "__main__":
    main()
