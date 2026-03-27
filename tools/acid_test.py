import os
import sys
import time
import torch
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader, Subset
from dataset import RafaGuerrillaDataset
from config import load_config
from lib_blackwell import NakedRAFA
import rafa_math_tools as rmt

# --- THE VALIDATION PROTOCOL: RAFA-M (80M) ACID TEST ---
# Goal: Defensible A/B comparison of Formal vs. Embedding interfaces.

def pearson_correlation_loss(pred, target, eps=1e-8):
    pred_mean = pred - pred.mean(dim=-1, keepdim=True)
    target_mean = target - target.mean(dim=-1, keepdim=True)
    cov = (pred_mean * target_mean).sum(dim=-1)
    pred_var = (pred_mean ** 2).sum(dim=-1)
    target_var = (target_mean ** 2).sum(dim=-1)
    corr = cov / torch.sqrt((pred_var * target_var).clamp_min(eps))
    return (1.0 - corr).mean()

def run_experiment(mode="formal"):
    cfg = load_config()
    dev = torch.device("cuda")
    
    # 1. Force RAFA-M (80M) Scale
    model = NakedRAFA(d_model=768, dev=dev)
    # Manually force migration to be sure
    for k in model.weights:
        model.weights[k] = model.weights[k].to(dev).detach().requires_grad_(True)
    
    print(f"DEBUG: Instantiated RAFA-M with d_model={model.d_model}")
    
    # 2. Constitutional Optimizer
    params = list(model.weights.values())
    total_params = sum(p.numel() for p in params)
    print(f"DEBUG: Total Trainable Parameters: {total_params/1e6:.2f}M")
    print(f"DEBUG: Parameter Device: {params[0].device}")
    opt = torch.optim.AdamW(params, lr=1e-5)
    scaler = torch.amp.GradScaler("cuda")
    
    # 3. 128/32 Workplace Split
    ds = RafaGuerrillaDataset(cfg, split="training")
    train_indices = list(range(0, 128))
    val_indices = list(range(128, 160))
    train_loader = DataLoader(Subset(ds, train_indices), batch_size=4, shuffle=True)
    val_loader = DataLoader(Subset(ds, val_indices), batch_size=4, shuffle=False)
    
    accum_steps = 2 # Effective BS = 8
    
    log_path = f"logs/acid_test_m_{mode}.csv"
    with open(log_path, "w") as f: f.write("epoch,train_corr,val_corr,vram_gb\n")

    print(f"DEBUG: Starting RAFA-M {mode.upper()} Experiment...")
    for ep in range(500):
        model_train_corrs = []
        opt.zero_grad()
        
        # --- TRAIN LOOP ---
        for i, batch in enumerate(train_loader):
            mag, phase, _, prompts, gt_tension = batch
            mag, phase, gt_tension = mag.to(dev), phase.to(dev), gt_tension.to(dev)
            
            with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                # Apply Interface Logic
                if mode == "formal":
                    # Bucket 1: Rule-based mode detection
                    mode_id = 1 if "engine" in str(prompts[0]).lower() else 0
                    pred_p, _, p_ext = model.forward(mag, phase, control_matrix={"mode_id": mode_id})
                else:
                    # Old Head (Bypassing control_matrix)
                    pred_p, _, p_ext = model.forward(mag, phase)
                
                zf = p_ext["phase_state"]
                h_e, i_e = rmt.harmonic_inharmonic_energy(zf, [tuple(x) for x in cfg['training']['rational_ratios']])
                actual = i_e / (h_e + i_e).clamp_min(1e-8)
                if actual.shape != gt_tension.shape:
                    actual = F.interpolate(actual.unsqueeze(1), size=gt_tension.size(1), mode="linear", align_corners=True).squeeze(1)
                
                loss = pearson_correlation_loss(actual, gt_tension) / accum_steps
            
            scaler.scale(loss).backward()
            
            if (i + 1) % accum_steps == 0:
                scaler.step(opt)
                scaler.update()
                opt.zero_grad()
                
            with torch.no_grad():
                c = np.corrcoef(actual[0].float().cpu().numpy(), gt_tension[0].cpu().numpy())[0, 1]
                if not np.isnan(c): model_train_corrs.append(c)

        # --- VAL LOOP (Held-out) ---
        val_corrs = []
        with torch.no_grad():
            for batch in val_loader:
                mag, phase, _, prompts, gt_tension = batch
                mag, phase, gt_tension = mag.to(dev), phase.to(dev), gt_tension.to(dev)
                mode_id = 1 if "engine" in str(prompts[0]).lower() else 0
                
                with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                    pred_p, _, p_ext = model.forward(mag, phase, control_matrix={"mode_id": mode_id} if mode=="formal" else None)
                    zf = p_ext["phase_state"]
                    h_e, i_e = rmt.harmonic_inharmonic_energy(zf, [tuple(x) for x in cfg['training']['rational_ratios']])
                    actual = i_e / (h_e + i_e).clamp_min(1e-8)
                    if actual.shape != gt_tension.shape:
                        actual = F.interpolate(actual.unsqueeze(1), size=gt_tension.size(1), mode="linear", align_corners=True).squeeze(1)
                
                c = np.corrcoef(actual[0].float().cpu().numpy(), gt_tension[0].cpu().numpy())[0, 1]
                if not np.isnan(c): val_corrs.append(c)
        
        t_c, v_c = np.mean(model_train_corrs), np.mean(val_corrs)
        vram = torch.cuda.memory_allocated() / 1e9
        
        if ep % 5 == 0:
            print(f"[{mode.upper()}] EP {ep} | TRAIN: {t_c:.4f} | VAL: {v_c:.4f} | VRAM: {vram:.2f}GB")
        
        with open(log_path, "a") as f:
            f.write(f"{ep},{t_c},{v_c},{vram}\n")

if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "formal"
    run_experiment(mode)
