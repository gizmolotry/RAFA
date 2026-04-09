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

# --- PHASE A/C: CONTROL CONTRACT + SUPERVISION (128/32 Split) ---

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

def run_ablation(variant="full"):
    cfg = load_config()
    dev = torch.device("cuda")
    model = NakedRAFA(d_model=768, dev=dev)
    for k in model.weights:
        model.weights[k] = model.weights[k].to(dev).detach().requires_grad_(True)
    model.qset = model.qset.to(dev); model.qw = model.qw.to(dev)

    # All Logic parameters unfrozen
    params = [p for n, p in model.weights.items() if "ps_" in n or "mode_seeds" in n or "spectrum_biases" in n or "lexicon_phasors" in n]
    opt = torch.optim.AdamW(params, lr=1e-4)
    scaler = torch.amp.GradScaler("cuda")
    
    ds = RafaGuerrillaDataset(cfg, split="training")
    train_indices = list(range(0, 128))
    val_indices = list(range(128, 160))
    train_loader = DataLoader(Subset(ds, train_indices), batch_size=8, shuffle=True)
    val_loader = DataLoader(Subset(ds, val_indices), batch_size=8, shuffle=False)
    
    log_path = f"logs/ablation_hardened_{variant}.csv"
    with open(log_path, "w") as f: f.write("epoch,train_corr,val_corr\n")

    print(f"DEBUG: Starting Hardened Ablation: {variant.upper()}")
    for ep in range(500):
        train_corrs = []
        for batch in train_loader:
            mag, phase, _, prompts, gt_tension = batch
            mag, phase, gt_tension = mag.to(dev), phase.to(dev), gt_tension.to(dev)
            
            with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                if variant == "old":
                    max_l = max(len(p) for p in prompts)
                    text_tokens = torch.zeros(len(prompts), max_l, dtype=torch.long, device=dev)
                    for b_idx, p in enumerate(prompts):
                        b = list(str(p).encode("utf-8", errors="ignore"))
                        text_tokens[b_idx, :len(b)] = torch.tensor(b, dtype=torch.long, device=dev)
                    pred_p, _, p_ext = model.forward(mag, phase, text_tokens=text_tokens)
                else:
                    ctrl = compile_buckets(prompts)
                    mode_id = ctrl["mode_id"][0].item()
                    spec_id = ctrl["spectrum_id"][0].item() if variant == "full" else None
                    pred_p, _, p_ext = model.forward(mag, phase, control_matrix={"mode_id": mode_id, "spectrum_id": spec_id})
                
                zf = p_ext["phase_state"]
                h_e, i_e = rmt.harmonic_inharmonic_energy(zf, [tuple(x) for x in cfg['training']['rational_ratios']])
                actual = i_e / (h_e + i_e).clamp_min(1e-8)
                if actual.shape != gt_tension.shape:
                    actual = F.interpolate(actual.unsqueeze(1), size=gt_tension.size(1), mode="linear", align_corners=True).squeeze(1)
                
                # --- PHASE C: HYBRID CONSTITUTIONAL LOSS ---
                l_mse = F.mse_loss(actual, gt_tension)
                l_pear = pearson_correlation_loss(actual, gt_tension)
                loss = 0.3 * l_mse + 0.7 * l_pear
            
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            opt.zero_grad()
            
            with torch.no_grad():
                c = np.corrcoef(actual[0].float().cpu().numpy(), gt_tension[0].cpu().numpy())[0, 1]
                if not np.isnan(c): train_corrs.append(c)

        # VALIDATE
        val_corrs = []
        with torch.no_grad():
            for batch in val_loader:
                mag, phase, _, prompts, gt_tension = batch
                mag, phase, gt_tension = mag.to(dev), phase.to(dev), gt_tension.to(dev)
                if variant == "old":
                    pred_p, _, p_ext = model.forward(mag, phase)
                else:
                    ctrl = compile_buckets(prompts)
                    mode_id = ctrl["mode_id"][0].item()
                    spec_id = ctrl["spectrum_id"][0].item() if variant == "full" else None
                    pred_p, _, p_ext = model.forward(mag, phase, control_matrix={"mode_id": mode_id, "spectrum_id": spec_id})
                zf = p_ext["phase_state"]
                h_e, i_e = rmt.harmonic_inharmonic_energy(zf, [tuple(x) for x in cfg['training']['rational_ratios']])
                actual = i_e / (h_e + i_e).clamp_min(1e-8)
                if actual.shape != gt_tension.shape:
                    actual = F.interpolate(actual.unsqueeze(1), size=gt_tension.size(1), mode="linear", align_corners=True).squeeze(1)
                c = np.corrcoef(actual[0].float().cpu().numpy(), gt_tension[0].cpu().numpy())[0, 1]
                if not np.isnan(c): val_corrs.append(c)
        
        t_c, v_c = np.mean(train_corrs), np.mean(val_corrs)
        if ep % 10 == 0:
            print(f"[{variant.upper()}] EP {ep} | TRAIN: {t_c:.4f} | VAL: {v_c:.4f}")
        with open(log_path, "a") as f: f.write(f"{ep},{t_c},{v_c}\n")

if __name__ == "__main__":
    variant = sys.argv[1] if len(sys.argv) > 1 else "full"
    run_ablation(variant)
