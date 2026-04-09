import os
import sys
import time
import torch
import torch.nn.functional as F
import numpy as np
import json
from datetime import datetime
from torch.utils.data import DataLoader, Subset
from dataset import RafaGuerrillaDataset
from config import load_config
from lib_blackwell import NakedRAFA
import rafa_math_tools as rmt

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

def run_experiment(variant="full", seed=42, epochs=500):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = f"validation_results/run_{variant}_s{seed}_{timestamp}"
    os.makedirs(run_dir, exist_ok=True)
    os.makedirs(f"{run_dir}/checkpoints", exist_ok=True)
    
    print(f"\n>>> STARTING: {variant.upper()} (Seed {seed})")
    try:
        torch.manual_seed(seed)
        np.random.seed(seed)
        cfg = load_config()
        dev = torch.device("cuda")
        
        model = NakedRAFA(d_model=768, dev=dev)
        for k in model.weights:
            model.weights[k] = model.weights[k].to(dev).detach().requires_grad_(True)
        model.qset = model.qset.to(dev); model.qw = model.qw.to(dev)

        params = [p for n, p in model.weights.items() if any(x in n for x in ["ps_", "mode_seeds", "spectrum_biases", "lexicon_phasors"])]
        opt = torch.optim.AdamW(params, lr=1e-4)
        scaler = torch.amp.GradScaler("cuda")
        
        # 128/32 Split
        ds = RafaGuerrillaDataset(cfg, split="training")
        train_loader = DataLoader(Subset(ds, list(range(0, 128))), batch_size=4, shuffle=True)
        val_loader = DataLoader(Subset(ds, list(range(128, 160))), batch_size=4, shuffle=False)
        
        history = []
        best_val_corr = -1.0
        
        for ep in range(epochs):
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
                        spec_id = ctrl["spectrum_id"][0].item() if variant in ["full", "no_env", "no_tex"] else None
                        pred_p, _, p_ext = model.forward(mag, phase, control_matrix={"mode_id": mode_id, "spectrum_id": spec_id})
                    
                    zf = p_ext["phase_state"]
                    h_e, i_e = rmt.harmonic_inharmonic_energy(zf, [tuple(x) for x in cfg['training']['rational_ratios']])
                    actual = i_e / (h_e + i_e).clamp_min(1e-8)
                    if actual.shape != gt_tension.shape:
                        actual = F.interpolate(actual.unsqueeze(1), size=gt_tension.size(1), mode="linear", align_corners=True).squeeze(1)
                    
                    l_mse = F.mse_loss(actual, gt_tension)
                    l_pear = pearson_correlation_loss(actual, gt_tension)
                    loss = 0.3 * l_mse + 0.7 * l_pear
                
                scaler.scale(loss).backward()
                torch.cuda.synchronize() # Sm_120 Stability
                scaler.step(opt)
                scaler.update()
                opt.zero_grad()
                
                with torch.no_grad():
                    c = np.corrcoef(actual[0].detach().float().cpu().numpy(), gt_tension[0].cpu().numpy())[0, 1]
                    if not np.isnan(c): train_corrs.append(c)

            # VALIDATION
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
                        spec_id = ctrl["spectrum_id"][0].item() if variant in ["full", "no_env", "no_tex"] else None
                        pred_p, _, p_ext = model.forward(mag, phase, control_matrix={"mode_id": mode_id, "spectrum_id": spec_id})
                    
                    zf = p_ext["phase_state"]
                    h_e, i_e = rmt.harmonic_inharmonic_energy(zf, [tuple(x) for x in cfg['training']['rational_ratios']])
                    actual = i_e / (h_e + i_e).clamp_min(1e-8)
                    if actual.shape != gt_tension.shape:
                        actual = F.interpolate(actual.unsqueeze(1), size=gt_tension.size(1), mode="linear", align_corners=True).squeeze(1)
                    
                    c = np.corrcoef(actual[0].detach().float().cpu().numpy(), gt_tension[0].cpu().numpy())[0, 1]
                    if not np.isnan(c): val_corrs.append(c)
            
            t_c, v_c = np.mean(train_corrs), np.mean(val_corrs)
            history.append({"epoch": ep, "train_corr": t_c, "val_corr": v_c})
            
            if v_c > best_val_corr:
                best_val_corr = v_c
                torch.save({'weights': model.weights, 'epoch': ep, 'val_corr': v_c}, f"{run_dir}/checkpoints/best.pt")
                
            if ep % 1 == 0:
                print(f"[{variant.upper()} s{seed}] EP {ep} | TRAIN: {t_c:.4f} | VAL: {v_c:.4f}")

        torch.save({'weights': model.weights, 'epoch': epochs-1, 'val_corr': history[-1]['val_corr']}, f"{run_dir}/checkpoints/final.pt")
        with open(f"{run_dir}/history.csv", "w") as f:
            f.write("epoch,train_corr,val_corr\n")
            for h in history: f.write(f"{h['epoch']},{h['train_corr']},{h['val_corr']}\n")
            
        last50 = [h['val_corr'] for h in history[-50:]]
        stats = {
            "variant": variant, "seed": seed,
            "best_val": best_val_corr,
            "final_val": history[-1]['val_corr'],
            "mean_last50": np.mean(last50),
            "std_last50": np.std(last50),
            "cross_0.2": next((h['epoch'] for h in history if h['val_corr'] >= 0.2), -1),
            "cross_0.3": next((h['epoch'] for h in history if h['val_corr'] >= 0.3), -1),
            "cross_0.4": next((h['epoch'] for h in history if h['val_corr'] >= 0.4), -1)
        }
        with open(f"{run_dir}/stats.json", "w") as f: json.dump(stats, f, indent=4)
        return stats
    except Exception as e:
        print(f"ERROR in run_experiment: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    v = sys.argv[1] if len(sys.argv) > 1 else "full"
    s = int(sys.argv[2]) if len(sys.argv) > 2 else 42
    e = int(sys.argv[3]) if len(sys.argv) > 3 else 500
    run_experiment(v, s, e)
