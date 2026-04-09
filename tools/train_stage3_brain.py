import os
import sys
import time
import torch
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader
from dataset import RafaGuerrillaDataset
from config import load_config
from lib_blackwell import NakedRAFA
import rafa_math_tools as rmt
from tools.invariance_utils import apply_acoustic_violence, calculate_invariance_loss

# STAGE 3: BRAIN BOOT CAMP (Auto-Formalization Edition)
# Goal: Use structured control fields instead of text embeddings.

def pearson_correlation_loss(pred, target, eps=1e-8):
    pred_mean = pred - pred.mean(dim=-1, keepdim=True)
    target_mean = target - target.mean(dim=-1, keepdim=True)
    cov = (pred_mean * target_mean).sum(dim=-1)
    pred_var = (pred_mean ** 2).sum(dim=-1)
    target_var = (target_mean ** 2).sum(dim=-1)
    corr = cov / torch.sqrt((pred_var * target_var).clamp_min(eps))
    return (1.0 - corr).mean()

def compile_formal_prompt(prompts):
    """
    AUTO-FORMALIZER: Maps text to the 5-Bucket Ontology.
    """
    modes = {"engine": 1, "voice": 2, "impact": 3, "drone": 4, "bell": 5}
    batch_modes = []
    for p in prompts:
        p_lower = str(p).lower()
        found = False
        for k, v in modes.items():
            if k in p_lower:
                batch_modes.append(v)
                found = True
                break
        if not found: batch_modes.append(0) # Default/Noise mode
    return {"mode_id": torch.tensor(batch_modes, dtype=torch.long)}

def main():
    cfg = load_config()
    dev = torch.device("cuda")
    model = NakedRAFA(dev=dev)
    
    # Brain parameters only
    params = [p for n, p in model.weights.items() if "ps_" in n or "mode_seeds" in n]
    opt = torch.optim.AdamW(params, lr=1e-4)
    
    # Setup Data (Reduced Batch for Parallel sm_120 Stability)
    train_ds = RafaGuerrillaDataset(cfg, split="training")
    loader = DataLoader(train_ds, batch_size=4, shuffle=True, num_workers=0)
    
    print("DEBUG: Grabbing a single 16-sample batch for the FORMALIZED Overfit...")
    single_batch = next(iter(loader))
    
    print("STAGE 3 START: Brain Boot Camp (Auto-Formalization Active)")
    for ep in range(1000):
        corrs = []
        mag_raw, phase_raw, _, prompts, gt_tension_raw = single_batch
        mag, phase, gt_tension = mag_raw.to(dev), phase_raw.to(dev), gt_tension_raw.to(dev)
        
        # 1. AUTO-FORMALIZE PROMPTS
        ctrl = compile_formal_prompt(prompts)
        ctrl["mode_id"] = ctrl["mode_id"].to(dev)
        
        for i in range(10):
            # Forward with Control Matrix
            mode_id = ctrl["mode_id"][0].item()
            pred_p, pred_m, p_ext = model.forward(mag, phase, control_matrix={"mode_id": mode_id})
            torch.cuda.synchronize()
            
            zf = p_ext["phase_state"]
            h_e, i_e = rmt.harmonic_inharmonic_energy(zf, [tuple(x) for x in cfg['training']['rational_ratios']])
            actual = i_e / (h_e + i_e).clamp_min(1e-8)
            
            if actual.shape != gt_tension.shape:
                actual = F.interpolate(actual.unsqueeze(1), size=gt_tension.size(1), mode="linear", align_corners=True).squeeze(1)
            
            # PURE CORRELATION LOSS
            loss = pearson_correlation_loss(actual, gt_tension)
            
            loss.backward()
            torch.cuda.synchronize()
            opt.step()
            opt.zero_grad()
            
            with torch.no_grad():
                for b_idx in range(actual.size(0)):
                    c = np.corrcoef(actual[b_idx].cpu().numpy(), gt_tension[b_idx].cpu().numpy())[0, 1]
                    if not np.isnan(c): corrs.append(c)

        if ep % 5 == 0:
            avg_c = np.mean(corrs) if corrs else 0.0
            print(f"EP {ep} | FORMAL CORR: {avg_c:.6f} | WIN: {avg_c > 0.99}")
            
if __name__ == "__main__":
    main()
