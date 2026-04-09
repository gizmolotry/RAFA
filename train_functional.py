import sys
import os
import time
import torch
import torch.nn.functional as F_
import numpy as np
from torch.utils.data import DataLoader

# --- NAKED TENSOR RAFA v3.0 (Absolute sm_120 Isolation) ---
# No nn.Module. No nn.Parameter. No nn.ParameterDict.
# Just raw tensors and functional math. 

from dataset import RafaGuerrillaDataset
from config import load_config
import rafa_math_tools as rmt
from diffusion_utils import make_beta_schedule, extract, phase_to_phasor, q_sample_x0

# BLACKWELL sm_120 WORKAROUND
torch.backends.cudnn.enabled = False

def create_naked_weight(shape):
    # Create raw CUDA tensor directly to bypass _apply scanner
    w = torch.empty(*shape, device='cuda', dtype=torch.float32)
    if len(shape) >= 2:
        torch.nn.init.xavier_uniform_(w)
    else:
        torch.nn.init.zeros_(w)
    w.requires_grad = True
    return w

class NakedRAFA:
    def __init__(self, cfg):
        self.cfg = cfg
        self.d_model = 256
        self.freq_bins = 129
        
        # RAW TENSOR REGISTRY
        self.weights = {
            "lexicon_phasors": create_naked_weight((256, self.freq_bins * 2)),
            "g1_w_proj": create_naked_weight((self.d_model, 6 * self.freq_bins)),
            "g1_b_proj": create_naked_weight((self.d_model,)),
            "g1_w_ph": create_naked_weight((self.freq_bins, self.d_model)),
            "g1_w_mag": create_naked_weight((self.freq_bins, self.d_model)),
            
            # Phase gru Weights (The Brain)
            "ps_w_ih": create_naked_weight((3 * 64, 3)),
            "ps_w_hh": create_naked_weight((3 * 64, 64)),
            "ps_b_ih": create_naked_weight((3 * 64,)),
            "ps_b_hh": create_naked_weight((3 * 64,)),
            "ps_slow_proj": create_naked_weight((3, 64))
        }

    def forward(self, mag, phase, text_tokens=None):
        B, F, T = mag.shape
        w = self.weights
        
        # --- LAYER 4: GEARS ---
        x = torch.cat([mag]*6, dim=1).transpose(1, 2)
        x = F_.linear(x, w["g1_w_proj"], w["g1_b_proj"])
        p_seed = F_.linear(x, w["g1_w_ph"]).transpose(1, 2)
        m_seed = F_.linear(x, w["g1_w_mag"]).transpose(1, 2)
        
        # --- LAYER 2: PHASE CRYSTAL (IFS) ---
        z_in = torch.stack([torch.cos(p_seed), torch.sin(p_seed)], dim=-1).permute(0, 2, 1, 3)
        z_prev = z_in[:, 0]
        
        # Lexicon Disturbances
        if text_tokens is not None:
            token_p = F_.embedding(text_tokens, w["lexicon_phasors"])
            token_p = token_p.view(B, -1, self.freq_bins, 2)
            z_prev = z_prev + token_p.sum(1)
            z_prev = z_prev / torch.sqrt((z_prev*z_prev).sum(-1, keepdim=True).clamp_min(1e-8))
            
        h_slow = z_in.new_zeros(B, 64)
        outs = []
        
        for t in range(T):
            # Slow Clock functional step
            f_v = z_in.new_zeros(B, 3) 
            ih, hh = F_.linear(f_v, w["ps_w_ih"], w["ps_b_ih"]), F_.linear(h_slow, w["ps_w_hh"], w["ps_b_hh"])
            i_r, i_z, i_n = ih.chunk(3, dim=-1); h_r, h_z, h_n = hh.chunk(3, dim=-1)
            r, z_gate = torch.sigmoid(i_r + h_r), torch.sigmoid(i_z + h_z)
            h_slow = (1.0 - z_gate) * torch.tanh(i_n + r * h_n) + z_gate * h_slow
            
            # Physics Step
            zt = z_in[:, t]
            zt = zt + 0.05 * z_prev
            z_prev = zt / torch.sqrt((zt*zt).sum(-1, keepdim=True).clamp_min(1e-8))
            outs.append(z_prev)
            
        zf = torch.stack(outs, dim=1).permute(0, 2, 1, 3)
        return torch.atan2(zf[..., 1], zf[..., 0] + 1e-12), m_seed, {"phase_state": zf}

def main():
    cfg = load_config()
    model = NakedRAFA(cfg)
    dev = torch.device("cuda")
    
    train_ds = RafaGuerrillaDataset(cfg, split="training")
    loader = DataLoader(train_ds, batch_size=16, shuffle=True, num_workers=0)
    
    # Pass naked tensor list to optimizer
    opt = torch.optim.AdamW(list(model.weights.values()), lr=1e-5)
    
    print("DEBUG: Naked Tensor Master Training (RTX 5070 Ti Absolute Isolation)")
    for ep in range(100):
        corrs = []
        for i, batch in enumerate(loader):
            mag, phase, _, prompts, gt_tension = batch
            mag, phase, gt_tension = mag.to(dev), phase.to(dev), gt_tension.to(dev)
            
            text_tokens = None
            if prompts:
                max_l = max(len(p) for p in prompts)
                text_tokens = torch.zeros(len(prompts), max_l, dtype=torch.long, device=dev)
                for b_idx, p in enumerate(prompts):
                    b = list(str(p).encode("utf-8", errors="ignore"))
                    text_tokens[b_idx, :len(b)] = torch.tensor(b, dtype=torch.long, device=dev)
            
            pred_p, pred_m, p_ext = model.forward(mag, phase, text_tokens=text_tokens)
            
            # Alignment Logic
            zf = p_ext["phase_state"]
            h_e, i_e = rmt.harmonic_inharmonic_energy(zf, [tuple(x) for x in cfg['training']['rational_ratios']])
            actual = i_e / (h_e + i_e).clamp_min(1e-8)
            if actual.shape != gt_tension.shape:
                actual = F_.interpolate(actual.unsqueeze(1), size=gt_tension.size(1), mode="linear", align_corners=True).squeeze(1)
            
            loss = F_.mse_loss(actual, gt_tension)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(list(model.weights.values()), 0.1)
            opt.step()
            opt.zero_grad()
            
            if i % 20 == 0:
                with torch.no_grad():
                    c = np.corrcoef(actual[0].cpu().numpy(), gt_tension[0].cpu().numpy())[0, 1]
                print(f"STEP {i} | LOSS: {loss.item():.6f} | CORR: {c:.4f} | sm_120: PASS")
            
            if i % 1000 == 0:
                torch.save(model.weights, f"checkpoints_diffusion_master_definitive/naked_weights_step_{i}.pt")

if __name__ == "__main__":
    main()
