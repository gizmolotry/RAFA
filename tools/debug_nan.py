
import torch
from model import RAFA
from config import load_config
import torch.nn.functional as F
import math

def circular_mse(pred, tgt):
    return torch.mean(1.0 - torch.cos(pred - tgt))

def compute_delta_phase(phs):
    td = phs[..., 1:] - phs[..., :-1]
    td = ((td + math.pi) % (2.0 * math.pi)) - math.pi
    return torch.cat([torch.zeros_like(td[..., :1]), td], dim=-1)

import rafa_math_tools as rmt

def debug_nan():
    cfg = load_config()
    dev = torch.device("cuda")
    model = RAFA(cfg).to(dev)
    
    # Mock data
    freq_bins = cfg["data"]["stft"]["n_fft"] // 2 + 1
    t_frames = 128
    mags = torch.randn(1, freq_bins, t_frames, device=dev).abs() + 0.1
    phs = torch.randn(1, freq_bins, t_frames, device=dev)
    td = compute_delta_phase(phs)
    
    opt = torch.optim.Adam(model.parameters(), lr=1e-4)
    
    torch.autograd.set_detect_anomaly(True)
    
    rat_ratios = [tuple(x) for x in cfg["training"]["rational_ratios"]]
    
    # Enable learned freqs for testing
    init_freqs = torch.linspace(0.01, torch.pi, freq_bins, device=dev)
    model.learned_freqs = torch.nn.Parameter(init_freqs)

    for i in range(20):
        opt.zero_grad()
        pd, md, p_ext, m_ext = model(mags, phs)
        
        l_phase = circular_mse(pd, td)
        l_mag = F.mse_loss(md, mags)
        
        d_pd = compute_delta_phase(pd)
        d_td = compute_delta_phase(td)
        l_kin_smooth = F.mse_loss(d_pd, d_td)
        
        l_r2 = rmt.modular_phase_embed_loss(pd, td, qs=(3, 4, 5))
        
        kappa = pd.new_tensor(1.0)
        l_r5 = rmt.von_mises_nll(pd, td, kappa).mean()
        
        harmonic_input = pd.mean(dim=0) # simplified projection
        l_harm = rmt.harmonic_coupling_loss(harmonic_input, rat_ratios)
        
        l_cry = rmt.ramanujan_crystal_loss(pd[0], qs=(2,3,4,5))
        
        l_rat = rmt.soft_rational_prior(model.learned_freqs, rat_ratios)
        
        l_clutch_iso = torch.zeros((), device=dev)
        if "phase_gates" in p_ext:
            gates = p_ext["phase_gates"]
            l_clutch_iso = F.mse_loss(gates[:, 0], gates.mean(dim=1).detach()) # dummy iso loss
            
        loss = l_phase + l_mag + 0.2 * l_kin_smooth + 0.1 * l_r2 + 0.1 * l_r5 + 0.05 * l_harm + 0.05 * l_cry + 0.05 * l_rat + 0.1 * l_clutch_iso
        
        try:
            loss.backward()
            opt.step()
            print(f"Step {i} OK")
        except Exception as e:
            print(f"Step {i} FAILED: {e}")
            break

if __name__ == "__main__":
    debug_nan()
