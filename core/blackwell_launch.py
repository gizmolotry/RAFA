import sys
import os

# TOTAL ECLIPSE: Clear all inherited environment pollution
for k in list(os.environ.keys()):
    if k.startswith("CONDA") or k.startswith("PYTHON"):
        del os.environ[k]

# Lock path to Sterile sm_120 Env only
STABLE_ENV = r"C:\Users\Andrew\miniconda3\envs\RAFA_SM120_STABLE"
sys.path = [
    os.path.join(STABLE_ENV, "python310.zip"),
    os.path.join(STABLE_ENV, "DLLs"),
    os.path.join(STABLE_ENV, "lib"),
    STABLE_ENV,
    os.path.join(STABLE_ENV, "lib", "site-packages"),
    r"D:\RAFA" # Local project
]

print(f"DEBUG: TOTAL ECLIPSE Active. Locked Path: {sys.path}")

# Now import established Blackwell logic
import torch
import torch.nn as nn
from blackwell_model import RAFA
from diffusion_models import RAFADenoiser
from dataset import RafaGuerrillaDataset
from config import load_config
from torch.utils.data import DataLoader
import rafa_math_tools as rmt
from diffusion_utils import make_beta_schedule, extract, phase_to_phasor, q_sample_x0

# BLACKWELL WORKAROUND: Force native rnns
torch.backends.cudnn.enabled = False

def train():
    cfg = load_config()
    dev = torch.device("cuda")
    
    print(f"DEBUG: Initializing Sterile RAFA on Blackwell (sm_120)")
    model = RAFADenoiser(cfg).to(dev)
    
    # Verify we are on sm_120 and using the sterile torch
    print(f"DEBUG: Torch Version: {torch.__version__}")
    print(f"DEBUG: Device Capability: {torch.cuda.get_device_capability(0)}")
    
    train_ds = RafaGuerrillaDataset(cfg, split="training")
    loader = DataLoader(train_ds, batch_size=8, shuffle=True, num_workers=0)
    
    opt = torch.optim.AdamW(model.parameters(), lr=1e-5)
    timesteps = 50
    betas = make_beta_schedule(timesteps, 1e-4, 0.02, dev)
    alphas_cumprod = torch.cumprod(1.0 - betas, dim=0)
    
    print("DEBUG: Training loop start (sm_120 Native)")
    for i, batch in enumerate(loader):
        mag, phase, _, _, gt_tension = batch
        mag, phase = mag.to(dev), phase.to(dev)
        if gt_tension is not None: gt_tension = gt_tension.to(dev)
        
        x0_mag = torch.log1p(mag)
        x0_z = phase_to_phasor(phase)
        t = torch.randint(0, timesteps, (mag.size(0),), device=dev)
        xt_mag, xt_z, eps_mag, eps_z = q_sample_x0(x0_mag, x0_z, t, alphas_cumprod)
        
        pred_mag, pred_z, p_ext, _ = model(xt_mag, xt_z, t, semantic_controls={"semantic_tension": gt_tension} if gt_tension is not None else None)
        
        # Simple Relational Loss
        zf = p_ext["phase_state"]
        h_e, i_e = rmt.harmonic_inharmonic_energy(zf, [tuple(x) for x in cfg['training']['rational_ratios']])
        actual = i_e / (h_e + i_e).clamp_min(1e-8)
        loss = torch.nn.functional.mse_loss(actual, gt_tension)
        
        loss.backward()
        opt.step()
        opt.zero_grad()
        
        if i % 5 == 0:
            print(f"STEP {i} | LOSS: {loss.item():.6f}")
        if i >= 100: break

if __name__ == "__main__":
    train()
