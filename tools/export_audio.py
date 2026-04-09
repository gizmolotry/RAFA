import torch
import torch.nn.functional as F
import numpy as np
import os
import wave
try:
    from lib_blackwell import NakedRAFA, NakedDenoiser
except ImportError:
    from core.lib_blackwell import NakedRAFA, NakedDenoiser
try:
    from dataset import RafaGuerrillaDataset
except ImportError:
    from core.dataset import RafaGuerrillaDataset
from config import load_config
from diffusion_utils import phase_to_phasor, q_sample_x0, make_beta_schedule, phasor_to_phase
import stft_utils

def save_wav(mag, phase, path, stft_cfg, sr=16000):
    m = torch.as_tensor(mag).cpu()
    p = torch.as_tensor(phase).cpu()
    if m.ndim == 2: m = m.unsqueeze(0)
    if p.ndim == 2: p = p.unsqueeze(0)
    wav = stft_utils.inverse_stft(m, p, stft_cfg)
    wav = wav.squeeze(0).cpu().numpy()
    wav = wav / (np.abs(wav).max() + 1e-8)
    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes((wav * 32767).astype(np.int16).tobytes())

def render_graduation_suite(ckpt_path, out_dir):
    cfg = load_config()
    dev = torch.device("cuda")
    stft_cfg = cfg["data"]["stft"]
    os.makedirs(out_dir, exist_ok=True)
    
    rafa_core = NakedRAFA(d_model=768, dev=dev)
    denoiser = NakedDenoiser(dev=dev)
    
    print(f"DEBUG: Loading Graduation Checkpoint from {ckpt_path}...")
    ckpt = torch.load(ckpt_path, map_location=dev, weights_only=False)
    rafa_core.weights = ckpt['core']
    denoiser.weights = ckpt['denoiser']
    rafa_core.qset = rafa_core.qset.to(dev); rafa_core.qw = rafa_core.qw.to(dev)

    ds = RafaGuerrillaDataset(cfg, split="training")
    
    # Graduation Suite: [SampleIdx, ModeName, ModeID]
    suite = [
        (42, "engine", 1),
        (100, "voice", 2),
        (0, "impact", 3),
        (200, "drone", 4)
    ]
    
    timesteps = 50
    betas = make_beta_schedule(timesteps, 1e-4, 0.02, dev)
    alphas_cumprod = torch.cumprod(1.0 - betas, dim=0)
    t = torch.ones(1, dtype=torch.long, device=dev) * 25 # High-fidelity mid-diffusion

    with torch.no_grad():
        for idx, name, m_id in suite:
            mag, phase, _, _, _ = ds[idx]
            mag, phase = mag.to(dev).unsqueeze(0), phase.to(dev).unsqueeze(0)
            x0_mag, x0_z = torch.log1p(mag), phase_to_phasor(phase)
            xt_mag, xt_z, _, _ = q_sample_x0(x0_mag, x0_z, t, alphas_cumprod)
            
            # Matched Guidance from the Mind
            p_p, p_m, _ = rafa_core.forward(mag, phase, control_matrix={"mode_id": m_id})
            
            # Rendering through the Mouth
            d_mag, d_z = denoiser.forward(xt_mag, xt_z, t, rafa_mag=p_m, rafa_z=torch.stack([torch.cos(p_p), torch.sin(p_p)], -1))
            
            out_path = f"{out_dir}/grad_{name}.wav"
            save_wav(torch.expm1(d_mag[0]), phasor_to_phase(d_z)[0], out_path, stft_cfg)
            print(f"RENDERED: {out_path}")

if __name__ == "__main__":
    render_graduation_suite("checkpoints_stage4/bound_weights_ep99_step300.pt", "outputs/graduation_pack")
