import torch
import torch.nn.functional as F
import numpy as np
import sys
import os
import wave
from lib_blackwell import NakedRAFA, NakedDenoiser
from dataset import RafaGuerrillaDataset
from config import load_config
from diffusion_utils import phase_to_phasor, q_sample_x0, make_beta_schedule, phasor_to_phase
import stft_utils

def save_wav(mag, phase, path, stft_cfg, sr=16000):
    # Inverse STFT to save the audio
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

def main():
    cfg = load_config()
    dev = torch.device("cuda")
    stft_cfg = cfg["data"]["stft"]
    
    # 1. Initialize and Load Target Checkpoint
    rafa_core = NakedRAFA(dev=dev)
    denoiser = NakedDenoiser(dev=dev)
    
    # USE LATEST EP 9 HARDENED WEIGHTS
    ckpt_path = "checkpoints_stage4/bound_weights_ep9_step900.pt" 
    if not os.path.exists(ckpt_path):
        print(f"ERROR: Checkpoint {ckpt_path} not found.")
        return
        
    print(f"DEBUG: Rendering comparison from {ckpt_path}...")
    ckpt = torch.load(ckpt_path, map_location=dev)
    rafa_core.weights = ckpt['core']
    denoiser.weights = ckpt['denoiser']

    # 2. Grab a Sample
    train_ds = RafaGuerrillaDataset(cfg, split="training")
    mag, phase, _, _, _ = train_ds[42]
    mag, phase = mag.to(dev).unsqueeze(0), phase.to(dev).unsqueeze(0)
    
    x0_mag = torch.log1p(mag)
    x0_z = phase_to_phasor(phase)
    
    timesteps = 50
    betas = make_beta_schedule(timesteps, 1e-4, 0.02, dev)
    alphas_cumprod = torch.cumprod(1.0 - betas, dim=0)
    t = torch.ones(1, dtype=torch.long, device=dev) * 25
    xt_mag, xt_z, _, _ = q_sample_x0(x0_mag, x0_z, t, alphas_cumprod)

    # 3. RENDER THREE CONDITIONS
    os.makedirs("outputs/binding_audit", exist_ok=True)
    
    with torch.no_grad():
        # A: Normal (Matched)
        p_p, p_m, _ = rafa_core.forward(mag, phase)
        d_mag_a, d_z_a = denoiser.forward(xt_mag, xt_z, t, rafa_mag=p_m, rafa_z=torch.stack([torch.cos(p_p), torch.sin(p_p)], -1))
        save_wav(torch.expm1(d_mag_a[0]), phasor_to_phase(d_z_a)[0], "outputs/binding_audit/matched.wav", stft_cfg)
        
        # B: Shuffled (Causal mismatch)
        mag_other, phase_other, _, _, _ = train_ds[0]
        mag_other, phase_other = mag_other.to(dev).unsqueeze(0), phase_other.to(dev).unsqueeze(0)
        p_p_shuf, p_m_shuf, _ = rafa_core.forward(mag_other, phase_other)
        d_mag_b, d_z_b = denoiser.forward(xt_mag, xt_z, t, rafa_mag=p_m_shuf, rafa_z=torch.stack([torch.cos(p_p_shuf), torch.sin(p_p_shuf)], -1))
        save_wav(torch.expm1(d_mag_b[0]), phasor_to_phase(d_z_b)[0], "outputs/binding_audit/shuffled.wav", stft_cfg)
        
        # C: Zeroed (Brain silenced)
        d_mag_c, d_z_c = denoiser.forward(xt_mag, xt_z, t, rafa_mag=None, rafa_z=None)
        save_wav(torch.expm1(d_mag_c[0]), phasor_to_phase(d_z_c)[0], "outputs/binding_audit/zeroed.wav", stft_cfg)

    print("\n" + "="*60)
    print("ACOUSTIC AUDIT COMPLETE")
    print("="*60)
    print("Files generated in outputs/binding_audit/:")
    print("- matched.wav")
    print("- shuffled.wav")
    print("- zeroed.wav")
    print("="*60)

if __name__ == "__main__":
    main()
