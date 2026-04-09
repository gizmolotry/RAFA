import torch
import torch.nn.functional as F
import numpy as np
import os
import wave
from lib_blackwell import NakedRAFA, NakedDenoiser
from config import load_config
from diffusion_utils import make_beta_schedule, phasor_to_phase
import stft_utils

def save_wav(mag, phase, path, stft_cfg, sr=16000):
    m = torch.as_tensor(mag).cpu()
    p = torch.as_tensor(phase).cpu()
    if m.ndim == 2: m = m.unsqueeze(0)
    # Inverse STFT
    wav = stft_utils.inverse_stft(m, p, stft_cfg)
    wav = wav.squeeze(0).cpu().numpy()
    # Peak normalize
    wav = wav / (np.abs(wav).max() + 1e-8)
    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes((wav * 32767).astype(np.int16).tobytes())

def compile_prompt(text):
    text = text.lower()
    m_id = 0
    if "engine" in text or "airplane" in text or "jet" in text: m_id = 1
    elif "voice" in text or "singing" in text: m_id = 2
    elif "impact" in text or "hit" in text: m_id = 3
    elif "drone" in text or "hum" in text: m_id = 4
    elif "piano" in text or "bell" in text or "note" in text: m_id = 5
    
    s_id = 1 
    if "bright" in text or "high" in text or "piano" in text: s_id = 2
    elif "dark" in text or "low" in text: s_id = 0
    elif "broadband" in text or "noise" in text or "airplane" in text: s_id = 3
    
    return {"mode_id": m_id, "spectrum_id": s_id}

def run_inference(prompt_text, out_path):
    cfg = load_config()
    dev = torch.device("cuda")
    stft_cfg = cfg["data"]["stft"]
    
    # 1. Load the FINAL Graduation Weights
    rafa_core = NakedRAFA(d_model=768, dev=dev)
    denoiser = NakedDenoiser(dev=dev)
    ckpt_path = "checkpoints_stage4/bound_weights_ep99_step200.pt"
    print(f"DEBUG: Loading Final Production Weights from {ckpt_path}...")
    ckpt = torch.load(ckpt_path, map_location=dev, weights_only=False)
    rafa_core.weights = ckpt['core']
    denoiser.weights = ckpt['denoiser']
    rafa_core.qset = rafa_core.qset.to(dev); rafa_core.qw = rafa_core.qw.to(dev)

    # 2. Compile Intent
    ctrl = compile_prompt(prompt_text)
    print(f"COMPILING: '{prompt_text}' -> Mode:{ctrl['mode_id']}, Spec:{ctrl['spectrum_id']}")

    # 3. Generate from Zero-State
    B, F, T = 1, 129, 251
    mag_init = torch.zeros(B, F, T, device=dev)
    phase_init = torch.zeros(B, F, T, device=dev)
    
    timesteps = 50
    betas = make_beta_schedule(timesteps, 1e-4, 0.02, dev)
    alphas_cumprod = torch.cumprod(1.0 - betas, dim=0)
    t = torch.ones(1, dtype=torch.long, device=dev) * 25 
    
    # Mid-diffusion noise
    xt_mag = torch.randn(B, F, T, device=dev) * 0.1
    xt_z = torch.randn(B, F, T, 2, device=dev)
    xt_z = xt_z / torch.sqrt((xt_z**2).sum(-1, keepdim=True) + 1e-8)

    with torch.no_grad():
        # Brain hallucinates the physical skeleton
        p_p, p_m, _ = rafa_core.forward(mag_init, phase_init, control_matrix=ctrl)
        
        # Voice renders the audio through the TRAINED adapter
        d_mag, d_z = denoiser.forward(xt_mag, xt_z, t, rafa_mag=p_m, rafa_z=torch.stack([torch.cos(p_p), torch.sin(p_p)], -1))
        
        # 4. HIGH-FIDELITY INVERSE SCALING
        # Reverse log1p(mag * 10.0)
        final_mag = (torch.expm1(d_mag[0]) / 10.0)
        final_phase = phasor_to_phase(d_z)[0]
        
        save_wav(final_mag, final_phase, out_path, stft_cfg)
        print(f"SUCCESS: Generated {out_path}")

if __name__ == "__main__":
    os.makedirs("outputs/inference", exist_ok=True)
    run_inference("airplane jet noise broadband", "outputs/inference/airplane_hardened.wav")
    run_inference("play a bright piano note", "outputs/inference/piano_hardened.wav")
