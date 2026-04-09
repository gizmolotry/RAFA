import torch
import torch.nn.functional as F
import numpy as np

def apply_acoustic_violence(mag, phase, severity=0.5):
    """
    Perturbs local timbre/texture while preserving macro-structure.
    Used for the Invariance Curriculum.
    """
    dev = mag.device
    B, F_bins, T = mag.shape
    
    # 1. Phase Shuffling (Destroy local phase-coherence/timbre)
    # We add random noise to the phase increment
    noise = torch.randn_like(phase) * severity * torch.pi
    perturbed_phase = (phase + noise) % (2 * torch.pi)
    
    # 2. Spectral Jitter (Destroy precise harmonic peaks)
    # We apply a random local smoothing/blur to the magnitude
    # This keeps the macro-energy (tension) but smears the timbre
    kernel_size = int(3 + 4 * severity)
    if kernel_size % 2 == 0: kernel_size += 1
    
    mag_pad = F.pad(mag, (0, 0, kernel_size//2, kernel_size//2), mode='reflect')
    # Use mean pool as a simple blur
    perturbed_mag = F.avg_pool2d(mag_pad.unsqueeze(1), (kernel_size, 1), stride=1).squeeze(1)
    
    # 3. Energy Normalization
    # Ensure the tension (spectral flatness) remains roughly in the same ballpark
    perturbed_mag = perturbed_mag * (mag.mean() / perturbed_mag.mean().clamp_min(1e-8))
    
    return perturbed_mag, perturbed_phase

def calculate_invariance_loss(h_slow_clean, h_slow_distorted):
    """
    Penalizes the Brain if its 'thought' changes under acoustic violence.
    h_slow: [B, T, Hidden]
    """
    # Simple MSE or Cosine Similarity on the slow latent trajectory
    return F.mse_loss(h_slow_clean, h_slow_distorted)
