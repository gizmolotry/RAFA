# stft_utils.py
import torch
import math

def _hann(win_length, device):
    return torch.hann_window(win_length, periodic=True, dtype=torch.float32, device=device)

@torch.no_grad()
def compute_stft(waveform, stft_cfg):
    """
    waveform: (B, T) or (T,) mono
    stft_cfg: dict with keys n_fft, hop, win_length
    returns: (mag, phase) each (B, F, T_frames)
    """
    if waveform.ndim == 1:
        waveform = waveform.unsqueeze(0)
    B, T = waveform.shape
    n_fft = int(stft_cfg["n_fft"])
    hop   = int(stft_cfg["hop"])
    winl  = int(stft_cfg["win_length"])
    device = waveform.device

    window = _hann(winl, device)
    Z = torch.stft(
        waveform,
        n_fft=n_fft,
        hop_length=hop,
        win_length=winl,
        window=window,
        center=True,
        return_complex=True,
        pad_mode="reflect",
    )  # (B, F, frames) complex
    mag   = Z.abs()
    phase = torch.angle(Z)  # (-pi, pi]
    return mag, phase

@torch.no_grad()
def inverse_stft(mag, phase, stft_cfg):
    """
    mag, phase: (B, F, T_frames)
    returns: waveform (B, T)
    """
    # Combine back to complex
    Z = torch.polar(torch.as_tensor(mag), torch.as_tensor(phase))
    
    n_fft = int(stft_cfg["n_fft"])
    hop   = int(stft_cfg["hop"])
    winl  = int(stft_cfg["win_length"])
    device = Z.device

    window = _hann(winl, device)
    wav = torch.istft(
        Z,
        n_fft=n_fft,
        hop_length=hop,
        win_length=winl,
        window=window,
        center=True
    )
    return wav
