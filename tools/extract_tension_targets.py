import os
import glob
import torch
import torchaudio
import numpy as np
from tqdm import tqdm
import wave

# --- TARGET 4 PHYSICS CALIBRATION ---
SAMPLE_RATE = 16000
N_FFT = 256
HOP_LENGTH = 128
WIN_LENGTH = 256

def _load_wav_native(path: str) -> tuple[torch.Tensor, int]:
    """Robust fallback loader using native wave module."""
    with wave.open(path, "rb") as wf:
        sr = wf.getframerate()
        ch = wf.getnchannels()
        sw = wf.getsampwidth()
        raw = wf.readframes(wf.getnframes())
        
    if sw == 1:
        arr = np.frombuffer(raw, dtype=np.uint8).astype(np.float32)
        arr = (arr - 128.0) / 128.0
    elif sw == 2:
        arr = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    elif sw == 4:
        arr = np.frombuffer(raw, dtype=np.int32).astype(np.float32) / 2147483648.0
    else:
        raise RuntimeError(f"Unsupported bit depth: {sw}")
        
    arr = arr.reshape(-1, ch).T
    return torch.from_numpy(arr), sr

def calculate_tension_envelope(waveform: torch.Tensor) -> torch.Tensor:
    """
    Calculates a 1D Tension (Inharmonicity) envelope using Spectral Flatness.
    Returns a tensor of shape [1, Time_Frames] bounded between [0, 1].
    """
    # 1. Push to STFT magnitude
    window = torch.hann_window(WIN_LENGTH).to(waveform.device)
    stft = torch.stft(
        waveform, 
        n_fft=N_FFT, 
        hop_length=HOP_LENGTH, 
        win_length=WIN_LENGTH, 
        window=window, 
        return_complex=True
    )
    mag = torch.abs(stft) # Shape: [1, Freq_Bins, Time_Frames]
    
    # 2. Calculate Spectral Flatness (Proxy for Dissonance/Inharmonicity)
    # Flatness = geometric_mean(power) / arithmetic_mean(power)
    power = mag ** 2
    # Add small epsilon to prevent log(0)
    eps = 1e-8
    
    arithmetic_mean = torch.mean(power, dim=1, keepdim=True)
    geometric_mean = torch.exp(torch.mean(torch.log(power + eps), dim=1, keepdim=True))
    
    flatness = geometric_mean / (arithmetic_mean + eps)
    
    # 3. Normalize to [0, 1] range
    flatness = flatness.squeeze(1) # Shape: [1, Time_Frames]
    flat_min, flat_max = flatness.min(), flatness.max()
    if flat_max > flat_min:
        tension_envelope = (flatness - flat_min) / (flat_max - flat_min)
    else:
        tension_envelope = flatness # Fallback if perfectly flat (silence)
        
    return tension_envelope

def process_dataset(data_dir: str):
    """Loops over all .wav files and saves parallel _tension.npy files."""
    wav_files = glob.glob(os.path.join(data_dir, "**/*.wav"), recursive=True)
    print(f"Found {len(wav_files)} audio files. Extracting Tension Targets...")
    
    # Create output dir if it doesn't exist
    out_root = "datasets/tension_tracks"
    os.makedirs(out_root, exist_ok=True)
    
    for wav_path in tqdm(wav_files):
        try:
            # Load audio with robust fallback
            try:
                wav, sr = torchaudio.load(wav_path)
            except Exception:
                wav, sr = _load_wav_native(wav_path)
            
            # Resample if necessary
            if sr != SAMPLE_RATE:
                wav = torchaudio.functional.resample(wav, sr, SAMPLE_RATE)
            
            # Convert stereo to mono if needed
            if wav.shape[0] > 1:
                wav = torch.mean(wav, dim=0, keepdim=True)
                
            # Calculate the 1D Target
            tension_track = calculate_tension_envelope(wav)
            
            # Save as .npy for fast loading in the established dataset.py
            rel_path = os.path.relpath(wav_path, data_dir)
            out_path = os.path.join(out_root, rel_path).replace(".wav", ".npy")
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            
            np.save(out_path, tension_track.squeeze(0).numpy().astype(np.float16))
            
        except Exception as e:
            print(f"Failed processing {wav_path}: {e}")

if __name__ == "__main__":
    # POINTED TO YOUR VERIFIED WAV DIRECTORY
    DATASET_DIRECTORY = "D:/RAFA/wav_files" 
    process_dataset(DATASET_DIRECTORY)
    print("DSP Labeling Complete. The Teacher is ready.")
