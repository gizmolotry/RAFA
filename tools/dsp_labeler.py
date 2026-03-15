
import os
import torch
import torchaudio
import numpy as np
from pathlib import Path
from tqdm import tqdm
import yaml
import wave
from stft_utils import compute_stft

def _load_wav_robust(path: Path):
    """Fallback loader using wave module if torchaudio fails."""
    try:
        wav, sr = torchaudio.load(str(path))
        return wav, sr
    except Exception:
        with wave.open(str(path), "rb") as wf:
            sr = int(wf.getframerate())
            ch = int(wf.getnchannels())
            sampwidth = int(wf.getsampwidth())
            nframes = int(wf.getnframes())
            raw = wf.readframes(nframes)

        if sampwidth == 1:
            arr = np.frombuffer(raw, dtype=np.uint8).astype(np.float32)
            arr = (arr - 128.0) / 128.0
        elif sampwidth == 2:
            arr = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
        elif sampwidth == 4:
            arr = np.frombuffer(raw, dtype=np.int32).astype(np.float32) / 2147483648.0
        else:
            raise RuntimeError(f"Unsupported sample width: {sampwidth}")

        arr = arr.reshape(-1, ch).T
        return torch.from_numpy(arr), sr

def _q_energy_trajectory(mag, phase, qset, harmonic_qs, inharmonic_qs):
    # mag/phase: [F, T]
    mag_t = mag.transpose(0, 1)  # [T, F]
    phase_t = phase.transpose(0, 1)  # [T, F]
    z = torch.stack([torch.cos(phase_t), torch.sin(phase_t)], dim=-1)  # [T,F,2]
    zi = z.unsqueeze(2)
    zj = z.unsqueeze(1)
    r_re = zi[..., 0] * zj[..., 0] + zi[..., 1] * zj[..., 1]
    r_im = zi[..., 1] * zj[..., 0] - zi[..., 0] * zj[..., 1]
    theta = torch.atan2(r_im, r_re)
    w = (mag_t.unsqueeze(2) * mag_t.unsqueeze(1)).clamp_min(1e-8)
    per_q = {}
    for q in qset:
        compat = 0.5 * (1.0 + torch.cos(float(q) * theta))
        per_q[int(q)] = (compat * w).sum(dim=(1, 2)) / w.sum(dim=(1, 2)).clamp_min(1e-8)
    harm = torch.stack([v for q, v in per_q.items() if q in harmonic_qs], dim=0).sum(dim=0)
    inharm = torch.stack([v for q, v in per_q.items() if q in inharmonic_qs], dim=0).sum(dim=0)
    return harm, inharm

def main():
    with open("config.yaml", "r") as f:
        cfg = yaml.safe_load(f)
    
    wav_root = Path(cfg["data"].get("local_wav_root", "wav_files"))
    out_root = Path("datasets/tension_tracks")
    out_root.mkdir(parents=True, exist_ok=True)
    
    qset = cfg.get("phase_native_ifs", {}).get("qset", [2, 3, 4, 5, 6, 8, 12])
    harmonic_qs = {2, 3, 4, 6, 8, 12}
    inharmonic_qs = {q for q in qset if q not in harmonic_qs}
    if not inharmonic_qs:
        inharmonic_qs = {5, 7, 11, 13}.intersection(set(qset))

    stft_cfg = cfg["data"]["stft"]
    sr = cfg["data"]["sample_rate"]
    clip_seconds = cfg["data"].get("clip_seconds", 1)
    
    wav_files = list(wav_root.rglob("*.wav"))
    print(f"Found {len(wav_files)} files. Processing tension tracks...")
    
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for wav_p in tqdm(wav_files):
        rel_path = wav_p.relative_to(wav_root)
        out_p = out_root / rel_path.with_suffix(".npy")
        out_p.parent.mkdir(parents=True, exist_ok=True)
        
        if out_p.exists():
            continue
            
        try:
            wav, orig_sr = _load_wav_robust(wav_p)
            if wav.shape[0] > 1:
                wav = wav.mean(dim=0, keepdim=True)
            if orig_sr != sr:
                wav = torchaudio.functional.resample(wav, orig_sr, sr)
            
            # Match clip length
            target_len = sr * clip_seconds
            if wav.shape[1] < target_len:
                wav = torch.nn.functional.pad(wav, (0, target_len - wav.shape[1]))
            else:
                wav = wav[:, :target_len]
                
            mag, phase = compute_stft(wav.to(dev), stft_cfg)
            mag = mag.squeeze(0)
            phase = phase.squeeze(0)
            
            harm, inharm = _q_energy_trajectory(mag, phase, qset, harmonic_qs, inharmonic_qs)
            ratio = inharm / (harm + inharm).clamp_min(1e-8)
            
            # Save as float16 to save space
            np.save(out_p, ratio.cpu().numpy().astype(np.float16))
            
        except Exception as e:
            pass # Silent skip for broken files

if __name__ == "__main__":
    main()
