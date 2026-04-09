"""Inference script to reconstruct audio from a checkpoint and prompt."""

import os
import wave

import numpy as np
import torch
import torchaudio
import yaml

from model import RAFA
from stft_utils import compute_stft


def reconstruct_phase(p0: torch.Tensor, delta: torch.Tensor) -> torch.Tensor:
    bsz, freq_bins, t_len = delta.size()
    ph = delta.new_zeros(bsz, freq_bins, t_len)
    ph[..., 0] = p0
    for t in range(1, t_len):
        ph[..., t] = ph[..., t - 1] + delta[..., t]
    return ph


def _load_wav_fallback(path: str) -> tuple[torch.Tensor, int]:
    with wave.open(path, "rb") as wf:
        sr = int(wf.getframerate())
        ch = int(wf.getnchannels())
        sw = int(wf.getsampwidth())
        n = int(wf.getnframes())
        raw = wf.readframes(n)
    if sw == 1:
        arr = (np.frombuffer(raw, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
    elif sw == 2:
        arr = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    elif sw == 4:
        arr = np.frombuffer(raw, dtype=np.int32).astype(np.float32) / 2147483648.0
    else:
        raise RuntimeError(f"Unsupported WAV sample width: {sw}")
    arr = arr.reshape(-1, ch).T
    return torch.from_numpy(arr), sr


def _save_wav_fallback(path: str, wav: torch.Tensor, sr: int) -> None:
    x = wav.detach().cpu()
    if x.dim() == 1:
        x = x.unsqueeze(0)
    x = x.clamp(-1.0, 1.0).numpy()
    pcm = (x * 32767.0).astype(np.int16)
    interleaved = pcm.T.reshape(-1).tobytes()
    with wave.open(path, "wb") as wf:
        wf.setnchannels(int(x.shape[0]))
        wf.setsampwidth(2)
        wf.setframerate(int(sr))
        wf.writeframes(interleaved)


def _wav_meta(path: str) -> dict[str, float]:
    with wave.open(path, "rb") as wf:
        sr = int(wf.getframerate())
        ch = int(wf.getnchannels())
        sw = int(wf.getsampwidth())
        n = int(wf.getnframes())
    return {
        "channels": ch,
        "sample_rate": sr,
        "sample_width": sw,
        "frames": n,
        "duration": float(n) / float(sr),
        "bytes": float(os.path.getsize(path)),
    }


def main() -> None:
    cfg = yaml.safe_load(open("config.yaml", "r"))
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = RAFA(cfg).to(dev)

    ckpts = [n for n in os.listdir(cfg["training"]["checkpoint_dir"]) if n.endswith(".pt")]
    if not ckpts:
        raise FileNotFoundError("No checkpoints found for inference.")
    
    def _ckpt_sort(n):
        if "rafa_ep" in n:
            try:
                return int(n.replace("rafa_ep", "").replace(".pt", ""))
            except: return 0
        return 0
    
    ckpts.sort(key=_ckpt_sort)
    latest_ckpt = ckpts[-1]
    print(f"Loading checkpoint: {latest_ckpt}")
    state = torch.load(os.path.join(cfg["training"]["checkpoint_dir"], latest_ckpt), map_location=dev)
    _missing, unexpected = model.load_state_dict(state, strict=False)
    if unexpected:
        print(f"Checkpoint compatibility warning: unexpected_keys={len(unexpected)}")
    model.eval()

    wav = None
    sr = None
    input_path = "input/input.wav"
    if not os.path.exists(input_path):
        input_path = "input.wav"

    try:
        wav, sr = torchaudio.load(input_path)
    except Exception:
        try:
            wav, sr = _load_wav_fallback(input_path)
        except Exception:
            fallback_root = cfg.get("data", {}).get("local_wav_root", "datasets/web_wavs")
            candidates = sorted(
                [os.path.join(fallback_root, n) for n in os.listdir(fallback_root) if n.lower().endswith(".wav")]
            )
            if not candidates:
                raise RuntimeError("No decodable input source found for inference.")
            wav, sr = _load_wav_fallback(candidates[0])
    wav = wav.mean(dim=0)
    if sr != cfg["data"]["sample_rate"]:
        wav = torchaudio.transforms.Resample(sr, cfg["data"]["sample_rate"])(wav)
    
    # Request duration or use 10 seconds if available
    target_duration = 10.0
    target_samples = int(target_duration * cfg["data"]["sample_rate"])
    if wav.shape[0] > target_samples:
        wav = wav[:target_samples]
    else:
        print(f"Input too short for 10s, using full length: {wav.shape[0]/cfg['data']['sample_rate']:.2f}s")

    mag, phase = compute_stft(wav.unsqueeze(0), cfg["data"]["stft"])
    mag = mag.squeeze(0).to(dev)
    phase = phase.squeeze(0).to(dev)

    tokens = None
    if cfg["model"]["text_conditioned"]:
        from transformers import CLIPTokenizer

        tok = CLIPTokenizer.from_pretrained(cfg["model"]["text_encoder"])
        cap = input("Prompt: ")
        tokens = tok(cap, return_tensors="pt").to(dev)

    with torch.no_grad():
        pred_phase, pred_mag, p_ext, m_ext = model(mag.unsqueeze(0), phase.unsqueeze(0), tokens)

    print(f"Pred Mag range: {pred_mag.min().item():.4f} to {pred_mag.max().item():.4f}, mean={pred_mag.mean().item():.4f}")
    print(f"Pred Phase range: {pred_phase.min().item():.4f} to {pred_phase.max().item():.4f}, mean={pred_phase.mean().item():.4f}")

    # Check for NaNs
    if not torch.isfinite(pred_mag).all() or not torch.isfinite(pred_phase).all():
        print("CRITICAL: NaNs detected in model output during inference!")

    # The new RAFA-PC model returns absolute phase from the IFS gru.
    # We no longer need to manually integrate deltas.
    phase_rec = pred_phase
    z_rec = pred_mag * torch.exp(1j * phase_rec)
    wav_rec = torch.istft(
        z_rec,
        n_fft=cfg["data"]["stft"]["n_fft"],
        hop_length=cfg["data"]["stft"]["hop"],
        win_length=cfg["data"]["stft"]["win_length"],
        window=torch.hann_window(cfg["data"]["stft"]["win_length"]).to(z_rec.device),
        length=wav.shape[0],
    )

    output_path = "outputs/reconstructed.wav"
    try:
        torchaudio.save(output_path, wav_rec.cpu(), cfg["data"]["sample_rate"])
    except Exception:
        _save_wav_fallback(output_path, wav_rec.cpu(), cfg["data"]["sample_rate"])
    expected_dur = float(cfg["data"]["sample_rate"]) / float(cfg["data"]["sample_rate"])
    meta = _wav_meta(output_path)
    print(
        f"Saved -> {output_path} | ch={meta['channels']} sr={int(meta['sample_rate'])} "
        f"sw={int(meta['sample_width'])} frames={int(meta['frames'])} "
        f"dur={meta['duration']:.6f}s bytes={int(meta['bytes'])}"
    )
    min_allowed = 0.8 * expected_dur
    if meta["duration"] < min_allowed:
        raise RuntimeError(
            "Reconstructed duration is suspiciously short. "
            f"expected~{expected_dur:.3f}s got={meta['duration']:.3f}s "
            f"frames={meta['frames']} bytes={meta['bytes']}"
        )
    if meta["bytes"] < 20000:
        raise RuntimeError(
            "Reconstructed WAV file size is suspiciously small. "
            f"bytes={meta['bytes']} duration={meta['duration']:.3f}s"
        )


if __name__ == "__main__":
    main()
