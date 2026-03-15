"""Audio I/O and DSP round-trip diagnostics for RAFA."""

from __future__ import annotations

import argparse
import os
import wave

import numpy as np
import torch
import yaml

from stft_utils import compute_stft


def _load_wav(path: str) -> tuple[torch.Tensor, int, int, int, int]:
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
        raise RuntimeError(f"Unsupported sample width: {sw} bytes")
    arr = arr.reshape(-1, ch).T
    return torch.from_numpy(arr), sr, ch, sw, n


def _save_wav(path: str, wav: torch.Tensor, sr: int) -> None:
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


def print_wav_metadata(path: str) -> dict[str, float]:
    wav, sr, ch, sw, n = _load_wav(path)
    duration = float(n) / float(sr)
    size = float(os.path.getsize(path))
    print(f"[wav] path={path}")
    print(f"[wav] channels={ch} sample_rate={sr} sample_width_bytes={sw} frames={n}")
    print(f"[wav] duration_sec={duration:.6f} bytes={int(size)}")
    return {
        "channels": ch,
        "sample_rate": sr,
        "sample_width_bytes": sw,
        "frames": n,
        "duration_sec": duration,
        "bytes": size,
        "rms": float(torch.sqrt(torch.mean(wav.mean(dim=0) ** 2) + 1e-12).item()),
        "peak": float(torch.max(torch.abs(wav)).item()),
    }


def stft_istft_roundtrip(path: str, cfg_path: str, out_path: str) -> dict[str, float]:
    cfg = yaml.safe_load(open(cfg_path, "r"))
    stft_cfg = cfg["data"]["stft"]
    wav, sr, ch, sw, n = _load_wav(path)
    mono = wav.mean(dim=0)
    mag, phase = compute_stft(mono, stft_cfg)
    z = mag * torch.exp(1j * phase)
    rec = torch.istft(
        z,
        n_fft=int(stft_cfg["n_fft"]),
        hop_length=int(stft_cfg["hop"]),
        win_length=int(stft_cfg["win_length"]),
        window=torch.hann_window(int(stft_cfg["win_length"])),
        center=True,
        length=mono.numel(),
    )
    _save_wav(out_path, rec.unsqueeze(0), sr)

    in_dur = float(n) / float(sr)
    out_wav, out_sr, _ch, _sw, out_n = _load_wav(out_path)
    out_dur = float(out_n) / float(out_sr)
    frame_tol = 1.0 / float(sr)
    dur_diff = abs(in_dur - out_dur)
    rms = float(torch.sqrt(torch.mean(out_wav.mean(dim=0) ** 2) + 1e-12).item())
    peak = float(torch.max(torch.abs(out_wav)).item())

    print(f"[rt] in_dur={in_dur:.6f}s out_dur={out_dur:.6f}s diff={dur_diff:.6f}s")
    print(f"[rt] out_rms={rms:.6f} out_peak={peak:.6f}")
    if dur_diff > frame_tol:
        raise RuntimeError(
            f"Round-trip duration mismatch too large: {dur_diff:.6f}s > frame_tol {frame_tol:.6f}s"
        )
    if not (1e-4 <= rms <= 1.0):
        raise RuntimeError(f"Round-trip RMS out of expected range: {rms:.6f}")
    if not (1e-3 <= peak <= 1.0):
        raise RuntimeError(f"Round-trip peak out of expected range: {peak:.6f}")
    return {"duration_diff_sec": dur_diff, "rms": rms, "peak": peak}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--input", default="input.wav")
    p.add_argument("--config", default="config.yaml")
    p.add_argument("--out", default="roundtrip.wav")
    args = p.parse_args()
    print_wav_metadata(args.input)
    stft_istft_roundtrip(args.input, args.config, args.out)
    print_wav_metadata(args.out)


if __name__ == "__main__":
    main()

