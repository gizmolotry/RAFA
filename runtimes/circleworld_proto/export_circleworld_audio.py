from __future__ import annotations

import argparse
import json
import sys
import wave
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
CORE = ROOT / "core"
LINEAGE = ROOT / "lineages" / "04_positive_replacement"
for path in (ROOT, CORE, LINEAGE):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from config import load_config
from circleworld import CircleworldConfig, recurse_circleworld, summarize_circleworld_run
from stft_utils import compute_stft, inverse_stft
from rafa_math_tools import phase_to_phasor, phasor_normalize


def _safe_device(requested: str) -> torch.device:
    if requested == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _load_circle_cfg(path: Path) -> CircleworldConfig:
    payload = json.loads(path.read_text(encoding="utf-8"))
    cfg = payload["config"] if "config" in payload else payload
    return CircleworldConfig(
        qset=tuple(cfg.get("qset", (2, 3, 4, 5, 6, 8, 12))),
        q_weights=tuple(cfg["q_weights"]),
        promotion_threshold=float(cfg["promotion_threshold"]),
        max_promotions=int(cfg["max_promotions"]),
        recursion_depth=int(cfg["recursion_depth"]),
        residue_scale=float(cfg.get("residue_scale", 0.75)),
        child_law_gain=float(cfg["child_law_gain"]),
        attack_window=int(cfg["attack_window"]),
        persistence_momentum=float(cfg["persistence_momentum"]),
    )


def _native_resample(wav: torch.Tensor, orig_sr: int, target_sr: int) -> torch.Tensor:
    if orig_sr == target_sr:
        return wav
    ratio = float(target_sr) / float(orig_sr)
    target_len = int(round(wav.size(1) * ratio))
    return F.interpolate(
        wav.unsqueeze(0),
        size=target_len,
        mode="linear",
        align_corners=False,
    ).squeeze(0)


def _load_local_pcm_wav(path: Path) -> tuple[torch.Tensor, int]:
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
    elif sampwidth == 3:
        b = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 3)
        v = (
            b[:, 0].astype(np.int32)
            | (b[:, 1].astype(np.int32) << 8)
            | (b[:, 2].astype(np.int32) << 16)
        )
        sign = 1 << 23
        v = (v ^ sign) - sign
        arr = v.astype(np.float32) / 8388608.0
    elif sampwidth == 4:
        arr = np.frombuffer(raw, dtype=np.int32).astype(np.float32) / 2147483648.0
    else:
        raise RuntimeError(f"Unsupported WAV sample width: {sampwidth} bytes")

    arr = arr.reshape(-1, ch).T
    wav = torch.from_numpy(arr)
    return wav, sr


def _save_wav(wav: torch.Tensor, path: Path, sr: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    mono = wav.detach().cpu().float().view(-1).numpy()
    mono = mono / (np.abs(mono).max() + 1e-8)
    pcm = (mono * 32767.0).clip(-32768, 32767).astype(np.int16)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(pcm.tobytes())


def _crop_or_pad(wav: torch.Tensor, target_len: int) -> torch.Tensor:
    if wav.size(1) < target_len:
        wav = F.pad(wav, (0, target_len - wav.size(1)))
    elif wav.size(1) > target_len:
        wav = wav[:, :target_len]
    return wav


def _phasor_to_phase(z: torch.Tensor) -> torch.Tensor:
    return torch.atan2(z[..., 1], z[..., 0])


def _blend_phase(original_phase: torch.Tensor, final_z: torch.Tensor, blend: float) -> torch.Tensor:
    blend = float(min(1.0, max(0.0, blend)))
    orig_z = phase_to_phasor(original_phase)
    mixed = phasor_normalize((1.0 - blend) * orig_z + blend * final_z)
    return _phasor_to_phase(mixed)


def render_circleworld_audio(
    wav_path: Path,
    config_path: Path,
    out_path: Path,
    device_name: str,
    phase_blend: float,
) -> dict[str, Any]:
    cfg = load_config()
    sr = int(cfg["data"]["sample_rate"])
    clip_seconds = int(cfg["data"].get("clip_seconds", 2))
    target_len = sr * clip_seconds
    stft_cfg = cfg["data"]["stft"]
    circle_cfg = _load_circle_cfg(config_path)
    device = _safe_device(device_name)

    wav, wav_sr = _load_local_pcm_wav(wav_path)
    wav = wav.float()
    if wav.size(0) > 1:
        wav = wav.mean(dim=0, keepdim=True)
    wav = _native_resample(wav, wav_sr, sr)
    wav = _crop_or_pad(wav, target_len)
    wav = wav.to(device)

    mag, phase = compute_stft(wav, stft_cfg)
    phase_state = phase_to_phasor(phase)
    run = recurse_circleworld(phase_state, cfg=circle_cfg, depth=circle_cfg.recursion_depth, mode="active_packets")
    summary = summarize_circleworld_run(run)
    final_phase = _blend_phase(phase, run["phase_state"], blend=phase_blend)
    out_wav = inverse_stft(mag, final_phase, stft_cfg).squeeze(0)
    _save_wav(out_wav, out_path, sr=sr)

    meta = {
        "source_wav": str(wav_path),
        "out_wav": str(out_path),
        "circleworld_config": str(config_path),
        "device": str(device),
        "sample_rate": sr,
        "clip_seconds": clip_seconds,
        "phase_blend": float(phase_blend),
        **summary,
    }
    return meta


def main() -> None:
    ap = argparse.ArgumentParser(description="Render Circleworld audio from a real local WAV anchor.")
    ap.add_argument("--wav-path", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--phase-blend", type=float, default=0.75)
    args = ap.parse_args()

    meta = render_circleworld_audio(
        wav_path=Path(args.wav_path),
        config_path=Path(args.config),
        out_path=Path(args.out),
        device_name=args.device,
        phase_blend=args.phase_blend,
    )
    meta_path = Path(args.out).with_suffix(".json")
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
