from __future__ import annotations

import argparse
import json
import math
import re
import sys
import wave
from dataclasses import fields
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
CORE = ROOT / "core"
LINEAGE = ROOT / "lineages" / "04_positive_replacement"
RUNTIME = ROOT / "runtimes" / "circleworld_proto"
for path in (ROOT, CORE, LINEAGE, RUNTIME):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from circleworld import CircleworldConfig, recurse_circleworld, summarize_circleworld_run
from config import load_config
from phase_audio_metrics import chunk_vectors as _chunk_vectors
from phase_audio_metrics import compare_waveforms as _compare_waveforms
from phase_audio_metrics import cosine as _cosine
from phase_audio_metrics import frame_rms as _frame_rms
from phase_audio_metrics import loop_reentry_metrics as _loop_reentry_metrics
from phase_audio_metrics import normalized_autocorr as _normalized_autocorr
from rafa_math_tools import phase_to_phasor
from stft_utils import compute_stft, inverse_stft


DEFAULT_CASES_JSON = ROOT / "outputs" / "circleworld_proto" / "expanded_anchor_cases_2026-04-09.json"
DEFAULT_CASES = {
    "airplane_takeoff": ROOT / "wav_files" / "soundbible_airplane-takeoff_c752b8ba8b.wav",
    "steam_engine": ROOT / "wav_files" / "soundbible_steam-engine-running_68e20d27d1.wav",
    "voice": ROOT / "wav_files" / "soundbible_mystic-chanting-4_a38f2bbe68.wav",
    "spooky_drone": ROOT / "wav_files" / "soundbible_spooky-drone_2efbfd965b.wav",
    "sax_like_clarinet": ROOT / "wav_files" / "freewavesamples_ensoniq-zr-76-clarinet-c5_19f8dd5ddd.wav",
}
MAGNITUDE_MODES = ("prefix_hold", "flat")
PHASE_SEED_POLICIES = (
    "velocity",
    "hold",
    "damped_velocity_0_5",
    "reverse_velocity",
    "mean_velocity_4",
    "mean_velocity_8",
    "weighted_mean_velocity_8",
    "linear_fit_velocity_8",
    "stft_bin_rotation",
    "stft_phase_period_loop",
    "copy_last_waveform_phase",
    "copy_last_waveform_phase_0_5s",
    "copy_last_waveform_phase_0_25s",
    "autocorr_loop_waveform_phase",
)


def _safe_device(requested: str) -> torch.device:
    if requested == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _jsonify(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonify(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonify(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if torch.is_tensor(value):
        if value.numel() == 1:
            return float(value.detach().cpu().item())
        return value.detach().cpu().tolist()
    return value


def _load_circle_cfg(path: Path) -> CircleworldConfig:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    raw = payload["config"] if isinstance(payload, dict) and "config" in payload else payload
    if not isinstance(raw, dict):
        raise RuntimeError(f"Circleworld config must be a JSON object: {path}")

    defaults = CircleworldConfig()
    kwargs: dict[str, Any] = {}
    for item in fields(CircleworldConfig):
        value = raw.get(item.name, getattr(defaults, item.name))
        default_value = getattr(defaults, item.name)
        if isinstance(default_value, tuple):
            value = tuple(value)
        kwargs[item.name] = value
    return CircleworldConfig(**kwargs)


def _load_cases(path: Path | None = None) -> dict[str, Path]:
    if path is None:
        path = DEFAULT_CASES_JSON if DEFAULT_CASES_JSON.exists() else None
    if path is None:
        cases = dict(DEFAULT_CASES)
    else:
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
        if not isinstance(payload, dict):
            raise RuntimeError(f"Case JSON must map names to WAV paths: {path}")
        cases = {str(name): Path(str(wav_path)) for name, wav_path in payload.items()}
    existing = {name: wav_path for name, wav_path in cases.items() if wav_path.exists()}
    if not existing:
        raise RuntimeError("No real anchor WAVs were found for the continuation benchmark")
    return existing


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
        values = (
            b[:, 0].astype(np.int32)
            | (b[:, 1].astype(np.int32) << 8)
            | (b[:, 2].astype(np.int32) << 16)
        )
        sign = 1 << 23
        arr = ((values ^ sign) - sign).astype(np.float32) / 8388608.0
    elif sampwidth == 4:
        arr = np.frombuffer(raw, dtype=np.int32).astype(np.float32) / 2147483648.0
    else:
        raise RuntimeError(f"Unsupported WAV sample width: {sampwidth} bytes")

    arr = arr.reshape(-1, ch).T
    return torch.from_numpy(arr), sr


def _native_resample(wav: torch.Tensor, orig_sr: int, target_sr: int) -> torch.Tensor:
    if orig_sr == target_sr:
        return wav
    target_len = int(round(wav.size(1) * float(target_sr) / float(orig_sr)))
    return F.interpolate(
        wav.unsqueeze(0),
        size=max(1, target_len),
        mode="linear",
        align_corners=False,
    ).squeeze(0)


def _crop_or_pad_zero(wav: torch.Tensor, target_len: int) -> torch.Tensor:
    if wav.size(-1) < target_len:
        wav = F.pad(wav, (0, target_len - wav.size(-1)))
    elif wav.size(-1) > target_len:
        wav = wav[..., :target_len]
    return wav


def _prepare_anchor(
    wav_path: Path,
    target_sr: int,
    total_samples: int,
) -> torch.Tensor:
    wav, wav_sr = _load_local_pcm_wav(wav_path)
    wav = wav.float()
    if wav.size(0) > 1:
        wav = wav.mean(dim=0, keepdim=True)
    wav = _native_resample(wav, wav_sr, target_sr)
    return _crop_or_pad_zero(wav, total_samples)


def _write_wav(path: Path, wav: np.ndarray | torch.Tensor, sr: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if torch.is_tensor(wav):
        arr = wav.detach().cpu().float().view(-1).numpy()
    else:
        arr = np.asarray(wav, dtype=np.float32).reshape(-1)
    pcm = np.clip(arr, -1.0, 1.0)
    pcm = (pcm * 32767.0).clip(-32768, 32767).astype(np.int16)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(pcm.tobytes())


def _sanitize_name(name: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", name.strip())
    return safe.strip("._") or "case"


def _wrap_angle(x: torch.Tensor) -> torch.Tensor:
    return (x + math.pi) % (2.0 * math.pi) - math.pi


def _phase_to_angle(z: torch.Tensor) -> torch.Tensor:
    return torch.atan2(z[..., 1], z[..., 0])


def _extend_phase_from_prefix(prefix_phase: torch.Tensor, future_frames: int) -> torch.Tensor:
    future_frames = max(1, int(future_frames))
    if prefix_phase.size(-1) >= 2:
        delta = _wrap_angle(prefix_phase[..., -1:] - prefix_phase[..., -2:-1])
    else:
        delta = torch.zeros_like(prefix_phase[..., -1:])
    steps = torch.arange(1, future_frames + 1, device=prefix_phase.device, dtype=prefix_phase.dtype)
    future = prefix_phase[..., -1:] + delta * steps.view(1, 1, -1)
    return torch.cat([prefix_phase, _wrap_angle(future)], dim=-1)


def _build_magnitude_canvas(
    prefix_mag: torch.Tensor,
    future_frames: int,
    mode: str,
) -> tuple[torch.Tensor, dict[str, Any]]:
    if mode == "prefix_hold":
        future_mag = prefix_mag[..., -1:].expand(-1, -1, future_frames).clone()
        source = "last_prefix_stft_magnitude_frame"
    elif mode == "flat":
        scalar = prefix_mag.mean(dim=(1, 2), keepdim=True)
        future_mag = scalar.expand(-1, prefix_mag.size(1), future_frames).clone()
        source = "prefix_mean_magnitude_scalar"
    else:
        raise ValueError(f"Unknown magnitude mode: {mode}")
    return torch.cat([prefix_mag, future_mag], dim=-1), {
        "magnitude_mode": mode,
        "future_magnitude_source": source,
        "future_target_magnitude_reused": False,
        "target_future_stft_magnitude_accessed": False,
    }


def _render_future_from_stft(
    mag_canvas: torch.Tensor,
    phase_canvas: torch.Tensor,
    stft_cfg: dict[str, Any],
    prefix_samples: int,
    future_samples: int,
) -> torch.Tensor:
    wav = inverse_stft(mag_canvas, phase_canvas, stft_cfg).squeeze(0)
    wav = _crop_or_pad_zero(wav, prefix_samples + future_samples)
    return wav[prefix_samples : prefix_samples + future_samples]


def _copy_recent_baseline(prefix: torch.Tensor, future_samples: int, sr: int, seconds: float) -> torch.Tensor:
    copy_len = min(prefix.size(-1), max(1, int(round(float(seconds) * sr))))
    seed = prefix[..., -copy_len:].reshape(-1)
    reps = int(math.ceil(float(future_samples) / float(max(1, seed.numel()))))
    return seed.repeat(reps)[:future_samples]


def _copy_last_baseline(prefix: torch.Tensor, future_samples: int, sr: int) -> torch.Tensor:
    return _copy_recent_baseline(prefix, future_samples=future_samples, sr=sr, seconds=1.0)


def _copy_autocorr_loop_baseline(prefix: torch.Tensor, future_samples: int, sr: int) -> tuple[torch.Tensor, dict[str, Any]]:
    x = prefix.detach().cpu().float().reshape(-1).numpy()
    if x.size < max(16, int(round(0.05 * sr))):
        return _copy_recent_baseline(prefix.detach().cpu(), future_samples, sr, seconds=0.25), {
            "autocorr_loop_lag_samples": 0,
            "autocorr_loop_corr": 0.0,
            "autocorr_fallback": True,
        }
    centered = x.astype(np.float64) - float(np.mean(x))
    denom = float(np.dot(centered, centered))
    if denom <= 1.0e-12:
        return _copy_recent_baseline(prefix.detach().cpu(), future_samples, sr, seconds=0.25), {
            "autocorr_loop_lag_samples": 0,
            "autocorr_loop_corr": 0.0,
            "autocorr_fallback": True,
        }
    corr = np.correlate(centered, centered, mode="full")[x.size - 1 :] / denom
    min_lag = max(1, int(round(0.02 * sr)))
    max_lag = min(x.size - 1, int(round(0.5 * sr)))
    if max_lag <= min_lag:
        lag = min(x.size, max(1, int(round(0.25 * sr))))
        score = 0.0
    else:
        local = corr[min_lag : max_lag + 1]
        lag = int(np.argmax(local) + min_lag)
        score = float(local[lag - min_lag])
    seed = prefix.detach().cpu()[..., -lag:].reshape(-1)
    reps = int(math.ceil(float(future_samples) / float(max(1, seed.numel()))))
    return seed.repeat(reps)[:future_samples], {
        "autocorr_loop_lag_samples": int(lag),
        "autocorr_loop_corr": score,
        "autocorr_fallback": False,
    }


def _fit_phase_frames(phase: torch.Tensor, frames: int) -> torch.Tensor:
    if phase.size(-1) == frames:
        return phase
    if phase.size(-1) > frames:
        return phase[..., :frames]
    pad = phase[..., -1:].expand(-1, -1, frames - phase.size(-1))
    return torch.cat([phase, pad], dim=-1)


def _future_from_delta(prefix_phase: torch.Tensor, future_frames: int, delta: torch.Tensor) -> torch.Tensor:
    steps = torch.arange(1, future_frames + 1, device=prefix_phase.device, dtype=prefix_phase.dtype)
    future = _wrap_angle(prefix_phase[..., -1:] + delta * steps.view(1, 1, -1))
    return torch.cat([prefix_phase, future], dim=-1)


def _mean_phase_delta(prefix_phase: torch.Tensor, history_frames: int) -> torch.Tensor:
    if prefix_phase.size(-1) < 2:
        return torch.zeros_like(prefix_phase[..., -1:])
    history = max(1, min(int(history_frames), int(prefix_phase.size(-1) - 1)))
    deltas = _wrap_angle(prefix_phase[..., -history:] - prefix_phase[..., -history - 1 : -1])
    sin_mean = torch.sin(deltas).mean(dim=-1, keepdim=True)
    cos_mean = torch.cos(deltas).mean(dim=-1, keepdim=True)
    return torch.atan2(sin_mean, cos_mean)


def _weighted_mean_phase_delta(prefix: torch.Tensor, prefix_phase: torch.Tensor, stft_cfg: dict[str, Any], history_frames: int) -> torch.Tensor:
    if prefix_phase.size(-1) < 2:
        return torch.zeros_like(prefix_phase[..., -1:])
    prefix_mag, _phase = compute_stft(prefix.detach(), stft_cfg)
    history = max(1, min(int(history_frames), int(prefix_phase.size(-1) - 1), int(prefix_mag.size(-1))))
    deltas = _wrap_angle(prefix_phase[..., -history:] - prefix_phase[..., -history - 1 : -1])
    weights = prefix_mag[..., -history:].clamp_min(1.0e-8)
    sin_mean = (torch.sin(deltas) * weights).sum(dim=-1, keepdim=True) / weights.sum(dim=-1, keepdim=True)
    cos_mean = (torch.cos(deltas) * weights).sum(dim=-1, keepdim=True) / weights.sum(dim=-1, keepdim=True)
    return torch.atan2(sin_mean, cos_mean)


def _linear_fit_phase_delta(prefix_phase: torch.Tensor, history_frames: int) -> torch.Tensor:
    if prefix_phase.size(-1) < 2:
        return torch.zeros_like(prefix_phase[..., -1:])
    history = max(2, min(int(history_frames), int(prefix_phase.size(-1))))
    hist = prefix_phase[..., -history:]
    increments = _wrap_angle(hist[..., 1:] - hist[..., :-1])
    unwrapped = torch.cat([hist[..., :1], hist[..., :1] + torch.cumsum(increments, dim=-1)], dim=-1)
    t = torch.arange(history, device=prefix_phase.device, dtype=prefix_phase.dtype).view(1, 1, -1)
    centered_t = t - t.mean(dim=-1, keepdim=True)
    centered_y = unwrapped - unwrapped.mean(dim=-1, keepdim=True)
    denom = (centered_t * centered_t).sum(dim=-1, keepdim=True).clamp_min(1.0e-8)
    return (centered_t * centered_y).sum(dim=-1, keepdim=True) / denom


def _stft_phase_period_loop(prefix: torch.Tensor, prefix_phase: torch.Tensor, future_frames: int, stft_cfg: dict[str, Any]) -> tuple[torch.Tensor, dict[str, Any]]:
    prefix_mag, _phase = compute_stft(prefix.detach(), stft_cfg)
    envelope = prefix_mag.detach().cpu().float().mean(dim=1).reshape(-1).numpy()
    frames = int(min(prefix_phase.size(-1), envelope.size))
    if frames < 4:
        future = prefix_phase[..., -1:].expand(-1, -1, future_frames).clone()
        return torch.cat([prefix_phase, future], dim=-1), {
            "period_loop_lag_frames": 0,
            "period_loop_corr": 0.0,
            "period_loop_fallback": True,
        }
    x = envelope[-frames:].astype(np.float64)
    x = x - float(np.mean(x))
    denom = float(np.dot(x, x))
    min_lag = 2
    max_lag = min(frames - 1, 64)
    if denom <= 1.0e-12 or max_lag <= min_lag:
        lag = min(frames, max(1, future_frames))
        score = 0.0
        fallback = True
    else:
        corr = np.correlate(x, x, mode="full")[frames - 1 :] / denom
        local = corr[min_lag : max_lag + 1]
        lag = int(np.argmax(local) + min_lag)
        score = float(local[lag - min_lag])
        fallback = False
    seed = prefix_phase[..., -lag:]
    reps = int(math.ceil(float(future_frames) / float(max(1, seed.size(-1)))))
    future = seed.repeat(1, 1, reps)[..., :future_frames].clone()
    return torch.cat([prefix_phase, future], dim=-1), {
        "period_loop_lag_frames": int(lag),
        "period_loop_corr": score,
        "period_loop_fallback": fallback,
    }


def _phase_from_future_waveform(
    *,
    prefix: torch.Tensor,
    prefix_phase: torch.Tensor,
    future_waveform: torch.Tensor,
    stft_cfg: dict[str, Any],
) -> torch.Tensor:
    total_frames = int(prefix_phase.size(-1) + max(1, math.ceil(float(future_waveform.numel()) / float(stft_cfg["hop"]))))
    canvas = torch.cat([prefix.detach(), future_waveform.reshape(1, -1).to(prefix.device)], dim=-1)
    _mag, phase = compute_stft(canvas, stft_cfg)
    phase = _fit_phase_frames(phase, total_frames)
    return torch.cat([prefix_phase, phase[..., prefix_phase.size(-1) :]], dim=-1)


def _build_phase_seed_canvas(
    prefix: torch.Tensor,
    prefix_phase: torch.Tensor,
    future_frames: int,
    future_samples: int,
    sr: int,
    stft_cfg: dict[str, Any],
    policy: str,
) -> tuple[torch.Tensor, dict[str, Any]]:
    if policy == "velocity":
        return _extend_phase_from_prefix(prefix_phase, future_frames=future_frames), {
            "phase_seed_policy": policy,
            "future_phase_source": "last_prefix_phase_velocity",
            "future_target_phase_reused": False,
            "target_future_stft_phase_accessed": False,
        }
    if policy == "hold":
        future = prefix_phase[..., -1:].expand(-1, -1, future_frames).clone()
        return torch.cat([prefix_phase, future], dim=-1), {
            "phase_seed_policy": policy,
            "future_phase_source": "last_prefix_phase_frame",
            "future_target_phase_reused": False,
            "target_future_stft_phase_accessed": False,
        }
    if policy in {"damped_velocity_0_5", "reverse_velocity"}:
        if prefix_phase.size(-1) >= 2:
            delta = _wrap_angle(prefix_phase[..., -1:] - prefix_phase[..., -2:-1])
        else:
            delta = torch.zeros_like(prefix_phase[..., -1:])
        scale = 0.5 if policy == "damped_velocity_0_5" else -1.0
        steps = torch.arange(1, future_frames + 1, device=prefix_phase.device, dtype=prefix_phase.dtype)
        future = _wrap_angle(prefix_phase[..., -1:] + float(scale) * delta * steps.view(1, 1, -1))
        return torch.cat([prefix_phase, future], dim=-1), {
            "phase_seed_policy": policy,
            "future_phase_source": f"last_prefix_phase_velocity_scaled_{scale}",
            "future_target_phase_reused": False,
            "target_future_stft_phase_accessed": False,
        }
    if policy in {"mean_velocity_4", "mean_velocity_8"}:
        history = 4 if policy == "mean_velocity_4" else 8
        delta = _mean_phase_delta(prefix_phase, history_frames=history)
        return _future_from_delta(prefix_phase, future_frames=future_frames, delta=delta), {
            "phase_seed_policy": policy,
            "future_phase_source": f"circular_mean_prefix_phase_velocity_last_{history}",
            "history_frames": history,
            "future_target_phase_reused": False,
            "target_future_stft_phase_accessed": False,
        }
    if policy == "weighted_mean_velocity_8":
        history = 8
        delta = _weighted_mean_phase_delta(prefix, prefix_phase, stft_cfg=stft_cfg, history_frames=history)
        return _future_from_delta(prefix_phase, future_frames=future_frames, delta=delta), {
            "phase_seed_policy": policy,
            "future_phase_source": "prefix_magnitude_weighted_circular_mean_phase_velocity_last_8",
            "history_frames": history,
            "future_target_phase_reused": False,
            "target_future_stft_phase_accessed": False,
        }
    if policy == "linear_fit_velocity_8":
        history = 8
        delta = _linear_fit_phase_delta(prefix_phase, history_frames=history)
        return _future_from_delta(prefix_phase, future_frames=future_frames, delta=delta), {
            "phase_seed_policy": policy,
            "future_phase_source": "least_squares_unwrapped_prefix_phase_velocity_last_8",
            "history_frames": history,
            "future_target_phase_reused": False,
            "target_future_stft_phase_accessed": False,
        }
    if policy == "stft_bin_rotation":
        n_fft = max(1, int(stft_cfg.get("n_fft", (prefix_phase.size(1) - 1) * 2)))
        hop = int(stft_cfg.get("hop", 1))
        bins = torch.arange(prefix_phase.size(1), device=prefix_phase.device, dtype=prefix_phase.dtype).view(1, -1, 1)
        delta = _wrap_angle((2.0 * math.pi * float(hop) / float(n_fft)) * bins)
        return _future_from_delta(prefix_phase, future_frames=future_frames, delta=delta), {
            "phase_seed_policy": policy,
            "future_phase_source": "analytic_stft_bin_rotation_from_prefix",
            "n_fft": n_fft,
            "hop": hop,
            "future_target_phase_reused": False,
            "target_future_stft_phase_accessed": False,
        }
    if policy == "stft_phase_period_loop":
        phase, loop_flags = _stft_phase_period_loop(
            prefix=prefix,
            prefix_phase=prefix_phase,
            future_frames=future_frames,
            stft_cfg=stft_cfg,
        )
        return phase, {
            "phase_seed_policy": policy,
            "future_phase_source": "prefix_stft_envelope_periodic_phase_loop_without_future_target",
            **loop_flags,
            "future_target_phase_reused": False,
            "target_future_stft_phase_accessed": False,
        }
    if policy == "copy_last_waveform_phase":
        copy_last = _copy_last_baseline(prefix.detach().cpu(), future_samples=future_samples, sr=sr).to(prefix.device)
        return _phase_from_future_waveform(
            prefix=prefix,
            prefix_phase=prefix_phase,
            future_waveform=copy_last,
            stft_cfg=stft_cfg,
        ), {
            "phase_seed_policy": policy,
            "future_phase_source": "copy_last_waveform_stft_phase_without_future_target",
            "copy_seconds": 1.0,
            "future_target_phase_reused": False,
            "target_future_stft_phase_accessed": False,
        }
    if policy in {"copy_last_waveform_phase_0_5s", "copy_last_waveform_phase_0_25s"}:
        seconds = 0.5 if policy == "copy_last_waveform_phase_0_5s" else 0.25
        copy_recent = _copy_recent_baseline(prefix.detach().cpu(), future_samples=future_samples, sr=sr, seconds=seconds).to(
            prefix.device
        )
        return _phase_from_future_waveform(
            prefix=prefix,
            prefix_phase=prefix_phase,
            future_waveform=copy_recent,
            stft_cfg=stft_cfg,
        ), {
            "phase_seed_policy": policy,
            "future_phase_source": f"copy_last_{seconds:g}s_waveform_stft_phase_without_future_target",
            "copy_seconds": seconds,
            "future_target_phase_reused": False,
            "target_future_stft_phase_accessed": False,
        }
    if policy == "autocorr_loop_waveform_phase":
        loop_future, loop_flags = _copy_autocorr_loop_baseline(
            prefix.detach().cpu(),
            future_samples=future_samples,
            sr=sr,
        )
        loop_future = loop_future.to(prefix.device)
        return _phase_from_future_waveform(
            prefix=prefix,
            prefix_phase=prefix_phase,
            future_waveform=loop_future,
            stft_cfg=stft_cfg,
        ), {
            "phase_seed_policy": policy,
            "future_phase_source": "prefix_autocorr_loop_waveform_stft_phase_without_future_target",
            **loop_flags,
            "future_target_phase_reused": False,
            "target_future_stft_phase_accessed": False,
        }
    raise ValueError(f"Unknown phase seed policy: {policy}")


def _method_summary(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    methods = sorted({method for row in rows for method in row.get("methods", {})})
    out: dict[str, Any] = {}
    for method in methods:
        metrics = [row["methods"][method]["metrics"] for row in rows if method in row.get("methods", {})]
        loop_metrics = [row["methods"][method]["loop_reentry"] for row in rows if method in row.get("methods", {})]
        flags = [row["methods"][method].get("magnitude_flags", {}) for row in rows if method in row.get("methods", {})]
        phase_flags = [
            row["methods"][method].get("phase_seed_flags", {}) for row in rows if method in row.get("methods", {})
        ]
        out[method] = {
            "case_count": len(metrics),
            "mean_corr": float(np.mean([m["corr"] for m in metrics])) if metrics else 0.0,
            "mean_mae": float(np.mean([m["mae"] for m in metrics])) if metrics else 0.0,
            "mean_mse": float(np.mean([m["mse"] for m in metrics])) if metrics else 0.0,
            "mean_loop_autocorr_peak": float(np.mean([m["loop_autocorr_peak"] for m in loop_metrics])) if loop_metrics else 0.0,
            "mean_first_chunk_reentry": float(np.mean([m["first_chunk_reentry"] for m in loop_metrics])) if loop_metrics else 0.0,
            "any_future_target_magnitude_reused": bool(any(bool(f.get("future_target_magnitude_reused")) for f in flags)),
            "any_target_future_stft_magnitude_accessed": bool(
                any(bool(f.get("target_future_stft_magnitude_accessed")) for f in flags)
            ),
            "any_future_target_phase_reused": bool(any(bool(f.get("future_target_phase_reused")) for f in phase_flags)),
            "any_target_future_stft_phase_accessed": bool(
                any(bool(f.get("target_future_stft_phase_accessed")) for f in phase_flags)
            ),
        }
    return out


def _markdown_summary(summary: dict[str, Any]) -> str:
    lines = [
        "# Circleworld Audio Continuation Benchmark",
        "",
        "This harness compares generated future audio against the held-out future waveform.",
        "The target future STFT magnitude is not used as an output carrier.",
        "",
        "## Run",
        "",
        f"- Config: `{summary['config_path']}`",
        f"- Device: `{summary['device']}`",
        f"- Cases: {summary['case_count']}",
        f"- Prefix seconds: {summary['prefix_seconds']}",
        f"- Future seconds: {summary['future_seconds']}",
        f"- Circleworld magnitude mode: `{summary['magnitude_mode']}`",
        f"- Phase seed policy: `{summary.get('phase_seed_policy', 'velocity')}`",
        f"- Future target magnitude reused: `{summary['future_target_magnitude_reused']}`",
        f"- Target future STFT magnitude accessed: `{summary.get('target_future_stft_magnitude_accessed', False)}`",
        f"- Future target phase reused: `{summary.get('future_target_phase_reused', False)}`",
        f"- Target future STFT phase accessed: `{summary.get('target_future_stft_phase_accessed', False)}`",
        "",
        "## Aggregate Metrics",
        "",
        "| Method | Corr | MAE | MSE | Loop peak | First reentry | Future mag reused | Future mag STFT accessed | Future phase reused | Future phase STFT accessed |",
        "| --- | ---: | ---: | ---: | ---: | ---: | --- | --- | --- | --- |",
    ]
    for method, item in sorted(summary["method_summary"].items()):
        lines.append(
            "| {method} | {corr:.6f} | {mae:.6f} | {mse:.6f} | {loop:.6f} | {reentry:.6f} | `{mag_flag}` | `{mag_stft_flag}` | `{phase_flag}` | `{phase_stft_flag}` |".format(
                method=method,
                corr=float(item["mean_corr"]),
                mae=float(item["mean_mae"]),
                mse=float(item["mean_mse"]),
                loop=float(item["mean_loop_autocorr_peak"]),
                reentry=float(item["mean_first_chunk_reentry"]),
                mag_flag=bool(item["any_future_target_magnitude_reused"]),
                mag_stft_flag=bool(item.get("any_target_future_stft_magnitude_accessed", False)),
                phase_flag=bool(item.get("any_future_target_phase_reused", False)),
                phase_stft_flag=bool(item.get("any_target_future_stft_phase_accessed", False)),
            )
        )
    lines.extend(
        [
            "",
            "## Cases",
            "",
            "| Case | Method | Corr | MAE | MSE | Output WAV |",
            "| --- | --- | ---: | ---: | ---: | --- |",
        ]
    )
    for row in summary["rows"]:
        for method, item in sorted(row["methods"].items()):
            metrics = item["metrics"]
            lines.append(
                "| {case} | {method} | {corr:.6f} | {mae:.6f} | {mse:.6f} | `{wav}` |".format(
                    case=row["name"],
                    method=method,
                    corr=float(metrics["corr"]),
                    mae=float(metrics["mae"]),
                    mse=float(metrics["mse"]),
                    wav=item.get("wav_path", ""),
                )
            )
    lines.extend(
        [
            "",
            "## Limitations",
            "",
            "- Circleworld is phase-only here; magnitude controls are prefix-derived carriers, not generated magnitude.",
            "- Continuation is seeded by a no-future phase policy because the current Circleworld runtime is not an autoregressive audio decoder.",
            "- Loop/reentry metrics are lightweight diagnostics, not a perceptual listening test.",
        ]
    )
    return "\n".join(lines) + "\n"


def _run_case(
    name: str,
    wav_path: Path,
    circle_cfg: CircleworldConfig,
    stft_cfg: dict[str, Any],
    sr: int,
    out_dir: Path,
    device: torch.device,
    prefix_seconds: float,
    future_seconds: float,
    magnitude_mode: str,
    phase_seed_policy: str,
) -> dict[str, Any]:
    prefix_samples = max(1, int(round(prefix_seconds * sr)))
    future_samples = max(1, int(round(future_seconds * sr)))
    total_samples = prefix_samples + future_samples
    safe_name = _sanitize_name(name)

    wav = _prepare_anchor(wav_path, target_sr=sr, total_samples=total_samples)
    prefix = wav[..., :prefix_samples].to(device)
    target_future = wav[..., prefix_samples:total_samples].reshape(-1).to(device)

    prefix_mag, prefix_phase = compute_stft(prefix, stft_cfg)
    hop = int(stft_cfg["hop"])
    future_frames = max(1, int(math.ceil(float(future_samples) / float(hop))))
    initial_phase, phase_seed_flags = _build_phase_seed_canvas(
        prefix=prefix,
        prefix_phase=prefix_phase,
        future_frames=future_frames,
        future_samples=future_samples,
        sr=sr,
        stft_cfg=stft_cfg,
        policy=phase_seed_policy,
    )
    initial_z = phase_to_phasor(initial_phase)

    run_mode = circle_cfg.branching_mode if circle_cfg.branching_mode in {"native_multimode", "native_multimode_childworld"} else "active_packets"
    run = recurse_circleworld(initial_z, cfg=circle_cfg, depth=circle_cfg.recursion_depth, mode=run_mode)
    final_phase = _phase_to_angle(run["phase_state"])
    final_phase = torch.cat([prefix_phase, final_phase[..., prefix_phase.size(-1) :]], dim=-1)

    mag_canvas, mag_flags = _build_magnitude_canvas(prefix_mag, future_frames=future_frames, mode=magnitude_mode)
    circle_future = _render_future_from_stft(
        mag_canvas=mag_canvas,
        phase_canvas=final_phase,
        stft_cfg=stft_cfg,
        prefix_samples=prefix_samples,
        future_samples=future_samples,
    )

    prefix_hold_mag_canvas, prefix_hold_flags = _build_magnitude_canvas(prefix_mag, future_frames=future_frames, mode="prefix_hold")
    flat_mag_canvas, flat_flags = _build_magnitude_canvas(prefix_mag, future_frames=future_frames, mode="flat")
    phase_seed_baseline = _render_future_from_stft(
        mag_canvas=prefix_hold_mag_canvas,
        phase_canvas=initial_phase,
        stft_cfg=stft_cfg,
        prefix_samples=prefix_samples,
        future_samples=future_samples,
    )
    flat_phase_seed_baseline = _render_future_from_stft(
        mag_canvas=flat_mag_canvas,
        phase_canvas=initial_phase,
        stft_cfg=stft_cfg,
        prefix_samples=prefix_samples,
        future_samples=future_samples,
    )
    copy_last = _copy_last_baseline(prefix.detach().cpu(), future_samples=future_samples, sr=sr).to(device)

    artifacts = {
        "target_future": out_dir / f"{safe_name}_target_future.wav",
        "circleworld": out_dir / f"{safe_name}_circleworld_continuation.wav",
        "baseline_copy_last": out_dir / f"{safe_name}_baseline_copy_last.wav",
        "baseline_prefix_magnitude_hold": out_dir / f"{safe_name}_baseline_prefix_magnitude_hold.wav",
        "baseline_flat_magnitude": out_dir / f"{safe_name}_baseline_flat_magnitude.wav",
    }
    _write_wav(artifacts["target_future"], target_future, sr)
    _write_wav(artifacts["circleworld"], circle_future, sr)
    _write_wav(artifacts["baseline_copy_last"], copy_last, sr)
    _write_wav(artifacts["baseline_prefix_magnitude_hold"], phase_seed_baseline, sr)
    _write_wav(artifacts["baseline_flat_magnitude"], flat_phase_seed_baseline, sr)

    methods = {
        "circleworld": {
            "wav_path": str(artifacts["circleworld"]),
            "metrics": _compare_waveforms(target_future, circle_future),
            "loop_reentry": _loop_reentry_metrics(circle_future, sr),
            "magnitude_flags": mag_flags,
            "phase_seed_flags": phase_seed_flags,
        },
        "baseline_copy_last": {
            "wav_path": str(artifacts["baseline_copy_last"]),
            "metrics": _compare_waveforms(target_future, copy_last),
            "loop_reentry": _loop_reentry_metrics(copy_last, sr),
            "magnitude_flags": {
                "magnitude_mode": "waveform_copy_last_prefix",
                "future_magnitude_source": "none_waveform_baseline",
                "future_target_magnitude_reused": False,
                "target_future_stft_magnitude_accessed": False,
            },
        },
        "baseline_prefix_magnitude_hold": {
            "wav_path": str(artifacts["baseline_prefix_magnitude_hold"]),
            "metrics": _compare_waveforms(target_future, phase_seed_baseline),
            "loop_reentry": _loop_reentry_metrics(phase_seed_baseline, sr),
            "magnitude_flags": prefix_hold_flags,
            "phase_seed_flags": phase_seed_flags,
        },
        "baseline_flat_magnitude": {
            "wav_path": str(artifacts["baseline_flat_magnitude"]),
            "metrics": _compare_waveforms(target_future, flat_phase_seed_baseline),
            "loop_reentry": _loop_reentry_metrics(flat_phase_seed_baseline, sr),
            "magnitude_flags": flat_flags,
            "phase_seed_flags": phase_seed_flags,
        },
    }

    return {
        "name": name,
        "source_wav": str(wav_path),
        "target_future_wav": str(artifacts["target_future"]),
        "sample_rate": sr,
        "prefix_samples": prefix_samples,
        "future_samples": future_samples,
        "prefix_stft_frames": int(prefix_mag.size(-1)),
        "future_stft_frames": int(future_frames),
        "continuation_strategy": f"{phase_seed_policy}_then_circleworld_rollout",
        "phase_seed_policy": phase_seed_policy,
        "phase_seed_flags": phase_seed_flags,
        "future_target_audio_used_for_metrics_only": True,
        "future_target_magnitude_reused": False,
        "target_future_stft_magnitude_accessed": False,
        "future_target_phase_reused": False,
        "target_future_stft_phase_accessed": False,
        "methods": methods,
        "circleworld_meta": _jsonify(summarize_circleworld_run(run)),
    }


def run_benchmark(
    config_path: Path,
    out_dir: Path,
    device_name: str,
    num_cases: int,
    prefix_seconds: float,
    future_seconds: float,
    magnitude_mode: str,
    phase_seed_policy: str,
    cases_json: Path | None = None,
) -> dict[str, Any]:
    if magnitude_mode not in MAGNITUDE_MODES:
        raise ValueError(f"magnitude_mode must be one of {MAGNITUDE_MODES}")
    if phase_seed_policy not in PHASE_SEED_POLICIES:
        raise ValueError(f"phase_seed_policy must be one of {PHASE_SEED_POLICIES}")
    if prefix_seconds <= 0.0 or future_seconds <= 0.0:
        raise ValueError("prefix_seconds and future_seconds must be positive")

    cfg = load_config(str(ROOT / "config.yaml")) if (ROOT / "config.yaml").exists() else load_config()
    sr = int(cfg["data"]["sample_rate"])
    stft_cfg = dict(cfg["data"]["stft"])
    circle_cfg = _load_circle_cfg(config_path)
    device = _safe_device(device_name)
    out_dir.mkdir(parents=True, exist_ok=True)

    cases = list(_load_cases(cases_json).items())
    if num_cases > 0:
        cases = cases[:num_cases]
    if not cases:
        raise RuntimeError("No benchmark cases selected")

    rows = [
        _run_case(
            name=name,
            wav_path=wav_path,
            circle_cfg=circle_cfg,
            stft_cfg=stft_cfg,
            sr=sr,
            out_dir=out_dir,
            device=device,
            prefix_seconds=prefix_seconds,
            future_seconds=future_seconds,
            magnitude_mode=magnitude_mode,
            phase_seed_policy=phase_seed_policy,
        )
        for name, wav_path in cases
    ]

    method_summary = _method_summary(rows)
    future_target_magnitude_reused = bool(
        any(bool(item.get("any_future_target_magnitude_reused")) for item in method_summary.values())
    )
    target_future_stft_magnitude_accessed = bool(
        any(bool(item.get("any_target_future_stft_magnitude_accessed")) for item in method_summary.values())
    )
    future_target_phase_reused = bool(
        any(bool(item.get("any_future_target_phase_reused")) for item in method_summary.values())
    )
    target_future_stft_phase_accessed = bool(
        any(bool(item.get("any_target_future_stft_phase_accessed")) for item in method_summary.values())
    )

    summary = {
        "runtime": "circleworld_proto",
        "schema": "circleworld_audio_continuation_v0",
        "config_path": str(config_path),
        "out_dir": str(out_dir),
        "device": str(device),
        "requested_device": device_name,
        "sample_rate": sr,
        "prefix_seconds": float(prefix_seconds),
        "future_seconds": float(future_seconds),
        "magnitude_mode": magnitude_mode,
        "phase_seed_policy": phase_seed_policy,
        "case_count": len(rows),
        "cases": {name: str(path) for name, path in cases},
        "future_target_magnitude_reused": future_target_magnitude_reused,
        "target_future_stft_magnitude_accessed": target_future_stft_magnitude_accessed,
        "future_target_phase_reused": future_target_phase_reused,
        "target_future_stft_phase_accessed": target_future_stft_phase_accessed,
        "future_target_audio_used_for_metrics_only": True,
        "method_summary": method_summary,
        "rows": rows,
    }
    summary = _jsonify(summary)
    (out_dir / "audio_continuation_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (out_dir / "audio_continuation_summary.md").write_text(_markdown_summary(summary), encoding="utf-8")
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Benchmark true Circleworld audio continuation without future target magnitude carriers."
    )
    ap.add_argument("--config", required=True, help="Circleworld JSON config path.")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--num-cases", type=int, default=3)
    ap.add_argument("--prefix-seconds", type=float, default=2.0)
    ap.add_argument("--future-seconds", type=float, default=2.0)
    ap.add_argument("--magnitude-mode", choices=MAGNITUDE_MODES, default="prefix_hold")
    ap.add_argument("--phase-seed-policy", choices=PHASE_SEED_POLICIES, default="velocity")
    ap.add_argument(
        "--cases-json",
        default=None,
        help="Optional JSON mapping case names to real anchor WAV paths. Defaults to the expanded Circleworld anchors if present.",
    )
    args = ap.parse_args()

    summary = run_benchmark(
        config_path=Path(args.config),
        out_dir=Path(args.out_dir),
        device_name=args.device,
        num_cases=int(args.num_cases),
        prefix_seconds=float(args.prefix_seconds),
        future_seconds=float(args.future_seconds),
        magnitude_mode=str(args.magnitude_mode),
        phase_seed_policy=str(args.phase_seed_policy),
        cases_json=Path(args.cases_json) if args.cases_json else None,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
