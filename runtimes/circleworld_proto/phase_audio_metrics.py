from __future__ import annotations

from typing import Any

import numpy as np
import torch


def compare_waveforms(target: torch.Tensor, pred: torch.Tensor) -> dict[str, float]:
    """Compare two waveform tensors with stable audio-continuation metrics."""
    n = min(int(target.numel()), int(pred.numel()))
    a = target[:n].detach().cpu().float().numpy()
    b = pred[:n].detach().cpu().float().numpy()
    diff = a - b
    if n == 0 or np.std(a) < 1e-12 or np.std(b) < 1e-12:
        corr = 0.0
    else:
        corr = float(np.corrcoef(a, b)[0, 1])
    return {
        "num_samples": float(n),
        "corr": corr,
        "mae": float(np.mean(np.abs(diff))) if n else 0.0,
        "mse": float(np.mean(diff * diff)) if n else 0.0,
        "target_rms": float(np.sqrt(np.mean(a * a) + 1e-12)) if n else 0.0,
        "output_rms": float(np.sqrt(np.mean(b * b) + 1e-12)) if n else 0.0,
    }


def frame_rms(x: np.ndarray, frame: int, hop: int) -> np.ndarray:
    if len(x) < frame:
        x = np.pad(x, (0, frame - len(x)))
    n = 1 + max(0, (len(x) - frame) // hop)
    out = np.empty(n, dtype=np.float32)
    for i in range(n):
        seg = x[i * hop : i * hop + frame]
        out[i] = float(np.sqrt(np.mean(seg * seg) + 1e-12))
    return out


def normalized_autocorr(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float64)
    x = x - x.mean()
    denom = float(np.dot(x, x))
    if denom <= 1e-12:
        return np.zeros(len(x), dtype=np.float64)
    return np.correlate(x, x, mode="full")[len(x) - 1 :] / denom


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom <= 1e-12:
        return 0.0
    return float(np.dot(a, b) / denom)


def chunk_vectors(x: np.ndarray, sr: int, chunk_seconds: float) -> list[np.ndarray]:
    chunk_len = max(1, int(round(sr * chunk_seconds)))
    chunks = []
    for start in range(0, len(x), chunk_len):
        seg = x[start : start + chunk_len]
        if len(seg) < chunk_len:
            break
        spec = np.fft.rfft(seg * np.hanning(len(seg)).astype(np.float32))
        chunks.append(np.log1p(np.abs(spec)).astype(np.float32))
    return chunks


def loop_reentry_metrics(wav: torch.Tensor, sr: int) -> dict[str, float]:
    """Estimate waveform loop/reentry pressure without target/future access."""
    x = wav.detach().cpu().float().view(-1).numpy()
    duration = float(len(x) / float(sr)) if sr else 0.0
    if len(x) < max(16, sr // 10):
        return {
            "duration_seconds": duration,
            "rms": float(np.sqrt(np.mean(x * x) + 1e-12)) if len(x) else 0.0,
            "loop_autocorr_peak": 0.0,
            "loop_period_seconds": 0.0,
            "adjacent_chunk_similarity": 0.0,
            "nonlocal_chunk_repeat": 0.0,
            "first_chunk_reentry": 0.0,
        }

    frame = min(2048, max(64, len(x) // 8))
    hop = max(16, frame // 4)
    env = frame_rms(x, frame=frame, hop=hop)
    ac = normalized_autocorr(env)
    hop_seconds = hop / float(sr)
    min_lag = max(1, int(round(0.25 / hop_seconds)))
    max_lag = min(len(ac) - 1, int(round(min(4.0, max(0.25, duration * 0.75)) / hop_seconds)))
    loop_peak = 0.0
    loop_period = 0.0
    if max_lag > min_lag:
        win = ac[min_lag : max_lag + 1]
        idx = int(np.argmax(win))
        loop_peak = float(win[idx])
        loop_period = float((min_lag + idx) * hop_seconds)

    chunk_seconds = min(1.0, max(0.1, duration / 4.0))
    chunks = chunk_vectors(x, sr=sr, chunk_seconds=chunk_seconds)
    adjacent = [cosine(chunks[i], chunks[i + 1]) for i in range(len(chunks) - 1)]
    nonlocal_repeat = [
        cosine(chunks[i], chunks[j])
        for i in range(len(chunks))
        for j in range(i + 2, len(chunks))
    ]
    first_reentry = [cosine(chunks[0], chunks[i]) for i in range(1, len(chunks))] if chunks else []
    return {
        "duration_seconds": duration,
        "rms": float(np.sqrt(np.mean(x * x) + 1e-12)),
        "loop_autocorr_peak": loop_peak,
        "loop_period_seconds": loop_period,
        "adjacent_chunk_similarity": float(np.mean(adjacent)) if adjacent else 0.0,
        "nonlocal_chunk_repeat": float(np.max(nonlocal_repeat)) if nonlocal_repeat else 0.0,
        "first_chunk_reentry": float(np.mean(first_reentry)) if first_reentry else 0.0,
    }


def aggregate_audio_metrics(rows: list[dict[str, Any]], key: str = "metrics") -> dict[str, float]:
    metrics = [row.get(key, {}) for row in rows if isinstance(row.get(key, {}), dict)]
    return {
        "mean_corr": float(np.mean([m.get("corr", 0.0) for m in metrics])) if metrics else 0.0,
        "mean_mae": float(np.mean([m.get("mae", 0.0) for m in metrics])) if metrics else 0.0,
        "mean_mse": float(np.mean([m.get("mse", 0.0) for m in metrics])) if metrics else 0.0,
    }
