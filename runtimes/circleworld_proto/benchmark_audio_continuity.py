from __future__ import annotations

import argparse
import json
import math
import wave
from pathlib import Path
from typing import Any

import numpy as np


def _read_pcm16_mono(path: Path) -> tuple[np.ndarray, int]:
    with wave.open(str(path), "rb") as wf:
        sr = int(wf.getframerate())
        ch = int(wf.getnchannels())
        sw = int(wf.getsampwidth())
        n = int(wf.getnframes())
        raw = wf.readframes(n)
    if sw != 2:
        raise RuntimeError(f"Expected 16-bit PCM WAV, got sample width={sw}")
    arr = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    arr = arr.reshape(-1, ch)
    if ch > 1:
        arr = arr.mean(axis=1)
    else:
        arr = arr[:, 0]
    return arr, sr


def _frame_rms(x: np.ndarray, frame: int, hop: int) -> np.ndarray:
    if len(x) < frame:
        x = np.pad(x, (0, frame - len(x)))
    n = 1 + max(0, (len(x) - frame) // hop)
    vals = np.empty(n, dtype=np.float32)
    for i in range(n):
        start = i * hop
        seg = x[start : start + frame]
        vals[i] = float(np.sqrt(np.mean(seg * seg) + 1e-12))
    return vals


def _normalized_autocorr(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float64)
    x = x - x.mean()
    denom = np.dot(x, x)
    if denom <= 1e-12:
        return np.zeros(len(x), dtype=np.float64)
    ac = np.correlate(x, x, mode="full")[len(x) - 1 :]
    return ac / denom


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom <= 1e-12:
        return 0.0
    return float(np.dot(a, b) / denom)


def _chunk_vectors(x: np.ndarray, sr: int, chunk_seconds: float) -> list[np.ndarray]:
    chunk_len = max(1, int(round(sr * chunk_seconds)))
    chunks: list[np.ndarray] = []
    for start in range(0, len(x), chunk_len):
        seg = x[start : start + chunk_len]
        if len(seg) < chunk_len:
            break
        win = np.hanning(len(seg)).astype(np.float32)
        spec = np.fft.rfft(seg * win)
        vec = np.log1p(np.abs(spec)).astype(np.float32)
        chunks.append(vec)
    return chunks


def analyze_wav(
    wav_path: Path,
    min_loop_seconds: float,
    max_loop_seconds: float,
    chunk_seconds: float,
) -> dict[str, Any]:
    x, sr = _read_pcm16_mono(wav_path)
    duration = float(len(x) / sr)

    frame = 2048
    hop = 512
    env = _frame_rms(x, frame=frame, hop=hop)
    ac = _normalized_autocorr(env)
    hop_seconds = hop / float(sr)
    min_lag = max(1, int(round(min_loop_seconds / hop_seconds)))
    max_lag = min(len(ac) - 1, int(round(max_loop_seconds / hop_seconds)))
    loop_peak = 0.0
    loop_period = 0.0
    if max_lag > min_lag:
        win = ac[min_lag : max_lag + 1]
        idx = int(np.argmax(win))
        loop_peak = float(win[idx])
        loop_period = float((min_lag + idx) * hop_seconds)

    chunks = _chunk_vectors(x, sr=sr, chunk_seconds=chunk_seconds)
    adj_sims: list[float] = []
    nonlocal_sims: list[float] = []
    first_vs_later: list[float] = []
    for i in range(len(chunks)):
        if i + 1 < len(chunks):
            adj_sims.append(_cosine(chunks[i], chunks[i + 1]))
        if i >= 1:
            first_vs_later.append(_cosine(chunks[0], chunks[i]))
        for j in range(i + 2, len(chunks)):
            nonlocal_sims.append(_cosine(chunks[i], chunks[j]))

    return {
        "wav_path": str(wav_path),
        "sample_rate": sr,
        "duration_seconds": duration,
        "rms": float(np.sqrt(np.mean(x * x) + 1e-12)),
        "loop_autocorr_peak": loop_peak,
        "loop_period_seconds": loop_period,
        "adjacent_chunk_similarity": float(np.mean(adj_sims)) if adj_sims else 0.0,
        "nonlocal_chunk_repeat": float(np.max(nonlocal_sims)) if nonlocal_sims else 0.0,
        "first_chunk_reentry": float(np.mean(first_vs_later)) if first_vs_later else 0.0,
        "chunk_count": len(chunks),
    }


def benchmark_folder(
    folder: Path,
    pattern: str,
    out_path: Path,
    min_loop_seconds: float,
    max_loop_seconds: float,
    chunk_seconds: float,
) -> dict[str, Any]:
    wavs = sorted(folder.glob(pattern))
    rows = [
        analyze_wav(
            wav_path=wav_path,
            min_loop_seconds=min_loop_seconds,
            max_loop_seconds=max_loop_seconds,
            chunk_seconds=chunk_seconds,
        )
        for wav_path in wavs
    ]
    if not rows:
        raise RuntimeError(f"No WAVs matched pattern {pattern} in {folder}")

    def _mean(key: str) -> float:
        return float(sum(float(r[key]) for r in rows) / len(rows))

    summary = {
        "folder": str(folder),
        "pattern": pattern,
        "min_loop_seconds": min_loop_seconds,
        "max_loop_seconds": max_loop_seconds,
        "chunk_seconds": chunk_seconds,
        "file_count": len(rows),
        "mean_loop_autocorr_peak": _mean("loop_autocorr_peak"),
        "mean_loop_period_seconds": _mean("loop_period_seconds"),
        "mean_adjacent_chunk_similarity": _mean("adjacent_chunk_similarity"),
        "mean_nonlocal_chunk_repeat": _mean("nonlocal_chunk_repeat"),
        "mean_first_chunk_reentry": _mean("first_chunk_reentry"),
        "rows": rows,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description="Benchmark macro-time continuity and loop tendency for a WAV folder.")
    ap.add_argument("--folder", required=True)
    ap.add_argument("--pattern", default="*.wav")
    ap.add_argument("--out", required=True)
    ap.add_argument("--min-loop-seconds", type=float, default=0.5)
    ap.add_argument("--max-loop-seconds", type=float, default=4.0)
    ap.add_argument("--chunk-seconds", type=float, default=2.0)
    args = ap.parse_args()

    summary = benchmark_folder(
        folder=Path(args.folder),
        pattern=args.pattern,
        out_path=Path(args.out),
        min_loop_seconds=float(args.min_loop_seconds),
        max_loop_seconds=float(args.max_loop_seconds),
        chunk_seconds=float(args.chunk_seconds),
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
