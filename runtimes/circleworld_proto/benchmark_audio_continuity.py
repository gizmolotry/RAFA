from __future__ import annotations

import argparse
import json
import wave
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np


DEFAULT_CONTINUITY_WINDOWS_SECONDS: tuple[float, ...] = (1.0, 2.0, 4.0)
DEFAULT_CONTINUITY_VIEW_SECONDS: tuple[float, ...] = (4.0, 10.0)
RECURRENCE_BAND_ORDER: tuple[str, ...] = ("low", "moderate", "high", "severe")
_SUMMARY_MEAN_KEYS: tuple[str, ...] = (
    "mean_duration_seconds",
    "mean_rms",
    "mean_chunk_count",
    "mean_loop_autocorr_peak",
    "mean_loop_period_seconds",
    "mean_adjacent_chunk_similarity",
    "mean_nonlocal_chunk_repeat",
    "mean_first_chunk_reentry",
    "mean_recurrence_score",
)
_SUMMARY_SCALAR_ALIASES: dict[str, str] = {
    "mean_duration_seconds": "duration_seconds",
    "mean_rms": "rms",
    "mean_chunk_count": "chunk_count",
    "mean_loop_autocorr_peak": "loop_autocorr_peak",
    "mean_loop_period_seconds": "loop_period_seconds",
    "mean_adjacent_chunk_similarity": "adjacent_chunk_similarity",
    "mean_nonlocal_chunk_repeat": "nonlocal_chunk_repeat",
    "mean_first_chunk_reentry": "first_chunk_reentry",
    "mean_recurrence_score": "recurrence_score",
}


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


def _seconds_label(seconds: float) -> str:
    rounded = round(float(seconds))
    if abs(float(seconds) - rounded) <= 1e-9:
        return f"{int(rounded)}s"
    return f"{float(seconds):g}s"


def _normalize_seconds(values: Sequence[float] | None, default: Sequence[float]) -> tuple[float, ...]:
    normalized: list[float] = []
    for raw in values if values is not None else default:
        value = float(raw)
        if value <= 0.0:
            continue
        if any(abs(value - prev) <= 1e-9 for prev in normalized):
            continue
        normalized.append(value)
    return tuple(sorted(normalized))


def parse_seconds_csv(raw: str | None, default: Sequence[float]) -> tuple[float, ...]:
    if raw is None:
        return _normalize_seconds(None, default)
    values: list[float] = []
    for piece in str(raw).split(","):
        token = piece.strip()
        if not token:
            continue
        values.append(float(token))
    return _normalize_seconds(values, default)


def _slice_signal(x: np.ndarray, sr: int, seconds: float) -> np.ndarray:
    if len(x) == 0:
        return x
    sample_count = max(1, int(round(float(seconds) * float(sr))))
    return x[: min(len(x), sample_count)]


def _recurrence_score(
    loop_peak: float,
    loop_period: float,
    max_loop_seconds: float,
    nonlocal_repeat: float,
    first_chunk_reentry: float,
) -> float:
    short_loop_pressure = 0.0
    if max_loop_seconds > 1e-9 and loop_period > 1e-9:
        short_loop_pressure = max(0.0, 1.0 - min(loop_period, max_loop_seconds) / max_loop_seconds)
    score = (
        0.40 * float(loop_peak)
        + 0.35 * float(nonlocal_repeat)
        + 0.15 * float(first_chunk_reentry)
        + 0.10 * float(short_loop_pressure)
    )
    return float(min(1.0, max(0.0, score)))


def _recurrence_band(score: float) -> str:
    value = float(score)
    if value < 0.40:
        return "low"
    if value < 0.58:
        return "moderate"
    if value < 0.74:
        return "high"
    return "severe"


def _delta_severity_band(max_abs_delta: float) -> str:
    value = abs(float(max_abs_delta))
    if value < 0.01:
        return "near_identity"
    if value < 0.03:
        return "minor_shift"
    if value < 0.07:
        return "moderate_shift"
    return "major_shift"


def _analyze_signal(
    x: np.ndarray,
    sr: int,
    min_loop_seconds: float,
    max_loop_seconds: float,
    chunk_seconds: float,
) -> dict[str, Any]:
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

    adjacent_chunk_similarity = float(np.mean(adj_sims)) if adj_sims else 0.0
    nonlocal_chunk_repeat = float(np.max(nonlocal_sims)) if nonlocal_sims else 0.0
    first_chunk_reentry = float(np.mean(first_vs_later)) if first_vs_later else 0.0
    recurrence_score = _recurrence_score(
        loop_peak=loop_peak,
        loop_period=loop_period,
        max_loop_seconds=max_loop_seconds,
        nonlocal_repeat=nonlocal_chunk_repeat,
        first_chunk_reentry=first_chunk_reentry,
    )

    return {
        "sample_rate": sr,
        "duration_seconds": duration,
        "rms": float(np.sqrt(np.mean(x * x) + 1e-12)),
        "analysis_chunk_seconds": float(chunk_seconds),
        "loop_autocorr_peak": loop_peak,
        "loop_period_seconds": loop_period,
        "adjacent_chunk_similarity": adjacent_chunk_similarity,
        "nonlocal_chunk_repeat": nonlocal_chunk_repeat,
        "first_chunk_reentry": first_chunk_reentry,
        "chunk_count": len(chunks),
        "recurrence_score": recurrence_score,
        "recurrence_severity_band": _recurrence_band(recurrence_score),
    }


def _mean(rows: Sequence[dict[str, Any]], key: str) -> float:
    if not rows:
        return 0.0
    return float(sum(float(row.get(key, 0.0)) for row in rows) / len(rows))


def _band_counts(rows: Sequence[dict[str, Any]]) -> dict[str, int]:
    counts = {band: 0 for band in RECURRENCE_BAND_ORDER}
    for row in rows:
        band = str(row.get("recurrence_severity_band", "low"))
        counts.setdefault(band, 0)
        counts[band] += 1
    return counts


def _dominant_band(counts: dict[str, int]) -> str:
    best_band = RECURRENCE_BAND_ORDER[0]
    best_count = counts.get(best_band, 0)
    for band in RECURRENCE_BAND_ORDER[1:]:
        count = counts.get(band, 0)
        if count > best_count:
            best_band = band
            best_count = count
    return best_band


def _summarize_metric_rows(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    counts = _band_counts(rows)
    mean_recurrence_score = _mean(rows, "recurrence_score")
    return {
        "mean_duration_seconds": _mean(rows, "duration_seconds"),
        "mean_rms": _mean(rows, "rms"),
        "mean_chunk_count": _mean(rows, "chunk_count"),
        "mean_loop_autocorr_peak": _mean(rows, "loop_autocorr_peak"),
        "mean_loop_period_seconds": _mean(rows, "loop_period_seconds"),
        "mean_adjacent_chunk_similarity": _mean(rows, "adjacent_chunk_similarity"),
        "mean_nonlocal_chunk_repeat": _mean(rows, "nonlocal_chunk_repeat"),
        "mean_first_chunk_reentry": _mean(rows, "first_chunk_reentry"),
        "mean_recurrence_score": mean_recurrence_score,
        "recurrence_band_counts": counts,
        "dominant_recurrence_band": _dominant_band(counts),
        "recurrence_severity_band": _recurrence_band(mean_recurrence_score),
    }


def summarize_continuity_rows(
    rows: Iterable[dict[str, Any]],
    *,
    folder: Path | str,
    pattern: str,
    min_loop_seconds: float,
    max_loop_seconds: float,
    chunk_seconds: float,
    continuity_windows: Sequence[float] | None = None,
    view_seconds: Sequence[float] | None = None,
) -> dict[str, Any]:
    row_list = list(rows)
    if not row_list:
        raise RuntimeError("Cannot summarize continuity rows: no rows provided")

    normalized_windows = _normalize_seconds(continuity_windows, DEFAULT_CONTINUITY_WINDOWS_SECONDS)
    normalized_views = _normalize_seconds(view_seconds, DEFAULT_CONTINUITY_VIEW_SECONDS)
    summary = {
        "folder": str(folder),
        "pattern": pattern,
        "min_loop_seconds": float(min_loop_seconds),
        "max_loop_seconds": float(max_loop_seconds),
        "chunk_seconds": float(chunk_seconds),
        "continuity_windows_seconds": [float(v) for v in normalized_windows],
        "continuity_view_seconds": [float(v) for v in normalized_views],
        "file_count": len(row_list),
        **_summarize_metric_rows(row_list),
        "rows": row_list,
    }
    summary["mean_loop_autocorr_peak"] = float(summary["mean_loop_autocorr_peak"])
    summary["mean_loop_period_seconds"] = float(summary["mean_loop_period_seconds"])
    summary["mean_adjacent_chunk_similarity"] = float(summary["mean_adjacent_chunk_similarity"])
    summary["mean_nonlocal_chunk_repeat"] = float(summary["mean_nonlocal_chunk_repeat"])
    summary["mean_first_chunk_reentry"] = float(summary["mean_first_chunk_reentry"])

    summary["window_summaries"] = {}
    for window_seconds in normalized_windows:
        label = _seconds_label(window_seconds)
        nested_rows = [row["window_summaries"][label] for row in row_list if label in row.get("window_summaries", {})]
        if not nested_rows:
            continue
        nested_summary = _summarize_metric_rows(nested_rows)
        nested_summary["file_count"] = len(nested_rows)
        nested_summary["chunk_seconds"] = float(window_seconds)
        summary["window_summaries"][label] = nested_summary

    summary["continuity_views"] = {}
    for requested_view_seconds in normalized_views:
        label = _seconds_label(requested_view_seconds)
        nested_rows = [row["continuity_views"][label] for row in row_list if label in row.get("continuity_views", {})]
        if not nested_rows:
            continue
        nested_summary = _summarize_metric_rows(nested_rows)
        nested_summary["file_count"] = len(nested_rows)
        nested_summary["requested_view_seconds"] = float(requested_view_seconds)
        nested_summary["mean_effective_view_seconds"] = _mean(nested_rows, "effective_view_seconds")
        nested_summary["chunk_seconds"] = float(chunk_seconds)
        nested_summary["window_summaries"] = {}
        for window_seconds in normalized_windows:
            window_label = _seconds_label(window_seconds)
            view_window_rows = [
                row["window_summaries"][window_label]
                for row in nested_rows
                if window_label in row.get("window_summaries", {})
            ]
            if not view_window_rows:
                continue
            view_window_summary = _summarize_metric_rows(view_window_rows)
            view_window_summary["file_count"] = len(view_window_rows)
            view_window_summary["chunk_seconds"] = float(window_seconds)
            nested_summary["window_summaries"][window_label] = view_window_summary
        summary["continuity_views"][label] = nested_summary

    return summary


def compare_continuity_summaries(
    lhs: dict[str, Any],
    rhs: dict[str, Any],
    *,
    lhs_label: str = "lhs",
    rhs_label: str = "rhs",
) -> dict[str, Any]:
    def metric_value(summary: dict[str, Any], mean_key: str) -> float:
        if mean_key in summary:
            return float(summary.get(mean_key, 0.0))
        scalar_key = _SUMMARY_SCALAR_ALIASES.get(mean_key)
        if scalar_key and scalar_key in summary:
            return float(summary.get(scalar_key, 0.0))
        return 0.0

    delta = {}
    for key in _SUMMARY_MEAN_KEYS:
        scalar_key = _SUMMARY_SCALAR_ALIASES.get(key)
        if key in lhs or key in rhs or (scalar_key and (scalar_key in lhs or scalar_key in rhs)):
            delta[key] = metric_value(lhs, key) - metric_value(rhs, key)
    max_abs_delta = max((abs(value) for value in delta.values()), default=0.0)
    result: dict[str, Any] = {
        "lhs_label": lhs_label,
        "rhs_label": rhs_label,
        "lhs_file_count": int(lhs.get("file_count", 0)),
        "rhs_file_count": int(rhs.get("file_count", 0)),
        "delta": delta,
        "max_abs_mean_delta": float(max_abs_delta),
        "delta_severity_band": _delta_severity_band(max_abs_delta),
        "lhs_recurrence_severity_band": str(lhs.get("recurrence_severity_band", lhs.get("dominant_recurrence_band", "low"))),
        "rhs_recurrence_severity_band": str(rhs.get("recurrence_severity_band", rhs.get("dominant_recurrence_band", "low"))),
        "recurrence_band_count_delta": {
            band: int(lhs.get("recurrence_band_counts", {}).get(band, 0)) - int(rhs.get("recurrence_band_counts", {}).get(band, 0))
            for band in RECURRENCE_BAND_ORDER
        },
        "window_summaries": {},
        "continuity_views": {},
    }
    for label in sorted(set(lhs.get("window_summaries", {})) | set(rhs.get("window_summaries", {}))):
        left_summary = lhs.get("window_summaries", {}).get(label)
        right_summary = rhs.get("window_summaries", {}).get(label)
        if left_summary is None or right_summary is None:
            continue
        result["window_summaries"][label] = compare_continuity_summaries(
            left_summary,
            right_summary,
            lhs_label=f"{lhs_label}:{label}",
            rhs_label=f"{rhs_label}:{label}",
        )
    for label in sorted(set(lhs.get("continuity_views", {})) | set(rhs.get("continuity_views", {}))):
        left_view = lhs.get("continuity_views", {}).get(label)
        right_view = rhs.get("continuity_views", {}).get(label)
        if left_view is None or right_view is None:
            continue
        view_compare = compare_continuity_summaries(
            left_view,
            right_view,
            lhs_label=f"{lhs_label}:{label}",
            rhs_label=f"{rhs_label}:{label}",
        )
        view_compare["requested_view_seconds"] = float(left_view.get("requested_view_seconds", right_view.get("requested_view_seconds", 0.0)))
        view_compare["mean_effective_view_seconds_delta"] = (
            float(left_view.get("mean_effective_view_seconds", 0.0)) - float(right_view.get("mean_effective_view_seconds", 0.0))
        )
        result["continuity_views"][label] = view_compare
    return result


def _continuity_metric_snapshot(summary: dict[str, Any]) -> dict[str, Any]:
    return {
        "sample_rate": int(summary.get("sample_rate", 0)),
        "duration_seconds": float(summary.get("duration_seconds", 0.0)),
        "rms": float(summary.get("rms", 0.0)),
        "analysis_chunk_seconds": float(summary.get("analysis_chunk_seconds", 0.0)),
        "chunk_count": int(summary.get("chunk_count", 0)),
        "loop_autocorr_peak": float(summary.get("loop_autocorr_peak", 0.0)),
        "loop_period_seconds": float(summary.get("loop_period_seconds", 0.0)),
        "adjacent_chunk_similarity": float(summary.get("adjacent_chunk_similarity", 0.0)),
        "nonlocal_chunk_repeat": float(summary.get("nonlocal_chunk_repeat", 0.0)),
        "first_chunk_reentry": float(summary.get("first_chunk_reentry", 0.0)),
        "recurrence_score": float(summary.get("recurrence_score", 0.0)),
        "recurrence_severity_band": str(summary.get("recurrence_severity_band", "low")),
    }


def continuity_snapshot(summary: dict[str, Any]) -> dict[str, Any]:
    snapshot = _continuity_metric_snapshot(summary)
    if "wav_path" in summary:
        snapshot["wav_path"] = str(summary.get("wav_path", ""))
    if "continuity_windows_seconds" in summary:
        snapshot["continuity_windows_seconds"] = [float(value) for value in summary.get("continuity_windows_seconds", [])]
    if "continuity_view_seconds" in summary:
        snapshot["continuity_view_seconds"] = [float(value) for value in summary.get("continuity_view_seconds", [])]
    if "recurrence_band_counts" in summary:
        snapshot["recurrence_band_counts"] = {
            str(band): int(count)
            for band, count in dict(summary.get("recurrence_band_counts", {})).items()
        }
    if "dominant_recurrence_band" in summary:
        snapshot["dominant_recurrence_band"] = str(summary.get("dominant_recurrence_band", "low"))
    if "requested_view_seconds" in summary:
        snapshot["requested_view_seconds"] = float(summary.get("requested_view_seconds", 0.0))
    if "effective_view_seconds" in summary:
        snapshot["effective_view_seconds"] = float(summary.get("effective_view_seconds", 0.0))
    if "window_summaries" in summary:
        snapshot["window_summaries"] = {
            str(label): _continuity_metric_snapshot(dict(window_summary))
            for label, window_summary in dict(summary.get("window_summaries", {})).items()
        }
    if "continuity_views" in summary:
        snapshot["continuity_views"] = {}
        for label, view_summary in dict(summary.get("continuity_views", {})).items():
            view_snapshot = _continuity_metric_snapshot(dict(view_summary))
            view_snapshot["requested_view_seconds"] = float(view_summary.get("requested_view_seconds", 0.0))
            view_snapshot["effective_view_seconds"] = float(view_summary.get("effective_view_seconds", 0.0))
            view_snapshot["window_summaries"] = {
                str(window_label): _continuity_metric_snapshot(dict(window_metrics))
                for window_label, window_metrics in dict(view_summary.get("window_summaries", {})).items()
            }
            snapshot["continuity_views"][str(label)] = view_snapshot
    return snapshot


def continuity_delta_snapshot(summary: dict[str, Any]) -> dict[str, Any]:
    snapshot = {
        "lhs_label": str(summary.get("lhs_label", "lhs")),
        "rhs_label": str(summary.get("rhs_label", "rhs")),
        "lhs_file_count": int(summary.get("lhs_file_count", 0)),
        "rhs_file_count": int(summary.get("rhs_file_count", 0)),
        "max_abs_mean_delta": float(summary.get("max_abs_mean_delta", 0.0)),
        "delta_severity_band": str(summary.get("delta_severity_band", "low")),
        "lhs_recurrence_severity_band": str(summary.get("lhs_recurrence_severity_band", "low")),
        "rhs_recurrence_severity_band": str(summary.get("rhs_recurrence_severity_band", "low")),
        "recurrence_band_count_delta": {
            str(band): int(count)
            for band, count in dict(summary.get("recurrence_band_count_delta", {})).items()
        },
        "delta": {
            str(key): float(value)
            for key, value in dict(summary.get("delta", {})).items()
        },
    }
    if "window_summaries" in summary:
        snapshot["window_summaries"] = {
            str(label): continuity_delta_snapshot(dict(window_summary))
            for label, window_summary in dict(summary.get("window_summaries", {})).items()
        }
    if "continuity_views" in summary:
        snapshot["continuity_views"] = {}
        for label, view_summary in dict(summary.get("continuity_views", {})).items():
            view_snapshot = continuity_delta_snapshot(dict(view_summary))
            if "requested_view_seconds" in view_summary:
                view_snapshot["requested_view_seconds"] = float(view_summary.get("requested_view_seconds", 0.0))
            if "mean_effective_view_seconds_delta" in view_summary:
                view_snapshot["mean_effective_view_seconds_delta"] = float(view_summary.get("mean_effective_view_seconds_delta", 0.0))
            snapshot["continuity_views"][str(label)] = view_snapshot
    return snapshot


def analyze_wav(
    wav_path: Path,
    min_loop_seconds: float,
    max_loop_seconds: float,
    chunk_seconds: float,
    continuity_windows: Sequence[float] | None = None,
    view_seconds: Sequence[float] | None = None,
) -> dict[str, Any]:
    x, sr = _read_pcm16_mono(wav_path)
    normalized_windows = _normalize_seconds(continuity_windows, DEFAULT_CONTINUITY_WINDOWS_SECONDS)
    normalized_views = _normalize_seconds(view_seconds, DEFAULT_CONTINUITY_VIEW_SECONDS)

    result = {
        "wav_path": str(wav_path),
        "continuity_windows_seconds": [float(v) for v in normalized_windows],
        "continuity_view_seconds": [float(v) for v in normalized_views],
        **_analyze_signal(
            x,
            sr=sr,
            min_loop_seconds=min_loop_seconds,
            max_loop_seconds=max_loop_seconds,
            chunk_seconds=chunk_seconds,
        ),
    }
    result["window_summaries"] = {
        _seconds_label(window_seconds): _analyze_signal(
            x,
            sr=sr,
            min_loop_seconds=min_loop_seconds,
            max_loop_seconds=max_loop_seconds,
            chunk_seconds=window_seconds,
        )
        for window_seconds in normalized_windows
    }
    result["continuity_views"] = {}
    for requested_view_seconds in normalized_views:
        label = _seconds_label(requested_view_seconds)
        view_signal = _slice_signal(x, sr=sr, seconds=requested_view_seconds)
        view_summary = _analyze_signal(
            view_signal,
            sr=sr,
            min_loop_seconds=min_loop_seconds,
            max_loop_seconds=max_loop_seconds,
            chunk_seconds=chunk_seconds,
        )
        view_summary["requested_view_seconds"] = float(requested_view_seconds)
        view_summary["effective_view_seconds"] = float(len(view_signal) / sr)
        view_summary["window_summaries"] = {
            _seconds_label(window_seconds): _analyze_signal(
                view_signal,
                sr=sr,
                min_loop_seconds=min_loop_seconds,
                max_loop_seconds=max_loop_seconds,
                chunk_seconds=window_seconds,
            )
            for window_seconds in normalized_windows
        }
        result["continuity_views"][label] = view_summary
    return result


def benchmark_folder(
    folder: Path,
    pattern: str,
    out_path: Path,
    min_loop_seconds: float,
    max_loop_seconds: float,
    chunk_seconds: float,
    continuity_windows: Sequence[float] | None = None,
    view_seconds: Sequence[float] | None = None,
) -> dict[str, Any]:
    wavs = sorted(folder.glob(pattern))
    rows = [
        analyze_wav(
            wav_path=wav_path,
            min_loop_seconds=min_loop_seconds,
            max_loop_seconds=max_loop_seconds,
            chunk_seconds=chunk_seconds,
            continuity_windows=continuity_windows,
            view_seconds=view_seconds,
        )
        for wav_path in wavs
    ]
    if not rows:
        raise RuntimeError(f"No WAVs matched pattern {pattern} in {folder}")

    summary = summarize_continuity_rows(
        rows,
        folder=folder,
        pattern=pattern,
        min_loop_seconds=min_loop_seconds,
        max_loop_seconds=max_loop_seconds,
        chunk_seconds=chunk_seconds,
        continuity_windows=continuity_windows,
        view_seconds=view_seconds,
    )
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
    ap.add_argument(
        "--continuity-windows",
        default="1,2,4",
        help="Comma-separated chunk sizes for additive multi-window continuity summaries.",
    )
    ap.add_argument(
        "--view-seconds",
        default="4,10",
        help="Comma-separated macro-time views to compute over truncated clips.",
    )
    args = ap.parse_args()

    summary = benchmark_folder(
        folder=Path(args.folder),
        pattern=args.pattern,
        out_path=Path(args.out),
        min_loop_seconds=float(args.min_loop_seconds),
        max_loop_seconds=float(args.max_loop_seconds),
        chunk_seconds=float(args.chunk_seconds),
        continuity_windows=parse_seconds_csv(args.continuity_windows, DEFAULT_CONTINUITY_WINDOWS_SECONDS),
        view_seconds=parse_seconds_csv(args.view_seconds, DEFAULT_CONTINUITY_VIEW_SECONDS),
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
