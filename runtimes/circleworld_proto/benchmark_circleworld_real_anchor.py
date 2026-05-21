from __future__ import annotations

import argparse
import hashlib
import json
import sys
import wave
from pathlib import Path
from typing import Any, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
RUNTIME = ROOT / "runtimes" / "circleworld_proto"
for path in (ROOT, RUNTIME):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from benchmark_audio_continuity import (
    DEFAULT_CONTINUITY_VIEW_SECONDS,
    DEFAULT_CONTINUITY_WINDOWS_SECONDS,
    analyze_wav,
    compare_continuity_summaries,
    parse_seconds_csv,
    summarize_continuity_rows,
)
from export_circleworld_audio import render_circleworld_audio, render_reference_audio


DEFAULT_CASES = {
    "airplane_takeoff": ROOT / "wav_files" / "soundbible_airplane-takeoff_c752b8ba8b.wav",
    "steam_engine": ROOT / "wav_files" / "soundbible_steam-engine-running_68e20d27d1.wav",
    "voice": ROOT / "wav_files" / "soundbible_mystic-chanting-4_a38f2bbe68.wav",
    "spooky_drone": ROOT / "wav_files" / "soundbible_spooky-drone_2efbfd965b.wav",
    "sax_like_clarinet": ROOT / "wav_files" / "freewavesamples_ensoniq-zr-76-clarinet-c5_19f8dd5ddd.wav",
}


def _load_cases_json(path: Path | None) -> dict[str, Path]:
    if path is None:
        return dict(DEFAULT_CASES)
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    return {str(name): Path(str(wav_path)) for name, wav_path in payload.items()}


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


def _wav_bytes(path: Path) -> bytes:
    return path.read_bytes()


def _sha256(path: Path) -> str:
    return hashlib.sha256(_wav_bytes(path)).hexdigest()


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
        arr = arr.mean(axis=1, keepdims=True)
    return arr.reshape(-1), sr


def _compare_audio(a_path: Path, b_path: Path) -> dict[str, float]:
    a, a_sr = _read_pcm16_mono(a_path)
    b, b_sr = _read_pcm16_mono(b_path)
    if a_sr != b_sr:
        raise RuntimeError(f"Sample-rate mismatch: {a_sr} vs {b_sr}")
    n = min(len(a), len(b))
    a = a[:n]
    b = b[:n]
    diff = a - b
    mse = float(np.mean(diff ** 2))
    mae = float(np.mean(np.abs(diff)))
    rms_a = float(np.sqrt(np.mean(a ** 2)))
    rms_b = float(np.sqrt(np.mean(b ** 2)))
    if np.std(a) < 1e-12 or np.std(b) < 1e-12:
        corr = 0.0
    else:
        corr = float(np.corrcoef(a, b)[0, 1])
    return {
        "num_samples": float(n),
        "mse": mse,
        "mae": mae,
        "corr": corr,
        "rms_ref": rms_a,
        "rms_out": rms_b,
    }


def run_benchmark(
    config_path: Path,
    out_dir: Path,
    device_name: str,
    clip_seconds: int,
    phase_blend: float,
    rerender_check: bool = True,
    cases: dict[str, Path] | None = None,
    include_continuity_diagnostics: bool = True,
    continuity_min_loop_seconds: float = 0.5,
    continuity_max_loop_seconds: float = 4.0,
    continuity_chunk_seconds: float = 2.0,
    continuity_windows: Sequence[float] | None = None,
    continuity_view_seconds: Sequence[float] | None = None,
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    reference_continuity_rows: list[dict[str, Any]] = []
    circleworld_continuity_rows: list[dict[str, Any]] = []
    active_cases = dict(cases or DEFAULT_CASES)
    normalized_continuity_windows = _normalize_seconds(continuity_windows, DEFAULT_CONTINUITY_WINDOWS_SECONDS)
    normalized_continuity_views = _normalize_seconds(continuity_view_seconds, DEFAULT_CONTINUITY_VIEW_SECONDS)

    for name, wav_path in active_cases.items():
        ref_path = out_dir / f"{name}_reference.wav"
        ref_meta = render_reference_audio(
            wav_path=wav_path,
            out_path=ref_path,
            device_name=device_name,
            clip_seconds_override=clip_seconds,
        )

        cw_path = out_dir / f"{name}_circleworld.wav"
        cw_meta = render_circleworld_audio(
            wav_path=wav_path,
            config_path=config_path,
            out_path=cw_path,
            device_name=device_name,
            phase_blend=phase_blend,
            clip_seconds_override=clip_seconds,
        )

        rerender_match = None
        rerender_hash = None
        if rerender_check:
            rerender_path = out_dir / f"{name}_circleworld_rerender.wav"
            render_circleworld_audio(
                wav_path=wav_path,
                config_path=config_path,
                out_path=rerender_path,
                device_name=device_name,
                phase_blend=phase_blend,
                clip_seconds_override=clip_seconds,
            )
            rerender_hash = _sha256(rerender_path)
            rerender_match = rerender_hash == _sha256(cw_path)

        macro_time_profile = None
        if include_continuity_diagnostics:
            reference_continuity = analyze_wav(
                wav_path=ref_path,
                min_loop_seconds=continuity_min_loop_seconds,
                max_loop_seconds=continuity_max_loop_seconds,
                chunk_seconds=continuity_chunk_seconds,
                continuity_windows=normalized_continuity_windows,
                view_seconds=normalized_continuity_views,
            )
            circleworld_continuity = analyze_wav(
                wav_path=cw_path,
                min_loop_seconds=continuity_min_loop_seconds,
                max_loop_seconds=continuity_max_loop_seconds,
                chunk_seconds=continuity_chunk_seconds,
                continuity_windows=normalized_continuity_windows,
                view_seconds=normalized_continuity_views,
            )
            reference_continuity_rows.append(reference_continuity)
            circleworld_continuity_rows.append(circleworld_continuity)
            macro_time_profile = {
                "reference": reference_continuity,
                "circleworld": circleworld_continuity,
            }

        row = {
            "name": name,
            "source_wav": str(wav_path),
            "reference_wav": str(ref_path),
            "circleworld_wav": str(cw_path),
            "reference_sha256": _sha256(ref_path),
            "circleworld_sha256": _sha256(cw_path),
            "circleworld_rerender_sha256": rerender_hash,
            "circleworld_bitwise_stable": rerender_match,
            **_compare_audio(ref_path, cw_path),
            "circleworld_meta": cw_meta,
            "reference_meta": ref_meta,
        }
        if macro_time_profile is not None:
            row["macro_time_profile"] = macro_time_profile
        rows.append(row)

    summary = {
        "runtime": "circleworld_proto",
        "schema": "circleworld_v0",
        "config_path": str(config_path),
        "out_dir": str(out_dir),
        "device": device_name,
        "clip_seconds": int(clip_seconds),
        "phase_blend": float(phase_blend),
        "rerender_check": bool(rerender_check),
        "cases": {name: str(path) for name, path in active_cases.items()},
        "mean_mse": float(sum(r["mse"] for r in rows) / len(rows)),
        "mean_mae": float(sum(r["mae"] for r in rows) / len(rows)),
        "mean_corr": float(sum(r["corr"] for r in rows) / len(rows)),
        "all_circleworld_bitwise_stable": bool(all(bool(r["circleworld_bitwise_stable"]) for r in rows if r["circleworld_bitwise_stable"] is not None)),
        "rows": rows,
    }
    if include_continuity_diagnostics and reference_continuity_rows and circleworld_continuity_rows:
        reference_summary = summarize_continuity_rows(
            reference_continuity_rows,
            folder=out_dir,
            pattern="*_reference.wav",
            min_loop_seconds=continuity_min_loop_seconds,
            max_loop_seconds=continuity_max_loop_seconds,
            chunk_seconds=continuity_chunk_seconds,
            continuity_windows=normalized_continuity_windows,
            view_seconds=normalized_continuity_views,
        )
        circleworld_summary = summarize_continuity_rows(
            circleworld_continuity_rows,
            folder=out_dir,
            pattern="*_circleworld.wav",
            min_loop_seconds=continuity_min_loop_seconds,
            max_loop_seconds=continuity_max_loop_seconds,
            chunk_seconds=continuity_chunk_seconds,
            continuity_windows=normalized_continuity_windows,
            view_seconds=normalized_continuity_views,
        )
        summary["macro_time_diagnostics"] = {
            "continuity_windows_seconds": [float(v) for v in normalized_continuity_windows],
            "continuity_view_seconds": [float(v) for v in normalized_continuity_views],
            "reference": reference_summary,
            "circleworld": circleworld_summary,
            "delta": compare_continuity_summaries(
                circleworld_summary,
                reference_summary,
                lhs_label="circleworld",
                rhs_label="reference",
            ),
        }
    (out_dir / "benchmark_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description="Benchmark Circleworld on real anchor audio using bit-comparable preprocessing.")
    ap.add_argument("--config", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--clip-seconds", type=int, default=10)
    ap.add_argument("--phase-blend", type=float, default=1.0)
    ap.add_argument("--no-rerender-check", action="store_true")
    ap.add_argument("--cases-json", default=None, help="Optional JSON file mapping case names to WAV paths.")
    ap.add_argument("--no-continuity-diagnostics", action="store_true")
    ap.add_argument("--continuity-min-loop-seconds", type=float, default=0.5)
    ap.add_argument("--continuity-max-loop-seconds", type=float, default=4.0)
    ap.add_argument("--continuity-chunk-seconds", type=float, default=2.0)
    ap.add_argument(
        "--continuity-windows",
        default="1,2,4",
        help="Comma-separated chunk sizes for additive continuity summaries.",
    )
    ap.add_argument(
        "--continuity-view-seconds",
        default="4,10",
        help="Comma-separated macro-time views to compute for each rendered clip.",
    )
    args = ap.parse_args()

    summary = run_benchmark(
        config_path=Path(args.config),
        out_dir=Path(args.out_dir),
        device_name=args.device,
        clip_seconds=args.clip_seconds,
        phase_blend=args.phase_blend,
        rerender_check=not bool(args.no_rerender_check),
        cases=_load_cases_json(Path(args.cases_json)) if args.cases_json else None,
        include_continuity_diagnostics=not bool(args.no_continuity_diagnostics),
        continuity_min_loop_seconds=float(args.continuity_min_loop_seconds),
        continuity_max_loop_seconds=float(args.continuity_max_loop_seconds),
        continuity_chunk_seconds=float(args.continuity_chunk_seconds),
        continuity_windows=parse_seconds_csv(args.continuity_windows, DEFAULT_CONTINUITY_WINDOWS_SECONDS),
        continuity_view_seconds=parse_seconds_csv(args.continuity_view_seconds, DEFAULT_CONTINUITY_VIEW_SECONDS),
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
