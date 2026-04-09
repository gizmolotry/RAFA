from __future__ import annotations

import argparse
import hashlib
import json
import sys
import wave
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
RUNTIME = ROOT / "runtimes" / "circleworld_proto"
for path in (ROOT, RUNTIME):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from export_circleworld_audio import render_circleworld_audio, render_reference_audio


DEFAULT_CASES = {
    "airplane_takeoff": ROOT / "wav_files" / "soundbible_airplane-takeoff_c752b8ba8b.wav",
    "steam_engine": ROOT / "wav_files" / "soundbible_steam-engine-running_68e20d27d1.wav",
    "voice": ROOT / "wav_files" / "soundbible_mystic-chanting-4_a38f2bbe68.wav",
    "spooky_drone": ROOT / "wav_files" / "soundbible_spooky-drone_2efbfd965b.wav",
    "sax_like_clarinet": ROOT / "wav_files" / "freewavesamples_ensoniq-zr-76-clarinet-c5_19f8dd5ddd.wav",
}


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
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []

    for name, wav_path in DEFAULT_CASES.items():
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
        "mean_mse": float(sum(r["mse"] for r in rows) / len(rows)),
        "mean_mae": float(sum(r["mae"] for r in rows) / len(rows)),
        "mean_corr": float(sum(r["corr"] for r in rows) / len(rows)),
        "all_circleworld_bitwise_stable": bool(all(bool(r["circleworld_bitwise_stable"]) for r in rows if r["circleworld_bitwise_stable"] is not None)),
        "rows": rows,
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
    args = ap.parse_args()

    summary = run_benchmark(
        config_path=Path(args.config),
        out_dir=Path(args.out_dir),
        device_name=args.device,
        clip_seconds=args.clip_seconds,
        phase_blend=args.phase_blend,
        rerender_check=not bool(args.no_rerender_check),
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
