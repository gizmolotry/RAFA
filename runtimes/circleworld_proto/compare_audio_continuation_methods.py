from __future__ import annotations

import argparse
import json
import wave
from pathlib import Path
from typing import Any

import numpy as np


BASELINE_BY_MODE = {
    "prefix_hold": "baseline_prefix_magnitude_hold",
    "flat": "baseline_flat_magnitude",
}
METHODS = (
    "circleworld",
    "baseline_copy_last",
    "baseline_flat_magnitude",
    "baseline_prefix_magnitude_hold",
)


def _as_float(value: Any, default: float = 0.0) -> float:
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _read_wav(path: Path) -> tuple[np.ndarray, int]:
    with wave.open(str(path), "rb") as wf:
        sr = int(wf.getframerate())
        channels = int(wf.getnchannels())
        sampwidth = int(wf.getsampwidth())
        frames = int(wf.getnframes())
        raw = wf.readframes(frames)
    if sampwidth != 2:
        raise RuntimeError(f"Expected 16-bit PCM WAV for continuation compare, got {sampwidth} bytes: {path}")
    arr = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    arr = arr.reshape(-1, channels)
    if channels > 1:
        arr = arr.mean(axis=1)
    else:
        arr = arr[:, 0]
    return arr, sr


def _compare_audio(a_path: Path, b_path: Path) -> dict[str, float]:
    a, a_sr = _read_wav(a_path)
    b, b_sr = _read_wav(b_path)
    if a_sr != b_sr:
        raise RuntimeError(f"Sample-rate mismatch: {a_sr} vs {b_sr}")
    n = min(len(a), len(b))
    a = a[:n]
    b = b[:n]
    diff = a - b
    if np.std(a) < 1.0e-12 or np.std(b) < 1.0e-12:
        corr = 0.0
    else:
        corr = float(np.corrcoef(a, b)[0, 1])
    return {
        "num_samples": float(n),
        "corr": corr,
        "mae": float(np.mean(np.abs(diff))),
        "mse": float(np.mean(diff * diff)),
        "rms_a": float(np.sqrt(np.mean(a * a) + 1.0e-12)),
        "rms_b": float(np.sqrt(np.mean(b * b) + 1.0e-12)),
    }


def _mean(rows: list[dict[str, Any]], key: str) -> float:
    if not rows:
        return 0.0
    return float(sum(_as_float(row.get(key)) for row in rows) / len(rows))


def _parse_summary_arg(raw: str) -> tuple[str, Path]:
    if "=" in raw:
        label, path = raw.split("=", 1)
        return label.strip(), Path(path.strip())
    path = Path(raw)
    return path.parent.name, path


def _method_summary(summary: dict[str, Any], method: str) -> dict[str, Any]:
    block = summary.get("method_summary", {}).get(method, {})
    return block if isinstance(block, dict) else {}


def analyze_one(label: str, summary_path: Path) -> dict[str, Any]:
    payload = json.loads(summary_path.read_text(encoding="utf-8-sig"))
    rows = payload.get("rows", []) if isinstance(payload.get("rows"), list) else []
    mode = str(payload.get("magnitude_mode", ""))
    phase_seed_policy = str(payload.get("phase_seed_policy", "velocity"))
    carrier_method = BASELINE_BY_MODE.get(mode, "baseline_prefix_magnitude_hold")
    future_target_magnitude_reused = bool(payload.get("future_target_magnitude_reused"))
    target_future_stft_magnitude_accessed = bool(payload.get("target_future_stft_magnitude_accessed"))
    future_target_phase_reused = bool(payload.get("future_target_phase_reused"))
    target_future_stft_phase_accessed = bool(payload.get("target_future_stft_phase_accessed"))
    per_case: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        phase_flags = row.get("phase_seed_flags") if isinstance(row.get("phase_seed_flags"), dict) else {}
        methods = row.get("methods") if isinstance(row.get("methods"), dict) else {}
        cw = methods.get("circleworld", {})
        carrier = methods.get(carrier_method, {})
        copy_last = methods.get("baseline_copy_last", {})
        flat = methods.get("baseline_flat_magnitude", {})
        prefix = methods.get("baseline_prefix_magnitude_hold", {})
        if not all(isinstance(item, dict) for item in (cw, carrier, copy_last, flat, prefix)):
            continue
        cw_path = Path(str(cw.get("wav_path", "")))
        carrier_path = Path(str(carrier.get("wav_path", "")))
        target_path = Path(str(row.get("target_future_wav", "")))
        if not (cw_path.exists() and carrier_path.exists() and target_path.exists()):
            continue
        cw_target = cw.get("metrics") if isinstance(cw.get("metrics"), dict) else {}
        carrier_target = carrier.get("metrics") if isinstance(carrier.get("metrics"), dict) else {}
        copy_target = copy_last.get("metrics") if isinstance(copy_last.get("metrics"), dict) else {}
        flat_target = flat.get("metrics") if isinstance(flat.get("metrics"), dict) else {}
        prefix_target = prefix.get("metrics") if isinstance(prefix.get("metrics"), dict) else {}
        cw_vs_carrier = _compare_audio(cw_path, carrier_path)
        cw_vs_target = _compare_audio(cw_path, target_path)
        carrier_vs_target = _compare_audio(carrier_path, target_path)
        cw_mse = _as_float(cw_target.get("mse"))
        carrier_mse = _as_float(carrier_target.get("mse"))
        copy_mse = _as_float(copy_target.get("mse"))
        flat_mse = _as_float(flat_target.get("mse"))
        prefix_mse = _as_float(prefix_target.get("mse"))
        cw_corr = _as_float(cw_target.get("corr"))
        best_corr = max(
            _as_float(copy_target.get("corr"), -1.0),
            _as_float(flat_target.get("corr"), -1.0),
            _as_float(prefix_target.get("corr"), -1.0),
        )
        best_mse = min(copy_mse, flat_mse, prefix_mse)
        per_case.append(
            {
                "name": row.get("name"),
                "phase_seed_policy": row.get("phase_seed_policy", phase_seed_policy),
                "phase_seed_source": phase_flags.get("future_phase_source"),
                "future_target_magnitude_reused": bool(row.get("future_target_magnitude_reused", False)),
                "target_future_stft_magnitude_accessed": bool(row.get("target_future_stft_magnitude_accessed", False)),
                "future_target_phase_reused": bool(row.get("future_target_phase_reused", False)),
                "target_future_stft_phase_accessed": bool(row.get("target_future_stft_phase_accessed", False)),
                "carrier_method": carrier_method,
                "cw_vs_carrier_corr": cw_vs_carrier["corr"],
                "cw_vs_carrier_mae": cw_vs_carrier["mae"],
                "cw_vs_carrier_mse": cw_vs_carrier["mse"],
                "cw_target_corr": _as_float(cw_target.get("corr")),
                "carrier_target_corr": _as_float(carrier_target.get("corr")),
                "cw_target_mae": _as_float(cw_target.get("mae")),
                "carrier_target_mae": _as_float(carrier_target.get("mae")),
                "cw_target_mse": cw_mse,
                "carrier_target_mse": carrier_mse,
                "cw_minus_carrier_mse": cw_mse - carrier_mse,
                "cw_minus_best_baseline_mse": cw_mse - best_mse,
                "cw_minus_best_baseline_corr": cw_corr - best_corr,
                "cw_wins_corr": bool(cw_corr >= best_corr),
                "cw_wins_mse": bool(cw_mse <= best_mse),
                "cw_vs_target_direct_corr": cw_vs_target["corr"],
                "carrier_vs_target_direct_corr": carrier_vs_target["corr"],
            }
        )
    carrier_corr = _mean(per_case, "cw_vs_carrier_corr")
    carrier_mae = _mean(per_case, "cw_vs_carrier_mae")
    carrier_mse = _mean(per_case, "cw_vs_carrier_mse")
    cw_wins_corr = sum(1 for row in per_case if row.get("cw_wins_corr"))
    cw_wins_mse = sum(1 for row in per_case if row.get("cw_wins_mse"))
    lock_score = float(max(0.0, min(1.0, 0.70 * max(0.0, carrier_corr) + 0.30 * max(0.0, 1.0 - 1000.0 * carrier_mse))))
    cw_win_fraction_corr = float(cw_wins_corr / max(1, len(per_case)))
    cw_win_fraction_mse = float(cw_wins_mse / max(1, len(per_case)))
    leakage = (
        future_target_magnitude_reused
        or target_future_stft_magnitude_accessed
        or future_target_phase_reused
        or target_future_stft_phase_accessed
    )
    if leakage:
        status = "invalid_future_leakage"
    elif lock_score >= 0.95 and cw_win_fraction_corr <= 0.10:
        status = "carrier_locked"
    else:
        status = "needs_review"
    return {
        "label": label,
        "summary_path": str(summary_path),
        "magnitude_mode": mode,
        "phase_seed_policy": phase_seed_policy,
        "carrier_method": carrier_method,
        "case_count": len(per_case),
        "future_target_magnitude_reused": future_target_magnitude_reused,
        "target_future_stft_magnitude_accessed": target_future_stft_magnitude_accessed,
        "future_target_phase_reused": future_target_phase_reused,
        "target_future_stft_phase_accessed": target_future_stft_phase_accessed,
        "circleworld_summary": _method_summary(payload, "circleworld"),
        "carrier_summary": _method_summary(payload, carrier_method),
        "copy_last_summary": _method_summary(payload, "baseline_copy_last"),
        "mean_cw_vs_carrier_corr": carrier_corr,
        "mean_cw_vs_carrier_mae": carrier_mae,
        "mean_cw_vs_carrier_mse": carrier_mse,
        "mean_cw_minus_carrier_mse": _mean(per_case, "cw_minus_carrier_mse"),
        "mean_cw_minus_best_baseline_mse": _mean(per_case, "cw_minus_best_baseline_mse"),
        "mean_cw_minus_best_baseline_corr": _mean(per_case, "cw_minus_best_baseline_corr"),
        "cw_win_fraction_corr": cw_win_fraction_corr,
        "cw_win_fraction_mse": cw_win_fraction_mse,
        "carrier_lock_score": lock_score,
        "status": status,
        "worst_cases_by_corr_gap": sorted(per_case, key=lambda row: row["cw_minus_best_baseline_corr"])[:5],
        "rows": per_case,
    }


def write_markdown(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# Audio Continuation Method Compare",
        "",
        f"- status: `{payload['status']}`",
        f"- runs: `{len(payload['runs'])}`",
        "",
        "| run | mode | phase seed | cases | carrier | cw/carrier corr | cw/carrier MSE | cw-best corr | cw-best MSE | future mag | future phase | corr wins | mse wins | status |",
        "|---|---|---|---:|---|---:|---:|---:|---:|---|---|---:|---:|---|",
    ]
    for run in payload["runs"]:
        lines.append(
            "| {label} | {mode} | {seed} | {cases} | {carrier} | {ccorr:.6g} | {cmse:.6g} | {bcorr:.6g} | {bmse:.6g} | `{fmag}` | `{fphase}` | {wcorr:.6g} | {wmse:.6g} | {status} |".format(
                label=run["label"],
                mode=run["magnitude_mode"],
                seed=run["phase_seed_policy"],
                cases=run["case_count"],
                carrier=run["carrier_method"],
                ccorr=run["mean_cw_vs_carrier_corr"],
                cmse=run["mean_cw_vs_carrier_mse"],
                bcorr=run["mean_cw_minus_best_baseline_corr"],
                bmse=run["mean_cw_minus_best_baseline_mse"],
                fmag=run["future_target_magnitude_reused"],
                fphase=run["future_target_phase_reused"],
                wcorr=run["cw_win_fraction_corr"],
                wmse=run["cw_win_fraction_mse"],
                status=run["status"],
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description="Compare Circleworld continuation against generated carrier baselines.")
    ap.add_argument("--summary", action="append", required=True, help="label=audio_continuation_summary.json")
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()
    runs = [analyze_one(*_parse_summary_arg(raw)) for raw in args.summary]
    if runs and any(run["status"] == "invalid_future_leakage" for run in runs):
        status = "invalid_future_leakage"
    elif runs and all(run["status"] == "carrier_locked" for run in runs):
        status = "carrier_locked"
    else:
        status = "needs_review"
    payload = {
        "schema": "circleworld_audio_continuation_method_compare_v0",
        "status": status,
        "runs": runs,
    }
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "audio_continuation_method_compare.json"
    md_path = out_dir / "AUDIO_CONTINUATION_METHOD_COMPARE.md"
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    write_markdown(md_path, payload)
    print(json.dumps({"json": str(json_path), "markdown": str(md_path), "status": payload["status"]}, indent=2))


if __name__ == "__main__":
    main()
