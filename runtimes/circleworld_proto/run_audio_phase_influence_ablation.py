from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
CORE = ROOT / "core"
LINEAGE = ROOT / "lineages" / "04_positive_replacement"
RUNTIME = ROOT / "runtimes" / "circleworld_proto"
for path in (ROOT, CORE, LINEAGE, RUNTIME):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from common_io import as_float as _as_float  # noqa: E402
from common_io import mean as _mean  # noqa: E402
from common_io import parse_gain_csv as _parse_gain_csv  # noqa: E402
from benchmark_audio_continuation import (  # noqa: E402
    MAGNITUDE_MODES,
    _build_magnitude_canvas,
    _compare_waveforms,
    _extend_phase_from_prefix,
    _jsonify,
    _load_cases,
    _load_circle_cfg,
    _loop_reentry_metrics,
    _phase_to_angle,
    _prepare_anchor,
    _render_future_from_stft,
    _safe_device,
    _wrap_angle,
)
from circleworld import recurse_circleworld, summarize_circleworld_run  # noqa: E402
from config import load_config  # noqa: E402
from rafa_math_tools import phase_to_phasor  # noqa: E402
from stft_utils import compute_stft  # noqa: E402


DEFAULT_PHASE_GAINS = (-1.0, 0.0, 0.25, 0.5, 1.0, 2.0)


def _parse_csv_modes(raw: str) -> list[str]:
    values = [item.strip() for item in raw.split(",") if item.strip()]
    if not values:
        raise ValueError("At least one magnitude mode is required")
    unknown = sorted(set(values) - set(MAGNITUDE_MODES))
    if unknown:
        raise ValueError(f"Unknown magnitude modes: {unknown}; expected one of {MAGNITUDE_MODES}")
    return values


def _phase_delta_stats(initial_phase: torch.Tensor, final_phase: torch.Tensor, prefix_frames: int) -> dict[str, float]:
    delta = _wrap_angle(final_phase - initial_phase)
    future_delta = delta[..., prefix_frames:]
    if future_delta.numel() == 0:
        future_delta = delta
    abs_delta = future_delta.abs()
    cos_delta = torch.cos(future_delta)
    return {
        "future_mean_abs_phase_delta": float(abs_delta.mean().detach().cpu().item()),
        "future_rms_phase_delta": float(torch.sqrt(torch.mean(future_delta * future_delta)).detach().cpu().item()),
        "future_max_abs_phase_delta": float(abs_delta.max().detach().cpu().item()),
        "future_mean_cos_phase_delta": float(cos_delta.mean().detach().cpu().item()),
    }


def _blend_phase(
    prefix_phase: torch.Tensor,
    initial_phase: torch.Tensor,
    final_phase: torch.Tensor,
    gain: float,
) -> torch.Tensor:
    delta = _wrap_angle(final_phase - initial_phase)
    phase = _wrap_angle(initial_phase + float(gain) * delta)
    prefix_frames = int(prefix_phase.size(-1))
    return torch.cat([prefix_phase, phase[..., prefix_frames:]], dim=-1)


def _aggregate(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, float], list[dict[str, Any]]] = defaultdict(list)
    carrier_by_mode: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["magnitude_mode"]), float(row["phase_gain"]))].append(row)
        if abs(float(row["phase_gain"])) < 1.0e-12:
            carrier_by_mode[str(row["magnitude_mode"])].append(row)

    out: list[dict[str, Any]] = []
    for (mode, gain), group in sorted(grouped.items(), key=lambda item: (item[0][0], item[0][1])):
        carrier_rows = {str(row["name"]): row for row in carrier_by_mode.get(mode, [])}
        corr_deltas: list[float] = []
        mse_deltas: list[float] = []
        corr_wins = 0
        mse_wins = 0
        for row in group:
            carrier = carrier_rows.get(str(row["name"]))
            if carrier is None:
                continue
            corr_delta = _as_float(row["target_corr"]) - _as_float(carrier["target_corr"])
            mse_delta = _as_float(row["target_mse"]) - _as_float(carrier["target_mse"])
            corr_deltas.append(corr_delta)
            mse_deltas.append(mse_delta)
            corr_wins += int(corr_delta > 1.0e-6)
            mse_wins += int(mse_delta < -1.0e-9)

        case_count = len(group)
        mean_vs_gain0_corr = _mean([_as_float(row.get("vs_gain0_corr"), 1.0) for row in group])
        mean_vs_gain0_mse = _mean([_as_float(row.get("vs_gain0_mse"), 0.0) for row in group])
        mean_corr_delta = _mean(corr_deltas)
        mean_mse_delta = _mean(mse_deltas)
        corr_win_fraction = float(corr_wins / max(1, len(corr_deltas)))
        mse_win_fraction = float(mse_wins / max(1, len(mse_deltas)))
        if abs(gain) < 1.0e-12:
            verdict = "carrier_seed"
        elif mean_corr_delta > 0.02 and corr_win_fraction >= 0.50:
            verdict = "phase_delta_helpful_candidate"
        elif mean_mse_delta < -1.0e-4 and mse_win_fraction >= 0.50:
            verdict = "phase_delta_mse_helpful_candidate"
        elif mean_vs_gain0_corr >= 0.95 and mean_corr_delta <= 0.005:
            verdict = "carrier_locked"
        else:
            verdict = "phase_changes_without_target_gain"
        out.append(
            {
                "magnitude_mode": mode,
                "phase_gain": gain,
                "case_count": case_count,
                "mean_target_corr": _mean([_as_float(row["target_corr"]) for row in group]),
                "mean_target_mae": _mean([_as_float(row["target_mae"]) for row in group]),
                "mean_target_mse": _mean([_as_float(row["target_mse"]) for row in group]),
                "mean_loop_autocorr_peak": _mean([_as_float(row["loop_autocorr_peak"]) for row in group]),
                "mean_first_chunk_reentry": _mean([_as_float(row["first_chunk_reentry"]) for row in group]),
                "mean_vs_gain0_corr": mean_vs_gain0_corr,
                "mean_vs_gain0_mse": mean_vs_gain0_mse,
                "mean_target_corr_delta_vs_gain0": mean_corr_delta,
                "mean_target_mse_delta_vs_gain0": mean_mse_delta,
                "target_corr_win_fraction_vs_gain0": corr_win_fraction,
                "target_mse_win_fraction_vs_gain0": mse_win_fraction,
                "verdict": verdict,
            }
        )
    return out


def _overall_status(aggregate: list[dict[str, Any]]) -> str:
    nonzero = [row for row in aggregate if abs(float(row["phase_gain"])) > 1.0e-12]
    if any(str(row["verdict"]).endswith("helpful_candidate") for row in nonzero):
        return "phase_influence_positive_candidate"
    if nonzero and all(row["verdict"] == "carrier_locked" for row in nonzero):
        return "phase_delta_inert_or_carrier_locked"
    if nonzero and all(row["verdict"] in {"carrier_locked", "phase_changes_without_target_gain"} for row in nonzero):
        return "phase_changes_without_target_gain"
    return "needs_review"


def _run_case(
    name: str,
    wav_path: Path,
    circle_cfg: Any,
    stft_cfg: dict[str, Any],
    sr: int,
    device: torch.device,
    prefix_seconds: float,
    future_seconds: float,
    magnitude_modes: Sequence[str],
    phase_gains: Sequence[float],
) -> dict[str, Any]:
    prefix_samples = max(1, int(round(prefix_seconds * sr)))
    future_samples = max(1, int(round(future_seconds * sr)))
    total_samples = prefix_samples + future_samples
    wav = _prepare_anchor(wav_path, target_sr=sr, total_samples=total_samples)
    prefix = wav[..., :prefix_samples].to(device)
    target_future = wav[..., prefix_samples:total_samples].reshape(-1).to(device)

    prefix_mag, prefix_phase = compute_stft(prefix, stft_cfg)
    hop = int(stft_cfg["hop"])
    future_frames = max(1, int(math.ceil(float(future_samples) / float(hop))))
    initial_phase = _extend_phase_from_prefix(prefix_phase, future_frames=future_frames)
    initial_z = phase_to_phasor(initial_phase)
    run_mode = (
        circle_cfg.branching_mode
        if circle_cfg.branching_mode in {"native_multimode", "native_multimode_childworld"}
        else "active_packets"
    )
    run = recurse_circleworld(initial_z, cfg=circle_cfg, depth=circle_cfg.recursion_depth, mode=run_mode)
    final_phase = _phase_to_angle(run["phase_state"])
    final_phase = torch.cat([prefix_phase, final_phase[..., prefix_phase.size(-1) :]], dim=-1)
    phase_stats = _phase_delta_stats(initial_phase, final_phase, prefix_frames=int(prefix_phase.size(-1)))

    rows: list[dict[str, Any]] = []
    carrier_future_by_mode: dict[str, torch.Tensor] = {}
    for mode in magnitude_modes:
        mag_canvas, flags = _build_magnitude_canvas(prefix_mag, future_frames=future_frames, mode=mode)
        carrier_phase = _blend_phase(prefix_phase, initial_phase, final_phase, gain=0.0)
        carrier_future = _render_future_from_stft(
            mag_canvas=mag_canvas,
            phase_canvas=carrier_phase,
            stft_cfg=stft_cfg,
            prefix_samples=prefix_samples,
            future_samples=future_samples,
        )
        carrier_future_by_mode[mode] = carrier_future
        for gain in phase_gains:
            phase = _blend_phase(prefix_phase, initial_phase, final_phase, gain=float(gain))
            pred = _render_future_from_stft(
                mag_canvas=mag_canvas,
                phase_canvas=phase,
                stft_cfg=stft_cfg,
                prefix_samples=prefix_samples,
                future_samples=future_samples,
            )
            target_metrics = _compare_waveforms(target_future, pred)
            carrier_metrics = _compare_waveforms(carrier_future_by_mode[mode], pred)
            loop = _loop_reentry_metrics(pred, sr)
            rows.append(
                {
                    "name": name,
                    "source_wav": str(wav_path),
                    "magnitude_mode": mode,
                    "phase_gain": float(gain),
                    "target_corr": target_metrics["corr"],
                    "target_mae": target_metrics["mae"],
                    "target_mse": target_metrics["mse"],
                    "target_rms": target_metrics["target_rms"],
                    "output_rms": target_metrics["output_rms"],
                    "vs_gain0_corr": carrier_metrics["corr"],
                    "vs_gain0_mae": carrier_metrics["mae"],
                    "vs_gain0_mse": carrier_metrics["mse"],
                    "loop_autocorr_peak": loop["loop_autocorr_peak"],
                    "first_chunk_reentry": loop["first_chunk_reentry"],
                    "magnitude_flags": flags,
                }
            )
    return {
        "name": name,
        "source_wav": str(wav_path),
        "prefix_samples": prefix_samples,
        "future_samples": future_samples,
        "prefix_stft_frames": int(prefix_mag.size(-1)),
        "future_stft_frames": int(future_frames),
        "phase_delta_stats": phase_stats,
        "circleworld_meta": _jsonify(summarize_circleworld_run(run)),
        "rows": rows,
    }


def run_ablation(
    config_path: Path,
    out_dir: Path,
    device_name: str,
    num_cases: int,
    prefix_seconds: float,
    future_seconds: float,
    magnitude_modes: Sequence[str],
    phase_gains: Sequence[float],
    cases_json: Path | None = None,
) -> dict[str, Any]:
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
        raise RuntimeError("No cases selected for audio phase influence ablation")

    case_rows = [
        _run_case(
            name=name,
            wav_path=wav_path,
            circle_cfg=circle_cfg,
            stft_cfg=stft_cfg,
            sr=sr,
            device=device,
            prefix_seconds=prefix_seconds,
            future_seconds=future_seconds,
            magnitude_modes=magnitude_modes,
            phase_gains=phase_gains,
        )
        for name, wav_path in cases
    ]
    flat_rows = [row for case in case_rows for row in case["rows"]]
    aggregate = _aggregate(flat_rows)
    status = _overall_status(aggregate)
    summary = _jsonify(
        {
            "runtime": "circleworld_proto",
            "schema": "circleworld_audio_phase_influence_ablation_v0",
            "status": status,
            "config_path": str(config_path),
            "out_dir": str(out_dir),
            "device": str(device),
            "requested_device": device_name,
            "sample_rate": sr,
            "prefix_seconds": float(prefix_seconds),
            "future_seconds": float(future_seconds),
            "magnitude_modes": list(magnitude_modes),
            "phase_gains": [float(gain) for gain in phase_gains],
            "case_count": len(case_rows),
            "future_target_audio_used_for_metrics_only": True,
            "future_target_magnitude_reused": False,
            "target_future_stft_magnitude_accessed": False,
            "mean_future_abs_phase_delta": _mean(
                [_as_float(case["phase_delta_stats"]["future_mean_abs_phase_delta"]) for case in case_rows]
            ),
            "mean_future_cos_phase_delta": _mean(
                [_as_float(case["phase_delta_stats"]["future_mean_cos_phase_delta"]) for case in case_rows]
            ),
            "aggregate": aggregate,
            "cases": case_rows,
        }
    )
    json_path = out_dir / "audio_phase_influence_ablation.json"
    md_path = out_dir / "AUDIO_PHASE_INFLUENCE_ABLATION.md"
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    md_path.write_text(_markdown_summary(summary), encoding="utf-8")
    return summary


def _markdown_summary(summary: dict[str, Any]) -> str:
    lines = [
        "# Audio Phase Influence Ablation",
        "",
        "This isolates Circleworld phase-law contribution from the magnitude carrier.",
        "Gain `0.0` is the prefix phase-velocity carrier; nonzero gains scale Circleworld's phase delta away from that carrier.",
        "",
        f"- status: `{summary['status']}`",
        f"- config: `{summary['config_path']}`",
        f"- cases: {summary['case_count']}",
        f"- magnitude modes: `{', '.join(summary['magnitude_modes'])}`",
        f"- phase gains: `{', '.join(str(g) for g in summary['phase_gains'])}`",
        f"- future target magnitude reused: `{summary['future_target_magnitude_reused']}`",
        f"- target future STFT magnitude accessed: `{summary['target_future_stft_magnitude_accessed']}`",
        f"- mean future abs phase delta: `{summary['mean_future_abs_phase_delta']:.6g}`",
        f"- mean future cos phase delta: `{summary['mean_future_cos_phase_delta']:.6g}`",
        "",
        "## Aggregate",
        "",
        "| mode | gain | cases | corr | MAE | MSE | vs gain0 corr | vs gain0 MSE | corr delta | MSE delta | corr wins | MSE wins | verdict |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in summary["aggregate"]:
        lines.append(
            "| {mode} | {gain:.6g} | {cases} | {corr:.6g} | {mae:.6g} | {mse:.6g} | {vcorr:.6g} | {vmse:.6g} | {dcorr:.6g} | {dmse:.6g} | {wcorr:.6g} | {wmse:.6g} | {verdict} |".format(
                mode=row["magnitude_mode"],
                gain=float(row["phase_gain"]),
                cases=int(row["case_count"]),
                corr=float(row["mean_target_corr"]),
                mae=float(row["mean_target_mae"]),
                mse=float(row["mean_target_mse"]),
                vcorr=float(row["mean_vs_gain0_corr"]),
                vmse=float(row["mean_vs_gain0_mse"]),
                dcorr=float(row["mean_target_corr_delta_vs_gain0"]),
                dmse=float(row["mean_target_mse_delta_vs_gain0"]),
                wcorr=float(row["target_corr_win_fraction_vs_gain0"]),
                wmse=float(row["target_mse_win_fraction_vs_gain0"]),
                verdict=row["verdict"],
            )
        )
    lines.extend(
        [
            "",
            "## Interpretation Guard",
            "",
            "- A positive phase-law claim requires nonzero gains to beat gain `0.0` on target metrics, not merely differ from it.",
            "- If nonzero gains stay highly correlated with gain `0.0`, the continuation is carrier-locked or phase-delta-inert under this assay.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Scale Circleworld phase deltas during no-future-magnitude audio continuation."
    )
    ap.add_argument("--config", required=True, help="Circleworld JSON config path.")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--num-cases", type=int, default=3)
    ap.add_argument("--prefix-seconds", type=float, default=2.0)
    ap.add_argument("--future-seconds", type=float, default=2.0)
    ap.add_argument("--magnitude-modes", default="prefix_hold,flat")
    ap.add_argument("--phase-gains", default=",".join(str(value) for value in DEFAULT_PHASE_GAINS))
    ap.add_argument(
        "--cases-json",
        default=None,
        help="Optional JSON mapping case names to real anchor WAV paths. Defaults to the expanded Circleworld anchors if present.",
    )
    args = ap.parse_args()
    summary = run_ablation(
        config_path=Path(args.config),
        out_dir=Path(args.out_dir),
        device_name=str(args.device),
        num_cases=int(args.num_cases),
        prefix_seconds=float(args.prefix_seconds),
        future_seconds=float(args.future_seconds),
        magnitude_modes=_parse_csv_modes(str(args.magnitude_modes)),
        phase_gains=_parse_gain_csv(
            str(args.phase_gains),
            empty_message="At least one phase gain is required",
            zero_epsilon=None,
        ),
        cases_json=Path(args.cases_json) if args.cases_json else None,
    )
    print(
        json.dumps(
            {
                "json": str(Path(args.out_dir) / "audio_phase_influence_ablation.json"),
                "markdown": str(Path(args.out_dir) / "AUDIO_PHASE_INFLUENCE_ABLATION.md"),
                "status": summary["status"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
