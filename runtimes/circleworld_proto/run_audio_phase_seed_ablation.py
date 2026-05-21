from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

import torch

ROOT = Path(__file__).resolve().parents[2]
CORE = ROOT / "core"
LINEAGE = ROOT / "lineages" / "04_positive_replacement"
RUNTIME = ROOT / "runtimes" / "circleworld_proto"
for path in (ROOT, CORE, LINEAGE, RUNTIME):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from benchmark_audio_continuation import (  # noqa: E402
    MAGNITUDE_MODES,
    PHASE_SEED_POLICIES,
    _build_magnitude_canvas,
    _build_phase_seed_canvas,
    _compare_waveforms,
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


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _mean(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    return float(sum(float(value) for value in values) / len(values))


def _parse_csv(raw: str, allowed: Sequence[str], label: str) -> list[str]:
    values = [item.strip() for item in raw.split(",") if item.strip()]
    if not values:
        raise ValueError(f"At least one {label} is required")
    unknown = sorted(set(values) - set(allowed))
    if unknown:
        raise ValueError(f"Unknown {label}: {unknown}; expected one of {tuple(allowed)}")
    return values


def _fit_phase_frames(phase: torch.Tensor, frames: int) -> torch.Tensor:
    if phase.size(-1) == frames:
        return phase
    if phase.size(-1) > frames:
        return phase[..., :frames]
    pad = phase[..., -1:].expand(-1, -1, frames - phase.size(-1))
    return torch.cat([phase, pad], dim=-1)


def _seed_phase(
    *,
    policy: str,
    prefix: torch.Tensor,
    prefix_phase: torch.Tensor,
    future_frames: int,
    future_samples: int,
    sr: int,
    stft_cfg: dict[str, Any],
) -> tuple[torch.Tensor, dict[str, Any]]:
    return _build_phase_seed_canvas(
        prefix=prefix,
        prefix_phase=prefix_phase,
        future_frames=future_frames,
        future_samples=future_samples,
        sr=sr,
        stft_cfg=stft_cfg,
        policy=policy,
    )


def _phase_delta_stats(seed_phase: torch.Tensor, final_phase: torch.Tensor, prefix_frames: int) -> dict[str, float]:
    delta = _wrap_angle(final_phase - seed_phase)
    future = delta[..., prefix_frames:]
    if future.numel() == 0:
        future = delta
    return {
        "future_mean_abs_phase_delta": float(future.abs().mean().detach().cpu().item()),
        "future_rms_phase_delta": float(torch.sqrt(torch.mean(future * future)).detach().cpu().item()),
        "future_mean_cos_phase_delta": float(torch.cos(future).mean().detach().cpu().item()),
    }


def _aggregate(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["magnitude_mode"]), str(row["phase_seed_policy"]))].append(row)

    out: list[dict[str, Any]] = []
    for (mode, policy), group in sorted(grouped.items(), key=lambda item: (item[0][0], item[0][1])):
        circle_corr_delta = [_as_float(row["circle_target_corr"]) - _as_float(row["seed_target_corr"]) for row in group]
        circle_mse_delta = [_as_float(row["circle_target_mse"]) - _as_float(row["seed_target_mse"]) for row in group]
        circle_loop_delta = [
            _as_float(row["circle_loop_autocorr_peak"]) - _as_float(row["seed_loop_autocorr_peak"]) for row in group
        ]
        circle_reentry_delta = [
            _as_float(row.get("circle_first_chunk_reentry")) - _as_float(row.get("seed_first_chunk_reentry"))
            for row in group
        ]
        circle_corr_wins = [delta > 1.0e-6 for delta in circle_corr_delta]
        circle_mse_wins = [delta < -1.0e-9 for delta in circle_mse_delta]
        circle_reentry_wins = [delta <= 0.0 for delta in circle_reentry_delta]
        mean_corr_delta = _mean(circle_corr_delta)
        mean_mse_delta = _mean(circle_mse_delta)
        if mean_corr_delta > 0.02 and _mean([float(v) for v in circle_corr_wins]) >= 0.5:
            verdict = "circleworld_seed_policy_helpful_candidate"
        elif mean_mse_delta < -1.0e-4 and _mean([float(v) for v in circle_mse_wins]) >= 0.5:
            verdict = "circleworld_seed_policy_mse_helpful_candidate"
        elif abs(mean_corr_delta) <= 0.005 and abs(mean_mse_delta) <= 1.0e-4:
            verdict = "circleworld_near_seed_policy"
        else:
            verdict = "circleworld_changes_without_policy_gain"
        out.append(
            {
                "magnitude_mode": mode,
                "phase_seed_policy": policy,
                "case_count": len(group),
                "mean_seed_target_corr": _mean([_as_float(row["seed_target_corr"]) for row in group]),
                "mean_circle_target_corr": _mean([_as_float(row["circle_target_corr"]) for row in group]),
                "mean_seed_target_mae": _mean([_as_float(row["seed_target_mae"]) for row in group]),
                "mean_circle_target_mae": _mean([_as_float(row["circle_target_mae"]) for row in group]),
                "mean_seed_target_mse": _mean([_as_float(row["seed_target_mse"]) for row in group]),
                "mean_circle_target_mse": _mean([_as_float(row["circle_target_mse"]) for row in group]),
                "mean_circle_minus_seed_corr": mean_corr_delta,
                "mean_circle_minus_seed_mse": mean_mse_delta,
                "mean_seed_loop_autocorr_peak": _mean([_as_float(row["seed_loop_autocorr_peak"]) for row in group]),
                "mean_circle_loop_autocorr_peak": _mean([_as_float(row["circle_loop_autocorr_peak"]) for row in group]),
                "mean_circle_minus_seed_loop_autocorr_peak": _mean(circle_loop_delta),
                "mean_seed_first_chunk_reentry": _mean([_as_float(row.get("seed_first_chunk_reentry")) for row in group]),
                "mean_circle_first_chunk_reentry": _mean([_as_float(row.get("circle_first_chunk_reentry")) for row in group]),
                "mean_circle_minus_seed_first_chunk_reentry": _mean(circle_reentry_delta),
                "circle_corr_win_fraction": _mean([float(v) for v in circle_corr_wins]),
                "circle_mse_win_fraction": _mean([float(v) for v in circle_mse_wins]),
                "circle_reentry_win_fraction": _mean([float(v) for v in circle_reentry_wins]),
                "mean_circle_vs_seed_corr": _mean([_as_float(row["circle_vs_seed_corr"]) for row in group]),
                "mean_phase_abs_delta": _mean([_as_float(row["phase_delta_stats"]["future_mean_abs_phase_delta"]) for row in group]),
                "mean_phase_cos_delta": _mean([_as_float(row["phase_delta_stats"]["future_mean_cos_phase_delta"]) for row in group]),
                "verdict": verdict,
            }
        )
    return out


def _best_seed_policy(aggregate: list[dict[str, Any]]) -> dict[str, Any]:
    if not aggregate:
        return {}
    return max(aggregate, key=lambda row: (_as_float(row.get("mean_seed_target_corr")), -_as_float(row.get("mean_seed_target_mse"))))


def _seed_policy_spread(aggregate: list[dict[str, Any]]) -> dict[str, Any]:
    by_mode: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in aggregate:
        by_mode[str(row.get("magnitude_mode"))].append(row)
    rows: list[dict[str, Any]] = []
    max_corr_spread = 0.0
    max_mse_spread = 0.0
    for mode, group in sorted(by_mode.items()):
        seed_corr = [_as_float(row.get("mean_seed_target_corr")) for row in group]
        seed_mse = [_as_float(row.get("mean_seed_target_mse")) for row in group]
        circle_minus_seed = [abs(_as_float(row.get("mean_circle_minus_seed_corr"))) for row in group]
        corr_spread = max(seed_corr) - min(seed_corr) if seed_corr else 0.0
        mse_spread = max(seed_mse) - min(seed_mse) if seed_mse else 0.0
        max_corr_spread = max(max_corr_spread, corr_spread)
        max_mse_spread = max(max_mse_spread, mse_spread)
        rows.append(
            {
                "magnitude_mode": mode,
                "seed_corr_spread": corr_spread,
                "seed_mse_spread": mse_spread,
                "max_abs_circle_minus_seed_corr": max(circle_minus_seed) if circle_minus_seed else 0.0,
                "seed_policy_matters": bool(corr_spread >= 0.02 or mse_spread >= 1.0e-3),
            }
        )
    return {
        "max_seed_corr_spread": max_corr_spread,
        "max_seed_mse_spread": max_mse_spread,
        "seed_policy_matters": bool(max_corr_spread >= 0.02 or max_mse_spread >= 1.0e-3),
        "rows": rows,
    }


def _overall_status(aggregate: list[dict[str, Any]], spread: dict[str, Any]) -> str:
    if any(str(row.get("verdict", "")).endswith("helpful_candidate") for row in aggregate):
        return "circleworld_policy_helpful_candidate"
    if bool(spread.get("seed_policy_matters")):
        return "seed_policy_matters_circleworld_not_helpful"
    if aggregate and all(row.get("verdict") == "circleworld_near_seed_policy" for row in aggregate):
        return "seed_policy_dominates_circleworld"
    return "seed_policy_effect_needs_review"


def _future_access_flags(rows: Sequence[dict[str, Any]]) -> dict[str, bool]:
    magnitude_flags = [row.get("magnitude_flags", {}) for row in rows]
    phase_flags = [row.get("phase_seed_flags", {}) for row in rows]
    return {
        "future_target_magnitude_reused": bool(
            any(bool(flags.get("future_target_magnitude_reused")) for flags in magnitude_flags)
        ),
        "target_future_stft_magnitude_accessed": bool(
            any(bool(flags.get("target_future_stft_magnitude_accessed")) for flags in magnitude_flags)
        ),
        "future_target_phase_reused": bool(any(bool(flags.get("future_target_phase_reused")) for flags in phase_flags)),
        "target_future_stft_phase_accessed": bool(
            any(bool(flags.get("target_future_stft_phase_accessed")) for flags in phase_flags)
        ),
    }


def _run_case(
    *,
    name: str,
    wav_path: Path,
    circle_cfg: Any,
    stft_cfg: dict[str, Any],
    sr: int,
    device: torch.device,
    prefix_seconds: float,
    future_seconds: float,
    magnitude_modes: Sequence[str],
    phase_seed_policies: Sequence[str],
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
    prefix_frames = int(prefix_phase.size(-1))

    rows: list[dict[str, Any]] = []
    run_meta_by_policy: dict[str, Any] = {}
    for policy in phase_seed_policies:
        seed_phase, seed_flags = _seed_phase(
            policy=policy,
            prefix=prefix,
            prefix_phase=prefix_phase,
            future_frames=future_frames,
            future_samples=future_samples,
            sr=sr,
            stft_cfg=stft_cfg,
        )
        seed_z = phase_to_phasor(seed_phase)
        run_mode = (
            circle_cfg.branching_mode
            if circle_cfg.branching_mode in {"native_multimode", "native_multimode_childworld"}
            else "active_packets"
        )
        run = recurse_circleworld(seed_z, cfg=circle_cfg, depth=circle_cfg.recursion_depth, mode=run_mode)
        final_phase = _phase_to_angle(run["phase_state"])
        final_phase = torch.cat([prefix_phase, final_phase[..., prefix_frames:]], dim=-1)
        run_meta_by_policy[policy] = _jsonify(summarize_circleworld_run(run))
        phase_stats = _phase_delta_stats(seed_phase, final_phase, prefix_frames=prefix_frames)
        for mode in magnitude_modes:
            mag_canvas, mag_flags = _build_magnitude_canvas(prefix_mag, future_frames=future_frames, mode=mode)
            seed_future = _render_future_from_stft(
                mag_canvas=mag_canvas,
                phase_canvas=seed_phase,
                stft_cfg=stft_cfg,
                prefix_samples=prefix_samples,
                future_samples=future_samples,
            )
            circle_future = _render_future_from_stft(
                mag_canvas=mag_canvas,
                phase_canvas=final_phase,
                stft_cfg=stft_cfg,
                prefix_samples=prefix_samples,
                future_samples=future_samples,
            )
            seed_metrics = _compare_waveforms(target_future, seed_future)
            circle_metrics = _compare_waveforms(target_future, circle_future)
            circle_vs_seed = _compare_waveforms(seed_future, circle_future)
            seed_loop = _loop_reentry_metrics(seed_future, sr)
            circle_loop = _loop_reentry_metrics(circle_future, sr)
            rows.append(
                {
                    "name": name,
                    "source_wav": str(wav_path),
                    "magnitude_mode": mode,
                    "phase_seed_policy": policy,
                    "seed_target_corr": seed_metrics["corr"],
                    "circle_target_corr": circle_metrics["corr"],
                    "seed_target_mae": seed_metrics["mae"],
                    "circle_target_mae": circle_metrics["mae"],
                    "seed_target_mse": seed_metrics["mse"],
                    "circle_target_mse": circle_metrics["mse"],
                    "circle_vs_seed_corr": circle_vs_seed["corr"],
                    "circle_vs_seed_mse": circle_vs_seed["mse"],
                    "seed_loop_autocorr_peak": seed_loop["loop_autocorr_peak"],
                    "circle_loop_autocorr_peak": circle_loop["loop_autocorr_peak"],
                    "seed_first_chunk_reentry": seed_loop["first_chunk_reentry"],
                    "circle_first_chunk_reentry": circle_loop["first_chunk_reentry"],
                    "phase_delta_stats": phase_stats,
                    "phase_seed_flags": seed_flags,
                    "magnitude_flags": mag_flags,
                }
            )
    return {
        "name": name,
        "source_wav": str(wav_path),
        "prefix_samples": prefix_samples,
        "future_samples": future_samples,
        "prefix_stft_frames": int(prefix_frames),
        "future_stft_frames": int(future_frames),
        "rows": rows,
        "circleworld_meta_by_policy": run_meta_by_policy,
    }


def run_ablation(
    *,
    config_path: Path,
    out_dir: Path,
    device_name: str,
    num_cases: int,
    prefix_seconds: float,
    future_seconds: float,
    magnitude_modes: Sequence[str],
    phase_seed_policies: Sequence[str],
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
        raise RuntimeError("No cases selected for audio phase seed ablation")

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
            phase_seed_policies=phase_seed_policies,
        )
        for name, wav_path in cases
    ]
    rows = [row for case in case_rows for row in case["rows"]]
    aggregate = _aggregate(rows)
    best_seed = _best_seed_policy(aggregate)
    spread = _seed_policy_spread(aggregate)
    access_flags = _future_access_flags(rows)
    status = _overall_status(aggregate, spread)
    if any(access_flags.values()):
        status = "invalid_future_leakage"
    summary = _jsonify(
        {
            "runtime": "circleworld_proto",
            "schema": "circleworld_audio_phase_seed_ablation_v0",
            "status": status,
            "config_path": str(config_path),
            "out_dir": str(out_dir),
            "device": str(device),
            "requested_device": str(device_name),
            "sample_rate": sr,
            "prefix_seconds": float(prefix_seconds),
            "future_seconds": float(future_seconds),
            "magnitude_modes": list(magnitude_modes),
            "phase_seed_policies": list(phase_seed_policies),
            "case_count": len(case_rows),
            "future_target_audio_used_for_metrics_only": True,
            **access_flags,
            "best_seed_policy_by_corr": best_seed,
            "best_seed_policy_by_eval_target_corr": best_seed,
            "seed_policy_spread": spread,
            "aggregate": aggregate,
            "cases": case_rows,
        }
    )
    json_path = out_dir / "audio_phase_seed_ablation.json"
    md_path = out_dir / "AUDIO_PHASE_SEED_ABLATION.md"
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    md_path.write_text(_markdown_summary(summary), encoding="utf-8")
    return summary


def _markdown_summary(summary: dict[str, Any]) -> str:
    best = summary.get("best_seed_policy_by_eval_target_corr", summary.get("best_seed_policy_by_corr", {}))
    lines = [
        "# Audio Phase Seed Ablation",
        "",
        "This tests whether the no-future audio path is dominated by the seed phase policy before Circleworld rollout.",
        "",
        f"- status: `{summary['status']}`",
        f"- config: `{summary['config_path']}`",
        f"- cases: {summary['case_count']}",
        f"- phase seed policies: `{', '.join(summary['phase_seed_policies'])}`",
        f"- magnitude modes: `{', '.join(summary['magnitude_modes'])}`",
        f"- future target magnitude reused: `{summary['future_target_magnitude_reused']}`",
        f"- target future STFT magnitude accessed: `{summary['target_future_stft_magnitude_accessed']}`",
        f"- future target phase reused: `{summary.get('future_target_phase_reused', False)}`",
        f"- target future STFT phase accessed: `{summary.get('target_future_stft_phase_accessed', False)}`",
        f"- best seed policy by eval target corr: `{best.get('phase_seed_policy')}` / `{best.get('magnitude_mode')}`",
        f"- max seed corr spread: `{summary['seed_policy_spread']['max_seed_corr_spread']:.6g}`",
        f"- max seed MSE spread: `{summary['seed_policy_spread']['max_seed_mse_spread']:.6g}`",
        "",
        "## Aggregate",
        "",
        "| mode | seed policy | seed corr | circle corr | delta corr | seed MSE | circle MSE | delta MSE | seed reentry | circle reentry | delta reentry | reentry wins | circle/seed corr | phase abs delta | corr wins | MSE wins | verdict |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in summary["aggregate"]:
        lines.append(
            "| {mode} | {policy} | {seed_corr:.6g} | {circle_corr:.6g} | {dcorr:.6g} | {seed_mse:.6g} | {circle_mse:.6g} | {dmse:.6g} | {seed_reentry:.6g} | {circle_reentry:.6g} | {dreentry:.6g} | {wreentry:.6g} | {vseed:.6g} | {pdelta:.6g} | {wcorr:.6g} | {wmse:.6g} | {verdict} |".format(
                mode=row["magnitude_mode"],
                policy=row["phase_seed_policy"],
                seed_corr=float(row["mean_seed_target_corr"]),
                circle_corr=float(row["mean_circle_target_corr"]),
                dcorr=float(row["mean_circle_minus_seed_corr"]),
                seed_mse=float(row["mean_seed_target_mse"]),
                circle_mse=float(row["mean_circle_target_mse"]),
                dmse=float(row["mean_circle_minus_seed_mse"]),
                seed_reentry=float(row.get("mean_seed_first_chunk_reentry", 0.0)),
                circle_reentry=float(row.get("mean_circle_first_chunk_reentry", 0.0)),
                dreentry=float(row.get("mean_circle_minus_seed_first_chunk_reentry", 0.0)),
                wreentry=float(row.get("circle_reentry_win_fraction", 0.0)),
                vseed=float(row["mean_circle_vs_seed_corr"]),
                pdelta=float(row["mean_phase_abs_delta"]),
                wcorr=float(row["circle_corr_win_fraction"]),
                wmse=float(row["circle_mse_win_fraction"]),
                verdict=row["verdict"],
            )
        )
    lines.extend(
        [
            "",
            "## Interpretation Guard",
            "",
            "- A seed-policy claim is not a Circleworld claim; gain comes from the phase prior before rollout.",
            "- `best_seed_policy_by_eval_target_corr` is a retrospective held-out metric selector, not a deployable no-future policy oracle.",
            "- A Circleworld continuation claim requires the Circleworld row to beat its own seed carrier, not just a weaker seed policy.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Compare no-future audio phase seed policies before and after Circleworld rollout."
    )
    ap.add_argument("--config", required=True, help="Circleworld JSON config path.")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--num-cases", type=int, default=3)
    ap.add_argument("--prefix-seconds", type=float, default=2.0)
    ap.add_argument("--future-seconds", type=float, default=2.0)
    ap.add_argument("--magnitude-modes", default="prefix_hold,flat")
    ap.add_argument("--phase-seed-policies", default=",".join(PHASE_SEED_POLICIES))
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
        magnitude_modes=_parse_csv(str(args.magnitude_modes), MAGNITUDE_MODES, "magnitude mode"),
        phase_seed_policies=_parse_csv(str(args.phase_seed_policies), PHASE_SEED_POLICIES, "phase seed policy"),
        cases_json=Path(args.cases_json) if args.cases_json else None,
    )
    print(
        json.dumps(
            {
                "json": str(Path(args.out_dir) / "audio_phase_seed_ablation.json"),
                "markdown": str(Path(args.out_dir) / "AUDIO_PHASE_SEED_ABLATION.md"),
                "status": summary["status"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
