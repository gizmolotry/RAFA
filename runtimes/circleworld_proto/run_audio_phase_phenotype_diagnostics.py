from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
RUNTIME = ROOT / "runtimes" / "circleworld_proto"
for path in (ROOT, RUNTIME):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from benchmark_audio_continuation import _loop_reentry_metrics, _prepare_anchor  # noqa: E402
from config import load_config  # noqa: E402
from run_audio_delta_mechanism_probe import OUTPUT_JSON as PROBE_JSON  # noqa: E402
from stft_utils import compute_stft  # noqa: E402


TOKENBURST_ROOT = Path(r"C:\Users\Andrew\AppData\Local\Temp\rafa_tokenburst")
DEFAULT_MANIFEST = (
    TOKENBURST_ROOT
    / "audio_steady_phenotype_manifest_cuda_2026_05_06"
    / "audio_steady_phenotype_manifest.json"
)
OUTPUT_JSON = "audio_phase_phenotype_diagnostics.json"
OUTPUT_MD = "AUDIO_PHASE_PHENOTYPE_DIAGNOSTICS.md"


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _mean(values: Iterable[float]) -> float:
    vals = [float(value) for value in values]
    return float(sum(vals) / len(vals)) if vals else 0.0


def _median(values: Iterable[float]) -> float:
    vals = sorted(float(value) for value in values)
    if not vals:
        return 0.0
    mid = len(vals) // 2
    if len(vals) % 2:
        return vals[mid]
    return float(0.5 * (vals[mid - 1] + vals[mid]))


def _std(values: Iterable[float]) -> float:
    vals = np.asarray([float(value) for value in values], dtype=np.float64)
    return float(vals.std()) if vals.size else 0.0


def _pearson(xs: Sequence[float], ys: Sequence[float]) -> float | None:
    if len(xs) < 3 or len(xs) != len(ys):
        return None
    x = np.asarray(xs, dtype=np.float64)
    y = np.asarray(ys, dtype=np.float64)
    if float(np.std(x)) <= 1.0e-12 or float(np.std(y)) <= 1.0e-12:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def _safe_div(num: float, den: float) -> float:
    return float(num / max(1.0e-12, den))


def _frame_rms_cv(wav: torch.Tensor, frame: int = 1024, hop: int = 256) -> float:
    x = wav.detach().cpu().float().view(-1)
    if x.numel() < frame:
        pad = torch.zeros(frame - x.numel(), dtype=x.dtype)
        x = torch.cat([x, pad], dim=0)
    frames = x.unfold(0, frame, hop)
    rms = torch.sqrt(torch.mean(frames * frames, dim=-1) + 1.0e-12)
    mean = float(rms.mean().item())
    return _safe_div(float(rms.std(unbiased=False).item()), mean)


def _zero_crossing_rate(wav: torch.Tensor) -> float:
    x = wav.detach().cpu().float().view(-1)
    if x.numel() < 2:
        return 0.0
    signs = torch.signbit(x)
    return float((signs[1:] != signs[:-1]).float().mean().item())


def _spectral_features(mag: torch.Tensor) -> dict[str, float]:
    # mag: [B, F, T]
    power = (mag.detach().float().clamp_min(0.0) ** 2).squeeze(0)
    if power.ndim != 2:
        power = power.reshape(power.shape[-2], power.shape[-1])
    f, t = int(power.size(0)), int(power.size(1))
    total = power.sum(dim=0).clamp_min(1.0e-12)
    probs = power / total.view(1, -1)
    entropy = -(probs * torch.log(probs.clamp_min(1.0e-12))).sum(dim=0) / math.log(max(2, f))
    bins = torch.linspace(0.0, 1.0, f, dtype=power.dtype, device=power.device).view(f, 1)
    centroid = (probs * bins).sum(dim=0)
    dominant_share = power.max(dim=0).values / total
    flux = torch.zeros(t, dtype=power.dtype, device=power.device)
    if t > 1:
        diff = torch.abs(power[:, 1:] - power[:, :-1]).sum(dim=0)
        denom = (power[:, 1:] + power[:, :-1]).sum(dim=0).clamp_min(1.0e-12)
        flux[1:] = diff / denom
    frame_energy = total
    return {
        "spectral_entropy_mean": float(entropy.mean().item()),
        "spectral_entropy_std": float(entropy.std(unbiased=False).item()),
        "spectral_centroid_mean": float(centroid.mean().item()),
        "spectral_centroid_std": float(centroid.std(unbiased=False).item()),
        "dominant_bin_share_mean": float(dominant_share.mean().item()),
        "spectral_flux_mean": float(flux.mean().item()),
        "spectral_flux_std": float(flux.std(unbiased=False).item()),
        "stft_energy_cv": _safe_div(float(frame_energy.std(unbiased=False).item()), float(frame_energy.mean().item())),
    }


def _phase_velocity_coherence(phase: torch.Tensor) -> float:
    if phase.size(-1) < 2:
        return 1.0
    diff = phase[..., 1:] - phase[..., :-1]
    diff = (diff + math.pi) % (2.0 * math.pi) - math.pi
    mean_cos = torch.cos(diff).mean(dim=-1)
    mean_sin = torch.sin(diff).mean(dim=-1)
    coh = torch.sqrt(mean_cos * mean_cos + mean_sin * mean_sin)
    return float(coh.mean().item())


def _audio_features(
    wav_path: Path,
    *,
    sr: int,
    stft_cfg: dict[str, Any],
    prefix_seconds: float,
    future_seconds: float,
) -> dict[str, float]:
    prefix_samples = max(1, int(round(prefix_seconds * sr)))
    future_samples = max(1, int(round(future_seconds * sr)))
    wav = _prepare_anchor(wav_path, target_sr=sr, total_samples=prefix_samples + future_samples)
    prefix = wav[..., :prefix_samples]
    future = wav[..., prefix_samples : prefix_samples + future_samples]
    prefix_mag, prefix_phase = compute_stft(prefix, stft_cfg)
    future_mag, future_phase = compute_stft(future, stft_cfg)
    prefix_loop = _loop_reentry_metrics(prefix, sr)
    future_loop = _loop_reentry_metrics(future, sr)
    prefix_rms = float(torch.sqrt(torch.mean(prefix * prefix) + 1.0e-12).item())
    future_rms = float(torch.sqrt(torch.mean(future * future) + 1.0e-12).item())
    out = {
        "prefix_rms": prefix_rms,
        "future_rms": future_rms,
        "future_prefix_rms_ratio": _safe_div(future_rms, prefix_rms),
        "prefix_frame_rms_cv": _frame_rms_cv(prefix),
        "future_frame_rms_cv": _frame_rms_cv(future),
        "prefix_zero_crossing_rate": _zero_crossing_rate(prefix),
        "future_zero_crossing_rate": _zero_crossing_rate(future),
        "prefix_phase_velocity_coherence": _phase_velocity_coherence(prefix_phase),
        "future_phase_velocity_coherence": _phase_velocity_coherence(future_phase),
        "prefix_loop_autocorr_peak": _as_float(prefix_loop.get("loop_autocorr_peak")),
        "future_loop_autocorr_peak": _as_float(future_loop.get("loop_autocorr_peak")),
        "prefix_first_chunk_reentry": _as_float(prefix_loop.get("first_chunk_reentry")),
        "future_first_chunk_reentry": _as_float(future_loop.get("first_chunk_reentry")),
    }
    out.update({f"prefix_{key}": value for key, value in _spectral_features(prefix_mag).items()})
    out.update({f"future_{key}": value for key, value in _spectral_features(future_mag).items()})
    return out


def _gain_rows(case: dict[str, Any], target_gain: float) -> tuple[dict[str, Any], dict[str, Any]]:
    by_gain = {
        _as_float(row.get("gain")): row
        for row in case.get("rows", []) or []
        if isinstance(row, dict)
    }
    return by_gain.get(0.0, {}), by_gain.get(float(target_gain), {})


def _load_probe_rows(manifest: dict[str, Any], target_gain: float) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    group_meta = {
        str(group.get("group")): group
        for group in manifest.get("groups", []) or []
        if isinstance(group, dict)
    }
    for group_name, group in group_meta.items():
        probe_json = Path(str(group.get("probe_json", "")))
        if not probe_json.exists():
            probe_json = Path(str(manifest.get("out_dir", ""))) / "groups" / group_name / PROBE_JSON
        if not probe_json.exists():
            continue
        probe = json.loads(probe_json.read_text(encoding="utf-8-sig"))
        for case in probe.get("cases", []) or []:
            if not isinstance(case, dict):
                continue
            base, target = _gain_rows(case, target_gain)
            if not base or not target:
                continue
            mechanism_flags = target.get("mechanism_flags") if isinstance(target.get("mechanism_flags"), dict) else {}
            delta_stats = target.get("phase_delta_stats") if isinstance(target.get("phase_delta_stats"), dict) else {}
            rows.append(
                {
                    "name": case.get("name"),
                    "group": group_name,
                    "role": group.get("role"),
                    "source_wav": case.get("source_wav"),
                    "gain0_target_corr": base.get("target_corr"),
                    "target_gain_corr": target.get("target_corr"),
                    "corr_delta": _as_float(target.get("target_corr")) - _as_float(base.get("target_corr")),
                    "gain0_target_mse": base.get("target_mse"),
                    "target_gain_mse": target.get("target_mse"),
                    "mse_delta": _as_float(target.get("target_mse")) - _as_float(base.get("target_mse")),
                    "vs_gain0_corr": target.get("vs_gain0_corr"),
                    "vs_gain0_mse": target.get("vs_gain0_mse"),
                    "loop_delta": _as_float(target.get("loop_autocorr_peak")) - _as_float(base.get("loop_autocorr_peak")),
                    "reentry_delta": _as_float(target.get("first_chunk_reentry")) - _as_float(base.get("first_chunk_reentry")),
                    "target_output_rms": target.get("output_rms"),
                    "target_rms": target.get("target_rms"),
                    "mean_abs_raw_delta": mechanism_flags.get("mean_abs_raw_delta"),
                    "mean_abs_shaped_delta": mechanism_flags.get("mean_abs_shaped_delta"),
                    "mean_energy_weight": mechanism_flags.get("mean_energy_weight"),
                    "mean_phase_velocity_coherence_probe": mechanism_flags.get("mean_phase_velocity_coherence"),
                    "future_mean_abs_phase_delta": delta_stats.get("future_mean_abs_phase_delta"),
                    "future_rms_phase_delta": delta_stats.get("future_rms_phase_delta"),
                    "future_mean_cos_phase_delta": delta_stats.get("future_mean_cos_phase_delta"),
                    "active_bin_mean_abs_phase_delta": delta_stats.get("active_bin_mean_abs_phase_delta"),
                }
            )
    return rows


def _summarize_cases(rows: Sequence[dict[str, Any]], key: str) -> list[dict[str, Any]]:
    groups = sorted({str(row.get(key)) for row in rows})
    out: list[dict[str, Any]] = []
    for group in groups:
        local = [row for row in rows if str(row.get(key)) == group]
        corr = [_as_float(row.get("corr_delta")) for row in local]
        mse = [_as_float(row.get("mse_delta")) for row in local]
        out.append(
            {
                key: group,
                "case_count": len(local),
                "mean_corr_delta": _mean(corr),
                "median_corr_delta": _median(corr),
                "corr_win_fraction": _mean([1.0 if value > 1.0e-6 else 0.0 for value in corr]),
                "mean_mse_delta": _mean(mse),
                "mse_win_fraction": _mean([1.0 if value < -1.0e-9 else 0.0 for value in mse]),
                "corr_delta_std": _std(corr),
                "top_positive_case": max(local, key=lambda row: _as_float(row.get("corr_delta")), default={}),
                "top_negative_case": min(local, key=lambda row: _as_float(row.get("corr_delta")), default={}),
            }
        )
    return out


def _feature_correlations(rows: Sequence[dict[str, Any]], *, exclude_single_source: bool) -> list[dict[str, Any]]:
    filtered = [
        row
        for row in rows
        if not (exclude_single_source and row.get("role") == "single_source_probe")
    ]
    if len(filtered) < 3:
        return []
    skip = {"name", "group", "role", "source_wav"}
    feature_keys = sorted(
        key
        for key, value in filtered[0].items()
        if key not in skip and isinstance(value, (int, float))
    )
    out: list[dict[str, Any]] = []
    y_corr = [_as_float(row.get("corr_delta")) for row in filtered]
    y_mse = [_as_float(row.get("mse_delta")) for row in filtered]
    for key in feature_keys:
        if key in {"corr_delta", "mse_delta"}:
            continue
        xs = [_as_float(row.get(key)) for row in filtered]
        corr_corr = _pearson(xs, y_corr)
        corr_mse = _pearson(xs, y_mse)
        if corr_corr is None and corr_mse is None:
            continue
        out.append(
            {
                "feature": key,
                "case_count": len(filtered),
                "exclude_single_source": bool(exclude_single_source),
                "pearson_with_corr_delta": corr_corr,
                "pearson_with_mse_delta": corr_mse,
                "feature_mean": _mean(xs),
                "feature_std": _std(xs),
            }
        )
    return sorted(out, key=lambda row: abs(_as_float(row.get("pearson_with_corr_delta"))), reverse=True)


def _write_markdown(summary: dict[str, Any], out_path: Path) -> None:
    def fmt(value: Any) -> str:
        if value is None:
            return "NA"
        if isinstance(value, (int, float)):
            return f"{float(value):.6g}"
        return str(value)

    lines = [
        "# Audio Phase Phenotype Diagnostics",
        "",
        "Post-hoc diagnostic over the frozen disjoint steady-phenotype manifest.",
        "Target audio is used only for correlation diagnostics, not mechanism selection.",
        "",
        f"- status: `{summary.get('status')}`",
        f"- manifest: `{summary.get('manifest_path')}`",
        f"- case count: `{fmt(summary.get('case_count'))}`",
        f"- target gain: `{fmt(summary.get('target_gain'))}`",
        "",
        "## Role Summary",
        "",
        "| role | cases | mean corr d | median corr d | corr wins | mean MSE d | top positive | top d | top negative | bottom d |",
        "|---|---:|---:|---:|---:|---:|---|---:|---|---:|",
    ]
    for row in summary.get("role_summary", []) or []:
        top_pos = row.get("top_positive_case") if isinstance(row.get("top_positive_case"), dict) else {}
        top_neg = row.get("top_negative_case") if isinstance(row.get("top_negative_case"), dict) else {}
        lines.append(
            "| {role} | {cases} | {mean} | {median} | {wins} | {mse} | {pos} | {posd} | {neg} | {negd} |".format(
                role=row.get("role"),
                cases=fmt(row.get("case_count")),
                mean=fmt(row.get("mean_corr_delta")),
                median=fmt(row.get("median_corr_delta")),
                wins=fmt(row.get("corr_win_fraction")),
                mse=fmt(row.get("mean_mse_delta")),
                pos=top_pos.get("name"),
                posd=fmt(top_pos.get("corr_delta")),
                neg=top_neg.get("name"),
                negd=fmt(top_neg.get("corr_delta")),
            )
        )
    lines.extend(
        [
            "",
            "## Top Feature Correlations",
            "",
            "| feature | corr with corr d | corr with MSE d | mean | std |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for row in (summary.get("feature_correlations_no_single_source", []) or [])[:20]:
        lines.append(
            "| {feature} | {ccorr} | {cmse} | {mean} | {std} |".format(
                feature=row.get("feature"),
                ccorr=fmt(row.get("pearson_with_corr_delta")),
                cmse=fmt(row.get("pearson_with_mse_delta")),
                mean=fmt(row.get("feature_mean")),
                std=fmt(row.get("feature_std")),
            )
        )
    lines.extend(
        [
            "",
            "## Top Cases",
            "",
            "| case | group | role | corr d | MSE d | gain0 corr | gain corr | shaped delta | prefix entropy | prefix flux |",
            "|---|---|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in summary.get("top_cases_by_corr_delta", []) or []:
        lines.append(
            "| {case} | {group} | {role} | {corr} | {mse} | {base} | {target} | {delta} | {entropy} | {flux} |".format(
                case=row.get("name"),
                group=row.get("group"),
                role=row.get("role"),
                corr=fmt(row.get("corr_delta")),
                mse=fmt(row.get("mse_delta")),
                base=fmt(row.get("gain0_target_corr")),
                target=fmt(row.get("target_gain_corr")),
                delta=fmt(row.get("mean_abs_shaped_delta")),
                entropy=fmt(row.get("prefix_spectral_entropy_mean")),
                flux=fmt(row.get("prefix_spectral_flux_mean")),
            )
        )
    lines.extend(
        [
            "",
            "## Interpretation Guard",
            "",
            "- This is post-hoc feature discovery over a weak mechanism signal.",
            "- Strong correlations here are hypotheses for a predeclared follow-up, not proof.",
            "- Single-source razor is excluded from the main correlation table.",
        ]
    )
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_phase_phenotype_diagnostics(
    *,
    manifest_path: Path,
    out_dir: Path,
    target_gain: float,
) -> dict[str, Any]:
    cfg = load_config(str(ROOT / "config.yaml")) if (ROOT / "config.yaml").exists() else load_config()
    sr = int(cfg["data"]["sample_rate"])
    stft_cfg = dict(cfg["data"]["stft"])
    manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    rows = _load_probe_rows(manifest, target_gain)
    prefix_seconds = _as_float(manifest.get("prefix_seconds"), 1.0)
    future_seconds = _as_float(manifest.get("future_seconds"), 1.0)
    enriched: list[dict[str, Any]] = []
    for row in rows:
        wav_path = Path(str(row.get("source_wav")))
        features = _audio_features(
            wav_path,
            sr=sr,
            stft_cfg=stft_cfg,
            prefix_seconds=prefix_seconds,
            future_seconds=future_seconds,
        )
        enriched.append({**row, **features})
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "runtime": "circleworld_proto",
        "schema": "circleworld_audio_phase_phenotype_diagnostics_v0",
        "status": "posthoc_diagnostic_only",
        "manifest_path": str(manifest_path),
        "out_dir": str(out_dir),
        "target_gain": float(target_gain),
        "case_count": len(enriched),
        "sample_rate": sr,
        "prefix_seconds": prefix_seconds,
        "future_seconds": future_seconds,
        "role_summary": _summarize_cases(enriched, "role"),
        "group_summary": _summarize_cases(enriched, "group"),
        "feature_correlations_all_cases": _feature_correlations(enriched, exclude_single_source=False),
        "feature_correlations_no_single_source": _feature_correlations(enriched, exclude_single_source=True),
        "top_cases_by_corr_delta": sorted(enriched, key=lambda row: _as_float(row.get("corr_delta")), reverse=True)[:15],
        "bottom_cases_by_corr_delta": sorted(enriched, key=lambda row: _as_float(row.get("corr_delta")))[:15],
        "cases": enriched,
    }
    (out_dir / OUTPUT_JSON).write_text(json.dumps(summary, indent=2), encoding="utf-8")
    _write_markdown(summary, out_dir / OUTPUT_MD)
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description="Post-hoc phase/audio phenotype diagnostics for a frozen mechanism manifest.")
    ap.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--target-gain", type=float, default=2.0)
    args = ap.parse_args()
    summary = run_phase_phenotype_diagnostics(
        manifest_path=Path(args.manifest),
        out_dir=Path(args.out_dir),
        target_gain=float(args.target_gain),
    )
    print(
        json.dumps(
            {
                "json": str(Path(args.out_dir) / OUTPUT_JSON),
                "markdown": str(Path(args.out_dir) / OUTPUT_MD),
                "status": summary.get("status"),
                "case_count": summary.get("case_count"),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
