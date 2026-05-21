from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
CORE = ROOT / "core"
LINEAGE = ROOT / "lineages" / "04_positive_replacement"
for path in (ROOT, CORE, LINEAGE):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from circleworld import recurse_circleworld
from export_circleworld_audio import _load_circle_cfg, prepare_reference_audio
from rafa_math_tools import phase_to_phasor, phasor_normalize
from stft_utils import compute_stft

COSINE_METRICS: tuple[tuple[str, str, str], ...] = (
    ("pooled_full_cos", "mean_pooled_full_cos", "pooled full cosine"),
    ("mean_prefix_cos", "mean_prefix_cos", "prefix cosine"),
    ("q_profile_cos", "mean_q_profile_cos", "q-profile cosine"),
    ("arc_profile_cos", "mean_arc_profile_cos", "arc-profile cosine"),
    ("temporal_profile_cos", "mean_temporal_profile_cos", "temporal-profile cosine"),
    ("support_profile_cos", "mean_support_profile_cos", "support-profile cosine"),
    ("branch_profile_cos", "mean_branch_profile_cos", "branch-profile cosine"),
)

DRIFT_METRICS: tuple[tuple[str, str, str], ...] = (
    ("signature_count_delta", "mean_signature_count_delta", "signature-count drift"),
    ("family_count_delta", "mean_family_count_delta", "family-count drift"),
    ("confidence_delta", "mean_confidence_delta", "confidence drift"),
    ("q_entropy_delta", "mean_q_entropy_delta", "q-entropy drift"),
    ("branch_mass_delta", "mean_branch_mass_delta", "branch-mass drift"),
    ("dominant_family_share_delta", "mean_dominant_family_share_delta", "dominant-family-share drift"),
)


def _safe_device(requested: str) -> torch.device:
    if requested == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _default_cases_path() -> Path:
    return ROOT / "outputs" / "circleworld_proto" / "expanded_anchor_cases_2026-04-09.json"


def _load_cases(path: Path) -> dict[str, str]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"Cases file must be a JSON object: {path}")
    return {str(k): str(v) for k, v in payload.items()}


def _crop_or_pad(wav: torch.Tensor, target_len: int) -> torch.Tensor:
    if wav.size(-1) < target_len:
        reps = int((target_len + max(1, wav.size(-1)) - 1) / max(1, wav.size(-1)))
        wav = wav.repeat(1, reps)[..., :target_len]
    elif wav.size(-1) > target_len:
        wav = wav[..., :target_len]
    return wav


def _pitch_resample(wav: torch.Tensor, semitones: float) -> torch.Tensor:
    ratio = 2.0 ** (float(semitones) / 12.0)
    src_len = wav.size(-1)
    target_len = max(8, int(round(src_len / ratio)))
    warped = F.interpolate(wav.unsqueeze(0), size=target_len, mode="linear", align_corners=False).squeeze(0)
    return _crop_or_pad(warped, src_len)


def _phasor_global_phase_offset(z: torch.Tensor, offset: float) -> torch.Tensor:
    delta = z.new_tensor([torch.cos(z.new_tensor(offset)), torch.sin(z.new_tensor(offset))])
    delta = delta.view(1, 1, 1, 2)
    real = z[..., 0] * delta[..., 0] - z[..., 1] * delta[..., 1]
    imag = z[..., 0] * delta[..., 1] + z[..., 1] * delta[..., 0]
    return phasor_normalize(torch.stack([real, imag], dim=-1))


def _run_circleworld(phase_state: torch.Tensor, cfg: Any) -> dict[str, Any]:
    run_mode = cfg.branching_mode if cfg.branching_mode in {"native_multimode", "native_multimode_childworld"} else "active_packets"
    return recurse_circleworld(phase_state, cfg=cfg, depth=cfg.recursion_depth, mode=run_mode)


def _pooled_signature(bank: Any) -> torch.Tensor | None:
    if bank is None or getattr(bank, "num_signatures", 0) <= 0:
        return None
    return F.normalize(bank.h.mean(dim=0, keepdim=True), dim=-1).squeeze(0)


def _pooled_feature(x: torch.Tensor | None) -> torch.Tensor | None:
    if x is None or x.numel() == 0:
        return None
    if x.dim() == 1:
        pooled = x
    else:
        pooled = x.float().reshape(x.size(0), -1).mean(dim=0)
    return F.normalize(pooled.view(1, -1), dim=-1).squeeze(0)


def _cos(a: torch.Tensor | None, b: torch.Tensor | None) -> float:
    if a is None or b is None:
        return 0.0
    if a.numel() != b.numel():
        size = min(a.numel(), b.numel())
        a = a[:size]
        b = b[:size]
    return float(F.cosine_similarity(a.view(1, -1), b.view(1, -1), dim=-1).item())


def _bank_metrics(base_bank: Any, transformed_bank: Any) -> dict[str, float]:
    if base_bank is None or transformed_bank is None or getattr(base_bank, "num_signatures", 0) <= 0 or getattr(transformed_bank, "num_signatures", 0) <= 0:
        return {
            "pooled_full_cos": 0.0,
            "q_profile_cos": 0.0,
            "arc_profile_cos": 0.0,
            "temporal_profile_cos": 0.0,
            "support_profile_cos": 0.0,
            "branch_profile_cos": 0.0,
            "mean_prefix_cos": 0.0,
        }
    base_h = _pooled_signature(base_bank)
    tr_h = _pooled_signature(transformed_bank)
    out: dict[str, float] = {
        "pooled_full_cos": _cos(base_h, tr_h),
        "q_profile_cos": _cos(_pooled_feature(base_bank.q_profile), _pooled_feature(transformed_bank.q_profile)),
        "arc_profile_cos": _cos(_pooled_feature(base_bank.arc_profile), _pooled_feature(transformed_bank.arc_profile)),
        "temporal_profile_cos": _cos(_pooled_feature(base_bank.temporal_profile), _pooled_feature(transformed_bank.temporal_profile)),
        "support_profile_cos": _cos(_pooled_feature(base_bank.support_profile), _pooled_feature(transformed_bank.support_profile)),
        "branch_profile_cos": _cos(_pooled_feature(base_bank.branch_profile), _pooled_feature(transformed_bank.branch_profile)),
    }
    prefix_cos = []
    for dim in getattr(base_bank, "prefix_dims", (128, 256, 384, 512, 640, 768)):
        dim = min(int(dim), base_bank.h.size(-1), transformed_bank.h.size(-1))
        key = f"prefix_cos_{dim}"
        out[key] = _cos(base_h[:dim], tr_h[:dim])
        prefix_cos.append(out[key])
    out["mean_prefix_cos"] = float(sum(prefix_cos) / max(1, len(prefix_cos)))
    return out


def _summary_metrics(run: dict[str, Any]) -> dict[str, float]:
    summary = run.get("summary_cache")
    if summary is None:
        from circleworld import summarize_circleworld_run

        summary = summarize_circleworld_run(run)
        run["summary_cache"] = summary
    return {
        "num_relational_signatures": float(summary.get("num_relational_signatures", 0.0)),
        "num_relational_signature_families": float(summary.get("num_relational_signature_families", 0.0)),
        "mean_relational_signature_confidence": float(summary.get("mean_relational_signature_confidence", 0.0)),
        "mean_relational_signature_q_entropy": float(summary.get("mean_relational_signature_q_entropy", 0.0)),
        "mean_relational_branch_mass": float(summary.get("mean_relational_branch_mass", 0.0)),
        "dominant_relational_family_share": float(summary.get("dominant_relational_family_share", 0.0)),
    }


def _transform_specs() -> list[dict[str, Any]]:
    return [
        {"name": "gain_070", "kind": "wav_gain", "scale": 0.70, "family": "gain"},
        {"name": "gain_130", "kind": "wav_gain", "scale": 1.30, "family": "gain"},
        {"name": "time_shift_1024", "kind": "wav_roll", "shift": 1024, "family": "time_shift"},
        {"name": "time_shift_2048", "kind": "wav_roll", "shift": 2048, "family": "time_shift"},
        {"name": "time_shift_neg_2048", "kind": "wav_roll", "shift": -2048, "family": "time_shift"},
        {"name": "pitch_down_1st", "kind": "wav_pitch", "semitones": -1.0, "family": "pitch"},
        {"name": "pitch_up_1st", "kind": "wav_pitch", "semitones": 1.0, "family": "pitch"},
        {"name": "pitch_up_2st", "kind": "wav_pitch", "semitones": 2.0, "family": "pitch"},
        {"name": "phase_half_pi", "kind": "phase_offset", "offset": math.pi / 2.0, "family": "phase"},
        {"name": "phase_pi", "kind": "phase_offset", "offset": math.pi, "family": "phase"},
    ]


def _apply_transform(
    *,
    wav: torch.Tensor,
    phase_state: torch.Tensor,
    transform: dict[str, Any],
    stft_cfg: dict[str, Any],
) -> torch.Tensor:
    kind = str(transform["kind"])
    if kind == "wav_gain":
        wav_t = wav * float(transform["scale"])
        _mag, phase = compute_stft(wav_t, stft_cfg)
        return phase_to_phasor(phase)
    if kind == "wav_roll":
        wav_t = torch.roll(wav, shifts=int(transform["shift"]), dims=-1)
        _mag, phase = compute_stft(wav_t, stft_cfg)
        return phase_to_phasor(phase)
    if kind == "wav_pitch":
        wav_t = _pitch_resample(wav, semitones=float(transform["semitones"]))
        _mag, phase = compute_stft(wav_t, stft_cfg)
        return phase_to_phasor(phase)
    if kind == "phase_offset":
        return _phasor_global_phase_offset(phase_state, float(transform["offset"]))
    raise RuntimeError(f"Unknown transform kind: {kind}")


def _mean_value(rows: list[dict[str, Any]], key: str) -> float:
    return float(sum(float(row.get(key, 0.0)) for row in rows) / max(1, len(rows)))


def _mean_abs_value(rows: list[dict[str, Any]], key: str) -> float:
    return float(sum(abs(float(row.get(key, 0.0))) for row in rows) / max(1, len(rows)))


def _rank_metric(values: dict[str, float], *, higher_is_better: bool = False, closer_to_zero: bool = False) -> tuple[list[str], dict[str, int]]:
    items = list(values.items())
    if closer_to_zero:
        items.sort(key=lambda item: (abs(float(item[1])), item[0]))
    elif higher_is_better:
        items.sort(key=lambda item: (-float(item[1]), item[0]))
    else:
        items.sort(key=lambda item: (float(item[1]), item[0]))
    ordered = [name for name, _value in items]
    return ordered, {name: idx for idx, name in enumerate(ordered, start=1)}


def _headline_verdict(stats: dict[str, Any]) -> dict[str, Any]:
    mean_cosine_suite = float(stats["mean_cosine_suite"])
    mean_abs_drift_suite = float(stats["mean_abs_drift_suite"])
    strongest_cos = max(COSINE_METRICS, key=lambda item: float(stats.get(item[1], 0.0)))
    weakest_drift = max(DRIFT_METRICS, key=lambda item: abs(float(stats.get(item[1], 0.0))))
    if mean_cosine_suite >= 0.999 and mean_abs_drift_suite <= 0.01:
        tier = "near_invariant"
        label = "near-invariant"
    elif mean_cosine_suite >= 0.995 and mean_abs_drift_suite <= 0.05:
        tier = "stable"
        label = "stable"
    elif mean_cosine_suite >= 0.985 and mean_abs_drift_suite <= 0.2:
        tier = "mixed"
        label = "mixed"
    else:
        tier = "fragile"
        label = "fragile"
    summary = (
        f"{stats['family']} transform stays {label}; strongest {strongest_cos[2]} "
        f"{float(stats.get(strongest_cos[1], 0.0)):.4f}, worst {weakest_drift[2]} "
        f"{float(stats.get(weakest_drift[1], 0.0)):+.4f}."
    )
    return {
        "tier": tier,
        "summary": summary,
        "strongest_cosine_metric": strongest_cos[1],
        "strongest_cosine_value": float(stats.get(strongest_cos[1], 0.0)),
        "largest_drift_metric": weakest_drift[1],
        "largest_drift_value": float(stats.get(weakest_drift[1], 0.0)),
    }


def evaluate_relational_metamers(
    *,
    config_path: Path,
    out_dir: Path,
    cases_path: Path,
    device_name: str,
    clip_seconds: int,
) -> dict[str, Any]:
    device = _safe_device(device_name)
    cfg = _load_circle_cfg(config_path)
    from config import load_config

    stft_cfg = load_config()["data"]["stft"]
    cases = _load_cases(cases_path)
    out_dir.mkdir(parents=True, exist_ok=True)

    case_rows: list[dict[str, Any]] = []
    transform_rows: list[dict[str, Any]] = []
    transforms = _transform_specs()
    transform_lookup = {str(spec["name"]): spec for spec in transforms}

    for case_name, wav_path_str in cases.items():
        wav_path = Path(wav_path_str)
        wav, _ = prepare_reference_audio(wav_path=wav_path, device_name=device_name, clip_seconds_override=clip_seconds)
        wav = wav.to(device)
        _mag, phase = compute_stft(wav, stft_cfg)
        phase_state = phase_to_phasor(phase)
        base_run = _run_circleworld(phase_state, cfg)
        base_bank = base_run["relational_signatures"]
        base_metrics = _summary_metrics(base_run)
        case_entry: dict[str, Any] = {
            "case": case_name,
            "wav_path": str(wav_path),
            "base": base_metrics,
            "transforms": [],
        }
        for transform in transforms:
            phase_state_t = _apply_transform(wav=wav, phase_state=phase_state, transform=transform, stft_cfg=stft_cfg)
            run_t = _run_circleworld(phase_state_t, cfg)
            bank_t = run_t["relational_signatures"]
            trans_metrics = _summary_metrics(run_t)
            compare = _bank_metrics(base_bank, bank_t)
            compare.update(
                {
                    "signature_count_delta": float(trans_metrics["num_relational_signatures"] - base_metrics["num_relational_signatures"]),
                    "family_count_delta": float(trans_metrics["num_relational_signature_families"] - base_metrics["num_relational_signature_families"]),
                    "confidence_delta": float(trans_metrics["mean_relational_signature_confidence"] - base_metrics["mean_relational_signature_confidence"]),
                    "q_entropy_delta": float(trans_metrics["mean_relational_signature_q_entropy"] - base_metrics["mean_relational_signature_q_entropy"]),
                    "branch_mass_delta": float(trans_metrics["mean_relational_branch_mass"] - base_metrics["mean_relational_branch_mass"]),
                    "dominant_family_share_delta": float(
                        trans_metrics["dominant_relational_family_share"] - base_metrics["dominant_relational_family_share"]
                    ),
                }
            )
            row = {
                "case": case_name,
                "transform": str(transform["name"]),
                "transform_family": str(transform.get("family", transform["kind"])),
                **trans_metrics,
                **compare,
            }
            transform_rows.append(row)
            case_entry["transforms"].append(row)
        case_rows.append(case_entry)

    by_transform: dict[str, dict[str, Any]] = {}
    for name in sorted({str(row["transform"]) for row in transform_rows}):
        subset = [row for row in transform_rows if row["transform"] == name]
        spec = transform_lookup.get(name, {"family": "unknown"})
        stats: dict[str, Any] = {
            "count": len(subset),
            "family": str(spec.get("family", "unknown")),
            "spec": spec,
            "mean_pooled_full_cos": _mean_value(subset, "pooled_full_cos"),
            "mean_prefix_cos": _mean_value(subset, "mean_prefix_cos"),
            "mean_mean_prefix_cos": _mean_value(subset, "mean_prefix_cos"),
            "mean_q_profile_cos": _mean_value(subset, "q_profile_cos"),
            "mean_arc_profile_cos": _mean_value(subset, "arc_profile_cos"),
            "mean_temporal_profile_cos": _mean_value(subset, "temporal_profile_cos"),
            "mean_support_profile_cos": _mean_value(subset, "support_profile_cos"),
            "mean_branch_profile_cos": _mean_value(subset, "branch_profile_cos"),
            "mean_signature_count_delta": _mean_value(subset, "signature_count_delta"),
            "mean_family_count_delta": _mean_value(subset, "family_count_delta"),
            "mean_confidence_delta": _mean_value(subset, "confidence_delta"),
            "mean_q_entropy_delta": _mean_value(subset, "q_entropy_delta"),
            "mean_branch_mass_delta": _mean_value(subset, "branch_mass_delta"),
            "mean_dominant_family_share_delta": _mean_value(subset, "dominant_family_share_delta"),
            "mean_abs_signature_count_delta": _mean_abs_value(subset, "signature_count_delta"),
            "mean_abs_family_count_delta": _mean_abs_value(subset, "family_count_delta"),
            "mean_abs_confidence_delta": _mean_abs_value(subset, "confidence_delta"),
            "mean_abs_q_entropy_delta": _mean_abs_value(subset, "q_entropy_delta"),
            "mean_abs_branch_mass_delta": _mean_abs_value(subset, "branch_mass_delta"),
            "mean_abs_dominant_family_share_delta": _mean_abs_value(subset, "dominant_family_share_delta"),
        }
        stats["mean_cosine_suite"] = float(sum(float(stats[key]) for _raw, key, _label in COSINE_METRICS) / len(COSINE_METRICS))
        stats["mean_abs_drift_suite"] = float(
            sum(float(stats[f"mean_abs_{raw}"]) for raw, _key, _label in DRIFT_METRICS) / len(DRIFT_METRICS)
        )
        stats["headline_verdict"] = _headline_verdict(stats)
        by_transform[name] = stats

    transform_rankings: dict[str, Any] = {"cosine_metrics": {}, "drift_metrics": {}}
    rank_totals = {name: 0 for name in by_transform}
    rank_counts = {name: 0 for name in by_transform}
    for _raw_key, mean_key, _label in COSINE_METRICS:
        ordered, ranks = _rank_metric({name: float(stats.get(mean_key, 0.0)) for name, stats in by_transform.items()}, higher_is_better=True)
        transform_rankings["cosine_metrics"][mean_key] = ordered
        for name, rank in ranks.items():
            rank_totals[name] += rank
            rank_counts[name] += 1
    for raw_key, mean_key, _label in DRIFT_METRICS:
        ordered, ranks = _rank_metric({name: float(stats.get(mean_key, 0.0)) for name, stats in by_transform.items()}, closer_to_zero=True)
        transform_rankings["drift_metrics"][mean_key] = ordered
        for name, rank in ranks.items():
            rank_totals[name] += rank
            rank_counts[name] += 1
            by_transform[name][f"{mean_key}_abs_rank"] = rank
            by_transform[name][f"mean_abs_{raw_key}"] = float(by_transform[name].get(f"mean_abs_{raw_key}", 0.0))
    overall_ranked = sorted(
        by_transform.keys(),
        key=lambda name: (rank_totals[name] / max(1, rank_counts[name]), name),
    )
    transform_rankings["overall_stability"] = overall_ranked
    for idx, name in enumerate(overall_ranked, start=1):
        by_transform[name]["overall_stability_rank"] = idx
        by_transform[name]["mean_rank_across_metrics"] = float(rank_totals[name] / max(1, rank_counts[name]))
        for _raw_key, mean_key, _label in COSINE_METRICS:
            ordered = transform_rankings["cosine_metrics"][mean_key]
            by_transform[name][f"{mean_key}_rank"] = ordered.index(name) + 1

    transform_headlines = [
        {
            "transform": name,
            "family": str(by_transform[name].get("family", "unknown")),
            "overall_stability_rank": int(by_transform[name]["overall_stability_rank"]),
            "headline_verdict": by_transform[name]["headline_verdict"],
        }
        for name in overall_ranked
    ]

    summary = {
        "config_path": str(config_path),
        "cases_path": str(cases_path),
        "device": str(device),
        "clip_seconds": int(clip_seconds),
        "num_cases": len(case_rows),
        "num_transforms": len(transforms),
        "transform_specs": transforms,
        "mean_pooled_full_cos": _mean_value(transform_rows, "pooled_full_cos"),
        "mean_prefix_cos": _mean_value(transform_rows, "mean_prefix_cos"),
        "mean_q_profile_cos": _mean_value(transform_rows, "q_profile_cos"),
        "mean_arc_profile_cos": _mean_value(transform_rows, "arc_profile_cos"),
        "mean_temporal_profile_cos": _mean_value(transform_rows, "temporal_profile_cos"),
        "mean_support_profile_cos": _mean_value(transform_rows, "support_profile_cos"),
        "mean_branch_profile_cos": _mean_value(transform_rows, "branch_profile_cos"),
        "mean_signature_count_delta": _mean_value(transform_rows, "signature_count_delta"),
        "mean_family_count_delta": _mean_value(transform_rows, "family_count_delta"),
        "mean_confidence_delta": _mean_value(transform_rows, "confidence_delta"),
        "mean_q_entropy_delta": _mean_value(transform_rows, "q_entropy_delta"),
        "mean_branch_mass_delta": _mean_value(transform_rows, "branch_mass_delta"),
        "mean_dominant_family_share_delta": _mean_value(transform_rows, "dominant_family_share_delta"),
        "mean_abs_signature_count_delta": _mean_abs_value(transform_rows, "signature_count_delta"),
        "mean_abs_family_count_delta": _mean_abs_value(transform_rows, "family_count_delta"),
        "mean_abs_confidence_delta": _mean_abs_value(transform_rows, "confidence_delta"),
        "mean_abs_q_entropy_delta": _mean_abs_value(transform_rows, "q_entropy_delta"),
        "mean_abs_branch_mass_delta": _mean_abs_value(transform_rows, "branch_mass_delta"),
        "mean_abs_dominant_family_share_delta": _mean_abs_value(transform_rows, "dominant_family_share_delta"),
        "by_transform": by_transform,
        "transform_rankings": transform_rankings,
        "transform_headlines": transform_headlines,
        "cases": case_rows,
    }
    (out_dir / "relational_metamer_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    md_lines = [
        "# Circleworld Relational Metamer Evaluation",
        "",
        f"- config: `{config_path}`",
        f"- cases: `{cases_path}`",
        f"- device: `{device}`",
        f"- clip seconds: `{clip_seconds}`",
        f"- cases: `{len(case_rows)}`",
        f"- transforms per case: `{len(transforms)}`",
        f"- mean pooled full cosine: `{summary['mean_pooled_full_cos']:.4f}`",
        f"- mean prefix cosine: `{summary['mean_prefix_cos']:.4f}`",
        f"- mean q-profile cosine: `{summary['mean_q_profile_cos']:.4f}`",
        f"- mean arc-profile cosine: `{summary['mean_arc_profile_cos']:.4f}`",
        f"- mean temporal-profile cosine: `{summary['mean_temporal_profile_cos']:.4f}`",
        f"- mean support-profile cosine: `{summary['mean_support_profile_cos']:.4f}`",
        f"- mean branch-profile cosine: `{summary['mean_branch_profile_cos']:.4f}`",
        f"- mean signature-count drift: `{summary['mean_signature_count_delta']:+.4f}`",
        f"- mean family-count drift: `{summary['mean_family_count_delta']:+.4f}`",
        f"- mean confidence drift: `{summary['mean_confidence_delta']:+.4f}`",
        "",
        "## Transform Rankings",
        "",
    ]
    for item in transform_headlines:
        verdict = item["headline_verdict"]
        md_lines.append(
            f"- rank {item['overall_stability_rank']}: `{item['transform']}` ({item['family']}) {verdict['summary']}"
        )
    md_lines.extend(["", "## By Transform", ""])
    for name in overall_ranked:
        stats = by_transform[name]
        verdict = stats["headline_verdict"]
        md_lines.extend(
            [
                f"### {name}",
                f"- family: `{stats['family']}`",
                f"- verdict: `{verdict['tier']}`",
                f"- headline: {verdict['summary']}",
                f"- stability rank: `{stats['overall_stability_rank']}`",
                f"- mean pooled full cosine: `{stats['mean_pooled_full_cos']:.4f}`",
                f"- mean prefix cosine: `{stats['mean_prefix_cos']:.4f}`",
                f"- mean q-profile cosine: `{stats['mean_q_profile_cos']:.4f}`",
                f"- mean arc-profile cosine: `{stats['mean_arc_profile_cos']:.4f}`",
                f"- mean temporal-profile cosine: `{stats['mean_temporal_profile_cos']:.4f}`",
                f"- mean support-profile cosine: `{stats['mean_support_profile_cos']:.4f}`",
                f"- mean branch-profile cosine: `{stats['mean_branch_profile_cos']:.4f}`",
                f"- mean signature-count drift: `{stats['mean_signature_count_delta']:+.4f}`",
                f"- mean family-count drift: `{stats['mean_family_count_delta']:+.4f}`",
                f"- mean confidence drift: `{stats['mean_confidence_delta']:+.4f}`",
                f"- mean q-entropy drift: `{stats['mean_q_entropy_delta']:+.4f}`",
                f"- mean branch-mass drift: `{stats['mean_branch_mass_delta']:+.4f}`",
                "",
            ]
        )
    (out_dir / "RELATIONAL_METAMER_EVAL.md").write_text("\n".join(md_lines), encoding="utf-8")
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate Circleworld relational signatures under transformed-anchor metamers.")
    ap.add_argument("--config", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--cases-json", default=str(_default_cases_path()))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--clip-seconds", type=int, default=4)
    args = ap.parse_args()

    summary = evaluate_relational_metamers(
        config_path=Path(args.config),
        out_dir=Path(args.out_dir),
        cases_path=Path(args.cases_json),
        device_name=args.device,
        clip_seconds=int(args.clip_seconds),
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
