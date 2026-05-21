from __future__ import annotations

import argparse
import json
import re
import sys
import wave
from collections import OrderedDict
from pathlib import Path
from typing import Any, Iterable, Sequence

ROOT = Path(__file__).resolve().parents[2]
RUNTIME = ROOT / "runtimes" / "circleworld_proto"
for path in (ROOT, RUNTIME):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from run_audio_delta_mechanism_probe import run_probe  # noqa: E402


TOKENBURST_ROOT = Path(r"C:\Users\Andrew\AppData\Local\Temp\rafa_tokenburst")
DEFAULT_WAV_DIR = ROOT / "wav_files"
DEFAULT_EXCLUDE_CASES = TOKENBURST_ROOT / "broad_wav_cases_18_2026_05_06.json"
OUTPUT_JSON = "audio_mechanism_family_sensitivity.json"
OUTPUT_MD = "AUDIO_MECHANISM_FAMILY_SENSITIVITY.md"

FAMILY_SPECS: "OrderedDict[str, tuple[str, ...]]" = OrderedDict(
    [
        (
            "electric_motor_hum",
            (
                "air conditioner",
                "air-conditioner",
                "razor",
                "electric razor",
                "vacuum",
                "air zoom vacuum",
                "engine start",
                "car engine",
                "hum",
                "air wrench",
                "can opener electric",
                "electric typewriter",
                "chain saw",
                "chainsaw",
                "buzz saw",
                "buzzsaw",
                "buzzer",
                "buzz fade",
            ),
        ),
        (
            "aircraft_transport",
            (
                "airplane",
                "aircraft",
                "fighter plane",
                "propeller",
                "jet",
                "takeoff",
                "landing",
                "ufo takeoff",
                "car drive",
                "car screech",
                "formula",
                "truck",
                "tractor",
            ),
        ),
        (
            "combustion_steam_engines",
            (
                "steam engine",
                "steam train",
                "train",
                "engine running",
                "engine revving",
                "engine start",
                "muscle car",
                "car engine",
                "tractor",
            ),
        ),
        (
            "tonal_instrument",
            (
                "piano",
                "guitar",
                "bell",
                "chime",
                "whistle",
                "trumpet",
                "flute",
                "accordion",
                "celesta",
                "harpsichord",
                "bontempi",
                "ensoniq",
                "sitar",
            ),
        ),
        (
            "impact_transient",
            (
                "gunshot",
                "shot",
                "crash",
                "thunder",
                "impact",
                "bomb",
                "explosion",
                "exploding",
                "slap",
                "pop",
                "drop",
                "ricochet",
                "thump",
                "whack",
            ),
        ),
        (
            "voice_creature",
            (
                "vocal",
                "vocalized",
                "cheer",
                "cheering",
                "laugh",
                "laughing",
                "gasp",
                "dog",
                "wolf",
                "roar",
                "scream",
                "baby",
                "yoda",
                "woah",
                "zombie",
                "bird",
            ),
        ),
        (
            "ambience_noise",
            (
                "rain",
                "storm",
                "wind",
                "cricket",
                "city",
                "street",
                "crowd",
                "applause",
                "water",
                "waterfall",
                "fireplace",
                "ocean",
            ),
        ),
    ]
)


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _safe_key(path: Path) -> str:
    return re.sub(r"[^a-z0-9]+", "_", path.stem.lower()).strip("_")[:64] or "case"


def _load_case_paths(path: Path | None) -> set[str]:
    if path is None or not path.exists():
        return set()
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if isinstance(payload, dict):
        return {str(value) for value in payload.values()}
    if isinstance(payload, list):
        out = set()
        for row in payload:
            if isinstance(row, dict) and row.get("path"):
                out.add(str(row["path"]))
            elif isinstance(row, str):
                out.add(row)
        return out
    return set()


def _valid_pcm_wav(path: Path) -> tuple[bool, str]:
    try:
        with wave.open(str(path), "rb") as wf:
            if wf.getnframes() <= 0:
                return False, "zero_frames"
            if wf.getframerate() <= 0:
                return False, "bad_sample_rate"
            if wf.getnchannels() <= 0:
                return False, "bad_channels"
        return True, ""
    except Exception as exc:  # noqa: BLE001 - report dataset hygiene, do not crash selection.
        return False, str(exc)


def _canonical_stem(path: Path) -> str:
    stem = re.sub(r"[^a-z0-9]+", "_", path.stem.lower()).strip("_")
    return re.sub(r"_[0-9a-f]{10}$", "", stem)


def _iter_valid_wavs(wav_dir: Path, exclude_paths: set[str]) -> tuple[list[Path], list[dict[str, str]], int]:
    candidates: list[Path] = []
    rejected: list[dict[str, str]] = []
    for path in sorted(wav_dir.glob("*.wav"), key=lambda item: item.name.lower()):
        if str(path) in exclude_paths:
            continue
        ok, reason = _valid_pcm_wav(path)
        if ok:
            candidates.append(path)
        else:
            rejected.append({"path": str(path), "reason": reason})
    candidates = sorted(
        candidates,
        key=lambda item: (_canonical_stem(item), item.name.lower(), item.stat().st_size, str(item).lower()),
    )
    valid: list[Path] = []
    seen: set[str] = set()
    duplicate_count = 0
    for path in candidates:
        key = _canonical_stem(path)
        if key in seen:
            duplicate_count += 1
            continue
        seen.add(key)
        valid.append(path)
    return valid, rejected, duplicate_count


def _family_match(path: Path, keywords: Sequence[str]) -> bool:
    name = re.sub(r"[^a-z0-9]+", " ", path.stem.lower()).strip()
    tokens = set(name.split())
    padded = f" {name} "
    for keyword in keywords:
        key = re.sub(r"[^a-z0-9]+", " ", keyword.lower()).strip()
        if not key:
            continue
        if " " in key:
            if f" {key} " in padded:
                return True
        elif key in tokens:
            return True
    return False


def _select_family_cases(
    wavs: Sequence[Path],
    *,
    cases_per_family: int,
) -> dict[str, list[Path]]:
    selected: dict[str, list[Path]] = {}
    for family, keywords in FAMILY_SPECS.items():
        rows = [path for path in wavs if _family_match(path, keywords)]
        selected[family] = rows[:cases_per_family]
    return selected


def _write_cases_json(paths: Sequence[Path], out_path: Path, family: str) -> dict[str, str]:
    cases: dict[str, str] = {}
    used_keys: set[str] = set()
    for path in paths:
        base = _safe_key(path)
        key = f"{family}_{len(cases) + 1:02d}_{base}"
        while key in used_keys:
            key = f"{key}_x"
        used_keys.add(key)
        cases[key] = str(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(cases, indent=2), encoding="utf-8")
    return cases


def _case_deltas(probe: dict[str, Any], target_gain: float) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for case in probe.get("cases", []) or []:
        if not isinstance(case, dict):
            continue
        by_gain = {
            _as_float(row.get("gain")): row
            for row in case.get("rows", []) or []
            if isinstance(row, dict)
        }
        base = by_gain.get(0.0)
        target = by_gain.get(float(target_gain))
        if not base or not target:
            continue
        rows.append(
            {
                "name": case.get("name"),
                "source_wav": case.get("source_wav"),
                "gain0_corr": base.get("target_corr"),
                "target_gain_corr": target.get("target_corr"),
                "corr_delta": _as_float(target.get("target_corr")) - _as_float(base.get("target_corr")),
                "gain0_mse": base.get("target_mse"),
                "target_gain_mse": target.get("target_mse"),
                "mse_delta": _as_float(target.get("target_mse")) - _as_float(base.get("target_mse")),
                "vs_gain0_corr": target.get("vs_gain0_corr"),
                "loop_delta": _as_float(target.get("loop_autocorr_peak"))
                - _as_float(base.get("loop_autocorr_peak")),
                "reentry_delta": _as_float(target.get("first_chunk_reentry"))
                - _as_float(base.get("first_chunk_reentry")),
            }
        )
    return rows


def _find_aggregate_row(probe: dict[str, Any], target_gain: float) -> dict[str, Any]:
    for row in probe.get("aggregate", []) or []:
        if isinstance(row, dict) and abs(_as_float(row.get("gain")) - target_gain) <= 1.0e-12:
            return row
    return {}


def _family_status(
    row: dict[str, Any],
    *,
    min_leave_one_out_corr_delta: float | None,
    positive_outlier_share: float,
) -> str:
    mean_corr = _as_float(row.get("mean_target_corr_delta_vs_gain0"))
    median_corr = _as_float(row.get("median_target_corr_delta_vs_gain0"))
    mean_mse = _as_float(row.get("mean_target_mse_delta_vs_gain0"))
    win_fraction = _as_float(row.get("target_corr_win_fraction_vs_gain0"))
    min_loo = _as_float(min_leave_one_out_corr_delta, default=-1.0)
    if (
        mean_corr > 0.02
        and median_corr > 0.005
        and win_fraction >= 0.67
        and mean_mse <= 0.0
        and min_loo > 0.0
        and positive_outlier_share <= 0.5
    ):
        return "family_candidate_signal"
    if mean_corr > 0.0 and mean_mse <= 0.0:
        return "family_tradeoff_signal"
    if mean_mse < 0.0:
        return "family_mse_only_or_corr_tradeoff"
    return "family_not_improved"


def _positive_outlier_share(case_rows: Sequence[dict[str, Any]]) -> float:
    positives = sorted((_as_float(row.get("corr_delta")) for row in case_rows if _as_float(row.get("corr_delta")) > 0.0), reverse=True)
    if not positives:
        return 0.0
    return float(positives[0] / max(1.0e-12, sum(positives)))


def _leave_one_out_mean(values: Sequence[float]) -> list[float]:
    vals = [float(value) for value in values]
    if len(vals) <= 1:
        return []
    total = sum(vals)
    denom = len(vals) - 1
    return [float((total - value) / denom) for value in vals]


def _write_markdown(summary: dict[str, Any], out_path: Path) -> None:
    def fmt(value: Any) -> str:
        if value is None:
            return "NA"
        if isinstance(value, (int, float)):
            return f"{float(value):.6g}"
        return str(value)

    lines = [
        "# Audio Mechanism Family Sensitivity",
        "",
        "This runs a frozen prefix-only mechanism row across deterministic valid-PCM sound-family lockboxes.",
        "The target future audio is used only for metrics; family selection uses filenames and RIFF validity only.",
        "",
        f"- status: `{summary.get('status')}`",
        f"- config: `{summary.get('config_path')}`",
        f"- wav dir: `{summary.get('wav_dir')}`",
        f"- excluded broad paths: `{summary.get('excluded_path_count')}`",
        f"- rejected non-PCM WAVs: `{summary.get('rejected_wav_count')}`",
        f"- deduped canonical stems: `{summary.get('deduped_duplicate_stem_count')}`",
        f"- mechanism: `{summary.get('mechanism')}`",
        f"- magnitude/mask/gain: `{summary.get('magnitude_mode')}` / `{summary.get('mask_mode')}` / `{fmt(summary.get('target_gain'))}`",
        "",
        "| family | cases | status | mean corr d | median corr d | min LOO mean | corr wins | mean MSE d | outlier share | top positive case | top positive d | top negative case | top negative d |",
        "|---|---:|---|---:|---:|---:|---:|---:|---:|---|---:|---|---:|",
    ]
    for row in summary.get("families", []) or []:
        top_pos = row.get("top_positive_case") if isinstance(row.get("top_positive_case"), dict) else {}
        top_neg = row.get("top_negative_case") if isinstance(row.get("top_negative_case"), dict) else {}
        lines.append(
            "| {family} | {cases} | {status} | {mean} | {median} | {loo} | {wins} | {mse} | {share} | {pos} | {posd} | {neg} | {negd} |".format(
                family=row.get("family"),
                cases=fmt(row.get("case_count")),
                status=row.get("status"),
                mean=fmt(row.get("mean_corr_delta")),
                median=fmt(row.get("median_corr_delta")),
                loo=fmt(row.get("min_leave_one_out_corr_delta")),
                wins=fmt(row.get("corr_win_fraction")),
                mse=fmt(row.get("mean_mse_delta")),
                share=fmt(row.get("positive_outlier_share")),
                pos=top_pos.get("name"),
                posd=fmt(top_pos.get("corr_delta")),
                neg=top_neg.get("name"),
                negd=fmt(top_neg.get("corr_delta")),
            )
        )
    lines.extend(
        [
            "",
            "## Interpretation Guard",
            "",
            "- A family candidate requires positive mean, positive median, broad win fraction, and non-regressing MSE.",
            "- High mean with low median or high outlier share is sound-family sensitivity, not general audio proof.",
            "- This is a retrospective family cut over filename groups; promotion requires a new predeclared family lockbox.",
        ]
    )
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_family_sensitivity(
    *,
    config_path: Path,
    out_dir: Path,
    wav_dir: Path,
    exclude_cases_json: Path | None,
    device_name: str,
    cases_per_family: int,
    min_cases: int,
    prefix_seconds: float,
    future_seconds: float,
    magnitude_mode: str,
    mask_mode: str,
    mechanism: str,
    target_gain: float,
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    exclude_paths = _load_case_paths(exclude_cases_json)
    wavs, rejected, duplicate_stem_count = _iter_valid_wavs(wav_dir, exclude_paths)
    family_paths = _select_family_cases(wavs, cases_per_family=cases_per_family)
    families: list[dict[str, Any]] = []
    for family, paths in family_paths.items():
        case_json = out_dir / "cases" / f"{family}_cases.json"
        cases = _write_cases_json(paths, case_json, family)
        if len(cases) < min_cases:
            families.append(
                {
                    "family": family,
                    "status": "insufficient_valid_cases",
                    "case_count": len(cases),
                    "case_json": str(case_json),
                    "cases": cases,
                }
            )
            continue
        probe_out = out_dir / "families" / family
        probe = run_probe(
            config_path=config_path,
            out_dir=probe_out,
            device_name=device_name,
            num_cases=len(cases),
            prefix_seconds=prefix_seconds,
            future_seconds=future_seconds,
            magnitude_modes=[magnitude_mode],
            mask_modes=[mask_mode],
            mechanisms=[mechanism],
            gains=[0.0, target_gain],
            cases_json=case_json,
        )
        aggregate = _find_aggregate_row(probe, target_gain)
        case_rows = _case_deltas(probe, target_gain)
        top_positive = max(case_rows, key=lambda row: _as_float(row.get("corr_delta")), default={})
        top_negative = min(case_rows, key=lambda row: _as_float(row.get("corr_delta")), default={})
        corr_deltas = [_as_float(row.get("corr_delta")) for row in case_rows]
        loo_means = _leave_one_out_mean(corr_deltas)
        min_loo = min(loo_means) if loo_means else None
        outlier_share = _positive_outlier_share(case_rows)
        status = _family_status(
            aggregate,
            min_leave_one_out_corr_delta=min_loo,
            positive_outlier_share=outlier_share,
        )
        families.append(
            {
                "family": family,
                "status": status,
                "raw_probe_status": probe.get("status"),
                "case_count": len(cases),
                "case_json": str(case_json),
                "probe_json": str(probe_out / "audio_delta_mechanism_probe.json"),
                "probe_markdown": str(probe_out / "AUDIO_DELTA_MECHANISM_PROBE.md"),
                "mean_corr_delta": aggregate.get("mean_target_corr_delta_vs_gain0"),
                "median_corr_delta": aggregate.get("median_target_corr_delta_vs_gain0"),
                "corr_win_fraction": aggregate.get("target_corr_win_fraction_vs_gain0"),
                "mean_mse_delta": aggregate.get("mean_target_mse_delta_vs_gain0"),
                "mse_win_fraction": aggregate.get("target_mse_win_fraction_vs_gain0"),
                "mean_vs_gain0_corr": aggregate.get("mean_vs_gain0_corr"),
                "positive_case_count": sum(1 for value in corr_deltas if value > 1.0e-6),
                "min_leave_one_out_corr_delta": min_loo,
                "mean_leave_one_out_corr_delta": _mean(loo_means) if loo_means else None,
                "mean_loop_autocorr_delta_vs_baseline": _mean([_as_float(row.get("loop_delta")) for row in case_rows]),
                "mean_reentry_delta_vs_baseline": _mean([_as_float(row.get("reentry_delta")) for row in case_rows]),
                "positive_outlier_share": outlier_share,
                "top_positive_case": top_positive,
                "top_negative_case": top_negative,
                "case_deltas": sorted(case_rows, key=lambda row: _as_float(row.get("corr_delta")), reverse=True),
                "cases": cases,
            }
        )
    best = max(families, key=lambda row: _as_float(row.get("mean_corr_delta")), default={})
    any_candidate = any(row.get("status") == "family_candidate_signal" for row in families)
    any_tradeoff = any(row.get("status") == "family_tradeoff_signal" for row in families)
    status = "family_candidate_found" if any_candidate else "family_tradeoff_signal_only" if any_tradeoff else "no_family_signal"
    summary = {
        "runtime": "circleworld_proto",
        "schema": "circleworld_audio_mechanism_family_sensitivity_v0",
        "status": status,
        "config_path": str(config_path),
        "out_dir": str(out_dir),
        "wav_dir": str(wav_dir),
        "exclude_cases_json": str(exclude_cases_json) if exclude_cases_json else None,
        "excluded_path_count": len(exclude_paths),
        "valid_wav_count": len(wavs),
        "rejected_wav_count": len(rejected),
        "deduped_duplicate_stem_count": duplicate_stem_count,
        "rejected_wavs_sample": rejected[:25],
        "device": device_name,
        "prefix_seconds": float(prefix_seconds),
        "future_seconds": float(future_seconds),
        "magnitude_mode": magnitude_mode,
        "mask_mode": mask_mode,
        "mechanism": mechanism,
        "target_gain": float(target_gain),
        "cases_per_family": int(cases_per_family),
        "min_cases": int(min_cases),
        "family_specs": {family: list(keywords) for family, keywords in FAMILY_SPECS.items()},
        "best_family": best.get("family"),
        "best_family_status": best.get("status"),
        "best_family_mean_corr_delta": best.get("mean_corr_delta"),
        "best_family_median_corr_delta": best.get("median_corr_delta"),
        "best_family_corr_win_fraction": best.get("corr_win_fraction"),
        "families": families,
    }
    (out_dir / OUTPUT_JSON).write_text(json.dumps(summary, indent=2), encoding="utf-8")
    _write_markdown(summary, out_dir / OUTPUT_MD)
    return summary


def _mean(values: Iterable[float]) -> float:
    vals = [float(value) for value in values]
    if not vals:
        return 0.0
    return float(sum(vals) / len(vals))


def main() -> None:
    ap = argparse.ArgumentParser(description="Run fixed audio delta mechanism across sound-family lockboxes.")
    ap.add_argument("--config", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--wav-dir", default=str(DEFAULT_WAV_DIR))
    ap.add_argument("--exclude-cases-json", default=str(DEFAULT_EXCLUDE_CASES))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--cases-per-family", type=int, default=12)
    ap.add_argument("--min-cases", type=int, default=4)
    ap.add_argument("--prefix-seconds", type=float, default=1.0)
    ap.add_argument("--future-seconds", type=float, default=1.0)
    ap.add_argument("--magnitude-mode", default="prefix_hold")
    ap.add_argument("--mask-mode", default="all_bins")
    ap.add_argument("--mechanism", default="time_smooth_3")
    ap.add_argument("--target-gain", type=float, default=2.0)
    args = ap.parse_args()
    summary = run_family_sensitivity(
        config_path=Path(args.config),
        out_dir=Path(args.out_dir),
        wav_dir=Path(args.wav_dir),
        exclude_cases_json=Path(args.exclude_cases_json) if args.exclude_cases_json else None,
        device_name=str(args.device),
        cases_per_family=int(args.cases_per_family),
        min_cases=int(args.min_cases),
        prefix_seconds=float(args.prefix_seconds),
        future_seconds=float(args.future_seconds),
        magnitude_mode=str(args.magnitude_mode),
        mask_mode=str(args.mask_mode),
        mechanism=str(args.mechanism),
        target_gain=float(args.target_gain),
    )
    print(
        json.dumps(
            {
                "json": str(Path(args.out_dir) / OUTPUT_JSON),
                "markdown": str(Path(args.out_dir) / OUTPUT_MD),
                "status": summary["status"],
                "best_family": summary.get("best_family"),
                "best_family_mean_corr_delta": summary.get("best_family_mean_corr_delta"),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
