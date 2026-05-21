from __future__ import annotations

import argparse
import hashlib
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
OUTPUT_JSON = "audio_electric_motor_holdout.json"
OUTPUT_MD = "AUDIO_ELECTRIC_MOTOR_HOLDOUT.md"

EXPLORATORY_SPECS: "OrderedDict[str, tuple[str, ...]]" = OrderedDict(
    [
        (
            "generic_electric_motor",
            (
                "electric motor",
                "electrical motor",
                "servo motor",
            ),
        ),
        (
            "hvac_fan_airflow_motor",
            (
                "air conditioner",
                "air-conditioner",
                "fan startup",
            ),
        ),
        (
            "household_appliance_motor",
            (
                "can opener electric",
                "dishwasher",
                "clothes dryer",
                "hair dryer",
                "electric toothbrush",
            ),
        ),
        (
            "rotary_tool_motor",
            (
                "air wrench",
                "air wrench short",
                "drill",
                "drilling",
                "drill press",
                "cordless drill",
                "dentist drill",
                "electric sander",
                "metal grinder",
            ),
        ),
        (
            "saw_chain_motor_tool",
            (
                "chain saw",
                "chainsaw",
                "chainsaw quick",
                "buzz saw",
                "buzzsaw",
                "electric saw",
                "electric table saw",
                "table saw buzz",
            ),
        ),
        (
            "uncertain_yard_pump_motor",
            (
                "lawn mower",
                "pump",
            ),
        ),
        (
            "electrical_nonmotor_hum_buzz_control",
            (
                "fluorescent light humming",
                "old electrical buzz alarm",
                "electrical sweep",
                "electricity",
                "buzzer",
                "buzz fade",
            ),
        ),
        (
            "typewriter_nonmotor_control",
            (
                "electric typewriter",
                "typewriter",
            ),
        ),
        (
            "combustion_vehicle_motor_control",
            (
                "motorcycle",
                "motorbike",
                "fast bike or motorcycle",
                "engine",
                "tractor",
            ),
        ),
        (
            "buzzy_synth_control",
            (
                "buzzy",
                "saw c",
                "saw f",
                "saw lead",
                "sawtooth",
                "synth lead",
                "buzzy pad",
                "buzzy synth",
            ),
        ),
    ]
)

HOLDOUT_SPECS: "OrderedDict[str, tuple[str, ...]]" = OrderedDict(
    [
        (
            "razor_only_holdout",
            (
                "electric razor",
                "razor",
            ),
        ),
        (
            "nonrazor_appliance_holdout",
            (
                "electric toothbrush",
                "can opener electric",
                "air conditioner",
                "air-conditioner",
                "electric typewriter",
                "dishwasher",
                "clothes dryer",
                "hair dryer",
            ),
        ),
        (
            "appliance_with_razor_diagnostic",
            (
                "electric razor",
                "razor",
                "electric toothbrush",
                "can opener electric",
                "air zoom vacuum",
                "vacuum",
                "air conditioner",
                "electric typewriter",
            ),
        ),
    ]
)

RAZOR_KEYWORDS = ("electric razor", "razor")
PRIMARY_NO_RAZOR_UNION = (
    "generic_electric_motor",
    "hvac_fan_airflow_motor",
    "household_appliance_motor",
    "rotary_tool_motor",
    "saw_chain_motor_tool",
)
EXPANDED_NO_RAZOR_UNION = (*PRIMARY_NO_RAZOR_UNION, "uncertain_yard_pump_motor")
STRUCTURAL_MOTOR_GROUPS = set(EXPANDED_NO_RAZOR_UNION)
EXCLUDED_FALSE_POSITIVE_KEYWORDS = (
    "piano",
    "guitar",
    "bass",
    "lead",
    "saw wave",
    "saw stack",
    "saw pulse",
    "sawzallbs",
    "sawmono",
    "saww",
    "fantasia",
    "fantasy",
    "fanfare",
    "rhumba",
    "human voice",
    "humantouch",
    "human",
    "hummingbird",
    "humpback",
    "humvee",
    "shotgun",
    "mossberg",
    "pellet gun",
    "thump",
    "bump",
    "sawing wood",
    "fast sawing",
    "hacksaw",
    "small bow saw",
    "wood saw",
    "bone saw",
    "shock zap",
    "windows roll",
    "heartbeat",
    "heart",
    "garbage bag",
    "dropping a wrench",
)


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _mean(values: Iterable[float]) -> float:
    vals = [float(value) for value in values]
    if not vals:
        return 0.0
    return float(sum(vals) / len(vals))


def _safe_key(path: Path) -> str:
    return re.sub(r"[^a-z0-9]+", "_", path.stem.lower()).strip("_")[:64] or "case"


def _canonical_stem(path: Path) -> str:
    stem = re.sub(r"[^a-z0-9]+", "_", path.stem.lower()).strip("_")
    stem = re.sub(r"_[0-9a-f]{10}$", "", stem)
    for prefix in ("soundbible_", "freewavesamples_"):
        if stem.startswith(prefix):
            stem = stem[len(prefix) :]
    return stem


def _load_case_paths(path: Path | None) -> set[str]:
    if path is None or not path.exists():
        return set()
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if isinstance(payload, dict):
        return {str(value) for value in payload.values()}
    if isinstance(payload, list):
        out: set[str] = set()
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
    except Exception as exc:  # noqa: BLE001 - dataset hygiene should be reported, not fatal.
        return False, str(exc)


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


def _select_cases(
    wavs: Sequence[Path],
    specs: "OrderedDict[str, tuple[str, ...]]",
    *,
    cases_per_family: int,
) -> dict[str, list[Path]]:
    selected: dict[str, list[Path]] = {}
    for family, keywords in specs.items():
        rows = [path for path in wavs if _family_match(path, keywords)]
        selected[family] = rows[:cases_per_family]
    return selected


def _unique_paths(paths: Iterable[Path]) -> list[Path]:
    seen: set[str] = set()
    out: list[Path] = []
    for path in paths:
        key = _canonical_stem(path)
        if key in seen:
            continue
        seen.add(key)
        out.append(path)
    return out


def _write_cases_json(paths: Sequence[Path], out_path: Path, group: str) -> dict[str, str]:
    cases: dict[str, str] = {}
    used_keys: set[str] = set()
    for path in paths:
        base = _safe_key(path)
        key = f"{group}_{len(cases) + 1:02d}_{base}"
        while key in used_keys:
            key = f"{key}_x"
        used_keys.add(key)
        cases[key] = str(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(cases, indent=2), encoding="utf-8")
    return cases


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _find_aggregate_row(probe: dict[str, Any], target_gain: float) -> dict[str, Any]:
    for row in probe.get("aggregate", []) or []:
        if isinstance(row, dict) and abs(_as_float(row.get("gain")) - target_gain) <= 1.0e-12:
            return row
    return {}


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


def _leave_one_out_mean(values: Sequence[float]) -> list[float]:
    vals = [float(value) for value in values]
    if len(vals) <= 1:
        return []
    total = sum(vals)
    denom = len(vals) - 1
    return [float((total - value) / denom) for value in vals]


def _positive_outlier_share(case_rows: Sequence[dict[str, Any]]) -> float:
    positives = sorted(
        (_as_float(row.get("corr_delta")) for row in case_rows if _as_float(row.get("corr_delta")) > 0.0),
        reverse=True,
    )
    if not positives:
        return 0.0
    return float(positives[0] / max(1.0e-12, sum(positives)))


def _group_status(
    aggregate: dict[str, Any],
    *,
    min_leave_one_out_corr_delta: float | None,
    positive_outlier_share: float,
    split: str,
) -> str:
    mean_corr = _as_float(aggregate.get("mean_target_corr_delta_vs_gain0"))
    median_corr = _as_float(aggregate.get("median_target_corr_delta_vs_gain0"))
    mean_mse = _as_float(aggregate.get("mean_target_mse_delta_vs_gain0"))
    win_fraction = _as_float(aggregate.get("target_corr_win_fraction_vs_gain0"))
    min_loo = _as_float(min_leave_one_out_corr_delta, default=-1.0)
    strict = (
        mean_corr > 0.02
        and median_corr > 0.005
        and win_fraction >= 0.67
        and mean_mse <= 0.0
        and min_loo > 0.0
        and positive_outlier_share <= 0.5
    )
    if strict:
        return f"{split}_candidate_signal"
    if mean_corr > 0.02 and median_corr > 0.0 and win_fraction >= 0.5 and mean_mse <= 0.0:
        if split == "holdout":
            return f"{split}_holdout_like_signal"
        return f"{split}_robust_signal"
    if mean_corr > 0.0 and mean_mse <= 0.0:
        return f"{split}_tradeoff_signal"
    if mean_mse < 0.0:
        return f"{split}_mse_only_or_corr_tradeoff"
    return f"{split}_not_improved"


def _is_signal(row: dict[str, Any]) -> bool:
    status = str(row.get("status", ""))
    return (
        status.endswith("_candidate_signal")
        or status.endswith("_holdout_like_signal")
        or status.endswith("_tradeoff_signal")
    )


def _is_strict_signal(row: dict[str, Any]) -> bool:
    status = str(row.get("status", ""))
    return (
        status.endswith("_candidate_signal")
        or status.endswith("_holdout_like_signal")
        or status.endswith("_robust_signal")
    )


def _run_group(
    *,
    split: str,
    group: str,
    paths: Sequence[Path],
    config_path: Path,
    out_dir: Path,
    device_name: str,
    min_cases: int,
    prefix_seconds: float,
    future_seconds: float,
    magnitude_mode: str,
    mask_mode: str,
    mechanism: str,
    target_gain: float,
) -> dict[str, Any]:
    case_json = out_dir / "cases" / split / f"{group}_cases.json"
    cases = _write_cases_json(paths, case_json, f"{split}_{group}")
    if len(cases) < min_cases:
        return {
            "split": split,
            "family": group,
            "status": "insufficient_valid_cases",
            "case_count": len(cases),
            "case_json": str(case_json),
            "cases": cases,
        }

    probe_out = out_dir / "groups" / split / group
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
    corr_deltas = [_as_float(row.get("corr_delta")) for row in case_rows]
    loo_means = _leave_one_out_mean(corr_deltas)
    min_loo = min(loo_means) if loo_means else None
    outlier_share = _positive_outlier_share(case_rows)
    top_positive = max(case_rows, key=lambda row: _as_float(row.get("corr_delta")), default={})
    top_negative = min(case_rows, key=lambda row: _as_float(row.get("corr_delta")), default={})
    return {
        "split": split,
        "family": group,
        "status": _group_status(
            aggregate,
            min_leave_one_out_corr_delta=min_loo,
            positive_outlier_share=outlier_share,
            split=split,
        ),
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
        "future_target_magnitude_reused": bool(probe.get("future_target_magnitude_reused")),
        "target_future_stft_magnitude_accessed": bool(probe.get("target_future_stft_magnitude_accessed")),
        "future_target_phase_reused": bool(probe.get("future_target_phase_reused")),
        "target_future_stft_phase_accessed": bool(probe.get("target_future_stft_phase_accessed")),
    }


def _overall_status(groups: Sequence[dict[str, Any]], leakage: bool) -> str:
    if leakage:
        return "invalid_target_leakage"
    holdouts = [row for row in groups if row.get("split") == "holdout"]
    exploratory = [row for row in groups if row.get("split") == "exploratory"]
    nonrazor_structural_names = {
        "primary_no_razor_motor_union",
        "expanded_no_razor_motor_union",
        "nonrazor_appliance_holdout",
    }
    nonrazor_rows = [row for row in groups if row.get("family") in nonrazor_structural_names]
    razor_rows = [row for row in groups if "razor" in str(row.get("family", ""))]
    if any(_is_strict_signal(row) for row in nonrazor_rows):
        return "nonrazor_motor_structural_signal"
    if any(_is_signal(row) for row in nonrazor_rows) and any(_is_strict_signal(row) for row in razor_rows):
        return "nonrazor_motor_tradeoff_with_razor_diagnostic_signal"
    if any(_is_signal(row) for row in nonrazor_rows):
        return "nonrazor_motor_structural_tradeoff_only"
    if any(_is_strict_signal(row) for row in razor_rows):
        return "razor_holdout_signal"
    if any(_is_signal(row) for row in razor_rows):
        return "razor_holdout_tradeoff_only"
    if any(_is_signal(row) for row in holdouts):
        return "other_holdout_tradeoff_only"
    if any(str(row.get("status", "")).endswith("_candidate_signal") for row in exploratory):
        return "exploratory_motor_candidate_no_holdout"
    if any(_is_signal(row) for row in exploratory):
        return "exploratory_motor_tradeoff_only"
    return "no_electric_motor_signal"


def _selected_reuse_summary(groups: Sequence[dict[str, Any]]) -> dict[str, Any]:
    by_path: dict[str, list[str]] = {}
    by_hash: dict[str, dict[str, Any]] = {}
    for row in groups:
        family = str(row.get("family"))
        split = str(row.get("split"))
        label = f"{split}/{family}"
        cases = row.get("cases") if isinstance(row.get("cases"), dict) else {}
        for path_text in cases.values():
            path = Path(str(path_text))
            by_path.setdefault(str(path), []).append(label)
            try:
                digest = _file_sha256(path)
            except OSError:
                digest = "unreadable"
            entry = by_hash.setdefault(digest, {"paths": set(), "groups": []})
            entry["paths"].add(str(path))
            entry["groups"].append(label)
    path_reuse = {
        path: groups
        for path, groups in by_path.items()
        if len(groups) > 1
    }
    hash_reuse: dict[str, dict[str, Any]] = {}
    for digest, entry in by_hash.items():
        groups = entry["groups"]
        if len(groups) <= 1:
            continue
        hash_reuse[digest] = {
            "paths": sorted(entry["paths"]),
            "groups": groups,
        }
    return {
        "selected_case_occurrences": sum(len(groups) for groups in by_path.values()),
        "selected_unique_paths": len(by_path),
        "selected_path_reuse_count": sum(max(0, len(groups) - 1) for groups in by_path.values()),
        "selected_unique_content_hashes": len(by_hash),
        "selected_content_hash_reuse_count": sum(max(0, len(entry["groups"]) - 1) for entry in by_hash.values()),
        "selected_reused_paths_sample": [
            {"path": path, "groups": groups}
            for path, groups in list(sorted(path_reuse.items()))[:25]
        ],
        "selected_reused_hashes_sample": [
            {"sha256": digest, **entry}
            for digest, entry in list(sorted(hash_reuse.items()))[:25]
        ],
    }


def _write_markdown(summary: dict[str, Any], out_path: Path) -> None:
    def fmt(value: Any) -> str:
        if value is None:
            return "NA"
        if isinstance(value, (int, float)):
            return f"{float(value):.6g}"
        return str(value)

    lines = [
        "# Audio Electric Motor Holdout",
        "",
        "This assay freezes the Circleworld audio delta mechanism before selection:",
        "`prefix_hold` magnitude, `all_bins` mask, `time_smooth_3`, gain 2.0 versus gain 0.0.",
        "Electric-razor cases are withheld from exploratory subfamilies and scored as holdout groups.",
        "",
        f"- status: `{summary.get('status')}`",
        f"- config: `{summary.get('config_path')}`",
        f"- wav dir: `{summary.get('wav_dir')}`",
        f"- valid WAVs after broad exclusion/dedupe: `{fmt(summary.get('valid_wav_count'))}`",
        f"- rejected WAVs: `{fmt(summary.get('rejected_wav_count'))}`",
        f"- deduped duplicate stems: `{fmt(summary.get('deduped_duplicate_stem_count'))}`",
        f"- broad excluded paths: `{fmt(summary.get('excluded_path_count'))}`",
        f"- withheld razor exploratory paths: `{fmt(summary.get('withheld_razor_exploratory_count'))}`",
        f"- leakage detected: `{summary.get('target_leakage_detected')}`",
        f"- selected unique paths / occurrences: `{fmt(summary.get('selected_unique_paths'))}` / `{fmt(summary.get('selected_case_occurrences'))}`",
        f"- selected content-hash reuse count: `{fmt(summary.get('selected_content_hash_reuse_count'))}`",
        f"- best no-razor structural family: `{summary.get('best_nonrazor_motor_family')}` / `{summary.get('best_nonrazor_motor_status')}` / mean corr delta `{fmt(summary.get('best_nonrazor_motor_mean_corr_delta'))}`",
        f"- best razor family: `{summary.get('best_razor_family')}` / `{summary.get('best_razor_status')}` / mean corr delta `{fmt(summary.get('best_razor_mean_corr_delta'))}`",
        "",
        "| split | family | cases | status | mean corr d | median corr d | min LOO d | corr wins | mean MSE d | outlier share | top positive | top positive d | top negative | top negative d |",
        "|---|---|---:|---|---:|---:|---:|---:|---:|---:|---|---:|---|---:|",
    ]
    for row in summary.get("groups", []) or []:
        top_pos = row.get("top_positive_case") if isinstance(row.get("top_positive_case"), dict) else {}
        top_neg = row.get("top_negative_case") if isinstance(row.get("top_negative_case"), dict) else {}
        lines.append(
            "| {split} | {family} | {cases} | {status} | {mean} | {median} | {loo} | {wins} | {mse} | {share} | {pos} | {posd} | {neg} | {negd} |".format(
                split=row.get("split"),
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
            "- This is not per-family tuning: the mechanism, gain, mask, and magnitude policy are fixed.",
            "- Exploratory groups cannot use electric-razor files when `exclude_razor_from_exploratory` is true.",
            "- A positive holdout driven by one razor case remains a research signal, not an audio-model proof.",
        ]
    )
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_electric_motor_holdout(
    *,
    config_path: Path,
    out_dir: Path,
    wav_dir: Path,
    exclude_cases_json: Path | None,
    device_name: str,
    cases_per_family: int,
    cases_per_holdout: int,
    cases_per_union: int,
    min_cases: int,
    prefix_seconds: float,
    future_seconds: float,
    magnitude_mode: str,
    mask_mode: str,
    mechanism: str,
    target_gain: float,
    exclude_razor_from_exploratory: bool,
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    exclude_paths = _load_case_paths(exclude_cases_json)
    wavs, rejected, duplicate_stem_count = _iter_valid_wavs(wav_dir, exclude_paths)
    razor_paths = [path for path in wavs if _family_match(path, RAZOR_KEYWORDS)]
    exploratory_wavs = [path for path in wavs if path not in set(razor_paths)] if exclude_razor_from_exploratory else list(wavs)
    exploratory_paths = _select_cases(exploratory_wavs, EXPLORATORY_SPECS, cases_per_family=cases_per_family)
    for family in STRUCTURAL_MOTOR_GROUPS:
        if family in exploratory_paths:
            exploratory_paths[family] = [
                path
                for path in exploratory_paths[family]
                if not _family_match(path, EXCLUDED_FALSE_POSITIVE_KEYWORDS)
            ]
    primary_union = _unique_paths(
        path
        for family in PRIMARY_NO_RAZOR_UNION
        for path in exploratory_paths.get(family, [])
    )
    expanded_union = _unique_paths(
        path
        for family in EXPANDED_NO_RAZOR_UNION
        for path in exploratory_paths.get(family, [])
    )
    exploratory_paths["primary_no_razor_motor_union"] = primary_union[:cases_per_union]
    exploratory_paths["expanded_no_razor_motor_union"] = expanded_union[:cases_per_union]
    holdout_paths = _select_cases(wavs, HOLDOUT_SPECS, cases_per_family=cases_per_holdout)

    groups: list[dict[str, Any]] = []
    for family, paths in exploratory_paths.items():
        groups.append(
            _run_group(
                split="exploratory",
                group=family,
                paths=paths,
                config_path=config_path,
                out_dir=out_dir,
                device_name=device_name,
                min_cases=min_cases,
                prefix_seconds=prefix_seconds,
                future_seconds=future_seconds,
                magnitude_mode=magnitude_mode,
                mask_mode=mask_mode,
                mechanism=mechanism,
                target_gain=target_gain,
            )
        )
    for family, paths in holdout_paths.items():
        groups.append(
            _run_group(
                split="holdout",
                group=family,
                paths=paths,
                config_path=config_path,
                out_dir=out_dir,
                device_name=device_name,
                min_cases=min_cases,
                prefix_seconds=prefix_seconds,
                future_seconds=future_seconds,
                magnitude_mode=magnitude_mode,
                mask_mode=mask_mode,
                mechanism=mechanism,
                target_gain=target_gain,
            )
        )

    leakage = any(
        bool(row.get("future_target_magnitude_reused"))
        or bool(row.get("target_future_stft_magnitude_accessed"))
        or bool(row.get("future_target_phase_reused"))
        or bool(row.get("target_future_stft_phase_accessed"))
        for row in groups
    )
    best_exploratory = max(
        (row for row in groups if row.get("split") == "exploratory"),
        key=lambda row: _as_float(row.get("mean_corr_delta")),
        default={},
    )
    best_holdout = max(
        (row for row in groups if row.get("split") == "holdout"),
        key=lambda row: _as_float(row.get("mean_corr_delta")),
        default={},
    )
    nonrazor_structural_names = {
        "primary_no_razor_motor_union",
        "expanded_no_razor_motor_union",
        "nonrazor_appliance_holdout",
    }
    nonrazor_motor_rows = [row for row in groups if row.get("family") in nonrazor_structural_names]
    razor_holdout_rows = [row for row in groups if "razor" in str(row.get("family", ""))]
    best_nonrazor_motor = max(
        nonrazor_motor_rows,
        key=lambda row: _as_float(row.get("mean_corr_delta")),
        default={},
    )
    best_razor = max(
        razor_holdout_rows,
        key=lambda row: _as_float(row.get("mean_corr_delta")),
        default={},
    )
    reuse_summary = _selected_reuse_summary(groups)
    summary = {
        "runtime": "circleworld_proto",
        "schema": "circleworld_audio_electric_motor_holdout_v0",
        "status": _overall_status(groups, leakage),
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
        "selected_fixed_mechanism_row": {
            "status": "fixed_predeclared",
            "magnitude_mode": magnitude_mode,
            "mask_mode": mask_mode,
            "mechanism": mechanism,
            "gain": float(target_gain),
            "baseline_gain": 0.0,
        },
        "cases_per_family": int(cases_per_family),
        "cases_per_holdout": int(cases_per_holdout),
        "cases_per_union": int(cases_per_union),
        "min_cases": int(min_cases),
        "exclude_razor_from_exploratory": bool(exclude_razor_from_exploratory),
        "withheld_razor_exploratory_count": len(razor_paths) if exclude_razor_from_exploratory else 0,
        "withheld_razor_exploratory_paths": [str(path) for path in razor_paths] if exclude_razor_from_exploratory else [],
        "exploratory_specs": {family: list(keywords) for family, keywords in EXPLORATORY_SPECS.items()},
        "holdout_specs": {family: list(keywords) for family, keywords in HOLDOUT_SPECS.items()},
        "structural_motor_groups": sorted(STRUCTURAL_MOTOR_GROUPS),
        "primary_no_razor_union_groups": list(PRIMARY_NO_RAZOR_UNION),
        "expanded_no_razor_union_groups": list(EXPANDED_NO_RAZOR_UNION),
        "excluded_false_positive_keywords": list(EXCLUDED_FALSE_POSITIVE_KEYWORDS),
        "target_leakage_detected": leakage,
        "future_target_magnitude_reused": any(bool(row.get("future_target_magnitude_reused")) for row in groups),
        "target_future_stft_magnitude_accessed": any(bool(row.get("target_future_stft_magnitude_accessed")) for row in groups),
        "future_target_phase_reused": any(bool(row.get("future_target_phase_reused")) for row in groups),
        "target_future_stft_phase_accessed": any(bool(row.get("target_future_stft_phase_accessed")) for row in groups),
        "best_exploratory_family": best_exploratory.get("family"),
        "best_exploratory_status": best_exploratory.get("status"),
        "best_exploratory_mean_corr_delta": best_exploratory.get("mean_corr_delta"),
        "best_exploratory_median_corr_delta": best_exploratory.get("median_corr_delta"),
        "best_exploratory_corr_win_fraction": best_exploratory.get("corr_win_fraction"),
        "best_exploratory_positive_outlier_share": best_exploratory.get("positive_outlier_share"),
        "best_holdout_family": best_holdout.get("family"),
        "best_holdout_status": best_holdout.get("status"),
        "best_holdout_mean_corr_delta": best_holdout.get("mean_corr_delta"),
        "best_holdout_median_corr_delta": best_holdout.get("median_corr_delta"),
        "best_holdout_corr_win_fraction": best_holdout.get("corr_win_fraction"),
        "best_holdout_positive_outlier_share": best_holdout.get("positive_outlier_share"),
        "best_nonrazor_motor_family": best_nonrazor_motor.get("family"),
        "best_nonrazor_motor_status": best_nonrazor_motor.get("status"),
        "best_nonrazor_motor_mean_corr_delta": best_nonrazor_motor.get("mean_corr_delta"),
        "best_nonrazor_motor_median_corr_delta": best_nonrazor_motor.get("median_corr_delta"),
        "best_nonrazor_motor_corr_win_fraction": best_nonrazor_motor.get("corr_win_fraction"),
        "best_nonrazor_motor_positive_outlier_share": best_nonrazor_motor.get("positive_outlier_share"),
        "best_razor_family": best_razor.get("family"),
        "best_razor_status": best_razor.get("status"),
        "best_razor_mean_corr_delta": best_razor.get("mean_corr_delta"),
        "best_razor_median_corr_delta": best_razor.get("median_corr_delta"),
        "best_razor_corr_win_fraction": best_razor.get("corr_win_fraction"),
        "best_razor_positive_outlier_share": best_razor.get("positive_outlier_share"),
        **reuse_summary,
        "groups": groups,
    }
    (out_dir / OUTPUT_JSON).write_text(json.dumps(summary, indent=2), encoding="utf-8")
    _write_markdown(summary, out_dir / OUTPUT_MD)
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description="Run fixed Circleworld audio delta mechanism on electric-motor subfamilies and withheld razor holdout.")
    ap.add_argument("--config", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--wav-dir", default=str(DEFAULT_WAV_DIR))
    ap.add_argument("--exclude-cases-json", default=str(DEFAULT_EXCLUDE_CASES))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--cases-per-family", type=int, default=12)
    ap.add_argument("--cases-per-holdout", type=int, default=12)
    ap.add_argument("--cases-per-union", type=int, default=36)
    ap.add_argument("--min-cases", type=int, default=2)
    ap.add_argument("--prefix-seconds", type=float, default=1.0)
    ap.add_argument("--future-seconds", type=float, default=1.0)
    ap.add_argument("--magnitude-mode", default="prefix_hold")
    ap.add_argument("--mask-mode", default="all_bins")
    ap.add_argument("--mechanism", default="time_smooth_3")
    ap.add_argument("--target-gain", type=float, default=2.0)
    ap.add_argument("--include-razor-in-exploratory", action="store_true")
    args = ap.parse_args()
    summary = run_electric_motor_holdout(
        config_path=Path(args.config),
        out_dir=Path(args.out_dir),
        wav_dir=Path(args.wav_dir),
        exclude_cases_json=Path(args.exclude_cases_json) if args.exclude_cases_json else None,
        device_name=str(args.device),
        cases_per_family=int(args.cases_per_family),
        cases_per_holdout=int(args.cases_per_holdout),
        cases_per_union=int(args.cases_per_union),
        min_cases=int(args.min_cases),
        prefix_seconds=float(args.prefix_seconds),
        future_seconds=float(args.future_seconds),
        magnitude_mode=str(args.magnitude_mode),
        mask_mode=str(args.mask_mode),
        mechanism=str(args.mechanism),
        target_gain=float(args.target_gain),
        exclude_razor_from_exploratory=not bool(args.include_razor_in_exploratory),
    )
    print(
        json.dumps(
            {
                "json": str(Path(args.out_dir) / OUTPUT_JSON),
                "markdown": str(Path(args.out_dir) / OUTPUT_MD),
                "status": summary.get("status"),
                "best_exploratory_family": summary.get("best_exploratory_family"),
                "best_holdout_family": summary.get("best_holdout_family"),
                "best_holdout_mean_corr_delta": summary.get("best_holdout_mean_corr_delta"),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
