from __future__ import annotations

import argparse
import json
import sys
from collections import OrderedDict
from pathlib import Path
from typing import Any, Iterable, Sequence

ROOT = Path(__file__).resolve().parents[2]
RUNTIME = ROOT / "runtimes" / "circleworld_proto"
for path in (ROOT, RUNTIME):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from run_audio_delta_mechanism_probe import run_probe  # noqa: E402
from run_audio_electric_motor_holdout import (  # noqa: E402
    DEFAULT_EXCLUDE_CASES,
    DEFAULT_WAV_DIR,
    TOKENBURST_ROOT,
    _as_float,
    _canonical_stem,
    _case_deltas,
    _family_match,
    _file_sha256,
    _find_aggregate_row,
    _leave_one_out_mean,
    _load_case_paths,
    _mean,
    _positive_outlier_share,
    _valid_pcm_wav,
    _write_cases_json,
)


OUTPUT_JSON = "audio_steady_phenotype_manifest.json"
OUTPUT_MD = "AUDIO_STEADY_PHENOTYPE_MANIFEST.md"

GROUP_SPECS: "OrderedDict[str, dict[str, Any]]" = OrderedDict(
    [
        (
            "steady_buzz_nonmotor",
            {
                "role": "steady_buzz_control",
                "min_cases": 6,
                "include": (
                    "fluorescent light humming",
                    "old electrical buzz alarm",
                    "electrical sweep",
                    "electricity",
                    "buzzer",
                    "buzz fade",
                    "buzz",
                    "door buzzer",
                    "alarm alert effect",
                    "industrial alarm",
                ),
                "exclude": (
                    "buzzard",
                    "fly buzzing",
                    "mosquito buzzing",
                    "electric razor",
                    "razor",
                    "synth",
                    "buzzy synth",
                    "buzzy pad",
                ),
            },
        ),
        (
            "small_appliance_motor_nonrazor",
            {
                "role": "no_razor_motor",
                "min_cases": 5,
                "include": (
                    "air conditioner",
                    "air-conditioner",
                    "fan startup",
                    "can opener electric",
                    "electric toothbrush",
                    "clothes dryer",
                    "hair dryer",
                    "dishwasher",
                ),
                "exclude": ("razor", "electric razor", "typewriter"),
            },
        ),
        (
            "rotary_tool_motor_nonrazor",
            {
                "role": "no_razor_motor",
                "min_cases": 6,
                "include": (
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
                "exclude": ("dropping a wrench", "razor"),
            },
        ),
        (
            "saw_chain_tool_motor",
            {
                "role": "tool_motor_or_saw_control",
                "min_cases": 6,
                "include": (
                    "chain saw",
                    "chainsaw",
                    "chainsaw quick",
                    "buzz saw",
                    "buzzsaw",
                    "electric saw",
                    "electric table saw",
                    "table saw buzz",
                ),
                "exclude": (
                    "hacksaw",
                    "wood saw",
                    "bone saw",
                    "small bow saw",
                    "sawing wood",
                    "fast sawing",
                ),
            },
        ),
        (
            "typewriter_nonmotor_control",
            {
                "role": "mechanical_control",
                "min_cases": 3,
                "include": (
                    "electric typewriter",
                    "typewriter",
                    "typewriter and bell",
                    "typing on old typewriter",
                ),
                "exclude": (),
            },
        ),
        (
            "combustion_engine_control",
            {
                "role": "combustion_control",
                "min_cases": 8,
                "include": (
                    "motorcycle",
                    "motorbike",
                    "fast bike or motorcycle",
                    "engine",
                    "tractor",
                    "muscle car",
                ),
                "exclude": (
                    "electric motor",
                    "servo motor",
                    "steam train",
                    "steam engine",
                ),
            },
        ),
        (
            "buzzy_synth_control",
            {
                "role": "synth_control",
                "min_cases": 4,
                "include": (
                    "buzzy",
                    "buzzy pad",
                    "buzzy synth",
                    "synth lead",
                    "saw lead",
                    "sawtooth",
                    "saw c",
                    "saw f",
                ),
                "exclude": ("buzz saw", "buzzsaw", "table saw", "chain saw", "chainsaw"),
            },
        ),
        (
            "razor_single_source_probe",
            {
                "role": "single_source_probe",
                "min_cases": 1,
                "include": ("electric razor", "razor"),
                "exclude": (),
            },
        ),
    ]
)


def _iter_deduped_valid_wavs(wav_dir: Path, exclude_paths: set[str]) -> tuple[list[Path], list[dict[str, str]], int]:
    raw: list[Path] = []
    rejected: list[dict[str, str]] = []
    for path in sorted(wav_dir.glob("*.wav"), key=lambda item: item.name.lower()):
        if str(path) in exclude_paths:
            continue
        ok, reason = _valid_pcm_wav(path)
        if ok:
            raw.append(path)
        else:
            rejected.append({"path": str(path), "reason": reason})
    raw = sorted(raw, key=lambda item: (_canonical_stem(item), item.name.lower(), item.stat().st_size, str(item).lower()))
    out: list[Path] = []
    seen_stems: set[str] = set()
    duplicate_stems = 0
    for path in raw:
        stem = _canonical_stem(path)
        if stem in seen_stems:
            duplicate_stems += 1
            continue
        seen_stems.add(stem)
        out.append(path)
    return out, rejected, duplicate_stems


def _matches_spec(path: Path, spec: dict[str, Any]) -> bool:
    include = tuple(str(item) for item in spec.get("include", ()))
    exclude = tuple(str(item) for item in spec.get("exclude", ()))
    if not include or not _family_match(path, include):
        return False
    if exclude and _family_match(path, exclude):
        return False
    return True


def _select_disjoint_cases(
    wavs: Sequence[Path],
    *,
    cases_per_group: int,
) -> tuple[dict[str, list[Path]], dict[str, Any]]:
    selected: dict[str, list[Path]] = {}
    used_hashes: set[str] = set()
    rejected_reuse: list[dict[str, str]] = []
    for group, spec in GROUP_SPECS.items():
        rows: list[Path] = []
        for path in wavs:
            if len(rows) >= cases_per_group:
                break
            if not _matches_spec(path, spec):
                continue
            try:
                digest = _file_sha256(path)
            except OSError:
                continue
            if digest in used_hashes:
                rejected_reuse.append({"group": group, "path": str(path), "sha256": digest})
                continue
            used_hashes.add(digest)
            rows.append(path)
        selected[group] = rows
    selected_paths = [path for rows in selected.values() for path in rows]
    return selected, {
        "selected_case_occurrences": len(selected_paths),
        "selected_unique_paths": len({str(path) for path in selected_paths}),
        "selected_unique_content_hashes": len(used_hashes),
        "selection_reuse_rejections": rejected_reuse[:50],
        "selection_reuse_rejection_count": len(rejected_reuse),
    }


def _status_for_row(row: dict[str, Any], *, min_loo: float | None, outlier_share: float, role: str) -> str:
    mean_corr = _as_float(row.get("mean_target_corr_delta_vs_gain0"))
    median_corr = _as_float(row.get("median_target_corr_delta_vs_gain0"))
    mean_mse = _as_float(row.get("mean_target_mse_delta_vs_gain0"))
    win_fraction = _as_float(row.get("target_corr_win_fraction_vs_gain0"))
    min_loo_value = _as_float(min_loo, default=-1.0)
    if role == "single_source_probe":
        if mean_corr > 0.02 and mean_mse <= 0.0:
            return "single_source_probe_signal"
        if mean_corr > 0.0 and mean_mse <= 0.0:
            return "single_source_probe_tradeoff"
        return "single_source_probe_not_improved"
    if (
        mean_corr > 0.02
        and median_corr > 0.005
        and win_fraction >= 0.67
        and mean_mse <= 0.0
        and min_loo_value > 0.0
        and outlier_share <= 0.5
    ):
        return f"{role}_candidate_signal"
    if mean_corr > 0.02 and median_corr > 0.0 and win_fraction >= 0.5 and mean_mse <= 0.0:
        return f"{role}_robust_tradeoff"
    if mean_corr > 0.0 and mean_mse <= 0.0:
        return f"{role}_tradeoff_signal"
    if mean_mse < 0.0:
        return f"{role}_mse_only_or_corr_tradeoff"
    return f"{role}_not_improved"


def _run_group(
    *,
    group: str,
    spec: dict[str, Any],
    paths: Sequence[Path],
    config_path: Path,
    out_dir: Path,
    device_name: str,
    prefix_seconds: float,
    future_seconds: float,
    magnitude_mode: str,
    mask_mode: str,
    mechanism: str,
    target_gain: float,
) -> dict[str, Any]:
    role = str(spec.get("role", "unknown"))
    min_cases = int(spec.get("min_cases", 2))
    case_json = out_dir / "cases" / f"{group}_cases.json"
    cases = _write_cases_json(paths, case_json, group)
    if len(cases) < min_cases:
        return {
            "group": group,
            "role": role,
            "status": "insufficient_valid_cases",
            "case_count": len(cases),
            "min_cases": min_cases,
            "case_json": str(case_json),
            "cases": cases,
        }
    probe_out = out_dir / "groups" / group
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
        "group": group,
        "role": role,
        "status": _status_for_row(aggregate, min_loo=min_loo, outlier_share=outlier_share, role=role),
        "raw_probe_status": probe.get("status"),
        "case_count": len(cases),
        "min_cases": min_cases,
        "case_json": str(case_json),
        "probe_json": str(probe_out / "audio_delta_mechanism_probe.json"),
        "mean_corr_delta": aggregate.get("mean_target_corr_delta_vs_gain0"),
        "median_corr_delta": aggregate.get("median_target_corr_delta_vs_gain0"),
        "corr_win_fraction": aggregate.get("target_corr_win_fraction_vs_gain0"),
        "mean_mse_delta": aggregate.get("mean_target_mse_delta_vs_gain0"),
        "mse_win_fraction": aggregate.get("target_mse_win_fraction_vs_gain0"),
        "positive_case_count": sum(1 for value in corr_deltas if value > 1.0e-6),
        "min_leave_one_out_corr_delta": min_loo,
        "mean_leave_one_out_corr_delta": _mean(loo_means) if loo_means else None,
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


def _overall_status(rows: Sequence[dict[str, Any]], leakage: bool) -> str:
    if leakage:
        return "invalid_target_leakage"
    steady = [row for row in rows if row.get("role") == "steady_buzz_control"]
    motors = [row for row in rows if row.get("role") == "no_razor_motor"]
    tools = [row for row in rows if row.get("role") == "tool_motor_or_saw_control"]
    if any(str(row.get("status", "")).endswith("_candidate_signal") for row in motors):
        return "no_razor_motor_candidate_signal"
    if any(str(row.get("status", "")).endswith("_candidate_signal") for row in steady):
        return "steady_buzz_candidate_motor_not_supported"
    if any("robust" in str(row.get("status", "")) for row in steady):
        return "steady_buzz_robust_tradeoff_motor_not_supported"
    if any("tradeoff_signal" in str(row.get("status", "")) for row in motors + tools + steady):
        return "disjoint_phenotype_tradeoff_only"
    return "no_disjoint_phenotype_signal"


def _best_row(rows: Sequence[dict[str, Any]], role: str | None = None) -> dict[str, Any]:
    candidates = [row for row in rows if role is None or row.get("role") == role]
    return max(candidates, key=lambda row: _as_float(row.get("mean_corr_delta")), default={})


def _write_markdown(summary: dict[str, Any], out_path: Path) -> None:
    def fmt(value: Any) -> str:
        if value is None:
            return "NA"
        if isinstance(value, (int, float)):
            return f"{float(value):.6g}"
        return str(value)

    lines = [
        "# Audio Steady Phenotype Manifest",
        "",
        "This assay allocates each selected WAV to at most one group by provider-neutral stem and content hash.",
        "The mechanism row is frozen: `prefix_hold` / `all_bins` / `time_smooth_3` / gain 2.0 versus gain 0.0.",
        "",
        f"- status: `{summary.get('status')}`",
        f"- valid WAVs after broad exclusion/dedupe: `{fmt(summary.get('valid_wav_count'))}`",
        f"- rejected WAVs: `{fmt(summary.get('rejected_wav_count'))}`",
        f"- provider-stem duplicates dropped: `{fmt(summary.get('deduped_duplicate_stem_count'))}`",
        f"- selected unique paths / occurrences: `{fmt(summary.get('selected_unique_paths'))}` / `{fmt(summary.get('selected_case_occurrences'))}`",
        f"- selected unique content hashes: `{fmt(summary.get('selected_unique_content_hashes'))}`",
        f"- leakage detected: `{summary.get('target_leakage_detected')}`",
        "",
        "| group | role | cases | status | mean corr d | median corr d | min LOO d | corr wins | mean MSE d | outlier share | top positive | top positive d | top negative | top negative d |",
        "|---|---|---:|---|---:|---:|---:|---:|---:|---:|---|---:|---|---:|",
    ]
    for row in summary.get("groups", []) or []:
        top_pos = row.get("top_positive_case") if isinstance(row.get("top_positive_case"), dict) else {}
        top_neg = row.get("top_negative_case") if isinstance(row.get("top_negative_case"), dict) else {}
        lines.append(
            "| {group} | {role} | {cases} | {status} | {mean} | {median} | {loo} | {wins} | {mse} | {share} | {pos} | {posd} | {neg} | {negd} |".format(
                group=row.get("group"),
                role=row.get("role"),
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
            "- This is still filename-derived, but selected cases are disjoint and content-hash unique.",
            "- A single-source razor probe is diagnostic only, never a family promotion.",
            "- A broad motor claim requires no-razor motor rows to beat steady-buzz and synth controls.",
        ]
    )
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_steady_phenotype_manifest(
    *,
    config_path: Path,
    out_dir: Path,
    wav_dir: Path,
    exclude_cases_json: Path | None,
    device_name: str,
    cases_per_group: int,
    prefix_seconds: float,
    future_seconds: float,
    magnitude_mode: str,
    mask_mode: str,
    mechanism: str,
    target_gain: float,
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    exclude_paths = _load_case_paths(exclude_cases_json)
    wavs, rejected, duplicate_stems = _iter_deduped_valid_wavs(wav_dir, exclude_paths)
    selected, selection_summary = _select_disjoint_cases(wavs, cases_per_group=cases_per_group)
    groups = [
        _run_group(
            group=group,
            spec=GROUP_SPECS[group],
            paths=paths,
            config_path=config_path,
            out_dir=out_dir,
            device_name=device_name,
            prefix_seconds=prefix_seconds,
            future_seconds=future_seconds,
            magnitude_mode=magnitude_mode,
            mask_mode=mask_mode,
            mechanism=mechanism,
            target_gain=target_gain,
        )
        for group, paths in selected.items()
    ]
    leakage = any(
        bool(row.get("future_target_magnitude_reused"))
        or bool(row.get("target_future_stft_magnitude_accessed"))
        or bool(row.get("future_target_phase_reused"))
        or bool(row.get("target_future_stft_phase_accessed"))
        for row in groups
    )
    summary = {
        "runtime": "circleworld_proto",
        "schema": "circleworld_audio_steady_phenotype_manifest_v0",
        "status": _overall_status(groups, leakage),
        "config_path": str(config_path),
        "out_dir": str(out_dir),
        "wav_dir": str(wav_dir),
        "exclude_cases_json": str(exclude_cases_json) if exclude_cases_json else None,
        "excluded_path_count": len(exclude_paths),
        "valid_wav_count": len(wavs),
        "rejected_wav_count": len(rejected),
        "deduped_duplicate_stem_count": duplicate_stems,
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
        "cases_per_group": int(cases_per_group),
        "target_leakage_detected": leakage,
        "future_target_magnitude_reused": any(bool(row.get("future_target_magnitude_reused")) for row in groups),
        "target_future_stft_magnitude_accessed": any(bool(row.get("target_future_stft_magnitude_accessed")) for row in groups),
        "future_target_phase_reused": any(bool(row.get("future_target_phase_reused")) for row in groups),
        "target_future_stft_phase_accessed": any(bool(row.get("target_future_stft_phase_accessed")) for row in groups),
        "group_specs": GROUP_SPECS,
        "best_overall_group": _best_row(groups).get("group"),
        "best_overall_status": _best_row(groups).get("status"),
        "best_overall_mean_corr_delta": _best_row(groups).get("mean_corr_delta"),
        "best_steady_buzz_group": _best_row(groups, "steady_buzz_control").get("group"),
        "best_steady_buzz_status": _best_row(groups, "steady_buzz_control").get("status"),
        "best_steady_buzz_mean_corr_delta": _best_row(groups, "steady_buzz_control").get("mean_corr_delta"),
        "best_no_razor_motor_group": _best_row(groups, "no_razor_motor").get("group"),
        "best_no_razor_motor_status": _best_row(groups, "no_razor_motor").get("status"),
        "best_no_razor_motor_mean_corr_delta": _best_row(groups, "no_razor_motor").get("mean_corr_delta"),
        **selection_summary,
        "groups": groups,
    }
    (out_dir / OUTPUT_JSON).write_text(json.dumps(summary, indent=2), encoding="utf-8")
    _write_markdown(summary, out_dir / OUTPUT_MD)
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description="Run a disjoint steady-buzz/motor phenotype manifest with a frozen Circleworld audio mechanism row.")
    ap.add_argument("--config", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--wav-dir", default=str(DEFAULT_WAV_DIR))
    ap.add_argument("--exclude-cases-json", default=str(DEFAULT_EXCLUDE_CASES))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--cases-per-group", type=int, default=12)
    ap.add_argument("--prefix-seconds", type=float, default=1.0)
    ap.add_argument("--future-seconds", type=float, default=1.0)
    ap.add_argument("--magnitude-mode", default="prefix_hold")
    ap.add_argument("--mask-mode", default="all_bins")
    ap.add_argument("--mechanism", default="time_smooth_3")
    ap.add_argument("--target-gain", type=float, default=2.0)
    args = ap.parse_args()
    summary = run_steady_phenotype_manifest(
        config_path=Path(args.config),
        out_dir=Path(args.out_dir),
        wav_dir=Path(args.wav_dir),
        exclude_cases_json=Path(args.exclude_cases_json) if args.exclude_cases_json else None,
        device_name=str(args.device),
        cases_per_group=int(args.cases_per_group),
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
                "status": summary.get("status"),
                "best_overall_group": summary.get("best_overall_group"),
                "best_no_razor_motor_group": summary.get("best_no_razor_motor_group"),
                "best_steady_buzz_group": summary.get("best_steady_buzz_group"),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
