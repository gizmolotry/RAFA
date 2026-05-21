from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Iterable, Sequence


OUTPUT_JSON = "audio_lockbox_result_compare.json"
OUTPUT_MD = "AUDIO_LOCKBOX_RESULT_COMPARE.md"

BASELINE_GAIN = 0.0
FIXED_BASELINE_BINS = (
    ("bad_baseline", float("-inf"), -0.05),
    ("weak_baseline", -0.05, 0.05),
    ("moderate_baseline", 0.05, 0.20),
    ("good_baseline", 0.20, float("inf")),
)


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    if math.isnan(out):
        return default
    return out


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


def _fmt(value: Any) -> str:
    if value is None:
        return "NA"
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, (int, float)):
        if value == float("inf"):
            return "inf"
        if value == float("-inf"):
            return "-inf"
        return f"{float(value):.6g}"
    return str(value)


def _safe_label(value: Any, fallback: str) -> str:
    text = str(value or "").strip()
    return text if text else fallback


def _norm_path(value: Any) -> str:
    if value is None:
        return ""
    raw = str(value).strip()
    if not raw:
        return ""
    try:
        return str(Path(raw).resolve()).lower()
    except (OSError, RuntimeError):
        return raw.replace("/", "\\").lower()


def _case_key(row: dict[str, Any]) -> str:
    path = _norm_path(row.get("source_wav") or row.get("path") or row.get("wav_path"))
    if path:
        return path
    return str(row.get("name") or row.get("case_name") or "")


def _row_key(row: dict[str, Any]) -> tuple[str, str, str, float]:
    return (
        str(row.get("magnitude_mode", "")),
        str(row.get("mask_mode", "")),
        str(row.get("mechanism", "")),
        float(_as_float(row.get("gain"))),
    )


def _baseline_key(row: dict[str, Any]) -> tuple[str, str, str]:
    return (
        str(row.get("magnitude_mode", "")),
        str(row.get("mask_mode", "")),
        str(row.get("mechanism", "")),
    )


def _positive_outlier_share(rows: Sequence[dict[str, Any]]) -> float:
    positives = sorted(
        (_as_float(row.get("corr_delta")) for row in rows if _as_float(row.get("corr_delta")) > 0.0),
        reverse=True,
    )
    if not positives:
        return 0.0
    return float(positives[0] / max(1.0e-12, sum(positives)))


def _positive_corr_sum(rows: Sequence[dict[str, Any]]) -> float:
    return float(sum(max(0.0, _as_float(row.get("corr_delta"))) for row in rows))


def _leave_one_out_mean(values: Sequence[float]) -> list[float]:
    vals = [float(value) for value in values]
    if len(vals) <= 1:
        return []
    total = sum(vals)
    denom = len(vals) - 1
    return [float((total - value) / denom) for value in vals]


def _bin_name(value: float) -> str:
    for name, lo, hi in FIXED_BASELINE_BINS:
        if value >= lo and value < hi:
            return name
    return "unbinned"


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"Expected JSON object at {path}")
    return payload


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    return value


def _parse_labeled_path(raw: str) -> tuple[str, Path]:
    if "=" in raw:
        label, path = raw.split("=", 1)
        return _safe_label(label, Path(path).stem), Path(path)
    path = Path(raw)
    return path.parent.name or path.stem, path


def _case_index_from_manifest(manifest: dict[str, Any] | None) -> dict[str, dict[str, Any]]:
    if not isinstance(manifest, dict):
        return {}
    out: dict[str, dict[str, Any]] = {}

    def add_case(case: dict[str, Any], group: str | None = None, role: str | None = None) -> None:
        meta = {
            "manifest_group": group or case.get("group") or case.get("family"),
            "manifest_role": role or case.get("role"),
            "manifest_case_status": case.get("status"),
        }
        for key in (
            case.get("source_wav"),
            case.get("wav_path"),
            case.get("path"),
            case.get("case_path"),
            case.get("name"),
            case.get("case_name"),
        ):
            norm = _norm_path(key) if str(key or "").lower().endswith(".wav") else str(key or "")
            if norm:
                out[norm] = {k: v for k, v in meta.items() if v is not None}

    for case in manifest.get("cases", []) or []:
        if isinstance(case, dict):
            add_case(case)
    for group_key in ("groups", "families"):
        raw_groups = manifest.get(group_key, []) or []
        group_iter = raw_groups.values() if isinstance(raw_groups, dict) else raw_groups
        for group in group_iter:
            if not isinstance(group, dict):
                continue
            group_name = str(group.get("group") or group.get("group_id") or group.get("family") or group.get("name") or "")
            role = str(group.get("role") or "") or None
            for case in group.get("cases", []) or []:
                if isinstance(case, dict):
                    add_case(case, group=group_name, role=role)
                else:
                    add_case({"path": case, "name": Path(str(case)).stem}, group=group_name, role=role)
    return out


def _declared_rows_from_manifest(manifest: dict[str, Any] | None) -> list[dict[str, Any]]:
    if not isinstance(manifest, dict):
        return []
    candidates: list[dict[str, Any]] = []
    for key in ("selected_fixed_mechanism_row", "fixed_mechanism_row", "predeclared_row", "declared_row"):
        value = manifest.get(key)
        if isinstance(value, dict):
            candidates.append(value)
    for key in ("predeclared_rows", "declared_rows", "fixed_rows", "selected_fixed_mechanism_rows"):
        for value in manifest.get(key, []) or []:
            if isinstance(value, dict):
                candidates.append(value)
    if {"magnitude_mode", "mask_mode", "mechanism"} <= set(manifest):
        candidates.append(
            {
                "magnitude_mode": manifest.get("magnitude_mode"),
                "mask_mode": manifest.get("mask_mode"),
                "mechanism": manifest.get("mechanism"),
                "gain": manifest.get("target_gain", manifest.get("gain")),
                "baseline_gain": manifest.get("baseline_gain", BASELINE_GAIN),
            }
        )

    out: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str, float]] = set()
    for row in candidates:
        if not {"magnitude_mode", "mask_mode", "mechanism"} <= set(row):
            continue
        gain = _as_float(row.get("gain", row.get("target_gain")), default=float("nan"))
        if math.isnan(gain):
            continue
        key = (
            str(row.get("magnitude_mode")),
            str(row.get("mask_mode")),
            str(row.get("mechanism")),
            float(gain),
        )
        if key in seen:
            continue
        seen.add(key)
        out.append(
            {
                "magnitude_mode": key[0],
                "mask_mode": key[1],
                "mechanism": key[2],
                "gain": key[3],
                "baseline_gain": _as_float(row.get("baseline_gain"), BASELINE_GAIN),
                "source": row.get("status", "manifest"),
            }
        )
    return out


def _matches_declared(row: dict[str, Any], declared_rows: Sequence[dict[str, Any]]) -> bool:
    if not declared_rows:
        return False
    for declared in declared_rows:
        if (
            str(row.get("magnitude_mode")) == str(declared.get("magnitude_mode"))
            and str(row.get("mask_mode")) == str(declared.get("mask_mode"))
            and str(row.get("mechanism")) == str(declared.get("mechanism"))
            and abs(_as_float(row.get("gain")) - _as_float(declared.get("gain"))) <= 1.0e-12
        ):
            return True
    return False


def _iter_probe_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    cases = payload.get("cases")
    if isinstance(cases, list):
        for case in cases:
            if not isinstance(case, dict):
                continue
            case_rows = case.get("rows")
            if not isinstance(case_rows, list):
                continue
            case_meta = {
                "name": case.get("name"),
                "source_wav": case.get("source_wav"),
                "sample_rate": case.get("sample_rate"),
                "phase_seed_policy": case.get("phase_seed_policy", payload.get("phase_seed_policy")),
                "prefix_samples": case.get("prefix_samples"),
                "future_samples": case.get("future_samples"),
            }
            for row in case_rows:
                if isinstance(row, dict):
                    merged = dict(case_meta)
                    merged.update(row)
                    rows.append(merged)
    if rows:
        return rows

    flat = payload.get("rows")
    if isinstance(flat, list):
        return [dict(row) for row in flat if isinstance(row, dict)]
    return []


def _iter_compatible_delta_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    candidates = payload.get("case_deltas")
    if not isinstance(candidates, list):
        candidates = payload.get("cases")
    if isinstance(candidates, list):
        for row in candidates:
            if not isinstance(row, dict):
                continue
            if "corr_delta" not in row and "gain0_target_corr" not in row:
                continue
            out.append(
                {
                    "name": row.get("name") or row.get("case_name"),
                    "source_wav": row.get("source_wav") or row.get("path"),
                    "magnitude_mode": row.get("magnitude_mode", payload.get("magnitude_mode")),
                    "mask_mode": row.get("mask_mode", payload.get("mask_mode")),
                    "mechanism": row.get("mechanism", payload.get("mechanism")),
                    "gain": row.get("gain", payload.get("target_gain")),
                    "target_corr": row.get("target_corr"),
                    "target_mse": row.get("target_mse"),
                    "baseline_target_corr": row.get("gain0_target_corr"),
                    "baseline_target_mse": row.get("gain0_target_mse"),
                    "corr_delta": row.get("corr_delta"),
                    "mse_delta": row.get("mse_delta"),
                    "group": row.get("group", payload.get("group")),
                    "role": row.get("role", payload.get("role")),
                }
            )
    return out


def _normalise_result_rows(
    *,
    label: str,
    result_path: Path,
    payload: dict[str, Any],
    manifest_index: dict[str, dict[str, Any]],
    declared_rows: Sequence[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    raw_rows = _iter_probe_rows(payload)
    source_schema = str(payload.get("schema", "compatible_delta_rows"))
    compatible_mode = False
    if not raw_rows:
        raw_rows = _iter_compatible_delta_rows(payload)
        compatible_mode = True

    baselines: dict[tuple[str, tuple[str, str, str]], dict[str, Any]] = {}
    for row in raw_rows:
        case_key = _case_key(row)
        if not case_key:
            continue
        if abs(_as_float(row.get("gain"))) <= 1.0e-12:
            baselines[(case_key, _baseline_key(row))] = row

    normalised: list[dict[str, Any]] = []
    for row in raw_rows:
        case_key = _case_key(row)
        if not case_key:
            case_key = str(row.get("name") or f"case_{len(normalised)}")
        baseline = baselines.get((case_key, _baseline_key(row)), {})
        manifest_meta = manifest_index.get(case_key) or manifest_index.get(str(row.get("name") or "")) or {}
        target_corr = _as_float(row.get("target_corr"), default=_as_float(row.get("target_correlation")))
        target_mse = _as_float(row.get("target_mse"), default=_as_float(row.get("mse")))
        baseline_corr = _as_float(
            row.get("baseline_target_corr", row.get("gain0_target_corr")),
            default=_as_float(baseline.get("target_corr"), default=target_corr),
        )
        baseline_mse = _as_float(
            row.get("baseline_target_mse", row.get("gain0_target_mse")),
            default=_as_float(baseline.get("target_mse"), default=target_mse),
        )
        corr_delta = _as_float(row.get("corr_delta"), default=target_corr - baseline_corr)
        mse_delta = _as_float(row.get("mse_delta"), default=target_mse - baseline_mse)
        out = {
            "run_label": label,
            "result_path": str(result_path),
            "schema": source_schema,
            "name": row.get("name") or row.get("case_name"),
            "case_key": case_key,
            "source_wav": row.get("source_wav") or row.get("path") or row.get("wav_path"),
            "group": row.get("group") or manifest_meta.get("manifest_group") or "all_cases",
            "role": row.get("role") or manifest_meta.get("manifest_role"),
            "magnitude_mode": row.get("magnitude_mode"),
            "mask_mode": row.get("mask_mode"),
            "mechanism": row.get("mechanism"),
            "gain": _as_float(row.get("gain")),
            "target_corr": target_corr,
            "target_mse": target_mse,
            "target_mae": _as_float(row.get("target_mae")),
            "baseline_target_corr": baseline_corr,
            "baseline_target_mse": baseline_mse,
            "corr_delta": corr_delta,
            "mse_delta": mse_delta,
            "baseline_corr_bin": _bin_name(baseline_corr),
            "vs_gain0_corr": _as_float(row.get("vs_gain0_corr")),
            "vs_gain0_mse": _as_float(row.get("vs_gain0_mse")),
            "loop_autocorr_peak": _as_float(row.get("loop_autocorr_peak")),
            "first_chunk_reentry": _as_float(row.get("first_chunk_reentry")),
            "mask_bin_fraction": _as_float(row.get("mask_bin_fraction")),
            "predeclared_row_match": _matches_declared(row, declared_rows),
            "compatible_delta_row": compatible_mode,
            "manifest_case_status": manifest_meta.get("manifest_case_status"),
            "future_target_audio_used_for_metrics_only": bool(row.get("future_target_audio_used_for_metrics_only", True)),
            "future_target_magnitude_reused": bool(row.get("future_target_magnitude_reused")),
            "target_future_stft_magnitude_accessed": bool(row.get("target_future_stft_magnitude_accessed")),
            "future_target_phase_reused": bool(row.get("future_target_phase_reused")),
            "target_future_stft_phase_accessed": bool(row.get("target_future_stft_phase_accessed")),
        }
        normalised.append(out)

    run_meta = {
        "label": label,
        "result_path": str(result_path),
        "schema": source_schema,
        "raw_status": payload.get("status"),
        "raw_case_count": payload.get("case_count"),
        "normalised_row_count": len(normalised),
        "normalised_case_count": len({row.get("case_key") for row in normalised}),
        "future_target_magnitude_reused": bool(payload.get("future_target_magnitude_reused"))
        or any(bool(row.get("future_target_magnitude_reused")) for row in normalised),
        "target_future_stft_magnitude_accessed": bool(payload.get("target_future_stft_magnitude_accessed"))
        or any(bool(row.get("target_future_stft_magnitude_accessed")) for row in normalised),
        "future_target_phase_reused": bool(payload.get("future_target_phase_reused"))
        or any(bool(row.get("future_target_phase_reused")) for row in normalised),
        "target_future_stft_phase_accessed": bool(payload.get("target_future_stft_phase_accessed"))
        or any(bool(row.get("target_future_stft_phase_accessed")) for row in normalised),
        "predeclared_match_count": sum(1 for row in normalised if row.get("predeclared_row_match")),
    }
    return normalised, run_meta


def _summarise_rows(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    corr = [_as_float(row.get("corr_delta")) for row in rows]
    mse = [_as_float(row.get("mse_delta")) for row in rows]
    loo = _leave_one_out_mean(corr)
    top_positive = max(rows, key=lambda row: _as_float(row.get("corr_delta")), default={})
    top_negative = min(rows, key=lambda row: _as_float(row.get("corr_delta")), default={})
    outlier_share = _positive_outlier_share(rows)
    return {
        "case_count": len(rows),
        "unique_case_count": len({row.get("case_key") for row in rows}),
        "mean_baseline_target_corr": _mean([_as_float(row.get("baseline_target_corr")) for row in rows]),
        "median_baseline_target_corr": _median([_as_float(row.get("baseline_target_corr")) for row in rows]),
        "mean_target_corr": _mean([_as_float(row.get("target_corr")) for row in rows]),
        "median_target_corr": _median([_as_float(row.get("target_corr")) for row in rows]),
        "mean_corr_delta": _mean(corr),
        "median_corr_delta": _median(corr),
        "corr_win_fraction": _mean([1.0 if value > 1.0e-6 else 0.0 for value in corr]),
        "mean_target_mse": _mean([_as_float(row.get("target_mse")) for row in rows]),
        "mean_baseline_target_mse": _mean([_as_float(row.get("baseline_target_mse")) for row in rows]),
        "mean_mse_delta": _mean(mse),
        "median_mse_delta": _median(mse),
        "mse_win_fraction": _mean([1.0 if value < -1.0e-9 else 0.0 for value in mse]),
        "positive_case_count": sum(1 for value in corr if value > 1.0e-6),
        "negative_case_count": sum(1 for value in corr if value < -1.0e-6),
        "positive_outlier_share": outlier_share,
        "positive_corr_sum": _positive_corr_sum(rows),
        "min_leave_one_out_corr_delta": min(loo) if loo else None,
        "mean_leave_one_out_corr_delta": _mean(loo) if loo else None,
        "top_positive_case": _case_brief(top_positive),
        "top_negative_case": _case_brief(top_negative),
        "outlier_pressure": {
            "positive_outlier_share": outlier_share,
            "largest_positive_corr_delta": max([0.0] + [value for value in corr if value > 0.0]),
            "min_leave_one_out_corr_delta": min(loo) if loo else None,
            "positive_case_count": sum(1 for value in corr if value > 1.0e-6),
            "status": _outlier_pressure_status(outlier_share, loo),
        },
    }


def _outlier_pressure_status(outlier_share: float, loo: Sequence[float]) -> str:
    if outlier_share <= 0.0:
        return "no_positive_cases"
    if outlier_share > 0.5:
        return "single_case_pressure"
    if loo and min(loo) <= 0.0:
        return "leave_one_out_fragile"
    return "distributed"


def _case_brief(row: dict[str, Any]) -> dict[str, Any]:
    if not row:
        return {}
    return {
        "run_label": row.get("run_label"),
        "name": row.get("name"),
        "source_wav": row.get("source_wav"),
        "group": row.get("group"),
        "magnitude_mode": row.get("magnitude_mode"),
        "mask_mode": row.get("mask_mode"),
        "mechanism": row.get("mechanism"),
        "gain": row.get("gain"),
        "baseline_target_corr": row.get("baseline_target_corr"),
        "target_corr": row.get("target_corr"),
        "corr_delta": row.get("corr_delta"),
        "mse_delta": row.get("mse_delta"),
    }


def _fixed_bin_summaries(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for name, lo, hi in FIXED_BASELINE_BINS:
        local = [
            row
            for row in rows
            if _as_float(row.get("baseline_target_corr")) >= lo and _as_float(row.get("baseline_target_corr")) < hi
        ]
        summary = _summarise_rows(local)
        summary.update({"bin": name, "lo": lo, "hi": hi})
        out.append(summary)
    return out


def _bad_baseline_dominated(rows: Sequence[dict[str, Any]], bins: Sequence[dict[str, Any]]) -> bool:
    total_positive = _positive_corr_sum(rows)
    if total_positive <= 0.0:
        return False
    bad_rows = [row for row in rows if row.get("baseline_corr_bin") == "bad_baseline"]
    bad_bin = next((row for row in bins if row.get("bin") == "bad_baseline"), {})
    bad_positive_share = _positive_corr_sum(bad_rows) / max(1.0e-12, total_positive)
    non_bad_signal = any(
        row.get("bin") != "bad_baseline"
        and int(row.get("case_count") or 0) > 0
        and _as_float(row.get("median_corr_delta")) > 0.005
        and _as_float(row.get("corr_win_fraction")) >= 0.67
        for row in bins
    )
    return bool(
        bad_rows
        and _as_float(bad_bin.get("mean_corr_delta")) > 0.02
        and bad_positive_share >= 0.67
        and not non_bad_signal
    )


def _status_for_summary(summary: dict[str, Any], bins: Sequence[dict[str, Any]], leakage: bool) -> str:
    if leakage:
        return "invalid_future_leakage"
    if int(summary.get("case_count") or 0) <= 0:
        return "missing_cases"
    mean_corr = _as_float(summary.get("mean_corr_delta"))
    median_corr = _as_float(summary.get("median_corr_delta"))
    corr_wins = _as_float(summary.get("corr_win_fraction"))
    mean_mse = _as_float(summary.get("mean_mse_delta"))
    min_loo = summary.get("min_leave_one_out_corr_delta")
    outlier_share = _as_float(summary.get("positive_outlier_share"))
    if (
        mean_corr > 0.02
        and median_corr > 0.005
        and corr_wins >= 0.67
        and mean_mse <= 0.0
        and (min_loo is None or _as_float(min_loo) > 0.0)
        and outlier_share <= 0.5
    ):
        return "robust_signal"
    if mean_corr > 0.0 and mean_mse <= 0.0:
        return "tradeoff_signal_only"
    if mean_mse < 0.0:
        return "mse_only_or_corr_tradeoff"
    return "no_signal"


def _group_summaries(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str, str, str, float], list[dict[str, Any]]] = {}
    for row in rows:
        key = (
            str(row.get("run_label")),
            str(row.get("group") or "all_cases"),
            str(row.get("magnitude_mode")),
            str(row.get("mask_mode")),
            str(row.get("mechanism")),
            _as_float(row.get("gain")),
        )
        grouped.setdefault(key, []).append(row)

    out: list[dict[str, Any]] = []
    for key, local in sorted(grouped.items(), key=lambda item: item[0]):
        bins = _fixed_bin_summaries(local)
        summary = _summarise_rows(local)
        bad_dominated = _bad_baseline_dominated(local, bins)
        leakage = any(
            bool(row.get("future_target_magnitude_reused"))
            or bool(row.get("target_future_stft_magnitude_accessed"))
            or bool(row.get("future_target_phase_reused"))
            or bool(row.get("target_future_stft_phase_accessed"))
            for row in local
        )
        status = "bad_baseline_rescue_dominated" if bad_dominated else _status_for_summary(summary, bins, leakage)
        summary.update(
            {
                "run_label": key[0],
                "group": key[1],
                "magnitude_mode": key[2],
                "mask_mode": key[3],
                "mechanism": key[4],
                "gain": key[5],
                "status": status,
                "bad_baseline_rescue_dominated": bad_dominated,
                "baseline_correlation_bins": bins,
            }
        )
        out.append(summary)
    return out


def _candidate_summaries(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str, str, float], list[dict[str, Any]]] = {}
    for row in rows:
        key = (
            str(row.get("run_label")),
            str(row.get("magnitude_mode")),
            str(row.get("mask_mode")),
            str(row.get("mechanism")),
            _as_float(row.get("gain")),
        )
        grouped.setdefault(key, []).append(row)

    out: list[dict[str, Any]] = []
    for key, local in sorted(grouped.items(), key=lambda item: item[0]):
        bins = _fixed_bin_summaries(local)
        summary = _summarise_rows(local)
        bad_dominated = _bad_baseline_dominated(local, bins)
        leakage = any(
            bool(row.get("future_target_magnitude_reused"))
            or bool(row.get("target_future_stft_magnitude_accessed"))
            or bool(row.get("future_target_phase_reused"))
            or bool(row.get("target_future_stft_phase_accessed"))
            for row in local
        )
        status = "bad_baseline_rescue_dominated" if bad_dominated else _status_for_summary(summary, bins, leakage)
        summary.update(
            {
                "run_label": key[0],
                "magnitude_mode": key[1],
                "mask_mode": key[2],
                "mechanism": key[3],
                "gain": key[4],
                "status": status,
                "bad_baseline_rescue_dominated": bad_dominated,
                "baseline_correlation_bins": bins,
            }
        )
        out.append(summary)
    out.sort(
        key=lambda row: (
            _as_float(row.get("mean_corr_delta")),
            _as_float(row.get("median_corr_delta")),
            _as_float(row.get("corr_win_fraction")),
            -_as_float(row.get("positive_outlier_share")),
        ),
        reverse=True,
    )
    return out


def _per_case_extremes(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault((str(row.get("run_label")), str(row.get("case_key"))), []).append(row)

    out: list[dict[str, Any]] = []
    for (run_label, case_key), local in sorted(grouped.items(), key=lambda item: item[0]):
        best_corr = max(local, key=lambda row: _as_float(row.get("corr_delta")), default={})
        worst_corr = min(local, key=lambda row: _as_float(row.get("corr_delta")), default={})
        best_mse = min(local, key=lambda row: _as_float(row.get("mse_delta")), default={})
        worst_mse = max(local, key=lambda row: _as_float(row.get("mse_delta")), default={})
        first = local[0] if local else {}
        out.append(
            {
                "run_label": run_label,
                "case_key": case_key,
                "name": first.get("name"),
                "source_wav": first.get("source_wav"),
                "group": first.get("group"),
                "row_count": len(local),
                "best_corr_row": _case_brief(best_corr),
                "worst_corr_row": _case_brief(worst_corr),
                "best_mse_row": _case_brief(best_mse),
                "worst_mse_row": _case_brief(worst_mse),
            }
        )
    return out


def _overall_status(groups: Sequence[dict[str, Any]]) -> str:
    statuses = {str(row.get("status")) for row in groups}
    if not statuses:
        return "missing_cases"
    if "invalid_future_leakage" in statuses:
        return "invalid_future_leakage"
    if "robust_signal" in statuses:
        return "robust_signal"
    if statuses and statuses <= {"bad_baseline_rescue_dominated", "missing_cases"}:
        return "bad_baseline_rescue_dominated"
    if "bad_baseline_rescue_dominated" in statuses and "tradeoff_signal_only" not in statuses:
        return "bad_baseline_rescue_dominated"
    if "tradeoff_signal_only" in statuses:
        return "tradeoff_signal_only"
    if "mse_only_or_corr_tradeoff" in statuses:
        return "mse_only_or_corr_tradeoff"
    if "missing_cases" in statuses:
        return "missing_cases"
    return "no_signal"


def compare_audio_lockbox_results(
    *,
    result_specs: Sequence[tuple[str, Path]],
    out_dir: Path,
    manifest_path: Path | None,
    include_all_rows: bool,
    exclude_roles: Sequence[str] = (),
) -> dict[str, Any]:
    manifest = _load_json(manifest_path) if manifest_path else None
    manifest_index = _case_index_from_manifest(manifest)
    declared_rows = _declared_rows_from_manifest(manifest)

    all_rows: list[dict[str, Any]] = []
    runs: list[dict[str, Any]] = []
    for label, result_path in result_specs:
        payload = _load_json(result_path)
        rows, run_meta = _normalise_result_rows(
            label=label,
            result_path=result_path,
            payload=payload,
            manifest_index=manifest_index,
            declared_rows=declared_rows,
        )
        all_rows.extend(rows)
        runs.append(run_meta)

    candidate_rows = [row for row in all_rows if abs(_as_float(row.get("gain"))) > 1.0e-12]
    if declared_rows and not include_all_rows:
        analysis_rows = [row for row in candidate_rows if row.get("predeclared_row_match")]
        comparison_scope = "manifest_declared_nonbaseline_rows"
    else:
        analysis_rows = candidate_rows
        comparison_scope = "all_nonbaseline_rows"
    excluded_role_set = {str(role) for role in exclude_roles if str(role)}
    excluded_by_role_count = 0
    if excluded_role_set:
        before = len(analysis_rows)
        analysis_rows = [row for row in analysis_rows if str(row.get("role") or "") not in excluded_role_set]
        excluded_by_role_count = before - len(analysis_rows)

    baseline_bins = _fixed_bin_summaries(analysis_rows)
    candidates = _candidate_summaries(analysis_rows)
    groups = _group_summaries(analysis_rows)
    summary = {
        "runtime": "circleworld_proto",
        "schema": "circleworld_audio_lockbox_result_compare_v0",
        "status": _overall_status(groups),
        "comparison_scope": comparison_scope,
        "excluded_roles": sorted(excluded_role_set),
        "excluded_by_role_count": excluded_by_role_count,
        "manifest_path": str(manifest_path) if manifest_path else None,
        "out_dir": str(out_dir),
        "result_count": len(result_specs),
        "run_count": len(runs),
        "case_count": len({row.get("case_key") for row in analysis_rows}),
        "analysis_row_count": len(analysis_rows),
        "source_row_count": len(all_rows),
        "declared_rows": declared_rows,
        "runs": runs,
        "overall_summary": _summarise_rows(analysis_rows),
        "baseline_correlation_bins": baseline_bins,
        "candidate_summaries": candidates,
        "groups": groups,
        "per_case_extremes": _per_case_extremes(analysis_rows),
        "interpretation_guard": [
            "This is a post-run summarizer; it never selects cases by target metrics.",
            "Baseline-correlation bins are computed from the gain-0 baseline for the same case and row key.",
            "Per-case best/worst rows rank already-run rows only and are diagnostic, not a selection rule.",
            "A robust_signal label requires positive mean, median, win fraction, MSE, leave-one-out, and outlier guards.",
            "bad_baseline_rescue_dominated means gains mainly helped cases where the gain-0 baseline was already bad.",
        ],
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / OUTPUT_JSON).write_text(json.dumps(_json_safe(summary), indent=2, allow_nan=False), encoding="utf-8")
    _write_markdown(summary, out_dir / OUTPUT_MD)
    return summary


def _write_markdown(summary: dict[str, Any], out_path: Path) -> None:
    lines = [
        "# Audio Lockbox Result Compare",
        "",
        "This post-run report summarizes existing RAFA audio lockbox results without selecting cases by target metrics.",
        "",
        f"- status: `{summary.get('status')}`",
        f"- comparison scope: `{summary.get('comparison_scope')}`",
        f"- excluded roles: `{', '.join(summary.get('excluded_roles') or [])}`",
        f"- excluded by role: `{summary.get('excluded_by_role_count')}`",
        f"- manifest: `{summary.get('manifest_path')}`",
        f"- results: `{summary.get('result_count')}`",
        f"- cases: `{summary.get('case_count')}`",
        f"- analysis rows: `{summary.get('analysis_row_count')}`",
        "",
        "## Candidate Summaries",
        "",
        "| run | mode | mask | mechanism | gain | cases | status | mean corr d | median corr d | corr wins | mean MSE d | outlier pressure | top positive | top d |",
        "|---|---|---|---|---:|---:|---|---:|---:|---:|---:|---:|---|---:|",
    ]
    for row in summary.get("candidate_summaries", []) or []:
        top_pos = row.get("top_positive_case") if isinstance(row.get("top_positive_case"), dict) else {}
        pressure = row.get("outlier_pressure") if isinstance(row.get("outlier_pressure"), dict) else {}
        lines.append(
            "| {run} | {mode} | {mask} | {mechanism} | {gain} | {cases} | {status} | {mean} | {median} | {wins} | {mse} | {pressure} | {pos} | {posd} |".format(
                run=row.get("run_label"),
                mode=row.get("magnitude_mode"),
                mask=row.get("mask_mode"),
                mechanism=row.get("mechanism"),
                gain=_fmt(row.get("gain")),
                cases=_fmt(row.get("case_count")),
                status=row.get("status"),
                mean=_fmt(row.get("mean_corr_delta")),
                median=_fmt(row.get("median_corr_delta")),
                wins=_fmt(row.get("corr_win_fraction")),
                mse=_fmt(row.get("mean_mse_delta")),
                pressure=_fmt(pressure.get("positive_outlier_share")),
                pos=top_pos.get("name"),
                posd=_fmt(top_pos.get("corr_delta")),
            )
        )
    lines.extend(
        [
            "",
            "## Group Summaries",
            "",
            "| run | group | mode | mask | mechanism | gain | cases | status | mean base corr | mean corr d | median corr d | corr wins | mean MSE d | MSE wins | outlier pressure | top positive | top d | top negative | neg d |",
            "|---|---|---|---|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---|---:|---|---:|",
        ]
    )
    for row in summary.get("groups", []) or []:
        top_pos = row.get("top_positive_case") if isinstance(row.get("top_positive_case"), dict) else {}
        top_neg = row.get("top_negative_case") if isinstance(row.get("top_negative_case"), dict) else {}
        pressure = row.get("outlier_pressure") if isinstance(row.get("outlier_pressure"), dict) else {}
        lines.append(
            "| {run} | {group} | {mode} | {mask} | {mechanism} | {gain} | {cases} | {status} | {base} | {mean} | {median} | {wins} | {mse} | {mwins} | {pressure} | {pos} | {posd} | {neg} | {negd} |".format(
                run=row.get("run_label"),
                group=row.get("group"),
                mode=row.get("magnitude_mode"),
                mask=row.get("mask_mode"),
                mechanism=row.get("mechanism"),
                gain=_fmt(row.get("gain")),
                cases=_fmt(row.get("case_count")),
                status=row.get("status"),
                base=_fmt(row.get("mean_baseline_target_corr")),
                mean=_fmt(row.get("mean_corr_delta")),
                median=_fmt(row.get("median_corr_delta")),
                wins=_fmt(row.get("corr_win_fraction")),
                mse=_fmt(row.get("mean_mse_delta")),
                mwins=_fmt(row.get("mse_win_fraction")),
                pressure=_fmt(pressure.get("positive_outlier_share")),
                pos=top_pos.get("name"),
                posd=_fmt(top_pos.get("corr_delta")),
                neg=top_neg.get("name"),
                negd=_fmt(top_neg.get("corr_delta")),
            )
        )

    lines.extend(
        [
            "",
            "## Baseline-Correlation Bins",
            "",
            "| bin | range | rows | mean base corr | mean corr d | median corr d | corr wins | mean MSE d | outlier pressure | top positive | top d |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|---|---:|",
        ]
    )
    for row in summary.get("baseline_correlation_bins", []) or []:
        top_pos = row.get("top_positive_case") if isinstance(row.get("top_positive_case"), dict) else {}
        pressure = row.get("outlier_pressure") if isinstance(row.get("outlier_pressure"), dict) else {}
        lines.append(
            "| {bin} | [{lo}, {hi}) | {cases} | {base} | {mean} | {median} | {wins} | {mse} | {pressure} | {pos} | {posd} |".format(
                bin=row.get("bin"),
                lo=_fmt(row.get("lo")),
                hi=_fmt(row.get("hi")),
                cases=_fmt(row.get("case_count")),
                base=_fmt(row.get("mean_baseline_target_corr")),
                mean=_fmt(row.get("mean_corr_delta")),
                median=_fmt(row.get("median_corr_delta")),
                wins=_fmt(row.get("corr_win_fraction")),
                mse=_fmt(row.get("mean_mse_delta")),
                pressure=_fmt(pressure.get("positive_outlier_share")),
                pos=top_pos.get("name"),
                posd=_fmt(top_pos.get("corr_delta")),
            )
        )

    lines.extend(
        [
            "",
            "## Per-Case Extremes",
            "",
            "| run | group | case | rows | best corr row | best corr d | worst corr row | worst corr d | best MSE row | best MSE d |",
            "|---|---|---|---:|---|---:|---|---:|---|---:|",
        ]
    )
    for row in summary.get("per_case_extremes", []) or []:
        best = row.get("best_corr_row") if isinstance(row.get("best_corr_row"), dict) else {}
        worst = row.get("worst_corr_row") if isinstance(row.get("worst_corr_row"), dict) else {}
        best_mse = row.get("best_mse_row") if isinstance(row.get("best_mse_row"), dict) else {}
        lines.append(
            "| {run} | {group} | {case} | {rows} | {best_row} | {best_d} | {worst_row} | {worst_d} | {mse_row} | {mse_d} |".format(
                run=row.get("run_label"),
                group=row.get("group"),
                case=row.get("name") or row.get("case_key"),
                rows=_fmt(row.get("row_count")),
                best_row=_brief_row_name(best),
                best_d=_fmt(best.get("corr_delta")),
                worst_row=_brief_row_name(worst),
                worst_d=_fmt(worst.get("corr_delta")),
                mse_row=_brief_row_name(best_mse),
                mse_d=_fmt(best_mse.get("mse_delta")),
            )
        )

    lines.extend(["", "## Interpretation Guard", ""])
    for item in summary.get("interpretation_guard", []) or []:
        lines.append(f"- {item}")
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _brief_row_name(row: dict[str, Any]) -> str:
    if not row:
        return "NA"
    return "{mode}/{mask}/{mechanism}/g{gain}".format(
        mode=row.get("magnitude_mode"),
        mask=row.get("mask_mode"),
        mechanism=row.get("mechanism"),
        gain=_fmt(row.get("gain")),
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Post-run comparison for predeclared RAFA audio lockbox result JSONs."
    )
    parser.add_argument(
        "--result",
        action="append",
        required=True,
        help="Result JSON path, optionally label=path. May be repeated.",
    )
    parser.add_argument("--manifest", default=None, help="Optional lockbox manifest JSON.")
    parser.add_argument("--out-dir", required=True, help="Directory for JSON and Markdown outputs.")
    parser.add_argument(
        "--all-rows",
        action="store_true",
        help="Compare all non-baseline rows even when the manifest declares frozen rows.",
    )
    parser.add_argument(
        "--exclude-role",
        action="append",
        default=None,
        help="Exclude manifest rows with this role from analysis. May be repeated.",
    )
    args = parser.parse_args()

    result_specs = [_parse_labeled_path(raw) for raw in args.result]
    summary = compare_audio_lockbox_results(
        result_specs=result_specs,
        out_dir=Path(args.out_dir),
        manifest_path=Path(args.manifest) if args.manifest else None,
        include_all_rows=bool(args.all_rows),
        exclude_roles=args.exclude_role or (),
    )
    print(
        json.dumps(
            {
                "json": str(Path(args.out_dir) / OUTPUT_JSON),
                "markdown": str(Path(args.out_dir) / OUTPUT_MD),
                "status": summary.get("status"),
                "comparison_scope": summary.get("comparison_scope"),
                "excluded_roles": summary.get("excluded_roles"),
                "excluded_by_role_count": summary.get("excluded_by_role_count"),
                "case_count": summary.get("case_count"),
                "analysis_row_count": summary.get("analysis_row_count"),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
