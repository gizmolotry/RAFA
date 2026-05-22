from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

from score_phase_native_audio_reentry_guard import _as_float, _key, _mean
from score_phase_native_audio_target_replay_oracle import (
    OUTPUT_JSON as TARGET_ORACLE_JSON,
    _aggregate_selected,
    _best_global,
    _best_per_group,
    _collect_rows,
    _row_objective,
)


OUTPUT_JSON = "phase_native_audio_prefix_router_scout.json"
OUTPUT_MD = "PHASE_NATIVE_AUDIO_PREFIX_ROUTER_SCOUT.md"

NO_FUTURE_ROW_FLAG_KEYS = (
    "future_target_audio_used",
    "future_target_magnitude_used",
    "future_target_magnitude_reused",
    "future_target_phase_used",
    "future_target_phase_reused",
    "future_target_metric_used",
    "target_future_audio_used",
    "target_future_magnitude_used",
    "target_future_stft_magnitude_accessed",
    "target_future_phase_used",
    "target_future_stft_phase_accessed",
    "target_future_metric_used",
)

REENTRY_MECHANISM_FLAG_KEYS = (
    "mean_phase_velocity_coherence",
    "mean_prefix_magnitude_stability",
    "mean_energy_weight",
    "mean_abs_raw_delta",
)


def _json_load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _case_group(case_name: str) -> str:
    return str(case_name).split("__", 1)[0]


def _key_to_dict(key: tuple[str, str, str, float]) -> dict[str, Any]:
    return {
        "magnitude_mode": key[0],
        "mask_mode": key[1],
        "mechanism": key[2],
        "gain": key[3],
    }


def _dict_to_key(row: dict[str, Any]) -> tuple[str, str, str, float]:
    return (
        str(row["magnitude_mode"]),
        str(row["mask_mode"]),
        str(row["mechanism"]),
        float(row["gain"]),
    )


def _row_matches_key(row: dict[str, Any], key: tuple[str, str, str, float]) -> bool:
    return _key(row) == key


def _best_case_rows(rows: Sequence[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    by_case: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if abs(_as_float(row.get("gain"))) <= 1.0e-12:
            continue
        by_case[str(row["case_name"])].append(row)
    return {case: max(case_rows, key=_row_objective) for case, case_rows in by_case.items() if case_rows}


def _case_rows_by_case(rows: Sequence[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        out[str(row["case_name"])].append(row)
    return out


def _select_row_for_key(
    case_name: str,
    key: tuple[str, str, str, float],
    rows_by_case: dict[str, list[dict[str, Any]]],
) -> tuple[dict[str, Any] | None, bool]:
    candidates = [row for row in rows_by_case.get(case_name, []) if _row_matches_key(row, key)]
    if candidates:
        return max(candidates, key=_row_objective), False
    fallback = rows_by_case.get(case_name, [])
    if not fallback:
        return None, True
    return max(fallback, key=_row_objective), True


def _finite_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return float(value)
    return None


def _row_has_future_leakage(row: dict[str, Any]) -> bool:
    for key in NO_FUTURE_ROW_FLAG_KEYS:
        if bool(row.get(key, False)):
            return True
    for section_name in ("mechanism_flags", "mask_flags", "phase_seed_flags", "magnitude_flags"):
        section = row.get(section_name, {})
        if not isinstance(section, dict):
            continue
        for key in NO_FUTURE_ROW_FLAG_KEYS:
            if bool(section.get(key, False)):
                return True
    return False


def _row_section_values(case: dict[str, Any], section_name: str, key: str) -> list[float]:
    values: list[float] = []
    for row in case.get("rows", []):
        if not isinstance(row, dict) or _row_has_future_leakage(row):
            continue
        section = row.get(section_name, {})
        if not isinstance(section, dict):
            continue
        value = _finite_float(section.get(key))
        if value is not None:
            values.append(value)
    return values


def _add_reentry_query_features(out: dict[str, float], case: dict[str, Any]) -> None:
    """Expose prefix/Circleworld route-query signals without target-future metrics."""
    observed: dict[str, float] = {}
    for key in REENTRY_MECHANISM_FLAG_KEYS:
        values = _row_section_values(case, "mechanism_flags", key)
        if values:
            observed[key] = _mean(values)

    phase_coherence = observed.get("mean_phase_velocity_coherence")
    if phase_coherence is not None:
        clipped = max(0.0, min(1.0, phase_coherence))
        out["circleworld_meta.reentry_query_phase_velocity_coherence_mean"] = clipped
        out["circleworld_meta.reentry_query_phase_velocity_incoherence_mean"] = 1.0 - clipped

    magnitude_stability = observed.get("mean_prefix_magnitude_stability")
    if magnitude_stability is not None:
        clipped = max(0.0, min(1.0, magnitude_stability))
        out["circleworld_meta.reentry_query_prefix_magnitude_stability_mean"] = clipped
        out["circleworld_meta.reentry_query_prefix_magnitude_instability_mean"] = 1.0 - clipped

    energy_weight = observed.get("mean_energy_weight")
    if energy_weight is not None:
        out["circleworld_meta.reentry_query_prefix_energy_weight_mean"] = energy_weight

    raw_delta = observed.get("mean_abs_raw_delta")
    if raw_delta is not None:
        out["circleworld_meta.reentry_query_circleworld_raw_delta_abs_mean"] = raw_delta


def _case_features(delta_jsons: Sequence[Path]) -> dict[str, dict[str, float]]:
    features: dict[str, dict[str, float]] = {}
    for delta_json in delta_jsons:
        payload = _json_load(delta_json)
        for case in payload.get("cases", []):
            name = str(case.get("name", ""))
            if not name or name in features:
                continue
            meta = case.get("circleworld_meta", {})
            row: dict[str, float] = {}
            for key, value in sorted(meta.items()):
                numeric = _finite_float(value)
                if numeric is not None:
                    row[f"circleworld_meta.{key}"] = numeric
            _add_reentry_query_features(row, case)
            for key in ("prefix_stft_frames", "future_stft_frames", "prefix_samples", "future_samples"):
                numeric = _finite_float(case.get(key))
                if numeric is not None:
                    row[f"case.{key}"] = numeric
            features[name] = row
    return features


def _feature_matrix(features: dict[str, dict[str, float]]) -> tuple[list[str], dict[str, list[float]]]:
    keys = sorted({key for row in features.values() for key in row})
    matrix = {case: [features[case].get(key, 0.0) for key in keys] for case in sorted(features)}
    return keys, matrix


def _scaled_distance(
    a: Sequence[float],
    b: Sequence[float],
    means: Sequence[float],
    stds: Sequence[float],
) -> float:
    total = 0.0
    for av, bv, mean, std in zip(a, b, means, stds):
        denom = std if std > 1.0e-9 else 1.0
        da = (av - mean) / denom
        db = (bv - mean) / denom
        total += (da - db) * (da - db)
    return math.sqrt(total)


def _fit_scaler(vectors: Sequence[Sequence[float]]) -> tuple[list[float], list[float]]:
    if not vectors:
        return [], []
    dims = len(vectors[0])
    means: list[float] = []
    stds: list[float] = []
    for idx in range(dims):
        vals = [float(vec[idx]) for vec in vectors]
        mean = sum(vals) / float(len(vals))
        var = sum((val - mean) ** 2 for val in vals) / float(len(vals))
        means.append(mean)
        stds.append(math.sqrt(var))
    return means, stds


def _centroid(vectors: Sequence[Sequence[float]]) -> list[float]:
    if not vectors:
        return []
    dims = len(vectors[0])
    return [sum(float(vec[idx]) for vec in vectors) / float(len(vectors)) for idx in range(dims)]


def _nearest_case_route(
    cases: Sequence[str],
    matrix: dict[str, list[float]],
    case_best_key: dict[str, tuple[str, str, str, float]],
) -> tuple[dict[str, tuple[str, str, str, float]], dict[str, Any]]:
    predictions: dict[str, tuple[str, str, str, float]] = {}
    nearest: dict[str, str] = {}
    for case in cases:
        train_cases = [other for other in cases if other != case and other in case_best_key]
        train_vectors = [matrix[other] for other in train_cases]
        means, stds = _fit_scaler(train_vectors)
        best_other = min(
            train_cases,
            key=lambda other: _scaled_distance(matrix[case], matrix[other], means, stds),
        )
        predictions[case] = case_best_key[best_other]
        nearest[case] = best_other
    return predictions, {"nearest_cases": nearest}


def _leave_one_case_family_centroid_route(
    cases: Sequence[str],
    matrix: dict[str, list[float]],
    family_key: dict[str, tuple[str, str, str, float]],
) -> tuple[dict[str, tuple[str, str, str, float]], dict[str, Any]]:
    predictions: dict[str, tuple[str, str, str, float]] = {}
    predicted_groups: dict[str, str] = {}
    for case in cases:
        train_cases = [other for other in cases if other != case]
        train_vectors = [matrix[other] for other in train_cases]
        means, stds = _fit_scaler(train_vectors)
        by_group: dict[str, list[list[float]]] = defaultdict(list)
        for other in train_cases:
            group = _case_group(other)
            if group in family_key:
                by_group[group].append(matrix[other])
        centroids = {group: _centroid(vecs) for group, vecs in by_group.items() if vecs}
        best_group = min(
            centroids,
            key=lambda group: _scaled_distance(matrix[case], centroids[group], means, stds),
        )
        predictions[case] = family_key[best_group]
        predicted_groups[case] = best_group
    return predictions, {"predicted_groups": predicted_groups}


def _leave_one_group_centroid_route(
    cases: Sequence[str],
    matrix: dict[str, list[float]],
    family_key: dict[str, tuple[str, str, str, float]],
) -> tuple[dict[str, tuple[str, str, str, float]], dict[str, Any]]:
    predictions: dict[str, tuple[str, str, str, float]] = {}
    predicted_groups: dict[str, str] = {}
    groups = sorted({_case_group(case) for case in cases})
    for held_group in groups:
        test_cases = [case for case in cases if _case_group(case) == held_group]
        train_cases = [case for case in cases if _case_group(case) != held_group]
        train_vectors = [matrix[case] for case in train_cases]
        means, stds = _fit_scaler(train_vectors)
        by_group: dict[str, list[list[float]]] = defaultdict(list)
        for case in train_cases:
            group = _case_group(case)
            if group in family_key:
                by_group[group].append(matrix[case])
        centroids = {group: _centroid(vecs) for group, vecs in by_group.items() if vecs}
        if not centroids:
            continue
        for case in test_cases:
            best_group = min(
                centroids,
                key=lambda group: _scaled_distance(matrix[case], centroids[group], means, stds),
            )
            predictions[case] = family_key[best_group]
            predicted_groups[case] = best_group
    return predictions, {"predicted_groups": predicted_groups}


def _evaluate_predictions(
    *,
    label: str,
    cases: Sequence[str],
    predictions: dict[str, tuple[str, str, str, float]],
    rows_by_case: dict[str, list[dict[str, Any]]],
    family_key: dict[str, tuple[str, str, str, float]],
    case_best_key: dict[str, tuple[str, str, str, float]],
    predicted_groups: dict[str, str] | None = None,
) -> dict[str, Any]:
    selected: list[dict[str, Any]] = []
    missing_cases: list[str] = []
    fallback_count = 0
    route_family_matches = 0
    route_case_matches = 0
    group_matches = 0
    comparable_group_count = 0
    for case in cases:
        key = predictions.get(case)
        if key is None:
            missing_cases.append(case)
            continue
        row, fallback = _select_row_for_key(case, key, rows_by_case)
        if row is None:
            missing_cases.append(case)
            continue
        selected.append(row)
        fallback_count += int(fallback)
        if family_key.get(_case_group(case)) == key:
            route_family_matches += 1
        if case_best_key.get(case) == key:
            route_case_matches += 1
        if predicted_groups is not None and case in predicted_groups:
            comparable_group_count += 1
            group_matches += int(predicted_groups[case] == _case_group(case))
    summary = _aggregate_selected(selected, label)
    summary.update(
        {
            "missing_case_count": len(missing_cases),
            "fallback_case_count": fallback_count,
            "route_family_key_match_fraction": route_family_matches / float(len(cases)) if cases else 0.0,
            "route_case_key_match_fraction": route_case_matches / float(len(cases)) if cases else 0.0,
            "predicted_group_match_fraction": (
                group_matches / float(comparable_group_count) if comparable_group_count else None
            ),
            "missing_cases": missing_cases[:20],
        }
    )
    return summary


def _markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# Phase-Native Audio Prefix Router Scout",
        "",
        "This scout evaluates whether target-normalized replay-oracle routes can be",
        "approximated from no-future prefix/Circleworld metadata.",
        "",
        "## Summary",
        "",
        f"- Status: `{summary['status']}`",
        f"- Future access clean: `{summary['future_access_clean']}`",
        f"- Feature count: `{summary['feature_count']}`",
        f"- Case count: `{summary['case_count']}`",
        "",
        "## Policies",
        "",
        "| policy | corr-copy | MSE-copy | loop-copy | harm-delta | corr-gain0 | family-key match | case-key match | group match | strict |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for label, row in summary["policies"].items():
        group_match = row.get("predicted_group_match_fraction")
        group_text = "" if group_match is None else f"{_as_float(group_match):.6f}"
        lines.append(
            "| `{label}` | {corr:.9f} | {mse:.9f} | {loop:.9f} | {harm:.9f} | {cgain:.9f} | {fmatch:.6f} | {cmatch:.6f} | {gmatch} | `{strict}` |".format(
                label=label,
                corr=_as_float(row.get("mean_corr_delta_vs_copy_last")),
                mse=_as_float(row.get("mean_mse_delta_vs_copy_last")),
                loop=_as_float(row.get("mean_loop_delta_vs_copy_last")),
                harm=_as_float(row.get("mean_harmful_replay_excess_delta_vs_copy_last")),
                cgain=_as_float(row.get("mean_corr_delta_vs_gain0")),
                fmatch=_as_float(row.get("route_family_key_match_fraction")),
                cmatch=_as_float(row.get("route_case_key_match_fraction")),
                gmatch=group_text,
                strict=bool(row.get("strict_target_replay_pass")),
            )
        )
    lines.extend(["", "## Interpretation", "", summary["interpretation"]])
    return "\n".join(lines) + "\n"


def score_prefix_router_scout(
    suite_json: Path,
    delta_jsons: Sequence[Path],
    out_dir: Path,
    *,
    margin: float,
) -> dict[str, Any]:
    meta, rows = _collect_rows(suite_json, delta_jsons, margin)
    rows_by_case = _case_rows_by_case(rows)
    cases = sorted(rows_by_case)
    features = _case_features(delta_jsons)
    feature_keys, matrix = _feature_matrix({case: features[case] for case in cases if case in features})
    cases = [case for case in cases if case in matrix]

    global_summary, _ = _best_global(rows)
    _, family_choices, _ = _best_per_group(rows)
    best_case_rows = _best_case_rows(rows)
    family_key = {group: _dict_to_key(choice["key"]) for group, choice in family_choices.items()}
    case_best_key = {case: _key(row) for case, row in best_case_rows.items()}

    policies: dict[str, dict[str, Any]] = {}
    diagnostics: dict[str, Any] = {}

    if global_summary and "key" in global_summary:
        global_key = _dict_to_key(global_summary["key"])
        global_pred = {case: global_key for case in cases}
        policies["global_route"] = _evaluate_predictions(
            label="global_route",
            cases=cases,
            predictions=global_pred,
            rows_by_case=rows_by_case,
            family_key=family_key,
            case_best_key=case_best_key,
        )

    family_pred = {case: family_key[_case_group(case)] for case in cases if _case_group(case) in family_key}
    policies["known_family_table"] = _evaluate_predictions(
        label="known_family_table",
        cases=cases,
        predictions=family_pred,
        rows_by_case=rows_by_case,
        family_key=family_key,
        case_best_key=case_best_key,
    )

    nn_pred, nn_diag = _nearest_case_route(cases, matrix, case_best_key)
    diagnostics["nearest_case"] = nn_diag
    policies["leave_one_case_nearest_case"] = _evaluate_predictions(
        label="leave_one_case_nearest_case",
        cases=cases,
        predictions=nn_pred,
        rows_by_case=rows_by_case,
        family_key=family_key,
        case_best_key=case_best_key,
    )

    loc_pred, loc_diag = _leave_one_case_family_centroid_route(cases, matrix, family_key)
    diagnostics["leave_one_case_family_centroid"] = loc_diag
    policies["leave_one_case_family_centroid"] = _evaluate_predictions(
        label="leave_one_case_family_centroid",
        cases=cases,
        predictions=loc_pred,
        rows_by_case=rows_by_case,
        family_key=family_key,
        case_best_key=case_best_key,
        predicted_groups=loc_diag["predicted_groups"],
    )

    log_pred, log_diag = _leave_one_group_centroid_route(cases, matrix, family_key)
    diagnostics["leave_one_group_family_centroid"] = log_diag
    policies["leave_one_group_family_centroid"] = _evaluate_predictions(
        label="leave_one_group_family_centroid",
        cases=cases,
        predictions=log_pred,
        rows_by_case=rows_by_case,
        family_key=family_key,
        case_best_key=case_best_key,
        predicted_groups=log_diag["predicted_groups"],
    )

    strict_prefix = [
        label
        for label in ("leave_one_case_nearest_case", "leave_one_case_family_centroid", "leave_one_group_family_centroid")
        if bool(policies[label].get("strict_target_replay_pass"))
    ]
    if strict_prefix:
        status = "prefix_router_target_replay_pass"
        interpretation = (
            "A no-future prefix-feature router passes the target-normalized replay guard in this scout. "
            "It should be frozen and rerun on a fresh lockbox before promotion."
        )
    elif bool(policies["known_family_table"].get("strict_target_replay_pass")):
        status = "known_family_route_table_pass"
        interpretation = (
            "The predeclared-family route table passes, but prefix-feature generalization does not. "
            "This supports a trainable router objective while withholding runtime promotion."
        )
    else:
        status = "prefix_router_not_yet"
        interpretation = (
            "The target replay oracle is not yet recoverable from this simple prefix-feature router. "
            "Next step is a learned selector with richer prefix descriptors or more training cases."
        )

    summary = {
        "schema": "phase_native_audio_prefix_router_scout_v1",
        "status": status,
        "suite_json": str(suite_json),
        "delta_jsons": [str(path) for path in delta_jsons],
        "target_oracle_json_name": TARGET_ORACLE_JSON,
        "future_access_clean": bool(meta["future_access_clean"]),
        "target_margin": float(margin),
        "case_count": len(cases),
        "feature_count": len(feature_keys),
        "feature_keys": feature_keys,
        "policies": policies,
        "family_route_keys": {group: _key_to_dict(key) for group, key in sorted(family_key.items())},
        "diagnostics": diagnostics,
        "interpretation": interpretation,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / OUTPUT_JSON).write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (out_dir / OUTPUT_MD).write_text(_markdown(summary), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Scout prefix-only routing for target replay mechanisms.")
    parser.add_argument("--suite-json", required=True, type=Path)
    parser.add_argument("--delta-json", required=True, action="append", type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--target-margin", type=float, default=0.01)
    args = parser.parse_args()
    summary = score_prefix_router_scout(args.suite_json, args.delta_json, args.out_dir, margin=args.target_margin)
    print(
        json.dumps(
            {
                "status": summary["status"],
                "json": str(Path(args.out_dir) / OUTPUT_JSON),
                "markdown": str(Path(args.out_dir) / OUTPUT_MD),
                "strict_policies": [
                    label for label, row in summary["policies"].items() if row.get("strict_target_replay_pass")
                ],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
