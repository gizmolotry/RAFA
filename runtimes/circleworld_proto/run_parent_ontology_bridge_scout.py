from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SCHEMA = "rafa_parent_ontology_bridge_scout_v0"
TARGET_EPSILON = 1.0e-9
DEFAULT_L2 = 1.0e-3

FEATURES = [
    "readout_sibling_response",
    "readout_only_baseline",
    "readout_minus_baseline",
    "coarse_env_corr",
    "fine_q_profile_corr",
    "fine_texture_distance",
    "branch_identity_qualified_carry",
    "branch_identity_support",
    "branch_identity_coherence",
    "branch_identity_budget_retained",
    "branch_identity_parent_divergence",
    "branch_identity_sibling_divergence",
    "child_predictive_positive_residual_reduction",
    "child_predictive_boundary_match",
    "child_predictive_ontology_binding_mass",
    "parent_ontology_charge",
    "parent_ontology_mode1_occupancy_after",
    "parent_ontology_gate_floor",
    "parent_ontology_charge_gain",
    "parent_ontology_mode1_logit_boost",
    "parent_ontology_support_boost",
    "mode1_replace_ratio",
    "mode1_replace_gate_floor",
    "mode1_replace_ratio_scale",
    "mode1_replace_support_gate_mean",
    "mode1_replace_support_gate_max",
    "child_volume_active_count",
    "child_volume_effective_count",
    "child_volume_top_score_share",
    "world_jump_penalty",
    "direct_readout_dependency",
    "final_direct_mix_shortcut_score",
]

NONLEAKY_FEATURES = [
    # Available before judging the post-hoc readout crossing. These are still
    # assay-derived in this scout, but avoid direct label/readout leakage.
    "branch_identity_qualified_carry",
    "branch_identity_support",
    "branch_identity_coherence",
    "branch_identity_budget_retained",
    "branch_identity_parent_divergence",
    "branch_identity_sibling_divergence",
    "child_predictive_boundary_match",
    "parent_ontology_gate_floor",
    "parent_ontology_charge_gain",
    "parent_ontology_mode1_logit_boost",
    "parent_ontology_support_boost",
    "mode1_replace_ratio",
    "mode1_replace_gate_floor",
    "mode1_replace_ratio_scale",
    "child_volume_active_count",
    "child_volume_effective_count",
    "child_volume_top_score_share",
]


@dataclass(frozen=True)
class ScoutRow:
    source_path: str
    source_kind: str
    case: str
    variant: str
    branch: str
    family: str
    label: str
    features: dict[str, float]
    target_nested_sibling: int
    target_readout_crossing: int
    target_bridge: int
    uses_oracle: bool


def _utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return float(default)
        if isinstance(value, bool):
            return 1.0 if value else 0.0
        result = float(value)
        if math.isfinite(result):
            return result
        return float(default)
    except (TypeError, ValueError):
        return float(default)


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _safe_name(value: Any) -> str:
    return str(value or "").strip()


def _read_json(path: Path) -> Any | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, UnicodeDecodeError):
        return None


def _iter_json_files(input_dirs: list[Path]) -> list[Path]:
    paths: list[Path] = []
    seen: set[str] = set()
    for root in input_dirs:
        if root.is_file() and root.suffix.lower() == ".json":
            candidates = [root]
        elif root.is_dir():
            candidates = sorted(root.rglob("*.json"))
        else:
            candidates = []
        for path in candidates:
            key = str(path.resolve())
            if key not in seen:
                seen.add(key)
                paths.append(path)
    return paths


def _nested_get(mapping: dict[str, Any], *keys: str) -> float:
    for key in keys:
        if key in mapping:
            return _as_float(mapping.get(key))
    return 0.0


def _extract_branch_features(branch_row: dict[str, Any], case_row: dict[str, Any], root: dict[str, Any]) -> dict[str, float]:
    identity = _as_dict(branch_row.get("branch_identity"))
    predictive = _as_dict(branch_row.get("child_predictive_assimilation"))
    replace = _as_dict(branch_row.get("mode1_replace_metrics"))
    partition = _as_dict(branch_row.get("assay_child_partition"))
    nested_response = _as_dict(case_row.get("case_nested_response"))
    root_assay = _as_dict(root.get("assay_killswitch"))

    readout = _as_float(branch_row.get("readout_sibling_response"))
    baseline = _as_float(branch_row.get("readout_only_baseline"))
    if baseline == 0.0:
        baseline = _as_float(nested_response.get("readout_only_baseline"))

    return {
        "readout_sibling_response": readout,
        "readout_only_baseline": baseline,
        "readout_minus_baseline": readout - baseline,
        "coarse_env_corr": _as_float(branch_row.get("coarse_env_corr")),
        "fine_q_profile_corr": _as_float(branch_row.get("fine_q_profile_corr")),
        "fine_texture_distance": _as_float(branch_row.get("fine_texture_distance")),
        "branch_identity_qualified_carry": _nested_get(
            identity,
            "same_child_qualified_carry_fraction",
            "same_child_carry_fraction",
        ),
        "branch_identity_support": _nested_get(identity, "same_child_support_mean", "same_child_min_support"),
        "branch_identity_coherence": _nested_get(identity, "same_child_coherence_mean", "same_child_min_coherence"),
        "branch_identity_budget_retained": _nested_get(
            identity,
            "same_child_qualified_budget_retained",
            "same_child_budget_retained",
        ),
        "branch_identity_parent_divergence": _as_float(identity.get("same_child_parent_divergence")),
        "branch_identity_sibling_divergence": _as_float(identity.get("same_child_sibling_divergence")),
        "child_predictive_positive_residual_reduction": _nested_get(
            predictive,
            "positive_residual_reduction",
            "residual_reduction",
        ),
        "child_predictive_boundary_match": _as_float(predictive.get("boundary_match")),
        "child_predictive_ontology_binding_mass": _as_float(predictive.get("ontology_binding_mass")),
        "parent_ontology_charge": _nested_get(
            replace,
            "parent_ontology_charge_mean",
            "mode1_replace_parent_ontology_charge",
        )
        or _as_float(predictive.get("parent_ontology_charge_mean")),
        "parent_ontology_mode1_occupancy_after": _nested_get(
            replace,
            "parent_ontology_mode1_occupancy_after",
            "mode1_replace_parent_ontology_mode1_occupancy_after",
        )
        or _as_float(predictive.get("parent_ontology_mode1_occupancy_after")),
        "parent_ontology_gate_floor": _as_float(root_assay.get("parent_ontology_gate_floor")),
        "parent_ontology_charge_gain": _as_float(root_assay.get("parent_ontology_charge_gain")),
        "parent_ontology_mode1_logit_boost": _as_float(root_assay.get("parent_ontology_mode1_logit_boost")),
        "parent_ontology_support_boost": _as_float(root_assay.get("parent_ontology_support_boost")),
        "mode1_replace_ratio": _as_float(replace.get("replace_ratio")),
        "mode1_replace_gate_floor": _as_float(replace.get("gate_floor")),
        "mode1_replace_ratio_scale": _as_float(replace.get("ratio_scale")) or _as_float(root_assay.get("mode_replace_ratio_scale")),
        "mode1_replace_support_gate_mean": _as_float(replace.get("support_gate_mean")),
        "mode1_replace_support_gate_max": _as_float(replace.get("support_gate_max")),
        "child_volume_active_count": _as_float(partition.get("after_active_count")),
        "child_volume_effective_count": _as_float(partition.get("after_effective_count")),
        "child_volume_top_score_share": _as_float(partition.get("after_top_score_share")),
        "world_jump_penalty": _as_float(branch_row.get("world_jump_penalty")),
        "direct_readout_dependency": _as_float(nested_response.get("direct_readout_dependency")),
        "final_direct_mix_shortcut_score": _as_float(nested_response.get("final_direct_mix_shortcut_score")),
    }


def _extract_flat_row_features(row: dict[str, Any], root: dict[str, Any]) -> dict[str, float]:
    metrics = _as_dict(row.get("metrics"))
    if not metrics:
        metrics = row
    assay = _as_dict(row.get("assay_killswitch")) or _as_dict(root.get("assay_killswitch"))
    readout = _nested_get(metrics, "readout_sibling_response", "mean_readout_sibling_response")
    baseline = _as_float(metrics.get("readout_only_baseline"))
    return {
        "readout_sibling_response": readout,
        "readout_only_baseline": baseline,
        "readout_minus_baseline": readout - baseline,
        "coarse_env_corr": _nested_get(metrics, "coarse_env_corr", "mean_coarse_env_corr"),
        "fine_q_profile_corr": _nested_get(metrics, "fine_q_profile_corr", "mean_fine_q_profile_corr"),
        "fine_texture_distance": _as_float(metrics.get("fine_texture_distance")),
        "branch_identity_qualified_carry": _nested_get(
            metrics,
            "same_child_qualified_carry_fraction",
            "same_child_carry_fraction",
            "mean_branch_identity_qualified_carry",
            "mean_branch_identity_carry",
        ),
        "branch_identity_support": _nested_get(metrics, "mean_branch_identity_support", "best_identity_support_mean"),
        "branch_identity_coherence": _nested_get(metrics, "mean_branch_identity_coherence", "best_identity_coherence_mean"),
        "branch_identity_budget_retained": _nested_get(
            metrics,
            "mean_branch_identity_budget_retained",
            "max_branch_identity_budget_retained",
        ),
        "branch_identity_parent_divergence": _nested_get(
            metrics,
            "same_child_parent_divergence",
            "mean_branch_identity_parent_div",
            "mean_max_branch_identity_parent_div",
        ),
        "branch_identity_sibling_divergence": _nested_get(
            metrics,
            "same_child_sibling_divergence",
            "mean_branch_identity_sibling_div",
            "mean_max_branch_identity_sibling_div",
        ),
        "child_predictive_positive_residual_reduction": _nested_get(
            metrics,
            "mean_child_predictive_positive_residual_reduction",
            "child_predictive_positive_residual_reduction",
        ),
        "child_predictive_boundary_match": _nested_get(metrics, "mean_child_predictive_boundary_match", "child_predictive_boundary_match"),
        "child_predictive_ontology_binding_mass": _nested_get(
            metrics,
            "mean_child_predictive_ontology_binding_mass",
            "child_predictive_ontology_binding_mass",
        ),
        "parent_ontology_charge": _nested_get(
            metrics,
            "mean_assay_mode_replace_mode1_replace_parent_ontology_charge",
            "mean_parent_ontology_charge",
            "parent_ontology_charge_mean",
        ),
        "parent_ontology_mode1_occupancy_after": _nested_get(
            metrics,
            "mean_assay_mode_replace_mode1_replace_parent_ontology_mode1_occupancy_after",
            "mean_parent_ontology_mode1_occupancy_after",
            "parent_ontology_mode1_occupancy_after",
        ),
        "parent_ontology_gate_floor": _as_float(assay.get("parent_ontology_gate_floor")),
        "parent_ontology_charge_gain": _as_float(assay.get("parent_ontology_charge_gain")),
        "parent_ontology_mode1_logit_boost": _as_float(assay.get("parent_ontology_mode1_logit_boost")),
        "parent_ontology_support_boost": _as_float(assay.get("parent_ontology_support_boost")),
        "mode1_replace_ratio": _nested_get(metrics, "mean_assay_mode_replace_mode1_replace_ratio", "mode1_replace_ratio"),
        "mode1_replace_gate_floor": _nested_get(
            metrics,
            "mean_assay_mode_replace_mode1_replace_gate_floor",
            "mode1_replace_gate_floor",
        ),
        "mode1_replace_ratio_scale": _as_float(assay.get("mode_replace_ratio_scale")) or _as_float(metrics.get("mode1_replace_ratio_scale")),
        "mode1_replace_support_gate_mean": _as_float(metrics.get("mean_assay_mode_replace_mode1_replace_gate")),
        "mode1_replace_support_gate_max": _as_float(metrics.get("mean_assay_mode_replace_max_nested_sibling_readiness")),
        "child_volume_active_count": _nested_get(metrics, "mean_child_volume_active_count", "child_volume_active_count"),
        "child_volume_effective_count": _nested_get(metrics, "mean_child_volume_effective_count", "child_volume_effective_count"),
        "child_volume_top_score_share": _nested_get(metrics, "mean_child_volume_top_score_share", "child_volume_top_score_share"),
        "world_jump_penalty": _nested_get(metrics, "mean_world_jump_penalty", "world_jump_penalty"),
        "direct_readout_dependency": _nested_get(metrics, "mean_direct_readout_dependency", "direct_readout_dependency"),
        "final_direct_mix_shortcut_score": _nested_get(
            metrics,
            "mean_final_direct_mix_shortcut_score",
            "final_direct_mix_shortcut_score",
        ),
    }


def _uses_oracle(*parts: Any) -> bool:
    text = " ".join(json.dumps(part, sort_keys=True, default=str) if isinstance(part, (dict, list)) else str(part) for part in parts)
    return "oracle" in text.lower()


def _make_row(
    *,
    source_path: Path,
    source_kind: str,
    case: str = "",
    variant: str = "",
    branch: str = "",
    family: str = "",
    label: str = "",
    features: dict[str, float],
    target_hint: float | None = None,
    oracle_parts: tuple[Any, ...] = (),
) -> ScoutRow:
    readout_crossing = int(features.get("readout_minus_baseline", 0.0) > TARGET_EPSILON)
    nested = int(label == "nested_sibling" or (target_hint is not None and target_hint > TARGET_EPSILON))
    bridge = int(nested or readout_crossing)
    return ScoutRow(
        source_path=str(source_path),
        source_kind=source_kind,
        case=case,
        variant=variant,
        branch=branch,
        family=family,
        label=label,
        features={name: _as_float(features.get(name)) for name in FEATURES},
        target_nested_sibling=nested,
        target_readout_crossing=readout_crossing,
        target_bridge=bridge,
        uses_oracle=_uses_oracle(str(source_path), *oracle_parts),
    )


def _extract_rows_from_nested_report(path: Path, payload: dict[str, Any]) -> list[ScoutRow]:
    rows: list[ScoutRow] = []
    variant = path.parent.name
    for case_row in payload.get("cases", []):
        if not isinstance(case_row, dict):
            continue
        case = _safe_name(case_row.get("case"))
        for branch, metrics in _as_dict(case_row.get("branch_metrics")).items():
            if not isinstance(metrics, dict):
                continue
            features = _extract_branch_features(metrics, case_row, payload)
            label = _safe_name(metrics.get("label"))
            rows.append(
                _make_row(
                    source_path=path,
                    source_kind="branch_metrics",
                    case=case,
                    variant=variant,
                    branch=str(branch),
                    family=_safe_name(metrics.get("family")),
                    label=label,
                    features=features,
                    oracle_parts=(payload.get("assay_killswitch"), metrics),
                )
            )
    return rows


def _extract_rows_from_compare(path: Path, payload: dict[str, Any]) -> list[ScoutRow]:
    rows: list[ScoutRow] = []
    for key, value in payload.items():
        if isinstance(value, dict) and isinstance(value.get("nested_rows"), list):
            for item in value.get("nested_rows", []):
                if not isinstance(item, dict):
                    continue
                features = _extract_flat_row_features(item, value)
                rows.append(
                    _make_row(
                        source_path=path,
                        source_kind="nested_rows",
                        case=_safe_name(item.get("case")),
                        variant=str(key),
                        branch=_safe_name(item.get("branch")),
                        family=_safe_name(item.get("family")),
                        label=_safe_name(item.get("label")),
                        features=features,
                        target_hint=_as_float(item.get("nested_sibling_fraction")) if "nested_sibling_fraction" in item else None,
                        oracle_parts=(value, item),
                    )
                )
        if isinstance(value, list) and ("branch" in key or "readout" in key or "nested" in key):
            for item in value:
                if not isinstance(item, dict):
                    continue
                features = _extract_flat_row_features(item, payload)
                rows.append(
                    _make_row(
                        source_path=path,
                        source_kind="flat_branch_list",
                        case=_safe_name(item.get("case")),
                        variant=str(key),
                        branch=_safe_name(item.get("branch")),
                        family=_safe_name(item.get("family")),
                        label=_safe_name(item.get("label")),
                        features=features,
                        oracle_parts=(item,),
                    )
                )
    return rows


def _extract_rows_from_sweep_summary(path: Path, payload: dict[str, Any]) -> list[ScoutRow]:
    rows: list[ScoutRow] = []
    for variant, value in _as_dict(payload.get("variants")).items():
        if not isinstance(value, dict):
            continue
        metrics = _as_dict(value.get("metrics"))
        if not metrics:
            continue
        features = _extract_flat_row_features(value, payload)
        target_hint = max(
            _as_float(metrics.get("mean_assay_mode_replace_nested_sibling_fraction")),
            _as_float(metrics.get("mean_parent_mode_conversion_sibling_fraction")),
            _as_float(metrics.get("mean_mode_replace_conversion_score")),
        )
        rows.append(
            _make_row(
                source_path=path,
                source_kind="sweep_variant_summary",
                variant=str(variant),
                branch="variant_summary",
                family="sweep_summary",
                label="nested_sibling" if target_hint > TARGET_EPSILON else "not_nested_sibling",
                features=features,
                target_hint=target_hint,
                oracle_parts=(value.get("assay_killswitch"), metrics, variant),
            )
        )
    return rows


def extract_rows(paths: list[Path]) -> tuple[list[ScoutRow], dict[str, int]]:
    rows: list[ScoutRow] = []
    counts = {
        "json_files_seen": 0,
        "json_files_read": 0,
        "nested_reports": 0,
        "sweep_summaries": 0,
        "compare_reports": 0,
    }
    for path in paths:
        counts["json_files_seen"] += 1
        payload = _read_json(path)
        if not isinstance(payload, dict):
            continue
        counts["json_files_read"] += 1
        before = len(rows)
        if isinstance(payload.get("cases"), list):
            rows.extend(_extract_rows_from_nested_report(path, payload))
            if len(rows) > before:
                counts["nested_reports"] += 1
        before = len(rows)
        if isinstance(payload.get("variants"), dict):
            rows.extend(_extract_rows_from_sweep_summary(path, payload))
            if len(rows) > before:
                counts["sweep_summaries"] += 1
        before = len(rows)
        rows.extend(_extract_rows_from_compare(path, payload))
        if len(rows) > before:
            counts["compare_reports"] += 1
    return rows, counts


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _std(values: list[float], mean: float) -> float:
    if not values:
        return 1.0
    var = sum((value - mean) ** 2 for value in values) / len(values)
    return math.sqrt(var) if var > 1.0e-24 else 1.0


def _solve_linear_system(matrix: list[list[float]], vector: list[float]) -> list[float]:
    n = len(vector)
    aug = [row[:] + [vector[i]] for i, row in enumerate(matrix)]
    for col in range(n):
        pivot = max(range(col, n), key=lambda row: abs(aug[row][col]))
        if abs(aug[pivot][col]) < 1.0e-12:
            aug[col][col] += 1.0e-8
            pivot = col
        if pivot != col:
            aug[col], aug[pivot] = aug[pivot], aug[col]
        scale = aug[col][col]
        if abs(scale) < 1.0e-12:
            continue
        for j in range(col, n + 1):
            aug[col][j] /= scale
        for row in range(n):
            if row == col:
                continue
            factor = aug[row][col]
            if factor == 0.0:
                continue
            for j in range(col, n + 1):
                aug[row][j] -= factor * aug[col][j]
    return [aug[i][n] for i in range(n)]


def _fit_ridge(rows: list[ScoutRow], feature_names: list[str], l2: float) -> dict[str, Any]:
    values_by_feature = {name: [row.features.get(name, 0.0) for row in rows] for name in feature_names}
    means = {name: _mean(values) for name, values in values_by_feature.items()}
    stds = {name: _std(values, means[name]) for name, values in values_by_feature.items()}
    width = len(feature_names) + 1
    xtx = [[0.0 for _ in range(width)] for _ in range(width)]
    xty = [0.0 for _ in range(width)]
    for row in rows:
        x = [1.0] + [(row.features.get(name, 0.0) - means[name]) / stds[name] for name in feature_names]
        y = float(row.target_bridge)
        for i in range(width):
            xty[i] += x[i] * y
            for j in range(width):
                xtx[i][j] += x[i] * x[j]
    for i in range(1, width):
        xtx[i][i] += l2
    coef = _solve_linear_system(xtx, xty)
    predictions = []
    for row in rows:
        x = [1.0] + [(row.features.get(name, 0.0) - means[name]) / stds[name] for name in feature_names]
        score = sum(c * value for c, value in zip(coef, x))
        predictions.append(score)
    labels = [row.target_bridge for row in rows]
    classified = [1 if score >= 0.5 else 0 for score in predictions]
    accuracy = sum(int(pred == label) for pred, label in zip(classified, labels)) / len(labels) if labels else 0.0
    mse = _mean([(score - label) ** 2 for score, label in zip(predictions, labels)])
    return {
        "model_type": "standardized_ridge_linear_probability_surrogate",
        "l2": l2,
        "intercept": coef[0],
        "coefficients": {name: coef[i + 1] for i, name in enumerate(feature_names)},
        "feature_means": means,
        "feature_stds": stds,
        "train_accuracy_at_0_5": accuracy,
        "train_mse": mse,
        "predictions": predictions,
    }


def _select_features(rows: list[ScoutRow]) -> list[str]:
    selected = []
    for name in FEATURES:
        values = [row.features.get(name, 0.0) for row in rows]
        if max(values, default=0.0) - min(values, default=0.0) > 1.0e-12:
            selected.append(name)
    return selected


def _select_named_features(rows: list[ScoutRow], names: list[str]) -> list[str]:
    selected = []
    for name in names:
        values = [row.features.get(name, 0.0) for row in rows]
        if max(values, default=0.0) - min(values, default=0.0) > 1.0e-12:
            selected.append(name)
    return selected


def _accuracy_for_threshold(rows: list[ScoutRow], name: str, threshold: float, direction: int) -> float:
    if not rows:
        return 0.0
    correct = 0
    for row in rows:
        value = row.features.get(name, 0.0)
        pred = int(value >= threshold) if direction >= 0 else int(value <= threshold)
        correct += int(pred == row.target_bridge)
    return correct / len(rows)


def _best_threshold(rows: list[ScoutRow], name: str) -> dict[str, Any]:
    values = sorted({row.features.get(name, 0.0) for row in rows})
    if not values:
        return {}
    candidates = [values[0] - 1.0e-9, values[-1] + 1.0e-9]
    candidates.extend(values)
    candidates.extend((a + b) / 2.0 for a, b in zip(values, values[1:]))
    best = {"feature": name, "threshold": candidates[0], "direction": ">=", "accuracy": -1.0}
    for threshold in candidates:
        for direction in (1, -1):
            accuracy = _accuracy_for_threshold(rows, name, threshold, direction)
            if accuracy > best["accuracy"]:
                best = {
                    "feature": name,
                    "threshold": threshold,
                    "direction": ">=" if direction >= 0 else "<=",
                    "accuracy": accuracy,
                }
    negatives = [row.features.get(name, 0.0) for row in rows if row.target_bridge == 0]
    positives = [row.features.get(name, 0.0) for row in rows if row.target_bridge == 1]
    best["negative_max"] = max(negatives) if negatives else None
    best["positive_min"] = min(positives) if positives else None
    return best


def _boundary(rows: list[ScoutRow], feature_names: list[str]) -> dict[str, Any]:
    thresholds = [_best_threshold(rows, name) for name in feature_names]
    thresholds = [row for row in thresholds if row]
    thresholds.sort(key=lambda row: (-_as_float(row.get("accuracy")), str(row.get("feature"))))
    gate_rows = [row for row in thresholds if row.get("feature") == "mode1_replace_gate_floor"]
    parent_gate_rows = [row for row in thresholds if row.get("feature") == "parent_ontology_gate_floor"]
    occ_rows = [row for row in thresholds if row.get("feature") == "parent_ontology_mode1_occupancy_after"]
    return {
        "best_univariate_boundaries": thresholds[:12],
        "estimated_parent_ontology_gate_floor_boundary": parent_gate_rows[0] if parent_gate_rows else None,
        "estimated_gate_floor_boundary": gate_rows[0] if gate_rows else None,
        "estimated_mode1_occupancy_boundary": occ_rows[0] if occ_rows else None,
    }


def _boundary_by_family(rows: list[ScoutRow], feature_names: list[str]) -> dict[str, Any]:
    families = sorted({row.family or "unknown" for row in rows})
    result: dict[str, Any] = {}
    for family in families:
        family_rows = [row for row in rows if (row.family or "unknown") == family]
        if len({row.target_bridge for row in family_rows}) < 2:
            continue
        result[family] = {
            "row_count": len(family_rows),
            "positive_bridge_count": sum(row.target_bridge for row in family_rows),
            "boundary": _boundary(family_rows, feature_names),
        }
    return result


def _recommend_next_target(rows: list[ScoutRow], ridge: dict[str, Any]) -> dict[str, Any]:
    predictions = ridge.get("predictions", [])
    scored = list(zip(rows, predictions))
    non_oracle_negatives = [
        (row, _as_float(score))
        for row, score in scored
        if row.target_bridge == 0 and not row.uses_oracle and "oracle" not in row.variant.lower() and "oracle" not in row.branch.lower()
    ]
    if not non_oracle_negatives:
        non_oracle_negatives = [
            (row, _as_float(score))
            for row, score in scored
            if row.target_bridge == 0 and "oracle" not in row.variant.lower() and "oracle" not in row.branch.lower()
        ]
    fallback_is_oracle_tainted = False
    if not non_oracle_negatives:
        non_oracle_negatives = [
            (row, _as_float(score))
            for row, score in scored
            if row.target_bridge == 0
        ]
        fallback_is_oracle_tainted = True
    if not non_oracle_negatives:
        return {"status": "no_negative_candidate_found"}
    candidate, score = max(non_oracle_negatives, key=lambda item: item[1])
    positives = [row for row in rows if row.target_bridge == 1]
    gate = candidate.features.get("parent_ontology_gate_floor", 0.0)
    occ = candidate.features.get("parent_ontology_mode1_occupancy_after", 0.0)
    positive_gates = sorted(row.features.get("parent_ontology_gate_floor", 0.0) for row in positives if row.features.get("parent_ontology_gate_floor", 0.0) > gate)
    positive_occs = sorted(row.features.get("parent_ontology_mode1_occupancy_after", 0.0) for row in positives if row.features.get("parent_ontology_mode1_occupancy_after", 0.0) > occ)
    next_gate = (gate + positive_gates[0]) / 2.0 if positive_gates else (gate + 0.01 if gate else None)
    next_occ = (occ + positive_occs[0]) / 2.0 if positive_occs else None
    return {
        "status": "non_oracle_target_from_oracle_tainted_calibration" if (fallback_is_oracle_tainted or candidate.uses_oracle) else "candidate",
        "source_path": candidate.source_path,
        "source_kind": candidate.source_kind,
        "variant": candidate.variant,
        "case": candidate.case,
        "branch": candidate.branch,
        "family": candidate.family,
        "source_uses_oracle": candidate.uses_oracle,
        "surrogate_score": score,
        "current_parent_ontology_gate_floor": gate,
        "current_mode1_occupancy_after": occ,
        "recommended_next_parent_ontology_gate_floor": next_gate,
        "recommended_next_mode1_occupancy_after": next_occ,
        "non_oracle_bridge_target": {
            "branch_family": candidate.family or "mode_replace",
            "branch": candidate.branch,
            "learned_head": "predict parent_ontology_charge / mode1 occupancy from non-oracle branch features",
            "avoid": [
                "child_predictive_candidate=oracle_parent_residual",
                "oracle residual labels at runtime",
                "promotion to default behavior",
            ],
        },
        "rationale": "Use the highest-scoring miss as the next learned-head target; if calibration evidence is oracle-tainted, only copy the boundary/occupancy target, not the oracle residual source.",
    }


def _summarize_rows(rows: list[ScoutRow]) -> dict[str, Any]:
    by_kind: dict[str, int] = {}
    by_family: dict[str, int] = {}
    for row in rows:
        by_kind[row.source_kind] = by_kind.get(row.source_kind, 0) + 1
        by_family[row.family or "unknown"] = by_family.get(row.family or "unknown", 0) + 1
    positives = [row for row in rows if row.target_bridge == 1]
    return {
        "row_count": len(rows),
        "positive_bridge_count": len(positives),
        "nested_sibling_count": sum(row.target_nested_sibling for row in rows),
        "readout_crossing_count": sum(row.target_readout_crossing for row in rows),
        "oracle_tainted_count": sum(1 for row in rows if row.uses_oracle),
        "source_kind_counts": dict(sorted(by_kind.items())),
        "family_counts": dict(sorted(by_family.items())),
    }


def run_bridge_scout(input_dirs: list[Path], out_path: Path, l2: float = DEFAULT_L2) -> dict[str, Any]:
    json_paths = _iter_json_files(input_dirs)
    rows, file_counts = extract_rows(json_paths)
    feature_names = _select_features(rows)
    if not rows:
        raise SystemExit("No usable parent ontology/nested report rows found.")
    if len({row.target_bridge for row in rows}) < 2:
        raise SystemExit("Need both positive and negative bridge targets to fit a boundary.")
    if not feature_names:
        raise SystemExit("No varying numeric features found.")
    ridge = _fit_ridge(rows, feature_names, l2=l2)
    nonleaky_feature_names = _select_named_features(rows, NONLEAKY_FEATURES)
    nonleaky_model: dict[str, Any] | None = None
    nonleaky_coefficients: dict[str, float] = {}
    nonleaky_saliency: list[dict[str, Any]] = []
    if nonleaky_feature_names:
        nonleaky_model = _fit_ridge(rows, nonleaky_feature_names, l2=l2)
        nonleaky_coefficients = dict(nonleaky_model.pop("coefficients"))
        nonleaky_saliency = [
            {"feature": name, "coefficient": nonleaky_coefficients[name], "abs_coefficient": abs(nonleaky_coefficients[name])}
            for name in nonleaky_feature_names
        ]
        nonleaky_saliency.sort(key=lambda row: (-row["abs_coefficient"], row["feature"]))
    coefficients = ridge.pop("coefficients")
    saliency = [
        {"feature": name, "coefficient": coefficients[name], "abs_coefficient": abs(coefficients[name])}
        for name in feature_names
    ]
    saliency.sort(key=lambda row: (-row["abs_coefficient"], row["feature"]))
    payload = {
        "schema": SCHEMA,
        "created_utc": _utc_timestamp(),
        "input_dirs": [str(path) for path in input_dirs],
        "out_path": str(out_path),
        "file_counts": file_counts,
        "row_summary": _summarize_rows(rows),
        "target_definition": {
            "target_bridge": "label == nested_sibling OR readout_sibling_response > readout_only_baseline OR positive mode-replace summary target",
            "target_nested_sibling": "branch/report label or positive mode-replace nested/conversion summary",
            "target_readout_crossing": "readout_sibling_response - readout_only_baseline > 0",
        },
        "feature_names": feature_names,
        "nonleaky_feature_names": nonleaky_feature_names,
        "model": {
            **ridge,
            "coefficients": coefficients,
            "feature_saliency": saliency,
        },
        "nonleaky_model": (
            {
                **nonleaky_model,
                "coefficients": nonleaky_coefficients,
                "feature_saliency": nonleaky_saliency,
            }
            if nonleaky_model is not None
            else None
        ),
        "learned_boundary": _boundary(rows, feature_names),
        "learned_boundary_by_family": _boundary_by_family(rows, feature_names),
        "nonleaky_boundary": _boundary(rows, nonleaky_feature_names) if nonleaky_feature_names else None,
        "nonleaky_boundary_by_family": _boundary_by_family(rows, nonleaky_feature_names) if nonleaky_feature_names else {},
        "recommended_next_non_oracle_bridge_target": _recommend_next_target(rows, {**ridge, "predictions": ridge["predictions"]}),
        "sample_rows": [
            {
                "source_path": row.source_path,
                "source_kind": row.source_kind,
                "variant": row.variant,
                "case": row.case,
                "branch": row.branch,
                "family": row.family,
                "label": row.label,
                "target_bridge": row.target_bridge,
                "uses_oracle": row.uses_oracle,
                "features": row.features,
            }
            for row in rows[:20]
        ],
        "notes": [
            "Scout only: reads existing JSON artifacts and writes this summary; it does not import or mutate Circleworld runtime code.",
            "Oracle-tainted rows are retained for boundary learning but excluded from the first-choice next-target recommendation when possible.",
            "Ridge coefficients are descriptive saliency, not a production policy.",
            "The nonleaky model excludes direct readout/label and post-replacement outcome fields; use it for runtime-head design, not the all-feature model.",
        ],
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return payload


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Learn a small parent-ontology bridge boundary from existing nested calibration JSON reports."
    )
    parser.add_argument(
        "--input-dir",
        dest="input_dirs",
        action="append",
        type=Path,
        required=True,
        help="Directory or JSON file to scan. Repeat for multiple calibration/report roots.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        required=True,
        help="Path for the JSON scout summary.",
    )
    parser.add_argument(
        "--l2",
        type=float,
        default=DEFAULT_L2,
        help=f"Ridge regularization strength. Default: {DEFAULT_L2}",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    payload = run_bridge_scout(args.input_dirs, args.out, l2=float(args.l2))
    print(
        json.dumps(
            {
                "schema": payload["schema"],
                "out_path": payload["out_path"],
                "row_summary": payload["row_summary"],
                "train_accuracy_at_0_5": payload["model"]["train_accuracy_at_0_5"],
                "nonleaky_train_accuracy_at_0_5": payload["nonleaky_model"]["train_accuracy_at_0_5"] if payload.get("nonleaky_model") else None,
                "best_boundary": payload["learned_boundary"]["best_univariate_boundaries"][:3],
                "best_nonleaky_boundary": payload["nonleaky_boundary"]["best_univariate_boundaries"][:3] if payload.get("nonleaky_boundary") else [],
                "recommendation": payload["recommended_next_non_oracle_bridge_target"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
