from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


PARENT_PHASOR_CARRIER_FEATURES: tuple[str, ...] = (
    "bias",
    "parent_cos",
    "parent_sin",
    "raw_delta",
    "sin_raw_delta",
    "cos_raw_delta",
    "raw_low_rank_delta",
    "parent_tangent_delta",
    "parent_tangent_low_rank_delta",
    "gate",
    "boundary_mask",
    "child_parent_alignment_centered",
    "parent_cos_x_cos_raw",
    "parent_sin_x_sin_raw",
    "parent_sin_x_cos_raw",
    "parent_cos_x_sin_raw",
    "gate_x_sin_raw",
    "gate_x_cos_raw",
    "boundary_x_sin_raw",
    "boundary_x_cos_raw",
)

PARENT_AUTHORITY_FEATURES: tuple[str, ...] = (
    "bias",
    "boundary_match",
    "boundary_score",
    "support_gate_mean",
    "support_score",
    "support_gate_max",
    "raw_low_rank_abs_mean",
    "raw_low_rank_sq_mean",
    "parent_tangent_abs_mean",
    "boundary_weighted_mean",
    "raw_abs_x_support",
    "raw_abs_x_boundary",
    "boundary_x_support",
    "support_x_boundary_score",
    "raw_minus_tangent_abs_mean",
)


def _npz_paths(inputs: list[str]) -> list[Path]:
    paths: list[Path] = []
    for raw in inputs:
        path = Path(raw)
        if path.is_dir():
            paths.extend(sorted(path.rglob("*.npz")))
        elif path.is_file():
            paths.append(path)
        else:
            paths.extend(sorted(Path().glob(raw)))
    return sorted(dict.fromkeys(paths))


def _item_string(value: Any, default: str) -> str:
    try:
        arr = np.asarray(value)
        if arr.shape == ():
            return str(arr.item())
    except Exception:
        pass
    return default


def _load_rows(
    paths: list[Path],
    *,
    target_key: str,
    min_gate: float,
    max_samples_per_file: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict[str, Any]], list[str]]:
    rng = np.random.default_rng(seed)
    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    ws: list[np.ndarray] = []
    groups: list[dict[str, Any]] = []
    feature_names: list[str] = []
    for path in paths:
        with np.load(path, allow_pickle=True) as payload:
            features = np.asarray(payload["features"], dtype=np.float64)
            target = np.asarray(payload[target_key], dtype=np.float64)
            gate = np.asarray(payload["gate"], dtype=np.float64)
            if not feature_names and "feature_names" in payload:
                feature_names = [str(item) for item in np.asarray(payload["feature_names"], dtype=object).tolist()]
            case = _item_string(payload["case"], path.stem)
            branch = _item_string(payload["branch"], path.stem)

        x = np.moveaxis(features, 0, -1).reshape(-1, features.shape[0])
        y = target.reshape(-1)
        w = gate.reshape(-1)
        keep = np.isfinite(x).all(axis=1) & np.isfinite(y) & np.isfinite(w) & (w >= float(min_gate))
        x = x[keep]
        y = y[keep]
        w = w[keep]
        if max_samples_per_file > 0 and len(y) > max_samples_per_file:
            idx = rng.choice(len(y), size=int(max_samples_per_file), replace=False)
            idx.sort()
            x = x[idx]
            y = y[idx]
            w = w[idx]
        xs.append(x)
        ys.append(y)
        ws.append(w)
        groups.extend({"case": case, "branch": branch, "path": str(path)} for _ in range(len(y)))
    if not xs:
        raise ValueError("No projector rows found")
    if not feature_names:
        feature_names = [f"feature_{idx}" for idx in range(xs[0].shape[1])]
    return np.vstack(xs), np.concatenate(ys), np.concatenate(ws), groups, feature_names


def _load_phasor_rows(
    paths: list[Path],
    *,
    min_gate: float,
    max_samples_per_file: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[dict[str, Any]], list[str]]:
    rng = np.random.default_rng(seed + 7919)
    xs: list[np.ndarray] = []
    y_cos: list[np.ndarray] = []
    y_sin: list[np.ndarray] = []
    ws: list[np.ndarray] = []
    groups: list[dict[str, Any]] = []
    for path in paths:
        with np.load(path, allow_pickle=True) as payload:
            parent = np.asarray(payload["parent_mode1"], dtype=np.float64)
            child = np.asarray(payload["child_phase"], dtype=np.float64)
            raw_delta = np.asarray(payload["raw_delta"], dtype=np.float64)
            gate = np.asarray(payload["gate"], dtype=np.float64)
            boundary = np.asarray(payload["boundary_mask"], dtype=np.float64)
            raw_low_rank = np.asarray(payload["raw_child_delta"], dtype=np.float64)
            parent_tangent = np.asarray(payload["parent_tangent_delta"], dtype=np.float64)
            parent_tangent_low_rank = np.asarray(payload["parent_tangent_low_rank_delta"], dtype=np.float64)
            case = _item_string(payload["case"], path.stem)
            branch = _item_string(payload["branch"], path.stem)

        parent_cos = parent[..., 0]
        parent_sin = parent[..., 1]
        child_cos = child[..., 0]
        child_sin = child[..., 1]
        sin_raw = np.sin(raw_delta)
        cos_raw = np.cos(raw_delta)
        align = (parent_cos * child_cos + parent_sin * child_sin).clip(-1.0, 1.0)
        align_centered = align
        features = np.stack(
            [
                np.ones_like(gate),
                parent_cos,
                parent_sin,
                raw_delta,
                sin_raw,
                cos_raw,
                raw_low_rank,
                parent_tangent,
                parent_tangent_low_rank,
                gate,
                boundary,
                align_centered,
                parent_cos * cos_raw,
                parent_sin * sin_raw,
                parent_sin * cos_raw,
                parent_cos * sin_raw,
                gate * sin_raw,
                gate * cos_raw,
                boundary * sin_raw,
                boundary * cos_raw,
            ],
            axis=0,
        )
        x = np.moveaxis(features, 0, -1).reshape(-1, features.shape[0])
        yc = child_cos.reshape(-1)
        ys = child_sin.reshape(-1)
        w = gate.reshape(-1)
        keep = np.isfinite(x).all(axis=1) & np.isfinite(yc) & np.isfinite(ys) & np.isfinite(w) & (w >= float(min_gate))
        x = x[keep]
        yc = yc[keep]
        ys = ys[keep]
        w = w[keep]
        if max_samples_per_file > 0 and len(w) > max_samples_per_file:
            idx = rng.choice(len(w), size=int(max_samples_per_file), replace=False)
            idx.sort()
            x = x[idx]
            yc = yc[idx]
            ys = ys[idx]
            w = w[idx]
        xs.append(x)
        y_cos.append(yc)
        y_sin.append(ys)
        ws.append(w)
        groups.extend({"case": case, "branch": branch, "path": str(path)} for _ in range(len(w)))
    if not xs:
        raise ValueError("No phasor carrier rows found")
    return (
        np.vstack(xs),
        np.concatenate(y_cos),
        np.concatenate(y_sin),
        np.concatenate(ws),
        groups,
        list(PARENT_PHASOR_CARRIER_FEATURES),
    )


def _weighted_mean(value: np.ndarray, weight: np.ndarray) -> float:
    denom = float(np.clip(weight.sum(), 1.0e-12, None))
    return float((value * weight).sum() / denom)


def _authority_features_from_payload(payload: Any, metrics: dict[str, Any]) -> list[float]:
    gate = np.asarray(payload["gate"], dtype=np.float64)
    boundary = np.asarray(payload["boundary_mask"], dtype=np.float64)
    raw_low_rank = np.asarray(payload["raw_child_delta"], dtype=np.float64)
    parent_tangent = np.asarray(payload["parent_tangent_low_rank_delta"], dtype=np.float64)
    weight = np.clip(gate, 0.0, None)
    boundary_match = float(metrics.get("boundary_match", 0.0))
    boundary_center = 0.952
    boundary_scale = 120.0
    boundary_score = float(1.0 / (1.0 + np.exp(-np.clip((boundary_match - boundary_center) * boundary_scale, -60.0, 60.0))))
    support_mean = float(metrics.get("support_gate_mean", float(np.mean(gate))))
    support_max = float(metrics.get("support_gate_max", float(np.max(gate))))
    support_score = float(np.clip(support_mean / 0.158, 0.0, 1.0))
    raw_abs = _weighted_mean(np.abs(raw_low_rank), weight)
    raw_sq = _weighted_mean(raw_low_rank * raw_low_rank, weight)
    tangent_abs = _weighted_mean(np.abs(parent_tangent), weight)
    boundary_mean = _weighted_mean(np.clip(boundary, 0.0, 1.0), weight)
    return [
        1.0,
        boundary_match,
        boundary_score,
        support_mean,
        support_score,
        support_max,
        raw_abs,
        raw_sq,
        tangent_abs,
        boundary_mean,
        raw_abs * support_mean,
        raw_abs * boundary_mean,
        boundary_mean * support_mean,
        support_mean * boundary_score,
        abs(raw_abs - tangent_abs),
    ]


def _load_authority_rows(paths: list[Path]) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict[str, Any]], list[str]]:
    xs: list[list[float]] = []
    ys: list[float] = []
    ws: list[float] = []
    groups: list[dict[str, Any]] = []
    for path in paths:
        with np.load(path, allow_pickle=True) as payload:
            metrics_raw = payload["metrics"].item() if "metrics" in payload else "{}"
            metrics = json.loads(str(metrics_raw)) if isinstance(metrics_raw, str) else dict(metrics_raw)
            case = _item_string(payload["case"], path.stem)
            branch = _item_string(payload["branch"], path.stem)
            features = _authority_features_from_payload(payload, metrics)
        oracle_reduction = max(0.0, float(metrics.get("oracle_residual_reduction", metrics.get("residual_reduction", 0.0))))
        boundary_match = max(0.0, min(1.0, float(metrics.get("boundary_match", 0.0))))
        target_acceptance = max(0.0, min(1.0, oracle_reduction * boundary_match))
        support_weight = max(1.0e-6, float(metrics.get("support_gate_mean", 0.0)))
        xs.append(features)
        ys.append(target_acceptance)
        ws.append(support_weight)
        groups.append({"case": case, "branch": branch, "path": str(path)})
    if not xs:
        raise ValueError("No authority rows found")
    return np.asarray(xs, dtype=np.float64), np.asarray(ys, dtype=np.float64), np.asarray(ws, dtype=np.float64), groups, list(PARENT_AUTHORITY_FEATURES)


def _fit_ridge(x: np.ndarray, y: np.ndarray, w: np.ndarray, ridge_lambda: float) -> np.ndarray:
    sqrt_w = np.sqrt(np.clip(w, 0.0, None))[:, None]
    xw = x * sqrt_w
    yw = y * sqrt_w[:, 0]
    xtx = xw.T @ xw
    reg = np.eye(x.shape[1], dtype=np.float64) * float(ridge_lambda)
    reg[0, 0] = 0.0
    xty = xw.T @ yw
    return np.linalg.solve(xtx + reg, xty)


def _weighted_mae(delta: np.ndarray, w: np.ndarray) -> float:
    denom = float(np.clip(w.sum(), 1.0e-12, None))
    return float((np.abs(delta) * w).sum() / denom)


def _score_split(x: np.ndarray, y: np.ndarray, w: np.ndarray, coeffs: np.ndarray) -> dict[str, float]:
    pred = x @ coeffs
    zero_mae = _weighted_mae(y, w)
    raw_mae = _weighted_mae(y - x[:, 1], w) if x.shape[1] > 1 else 0.0
    tangent_mae = _weighted_mae(y - x[:, 2], w) if x.shape[1] > 2 else 0.0
    signed_tangent_mae = _weighted_mae(y - x[:, 3], w) if x.shape[1] > 3 else 0.0
    pred_mae = _weighted_mae(y - pred, w)
    pred_abs = _weighted_mae(pred, w)
    target_abs = _weighted_mae(y, w)
    return {
        "num_samples": float(len(y)),
        "weight_mean": float(np.mean(w)) if len(w) else 0.0,
        "target_abs_mean": float(target_abs),
        "prediction_abs_mean": float(pred_abs),
        "zero_mae": float(zero_mae),
        "raw_child_delta_mae": float(raw_mae),
        "parent_tangent_mae": float(tangent_mae),
        "signed_parent_tangent_mae": float(signed_tangent_mae),
        "projector_mae": float(pred_mae),
        "projector_reduction_vs_zero": float((zero_mae - pred_mae) / max(1.0e-12, zero_mae)),
        "projector_reduction_vs_raw": float((raw_mae - pred_mae) / max(1.0e-12, raw_mae)) if raw_mae else 0.0,
        "projector_reduction_vs_parent_tangent": float((tangent_mae - pred_mae) / max(1.0e-12, tangent_mae)) if tangent_mae else 0.0,
    }


def _score_authority_split(x: np.ndarray, y: np.ndarray, w: np.ndarray, coeffs: np.ndarray) -> dict[str, float]:
    pred = np.clip(x @ coeffs, 0.0, 1.0)
    static_acceptance = np.clip(0.13 * x[:, 2], 0.0, 1.0) if x.shape[1] > 2 else np.zeros_like(y)
    support_proxy = np.clip(x[:, 3] * x[:, 1], 0.0, 1.0) if x.shape[1] > 3 else np.zeros_like(y)
    zero_mae = _weighted_mae(y, w)
    static_mae = _weighted_mae(y - static_acceptance, w)
    support_mae = _weighted_mae(y - support_proxy, w)
    pred_mae = _weighted_mae(y - pred, w)
    return {
        "num_samples": float(len(y)),
        "weight_mean": float(np.mean(w)) if len(w) else 0.0,
        "target_acceptance_mean": float(_weighted_mae(y, w)),
        "prediction_acceptance_mean": float(_weighted_mae(pred, w)),
        "zero_mae": float(zero_mae),
        "static_bridge_mae": float(static_mae),
        "support_boundary_proxy_mae": float(support_mae),
        "authority_mae": float(pred_mae),
        "authority_reduction_vs_zero": float((zero_mae - pred_mae) / max(1.0e-12, zero_mae)),
        "authority_reduction_vs_static_bridge": float((static_mae - pred_mae) / max(1.0e-12, static_mae)) if static_mae else 0.0,
        "authority_reduction_vs_support_boundary": float((support_mae - pred_mae) / max(1.0e-12, support_mae)) if support_mae else 0.0,
    }


def _score_phasor_split(
    x: np.ndarray,
    y_cos: np.ndarray,
    y_sin: np.ndarray,
    w: np.ndarray,
    cos_coeffs: np.ndarray,
    sin_coeffs: np.ndarray,
) -> dict[str, float]:
    pred_cos = x @ cos_coeffs
    pred_sin = x @ sin_coeffs
    norm = np.sqrt(pred_cos * pred_cos + pred_sin * pred_sin)
    pred_cos = pred_cos / np.clip(norm, 1.0e-12, None)
    pred_sin = pred_sin / np.clip(norm, 1.0e-12, None)
    dot = np.clip(pred_cos * y_cos + pred_sin * y_sin, -1.0, 1.0)
    parent_dot = np.clip(x[:, 1] * y_cos + x[:, 2] * y_sin, -1.0, 1.0)
    phasor_l1 = np.abs(pred_cos - y_cos) + np.abs(pred_sin - y_sin)
    parent_l1 = np.abs(x[:, 1] - y_cos) + np.abs(x[:, 2] - y_sin)
    angular = np.arccos(dot)
    parent_angular = np.arccos(parent_dot)
    model_mae = _weighted_mae(phasor_l1, w)
    parent_mae = _weighted_mae(parent_l1, w)
    angular_mae = _weighted_mae(angular, w)
    parent_angular_mae = _weighted_mae(parent_angular, w)
    return {
        "num_samples": float(len(w)),
        "weight_mean": float(np.mean(w)) if len(w) else 0.0,
        "phasor_l1": float(model_mae),
        "parent_phasor_l1": float(parent_mae),
        "angular_mae": float(angular_mae),
        "parent_angular_mae": float(parent_angular_mae),
        "angular_reduction_vs_parent": float((parent_angular_mae - angular_mae) / max(1.0e-12, parent_angular_mae)),
    }


def _case_scores(x: np.ndarray, y: np.ndarray, w: np.ndarray, groups: list[dict[str, Any]], coeffs: np.ndarray) -> dict[str, Any]:
    cases = sorted({str(row["case"]) for row in groups})
    out: dict[str, Any] = {}
    group_cases = np.array([str(row["case"]) for row in groups], dtype=object)
    for case in cases:
        mask = group_cases == case
        out[case] = _score_split(x[mask], y[mask], w[mask], coeffs)
    return out


def _authority_case_scores(x: np.ndarray, y: np.ndarray, w: np.ndarray, groups: list[dict[str, Any]], coeffs: np.ndarray) -> dict[str, Any]:
    cases = sorted({str(row["case"]) for row in groups})
    out: dict[str, Any] = {}
    group_cases = np.array([str(row["case"]) for row in groups], dtype=object)
    for case in cases:
        mask = group_cases == case
        out[case] = _score_authority_split(x[mask], y[mask], w[mask], coeffs)
    return out


def _phasor_case_scores(
    x: np.ndarray,
    y_cos: np.ndarray,
    y_sin: np.ndarray,
    w: np.ndarray,
    groups: list[dict[str, Any]],
    cos_coeffs: np.ndarray,
    sin_coeffs: np.ndarray,
) -> dict[str, Any]:
    cases = sorted({str(row["case"]) for row in groups})
    out: dict[str, Any] = {}
    group_cases = np.array([str(row["case"]) for row in groups], dtype=object)
    for case in cases:
        mask = group_cases == case
        out[case] = _score_phasor_split(x[mask], y_cos[mask], y_sin[mask], w[mask], cos_coeffs, sin_coeffs)
    return out


def _phasor_leave_one_case_scores(
    x: np.ndarray,
    y_cos: np.ndarray,
    y_sin: np.ndarray,
    w: np.ndarray,
    groups: list[dict[str, Any]],
    *,
    ridge_lambda: float,
) -> dict[str, Any]:
    group_cases = np.array([str(row["case"]) for row in groups], dtype=object)
    out: dict[str, Any] = {}
    for case in sorted(set(group_cases.tolist())):
        test = group_cases == case
        train = ~test
        if int(train.sum()) == 0 or int(test.sum()) == 0:
            continue
        cos_coeffs = _fit_ridge(x[train], y_cos[train], w[train], ridge_lambda)
        sin_coeffs = _fit_ridge(x[train], y_sin[train], w[train], ridge_lambda)
        out[str(case)] = _score_phasor_split(x[test], y_cos[test], y_sin[test], w[test], cos_coeffs, sin_coeffs)
    return out


def _fit_phasor_ridge(
    x: np.ndarray,
    y_cos: np.ndarray,
    y_sin: np.ndarray,
    w: np.ndarray,
    ridge_lambda: float,
) -> tuple[np.ndarray, np.ndarray]:
    return _fit_ridge(x, y_cos, w, ridge_lambda), _fit_ridge(x, y_sin, w, ridge_lambda)


def _leave_one_case_scores(
    x: np.ndarray,
    y: np.ndarray,
    w: np.ndarray,
    groups: list[dict[str, Any]],
    *,
    ridge_lambda: float,
) -> dict[str, Any]:
    group_cases = np.array([str(row["case"]) for row in groups], dtype=object)
    out: dict[str, Any] = {}
    for case in sorted(set(group_cases.tolist())):
        test = group_cases == case
        train = ~test
        if int(train.sum()) == 0 or int(test.sum()) == 0:
            continue
        coeffs = _fit_ridge(x[train], y[train], w[train], ridge_lambda)
        out[str(case)] = _score_split(x[test], y[test], w[test], coeffs)
    return out


def _authority_leave_one_case_scores(
    x: np.ndarray,
    y: np.ndarray,
    w: np.ndarray,
    groups: list[dict[str, Any]],
    *,
    ridge_lambda: float,
) -> dict[str, Any]:
    group_cases = np.array([str(row["case"]) for row in groups], dtype=object)
    out: dict[str, Any] = {}
    for case in sorted(set(group_cases.tolist())):
        test = group_cases == case
        train = ~test
        if int(train.sum()) == 0 or int(test.sum()) == 0:
            continue
        coeffs = _fit_ridge(x[train], y[train], w[train], ridge_lambda)
        out[str(case)] = _score_authority_split(x[test], y[test], w[test], coeffs)
    return out


def _write_markdown(report: dict[str, Any], path: Path) -> None:
    metrics = report["metrics"]
    lines = [
        "# Parent Phase Projector Scout",
        "",
        "Tiny ridge projector trained from assay-exported tensor rows.",
        "",
        "## Fit",
        "",
        f"- files: {report['num_files']}",
        f"- rows: {int(metrics['all']['num_samples'])}",
        f"- features: {len(report['feature_names'])}",
        f"- ridge_lambda: {report['ridge_lambda']}",
        "",
        "## Metrics",
        "",
        "| split | projector_mae | zero_mae | raw_mae | tangent_mae | reduction_vs_zero | reduction_vs_raw | reduction_vs_tangent |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, row in [("all", metrics["all"])] + [(f"loo:{k}", v) for k, v in metrics.get("leave_one_case", {}).items()]:
        lines.append(
            "| {name} | {projector_mae:.8f} | {zero_mae:.8f} | {raw_child_delta_mae:.8f} | {parent_tangent_mae:.8f} | {projector_reduction_vs_zero:.6f} | {projector_reduction_vs_raw:.6f} | {projector_reduction_vs_parent_tangent:.6f} |".format(
                name=name,
                **row,
            )
        )
    lines.extend(
        [
            "",
            "## Carrier Map Metrics",
            "",
            "| split | carrier_mae | zero_mae | raw_low_rank_mae | reduction_vs_zero | reduction_vs_raw_low_rank |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    carrier_rows = [("carrier_all", metrics.get("carrier_all", {}))]
    carrier_rows.extend((f"carrier_loo:{k}", v) for k, v in metrics.get("carrier_leave_one_case", {}).items())
    for name, row in carrier_rows:
        if not row:
            continue
        lines.append(
            "| {name} | {projector_mae:.8f} | {zero_mae:.8f} | {raw_child_delta_mae:.8f} | {projector_reduction_vs_zero:.6f} | {projector_reduction_vs_raw:.6f} |".format(
                name=name,
                **row,
            )
        )
    lines.extend(
        [
            "",
            "## Phasor Carrier Metrics",
            "",
            "| split | angular_mae | parent_angular_mae | angular_reduction_vs_parent | phasor_l1 | parent_phasor_l1 |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    phasor_rows = [("phasor_carrier_all", metrics.get("phasor_carrier_all", {}))]
    phasor_rows.extend((f"phasor_carrier_loo:{k}", v) for k, v in metrics.get("phasor_carrier_leave_one_case", {}).items())
    for name, row in phasor_rows:
        if not row:
            continue
        lines.append(
            "| {name} | {angular_mae:.8f} | {parent_angular_mae:.8f} | {angular_reduction_vs_parent:.6f} | {phasor_l1:.8f} | {parent_phasor_l1:.8f} |".format(
                name=name,
                **row,
            )
        )
    lines.extend(
        [
            "",
            "## Authority Head Metrics",
            "",
            "| split | authority_mae | static_bridge_mae | support_boundary_mae | reduction_vs_static | reduction_vs_support_boundary |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    authority_rows = [("authority_all", metrics.get("authority_all", {}))]
    authority_rows.extend((f"authority_loo:{k}", v) for k, v in metrics.get("authority_leave_one_case", {}).items())
    for name, row in authority_rows:
        if not row:
            continue
        lines.append(
            "| {name} | {authority_mae:.8f} | {static_bridge_mae:.8f} | {support_boundary_proxy_mae:.8f} | {authority_reduction_vs_static_bridge:.6f} | {authority_reduction_vs_support_boundary:.6f} |".format(
                name=name,
                **row,
            )
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            report["interpretation"],
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def _interpret(metrics: dict[str, Any]) -> str:
    all_metrics = metrics["all"]
    zero = float(all_metrics.get("projector_reduction_vs_zero", 0.0))
    raw = float(all_metrics.get("projector_reduction_vs_raw", 0.0))
    tangent = float(all_metrics.get("projector_reduction_vs_parent_tangent", 0.0))
    loo = metrics.get("leave_one_case", {})
    loo_zero = [float(row.get("projector_reduction_vs_zero", 0.0)) for row in loo.values()]
    if zero > 0.25 and raw > 0.05 and tangent > 0.05 and loo_zero and min(loo_zero) > 0.0:
        return "Projector learns a reusable residual direction better than zero/raw/tangent baselines; replay in nested assay is justified."
    if zero > 0.0:
        return "Projector learns some in-sample residual direction, but leave-one-case or baseline margins are weak; replay is exploratory, not proof."
    return "Projector does not beat the zero residual baseline; do not expect ontology conversion from replay."


def main() -> None:
    ap = argparse.ArgumentParser(description="Train a tiny parent phase projector from Circleworld assay NPZ rows.")
    ap.add_argument("inputs", nargs="+", help="NPZ files, directories, or glob patterns.")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--ridge-lambda", type=float, default=1.0e-4)
    ap.add_argument("--min-gate", type=float, default=0.02)
    ap.add_argument("--max-samples-per-file", type=int, default=12000)
    ap.add_argument("--seed", type=int, default=1313)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = _npz_paths([str(item) for item in args.inputs])
    if not paths:
        raise ValueError("No NPZ files matched inputs")
    x, y, w, groups, feature_names = _load_rows(
        paths,
        target_key="target_delta",
        min_gate=float(args.min_gate),
        max_samples_per_file=int(args.max_samples_per_file),
        seed=int(args.seed),
    )
    carrier_x, carrier_y, carrier_w, carrier_groups, carrier_feature_names = _load_rows(
        paths,
        target_key="raw_delta",
        min_gate=float(args.min_gate),
        max_samples_per_file=int(args.max_samples_per_file),
        seed=int(args.seed),
    )
    phasor_x, phasor_y_cos, phasor_y_sin, phasor_w, phasor_groups, phasor_feature_names = _load_phasor_rows(
        paths,
        min_gate=float(args.min_gate),
        max_samples_per_file=int(args.max_samples_per_file),
        seed=int(args.seed),
    )
    authority_x, authority_y, authority_w, authority_groups, authority_feature_names = _load_authority_rows(paths)
    coeffs = _fit_ridge(x, y, w, float(args.ridge_lambda))
    carrier_coeffs = _fit_ridge(carrier_x, carrier_y, carrier_w, float(args.ridge_lambda))
    authority_coeffs = _fit_ridge(authority_x, authority_y, authority_w, float(args.ridge_lambda))
    phasor_cos_coeffs, phasor_sin_coeffs = _fit_phasor_ridge(
        phasor_x,
        phasor_y_cos,
        phasor_y_sin,
        phasor_w,
        float(args.ridge_lambda),
    )
    metrics = {
        "all": _score_split(x, y, w, coeffs),
        "by_case": _case_scores(x, y, w, groups, coeffs),
        "leave_one_case": _leave_one_case_scores(x, y, w, groups, ridge_lambda=float(args.ridge_lambda)),
        "carrier_all": _score_split(carrier_x, carrier_y, carrier_w, carrier_coeffs),
        "carrier_by_case": _case_scores(carrier_x, carrier_y, carrier_w, carrier_groups, carrier_coeffs),
        "carrier_leave_one_case": _leave_one_case_scores(
            carrier_x,
            carrier_y,
            carrier_w,
            carrier_groups,
            ridge_lambda=float(args.ridge_lambda),
        ),
        "phasor_carrier_all": _score_phasor_split(
            phasor_x,
            phasor_y_cos,
            phasor_y_sin,
            phasor_w,
            phasor_cos_coeffs,
            phasor_sin_coeffs,
        ),
        "phasor_carrier_by_case": _phasor_case_scores(
            phasor_x,
            phasor_y_cos,
            phasor_y_sin,
            phasor_w,
            phasor_groups,
            phasor_cos_coeffs,
            phasor_sin_coeffs,
        ),
        "phasor_carrier_leave_one_case": _phasor_leave_one_case_scores(
            phasor_x,
            phasor_y_cos,
            phasor_y_sin,
            phasor_w,
            phasor_groups,
            ridge_lambda=float(args.ridge_lambda),
        ),
        "authority_all": _score_authority_split(authority_x, authority_y, authority_w, authority_coeffs),
        "authority_by_case": _authority_case_scores(authority_x, authority_y, authority_w, authority_groups, authority_coeffs),
        "authority_leave_one_case": _authority_leave_one_case_scores(
            authority_x,
            authority_y,
            authority_w,
            authority_groups,
            ridge_lambda=float(args.ridge_lambda),
        ),
    }
    report = {
        "schema": "circleworld_parent_phase_projector_model.v1",
        "num_files": int(len(paths)),
        "input_files": [str(path) for path in paths],
        "ridge_lambda": float(args.ridge_lambda),
        "min_gate": float(args.min_gate),
        "max_samples_per_file": int(args.max_samples_per_file),
        "feature_names": feature_names,
        "coefficients": [float(item) for item in coeffs.tolist()],
        "carrier_target": "raw_delta",
        "carrier_feature_names": carrier_feature_names,
        "carrier_coefficients": [float(item) for item in carrier_coeffs.tolist()],
        "phasor_carrier_target": "child_phase",
        "phasor_carrier_feature_names": phasor_feature_names,
        "phasor_carrier_cos_coefficients": [float(item) for item in phasor_cos_coeffs.tolist()],
        "phasor_carrier_sin_coefficients": [float(item) for item in phasor_sin_coeffs.tolist()],
        "authority_target": "max(0, oracle_residual_reduction) * boundary_match",
        "authority_feature_names": authority_feature_names,
        "authority_coefficients": [float(item) for item in authority_coeffs.tolist()],
        "metrics": metrics,
        "interpretation": _interpret(metrics),
    }
    model_path = out_dir / "parent_phase_projector_model.json"
    model_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    killswitch = {
        "enable_child_predictive_probe": True,
        "enable_child_predictive_assimilation": True,
        "child_predictive_candidate": "learned_parent_bridge",
        "child_predictive_bind_gain": 1.00,
        "child_predictive_logit_gain": 0.60,
        "child_predictive_gate_floor": 0.05,
        "learned_parent_bridge_delta_source": "trained_linear_projector",
        "learned_parent_bridge_delta_scale": 1.0,
        "learned_parent_bridge_projector_feature_names": feature_names,
        "learned_parent_bridge_projector_coeffs": [float(item) for item in coeffs.tolist()],
        "learned_parent_bridge_carrier_feature_names": carrier_feature_names,
        "learned_parent_bridge_carrier_coeffs": [float(item) for item in carrier_coeffs.tolist()],
        "learned_parent_bridge_phasor_carrier_feature_names": phasor_feature_names,
        "learned_parent_bridge_phasor_carrier_cos_coeffs": [float(item) for item in phasor_cos_coeffs.tolist()],
        "learned_parent_bridge_phasor_carrier_sin_coeffs": [float(item) for item in phasor_sin_coeffs.tolist()],
        "learned_parent_bridge_authority_source": "trained_linear_authority",
        "learned_parent_bridge_authority_feature_names": authority_feature_names,
        "learned_parent_bridge_authority_coeffs": [float(item) for item in authority_coeffs.tolist()],
        "enable_parent_ontology_field": True,
        "parent_ontology_charge_gain": 7.0,
        "parent_ontology_use_in_mode_replace": True,
        "parent_ontology_mode_replace_source": "child_phase",
        "parent_ontology_mode1_logit_boost": 22.0,
        "parent_ontology_mode0_logit_suppress": 14.5,
        "parent_ontology_support_boost": 11.0,
        "parent_ontology_gate_floor": 0.55,
        "learned_parent_bridge_score_threshold": 0.50,
        "learned_parent_bridge_acceptance_target": 0.13,
    }
    killswitch_path = out_dir / "assay_killswitch_trained_projector.json"
    killswitch_path.write_text(json.dumps(killswitch, indent=2), encoding="utf-8")
    carrier_killswitch = dict(killswitch)
    carrier_killswitch["parent_ontology_mode_replace_source"] = "trained_carrier_map"
    carrier_killswitch_path = out_dir / "assay_killswitch_trained_carrier.json"
    carrier_killswitch_path.write_text(json.dumps(carrier_killswitch, indent=2), encoding="utf-8")
    phasor_carrier_killswitch = dict(killswitch)
    phasor_carrier_killswitch["parent_ontology_mode_replace_source"] = "trained_phasor_carrier_map"
    phasor_carrier_killswitch_path = out_dir / "assay_killswitch_trained_phasor_carrier.json"
    phasor_carrier_killswitch_path.write_text(json.dumps(phasor_carrier_killswitch, indent=2), encoding="utf-8")
    carrier_only_killswitch = dict(phasor_carrier_killswitch)
    carrier_only_killswitch["learned_parent_bridge_prewrite_policy"] = "carrier_only"
    carrier_only_killswitch_path = out_dir / "assay_killswitch_trained_phasor_carrier_authority_carrier_only.json"
    carrier_only_killswitch_path.write_text(json.dumps(carrier_only_killswitch, indent=2), encoding="utf-8")
    coeffs_only = {
        "learned_parent_bridge_projector_feature_names": feature_names,
        "learned_parent_bridge_projector_coeffs": [float(item) for item in coeffs.tolist()],
        "learned_parent_bridge_carrier_feature_names": carrier_feature_names,
        "learned_parent_bridge_carrier_coeffs": [float(item) for item in carrier_coeffs.tolist()],
        "learned_parent_bridge_phasor_carrier_feature_names": phasor_feature_names,
        "learned_parent_bridge_phasor_carrier_cos_coeffs": [float(item) for item in phasor_cos_coeffs.tolist()],
        "learned_parent_bridge_phasor_carrier_sin_coeffs": [float(item) for item in phasor_sin_coeffs.tolist()],
        "learned_parent_bridge_authority_source": "trained_linear_authority",
        "learned_parent_bridge_authority_feature_names": authority_feature_names,
        "learned_parent_bridge_authority_coeffs": [float(item) for item in authority_coeffs.tolist()],
    }
    (out_dir / "assay_killswitch_projector_carrier_coeffs_only.json").write_text(json.dumps(coeffs_only, indent=2), encoding="utf-8")
    coeffs_no_authority = dict(coeffs_only)
    coeffs_no_authority.pop("learned_parent_bridge_authority_source", None)
    (out_dir / "assay_killswitch_projector_carrier_coeffs_no_authority.json").write_text(json.dumps(coeffs_no_authority, indent=2), encoding="utf-8")
    _write_markdown(report, out_dir / "PARENT_PHASE_PROJECTOR_SCOUT.md")
    print(
        json.dumps(
            {
                "model_path": str(model_path),
                "killswitch_path": str(killswitch_path),
                "carrier_killswitch_path": str(carrier_killswitch_path),
                "phasor_carrier_killswitch_path": str(phasor_carrier_killswitch_path),
                "carrier_only_killswitch_path": str(carrier_only_killswitch_path),
                "metrics": metrics,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
