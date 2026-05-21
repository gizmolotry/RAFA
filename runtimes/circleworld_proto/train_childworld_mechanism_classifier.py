from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


SCHEMA = "circleworld_childworld_mechanism_classifier_v1"

FEATURE_KEYS: tuple[str, ...] = (
    "continuation_fraction",
    "mode_replace_fraction",
    "readout_fraction",
    "continuation_write_enabled",
    "continuation_write_delta",
    "branch_identity_qualified_carry",
    "branch_identity_budget_retained",
    "fine_q_profile_corr",
    "readout_sibling_response",
    "world_jump_penalty",
    "mode_replace_readiness",
    "mode_replace_enabled",
    "mode_replace_gate",
    "mode_replace_ratio",
    "mode_replace_static_record",
    "direct_readout_dependency",
    "final_direct_mix_shortcut_score",
    "child_record_survival_score",
    "parent_mode_conversion_sibling_fraction",
    "mode_replace_conversion_score",
)

TARGET_KEYS: tuple[str, ...] = (
    "target_direct_carrier",
    "target_writeback_sufficient",
    "target_phase_specific",
    "target_child_record_required",
    "target_mode_replace_supported",
)


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return payload


def _num(value: Any) -> float | None:
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return float(value)
    if isinstance(value, str):
        try:
            parsed = float(value)
        except ValueError:
            return None
        return parsed if math.isfinite(parsed) else None
    return None


def _bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "1", "yes"}:
            return True
        if lowered in {"false", "0", "no"}:
            return False
    return None


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -50.0, 50.0)))


def _feature_matrix(cases: list[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray]:
    raw = np.zeros((len(cases), len(FEATURE_KEYS)), dtype=np.float64)
    mask = np.zeros_like(raw)
    for row_idx, case in enumerate(cases):
        for col_idx, key in enumerate(FEATURE_KEYS):
            value = _num(case.get(key))
            if value is not None:
                raw[row_idx, col_idx] = value
                mask[row_idx, col_idx] = 1.0
    means = np.divide(raw.sum(axis=0), np.maximum(mask.sum(axis=0), 1.0))
    filled = np.where(mask > 0.0, raw, means[None, :])
    return filled, mask


def _standardize(x: np.ndarray) -> tuple[np.ndarray, dict[str, list[float]]]:
    mean = x.mean(axis=0)
    std = x.std(axis=0)
    std = np.where(std < 1.0e-8, 1.0, std)
    return (x - mean[None, :]) / std[None, :], {
        "mean": mean.tolist(),
        "std": std.tolist(),
    }


def _fit_logistic(
    x: np.ndarray,
    y: np.ndarray,
    *,
    epochs: int,
    lr: float,
    l2: float,
) -> tuple[np.ndarray, float, list[float]]:
    w = np.zeros((x.shape[1],), dtype=np.float64)
    b = 0.0
    losses: list[float] = []
    for _ in range(max(1, int(epochs))):
        pred = _sigmoid(x @ w + b)
        eps = 1.0e-8
        loss = float(-(y * np.log(pred + eps) + (1.0 - y) * np.log(1.0 - pred + eps)).mean() + l2 * np.dot(w, w))
        grad = pred - y
        grad_w = (x.T @ grad) / max(1, len(y)) + 2.0 * l2 * w
        grad_b = float(grad.mean())
        w -= float(lr) * grad_w
        b -= float(lr) * grad_b
        losses.append(loss)
    return w, b, losses


def _metrics(y: np.ndarray, prob: np.ndarray) -> dict[str, Any]:
    pred = (prob >= 0.5).astype(np.float64)
    tp = float(((pred == 1.0) & (y == 1.0)).sum())
    tn = float(((pred == 0.0) & (y == 0.0)).sum())
    fp = float(((pred == 1.0) & (y == 0.0)).sum())
    fn = float(((pred == 0.0) & (y == 1.0)).sum())
    accuracy = float((tp + tn) / max(1.0, tp + tn + fp + fn))
    precision = float(tp / max(1.0, tp + fp))
    recall = float(tp / max(1.0, tp + fn))
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "positive_count": float(y.sum()),
        "negative_count": float((1.0 - y).sum()),
        "mean_positive_probability": float(prob[y == 1.0].mean()) if bool((y == 1.0).any()) else None,
        "mean_negative_probability": float(prob[y == 0.0].mean()) if bool((y == 0.0).any()) else None,
    }


def _top_weights(w: np.ndarray, limit: int = 8) -> list[dict[str, Any]]:
    order = np.argsort(-np.abs(w))[:limit]
    return [
        {
            "feature": FEATURE_KEYS[int(idx)],
            "weight": float(w[int(idx)]),
            "abs_weight": float(abs(w[int(idx)])),
        }
        for idx in order
    ]


def train(dataset_path: Path, *, epochs: int, lr: float, l2: float) -> dict[str, Any]:
    dataset = _load_json(dataset_path)
    cases = [case for case in dataset.get("cases", []) if isinstance(case, dict)]
    x_raw, present_mask = _feature_matrix(cases)
    x, scaler = _standardize(x_raw)

    target_results: dict[str, Any] = {}
    for target in TARGET_KEYS:
        idxs: list[int] = []
        ys: list[float] = []
        for idx, case in enumerate(cases):
            label = _bool(case.get(target))
            if label is None:
                continue
            idxs.append(idx)
            ys.append(1.0 if label else 0.0)
        if len(set(ys)) < 2:
            target_results[target] = {
                "status": "skipped_single_class_or_empty",
                "example_count": len(ys),
                "positive_count": float(sum(ys)),
            }
            continue
        x_t = x[np.array(idxs, dtype=np.int64)]
        y_t = np.array(ys, dtype=np.float64)
        w, b, losses = _fit_logistic(x_t, y_t, epochs=epochs, lr=lr, l2=l2)
        prob = _sigmoid(x_t @ w + b)
        target_results[target] = {
            "status": "trained",
            "example_count": int(len(y_t)),
            "metrics": _metrics(y_t, prob),
            "bias": float(b),
            "top_weights": _top_weights(w),
            "final_loss": float(losses[-1]),
            "initial_loss": float(losses[0]),
        }

    return {
        "schema": SCHEMA,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_path": str(dataset_path),
        "dataset_schema": dataset.get("schema"),
        "case_count": len(cases),
        "feature_keys": list(FEATURE_KEYS),
        "target_keys": list(TARGET_KEYS),
        "feature_present_fraction": {
            key: float(present_mask[:, idx].mean()) if present_mask.size else 0.0
            for idx, key in enumerate(FEATURE_KEYS)
        },
        "scaler": scaler,
        "fit": {
            "epochs": int(epochs),
            "lr": float(lr),
            "l2": float(l2),
            "model": "artifact_logistic_baseline",
        },
        "targets": target_results,
        "notes": [
            "This is an artifact-level classifier over summary metrics, not the final tensor replay branch-law model.",
            "Use this to validate label separability and target plumbing before replaying branch-law feature tensors.",
        ],
    }


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Childworld Mechanism Classifier",
        "",
        f"- schema: `{report.get('schema')}`",
        f"- dataset: `{report.get('dataset_path')}`",
        f"- case_count: `{report.get('case_count')}`",
        "",
        "## Targets",
        "",
        "| target | status | n | acc | precision | recall | top weights |",
        "| --- | --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for target, payload in report.get("targets", {}).items():
        if not isinstance(payload, dict):
            continue
        metrics = payload.get("metrics") if isinstance(payload.get("metrics"), dict) else {}
        weights = ", ".join(
            f"{item['feature']}={item['weight']:.3g}"
            for item in payload.get("top_weights", [])[:4]
            if isinstance(item, dict)
        )
        lines.append(
            "| "
            + " | ".join(
                [
                    str(target),
                    str(payload.get("status")),
                    str(payload.get("example_count", "")),
                    "" if metrics.get("accuracy") is None else f"{float(metrics.get('accuracy')):.3f}",
                    "" if metrics.get("precision") is None else f"{float(metrics.get('precision')):.3f}",
                    "" if metrics.get("recall") is None else f"{float(metrics.get('recall')):.3f}",
                    weights,
                ]
            )
            + " |"
        )
    return "\n".join(lines).rstrip() + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Train an artifact-level childworld mechanism classifier.")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md", default=None)
    parser.add_argument("--epochs", type=int, default=800)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--l2", type=float, default=1.0e-3)
    args = parser.parse_args()

    report = train(Path(args.dataset), epochs=int(args.epochs), lr=float(args.lr), l2=float(args.l2))
    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    if args.output_md:
        output_md = Path(args.output_md)
        output_md.parent.mkdir(parents=True, exist_ok=True)
        output_md.write_text(render_markdown(report), encoding="utf-8")
    print(json.dumps({"schema": report["schema"], "case_count": report["case_count"], "targets": {k: v.get("status") for k, v in report["targets"].items()}}, indent=2))


if __name__ == "__main__":
    main()
