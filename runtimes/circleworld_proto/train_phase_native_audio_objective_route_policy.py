from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Sequence

ROOT = Path(__file__).resolve().parents[2]
RUNTIME = ROOT / "runtimes" / "circleworld_proto"
for path in (ROOT, RUNTIME):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from score_phase_native_audio_prefix_router_scout import _case_features  # noqa: E402
from score_phase_native_audio_reentry_guard import _as_float, _key  # noqa: E402
from score_phase_native_audio_target_replay_oracle import _collect_rows, _row_objective  # noqa: E402
from profile_registry import case_group as _case_group  # noqa: E402
from profile_registry import route_dict as _route_dict  # noqa: E402
from profile_registry import route_from_id as _route_from_id  # noqa: E402
from profile_registry import route_id as _route_id  # noqa: E402


OUTPUT_JSON = "phase_native_audio_objective_route_policy.json"
OUTPUT_MD = "PHASE_NATIVE_AUDIO_OBJECTIVE_ROUTE_POLICY.md"
OBJECTIVE_ROUTE_MODELS = (
    "objective_centroid_v1",
    "objective_knn1_v1",
    "objective_knn5_v1",
    "objective_mlp_v1",
)


def _parse_train_set(raw: str) -> tuple[Path, list[Path]]:
    if "::" not in raw:
        raise ValueError("--train-set must be formatted suite_json::delta_json1,delta_json2")
    suite_raw, delta_raw = raw.split("::", 1)
    deltas = [Path(item.strip()) for item in delta_raw.split(",") if item.strip()]
    if not deltas:
        raise ValueError("--train-set needs at least one delta JSON")
    return Path(suite_raw.strip()), deltas


def _feature_keys(feature_sets: Sequence[dict[str, dict[str, float]]]) -> list[str]:
    keys: set[str] = set()
    for features in feature_sets:
        for row in features.values():
            keys.update(row)
    return sorted(keys)


def _vector(row: dict[str, float], keys: Sequence[str]) -> list[float]:
    return [float(row.get(key, 0.0)) for key in keys]


def _fit_scaler(vectors: Sequence[Sequence[float]]) -> tuple[list[float], list[float]]:
    if not vectors:
        return [], []
    dims = len(vectors[0])
    means: list[float] = []
    stds: list[float] = []
    for idx in range(dims):
        vals = [float(vec[idx]) for vec in vectors if math.isfinite(float(vec[idx]))]
        if not vals:
            means.append(0.0)
            stds.append(1.0)
            continue
        mean = sum(vals) / float(len(vals))
        var = sum((val - mean) ** 2 for val in vals) / float(len(vals))
        means.append(mean)
        stds.append(math.sqrt(var) if var > 1.0e-18 else 1.0)
    return means, stds


def _scale(vec: Sequence[float], means: Sequence[float], stds: Sequence[float]) -> list[float]:
    return [(float(value) - means[idx]) / stds[idx] for idx, value in enumerate(vec)]


def _centroid(vectors: Sequence[Sequence[float]]) -> list[float]:
    if not vectors:
        return []
    dims = len(vectors[0])
    return [sum(float(vec[idx]) for vec in vectors) / float(len(vectors)) for idx in range(dims)]


def _distance(a: Sequence[float], b: Sequence[float]) -> float:
    return math.sqrt(sum((float(av) - float(bv)) ** 2 for av, bv in zip(a, b)))


def _best_rows_from_train_set(suite_json: Path, delta_jsons: Sequence[Path], *, margin: float) -> dict[str, dict[str, Any]]:
    _, rows = _collect_rows(suite_json, delta_jsons, margin)
    by_case: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if abs(_as_float(row.get("gain"))) <= 1.0e-12:
            continue
        by_case[str(row["case_name"])].append(row)
    return {case: max(case_rows, key=_row_objective) for case, case_rows in by_case.items() if case_rows}


def _training_rows(
    train_sets: Sequence[tuple[Path, list[Path]]],
    keys: Sequence[str],
    *,
    margin: float,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for suite_json, delta_jsons in train_sets:
        best_by_case = _best_rows_from_train_set(suite_json, delta_jsons, margin=margin)
        features = _case_features([delta_jsons[0]])
        for case_name, best in sorted(best_by_case.items()):
            if case_name not in features:
                continue
            route = _key(best)
            rows.append(
                {
                    "case_name": case_name,
                    "group": _case_group(case_name),
                    "suite_json": str(suite_json),
                    "feature_delta_json": str(delta_jsons[0]),
                    "route_id": _route_id(route),
                    "route": _route_dict(route),
                    "row_objective": _row_objective(best),
                    "label_metrics": {
                        "corr_delta_vs_copy_last": best.get("corr_delta_vs_copy_last"),
                        "mse_delta_vs_copy_last": best.get("mse_delta_vs_copy_last"),
                        "loop_delta_vs_copy_last": best.get("loop_delta_vs_copy_last"),
                        "harmful_replay_excess_delta_vs_copy_last": best.get(
                            "harmful_replay_excess_delta_vs_copy_last"
                        ),
                        "corr_delta_vs_gain0": best.get("corr_delta_vs_gain0"),
                    },
                    "vector": _vector(features[case_name], keys),
                }
            )
    return rows


def _fit_route_centroids(training_rows: Sequence[dict[str, Any]], means: Sequence[float], stds: Sequence[float]) -> dict[str, list[float]]:
    by_route: dict[str, list[list[float]]] = defaultdict(list)
    for row in training_rows:
        by_route[str(row["route_id"])].append(_scale(row["vector"], means, stds))
    return {route_id: _centroid(vectors) for route_id, vectors in sorted(by_route.items())}


def _predict_centroid(
    vector: Sequence[float],
    *,
    means: Sequence[float],
    stds: Sequence[float],
    centroids: dict[str, list[float]],
) -> tuple[str, float]:
    scaled = _scale(vector, means, stds)
    route_id = min(centroids, key=lambda rid: _distance(scaled, centroids[rid]))
    return route_id, _distance(scaled, centroids[route_id])


def _predict_knn(
    vector: Sequence[float],
    *,
    means: Sequence[float],
    stds: Sequence[float],
    training_rows: Sequence[dict[str, Any]],
    k: int,
) -> tuple[str, float]:
    scaled = _scale(vector, means, stds)
    scored = [
        (
            _distance(scaled, _scale(row["vector"], means, stds)),
            str(row["route_id"]),
            _as_float(row.get("row_objective")),
        )
        for row in training_rows
    ]
    scored.sort(key=lambda item: item[0])
    nearest = scored[: max(1, min(int(k), len(scored)))]
    votes: dict[str, float] = defaultdict(float)
    for distance, route_id, objective in nearest:
        votes[route_id] += (1.0 + max(objective, 0.0)) / max(distance, 1.0e-6)
    route_id = max(votes, key=votes.get)
    return route_id, sum(distance for distance, _, _ in nearest) / float(len(nearest))


def _fit_mlp_route_classifier(
    training_rows: Sequence[dict[str, Any]],
    *,
    means: Sequence[float],
    stds: Sequence[float],
    fallback_route_id: str,
) -> dict[str, Any]:
    route_ids = sorted({str(row["route_id"]) for row in training_rows})
    diagnostics: dict[str, Any] = {
        "model_family": "mlp",
        "class_count": len(route_ids),
        "epochs": 0,
        "final_loss": None,
        "training_accuracy": 0.0,
        "route_label_accuracy": 0.0,
        "fallback_to_most_common_source_route": False,
        "fallback_reason": None,
        "hidden_dim": 0,
        "seed": 1701,
    }
    if len(route_ids) <= 1:
        diagnostics.update(
            {
                "fallback_to_most_common_source_route": True,
                "fallback_reason": "single_class_training_set",
                "training_accuracy": 1.0 if training_rows else 0.0,
                "route_label_accuracy": 1.0 if training_rows else 0.0,
            }
        )
        return {"mode": "fallback", "fallback_route_id": fallback_route_id, "diagnostics": diagnostics}
    input_dim = len(means)
    if input_dim <= 0:
        diagnostics.update(
            {
                "fallback_to_most_common_source_route": True,
                "fallback_reason": "empty_feature_vector",
            }
        )
        return {"mode": "fallback", "fallback_route_id": fallback_route_id, "diagnostics": diagnostics}

    try:
        import torch
    except ImportError as exc:
        diagnostics.update(
            {
                "fallback_to_most_common_source_route": True,
                "fallback_reason": f"torch_import_failed:{exc.__class__.__name__}",
            }
        )
        return {"mode": "fallback", "fallback_route_id": fallback_route_id, "diagnostics": diagnostics}

    torch.manual_seed(int(diagnostics["seed"]))
    try:
        torch.use_deterministic_algorithms(True)
    except Exception:
        diagnostics["deterministic_algorithms_unavailable"] = True

    class_index = {route_id: idx for idx, route_id in enumerate(route_ids)}
    x = torch.tensor(
        [_scale(row["vector"], means, stds) for row in training_rows],
        dtype=torch.float32,
    )
    y = torch.tensor([class_index[str(row["route_id"])] for row in training_rows], dtype=torch.long)
    if not bool(torch.isfinite(x).all()):
        diagnostics.update(
            {
                "fallback_to_most_common_source_route": True,
                "fallback_reason": "nonfinite_training_features",
            }
        )
        return {"mode": "fallback", "fallback_route_id": fallback_route_id, "diagnostics": diagnostics}

    hidden_dim = min(16, max(4, len(route_ids) * 2, input_dim // 2))
    epochs = 96
    mlp = torch.nn.Sequential(
        torch.nn.Linear(input_dim, hidden_dim),
        torch.nn.Tanh(),
        torch.nn.Linear(hidden_dim, len(route_ids)),
    )
    optimizer = torch.optim.AdamW(mlp.parameters(), lr=0.03, weight_decay=1.0e-4)
    final_loss = float("nan")
    for _ in range(epochs):
        optimizer.zero_grad(set_to_none=True)
        logits = mlp(x)
        loss = torch.nn.functional.cross_entropy(logits, y)
        if not bool(torch.isfinite(loss)):
            diagnostics.update(
                {
                    "fallback_to_most_common_source_route": True,
                    "fallback_reason": "nonfinite_training_loss",
                    "epochs": _,
                }
            )
            return {"mode": "fallback", "fallback_route_id": fallback_route_id, "diagnostics": diagnostics}
        loss.backward()
        optimizer.step()
        final_loss = float(loss.detach().cpu())

    with torch.no_grad():
        logits = mlp(x)
        predicted = torch.argmax(logits, dim=1)
        accuracy = float((predicted == y).to(torch.float32).mean().cpu())

    diagnostics.update(
        {
            "epochs": epochs,
            "final_loss": final_loss,
            "training_accuracy": accuracy,
            "route_label_accuracy": accuracy,
            "hidden_dim": hidden_dim,
            "route_ids": route_ids,
        }
    )
    return {
        "mode": "mlp",
        "model": mlp,
        "route_ids": route_ids,
        "diagnostics": diagnostics,
        "torch": torch,
    }


def _predict_mlp(
    vector: Sequence[float],
    *,
    means: Sequence[float],
    stds: Sequence[float],
    mlp_state: dict[str, Any],
) -> tuple[str, float]:
    if mlp_state.get("mode") != "mlp":
        return str(mlp_state["fallback_route_id"]), 1.0
    torch = mlp_state["torch"]
    model = mlp_state["model"]
    model.eval()
    with torch.no_grad():
        x = torch.tensor([_scale(vector, means, stds)], dtype=torch.float32)
        logits = model(x)
        probs = torch.softmax(logits, dim=1)[0]
        class_idx = int(torch.argmax(probs).cpu())
        confidence = float(probs[class_idx].cpu())
    route_ids = mlp_state["route_ids"]
    if class_idx < 0 or class_idx >= len(route_ids):
        return str(mlp_state["fallback_route_id"]), 1.0
    return str(route_ids[class_idx]), 1.0 - confidence


def _predict(
    vector: Sequence[float],
    *,
    model: str,
    means: Sequence[float],
    stds: Sequence[float],
    training_rows: Sequence[dict[str, Any]],
    centroids: dict[str, list[float]],
    mlp_state: dict[str, Any] | None = None,
) -> tuple[str, float]:
    if model == "objective_centroid_v1":
        return _predict_centroid(vector, means=means, stds=stds, centroids=centroids)
    if model == "objective_knn1_v1":
        return _predict_knn(vector, means=means, stds=stds, training_rows=training_rows, k=1)
    if model == "objective_knn5_v1":
        return _predict_knn(vector, means=means, stds=stds, training_rows=training_rows, k=5)
    if model == "objective_mlp_v1":
        if mlp_state is None:
            raise ValueError("objective_mlp_v1 requires a trained mlp_state")
        return _predict_mlp(vector, means=means, stds=stds, mlp_state=mlp_state)
    raise ValueError(f"Unknown objective route model: {model}")


def _leave_one_accuracy(training_rows: Sequence[dict[str, Any]], *, model: str) -> dict[str, Any]:
    correct = 0
    total = 0
    for idx, held in enumerate(training_rows):
        train = [row for j, row in enumerate(training_rows) if j != idx]
        if not train:
            continue
        means, stds = _fit_scaler([row["vector"] for row in train])
        centroids = _fit_route_centroids(train, means, stds)
        pred, _ = _predict(
            held["vector"],
            model=model,
            means=means,
            stds=stds,
            training_rows=train,
            centroids=centroids,
        )
        correct += int(pred == held["route_id"])
        total += 1
    return {
        "total": total,
        "route_label_accuracy": correct / float(total) if total else 0.0,
    }


def _markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# Phase-Native Audio Objective Route Policy",
        "",
        "This policy derives route labels from source-lockbox route outcomes, then",
        "predicts target routes from no-future Circleworld/prefix metadata.",
        "",
        "## Summary",
        "",
        f"- Status: `{summary['status']}`",
        f"- Model: `{summary['model']}`",
        f"- Training cases: `{summary['train_case_count']}`",
        f"- Target cases: `{summary['target_case_count']}`",
        f"- Feature count: `{summary['feature_count']}`",
        f"- Leave-one route-label accuracy: `{summary['training_diagnostics']['route_label_accuracy']:.6f}`",
        "",
        "## Predicted Routes",
        "",
        "```json",
        json.dumps(summary["predicted_route_counts"], indent=2),
        "```",
        "",
        "## Interpretation",
        "",
        summary["interpretation"],
    ]
    return "\n".join(lines) + "\n"


def train_objective_route_policy(
    *,
    train_sets: Sequence[tuple[Path, list[Path]]],
    target_delta_json: Path,
    out_dir: Path,
    model: str,
    margin: float,
) -> dict[str, Any]:
    train_feature_sets = [_case_features([delta_jsons[0]]) for _, delta_jsons in train_sets]
    target_features = _case_features([target_delta_json])
    keys = _feature_keys([*train_feature_sets, target_features])
    training_rows = _training_rows(train_sets, keys, margin=margin)
    if not training_rows:
        raise RuntimeError("No objective-labeled training rows produced")
    means, stds = _fit_scaler([row["vector"] for row in training_rows])
    centroids = _fit_route_centroids(training_rows, means, stds)
    fallback_route_id = Counter(str(row["route_id"]) for row in training_rows).most_common(1)[0][0]
    mlp_state = (
        _fit_mlp_route_classifier(
            training_rows,
            means=means,
            stds=stds,
            fallback_route_id=fallback_route_id,
        )
        if model == "objective_mlp_v1"
        else None
    )

    case_routes: dict[str, dict[str, Any]] = {}
    predictions: list[dict[str, Any]] = []
    fallback_prediction_count = 0
    for case_name, features in sorted(target_features.items()):
        route_id, distance = _predict(
            _vector(features, keys),
            model=model,
            means=means,
            stds=stds,
            training_rows=training_rows,
            centroids=centroids,
            mlp_state=mlp_state,
        )
        if route_id not in centroids:
            route_id = fallback_route_id
            fallback_prediction_count += 1
        route = _route_from_id(route_id)
        case_routes[case_name] = _route_dict(route)
        predictions.append(
            {
                "case_name": case_name,
                "group": _case_group(case_name),
                "route_id": route_id,
                "route": _route_dict(route),
                "distance": distance,
            }
        )

    if mlp_state is not None and mlp_state.get("mode") != "mlp":
        fallback_prediction_count = len(predictions)
    diagnostics = (
        {**mlp_state["diagnostics"], "total": len(training_rows), "fallback_prediction_count": fallback_prediction_count}
        if mlp_state is not None
        else _leave_one_accuracy(training_rows, model=model)
    )
    summary = {
        "schema": "phase_native_audio_objective_route_policy_v1",
        "status": "objective_route_policy_ready",
        "model": model,
        "train_sets": [
            {
                "suite_json": str(suite_json),
                "delta_jsons": [str(path) for path in delta_jsons],
            }
            for suite_json, delta_jsons in train_sets
        ],
        "target_delta_json": str(target_delta_json),
        "target_margin": float(margin),
        "feature_count": len(keys),
        "feature_keys": keys,
        "train_case_count": len(training_rows),
        "target_case_count": len(predictions),
        "training_route_counts": dict(sorted(Counter(str(row["route_id"]) for row in training_rows).items())),
        "predicted_route_counts": dict(sorted(Counter(str(pred["route_id"]) for pred in predictions).items())),
        "fallback_route": _route_dict(_route_from_id(fallback_route_id)),
        "case_routes": case_routes,
        "predictions": predictions,
        "route_centroids": {
            route_id: {
                "route": _route_dict(_route_from_id(route_id)),
                "centroid": centroid,
            }
            for route_id, centroid in sorted(centroids.items())
        },
        "scaler": {
            "means": means,
            "stds": stds,
        },
        "training_diagnostics": diagnostics,
        "source_future_metrics_used_for_route_training_labels": True,
        "target_future_audio_used_for_route_selection": False,
        "target_future_metrics_used_for_route_selection": False,
        "interpretation": (
            "The policy is ready for live selected-route rendering. Source lockbox outcomes define labels; "
            "target route selection uses only no-future Circleworld/prefix metadata."
        ),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / OUTPUT_JSON).write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (out_dir / OUTPUT_MD).write_text(_markdown(summary), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Train objective-aware phase-native route policy.")
    parser.add_argument("--train-set", required=True, action="append", help="suite_json::delta_json1,delta_json2,...")
    parser.add_argument("--target-delta-json", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument(
        "--model",
        choices=OBJECTIVE_ROUTE_MODELS,
        default="objective_knn5_v1",
    )
    parser.add_argument("--target-margin", type=float, default=0.01)
    args = parser.parse_args()
    summary = train_objective_route_policy(
        train_sets=[_parse_train_set(raw) for raw in args.train_set],
        target_delta_json=args.target_delta_json,
        out_dir=args.out_dir,
        model=str(args.model),
        margin=float(args.target_margin),
    )
    print(
        json.dumps(
            {
                "status": summary["status"],
                "json": str(Path(args.out_dir) / OUTPUT_JSON),
                "markdown": str(Path(args.out_dir) / OUTPUT_MD),
                "train_case_count": summary["train_case_count"],
                "target_case_count": summary["target_case_count"],
                "route_label_accuracy": summary["training_diagnostics"]["route_label_accuracy"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
