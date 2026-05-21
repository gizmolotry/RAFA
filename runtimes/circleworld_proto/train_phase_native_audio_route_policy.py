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
from profile_registry import case_group as _case_group  # noqa: E402
from profile_registry import route_dict as _route_dict  # noqa: E402
from profile_registry import route_from_id as _route_from_id  # noqa: E402
from profile_registry import route_id as _route_id  # noqa: E402
from profile_registry import route_key_from_dict as _route_key_from_dict  # noqa: E402


OUTPUT_JSON = "phase_native_audio_learned_route_policy.json"
OUTPUT_MD = "PHASE_NATIVE_AUDIO_LEARNED_ROUTE_POLICY.md"


def _json_load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _family_routes(oracle_json: Path) -> dict[str, tuple[str, str, str, float]]:
    oracle = _json_load(oracle_json)
    routes: dict[str, tuple[str, str, str, float]] = {}
    for group, choice in oracle.get("family_choices", {}).items():
        key = choice.get("key", {})
        if key:
            routes[str(group)] = _route_key_from_dict(key)
    if not routes:
        raise ValueError(f"No family route choices found in {oracle_json}")
    return routes


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


def _training_rows(
    train_oracle_jsons: Sequence[Path],
    train_delta_jsons: Sequence[Path],
    keys: Sequence[str],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for oracle_json, delta_json in zip(train_oracle_jsons, train_delta_jsons):
        routes = _family_routes(oracle_json)
        features = _case_features([delta_json])
        for case_name, row in sorted(features.items()):
            group = _case_group(case_name)
            if group not in routes:
                continue
            route = routes[group]
            rows.append(
                {
                    "case_name": case_name,
                    "group": group,
                    "oracle_json": str(oracle_json),
                    "delta_json": str(delta_json),
                    "route_id": _route_id(route),
                    "route": _route_dict(route),
                    "vector": _vector(row, keys),
                }
            )
    return rows


def _fit_route_centroids(training_rows: Sequence[dict[str, Any]], means: Sequence[float], stds: Sequence[float]) -> dict[str, list[float]]:
    by_route: dict[str, list[list[float]]] = defaultdict(list)
    for row in training_rows:
        by_route[str(row["route_id"])].append(_scale(row["vector"], means, stds))
    return {route_id: _centroid(vectors) for route_id, vectors in sorted(by_route.items())}


def _predict_centroid_route(
    vector: Sequence[float],
    *,
    means: Sequence[float],
    stds: Sequence[float],
    centroids: dict[str, list[float]],
) -> tuple[str, float]:
    scaled = _scale(vector, means, stds)
    route_id = min(centroids, key=lambda rid: _distance(scaled, centroids[rid]))
    return route_id, _distance(scaled, centroids[route_id])


def _predict_knn_route(
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
        )
        for row in training_rows
    ]
    scored.sort(key=lambda item: item[0])
    nearest = scored[: max(1, min(int(k), len(scored)))]
    votes: dict[str, float] = defaultdict(float)
    for distance, route_id in nearest:
        votes[route_id] += 1.0 / max(distance, 1.0e-6)
    route_id = max(votes, key=votes.get)
    mean_distance = sum(distance for distance, _ in nearest) / float(len(nearest))
    return route_id, mean_distance


def _leave_one_training_accuracy(training_rows: Sequence[dict[str, Any]], keys: Sequence[str]) -> dict[str, Any]:
    correct = 0
    total = 0
    by_group: dict[str, list[int]] = defaultdict(list)
    for idx, held in enumerate(training_rows):
        train = [row for j, row in enumerate(training_rows) if j != idx]
        if not train:
            continue
        means, stds = _fit_scaler([row["vector"] for row in train])
        centroids = _fit_route_centroids(train, means, stds)
        pred, _ = _predict_centroid_route(held["vector"], means=means, stds=stds, centroids=centroids)
        ok = int(pred == held["route_id"])
        correct += ok
        total += 1
        by_group[str(held["group"])].append(ok)
    return {
        "total": total,
        "route_accuracy": correct / float(total) if total else 0.0,
        "group_route_accuracy": {
            group: sum(vals) / float(len(vals)) for group, vals in sorted(by_group.items()) if vals
        },
        "feature_count": len(keys),
    }


def _markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# Phase-Native Audio Learned Route Policy",
        "",
        "This policy predicts phase-only continuation routes from no-future",
        "Circleworld/prefix metadata using nearest-centroid route labels.",
        "",
        "## Summary",
        "",
        f"- Status: `{summary['status']}`",
        f"- Model: `{summary['model']}`",
        f"- Train cases: `{summary['train_case_count']}`",
        f"- Target cases: `{summary['target_case_count']}`",
        f"- Feature count: `{summary['feature_count']}`",
        f"- Leave-one training route accuracy: `{summary['training_diagnostics']['route_accuracy']:.6f}`",
        "",
        "## Predicted Route Counts",
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


def train_route_policy(
    *,
    train_oracle_jsons: Sequence[Path],
    train_delta_jsons: Sequence[Path],
    target_delta_json: Path,
    out_dir: Path,
    model: str,
) -> dict[str, Any]:
    if len(train_oracle_jsons) != len(train_delta_jsons):
        raise ValueError("--train-oracle-json and --train-delta-json counts must match")
    train_feature_sets = [_case_features([path]) for path in train_delta_jsons]
    target_features = _case_features([target_delta_json])
    keys = _feature_keys([*train_feature_sets, target_features])
    training_rows = _training_rows(train_oracle_jsons, train_delta_jsons, keys)
    if not training_rows:
        raise RuntimeError("No labeled training rows were produced")
    means, stds = _fit_scaler([row["vector"] for row in training_rows])
    centroids = _fit_route_centroids(training_rows, means, stds)
    fallback_route = Counter(str(row["route_id"]) for row in training_rows).most_common(1)[0][0]

    case_routes: dict[str, dict[str, Any]] = {}
    predictions: list[dict[str, Any]] = []
    for case_name, row in sorted(target_features.items()):
        vec = _vector(row, keys)
        if model == "nearest_centroid_route_v1":
            route_id, distance = _predict_centroid_route(vec, means=means, stds=stds, centroids=centroids)
        elif model == "nearest_case_route_v1":
            route_id, distance = _predict_knn_route(vec, means=means, stds=stds, training_rows=training_rows, k=1)
        elif model == "knn5_route_vote_v1":
            route_id, distance = _predict_knn_route(vec, means=means, stds=stds, training_rows=training_rows, k=5)
        else:
            raise ValueError(f"Unknown route policy model: {model}")
        route = _route_from_id(route_id)
        case_routes[case_name] = _route_dict(route)
        predictions.append(
            {
                "case_name": case_name,
                "group": _case_group(case_name),
                "route_id": route_id,
                "route": _route_dict(route),
                "centroid_distance": distance,
            }
        )

    route_counts = Counter(pred["route_id"] for pred in predictions)
    diagnostics = _leave_one_training_accuracy(training_rows, keys)
    summary = {
        "schema": "phase_native_audio_learned_route_policy_v1",
        "status": "learned_route_policy_ready",
        "model": model,
        "train_oracle_jsons": [str(path) for path in train_oracle_jsons],
        "train_delta_jsons": [str(path) for path in train_delta_jsons],
        "target_delta_json": str(target_delta_json),
        "feature_count": len(keys),
        "feature_keys": keys,
        "train_case_count": len(training_rows),
        "target_case_count": len(predictions),
        "training_route_counts": dict(sorted(Counter(str(row["route_id"]) for row in training_rows).items())),
        "predicted_route_counts": dict(sorted(route_counts.items())),
        "fallback_route": _route_dict(_route_from_id(fallback_route)),
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
        "future_target_audio_used_for_route_training_labels_only": True,
        "target_future_audio_used_for_route_selection": False,
        "target_future_metrics_used_for_route_selection": False,
        "interpretation": (
            "The policy is ready for live selected-route rendering. Training labels come from source-lockbox "
            "family oracle tables; target route selection uses only target prefix/Circleworld metadata features."
        ),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / OUTPUT_JSON).write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (out_dir / OUTPUT_MD).write_text(_markdown(summary), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a no-future prefix route policy for phase-native audio.")
    parser.add_argument("--train-oracle-json", required=True, action="append", type=Path)
    parser.add_argument("--train-delta-json", required=True, action="append", type=Path)
    parser.add_argument("--target-delta-json", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument(
        "--model",
        choices=("nearest_centroid_route_v1", "nearest_case_route_v1", "knn5_route_vote_v1"),
        default="nearest_centroid_route_v1",
    )
    args = parser.parse_args()
    summary = train_route_policy(
        train_oracle_jsons=args.train_oracle_json,
        train_delta_jsons=args.train_delta_json,
        target_delta_json=args.target_delta_json,
        out_dir=args.out_dir,
        model=str(args.model),
    )
    print(
        json.dumps(
            {
                "status": summary["status"],
                "json": str(Path(args.out_dir) / OUTPUT_JSON),
                "markdown": str(Path(args.out_dir) / OUTPUT_MD),
                "train_case_count": summary["train_case_count"],
                "target_case_count": summary["target_case_count"],
                "route_accuracy": summary["training_diagnostics"]["route_accuracy"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
