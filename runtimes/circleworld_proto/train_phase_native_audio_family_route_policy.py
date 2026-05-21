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
from profile_registry import route_key_from_dict as _route_key_from_dict  # noqa: E402


OUTPUT_JSON = "phase_native_audio_family_route_policy.json"
OUTPUT_MD = "PHASE_NATIVE_AUDIO_FAMILY_ROUTE_POLICY.md"


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
        raise ValueError(f"No family choices found in route source oracle: {oracle_json}")
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


def _training_rows(train_delta_jsons: Sequence[Path], keys: Sequence[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for delta_json in train_delta_jsons:
        features = _case_features([delta_json])
        for case_name, row in sorted(features.items()):
            rows.append(
                {
                    "case_name": case_name,
                    "group": _case_group(case_name),
                    "delta_json": str(delta_json),
                    "vector": _vector(row, keys),
                }
            )
    return rows


def _fit_group_centroids(training_rows: Sequence[dict[str, Any]], means: Sequence[float], stds: Sequence[float]) -> dict[str, list[float]]:
    by_group: dict[str, list[list[float]]] = defaultdict(list)
    for row in training_rows:
        by_group[str(row["group"])].append(_scale(row["vector"], means, stds))
    return {group: _centroid(vectors) for group, vectors in sorted(by_group.items())}


def _predict_group_centroid(
    vector: Sequence[float],
    *,
    means: Sequence[float],
    stds: Sequence[float],
    centroids: dict[str, list[float]],
) -> tuple[str, float]:
    scaled = _scale(vector, means, stds)
    group = min(centroids, key=lambda item: _distance(scaled, centroids[item]))
    return group, _distance(scaled, centroids[group])


def _predict_group_knn(
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
            str(row["group"]),
        )
        for row in training_rows
    ]
    scored.sort(key=lambda item: item[0])
    nearest = scored[: max(1, min(int(k), len(scored)))]
    votes: dict[str, float] = defaultdict(float)
    for distance, group in nearest:
        votes[group] += 1.0 / max(distance, 1.0e-6)
    group = max(votes, key=votes.get)
    mean_distance = sum(distance for distance, _ in nearest) / float(len(nearest))
    return group, mean_distance


def _predict_group(
    vector: Sequence[float],
    *,
    model: str,
    means: Sequence[float],
    stds: Sequence[float],
    training_rows: Sequence[dict[str, Any]],
    centroids: dict[str, list[float]],
) -> tuple[str, float]:
    if model == "group_centroid_v1":
        return _predict_group_centroid(vector, means=means, stds=stds, centroids=centroids)
    if model == "group_knn1_v1":
        return _predict_group_knn(vector, means=means, stds=stds, training_rows=training_rows, k=1)
    if model == "group_knn5_v1":
        return _predict_group_knn(vector, means=means, stds=stds, training_rows=training_rows, k=5)
    raise ValueError(f"Unknown family-route model: {model}")


def _leave_one_group_accuracy(training_rows: Sequence[dict[str, Any]], *, model: str) -> dict[str, Any]:
    correct = 0
    total = 0
    by_group: dict[str, list[int]] = defaultdict(list)
    for idx, held in enumerate(training_rows):
        train = [row for j, row in enumerate(training_rows) if j != idx]
        if not train:
            continue
        means, stds = _fit_scaler([row["vector"] for row in train])
        centroids = _fit_group_centroids(train, means, stds)
        pred, _ = _predict_group(
            held["vector"],
            model=model,
            means=means,
            stds=stds,
            training_rows=train,
            centroids=centroids,
        )
        ok = int(pred == held["group"])
        correct += ok
        total += 1
        by_group[str(held["group"])].append(ok)
    return {
        "total": total,
        "group_accuracy": correct / float(total) if total else 0.0,
        "group_accuracy_by_group": {
            group: sum(vals) / float(len(vals)) for group, vals in sorted(by_group.items()) if vals
        },
    }


def _markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# Phase-Native Audio Family Route Policy",
        "",
        "This policy predicts a no-future acoustic family from Circleworld/prefix",
        "metadata, then applies a frozen source-oracle family route table.",
        "",
        "## Summary",
        "",
        f"- Status: `{summary['status']}`",
        f"- Model: `{summary['model']}`",
        f"- Training cases: `{summary['train_case_count']}`",
        f"- Target cases: `{summary['target_case_count']}`",
        f"- Feature count: `{summary['feature_count']}`",
        f"- Leave-one group accuracy: `{summary['training_diagnostics']['group_accuracy']:.6f}`",
        "",
        "## Predicted Groups",
        "",
        "```json",
        json.dumps(summary["predicted_group_counts"], indent=2),
        "```",
        "",
        "## Interpretation",
        "",
        summary["interpretation"],
    ]
    return "\n".join(lines) + "\n"


def train_family_route_policy(
    *,
    train_delta_jsons: Sequence[Path],
    route_oracle_json: Path,
    target_delta_json: Path,
    out_dir: Path,
    model: str,
) -> dict[str, Any]:
    route_table = _family_routes(route_oracle_json)
    train_feature_sets = [_case_features([path]) for path in train_delta_jsons]
    target_features = _case_features([target_delta_json])
    keys = _feature_keys([*train_feature_sets, target_features])
    training_rows = _training_rows(train_delta_jsons, keys)
    if not training_rows:
        raise RuntimeError("No family training rows produced")
    means, stds = _fit_scaler([row["vector"] for row in training_rows])
    centroids = _fit_group_centroids(training_rows, means, stds)
    fallback_group = Counter(str(row["group"]) for row in training_rows).most_common(1)[0][0]

    case_routes: dict[str, dict[str, Any]] = {}
    predictions: list[dict[str, Any]] = []
    for case_name, row in sorted(target_features.items()):
        vector = _vector(row, keys)
        group, distance = _predict_group(
            vector,
            model=model,
            means=means,
            stds=stds,
            training_rows=training_rows,
            centroids=centroids,
        )
        route_group = group if group in route_table else fallback_group
        if route_group not in route_table:
            route_group = sorted(route_table)[0]
        route = route_table[route_group]
        case_routes[case_name] = _route_dict(route)
        predictions.append(
            {
                "case_name": case_name,
                "true_group_from_case_name_for_audit": _case_group(case_name),
                "predicted_group": group,
                "route_group": route_group,
                "group_distance": distance,
                "route": _route_dict(route),
                "group_match_for_audit": group == _case_group(case_name),
            }
        )

    diagnostics = _leave_one_group_accuracy(training_rows, model=model)
    group_counts = Counter(str(pred["predicted_group"]) for pred in predictions)
    summary = {
        "schema": "phase_native_audio_family_route_policy_v1",
        "status": "family_route_policy_ready",
        "model": model,
        "route_oracle_json": str(route_oracle_json),
        "train_delta_jsons": [str(path) for path in train_delta_jsons],
        "target_delta_json": str(target_delta_json),
        "feature_count": len(keys),
        "feature_keys": keys,
        "train_case_count": len(training_rows),
        "target_case_count": len(predictions),
        "training_group_counts": dict(sorted(Counter(str(row["group"]) for row in training_rows).items())),
        "predicted_group_counts": dict(sorted(group_counts.items())),
        "route_family_table": {group: _route_dict(route) for group, route in sorted(route_table.items())},
        "fallback_group": fallback_group,
        "fallback_route": _route_dict(route_table[fallback_group]) if fallback_group in route_table else None,
        "case_routes": case_routes,
        "predictions": predictions,
        "group_centroids": centroids,
        "scaler": {
            "means": means,
            "stds": stds,
        },
        "training_diagnostics": diagnostics,
        "future_target_audio_used_for_family_training_labels_only": False,
        "target_future_audio_used_for_route_selection": False,
        "target_future_metrics_used_for_route_selection": False,
        "interpretation": (
            "The policy is ready for live selected-route rendering. It predicts family from no-future "
            "Circleworld/prefix metadata and maps that family to a frozen route table."
        ),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / OUTPUT_JSON).write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (out_dir / OUTPUT_MD).write_text(_markdown(summary), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Train no-future family classifier and route table policy.")
    parser.add_argument("--train-delta-json", required=True, action="append", type=Path)
    parser.add_argument("--route-oracle-json", required=True, type=Path)
    parser.add_argument("--target-delta-json", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument(
        "--model",
        choices=("group_centroid_v1", "group_knn1_v1", "group_knn5_v1"),
        default="group_knn5_v1",
    )
    args = parser.parse_args()
    summary = train_family_route_policy(
        train_delta_jsons=args.train_delta_json,
        route_oracle_json=args.route_oracle_json,
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
                "group_accuracy": summary["training_diagnostics"]["group_accuracy"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
